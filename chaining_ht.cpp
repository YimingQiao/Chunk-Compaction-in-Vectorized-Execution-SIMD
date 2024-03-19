#include "chaining_ht.h"

namespace simd_compaction {
HashTable::HashTable(size_t n_rhs_tuples, size_t chunk_factor) {
  n_buckets_ = 1;
  while (n_buckets_ < 2 * n_rhs_tuples) n_buckets_ *= 2;
  linked_lists_.resize(n_buckets_);
  for (auto &bucket : linked_lists_) bucket = std::make_unique<list<Key>>();

  // bucket mask
  SCALAR_BUCKET_MASK = n_buckets_ - 1;
  SIMD_BUCKET_MASK = _mm512_set1_epi64(n_buckets_ - 1);

  // Tuple in Hash Table
  vector<Tuple> rhs_table(n_rhs_tuples);
  size_t cnt = 0;
  const size_t num_unique = n_rhs_tuples / chunk_factor + (n_rhs_tuples % chunk_factor != 0);
  for (size_t i = 0; i < num_unique; ++i) {
    auto unique_value = i * (n_rhs_tuples / num_unique);
    for (size_t j = 0; j < chunk_factor && cnt < n_rhs_tuples; ++j) {
      auto payload = cnt + 10000000;
      rhs_table[cnt].emplace_back(unique_value);
      rhs_table[cnt].emplace_back(payload);
      ++cnt;
    }
  }

  // build hash table
  for (size_t i = 0; i < n_rhs_tuples; ++i) {
    auto &tuple = rhs_table[i];
    Attribute value = tuple[0];
    auto bucket_idx = murmurhash64(value) & SCALAR_BUCKET_MASK;
    auto &bucket = linked_lists_[bucket_idx];
    bucket->push_back(tuple[0]);
  }
}

ScanStructure HashTable::Probe(Vector &join_key) {
  vector<list<Key> *> ptrs(kBlockSize);
  vector<uint32_t> ptrs_sel_vector(kBlockSize);

  CycleProfiler::Get().Start();

  for (size_t i = 0; i < join_key.count_; ++i) {
    auto attr = join_key.GetValue(join_key.selection_vector_[i]);
    auto bucket_idx = murmurhash64(attr) & SCALAR_BUCKET_MASK;
    ptrs[i] = linked_lists_[bucket_idx].get();
  }

  CycleProfiler::Get().End(0);

  size_t n_non_empty = 0;
  for (size_t i = 0; i < join_key.count_; ++i) {
    if (!ptrs[i]->empty()) ptrs_sel_vector[n_non_empty++] = i;
  }

  return ScanStructure(n_non_empty, ptrs_sel_vector, ptrs, join_key.selection_vector_, this);
}

size_t ScanStructure::Next(Vector &join_key, DataChunk &input, DataChunk &result, bool compact_mode) {
  // reset the result chunk
  result.Reset();

  if (compact_mode) {
    // take the buffer data if the buffer is not empty
    if (HasBuffer()) {
      result.data_.swap(buffer_->data_);
      std::swap(result.count_, buffer_->count_);
    }

    // compact result chunks without extra memory copy
    while (HasNext() && !HasBuffer()) { NextInternal(join_key, input, result); }
  } else {
    NextInternal(join_key, input, result);
  }

  return result.count_;
}

void ScanStructure::NextInternal(Vector &join_key, DataChunk &input, DataChunk &result) {
  if (count_ == 0) {
    // no pointers left to chase
    return;
  }

  vector<uint32_t> result_vector(kBlockSize);
  size_t result_count = ScanInnerJoin(join_key, result_vector);

  if (result_count > 0) {
    if (result.count_ + result_count <= kBlockSize) {
      // matches were found
      // construct the result
      // on the LHS, we create a slice using the result vector
      result.Slice(input, result_vector, result_count);

      // on the RHS, we need to fetch the data from the hash table
      vector<Vector *> cols{&result.data_[input.data_.size()], &result.data_[input.data_.size() + 1]};
      GatherResult(cols, result_vector, result_count);
    } else {
      // init the buffer
      if (buffer_ == nullptr) buffer_ = std::make_unique<DataChunk>(result.types_);

      // buffer the result
      buffer_->Slice(input, result_vector, result_count);
      vector<Vector *> cols{&buffer_->data_[input.data_.size()], &buffer_->data_[input.data_.size() + 1]};
      GatherResult(cols, result_vector, result_count);
    }
  }
  AdvancePointers();
}

size_t ScanStructure::ScanInnerJoin(Vector &join_key, vector<uint32_t> &result_vector) {
  while (true) {
    CycleProfiler::Get().Start();

    // Match
    size_t result_count = 0;
    for (size_t i = 0; i < count_; ++i) {
      size_t idx = bucket_sel_vector_[i];
      auto &l_key = join_key.GetValue(key_sel_vector_[idx]);
      auto &r_key = (*iterators_[idx]);

      // remove branch prediction
      result_vector[result_count] = idx;
      result_count += (l_key == r_key);
      // if (l_key == r_key) result_vector[result_count++] = idx;
    }

    CycleProfiler::Get().End(1);

    if (result_count > 0) return result_count;

    // no matches found: check the next set of pointers
    AdvancePointers();
    if (count_ == 0) return 0;
  }
}

void ScanStructure::AdvancePointers() {
  CycleProfiler::Get().Start();

  size_t new_count = 0;
  for (size_t i = 0; i < count_; i++) {
    auto idx = bucket_sel_vector_[i];

    // remove branch prediction
    bucket_sel_vector_[new_count] = idx;
    new_count += (++iterators_[idx] != iterator_ends_[idx]);
    // if (++iterators_[idx] != iterator_ends_[idx]) bucket_sel_vector_[new_count++] = idx;
  }
  count_ = new_count;

  CycleProfiler::Get().End(3);
}

void ScanStructure::GatherResult(vector<Vector *> cols, vector<uint32_t> &sel_vector, size_t count) {
  CycleProfiler::Get().Start();

  auto col_count = cols[0]->count_;
  for (size_t i = 0; i < count; ++i) {
    auto idx = sel_vector[i];
    cols[1]->GetValue(i + col_count) = (*iterators_[idx]);
  }
  cols[0]->count_ += count;
  cols[1]->count_ += count;

  CycleProfiler::Get().End(2);
}

size_t ScanStructure::InOneNext(Vector &join_key, DataChunk &input, DataChunk &result, bool compact_mode) {
  // reset the result chunk
  result.Reset();

  if (compact_mode) {
    // take the buffer data if the buffer is not empty
    if (HasBuffer()) {
      result.data_.swap(buffer_->data_);
      std::swap(result.count_, buffer_->count_);
    }

    // compact result chunks without extra memory copy
    while (HasNext() && !HasBuffer()) { InOneNextInternal(join_key, input, result); }
  } else {
    InOneNextInternal(join_key, input, result);
  }

  return result.count_;
}

void ScanStructure::InOneNextInternal(Vector &join_key, DataChunk &input, DataChunk &result) {
  vector<uint32_t> result_vector(kBlockSize);
  size_t result_count = 0;
  size_t new_count = 0;

  CycleProfiler::Get().Start();

  vector<Vector *> cols{&result.data_[input.data_.size()], &result.data_[input.data_.size() + 1]};
  for (size_t i = 0; i < count_; ++i) {
    size_t idx = bucket_sel_vector_[i];
    auto &l_key = join_key.GetValue(key_sel_vector_[idx]);
    auto &r_key = (*iterators_[idx]);

    // match & gather
    cols[1]->GetValue(cols[1]->count_ + result_count) = r_key;
    result_vector[result_count] = idx;
    result_count += (l_key == r_key);

    // advance, remove branch prediction
    bucket_sel_vector_[new_count] = idx;
    new_count += (++iterators_[idx] != iterator_ends_[idx]);
  }
  result.Slice(input, result_vector, result_count);

  // update count
  cols[1]->count_ = result_count;
  cols[0]->count_ = result_count;
  count_ = new_count;

  CycleProfiler::Get().End(1);
}

// --------------------------------------------  SIMD  --------------------------------------------
ScanStructure HashTable::SIMDProbe(Vector &join_key) {
  vector<list<Key> *> ptrs(kBlockSize);
  vector<uint32_t> ptrs_sel_vector(kBlockSize);

  CycleProfiler::Get().Start();

  int32_t tail = join_key.count_ & 7;
  for (uint64_t i = 0; i < join_key.count_ - tail; i += 8) {
    __m256i indices = _mm256_loadu_epi32(join_key.selection_vector_.data() + i);
    auto keys = _mm512_i32gather_epi64(indices, join_key.Data(), 8);
    __m512i hashes = mm512_murmurhash64(keys);
    __m512i bucket_indices = _mm512_and_si512(hashes, SIMD_BUCKET_MASK);
    __m512i address = _mm512_i64gather_epi64(bucket_indices, linked_lists_.data(), 8);
    _mm512_storeu_epi64(ptrs.data() + i, address);
  }

  CycleProfiler::Get().End(0);

  for (size_t i = join_key.count_ - tail; i < join_key.count_; ++i) {
    auto attr = join_key.GetValue(join_key.selection_vector_[i]);
    auto bucket_idx = murmurhash64(attr) & SCALAR_BUCKET_MASK;
    ptrs[i] = linked_lists_[bucket_idx].get();
  }

  size_t n_non_empty = 0;
  for (size_t i = 0; i < join_key.count_; ++i) {
    if (!ptrs[i]->empty()) ptrs_sel_vector[n_non_empty++] = i;
  }

  return ScanStructure(n_non_empty, ptrs_sel_vector, ptrs, join_key.selection_vector_, this);
}

size_t ScanStructure::SIMDNext(Vector &join_key, DataChunk &input, DataChunk &result, bool compact_mode) {
  // reset the result chunk
  result.Reset();

  if (compact_mode) {
    // take the buffer data if the buffer is not empty
    if (HasBuffer()) {
      result.data_.swap(buffer_->data_);
      std::swap(result.count_, buffer_->count_);
    }

    // compact result chunks without extra memory copy
    while (HasNext() && !HasBuffer()) { SIMDNextInternal(join_key, input, result); }
  } else {
    SIMDNextInternal(join_key, input, result);
  }

  return result.count_;
}

void ScanStructure::SIMDNextInternal(Vector &join_key, DataChunk &input, DataChunk &result) {
  if (count_ == 0) {
    // no pointers left to chase
    return;
  }

  vector<uint32_t> result_vector(kBlockSize);
  size_t result_count = SIMDScanInnerJoin(join_key, result_vector);

  if (result_count > 0) {
    if (result.count_ + result_count <= kBlockSize) {
      // matches were found
      // construct the result
      // on the LHS, we create a slice using the result vector
      result.Slice(input, result_vector, result_count);

      // on the RHS, we need to fetch the data from the hash table
      vector<Vector *> cols{&result.data_[input.data_.size()], &result.data_[input.data_.size() + 1]};
      SIMDGatherResult(cols, result_vector, result_count);
    } else {
      // init the buffer
      if (buffer_ == nullptr) buffer_ = std::make_unique<DataChunk>(result.types_);

      // buffer the result
      buffer_->Slice(input, result_vector, result_count);
      vector<Vector *> cols{&buffer_->data_[input.data_.size()], &buffer_->data_[input.data_.size() + 1]};
      SIMDGatherResult(cols, result_vector, result_count);
    }
  }
  SIMDAdvancePointers();
}

size_t ScanStructure::SIMDScanInnerJoin(Vector &join_key, vector<uint32_t> &result_vector) {
  while (true) {
    CycleProfiler::Get().Start();

    size_t result_count = 0;
    int64_t tail = count_ & 7;
    for (size_t i = 0; i < count_ - tail; i += 8) {
      auto indices = _mm256_loadu_epi32(bucket_sel_vector_.data() + i);
      auto key_indices = _mm256_i32gather_epi32((int *) key_sel_vector_.data(), indices, 4);
      auto l_keys = _mm512_i32gather_epi64(key_indices, join_key.Data(), 8);

      auto iterators = _mm512_i32gather_epi64(indices, iterators_.data(), 8);
      iterators = _mm512_add_epi64(iterators, ALL_SIXTEEN);
      auto r_keys = _mm512_i64gather_epi64(iterators, nullptr, 1);
      __mmask8 match = _mm512_cmpeq_epi64_mask(l_keys, r_keys);

      _mm256_mask_compressstoreu_epi32(result_vector.data() + result_count, match, indices);
      result_count += _mm_popcnt_u32(match);
    }

    for (size_t i = count_ - tail; i < count_; ++i) {
      size_t idx = bucket_sel_vector_[i];
      auto &l_key = join_key.GetValue(key_sel_vector_[idx]);
      auto &r_key = (*iterators_[idx]);
      if (l_key == r_key) result_vector[result_count++] = idx;
    }

    CycleProfiler::Get().End(1);

    if (result_count > 0) return result_count;

    // no matches found: check the next set of pointers
    SIMDAdvancePointers();
    if (count_ == 0) return 0;
  }
}

void ScanStructure::SIMDAdvancePointers() {
  CycleProfiler::Get().Start();

  size_t tail = count_ & 7;
  size_t new_count = 0;
  for (size_t i = 0; i < count_ - tail; i += 8) {
    auto indices = _mm256_loadu_epi32(bucket_sel_vector_.data() + i);
    auto iterators = _mm512_i32gather_epi64(indices, iterators_.data(), 8);
    auto next_its = _mm512_i64gather_epi64(iterators, nullptr, 1);
    _mm512_i32scatter_epi64(iterators_.data(), indices, next_its, 8);
    auto its_ends = _mm512_i32gather_epi64(indices, iterator_ends_.data(), 8);
    __mmask8 valid = _mm512_cmpneq_epi64_mask(next_its, its_ends);
    _mm256_mask_compressstoreu_epi32(bucket_sel_vector_.data() + new_count, valid, indices);
    new_count += _mm_popcnt_u32(valid);
  }

  for (size_t i = count_ - tail; i < count_; ++i) {
    auto idx = bucket_sel_vector_[i];
    if (++iterators_[idx] != buckets_[idx]->end()) bucket_sel_vector_[new_count++] = idx;
  }
  count_ = new_count;

  CycleProfiler::Get().End(3);
}

void ScanStructure::SIMDGatherResult(vector<Vector *> cols, vector<uint32_t> &sel_vector, size_t count) {
  CycleProfiler::Get().Start();

  auto &col = *cols[1];
  int32_t tail = count & 7;
  for (size_t i = 0; i < count - tail; i += 8) {
    __m256i indices = _mm256_loadu_epi32(sel_vector.data() + i);
    auto iterators = _mm512_i32gather_epi64(indices, iterators_.data(), 8);
    iterators = _mm512_add_epi64(iterators, ALL_SIXTEEN);
    auto keys = _mm512_i64gather_epi64(iterators, nullptr, 1);
    _mm512_storeu_epi64(col.Data() + i + col.count_, keys);
  }

  for (size_t i = count - tail; i < count; i++) {
    auto idx = sel_vector[i];
    col.GetValue(i + col.count_) = (*iterators_[idx]);
  }
  cols[0]->count_ += count;
  cols[1]->count_ += count;

  CycleProfiler::Get().End(2);
}

size_t ScanStructure::SIMDInOneNext(Vector &join_key, DataChunk &input, DataChunk &result, bool compact_mode) {
  // reset the result chunk
  result.Reset();

  if (compact_mode) {
    // take the buffer data if the buffer is not empty
    if (HasBuffer()) {
      result.data_.swap(buffer_->data_);
      std::swap(result.count_, buffer_->count_);
    }

    // compact result chunks without extra memory copy
    while (HasNext() && !HasBuffer()) { SIMDInOneNextInternal(join_key, input, result); }
  } else {
    SIMDInOneNextInternal(join_key, input, result);
  }

  return result.count_;
}

void ScanStructure::SIMDInOneNextInternal(Vector &join_key, DataChunk &input, DataChunk &result) {
  vector<uint32_t> result_vector(kBlockSize);
  size_t result_count = 0;
  size_t tail = count_ & 7;
  size_t new_count = 0;

  CycleProfiler::Get().Start();
  vector<Vector *> cols{&result.data_[input.data_.size()], &result.data_[input.data_.size() + 1]};
  for (size_t i = 0; i < count_ - tail; i += 8) {
    auto indices = _mm256_loadu_epi32(bucket_sel_vector_.data() + i);
    auto key_indices = _mm256_i32gather_epi32((int *) key_sel_vector_.data(), indices, 4);
    auto l_keys = _mm512_i32gather_epi64(key_indices, join_key.Data(), 8);

    auto iterators = _mm512_i32gather_epi64(indices, iterators_.data(), 8);
    auto node = _mm512_add_epi64(iterators, ALL_SIXTEEN);
    auto r_keys = _mm512_i64gather_epi64(node, nullptr, 1);

    // match & gather
    __mmask8 match = _mm512_cmpeq_epi64_mask(l_keys, r_keys);
    auto r_payload = _mm512_i64gather_epi64(node, nullptr, 1);
    _mm512_mask_storeu_epi64(cols[1]->Data() + cols[1]->count_ + result_count, match, r_payload);
    _mm256_mask_compressstoreu_epi32(result_vector.data() + result_count, match, indices);
    result_count += _mm_popcnt_u32(match);

    // advance
    auto next_its = _mm512_i64gather_epi64(iterators, nullptr, 1);
    _mm512_i32scatter_epi64(iterators_.data(), indices, next_its, 8);
    auto its_ends = _mm512_i32gather_epi64(indices, iterator_ends_.data(), 8);
    __mmask8 valid = _mm512_cmpneq_epi64_mask(next_its, its_ends);
    _mm256_mask_compressstoreu_epi32(bucket_sel_vector_.data() + new_count, valid, indices);
    new_count += _mm_popcnt_u32(valid);
  }

  for (size_t i = count_ - tail; i < count_; ++i) {
    size_t idx = bucket_sel_vector_[i];
    auto &l_key = join_key.GetValue(key_sel_vector_[idx]);
    auto &r_key = (*iterators_[idx]);

    // match & gather
    cols[1]->GetValue(cols[1]->count_ + result_count) = r_key;
    result_vector[result_count] = idx;
    result_count += (l_key == r_key);

    // advance, remove branch prediction
    bucket_sel_vector_[new_count] = idx;
    new_count += (++iterators_[idx] != iterator_ends_[idx]);
  }
  result.Slice(input, result_vector, result_count);

  // update count
  cols[1]->count_ = result_count;
  cols[0]->count_ = result_count;
  count_ = new_count;

  CycleProfiler::Get().End(1);
}
}// namespace simd_compaction
