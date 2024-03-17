#include "hash_table.h"

namespace simd_compaction {
HashTable::HashTable(size_t n_rhs_tuples, size_t chunk_factor)
    : loaded_keys(kBlockSize), bucket_ids(kBlockSize), ptrs_sel_vector(kBlockSize), result_vector(kBlockSize),
      bucket_sel_vector_(kBlockSize), iterators_(kBlockSize), iterators_end_(kBlockSize) {
  // number of buckets should be the minimum exp number of 2.
  n_buckets_ = 1;
  while (n_buckets_ < 2 * n_rhs_tuples) n_buckets_ *= 2;

  SCALAR_BUCKET_MASK = n_buckets_ - 1;
  BUCKET_MASK = _mm512_set1_epi64(n_buckets_ - 1);

  linked_lists_.resize(n_buckets_);
  for (auto &bucket : linked_lists_) bucket = std::make_unique<list<Tuple>>();

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
    auto bucket_idx = murmurhash64(value) % n_buckets_;
    // auto bucket_idx = value % n_buckets_;
    auto &bucket = linked_lists_[bucket_idx];
    bucket->push_back(tuple);
  }

  // get list ends.
  list_sizes_.resize(n_buckets_);
  for (size_t i = 0; i < n_buckets_; ++i) list_sizes_[i] = linked_lists_[i]->size();
}

void HashTable::Probe(Vector &join_key, int64_t count, vector<uint32_t> &sel_vector) {
  ref_sel_vector = &sel_vector;
  count_ = count;

  CycleProfiler::Get().Start();

  // gather, hash and find buckets
  for (size_t i = 0; i < count; i++) {
    int64_t key = join_key[sel_vector[i]];
    bucket_ids[i] = murmurhash64(key) & SCALAR_BUCKET_MASK;
  }

  CycleProfiler::Get().End(0);
}

void HashTable::SIMDProbe(Vector &join_key, int64_t count, vector<uint32_t> &sel_vector) {
  CycleProfiler::Get().Start();

  ref_sel_vector = &sel_vector;
  count_ = count;

  int32_t tail = count & 7;
  for (uint64_t i = 0; i < count - tail; i += 8) {
    __m256i indices = _mm256_loadu_epi32(sel_vector.data() + i);
    auto keys = _mm512_i32gather_epi64(indices, join_key.data_->data(), 8);
    __m512i hashes = mm512_murmurhash64(keys);
    __m512i bucket_indices = _mm512_and_si512(hashes, BUCKET_MASK);
    _mm512_storeu_epi64(bucket_ids.data() + i, bucket_indices);
  }

  for (size_t i = count - tail; i < count; i++) {
    int64_t key = join_key[sel_vector[i]];
    bucket_ids[i] = murmurhash64(key) & SCALAR_BUCKET_MASK;
  }

  CycleProfiler::Get().End(0);
}

ScanStructure HashTable::GetScanStructure() {
  int64_t n_valid = 0;
  for (size_t i = 0; i < count_; ++i) {
    auto *ptr = linked_lists_[bucket_ids[i]].get();

    if (ptr && !ptr->empty()) {
      bucket_sel_vector_[n_valid++] = i;
      iterators_[i] = ptr->begin();
      iterators_end_[i] = ptr->end();
    }
  }

  return ScanStructure(n_valid, *ref_sel_vector, result_vector, bucket_sel_vector_, iterators_, iterators_end_);
}

uint32_t ScanStructure::Next(Vector &join_key, DataChunk &input, DataChunk &result) {
  size_t result_count = Match(join_key, result_vector_);

  if (result_count > 0) {
    // on the LHS, we create a slice using the result vector
    result.Slice(input, result_vector_, result_count);

    // on the RHS, we need to fetch the data from the hash table
    vector<Vector *> cols{&result.data_[input.data_.size()], &result.data_[input.data_.size() + 1]};

    CycleProfiler::Get().Start();

    GatherResult(cols, input.selection_vector_, result_vector_, result_count);

    CycleProfiler::Get().End(2);
  }
  AdvancePointers();

  return result_count;
}

uint32_t ScanStructure::SIMDNext(Vector &join_key, DataChunk &input, DataChunk &result) {
  size_t result_count = SIMDMatch(join_key, result_vector_);

  if (result_count > 0) {
    // matches were found
    // construct the result
    // on the LHS, we create a slice using the result vector
    result.SIMDSlice(input, result_vector_, result_count);

    // on the RHS, we need to fetch the data from the hash table
    vector<Vector *> cols{&result.data_[input.data_.size()], &result.data_[input.data_.size() + 1]};

    CycleProfiler::Get().Start();

    SIMDGatherResult(cols, input.selection_vector_, result_vector_, result_count);

    CycleProfiler::Get().End(2);
  }
  SIMDAdvancePointers();

  return result_count;
}

size_t ScanStructure::Match(Vector &join_key, vector<uint32_t> &result_vector) {
  CycleProfiler::Get().Start();

  // Match
  size_t result_count = 0;
  int64_t tail = count_ & 15;

  for (size_t i = 0; i < count_ - tail; ++i) {
    size_t idx = bucket_sel_vector_[i];
    auto &l_key = join_key[key_sel_vector_[idx]];
    auto &r_key = (*iterators_[idx])[0];

    // if (l_key == r_key) result_vector[result_count++] = idx;
    result_vector[result_count] = idx;
    result_count += (l_key == r_key);
  }

  CycleProfiler::Get().End(1);
  return result_count;
}

size_t ScanStructure::SIMDMatch(Vector &join_key, vector<uint32_t> &result_vector) {
  CycleProfiler::Get().Start();

  size_t result_count = 0;
  int64_t tail = count_ & 15;

  // Memory format of list::iterators
  //      struct ListNode {
  //        ListNode *next;
  //        ListNode *prev;
  //        T* data;
  //      };
  //
  //      struct ListIterator {
  //        ListNode<T> currentNode;
  //      };
  //
  for (int32_t i = 0; i < count_ - tail; i += 16) {
    //    auto indices = _mm256_loadu_epi32(bucket_sel_vector_.data() + i);
    //    auto l_keys = _mm512_i32gather_epi64(indices, join_key.data_->data(), 8);
    //    auto iterators = _mm512_i32gather_epi64(indices, iterators_.data(), 8);
    //    iterators = _mm512_add_epi64(iterators, ALL_SIXTEEN);
    //    auto tuple = _mm512_i64gather_epi64(iterators, nullptr, 1);
    //    auto r_keys = _mm512_i64gather_epi64(nodes, nullptr, 1);
    //    __mmask8 match = _mm512_cmpeq_epi64_mask(l_keys, r_keys);
    //
    //    _mm256_mask_compressstoreu_epi32(result_vector_.data() + result_count, match, indices);
    //    result_count += _mm_popcnt_u32(match);

    __m512i indices = _mm512_loadu_epi32(bucket_sel_vector_.data() + i);
    auto high_indices = _mm512_extracti32x8_epi32(indices, 1);
    auto l_keys = _mm512_i32gather_epi64(high_indices, join_key.data_->data(), 8);
    auto iterators = _mm512_i32gather_epi64(high_indices, iterators_.data(), 8);
    iterators = _mm512_add_epi64(iterators, ALL_SIXTEEN);
    auto tuple = _mm512_i64gather_epi64(iterators, nullptr, 1);
    auto r_keys = _mm512_i64gather_epi64(tuple, nullptr, 1);
    __mmask16 match = ((__mmask16) _mm512_cmpeq_epi64_mask(l_keys, r_keys)) << 8;

    auto low_indices = _mm512_extracti32x8_epi32(indices, 0);
    auto l_keys1 = _mm512_i32gather_epi64(low_indices, join_key.data_->data(), 8);
    auto iterators1 = _mm512_i32gather_epi64(low_indices, iterators_.data(), 8);
    iterators1 = _mm512_add_epi64(iterators1, ALL_SIXTEEN);
    auto tuple1 = _mm512_i64gather_epi64(iterators1, nullptr, 1);
    auto r_keys1 = _mm512_i64gather_epi64(tuple1, nullptr, 1);
    match = match | _mm512_cmpeq_epi64_mask(l_keys1, r_keys1);

    _mm512_mask_compressstoreu_epi32(result_vector_.data() + result_count, match, indices);
    result_count += _mm_popcnt_u32(match);
  }

  //  for (int32_t i = count_ - tail; i < count_; ++i) {
  //    size_t idx = bucket_sel_vector_[i];
  //    auto &l_key = join_key.GetValue(key_sel_vector_[idx]);
  //    auto &r_key = (*iterators_[idx])[0];
  //    if (l_key == r_key) result_vector[result_count++] = idx;
  //  }

  CycleProfiler::Get().End(1);

  return result_count;
}

void ScanStructure::AdvancePointers() {
  CycleProfiler::Get().Start();
  size_t new_count = 0;
  int32_t tail = count_ & 15;
  for (size_t i = 0; i < count_ - tail; i++) {
    auto idx = bucket_sel_vector_[i];

    bucket_sel_vector_[new_count] = idx;
    new_count += ++iterators_[idx] != iterators_end_[idx];
    // if (++iterators_[idx] != iterators_end_[idx]) { bucket_sel_vector_[new_count++] = idx; }
  }
  count_ = new_count;
  CycleProfiler::Get().End(3);
}

void ScanStructure::SIMDAdvancePointers() {
  CycleProfiler::Get().Start();

  int32_t tail = count_ & 15;
  size_t new_count = 0;

  for (int32_t i = 0; i < count_ - tail; i += 16) {
    __m512i indices = _mm512_loadu_epi32(bucket_sel_vector_.data() + i);

    auto high_indices = _mm512_extracti32x8_epi32(indices, 1);
    auto iterators = _mm512_i32gather_epi64(high_indices, iterators_.data(), 8);
    auto next_its = _mm512_i64gather_epi64(iterators, nullptr, 1);
    _mm512_i32scatter_epi64(iterators_.data(), high_indices, next_its, 8);
    auto its_ends = _mm512_i32gather_epi64(high_indices, iterators_end_.data(), 8);
    __mmask16 valid = ((__mmask16) _mm512_cmpneq_epi64_mask(next_its, its_ends)) << 8;

    auto low_indices = _mm512_extracti32x8_epi32(indices, 0);
    auto iterators1 = _mm512_i32gather_epi64(low_indices, iterators_.data(), 8);
    auto next_its1 = _mm512_i64gather_epi64(iterators1, nullptr, 1);
    _mm512_i32scatter_epi64(iterators_.data(), low_indices, next_its1, 8);
    auto its_ends1 = _mm512_i32gather_epi64(low_indices, iterators_end_.data(), 8);
    valid = valid | _mm512_cmpneq_epi64_mask(next_its1, its_ends1);

    _mm512_mask_compressstoreu_epi32(bucket_sel_vector_.data() + new_count, valid, indices);
    new_count += _mm_popcnt_u32(valid);
  }

  //  for (size_t i = count_ - tail; i < count_; i++) {
  //    auto idx = bucket_sel_vector_[i];
  //    if (++iterators_[idx] != iterators_end_[idx]) bucket_sel_vector_[new_count++] = idx;
  //  }

  count_ = new_count;

  CycleProfiler::Get().End(3);
}

void ScanStructure::GatherResult(vector<Vector *> cols, vector<uint32_t> &sel_vector, vector<uint32_t> &result_vector,
                                 size_t count) {
  int32_t tail = count & 7;
  for (size_t i = 0; i < count - tail; ++i) {
    auto idx = result_vector[i];
    auto &tuple = *iterators_[idx];

    // columns from the right table align with the selection vector given by the left table
    size_t pos = sel_vector[idx];
    for (size_t j = 0; j < cols.size(); ++j) { (*cols[j])[pos] = tuple[j]; }
  }
}

void ScanStructure::SIMDGatherResult(vector<Vector *> cols, vector<uint32_t> &sel_vector,
                                     vector<uint32_t> &result_vector, int32_t count) {
  int32_t tail = count & 7;
  for (int32_t i = 0; i < count - tail; i += 8) {
    __m256i indices = _mm256_loadu_epi32(result_vector.data() + i);
    auto positions = _mm256_i32gather_epi32((int *) sel_vector.data(), indices, 4);
    auto iterators = _mm512_i32gather_epi64(indices, iterators_.data(), 8);
    iterators = _mm512_add_epi64(iterators, ALL_SIXTEEN);
    auto nodes = _mm512_i64gather_epi64(iterators, nullptr, 1);

    for (size_t j = 0; j < cols.size(); ++j) {
      __m512i attribute = _mm512_add_epi64(nodes, _mm512_set1_epi64(8 * j));
      auto keys = _mm512_i64gather_epi64(attribute, nullptr, 1);
      _mm512_i32scatter_epi64(cols[j]->data_->data(), positions, keys, 8);
    }
  }

  //  for (int32_t i = count - tail; i < count; ++i) {
  //    auto idx = result_vector[i];
  //    auto &tuple = *iterators_[idx];
  //    size_t pos = sel_vector[idx];
  //
  //    for (size_t j = 0; j < cols.size(); ++j) {
  //      auto &col = *cols[j];
  //      col.GetValue(pos) = tuple[j];
  //    }
  //  }
}
}// namespace simd_compaction
