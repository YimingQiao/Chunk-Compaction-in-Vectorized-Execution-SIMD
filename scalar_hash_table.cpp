#include "scalar_hash_table.h"

namespace simd_compaction {
HashTable::HashTable(size_t n_rhs_tuples, size_t chunk_factor)
    : loaded_keys(kBlockSize), bucket_ids(kBlockSize), ptrs(kBlockSize), ptrs_sel_vector(kBlockSize),
      bucket_offset(kBlockSize, 0), bucket_sizes(kBlockSize) {
  // number of buckets should be the minimum exp number of 2.
  n_buckets_ = 1;
  while (n_buckets_ < 2 * n_rhs_tuples) n_buckets_ *= 2;

  SCALAR_BUCKET_MASK = n_buckets_ - 1;
  BUCKET_MASK = _mm512_set1_epi64(n_buckets_ - 1);
  linked_lists_.resize(n_buckets_);
  for (auto &bucket : linked_lists_) bucket = std::make_unique<vector<Tuple>>();

  // Tuple in Hash Table
  vector<Tuple> rhs_table(n_rhs_tuples);
  size_t cnt = 0;
  const size_t num_unique = n_rhs_tuples / chunk_factor + (n_rhs_tuples % chunk_factor != 0);
  for (size_t i = 0; i < num_unique; ++i) {
    auto unique_value = i * (n_rhs_tuples / num_unique);
    for (size_t j = 0; j < chunk_factor && cnt < n_rhs_tuples; ++j) {
      auto payload = cnt + 10000000;
      rhs_table[cnt].attrs_.emplace_back(unique_value);
      rhs_table[cnt].attrs_.emplace_back(payload);
      ++cnt;
    }
  }

  // build hash table
  for (size_t i = 0; i < n_rhs_tuples; ++i) {
    auto &tuple = rhs_table[i];
    Attribute value = tuple.attrs_[0];
    auto bucket_idx = murmurhash64(value) % n_buckets_;
    // auto bucket_idx = value % n_buckets_;
    auto &bucket = linked_lists_[bucket_idx];
    bucket->push_back(tuple);
  }

  list_sizes_.resize(n_buckets_);
  for (size_t i = 0; i < n_buckets_; ++i) { list_sizes_[i] = linked_lists_[i]->size(); }
}

ScanStructure HashTable::Probe(Vector &join_key, size_t count, vector<uint32_t> &sel_vector) {
  //  Profiler profiler;
  //  profiler.Start();

  // gather, hash and find buckets
  size_t n_non_empty = 0;
  for (size_t i = 0; i < count; i++) {
    int64_t key = join_key[sel_vector[i]];
    uint64_t hash = murmurhash64(key);
    // uint64_t hash = key;
    ptrs[i] = linked_lists_[hash & SCALAR_BUCKET_MASK].get();
    if (!ptrs[i]->empty()) {
      ptrs_sel_vector[n_non_empty++] = i;
      bucket_sizes[i] = ptrs[i]->size();
    }
  }

  auto ret = ScanStructure(n_non_empty, ptrs_sel_vector, ptrs, bucket_sizes, sel_vector, bucket_offset, this);

  //  double time = profiler.Elapsed();
  //  BeeProfiler::Get().InsertStatRecord("[Join - Probe] 0x" + std::to_string(size_t(this)), time);
  //  ZebraProfiler::Get().InsertRecord("[Join - Probe] 0x" + std::to_string(size_t(this)), count, time);
  return ret;
}

ScanStructure HashTable::SIMDProbe(Vector &join_key, size_t count, vector<uint32_t> &sel_vector) {
  //  Profiler profiler;
  //  profiler.Start();

  // suppose count is the times of 8
  size_t n_non_empty = 0;
  for (uint64_t i = 0; i < count; i += 8) {
    __m256i indices = _mm256_loadu_epi32(sel_vector.data() + i);
    __m512i gathered_values = _mm512_i32gather_epi64(indices, join_key.data_->data(), 8);
    __m512i hashes = mm512_murmurhash64(gathered_values);
    // __m512i hashes = gathered_values;
    __m512i bucket_indices = _mm512_and_si512(hashes, BUCKET_MASK);
    __m512i gathered_buckets = _mm512_i64gather_epi64(bucket_indices, linked_lists_.data(), 8);
    _mm512_storeu_epi64(ptrs.data() + i, gathered_buckets);

    __m512i sizes = _mm512_i64gather_epi64(bucket_indices, list_sizes_.data(), 8);
    __mmask8 not_zero = _mm512_cmpneq_epi64_mask(sizes, _mm512_set1_epi64(0));
    _mm256_mask_compressstoreu_epi32(ptrs_sel_vector.data() + n_non_empty, not_zero, indices);
    n_non_empty += _mm_popcnt_u32(not_zero);
  }

  auto ret = ScanStructure(n_non_empty, ptrs_sel_vector, ptrs, bucket_sizes, sel_vector, bucket_offset, this);
  //
  //  double time = profiler.Elapsed();
  //  BeeProfiler::Get().InsertStatRecord("[Join - Probe] 0x" + std::to_string(size_t(this)), time);
  //  ZebraProfiler::Get().InsertRecord("[Join - Probe] 0x" + std::to_string(size_t(this)), count, time);
  return ret;
}

void ScanStructure::Next(Vector &join_key, DataChunk &input, DataChunk &result) {
  if (count_ == 0) {
    // no pointers left to chase
    return;
  }

  //  Profiler profiler;
  //  profiler.Start();

  vector<uint32_t> result_vector(kBlockSize);
  size_t result_count = ScanInnerJoin(join_key, result_vector);

  if (result_count > 0) {
    // matches were found
    // construct the result
    // on the LHS, we create a slice using the result vector
    result.Slice(input, result_vector, result_count);

    // on the RHS, we need to fetch the data from the hash table
    vector<Vector *> cols{&result.data_[input.data_.size()], &result.data_[input.data_.size() + 1]};
    GatherResult(cols, input.selection_vector_, result_vector, result_count);
  }
  AdvancePointers();

  //  double time = profiler.Elapsed();
  //  BeeProfiler::Get().InsertStatRecord("[Join - Next] 0x" + std::to_string(size_t(ht_)), time);
  //  ZebraProfiler::Get().InsertRecord("[Join - Next] 0x" + std::to_string(size_t(ht_)), input.count_, time);
}

void ScanStructure::SIMDNext(Vector &join_key, DataChunk &input, DataChunk &result) {
  if (count_ == 0) return;

  //  Profiler profiler;
  //  profiler.Start();

  size_t tail = count_ & 7;
  vector<uint32_t> result_vector(kBlockSize);
  size_t result_count = 0;
  while (true) {
    for (size_t i = 0; i < count_ - tail; i += 8) {
      __m256i indices = _mm256_loadu_epi32(bucket_sel_vector_.data() + i);
      __m512i l_keys = _mm512_i32gather_epi64(indices, join_key.data_->data(), 8);
      __m512i buckets = _mm512_i32gather_epi64(indices, buckets_.data(), 8);
      __m512i offsets = _mm512_i32gather_epi64(indices, offsets_.data(), 8);

      __m512i r_keys;
      for (int j = 0; j < 8; j++) {
        auto *bucket = reinterpret_cast<vector<Tuple> *>(buckets[j]);
        r_keys[j] = (*bucket)[offsets[j]].attrs_[0];
      }

      __mmask8 match = _mm512_cmpeq_epi64_mask(l_keys, r_keys);
      _mm256_mask_compressstoreu_epi32(result_vector.data() + result_count, match, indices);
      result_count += _mm_popcnt_u32(match);
    }

    if (tail != 0) {
      size_t index = count_ - tail;
      for (size_t i = index; i < count_; ++i) {
        size_t idx = bucket_sel_vector_[i];
        auto &l_key = join_key.GetValue(key_sel_vector_[idx]);
        uint32_t offset = offsets_[idx];
        auto &r_key = (*buckets_[idx])[offset].attrs_[0];
        if (l_key == r_key) result_vector[result_count++] = idx;
      }
    }

    if (result_count > 0) break;

    // no matches found: check the next set of pointers
    SIMDAdvancePointers();
    if (count_ == 0) break;
  }

  if (result_count > 0) {
    // matches were found
    // construct the result
    // on the LHS, we create a slice using the result vector
    result.SIMDSlice(input, result_vector, result_count);

    // on the RHS, we need to fetch the data from the hash table
    vector<Vector *> cols{&result.data_[input.data_.size()], &result.data_[input.data_.size() + 1]};
    GatherResult(cols, input.selection_vector_, result_vector, result_count);
  }
  SIMDAdvancePointers();

  //  double time = profiler.Elapsed();
  //  BeeProfiler::Get().InsertStatRecord("[Join - Next] 0x" + std::to_string(size_t(ht_)), time);
  //  ZebraProfiler::Get().InsertRecord("[Join - Next] 0x" + std::to_string(size_t(ht_)), input.count_, time);
}

size_t ScanStructure::ScanInnerJoin(Vector &join_key, vector<uint32_t> &result_vector) {
  while (true) {
    // Match
    size_t result_count = 0;
    for (size_t i = 0; i < count_; ++i) {
      size_t idx = bucket_sel_vector_[i];
      auto &l_key = join_key.GetValue(key_sel_vector_[idx]);
      uint32_t offset = offsets_[idx];
      auto &r_key = (*buckets_[idx])[offset].attrs_[0];
      if (l_key == r_key) result_vector[result_count++] = idx;
    }

    if (result_count > 0) return result_count;

    // no matches found: check the next set of pointers
    AdvancePointers();
    if (count_ == 0) return 0;
  }
}

void ScanStructure::AdvancePointers() {
  size_t new_count = 0;
  for (size_t i = 0; i < count_; i++) {
    auto idx = bucket_sel_vector_[i];
    if (++offsets_[idx] != buckets_[idx]->size()) { bucket_sel_vector_[new_count++] = idx; }
  }
  count_ = new_count;
}

void ScanStructure::SIMDAdvancePointers() {
  int32_t tail = count_ & 7;
  size_t new_count = 0;
  for (size_t i = 0; i < count_ - tail; i += 8) {
    __m256i indices = _mm256_loadu_epi32(bucket_sel_vector_.data() + i);
    auto offsets = _mm512_i32gather_epi64(indices, offsets_.data(), 8);
    offsets = _mm512_add_epi64(offsets, _mm512_set1_epi64(1));
    _mm512_i32scatter_epi64(offsets_.data(), indices, offsets, 8);
    auto bucket_sizes = _mm512_i32gather_epi64(indices, bucket_sizes_.data(), 8);

    __mmask8 match = _mm512_cmplt_epi64_mask(offsets, bucket_sizes);
    _mm256_mask_compressstoreu_epi32(bucket_sel_vector_.data() + new_count, match, indices);
    new_count += _mm_popcnt_u32(match);
  }

  if (tail) {
    for (size_t i = count_ - tail; i < count_; i++) {
      auto idx = bucket_sel_vector_[i];
      if (++offsets_[idx] != buckets_[idx]->size()) { bucket_sel_vector_[new_count++] = idx; }
    }
  }

  count_ = new_count;
}

void ScanStructure::GatherResult(vector<Vector *> cols, vector<uint32_t> &sel_vector, vector<uint32_t> &result_vector,
                                 size_t count) {
  for (size_t i = 0; i < count; ++i) {
    auto idx = result_vector[i];
    uint32_t offset = offsets_[idx];
    auto &tuple = (*buckets_[idx])[offset];

    // columns from the right table align with the selection vector given by the left table
    size_t pos = sel_vector[idx];
    for (size_t j = 0; j < cols.size(); ++j) {
      auto &col = *cols[j];
      col.GetValue(pos) = tuple.attrs_[j];
    }
  }
}

void ScanStructure::SIMDGatherResult(vector<Vector *> cols, vector<uint32_t> &sel_vector,
                                     vector<uint32_t> &result_vector, size_t count) {
  for (size_t j = 0; j < cols.size(); ++j) {
    auto &col = *cols[j];

    uint32_t tail = count & 7;
    for (size_t i = 0; i < count - tail; i += 8) {
      __m256i ids = _mm256_loadu_epi32(result_vector.data() + i);
      __m512i offsets = _mm512_i32gather_epi64(ids, offsets_.data(), 8);
      __m512i buckets = _mm512_i32gather_epi64(ids, buckets_.data(), 8);
      __m256i positions = _mm256_i32gather_epi32(reinterpret_cast<int *>(sel_vector.data()), ids, 8);

      __m512i keys;
      for (int k = 0; k < 8; k++) {
        auto *bucket = reinterpret_cast<vector<Tuple> *>(buckets[k]);
        keys[k] = (*bucket)[offsets[k]].attrs_[j];
      }

      _mm512_i32scatter_epi64(col.data_->data(), positions, keys, 8);
    }

    if (tail) {
      for (size_t i = count - tail; i < count; ++i) {
        auto idx = result_vector[i];
        uint32_t offset = offsets_[idx];
        auto &tuple = (*buckets_[idx])[offset];
        size_t pos = sel_vector[idx];
        col.GetValue(pos) = tuple.attrs_[j];
      }
    }
  }
}
}// namespace simd_compaction
