#include "linear_probing_hash_table.h"

namespace simd_compaction {
simd_compaction::LPHashTable::LPHashTable(size_t n_rhs_tuples, size_t chunk_factor) {
  size_t n_slots_ = 1;
  while (n_slots_ < (n_rhs_tuples << 2)) n_slots_ <<= 1;
  slots_.resize(n_slots_, -1);

  // slot mask
  SCALAR_SLOT_MASK = n_slots_ - 1;
  SIMD_SLOT_MASK = _mm512_set1_epi64(n_slots_ - 1);

  // tuple in hash table
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
    auto slot_id = murmurhash64(value) & SCALAR_SLOT_MASK;

    // find an empty slot
    while (slots_[slot_id] != -1) { slot_id = (slot_id + 1) & SCALAR_SLOT_MASK; }
    slots_[slot_id] = tuple[0];
  }
}

LPScanStructure LPHashTable::Probe(Vector &join_key) {
  vector<uint64_t> slot_ids(kBlockSize);
  vector<uint32_t> slot_sel_vector(kBlockSize);

  CycleProfiler::Get().Start();

  for (size_t i = 0; i < join_key.count_; ++i) {
    auto &key = join_key.GetValue(join_key.selection_vector_[i]);
    auto slot_id = murmurhash64(key) & SCALAR_SLOT_MASK;
    slot_ids[i] = slot_id;
  }

  CycleProfiler::Get().End(0);

  size_t count = 0;
  for (size_t i = 0; i < join_key.count_; ++i) {
    auto idx = slot_ids[i];
    if (slots_[idx] != -1) slot_sel_vector[count++] = i;
  }

  return LPScanStructure(count, slot_ids, slot_sel_vector, slots_);
}

size_t LPScanStructure::Next(Vector &join_key, DataChunk &input, DataChunk &result) {
  vector<uint32_t> result_vector(kBlockSize);
  size_t result_count = 0;

  CycleProfiler::Get().Start();

  // match
  for (size_t i = 0; i < count_; ++i) {
    auto idx = slot_sel_vector_[i];
    auto &l_key = join_key.GetValue(join_key.selection_vector_[idx]);
    auto &r_key = slots_[slot_ids_[idx]];

    // branch prediction
    result_vector[result_count] = idx;
    result_count += (l_key == r_key);
  }

  CycleProfiler::Get().End(1);

  // gather
  result.Slice(input, result_vector, result_count);
  vector<Vector *> cols{&result.data_[input.data_.size()], &result.data_[input.data_.size() + 1]};

  CycleProfiler::Get().Start();

  for (size_t i = 0; i < result_count; ++i) {
    auto idx = result_vector[i];
    auto &r_payload = slots_[slot_ids_[idx]];
    cols[1]->GetValue(i + cols[1]->count_) = r_payload;
  }
  cols[1]->count_ += result_count;

  CycleProfiler::Get().End(2);
  CycleProfiler::Get().Start();

  // advance
  size_t new_count = 0;
  for (size_t i = 0; i < count_; ++i) {
    auto idx = slot_sel_vector_[i];
    auto id = (slot_ids_[idx] + 1) & SCALAR_SLOT_MASK;
    slot_ids_[idx] = id;

    // branch prediction
    slot_sel_vector_[new_count] = idx;
    new_count += (slots_[id] != -1);
  }
  count_ = new_count;

  CycleProfiler::Get().End(3);

  return result_count;
}

size_t LPScanStructure::InOneNext(Vector &join_key, DataChunk &input, DataChunk &result) {
  size_t n_start_tuple = result.count_;
  size_t new_count = 0;

  CycleProfiler::Get().Start();

  vector<Vector *> cols{&result.data_[input.data_.size()], &result.data_[input.data_.size() + 1]};
  for (size_t i = 0; i < count_; ++i) {
    auto idx = slot_sel_vector_[i];
    auto &l_key = join_key.GetValue(join_key.selection_vector_[idx]);
    auto &r_key = slots_[slot_ids_[idx]];

    auto &col = *cols[1];
    col.GetValue(col.count_) = r_key;
    col.count_ += (l_key == r_key);

    auto id = (slot_ids_[idx] + 1) & SCALAR_SLOT_MASK;
    slot_ids_[idx] = id;

    // branch prediction
    slot_sel_vector_[new_count] = idx;
    new_count += (slots_[id] != -1);
  }
  count_ = new_count;

  CycleProfiler::Get().End(1);

  size_t n_end_tuple = cols[1]->count_;
  return n_end_tuple - n_start_tuple;
}

// --------------------------------------------  SIMD  --------------------------------------------
LPScanStructure LPHashTable::SIMDProbe(Vector &join_key) {
  vector<uint64_t> slot_ids(kBlockSize);
  vector<uint32_t> slot_sel_vector(kBlockSize);

  CycleProfiler::Get().Start();

  size_t tail = join_key.count_ & 7;
  for (size_t i = 0; i < join_key.count_ - tail; i += 8) {
    __m256i indices = _mm256_loadu_epi32(join_key.selection_vector_.data() + i);
    auto keys = _mm512_i32gather_epi64(indices, join_key.Data(), 8);
    auto slot_idv = _mm512_and_si512(mm512_murmurhash64(keys), SIMD_SLOT_MASK);
    _mm512_storeu_epi64(slot_ids.data() + i, slot_idv);
  }

  CycleProfiler::Get().End(0);

  for (size_t i = join_key.count_ - tail; i < join_key.count_; ++i) {
    auto &key = join_key.GetValue(join_key.selection_vector_[i]);
    auto slot_id = murmurhash64(key) & SCALAR_SLOT_MASK;
    slot_ids[i] = slot_id;
  }

  size_t count = 0;
  for (size_t i = 0; i < join_key.count_; ++i) {
    auto idx = slot_ids[i];
    if (slots_[idx] != -1) slot_sel_vector[count++] = i;
  }

  return LPScanStructure(count, slot_ids, slot_sel_vector, slots_);
}

size_t LPScanStructure::SIMDNext(Vector &join_key, DataChunk &input, DataChunk &result) {
  vector<uint32_t> result_vector(kBlockSize);
  size_t result_count = 0;
  size_t tail = count_ & 7;

  CycleProfiler::Get().Start();

  // ------------------------------------------------  Match  ------------------------------------------------
  for (size_t i = 0; i < count_ - tail; i += 8) {
    auto indices = _mm256_loadu_epi32(slot_sel_vector_.data() + i);
    auto l_keys = _mm512_i32gather_epi64(indices, join_key.Data(), 8);

    auto slot_ids = _mm512_i32gather_epi64(indices, slot_ids_.data(), 8);
    auto r_keys = _mm512_i64gather_epi64(slot_ids, slots_.data(), 8);

    __mmask8 match = _mm512_cmpeq_epi64_mask(l_keys, r_keys);
    _mm256_mask_compressstoreu_epi32(result_vector.data() + result_count, match, indices);
    result_count += _mm_popcnt_u32(match);
  }

  for (size_t i = count_ - tail; i < count_; i++) {
    auto idx = slot_sel_vector_[i];
    auto &l_key = join_key.GetValue(join_key.selection_vector_[idx]);
    auto &r_key = slots_[slot_ids_[idx]];

    // branch prediction
    result_vector[result_count] = idx;
    result_count += (l_key == r_key);
  }

  CycleProfiler::Get().End(1);

  // ------------------------------------------------  Gather  ------------------------------------------------
  result.Slice(input, result_vector, result_count);
  vector<Vector *> cols{&result.data_[input.data_.size()], &result.data_[input.data_.size() + 1]};

  CycleProfiler::Get().Start();

  tail = result_count & 7;
  for (size_t i = 0; i < result_count - tail; i += 8) {
    auto indices = _mm256_loadu_epi32(result_vector.data() + i);
    auto ids = _mm512_i32gather_epi64(indices, slot_ids_.data(), 8);
    auto r_payload = _mm512_i64gather_epi64(ids, slots_.data(), 8);
    _mm512_storeu_epi64(cols[1]->Data() + i + cols[1]->count_, r_payload);
  }

  for (size_t i = result_count - tail; i < result_count; ++i) {
    auto idx = result_vector[i];
    auto &r_tuple = slots_[slot_ids_[idx]];
    auto &col = *cols[1];
    col.GetValue(i + col.count_) = r_tuple;
  }
  cols[1]->count_ += result_count;

  CycleProfiler::Get().End(2);
  CycleProfiler::Get().Start();

  // ------------------------------------------------  Advance  ------------------------------------------------
  size_t new_count = 0;
  tail = count_ & 7;
  for (size_t i = 0; i < count_ - tail; i += 8) {
    auto indices = _mm256_loadu_epi32(slot_sel_vector_.data() + i);
    auto ids = _mm512_i32gather_epi64(indices, slot_ids_.data(), 8);
    ids = _mm512_and_epi64(SIMD_SLOT_MASK, _mm512_add_epi64(ids, ALL_ONE));
    _mm512_i32scatter_epi64(slot_ids_.data(), indices, ids, 8);

    auto keys = _mm512_i64gather_epi64(ids, slots_.data(), 8);
    __mmask8 valid = _mm512_cmpneq_epi64_mask(keys, ALL_NEG_ONE);
    _mm256_mask_compressstoreu_epi32(slot_sel_vector_.data() + new_count, valid, indices);
    new_count += _mm_popcnt_u32(valid);
  }

  for (size_t i = count_ - tail; i < count_; ++i) {
    auto idx = slot_sel_vector_[i];
    auto id = (slot_ids_[idx] + 1) & SCALAR_SLOT_MASK;
    slot_ids_[idx] = id;

    // branch prediction
    slot_sel_vector_[new_count] = idx;
    new_count += (slots_[id] != -1);
  }
  count_ = new_count;

  CycleProfiler::Get().End(3);

  return result_count;
}

size_t LPScanStructure::SIMDInOneNext(Vector &join_key, DataChunk &input, DataChunk &result) {
  size_t n_start_tuple = result.count_;
  size_t tail = count_ & 7;
  size_t new_count = 0;

  CycleProfiler::Get().Start();
  vector<Vector *> cols{&result.data_[input.data_.size()], &result.data_[input.data_.size() + 1]};
  for (size_t i = 0; i < count_ - tail; i += 8) {
    auto indices = _mm256_loadu_epi32(slot_sel_vector_.data() + i);
    auto l_keys = _mm512_i32gather_epi64(indices, join_key.Data(), 8);

    auto slot_ids = _mm512_i32gather_epi64(indices, slot_ids_.data(), 8);
    auto r_keys = _mm512_i64gather_epi64(slot_ids, slots_.data(), 8);

    // match & gather
    __mmask8 match = _mm512_cmpeq_epi64_mask(l_keys, r_keys);
    auto r_payload = _mm512_i64gather_epi64(slot_ids, slots_.data(), 8);
    _mm512_mask_storeu_epi64(cols[1]->Data() + cols[1]->count_, match, r_payload);
    cols[1]->count_ += _mm_popcnt_u32(match);

    // advance
    auto ids = _mm512_and_epi64(SIMD_SLOT_MASK, _mm512_add_epi64(slot_ids, ALL_ONE));
    _mm512_i32scatter_epi64(slot_ids_.data(), indices, ids, 8);
    auto next_keys = _mm512_i64gather_epi64(ids, slots_.data(), 8);
    __mmask8 valid = _mm512_cmpneq_epi64_mask(next_keys, ALL_NEG_ONE);
    _mm256_mask_compressstoreu_epi32(slot_sel_vector_.data() + new_count, valid, indices);
    new_count += _mm_popcnt_u32(valid);
  }

  for (size_t i = count_ - tail; i < count_; i++) {
    auto idx = slot_sel_vector_[i];
    auto &l_key = join_key.GetValue(join_key.selection_vector_[idx]);
    auto &r_key = slots_[slot_ids_[idx]];

    auto &col = *cols[1];
    col.GetValue(col.count_) = r_key;
    col.count_ += (l_key == r_key);

    auto id = (slot_ids_[idx] + 1) & SCALAR_SLOT_MASK;
    slot_ids_[idx] = id;

    // branch prediction
    slot_sel_vector_[new_count] = idx;
    new_count += (slots_[id] != -1);
  }
  count_ = new_count;
  CycleProfiler::Get().End(1);

  size_t n_end_tuple = cols[1]->count_;
  return n_end_tuple - n_start_tuple;
}
}// namespace simd_compaction