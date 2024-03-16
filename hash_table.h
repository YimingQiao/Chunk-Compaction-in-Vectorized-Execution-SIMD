//===----------------------------------------------------------------------===//
//
//                         SIMD Compaction
//
// hash_table.h
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include <functional>
#include <list>
#include <unordered_map>
#include <utility>

#include "base.h"
#include "gather_functions.h"
#include "hash_functions.h"
#include "profiler.h"
#include "setting.h"

namespace simd_compaction {

class HashTable;

using Tuple = vector<Attribute>;

class ScanStructure {
 public:
  explicit ScanStructure(int64_t count_, vector<uint32_t> &key_sel_vec, vector<uint32_t> &result_vec,
                         vector<uint32_t> &bucket_sel_vector, vector<list<Tuple>::iterator> &iterators,
                         vector<list<Tuple>::iterator> &iterators_end)
      : count_(count_), key_sel_vector_(key_sel_vec), result_vector_(result_vec), bucket_sel_vector_(bucket_sel_vector),
        iterators_(iterators), iterators_end_(iterators_end) {}

  uint32_t Next(Vector &join_key, DataChunk &input, DataChunk &result);

  uint32_t SIMDNext(Vector &join_key, DataChunk &input, DataChunk &result);

  bool HasNext() const { return count_ > 0; }

 private:
  int64_t count_;
  vector<uint32_t> &bucket_sel_vector_;
  vector<uint32_t> key_sel_vector_;
  vector<list<Tuple>::iterator> &iterators_;
  vector<list<Tuple>::iterator> &iterators_end_;
  vector<uint32_t> result_vector_;

  size_t Match(Vector &join_key, vector<uint32_t> &result_vector);

  inline size_t SIMDMatch(Vector &join_key, vector<uint32_t> &result_vector);

  inline void AdvancePointers();

  inline void SIMDAdvancePointers();

  inline void GatherResult(vector<Vector *> cols, vector<uint32_t> &sel_vector, vector<uint32_t> &result_vector,
                           size_t count);

  inline void SIMDGatherResult(vector<Vector *> cols, vector<uint32_t> &sel_vector, vector<uint32_t> &result_vector,
                               int32_t count);
};

class HashTable {
 public:
  size_t n_buckets_;
  vector<unique_ptr<list<Tuple>>> linked_lists_;
  vector<uint64_t> list_sizes_;

  HashTable(size_t n_rhs_tuples, size_t chunk_factor);

  void Probe(Vector &join_key, int64_t count, vector<uint32_t> &sel_vector);

  ScanStructure GetScanStructure();

  void SIMDProbe(Vector &join_key, int64_t count, vector<uint32_t> &sel_vector);

 private:
  uint64_t SCALAR_BUCKET_MASK;
  __m512i BUCKET_MASK;

  // helper vectors
  int64_t count_;
  vector<int64_t> loaded_keys;
  vector<uint64_t> bucket_ids;
  vector<uint32_t> ptrs_sel_vector;
  vector<uint32_t> bucket_sel_vector_;
  vector<list<Tuple>::iterator> iterators_;
  vector<list<Tuple>::iterator> iterators_end_;

  vector<uint32_t> *ref_sel_vector = nullptr;
  vector<uint32_t> result_vector;
};
}// namespace simd_compaction
