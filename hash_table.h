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

struct Tuple {
  vector<Attribute> attrs_;
};

class ScanStructure {
 public:
  explicit ScanStructure(size_t count, vector<uint32_t> &bucket_sel_vec, vector<list<Tuple> *> &buckets,
                         vector<uint32_t> &key_sel_vec, HashTable *ht, vector<uint32_t> &result_vec)
      : count_(count), buckets_(buckets), bucket_sel_vector_(bucket_sel_vec), key_sel_vector_(key_sel_vec), ht_(ht),
        result_vector_(result_vec) {
    iterators_.resize(kBlockSize);
    iterators_end_.resize(kBlockSize);
    for (size_t i = 0; i < count; ++i) {
      size_t idx = bucket_sel_vector_[i];
      iterators_[idx] = buckets_[idx]->begin();
      iterators_end_[idx] = buckets_[idx]->end();
    }
  }

  void Next(Vector &join_key, DataChunk &input, DataChunk &result);

  void SIMDNext(Vector &join_key, DataChunk &input, DataChunk &result);

  bool HasNext() const { return count_ > 0; }

 private:
  int64_t count_;
  vector<list<Tuple> *> buckets_;
  vector<uint32_t> bucket_sel_vector_;
  vector<uint32_t> &key_sel_vector_;
  vector<list<Tuple>::iterator> iterators_;
  vector<list<Tuple>::iterator> iterators_end_;
  HashTable *ht_;
  vector<uint32_t> result_vector_;

  size_t ScanInnerJoin(Vector &join_key, vector<uint32_t> &result_vector);

  inline size_t SIMDScanInnerJoin(Vector &join_key, vector<uint32_t> &result_vector);

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

  void Probe(Vector &join_key, size_t count, vector<uint32_t> &sel_vector);

  ScanStructure GetScanStructure() {
    return ScanStructure(n_valid, ptrs_sel_vector, ptrs, *ref_sel_vector, this, result_vector);
  }

  void SIMDProbe(Vector &join_key, size_t count, vector<uint32_t> &sel_vector);

 private:
  uint64_t SCALAR_BUCKET_MASK;
  __m512i BUCKET_MASK;

  // helper vectors
  vector<int64_t> loaded_keys;
  vector<uint64_t> bucket_ids;
  vector<list<Tuple> *> ptrs;
  vector<uint32_t> ptrs_sel_vector;

  uint32_t n_valid = 0;
  vector<uint32_t> *ref_sel_vector = nullptr;
  vector<uint32_t> result_vector;
};
}// namespace simd_compaction
