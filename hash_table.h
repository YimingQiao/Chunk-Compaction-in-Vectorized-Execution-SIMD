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
  explicit ScanStructure(size_t count, vector<uint32_t> bucket_sel_vector, vector<list<Tuple> *> buckets,
                         vector<uint32_t> &key_sel_vector)
      : count_(count), bucket_sel_vector_(std::move(bucket_sel_vector)), buckets_(std::move(buckets)),
        key_sel_vector_(key_sel_vector) {
    iterators_.resize(kBlockSize);
    iterator_ends_.resize(kBlockSize);
    for (size_t i = 0; i < count; ++i) {
      auto idx = bucket_sel_vector_[i];
      iterators_[idx] = buckets_[idx]->begin();
      iterator_ends_[idx] = buckets_[idx]->end();
    }
  };

  uint32_t Next(Vector &join_key, DataChunk &input, DataChunk &result);

  bool HasNext() const { return count_ > 0; }

 private:
  size_t count_;
  vector<list<Tuple> *> buckets_;
  vector<uint32_t> bucket_sel_vector_;
  vector<uint32_t> key_sel_vector_;
  vector<list<Tuple>::iterator> iterators_;
  vector<list<Tuple>::iterator> iterator_ends_;

  size_t Match(Vector &join_key, vector<uint32_t> &result_vector);

  inline void AdvancePointers();

  inline void GatherResult(vector<Vector *> cols, vector<uint32_t> &sel_vector, vector<uint32_t> &result_vector,
                           size_t count);

  // --------------------------------------------  SIMD  --------------------------------------------

 public:
  uint32_t SIMDNext(Vector &join_key, DataChunk &input, DataChunk &result);

 private:
  inline size_t SIMDMatch(Vector &join_key, vector<uint32_t> &result_vector);

  inline void SIMDAdvancePointers();

  inline void SIMDGatherResult(vector<Vector *> cols, vector<uint32_t> &sel_vector, vector<uint32_t> &result_vector,
                               int32_t count);
};

class HashTable {
 public:

  HashTable(size_t n_rhs_tuples, size_t chunk_factor);

  ScanStructure Probe(Vector &join_key, int64_t count, vector<uint32_t> &sel_vector);

  ScanStructure SIMDProbe(Vector &join_key, int64_t count, vector<uint32_t> &sel_vector);

 private:
  size_t n_buckets_;
  vector<unique_ptr<list<Tuple>>> linked_lists_;

  // we use & mask to replace % n.
  uint64_t SCALAR_BUCKET_MASK;
  __m512i SIMD_BUCKET_MASK;

  // helper vectors
  vector<uint64_t> bucket_ids;
};
}// namespace simd_compaction
