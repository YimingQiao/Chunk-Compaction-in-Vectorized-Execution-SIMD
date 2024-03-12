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

namespace simd_compaction {

class HashTable;

struct Tuple {
  vector<Attribute> attrs_;
};

class ScanStructure {
 public:
  explicit ScanStructure(size_t count, vector<uint32_t> &bucket_sel_vector, vector<vector<Tuple> *> &buckets,
                         vector<uint64_t> &bucket_sizes, vector<uint32_t> &key_format, vector<uint64_t> &offsets,
                         HashTable *ht)
      : count_(count), buckets_(buckets), bucket_sizes_(bucket_sizes), bucket_sel_vector_(bucket_sel_vector),
        key_sel_vector_(key_format), offsets_(offsets), ht_(ht) {
    for (unsigned long &offset : offsets) offset = 0;
  }

  void Next(Vector &join_key, DataChunk &input, DataChunk &result);

  void SIMDNext(Vector &join_key, DataChunk &input, DataChunk &result);

  bool HasNext() const { return count_ > 0; }

 private:
  size_t count_;
  vector<vector<Tuple> *> buckets_;
  vector<uint64_t> bucket_sizes_;
  vector<uint32_t> bucket_sel_vector_;
  vector<uint32_t> &key_sel_vector_;
  vector<uint64_t> offsets_;
  HashTable *ht_;

  size_t ScanInnerJoin(Vector &join_key, vector<uint32_t> &result_vector);

  inline void AdvancePointers();

  inline void SIMDAdvancePointers();

  inline void GatherResult(vector<Vector *> cols, vector<uint32_t> &sel_vector, vector<uint32_t> &result_vector,
                           size_t count);

  inline void SIMDGatherResult(vector<Vector *> cols, vector<uint32_t> &sel_vector, vector<uint32_t> &result_vector,
                           size_t count);
};

class HashTable {
 public:
  size_t n_buckets_;
  vector<unique_ptr<vector<Tuple>>> linked_lists_;
  vector<uint64_t> list_sizes_;

  HashTable(size_t n_rhs_tuples, size_t chunk_factor);

  ScanStructure Probe(Vector &join_key, size_t count, vector<uint32_t> &sel_vector);

  ScanStructure SIMDProbe(Vector &join_key, size_t count, vector<uint32_t> &sel_vector);

 private:
  uint64_t SCALAR_BUCKET_MASK;
  __m512i BUCKET_MASK;

  // helper vectors
  vector<int64_t> loaded_keys;
  vector<uint64_t> bucket_ids;
  vector<vector<Tuple> *> ptrs;
  vector<uint32_t> ptrs_sel_vector;
  vector<uint64_t> bucket_offset;
  vector<uint64_t> bucket_sizes;
};
}// namespace simd_compaction
