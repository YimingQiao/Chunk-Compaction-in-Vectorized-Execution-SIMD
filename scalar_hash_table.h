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
  explicit ScanStructure(size_t count,
                         vector<uint32_t> bucket_sel_vector,
                         vector<list<Tuple> *> buckets,
                         vector<uint32_t> &key_format,
                         HashTable *ht)
      : count_(count), buckets_(std::move(buckets)),
        bucket_sel_vector_(std::move(bucket_sel_vector)), key_sel_vector_(key_format), ht_(ht) {
    iterators_.resize(kBlockSize);
    for (size_t i = 0; i < count; ++i) {
      size_t idx = bucket_sel_vector_[i];
      iterators_[idx] = buckets_[idx]->begin();
    }
  }

  void Next(Vector &join_key, DataChunk &input, DataChunk &result);

  bool HasNext() const { return count_ > 0; }

 private:
  size_t count_;
  vector<list<Tuple> *> buckets_;
  vector<uint32_t> bucket_sel_vector_;
  vector<uint32_t> &key_sel_vector_;
  vector<list<Tuple>::iterator> iterators_;
  HashTable *ht_;

  size_t ScanInnerJoin(Vector &join_key, vector<uint32_t> &result_vector);

  inline void AdvancePointers();

  inline void GatherResult(vector<Vector *> cols, vector<uint32_t> &sel_vector, vector<uint32_t> &result_vector,
                           size_t count);
};

class HashTable {
 public:
  size_t n_buckets_;
  vector<unique_ptr<list<Tuple>>> linked_lists_;

  HashTable(size_t n_rhs_tuples, size_t chunk_factor);

  ScanStructure Probe(Vector &join_key, size_t count, vector<uint32_t> &sel_vector);

 private:
  // Use murmur hash function
  using HashFn = uint64_t (*)(uint64_t);
  HashFn hash_ = &murmurhash64;

  uint64_t bucket_mask_;
};
}// namespace simd_compaction
