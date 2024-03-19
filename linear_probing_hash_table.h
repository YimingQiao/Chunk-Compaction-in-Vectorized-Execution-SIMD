//===----------------------------------------------------------------------===//
//
//                         SIMD Compaction
//
// linear_probing_hash_table
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include <list>
#include <unordered_map>
#include <utility>

#include "base.h"
#include "hash_functions.h"
#include "profiler.h"

namespace simd_compaction {
using Tuple = vector<int64_t>;
using Key = int64_t;

class LPScanStructure {
 public:
  LPScanStructure(size_t count, vector<uint64_t> slot_ids, vector<uint32_t> slot_sel_vector, vector<Key> &slots)
      : count_(count), slot_ids_(std::move(slot_ids)), slot_sel_vector_(std::move(slot_sel_vector)), slots_(slots) {
    SCALAR_SLOT_MASK = slots_.size() - 1;
    SIMD_SLOT_MASK = SIMD_SLOT_MASK = _mm512_set1_epi64(slots_.size() - 1);
  }

  size_t Next(Vector &join_key, DataChunk &input, DataChunk &result);

  size_t InOneNext(Vector &join_key, DataChunk &input, DataChunk &result);

  size_t SIMDNext(Vector &join_key, DataChunk &input, DataChunk &result);

  size_t SIMDInOneNext(Vector &join_key, DataChunk &input, DataChunk &result);

  inline bool HasNext() const { return count_ > 0; }

 private:
  size_t count_;
  vector<uint64_t> slot_ids_;
  vector<uint32_t> slot_sel_vector_;
  vector<Key> &slots_;

  // we use & mask to replace % n.
  uint64_t SCALAR_SLOT_MASK;
  __m512i SIMD_SLOT_MASK;
};

class LPHashTable {
 public:
  LPHashTable(size_t n_rhs_tuples, size_t chunk_factor);

  LPScanStructure Probe(Vector &join_key);

  LPScanStructure SIMDProbe(Vector &join_key);

 private:
  // All empty slots have the value -1.
  vector<Key> slots_;

  // we use & mask to replace % n.
  uint64_t SCALAR_SLOT_MASK;
  __m512i SIMD_SLOT_MASK;
};
}// namespace simd_compaction
