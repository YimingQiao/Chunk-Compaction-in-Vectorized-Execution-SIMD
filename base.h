//===----------------------------------------------------------------------===//
//
//                         SIMD Compaction
//
// base.h
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cassert>
#include <immintrin.h>
#include <iostream>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace simd_compaction {
const static __m512i ALL_NEG_ONE = _mm512_set1_epi64(-1);
const static __m512i ALL_ZERO = _mm512_set1_epi64(0);
const static __m512i ALL_ONE = _mm512_set1_epi64(1);
const static __m512i ALL_EIGHT = _mm512_set1_epi64(8);
const static __m512i ALL_SIXTEEN = _mm512_set1_epi64(16);

// Some data structures
using std::list;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::unordered_map;
using std::vector;

inline size_t kScale = 0;
using std::vector;
using idx_t = size_t;

// work set = left data chunk (block) + right hash table
inline size_t kBlockSize = 256 << kScale;
inline size_t kRHSTuples = 128 << kScale;
inline size_t kLHSTuples = 1024 << 17;
inline size_t kHitFreq = 1;

// query setting
inline size_t kJoins = 3;
inline size_t kLHSTupleSize = 2e7;
inline size_t kRHSTupleSize = 2e6;
inline size_t kChunkFactor = 1;

// Attribute includes three types: integer, float-point number, and the string.
using Attribute = int64_t;

enum class AttributeType : uint8_t { INTEGER = 0, INVALID = 3 };

// The vector uses Row ID.
class Vector {
 public:
  AttributeType type_;

  shared_ptr<vector<Attribute>> data_;

  Vector() : type_(AttributeType::INTEGER), data_(std::make_shared<vector<Attribute>>(kBlockSize)) {}

  explicit Vector(AttributeType type) : type_(type), data_(std::make_shared<vector<Attribute>>(kBlockSize)) {}

  inline void Reference(Vector &other);

  Attribute &GetValue(size_t idx) { return (*data_)[idx]; }

  inline auto *Data() { return data_->data(); }

  Attribute &operator[](size_t idx) { return (*data_)[idx]; }
};

// A data chunk has some columns.
class DataChunk {
 public:
  size_t count_;
  vector<Vector> data_;
  vector<AttributeType> types_;
  vector<uint32_t> selection_vector_;

  explicit DataChunk(const vector<AttributeType> &types);

  void Append(DataChunk &chunk, size_t num, size_t offset = 0);

  void AppendTuple(vector<Attribute> &tuple);

  void Slice(DataChunk &other, vector<uint32_t> &selection_vector, size_t count);

  void SIMDSlice(DataChunk &other, vector<uint32_t> &selection_vector, size_t count);

  void Reset() {
    count_ = 0;
    for (size_t i = 0; i < kBlockSize; ++i) selection_vector_[i] = i;
  };
};
}// namespace simd_compaction