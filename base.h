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
#include <unordered_map>
#include <variant>
#include <vector>

namespace simd_compaction {

const uint64_t kNumKeys = 204800;
const uint64_t kRHSTuples = 512000;
const uint64_t kRunTimes = 32;
const uint64_t kLanes = 8;
const uint64_t kChunkFactor = 1;

const static __m512i ALL_ZERO = _mm512_set1_epi64(0);
const static __m512i ALL_ONE = _mm512_set1_epi64(1);
const static __m512i ALL_EIGHT = _mm512_set1_epi64(8);

// Some data structures
using std::list;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::unordered_map;
using std::vector;

constexpr size_t kBlockSize = 204800;

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