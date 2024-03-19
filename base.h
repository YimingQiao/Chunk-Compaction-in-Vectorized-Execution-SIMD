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

constexpr uint64_t kScale = 7;
using std::vector;
using idx_t = size_t;

// work set = left data chunk (block) + right hash table
constexpr size_t kBlockSize = 256 << kScale;
constexpr uint64_t kRHSTuples = 128 << kScale;
constexpr uint64_t kLHSTuples = 1024 << 15;
constexpr uint64_t kHitFreq = 1;

// query setting
static size_t kJoins = 3;
static size_t kLHSTupleSize = 8;
static size_t kRHSTupleSize = 4;
static size_t kChunkFactor = 2;

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

  inline void Reset() { count_ = 0; }

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