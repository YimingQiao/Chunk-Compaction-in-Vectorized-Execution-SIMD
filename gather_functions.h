#pragma once

#include <immintrin.h>

#include "setting.h"

namespace simd_compaction {

// It uses AVX2 gather if inlined, otherwise use scalar gather.
inline std::vector<int64_t> &ScalarGather(const vector<Attribute> &keys, const vector<uint32_t> &sel_vector, vector<int64_t> &loaded_keys) {
  for (uint64_t i = 0; i < kLHSTuples; i += 8) {
    int64_t key0 = keys[sel_vector[i + 0]];
    loaded_keys[i + 0] = key0;

    int64_t key1 = keys[sel_vector[i + 1]];
    loaded_keys[i + 1] = key1;

    int64_t key2 = keys[sel_vector[i + 2]];
    loaded_keys[i + 2] = key2;

    int64_t key3 = keys[sel_vector[i + 3]];
    loaded_keys[i + 3] = key3;

    int64_t key4 = keys[sel_vector[i + 4]];
    loaded_keys[i + 4] = key4;

    int64_t key5 = keys[sel_vector[i + 5]];
    loaded_keys[i + 5] = key5;

    int64_t key6 = keys[sel_vector[i + 6]];
    loaded_keys[i + 6] = key6;

    int64_t key7 = keys[sel_vector[i + 7]];
    loaded_keys[i + 7] = key7;
  }
  return loaded_keys;
}

inline std::vector<int64_t> &SIMDGather(const vector<Attribute> &keys, const vector<uint32_t> &sel_vector, vector<int64_t> &loaded_keys, uint64_t n_keys) {
  auto *keys_data = keys.data();
  uint64_t i;
  uint64_t n_full_chunks = n_keys / 8;// number of full chunks that fit into data

  // process full chunks using SIMD
  for (i = 0; i < n_full_chunks * 8; i += 8) {
    __m256i indices = _mm256_loadu_si256((__m256i_u *) (sel_vector.data() + i));
    __m512i gathered_values = _mm512_i32gather_epi64(indices, (void *) keys_data, 8);
    _mm512_storeu_si512((__m512i *) (loaded_keys.data() + i), gathered_values);
  }

  // Scalar cleanup loop
  for (; i < n_keys; ++i) {
    uint32_t index = sel_vector[i];
    int64_t value = keys_data[index];
    loaded_keys[i] = value;
  }

  return loaded_keys;
}

}// namespace simd_compaction