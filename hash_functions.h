#pragma once

#include <cstdint>
#include <immintrin.h>

namespace simd_compaction {

inline uint64_t murmurhash64(uint64_t x) {
  x ^= x >> 32;
  x *= 0xd6e8feb86659fd93U;
  x ^= x >> 32;
  x *= 0xd6e8feb86659fd93U;
  x ^= x >> 32;

  return x;
}


inline __m512i mm512_murmurhash64(__m512i x) {
  __m512i MAGIC_NUMBER = _mm512_set1_epi64(0xd6e8feb86659fd93U);

  x = _mm512_xor_si512(x, _mm512_srli_epi64(x, 32));
  x = _mm512_mullo_epi64(x, MAGIC_NUMBER);
  x = _mm512_xor_si512(x, _mm512_srli_epi64(x, 32));
  x = _mm512_mullo_epi64(x, MAGIC_NUMBER);
  x = _mm512_xor_si512(x, _mm512_srli_epi64(x, 32));

  return x;
}
}// namespace simd_compaction
