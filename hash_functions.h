#pragma once

#include <cstdint>
#include <immintrin.h>

namespace simd_compaction {

#define BIG_CONSTANT(x) (x##LLU)
/*-----------------------------------------------------------------------------
// MurmurHash2, 64-bit versions, by Austin Appleby
//
// The same caveats as 32-bit MurmurHash2 apply here - beware of alignment
// and endian-ness issues if used across multiple platforms.
//
// 64-bit hash for 64-bit platforms
*/
inline uint64_t MurmurHash64A(const void *key, int len, uint64_t seed) {
  const uint64_t m = BIG_CONSTANT(0xc6a4a7935bd1e995);
  const int r = 47;

  uint64_t h = seed ^ (len * m);

  const uint64_t *data = (const uint64_t *) key;
  const uint64_t *end = data + (len / 8);

  while (data != end) {
    uint64_t k = *data++;

    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;
  }

  const unsigned char *data2 = (const unsigned char *) data;

  switch (len & 7) {
    case 7: h ^= ((uint64_t) data2[6]) << 48;
    case 6: h ^= ((uint64_t) data2[5]) << 40;
    case 5: h ^= ((uint64_t) data2[4]) << 32;
    case 4: h ^= ((uint64_t) data2[3]) << 24;
    case 3: h ^= ((uint64_t) data2[2]) << 16;
    case 2: h ^= ((uint64_t) data2[1]) << 8;
    case 1: h ^= ((uint64_t) data2[0]); h *= m;
  };

  h ^= h >> r;
  h *= m;
  h ^= h >> r;

  return h;
}

inline uint64_t murmurhash2(uint64_t x) { return MurmurHash64A(&x, 8, 0); }

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
