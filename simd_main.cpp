#include <cstdint>
#include <immintrin.h>
#include <iostream>
#include <variant>
#include <vector>
#include <x86intrin.h>

#include "profiler.h"

// using Attribute = std::variant<int64_t, double, char[24]>;
using Attribute = int64_t;

bool IsAligned(void *ptr, std::size_t byte_align) {
  return reinterpret_cast<uintptr_t>(ptr) % byte_align == 0;
}

inline __m512i mm512_murmurhash64(__m512i x) {
  __m512i mult = _mm512_set1_epi64(0xd6e8feb86659fd93U);
  x = _mm512_xor_si512(x, _mm512_srli_epi64(x, 32));
  x = _mm512_mullo_epi64(x, mult);
  x = _mm512_xor_si512(x, _mm512_srli_epi64(x, 32));
  x = _mm512_mullo_epi64(x, mult);
  x = _mm512_xor_si512(x, _mm512_srli_epi64(x, 32));
  return x;
}

inline uint64_t murmurhash64(uint64_t x) {
  x ^= x >> 32;
  x *= 0xd6e8feb86659fd93U;
  x ^= x >> 32;
  x *= 0xd6e8feb86659fd93U;
  x ^= x >> 32;
  return x;
}

const uint64_t kNumKeys = 1024;
const uint64_t kNumBuckets = (1 << 21);

int main(int argc, char *argv[]) {
  // use avx512 gather to gather effective values
  std::vector<Attribute> keys(kNumKeys);
  for (uint64_t i = 0; i < kNumKeys; ++i) keys[i] = int64_t(i);

  std::vector<uint32_t> sel_vector(kNumKeys);
  for (uint64_t i = 0; i < kNumKeys; ++i) sel_vector[i] = uint32_t(i);
  std::cout << "Sel Vector is aligned: " << IsAligned(sel_vector.data(), 64) << "\n";
  std::cout << "Sizeof(key): " << sizeof(keys[0]) << "\n";
  std::cout << "Workload Size: " << 8 * kNumKeys / 1024 << " KB\n";

  compaction::Profiler profiler;
  // -----------------------------------  SIMD  -----------------------------------
  std::cout << "--------------- SIMD ---------------\n";
  {
    std::vector<uint64_t> simd_buckets(kNumKeys);
    const uint64_t kLanes = 8;
    uint64_t start_cycles, end_cycles;

    profiler.Start();
    start_cycles = __rdtsc();

    for (uint64_t i = 0; i < kNumKeys; i += kLanes) {
      __m256i indices = _mm256_loadu_epi32(sel_vector.data() + i);
      __m512i gathered_values = _mm512_i32gather_epi64(indices, keys.data(), 8);
      __m512i hashes = mm512_murmurhash64(gathered_values);
      __m512i bucket_indices = _mm512_and_si512(hashes, _mm512_set1_epi64(kNumBuckets - 1));
      _mm512_storeu_epi64(simd_buckets.data() + i, bucket_indices);
    }

    end_cycles = __rdtsc();
    double time = profiler.Elapsed();

    uint64_t total_cycles = end_cycles - start_cycles;
    double cycles_per_tuple = static_cast<double>(total_cycles) / static_cast<double>(kNumKeys);
    std::cout << "SIMD Probing Cycles per tuple: " << cycles_per_tuple << "\n";
    std::cout << "SIMD Probing Time: " << time << "\n";
  }

  {
    std::vector<uint64_t> simd_hashes(kNumKeys);
    const uint64_t kLanes = 8;

    uint64_t start_cycles, end_cycles;

    profiler.Start();
    start_cycles = __rdtsc();

    for (uint64_t i = 0; i < kNumKeys; i += kLanes) {
      __m512i loaded_keys = _mm512_loadu_epi64(keys.data() + i);
      __m512i hashes = mm512_murmurhash64(loaded_keys);
      _mm512_storeu_epi64(simd_hashes.data() + i, hashes);
    }

    end_cycles = __rdtsc();
    double time = profiler.Elapsed();

    uint64_t total_cycles = end_cycles - start_cycles;
    double cycles_per_tuple = static_cast<double>(total_cycles) / static_cast<double>(kNumKeys);
    std::cout << "SIMD Hash Cycles per tuple: " << cycles_per_tuple << "\n";
    std::cout << "SIMD Hash Time: " << time << "\n";
  }

  // -----------------------------------  Scalar  -----------------------------------
  std::cout << "--------------- Scalar ---------------\n";
  {
    std::vector<uint64_t> scalar_buckets(kNumKeys);
    uint64_t start_cycles, end_cycles;


    profiler.Start();
    start_cycles = __rdtsc();

    for (uint64_t i = 0; i < kNumKeys; ++i) {
      int64_t key = keys[sel_vector[i]];
      uint64_t hash = murmurhash64(key);
      uint64_t bucket_index = hash & (kNumBuckets - 1);
      scalar_buckets[i] = bucket_index;
    }

    end_cycles = __rdtsc();
    double time = profiler.Elapsed();

    uint64_t total_cycles = end_cycles - start_cycles;
    double cycles_per_tuple = static_cast<double>(total_cycles) / static_cast<double>(kNumKeys);
    std::cout << "Scalar Probing Cycles per tuple: " << cycles_per_tuple << "\n";
    std::cout << "Scalar Probing Time: " << time << "\n";
  }

  {
    std::vector<uint64_t> scalar_hashes(kNumKeys);
    uint64_t start_cycles, end_cycles;

    profiler.Start();
    start_cycles = __rdtsc();

    for (uint64_t i = 0; i < kNumKeys; ++i) {
      uint64_t loaded_key = keys[i];
      uint64_t hash = murmurhash64(loaded_key);
      scalar_hashes[i] = hash;
    }

    end_cycles = __rdtsc();
    double time = profiler.Elapsed();

    uint64_t total_cycles = end_cycles - start_cycles;
    double cycles_per_tuple = static_cast<double>(total_cycles) / static_cast<double>(kNumKeys);
    std::cout << "Scalar Hash Cycles per tuple: " << cycles_per_tuple << "\n";
    std::cout << "Scalar Hash Time: " << time << "\n";
  }
}
