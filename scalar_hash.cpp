#include <cstdint>
#include <immintrin.h>
#include <iostream>
#include <variant>
#include <vector>
#include <x86intrin.h>

#include "profiler.h"
#include "scalar_gather.h"

inline uint64_t murmurhash64(uint64_t x) {
  x ^= x >> 32;
  x *= 0xd6e8feb86659fd93U;
  x ^= x >> 32;
  x *= 0xd6e8feb86659fd93U;
  x ^= x >> 32;

  return x;
}

void PrintCacheSizes() {
  for (int i = 0; i < 3; ++i) {
    std::string path = "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(i) + "/size";

    std::ifstream file(path);
    if (!file.is_open()) {
      std::cerr << "Failed to open file: " << path << '\n';
      return;
    }

    std::string size;
    std::getline(file, size);
    std::cout << "L" << (i + 1) << " cache size: " << size << '\n';
  }
}

int main(int argc, char *argv[]) {
  PrintCacheSizes();

  // use avx512 gather to gather effective values
  std::vector<Attribute> keys(kNumKeys);
  for (uint64_t i = 0; i < kNumKeys; ++i) keys[i] = int64_t(i);

  std::vector<uint32_t> sel_vector(kNumKeys);
  for (uint64_t i = 0; i < kNumKeys; ++i) sel_vector[i] = uint32_t(murmurhash64(i) % kNumKeys);
  std::cout << "Sizeof(key): " << sizeof(keys[0]) << "\n";
  std::cout << "Workload Size: " << 8 * kNumKeys / 1024 << " KB\n";

  std::cout << "--------------- Scalar ---------------\n";
  {
    std::vector<uint64_t> scalar_buckets(kNumKeys, 0);
    std::vector<int64_t> loaded_keys(kNumKeys, 0);
    const uint64_t run_times = 65536;
    uint64_t start_cycles, end_cycles;

    start_cycles = __rdtsc();

    for (uint32_t j = 0; j < run_times; j++) {
      loaded_keys = GetVector(keys, sel_vector, loaded_keys);

      for (uint64_t i = 0; i < kNumKeys; i += 8) {
        int64_t key0 = loaded_keys[i];
        uint64_t hash0 = murmurhash64(key0);
        scalar_buckets[i] = hash0 & (kNumBuckets - 1) + j;

        int64_t key1 = loaded_keys[i + 1];
        uint64_t hash1 = murmurhash64(key1);
        scalar_buckets[i + 1] = hash1 & (kNumBuckets - 1) + j;

        int64_t key2 = loaded_keys[i + 2];
        uint64_t hash2 = murmurhash64(key2);
        scalar_buckets[i + 2] = hash2 & (kNumBuckets - 1) + j;

        int64_t key3 = loaded_keys[i + 3];
        uint64_t hash3 = murmurhash64(key3);
        scalar_buckets[i + 3] = hash3 & (kNumBuckets - 1) + j;

        int64_t key4 = loaded_keys[i + 4];
        uint64_t hash4 = murmurhash64(key4);
        scalar_buckets[i + 4] = hash4 & (kNumBuckets - 1) + j;

        int64_t key5 = loaded_keys[i + 5];
        uint64_t hash5 = murmurhash64(key5);
        scalar_buckets[i + 5] = hash5 & (kNumBuckets - 1) + j;

        int64_t key6 = loaded_keys[i + 6];
        uint64_t hash6 = murmurhash64(key6);
        scalar_buckets[i + 6] = hash6 & (kNumBuckets - 1) + j;

        int64_t key7 = loaded_keys[i + 7];
        uint64_t hash7 = murmurhash64(key7);
        scalar_buckets[i + 7] = hash7 & (kNumBuckets - 1) + j;
      }
      //      for (uint64_t i = 0; i < kNumKeys; i += 8) {
      //        int64_t key0 = keys[sel_vector[i + 0]];
      //        uint64_t hash0 = murmurhash64(key0);
      //        scalar_buckets[i + 0] = hash0 & (kNumBuckets - 1) + j;
      //
      //        int64_t key1 = keys[sel_vector[i + 1]];
      //        uint64_t hash1 = murmurhash64(key1);
      //        scalar_buckets[i + 1] = hash1 & (kNumBuckets - 1) + j;
      //
      //        int64_t key2 = keys[sel_vector[i + 2]];
      //        uint64_t hash2 = murmurhash64(key2);
      //        scalar_buckets[i + 2] = hash2 & (kNumBuckets - 1) + j;
      //
      //        int64_t key3 = keys[sel_vector[i + 3]];
      //        uint64_t hash3 = murmurhash64(key3);
      //        scalar_buckets[i + 3] = hash3 & (kNumBuckets - 1) + j;
      //
      //        int64_t key4 = keys[sel_vector[i + 4]];
      //        uint64_t hash4 = murmurhash64(key4);
      //        scalar_buckets[i + 4] = hash4 & (kNumBuckets - 1) + j;
      //
      //        int64_t key5 = keys[sel_vector[i + 5]];
      //        uint64_t hash5 = murmurhash64(key5);
      //        scalar_buckets[i + 5] = hash5 & (kNumBuckets - 1) + j;
      //
      //        int64_t key6 = keys[sel_vector[i + 6]];
      //        uint64_t hash6 = murmurhash64(key6);
      //        scalar_buckets[i + 6] = hash6 & (kNumBuckets - 1) + j;
      //
      //        int64_t key7 = keys[sel_vector[i + 7]];
      //        uint64_t hash7 = murmurhash64(key7);
      //        scalar_buckets[i + 7] = hash7 & (kNumBuckets - 1) + j;
      //      }
    }

    end_cycles = __rdtsc();

    uint64_t total_cycles = end_cycles - start_cycles;
    double cycles_per_tuple = static_cast<double>(total_cycles) / static_cast<double>(kNumKeys * run_times);
    std::cout << "Scalar Probing Cycles per tuple: " << cycles_per_tuple << "\n";
  }

  {
    std::vector<uint64_t> scalar_hashes(kNumKeys, 0);
    const uint64_t run_times = 65536;
    uint64_t start_cycles, end_cycles;

    start_cycles = __rdtsc();

    for (uint32_t j = 0; j < run_times; j++)
      for (uint64_t i = 0; i < kNumKeys; i += 8) {
        scalar_hashes[i] = murmurhash64(keys[i]) + j;
        scalar_hashes[i + 1] = murmurhash64(keys[i + 1]) + j;
        scalar_hashes[i + 2] = murmurhash64(keys[i + 2]) + j;
        scalar_hashes[i + 3] = murmurhash64(keys[i + 3]) + j;
        scalar_hashes[i + 4] = murmurhash64(keys[i + 4]) + j;
        scalar_hashes[i + 5] = murmurhash64(keys[i + 5]) + j;
        scalar_hashes[i + 6] = murmurhash64(keys[i + 6]) + j;
        scalar_hashes[i + 7] = murmurhash64(keys[i + 7]) + j;
      }

    end_cycles = __rdtsc();

    uint64_t total_cycles = end_cycles - start_cycles;
    double cycles_per_tuple = static_cast<double>(total_cycles) / static_cast<double>(kNumKeys * run_times);
    std::cout << "Scalar Hash Cycles per tuple: " << cycles_per_tuple << "\n";
  }
}
