#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <vector>

#include "base.h"
#include "setting.h"
#include "hash_functions.h"

using namespace simd_compaction;

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
  Vector keys;
  std::vector<uint32_t> sel_vector(kLHSTuples);
  for (uint64_t i = 0; i < kLHSTuples; ++i) {
    keys[i] = int64_t(i);
    sel_vector[i] = i;
  }

  std::cout << "Sizeof(key): " << sizeof(keys[0]) << "\n";
  std::cout << "Workload Size: " << 8 * kLHSTuples / 1024 << " KB\n";

  std::cout << "--------------- SIMD with two loop ---------------\n";
  {
    std::vector<uint64_t> simd_buckets(kLHSTuples, 0);
    std::vector<int64_t> loaded_keys(kLHSTuples, 0);
    const uint64_t kLanes = 8;
    uint64_t start_cycles, end_cycles, gather_cycles = 0, hash_cycles = 0;

    __m512i BUCKET_MASK = _mm512_set1_epi64(kRHSTuples - 1);
    for (uint32_t j = 0; j < kRunTimes; j++) {

      start_cycles = __rdtsc();

      // gather
      int64_t *keys_data = keys.data_->data();
      for (uint64_t i = 0; i < kLHSTuples; i += kLanes * 64) {
        __m256i indices0 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 0);
        __m512i gathered_values0 = _mm512_i32gather_epi64(indices0, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 0, gathered_values0);

        __m256i indices1 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 1);
        __m512i gathered_values1 = _mm512_i32gather_epi64(indices1, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 1, gathered_values1);

        __m256i indices2 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 2);
        __m512i gathered_values2 = _mm512_i32gather_epi64(indices2, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 2, gathered_values2);

        __m256i indices3 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 3);
        __m512i gathered_values3 = _mm512_i32gather_epi64(indices3, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 3, gathered_values3);

        __m256i indices4 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 4);
        __m512i gathered_values4 = _mm512_i32gather_epi64(indices4, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 4, gathered_values4);

        __m256i indices5 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 5);
        __m512i gathered_values5 = _mm512_i32gather_epi64(indices5, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 5, gathered_values5);

        __m256i indices6 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 6);
        __m512i gathered_values6 = _mm512_i32gather_epi64(indices6, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 6, gathered_values6);

        __m256i indices7 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 7);
        __m512i gathered_values7 = _mm512_i32gather_epi64(indices7, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 7, gathered_values7);

        __m256i indices8 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 8);
        __m512i gathered_values8 = _mm512_i32gather_epi64(indices8, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 8, gathered_values8);

        __m256i indices9 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 9);
        __m512i gathered_values9 = _mm512_i32gather_epi64(indices9, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 9, gathered_values9);

        __m256i indices10 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 10);
        __m512i gathered_values10 = _mm512_i32gather_epi64(indices10, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 10, gathered_values10);

        __m256i indices11 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 11);
        __m512i gathered_values11 = _mm512_i32gather_epi64(indices11, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 11, gathered_values11);

        __m256i indices12 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 12);
        __m512i gathered_values12 = _mm512_i32gather_epi64(indices12, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 12, gathered_values12);

        __m256i indices13 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 13);
        __m512i gathered_values13 = _mm512_i32gather_epi64(indices13, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 13, gathered_values13);

        __m256i indices14 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 14);
        __m512i gathered_values14 = _mm512_i32gather_epi64(indices14, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 14, gathered_values14);

        __m256i indices15 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 15);
        __m512i gathered_values15 = _mm512_i32gather_epi64(indices15, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 15, gathered_values15);

        __m256i indices16 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 16);
        __m512i gathered_values16 = _mm512_i32gather_epi64(indices16, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 16, gathered_values16);

        __m256i indices17 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 17);
        __m512i gathered_values17 = _mm512_i32gather_epi64(indices17, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 17, gathered_values17);

        __m256i indices18 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 18);
        __m512i gathered_values18 = _mm512_i32gather_epi64(indices18, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 18, gathered_values18);

        __m256i indices19 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 19);
        __m512i gathered_values19 = _mm512_i32gather_epi64(indices19, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 19, gathered_values19);

        __m256i indices20 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 20);
        __m512i gathered_values20 = _mm512_i32gather_epi64(indices20, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 20, gathered_values20);

        __m256i indices21 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 21);
        __m512i gathered_values21 = _mm512_i32gather_epi64(indices21, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 21, gathered_values21);

        __m256i indices22 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 22);
        __m512i gathered_values22 = _mm512_i32gather_epi64(indices22, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 22, gathered_values22);

        __m256i indices23 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 23);
        __m512i gathered_values23 = _mm512_i32gather_epi64(indices23, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 23, gathered_values23);

        __m256i indices24 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 24);
        __m512i gathered_values24 = _mm512_i32gather_epi64(indices24, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 24, gathered_values24);

        __m256i indices25 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 25);
        __m512i gathered_values25 = _mm512_i32gather_epi64(indices25, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 25, gathered_values25);

        __m256i indices26 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 26);
        __m512i gathered_values26 = _mm512_i32gather_epi64(indices26, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 26, gathered_values26);

        __m256i indices27 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 27);
        __m512i gathered_values27 = _mm512_i32gather_epi64(indices27, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 27, gathered_values27);

        __m256i indices28 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 28);
        __m512i gathered_values28 = _mm512_i32gather_epi64(indices28, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 28, gathered_values28);

        __m256i indices29 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 29);
        __m512i gathered_values29 = _mm512_i32gather_epi64(indices29, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 29, gathered_values29);

        __m256i indices30 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 30);
        __m512i gathered_values30 = _mm512_i32gather_epi64(indices30, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 30, gathered_values30);

        __m256i indices31 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 31);
        __m512i gathered_values31 = _mm512_i32gather_epi64(indices31, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 31, gathered_values31);

        __m256i indices32 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 32);
        __m512i gathered_values32 = _mm512_i32gather_epi64(indices32, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 32, gathered_values32);

        __m256i indices33 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 33);
        __m512i gathered_values33 = _mm512_i32gather_epi64(indices33, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 33, gathered_values33);

        __m256i indices34 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 34);
        __m512i gathered_values34 = _mm512_i32gather_epi64(indices34, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 34, gathered_values34);

        __m256i indices35 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 35);
        __m512i gathered_values35 = _mm512_i32gather_epi64(indices35, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 35, gathered_values35);

        __m256i indices36 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 36);
        __m512i gathered_values36 = _mm512_i32gather_epi64(indices36, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 36, gathered_values36);

        __m256i indices37 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 37);
        __m512i gathered_values37 = _mm512_i32gather_epi64(indices37, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 37, gathered_values37);

        __m256i indices38 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 38);
        __m512i gathered_values38 = _mm512_i32gather_epi64(indices38, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 38, gathered_values38);

        __m256i indices39 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 39);
        __m512i gathered_values39 = _mm512_i32gather_epi64(indices39, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 39, gathered_values39);

        __m256i indices40 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 40);
        __m512i gathered_values40 = _mm512_i32gather_epi64(indices40, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 40, gathered_values40);

        __m256i indices41 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 41);
        __m512i gathered_values41 = _mm512_i32gather_epi64(indices41, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 41, gathered_values41);

        __m256i indices42 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 42);
        __m512i gathered_values42 = _mm512_i32gather_epi64(indices42, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 42, gathered_values42);

        __m256i indices43 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 43);
        __m512i gathered_values43 = _mm512_i32gather_epi64(indices43, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 43, gathered_values43);

        __m256i indices44 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 44);
        __m512i gathered_values44 = _mm512_i32gather_epi64(indices44, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 44, gathered_values44);

        __m256i indices45 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 45);
        __m512i gathered_values45 = _mm512_i32gather_epi64(indices45, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 45, gathered_values45);

        __m256i indices46 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 46);
        __m512i gathered_values46 = _mm512_i32gather_epi64(indices46, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 46, gathered_values46);

        __m256i indices47 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 47);
        __m512i gathered_values47 = _mm512_i32gather_epi64(indices47, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 47, gathered_values47);

        __m256i indices48 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 48);
        __m512i gathered_values48 = _mm512_i32gather_epi64(indices48, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 48, gathered_values48);

        __m256i indices49 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 49);
        __m512i gathered_values49 = _mm512_i32gather_epi64(indices49, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 49, gathered_values49);

        __m256i indices50 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 50);
        __m512i gathered_values50 = _mm512_i32gather_epi64(indices50, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 50, gathered_values50);

        __m256i indices51 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 51);
        __m512i gathered_values51 = _mm512_i32gather_epi64(indices51, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 51, gathered_values51);

        __m256i indices52 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 52);
        __m512i gathered_values52 = _mm512_i32gather_epi64(indices52, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 52, gathered_values52);

        __m256i indices53 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 53);
        __m512i gathered_values53 = _mm512_i32gather_epi64(indices53, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 53, gathered_values53);

        __m256i indices54 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 54);
        __m512i gathered_values54 = _mm512_i32gather_epi64(indices54, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 54, gathered_values54);

        __m256i indices55 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 55);
        __m512i gathered_values55 = _mm512_i32gather_epi64(indices55, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 55, gathered_values55);

        __m256i indices56 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 56);
        __m512i gathered_values56 = _mm512_i32gather_epi64(indices56, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 56, gathered_values56);

        __m256i indices57 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 57);
        __m512i gathered_values57 = _mm512_i32gather_epi64(indices57, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 57, gathered_values57);

        __m256i indices58 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 58);
        __m512i gathered_values58 = _mm512_i32gather_epi64(indices58, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 58, gathered_values58);

        __m256i indices59 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 59);
        __m512i gathered_values59 = _mm512_i32gather_epi64(indices59, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 59, gathered_values59);

        __m256i indices60 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 60);
        __m512i gathered_values60 = _mm512_i32gather_epi64(indices60, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 60, gathered_values60);

        __m256i indices61 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 61);
        __m512i gathered_values61 = _mm512_i32gather_epi64(indices61, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 61, gathered_values61);

        __m256i indices62 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 62);
        __m512i gathered_values62 = _mm512_i32gather_epi64(indices62, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 62, gathered_values62);

        __m256i indices63 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 63);
        __m512i gathered_values63 = _mm512_i32gather_epi64(indices63, keys_data, 8);
        _mm512_storeu_epi64(loaded_keys.data() + i + kLanes * 63, gathered_values63);
      }

      end_cycles = __rdtsc();
      gather_cycles += end_cycles - start_cycles;
      start_cycles = __rdtsc();

      // hash
      for (uint64_t i = 0; i < kLHSTuples; i += 8) {
        __m512i gathered_values0 = _mm512_loadu_epi64(loaded_keys.data() + i + kLanes * 0);
        __m512i hashes0 = mm512_murmurhash64(gathered_values0);
        __m512i bucket_indices0 = _mm512_and_si512(hashes0, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 0, bucket_indices0);
      }

      end_cycles = __rdtsc();
      hash_cycles += end_cycles - start_cycles;
    }

    double gather_cycles_per_tuple = static_cast<double>(gather_cycles) / static_cast<double>(kLHSTuples * kRunTimes);
    double hash_cycles_per_tuple = static_cast<double>(hash_cycles) / static_cast<double>(kLHSTuples * kRunTimes);
    std::cout << "Gather: " << gather_cycles_per_tuple << "\t";
    std::cout << "Hash: " << hash_cycles_per_tuple << "\t";
    std::cout << "Probe: " << gather_cycles_per_tuple + hash_cycles_per_tuple << "\n";
  }

  std::cout << "--------------- SIMD with one loop ---------------\n";
  {
    std::vector<uint64_t> simd_buckets(kLHSTuples, 0);
    std::vector<int64_t> loaded_keys(kLHSTuples, 0);
    const uint64_t kLanes = 8;
    uint64_t start_cycles, end_cycles;

    start_cycles = __rdtsc();

    __m512i BUCKET_MASK = _mm512_set1_epi64(kRHSTuples - 1);
    for (uint32_t j = 0; j < kRunTimes; j++) {
      int64_t *keys_data = keys.data_->data();
      for (uint64_t i = 0; i < kLHSTuples; i += kLanes * 64) {
        __m256i indices0 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 0);
        __m512i gathered_values0 = _mm512_i32gather_epi64(indices0, keys_data, 8);
        __m512i hashes0 = mm512_murmurhash64(gathered_values0);
        __m512i bucket_indices0 = _mm512_and_si512(hashes0, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 0, bucket_indices0);

        __m256i indices1 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 1);
        __m512i gathered_values1 = _mm512_i32gather_epi64(indices1, keys_data, 8);
        __m512i hashes1 = mm512_murmurhash64(gathered_values1);
        __m512i bucket_indices1 = _mm512_and_si512(hashes1, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 1, bucket_indices1);

        __m256i indices2 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 2);
        __m512i gathered_values2 = _mm512_i32gather_epi64(indices2, keys_data, 8);
        __m512i hashes2 = mm512_murmurhash64(gathered_values2);
        __m512i bucket_indices2 = _mm512_and_si512(hashes2, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 2, bucket_indices2);

        __m256i indices3 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 3);
        __m512i gathered_values3 = _mm512_i32gather_epi64(indices3, keys_data, 8);
        __m512i hashes3 = mm512_murmurhash64(gathered_values3);
        __m512i bucket_indices3 = _mm512_and_si512(hashes3, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 3, bucket_indices3);

        __m256i indices4 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 4);
        __m512i gathered_values4 = _mm512_i32gather_epi64(indices4, keys_data, 8);
        __m512i hashes4 = mm512_murmurhash64(gathered_values4);
        __m512i bucket_indices4 = _mm512_and_si512(hashes4, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 4, bucket_indices4);

        __m256i indices5 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 5);
        __m512i gathered_values5 = _mm512_i32gather_epi64(indices5, keys_data, 8);
        __m512i hashes5 = mm512_murmurhash64(gathered_values5);
        __m512i bucket_indices5 = _mm512_and_si512(hashes5, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 5, bucket_indices5);

        __m256i indices6 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 6);
        __m512i gathered_values6 = _mm512_i32gather_epi64(indices6, keys_data, 8);
        __m512i hashes6 = mm512_murmurhash64(gathered_values6);
        __m512i bucket_indices6 = _mm512_and_si512(hashes6, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 6, bucket_indices6);

        __m256i indices7 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 7);
        __m512i gathered_values7 = _mm512_i32gather_epi64(indices7, keys_data, 8);
        __m512i hashes7 = mm512_murmurhash64(gathered_values7);
        __m512i bucket_indices7 = _mm512_and_si512(hashes7, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 7, bucket_indices7);

        __m256i indices8 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 8);
        __m512i gathered_values8 = _mm512_i32gather_epi64(indices8, keys_data, 8);
        __m512i hashes8 = mm512_murmurhash64(gathered_values8);
        __m512i bucket_indices8 = _mm512_and_si512(hashes8, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 8, bucket_indices8);

        __m256i indices9 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 9);
        __m512i gathered_values9 = _mm512_i32gather_epi64(indices9, keys_data, 8);
        __m512i hashes9 = mm512_murmurhash64(gathered_values9);
        __m512i bucket_indices9 = _mm512_and_si512(hashes9, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 9, bucket_indices9);

        __m256i indices10 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 10);
        __m512i gathered_values10 = _mm512_i32gather_epi64(indices10, keys_data, 8);
        __m512i hashes10 = mm512_murmurhash64(gathered_values10);
        __m512i bucket_indices10 = _mm512_and_si512(hashes10, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 10, bucket_indices10);

        __m256i indices11 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 11);
        __m512i gathered_values11 = _mm512_i32gather_epi64(indices11, keys_data, 8);
        __m512i hashes11 = mm512_murmurhash64(gathered_values11);
        __m512i bucket_indices11 = _mm512_and_si512(hashes11, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 11, bucket_indices11);

        __m256i indices12 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 12);
        __m512i gathered_values12 = _mm512_i32gather_epi64(indices12, keys_data, 8);
        __m512i hashes12 = mm512_murmurhash64(gathered_values12);
        __m512i bucket_indices12 = _mm512_and_si512(hashes12, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 12, bucket_indices12);

        __m256i indices13 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 13);
        __m512i gathered_values13 = _mm512_i32gather_epi64(indices13, keys_data, 8);
        __m512i hashes13 = mm512_murmurhash64(gathered_values13);
        __m512i bucket_indices13 = _mm512_and_si512(hashes13, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 13, bucket_indices13);

        __m256i indices14 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 14);
        __m512i gathered_values14 = _mm512_i32gather_epi64(indices14, keys_data, 8);
        __m512i hashes14 = mm512_murmurhash64(gathered_values14);
        __m512i bucket_indices14 = _mm512_and_si512(hashes14, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 14, bucket_indices14);

        __m256i indices15 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 15);
        __m512i gathered_values15 = _mm512_i32gather_epi64(indices15, keys_data, 8);
        __m512i hashes15 = mm512_murmurhash64(gathered_values15);
        __m512i bucket_indices15 = _mm512_and_si512(hashes15, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 15, bucket_indices15);

        __m256i indices16 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 16);
        __m512i gathered_values16 = _mm512_i32gather_epi64(indices16, keys_data, 8);
        __m512i hashes16 = mm512_murmurhash64(gathered_values16);
        __m512i bucket_indices16 = _mm512_and_si512(hashes16, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 16, bucket_indices16);

        __m256i indices17 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 17);
        __m512i gathered_values17 = _mm512_i32gather_epi64(indices17, keys_data, 8);
        __m512i hashes17 = mm512_murmurhash64(gathered_values17);
        __m512i bucket_indices17 = _mm512_and_si512(hashes17, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 17, bucket_indices17);

        __m256i indices18 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 18);
        __m512i gathered_values18 = _mm512_i32gather_epi64(indices18, keys_data, 8);
        __m512i hashes18 = mm512_murmurhash64(gathered_values18);
        __m512i bucket_indices18 = _mm512_and_si512(hashes18, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 18, bucket_indices18);

        __m256i indices19 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 19);
        __m512i gathered_values19 = _mm512_i32gather_epi64(indices19, keys_data, 8);
        __m512i hashes19 = mm512_murmurhash64(gathered_values19);
        __m512i bucket_indices19 = _mm512_and_si512(hashes19, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 19, bucket_indices19);

        __m256i indices20 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 20);
        __m512i gathered_values20 = _mm512_i32gather_epi64(indices20, keys_data, 8);
        __m512i hashes20 = mm512_murmurhash64(gathered_values20);
        __m512i bucket_indices20 = _mm512_and_si512(hashes20, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 20, bucket_indices20);

        __m256i indices21 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 21);
        __m512i gathered_values21 = _mm512_i32gather_epi64(indices21, keys_data, 8);
        __m512i hashes21 = mm512_murmurhash64(gathered_values21);
        __m512i bucket_indices21 = _mm512_and_si512(hashes21, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 21, bucket_indices21);

        __m256i indices22 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 22);
        __m512i gathered_values22 = _mm512_i32gather_epi64(indices22, keys_data, 8);
        __m512i hashes22 = mm512_murmurhash64(gathered_values22);
        __m512i bucket_indices22 = _mm512_and_si512(hashes22, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 22, bucket_indices22);

        __m256i indices23 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 23);
        __m512i gathered_values23 = _mm512_i32gather_epi64(indices23, keys_data, 8);
        __m512i hashes23 = mm512_murmurhash64(gathered_values23);
        __m512i bucket_indices23 = _mm512_and_si512(hashes23, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 23, bucket_indices23);

        __m256i indices24 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 24);
        __m512i gathered_values24 = _mm512_i32gather_epi64(indices24, keys_data, 8);
        __m512i hashes24 = mm512_murmurhash64(gathered_values24);
        __m512i bucket_indices24 = _mm512_and_si512(hashes24, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 24, bucket_indices24);

        __m256i indices25 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 25);
        __m512i gathered_values25 = _mm512_i32gather_epi64(indices25, keys_data, 8);
        __m512i hashes25 = mm512_murmurhash64(gathered_values25);
        __m512i bucket_indices25 = _mm512_and_si512(hashes25, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 25, bucket_indices25);

        __m256i indices26 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 26);
        __m512i gathered_values26 = _mm512_i32gather_epi64(indices26, keys_data, 8);
        __m512i hashes26 = mm512_murmurhash64(gathered_values26);
        __m512i bucket_indices26 = _mm512_and_si512(hashes26, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 26, bucket_indices26);

        __m256i indices27 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 27);
        __m512i gathered_values27 = _mm512_i32gather_epi64(indices27, keys_data, 8);
        __m512i hashes27 = mm512_murmurhash64(gathered_values27);
        __m512i bucket_indices27 = _mm512_and_si512(hashes27, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 27, bucket_indices27);

        __m256i indices28 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 28);
        __m512i gathered_values28 = _mm512_i32gather_epi64(indices28, keys_data, 8);
        __m512i hashes28 = mm512_murmurhash64(gathered_values28);
        __m512i bucket_indices28 = _mm512_and_si512(hashes28, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 28, bucket_indices28);

        __m256i indices29 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 29);
        __m512i gathered_values29 = _mm512_i32gather_epi64(indices29, keys_data, 8);
        __m512i hashes29 = mm512_murmurhash64(gathered_values29);
        __m512i bucket_indices29 = _mm512_and_si512(hashes29, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 29, bucket_indices29);

        __m256i indices30 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 30);
        __m512i gathered_values30 = _mm512_i32gather_epi64(indices30, keys_data, 8);
        __m512i hashes30 = mm512_murmurhash64(gathered_values30);
        __m512i bucket_indices30 = _mm512_and_si512(hashes30, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 30, bucket_indices30);

        __m256i indices31 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 31);
        __m512i gathered_values31 = _mm512_i32gather_epi64(indices31, keys_data, 8);
        __m512i hashes31 = mm512_murmurhash64(gathered_values31);
        __m512i bucket_indices31 = _mm512_and_si512(hashes31, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 31, bucket_indices31);

        __m256i indices32 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 32);
        __m512i gathered_values32 = _mm512_i32gather_epi64(indices32, keys_data, 8);
        __m512i hashes32 = mm512_murmurhash64(gathered_values32);
        __m512i bucket_indices32 = _mm512_and_si512(hashes32, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 32, bucket_indices32);

        __m256i indices33 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 33);
        __m512i gathered_values33 = _mm512_i32gather_epi64(indices33, keys_data, 8);
        __m512i hashes33 = mm512_murmurhash64(gathered_values33);
        __m512i bucket_indices33 = _mm512_and_si512(hashes33, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 33, bucket_indices33);

        __m256i indices34 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 34);
        __m512i gathered_values34 = _mm512_i32gather_epi64(indices34, keys_data, 8);
        __m512i hashes34 = mm512_murmurhash64(gathered_values34);
        __m512i bucket_indices34 = _mm512_and_si512(hashes34, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 34, bucket_indices34);

        __m256i indices35 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 35);
        __m512i gathered_values35 = _mm512_i32gather_epi64(indices35, keys_data, 8);
        __m512i hashes35 = mm512_murmurhash64(gathered_values35);
        __m512i bucket_indices35 = _mm512_and_si512(hashes35, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 35, bucket_indices35);

        __m256i indices36 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 36);
        __m512i gathered_values36 = _mm512_i32gather_epi64(indices36, keys_data, 8);
        __m512i hashes36 = mm512_murmurhash64(gathered_values36);
        __m512i bucket_indices36 = _mm512_and_si512(hashes36, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 36, bucket_indices36);

        __m256i indices37 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 37);
        __m512i gathered_values37 = _mm512_i32gather_epi64(indices37, keys_data, 8);
        __m512i hashes37 = mm512_murmurhash64(gathered_values37);
        __m512i bucket_indices37 = _mm512_and_si512(hashes37, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 37, bucket_indices37);

        __m256i indices38 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 38);
        __m512i gathered_values38 = _mm512_i32gather_epi64(indices38, keys_data, 8);
        __m512i hashes38 = mm512_murmurhash64(gathered_values38);
        __m512i bucket_indices38 = _mm512_and_si512(hashes38, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 38, bucket_indices38);

        __m256i indices39 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 39);
        __m512i gathered_values39 = _mm512_i32gather_epi64(indices39, keys_data, 8);
        __m512i hashes39 = mm512_murmurhash64(gathered_values39);
        __m512i bucket_indices39 = _mm512_and_si512(hashes39, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 39, bucket_indices39);

        __m256i indices40 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 40);
        __m512i gathered_values40 = _mm512_i32gather_epi64(indices40, keys_data, 8);
        __m512i hashes40 = mm512_murmurhash64(gathered_values40);
        __m512i bucket_indices40 = _mm512_and_si512(hashes40, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 40, bucket_indices40);

        __m256i indices41 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 41);
        __m512i gathered_values41 = _mm512_i32gather_epi64(indices41, keys_data, 8);
        __m512i hashes41 = mm512_murmurhash64(gathered_values41);
        __m512i bucket_indices41 = _mm512_and_si512(hashes41, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 41, bucket_indices41);

        __m256i indices42 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 42);
        __m512i gathered_values42 = _mm512_i32gather_epi64(indices42, keys_data, 8);
        __m512i hashes42 = mm512_murmurhash64(gathered_values42);
        __m512i bucket_indices42 = _mm512_and_si512(hashes42, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 42, bucket_indices42);

        __m256i indices43 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 43);
        __m512i gathered_values43 = _mm512_i32gather_epi64(indices43, keys_data, 8);
        __m512i hashes43 = mm512_murmurhash64(gathered_values43);
        __m512i bucket_indices43 = _mm512_and_si512(hashes43, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 43, bucket_indices43);

        __m256i indices44 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 44);
        __m512i gathered_values44 = _mm512_i32gather_epi64(indices44, keys_data, 8);
        __m512i hashes44 = mm512_murmurhash64(gathered_values44);
        __m512i bucket_indices44 = _mm512_and_si512(hashes44, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 44, bucket_indices44);

        __m256i indices45 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 45);
        __m512i gathered_values45 = _mm512_i32gather_epi64(indices45, keys_data, 8);
        __m512i hashes45 = mm512_murmurhash64(gathered_values45);
        __m512i bucket_indices45 = _mm512_and_si512(hashes45, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 45, bucket_indices45);

        __m256i indices46 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 46);
        __m512i gathered_values46 = _mm512_i32gather_epi64(indices46, keys_data, 8);
        __m512i hashes46 = mm512_murmurhash64(gathered_values46);
        __m512i bucket_indices46 = _mm512_and_si512(hashes46, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 46, bucket_indices46);

        __m256i indices47 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 47);
        __m512i gathered_values47 = _mm512_i32gather_epi64(indices47, keys_data, 8);
        __m512i hashes47 = mm512_murmurhash64(gathered_values47);
        __m512i bucket_indices47 = _mm512_and_si512(hashes47, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 47, bucket_indices47);

        __m256i indices48 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 48);
        __m512i gathered_values48 = _mm512_i32gather_epi64(indices48, keys_data, 8);
        __m512i hashes48 = mm512_murmurhash64(gathered_values48);
        __m512i bucket_indices48 = _mm512_and_si512(hashes48, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 48, bucket_indices48);

        __m256i indices49 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 49);
        __m512i gathered_values49 = _mm512_i32gather_epi64(indices49, keys_data, 8);
        __m512i hashes49 = mm512_murmurhash64(gathered_values49);
        __m512i bucket_indices49 = _mm512_and_si512(hashes49, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 49, bucket_indices49);

        __m256i indices50 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 50);
        __m512i gathered_values50 = _mm512_i32gather_epi64(indices50, keys_data, 8);
        __m512i hashes50 = mm512_murmurhash64(gathered_values50);
        __m512i bucket_indices50 = _mm512_and_si512(hashes50, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 50, bucket_indices50);

        __m256i indices51 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 51);
        __m512i gathered_values51 = _mm512_i32gather_epi64(indices51, keys_data, 8);
        __m512i hashes51 = mm512_murmurhash64(gathered_values51);
        __m512i bucket_indices51 = _mm512_and_si512(hashes51, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 51, bucket_indices51);

        __m256i indices52 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 52);
        __m512i gathered_values52 = _mm512_i32gather_epi64(indices52, keys_data, 8);
        __m512i hashes52 = mm512_murmurhash64(gathered_values52);
        __m512i bucket_indices52 = _mm512_and_si512(hashes52, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 52, bucket_indices52);

        __m256i indices53 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 53);
        __m512i gathered_values53 = _mm512_i32gather_epi64(indices53, keys_data, 8);
        __m512i hashes53 = mm512_murmurhash64(gathered_values53);
        __m512i bucket_indices53 = _mm512_and_si512(hashes53, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 53, bucket_indices53);

        __m256i indices54 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 54);
        __m512i gathered_values54 = _mm512_i32gather_epi64(indices54, keys_data, 8);
        __m512i hashes54 = mm512_murmurhash64(gathered_values54);
        __m512i bucket_indices54 = _mm512_and_si512(hashes54, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 54, bucket_indices54);

        __m256i indices55 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 55);
        __m512i gathered_values55 = _mm512_i32gather_epi64(indices55, keys_data, 8);
        __m512i hashes55 = mm512_murmurhash64(gathered_values55);
        __m512i bucket_indices55 = _mm512_and_si512(hashes55, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 55, bucket_indices55);

        __m256i indices56 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 56);
        __m512i gathered_values56 = _mm512_i32gather_epi64(indices56, keys_data, 8);
        __m512i hashes56 = mm512_murmurhash64(gathered_values56);
        __m512i bucket_indices56 = _mm512_and_si512(hashes56, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 56, bucket_indices56);

        __m256i indices57 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 57);
        __m512i gathered_values57 = _mm512_i32gather_epi64(indices57, keys_data, 8);
        __m512i hashes57 = mm512_murmurhash64(gathered_values57);
        __m512i bucket_indices57 = _mm512_and_si512(hashes57, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 57, bucket_indices57);

        __m256i indices58 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 58);
        __m512i gathered_values58 = _mm512_i32gather_epi64(indices58, keys_data, 8);
        __m512i hashes58 = mm512_murmurhash64(gathered_values58);
        __m512i bucket_indices58 = _mm512_and_si512(hashes58, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 58, bucket_indices58);

        __m256i indices59 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 59);
        __m512i gathered_values59 = _mm512_i32gather_epi64(indices59, keys_data, 8);
        __m512i hashes59 = mm512_murmurhash64(gathered_values59);
        __m512i bucket_indices59 = _mm512_and_si512(hashes59, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 59, bucket_indices59);

        __m256i indices60 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 60);
        __m512i gathered_values60 = _mm512_i32gather_epi64(indices60, keys_data, 8);
        __m512i hashes60 = mm512_murmurhash64(gathered_values60);
        __m512i bucket_indices60 = _mm512_and_si512(hashes60, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 60, bucket_indices60);

        __m256i indices61 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 61);
        __m512i gathered_values61 = _mm512_i32gather_epi64(indices61, keys_data, 8);
        __m512i hashes61 = mm512_murmurhash64(gathered_values61);
        __m512i bucket_indices61 = _mm512_and_si512(hashes61, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 61, bucket_indices61);

        __m256i indices62 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 62);
        __m512i gathered_values62 = _mm512_i32gather_epi64(indices62, keys_data, 8);
        __m512i hashes62 = mm512_murmurhash64(gathered_values62);
        __m512i bucket_indices62 = _mm512_and_si512(hashes62, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 62, bucket_indices62);

        __m256i indices63 = _mm256_loadu_epi32(sel_vector.data() + i + kLanes * 63);
        __m512i gathered_values63 = _mm512_i32gather_epi64(indices63, keys_data, 8);
        __m512i hashes63 = mm512_murmurhash64(gathered_values63);
        __m512i bucket_indices63 = _mm512_and_si512(hashes63, BUCKET_MASK);
        _mm512_storeu_epi64(simd_buckets.data() + i + kLanes * 63, bucket_indices63);
      }
    }

    end_cycles = __rdtsc();

    uint64_t total_cycles = end_cycles - start_cycles;
    double cycles_per_tuple = static_cast<double>(total_cycles) / static_cast<double>(kLHSTuples * kRunTimes);
    std::cout << "Probe Cycles per tuple: " << cycles_per_tuple << "\n";
  }
}
