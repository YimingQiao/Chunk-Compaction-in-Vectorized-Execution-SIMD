#include <immintrin.h>
#include <iostream>
#include <vector>

#include "base.h"
#include "hash_table.h"
#include "setting.h"

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

  std::vector<int64_t> keys(kNumKeys);
  for (uint64_t i = 0; i < kNumKeys; ++i) { keys[i] = rand() & (kRHSTuples * kHitFreq - 1); }
  std::vector<uint32_t> sel_vector(kBlockSize);
  for (uint32_t i = 0; i < kBlockSize; ++i) { sel_vector[i] = i; }

  // LHS: 64-bit keys. RHS: 64-bit keys, and 64-bit payloads
  std::cout << "LHS Table Size: " << 8 * kNumKeys / 1024 << " KB\n";
  std::cout << "RHS Table Size: " << 8 * kRHSTuples * 2 / 1024 << " KB\n";

  // this code is to be tested.
  std::cout << "--------------- SIMD HashJoin ---------------\n";
  {
    CycleProfiler::Get().Init();
    HashTable hash_table(kRHSTuples, kChunkFactor);
    DataChunk input(vector<AttributeType>{AttributeType::INTEGER});
    DataChunk output(vector<AttributeType>{AttributeType::INTEGER, AttributeType::INTEGER, AttributeType::INTEGER});
    Vector keys_block;

    uint64_t n_tuples = 0;

    for (uint64_t j = 0; j < kRunTimes; j++) {
      for (uint32_t k = 0; k < kNumKeys; k += kBlockSize) {
        // Load one block
        size_t n_filling = std::min(kBlockSize, kNumKeys - k);
        for (uint32_t i = 0; i < n_filling; ++i) {
          keys_block[i] = keys[k + i];
          input.data_[0] = keys_block;
          input.count_ = n_filling;
        }

        // Function Probe.
        hash_table.SIMDProbe(keys_block, n_filling, sel_vector);

        // Function Next.
        auto scan_structure = hash_table.GetScanStructure();
        while (scan_structure.HasNext()) { n_tuples += scan_structure.SIMDNext(keys_block, input, output); }
      }
    }
    std::vector<double> cpts(4);
    for (size_t i = 0; i < 4; i++) { cpts[i] = double(CycleProfiler::Get().Data(i)) / double(kNumKeys * kRunTimes); }
    CycleProfiler::Get().Init();
    std::cout << "Hash & Find Bucket: " << cpts[0] << "\t";
    std::cout << "Match Tuples: " << cpts[1] << "\t";
    std::cout << "Gather Tuples: " << cpts[2] << "\t";
    std::cout << "Advance Pointers: " << cpts[3] << "\n";
    std::cout << "Total: " << cpts[0] + cpts[1] + cpts[2] + cpts[3] << "\n";
    std::cout << "#tuples: " << n_tuples << "\n";
  }

  std::cout << "--------------- Scalar HashJoin ---------------\n";
  {
    CycleProfiler::Get().Init();
    HashTable hash_table(kRHSTuples, kChunkFactor);
    DataChunk input(vector<AttributeType>{AttributeType::INTEGER});
    DataChunk output(vector<AttributeType>{AttributeType::INTEGER, AttributeType::INTEGER, AttributeType::INTEGER});
    Vector keys_block;

    uint64_t n_tuples = 0;

    for (uint32_t j = 0; j < kRunTimes; j++) {
      for (uint32_t k = 0; k < kNumKeys; k += kBlockSize) {
        // load one block
        size_t n_filling = std::min(kBlockSize, kNumKeys - k);
        for (uint32_t i = 0; i < n_filling; ++i) {
          keys_block[i] = keys[k + i];
          input.data_[0] = keys_block;
          input.count_ = n_filling;
        }

        // Function Probe.
        hash_table.Probe(keys_block, n_filling, sel_vector);

        // Function Next.
        auto scan_structure = hash_table.GetScanStructure();
        while (scan_structure.HasNext()) { n_tuples += scan_structure.Next(keys_block, input, output); }
      }
    }

    std::vector<double> cpts(4);
    for (size_t i = 0; i < 4; i++) { cpts[i] = double(CycleProfiler::Get().Data(i)) / double(kNumKeys * kRunTimes); }
    std::cout << "Hash & Find Bucket: " << cpts[0] << "\t";
    std::cout << "Match Tuples: " << cpts[1] << "\t";
    std::cout << "Gather Tuples: " << cpts[2] << "\t";
    std::cout << "Advance Pointers: " << cpts[3] << "\n";
    std::cout << "Total: " << cpts[0] + cpts[1] + cpts[2] + cpts[3] << "\n";
    std::cout << "#tuples: " << n_tuples << "\n";
  }

  std::cout << "--------------- Scalar with two loops ---------------\n";
  {
    std::vector<uint64_t> scalar_buckets(kNumKeys, 0);
    std::vector<int64_t> loaded_keys(kNumKeys, 0);
    std::vector<uint32_t> sel_vector(kNumKeys, 0);
    for (size_t i = 0; i < kNumKeys; i++) sel_vector[i] = i;
    uint64_t start_cycles, end_cycles, gather_cycles = 0, hash_cycles = 0;
    uint64_t BUCKET_MASK = (kRHSTuples - 1);

    for (uint32_t j = 0; j < kRunTimes; j++) {
      start_cycles = __rdtsc();

      loaded_keys = ScalarGather(keys, sel_vector, loaded_keys);

      end_cycles = __rdtsc();
      gather_cycles += end_cycles - start_cycles;
      start_cycles = __rdtsc();

      for (uint64_t i = 0; i < kNumKeys; i++) {
        int64_t key = loaded_keys[i];
        uint64_t hash = murmurhash64(key);
        scalar_buckets[i] = hash & BUCKET_MASK;
      }

      end_cycles = __rdtsc();
      hash_cycles += end_cycles - start_cycles;
    }

    double gather_cycles_per_tuple = static_cast<double>(gather_cycles) / static_cast<double>(kNumKeys * kRunTimes);
    double hash_cycles_per_tuple = static_cast<double>(hash_cycles) / static_cast<double>(kNumKeys * kRunTimes);
    std::cout << "Gather: " << gather_cycles_per_tuple << "\t";
    std::cout << "Hash: " << hash_cycles_per_tuple << "\t";
    std::cout << "Probe: " << gather_cycles_per_tuple + hash_cycles_per_tuple << "\n";
  }

  std::cout << "--------------- Scalar with one loop ---------------\n";
  {
    std::vector<uint64_t> scalar_buckets(kNumKeys, 0);
    std::vector<uint32_t> sel_vector(kNumKeys, 0);
    for (size_t i = 0; i < kNumKeys; i++) sel_vector[i] = i;
    uint64_t start_cycles, end_cycles, total_cycles = 0;
    uint64_t BUCKET_MASK = (kRHSTuples - 1);

    for (uint32_t j = 0; j < kRunTimes; j++) {
      start_cycles = __rdtsc();
      for (uint64_t i = 0; i < kNumKeys; i++) {
        int64_t key = keys[sel_vector[i]];
        uint64_t hash = murmurhash64(key);
        scalar_buckets[i] = hash & BUCKET_MASK;
      }
      end_cycles = __rdtsc();
      total_cycles += end_cycles - start_cycles;
    }
    double cycles_per_tuple = static_cast<double>(total_cycles) / static_cast<double>(kNumKeys * kRunTimes);
    std::cout << "Probe: " << cycles_per_tuple << "\n";
  }
}
