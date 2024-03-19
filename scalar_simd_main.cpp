#include <immintrin.h>
#include <iostream>
#include <vector>

#include "base.h"
#include "hash_table.h"
#include "linear_probing_hash_table.h"
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

  std::vector<int64_t> keys(kLHSTuples);
  for (uint64_t i = 0; i < kLHSTuples; ++i) { keys[i] = rand() & (kRHSTuples * kHitFreq - 1); }
  std::vector<uint32_t> sel_vector(kBlockSize);
  for (uint32_t i = 0; i < kBlockSize; ++i) { sel_vector[i] = i; }

  // LHS: 64-bit keys. RHS: 64-bit keys, and 64-bit payloads
  std::cout << "Work set size (LHS Block Size + RHS Hash Table Size): " << 8 * (kBlockSize) / 1024.0 << " KB + "
            << 8 * (kRHSTuples * 2) / 1024.0 << " KB\n";

  // this code is to be tested.
  std::cout << "--------------- SIMD Chaining HashJoin ---------------\n";
  {
    CycleProfiler::Get().Init();
    HashTable hash_table(kRHSTuples, kChunkFactor);
    DataChunk input(vector<AttributeType>{AttributeType::INTEGER});
    DataChunk output(vector<AttributeType>{AttributeType::INTEGER, AttributeType::INTEGER, AttributeType::INTEGER});
    Vector keys_block(AttributeType::INTEGER);
    uint64_t n_tuples = 0;

    for (uint32_t k = 0; k < kLHSTuples; k += kBlockSize) {
      // load one block
      size_t n_filling = std::min(kBlockSize, kLHSTuples - k);
      for (uint32_t i = 0; i < n_filling; ++i) { keys_block.GetValue(i) = keys[k + i]; }
      keys_block.count_ = n_filling;
      input.data_[0] = keys_block;
      input.count_ = n_filling;

      for (uint32_t i = 0; i < 1; i++) {
        // Function Probe.
        auto scan_structure = hash_table.SIMDProbe(keys_block);

        // Function Next.
        while (scan_structure.HasNext()) { n_tuples += scan_structure.SIMDNext(keys_block, input, output); }
      }
    }

    std::vector<double> cpts(4);
    for (size_t i = 0; i < 4; i++) { cpts[i] = double(CycleProfiler::Get().Data(i)) / double(kLHSTuples); }
    CycleProfiler::Get().Init();
    std::cout << "Hash & Find Bucket: " << cpts[0] << "\t";
    std::cout << "Match Tuples: " << cpts[1] << "\t";
    std::cout << "Gather Tuples: " << cpts[2] << "\t";
    std::cout << "Advance Pointers: " << cpts[3] << "\n";
    std::cout << "Total: " << cpts[0] + cpts[1] + cpts[2] + cpts[3] << "\n";
    std::cout << "#tuples: " << n_tuples << "\n";
  }

  std::cout << "--------------- Scalar Chaining HashJoin ---------------\n";
  {
    CycleProfiler::Get().Init();
    HashTable hash_table(kRHSTuples, kChunkFactor);
    DataChunk input(vector<AttributeType>{AttributeType::INTEGER});
    DataChunk output(vector<AttributeType>{AttributeType::INTEGER, AttributeType::INTEGER, AttributeType::INTEGER});
    Vector keys_block(AttributeType::INTEGER);

    uint64_t n_tuples = 0;

    for (uint32_t k = 0; k < kLHSTuples; k += kBlockSize) {
      // load one block
      size_t n_filling = std::min(kBlockSize, kLHSTuples - k);
      for (uint32_t i = 0; i < n_filling; ++i) { keys_block.GetValue(i) = keys[k + i]; }
      keys_block.count_ = n_filling;
      input.data_[0] = keys_block;
      input.count_ = n_filling;

      for (uint32_t i = 0; i < 1; i++) {
        // Function Probe.
        auto scan_structure = hash_table.Probe(keys_block);

        // Function Next.
        while (scan_structure.HasNext()) { n_tuples += scan_structure.Next(keys_block, input, output); }
      }
    }

    std::vector<double> cpts(4);
    for (size_t i = 0; i < 4; i++) { cpts[i] = double(CycleProfiler::Get().Data(i)) / double(kLHSTuples); }
    std::cout << "Hash & Find Bucket: " << cpts[0] << "\t";
    std::cout << "Match Tuples: " << cpts[1] << "\t";
    std::cout << "Gather Tuples: " << cpts[2] << "\t";
    std::cout << "Advance Pointers: " << cpts[3] << "\n";
    std::cout << "Total: " << cpts[0] + cpts[1] + cpts[2] + cpts[3] << "\n";
    std::cout << "#tuples: " << n_tuples << "\n";
  }

  std::cout << "--------------- SIMD Linear Probing HashJoin ---------------\n";
  {
    CycleProfiler::Get().Init();
    LPHashTable hash_table(kRHSTuples, kChunkFactor);
    DataChunk input(vector<AttributeType>{AttributeType::INTEGER});
    DataChunk output(vector<AttributeType>{AttributeType::INTEGER, AttributeType::INTEGER, AttributeType::INTEGER});
    Vector keys_block(AttributeType::INTEGER);

    uint64_t n_tuples = 0;

    for (uint32_t k = 0; k < kLHSTuples; k += kBlockSize) {
      // load one block
      size_t n_filling = std::min(kBlockSize, kLHSTuples - k);
      for (uint32_t i = 0; i < n_filling; ++i) { keys_block.GetValue(i) = keys[k + i]; }
      keys_block.count_ = n_filling;
      input.data_[0] = keys_block;
      input.count_ = n_filling;

      for (uint32_t i = 0; i < 1; i++) {
        // Function Probe.
        auto scan_structure = hash_table.SIMDProbe(keys_block);

        // Function Next.
        while (scan_structure.HasNext()) {
          n_tuples += scan_structure.SIMDNext(keys_block, input, output);
          output.Reset();
        }
      }
    }

    std::vector<double> cpts(4);
    for (size_t i = 0; i < 4; i++) { cpts[i] = double(CycleProfiler::Get().Data(i)) / double(kLHSTuples); }
    std::cout << "Hash & Find Bucket: " << cpts[0] << "\t";
    std::cout << "Match Tuples: " << cpts[1] << "\t";
    std::cout << "Gather Tuples: " << cpts[2] << "\t";
    std::cout << "Advance Pointers: " << cpts[3] << "\n";
    std::cout << "Total: " << cpts[0] + cpts[1] + cpts[2] + cpts[3] << "\n";
    std::cout << "#tuples: " << n_tuples << "\n";
  }

  std::cout << "--------------- Scalar Linear Probing HashJoin ---------------\n";
  {
    CycleProfiler::Get().Init();
    LPHashTable hash_table(kRHSTuples, kChunkFactor);
    DataChunk input(vector<AttributeType>{AttributeType::INTEGER});
    DataChunk output(vector<AttributeType>{AttributeType::INTEGER, AttributeType::INTEGER, AttributeType::INTEGER});
    Vector keys_block(AttributeType::INTEGER);

    uint64_t n_tuples = 0;

    for (uint32_t k = 0; k < kLHSTuples; k += kBlockSize) {
      // load one block
      size_t n_filling = std::min(kBlockSize, kLHSTuples - k);
      for (uint32_t i = 0; i < n_filling; ++i) { keys_block.GetValue(i) = keys[k + i]; }
      keys_block.count_ = n_filling;
      input.data_[0] = keys_block;
      input.count_ = n_filling;

      for (uint32_t i = 0; i < 1; i++) {
        // Function Probe.
        auto scan_structure = hash_table.Probe(keys_block);

        // Function Next.
        while (scan_structure.HasNext()) {
          n_tuples += scan_structure.Next(keys_block, input, output);
          output.Reset();
        }
      }
    }

    std::vector<double> cpts(4);
    for (size_t i = 0; i < 4; i++) { cpts[i] = double(CycleProfiler::Get().Data(i)) / double(kLHSTuples); }
    std::cout << "Hash & Find Bucket: " << cpts[0] << "\t";
    std::cout << "Match Tuples: " << cpts[1] << "\t";
    std::cout << "Gather Tuples: " << cpts[2] << "\t";
    std::cout << "Advance Pointers: " << cpts[3] << "\n";
    std::cout << "Total: " << cpts[0] + cpts[1] + cpts[2] + cpts[3] << "\n";
    std::cout << "#tuples: " << n_tuples << "\n";
  }

  std::cout << "--------------- SIMD (In One) Linear Probing HashJoin ---------------\n";
  {
    CycleProfiler::Get().Init();
    LPHashTable hash_table(kRHSTuples, kChunkFactor);
    DataChunk input(vector<AttributeType>{AttributeType::INTEGER});
    DataChunk output(vector<AttributeType>{AttributeType::INTEGER, AttributeType::INTEGER, AttributeType::INTEGER});
    Vector keys_block(AttributeType::INTEGER);

    uint64_t n_tuples = 0;

    for (uint32_t k = 0; k < kLHSTuples; k += kBlockSize) {
      // load one block
      size_t n_filling = std::min(kBlockSize, kLHSTuples - k);
      for (uint32_t i = 0; i < n_filling; ++i) { keys_block.GetValue(i) = keys[k + i]; }
      keys_block.count_ = n_filling;
      input.data_[0] = keys_block;
      input.count_ = n_filling;

      for (uint32_t i = 0; i < 1; i++) {
        // Function Probe.
        auto scan_structure = hash_table.SIMDProbe(keys_block);

        // Function Next.
        while (scan_structure.HasNext()) {
          n_tuples += scan_structure.SIMDInOneNext(keys_block, input, output);
          output.Reset();
        }
      }
    }

    std::vector<double> cpts(4);
    for (size_t i = 0; i < 4; i++) { cpts[i] = double(CycleProfiler::Get().Data(i)) / double(kLHSTuples); }
    std::cout << "Hash & Find Bucket: " << cpts[0] << "\t";
    std::cout << "Match & Gather & Advance: " << cpts[1] << "\n";
    std::cout << "Total: " << cpts[0] + cpts[1] + cpts[2] + cpts[3] << "\n";
    std::cout << "#tuples: " << n_tuples << "\n";
  }

  std::cout << "--------------- Scalar (In One) Linear Probing HashJoin ---------------\n";
  {
    CycleProfiler::Get().Init();
    LPHashTable hash_table(kRHSTuples, kChunkFactor);
    DataChunk input(vector<AttributeType>{AttributeType::INTEGER});
    DataChunk output(vector<AttributeType>{AttributeType::INTEGER, AttributeType::INTEGER, AttributeType::INTEGER});
    Vector keys_block(AttributeType::INTEGER);

    uint64_t n_tuples = 0;

    for (uint32_t k = 0; k < kLHSTuples; k += kBlockSize) {
      // load one block
      size_t n_filling = std::min(kBlockSize, kLHSTuples - k);
      for (uint32_t i = 0; i < n_filling; ++i) { keys_block.GetValue(i) = keys[k + i]; }
      keys_block.count_ = n_filling;
      input.data_[0] = keys_block;
      input.count_ = n_filling;

      for (uint32_t i = 0; i < 1; i++) {
        // Function Probe.
        auto scan_structure = hash_table.Probe(keys_block);

        // Function Next.
        while (scan_structure.HasNext()) {
          n_tuples += scan_structure.InOneNext(keys_block, input, output);
          output.Reset();
        }
      }
    }

    std::vector<double> cpts(4);
    for (size_t i = 0; i < 4; i++) { cpts[i] = double(CycleProfiler::Get().Data(i)) / double(kLHSTuples); }
    std::cout << "Hash & Find Bucket: " << cpts[0] << "\t";
    std::cout << "Match & Gather & Advance: " << cpts[1] << "\n";
    std::cout << "Total: " << cpts[0] + cpts[1] + cpts[2] + cpts[3] << "\n";
    std::cout << "#tuples: " << n_tuples << "\n";
  }
}