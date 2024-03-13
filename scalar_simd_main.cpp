#include <immintrin.h>
#include <iostream>
#include <vector>

#include "base.h"
#include "gather_functions.h"
#include "hash_functions.h"
#include "hash_table.h"
#include "profiler.h"
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

  Vector keys;
  std::vector<uint32_t> sel_vector(kNumKeys);
  for (uint64_t i = 0; i < kNumKeys; ++i) {
    keys[i] = int64_t(i);
    sel_vector[i] = i;
  }

  std::cout << "Join Key Size: " << 8 * kNumKeys / 1024 << " KB\n";
  std::cout << "Hash Table Size: " << 8 * kRHSTuples * 2 / 1024 << " KB\n";

  // this code is to be tested.
  std::cout << "--------------- SIMD HashJoin ---------------\n";
  {
    HashTable hash_table(kRHSTuples, 1);
    DataChunk input(vector<AttributeType>{AttributeType::INTEGER});
    input.data_[0] = keys;
    input.count_ = kNumKeys;
    DataChunk output(vector<AttributeType>{AttributeType::INTEGER, AttributeType::INTEGER, AttributeType::INTEGER});

    uint64_t start_cycles, end_cycles, probe_cycles = 0, next_cycles = 0;
    uint32_t next_times = 0;

    for (uint32_t j = 0; j < kRunTimes; j++) {
      // Function Probe.
      start_cycles = __rdtsc();

      hash_table.SIMDProbe(keys, kNumKeys, sel_vector);

      end_cycles = __rdtsc();
      probe_cycles += end_cycles - start_cycles;

      // Function Next.
      auto scan_structure = hash_table.GetScanStructure();
      while (scan_structure.HasNext()) {
        start_cycles = __rdtsc();

        scan_structure.SIMDNext(keys, input, output);

        end_cycles = __rdtsc();
        next_cycles += end_cycles - start_cycles;

        next_times++;
      }
    }
    double probe_cycles_per_tuple = static_cast<double>(probe_cycles) / static_cast<double>(kNumKeys * kRunTimes);
    double next_cycles_per_tuple = static_cast<double>(next_cycles) / static_cast<double>(kNumKeys * kRunTimes);
    std::cout << "Probe: " << probe_cycles_per_tuple << "\n";
    std::cout << "Next: " << next_cycles_per_tuple << "\n";
    std::cout << "#chunks: " << next_times << "\n";
  }

  // this code is to be tested.
  std::cout << "--------------- Scalar HashJoin ---------------\n";
  {
    HashTable hash_table(kRHSTuples, 1);
    DataChunk input(vector<AttributeType>{AttributeType::INTEGER});
    input.data_[0] = keys;
    input.count_ = kNumKeys;
    DataChunk output(vector<AttributeType>{AttributeType::INTEGER, AttributeType::INTEGER, AttributeType::INTEGER});

    uint64_t start_cycles, end_cycles, probe_cycles = 0, next_cycles = 0;
    uint32_t next_times = 0;

    for (uint32_t j = 0; j < kRunTimes; j++) {
      // Function Probe.
      start_cycles = __rdtsc();

      hash_table.Probe(keys, kNumKeys, sel_vector);

      end_cycles = __rdtsc();
      probe_cycles += end_cycles - start_cycles;

      // Function Next.
      auto scan_structure = hash_table.GetScanStructure();
      start_cycles = __rdtsc();
      while (scan_structure.HasNext()) {
        start_cycles = __rdtsc();

        scan_structure.Next(keys, input, output);

        end_cycles = __rdtsc();
        next_cycles += end_cycles - start_cycles;

        next_times++;
      }
    }
    double probe_cycles_per_tuple = static_cast<double>(probe_cycles) / static_cast<double>(kNumKeys * kRunTimes);
    double next_cycles_per_tuple = static_cast<double>(next_cycles) / static_cast<double>(kNumKeys * kRunTimes);
    std::cout << "Probe: " << probe_cycles_per_tuple << "\n";
    std::cout << "Next: " << next_cycles_per_tuple << "\n";
    std::cout << "#chunks: " << next_times << "\n";
  }

  std::cout << "--------------- Scalar with two loops ---------------\n";
  {
    std::vector<uint64_t> scalar_buckets(kNumKeys, 0);
    std::vector<int64_t> loaded_keys(kNumKeys, 0);
    uint64_t start_cycles, end_cycles, gather_cycles = 0, hash_cycles = 0;

    for (uint32_t j = 0; j < kRunTimes; j++) {

      start_cycles = __rdtsc();

      loaded_keys = ScalarGather(*keys.data_, sel_vector, loaded_keys);

      end_cycles = __rdtsc();
      gather_cycles += end_cycles - start_cycles;
      start_cycles = __rdtsc();

      uint64_t BUCKET_MASK = (kRHSTuples - 1);
      for (uint64_t i = 0; i < kNumKeys; i += 8) {
        int64_t key0 = loaded_keys[i];
        uint64_t hash0 = murmurhash64(key0);
        scalar_buckets[i] = hash0 & BUCKET_MASK + j;

        int64_t key1 = loaded_keys[i + 1];
        uint64_t hash1 = murmurhash64(key1);
        scalar_buckets[i + 1] = hash1 & BUCKET_MASK + j;

        int64_t key2 = loaded_keys[i + 2];
        uint64_t hash2 = murmurhash64(key2);
        scalar_buckets[i + 2] = hash2 & BUCKET_MASK + j;

        int64_t key3 = loaded_keys[i + 3];
        uint64_t hash3 = murmurhash64(key3);
        scalar_buckets[i + 3] = hash3 & BUCKET_MASK + j;

        int64_t key4 = loaded_keys[i + 4];
        uint64_t hash4 = murmurhash64(key4);
        scalar_buckets[i + 4] = hash4 & BUCKET_MASK + j;

        int64_t key5 = loaded_keys[i + 5];
        uint64_t hash5 = murmurhash64(key5);
        scalar_buckets[i + 5] = hash5 & BUCKET_MASK + j;

        int64_t key6 = loaded_keys[i + 6];
        uint64_t hash6 = murmurhash64(key6);
        scalar_buckets[i + 6] = hash6 & BUCKET_MASK + j;

        int64_t key7 = loaded_keys[i + 7];
        uint64_t hash7 = murmurhash64(key7);
        scalar_buckets[i + 7] = hash7 & BUCKET_MASK + j;
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
    uint64_t start_cycles, end_cycles;

    start_cycles = __rdtsc();

    for (uint32_t j = 0; j < kRunTimes; j++) {
      uint64_t BUCKET_MASK = (kRHSTuples - 1);
      for (uint64_t i = 0; i < kNumKeys; i += 8) {
        int64_t key0 = keys[sel_vector[i + 0]];
        uint64_t hash0 = murmurhash64(key0);
        scalar_buckets[i + 0] = hash0 & BUCKET_MASK + j;

        int64_t key1 = keys[sel_vector[i + 1]];
        uint64_t hash1 = murmurhash64(key1);
        scalar_buckets[i + 1] = hash1 & BUCKET_MASK + j;

        int64_t key2 = keys[sel_vector[i + 2]];
        uint64_t hash2 = murmurhash64(key2);
        scalar_buckets[i + 2] = hash2 & BUCKET_MASK + j;

        int64_t key3 = keys[sel_vector[i + 3]];
        uint64_t hash3 = murmurhash64(key3);
        scalar_buckets[i + 3] = hash3 & BUCKET_MASK + j;

        int64_t key4 = keys[sel_vector[i + 4]];
        uint64_t hash4 = murmurhash64(key4);
        scalar_buckets[i + 4] = hash4 & BUCKET_MASK + j;

        int64_t key5 = keys[sel_vector[i + 5]];
        uint64_t hash5 = murmurhash64(key5);
        scalar_buckets[i + 5] = hash5 & BUCKET_MASK + j;

        int64_t key6 = keys[sel_vector[i + 6]];
        uint64_t hash6 = murmurhash64(key6);
        scalar_buckets[i + 6] = hash6 & BUCKET_MASK + j;

        int64_t key7 = keys[sel_vector[i + 7]];
        uint64_t hash7 = murmurhash64(key7);
        scalar_buckets[i + 7] = hash7 & BUCKET_MASK + j;
      }
    }

    end_cycles = __rdtsc();

    uint64_t total_cycles = end_cycles - start_cycles;
    double cycles_per_tuple = static_cast<double>(total_cycles) / static_cast<double>(kNumKeys * kRunTimes);
    std::cout << "Probe: " << cycles_per_tuple << "\n";
  }
}
