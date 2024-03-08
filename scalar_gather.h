#include <vector>
#include <cstdint>

// using Attribute = std::variant<int64_t, double, char[24]>;
using Attribute = int64_t;
const uint64_t kNumKeys = 1024;
const uint64_t kNumBuckets = 1024;

// It uses AVX2 gather if inlined, otherwise use scalar.
inline std::vector<int64_t> &GetVector(const std::vector<Attribute> &keys, const std::vector<uint32_t> &sel_vector, std::vector<int64_t> &loaded_keys) {
  for (uint64_t i = 0; i < kNumKeys; i += 8) {
    int64_t key0 = keys[sel_vector[i]];
    loaded_keys[i] = key0;

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