#include <cstdint>
#include <vector>

// using Attribute = std::variant<int64_t, double, char[24]>;
using Attribute = int64_t;
const uint64_t kNumKeys = 1024;
const uint64_t kNumBuckets = 1024;