#pragma once

#include "immintrin.h"

#include "base.h"
#include "profiler.h"

namespace compaction {
class SIMDHashTable {
  SIMDHashTable(size_t n_rhs_tuples, size_t chunk_factor, size_t payload_length);

};
}

