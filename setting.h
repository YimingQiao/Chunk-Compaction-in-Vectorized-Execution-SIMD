//===----------------------------------------------------------------------===//
//
//                         SIMD Compaction
//
// setting.h
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "compactor.h"

namespace simd_compaction {

// compaction setting
#ifdef flag_full_compact
using Compactor = NaiveCompactor;
const string strategy_name = "full_compaction";
#elif defined(flag_binary_compact)
using Compactor = BinaryCompactor;
const string strategy_name = "binary_compaction";
#elif defined(flag_dynamic_compact)
using Compactor = DynamicCompactor;
const string strategy_name = "dynamic_compaction";
#else
using Compactor = NaiveCompactor;
const string strategy_name = "no_compaction";
#endif

const bool flag_collect_tuples = true;
}// namespace simd_compaction