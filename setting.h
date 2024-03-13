//===----------------------------------------------------------------------===//
//
//                         Compaction
//
// setting.h
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "compactor.h"

// This file contains all parameters used in the project

namespace simd_compaction {

static uint64_t kNumKeys = 20480;
static uint64_t kRHSTuples = 5120;
static uint64_t kRunTimes = 32;
static uint64_t kLanes = 8;

// query setting
static size_t kJoins = 3;
static size_t kLHSTupleSize = 2e7;
static size_t kRHSTupleSize = 2e6;
static size_t kChunkFactor = 6;
static vector<size_t> kRHSPayLoadLength{0, 0, 0, 0};

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

const bool flag_collect_tuples = false;
}// namespace simd_compaction