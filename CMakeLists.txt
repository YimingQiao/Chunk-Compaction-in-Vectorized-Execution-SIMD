cmake_minimum_required(VERSION 3.10)
project(Compaction-Project-SIMD)

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# add options for each compaction setting
option(USE_FULL_COMPACT "Use full compact method" OFF)
option(USE_BINARY_COMPACT "Use binary compact method" OFF)
option(USE_DYNAMIC_COMPACT "Use dynamic compact method" OFF)

# check which compaction method was chosen
if (USE_FULL_COMPACT)
    add_definitions(-Dflag_full_compact)
elseif (USE_BINARY_COMPACT)
    add_definitions(-Dflag_binary_compact)
elseif (USE_DYNAMIC_COMPACT)
    add_definitions(-Dflag_dynamic_compact)
else ()
    # default to no_compact
    add_definitions(-Dflag_no_compact)
endif ()

add_executable(simd_compaction_sevector main.cpp profiler.h base.cpp chaining_ht.cpp data_collection.cpp compactor.cpp negative_feedback.hpp)
target_compile_options(simd_compaction_sevector PRIVATE -march=native -funroll-loops)

add_executable(simd_bench_sevector simd_micro_bench.cpp base.cpp chaining_ht.cpp linear_probing_ht.cpp)
target_compile_options(simd_bench_sevector PRIVATE -funroll-loops -mavx512f -mavx512dq -mavx512vl -march=native)




