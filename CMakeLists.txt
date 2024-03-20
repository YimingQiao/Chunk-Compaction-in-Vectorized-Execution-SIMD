cmake_minimum_required(VERSION 3.10)
project(Compaction-Project-SIMD)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

add_executable(simd_compaction_dicodes main.cpp chaining_ht.cpp data_collection.cpp compactor.cpp base.cpp)
target_compile_options(simd_compaction_dicodes PRIVATE -march=native -funroll-loops)

add_executable(simd_bench_dicodes simd_micro_bench.cpp chaining_ht.cpp linear_probing_ht.cpp base.cpp)
target_compile_options(simd_bench_dicodes PRIVATE -funroll-loops -mavx512f -mavx512dq -mavx512vl -march=native)
