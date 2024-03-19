cmake_minimum_required(VERSION 3.10)
project(Compaction-Project-SIMD)

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

add_executable(simd_compaction
        main.cpp
        profiler.h
        base.cpp
        chaining_ht.cpp
        data_collection.cpp
        compactor.cpp
)
target_compile_options(simd_compaction PRIVATE -march=native -funroll-loops)

# Scalar code, w/ -march=native
add_executable(scalar_hash_native simd_micro_bench.cpp base.cpp chaining_ht.cpp
        hash_functions.h
        linear_probing_ht.cpp
        linear_probing_ht.h)
target_compile_options(scalar_hash_native PRIVATE -march=native -funroll-loops)

# Scalar code, w/o -march=native
add_executable(scalar_hash simd_micro_bench.cpp base.cpp chaining_ht.cpp
        hash_functions.h
        linear_probing_ht.cpp
        linear_probing_ht.h)
target_compile_options(scalar_hash PRIVATE -funroll-loops -mavx512f -mavx512dq -mavx512vl)
