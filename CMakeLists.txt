cmake_minimum_required(VERSION 3.10)
project(Compaction-Project-SIMD)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

add_executable(compaction
        main.cpp
        profiler.h
        base.cpp
        hash_table.cpp
        data_collection.cpp
        compactor.cpp
        simd_hash_table.cpp
        simd_hash_table.h)

# Scalar code, w/ -march=native
add_executable(scalar_hash_native scalar_simd_main.cpp base.cpp hash_table.cpp
        hash_functions.h
        linear_probing_hash_table.cpp
        linear_probing_hash_table.h)
target_compile_options(scalar_hash_native PRIVATE -march=native -funroll-loops)

# Scalar code, w/o -march=native
add_executable(scalar_hash scalar_simd_main.cpp base.cpp hash_table.cpp
        hash_functions.h
        linear_probing_hash_table.cpp
        linear_probing_hash_table.h)
target_compile_options(scalar_hash PRIVATE -funroll-loops -mavx512f -mavx512dq -mavx512vl)
