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

# AVX512 hand-writing code, w/ -march=native
add_executable(avx512_hash avx512_main.cpp hash_functions.h)
target_compile_options(avx512_hash PRIVATE -march=native -funroll-loops)

# Scalar code, w/ -march=native
add_executable(scalar_hash_native scalar_simd_main.cpp base.cpp hash_table.cpp)
target_compile_options(scalar_hash_native PRIVATE -march=native -funroll-loops)

# Scalar code, w/o -march=native
add_executable(scalar_hash scalar_simd_main.cpp base.cpp hash_table.cpp)
target_compile_options(scalar_hash PRIVATE -funroll-loops)


## Scalar code, w/ -march=native
#add_executable(scalar_hash_native scalar_hash.cpp)
#target_compile_options(scalar_hash_native PRIVATE -march=native -funroll-loops)
#
## Gather code, without -march=native
#add_library(scalar_gather scalar_gather.cpp)
#target_link_libraries(scalar_hash_native scalar_gather)





