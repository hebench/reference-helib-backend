# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.12)
project(hebench_helib_backend LANGUAGES C CXX)

#OpenMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

find_package (OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

# third-party # pulls down updated code
include(FetchContent)
set(FETCHCONTENT_BASE_DIR ${CMAKE_BINARY_DIR}/third-party)

set(CMAKE_PROJECT_NAME ${PROJECT_NAME})

# General Setup
if(CMAKE_BUILD_TYPE)
  set(RELEASE_TYPES
      Debug
      Release
      RelWithDebInfo
      MinSizeRel)
  list(FIND RELEASE_TYPES ${CMAKE_BUILD_TYPE} INDEX_FOUND)
  if(${INDEX_FOUND} EQUAL -1)
    message(
      FATAL_ERROR
        "CMAKE_BUILD_TYPE must be one of Debug, Release, RelWithDebInfo, or MinSizeRel"
      )
  endif()
else()
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of Build" FORCE)
endif()

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

# C++ version (SEAL requires 17)
set(CMAKE_CXX_STANDARD 17) # C++ standard C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

###########################################################################################

# benchmarks

set(${PROJECT_NAME}_HEADERS
   "${CMAKE_CURRENT_SOURCE_DIR}/include/benchmarks/ckks/helib_ckks_element_wise_benchmark.h"
   "${CMAKE_CURRENT_SOURCE_DIR}/include/benchmarks/ckks/helib_ckks_dot_product_benchmark.h"
   "${CMAKE_CURRENT_SOURCE_DIR}/include/benchmarks/ckks/helib_ckks_matmultval_benchmark.h"
   "${CMAKE_CURRENT_SOURCE_DIR}/include/benchmarks/ckks/helib_ckks_logreg_horner.h"

   "${CMAKE_CURRENT_SOURCE_DIR}/include/benchmarks/bgv/helib_bgv_element_wise_benchmark.h"
   "${CMAKE_CURRENT_SOURCE_DIR}/include/benchmarks/bgv/helib_bgv_dot_product_benchmark.h"
   "${CMAKE_CURRENT_SOURCE_DIR}/include/benchmarks/bgv/helib_bgv_matmultval_benchmark.h"
   )

set(${PROJECT_NAME}_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/benchmarks/ckks/helib_ckks_element_wise_benchmark.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/benchmarks/ckks/helib_ckks_dot_product_benchmark.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/benchmarks/ckks/helib_ckks_matmultval_benchmark.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/benchmarks/ckks/helib_ckks_logreg_horner.cpp"

    "${CMAKE_CURRENT_SOURCE_DIR}/src/benchmarks/bgv/helib_bgv_element_wise_benchmark.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/benchmarks/bgv/helib_bgv_dot_product_benchmark.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/benchmarks/bgv/helib_bgv_matmultval_benchmark.cpp"
    )

#engine

list(APPEND ${PROJECT_NAME}_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/engine/helib_context.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/engine/helib_engine.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/engine/helib_types.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/engine/helib_version.h"
    )

list(APPEND ${PROJECT_NAME}_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/engine/helib_context.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/engine/helib_engine.cpp"
    )

add_library(${PROJECT_NAME} SHARED)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# All external libraries are assumed to be pre-compiled for tutorial simplicity.

# extra compile options
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra) # show warnings

# third-party
include(FetchContent)
set(FETCHCONTENT_BASE_DIR ${CMAKE_BINARY_DIR}/third-party)

## API-BRIDGE
set(_COMPONENT_NAME "API_BRIDGE")
set(_COMPONENT_HEADER "hebench/api_bridge/api.h")
set(_COMPONENT_LIB_NAME "hebench_cpp")
set(_HEADER_ONLY FALSE)
include(cmake/utils/import-library.cmake)
include(cmake/third-party/${_COMPONENT_NAME}_check.cmake)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/engine/helib_version.h.in" "${CMAKE_CURRENT_BINARY_DIR}/include/engine/helib_version.h.in1")
configure_file("${CMAKE_CURRENT_BINARY_DIR}/include/engine/helib_version.h.in1" "${CMAKE_CURRENT_SOURCE_DIR}/include/engine/helib_version.h")

target_sources(${PROJECT_NAME} PRIVATE ${${PROJECT_NAME}_SOURCES} ${${PROJECT_NAME}_HEADERS})

# install options
include(GNUInstallDirs)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
install(TARGETS ${PROJECT_NAME} DESTINATION lib)

# NTL
target_link_libraries(${PROJECT_NAME} PUBLIC "-lntl")

#HElib
set(_COMPONENT_NAME "HELIB")
set(_COMPONENT_HEADER "helib/helib.h")
set(_COMPONENT_LIB_NAME "helib")
set(_HEADER_ONLY FALSE)
include(cmake/utils/import-library.cmake)
