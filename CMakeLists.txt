cmake_minimum_required(VERSION 3.16)
project(tickstream VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(tickstream INTERFACE)
target_include_directories(tickstream INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

option(TICKSTREAM_BUILD_TESTS "Build C++ tests" ON)
option(TICKSTREAM_BUILD_BENCH "Build benchmark" ON)

if(TICKSTREAM_BUILD_TESTS)
  enable_testing()
  add_executable(test_core tests/test_core.cpp)
  target_link_libraries(test_core PRIVATE tickstream)
  target_compile_options(test_core PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME test_core COMMAND test_core)
endif()

if(TICKSTREAM_BUILD_BENCH)
  add_executable(bench bench/bench.cpp)
  target_link_libraries(bench PRIVATE tickstream)
  target_compile_options(bench PRIVATE -O3 -march=native)
endif()
