cmake_minimum_required(VERSION 3.18)
project(NiftyRegAppleSSECheck LANGUAGES NONE)

# Exercise the actual selection blocks without configuring imaging dependencies.
file(READ "${CMAKE_CURRENT_LIST_DIR}/../../../CMakeLists.txt" source)
string(FIND "${source}" "# SSE is an x86-only instruction set" selection_start)
string(FIND "${source}" "option(CHECK_GPU" selection_end)
math(EXPR selection_length "${selection_end} - ${selection_start}")
string(SUBSTRING "${source}" ${selection_start} ${selection_length} selection)
string(FIND "${source}" "if(USE_SSE AND NOT NR_SSE_DEFAULT)" flags_start)
string(FIND "${source}" "endif(USE_SSE)" flags_end)
string(LENGTH "endif(USE_SSE)" terminator_length)
math(EXPR flags_length "${flags_end} + ${terminator_length} - ${flags_start}")
string(SUBSTRING "${source}" ${flags_start} ${flags_length} flags)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/selection.cmake" "${selection}\n${flags}\n")

# Simulate an Intel Mac requesting a universal build, including a stale cache.
set(APPLE TRUE)
set(CMAKE_SYSTEM_PROCESSOR x86_64)
set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64")
set(USE_SSE ON CACHE BOOL "Simulated existing cache" FORCE)
include("${CMAKE_CURRENT_BINARY_DIR}/selection.cmake")
get_directory_property(definitions COMPILE_DEFINITIONS)
if(USE_SSE OR CMAKE_C_FLAGS MATCHES "-msse3" OR
   CMAKE_CXX_FLAGS MATCHES "-msse3" OR "USE_SSE" IN_LIST definitions)
  message(FATAL_ERROR "ARM64-containing targets must not enable SSE: ${CMAKE_C_FLAGS}; ${CMAKE_CXX_FLAGS}; ${definitions}")
endif()
message(STATUS "ARM64-containing target disables cached SSE and x86 flags")
