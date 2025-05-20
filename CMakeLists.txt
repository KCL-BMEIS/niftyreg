cmake_minimum_required(VERSION 3.18)
#-----------------------------------------------------------------------------
project(NiftyReg)
#-----------------------------------------------------------------------------
# Set C++ standard version
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
#-----------------------------------------------------------------------------
if(APPLE)
  set(CMAKE_MACOSX_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif(APPLE)
#-----------------------------------------------------------------------------
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message("In-source builds not allowed by NiftyReg police.")
  message("Please create a new directory (called a build directory) and run CMake from there.")
  message(FATAL_ERROR "You may need to remove CMakeCache.txt and CMakeFiles.")
endif(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
#-----------------------------------------------------------------------------
# Set the NiftyReg version using Git tags in the format:
# <tag>+<commits since tag>.<commit hash> or <tag> if no commits since tag
# Example: 1.0.0+2.abc1234 or 1.0.0 if no commits since tag
find_package(Git)
if(GIT_FOUND)
  # Get last tag from git
  execute_process(COMMAND ${GIT_EXECUTABLE} describe --abbrev=0 --tags
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE ${PROJECT_NAME}_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  # How many commits since last tag
  execute_process(COMMAND ${GIT_EXECUTABLE} rev-list ${${PROJECT_NAME}_VERSION}..HEAD --count
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE ${PROJECT_NAME}_VERSION_AHEAD
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  # Get current commit hash from git
  execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE ${PROJECT_NAME}_VERSION_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  # Remove 'v' prefix from the version string if present
  if(${PROJECT_NAME}_VERSION MATCHES "^v")
    string(SUBSTRING ${${PROJECT_NAME}_VERSION} 1 -1 ${PROJECT_NAME}_VERSION)
  endif()

  # Set full project version string
  if(NOT ${${PROJECT_NAME}_VERSION_AHEAD} EQUAL 0)
    set(${PROJECT_NAME}_VERSION
      ${${PROJECT_NAME}_VERSION}+${${PROJECT_NAME}_VERSION_AHEAD}.${${PROJECT_NAME}_VERSION_HASH})
  endif()
else(GIT_FOUND)
  message(FATAL_ERROR "Git not found. Please install Git to get the version information.")
endif(GIT_FOUND)
set(NR_VERSION ${${PROJECT_NAME}_VERSION})
add_definitions(-DNR_VERSION="${NR_VERSION}")
message(STATUS "NiftyReg version: ${NR_VERSION}")
#-----------------------------------------------------------------------------
if(MSVC)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
  set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /bigobj")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /bigobj")
endif(MSVC)
#-----------------------------------------------------------------------------
if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  add_definitions(-fPIC)
endif(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Windows")
#-----------------------------------------------------------------------------
option(BUILD_ALL_DEP "All the dependencies are build" OFF)
option(BUILD_SHARED_LIBS "Build the libraries as shared" OFF)
option(BUILD_TESTING "To build the unit tests" OFF)
option(USE_CUDA "To use the CUDA platform" OFF)
option(USE_OPENCL "To use the OpenCL platform" OFF)
option(USE_OPENMP "To use OpenMP for multi-CPU processing" ON)
option(USE_SSE "To enable SSE computation in some case" ON)
option(CHECK_GPU "To check if a GPU is available" ON)
#-----------------------------------------------------------------------------
option(USE_NRRD "To use the NRRD file format" OFF)
mark_as_advanced(USE_NRRD)
#-----------------------------------------------------------------------------
if(WIN32)
  set(BUILD_ALL_DEP ON CACHE BOOL "All the dependencies are built" FORCE)
endif(WIN32)
#-----------------------------------------------------------------------------
# All dependencies are built to create the 3DSlicer package
if(BUILD_NR_SLICER_EXT)
  set(BUILD_ALL_DEP ON)
  mark_as_advanced(FORCE BUILD_ALL_DEP)
else(BUILD_NR_SLICER_EXT)
  mark_as_advanced(CLEAR BUILD_ALL_DEP)
endif(BUILD_NR_SLICER_EXT)
#-----------------------------------------------------------------------------
# Z library
# Try first to find the z library on the system and build it from the sources if it can not be found
if(NOT BUILD_ALL_DEP)
  find_package(ZLIB)
  if(ZLIB_FOUND)
    include_directories(SYSTEM ${ZLIB_INCLUDE_DIR})
    message(STATUS "Found zlib - the z library will not be built")
  else(ZLIB_FOUND)
    include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/reg-io/zlib)
    message(STATUS "zlib not found - the z library will be built")
  endif(ZLIB_FOUND)
else(NOT BUILD_ALL_DEP)
  include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/reg-io/zlib)
endif(NOT BUILD_ALL_DEP)
#-----------------------------------------------------------------------------
# Try to find the png library and header on the system
if(NOT BUILD_ALL_DEP)
  # PNG support - First try to find the PNG library on the system and build it if it is not found
  # I did not use the FindPNG.cmake here as the zlib is also included into the project
  if(CYGWIN)
    if(NOT BUILD_SHARED_LIBS)
      set(PNG_DEFINITIONS -DPNG_STATIC)
    endif(NOT BUILD_SHARED_LIBS)
  endif(CYGWIN)
  set(PNG_NAMES ${PNG_NAMES} png libpng png15 libpng15 png15d libpng15d png14 libpng14 png14d libpng14d png12 libpng12 png12d libpng12d)
  find_library(PNG_LIBRARY NAMES ${PNG_NAMES})
  find_path(PNG_INCLUDE_DIR png.h
    /usr/local/include/libpng
    /sw/include
  )
  # If the png library and header can not be found, it is build from the sources
  if(NOT PNG_LIBRARY OR NOT PNG_INCLUDE_DIR)
    message(STATUS "libpng not found - the png library will be built")
    set(PNG_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/reg-io/png/lpng)
    set(PNG_LIBRARY png)
    set(BUILD_INTERNAL_PNG true)
  else(NOT PNG_LIBRARY OR NOT PNG_INCLUDE_DIR)
    message(STATUS "Found libpng - the png library will not be built")
    set(BUILD_INTERNAL_PNG false)
  endif(NOT PNG_LIBRARY OR NOT PNG_INCLUDE_DIR)
else(NOT BUILD_ALL_DEP)
  set(PNG_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/reg-io/png/lpng)
  set(PNG_LIBRARY png)
endif(NOT BUILD_ALL_DEP)
include_directories(SYSTEM ${PNG_INCLUDE_DIR})
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/reg-io/png)
#-----------------------------------------------------------------------------
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR}/reg-io)
include_directories(${CMAKE_SOURCE_DIR}/reg-lib)
include_directories(${CMAKE_SOURCE_DIR}/reg-lib/cpu)
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/third-party)
#-----------------------------------------------------------------------------
if(USE_OPENCL)
  # Find the OpenCL package
  find_package(OpenCL REQUIRED)
  if(NOT OpenCL_FOUND)
    set(USE_OPENCL OFF CACHE BOOL "To use the OpenCL platform" FORCE)
    message(SEND_ERROR "OpenCL not found. The USE_OPENCL flag is turned OFF")
  else(NOT OpenCL_FOUND)
    message(STATUS "Found OpenCL")
    include_directories(SYSTEM ${OpenCL_INCLUDE_DIRS})
    include_directories(${CMAKE_SOURCE_DIR}/reg-lib/cl)
    add_definitions(-DUSE_OPENCL)
  endif(NOT OpenCL_FOUND)
endif(USE_OPENCL)
#-----------------------------------------------------------------------------
if(USE_CUDA)
  # Check if the CUDA Toolkit is available
  enable_language(CUDA)
  find_package(CUDAToolkit)
  option(CUDA_FAST_MATH "To use the fast math flag" OFF)
  mark_as_advanced(CUDA_FAST_MATH)
  if(NOT CMAKE_CUDA_COMPILER)
    set(USE_CUDA OFF CACHE BOOL "To use the CUDA platform" FORCE)
    message(SEND_ERROR "CUDA not found. The USE_CUDA flag is turned OFF")
  else(NOT CMAKE_CUDA_COMPILER)
    include_directories(SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    include_directories(${CMAKE_SOURCE_DIR}/reg-lib/cuda)
    add_definitions(-DUSE_CUDA)
  endif(NOT CMAKE_CUDA_COMPILER)
endif(USE_CUDA)
#-----------------------------------------------------------------------------
if(USE_SSE)
  if(NOT MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3")
  endif(NOT MSVC)
  add_definitions(-DUSE_SSE)
endif(USE_SSE)
#-----------------------------------------------------------------------------
if(USE_OPENMP)
  find_package(OpenMP)
  if(NOT OPENMP_FOUND)
    set(USE_OPENMP OFF CACHE BOOL "To use OpenMP for multi-CPU processing" FORCE)
    message(WARNING "OpenMP does not appear to be supported by your compiler, forcing USE_OPENMP to OFF")
  else(NOT OPENMP_FOUND)
    message(STATUS "Found OpenMP")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    if (OpenMP_C_INCLUDE_DIRS)
      include_directories(SYSTEM ${OpenMP_C_INCLUDE_DIRS})
    endif(OpenMP_C_INCLUDE_DIRS)
    if (OpenMP_CXX_INCLUDE_DIRS)
      include_directories(SYSTEM ${OpenMP_CXX_INCLUDE_DIRS})
    endif(OpenMP_CXX_INCLUDE_DIRS)
    link_libraries(${OpenMP_CXX_LIBRARIES})
  endif(NOT OPENMP_FOUND)
endif(USE_OPENMP)
#-----------------------------------------------------------------------------
if(BUILD_SHARED_LIBS)
  if(USE_CUDA)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build the libraries as shared." FORCE)
    message(WARNING "CUDA is not compatible with shared libraries. Forcing BUILD_SHARED_LIBS to OFF")
    set(NIFTYREG_LIBRARY_TYPE STATIC)
  else(USE_CUDA)
    set(NIFTYREG_LIBRARY_TYPE SHARED)
  endif(USE_CUDA)
else(BUILD_SHARED_LIBS)
  set(NIFTYREG_LIBRARY_TYPE STATIC)
endif(BUILD_SHARED_LIBS)
#-----------------------------------------------------------------------------
if(BUILD_TESTING)
  enable_testing()
  add_definitions(-DBUILD_TESTS)
endif(BUILD_TESTING)
#-----------------------------------------------------------------------------
add_subdirectory(reg-io)
add_subdirectory(reg-lib)
add_subdirectory(reg-apps)
add_subdirectory(cmake)
if(BUILD_TESTING)
  add_subdirectory(${CMAKE_SOURCE_DIR}/reg-test)
endif(BUILD_TESTING)
#-----------------------------------------------------------------------------
# add a target to generate API documentation with Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
  set(DOXY_EXCLUDED_PATTERNS "*/build/* */reg-io/RNifti/*")
  if(NOT BUILD_TESTING)
    set(DOXY_EXCLUDED_PATTERNS "${DOXY_EXCLUDED_PATTERNS} */reg-test/*")
  endif(NOT BUILD_TESTING)
  if(NOT USE_NRRD)
    set(DOXY_EXCLUDED_PATTERNS "${DOXY_EXCLUDED_PATTERNS} */reg-io/nrrd/*")
  endif(NOT USE_NRRD)
  if(NOT USE_CUDA)
    set(DOXY_EXCLUDED_PATTERNS "${DOXY_EXCLUDED_PATTERNS} */reg-lib/cuda/*")
  endif(NOT USE_CUDA)
  if(NOT USE_OPENCL)
    set(DOXY_EXCLUDED_PATTERNS "${DOXY_EXCLUDED_PATTERNS} */reg-lib/cl/*")
  endif(NOT USE_OPENCL)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
  add_custom_target(doc
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM
  )
  message(STATUS "Found doxygen")
endif(DOXYGEN_FOUND)
#-----------------------------------------------------------------------------