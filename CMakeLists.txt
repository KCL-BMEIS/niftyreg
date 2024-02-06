cmake_minimum_required(VERSION 3.18)
#-----------------------------------------------------------------------------
project(NiftyReg)
#-----------------------------------------------------------------------------
# Set C++ standard version
set(CMAKE_CXX_STANDARD 17)
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
if(NOT MSVC)
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
  endif(NOT CMAKE_BUILD_TYPE)
  string(TOLOWER "${CMAKE_BUILD_TYPE}" cmake_build_type_tolower)
  if(NOT cmake_build_type_tolower STREQUAL "debug"
     AND NOT cmake_build_type_tolower STREQUAL "release"
     AND NOT cmake_build_type_tolower STREQUAL "relwithdebinfo")
    message("Unknown build type \"${CMAKE_BUILD_TYPE}\".")
    message(FATAL_ERROR "Allowed values are Debug, Release, RelWithDebInfo (case-insensitive).")
  endif(NOT cmake_build_type_tolower STREQUAL "debug"
     AND NOT cmake_build_type_tolower STREQUAL "release"
     AND NOT cmake_build_type_tolower STREQUAL "relwithdebinfo")
  if(cmake_build_type_tolower STREQUAL "debug")
    set(DEBUG_MODE ON)
  elseif(cmake_build_type_tolower STREQUAL "release")
    set(DEBUG_MODE OFF)
  endif(cmake_build_type_tolower STREQUAL "debug")
endif(NOT MSVC)
#-----------------------------------------------------------------------------
# Set the NiftyReg version
set(NR_VERSION_MAJOR 1)
set(NR_VERSION_MINOR 6)
file(STRINGS "niftyreg_build_version.txt" NR_VERSION_BUILD)
set(NR_VERSION "${NR_VERSION_MAJOR}.${NR_VERSION_MINOR}.${NR_VERSION_BUILD}")
add_definitions(-DNR_VERSION="${NR_VERSION}")
# Define the pre-commit hook for developer
find_package(Git)
if(GIT_FOUND)
  message(STATUS "Found Git")
  file(COPY "${CMAKE_SOURCE_DIR}/update_version_hook" DESTINATION "${CMAKE_SOURCE_DIR}/.git/hooks" USE_SOURCE_PERMISSIONS)
  file(RENAME "${CMAKE_SOURCE_DIR}/.git/hooks/update_version_hook" "${CMAKE_SOURCE_DIR}/.git/hooks/pre-commit")
endif(GIT_FOUND)
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
option(USE_OPENMP "To use openMP for multi-CPU processing" ON)
option(USE_SSE "To enable SSE computation in some case" ON)
option(CHECK_GPU "To check if a GPU is available" ON)
#-----------------------------------------------------------------------------
option(USE_NRRD "To use the NRRD file format" OFF)
mark_as_advanced(USE_NRRD)
#-----------------------------------------------------------------------------
if(WIN32)
    set(BUILD_ALL_DEP ON CACHE BOOL "All the dependencies are build" FORCE)
endif(WIN32)
#-----------------------------------------------------------------------------
# All dependencies are build to create the 3DSlicer package
if(BUILD_NR_SLICER_EXT)
    set(BUILD_ALL_DEP ON)
    mark_as_advanced(FORCE BUILD_ALL_DEP)
else(BUILD_NR_SLICER_EXT)
    mark_as_advanced(CLEAR BUILD_ALL_DEP)
endif(BUILD_NR_SLICER_EXT)
#-----------------------------------------------------------------------------
# Z library
# Try first to find the z library on the system and built is from the sources if it can not be find
if(NOT BUILD_ALL_DEP)
    find_package(ZLIB)
    if(ZLIB_FOUND)
        include_directories(${ZLIB_INCLUDE_DIR})
        message(STATUS "Found zlib - the z library will not be built")
    else(ZLIB_FOUND)
        include_directories(${CMAKE_SOURCE_DIR}/reg-io/zlib)
        message(STATUS "zlib not found - the z library will be built")
    endif(ZLIB_FOUND)
else(NOT BUILD_ALL_DEP)
    include_directories(${CMAKE_SOURCE_DIR}/reg-io/zlib)
endif(NOT BUILD_ALL_DEP)
#-----------------------------------------------------------------------------
# Try to find the png library and header on the system
if(NOT BUILD_ALL_DEP)
    ## PNG support - First try to find the PNG library on the system and build it if it is not found
    ## I did not use the FindPNG.cmake here as the zlib is also included into the project
    if(CYGWIN)
        if(NOT BUILD_SHARED_LIBS)
            set (PNG_DEFINITIONS -DPNG_STATIC)
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
        set(PNG_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/reg-io/png/lpng1510)
        set(PNG_LIBRARY png)
        set(BUILD_INTERNAL_PNG true)
    else(NOT PNG_LIBRARY OR NOT PNG_INCLUDE_DIR)
        message(STATUS "Found libpng - the png library will not be built")
        set(BUILD_INTERNAL_PNG false)
    endif(NOT PNG_LIBRARY OR NOT PNG_INCLUDE_DIR)
else(NOT BUILD_ALL_DEP)
    set(PNG_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/reg-io/png/lpng1510)
    set(PNG_LIBRARY png)
endif(NOT BUILD_ALL_DEP)
include_directories(${CMAKE_SOURCE_DIR}/reg-io/png)
include_directories(${PNG_INCLUDE_DIR})
#-----------------------------------------------------------------------------
include_directories(${CMAKE_SOURCE_DIR}/reg-lib)
include_directories(${CMAKE_SOURCE_DIR}/reg-lib/cpu)
include_directories(${CMAKE_SOURCE_DIR}/reg-io)
include_directories(${CMAKE_SOURCE_DIR}/third-party)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/third-party/eigen3)
#-----------------------------------------------------------------------------
if(USE_OPENCL)
  # Find the OpenCL package
  find_package(OpenCL REQUIRED)
  if(NOT OpenCL_FOUND)
    set(USE_OPENCL OFF CACHE BOOL "To use the OpenCL platform" FORCE)
    message(SEND_ERROR "OpenCL not found. The USE_OPENCL flag is turned OFF")
  else(NOT OpenCL_FOUND)
    message(STATUS "Found OpenCL")
    include_directories(${CMAKE_SOURCE_DIR}/reg-lib/cl)
    include_directories(${OpenCL_INCLUDE_DIRS})
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
    include_directories(${CMAKE_SOURCE_DIR}/reg-lib/cuda)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
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
    set(USE_OPENMP OFF CACHE BOOL "To use openMP for multi-CPU processing" FORCE)
    message(WARNING "OpenMP does not appear to be supported by your compiler, forcing USE_OPENMP to OFF")
  else(NOT OPENMP_FOUND)
    message(STATUS "Found OpenMP")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
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
add_subdirectory(third-party)
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
  set(DOXY_EXCLUDED_PATTERNS "")
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