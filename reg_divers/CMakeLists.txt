PROJECT(nifty_reg_divers)

cmake_minimum_required(VERSION 2.6)

IF("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" MATCHES "^2\\.6$")
 MARK_AS_ADVANCED(FORCE CMAKE_BACKWARDS_COMPATIBILITY)
ELSE("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" MATCHES "^2\\.6$")
 MARK_AS_ADVANCED(CLEAR CMAKE_BACKWARDS_COMPATIBILITY)
ENDIF("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" MATCHES "^2\\.6$")

#-----------------------------------------------------------------------------

if(COMMAND cmake_policy)
      cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

#-----------------------------------------------------------------------------
IF(WIN32)
    SET(ZLIB "zlib")
ELSE(WIN32)
    SET(ZLIB "z")
ENDIF(WIN32)
#-----------------------------------------------------------------------------

OPTION(BUILD_BENCHMARK "To build the reg_benchmark executable" ON)

#-----------------------------------------------------------------------------

SET(BUILD_F3D "ON")
SET(BUILD_ALADIN "ON")
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/../nifti)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/../reg-lib)

#-----------------------------------------------------------------------------

ADD_DEFINITIONS(-D_VERBOSE)
ADD_DEFINITIONS(-D_USE_SSE)

INCLUDE(${CMAKE_SOURCE_DIR}/../CMake/cuda/FindCUDA.cmake)
ADD_DEFINITIONS(-D_USE_CUDA)
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-D_USE_CUDA")
INCLUDE_DIRECTORIES(${CUDA_CUT_INCLUDE_DIR})
CUDA_INCLUDE_DIRECTORIES(${CUDA_CUT_INCLUDE_DIR})
CUDA_INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/../nifti)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/../reg-lib_gpu)

#-----------------------------------------------------------------------------

SUBDIRS(../nifti)
SUBDIRS(../reg-lib)
SUBDIRS(../reg-lib_gpu)

#-----------------------------------------------------------------------------
IF(BUILD_BENCHMARK)
	CUDA_ADD_EXECUTABLE(reg_benchmark reg_benchmark.cpp)
	TARGET_LINK_LIBRARIES(reg_benchmark _reg_resampling _reg_bspline _reg_mutualinformation _reg_ssd _reg_affineTransformation _reg_blockMatching _reg_tools ${ZLIB} reg_nifti _reg_resampling_gpu _reg_bspline_gpu _reg_mutualinformation_gpu _reg_affineTransformation_gpu _reg_blockMatching_gpu _reg_tools_gpu _reg_cudaCommon)
	INSTALL_TARGETS(/bin reg_benchmark)
ENDIF(BUILD_BENCHMARK)
#-----------------------------------------------------------------------------