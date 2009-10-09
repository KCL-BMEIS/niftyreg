PROJECT(nifty_reg)

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

OPTION(BUILD_ALADIN "To build the reg_aladin executable" ON)
OPTION(BUILD_F3D "To build the reg_f3d executable" ON)
OPTION(BUILD_RESAMPLE "To build the reg_resample executable" ON)
OPTION(BUILD_TOOLS "To build the reg_tools executable" ON)

OPTION(USE_VERBOSE "To print out extra information" OFF)

OPTION(USE_SSE "to use SSE computation in some case" OFF)

OPTION(USE_CUDA "Use CUDA for a GPU implementation of the code" OFF)
#-----------------------------------------------------------------------------

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/reg-lib)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/nifti)

#-----------------------------------------------------------------------------

IF(USE_VERBOSE)
	ADD_DEFINITIONS(-D_VERBOSE)
ENDIF(USE_VERBOSE)

IF(USE_SSE)
	ADD_DEFINITIONS(-D_USE_SSE)
ENDIF(USE_SSE)

IF(USE_CUDA)
	INCLUDE(${CMAKE_SOURCE_DIR}/CMake/cuda/FindCUDA.cmake)
	ADD_DEFINITIONS(-D_USE_CUDA)
	SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-D_USE_CUDA")
	INCLUDE_DIRECTORIES(${CUDA_CUT_INCLUDE_DIR})
	CUDA_INCLUDE_DIRECTORIES(${CUDA_CUT_INCLUDE_DIR})
	CUDA_INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/nifti)
	INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/reg-lib_gpu)
ENDIF(USE_CUDA)

#-----------------------------------------------------------------------------

SUBDIRS(nifti)
SUBDIRS(reg-lib)
SUBDIRS(reg-apps)
IF(USE_CUDA)
	SUBDIRS(reg-lib_gpu)
ENDIF(USE_CUDA)

#-----------------------------------------------------------------------------
