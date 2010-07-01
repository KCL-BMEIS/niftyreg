PROJECT(nifty_reg)

cmake_minimum_required(VERSION 2.6.4)

IF("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION}" MATCHES "^2\\.6\\.4$")
 MARK_AS_ADVANCED(FORCE CMAKE_BACKWARDS_COMPATIBILITY)
ELSE("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION}" MATCHES "^2\\.6\\.4$")
 MARK_AS_ADVANCED(CLEAR CMAKE_BACKWARDS_COMPATIBILITY)
ENDIF("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION}" MATCHES "^2\\.6\\.4$")

#-----------------------------------------------------------------------------

if(COMMAND cmake_policy)
      cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

#-----------------------------------------------------------------------------
SET(ZLIB "z")
#-----------------------------------------------------------------------------

OPTION(BUILD_ALADIN "To build the reg_aladin executable" ON)
OPTION(BUILD_BENCHMARK "To build the reg_benchmark executable" ON)
OPTION(BUILD_F3D "To build the reg_f3d executable" ON)
#OPTION(BUILD_F3D2 "To build the reg_f3d2 executable" ON)
OPTION(BUILD_RESAMPLE "To build the reg_resample executable" ON)
OPTION(BUILD_TOOLS "To build the reg_tools executable" ON)

OPTION(USE_SSE "to use SSE computation in some case" OFF)
OPTION(USE_CUDA "Use CUDA for a GPU implementation of the code" OFF)
#-----------------------------------------------------------------------------

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/nifti)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/reg-lib)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/zlib)

#-----------------------------------------------------------------------------

IF(USE_SSE)
	ADD_DEFINITIONS(-D_USE_SSE)
ENDIF(USE_SSE)

IF(USE_CUDA)
    OPTION(CUDA_CHECK_CAP "Check the cuda card capability and adjust the block sizes accordingly" OFF)

	INCLUDE(${CMAKE_SOURCE_DIR}/CMake/cuda/FindCUDA.cmake)
	INCLUDE_DIRECTORIES(${CUDA_CUT_INCLUDE_DIR})
	CUDA_INCLUDE_DIRECTORIES(${CUDA_CUT_INCLUDE_DIR})

	
    if(CUDA_CHECK_CAP)
        try_run(RUN_RESULT_VAR COMPILE_RESULT_VAR
            ${CMAKE_BINARY_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/reg-apps/checkCudaCard.cpp
            CMAKE_FLAGS
                -DINCLUDE_DIRECTORIES:STRING=${CUDA_TOOLKIT_INCLUDE}
                -DLINK_LIBRARIES:STRING=${CUDA_CUDART_LIBRARY}
            COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT_VAR
            RUN_OUTPUT_VARIABLE RUN_OUTPUT_VAR
        )
        if(NOT COMPILE_RESULT_VAR)
            MESSAGE("Cuda test code did not compile. Exit")
            MESSAGE(FATAL_ERROR "\nError is the following:\n${COMPILE_RESULT_VAR}")
        endif(NOT COMPILE_RESULT_VAR)
        if(RUN_RESULT_VAR)
            MESSAGE("There are no cuda-enabled device detected on your machine.")
            MESSAGE("Please turn the USE_CUDA flag off.")
            MESSAGE(FATAL_ERROR "Turn the CUDA_CHECK_CAP flag OFF if you think you have the appropriate card to ignore the error.")
        endif(RUN_RESULT_VAR)

        MESSAGE("*** Cuda card capability ${RUN_OUTPUT_VAR} ***")
        # If the capability is equal or higher than 1.2
        IF(${RUN_OUTPUT_VAR} GREATER "1.1")
            ADD_DEFINITIONS(-D_HIGH_CAPA)
            SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-D_HIGH_CAPA")
        ENDIF(${RUN_OUTPUT_VAR} GREATER "1.1")
    ENDIF(CUDA_CHECK_CAP)
	
	ADD_DEFINITIONS(-D_USE_CUDA)
	SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-D_USE_CUDA")
	CUDA_INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/nifti)
	INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/reg-lib_gpu)
ENDIF(USE_CUDA)

#-----------------------------------------------------------------------------

IF(BUILD_BENCHMARK)
	IF(NOT BUILD_F3D)
		MESSAGE(FATAL_ERROR "BUILD_F3D is needed to build reg_benchmark. Exit")
	ENDIF(NOT BUILD_F3D)
	IF(NOT BUILD_ALADIN)
		MESSAGE(FATAL_ERROR "BUILD_ALADIN is needed to build reg_benchmark. Exit")
	ENDIF(NOT BUILD_ALADIN)
ENDIF(BUILD_BENCHMARK)

#-----------------------------------------------------------------------------

SUBDIRS(zlib)
SUBDIRS(nifti)
SUBDIRS(reg-lib)
IF(USE_CUDA)
	SUBDIRS(reg-lib_gpu)
ENDIF(USE_CUDA)
SUBDIRS(reg-apps)

#-----------------------------------------------------------------------------
