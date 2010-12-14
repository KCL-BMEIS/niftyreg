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

OPTION(BUILD_NIFTYREG "To build the NiftyReg executables" ON)
OPTION(BUILD_NR_DEV "To build all the templated functions (longer). Useful for developpers only" OFF)

OPTION(USE_SSE "to use SSE computation in some case" OFF)
OPTION(USE_CUDA "Use CUDA for a GPU implementation of the code" OFF)
#-----------------------------------------------------------------------------

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/nifti)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/reg-lib)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/zlib)

#-----------------------------------------------------------------------------

ADD_DEFINITIONS(-fPIC)

IF(USE_SSE)
    ADD_DEFINITIONS(-D_USE_SSE)
ENDIF(USE_SSE)

IF(BUILD_NR_DEV)
    ADD_DEFINITIONS(-D_NR_DEV)
ENDIF(BUILD_NR_DEV)

IF(USE_CUDA)
    OPTION(CUDA_CHECK_CAP "Check the cuda card capability and adjust the NiftyReg block sizes accordingly" ON)

    INCLUDE(${CMAKE_SOURCE_DIR}/CMake/cuda/FindCUDA.cmake)

    # Check the cuda version to make sure it is compatible with NiftyReg
    IF("${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}" MATCHES "^3\\.2$")
        MESSAGE(ERROR "\nNiftyReg is not compatible with the 3.2 cuda drivers\nThe USE_CUDA flag has been turned off")
        SET(USE_CUDA OFF)
    ENDIF("${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}" MATCHES "^3\\.2$")

    INCLUDE_DIRECTORIES(${CUDA_CUT_INCLUDE_DIR})
    CUDA_INCLUDE_DIRECTORIES(${CUDA_CUT_INCLUDE_DIR})

    IF(NOT EXISTS "${CUDA_SDK_ROOT_DIR}/C/common/inc/cutil.h")
        MESSAGE(FATAL_ERROR "Please define the CUDA_SDK_ROOT_DIR variable\nsuch as CUDA_SDK_ROOT_DIR/C/common/inc/cutil.h exists")
    ENDIF(NOT EXISTS "${CUDA_SDK_ROOT_DIR}/C/common/inc/cutil.h")

    INCLUDE_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/C/common/inc)
    CUDA_INCLUDE_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/C/common/inc)
	
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
        ENDIF(NOT COMPILE_RESULT_VAR)
        if(RUN_RESULT_VAR)
            MESSAGE("There are no cuda-enabled device detected on your machine.")
            MESSAGE("Please turn the USE_CUDA flag off.")
            MESSAGE(FATAL_ERROR "Turn the CUDA_CHECK_CAP flag OFF if you think you have the appropriate card to ignore the error.")
        ENDIF(RUN_RESULT_VAR)

        MESSAGE("*** Cuda card capability ${RUN_OUTPUT_VAR} ***")
        # If the capability is equal or higher than 1.2
        IF(${RUN_OUTPUT_VAR} GREATER "1.2" OR ${RUN_OUTPUT_VAR} STREQUAL "1.2")
            ADD_DEFINITIONS(-D_HIGH_CAPA)
            SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-D_HIGH_CAPA")
            IF(${RUN_OUTPUT_VAR} GREATER "2.0" OR ${RUN_OUTPUT_VAR} STREQUAL "2.0")
                ADD_DEFINITIONS(-D_20_CAPA)
                SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-D_20_CAPA")
            ENDIF(${RUN_OUTPUT_VAR} GREATER "2.0" OR ${RUN_OUTPUT_VAR} STREQUAL "2.0")
        ENDIF(${RUN_OUTPUT_VAR} GREATER "1.2" OR ${RUN_OUTPUT_VAR} STREQUAL "1.2")
    ENDIF(CUDA_CHECK_CAP)
	
	ADD_DEFINITIONS(-D_USE_CUDA)
	SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-D_USE_CUDA")
    CUDA_INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/nifti)
ENDIF(USE_CUDA)

#-----------------------------------------------------------------------------

SUBDIRS(zlib)
SUBDIRS(nifti)
IF(BUILD_NIFTYREG)
	SUBDIRS(reg-lib)
	SUBDIRS(reg-apps)
ENDIF(BUILD_NIFTYREG)

#-----------------------------------------------------------------------------
