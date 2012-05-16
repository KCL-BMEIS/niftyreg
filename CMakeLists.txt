PROJECT(NiftyReg)

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

IF (${CMAKE_SYSTEM_NAME} MATCHES "Windows")	
	ADD_DEFINITIONS(/W1)
ELSE (${CMAKE_SYSTEM_NAME} MATCHES "Windows")	
	ADD_DEFINITIONS(-fPIC)
ENDIF()	

#-----------------------------------------------------------------------------

OPTION(BUILD_NR_DEV "To build code under development" OFF)
OPTION(BUILD_TESTING "To build the unit tests" OFF)

OPTION(USE_SSE "To use SSE computation in some case" OFF)
OPTION(USE_CUDA "To use CUDA for a GPU implementation of the code" OFF)
OPTION(USE_DOUBLE "To use double precision instead of single precision" OFF)
IF(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    IF(NOT CMAKE_SYSTEM_VERSION VERSION_GREATER 11.0 OR CMAKE_SYSTEM_VERSION VERSION_EQUAL 11.0.0)
        OPTION(USE_OPENMP "To use openMP for multi-CPU processing" OFF)
    ENDIF()
ENDIF()
#-----------------------------------------------------------------------------

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/nifti)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/reg-lib)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/zlib)

#-----------------------------------------------------------------------------

IF(USE_DOUBLE)
    ADD_DEFINITIONS(-D_USE_NR_DOUBLE)
    IF(USE_SSE)
        MESSAGE("Can not use double precision and SSE implementation concurrenlty")
        MESSAGE(FATAL_ERROR "Please turn USE_SSE OFF to use double precision")
    ENDIF(USE_SSE)
    IF(USE_CUDA)
        MESSAGE("Can not use double precision and CUDA implementation concurrenlty")
        MESSAGE(FATAL_ERROR "Please turn USE_CUDA OFF to use double precision")
    ENDIF(USE_CUDA)
ENDIF(USE_DOUBLE)

IF(USE_SSE)
    ADD_DEFINITIONS(-D_USE_SSE)
ENDIF(USE_SSE)

IF(BUILD_NR_DEV)
    ADD_DEFINITIONS(-D_BUILD_NR_DEV)
ENDIF(BUILD_NR_DEV)

IF(USE_OPENMP)
    IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        IF(CMAKE_SYSTEM_VERSION VERSION_GREATER 11.0 OR CMAKE_SYSTEM_VERSION VERSION_EQUAL 11.0.0)
            MESSAGE("OpenMP appears to be broken using Mac OS Lion")
            MESSAGE(FATAL_ERROR "Please turn USE_OPENMP OFF")
        ENDIF()
    ENDIF()
	
    EXECUTE_PROCESS(COMMAND ${CMAKE_C_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)

    IF(GCC_VERSION VERSION_GREATER 4.2 OR GCC_VERSION VERSION_EQUAL 4.2)
        SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    ENDIF()
	
    IF(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
        SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /openmp")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
    ENDIF()
ENDIF(USE_OPENMP)

#-----------------------------------------------------------------------------
IF(USE_CUDA)
   INCLUDE(${CMAKE_SOURCE_DIR}/CMake/cuda/FindCUDA.cmake)
   INCLUDE_DIRECTORIES(${CUDA_CUT_INCLUDE_DIR})
    CUDA_INCLUDE_DIRECTORIES(${CUDA_CUT_INCLUDE_DIR})
    ADD_DEFINITIONS(-D_USE_CUDA)
    SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-D_USE_CUDA")
    CUDA_INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/nifti)
ENDIF(USE_CUDA)
#-----------------------------------------------------------------------------
IF(IS_DIRECTORY ${CMAKE_SOURCE_DIR}/.svn)
    FIND_PACKAGE(Subversion)
    IF(Subversion_FOUND)
        Subversion_WC_INFO(${CMAKE_SOURCE_DIR} NiftyReg)
        ADD_DEFINITIONS(-D_SVN_REV=${NiftyReg_WC_REVISION})
    ENDIF(Subversion_FOUND)
ENDIF(IS_DIRECTORY ${CMAKE_SOURCE_DIR}/.svn)
#-----------------------------------------------------------------------------

SUBDIRS(zlib)
SUBDIRS(nifti)
SUBDIRS(reg-lib)
SUBDIRS(reg-apps)
if(BUILD_TESTING)
    ENABLE_TESTING()
    INCLUDE(${CMAKE_ROOT}/Modules/Dart.cmake)
    SUBDIRS(reg-test)
ENDIF(BUILD_TESTING)

#=============================================================================
