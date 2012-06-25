PROJECT(NiftyReg)

#-----------------------------------------------------------------------------

cmake_minimum_required(VERSION 2.8.0)
IF("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION}" MATCHES "^2\\.8\\.0$")
 MARK_AS_ADVANCED(FORCE CMAKE_BACKWARDS_COMPATIBILITY)
ELSE("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION}" MATCHES "^2\\.8\\.0$")
 MARK_AS_ADVANCED(CLEAR CMAKE_BACKWARDS_COMPATIBILITY)
ENDIF("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION}" MATCHES "^2\\.8\\.0$")

#-----------------------------------------------------------------------------

if(COMMAND cmake_policy)
      cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

#-----------------------------------------------------------------------------

IF(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    ADD_DEFINITIONS(/W1)
ELSE(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    ADD_DEFINITIONS(-fPIC)
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Windows")

#-----------------------------------------------------------------------------

OPTION(BUILD_NR_DEV "To build NiftyReg code under development" ON)
OPTION(BUILD_TESTING "To build the unit tests" ON)
OPTION(BUILD_NR_SLICER_EXT "Where NiftyReg meets 3DSlicer" ${NiftyRegExtension_BUILD_SLICER_EXTENSION})

OPTION(USE_SSE "To use SSE computation in some case" OFF)
OPTION(USE_CUDA "To use CUDA for a GPU implementation of the code" OFF)
OPTION(USE_DOUBLE "To use double precision instead of single precision" OFF)
IF(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    IF(NOT CMAKE_SYSTEM_VERSION VERSION_GREATER 11.0 OR CMAKE_SYSTEM_VERSION VERSION_EQUAL 11.0.0)
        OPTION(USE_OPENMP "To use openMP for multi-CPU processing" ON)
    ENDIF()
ENDIF()
#-----------------------------------------------------------------------------

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/reg-lib)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/reg-io)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/reg-io/nifti)

#-----------------------------------------------------------------------------
# Z library
# Try first to find the z library on the system and built is from the sources if it can not be find
FIND_PACKAGE(ZLIB)
IF(ZLIB_FOUND)
    INCLUDE_DIRECTORIES(${ZLIB_INCLUDE_DIR})
        MESSAGE(STATUS "Found zlib - the z library will not be built")
ELSE(ZLIB_FOUND)
    INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/reg-io/zlib)
        MESSAGE(STATUS "zlib not found - the z library will be built")
ENDIF(ZLIB_FOUND)
#-----------------------------------------------------------------------------
## PNG support - First try to find the PNG library on the system and build it if it is not found
## I did not use the FindPNG.cmake here as the zlib is also included into the project

# Try to find the png library and header on the system
set(PNG_NAMES ${PNG_NAMES} png libpng png15 libpng15 png15d libpng15d png14 libpng14 png14d libpng14d png12 libpng12 png12d libpng12d)
FIND_LIBRARY(PNG_LIBRARY NAMES ${PNG_NAMES})
FIND_PATH(PNG_INCLUDE_DIR png.h
    /usr/local/include/libpng
    /sw/include
)
IF(CYGWIN)
    IF(NOT BUILD_SHARED_LIBS)
        set (PNG_DEFINITIONS -DPNG_STATIC)
    ENDIF(NOT BUILD_SHARED_LIBS)
ENDIF(CYGWIN)
# If the png library and header can not be found, it is build from the sources
IF(NOT PNG_LIBRARY OR NOT PNG_INCLUDE_DIR)
    MESSAGE(STATUS "libpng not found - the png library will be built")
    set(PNG_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/reg-io/png/lpng1510)
    set(PNG_LIBRARY png)
    set(BUILD_INTERNAL_PNG true)
ELSE(NOT PNG_LIBRARY OR NOT PNG_INCLUDE_DIR)
    MESSAGE(STATUS "Found libpng - the png library will not be built")
    set(BUILD_INTERNAL_PNG false)
ENDIF(NOT PNG_LIBRARY OR NOT PNG_INCLUDE_DIR)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/reg-io/png)
INCLUDE_DIRECTORIES(${PNG_INCLUDE_DIR})


#-----------------------------------------------------------------------------
# NRRD file format support - The nrrd file format has been embedded into the project
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR})
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/reg-io/nrrd)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/reg-io/nrrd/NrrdIO)

#-----------------------------------------------------------------------------
# Double can not be used with SSE as it has only been implemented for single precision
# Double can not be used with CUDA
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
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    ENDIF(GCC_VERSION VERSION_GREATER 4.2 OR GCC_VERSION VERSION_EQUAL 4.2)
	
    IF(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /openmp")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
    ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
ENDIF(USE_OPENMP)

#-----------------------------------------------------------------------------
IF(USE_CUDA)
    FIND_PACKAGE(CUDA)
    IF(CUDA_FOUND)
        ADD_DEFINITIONS(-D_USE_CUDA)
        MESSAGE(STATUS "Found CUDA - the GPU code will be compiled.")
    ELSE(CUDA_FOUND)
        MESSAGE(FATAL_ERROR "CUDA not found. Please turn the USE_CUDA flag off.")
    ENDIF(CUDA_FOUND)
ENDIF(USE_CUDA)
#-----------------------------------------------------------------------------
IF(IS_DIRECTORY ${CMAKE_SOURCE_DIR}/.svn)
    FIND_PACKAGE(Subversion)
    IF(Subversion_FOUND)
        Subversion_WC_INFO(${CMAKE_SOURCE_DIR} NiftyReg)
        ADD_DEFINITIONS(-D_SVN_REV=${NiftyReg_WC_REVISION})
        MESSAGE(STATUS "Found subversion - the revision number is added to the source.")
    ENDIF(Subversion_FOUND)
ENDIF(IS_DIRECTORY ${CMAKE_SOURCE_DIR}/.svn)
#-----------------------------------------------------------------------------

SUBDIRS(reg-io)
SUBDIRS(reg-lib)
SUBDIRS(reg-apps)
if(BUILD_TESTING)
    ENABLE_TESTING()
    INCLUDE(${CMAKE_ROOT}/Modules/Dart.cmake)
    SUBDIRS(reg-test)
ENDIF(BUILD_TESTING)
#-----------------------------------------------------------------------------

# add a target to generate API documentation with Doxygen
FIND_PACKAGE(Doxygen)
IF(DOXYGEN_FOUND)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(doc
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen" VERBATIM
    )
ENDIF(DOXYGEN_FOUND)
#=============================================================================
