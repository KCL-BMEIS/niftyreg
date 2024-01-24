find_package(Catch2 3)
if(NOT Catch2_FOUND)
  set(BUILD_TESTING OFF CACHE BOOL "To build the unit tests" FORCE)
  message(STATUS "Catch2 not found")
  message(SEND_ERROR "Catch2 is required to generate the unit test.
  The BUILD_TESTING flag is turned OFF")
  return()
else(NOT Catch2_FOUND)
  message(STATUS "Found Catch2")
endif(NOT Catch2_FOUND)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# Build the coverage test
option(WITH_COVERAGE "Set up the C, CXX and linker flags to run the coverage test" OFF)
if(WITH_COVERAGE)
  if(NOT MSVC)
    # Check prerequisites
    find_program(LCOV lcov REQUIRED)
    find_program(GENHTML genhtml REQUIRED)

    if(NOT LCOV)
      message(FATAL_ERROR "lcov not found! Aborting...")
    endif()

    if(NOT GENHTML)
      message(FATAL_ERROR "genhtml not found! Aborting...")
    endif()

    # Set the build type to debug
    set(CTEST_START_WITH_EMPTY_BINARY_DIRECTORY_ONCE TRUE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
    set(BUILD_ALL_DEP ON CACHE BOOL "All the dependencies are build" FORCE)

    # Set the flags for coverage
    set(CMAKE_CXX_FLAGS_DEBUG
      "-g -O0 -coverage"
      CACHE STRING "Force the debug CXX flags for the coverage test" FORCE)
    set(CMAKE_C_FLAGS_DEBUG
      ${CMAKE_CXX_FLAGS_DEBUG}
      CACHE STRING "Force the debug C flags for the coverage test" FORCE)

    # Add the coverage target
    add_custom_target(coverage
      # Gather data only for the reg-lib directory
      COMMAND ${LCOV} --directory . --capture --output-file coverage.info --include '*/reg-lib/*'
      # Generate report
      COMMAND ${GENHTML} --demangle-cpp -o coverage coverage.info
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

    # Add the clean target
    add_custom_target(clean_coverage
      COMMAND ${LCOV} --directory . --zerocounters
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  else(NOT MSVC)
    # Check prerequisites
    find_program(OPENCPPCOVERAGE OpenCppCoverage REQUIRED)

    if(NOT OPENCPPCOVERAGE)
      message(FATAL_ERROR "OpenCppCoverage not found! Aborting...")
    endif()

    # Set the build type to debug
    set(CTEST_START_WITH_EMPTY_BINARY_DIRECTORY_ONCE TRUE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
    set(BUILD_ALL_DEP ON CACHE BOOL "All the dependencies are build" FORCE)

    # Only include the reg-lib directory as coverage source
    string(REPLACE "/" "\\" COVERAGE_SOURCE "${CMAKE_SOURCE_DIR}/reg-lib")

    # Add the coverage target
    add_custom_target(coverage
      # Gather data only for the reg-lib directory
      COMMAND ${OPENCPPCOVERAGE} --sources=${COVERAGE_SOURCE} --cover_children -- ctest -C Debug
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  endif(NOT MSVC)
endif(WITH_COVERAGE)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# Set the build name
set(CTEST_BUILD_NAME "${CMAKE_SYSTEM}_${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}_cmake-${CMAKE_VERSION}_${CMAKE_BUILD_TYPE}")
if(USE_SSE)
  set(CTEST_BUILD_NAME "${CTEST_BUILD_NAME}_sse")
endif(USE_SSE)
if(USE_OPENMP)
  set(CTEST_BUILD_NAME "${CTEST_BUILD_NAME}_openmp")
endif(USE_OPENMP)
if(USE_CUDA)
  set(CTEST_BUILD_NAME "${CTEST_BUILD_NAME}_cuda-${CUDA_VERSION}")
endif(USE_CUDA)
if(USE_OPENCL)
  set(CTEST_BUILD_NAME "${CTEST_BUILD_NAME}_opencl")
endif(USE_OPENCL)
if(NOT MSVC)
  unset(BUILDNAME CACHE)
  unset(BUILDNAME)
  set(BUILDNAME ${CTEST_BUILD_NAME} CACHE STRING "Build name variable for CDash" FORCE)
else(MSVC)
  set(BUILDNAME ${CTEST_BUILD_NAME} CACHE STRING "Build name variable for CDash")
  message(STATUS "The buildname might need manual editing")
endif(NOT MSVC)
mark_as_advanced(BUILDNAME)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
include(CTest)
include(Catch)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
set(EXEC_LIST reg_test_affineDeformationField)
set(EXEC_LIST reg_test_be ${EXEC_LIST})
set(EXEC_LIST reg_test_blockMatching ${EXEC_LIST})
set(EXEC_LIST reg_test_conjugateGradient ${EXEC_LIST})
# set(EXEC_LIST reg_test_getDeformationField ${EXEC_LIST})
set(EXEC_LIST reg_test_composeField ${EXEC_LIST})
set(EXEC_LIST reg_test_imageGradient ${EXEC_LIST})
set(EXEC_LIST reg_test_interpolation ${EXEC_LIST})
set(EXEC_LIST reg_test_lncc ${EXEC_LIST})
set(EXEC_LIST reg_test_nmi ${EXEC_LIST})
set(EXEC_LIST reg_test_nmi_gradient ${EXEC_LIST})
set(EXEC_LIST reg_test_normaliseGradient ${EXEC_LIST})
set(EXEC_LIST reg_test_regr_getDeformationField ${EXEC_LIST})
set(EXEC_LIST reg_test_voxelCentricToNodeCentric ${EXEC_LIST})
if(USE_CUDA)
  set(EXEC_LIST reg_test_regr_approxBendingEnergyGradient ${EXEC_LIST})
  set(EXEC_LIST reg_test_regr_approxLinearEnergyGradient ${EXEC_LIST})
  set(EXEC_LIST reg_test_regr_blockMatching ${EXEC_LIST})
  set(EXEC_LIST reg_test_regr_exponentiateGradient ${EXEC_LIST})
  set(EXEC_LIST reg_test_regr_kernelConvolution ${EXEC_LIST})
  set(EXEC_LIST reg_test_regr_lts ${EXEC_LIST})
  set(EXEC_LIST reg_test_regr_measure ${EXEC_LIST})
  set(EXEC_LIST reg_test_regr_resampleGradient ${EXEC_LIST})
  set(EXEC_LIST reg_test_regr_symmetriseVelocityFields ${EXEC_LIST})
  set(EXEC_LIST reg_test_regr_updateVelocityField ${EXEC_LIST})
endif(USE_CUDA)

foreach(EXEC ${EXEC_LIST})
  add_executable(${EXEC} ${EXEC}.cpp)
  target_link_libraries(${EXEC} PRIVATE Catch2::Catch2WithMain _reg_aladin _reg_f3d)
  catch_discover_tests(${EXEC})
endforeach(EXEC)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------