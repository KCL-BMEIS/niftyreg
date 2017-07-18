#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# Define the folder to use to store all data
set(DFOLDER ${CMAKE_BINARY_DIR}/reg-test/test-data)
#-----------------------------------------------------------------------------
# Set the 2D image input file
if(EXISTS "${DFOLDER}/refImg2D.nii.gz")
  set(TESTING_2D_FILE "${DFOLDER}/refImg2D.nii.gz" CACHE FILEPATH "2D nifti file used to generate testing data")
else(EXISTS "${DFOLDER}/refImg2D.nii.gz")
  set(TESTING_2D_FILE "" CACHE FILEPATH "2D nifti file used to generate testing data")
endif(EXISTS "${DFOLDER}/refImg2D.nii.gz")
if(NOT EXISTS "${TESTING_2D_FILE}" AND NOT EXISTS "${DFOLDER}/refImg2D.nii.gz")
  set(BUILD_TESTING OFF CACHE BOOL "To build the unit tests" FORCE)
  message(STATUS "The specified 2D image file does not exists")
  message(STATUS "TESTING_2D_FILE=${TESTING_2D_FILE}")
  message(SEND_ERROR "The BUILD_TESTING flag is turned OFF")
  return()
endif(NOT EXISTS "${TESTING_2D_FILE}" AND NOT EXISTS "${DFOLDER}/refImg2D.nii.gz")
#-----------------------------------------------------------------------------
# Set the 3D image input file
if(EXISTS "${DFOLDER}/refImg3D.nii.gz")
  set(TESTING_3D_FILE "${DFOLDER}/refImg3D.nii.gz" CACHE FILEPATH "3D nifti file used to generate testing data")
else(EXISTS "${DFOLDER}/refImg3D.nii.gz")
  set(TESTING_3D_FILE "" CACHE FILEPATH "3D nifti file used to generate testing data")
endif(EXISTS "${DFOLDER}/refImg3D.nii.gz")
if(NOT EXISTS "${TESTING_3D_FILE}" AND NOT EXISTS "${DFOLDER}/refImg3D.nii.gz")
  set(BUILD_TESTING OFF CACHE BOOL "To build the unit tests" FORCE)
  message(STATUS "The specified 3D image file does not exists")
  message(STATUS "TESTING_3D_FILE=${TESTING_3D_FILE}")
  message(SEND_ERROR "The BUILD_TESTING flag is turned OFF")
  return()
endif(NOT EXISTS "${TESTING_3D_FILE}" AND NOT EXISTS "${DFOLDER}/refImg3D.nii.gz")
#-----------------------------------------------------------------------------
# Run the matlab command if required
if(NOT EXISTS "${DFOLDER}/refImg2D.nii.gz" OR NOT EXISTS "${DFOLDER}/refImg3D.nii.gz" OR
   NOT EXISTS "${DFOLDER}/affine_mat2D.txt" OR NOT EXISTS "${DFOLDER}/affine_mat3D.txt" OR
   NOT EXISTS "${DFOLDER}/affine_def2D.nii.gz" OR NOT EXISTS "${DFOLDER}/affine_def3D.nii.gz" OR
   NOT EXISTS "${DFOLDER}/bspline_def2D.nii.gz" OR NOT EXISTS "${DFOLDER}/bspline_def3D.nii.gz" OR
   NOT EXISTS "${DFOLDER}/bspline_grid2D.nii.gz" OR NOT EXISTS "${DFOLDER}/bspline_grid3D.nii.gz" OR
   NOT EXISTS "${DFOLDER}/refImg2D_uchar.nii.gz" OR NOT EXISTS "${DFOLDER}/refImg3D_uchar.nii.gz" OR
   NOT EXISTS "${DFOLDER}/refImg2D_float.nii.gz" OR NOT EXISTS "${DFOLDER}/refImg3D_float.nii.gz" OR
   NOT EXISTS "${DFOLDER}/refImg2D_double.nii.gz" OR NOT EXISTS "${DFOLDER}/refImg3D_double.nii.gz" OR
   NOT EXISTS "${DFOLDER}/warped_nearest2D.nii.gz" OR NOT EXISTS "${DFOLDER}/warped_nearest3D.nii.gz" OR
   NOT EXISTS "${DFOLDER}/warped_linear2D.nii.gz" OR NOT EXISTS "${DFOLDER}/warped_linear3D.nii.gz" OR
   NOT EXISTS "${DFOLDER}/warped_cubic2D.nii.gz" OR NOT EXISTS "${DFOLDER}/warped_cubic3D.nii.gz" OR
   NOT EXISTS "${DFOLDER}/expectedBlockMatching_mat2D.txt" OR NOT EXISTS "${DFOLDER}/expectedBlockMatching_mat3D.txt" OR
   NOT EXISTS "${DFOLDER}/warpedBlockMatchingImg2D.nii.gz" OR NOT EXISTS "${DFOLDER}/warpedBlockMatchingImg3D.nii.gz" OR
   NOT EXISTS "${DFOLDER}/inputSVDMatrix.txt" OR NOT EXISTS "${DFOLDER}/expectedSMatrix.txt" OR
   NOT EXISTS "${DFOLDER}/inputMatrix1.txt" OR NOT EXISTS "${DFOLDER}/inputMatrix2.txt" OR
   NOT EXISTS "${DFOLDER}/expectedVMatrix.txt" OR NOT EXISTS "${DFOLDER}/expectedUMatrix.txt" OR
   NOT EXISTS "${DFOLDER}/expectedSubMatrix.txt" OR NOT EXISTS "${DFOLDER}/expectedAddMatrix.txt" OR
   NOT EXISTS "${DFOLDER}/expectedMulMatrix.txt" OR NOT EXISTS "${DFOLDER}/expectedInvMatrix.txt" OR
   NOT EXISTS "${DFOLDER}/expectedLogMatrix.txt" OR NOT EXISTS "${DFOLDER}/expectedExpMatrix.txt" OR
   NOT EXISTS "${DFOLDER}/expectedRigidLTS_2D_70.txt" OR NOT EXISTS "${DFOLDER}/expectedRigidLTS_3D_70.txt" OR
   NOT EXISTS "${DFOLDER}/expectedRigidLTS_2D_100.txt" OR NOT EXISTS "${DFOLDER}/expectedRigidLTS_3D_100.txt" OR
   NOT EXISTS "${DFOLDER}/expectedAffineLTS_2D_70.txt" OR NOT EXISTS "${DFOLDER}/expectedAffineLTS_3D_70.txt" OR
   NOT EXISTS "${DFOLDER}/expectedAffineLTS_2D_100.txt" OR NOT EXISTS "${DFOLDER}/expectedAffineLTS_3D_100.txt" OR
   NOT EXISTS "${DFOLDER}/expectedMINDDescriptor2D_1.nii.gz" OR NOT EXISTS "${DFOLDER}/expectedMINDDescriptor3D_1.nii.gz" OR
   NOT EXISTS "${DFOLDER}/expectedMINDDescriptor2D_2.nii.gz" OR NOT EXISTS "${DFOLDER}/expectedMINDDescriptor3D_2.nii.gz" OR
   NOT EXISTS "${DFOLDER}/P1_2D.txt" OR NOT EXISTS "${DFOLDER}/P1_3D.txt" OR
   NOT EXISTS "${DFOLDER}/P2_2D.txt" OR NOT EXISTS "${DFOLDER}/P2_3D.txt" OR
   NOT EXISTS "${DFOLDER}/expectedSSDValue2D.txt" OR NOT EXISTS "${DFOLDER}/expectedSSDValue3D.txt" OR
   NOT EXISTS "${DFOLDER}/expectedMINDSSDValue2D.txt" OR NOT EXISTS "${DFOLDER}/expectedMINDSSDValue3D.txt" OR
   NOT EXISTS "${DFOLDER}/expectedMINDSSCDescriptor2D_1.nii.gz" OR NOT EXISTS "${DFOLDER}/expectedMINDSSCDescriptor3D_1.nii.gz" OR
   NOT EXISTS "${DFOLDER}/expectedMINDSSCDescriptor2D_2.nii.gz" OR NOT EXISTS "${DFOLDER}/expectedMINDSSCDescriptor3D_2.nii.gz" OR
   NOT EXISTS "${DFOLDER}/expectedImageGradient2D.nii.gz" OR NOT EXISTS "${DFOLDER}/expectedImageGradient3D.nii.gz" OR
   NOT EXISTS "${DFOLDER}/convolution2D_mea.nii.gz" OR NOT EXISTS "${DFOLDER}/convolution3D_mea.nii.gz" OR
   NOT EXISTS "${DFOLDER}/convolution2D_lin.nii.gz" OR NOT EXISTS "${DFOLDER}/convolution3D_lin.nii.gz" OR
   NOT EXISTS "${DFOLDER}/convolution2D_gau.nii.gz" OR NOT EXISTS "${DFOLDER}/convolution3D_gau.nii.gz" OR
   NOT EXISTS "${DFOLDER}/convolution2D_spl.nii.gz" OR NOT EXISTS "${DFOLDER}/convolution3D_spl.nii.gz" OR
   NOT EXISTS "${DFOLDER}/le_spline_approx2D.txt" OR NOT EXISTS "${DFOLDER}/le_spline_approx3D.txt" OR
   NOT EXISTS "${DFOLDER}/le_spline_dense2D.txt" OR NOT EXISTS "${DFOLDER}/le_spline_dense3D.txt" OR
   NOT EXISTS "${DFOLDER}/le_field_dense2D.txt" OR NOT EXISTS "${DFOLDER}/le_field_dense3D.txt" OR
   NOT EXISTS "${DFOLDER}/le_grad_spline_approx2D.nii.gz" OR NOT EXISTS "${DFOLDER}/le_grad_spline_approx3D.nii.gz" OR
   NOT EXISTS "${DFOLDER}/le_grad_spline_dense2D.nii.gz" OR NOT EXISTS "${DFOLDER}/le_grad_spline_dense3D.nii.gz" OR
   NOT EXISTS "${DFOLDER}/le_grad_field_dense2D.nii.gz" OR NOT EXISTS "${DFOLDER}/le_grad_field_dense3D.nii.gz"
  )
  # Generate the test data using matlab
  find_package(Matlab)
  if(NOT Matlab_FOUND)
    set(BUILD_TESTING OFF CACHE BOOL "To build the unit tests" FORCE)
    message(STATUS "Matlab not found")
    message(SEND_ERROR "Matlab is required to generate the test data. The BUILD_TESTING flag is turned OFF")
    return()
  else(NOT Matlab_FOUND)
    message(STATUS "Found Matlab")
  endif(NOT Matlab_FOUND)
  message(STATUS "Generating the test data - Takes several minutes")
  if(MSVC)
    execute_process(COMMAND
      ${Matlab_ROOT_DIR}/bin/matlab -noFigureWindows -nosplash -wait -r "create_test_data('${TESTING_2D_FILE}', '${TESTING_3D_FILE}', '${DFOLDER}');exit"
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/reg-test/matlab_tests"
      OUTPUT_VARIABLE MATLAB_CMD_OUTPUT
      )
  else(MSVC)
    execute_process(COMMAND
      ${Matlab_ROOT_DIR}/bin/matlab "-nodisplay -nosplash -nodesktop -r \";create_test_data(\'${TESTING_2D_FILE}\',\'${TESTING_3D_FILE}\',\'${DFOLDER}\');exit\""
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/reg-test/matlab_tests"
      OUTPUT_VARIABLE MATLAB_CMD_OUTPUT
      )
  endif(MSVC)

  message(STATUS "Maltab command output:")
  message(STATUS "${MATLAB_CMD_OUTPUT}")
else()
  message(STATUS "All test data already exists")
endif()
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# Build the coverage test
if(NOT MSVC)
  option(WITH_COVERAGE "Set up the C, CXX and linker flags to run the coverage test" OFF)
  if(WITH_COVERAGE)
    set(CTEST_START_WITH_EMPTY_BINARY_DIRECTORY_ONCE TRUE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
    set(CMAKE_CXX_FLAGS_DEBUG
      "-g -O0 -Wall -W -Wunused-variable -Wunused-parameter -Wunused-function -Wunused -Wno-system-headers -Wno-deprecated -Woverloaded-virtual -Wwrite-strings -fprofile-arcs -ftest-coverage"
      CACHE STRING "Force the debug CXX flags for the coverage test" FORCE)
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG
      "-fprofile-arcs -ftest-coverage"
      CACHE STRING "Force the debug linker flags for the coverage test" FORCE)
    set(BUILD_ALL_DEP ON CACHE BOOL "All the dependencies are build" FORCE)
    configure_file(${CMAKE_SOURCE_DIR}/reg-test/CTestCustom.cmake.in
                   ${CMAKE_BINARY_DIR}/CTestCustom.cmake)
  endif(WITH_COVERAGE)
endif(NOT MSVC)
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
set(PLATFORM_LIST 0)
if(USE_CUDA)
  set(PLATFORM_LIST ${PLATFORM_LIST} 1)
endif(USE_CUDA)
if(USE_OPENCL)
  set(PLATFORM_LIST ${PLATFORM_LIST} 2)
endif(USE_OPENCL)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
set(EXEC reg_test_affine_deformation_field)
add_executable(${EXEC} ${EXEC}.cpp)
target_link_libraries(${EXEC} _reg_aladin)
foreach(CURRENT_PLATFORM ${PLATFORM_LIST})
  add_test(${EXEC}_2D_${CURRENT_PLATFORM} ${EXEC} ${DFOLDER}/refImg2D.nii.gz ${DFOLDER}/affine_mat2D.txt ${DFOLDER}/affine_def2D.nii.gz ${CURRENT_PLATFORM})
  add_test(${EXEC}_3D_${CURRENT_PLATFORM} ${EXEC} ${DFOLDER}/refImg3D.nii.gz ${DFOLDER}/affine_mat3D.txt ${DFOLDER}/affine_def3D.nii.gz ${CURRENT_PLATFORM})
endforeach(CURRENT_PLATFORM)
#-----------------------------------------------------------------------------
set(EXEC reg_test_bspline_deformation_field)
add_executable(${EXEC} ${EXEC}.cpp)
target_link_libraries(${EXEC} _reg_f3d)
foreach(CURRENT_PLATFORM "0")
  add_test(${EXEC}_2D_${CURRENT_PLATFORM} ${EXEC} ${DFOLDER}/refImg2D.nii.gz ${DFOLDER}/bspline_grid2D.nii.gz ${DFOLDER}/bspline_def2D.nii.gz 0 ${CURRENT_PLATFORM})
  add_test(${EXEC}_3D_${CURRENT_PLATFORM} ${EXEC} ${DFOLDER}/refImg3D.nii.gz ${DFOLDER}/bspline_grid3D.nii.gz ${DFOLDER}/bspline_def3D.nii.gz 0 ${CURRENT_PLATFORM})
  add_test(${EXEC}_comp_2D_${CURRENT_PLATFORM} ${EXEC} ${DFOLDER}/refImg2D.nii.gz ${DFOLDER}/bspline_grid2D.nii.gz ${DFOLDER}/bspline_def2D.nii.gz 1 ${CURRENT_PLATFORM})
  add_test(${EXEC}_comp_3D_${CURRENT_PLATFORM} ${EXEC} ${DFOLDER}/refImg3D.nii.gz ${DFOLDER}/bspline_grid3D.nii.gz ${DFOLDER}/bspline_def3D.nii.gz 1 ${CURRENT_PLATFORM})
endforeach(CURRENT_PLATFORM)
#-----------------------------------------------------------------------------
set(EXEC reg_test_blockMatching)
add_executable(${EXEC} ${EXEC}.cpp)
target_link_libraries(${EXEC} _reg_aladin)
foreach(CURRENT_PLATFORM ${PLATFORM_LIST})
  add_test(${EXEC}_2D_${CURRENT_PLATFORM} ${EXEC} ${DFOLDER}/refImg2D.nii.gz ${DFOLDER}/warpedBlockMatchingImg2D.nii.gz ${DFOLDER}/expectedBlockMatching_mat2D.txt ${CURRENT_PLATFORM})
  add_test(${EXEC}_3D_${CURRENT_PLATFORM} ${EXEC} ${DFOLDER}/refImg3D.nii.gz ${DFOLDER}/warpedBlockMatchingImg3D.nii.gz ${DFOLDER}/expectedBlockMatching_mat3D.txt ${CURRENT_PLATFORM})
endforeach(CURRENT_PLATFORM)
#-----------------------------------------------------------------------------
  set(EXEC reg_test_changeDataType)
  add_executable(${EXEC} ${EXEC}.cpp)
  target_link_libraries(${EXEC} _reg_aladin)
  add_test(${EXEC}_uchar_2D  ${EXEC} ${DFOLDER}/refImg2D.nii.gz uchar  ${DFOLDER}/refImg2D_uchar.nii.gz)
  add_test(${EXEC}_float_2D  ${EXEC} ${DFOLDER}/refImg2D.nii.gz float  ${DFOLDER}/refImg2D_float.nii.gz)
  add_test(${EXEC}_double_2D ${EXEC} ${DFOLDER}/refImg2D.nii.gz double ${DFOLDER}/refImg2D_double.nii.gz)
  add_test(${EXEC}_uchar_3D  ${EXEC} ${DFOLDER}/refImg3D.nii.gz uchar  ${DFOLDER}/refImg3D_uchar.nii.gz)
  add_test(${EXEC}_float_3D  ${EXEC} ${DFOLDER}/refImg3D.nii.gz float  ${DFOLDER}/refImg3D_float.nii.gz)
  add_test(${EXEC}_double_3D ${EXEC} ${DFOLDER}/refImg3D.nii.gz double ${DFOLDER}/refImg3D_double.nii.gz)
#-----------------------------------------------------------------------------
  set(EXEC reg_test_interpolation)
  add_executable(${EXEC} ${EXEC}.cpp)
  target_link_libraries(${EXEC} _reg_aladin)
  foreach(CURRENT_PLATFORM ${PLATFORM_LIST})
    add_test(${EXEC}_nea_2D_${CURRENT_PLATFORM} ${EXEC} ${DFOLDER}/refImg2D.nii.gz ${DFOLDER}/affine_def2D.nii.gz ${DFOLDER}/warped_nearest2D.nii.gz 0 ${CURRENT_PLATFORM})
    add_test(${EXEC}_nea_3D_${CURRENT_PLATFORM} ${EXEC} ${DFOLDER}/refImg3D.nii.gz ${DFOLDER}/affine_def3D.nii.gz ${DFOLDER}/warped_nearest3D.nii.gz 0 ${CURRENT_PLATFORM})
    add_test(${EXEC}_lin_2D_${CURRENT_PLATFORM} ${EXEC} ${DFOLDER}/refImg2D.nii.gz ${DFOLDER}/affine_def2D.nii.gz ${DFOLDER}/warped_linear2D.nii.gz 1 ${CURRENT_PLATFORM})
    add_test(${EXEC}_lin_3D_${CURRENT_PLATFORM} ${EXEC} ${DFOLDER}/refImg3D.nii.gz ${DFOLDER}/affine_def3D.nii.gz ${DFOLDER}/warped_linear3D.nii.gz 1 ${CURRENT_PLATFORM})
    add_test(${EXEC}_cub_2D_${CURRENT_PLATFORM} ${EXEC} ${DFOLDER}/refImg2D.nii.gz ${DFOLDER}/affine_def2D.nii.gz ${DFOLDER}/warped_cubic2D.nii.gz 3 ${CURRENT_PLATFORM})
    add_test(${EXEC}_cub_3D_${CURRENT_PLATFORM} ${EXEC} ${DFOLDER}/refImg3D.nii.gz ${DFOLDER}/affine_def3D.nii.gz ${DFOLDER}/warped_cubic3D.nii.gz 3 ${CURRENT_PLATFORM})
  endforeach(CURRENT_PLATFORM)
#-----------------------------------------------------------------------------
  set(EXEC reg_test_leastTrimmedSquares)
  add_executable(${EXEC} ${EXEC}.cpp)
  target_link_libraries(${EXEC} _reg_aladin)
  foreach(CURRENT_PLATFORM ${PLATFORM_LIST})
    add_test(${EXEC}_RigidLS2D_${CURRENT_PLATFORM} ${EXEC} ${DFOLDER}/P1_2D.txt ${DFOLDER}/P2_2D.txt 100 0 ${DFOLDER}/expectedRigidLTS_2D_100.txt ${CURRENT_PLATFORM})
    add_test(${EXEC}_AffineLS2D_${CURRENT_PLATFORM} ${EXEC} ${DFOLDER}/P1_2D.txt ${DFOLDER}/P2_2D.txt 100 1 ${DFOLDER}/expectedAffineLTS_2D_100.txt ${CURRENT_PLATFORM})
    add_test(${EXEC}_RigidLS3D_${CURRENT_PLATFORM} ${EXEC} ${DFOLDER}/P1_3D.txt ${DFOLDER}/P2_3D.txt 100 0 ${DFOLDER}/expectedRigidLTS_3D_100.txt ${CURRENT_PLATFORM})
    add_test(${EXEC}_AffineLS3D_${CURRENT_PLATFORM} ${EXEC} ${DFOLDER}/P1_3D.txt ${DFOLDER}/P2_3D.txt 100 1 ${DFOLDER}/expectedAffineLTS_3D_100.txt ${CURRENT_PLATFORM})
    add_test(${EXEC}_RigidLTS2D_${CURRENT_PLATFORM} ${EXEC} ${DFOLDER}/P1_2D.txt ${DFOLDER}/P2_2D.txt 70 0 ${DFOLDER}/expectedRigidLTS_2D_70.txt ${CURRENT_PLATFORM})
    add_test(${EXEC}_AffineLTS2D_${CURRENT_PLATFORM} ${EXEC} ${DFOLDER}/P1_2D.txt ${DFOLDER}/P2_2D.txt 70 1 ${DFOLDER}/expectedAffineLTS_2D_70.txt ${CURRENT_PLATFORM})
    add_test(${EXEC}_RigidLTS3D_${CURRENT_PLATFORM} ${EXEC} ${DFOLDER}/P1_3D.txt ${DFOLDER}/P2_3D.txt 70 0 ${DFOLDER}/expectedRigidLTS_3D_70.txt ${CURRENT_PLATFORM})
    add_test(${EXEC}_AffineLTS3D_${CURRENT_PLATFORM} ${EXEC} ${DFOLDER}/P1_3D.txt ${DFOLDER}/P2_3D.txt 70 1 ${DFOLDER}/expectedAffineLTS_3D_70.txt ${CURRENT_PLATFORM})
  endforeach(CURRENT_PLATFORM)
#-----------------------------------------------------------------------------
  set(EXEC reg_test_matrix_operation)
  add_executable(${EXEC} ${EXEC}.cpp)
  target_link_libraries(${EXEC} _reg_maths _reg_ReadWriteImage)
  add_test(${EXEC} ${EXEC} ${DFOLDER}/inputMatrix1.txt ${DFOLDER}/inputMatrix2.txt ${DFOLDER}/expectedMulMatrix.txt ${DFOLDER}/expectedAddMatrix.txt ${DFOLDER}/expectedSubMatrix.txt ${DFOLDER}/expectedExpMatrix.txt ${DFOLDER}/expectedLogMatrix.txt ${DFOLDER}/expectedInvMatrix.txt)
#-----------------------------------------------------------------------------
set(EXEC reg_test_svd)
add_executable(${EXEC} ${EXEC}.cpp)
target_link_libraries(${EXEC} _reg_maths _reg_ReadWriteImage)
add_test(${EXEC}_0 ${EXEC} ${DFOLDER}/inputSVDMatrix.txt ${DFOLDER}/expectedUMatrix.txt ${DFOLDER}/expectedSMatrix.txt ${DFOLDER}/expectedVMatrix.txt)
#-----------------------------------------------------------------------------
if(USE_CUDA OR USE_OPENCL)
  set(EXEC reg_test_coherence_blockMatching)
  add_executable(${EXEC} ${EXEC}.cpp)
  target_link_libraries(${EXEC} _reg_aladin)
  if(USE_CUDA)
    add_test(${EXEC}_2D_1 ${EXEC} ${DFOLDER}/refImg2D.nii.gz ${DFOLDER}/warped_cubic2D.nii.gz 1)
    add_test(${EXEC}_3D_1 ${EXEC} ${DFOLDER}/refImg3D.nii.gz ${DFOLDER}/warped_cubic3D.nii.gz 1)
  endif(USE_CUDA)
  if(USE_OPENCL)
    add_test(${EXEC}_2D_2 ${EXEC} ${DFOLDER}/refImg2D.nii.gz ${DFOLDER}/warped_cubic2D.nii.gz 2)
    add_test(${EXEC}_3D_2 ${EXEC} ${DFOLDER}/refImg3D.nii.gz ${DFOLDER}/warped_cubic3D.nii.gz 2)
  endif(USE_OPENCL)
  #
  set(EXEC reg_test_coherence_interpolation)
  add_executable(${EXEC} ${EXEC}.cpp)
  target_link_libraries(${EXEC} _reg_aladin)
  if(USE_CUDA)
    add_test(${EXEC}_nea_2D_1 ${EXEC} ${DFOLDER}/refImg2D.nii.gz ${DFOLDER}/affine_def2D.nii.gz 0 1)
    add_test(${EXEC}_lin_2D_1 ${EXEC} ${DFOLDER}/refImg2D.nii.gz ${DFOLDER}/affine_def2D.nii.gz 1 1)
    add_test(${EXEC}_cub_2D_1 ${EXEC} ${DFOLDER}/refImg2D.nii.gz ${DFOLDER}/affine_def2D.nii.gz 3 1)
    add_test(${EXEC}_nea_3D_1 ${EXEC} ${DFOLDER}/refImg3D.nii.gz ${DFOLDER}/affine_def3D.nii.gz 0 1)
    add_test(${EXEC}_lin_3D_1 ${EXEC} ${DFOLDER}/refImg3D.nii.gz ${DFOLDER}/affine_def3D.nii.gz 1 1)
    add_test(${EXEC}_cub_3D_1 ${EXEC} ${DFOLDER}/refImg3D.nii.gz ${DFOLDER}/affine_def3D.nii.gz 3 1)
  endif(USE_CUDA)
  if(USE_OPENCL)
    add_test(${EXEC}_nea_2D_2 ${EXEC} ${DFOLDER}/refImg2D.nii.gz ${DFOLDER}/affine_def2D.nii.gz 0 2)
    add_test(${EXEC}_lin_2D_2 ${EXEC} ${DFOLDER}/refImg2D.nii.gz ${DFOLDER}/affine_def2D.nii.gz 1 2)
    add_test(${EXEC}_cub_2D_2 ${EXEC} ${DFOLDER}/refImg2D.nii.gz ${DFOLDER}/affine_def2D.nii.gz 3 2)
    add_test(${EXEC}_nea_3D_2 ${EXEC} ${DFOLDER}/refImg3D.nii.gz ${DFOLDER}/affine_def3D.nii.gz 0 2)
    add_test(${EXEC}_lin_3D_2 ${EXEC} ${DFOLDER}/refImg3D.nii.gz ${DFOLDER}/affine_def3D.nii.gz 1 2)
    add_test(${EXEC}_cub_3D_2 ${EXEC} ${DFOLDER}/refImg3D.nii.gz ${DFOLDER}/affine_def3D.nii.gz 3 2)
  endif(USE_OPENCL)
  #
  set(EXEC reg_test_coherence_affine_deformation_field)
  add_executable(${EXEC} ${EXEC}.cpp)
  target_link_libraries(${EXEC} _reg_aladin)
  if(USE_CUDA)
    add_test(${EXEC}_2D_1 ${EXEC} ${DFOLDER}/refImg2D.nii.gz ${DFOLDER}/affine_mat2D.txt ${DFOLDER}/affine_def2D.nii.gz 1)
    add_test(${EXEC}_3D_1 ${EXEC} ${DFOLDER}/refImg3D.nii.gz ${DFOLDER}/affine_mat3D.txt ${DFOLDER}/affine_def3D.nii.gz 1)
  endif(USE_CUDA)
  if(USE_OPENCL)
    add_test(${EXEC}_2D_2 ${EXEC} ${DFOLDER}/refImg2D.nii.gz ${DFOLDER}/affine_mat2D.txt ${DFOLDER}/affine_def2D.nii.gz 2)
    add_test(${EXEC}_3D_2 ${EXEC} ${DFOLDER}/refImg3D.nii.gz ${DFOLDER}/affine_mat3D.txt ${DFOLDER}/affine_def3D.nii.gz 2)
  endif(USE_OPENCL)
endif(USE_CUDA OR USE_OPENCL)
#-----------------------------------------------------------------------------
set(EXEC reg_test_mindDescriptor)
add_executable(${EXEC} ${EXEC}.cpp)
target_link_libraries(${EXEC} _reg_f3d)
add_test(${EXEC}_MINDDescriptor_2D1 ${EXEC} ${DFOLDER}/refImg2D.nii.gz ${DFOLDER}/expectedMINDDescriptor2D_1.nii.gz)
add_test(${EXEC}_MINDDescriptor_3D1 ${EXEC} ${DFOLDER}/refImg3D.nii.gz ${DFOLDER}/expectedMINDDescriptor3D_1.nii.gz)
add_test(${EXEC}_MINDDescriptor_2D2 ${EXEC} ${DFOLDER}/warped_linear2D.nii.gz ${DFOLDER}/expectedMINDDescriptor2D_2.nii.gz)
add_test(${EXEC}_MINDDescriptor_3D2 ${EXEC} ${DFOLDER}/warped_linear3D.nii.gz ${DFOLDER}/expectedMINDDescriptor3D_2.nii.gz)
#-----------------------------------------------------------------------------
set(EXEC reg_test_mindsscDescriptor)
add_executable(${EXEC} ${EXEC}.cpp)
target_link_libraries(${EXEC} _reg_f3d)
add_test(${EXEC}_MINDSSCDescriptor_2D1 ${EXEC} ${DFOLDER}/refImg2D.nii.gz ${DFOLDER}/expectedMINDSSCDescriptor2D_1.nii.gz)
add_test(${EXEC}_MINDSSCDescriptor_3D1 ${EXEC} ${DFOLDER}/refImg3D.nii.gz ${DFOLDER}/expectedMINDSSCDescriptor3D_1.nii.gz)
add_test(${EXEC}_MINDSSCDescriptor_2D2 ${EXEC} ${DFOLDER}/warped_linear2D.nii.gz ${DFOLDER}/expectedMINDSSCDescriptor2D_2.nii.gz)
add_test(${EXEC}_MINDSSCDescriptor_3D2 ${EXEC} ${DFOLDER}/warped_linear3D.nii.gz ${DFOLDER}/expectedMINDSSCDescriptor3D_2.nii.gz)
#-----------------------------------------------------------------------------
set(EXEC reg_test_measure)
add_executable(${EXEC} ${EXEC}.cpp)
target_link_libraries(${EXEC} _reg_f3d)
add_test(${EXEC}_SSD_2D ${EXEC} ${DFOLDER}/expectedMINDDescriptor2D_1.nii.gz ${DFOLDER}/expectedMINDDescriptor2D_2.nii.gz SSD ${DFOLDER}/expectedSSDValue2D.txt)
add_test(${EXEC}_MINDSSD_2D ${EXEC} ${DFOLDER}/refImg2D.nii.gz ${DFOLDER}/warped_linear2D.nii.gz MIND ${DFOLDER}/expectedMINDSSDValue2D.txt)
add_test(${EXEC}_SSD_3D ${EXEC} ${DFOLDER}/expectedMINDDescriptor3D_1.nii.gz ${DFOLDER}/expectedMINDDescriptor3D_2.nii.gz SSD ${DFOLDER}/expectedSSDValue3D.txt)
add_test(${EXEC}_MINDSSD_3D ${EXEC} ${DFOLDER}/refImg3D.nii.gz ${DFOLDER}/warped_linear3D.nii.gz MIND ${DFOLDER}/expectedMINDSSDValue3D.txt)
#-----------------------------------------------------------------------------
set(EXEC reg_test_imageGradient)
add_executable(${EXEC} ${EXEC}.cpp)
target_link_libraries(${EXEC} _reg_f3d)
add_test(${EXEC}_imageGradient_SYD_2D ${EXEC} ${DFOLDER}/expectedMINDDescriptor2D_1.nii.gz ${DFOLDER}/expectedImageGradient2D.nii.gz 0)
add_test(${EXEC}_imageGradient_SYD_3D ${EXEC} ${DFOLDER}/expectedMINDDescriptor3D_1.nii.gz ${DFOLDER}/expectedImageGradient3D.nii.gz 0)
add_test(${EXEC}_imageGradient_SPL_2D ${EXEC} ${DFOLDER}/expectedMINDDescriptor2D_1.nii.gz ${DFOLDER}/expectedImageGradient2D.nii.gz 3)
add_test(${EXEC}_imageGradient_SPL_3D ${EXEC} ${DFOLDER}/expectedMINDDescriptor3D_1.nii.gz ${DFOLDER}/expectedImageGradient3D.nii.gz 3)
#-----------------------------------------------------------------------------
set(EXEC reg_test_convolution)
add_executable(${EXEC} ${EXEC}.cpp)
target_link_libraries(${EXEC} _reg_f3d)
add_test(${EXEC}_convolution_MEA_2D ${EXEC} ${DFOLDER}/refImg2D.nii.gz ${DFOLDER}/convolution2D_mea.nii.gz 0)
add_test(${EXEC}_convolution_LIN_2D ${EXEC} ${DFOLDER}/refImg2D.nii.gz ${DFOLDER}/convolution2D_lin.nii.gz 1)
add_test(${EXEC}_convolution_GAU_2D ${EXEC} ${DFOLDER}/refImg2D.nii.gz ${DFOLDER}/convolution2D_gau.nii.gz 2)
add_test(${EXEC}_convolution_SPL_2D ${EXEC} ${DFOLDER}/refImg2D.nii.gz ${DFOLDER}/convolution2D_spl.nii.gz 3)
add_test(${EXEC}_convolution_MEA_3D ${EXEC} ${DFOLDER}/refImg3D.nii.gz ${DFOLDER}/convolution3D_mea.nii.gz 0)
add_test(${EXEC}_convolution_LIN_3D ${EXEC} ${DFOLDER}/refImg3D.nii.gz ${DFOLDER}/convolution3D_lin.nii.gz 1)
add_test(${EXEC}_convolution_GAU_3D ${EXEC} ${DFOLDER}/refImg3D.nii.gz ${DFOLDER}/convolution3D_gau.nii.gz 2)
add_test(${EXEC}_convolution_SPL_3D ${EXEC} ${DFOLDER}/refImg3D.nii.gz ${DFOLDER}/convolution3D_spl.nii.gz 3)
#-----------------------------------------------------------------------------
set(EXEC reg_test_linearElasticity)
add_executable(${EXEC} ${EXEC}.cpp)
target_link_libraries(${EXEC} _reg_f3d)
add_test(${EXEC}_SPL_APP_2D ${EXEC} ${DFOLDER}/refImg2D.nii.gz ${DFOLDER}/bspline_grid2D.nii.gz ${DFOLDER}/le_spline_approx2D.txt 0)
add_test(${EXEC}_SPL_APP_3D ${EXEC} ${DFOLDER}/refImg3D.nii.gz ${DFOLDER}/bspline_grid3D.nii.gz ${DFOLDER}/le_spline_approx3D.txt 0)
add_test(${EXEC}_SPL_DEN_2D ${EXEC} ${DFOLDER}/refImg2D.nii.gz ${DFOLDER}/bspline_grid2D.nii.gz ${DFOLDER}/le_spline_dense2D.txt 1)
add_test(${EXEC}_SPL_DEN_3D ${EXEC} ${DFOLDER}/refImg3D.nii.gz ${DFOLDER}/bspline_grid3D.nii.gz ${DFOLDER}/le_spline_dense3D.txt 1)
add_test(${EXEC}_DEF_DEN_2D ${EXEC} ${DFOLDER}/refImg2D.nii.gz ${DFOLDER}/bspline_def2D.nii.gz ${DFOLDER}/le_field_dense2D.txt 2)
add_test(${EXEC}_DEF_DEN_3D ${EXEC} ${DFOLDER}/refImg3D.nii.gz ${DFOLDER}/bspline_def3D.nii.gz ${DFOLDER}/le_field_dense3D.txt 2)
#-----------------------------------------------------------------------------
set(EXEC reg_test_linearElasticityGradient)
add_executable(${EXEC} ${EXEC}.cpp)
target_link_libraries(${EXEC} _reg_f3d)
add_test(${EXEC}_SPL_APP_2D ${EXEC} ${DFOLDER}/refImg2D.nii.gz ${DFOLDER}/bspline_grid2D.nii.gz ${DFOLDER}/le_grad_spline_approx2D.nii.gz 0)
add_test(${EXEC}_SPL_APP_3D ${EXEC} ${DFOLDER}/refImg3D.nii.gz ${DFOLDER}/bspline_grid3D.nii.gz ${DFOLDER}/le_grad_spline_approx3D.nii.gz 0)
add_test(${EXEC}_SPL_DEN_2D ${EXEC} ${DFOLDER}/refImg2D.nii.gz ${DFOLDER}/bspline_grid2D.nii.gz ${DFOLDER}/le_grad_spline_dense2D.nii.gz 1)
add_test(${EXEC}_SPL_DEN_3D ${EXEC} ${DFOLDER}/refImg3D.nii.gz ${DFOLDER}/bspline_grid3D.nii.gz ${DFOLDER}/le_grad_spline_dense3D.nii.gz 1)
add_test(${EXEC}_DEF_DEN_2D ${EXEC} ${DFOLDER}/refImg2D.nii.gz ${DFOLDER}/bspline_def2D.nii.gz ${DFOLDER}/le_grad_field_dense2D.nii.gz 2)
add_test(${EXEC}_DEF_DEN_3D ${EXEC} ${DFOLDER}/refImg3D.nii.gz ${DFOLDER}/bspline_def3D.nii.gz ${DFOLDER}/le_grad_field_dense3D.nii.gz 2)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
set(EXEC reg_test_computation_time)
add_executable(${EXEC} ${EXEC}.cpp)
target_link_libraries(${EXEC} _reg_f3d)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
