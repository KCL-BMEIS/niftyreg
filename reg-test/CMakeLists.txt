
	# The test data are dowloaded
	set(DATA_FOLDER ${CMAKE_BINARY_DIR}/reg-test/test-data)
	if(NOT EXISTS "${DATA_FOLDER}/test_data.tar.gz")
		file(MAKE_DIRECTORY ${DATA_FOLDER})
		set(test_data_url "http://www.cs.ucl.ac.uk/staff/mmodat/niftyreg_test_data.tar.gz")
		message(STATUS "Downloading the test data: ${test_data_url}")
		file(DOWNLOAD
				 ${test_data_url}
				 "${DATA_FOLDER}/test_data.tar.gz"
				 SHOW_PROGRESS STATUS download_status)
		list(GET download_status 0 test_data_download_status)
		if(${test_data_download_status})
			message(WARNING "Error when downloading the test data, forcing BUILD_TESTING to OFF")
			set(BUILD_TESTING OFF CACHE BOOL "To build the unit tests" FORCE)
			endif(${test_data_download_status})
	endif(NOT EXISTS "${DATA_FOLDER}/test_data.tar.gz")

	if(NOT EXISTS "${DATA_FOLDER}/refImg2D.nii.gz")
		message(STATUS "Extracting test data in ${DATA_FOLDER}")
		execute_process(
			COMMAND ${CMAKE_COMMAND} -E tar xzf ${DATA_FOLDER}/test_data.tar.gz
			WORKING_DIRECTORY ${DATA_FOLDER}
			ERROR_VARIABLE test_data_extract_status
		)
		if(NOT ${test_data_extract_status} STREQUAL "")
			message(WARNING "Error when extracting the test data, forcing BUILD_TESTING to OFF")
			set(BUILD_TESTING OFF CACHE BOOL "To build the unit tests" FORCE)
		endif(NOT ${test_data_extract_status} STREQUAL "")
	endif(NOT EXISTS "${DATA_FOLDER}/refImg2D.nii.gz")

	set(EXEC reg_test_affine_deformation_field)
	add_executable(${EXEC} ${EXEC}.cpp)
	target_link_libraries(${EXEC} _reg_f3d)
	add_test(${EXEC}_2D ${EXEC} ${DATA_FOLDER}/refImg2D.nii.gz ${DATA_FOLDER}/affine_mat2D.txt ${DATA_FOLDER}/affine_def2D.nii.gz)
	add_test(${EXEC}_3D ${EXEC} ${DATA_FOLDER}/refImg3D.nii.gz ${DATA_FOLDER}/affine_mat3D.txt ${DATA_FOLDER}/affine_def3D.nii.gz)

	set(EXEC reg_test_nonlinear_deformation_field)
	add_executable(${EXEC} ${EXEC}.cpp)
	target_link_libraries(${EXEC} _reg_f3d)
	add_test(${EXEC}_2D ${EXEC} ${DATA_FOLDER}/refImg2D.nii.gz ${DATA_FOLDER}/nonlin_cpp2D.nii.gz ${DATA_FOLDER}/nonlin_def2D.nii.gz)
	add_test(${EXEC}_3D ${EXEC} ${DATA_FOLDER}/refImg3D.nii.gz ${DATA_FOLDER}/nonlin_cpp3D.nii.gz ${DATA_FOLDER}/nonlin_def3D.nii.gz)

	set(EXEC reg_test_interpolation)
	add_executable(${EXEC} ${EXEC}.cpp)
	target_link_libraries(${EXEC} _reg_f3d)
	add_test(${EXEC}_0_2D ${EXEC} ${DATA_FOLDER}/floImg2D.nii.gz ${DATA_FOLDER}/nonlin_def2D.nii.gz ${DATA_FOLDER}/warped_nea2D.nii.gz 0)
	add_test(${EXEC}_0_3D ${EXEC} ${DATA_FOLDER}/floImg3D.nii.gz ${DATA_FOLDER}/nonlin_def3D.nii.gz ${DATA_FOLDER}/warped_nea3D.nii.gz 0)
	add_test(${EXEC}_1_2D ${EXEC} ${DATA_FOLDER}/floImg2D.nii.gz ${DATA_FOLDER}/nonlin_def2D.nii.gz ${DATA_FOLDER}/warped_lin2D.nii.gz 1)
	add_test(${EXEC}_1_3D ${EXEC} ${DATA_FOLDER}/floImg3D.nii.gz ${DATA_FOLDER}/nonlin_def3D.nii.gz ${DATA_FOLDER}/warped_lin3D.nii.gz 1)
	add_test(${EXEC}_3_2D ${EXEC} ${DATA_FOLDER}/floImg2D.nii.gz ${DATA_FOLDER}/nonlin_def2D.nii.gz ${DATA_FOLDER}/warped_cub2D.nii.gz 3)
	add_test(${EXEC}_3_3D ${EXEC} ${DATA_FOLDER}/floImg3D.nii.gz ${DATA_FOLDER}/nonlin_def3D.nii.gz ${DATA_FOLDER}/warped_cub3D.nii.gz 3)
	add_test(${EXEC}_4_2D ${EXEC} ${DATA_FOLDER}/floImg2D.nii.gz ${DATA_FOLDER}/nonlin_def2D.nii.gz ${DATA_FOLDER}/warped_sin2D.nii.gz 4)
	add_test(${EXEC}_4_3D ${EXEC} ${DATA_FOLDER}/floImg3D.nii.gz ${DATA_FOLDER}/nonlin_def3D.nii.gz ${DATA_FOLDER}/warped_sin3D.nii.gz 4)

	set(EXEC reg_test_compose_deformation_field)
	add_executable(${EXEC} ${EXEC}.cpp)
	target_link_libraries(${EXEC} _reg_f3d)
	add_test(${EXEC}_2D ${EXEC} ${DATA_FOLDER}/nonlin_def2D.nii.gz ${DATA_FOLDER}/nonlin_com2D.nii.gz)
	add_test(${EXEC}_3D ${EXEC} ${DATA_FOLDER}/nonlin_def3D.nii.gz ${DATA_FOLDER}/nonlin_com3D.nii.gz)

	set(EXEC reg_test_fullAffine)
	add_executable(${EXEC} ${EXEC}.cpp)
	target_link_libraries(${EXEC} _reg_aladin)
	add_test(${EXEC}_2D ${EXEC} ${DATA_FOLDER}/refImg2D.nii.gz ${DATA_FOLDER}/floImg2D.nii.gz ${DATA_FOLDER}/affine_mat2D.txt)
	add_test(${EXEC}_3D ${EXEC} ${DATA_FOLDER}/refImg3D.nii.gz ${DATA_FOLDER}/floImg3D.nii.gz ${DATA_FOLDER}/affine_mat3D.txt)

	set(EXEC reg_test_fullNonlinear)
	add_executable(${EXEC} ${EXEC}.cpp)
	target_link_libraries(${EXEC} _reg_f3d)
	add_test(${EXEC}_2D ${EXEC} ${DATA_FOLDER}/refImg2D.nii.gz ${DATA_FOLDER}/floImg2D.nii.gz ${DATA_FOLDER}/affine_mat2D.txt ${DATA_FOLDER}/nonlin_cpp2D.nii.gz)
	add_test(${EXEC}_3D ${EXEC} ${DATA_FOLDER}/refImg3D.nii.gz ${DATA_FOLDER}/floImg3D.nii.gz ${DATA_FOLDER}/affine_mat3D.txt ${DATA_FOLDER}/nonlin_cpp3D.nii.gz)

	set(EXEC reg_test_matrix_operation)
	add_executable(${EXEC} ${EXEC}.cpp)
	target_link_libraries(${EXEC} _reg_maths reg_nifti z)
	add_test(${EXEC} ${EXEC})

	set(EXEC reg_test_svd)
	add_executable(${EXEC} ${EXEC}.cpp)
	target_link_libraries(${EXEC} _reg_maths)
	add_test(${EXEC} ${EXEC})

	set(EXEC reg_test_measure)
	add_executable(${EXEC} ${EXEC}.cpp)
	target_link_libraries(${EXEC} _reg_f3d)
	add_test(${EXEC}_LNCC_2D ${EXEC} ${DATA_FOLDER}/refImg2D.nii.gz ${DATA_FOLDER}/warped_sin2D.nii.gz LNCC)
	add_test(${EXEC}_NMI_2D ${EXEC} ${DATA_FOLDER}/refImg2D.nii.gz ${DATA_FOLDER}/warped_sin2D.nii.gz NMI)
	add_test(${EXEC}_SSD_2D ${EXEC} ${DATA_FOLDER}/refImg2D.nii.gz ${DATA_FOLDER}/warped_sin2D.nii.gz SSD)
	add_test(${EXEC}_LNCC_3D ${EXEC} ${DATA_FOLDER}/refImg3D.nii.gz ${DATA_FOLDER}/warped_sin3D.nii.gz LNCC)
	add_test(${EXEC}_NMI_3D ${EXEC} ${DATA_FOLDER}/refImg3D.nii.gz ${DATA_FOLDER}/warped_sin3D.nii.gz NMI)
	add_test(${EXEC}_SSD_3D ${EXEC} ${DATA_FOLDER}/refImg3D.nii.gz ${DATA_FOLDER}/warped_sin3D.nii.gz SSD)
	
	#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
set(EXEC clInfo)
add_executable(${EXEC} ${EXEC}.cpp)
target_link_libraries(${EXEC} _reg_aladin)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
set(EXEC testOptimizerKernel)
add_executable(${EXEC} ${EXEC}.cpp)
target_link_libraries(${EXEC} _reg_aladin)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
set(EXEC testConvolutionKernel)
add_executable(${EXEC} ${EXEC}.cpp)
target_link_libraries(${EXEC} _reg_aladin)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
set(EXEC testAffineDeformationKernel)
add_executable(${EXEC} ${EXEC}.cpp)
target_link_libraries(${EXEC} _reg_aladin)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
set(EXEC testResamplingKernel)
add_executable(${EXEC} ${EXEC}.cpp)
target_link_libraries(${EXEC} _reg_aladin)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
set(EXEC testBlockMatchingKernel)
add_executable(${EXEC} ${EXEC}.cpp)
target_link_libraries(${EXEC} _reg_aladin)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
if(USE_OPENCL)
  set(EXEC reg_test_affine_deformation_field_cl)
  add_executable(${EXEC} ${EXEC}.cpp)
  target_link_libraries(${EXEC} _reg_aladin)
  add_test(${EXEC}_3D ${EXEC} ${DATA_FOLDER}/refImg3D.nii.gz ${DATA_FOLDER}/affine_mat3D.txt ${DATA_FOLDER}/affine_def3D.nii.gz)
endif(USE_OPENCL)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
if(USE_CUDA)
  set(EXEC reg_test_affine_deformation_field_cuda)
  add_executable(${EXEC} ${EXEC}.cpp)
  target_link_libraries(${EXEC} _reg_aladin)
  add_test(${EXEC}_3D ${EXEC} ${DATA_FOLDER}/refImg3D.nii.gz ${DATA_FOLDER}/affine_mat3D.txt ${DATA_FOLDER}/affine_def3D.nii.gz)
endif(USE_CUDA)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
set(EXEC reg_test_interpolation_cl)
add_executable(${EXEC} ${EXEC}.cpp)
target_link_libraries(${EXEC} _reg_aladin)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
set(EXEC reg_test_interpolation_cuda)
add_executable(${EXEC} ${EXEC}.cpp)
target_link_libraries(${EXEC} _reg_aladin)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
