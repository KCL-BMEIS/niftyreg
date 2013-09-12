set(DATA_FOLDER ${CMAKE_SOURCE_DIR}/reg-test/reg-test-data)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
set(EXEC reg_test_interp)
add_executable(${EXEC} ${EXEC}.cpp)
target_link_libraries(${EXEC} _reg_f3d)
#-----------------------------------------------------------------------------
add_test(interpolation_3D_NN ${EXEC} ${DATA_FOLDER}/brainweb_3D.nii.gz
    ${DATA_FOLDER}/test_DEF_BW_3D.nii.gz 0
    ${DATA_FOLDER}/test_warped_BW_3D_NN.nii.gz)
add_test(interpolation_3D_LIN ${EXEC} ${DATA_FOLDER}/brainweb_3D.nii.gz
    ${DATA_FOLDER}/test_DEF_BW_3D.nii.gz 1
    ${DATA_FOLDER}/test_warped_BW_3D_LIN.nii.gz)
add_test(interpolation_3D_SPL ${EXEC} ${DATA_FOLDER}/brainweb_3D.nii.gz
    ${DATA_FOLDER}/test_DEF_BW_3D.nii.gz 3
    ${DATA_FOLDER}/test_warped_BW_3D_SPL.nii.gz)
add_test(interpolation_2D_NN ${EXEC} ${DATA_FOLDER}/brainweb_2D.nii.gz
    ${DATA_FOLDER}/test_DEF_BW_2D.nii.gz 0
    ${DATA_FOLDER}/test_warped_BW_2D_NN.nii.gz)
add_test(interpolation_2D_LIN ${EXEC} ${DATA_FOLDER}/brainweb_2D.nii.gz
    ${DATA_FOLDER}/test_DEF_BW_2D.nii.gz 1
    ${DATA_FOLDER}/test_warped_BW_2D_LIN.nii.gz)
add_test(interpolation_2D_SPL ${EXEC} ${DATA_FOLDER}/brainweb_2D.nii.gz
    ${DATA_FOLDER}/test_DEF_BW_2D.nii.gz 3
    ${DATA_FOLDER}/test_warped_BW_2D_SPL.nii.gz)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
set(EXEC reg_test_dti)
add_executable(${EXEC} ${EXEC}.cpp)
target_link_libraries(${EXEC} _reg_f3d)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
set(EXEC reg_test_mat44_operations)
add_executable(${EXEC} ${EXEC}.cpp)
target_link_libraries(${EXEC} _reg_f3d)
#-----------------------------------------------------------------------------
add_test(mat44_operations ${EXEC})
#-----------------------------------------------------------------------------
set(EXEC reg_test_gradient)
add_executable(${EXEC} ${EXEC}.cpp)
target_link_libraries(${EXEC} _reg_f3d)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
if(USE_CUDA)
    set(EXEC reg_test_def_gpu)
    cuda_add_executable(${EXEC} ${EXEC}.cpp)
        target_link_libraries(${EXEC} _reg_f3d)
    foreach(dim 2 3)
        foreach(type def comp exp be beg ajac ajac jac ajacg jacg)
            add_test(${EXEC}_${dim}D_${type} ${EXEC} ${dim} ${type})
        endforeach(type)
    endforeach(dim)
#-----------------------------------------------------------------------------
	set(EXEC reg_test_measure_gpu)
	cuda_add_executable(${EXEC} ${EXEC}.cpp)
	target_link_libraries(${EXEC} _reg_f3d)
	foreach(dim 2 3)
		foreach(type spa nmig ssd ssdg)
			add_test(${EXEC}_${dim}D_${type} ${EXEC} ${dim} ${type})
		endforeach(type)
	endforeach(dim)
#-----------------------------------------------------------------------------
    set(EXEC reg_test_convolution_gpu)
    cuda_add_executable(${EXEC} ${EXEC}.cpp)
    target_link_libraries(${EXEC} _reg_f3d)
    foreach(dim 2 3)
        foreach(type gaussian spline)
            add_test(${EXEC}_${dim}D_${type} ${EXEC} ${dim} ${type})
        endforeach(type)
    endforeach(dim)
#-----------------------------------------------------------------------------
    set(EXEC reg_test_optimiser_gpu)
    cuda_add_executable(${EXEC} ${EXEC}.cpp)
    target_link_libraries(${EXEC} _reg_f3d)
    foreach(dim 2 3)
        foreach(type descent conjugate)
                add_test(${EXEC}_${dim}D_${type} ${EXEC} ${dim} ${type})
        endforeach(type)
    endforeach(dim)
#-----------------------------------------------------------------------------
    set(EXEC reg_test_interp_gpu)
    cuda_add_executable(${EXEC} ${EXEC}.cpp)
    target_link_libraries(${EXEC} _reg_f3d)
    add_test(reg_test_interp_gpu_2D_lin ${EXEC}
        ${DATA_FOLDER}/brainweb_2D.nii.gz
        ${DATA_FOLDER}/test_DEF_BW_2D.nii.gz 2 lin)
    add_test(reg_test_interp_gpu_3D_lin ${EXEC}
        ${DATA_FOLDER}/brainweb_3D.nii.gz
        ${DATA_FOLDER}/test_DEF_BW_3D.nii.gz 3 lin)
    add_test(reg_test_interp_gpu_2D_lgrad ${EXEC}
        ${DATA_FOLDER}/brainweb_2D.nii.gz
        ${DATA_FOLDER}/test_DEF_BW_2D.nii.gz 2 lgrad)
    add_test(reg_test_interp_gpu_3D_lgrad ${EXEC}
        ${DATA_FOLDER}/brainweb_3D.nii.gz
		${DATA_FOLDER}/test_DEF_BW_3D.nii.gz 3 lgrad)
#-----------------------------------------------------------------------------
endif(USE_CUDA)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

