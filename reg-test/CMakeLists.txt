SET(DATA_FOLDER ${CMAKE_SOURCE_DIR}/reg-test/reg-test-data)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
SET(EXEC reg_test_interp)
ADD_EXECUTABLE(${EXEC} ${EXEC}.cpp)
TARGET_LINK_LIBRARIES(${EXEC} _reg_resampling _reg_maths reg_nifti ${ZLIB})
#-----------------------------------------------------------------------------
ADD_TEST(interpolation_3D_NN ${EXEC} ${DATA_FOLDER}/brainweb_3D.nii.gz
    ${DATA_FOLDER}/test_DEF_BW_3D.nii.gz 0
    ${DATA_FOLDER}/test_warped_BW_3D_NN.nii.gz)
ADD_TEST(interpolation_3D_LIN ${EXEC} ${DATA_FOLDER}/brainweb_3D.nii.gz
    ${DATA_FOLDER}/test_DEF_BW_3D.nii.gz 1
    ${DATA_FOLDER}/test_warped_BW_3D_LIN.nii.gz)
ADD_TEST(interpolation_3D_SPL ${EXEC} ${DATA_FOLDER}/brainweb_3D.nii.gz
    ${DATA_FOLDER}/test_DEF_BW_3D.nii.gz 3
    ${DATA_FOLDER}/test_warped_BW_3D_SPL.nii.gz)
ADD_TEST(interpolation_2D_NN ${EXEC} ${DATA_FOLDER}/brainweb_2D.nii.gz
    ${DATA_FOLDER}/test_DEF_BW_2D.nii.gz 0
    ${DATA_FOLDER}/test_warped_BW_2D_NN.nii.gz)
ADD_TEST(interpolation_2D_LIN ${EXEC} ${DATA_FOLDER}/brainweb_2D.nii.gz
    ${DATA_FOLDER}/test_DEF_BW_2D.nii.gz 1
    ${DATA_FOLDER}/test_warped_BW_2D_LIN.nii.gz)
ADD_TEST(interpolation_2D_SPL ${EXEC} ${DATA_FOLDER}/brainweb_2D.nii.gz
    ${DATA_FOLDER}/test_DEF_BW_2D.nii.gz 3
    ${DATA_FOLDER}/test_warped_BW_2D_SPL.nii.gz)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
SET(EXEC reg_test_mat44_operations)
ADD_EXECUTABLE(${EXEC} ${EXEC}.cpp)
TARGET_LINK_LIBRARIES(${EXEC} _reg_maths reg_nifti ${ZLIB})
#-----------------------------------------------------------------------------
ADD_TEST(mat44_operations ${EXEC})
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------



