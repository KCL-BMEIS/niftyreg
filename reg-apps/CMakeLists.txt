IF(BUILD_NIFTYREG)
#-----------------------------------------------------------------------------
        ADD_EXECUTABLE(reg_average reg_average.cpp)
        TARGET_LINK_LIBRARIES(reg_average _reg_tools reg_nifti ${ZLIB})
        INSTALL_TARGETS(/bin reg_average)
#-----------------------------------------------------------------------------
        ADD_EXECUTABLE(reg_tools reg_tools.cpp)
        TARGET_LINK_LIBRARIES(reg_tools _reg_resampling _reg_localTransformation _reg_tools _reg_globalTransformation reg_nifti ${ZLIB})
        INSTALL_TARGETS(/bin reg_tools)
#-----------------------------------------------------------------------------
	ADD_EXECUTABLE(reg_resample reg_resample.cpp)
        TARGET_LINK_LIBRARIES(reg_resample _reg_resampling _reg_localTransformation _reg_tools _reg_globalTransformation reg_nifti ${ZLIB})
	INSTALL_TARGETS(/bin reg_resample)
#-----------------------------------------------------------------------------
	ADD_EXECUTABLE(reg_transform reg_transform.cpp)
        TARGET_LINK_LIBRARIES(reg_transform _reg_resampling _reg_localTransformation _reg_tools _reg_globalTransformation _reg_maths reg_nifti ${ZLIB})
	INSTALL_TARGETS(/bin reg_transform)
#-----------------------------------------------------------------------------
	ADD_EXECUTABLE(reg_jacobian reg_jacobian.cpp)
        TARGET_LINK_LIBRARIES(reg_jacobian _reg_resampling _reg_localTransformation _reg_tools _reg_globalTransformation reg_nifti ${ZLIB})
	INSTALL_TARGETS(/bin reg_jacobian)
#-----------------------------------------------------------------------------
	IF(USE_CUDA)
		CUDA_ADD_EXECUTABLE(reg_f3d reg_f3d.cpp)
	ELSE(USE_CUDA)
		ADD_EXECUTABLE(reg_f3d reg_f3d.cpp)
	ENDIF(USE_CUDA)
	TARGET_LINK_LIBRARIES(reg_f3d _reg_f3d)
	INSTALL_TARGETS(/bin reg_f3d)
#-----------------------------------------------------------------------------
	IF(USE_CUDA)
		CUDA_ADD_EXECUTABLE(reg_aladin reg_aladin.cpp)
        TARGET_LINK_LIBRARIES(reg_aladin _reg_resampling _reg_globalTransformation _reg_blockMatching _reg_tools ${ZLIB} reg_nifti _reg_resampling_gpu _reg_globalTransformation_gpu _reg_blockMatching_gpu _reg_cudaCommon _reg_maths)
	ELSE(USE_CUDA)
		ADD_EXECUTABLE(reg_aladin reg_aladin.cpp)
        TARGET_LINK_LIBRARIES(reg_aladin _reg_resampling _reg_globalTransformation _reg_blockMatching _reg_tools ${ZLIB} reg_nifti _reg_maths)
	ENDIF(USE_CUDA)
	INSTALL_TARGETS(/bin reg_aladin)
#-----------------------------------------------------------------------------
#	IF(USE_CUDA)
#		CUDA_ADD_EXECUTABLE(reg_benchmark reg_benchmark.cpp)
#		TARGET_LINK_LIBRARIES(reg_benchmark _reg_resampling _reg_bspline _reg_mutualinformation _reg_ssd _reg_affineTransformation _reg_blockMatching _reg_tools ${ZLIB} reg_nifti _reg_resampling_gpu _reg_bspline_gpu _reg_mutualinformation_gpu _reg_affineTransformation_gpu _reg_blockMatching_gpu _reg_tools_gpu _reg_cudaCommon)
#	ELSE(USE_CUDA)
#		ADD_EXECUTABLE(reg_benchmark reg_benchmark.cpp)
#		TARGET_LINK_LIBRARIES(reg_benchmark _reg_resampling _reg_bspline _reg_mutualinformation _reg_ssd _reg_affineTransformation _reg_blockMatching _reg_tools ${ZLIB} reg_nifti)
#	ENDIF(USE_CUDA)
#	INSTALL_TARGETS(/bin reg_benchmark)
#-----------------------------------------------------------------------------
ENDIF(BUILD_NIFTYREG)

