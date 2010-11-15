IF(BUILD_NIFTYREG AND USE_CUDA)
#-----------------------------------------------------------------------------
	SET(LIST _reg_affineTransformation)
	SET(LIST ${LIST} _reg_blockMatching)
	SET(LIST ${LIST} _reg_bspline)
	SET(LIST ${LIST} _reg_mutualinformation)
	SET(LIST ${LIST} _reg_resampling)
	SET(LIST ${LIST} _reg_tools)
	FOREACH(NAME ${LIST})
		CUDA_ADD_LIBRARY(${NAME}_gpu ${NAME}_gpu.h ${NAME}_gpu.cu ${NAME}_kernels.cu)
		INSTALL(TARGETS ${NAME}_gpu
			RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
			LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
			ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
		)
	ENDFOREACH(NAME)
#-----------------------------------------------------------------------------
	SET(NAME _reg_cudaCommon)
	CUDA_ADD_LIBRARY(${NAME} ${NAME}.h ${NAME}.cu)
	INSTALL(TARGETS ${NAME}
		RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
		LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
		ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	)
#-----------------------------------------------------------------------------
ENDIF(BUILD_NIFTYREG AND USE_CUDA)