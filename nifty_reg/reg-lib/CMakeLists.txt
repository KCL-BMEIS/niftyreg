#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
IF(BUILD_NIFTYREG)
##BUILD THE CPU LIBRARIES
#-----------------------------------------------------------------------------
	SET(LIST _reg_resampling)
	SET(LIST ${LIST} _reg_affineTransformation)
	SET(LIST ${LIST} _reg_bspline)
	SET(LIST ${LIST} _reg_mutualinformation)
	SET(LIST ${LIST} _reg_ssd)
	SET(LIST ${LIST} _reg_tools)
	SET(LIST ${LIST} _reg_blockMatching)
	FOREACH(NAME ${LIST})
		ADD_LIBRARY(${NAME} ${NAME}.h ${NAME}.cpp)
		INSTALL(TARGETS ${NAME}
			RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
			LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
			ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
		)
		INSTALL(FILES ${NAME}.h DESTINATION include)
	ENDFOREACH(NAME)
#-----------------------------------------------------------------------------
	SET(NAME _reg_bspline_comp)
	ADD_LIBRARY(${NAME} ${NAME}.h ${NAME}.cpp)
	TARGET_LINK_LIBRARIES(${NAME} _reg_bspline)
	INSTALL(TARGETS ${NAME}
		RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
		LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
		ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	)
	INSTALL(FILES ${NAME}.h DESTINATION include)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
##BUILD THE GPU LIBRARIES
	IF(USE_CUDA)
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
			INSTALL(FILES ${NAME}_gpu.h DESTINATION include)
		ENDFOREACH(NAME)
#-----------------------------------------------------------------------------
		SET(NAME _reg_cudaCommon)
		CUDA_ADD_LIBRARY(${NAME} ${NAME}.h ${NAME}.cu)
		INSTALL(TARGETS ${NAME}
			RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
			LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
			ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
		)
		INSTALL(FILES ${NAME}.h DESTINATION include)
#-----------------------------------------------------------------------------
	ENDIF(USE_CUDA)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
## BUILD THE F3D LIBRARY CPU OR GPU
	SET(NAME _reg_f3d)
	IF(USE_CUDA)
#-----------------------------------------------------------------------------
		CUDA_ADD_LIBRARY(${NAME} SHARED ${NAME}.h ${NAME}.cpp ${NAME}_gpu.h ${NAME}_gpu.cpp)
		TARGET_LINK_LIBRARIES(${NAME} _reg_bspline_comp _reg_resampling _reg_affineTransformation _reg_ssd _reg_mutualinformation _reg_tools reg_nifti ${ZLIB} _reg_bspline_gpu _reg_resampling_gpu _reg_affineTransformation_gpu _reg_mutualinformation_gpu _reg_tools_gpu _reg_cudaCommon)
		INSTALL(FILES ${NAME}_gpu.h DESTINATION include)
#-----------------------------------------------------------------------------
	ELSE(USE_CUDA)
#-----------------------------------------------------------------------------
		ADD_LIBRARY(${NAME} SHARED ${NAME}.h ${NAME}.cpp)
		TARGET_LINK_LIBRARIES(${NAME} _reg_bspline_comp _reg_resampling _reg_affineTransformation _reg_ssd _reg_mutualinformation _reg_tools reg_nifti ${ZLIB})
#-----------------------------------------------------------------------------
	ENDIF(USE_CUDA)
#-----------------------------------------------------------------------------
	INSTALL(TARGETS ${NAME}
		RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
		LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
		ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	)
	INSTALL(FILES ${NAME}.h DESTINATION include)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
ENDIF(BUILD_NIFTYREG)
