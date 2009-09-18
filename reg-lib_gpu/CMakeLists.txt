#-----------------------------------------------------------------------------
SET(NAME _reg_affineTransformation)
CUDA_ADD_LIBRARY(${NAME}_gpu ${NAME}_gpu.h ${NAME}_gpu.cu ${NAME}_kernels.cu)
INSTALL(TARGETS ${NAME}_gpu
	RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
	LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
#-----------------------------------------------------------------------------
IF(BUILD_ALADIN)
	SET(NAME _reg_blockMatching)
	CUDA_ADD_LIBRARY(${NAME}_gpu ${NAME}_gpu.h ${NAME}_gpu.cu ${NAME}_kernels.cu)
	INSTALL(TARGETS ${NAME}_gpu
		RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
		LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
		ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	)
ENDIF(BUILD_ALADIN)
#-----------------------------------------------------------------------------
IF(BUILD_F3D)
	SET(NAME _reg_bspline)
	CUDA_ADD_LIBRARY(${NAME}_gpu ${NAME}_gpu.h ${NAME}_gpu.cu ${NAME}_kernels.cu)
	INSTALL(TARGETS ${NAME}_gpu
		RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
		LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
		ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	)
ENDIF(BUILD_F3D)
#-----------------------------------------------------------------------------
SET(NAME _reg_cudaCommon)
CUDA_ADD_LIBRARY(${NAME} ${NAME}.h ${NAME}.cu)
INSTALL(TARGETS ${NAME}
	RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
	LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
#-----------------------------------------------------------------------------
IF(BUILD_F3D)
	SET(NAME _reg_mutualinformation)
	CUDA_ADD_LIBRARY(${NAME}_gpu ${NAME}_gpu.h ${NAME}_gpu.cu ${NAME}_kernels.cu)
	INSTALL(TARGETS ${NAME}_gpu
		RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
		LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
		ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	)
ENDIF(BUILD_F3D)
#-----------------------------------------------------------------------------
SET(NAME _reg_resampling)
CUDA_ADD_LIBRARY(${NAME}_gpu ${NAME}_gpu.h ${NAME}_gpu.cu ${NAME}_kernels.cu)
INSTALL(TARGETS ${NAME}_gpu
	RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
	LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
#-----------------------------------------------------------------------------
SET(NAME _reg_tools)
CUDA_ADD_LIBRARY(${NAME}_gpu ${NAME}_gpu.h ${NAME}_gpu.cu ${NAME}_kernels.cu)
INSTALL(TARGETS ${NAME}_gpu
	RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
	LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
#-----------------------------------------------------------------------------