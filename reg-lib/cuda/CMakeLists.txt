IF(USE_CUDA)
	set(NAME _reg_cuda_kernels)
		CUDA_ADD_LIBRARY(${NAME} ${NIFTYREG_LIBRARY_TYPE} 
		
		_reg_blockMatching_gpu.cu
		_reg_blockMatching_kernels.cu
		CudaContext.cpp
		CudaKernelFactory.cpp
		CudaKernelFuncs.cu
		CudaKernels.cpp
		CudaPlatform.cpp

		)
		target_link_libraries(${NAME} ${CUDA_CUDA_LIBRARY} _reg_common_gpu  )
		
		install(TARGETS ${NAME}
			RUNTIME DESTINATION lib
			LIBRARY DESTINATION lib
			ARCHIVE DESTINATION lib
		)
		install(FILES  _reg_blockMatching_gpu.h CudaContext.cpp CudaKernelFactory.h CudaKernelFuncs.h CudaKernels.h CudaPlatform.h DESTINATION include)

ENDIF()