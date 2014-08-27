IF(USE_CUDA)
	set(NAME _reg_cuda_kernels)
		CUDA_ADD_LIBRARY(${NAME} ${NIFTYREG_LIBRARY_TYPE} 
		CudaKernelFactory.cpp
		CudaKernels.cpp
		CudaPlatform.cpp
		ConvolutionKernel.cu
		)
		target_link_libraries(${NAME} ${CUDA_CUDA_LIBRARY} _reg_common_gpu)
		
		install(TARGETS ${NAME}
			RUNTIME DESTINATION lib
			LIBRARY DESTINATION lib
			ARCHIVE DESTINATION lib
		)
		install(FILES  CudaPlatform.h CudaKernelFactory.h CudaKernels.h DESTINATION include)

ENDIF()