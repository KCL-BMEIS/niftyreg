IF (USE_OPENCL)
	set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")#bring this to dir
	FIND_PACKAGE(OPENCL REQUIRED)
	IF (OPENCL_FOUND)
		set(NAME _reg_opencl_kernels)
		ADD_LIBRARY(${NAME} ${NIFTYREG_LIBRARY_TYPE} 
		CLKernelFactory.cpp
		CLKernels.cpp
		CLPlatform.cpp
		CLContextSingletton.cpp
		)
		
		target_link_libraries(${NAME} ${OPENCL_LIBRARIES} )
		configure_file(resampleKernel.cl          ${CMAKE_CURRENT_BINARY_DIR}/resampleKernel.cl          COPYONLY)
		configure_file(affineDeformationKernel.cl ${CMAKE_CURRENT_BINARY_DIR}/affineDeformationKernel.cl COPYONLY)
		
		
		install(TARGETS ${NAME}
			RUNTIME DESTINATION lib
			LIBRARY DESTINATION lib
			ARCHIVE DESTINATION lib
		)
		install(FILES  CLPlatform.h CLKernelFactory.h CPUKernelFactory.h CLKernels.h CLContextSingletton.h DESTINATION include)
	ENDIF()
ENDIF()