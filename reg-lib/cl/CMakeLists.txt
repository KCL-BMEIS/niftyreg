# Find the OpenCL package
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(OPENCL REQUIRED)
if(NOT OPENCL_FOUND)
	set(USE_OPENCL OFF CACHE BOOL "To enable the OpenCL platform" FORCE)
	message(ERROR "OpenCL not found. The USE_OPENCL flag is turned OFF")
endif(NOT OPENCL_FOUND)

set(SOURCE_PATH ${CMAKE_BINARY_DIR})

configure_file(config.h.in ${CMAKE_BINARY_DIR}/config.h CONFIGURE IMMEDIATE)

include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR}/reg-lib/cl)
include_directories(${OPENCL_INCLUDE_DIRS})

# Build the _reg_opencl_kernels library
set(NAME _reg_opencl_kernels)
add_library(${NAME} ${NIFTYREG_LIBRARY_TYPE}
	CLContextSingletton.cpp
	CLContent.cpp
	CLKernelFactory.cpp
	CLKernels.cpp
	../Content.cpp
	../Platform.cpp
	
	#for debugging
	../cpu/_reg_resampling.cpp
	../cpu/_reg_globalTransformation.cpp
	../cpu/_reg_blockMatching.cpp
)
target_link_libraries(${NAME} ${OPENCL_LIBRARIES} )
configure_file(resampleKernel.cl ${CMAKE_CURRENT_BINARY_DIR}/resampleKernel.cl COPYONLY)
configure_file(affineDeformationKernel.cl ${CMAKE_CURRENT_BINARY_DIR}/affineDeformationKernel.cl COPYONLY)
install(TARGETS ${NAME}
	RUNTIME DESTINATION lib
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
)
install(FILES CLContextSingletton.h CLContent.h CLKernelFactory.h CLKernels.h DESTINATION include)
install(FILES resampleKernel.cl affineDeformationKernel.cl blockMatchingKernel.cl DESTINATION include/cl)
