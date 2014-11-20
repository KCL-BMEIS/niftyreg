# Check if the CUDA drivers are available
find_package(CUDA)
mark_as_advanced(CUDA_SDK_ROOT_DIR)
if(NOT CUDA_FOUND)
	set(USE_CUDA OFF CACHE BOOL "To enable the CUDA platform" FORCE)
	message(ERROR "CUDA not found. The USE_OPENCL flag is turned OFF")
endif(NOT CUDA_FOUND)

include_directories(${CMAKE_SOURCE_DIR}/reg-lib/cuda)

# Display the CUDA version
message(STATUS "The CUDA version you have is v${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}")
# Compile an executable to check if there is at least one suitable graphical card
try_run(RUN_RESULT_VAR COMPILE_RESULT_VAR ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/checkCudaCard.cpp
	CMAKE_FLAGS -DINCLUDE_DIRECTORIES:STRING=${CUDA_INCLUDE_DIRS} -DLINK_LIBRARIES:STRING=${CUDA_LIBRARIES}
	COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT_VAR
	RUN_OUTPUT_VARIABLE RUN_OUTPUT_VAR
)
# Check if the executable could not compile
if(NOT COMPILE_RESULT_VAR)
	message(WARNING "The code to check the presence of a CUDA-enabled card failed.")
	message(WARNING "The USE_CUDA flag has been turned OFF.")
	set(USE_CUDA OFF CACHE BOOL "To enable CUDA for a GPU implementation of the code" FORCE)
	# Check if the executable return failure
elseif(RUN_RESULT_VAR)
	message(WARNING "No CUDA-enabled card has been detected\nResult code: ${RUN_RESULT_VAR}\nError message: ${RUN_OUTPUT_VAR}\nThe USE_CUDA flag has been turned OFF.")
	set(USE_CUDA OFF CACHE BOOL "To enable CUDA for a GPU implementation of the code" FORCE)
else(NOT COMPILE_RESULT_VAR)
	message(STATUS "Found CUDA and a CUDA-enabled card:  ${RUN_OUTPUT_VAR}")
	set(CUDA_NVCC_FLAGS "")
	#check cuda version and adjust compile flags
	if(RUN_OUTPUT_VAR EQUAL "1.0")
		set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_10,code=sm_10")
	elseif(RUN_OUTPUT_VAR EQUAL "2.0")
		set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_20,code=sm_20")
	elseif(RUN_OUTPUT_VAR EQUAL "3.0")
		set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30")
	elseif(RUN_OUTPUT_VAR EQUAL "3.5")
		set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=sm_35")
	elseif(RUN_OUTPUT_VAR EQUAL "5.0")
		set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50")
	endif(RUN_OUTPUT_VAR EQUAL "1.0")
	#adjust for debug and release versions
	if(DEBUG_MODE)
		#common debug flags for all architectures
		set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --ptxas-options=-v -g -G -lineinfo")
	else(DEBUG_MODE)
		set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -use_fast_math -O3 -Xcompiler")
	endif(DEBUG_MODE)
	message(STATUS "Debug: ${DEBUG_MODE} | NVCC Flags: ${CUDA_NVCC_FLAGS}")
	# The CUDA implementation is used
	#add_definitions(-D_USE_CUDA)
endif(NOT COMPILE_RESULT_VAR)


#-----------------------------------------------------------------------------
#install(FILES _reg_blocksize_gpu.h DESTINATION include)
#-----------------------------------------------------------------------------
set(NAME _reg_common_gpu)
cuda_add_library(${NAME} ${NIFTYREG_LIBRARY_TYPE} ${NAME}.h ${NAME}.cu )
if(WIN32)
	target_link_libraries(${NAME} ${CUDA_CUDA_LIBRARY})
endif(WIN32)
install(TARGETS ${NAME}
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
)
install(FILES ${NAME}.h DESTINATION include)
#-----------------------------------------------------------------------------
#set(LIST _reg_globalTransformation)
#set(LIST ${LIST} _reg_localTransformation)
#set(LIST ${LIST} _reg_nmi)
#set(LIST ${LIST} _reg_ssd)
#set(LIST ${LIST} _reg_resampling)
#set(LIST ${LIST} _reg_tools)
#set(LIST ${LIST} _reg_optimiser)
#foreach(NAME ${LIST})
#	cuda_add_library(${NAME}_gpu ${NIFTYREG_LIBRARY_TYPE} ${NAME}_gpu.h ${NAME}_gpu.cu ${NAME}_kernels.cu)
#	install(TARGETS ${NAME}_gpu
#		RUNTIME DESTINATION bin
#		LIBRARY DESTINATION lib
#		ARCHIVE DESTINATION lib
#	)
#	install(FILES ${NAME}_gpu.h DESTINATION include)
#endforeach(NAME)
#-----------------------------------------------------------------------------
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
install(FILES _reg_blockMatching_gpu.h CudaContext.h DESTINATION include)
install(FILES CudaKernelFactory.h CudaKernelFuncs.h CudaKernels.h CudaPlatform.h DESTINATION include)
#-----------------------------------------------------------------------------
