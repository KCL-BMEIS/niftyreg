#-----------------------------------------------------------------------------
# Check if the CUDA drivers are available
find_package(CUDA)
mark_as_advanced(CUDA_SDK_ROOT_DIR)
#-----------------------------------------------------------------------------
if(NOT CUDA_FOUND)
	set(USE_CUDA OFF CACHE BOOL "To use the CUDA platform" FORCE)
	message(ERROR "CUDA not found. The USE_CUDA flag is turned OFF")
endif(NOT CUDA_FOUND)
#-----------------------------------------------------------------------------
include_directories(${CMAKE_SOURCE_DIR}/reg-lib/cuda)
#-----------------------------------------------------------------------------
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
	set(USE_CUDA OFF CACHE BOOL "To use the CUDA platform" FORCE)
	# Check if the executable return failure
elseif(RUN_RESULT_VAR)
	message(WARNING "No CUDA-enabled card has been detected\nResult code: ${RUN_RESULT_VAR}\nError message: ${RUN_OUTPUT_VAR}\nThe USE_CUDA flag has been turned OFF.")
	set(USE_CUDA OFF CACHE BOOL "To enable CUDA for a GPU implementation of the code" FORCE)
else(NOT COMPILE_RESULT_VAR)
	message(STATUS "Found CUDA (v${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}) and a CUDA-enabled card (capability ${RUN_OUTPUT_VAR})")
	set(CUDA_NVCC_FLAGS "")
	#check cuda version and adjust compile flags
	string(REPLACE "." "" CAPABILITY_CODE ${RUN_OUTPUT_VAR})
	if(NOT "${RUN_OUTPUT_VAR}" MATCHES "^3\\.0$")
		set(USE_CUDA OFF CACHE BOOL "To use the CUDA platform" FORCE)
		message(ERROR "CUDA card with capability 1.x are not supported. The USE_CUDA flag is turned OFF")
	endif(NOT "${RUN_OUTPUT_VAR}" MATCHES "^3\\.0$")
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_${CAPABILITY_CODE},code=sm_${CAPABILITY_CODE}")
	#adjust for debug and release versions
	if(CMAKE_BUILD_TYPE STREQUAL "Debug")
		set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --ptxas-options=-v -g -G -lineinfo")
	else()
		set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --ptxas-options=-O3")
	endif(CMAKE_BUILD_TYPE STREQUAL "Debug")
#	if(CUDA_PRECISE_SQRT)
#	    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --prec-sqrt=true")
#	elseif(CUDA_PRECISE_SQRT)
#	    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --prec-sqrt=false")
#	endif(CUDA_PRECISE_SQRT)
#	if(CUDA_PRECISE_DIV)
#	    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --prec-div=true")
#	elseif(CUDA_PRECISE_DIV)
#	    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --prec-div=false")
#	endif(CUDA_PRECISE_DIV)
	if(CUDA_FAST_MATH AND CUDA_PRECISE_SQRT EQUAL "OFF" AND CUDA_PRECISE_DIV EQUAL "OFF")
		set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -use_fast_math")
	endif(CUDA_FAST_MATH AND CUDA_PRECISE_SQRT EQUAL "OFF" AND CUDA_PRECISE_DIV EQUAL "OFF")
endif(NOT COMPILE_RESULT_VAR)
#-----------------------------------------------------------------------------
set(NAME _reg_common_cuda)
cuda_add_library(${NAME} ${NIFTYREG_LIBRARY_TYPE} ${NAME}.h ${NAME}.cu)
target_link_libraries(${NAME} ${CUDA_CUDA_LIBRARY})
install(TARGETS ${NAME}
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
)
install(FILES ${NAME}.h DESTINATION include/cuda)
#-----------------------------------------------------------------------------
set(NAME _reg_cuda_kernels)
cuda_add_library(${NAME} ${NIFTYREG_LIBRARY_TYPE}
	    blockMatchingKernel.cu
        CUDAContent.cpp
        CUDAKernelFactory.cpp
        resampleKernel.cu
        affineDeformationKernel.cu
        CUDAAffineDeformationFieldKernel.cpp
        CUDABlockMatchingKernel.cpp
        CUDAConvolutionKernel.cpp
        CUDAOptimiseKernel.cpp
        CUDAResampleImageKernel.cpp
	../Content.cpp
    ../Platform.cpp
)
target_link_libraries(${NAME} ${CUDA_CUDA_LIBRARY} _reg_common_cuda)
install(TARGETS ${NAME}
	RUNTIME DESTINATION lib
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
)
install(FILES blockMatchingKernel.h CUDAContent.h DESTINATION include/cuda)
install(FILES CUDAKernelFactory.h affineDeformationKernel.h resampleKernel.h CUDAAffineDeformationFieldKernel.h CUDABlockMatchingKernel.h CUDAConvolutionKernel.h CUDAOptimiseKernel.h CUDAResampleImageKernel.h DESTINATION include/cuda)
#-----------------------------------------------------------------------------
