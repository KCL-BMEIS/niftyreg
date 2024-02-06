if(CHECK_GPU)
    # Compile an executable to check if there is at least one suitable CUDA card
    try_run(RUN_RESULT_VAR COMPILE_RESULT_VAR ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/checkCudaCard.cu
        CMAKE_FLAGS -DINCLUDE_DIRECTORIES:STRING=${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT_VAR
        RUN_OUTPUT_VARIABLE RUN_OUTPUT_VAR
    )
    # Check if the executable won't compile
    if(NOT COMPILE_RESULT_VAR)
        message(WARNING "The code to check the presence of a CUDA-enabled card failed.")
        message("The USE_CUDA flag has been turned OFF.")
        set(USE_CUDA OFF CACHE BOOL "To use the CUDA platform" FORCE)
        return()
    # Check if the executable returns failure
    elseif(RUN_RESULT_VAR)
        message(WARNING "No CUDA-enabled card has been detected")
        message("Result code: ${RUN_RESULT_VAR}")
        message("Error message: ${RUN_OUTPUT_VAR}")
        message("The USE_CUDA flag has been turned OFF.")
        set(USE_CUDA OFF CACHE BOOL "To use the CUDA platform" FORCE)
        return()
    endif(NOT COMPILE_RESULT_VAR)
    message(STATUS "Found a CUDA-enabled card (capability ${RUN_OUTPUT_VAR})")
    string(REPLACE "." "" CAPABILITY_CODE ${RUN_OUTPUT_VAR})
    # Check CUDA version and adjust compile flags
    if("${CAPABILITY_CODE}" LESS "60")
        set(USE_CUDA OFF CACHE BOOL "To use the CUDA platform" FORCE)
        message(SEND_ERROR "CUDA cards with capability less than 6.0 are not supported. The USE_CUDA flag is turned OFF")
        return()
    endif("${CAPABILITY_CODE}" LESS "60")
    set(CMAKE_CUDA_ARCHITECTURES "${CAPABILITY_CODE}-real")
else(CHECK_GPU)
    # If no GPU check is performed, assume a minimum capability of 6.0
    # Generate compiled code for all architectures supported by CUDA 11.8
    # Also, generate PTX code for future architectures
    # Therefore, the code should run on any GPU with a capability of 6.0 or higher
    set(CMAKE_CUDA_ARCHITECTURES "60-real;61-real;70-real;75-real;80-real;86-real;89")
endif(CHECK_GPU)
#-----------------------------------------------------------------------------
# Set C++ standard version for CUDA, and enable extended lambdas and relaxed constexpr support
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr")
# If desired, add PIC flags
if(CMAKE_POSITION_INDEPENDENT_CODE AND DEFINED CMAKE_C_COMPILE_OPTIONS_PIC)
    # Add (undocumented) CMake flag that should tell the host compiler to generate position independent code
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options ${CMAKE_C_COMPILE_OPTIONS_PIC}")
endif()
# Adjust for debug and release versions
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v -g -G")
else(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-O3")
endif(CMAKE_BUILD_TYPE STREQUAL "Debug")
if(CUDA_FAST_MATH AND CUDA_PRECISE_SQRT EQUAL "OFF" AND CUDA_PRECISE_DIV EQUAL "OFF")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math")
    message(STATUS "CUDA fast math enabled")
endif(CUDA_FAST_MATH AND CUDA_PRECISE_SQRT EQUAL "OFF" AND CUDA_PRECISE_DIV EQUAL "OFF")
#-----------------------------------------------------------------------------
set(NAME _reg_cuda_kernels)
add_library(${NAME} ${NIFTYREG_LIBRARY_TYPE}
    ../AladinContent.cpp
    affineDeformationKernel.cu
    blockMatchingKernel.cu
    CudaAffineDeformationFieldKernel.cpp
    CudaAladinContent.cpp
    CudaBlockMatchingKernel.cpp
    CudaCommon.cu
    CudaCompute.cu
    CudaContent.cpp
    CudaContext.cpp
    CudaConvolutionKernel.cpp
    CudaDefContent.cpp
    CudaF3dContent.cpp
    CudaGlobalTransformation.cu
    CudaKernelConvolution.cu
    CudaKernelFactory.cpp
    CudaLocalTransformation.cu
    CudaLtsKernel.cpp
    CudaMeasure.cpp
    CudaNormaliseGradient.cu
    CudaOptimiser.cu
    CudaResampleImageKernel.cpp
    CudaResampling.cu
    CudaTools.cu
    resampleKernel.cu
    _reg_nmi_gpu.cu
    _reg_ssd_gpu.cu
)
target_link_libraries(${NAME} CUDA::cuda_driver)
install(TARGETS ${NAME}
    RUNTIME DESTINATION lib
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES};${NAME}")
#-----------------------------------------------------------------------------
set(NAME _reg_cudainfo)
add_library(${NAME} ${NIFTYREG_LIBRARY_TYPE} ${NAME}.cu)
target_link_libraries(${NAME} CUDA::cuda_driver)
install(TARGETS ${NAME}
    RUNTIME DESTINATION lib
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES};${NAME}")
#-----------------------------------------------------------------------------
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES}" PARENT_SCOPE)