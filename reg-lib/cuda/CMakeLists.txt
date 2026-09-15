# Default architecture fan (used when no configure-time device probe runs):
# compiled code for every architecture the toolkit still supports, plus PTX so
# the code JIT-compiles on future architectures
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13)
    # CUDA 13 removed compilation for pre-Turing architectures (capability < 7.5).
    # SASS runs on any later minor revision of the same major (86 covers 87, 120
    # covers 121, ...); the trailing plain 90 embeds PTX for everything >= 9.0
    # without native code (e.g. capability 10.x/11.x)
    set(NR_DEFAULT_CUDA_ARCHITECTURES "75-real;80-real;86-real;89-real;90-real;120-real;90")
else()
    # All architectures supported by CUDA 11.8, so the code runs on any GPU with
    # a capability of 6.0 or higher
    set(NR_DEFAULT_CUDA_ARCHITECTURES "60-real;61-real;70-real;75-real;80-real;86-real;89")
endif()
if(USE_CUDA_PLUGIN)
    # Plugin builds target machines whose GPUs are unknown at configure time, so a
    # configure-time device probe is meaningless: build the default architecture fan
    # and decide availability at run time (the plugin's IsAvailable device probe)
    set(CMAKE_CUDA_ARCHITECTURES "${NR_DEFAULT_CUDA_ARCHITECTURES}")
elseif(CHECK_GPU)
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
else()
    set(CMAKE_CUDA_ARCHITECTURES "${NR_DEFAULT_CUDA_ARCHITECTURES}")
endif()
#-----------------------------------------------------------------------------
# Set C++ standard version for CUDA, and enable extended lambdas and relaxed constexpr support
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr")
# nvcc always drives cl.exe as the host compiler on Windows, and CCCL (the Thrust/CUB/
# libcu++ bundled with CUDA >= 13) refuses cl.exe's traditional preprocessor
if(MSVC AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/Zc:preprocessor")
endif()
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
set(_reg_cuda_kernel_sources
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
    CudaLts.cu
    CudaLtsKernel.cpp
    CudaMeasureCreator.cpp
    CudaNormaliseGradient.cu
    CudaOptimiser.cu
    CudaResampleImageKernel.cpp
    CudaResampling.cu
    CudaTools.cu
    _reg_lncc_gpu.cu
    _reg_nmi_gpu.cu
    _reg_ssd_gpu.cu
)
if(USE_CUDA_PLUGIN)
    # Runtime plugin: the CUDA kernels plus the plugin bridge in one shared library
    # the host binaries load at run time. With static host libraries (the default)
    # their objects are linked in so the plugin is self-contained; with shared host
    # libraries the plugin links the same shared libraries as the executables, so
    # nothing is duplicated. _reg_cudainfo.cu is included so reg_gpuinfo can report
    # device information through the plugin instead of linking CUDA itself.
    add_library(niftyreg_cuda SHARED
        CudaPluginImpl.cpp
        _reg_cudainfo.cu
        ${_reg_cuda_kernel_sources}
    )
    # One canonical file name per platform - the loader looks the plugin up by name
    # (MSVC and MinGW would otherwise disagree about the lib prefix on Windows)
    if(WIN32)
        set_target_properties(niftyreg_cuda PROPERTIES PREFIX "" OUTPUT_NAME "niftyreg_cuda")
    else()
        set_target_properties(niftyreg_cuda PROPERTIES PREFIX "lib" OUTPUT_NAME "niftyreg_cuda")
    endif()
    # The compiled architecture list backs the plugin's run-time device probe
    string(JOIN "," NR_CUDA_ARCHS_JOINED ${CMAKE_CUDA_ARCHITECTURES})
    target_compile_definitions(niftyreg_cuda PRIVATE "NR_CUDA_COMPILED_ARCHS=\"${NR_CUDA_ARCHS_JOINED}\"")
    target_link_libraries(niftyreg_cuda
        _reg_blockMatching
        _reg_compute
        _reg_content
        _reg_measure
        _reg_tools
        CUDA::cuda_driver CUDA::cusolver CUDA::cublas
    )
    if(BUILD_SHARED_LIBS)
        target_link_libraries(niftyreg_cuda CUDA::cudart)
    endif(BUILD_SHARED_LIBS)
    # Linux: with the libraries, where installed executables (bin/../lib) and
    # external consumers (next to lib_reg_platform) find it. Windows: next to the
    # executables, where LoadLibrary searches natively.
    install(TARGETS niftyreg_cuda
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
    )
else()
    set(NAME _reg_cuda_kernels)
    add_library(${NAME} ${NIFTYREG_LIBRARY_TYPE} ${_reg_cuda_kernel_sources})
    target_link_libraries(${NAME}
        _reg_blockMatching
        _reg_compute
        _reg_content
        _reg_measure
        _reg_tools
        CUDA::cuda_driver CUDA::cusolver CUDA::cublas
    )
    if(BUILD_SHARED_LIBS)
        target_link_libraries(${NAME} CUDA::cudart)
    endif(BUILD_SHARED_LIBS)
    install(TARGETS ${NAME}
        RUNTIME DESTINATION lib
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
    set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES};${NAME}")
    #-------------------------------------------------------------------------
    # In plugin mode the device-information code is part of the plugin instead
    # (a standalone _reg_cudainfo would put a hard CUDA dependency on reg_gpuinfo)
    set(NAME _reg_cudainfo)
    add_library(${NAME} ${NIFTYREG_LIBRARY_TYPE} ${NAME}.cu)
    target_link_libraries(${NAME} CUDA::cuda_driver)
    install(TARGETS ${NAME}
        RUNTIME DESTINATION lib
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
    set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES};${NAME}")
endif()
#-----------------------------------------------------------------------------
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES}" PARENT_SCOPE)