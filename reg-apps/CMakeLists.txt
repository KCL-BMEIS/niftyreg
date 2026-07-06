include_directories(${CMAKE_CURRENT_BINARY_DIR})
#-----------------------------------------------------------------------------
add_executable(reg_average reg_average.cpp)
target_link_libraries(reg_average _reg_resampling _reg_globalTrans _reg_localTrans _reg_maths _reg_tools _reg_ReadWriteImage)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/reg_average.h.in ${CMAKE_CURRENT_BINARY_DIR}/reg_average.h @ONLY)
#-----------------------------------------------------------------------------
add_executable(reg_tools reg_tools.cpp)
target_link_libraries(reg_tools _reg_measure _reg_blockMatching _reg_resampling _reg_maths _reg_tools _reg_ReadWriteImage)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/reg_tools.h.in ${CMAKE_CURRENT_BINARY_DIR}/reg_tools.h @ONLY)
#-----------------------------------------------------------------------------
add_executable(reg_resample reg_resample.cpp)
set(_reg_resample_libraries
  _reg_compute
  _reg_kernels
  _reg_resampling
  _reg_localTrans
  _reg_tools
  _reg_globalTrans
  _reg_ReadWriteImage
)
if(USE_CUDA)
  set(_reg_resample_libraries ${_reg_resample_libraries} _reg_cuda_kernels)
endif(USE_CUDA)
target_link_libraries(reg_resample ${_reg_resample_libraries})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/reg_resample.h.in ${CMAKE_CURRENT_BINARY_DIR}/reg_resample.h @ONLY)
#-----------------------------------------------------------------------------
add_executable(reg_measure reg_measure.cpp)
target_link_libraries(reg_measure _reg_resampling _reg_tools _reg_measure _reg_ReadWriteImage)
#-----------------------------------------------------------------------------
add_executable(reg_transform reg_transform.cpp)
target_link_libraries(reg_transform _reg_resampling _reg_localTrans _reg_tools _reg_globalTrans _reg_maths _reg_ReadWriteImage)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/reg_transform.h.in ${CMAKE_CURRENT_BINARY_DIR}/reg_transform.h @ONLY)
#-----------------------------------------------------------------------------
add_executable(reg_jacobian reg_jacobian.cpp)
target_link_libraries(reg_jacobian _reg_resampling _reg_localTrans _reg_tools _reg_globalTrans _reg_ReadWriteImage)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/reg_jacobian.h.in ${CMAKE_CURRENT_BINARY_DIR}/reg_jacobian.h @ONLY)
#-----------------------------------------------------------------------------
add_executable(reg_f3d reg_f3d.cpp)
target_link_libraries(reg_f3d _reg_f3d)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/reg_f3d.h.in ${CMAKE_CURRENT_BINARY_DIR}/reg_f3d.h @ONLY)
#-----------------------------------------------------------------------------
add_executable(reg_aladin reg_aladin.cpp)
target_link_libraries(reg_aladin _reg_aladin)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/reg_aladin.h.in ${CMAKE_CURRENT_BINARY_DIR}/reg_aladin.h @ONLY)
#-----------------------------------------------------------------------------
# Standalone CPU-vs-CUDA benchmark (own main; CUDA-only). Independent of the test infrastructure,
# so it builds whenever USE_CUDA is on, regardless of BUILD_TESTING. Not installed (dev tool).
if(USE_CUDA)
  add_executable(reg_benchmark reg_benchmark.cpp)
  target_link_libraries(reg_benchmark _reg_aladin _reg_f3d CUDA::cudart)
endif(USE_CUDA)
#-----------------------------------------------------------------------------
set(MODULE_LIST
  reg_average
  reg_tools
  reg_resample
  reg_measure
  reg_transform
  reg_jacobian
  reg_aladin
  reg_f3d
)
#-----------------------------------------------------------------------------
if(USE_CUDA OR USE_OPENCL)
  set(gpuinfo_libraries "")
  if(USE_CUDA)
    set(gpuinfo_libraries ${gpuinfo_libraries} _reg_cudainfo)
  endif(USE_CUDA)
  if(USE_OPENCL)
    set(gpuinfo_libraries ${gpuinfo_libraries} _reg_openclinfo)
  endif(USE_OPENCL)
  add_executable(reg_gpuinfo reg_gpuinfo.cpp)
  target_link_libraries(reg_gpuinfo ${gpuinfo_libraries})
  set(MODULE_LIST ${MODULE_LIST} reg_gpuinfo)
endif(USE_CUDA OR USE_OPENCL)
#-----------------------------------------------------------------------------
foreach(MODULE_NAME ${MODULE_LIST})
  install(TARGETS ${MODULE_NAME}
    RUNTIME DESTINATION bin COMPONENT Runtime
    LIBRARY DESTINATION lib COMPONENT Runtime
    ARCHIVE DESTINATION lib COMPONENT Runtime
  )
endforeach(MODULE_NAME)
#-----------------------------------------------------------------------------
install(PROGRAMS groupwise_niftyreg_params.sh DESTINATION bin COMPONENT Runtime)
install(PROGRAMS groupwise_niftyreg_run.sh DESTINATION bin COMPONENT Runtime)
#-----------------------------------------------------------------------------
#add_executable(reg_ppcnr reg_ppcnr.cpp)
#target_link_libraries(reg_ppcnr _reg_ReadWriteImage)
#-----------------------------------------------------------------------------