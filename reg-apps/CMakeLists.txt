#-----------------------------------------------------------------------------
add_executable(reg_average reg_average.cpp)
target_link_libraries(reg_average _reg_resampling _reg_globalTrans _reg_localTrans _reg_maths _reg_tools _reg_ReadWriteImage)
#-----------------------------------------------------------------------------
add_executable(reg_tools reg_tools.cpp)
target_link_libraries(reg_tools _reg_aladin)
#-----------------------------------------------------------------------------
add_executable(reg_resample reg_resample.cpp)
target_link_libraries(reg_resample _reg_resampling _reg_localTrans _reg_tools _reg_globalTrans _reg_ReadWriteImage)
#-----------------------------------------------------------------------------
add_executable(reg_measure reg_measure.cpp)
target_link_libraries(reg_measure _reg_resampling _reg_tools _reg_nmi _reg_ssd _reg_KLdivergence _reg_lncc _reg_dti _reg_ReadWriteImage)
#-----------------------------------------------------------------------------
#add_executable(reg_ppcnr reg_ppcnr.cpp)
#target_link_libraries(reg_ppcnr _reg_ReadWriteImage)
#-----------------------------------------------------------------------------
add_executable(reg_transform reg_transform.cpp)
target_link_libraries(reg_transform _reg_resampling _reg_localTrans _reg_tools _reg_globalTrans _reg_maths _reg_ReadWriteImage)
#-----------------------------------------------------------------------------
add_executable(reg_jacobian reg_jacobian.cpp)
target_link_libraries(reg_jacobian _reg_resampling _reg_localTrans _reg_tools _reg_globalTrans _reg_ReadWriteImage)
#-----------------------------------------------------------------------------
if(USE_CUDA)
  cuda_add_executable(reg_f3d reg_f3d.cpp)
else(USE_CUDA)
  add_executable(reg_f3d reg_f3d.cpp)
endif(USE_CUDA)
target_link_libraries(reg_f3d _reg_f3d)
#-----------------------------------------------------------------------------
if(USE_CUDA)
  cuda_add_executable(reg_aladin reg_aladin.cpp)
else(USE_CUDA)
  add_executable(reg_aladin reg_aladin.cpp)
endif(USE_CUDA)
target_link_libraries(reg_aladin _reg_aladin)
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
