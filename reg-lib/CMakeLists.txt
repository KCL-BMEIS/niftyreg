#-----------------------------------------------------------------------------
if(USE_CUDA)
  add_subdirectory(cuda)
  if(NOT USE_CUDA_PLUGIN)
    # Static CUDA: the kernels are linked into the host libraries. In plugin mode
    # NR_CUDA_LIBRARIES stays empty - CUDA lives in the niftyreg_cuda plugin, loaded
    # at run time - so no host target links any CUDA library.
    set(NR_CUDA_LIBRARIES _reg_cuda_kernels)
  endif()
endif(USE_CUDA)
#-----------------------------------------------------------------------------
if(USE_OPENCL)
  add_subdirectory(cl)
  set(NR_OPENCL_LIBRARIES _reg_opencl_kernels)
endif(USE_OPENCL)
#-----------------------------------------------------------------------------
##BUILD THE CPU LIBRARIES
#-----------------------------------------------------------------------------
add_library(_reg_maths ${NIFTYREG_LIBRARY_TYPE}
  cpu/Maths.cpp
)
install(TARGETS _reg_maths
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES};_reg_maths")
#-----------------------------------------------------------------------------
add_library(_reg_tools ${NIFTYREG_LIBRARY_TYPE} cpu/_reg_tools.cpp)
target_link_libraries(_reg_tools
  _reg_maths
  reg_nifti
)
install(TARGETS _reg_tools
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES};_reg_tools")
#-----------------------------------------------------------------------------
add_library(_reg_globalTrans ${NIFTYREG_LIBRARY_TYPE} cpu/_reg_globalTrans.cpp)
target_link_libraries(_reg_globalTrans _reg_tools)
install(TARGETS _reg_globalTrans
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES};_reg_globalTrans")
#-----------------------------------------------------------------------------
add_library(_reg_localTrans ${NIFTYREG_LIBRARY_TYPE}
  cpu/_reg_splineBasis.cpp
  cpu/_reg_localTrans.cpp
  cpu/_reg_localTrans_regul.cpp
  cpu/_reg_localTrans_jac.cpp
)
target_link_libraries(_reg_localTrans
  _reg_tools
  _reg_globalTrans
)
install(TARGETS _reg_localTrans
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES};_reg_localTrans")
#-----------------------------------------------------------------------------
set(measure_files
  cpu/_reg_nmi.cpp
  cpu/_reg_ssd.cpp
  cpu/_reg_kld.cpp
  cpu/_reg_lncc.cpp
  cpu/_reg_dti.cpp
  cpu/_reg_mind.cpp
)
add_library(_reg_measure ${NIFTYREG_LIBRARY_TYPE} ${measure_files})
target_link_libraries(_reg_measure _reg_tools _reg_resampling)
install(TARGETS _reg_measure
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES};_reg_measure")
#-----------------------------------------------------------------------------
add_library(_reg_resampling ${NIFTYREG_LIBRARY_TYPE} cpu/_reg_resampling.cpp)
target_link_libraries(_reg_resampling _reg_globalTrans)
install(TARGETS _reg_resampling
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES};_reg_resampling")
#-----------------------------------------------------------------------------
add_library(_reg_blockMatching ${NIFTYREG_LIBRARY_TYPE} cpu/_reg_blockMatching.cpp)
target_link_libraries(_reg_blockMatching _reg_globalTrans)
install(TARGETS _reg_blockMatching
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES};_reg_blockMatching")
#-----------------------------------------------------------------------------
add_library(_reg_content ${NIFTYREG_LIBRARY_TYPE}
  AladinContent.cpp
  Content.cpp
  DefContent.cpp
  F3dContent.cpp
)
target_link_libraries(_reg_content
  _reg_blockMatching
  _reg_resampling
  _reg_tools
)
install(TARGETS _reg_content
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES};_reg_content")
#-----------------------------------------------------------------------------
add_library(_reg_compute ${NIFTYREG_LIBRARY_TYPE}
  Compute.cpp
  Optimiser.cpp
  MeasureCreator.cpp
)
target_link_libraries(_reg_compute
  _reg_blockMatching
  _reg_content
  _reg_localTrans
  _reg_measure
  _reg_resampling
)
install(TARGETS _reg_compute
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES};_reg_compute")
#-----------------------------------------------------------------------------
add_library(_reg_kernels ${NIFTYREG_LIBRARY_TYPE}
  cpu/CpuKernelFactory.cpp
  cpu/CpuAffineDeformationFieldKernel.cpp
  cpu/CpuBlockMatchingKernel.cpp
  cpu/CpuConvolutionKernel.cpp
  cpu/CpuLtsKernel.cpp
  cpu/CpuResampleImageKernel.cpp
)
target_link_libraries(_reg_kernels
  _reg_blockMatching
  _reg_content
  _reg_globalTrans
  _reg_resampling
  _reg_tools
)
install(TARGETS _reg_kernels
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES};_reg_kernels")
#-----------------------------------------------------------------------------
add_library(_reg_platform ${NIFTYREG_LIBRARY_TYPE} Platform.cpp)
target_link_libraries(_reg_platform
  _reg_compute
  _reg_kernels
  ${NR_OPENCL_LIBRARIES}
  ${NR_CUDA_LIBRARIES}
)
if(USE_CUDA_PLUGIN)
  # Platform reaches the CUDA implementation through the runtime plugin loader.
  # NR_INSTALL_LIBDIR is the loader's last-resort search path (external programs
  # linking static NiftyReg, whose executable lives outside the install prefix).
  target_sources(_reg_platform PRIVATE CudaPluginLoader.cpp)
  target_link_libraries(_reg_platform ${CMAKE_DL_LIBS})
  target_compile_definitions(_reg_platform PRIVATE "NR_INSTALL_LIBDIR=\"${CMAKE_INSTALL_PREFIX}/lib\"")
  # Build-tree runs with shared host libraries: dlopen resolves a bare name through
  # the CALLING module's RUNPATH - which is lib_reg_platform.so, not the executable -
  # so the plugin's build directory must be on this target's RPATH (installed trees
  # need nothing: both libraries land in lib/, found module-relative)
  set_target_properties(_reg_platform PROPERTIES BUILD_RPATH "$<TARGET_FILE_DIR:niftyreg_cuda>")
endif()
install(TARGETS _reg_platform
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES};_reg_platform")
#-----------------------------------------------------------------------------
## BUILD THE ALADIN LIBRARY
add_library(_reg_aladin ${NIFTYREG_LIBRARY_TYPE} _reg_aladin.cpp _reg_aladin_sym.cpp)
target_link_libraries(_reg_aladin
  _reg_blockMatching
  _reg_compute
  _reg_content
  _reg_kernels
  _reg_platform
  _reg_localTrans
  _reg_resampling
  _reg_globalTrans
  _reg_tools
  _reg_ReadWriteImage
  ${NR_OPENCL_LIBRARIES}
  ${NR_CUDA_LIBRARIES}
)
install(TARGETS _reg_aladin
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES};_reg_aladin")
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
## BUILD THE F3D LIBRARY
set(_reg_f3d_files
  _reg_base.cpp
  _reg_f3d.cpp
  _reg_f3d2.cpp
)
set(_reg_f3d_libraries
  _reg_blockMatching
  _reg_compute
  _reg_content
  _reg_kernels
  _reg_platform
  _reg_localTrans
  _reg_globalTrans
  _reg_resampling
  _reg_measure
  _reg_tools
  _reg_ReadWriteImage
  ${NR_OPENCL_LIBRARIES}
  ${NR_CUDA_LIBRARIES}
)
add_library(_reg_f3d ${NIFTYREG_LIBRARY_TYPE} ${_reg_f3d_files})
target_link_libraries(_reg_f3d ${_reg_f3d_libraries})
install(TARGETS _reg_f3d
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES};_reg_f3d")
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES}" PARENT_SCOPE)