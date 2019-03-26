#-----------------------------------------------------------------------------
if(USE_CUDA)
  add_subdirectory(cuda)
  set(NR_CUDA_LIBRARIES
    _reg_common_cuda
    _reg_cuda_kernels
  )
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
  cpu/_reg_maths.cpp
  cpu/_reg_maths_eigen.cpp
)
install(TARGETS _reg_maths
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
install(FILES cpu/_reg_maths.h cpu/_reg_maths_eigen.h DESTINATION include)
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES};_reg_maths")
#-----------------------------------------------------------------------------
add_library(_reg_tools ${NIFTYREG_LIBRARY_TYPE}
  cpu/_reg_tools.cpp
)
target_link_libraries(_reg_tools
  _reg_maths
  reg_nifti
)
install(TARGETS _reg_tools
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
install(FILES cpu/_reg_tools.h DESTINATION include)
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES};_reg_tools")
#-----------------------------------------------------------------------------
add_library(_reg_globalTrans
  ${NIFTYREG_LIBRARY_TYPE}
  cpu/_reg_globalTrans.cpp
)
target_link_libraries(_reg_globalTrans
  _reg_tools
)
install(TARGETS _reg_globalTrans
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
install(FILES cpu/_reg_globalTrans.h DESTINATION include)
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES};_reg_globalTrans")
#-----------------------------------------------------------------------------
add_library(_reg_localTrans
  ${NIFTYREG_LIBRARY_TYPE}
  cpu/_reg_splineBasis.h
  cpu/_reg_splineBasis.cpp
  cpu/_reg_localTrans.h
  cpu/_reg_localTrans.cpp
  cpu/_reg_localTrans_regul.h
  cpu/_reg_localTrans_regul.cpp
  cpu/_reg_localTrans_jac.h
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
install(FILES cpu/_reg_localTrans.h cpu/_reg_splineBasis.h cpu/_reg_localTrans_regul.h cpu/_reg_localTrans_jac.h DESTINATION include)
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES};_reg_localTrans")
#-----------------------------------------------------------------------------
set(measure_files
  cpu/_reg_measure.h
  cpu/_reg_nmi.h
  cpu/_reg_nmi.cpp
  cpu/_reg_ssd.h
  cpu/_reg_ssd.cpp
  cpu/_reg_kld.h
  cpu/_reg_kld.cpp
  cpu/_reg_lncc.h
  cpu/_reg_lncc.cpp
  cpu/_reg_dti.h
  cpu/_reg_dti.cpp
  cpu/_reg_mind.h
  cpu/_reg_mind.cpp
)
add_library(_reg_measure ${NIFTYREG_LIBRARY_TYPE} ${measure_files})
target_link_libraries(_reg_measure _reg_tools _reg_resampling)
install(TARGETS _reg_measure
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
install(FILES cpu/_reg_measure.h cpu/_reg_nmi.h cpu/_reg_ssd.h cpu/_reg_kld.h cpu/_reg_lncc.h cpu/_reg_dti.h cpu/_reg_mind.h DESTINATION include)
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES};_reg_measure")
#-----------------------------------------------------------------------------
add_library(_reg_resampling ${NIFTYREG_LIBRARY_TYPE} cpu/_reg_resampling.cpp)
target_link_libraries(_reg_resampling _reg_globalTrans
)
install(TARGETS _reg_resampling
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  )
install(FILES cpu/_reg_resampling.h DESTINATION include)
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES};_reg_resampling")
#-----------------------------------------------------------------------------
add_library(_reg_blockMatching ${NIFTYREG_LIBRARY_TYPE} cpu/_reg_blockMatching.cpp)
target_link_libraries(_reg_blockMatching _reg_globalTrans
)
install(TARGETS _reg_blockMatching
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  )
install(FILES cpu/_reg_blockMatching.h DESTINATION include)
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES};_reg_blockMatching")
#-----------------------------------------------------------------------------
add_library(_reg_femTrans ${NIFTYREG_LIBRARY_TYPE} cpu/_reg_femTrans.cpp)
target_link_libraries(_reg_femTrans _reg_globalTrans
)
install(TARGETS _reg_femTrans
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  )
install(FILES cpu/_reg_femTrans.h DESTINATION include)
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES};_reg_femTrans")
#-----------------------------------------------------------------------------
## BUILD THE ALADIN LIBRARY
set(_reg_aladin_files
  AladinContent.cpp
  AladinContent.h
  Platform.cpp
  Platform.h
  Kernel.h
  cpu/CPUAffineDeformationFieldKernel.h
  cpu/CPUAffineDeformationFieldKernel.cpp
  cpu/CPUBlockMatchingKernel.h
  cpu/CPUBlockMatchingKernel.cpp
  cpu/CPUConvolutionKernel.h
  cpu/CPUConvolutionKernel.cpp
  cpu/CPUOptimiseKernel.h
  cpu/CPUOptimiseKernel.cpp
  cpu/CPUResampleImageKernel.h
  cpu/CPUResampleImageKernel.cpp
  cpu/CPUKernelFactory.cpp
  cpu/CPUKernelFactory.h
)
set(_reg_aladin_libraries
  _reg_localTrans
  _reg_blockMatching
  _reg_resampling
  _reg_globalTrans
  _reg_tools
  _reg_ReadWriteImage
  ${NR_OPENCL_LIBRARIES}
  ${NR_CUDA_LIBRARIES}
)
add_library(_reg_aladin ${NIFTYREG_LIBRARY_TYPE} ${_reg_aladin_files})
target_link_libraries(_reg_aladin ${_reg_aladin_libraries})
install(TARGETS _reg_aladin
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  )
install(FILES cpu/_reg_macros.h DESTINATION include)
install(FILES _reg_aladin.h _reg_aladin_sym.h DESTINATION include)
install(FILES _reg_aladin.cpp _reg_aladin_sym.cpp DESTINATION include)
install(FILES AladinContent.h Platform.h  DESTINATION include)
install(FILES
        Kernel.h
        AffineDeformationFieldKernel.h
        BlockMatchingKernel.h
        ConvolutionKernel.h
        OptimiseKernel.h
        ResampleImageKernel.h
        cpu/CPUAffineDeformationFieldKernel.h
        cpu/CPUBlockMatchingKernel.h
        cpu/CPUConvolutionKernel.h
        cpu/CPUOptimiseKernel.h
        cpu/CPUResampleImageKernel.h
        KernelFactory.h cpu/CPUKernelFactory.h DESTINATION include)
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES};_reg_aladin")
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
## BUILD THE F3D LIBRARY
set(_reg_f3d_files
    _reg_base.h
    _reg_base.cpp
    _reg_f3d.h
    _reg_f3d.cpp
    _reg_f3d2.h
    _reg_f3d2.cpp
    _reg_f3d_sym.h
    _reg_f3d_sym.cpp
)
set(_reg_f3d_libraries
  _reg_localTrans
  _reg_globalTrans
  _reg_resampling
  _reg_measure
  _reg_tools
  _reg_ReadWriteImage
)
add_library(_reg_f3d ${NIFTYREG_LIBRARY_TYPE} ${_reg_f3d_files})
target_link_libraries(_reg_f3d ${_reg_f3d_libraries})
install(TARGETS _reg_f3d
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  )
install(FILES _reg_base.h DESTINATION include)
install(FILES _reg_f3d.h DESTINATION include)
install(FILES _reg_f3d2.h DESTINATION include)
install(FILES _reg_f3d_sym.h DESTINATION include)
install(FILES cpu/_reg_optimiser.cpp cpu/_reg_optimiser.h DESTINATION include)
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES};_reg_f3d")
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# BUILD THE TPS LIBRARY
#set(NAME _reg_thinPlateSpline)
#if(APPLE)
#	add_library(${NAME} SHARED cpu/${NAME}.h cpu/${NAME}.cpp)
#else(APPLE)
#	add_library(${NAME} ${NIFTYREG_LIBRARY_TYPE} cpu/${NAME}.h cpu/${NAME}.cpp)
#endif(APPLE)
#target_link_libraries(${NAME} _reg_tools _reg_ReadWriteImage)
#install(TARGETS ${NAME}
#	RUNTIME DESTINATION bin
#	LIBRARY DESTINATION lib
#	ARCHIVE DESTINATION lib
#	)
#install(FILES cpu/${NAME}.h  DESTINATION include)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
## BUILD THE POLYAFFINE LIBRARY
#set(NAME _reg_polyAffine)
#if(APPLE)
#	add_library(${NAME} SHARED _reg_base.h _reg_base.cpp ${NAME}.h ${NAME}.cpp)
#else(APPLE)
#	add_library(${NAME} ${NIFTYREG_LIBRARY_TYPE} _reg_base.h _reg_base.cpp ${NAME}.h ${NAME}.cpp)
#endif(APPLE)
#target_link_libraries(${NAME} _reg_tools _reg_ReadWriteImage)
#install(TARGETS ${NAME}
#	RUNTIME DESTINATION bin
#	LIBRARY DESTINATION lib
#	ARCHIVE DESTINATION lib
#	)
#install(FILES ${NAME}.h  DESTINATION include)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES}" PARENT_SCOPE)