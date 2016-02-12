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
  cpu/_reg_maths.h
  cpu/_reg_maths.cpp
  cpu/_reg_maths_eigen.h
  cpu/_reg_maths_eigen.cpp
)
install(TARGETS _reg_maths
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
install(FILES cpu/_reg_maths.h cpu/_reg_maths_eigen.h DESTINATION include)
#-----------------------------------------------------------------------------
add_library(_reg_tools ${NIFTYREG_LIBRARY_TYPE}
  cpu/_reg_tools.h
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
#-----------------------------------------------------------------------------
add_library(_reg_globalTrans
  ${NIFTYREG_LIBRARY_TYPE}
  cpu/_reg_globalTrans.h
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
#-----------------------------------------------------------------------------
add_library(_reg_measure
  ${NIFTYREG_LIBRARY_TYPE}
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
)
target_link_libraries(_reg_measure
  _reg_tools
)
install(TARGETS _reg_measure
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
install(FILES cpu/_reg_measure.h cpu/_reg_nmi.h cpu/_reg_ssd.h cpu/_reg_kld.h cpu/_reg_lncc.h cpu/_reg_dti.h DESTINATION include)
#-----------------------------------------------------------------------------
set(LIB_LIST
  _reg_resampling
  _reg_blockMatching
  _reg_femTrans
)
foreach(NAME ${LIB_LIST})
  add_library(${NAME} ${NIFTYREG_LIBRARY_TYPE}
  cpu/${NAME}.h
  cpu/${NAME}.cpp)
  target_link_libraries(${NAME}
  _reg_globalTrans
  )
  install(TARGETS ${NAME}
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  )
  install(FILES cpu/${NAME}.h DESTINATION include)
endforeach(NAME)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
## BUILD THE ALADIN LIBRARY
set(NAME _reg_aladin)
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
add_library(${NAME} ${NIFTYREG_LIBRARY_TYPE} ${_reg_aladin_files})
target_link_libraries(${NAME} ${_reg_aladin_libraries})
install(TARGETS ${NAME}
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  )
install(FILES ${NAME}.h ${NAME}_sym.h cpu/_reg_macros.h DESTINATION include)
install(FILES ${NAME}.cpp ${NAME}_sym.cpp DESTINATION include)
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
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
## BUILD THE F3D LIBRARY
set(NAME _reg_f3d)
set(_reg_f3d_files
    _reg_base.h
    _reg_base.cpp
    ${NAME}.h
    ${NAME}.cpp
    ${NAME}2.h
    ${NAME}2.cpp
    ${NAME}_sym.h
    ${NAME}_sym.cpp
)
set(_reg_f3d_libraries
  _reg_localTrans
  _reg_resampling
  _reg_globalTrans
  _reg_measure
  _reg_tools
  _reg_ReadWriteImage
)
add_library(${NAME} ${NIFTYREG_LIBRARY_TYPE} ${_reg_f3d_files})
target_link_libraries(${NAME} ${_reg_f3d_libraries})
install(TARGETS ${NAME}
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  )
install(FILES _reg_base.cpp _reg_base.h DESTINATION include)
install(FILES ${NAME}.cpp ${NAME}.h DESTINATION include)
install(FILES ${NAME}_sym.cpp ${NAME}_sym.h DESTINATION include)
install(FILES ${NAME}2.cpp ${NAME}2.h DESTINATION include)
install(FILES cpu/_reg_optimiser.cpp cpu/_reg_optimiser.h DESTINATION include)
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
