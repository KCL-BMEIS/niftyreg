#-----------------------------------------------------------------------------
configure_file(config.h.in ${CMAKE_BINARY_DIR}/config.h IMMEDIATE)
#-----------------------------------------------------------------------------
# Build the _reg_opencl_kernels library
set(NAME _reg_opencl_kernels)
add_library(${NAME} ${NIFTYREG_LIBRARY_TYPE}
  ClCompute.cpp
  ClContextSingleton.cpp
  ClAladinContent.cpp
  ClKernelFactory.cpp
  ClAffineDeformationFieldKernel.cpp
  ClBlockMatchingKernel.cpp
  ClConvolutionKernel.cpp
  ClLtsKernel.cpp
  ClResampleImageKernel.cpp
  ../AladinContent.cpp
)
target_link_libraries(${NAME} ${OpenCL_LIBRARIES})
install(TARGETS ${NAME}
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES};${NAME}")
#-----------------------------------------------------------------------------
set(NAME _reg_openclinfo)
add_library(${NAME} ${NIFTYREG_LIBRARY_TYPE} ${NAME}.cpp ClContextSingleton.cpp)
target_link_libraries(${NAME} ${OpenCL_LIBRARIES})
install(TARGETS ${NAME}
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES};${NAME}")
#-----------------------------------------------------------------------------
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES}" PARENT_SCOPE)