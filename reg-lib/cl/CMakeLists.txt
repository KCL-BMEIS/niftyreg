#-----------------------------------------------------------------------------
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
#-----------------------------------------------------------------------------
# Find the OpenCL package
find_package(OPENCL REQUIRED)
if(NOT OPENCL_FOUND)
  set(USE_OPENCL OFF CACHE BOOL "To use the OpenCL platform" FORCE)
  message(SEND_ERROR "OpenCL not found. The USE_OPENCL flag is turned OFF")
  return()
else(NOT OPENCL_FOUND)
  message(STATUS "Found OpenCL")
endif(NOT OPENCL_FOUND)
#-----------------------------------------------------------------------------
set(SOURCE_PATH ${CMAKE_BINARY_DIR})
#-----------------------------------------------------------------------------
configure_file(config.h.in ${CMAKE_BINARY_DIR}/config.h IMMEDIATE)
mark_as_advanced(_OPENCL_CPP_INCLUDE_DIRS)
#-----------------------------------------------------------------------------
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR}/reg-lib/cl)
include_directories(${OPENCL_INCLUDE_DIRS})
#-----------------------------------------------------------------------------
# Build the _reg_opencl_kernels library
set(NAME _reg_opencl_kernels)
add_library(${NAME} ${NIFTYREG_LIBRARY_TYPE}
  CLContextSingletton.cpp
  CLAladinContent.cpp
  CLKernelFactory.cpp
  CLAffineDeformationFieldKernel.cpp
  CLBlockMatchingKernel.cpp
  CLConvolutionKernel.cpp
  CLOptimiseKernel.cpp
  CLResampleImageKernel.cpp
  ../AladinContent.cpp
  ../Platform.cpp
)
target_link_libraries(${NAME} ${OPENCL_LIBRARIES})
install(TARGETS ${NAME}
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES};${NAME}")
#-----------------------------------------------------------------------------
install(FILES CLContextSingletton.h CLAladinContent.h CLKernelFactory.h
        CLAffineDeformationFieldKernel.h
        CLBlockMatchingKernel.h
        CLConvolutionKernel.h
        CLOptimiseKernel.h
        CLResampleImageKernel.h
        DESTINATION include/cl)
install(FILES resampleKernel.cl affineDeformationKernel.cl blockMatchingKernel.cl DESTINATION include/cl)
#-----------------------------------------------------------------------------
set(NAME _reg_openclinfo)
add_library(${NAME} ${NIFTYREG_LIBRARY_TYPE} ${NAME}.cpp ${NAME}.h InfoDevice.h CLContextSingletton.cpp)
target_link_libraries(${NAME} ${OPENCL_LIBRARIES})
install(TARGETS ${NAME}
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
install(FILES ${NAME}.h DESTINATION include/cl)
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES};${NAME}")
#-----------------------------------------------------------------------------
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES}" PARENT_SCOPE)