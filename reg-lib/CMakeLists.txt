if(USE_OPENCL)
	add_subdirectory(cl)
	set(NR_OPENCL_LIBRARY
		"_reg_opencl_kernels")
endif(USE_OPENCL)
if(USE_CUDA)
	add_subdirectory(cuda)
	set(NR_CUDA_LIBRARY
#		"_reg_globalTransformation_gpu"
#		"_reg_localTransformation_gpu"
#		"_reg_nmi_gpu"
#		"_reg_ssd_gpu"
#		"_reg_resampling_gpu"
#		"_reg_tools_gpu"
#		"_reg_optimiser_gpu"
		"_reg_common_gpu"
		"_reg_cuda_kernels"
	)
endif(USE_CUDA)
#-----------------------------------------------------------------------------
##BUILD THE CPU LIBRARIES
#-----------------------------------------------------------------------------
add_library(_reg_maths ${NIFTYREG_LIBRARY_TYPE}
	_reg_maths.h
	_reg_maths.cpp
)
target_link_libraries(_reg_maths
)
install(TARGETS _reg_maths
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
)
install(FILES _reg_maths.h DESTINATION include)
#-----------------------------------------------------------------------------
add_library(_reg_tools ${NIFTYREG_LIBRARY_TYPE}
	_reg_tools.h
	_reg_tools.cpp
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
install(FILES _reg_tools.h DESTINATION include)

#-----------------------------------------------------------------------------
add_library(_reg_globalTransformation
	${NIFTYREG_LIBRARY_TYPE}
	_reg_globalTransformation.h
	_reg_globalTransformation.cpp
)
target_link_libraries(_reg_globalTransformation
	_reg_tools
)
install(TARGETS _reg_globalTransformation
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
)
install(FILES _reg_globalTransformation.h DESTINATION include)

#-----------------------------------------------------------------------------
set(LIB_LIST
	_reg_localTransformation
	_reg_resampling
	_reg_nmi
	_reg_ssd
	_reg_KLdivergence
	_reg_lncc
	_reg_dti
	_reg_blockMatching
	_reg_femTransformation
)
foreach(NAME ${LIB_LIST})
	add_library(${NAME} ${NIFTYREG_LIBRARY_TYPE}
	${NAME}.h
	${NAME}.cpp)
	target_link_libraries(${NAME}
	_reg_globalTransformation
	)
	install(TARGETS ${NAME}
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	)
	install(FILES ${NAME}.h DESTINATION include)
endforeach(NAME)

install(FILES _reg_measure.h DESTINATION include)

#-----------------------------------------------------------------------------
set(NAME _reg_optimiser)
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	add_library(${NAME} SHARED ${NAME}.h ${NAME}.cpp)
else(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	add_library(${NAME} ${NIFTYREG_LIBRARY_TYPE} ${NAME}.h ${NAME}.cpp)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
target_link_libraries(${NAME})
install(TARGETS ${NAME}
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	)
install(FILES ${NAME}.h ${NAME}.cpp DESTINATION include)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
# when building, don't use the install RPATH already
# (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# the RPATH to be used when installing, but only if it's not a system directory
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
	set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif("${isSystemDir}" STREQUAL "-1")
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
	_reg_localTransformation
	_reg_resampling
	_reg_globalTransformation
	_reg_lncc
	_reg_dti
	_reg_KLdivergence
	_reg_ssd _reg_nmi
	_reg_optimiser
	_reg_tools
	_reg_ReadWriteImage
)
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	add_library(${NAME} SHARED ${_reg_f3d_files})
else(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	add_library(${NAME} ${NIFTYREG_LIBRARY_TYPE} ${_reg_f3d_files})
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
target_link_libraries(${NAME} ${_reg_f3d_libraries})
install(TARGETS ${NAME}
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	)
install(FILES _reg_base.h _reg_base.cpp DESTINATION include)
install(FILES ${NAME}.h ${NAME}.cpp DESTINATION include)
install(FILES ${NAME}2.h ${NAME}2.cpp DESTINATION include)
install(FILES ${NAME}_sym.h ${NAME}_sym.cpp DESTINATION include)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
## BUILD THE ALADIN LIBRARY
set(NAME _reg_aladin)

set(_reg_aladin_files
	${NAME}.h
	${NAME}.cpp
	${NAME}_sym.h
	${NAME}_sym.cpp
	Context.cpp
	Platform.cpp
	CPUPlatfom.cpp
	CPUKernelFactory.cpp
	CPUKernels.cpp
)
set(_reg_aladin_libraries
	_reg_localTransformation
	_reg_blockMatching
	_reg_resampling
	_reg_globalTransformation
	_reg_ssd
	_reg_nmi
	_reg_tools
	_reg_ReadWriteImage
	${NR_OPENCL_LIBRARY}
	${NR_CUDA_LIBRARY}
)
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	add_library(${NAME} SHARED ${_reg_aladin_files})
else(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	add_library(${NAME} ${NIFTYREG_LIBRARY_TYPE} ${_reg_aladin_files})
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
target_link_libraries(${NAME} ${_reg_aladin_libraries})
install(TARGETS ${NAME}
	RUNTIME DESTINATION lib
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	)
install(FILES ${NAME}.h ${NAME}_sym.h _reg_macros.h DESTINATION include)
install(FILES Context.h Platform.h CPUPlatform.h DESTINATION include)
install(FILES Kernel.h CPUKernels.h KernelFactory.h CPUKernelFactory.h DESTINATION include)
#-----------------------------------------------------------------------------
## BUILD THE TPS LIBRARY
set(NAME _reg_thinPlateSpline)
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	add_library(${NAME} SHARED ${NAME}.h ${NAME}.cpp)
else(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	add_library(${NAME} ${NIFTYREG_LIBRARY_TYPE} ${NAME}.h ${NAME}.cpp)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
target_link_libraries(${NAME} _reg_tools _reg_ReadWriteImage)
install(TARGETS ${NAME}
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	)
install(FILES ${NAME}.h ${NAME}.cpp DESTINATION include)
#-----------------------------------------------------------------------------
## BUILD THE POLYAFFINE LIBRARY
set(NAME _reg_polyAffine)
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	add_library(${NAME} SHARED _reg_base.h _reg_base.cpp ${NAME}.h ${NAME}.cpp)
else(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	add_library(${NAME} ${NIFTYREG_LIBRARY_TYPE} _reg_base.h _reg_base.cpp ${NAME}.h ${NAME}.cpp)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
target_link_libraries(${NAME} _reg_tools _reg_ReadWriteImage)
install(TARGETS ${NAME}
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	)
install(FILES ${NAME}.h ${NAME}.cpp DESTINATION include)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
