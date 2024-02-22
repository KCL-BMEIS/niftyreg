#-----------------------------------------------------------------------------
# Eigen version 3.3.*
if(NOT EXISTS ${CMAKE_BINARY_DIR}/third-party/eigen)
  message(STATUS "Cloning Eigen...")
  execute_process(
    COMMAND git clone -q -b 3.3 https://gitlab.com/libeigen/eigen.git
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/third-party
    RESULT_VARIABLE result
  )
  if(result)
    message(FATAL_ERROR "Failed to clone Eigen!")
  endif(result)
  message(STATUS "Eigen is cloned into ${CMAKE_BINARY_DIR}/third-party/eigen")
endif(NOT EXISTS ${CMAKE_BINARY_DIR}/third-party/eigen)
#-----------------------------------------------------------------------------
if(OPENMP_FOUND)
	if(USE_OPENMP)
		add_definitions(-DEIGEN_HAS_OPENMP)
	endif(USE_OPENMP)
endif(OPENMP_FOUND)
#-----------------------------------------------------------------------------
