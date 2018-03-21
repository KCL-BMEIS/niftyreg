#-----------------------------------------------------------------------------
# Eigen version 3.3.3 - 67e894c6cd8f
if(NOT EXISTS ${CMAKE_BINARY_DIR}/third-party/eigen3)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar x ${CMAKE_SOURCE_DIR}/third-party/eigen_3.3.3.tar.gz
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/third-party
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E rename
    ${CMAKE_BINARY_DIR}/third-party/eigen-eigen-67e894c6cd8f
    ${CMAKE_BINARY_DIR}/third-party/eigen3
  )
  message(STATUS "Eigen3 files are copied in ${CMAKE_BINARY_DIR}/third-party/eigen3")
endif(NOT EXISTS ${CMAKE_BINARY_DIR}/third-party/eigen3)
#-----------------------------------------------------------------------------
if(MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj" )
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /bigobj" )
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /wd4127 /wd4505 /wd4714")
	string(REGEX REPLACE "/W[0-9]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()
#-----------------------------------------------------------------------------
if(OPENMP_FOUND)
	if(USE_OPENMP)
		add_definitions(-DEIGEN_HAS_OPENMP)
	endif(USE_OPENMP)
endif(OPENMP_FOUND)
#-----------------------------------------------------------------------------
