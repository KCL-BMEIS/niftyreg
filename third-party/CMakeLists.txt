## Eigen version 3.2.2 - 1306d75b4a21
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
