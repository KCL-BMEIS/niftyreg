#
# Try to find GLEW library and include path.
# Once done this will define
#
# GLEW_FOUND
# GLEW_INCLUDE_PATH
# GLEW_LIBRARY
# 



IF (WIN32)
    STRING(REGEX MATCH "Win64" ISWIN64 ${CMAKE_GENERATOR})
    IF("${ISWIN64}" STREQUAL "Win64") 
            SET(GLEW_LIB_NAMES glew64 glew64s )
    ELSE("${ISWIN64}" STREQUAL "Win64") 
            SET(GLEW_LIB_NAMES glew32 glew32s )
    ENDIF("${ISWIN64}" STREQUAL "Win64") 

    FIND_PATH( GLEW_INCLUDE_PATH GL/glew.h
        $ENV{PROGRAMFILES}/GLEW/include
        ${PROJECT_SOURCE_DIR}/src/nvgl/glew/include
        DOC "The directory where GL/glew.h resides")
    FIND_LIBRARY( GLEW_LIBRARY
	    NAMES glew ${GLEW_LIB_NAMES}
        PATHS
        $ENV{PROGRAMFILES}/GLEW/lib
        $ENV{NVSDKCOMPUTE_ROOT}/shared/lib                  
        $ENV{NVSDKCOMPUTE_ROOT}/shared/lib/Win32          
        ${PROJECT_SOURCE_DIR}/src/nvgl/glew/bin
        ${PROJECT_SOURCE_DIR}/src/nvgl/glew/lib
        DOC "The GLEW library")
ELSE (WIN32)
    FIND_PATH( GLEW_INCLUDE_PATH GL/glew.h
        /usr/include
        /usr/local/include
        /sw/include
        /opt/local/include
        $ENV{NVSDKCOMPUTE_ROOT}/shared/inc
        DOC "The directory where GL/glew.h resides")
    FIND_LIBRARY( GLEW_LIBRARY
        NAMES GLEW glew
        PATHS
        /usr/lib64
        /usr/lib
        /usr/local/lib64
        /usr/local/lib
        /sw/lib
        /opt/local/lib
        $ENV{NVSDKCOMPUTE_ROOT}/shared/lib/linux 
        DOC "The GLEW library")
ENDIF (WIN32)

IF (GLEW_INCLUDE_PATH)
    SET( GLEW_FOUND 1 CACHE STRING "Set to 1 if GLEW is found, 0 otherwise")
ELSE (GLEW_INCLUDE_PATH)
    SET( GLEW_FOUND 0 CACHE STRING "Set to 1 if GLEW is found, 0 otherwise")
ENDIF (GLEW_INCLUDE_PATH)

MARK_AS_ADVANCED( GLEW_FOUND )
