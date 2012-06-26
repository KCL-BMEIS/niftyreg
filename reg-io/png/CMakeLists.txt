IF(BUILD_INTERNAL_PNG OR BUILD_ALL_DEP)
    # If the png library is not present on the machine, it is build from the sources
    # Most of the following lines are extracted from the libpng1510 CMakeLists.txt
    SET(PNGLIB_MAJOR 1)
    SET(PNGLIB_MINOR 5)
    SET(PNGLIB_RELEASE 10)
    SET(PNGLIB_NAME libpng${PNGLIB_MAJOR}${PNGLIB_MINOR})
    SET(PNGLIB_VERSION ${PNGLIB_MAJOR}.${PNGLIB_MINOR}.${PNGLIB_RELEASE})
    # Check if the m library is present
    IF(NOT WIN32)
      FIND_LIBRARY(M_LIBRARY
        NAMES m
        PATHS /usr/lib /usr/local/lib
      )
      IF(NOT M_LIBRARY)
        MESSAGE(STATUS
          "math library 'libm' not found - floating point support disabled")
      ENDIF(NOT M_LIBRARY)
    ELSE(NOT WIN32)
      # the m library is not needed on windows
      SET(M_LIBRARY "")
    ENDIF(NOT WIN32)
    # generate the config file for libpng and set the path to use it
    CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/lpng1510/pnglibconf.h.prebuilt
                   ${CMAKE_BINARY_DIR}/pnglibconf.h)
    INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR})
    # Set the libpng sources
    SET(png_SRCS
        lpng1510/png.c
        lpng1510/png.h
        lpng1510/pngconf.h
        lpng1510/pngdebug.h
        lpng1510/pngerror.c
        lpng1510/pngget.c
        lpng1510/pnginfo.h
        lpng1510/pngmem.c
        lpng1510/pngpread.c
        lpng1510/pngpriv.h
        lpng1510/pngread.c
        lpng1510/pngrio.c
        lpng1510/pngrtran.c
        lpng1510/pngrutil.c
        lpng1510/pngset.c
        lpng1510/pngstruct.h
        lpng1510/pngtest.c
        lpng1510/pngtrans.c
        lpng1510/pngwio.c
        lpng1510/pngwrite.c
        lpng1510/pngwtran.c
        lpng1510/pngwutil.c
    )
    # Build the library
    ADD_LIBRARY(png ${png_SRCS})
    TARGET_LINK_LIBRARIES(png z)
ENDIF(BUILD_INTERNAL_PNG OR BUILD_ALL_DEP)

ADD_LIBRARY(reg_png reg_png.cpp reg_png.h readpng.cpp readpng.h)
TARGET_LINK_LIBRARIES(reg_png ${PNG_LIBRARY} _reg_tools reg_nifti z)
INSTALL(TARGETS reg_png
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)
INSTALL(FILES reg_png.h DESTINATION include)
