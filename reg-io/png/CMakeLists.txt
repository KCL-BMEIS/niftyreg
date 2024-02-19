if(BUILD_INTERNAL_PNG OR BUILD_ALL_DEP)
    # If the png library is not present on the machine, it is build from the sources
    # Most of the following lines are extracted from the libpng CMakeLists.txt
    # Check if the m library is present
    if(NOT WIN32)
      find_library(M_LIBRARY
        NAMES m
        PATHS /usr/lib /usr/local/lib
      )
      if(NOT M_LIBRARY)
        message(STATUS "math library 'libm' not found - floating point support disabled")
      endif(NOT M_LIBRARY)
    else(NOT WIN32)
      # the m library is not needed on windows
      set(M_LIBRARY "")
    endif(NOT WIN32)
    # generate the config file for libpng and set the path to use it
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/lpng/pnglibconf.h.prebuilt
                   ${CMAKE_BINARY_DIR}/pnglibconf.h)
    include_directories(${CMAKE_BINARY_DIR})
    # Set the libpng sources
    set(png_hdrs
      lpng/png.h
      lpng/pngconf.h
      lpng/pngdebug.h
      lpng/pnginfo.h
      lpng/pngpriv.h
      lpng/pngstruct.h
    )
    set(png_srcs
      lpng/png.c
      lpng/pngerror.c
      lpng/pngget.c
      lpng/pngmem.c
      lpng/pngpread.c
      lpng/pngread.c
      lpng/pngrio.c
      lpng/pngrtran.c
      lpng/pngrutil.c
      lpng/pngset.c
      lpng/pngtrans.c
      lpng/pngwio.c
      lpng/pngwrite.c
      lpng/pngwtran.c
      lpng/pngwutil.c
    )
    # Build the library
    add_library(png STATIC ${png_srcs})
    target_link_libraries(png z)
    install(TARGETS png
        LIBRARY DESTINATION lib COMPONENT Development
        ARCHIVE DESTINATION lib COMPONENT Development
    )
    install(FILES ${png_hdrs} ${CMAKE_BINARY_DIR}/pnglibconf.h DESTINATION include COMPONENT Development)
endif(BUILD_INTERNAL_PNG OR BUILD_ALL_DEP)

add_library(reg_png reg_png.cpp)
target_link_libraries(reg_png ${PNG_LIBRARY} _reg_tools)
install(TARGETS reg_png
        RUNTIME DESTINATION bin COMPONENT Development
        LIBRARY DESTINATION lib COMPONENT Development
        ARCHIVE DESTINATION lib COMPONENT Development
)
