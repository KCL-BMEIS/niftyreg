FIND_LIBRARY(PNG_LIBRARY
             NAMES png libpng
             PATHS /sw/lib
             )

IF(WIN32)
    ADD_LIBRARY(reg_png reg_png.cpp reg_png.h readpng.cpp readpng.h)
ELSE(WIN32)
    ADD_LIBRARY(reg_png SHARED reg_png.cpp reg_png.h readpng.cpp readpng.h)
ENDIF(WIN32)
    TARGET_LINK_LIBRARIES(reg_png ${PNG_LIBRARY} _reg_tools reg_nifti z)
INSTALL(TARGETS reg_png
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
INSTALL(FILES reg_png.h DESTINATION include)
