ADD_LIBRARY(reg_png reg_png.cpp reg_png.h readpng.cpp readpng.h)
TARGET_LINK_LIBRARIES(reg_png ${PNG_LIBRARY} _reg_tools reg_nifti z)
INSTALL(TARGETS reg_png
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
INSTALL(FILES reg_png.h DESTINATION include)
