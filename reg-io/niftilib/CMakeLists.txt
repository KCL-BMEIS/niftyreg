#-----------------------------------------------------------------------------
option(USE_NII_NAN "To enable NaN support with nifti images" ON)
mark_as_advanced(FORCE USE_NII_NAN)
if(USE_NII_NAN)
    add_definitions(-DUSE_NII_NAN)
endif(USE_NII_NAN)
set(NAME reg_nifti)
add_library(${NAME} nifti1_io.c)
target_link_libraries(${NAME} znz)
install(TARGETS ${NAME}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)
#-----------------------------------------------------------------------------
