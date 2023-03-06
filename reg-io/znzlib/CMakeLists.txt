#-----------------------------------------------------------------------------
add_library(znz znzlib.c)
install(TARGETS znz
    RUNTIME DESTINATION bin COMPONENT Development
    LIBRARY DESTINATION lib COMPONENT Development
    ARCHIVE DESTINATION lib COMPONENT Development
)
#-----------------------------------------------------------------------------
