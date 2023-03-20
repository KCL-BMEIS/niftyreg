#-----------------------------------------------------------------------------
add_library(znz znzlib.c)
target_link_libraries(znz z)
install(TARGETS znz
    RUNTIME DESTINATION bin COMPONENT Development
    LIBRARY DESTINATION lib COMPONENT Development
    ARCHIVE DESTINATION lib COMPONENT Development
)
#-----------------------------------------------------------------------------
