#-----------------------------------------------------------------------------
add_executable(reg_mrf reg_mrf.cpp)
target_link_libraries(reg_mrf _reg_f3d)
#-----------------------------------------------------------------------------
add_executable(reg_discrete_init reg_discrete_init.cpp)
target_link_libraries(reg_discrete_init _reg_f3d)
#-----------------------------------------------------------------------------
set(MODULE_LIST
  reg_discrete_init
  reg_mrf
  )
#-----------------------------------------------------------------------------
foreach(MODULE_NAME ${MODULE_LIST})
    install(TARGETS ${MODULE_NAME}
        RUNTIME DESTINATION bin COMPONENT Runtime
        LIBRARY DESTINATION lib COMPONENT Runtime
        ARCHIVE DESTINATION lib COMPONENT Runtime
        )
endforeach(MODULE_NAME)
#-----------------------------------------------------------------------------
