#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
IF(BUILD_NIFTYREG)
##BUILD THE CPU LIBRARIES
#-----------------------------------------------------------------------------
    SET(LIST _reg_resampling)
    SET(LIST ${LIST} _reg_maths)
    SET(LIST ${LIST} _reg_globalTransformation)
    SET(LIST ${LIST} _reg_mutualinformation)
    SET(LIST ${LIST} _reg_ssd)
    SET(LIST ${LIST} _reg_tools)
    SET(LIST ${LIST} _reg_blockMatching)
    SET(LIST ${LIST} _reg_femTransformation)
#    SET(LIST ${LIST} _reg_thinPlateSpline)
    FOREACH(NAME ${LIST})
        ADD_LIBRARY(${NAME} ${NAME}.h ${NAME}.cpp)
        TARGET_LINK_LIBRARIES(${NAME} _reg_maths)
        INSTALL(TARGETS ${NAME}
                RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
                LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
                ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        )
        INSTALL(FILES ${NAME}.h DESTINATION include)
    ENDFOREACH(NAME)
#-----------------------------------------------------------------------------
    SET(NAME _reg_localTransformation)
    ADD_LIBRARY(${NAME} ${NAME}.h ${NAME}.cpp)
    TARGET_LINK_LIBRARIES(${NAME} _reg_maths)
    INSTALL(TARGETS ${NAME}
            RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
            LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
            ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    )
    INSTALL(FILES ${NAME}.h DESTINATION include)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
##BUILD THE GPU LIBRARIES
    IF(USE_CUDA)
#-----------------------------------------------------------------------------
        SET(LIST _reg_globalTransformation)
        SET(LIST ${LIST} _reg_blockMatching)
        SET(LIST ${LIST} _reg_localTransformation)
        SET(LIST ${LIST} _reg_mutualinformation)
        SET(LIST ${LIST} _reg_resampling)
        SET(LIST ${LIST} _reg_tools)
        FOREACH(NAME ${LIST})
            CUDA_ADD_LIBRARY(${NAME}_gpu ${NAME}_gpu.h ${NAME}_gpu.cu ${NAME}_kernels.cu)
            INSTALL(TARGETS ${NAME}_gpu
                    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
                    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
                    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
            )
            INSTALL(FILES ${NAME}_gpu.h DESTINATION include)
        ENDFOREACH(NAME)
#-----------------------------------------------------------------------------
        SET(NAME _reg_cudaCommon)
        CUDA_ADD_LIBRARY(${NAME} ${NAME}.h ${NAME}.cu)
        INSTALL(TARGETS ${NAME}
                RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
                LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
                ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        )
        INSTALL(FILES ${NAME}.h DESTINATION include)
#-----------------------------------------------------------------------------
        INSTALL(FILES _reg_blocksize_gpu.h DESTINATION include)
#-----------------------------------------------------------------------------
    ENDIF(USE_CUDA)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
## BUILD THE F3D LIBRARY CPU OR GPU
    SET(NAME _reg_f3d)
    IF(USE_CUDA)
#-----------------------------------------------------------------------------
        IF(WIN32)
            CUDA_ADD_LIBRARY(${NAME} ${NAME}.h ${NAME}.cpp ${NAME}2.h ${NAME}2.cpp ${NAME}_gpu.h ${NAME}_gpu.cpp)
        ELSE(WIN32)
            CUDA_ADD_LIBRARY(${NAME} SHARED ${NAME}.h ${NAME}.cpp ${NAME}2.h ${NAME}2.cpp ${NAME}_gpu.h ${NAME}_gpu.cpp)
        ENDIF(WIN32)
            TARGET_LINK_LIBRARIES(${NAME} _reg_localTransformation _reg_resampling _reg_globalTransformation _reg_ssd _reg_mutualinformation _reg_tools reg_nifti ${ZLIB} _reg_localTransformation_gpu _reg_resampling_gpu _reg_globalTransformation_gpu _reg_mutualinformation_gpu _reg_tools_gpu _reg_cudaCommon)
            INSTALL(FILES ${NAME}_gpu.h DESTINATION include)
#-----------------------------------------------------------------------------
    ELSE(USE_CUDA)
#-----------------------------------------------------------------------------
        IF(WIN32)
            ADD_LIBRARY(${NAME} ${NAME}.h ${NAME}.cpp ${NAME}2.h ${NAME}2.cpp)
        ELSE(WIN32)
            ADD_LIBRARY(${NAME} SHARED ${NAME}.h ${NAME}.cpp ${NAME}2.h ${NAME}2.cpp)
        ENDIF(WIN32)
            TARGET_LINK_LIBRARIES(${NAME} _reg_localTransformation _reg_resampling _reg_globalTransformation _reg_ssd _reg_mutualinformation _reg_tools reg_nifti ${ZLIB})
#-----------------------------------------------------------------------------
    ENDIF(USE_CUDA)
#-----------------------------------------------------------------------------
    INSTALL(TARGETS ${NAME}
            RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
            LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
            ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    )
    INSTALL(FILES ${NAME}.h DESTINATION include)
    INSTALL(FILES ${NAME}2.h DESTINATION include)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
ENDIF(BUILD_NIFTYREG)
