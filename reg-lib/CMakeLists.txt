#-----------------------------------------------------------------------------
##BUILD THE CPU LIBRARIES
#-----------------------------------------------------------------------------
set(LIST _reg_resampling)
set(LIST ${LIST} _reg_maths)
set(LIST ${LIST} _reg_globalTransformation)
set(LIST ${LIST} _reg_mutualinformation)
set(LIST ${LIST} _reg_ssd)
set(LIST ${LIST} _reg_KLdivergence)
set(LIST ${LIST} _reg_blockMatching)
set(LIST ${LIST} _reg_femTransformation)
FOREACH(NAME ${LIST})
    ADD_LIBRARY(${NAME} ${NAME}.h ${NAME}.cpp)
    TARGET_LINK_LIBRARIES(${NAME})
    INSTALL(TARGETS ${NAME}
            RUNTIME DESTINATION bin COMPONENT Development
            LIBRARY DESTINATION lib COMPONENT Development
            ARCHIVE DESTINATION lib COMPONENT Development
    )
    INSTALL(FILES ${NAME}.h DESTINATION include COMPONENT Development)
ENDFOREACH(NAME)
#-----------------------------------------------------------------------------
set(LIST _reg_tools)
set(LIST ${LIST} _reg_localTransformation)
FOREACH(NAME ${LIST})
    ADD_LIBRARY(${NAME} ${NAME}.h ${NAME}.cpp)
    TARGET_LINK_LIBRARIES(${NAME} _reg_maths)
    INSTALL(TARGETS ${NAME}
            RUNTIME DESTINATION bin COMPONENT Development
            LIBRARY DESTINATION lib COMPONENT Development
            ARCHIVE DESTINATION lib COMPONENT Development
    )
    INSTALL(FILES ${NAME}.h DESTINATION include COMPONENT Development)
ENDFOREACH(NAME)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
##BUILD THE GPU LIBRARIES
IF(USE_CUDA)
#-----------------------------------------------------------------------------
    set(LIST _reg_globalTransformation)
    set(LIST ${LIST} _reg_blockMatching)
    set(LIST ${LIST} _reg_localTransformation)
    set(LIST ${LIST} _reg_mutualinformation)
    set(LIST ${LIST} _reg_resampling)
    set(LIST ${LIST} _reg_tools)
    FOREACH(NAME ${LIST})
        CUDA_ADD_LIBRARY(${NAME}_gpu ${NAME}_gpu.h ${NAME}_gpu.cu ${NAME}_kernels.cu)
        INSTALL(TARGETS ${NAME}_gpu
                RUNTIME DESTINATION bin COMPONENT Development
                LIBRARY DESTINATION lib COMPONENT Development
                ARCHIVE DESTINATION lib COMPONENT Development
        )
        INSTALL(FILES ${NAME}_gpu.h DESTINATION include COMPONENT Development)
    ENDFOREACH(NAME)
#-----------------------------------------------------------------------------
    set(NAME _reg_cudaCommon)
    CUDA_ADD_LIBRARY(${NAME} ${NAME}.h ${NAME}.cu)
    INSTALL(TARGETS ${NAME}
            RUNTIME DESTINATION bin COMPONENT Development
            LIBRARY DESTINATION lib COMPONENT Development
            ARCHIVE DESTINATION lib COMPONENT Development
    )
    INSTALL(FILES ${NAME}.h DESTINATION include)
    INSTALL(FILES _reg_blocksize_gpu.h DESTINATION include COMPONENT Development)
#-----------------------------------------------------------------------------
ENDIF(USE_CUDA)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
## BUILD THE F3D LIBRARY CPU OR GPU
set(NAME _reg_f3d)
IF(USE_CUDA)
#-----------------------------------------------------------------------------
    IF(WIN32)
        CUDA_ADD_LIBRARY(${NAME} ${NAME}.h ${NAME}.cpp ${NAME}_sym.h ${NAME}_sym.cpp ${NAME}2.h ${NAME}2.cpp ${NAME}_gpu.h ${NAME}_gpu.cpp)
    ELSE(WIN32)
        CUDA_ADD_LIBRARY(${NAME} SHARED ${NAME}.h ${NAME}.cpp ${NAME}_sym.h ${NAME}_sym.cpp ${NAME}2.h ${NAME}2.cpp ${NAME}_gpu.h ${NAME}_gpu.cpp)
    ENDIF(WIN32)
    TARGET_LINK_LIBRARIES(${NAME} _reg_localTransformation _reg_resampling _reg_globalTransformation _reg_KLdivergence _reg_ssd _reg_mutualinformation _reg_tools _reg_ReadWriteImage _reg_localTransformation_gpu _reg_resampling_gpu _reg_globalTransformation_gpu _reg_mutualinformation_gpu _reg_tools_gpu _reg_cudaCommon)
    INSTALL(FILES ${NAME}_gpu.h ${NAME}_gpu.cpp DESTINATION include COMPONENT Development)
#-----------------------------------------------------------------------------
ELSE(USE_CUDA)
#-----------------------------------------------------------------------------
    IF(WIN32)
        ADD_LIBRARY(${NAME} ${NAME}.h ${NAME}.cpp ${NAME}2.h ${NAME}2.cpp ${NAME}_sym.h ${NAME}_sym.cpp)
    ELSE(WIN32)
        ADD_LIBRARY(${NAME} SHARED ${NAME}.h ${NAME}.cpp ${NAME}2.h ${NAME}2.cpp ${NAME}_sym.h ${NAME}_sym.cpp)
    ENDIF(WIN32)
    TARGET_LINK_LIBRARIES(${NAME} _reg_localTransformation _reg_resampling _reg_globalTransformation _reg_KLdivergence _reg_ssd _reg_mutualinformation _reg_tools _reg_ReadWriteImage)
#-----------------------------------------------------------------------------
ENDIF(USE_CUDA)
#-----------------------------------------------------------------------------
INSTALL(TARGETS ${NAME}
        RUNTIME DESTINATION bin COMPONENT RuntimeLibraries
        LIBRARY DESTINATION lib COMPONENT RuntimeLibraries
        ARCHIVE DESTINATION lib COMPONENT RuntimeLibraries
)
INSTALL(FILES ${NAME}.h ${NAME}.cpp DESTINATION include COMPONENT Development)
INSTALL(FILES ${NAME}2.h ${NAME}2.cpp DESTINATION include COMPONENT Development)
INSTALL(FILES ${NAME}_sym.h ${NAME}_sym.cpp DESTINATION include COMPONENT Development)
#-----------------------------------------------------------------------------
## BUILD THE ALADIN LIBRARY CPU
set(NAME _reg_aladin)
IF(WIN32)
    ADD_LIBRARY(${NAME} ${NAME}.h ${NAME}.cpp ${NAME}_sym.h ${NAME}_sym.cpp)
ELSE(WIN32)
    ADD_LIBRARY(${NAME} SHARED ${NAME}.h ${NAME}.cpp ${NAME}_sym.h ${NAME}_sym.cpp)
ENDIF(WIN32)
TARGET_LINK_LIBRARIES(${NAME} _reg_localTransformation _reg_blockMatching _reg_resampling _reg_globalTransformation _reg_ssd _reg_mutualinformation _reg_tools _reg_ReadWriteImage)
INSTALL(TARGETS ${NAME}
        RUNTIME DESTINATION bin COMPONENT RuntimeLibraries
        LIBRARY DESTINATION lib COMPONENT RuntimeLibraries
        ARCHIVE DESTINATION lib COMPONENT RuntimeLibraries
)
INSTALL(FILES ${NAME}.h ${NAME}.cpp _reg_macros.h DESTINATION include COMPONENT Development)
#-----------------------------------------------------------------------------
## BUILD THE TPS LIBRARY CPU
set(NAME _reg_thinPlateSpline)
IF(WIN32)
    ADD_LIBRARY(${NAME} ${NAME}.h ${NAME}.cpp)
ELSE(WIN32)
    ADD_LIBRARY(${NAME} SHARED ${NAME}.h ${NAME}.cpp)
ENDIF(WIN32)
    TARGET_LINK_LIBRARIES(${NAME} _reg_tools _reg_ReadWriteImage)
INSTALL(TARGETS ${NAME}
        RUNTIME DESTINATION bin COMPONENT RuntimeLibraries
        LIBRARY DESTINATION lib COMPONENT RuntimeLibraries
        ARCHIVE DESTINATION lib COMPONENT RuntimeLibraries
)
INSTALL(FILES ${NAME}.h ${NAME}.cpp DESTINATION include COMPONENT Development)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
