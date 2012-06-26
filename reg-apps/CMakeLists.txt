#-----------------------------------------------------------------------------
ADD_EXECUTABLE(reg_average reg_average.cpp)
TARGET_LINK_LIBRARIES(reg_average _reg_globalTransformation _reg_maths _reg_tools _reg_ReadWriteImage)
#-----------------------------------------------------------------------------
ADD_EXECUTABLE(reg_tools reg_tools.cpp)
TARGET_LINK_LIBRARIES(reg_tools _reg_resampling _reg_localTransformation _reg_tools _reg_globalTransformation _reg_ReadWriteImage)
#-----------------------------------------------------------------------------
ADD_EXECUTABLE(reg_resample reg_resample.cpp)
TARGET_LINK_LIBRARIES(reg_resample _reg_resampling _reg_localTransformation _reg_tools _reg_globalTransformation _reg_ReadWriteImage)
#-----------------------------------------------------------------------------
ADD_EXECUTABLE(reg_transform reg_transform.cpp)
TARGET_LINK_LIBRARIES(reg_transform _reg_thinPlateSpline _reg_resampling _reg_localTransformation _reg_tools _reg_globalTransformation _reg_maths _reg_ReadWriteImage)
#-----------------------------------------------------------------------------
ADD_EXECUTABLE(reg_jacobian reg_jacobian.cpp)
TARGET_LINK_LIBRARIES(reg_jacobian _reg_resampling _reg_localTransformation _reg_tools _reg_globalTransformation _reg_ReadWriteImage)
#-----------------------------------------------------------------------------
if(USE_CUDA)
        CUDA_ADD_EXECUTABLE(reg_f3d reg_f3d.cpp)
else(USE_CUDA)
        ADD_EXECUTABLE(reg_f3d reg_f3d.cpp)
endif(USE_CUDA)
TARGET_LINK_LIBRARIES(reg_f3d _reg_f3d)
#-----------------------------------------------------------------------------
if(USE_CUDA)
        CUDA_ADD_EXECUTABLE(reg_aladin reg_aladin.cpp)
else(USE_CUDA)
        ADD_EXECUTABLE(reg_aladin reg_aladin.cpp)
endif(USE_CUDA)
TARGET_LINK_LIBRARIES(reg_aladin _reg_aladin)
#-----------------------------------------------------------------------------
SET(MODULE_LIST
    reg_resample
    reg_transform
    reg_jacobian
    reg_aladin
    reg_f3d
    )
#-----------------------------------------------------------------------------
if(BUILD_NR_SLICER_EXT)
    set(EXTENSION_NAME "NiftyRegExtension")
    set(EXTENSION_HOMEPAGE "https://sourceforge.net/projects/niftyreg")
    set(EXTENSION_CATEGORY "Registration")
    set(EXTENSION_CONTRIBUTORS "Marc Modat, Pankaj Daga, David Cash, Sebastien Ourselin (UCL)")
    set(EXTENSION_DESCRIPTION "NiftyReg meets slicer, integration of a NiftyReg module within 3DSlicer")
    set(EXTENSION_ICONURL "https://dl.dropbox.com/u/2962833/nifty_reg_logo_128.png")
    set(EXTENSION_SCREENSHOTURLS "http://www.na-mic.org/Wiki/images/4/4c/Nifty_reg_logo.png")
    set(EXTENSION_ACKNOWLEDGEMENTS "This work is partially funded through CBRC grant 168 (UK)")
    set(EXTENSION_MAJOR_VERSION 0)
    set(EXTENSION_MINOR_VERSION 1)
    set(EXTENSION_PATCH_VERSION 0)

    find_package(Slicer REQUIRED)
    include(${Slicer_USE_FILE})

    foreach(MODULE_NAME ${MODULE_LIST})
        install(TARGETS ${MODULE_NAME}
            RUNTIME DESTINATION ${Slicer_INSTALL_CLIMODULES_BIN_DIR} COMPONENT Runtime
            LIBRARY DESTINATION ${Slicer_INSTALL_CLIMODULES_LIB_DIR} COMPONENT Runtime
            ARCHIVE DESTINATION ${Slicer_INSTALL_CLIMODULES_LIB_DIR} COMPONENT Runtime
            )
    endforeach(MODULE_NAME)
    include(${Slicer_EXTENSION_CPACK})
endif(BUILD_NR_SLICER_EXT)
#-----------------------------------------------------------------------------
SET(MODULE_LIST
    ${MODULE_LIST}
    reg_average
    reg_tools
    )
foreach(MODULE_NAME ${MODULE_LIST})
    install(TARGETS ${MODULE_NAME}
        RUNTIME DESTINATION bin COMPONENT Runtime
        LIBRARY DESTINATION lib COMPONENT Runtime
        ARCHIVE DESTINATION lib COMPONENT Runtime
        )
endforeach(MODULE_NAME)
#-----------------------------------------------------------------------------

