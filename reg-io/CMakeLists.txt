# The z lib library is build only if it was not found on the system
IF(NOT ZLIB_FOUND)
    SUBDIRS(zlib)
ENDIF(NOT ZLIB_FOUND)

# Build the nifti file format library
SUBDIRS(nifti)

SET(LIBRARIES reg_nifti ${ZLIB_LIBRARY})

# Build the png library if required
IF(USE_PNG)
    SUBDIRS(png)
    SET(LIBRARIES ${LIBRARIES} reg_png)
ENDIF(USE_PNG)

# Build the NRRD file format library if required
IF(USE_NRRD)
    SUBDIRS(nrrd)
    SET(LIBRARIES ${LIBRARIES} reg_nrrd)
ENDIF(USE_NRRD)

# Create the reg_io library
ADD_LIBRARY(_reg_ReadWriteImage _reg_ReadWriteImage.h _reg_ReadWriteImage.cpp)
TARGET_LINK_LIBRARIES(_reg_ReadWriteImage ${LIBRARIES})
INSTALL(TARGETS _reg_ReadWriteImage
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
INSTALL(FILES _reg_ReadWriteImage.h DESTINATION include)
