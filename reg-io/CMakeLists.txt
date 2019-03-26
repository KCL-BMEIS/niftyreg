# The z lib library is build only if it was not found on the system
if(NOT ZLIB_FOUND OR BUILD_ALL_DEP)
    subdirs(zlib)
endif(NOT ZLIB_FOUND OR BUILD_ALL_DEP)

# Build the nifti file format library
subdirs(nifti)

set(LIBRARIES reg_nifti z)

# Build the png library if required
subdirs(png)
set(LIBRARIES ${LIBRARIES} reg_png)

# Build the NRRD file format library if required
if(USE_NRRD)
  add_definitions(-D_USE_NRRD)
  subdirs(nrrd)
  set(LIBRARIES ${LIBRARIES} reg_nrrd)
endif(USE_NRRD)

SET(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES};${LIBRARIES}")

# Create the reg_io library
add_library(_reg_ReadWriteImage _reg_ReadWriteImage.h _reg_ReadWriteImage.cpp
_reg_ReadWriteMatrix.h _reg_ReadWriteMatrix.cpp _reg_ReadWriteBinary.h
_reg_ReadWriteBinary.cpp _reg_stringFormat.h _reg_stringFormat.cpp)
target_link_libraries(_reg_ReadWriteImage ${LIBRARIES})
install(TARGETS _reg_ReadWriteImage
        RUNTIME DESTINATION bin COMPONENT Development
        LIBRARY DESTINATION lib COMPONENT Development
        ARCHIVE DESTINATION lib COMPONENT Development
)
install(FILES _reg_ReadWriteImage.h _reg_ReadWriteMatrix.h _reg_stringFormat.h DESTINATION include COMPONENT Development)
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES};_reg_ReadWriteImage")
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES}" PARENT_SCOPE)