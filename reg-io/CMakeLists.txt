# The z lib library is build only if it was not found on the system
if(NOT ZLIB_FOUND OR BUILD_ALL_DEP)
    subdirs(zlib)
endif(NOT ZLIB_FOUND OR BUILD_ALL_DEP)
add_definitions(-DHAVE_ZLIB)

# Build the znz library
subdirs(znzlib)

# Build the nifti file format library
subdirs(niftilib)

# Build the png library if required
subdirs(png)

set(LIBRARIES reg_nifti reg_png)

# Build the NRRD file format library if required
if(USE_NRRD)
  add_definitions(-DUSE_NRRD)
  subdirs(nrrd)
  set(LIBRARIES ${LIBRARIES} reg_nrrd)
endif(USE_NRRD)

SET(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES};${LIBRARIES}")

# Create the reg_io library
add_library(_reg_ReadWriteImage _reg_ReadWriteImage.cpp _reg_ReadWriteMatrix.cpp)
target_link_libraries(_reg_ReadWriteImage ${LIBRARIES})
install(TARGETS _reg_ReadWriteImage
        RUNTIME DESTINATION bin COMPONENT Development
        LIBRARY DESTINATION lib COMPONENT Development
        ARCHIVE DESTINATION lib COMPONENT Development
)
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES};_reg_ReadWriteImage")
set(NIFTYREG_LIBRARIES "${NIFTYREG_LIBRARIES}" PARENT_SCOPE)