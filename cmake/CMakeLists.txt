#### export configuration for external projects that want to use NiftyReg
# See https://cmake.org/cmake/help/v3.0/manual/cmake-packages.7.html
# Also https://rix0r.nl/blog/2015/08/13/cmake-guide/
# https://coderwall.com/p/qej45g/use-cmake-enabled-libraries-in-your-cmake-project-iii

include(CMakePackageConfigHelpers)
set(ConfigPackageLocation lib/cmake/NiftyReg)

WRITE_BASIC_PACKAGE_VERSION_FILE(${CMAKE_CURRENT_BINARY_DIR}/NIFTYREGConfigVersion.cmake
                                 VERSION ${NR_VERSION}
                                 COMPATIBILITY SameMajorVersion )

# Set STIR_INCLUDE_DIRS before exporting such that it will refer to
# the installed files, not the source.
set (NIFTYREG_INCLUDE_DIRS "include")

# There's always a leading ; in NIFTYREG_LIBRARIES. Remove it
string(SUBSTRING "${NIFTYREG_LIBRARIES}" 1 -1 NIFTYREG_LIBRARIES)

CONFIGURE_PACKAGE_CONFIG_FILE(
  "NIFTYREGConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/NIFTYREGConfig.cmake"
   INSTALL_DESTINATION "${ConfigPackageLocation}"
   PATH_VARS NIFTYREG_INCLUDE_DIRS
)

# install NiftyReg components
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/NIFTYREGConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/NIFTYREGConfigVersion.cmake"
  DESTINATION "${ConfigPackageLocation}"
)

# install our own Find* cmake files
install(
  DIRECTORY
    "${PROJECT_SOURCE_DIR}/src/cmake/"
  DESTINATION "${ConfigPackageLocation}"
    FILES_MATCHING PATTERN "Find*"
)

# install the public headers so that NIFTYREG_INCLUDE_DIRS is usable by external projects
# The in-tree include directories are flat (reg-lib, reg-lib/cpu, reg-io), so the headers
# are installed flat into include/; niftilib, RNifti and znzlib keep their subdirectory
# as they are included with a qualified path (e.g. <znzlib/znzlib.h>)
file(GLOB NR_PUBLIC_HEADERS
  "${PROJECT_SOURCE_DIR}/reg-lib/*.h"
  "${PROJECT_SOURCE_DIR}/reg-lib/*.hpp"
  "${PROJECT_SOURCE_DIR}/reg-lib/cpu/*.h"
  "${PROJECT_SOURCE_DIR}/reg-lib/cpu/*.hpp"
)
install(FILES
  ${NR_PUBLIC_HEADERS}
  "${PROJECT_SOURCE_DIR}/reg-io/_reg_ReadWriteImage.h"
  "${PROJECT_SOURCE_DIR}/reg-io/_reg_ReadWriteMatrix.h"
  "${PROJECT_SOURCE_DIR}/reg-io/RNifti.h"
  "${PROJECT_SOURCE_DIR}/reg-io/png/reg_png.h"
  DESTINATION include COMPONENT Development
)
install(DIRECTORY
  "${PROJECT_SOURCE_DIR}/reg-io/niftilib"
  "${PROJECT_SOURCE_DIR}/reg-io/RNifti"
  "${PROJECT_SOURCE_DIR}/reg-io/znzlib"
  DESTINATION include COMPONENT Development
  FILES_MATCHING PATTERN "*.h"
)
# znzlib.h includes zlib/zlib.h when HAVE_ZLIB is defined
install(FILES
  "${PROJECT_SOURCE_DIR}/reg-io/zlib/zlib.h"
  "${PROJECT_SOURCE_DIR}/reg-io/zlib/zconf.h"
  DESTINATION include/zlib COMPONENT Development
)
if(USE_NRRD)
  install(FILES "${PROJECT_SOURCE_DIR}/reg-io/nrrd/reg_nrrd.h"
    DESTINATION include COMPONENT Development
  )
endif(USE_NRRD)