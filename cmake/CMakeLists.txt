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