

############################################################
############################################################
# The following lines have been extracted from ITK4
# http://www.itk.org
# and slighlty modified for some of them to fit NiftyReg
############################################################
############################################################
# Turn on TEEM_BUILD so that the proper dll export def's are
# used on windows builds.
ADD_DEFINITIONS(-DTEEM_BUILD=1)
ADD_DEFINITIONS(-DTEEM_DIO=0)
ADD_DEFINITIONS(-DTEEM_ZLIB=1)

include(TestBigEndian)
test_big_endian(CMAKE_WORDS_BIGENDIAN)

# Configure files with settings for use by the build.
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/NrrdIO/NrrdConfigure.h.in
  ${CMAKE_BINARY_DIR}/NrrdConfigure.h)
# Need to be able to fine NrrdConfig.h
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR})

# Find platform-specific differences in the handling of IEEE floating point
# special values.
include(${CMAKE_CURRENT_SOURCE_DIR}/NrrdIO/CMake/TestQnanhibit.cmake)
test_qnanhibit(QNANHIBIT_VALUE ${CMAKE_CURRENT_SOURCE_DIR}/NrrdIO/CMake)
if(QNANHIBIT_VALUE)
   set(QNANHIBIT 1 CACHE INTERNAL "The 22nd bit of 32-bit floating-point quiet NaN.")
else(QNANHIBIT_VALUE)
   set(QNANHIBIT 0 CACHE INTERNAL "The 22nd bit of 32-bit floating-point quiet NaN.")
endif(QNANHIBIT_VALUE)

#The QNANHIBIT variable is configured by the root level CMakeLists.txt
IF(QNANHIBIT)
  ADD_DEFINITIONS(-DTEEM_QNANHIBIT=1)
ELSE(QNANHIBIT)
  ADD_DEFINITIONS(-DTEEM_QNANHIBIT=0)
ENDIF(QNANHIBIT)

SET(nrrdio_SRCS NrrdIO/754.c NrrdIO/mop.c NrrdIO/array.c NrrdIO/parseAir.c NrrdIO/dio.c
NrrdIO/sane.c NrrdIO/endianAir.c NrrdIO/string.c NrrdIO/enum.c NrrdIO/miscAir.c
NrrdIO/biffbiff.c NrrdIO/biffmsg.c NrrdIO/accessors.c NrrdIO/defaultsNrrd.c
NrrdIO/enumsNrrd.c NrrdIO/arraysNrrd.c NrrdIO/methodsNrrd.c NrrdIO/reorder.c NrrdIO/axis.c
NrrdIO/simple.c NrrdIO/comment.c NrrdIO/keyvalue.c NrrdIO/endianNrrd.c NrrdIO/parseNrrd.c
NrrdIO/gzio.c NrrdIO/read.c NrrdIO/write.c NrrdIO/format.c NrrdIO/formatNRRD.c NrrdIO/encoding.c
NrrdIO/encodingRaw.c NrrdIO/encodingAscii.c NrrdIO/encodingHex.c NrrdIO/encodingGzip.c
NrrdIO/subset.c NrrdIO/encodingBzip2.c NrrdIO/formatEPS.c NrrdIO/formatPNG.c NrrdIO/formatPNM.c
NrrdIO/formatText.c NrrdIO/formatVTK.c NrrdIO/biff.h NrrdIO/itk_NrrdIO_mangle.h NrrdIO/NrrdIO.h
NrrdIO/privateAir.h NrrdIO/privateNrrd.h NrrdIO/teem32bit.h NrrdIO/teemDio.h NrrdIO/teemEndian.h
NrrdIO/teemPng.h NrrdIO/teemQnanhibit.h)

ADD_LIBRARY(reg_NrrdIO ${nrrdio_SRCS})
TARGET_LINK_LIBRARIES(reg_NrrdIO ${ITKZLIB_LIBRARIES})
############################################################
############################################################
IF(WIN32)
    ADD_LIBRARY(reg_nrrd reg_nrrd.cpp reg_nrrd.h)
ELSE(WIN32)
    ADD_LIBRARY(reg_nrrd SHARED reg_nrrd.cpp reg_nrrd.h)
ENDIF(WIN32)
    TARGET_LINK_LIBRARIES(reg_nrrd _reg_tools _reg_maths reg_NrrdIO reg_nifti z)
INSTALL(TARGETS reg_nrrd
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
INSTALL(FILES reg_nrrd.h DESTINATION include)
