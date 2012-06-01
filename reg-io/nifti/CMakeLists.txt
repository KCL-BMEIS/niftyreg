#-----------------------------------------------------------------------------
ADD_DEFINITIONS(-DHAVE_ZLIB)
SET(NAME reg_nifti)
ADD_LIBRARY(${NAME} nifti1.h nifti1_io.c nifti1_io.h znzlib.c znzlib.h)
TARGET_LINK_LIBRARIES(${NAME} z)
INSTALL(TARGETS ${NAME}
	RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
	LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
INSTALL(FILES nifti1_io.h znzlib.h nifti1.h DESTINATION include)
#-----------------------------------------------------------------------------
