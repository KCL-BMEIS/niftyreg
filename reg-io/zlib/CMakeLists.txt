#-----------------------------------------------------------------------------
SET(NAME z)
ADD_LIBRARY(${NAME} adler32.c compress.c crc32.c gzio.c uncompr.c deflate.c trees.c zutil.c inflate.c infback.c inftrees.c inffast.c)
INSTALL(TARGETS ${NAME}
	RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
	LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
INSTALL(FILES zlib.h zutil.h DESTINATION include)
#-----------------------------------------------------------------------------
