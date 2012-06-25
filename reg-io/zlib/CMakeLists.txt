#-----------------------------------------------------------------------------
IF(NOT ZLIB_FOUND)
    ADD_LIBRARY(z adler32.c compress.c crc32.c gzio.c uncompr.c deflate.c trees.c zutil.c inflate.c infback.c inftrees.c inffast.c)
    INSTALL(TARGETS z
            RUNTIME DESTINATION bin
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
    )
    INSTALL(FILES zlib.h zutil.h DESTINATION include)
    SET(ZLIB_LIBRARY "z")
ENDIF(NOT ZLIB_FOUND)
#-----------------------------------------------------------------------------
