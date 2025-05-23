#-----------------------------------------------------------------------------
if(NOT ZLIB_FOUND)
    add_library(z adler32.c compress.c crc32.c deflate.c gzclose.c gzlib.c gzread.c gzwrite.c infback.c inffast.c inflate.c inftrees.c trees.c uncompr.c zutil.c)
    install(TARGETS z
            RUNTIME DESTINATION bin COMPONENT Development
            LIBRARY DESTINATION lib COMPONENT Development
            ARCHIVE DESTINATION lib COMPONENT Development
    )
    set(ZLIB_LIBRARY "z")
endif(NOT ZLIB_FOUND)
#-----------------------------------------------------------------------------
