#-----------------------------------------------------------------------------
SET(NAME _reg_resampling)
ADD_LIBRARY(${NAME} ${NAME}.h ${NAME}.cpp)
INSTALL(TARGETS ${NAME}
	RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
	LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
#-----------------------------------------------------------------------------
SET(NAME _reg_affineTransformation)
ADD_LIBRARY(${NAME} ${NAME}.h ${NAME}.cpp)
INSTALL(TARGETS ${NAME}
	RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
	LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
#-----------------------------------------------------------------------------
IF(BUILD_F3D)
	SET(NAME _reg_bspline)
	ADD_LIBRARY(${NAME} ${NAME}.h ${NAME}.cpp)
	INSTALL(TARGETS ${NAME}
		RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
		LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
		ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	)
#-----------------------------------------------------------------------------
	SET(NAME _reg_mutualinformation)
	ADD_LIBRARY(${NAME} ${NAME}.h ${NAME}.cpp)
	INSTALL(TARGETS ${NAME}
		RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
		LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
		ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	)
#-----------------------------------------------------------------------------
        SET(NAME _reg_ssd)
        ADD_LIBRARY(${NAME} ${NAME}.h ${NAME}.cpp)
        INSTALL(TARGETS ${NAME}
                RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
                LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
                ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        )
ENDIF(BUILD_F3D)
#-----------------------------------------------------------------------------
SET(NAME _reg_tools)
ADD_LIBRARY(${NAME} ${NAME}.h ${NAME}.cpp)
INSTALL(TARGETS ${NAME}
	RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
	LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
#-----------------------------------------------------------------------------
IF(BUILD_ALADIN)
	SET(NAME _reg_blockMatching)
	ADD_LIBRARY(${NAME} ${NAME}.h ${NAME}.cpp)
	INSTALL(TARGETS ${NAME}
		RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
		LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
		ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	)
ENDIF(BUILD_ALADIN)
#-----------------------------------------------------------------------------
