IF(BUILD_NIFTYREG)
#-----------------------------------------------------------------------------
	SET(LIST _reg_resampling)
	SET(LIST ${LIST} _reg_affineTransformation)
	SET(LIST ${LIST} _reg_bspline)
	SET(LIST ${LIST} _reg_mutualinformation)
	SET(LIST ${LIST} _reg_ssd)
	SET(LIST ${LIST} _reg_tools)
	SET(LIST ${LIST} _reg_blockMatching)
	FOREACH(NAME ${LIST})
		ADD_LIBRARY(${NAME} ${NAME}.h ${NAME}.cpp)
		INSTALL(TARGETS ${NAME}
			RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
			LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
			ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
		)
	ENDFOREACH(NAME)
#-----------------------------------------------------------------------------
	SET(NAME _reg_bspline_comp)
	ADD_LIBRARY(${NAME} ${NAME}.h ${NAME}.cpp)
	TARGET_LINK_LIBRARIES(_reg_bspline)
	INSTALL(TARGETS ${NAME}
		RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
		LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
		ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	)
#-----------------------------------------------------------------------------
ENDIF(BUILD_NIFTYREG)

