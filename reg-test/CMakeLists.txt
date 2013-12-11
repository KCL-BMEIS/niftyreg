set(DATA_FOLDER ${CMAKE_SOURCE_DIR}/reg-test/reg-test-data)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
set(EXEC reg_test_interp)
if(USE_CUDA)
		cuda_add_executable(${EXEC} ${EXEC}.cpp)
else(USE_CUDA)
	add_executable(${EXEC} ${EXEC}.cpp)
endif(USE_CUDA)
target_link_libraries(${EXEC} _reg_f3d)
foreach(dim 2 3)
	foreach(type 0 1 3)
		add_test(${EXEC}_${dim}D_${type} ${EXEC} ${dim} ${type})
	endforeach(type)
	if(USE_CUDA)
		add_test(${EXEC}_${dim}D_4 ${EXEC} ${dim} 4)
	endif(USE_CUDA)
endforeach(dim)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
set(EXEC reg_test_convolution)
add_executable(${EXEC} ${EXEC}.cpp)
target_link_libraries(${EXEC} _reg_f3d)
foreach(dim 2 3)
	foreach(type 0 1 2)
		add_test(${EXEC}_${dim}D_${type} ${EXEC} ${dim} ${type})
	endforeach(type)
endforeach(dim)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
set(EXEC reg_test_dti)
add_executable(${EXEC} ${EXEC}.cpp)
target_link_libraries(${EXEC} _reg_f3d)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
set(EXEC reg_test_mat44_operations)
add_executable(${EXEC} ${EXEC}.cpp)
target_link_libraries(${EXEC} _reg_f3d)
add_test(mat44_operations ${EXEC})
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
set(EXEC reg_test_fullReg)
add_executable(${EXEC} ${EXEC}.cpp)
target_link_libraries(${EXEC} _reg_f3d)
foreach(dim 2 3)
	foreach(type 0 1 2 3 4)
		add_test(${EXEC}_${dim}D_${type} ${EXEC} ${dim} ${type})
	endforeach(type)
endforeach(dim)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
if(USE_CUDA)
		set(EXEC reg_test_def_gpu)
		cuda_add_executable(${EXEC} ${EXEC}.cpp)
				target_link_libraries(${EXEC} _reg_f3d)
		foreach(dim 2 3)
				foreach(type def comp exp be beg ajac ajac jac ajacg jacg)
						add_test(${EXEC}_${dim}D_${type} ${EXEC} ${dim} ${type})
				endforeach(type)
		endforeach(dim)
#-----------------------------------------------------------------------------
	set(EXEC reg_test_measure_gpu)
	cuda_add_executable(${EXEC} ${EXEC}.cpp)
	target_link_libraries(${EXEC} _reg_f3d)
	foreach(dim 2 3)
		foreach(type spa nmig ssd ssdg)
			add_test(${EXEC}_${dim}D_${type} ${EXEC} ${dim} ${type})
		endforeach(type)
	endforeach(dim)
#-----------------------------------------------------------------------------
		set(EXEC reg_test_optimiser_gpu)
		cuda_add_executable(${EXEC} ${EXEC}.cpp)
		target_link_libraries(${EXEC} _reg_f3d)
		foreach(dim 2 3)
				foreach(type descent conjugate)
								add_test(${EXEC}_${dim}D_${type} ${EXEC} ${dim} ${type})
				endforeach(type)
		endforeach(dim)
#-----------------------------------------------------------------------------
endif(USE_CUDA)
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

