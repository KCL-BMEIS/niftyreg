__inline void getPosition(float* position, float* matrix, float* voxel, const unsigned int idx) {
	position[idx] =
		matrix[idx * 4 + 0] * voxel[0] +
		matrix[idx * 4 + 1] * voxel[1] +
		matrix[idx * 4 + 2] * voxel[2] +
		matrix[idx * 4 + 3];
}

__kernel void affineKernel(__global float* transformationMatrix, __global  float* defField,__global  int* mask, const uint3 params, const unsigned long voxelNumber, const bool composition) {

	float *deformationFieldPtrX =  defField ;
	float *deformationFieldPtrY = &deformationFieldPtrX[voxelNumber];
	float *deformationFieldPtrZ = &deformationFieldPtrY[voxelNumber]; 

	float voxel[3], position[3];

	
	const unsigned int z = get_group_id(2)*get_local_size(2) + get_local_id(2);
	const unsigned int y = get_group_id(1)*get_local_size(1) + get_local_id(1);
	const unsigned int x = get_group_id(0)*get_local_size(0) + get_local_id(0);
	const unsigned long index = x + y*params.x + z * params.x * params.y;
	if( z<params.z && y<params.y && x<params.x &&  mask[index] >= 0 ) {

					voxel[0] = composition ? deformationFieldPtrX[index] : x;
					voxel[1] = composition ? deformationFieldPtrY[index] : y;
					voxel[2] = composition ? deformationFieldPtrZ[index] : z;

					getPosition(position, transformationMatrix, voxel, 0);
					getPosition(position, transformationMatrix, voxel, 1);
					getPosition(position, transformationMatrix, voxel, 2);

					/* the deformation field (real coordinates) is stored */
					deformationFieldPtrX[index] = position[0];
					deformationFieldPtrY[index] = position[1];
					deformationFieldPtrZ[index] = position[2];

	}	
}