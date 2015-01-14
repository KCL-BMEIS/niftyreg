
__inline void getPosition1(float* position, __global float* matrix, float* voxel, const unsigned int idx) {
	position[idx] =
		matrix[idx * 4 + 0] * voxel[0] +
		matrix[idx * 4 + 1] * voxel[1] +
		matrix[idx * 4 + 2] * voxel[2] +
		matrix[idx * 4 + 3];
}
__inline__ float getPosition( __global float* matrix, float* voxel, const unsigned int idx) {
//	if ( voxel[0] == 126.0f && voxel[1] == 90.0f && voxel[2]==59.0f ) printf("(%d): (%f-%f-%f-%f)\n",idx, matrix[idx * 4 + 0], matrix[idx * 4 + 1], matrix[idx * 4 + 2], matrix[idx * 4 + 3]);
	return
		matrix[idx * 4 + 0] * voxel[0] +
		matrix[idx * 4 + 1] * voxel[1] +
		matrix[idx * 4 + 2] * voxel[2] +
		matrix[idx * 4 + 3];
}


__kernel void affineKernel(__global float* transformationMatrix, __global  float* defField,__global  int* mask, const uint3 params, const unsigned int composition) {

    const unsigned long voxelNumber = params.x*params.y*params.z;
    
	__global float *deformationFieldPtrX =  defField ;
	__global float *deformationFieldPtrY = &deformationFieldPtrX[voxelNumber];
	__global float *deformationFieldPtrZ = &deformationFieldPtrY[voxelNumber]; 

	float voxel[3];

	
	const unsigned int z = get_group_id(2)*get_local_size(2) + get_local_id(2);
	const unsigned int y = get_group_id(1)*get_local_size(1) + get_local_id(1);
	const unsigned int x = get_group_id(0)*get_local_size(0) + get_local_id(0);
	const unsigned long index = x + y*params.x + z * params.x * params.y;
	if( z<params.z && y<params.y && x<params.x &&  mask[index] >= 0 ) {

		voxel[0] = composition ? deformationFieldPtrX[index] : (float)x;
		voxel[1] = composition ? deformationFieldPtrY[index] : (float)y;
		voxel[2] = composition ? deformationFieldPtrZ[index] : (float)z;

		/* the deformation field (real coordinates) is stored */
		deformationFieldPtrX[index] = getPosition( transformationMatrix, voxel, 0);
		deformationFieldPtrY[index] = getPosition( transformationMatrix, voxel, 1);
		deformationFieldPtrZ[index] = getPosition( transformationMatrix, voxel, 2);
	
	}
}
