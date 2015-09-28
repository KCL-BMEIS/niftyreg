#pragma OPENCL EXTENSION cl_khr_fp64 : enable
/* *************************************************************** */
/* *************************************************************** */
__inline__ double getPosition(__global float* matrix,
														 double* voxel,
														 const unsigned int idx)
{
	size_t index = idx*4;
	return (double)matrix[index++] * voxel[0] +
				 (double)matrix[index++] * voxel[1] +
				 (double)matrix[index++] * voxel[2] +
				 (double)matrix[index];
}
/* *************************************************************** */
/* *************************************************************** */
__kernel void affineKernel2D(__global float* transformationMatrix,
														 __global float* defField,
														 __global int *mask,
														 const uint3 params,
														 const unsigned int composition)
{
	// Get the current coordinate
	const unsigned int x = get_group_id(0)*get_local_size(0) + get_local_id(0);
	const unsigned int y = get_group_id(1)*get_local_size(1) + get_local_id(1);
	const unsigned long index = x + params.x * y;

	if(y<params.y && x<params.x &&  mask[index] >= 0 )
	{
		double voxel[3];
		voxel[2]=0.;

		const unsigned long voxelNumber = params.x*params.y;
		__global float *deformationFieldPtrX = &defField[index] ;
		__global float *deformationFieldPtrY = &deformationFieldPtrX[voxelNumber];

		voxel[0] = composition ? *deformationFieldPtrX : (double)x;
		voxel[1] = composition ? *deformationFieldPtrY : (double)y;

		/* the deformation field (real coordinates) is stored */
		*deformationFieldPtrX = (float)getPosition( transformationMatrix, voxel, 0);
		*deformationFieldPtrY = (float)getPosition( transformationMatrix, voxel, 1);
	}
}
/* *************************************************************** */
__kernel void affineKernel3D(__global float* transformationMatrix,
														 __global float* defField,
														 __global int *mask,
														 const uint3 params,
														 const unsigned int composition)
{
	// Get the current coordinate
	const unsigned int x = get_group_id(0)*get_local_size(0) + get_local_id(0);
	const unsigned int y = get_group_id(1)*get_local_size(1) + get_local_id(1);
	const unsigned int z = get_group_id(2)*get_local_size(2) + get_local_id(2);
	const unsigned long index = x + params.x * ( y + z * params.y);

	if( z<params.z && y<params.y && x<params.x &&  mask[index] >= 0 )
	{
		double voxel[3];

		const unsigned long voxelNumber = params.x*params.y*params.z;
		__global float *deformationFieldPtrX = &defField[index] ;
		__global float *deformationFieldPtrY = &deformationFieldPtrX[voxelNumber];
		__global float *deformationFieldPtrZ = &deformationFieldPtrY[voxelNumber];

		voxel[0] = composition ? *deformationFieldPtrX : (double)x;
		voxel[1] = composition ? *deformationFieldPtrY : (double)y;
		voxel[2] = composition ? *deformationFieldPtrZ : (double)z;

		/* the deformation field (real coordinates) is stored */
		*deformationFieldPtrX = (float)getPosition( transformationMatrix, voxel, 0);
		*deformationFieldPtrY = (float)getPosition( transformationMatrix, voxel, 1);
		*deformationFieldPtrZ = (float)getPosition( transformationMatrix, voxel, 2);
	}
}
/* *************************************************************** */
/* *************************************************************** */
