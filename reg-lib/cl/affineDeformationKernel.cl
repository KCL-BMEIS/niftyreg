//To enable double precision
#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define DOUBLE_SUPPORT_AVAILABLE
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#define DOUBLE_SUPPORT_AVAILABLE
#else
#warning "double precision floating point not supported by OpenCL implementation.";
#endif

#if defined(DOUBLE_SUPPORT_AVAILABLE)
// double
typedef double real_t;
typedef double2 real2_t;
typedef double3 real3_t;
typedef double4 real4_t;
typedef double8 real8_t;
typedef double16 real16_t;
#define PI 3.14159265358979323846

#else

// float
typedef float real_t;
typedef float2 real2_t;
typedef float3 real3_t;
typedef float4 real4_t;
typedef float8 real8_t;
typedef float16 real16_t;
#define PI 3.14159265359f

#endif
/* *************************************************************** */
/* *************************************************************** */
__inline__ real_t getPosition(__global float* matrix,
                              real_t* voxel,
                              const unsigned int idx)
{
   size_t index = idx*4;
   return (real_t)matrix[index++] * voxel[0] +
          (real_t)matrix[index++] * voxel[1] +
          (real_t)matrix[index++] * voxel[2] +
          (real_t)matrix[index];
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
		real_t voxel[3];
		voxel[2]=0.;

		const unsigned long voxelNumber = params.x*params.y;
		__global float *deformationFieldPtrX = &defField[index] ;
		__global float *deformationFieldPtrY = &deformationFieldPtrX[voxelNumber];

		voxel[0] = composition ? *deformationFieldPtrX : (real_t)x;
		voxel[1] = composition ? *deformationFieldPtrY : (real_t)y;

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
		real_t voxel[3];

		const unsigned long voxelNumber = params.x*params.y*params.z;
		__global float *deformationFieldPtrX = &defField[index] ;
		__global float *deformationFieldPtrY = &deformationFieldPtrX[voxelNumber];
		__global float *deformationFieldPtrZ = &deformationFieldPtrY[voxelNumber];

		voxel[0] = composition ? *deformationFieldPtrX : (real_t)x;
		voxel[1] = composition ? *deformationFieldPtrY : (real_t)y;
		voxel[2] = composition ? *deformationFieldPtrZ : (real_t)z;

		/* the deformation field (real coordinates) is stored */
		*deformationFieldPtrX = (float)getPosition( transformationMatrix, voxel, 0);
		*deformationFieldPtrY = (float)getPosition( transformationMatrix, voxel, 1);
		*deformationFieldPtrZ = (float)getPosition( transformationMatrix, voxel, 2);
	}
}
/* *************************************************************** */
/* *************************************************************** */
