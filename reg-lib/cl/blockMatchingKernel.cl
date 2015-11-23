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


#define REDUCE reduceCustom
#define REDUCE2D reduce2DCustom
#define BLOCK_WIDTH 4


/* *************************************************************** */
/* *************************************************************** */
__inline__
void reg2D_mat44_mul_cl(__global float* mat,
                        float const* in,
                        __global float *out)
{
   out[0] = (float)((real_t)mat[0] * (real_t)in[0] +
         (real_t)mat[1] * (real_t)in[1] + (real_t)mat[3]);
   out[1] = (float)((real_t)mat[4] * (real_t)in[0] +
         (real_t)mat[5] * (real_t)in[1] + (real_t)mat[7]);
}
/* *************************************************************** */
/* *************************************************************** */
__inline__
void reg_mat44_mul_cl(__global float* mat,
                      float const* in,
                      __global float *out)
{
	out[0] = (float)((real_t)mat[0] * in[0] + (real_t)mat[1] * in[1] +
			(real_t)mat[2] * in[2] + (real_t)mat[3]);
	out[1] = (float)((real_t)mat[4] * in[0] + (real_t)mat[5] * in[1] +
			(real_t)mat[6] * in[2] + (real_t)mat[7]);
	out[2] = (float)((real_t)mat[8] * in[0] + (real_t)mat[9] * in[1] +
			(real_t)mat[10] * in[2] + (real_t)mat[11]);
}
/* *************************************************************** */
/* *************************************************************** */
__inline__ float reduce2DCustom(__local float* sData2,
                                float data,
                                const unsigned int tid)
{
	sData2[tid] = data;
	barrier(CLK_LOCAL_MEM_FENCE);

	for (unsigned int i = 8; i > 0; i >>= 1){
		if (tid < i) sData2[tid] += sData2[tid + i];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	const float temp = sData2[0];
	barrier(CLK_LOCAL_MEM_FENCE);

	return temp;
}
/* *************************************************************** */
/* *************************************************************** */
__inline__ float reduceCustom(__local float* sData2,
                              float data,
                              const unsigned int tid)
{
	sData2[tid] = data;
	barrier(CLK_LOCAL_MEM_FENCE);

	for (unsigned int i = 32; i > 0; i >>= 1){
		if (tid < i) sData2[tid] += sData2[tid + i];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	const float temp = sData2[0];
	barrier(CLK_LOCAL_MEM_FENCE);

	return temp;
}
/* *************************************************************** */
/* *************************************************************** */
__kernel void blockMatchingKernel2D(__local float *sWarpedValues,
                                    __global float* warpedImageArray,
                                    __global float* referenceImageArray,
                                    __global float *warpedPosition,
                                    __global float *referencePosition,
                                    __global int *totalBlock,
                                    __global int* mask,
                                    __global float* referenceMatrix_xyz,
                                    __global int* definedBlock,
                                    uint4 c_ImageSize)
{
	// Allocate some shared memory
	__local float sData[16];

	// Compute the current block index
	const unsigned int bid = get_group_id(1) * get_num_groups(0) + get_group_id(0);

	// Check if the current block is active
	const int currentBlockIndex = totalBlock[bid];
	if (currentBlockIndex > -1){

		// Assign the current coordonate of the voxel in the block
		const unsigned int idx = get_local_id(0);
		const unsigned int idy = get_local_id(1);
		const unsigned int tid = idy * 4 + idx;

		// Compute the coordinate of the current voxel in the whole image
		const unsigned int xImage = get_group_id(0) * 4 + idx;
		const unsigned int yImage = get_group_id(1) * 4 + idy;

		// Populate shared memory with the warped image values
		for (int y=-1; y<2; ++y) {
			const int yImageIn = yImage + y * 4;
			for (int x=-1; x<2; ++x) {
				const int xImageIn = xImage + x * 4;

				// Compute the index in the local shared memory
				const int sharedIndex = ((y+1)*4+idy)*12+(x+1)*4+idx;

				// Compute the index of the voxel under consideration
				const int indexXYIn = yImageIn * c_ImageSize.x + xImageIn;

				// Check if the current voxel belongs to the image
				const bool valid =
						(xImageIn > -1 && xImageIn < (int)c_ImageSize.x) &&
						(yImageIn > -1 && yImageIn < (int)c_ImageSize.y);
				// Copy the value from the global to the local shared memory
				sWarpedValues[sharedIndex] = (valid && mask[indexXYIn] > -1) ?
							warpedImageArray[indexXYIn] : NAN;
			}
		}

		// Compute the index of the current voxel in the whole image
		const unsigned long voxIndex = yImage * c_ImageSize.x + xImage;
		// Define a boolean to check if the current voxel is in the input image space
		const bool referenceInBounds =
				xImage < c_ImageSize.x &&
				yImage < c_ImageSize.y;
		// Get the value at the current voxel in the reference image
		float rReferenceValue = (referenceInBounds && mask[voxIndex] > -1) ?
					referenceImageArray[voxIndex] : NAN;
		// Check if the reference value is finite
		const bool finiteReference = isfinite(rReferenceValue);
		// The reference value is replace by 0 if non finite so that it has no influence on mean and variance
		rReferenceValue = finiteReference ? rReferenceValue : 0.0f;

		// Compute the number of voxel different from 0
		const unsigned int referenceSize = REDUCE2D(sData, finiteReference ? 1.0f : 0.0f, tid);

		// Define temp variables to store the displacements and measure of similarity
                float bestDisplacement[2] = {NAN, 0.0f};
                float bestCC = 0.0f;

		// Following computation is perform if there are at last half of the voxel are defined
		if (referenceSize > 8){

			// Compute the mean value in the reference image
			const float referenceMean = REDUCE2D(sData, rReferenceValue, tid) / (float)referenceSize;
			// Compute the difference to the mean
			const float referenceTemp = finiteReference ? rReferenceValue - referenceMean : 0.0f;
			// Compute the reference variance
			const float referenceVar = REDUCE2D(sData, referenceTemp*referenceTemp, tid);

			// Iteration of the 7 x 7 blocks in the neighborhood (3*2+1)^2
			// Starts at 1 since we stored to many voxels in the shared
			for (unsigned int y=1; y<8; ++y){
				for (unsigned int x=1; x<8; ++x){

					// Compute the coordinate of the voxel in the shared memory
					const unsigned int sharedIndex = ( y + idy ) * 12 + x + idx;
					// Get the warped value
					const float rWarpedValue = sWarpedValues[sharedIndex];
					// Check if the warped and reference are defined
					const bool overlap = isfinite(rWarpedValue) && finiteReference;
					// Compute the number of defined value in the block
					const unsigned int currentWarpedSize = REDUCE2D(sData, overlap ? 1.0f : 0.0f, tid);

					// Subsequent computation is performed if the more than half the voxel are defined
					if (currentWarpedSize > 8){

						// Store the reference variance and reference difference to the mean
						float newReferenceTemp = referenceTemp;
						float newReferenceVar = referenceVar;
						// If the defined voxels are different the reference mean and variance are recomputed
						if (currentWarpedSize != referenceSize){
							const float newReferenceValue = overlap ? rReferenceValue : 0.0f;
							const float newReferenceMean = REDUCE2D(sData, newReferenceValue, tid) / (float)currentWarpedSize;
							newReferenceTemp = overlap ? newReferenceValue - newReferenceMean : 0.0f;
							newReferenceVar = REDUCE2D(sData, newReferenceTemp*newReferenceTemp, tid);
						}

						const float rChecked = overlap ? rWarpedValue : 0.0f;
						const float warpedMean = REDUCE2D(sData, rChecked, tid) / (float)currentWarpedSize;
						const float warpedTemp = overlap ? rWarpedValue - warpedMean : 0.0f;
						const float warpedVar = REDUCE2D(sData, warpedTemp*warpedTemp, tid);

						const float sumReferenceWarped = REDUCE2D(sData, (newReferenceTemp)*(warpedTemp), tid);
                                                const float localCC = (newReferenceVar * warpedVar) > 0.0 ? fabs(sumReferenceWarped / sqrt(newReferenceVar*warpedVar)) : 0.0;

                  // Only the first thread of the block can update the final value
                  if (tid == 0 && localCC > bestCC) {
                     bestCC = localCC + 1.0e-7f;
                     bestDisplacement[0] = x - 4.f;
                     bestDisplacement[1] = y - 4.f;
                  }
               }
            }
         }
      }

		// Only the first thread can update the global array with the new result
		if(tid==0){
			const unsigned int posIdx = 2 * currentBlockIndex;
			const float referencePosition_temp[2] = { (float)(xImage), (float)(yImage)};

			bestDisplacement[0] += referencePosition_temp[0];
			bestDisplacement[1] += referencePosition_temp[1];

			reg2D_mat44_mul_cl(referenceMatrix_xyz, referencePosition_temp, &referencePosition[posIdx]);
			reg2D_mat44_mul_cl(referenceMatrix_xyz, bestDisplacement, &warpedPosition[posIdx]);

			if (isfinite(bestDisplacement[0])) {
				atomic_add(definedBlock, 1);
			}
		}
	}
}
/* *************************************************************** */
/* *************************************************************** */
__kernel void blockMatchingKernel3D(__local float *sWarpedValues,
                                    __global float* warpedImageArray,
                                    __global float* referenceImageArray,
                                    __global float *warpedPosition,
                                    __global float *referencePosition,
                                    __global int *totalBlock,
                                    __global int* mask,
                                    __global float* referenceMatrix_xyz,
                                    __global int* definedBlock,
                                    uint4 c_ImageSize)
{
	// Allocate some shared memory
	__local float sData[64];

	// Compute the current block index
	const unsigned int bid = (get_group_id(2)*get_num_groups(1)+get_group_id(1) ) *
			get_num_groups(0) + get_group_id(0);

	// Check if the current block is active
	const int currentBlockIndex = totalBlock[bid];
	if (currentBlockIndex > -1){

		// Assign the current coordonate of the voxel in the block
		const unsigned int idx = get_local_id(0);
		const unsigned int idy = get_local_id(1);
		const unsigned int idz = get_local_id(2);

		// Compute the current voxel index in the block
		const unsigned int tid = idz * 16 + idy * 4 + idx;

		// Compute the coordinate of the current voxel in the whole image
		const unsigned int xImage = get_group_id(0) * 4 + idx;
		const unsigned int yImage = get_group_id(1) * 4 + idy;
		const unsigned int zImage = get_group_id(2) * 4 + idz;

		// Populate shared memory with the warped image values
		for (int n=-1; n<2; ++n) {
			const int zImageIn = zImage + n * 4;
			for (int m=-1; m<2; ++m) {
				const int yImageIn = yImage + m * 4;
				for (int l=-1; l<2; ++l) {
					const int xImageIn = xImage + l * 4;

					// Compute the index in the local shared memory
					const int sharedIndex = (((n+1)*4+idz)*12+(m+1)*4+idy)*12+(l+1)*4+idx;

					// Compute the index of the voxel under consideration
					const unsigned int indexXYZIn = xImageIn + c_ImageSize.x *
							(yImageIn + zImageIn * c_ImageSize.y);

					// Check if the current voxel belongs to the image
					const bool valid =
							(xImageIn > -1 && xImageIn < (int)c_ImageSize.x) &&
							(yImageIn > -1 && yImageIn < (int)c_ImageSize.y) &&
							(zImageIn > -1 && zImageIn < (int)c_ImageSize.z);
					// Copy the value from the global to the local shared memory
					sWarpedValues[sharedIndex] = (valid && mask[indexXYZIn] > -1) ?
								warpedImageArray[indexXYZIn] : NAN;
				}
			}
		}

		// Compute the index of the current voxel in the whole image
		const unsigned int voxIndex = ( zImage * c_ImageSize.y + yImage ) *
				c_ImageSize.x + xImage;
		// Define a boolean to check if the current voxel is in the input image space
		const bool referenceInBounds =
				xImage < c_ImageSize.x &&
				yImage < c_ImageSize.y &&
				zImage < c_ImageSize.z;
		// Get the value at the current voxel in the reference image
		float rReferenceValue = (referenceInBounds && mask[voxIndex] > -1) ?
					referenceImageArray[voxIndex] : NAN;
		// Check if the reference value is finite
		const bool finiteReference = isfinite(rReferenceValue);
		// The reference value is replace by 0 if non finite so that it has no influence on mean and variance
		rReferenceValue = finiteReference ? rReferenceValue : 0.0f;

		// Compute the number of voxel different from 0
		const unsigned int referenceSize = REDUCE(sData, finiteReference ? 1.0f : 0.0f, tid);

		// Define temp variables to store the displacements and measure of similarity
                float bestDisplacement[3] = {NAN, 0.0f, 0.0f };
                float bestCC = 0.0f;

		// Following computation is perform if there are at last half of the voxel are defined
		if (referenceSize > 32){

			// Compute the mean value in the reference image
			const float referenceMean = REDUCE(sData, rReferenceValue, tid) / referenceSize;
			// Compute the difference to the mean
			const float referenceTemp = finiteReference ? rReferenceValue - referenceMean : 0.0f;
			// Compute the reference variance
			const float referenceVar = REDUCE(sData, referenceTemp*referenceTemp, tid);

			// Iteration of the 7 x 7 x 7 blocks in the neighborhood (3*2+1)^3
			// Starts at 1 since we stored to many voxel in the shared
			for (int n=1; n < 8; ++n) {
				for (int m=1; m < 8; ++m) {
					for (int l=1; l < 8; ++l) {

						// Compute the coordinate of the voxel in the shared memory
						const unsigned int sharedIndex = ( (n+idz) * 12 + m + idy ) * 12 + l + idx;

						// Get the warped value
						const float rWarpedValue = sWarpedValues[sharedIndex];
						// Check if the warped and reference are defined
						const bool overlap = isfinite(rWarpedValue) && finiteReference;
						// Compute the number of defined value in the block
						const unsigned int currentWarpedSize = REDUCE(sData, overlap ? 1.0f : 0.0f, tid);

						// Subsequent computation is performed if the more than half the voxel are defined
						if (currentWarpedSize > 32){

							// Store the reference variance and reference difference to the mean
							float newReferenceTemp = referenceTemp;
							float newReferenceVar = referenceVar;
							// If the defined voxels are different the reference mean and variance are recomputed
							if (currentWarpedSize != referenceSize){
								const float newReferenceValue = overlap ? rReferenceValue : 0.0f;
								const float newReferenceMean = REDUCE(sData, newReferenceValue, tid) / currentWarpedSize;
								newReferenceTemp = overlap ? newReferenceValue - newReferenceMean : 0.0f;
								newReferenceVar = REDUCE(sData, newReferenceTemp*newReferenceTemp, tid);
							}

							const float rChecked = overlap ? rWarpedValue : 0.0f;
							const float warpedMean = REDUCE(sData, rChecked, tid) / currentWarpedSize;
							const float warpedTemp = overlap ? rWarpedValue - warpedMean : 0.0f;
							const float warpedVar = REDUCE(sData, warpedTemp*warpedTemp, tid);

							const float sumReferenceWarped = REDUCE(sData, (newReferenceTemp)*(warpedTemp), tid);
                                                        const float localCC = (newReferenceVar * warpedVar) > 0.0 ? fabs((sumReferenceWarped) / sqrt(newReferenceVar*warpedVar)) : 0.0;

							// Only the first thread of the block can update the final value
                                                        if (tid == 0 && localCC > bestCC) {
                                                                bestCC = localCC + 1.0e-7f;
                                                                bestDisplacement[0] = l - 4.f;
                                                                bestDisplacement[1] = m - 4.f;
                                                                bestDisplacement[2] = n - 4.f;
                                                        }
						}
					}
				}
			}
		}

		// Only the first thread can update the global array with the new result
		if (tid==0){
			const unsigned int posIdx = 3 * currentBlockIndex;
			const float referencePosition_temp[3] = { (float)xImage, (float)yImage, (float)zImage};

			bestDisplacement[0] += referencePosition_temp[0];
			bestDisplacement[1] += referencePosition_temp[1];
			bestDisplacement[2] += referencePosition_temp[2];

			reg_mat44_mul_cl(referenceMatrix_xyz, referencePosition_temp, &referencePosition[posIdx]);
			reg_mat44_mul_cl(referenceMatrix_xyz, bestDisplacement, &warpedPosition[posIdx]);
			if (isfinite(bestDisplacement[0])) {
				atomic_add(definedBlock, 1);
			}
		}
	}
}
/* *************************************************************** */
/* *************************************************************** */
