

#define REDUCE reduceCustom

 __inline__ float reduceCustom(float data, const unsigned int tid){
	static __local float sData2[64];

	sData2[tid] = data;
	__syncthreads();

	if (tid < 32) sData2[tid] += sData2[tid + 32];
	if (tid < 16) sData2[tid] += sData2[tid + 16];
	if (tid < 8) sData2[tid] += sData2[tid + 8];
	if (tid < 4) sData2[tid] += sData2[tid + 4];
	if (tid < 2) sData2[tid] += sData2[tid + 2];
	if (tid == 0) sData2[0] += sData2[1];

	__syncthreads();
	return sData2[0];
}



__kernel void blockMatchingKernel(__global float *resultPosition, __global float *targetPosition, __global int* mask, __global float* targetMatrix_xyz, uint3 blockDims, unsigned int* definedBlock){

	__shared__ float sResultValues[12 * 12 * 12];

	//const bool is_7_21_11 = blockIdx.x == 7 && blockIdx.y == 21 && blockIdx.z == 11;
//	bool b2_13_10 = blockIdx.x==2&&blockIdx.y==13&&blockIdx.z==10;
	const bool border = blockIdx.x == gridDim.x - 1 || blockIdx.y == gridDim.y - 1 || blockIdx.z == gridDim.z - 1;

	const unsigned int idz = threadIdx.x / 16;
	const unsigned int idy = (threadIdx.x - 16 * idz) / 4;
	const unsigned int idx = threadIdx.x - 16 * idz - 4 * idy;

	const unsigned int bid = blockIdx.x + gridDim.x * blockIdx.y + (gridDim.x * gridDim.y) * blockIdx.z;

	const unsigned int xBaseImage = blockIdx.x * 4;
	const unsigned int yBaseImage = blockIdx.y * 4;
	const unsigned int zBaseImage = blockIdx.z * 4;


	const unsigned int tid = threadIdx.x;//0-blockSize

	const unsigned int xImage = xBaseImage + idx;
	const unsigned int yImage = yBaseImage + idy;
	const unsigned int zImage = zBaseImage + idz;

	const unsigned long imgIdx = xImage + yImage *(c_ImageSize.x) + zImage * (c_ImageSize.x * c_ImageSize.y);
	const bool targetInBounds = xImage < c_ImageSize.x && yImage < c_ImageSize.y && zImage < c_ImageSize.z;

	const int currentBlockIndex = tex1Dfetch(activeBlock_texture, bid);

	if (currentBlockIndex > -1){

		for (int n = -1; n <= 1; n += 1)
		{
			for (int m = -1; m <= 1; m += 1)
			{
				for (int l = -1; l <= 1; l += 1)
				{
					const int x = l * 4 + idx;
					const int y = m * 4 + idy;
					const int z = n * 4 + idz;

					const unsigned int sIdx = (z + 4) * 12 * 12 + (y + 4) * 12 + (x + 4);

					const int xImageIn = xBaseImage + x;
					const int yImageIn = yBaseImage + y;
					const int zImageIn = zBaseImage + z;

					const int indexXYZIn = xImageIn + yImageIn *(c_ImageSize.x) + zImageIn * (c_ImageSize.x * c_ImageSize.y);

					const bool valid = (xImageIn >= 0 && xImageIn < c_ImageSize.x) && (yImageIn >= 0 && yImageIn < c_ImageSize.y) && (zImageIn >= 0 && zImageIn < c_ImageSize.z);
					sResultValues[sIdx] = (valid) ? tex1Dfetch(resultImageArray_texture, indexXYZIn) : nanf("sNaN");

				}
			}
		}

		const float rTargetValue = (targetInBounds) ? tex1Dfetch(targetImageArray_texture, imgIdx) : nanf("sNaN");

		const float targetMean = REDUCE(rTargetValue, tid) / 64;
		const float targetTemp = rTargetValue - targetMean;
		const float targetVar = REDUCE(targetTemp*targetTemp, tid);

		float bestDisplacement[3] = { nanf("sNaN"),0.0f,0.0f };
		float bestCC = 0.0f;

		// iteration over the result blocks
		for (unsigned int n = 1; n < 8; n += 1)
		{
			const bool nBorder = (n < 4 && blockIdx.z == 0) || (n>4 && blockIdx.z >= gridDim.z - 2);
			for (unsigned int m = 1; m < 8; m += 1)
			{
				const bool mBorder = (m < 4 && blockIdx.y == 0) || (m>4 && blockIdx.y >= gridDim.y - 2);
				for (unsigned int l = 1; l < 8; l += 1)
				{

					/*bool nIs_1_0_m3 = l == 1 + 4 && m == 0 + 4 && n == -3 + 4;
					bool nIs_1_0_m2 = l == 1 + 4 && m == 0 + 4 && n == -2 + 4;

					bool condition1 = b2_13_10  && nIs_1_0_m3 && tid==0;
					bool condition2 = b2_13_10  && nIs_1_0_m2 && tid==0;*/

					const bool lBorder = (l < 4 && blockIdx.x == 0) || (l>4 && blockIdx.x >= gridDim.x - 2);

					const unsigned int x = idx + l;
					const unsigned int y = idy + m;
					const unsigned int z = idz + n;

					const unsigned int sIdxIn = z * 144 /*12*12*/ + y * 12 + x;

					const float rResultValue = sResultValues[sIdxIn];
					const bool overlap = isfinite(rResultValue) && targetInBounds;
//					const unsigned int bSize = (nBorder || mBorder || lBorder || border) ? countNans(rResultValue, tid, targetInBounds) : 64;//out
					const unsigned int bSize = (nBorder || mBorder || lBorder || border) ? (unsigned int)REDUCE(overlap?1.0f:0.0f, tid) : 64;//out


					if (bSize > 32 && bSize <= 64){

						const float rChecked = overlap ? rResultValue : 0.0f;
						float newTargetTemp = targetTemp;
						float ttargetvar = targetVar;
						if (bSize < 64){

							const float tChecked = overlap ? rTargetValue : 0.0f;
							const float ttargetMean = REDUCE(tChecked, tid) / bSize;
							newTargetTemp = overlap ? tChecked - ttargetMean : 0.0f;
							ttargetvar = REDUCE(newTargetTemp*newTargetTemp, tid);
						}

						const float resultMean = REDUCE(rChecked, tid) / bSize;
						const float resultTemp = overlap ? rResultValue - resultMean : 0.0f;
						const float resultVar = REDUCE(resultTemp*resultTemp, tid);

						const float sumTargetResult = REDUCE((newTargetTemp)*(resultTemp), tid);
						const float localCC = fabs((sumTargetResult) / sqrtf(ttargetvar*resultVar));

						/*if (condition1) printf("GPU -3 | sze: %d | TMN: %f | TVR: %f | RMN: %f |RVR %f | STR: %f | LCC: %f\n", bSize, targetMean, targetVar, resultMean, resultVar, sumTargetResult, localCC);
						if (condition2) printf("GPU -2 | sze: %d | TMN: %f | TVR: %f | RMN: %f |RVR %f | STR: %f | LCC: %f\n", bSize, targetMean, targetVar, resultMean, resultVar, sumTargetResult, localCC);
*/
						if (tid == 0 && localCC > bestCC) {
							bestCC = localCC;
							bestDisplacement[0] = l - 4.0f;
							bestDisplacement[1] = m - 4.0f;
							bestDisplacement[2] = n - 4.0f;
						}

					}
				}
			}
		}

		if (tid == 0 && isfinite(bestDisplacement[0])) {

//			if (b2_13_10) printf("disp: %f-%f-%f\n", bestDisplacement[0],bestDisplacement[1], bestDisplacement[2]);
			const unsigned int posIdx = 3 * atomicAdd(&(definedBlock[0]), 1);
			//printf("%d: %d \n", definedBlock[0], bid);
			resultPosition += posIdx;
			targetPosition += posIdx;

			const float targetPosition_temp[3] = {blockIdx.x*BLOCK_WIDTH,blockIdx.y*BLOCK_WIDTH, blockIdx.z*BLOCK_WIDTH };

			bestDisplacement[0] += targetPosition_temp[0];
			bestDisplacement[1] += targetPosition_temp[1];
			bestDisplacement[2] += targetPosition_temp[2];

			//float  tempPosition[3];
			reg_mat44_mul_cuda(targetMatrix_xyz, targetPosition_temp, targetPosition);
			reg_mat44_mul_cuda(targetMatrix_xyz, bestDisplacement, resultPosition);

		}
	}

}