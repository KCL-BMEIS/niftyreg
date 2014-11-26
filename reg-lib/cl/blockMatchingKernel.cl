

#define REDUCE reduceCustom2
#define BLOCK_WIDTH 4


__inline__
void reg_mat44_mul_cl(__global float* mat, float const* in, __global float *out)
{
	out[0] = mat[0 * 4 + 0] * in[0] +
		mat[0 * 4 + 1] * in[1] +
		mat[0 * 4 + 2] * in[2] +
		mat[0 * 4 + 3];
	out[1] = mat[1 * 4 + 0] * in[0] +
		mat[1 * 4 + 1] * in[1] +
		mat[1 * 4 + 2] * in[2] +
		mat[1 * 4 + 3];
	out[2] = mat[2 * 4 + 0] * in[0] +
		mat[2 * 4 + 1] * in[1] +
		mat[2 * 4 + 2] * in[2] +
		mat[2 * 4 + 3];
}

// __inline__ float reduceCustom(__local float* sData2, const unsigned int tid){
//	
//
//    barrier(CLK_LOCAL_MEM_FENCE);
//     
//	if (tid < 32) sData2[tid] += sData2[tid + 32];
//	if (tid < 16) sData2[tid] += sData2[tid + 16];
//	if (tid < 8) sData2[tid] += sData2[tid + 8];
//	if (tid < 4) sData2[tid] += sData2[tid + 4];
//	if (tid < 2) sData2[tid] += sData2[tid + 2];
//	if (tid == 0) sData2[0] += sData2[1];
//
//	barrier(CLK_LOCAL_MEM_FENCE);
//	return sData2[0];
//}

void reduce(__local float* sData2, float data, const unsigned int tid) {

    
    // Perform parallel reduction

//    sData2[tid] = data;
//    barrier(CLK_LOCAL_MEM_FENCE);
//    for(int offset = get_local_size(0) / 2;
//        offset > 0;
//        offset = offset / 2) {
//        if (local_index < offset) {
//            float other = sData2[local_index + offset];
//            float mine = sData2[local_index];
//            sData2[local_index] = (mine < other) ? mine : other;
//        }
//        barrier(CLK_LOCAL_MEM_FENCE);
//    }
//    if (local_index == 0) {
//        result[get_group_id(0)] = sData2[0];
//    }
}

__inline__ float reduceCustom2(__local float* sData2, float data, const unsigned int tid){

    sData2[tid] = data;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    float temp =0.0f;
//    if (tid < 32) sData2[tid] += sData2[tid + 32];
//    if (tid < 16) sData2[tid] += sData2[tid + 16];
//    if (tid < 8) sData2[tid] += sData2[tid + 8];
//    if (tid < 4) sData2[tid] += sData2[tid + 4];
//    if (tid < 2) sData2[tid] += sData2[tid + 2];
//    if (tid == 0) sData2[0] += sData2[1];
    for(int i =0;i<64;i++)
        temp += sData2[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    return temp;
}

//__kernel void blockMatchingKernel(
//                                  __global float* resultImageArray,
//                                  __global float* targetImageArray,
//                                  __global float *resultPosition,
//                                  __global float *targetPosition,
//                                  __global int *activeBlock,
//                                  __global int* mask,
//                                  __global float* targetMatrix_xyz,
//                                  __global unsigned int* definedBlock,
//                                  uint3 c_ImageSize
//                                  ){
//    
//    __local float sResultValues[12 * 12 * 12];
//    __local float sData[64];
//    
//    
//    
//    
//    const bool border = (get_group_id(0) == get_num_groups(0) - 1 )|| (get_group_id(1) == get_num_groups(1) - 1 )|| (get_group_id(2) == get_num_groups(2) - 1);
//    
//    const unsigned int idx = get_local_id(0) ;
//    const unsigned int idy = get_local_id(1) ;
//    const unsigned int idz = get_local_id(2);
//    
//    const unsigned int bid = get_group_id(0) + get_num_groups(0) * get_group_id(1) + (get_num_groups(0) * get_num_groups(1)) * get_group_id(2);
//    
//    const unsigned int xBaseImage = get_group_id(0) * 4;
//    const unsigned int yBaseImage = get_group_id(1) * 4;
//    const unsigned int zBaseImage = get_group_id(2) * 4;
//    
//    
//    const unsigned int tid = idz*16 + idy*4 + idx;
//    
//    const unsigned int xImage = xBaseImage + idx;
//    const unsigned int yImage = yBaseImage + idy;
//    const unsigned int zImage = zBaseImage + idz;
//    
//    const unsigned long imgIdx = xImage + yImage *(c_ImageSize.x) + zImage * (c_ImageSize.x * c_ImageSize.y);
//    const bool targetInBounds = xImage < c_ImageSize.x && yImage < c_ImageSize.y && zImage < c_ImageSize.z;
//    
//    const int currentBlockIndex = activeBlock[bid];
//
//    if (currentBlockIndex > -1){
//        
//        for (int n = -1; n <= 1; n += 1)
//        {
//            for (int m = -1; m <= 1; m += 1)
//            {
//                for (int l = -1; l <= 1; l += 1)
//                {
//                    const int x = l * 4 + idx;
//                    const int y = m * 4 + idy;
//                    const int z = n * 4 + idz;
//                    
//                    const unsigned int sIdx = (z + 4) * 12 * 12 + (y + 4) * 12 + (x + 4);
//                    
//                    const int xImageIn = xBaseImage + x;
//                    const int yImageIn = yBaseImage + y;
//                    const int zImageIn = zBaseImage + z;
//                    
//                    const int indexXYZIn = xImageIn + yImageIn *(c_ImageSize.x) + zImageIn * (c_ImageSize.x * c_ImageSize.y);
//                    
//                    const bool valid = (xImageIn >= 0 && xImageIn < c_ImageSize.x) && (yImageIn >= 0 && yImageIn < c_ImageSize.y) && (zImageIn >= 0 && zImageIn < c_ImageSize.z);
//                    sResultValues[sIdx] = (valid) ? resultImageArray[ indexXYZIn] : NAN;
//                    
//                }
//            }
//        }
//        
//        sData[tid] = targetInBounds?1.0f:0.0f;
//        const unsigned int tSize = ( border) ? (unsigned int)reduceCustom(sData, tid) : 64;//out
//        
//        const float rTargetValue = targetInBounds ? targetImageArray[imgIdx] : 0.0f;
//        
//        sData[tid] = rTargetValue;
//        const float targetMean = reduceCustom(sData, tid) / tSize;
//        const float targetTemp = targetInBounds ? rTargetValue - targetMean:0.0f;
//        
//        sData[tid] = targetTemp*targetTemp;
//        const float targetVar = reduceCustom(sData, tid);
//        
//        float bestDisplacement[3] = { NAN,0.0f,0.0f };
//        float bestCC = 0.0f;
//
//        // iteration over the result blocks
//        for (unsigned int n = 1; n < 8; n += 1)
//        {
//            const bool nBorder = (n < 4 && get_group_id(2) == 0) || (n>4 && get_group_id(2) >= get_num_groups(2) - 2);
//            for (unsigned int m = 1; m < 8; m += 1)
//            {
//                const bool mBorder = (m < 4 && get_group_id(1) == 0) || (m>4 && get_group_id(1) >= get_num_groups(1) - 2);
//                for (unsigned int l = 1; l < 8; l += 1)
//                {
//                    
//                    
//                    const bool lBorder = (l < 4 && get_group_id(0) == 0) || (l>4 && get_group_id(0) >= get_num_groups(0) - 2);
//                    
//                    const unsigned int x = idx + l;
//                    const unsigned int y = idy + m;
//                    const unsigned int z = idz + n;
//                    
//                    const unsigned int sIdxIn = z * 144 /*12*12*/ + y * 12 + x;
//                    
//                    const float rResultValue = sResultValues[sIdxIn];
//                    const bool overlap = isfinite(rResultValue) && targetInBounds;
//                    
//                    sData[tid] = overlap?1.0f:0.0f;
//                    const unsigned int bSize = (nBorder || mBorder || lBorder || border) ? (unsigned int)reduceCustom(sData, tid) : 64;//out
//                    
//                    
//                    if (bSize > 32 ){
//                        
//                        const float rChecked = overlap ? rResultValue : 0.0f;
//                        float newTargetTemp = targetTemp;
//                        float ttargetvar = targetVar;
//                        if (bSize < 64){
//                            
//                            const float tChecked = overlap ? rTargetValue : 0.0f;
//                            
//                            sData[tid] = tChecked;
//                            const float ttargetMean = reduceCustom(sData, tid) / bSize;
//                            newTargetTemp = overlap ? tChecked - ttargetMean : 0.0f;
//                            
//                            sData[tid] = newTargetTemp*newTargetTemp;
//                            ttargetvar = reduceCustom(sData, tid);
//                        }
//                        
//                        sData[tid] = rChecked;
//                        const float resultMean = reduceCustom(sData, tid)/ bSize;
//
//                        const float resultTemp = overlap ? rResultValue - resultMean : 0.0f;
//                        
//                        
//                        sData[tid] = resultTemp*resultTemp;
//                        const float resultVar = reduceCustom(sData, tid);
//                        
//                        
//                        sData[tid] = (newTargetTemp)*(resultTemp);
//                        const float sumTargetResult = reduceCustom(sData, tid);
//                        const float localCC = fabs((sumTargetResult) / sqrt(ttargetvar*resultVar));
//                        
//                        
//                        if (tid == 0 && localCC > bestCC) {
//                            bestCC = localCC;
//                            bestDisplacement[0] = l - 4.0f;
//                            bestDisplacement[1] = m - 4.0f;
//                            bestDisplacement[2] = n - 4.0f;
//                        }
//                        
//                    }
//                }
//            }
//        }
//        
//        if (tid == 0 && isfinite(bestDisplacement[0])) {
//            
//            const unsigned int posIdx = 3 * atomic_add(definedBlock, 1);
//            resultPosition += posIdx;
//            targetPosition += posIdx;
//            
//            const float targetPosition_temp[3] = {get_group_id(0)*BLOCK_WIDTH,get_group_id(1)*BLOCK_WIDTH, get_group_id(2)*BLOCK_WIDTH };
//            
//            bestDisplacement[0] += targetPosition_temp[0];
//            bestDisplacement[1] += targetPosition_temp[1];
//            bestDisplacement[2] += targetPosition_temp[2];
//            
//            //float  tempPosition[3];
//            reg_mat44_mul_cl(targetMatrix_xyz, targetPosition_temp, targetPosition);
//            reg_mat44_mul_cl(targetMatrix_xyz, bestDisplacement, resultPosition);
//            
//        }
//    }
//}

//__kernel void blockMatchingKernel2(__global float* resultImageArray, __global float* targetImageArray,  __global float *resultPosition, __global float *targetPosition,__global int *activeBlock, __global int* mask, __global float* targetMatrix_xyz, __global unsigned int* definedBlock, uint3 c_ImageSize){
//
//	__local float sResultValues[12 * 12 * 12];
//
//	const bool border = (get_group_id(0) == get_num_groups(0) - 1 )|| (get_group_id(1) == get_num_groups(1) - 1 )|| (get_group_id(2) == get_num_groups(2) - 1);
//
//	const unsigned int idz = get_local_id(0) / 16;
//	const unsigned int idy = (get_local_id(0) - 16 * idz) / 4;
//	const unsigned int idx = get_local_id(0) - 16 * idz - 4 * idy;
//
//	const unsigned int bid = get_group_id(0) + get_num_groups(0) * get_group_id(1) + (get_num_groups(0) * get_num_groups(1)) * get_group_id(2);
//
//	const unsigned int xBaseImage = get_group_id(0) * 4;
//	const unsigned int yBaseImage = get_group_id(1) * 4;
//	const unsigned int zBaseImage = get_group_id(2) * 4;
//
//
//	const unsigned int tid = get_local_id(0);//0-blockSize
//
//	const unsigned int xImage = xBaseImage + idx;
//	const unsigned int yImage = yBaseImage + idy;
//	const unsigned int zImage = zBaseImage + idz;
//
//	const unsigned long imgIdx = xImage + yImage *(c_ImageSize.x) + zImage * (c_ImageSize.x * c_ImageSize.y);
//	const bool targetInBounds = xImage < c_ImageSize.x && yImage < c_ImageSize.y && zImage < c_ImageSize.z;
//
//	const int currentBlockIndex = activeBlock[bid];
//
//	if (currentBlockIndex > -1){
//
//		for (int n = -1; n <= 1; n += 1)
//		{
//			for (int m = -1; m <= 1; m += 1)
//			{
//				for (int l = -1; l <= 1; l += 1)
//				{
//					const int x = l * 4 + idx;
//					const int y = m * 4 + idy;
//					const int z = n * 4 + idz;
//
//					const unsigned int sIdx = (z + 4) * 12 * 12 + (y + 4) * 12 + (x + 4);
//
//					const int xImageIn = xBaseImage + x;
//					const int yImageIn = yBaseImage + y;
//					const int zImageIn = zBaseImage + z;
//
//					const int indexXYZIn = xImageIn + yImageIn *(c_ImageSize.x) + zImageIn * (c_ImageSize.x * c_ImageSize.y);
//
//					const bool valid = (xImageIn >= 0 && xImageIn < c_ImageSize.x) && (yImageIn >= 0 && yImageIn < c_ImageSize.y) && (zImageIn >= 0 && zImageIn < c_ImageSize.z);
//					sResultValues[sIdx] = (valid) ? resultImageArray[ indexXYZIn] : NAN;
//
//				}
//			}
//		}
//
//		const float rTargetValue = targetInBounds ? targetImageArray[imgIdx] : NAN;
//
//		const float targetMean = reduceCustom2(rTargetValue, tid) / 64;
//		const float targetTemp = rTargetValue - targetMean;
//		const float targetVar = reduceCustom2(targetTemp*targetTemp, tid);
//
//		float bestDisplacement[3] = { NAN,0.0f,0.0f };
//		float bestCC = 0.0f;
//
//		// iteration over the result blocks
//		for (unsigned int n = 1; n < 8; n += 1)
//		{
//			const bool nBorder = (n < 4 && get_group_id(2) == 0) || (n>4 && get_group_id(2) >= get_num_groups(2) - 2);
//			for (unsigned int m = 1; m < 8; m += 1)
//			{
//				const bool mBorder = (m < 4 && get_group_id(1) == 0) || (m>4 && get_group_id(1) >= get_num_groups(1) - 2);
//				for (unsigned int l = 1; l < 8; l += 1)
//				{
//
//
//					const bool lBorder = (l < 4 && get_group_id(0) == 0) || (l>4 && get_group_id(0) >= get_num_groups(0) - 2);
//
//					const unsigned int x = idx + l;
//					const unsigned int y = idy + m;
//					const unsigned int z = idz + n;
//
//					const unsigned int sIdxIn = z * 144 /*12*12*/ + y * 12 + x;
//
//					const float rResultValue = sResultValues[sIdxIn];
//					const bool overlap = isfinite(rResultValue) && targetInBounds;
////					const unsigned int bSize = (nBorder || mBorder || lBorder || border) ? countNans(rResultValue, tid, targetInBounds) : 64;//out
//					const unsigned int bSize = (nBorder || mBorder || lBorder || border) ? (unsigned int)reduceCustom2(overlap?1.0f:0.0f, tid) : 64;//out
//
//
//					if (bSize > 32 && bSize <= 64){
//
//						const float rChecked = overlap ? rResultValue : 0.0f;
//						float newTargetTemp = targetTemp;
//						float ttargetvar = targetVar;
//						if (bSize < 64){
//
//							const float tChecked = overlap ? rTargetValue : 0.0f;
//							const float ttargetMean = reduceCustom2(tChecked, tid) / bSize;
//							newTargetTemp = overlap ? tChecked - ttargetMean : 0.0f;
//							ttargetvar = reduceCustom2(newTargetTemp*newTargetTemp, tid);
//						}
//
//						const float resultMean = reduceCustom2(rChecked, tid) / bSize;
//						const float resultTemp = overlap ? rResultValue - resultMean : 0.0f;
//						const float resultVar = reduceCustom2(resultTemp*resultTemp, tid);
//
//						const float sumTargetResult = reduceCustom2((newTargetTemp)*(resultTemp), tid);
//						const float localCC = fabs((sumTargetResult) / sqrt(ttargetvar*resultVar));
//
//
//						if (tid == 0 && localCC > bestCC) {
//							bestCC = localCC;
//							bestDisplacement[0] = l - 4.0f;
//							bestDisplacement[1] = m - 4.0f;
//							bestDisplacement[2] = n - 4.0f;
//						}
//
//					}
//				}
//			}
//		}
//
//		if (tid == 0 && isfinite(bestDisplacement[0])) {
//
//			const unsigned int posIdx = 3 * atomic_add(&(definedBlock[0]), 1);
//			resultPosition += posIdx;
//			targetPosition += posIdx;
//
//			const float targetPosition_temp[3] = {get_group_id(0)*BLOCK_WIDTH,get_group_id(1)*BLOCK_WIDTH, get_group_id(2)*BLOCK_WIDTH };
//
//			bestDisplacement[0] += targetPosition_temp[0];
//			bestDisplacement[1] += targetPosition_temp[1];
//			bestDisplacement[2] += targetPosition_temp[2];
//
//			//float  tempPosition[3];
//			reg_mat44_mul_cl(targetMatrix_xyz, targetPosition_temp, targetPosition);
//			reg_mat44_mul_cl(targetMatrix_xyz, bestDisplacement, resultPosition);
//
//		}
//	}
//
//}

__kernel void blockMatchingKernel3(__global float* resultImageArray, __global float* targetImageArray,  __global float *resultPosition, __global float *targetPosition,__global int *activeBlock, __global int* mask, __global float* targetMatrix_xyz, __global unsigned int* definedBlock, uint3 c_ImageSize){
    
    __local float sResultValues[12 * 12 * 12];
    __local float sData[64];
    
    const bool border = (get_group_id(0) == get_num_groups(0) - 1 )|| (get_group_id(1) == get_num_groups(1) - 1 )|| (get_group_id(2) == get_num_groups(2) - 1);
    
    const unsigned int idx = get_local_id(0) ;
    const unsigned int idy = get_local_id(1) ;
    const unsigned int idz = get_local_id(2);
    
    const unsigned int bid = get_group_id(0) + get_num_groups(0) * get_group_id(1) + (get_num_groups(0) * get_num_groups(1)) * get_group_id(2);
    
   
    const unsigned int xBaseImage = get_group_id(0) * 4;
    const unsigned int yBaseImage = get_group_id(1) * 4;
    const unsigned int zBaseImage = get_group_id(2) * 4;
    
    
    const unsigned int tid = idz*16 + idy*4 + idx;
//     if(bid==0)printf("tid: %d \n", tid);
    const unsigned int xImage = xBaseImage + idx;
    const unsigned int yImage = yBaseImage + idy;
    const unsigned int zImage = zBaseImage + idz;
    
    const unsigned long imgIdx = xImage + yImage *(c_ImageSize.x) + zImage * (c_ImageSize.x * c_ImageSize.y);
    const bool targetInBounds = xImage < c_ImageSize.x && yImage < c_ImageSize.y && zImage < c_ImageSize.z;
    
    const int currentBlockIndex = activeBlock[bid];
    
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
                    sResultValues[sIdx] = (valid) ? resultImageArray[ indexXYZIn] : NAN;
                    
                }
            }
        }
        
//        const unsigned int tSize = ( border) ? (unsigned int)reduceCustom2(sData,targetInBounds?1.0f:0.0f, tid) : 64;//out
        
        const float rTargetValue = targetInBounds ? targetImageArray[imgIdx] : 0.0f;
        const float targetMean = reduceCustom2(sData,rTargetValue, tid) / 64;
//        if (targetMean-sData[0]/64 != 0)printf("tid: %d | %f-%f\n", tid,targetMean, sData[0]/64);
        const float targetTemp = targetInBounds ? rTargetValue - targetMean:0.0f;
        const float targetVar = reduceCustom2(sData,targetTemp*targetTemp, tid);
//        if (targetVar-sData[0] != 0)printf("tid: %d | %f-%f\n", tid,targetVar, sData[0]);
        float bestDisplacement[3] = { NAN,0.0f,0.0f };
        float bestCC = 0.0f;
        
        // iteration over the result blocks
        for (unsigned int n = 1; n < 8; n += 1)
        {
            const bool nBorder = (n < 4 && get_group_id(2) == 0) || (n>4 && get_group_id(2) >= get_num_groups(2) - 2);
            for (unsigned int m = 1; m < 8; m += 1)
            {
                const bool mBorder = (m < 4 && get_group_id(1) == 0) || (m>4 && get_group_id(1) >= get_num_groups(1) - 2);
                for (unsigned int l = 1; l < 8; l += 1)
                {
                    
                    
                    const bool lBorder = (l < 4 && get_group_id(0) == 0) || (l>4 && get_group_id(0) >= get_num_groups(0) - 2);
                    
                    const unsigned int x = idx + l;
                    const unsigned int y = idy + m;
                    const unsigned int z = idz + n;
                    
                    const unsigned int sIdxIn = z * 144 /*12*12*/ + y * 12 + x;
                    
                    const float rResultValue = sResultValues[sIdxIn];
                    const bool overlap = isfinite(rResultValue) && targetInBounds;
                    //					const unsigned int bSize = (nBorder || mBorder || lBorder || border) ? countNans(rResultValue, tid, targetInBounds) : 64;//out
                    const float val = overlap?1.0f:0.0f;
                    const unsigned int bSize = (nBorder || mBorder || lBorder || border) ? (unsigned int)reduceCustom2( sData, val, tid) : 64;//out
//                     if ((nBorder || mBorder || lBorder || border) && ((float)bSize-sData[0]) != 0)printf("bSize tid: %d | %d-%f\n", tid,bSize, sData[0]);
                    
                    if (bSize > 32 ){
                        
                        const float rChecked = overlap ? rResultValue : 0.0f;
                        float newTargetTemp = targetTemp;
                        float ttargetvar = targetVar;
                        if (bSize < 64){
                            
                            const float tChecked = overlap ? rTargetValue : 0.0f;
                            const float ttargetMean = reduceCustom2( sData, tChecked, tid) / bSize;
//                            if (ttargetMean-sData[0]/bSize != 0)printf("ttargetMean tid: %d | %f-%f\n", tid,ttargetMean, sData[0]/bSize);
                            newTargetTemp = overlap ? tChecked - ttargetMean : 0.0f;
                            ttargetvar = reduceCustom2( sData, newTargetTemp*newTargetTemp, tid);
//                            if (ttargetvar-sData[0] != 0)printf("ttargetvar tid: %d | %f-%f\n", tid,ttargetvar, sData[0]);
                        }else if (bSize>64)printf("%d\n", bSize);
                        
                        const float resultMean = reduceCustom2( sData, rChecked, tid) / bSize;
//                        if (resultMean-sData[0]/bSize != 0)printf("resultMean tid: %d | %f-%f\n", tid,resultMean, sData[0]/bSize);
                        const float resultTemp = overlap ? rResultValue - resultMean : 0.0f;
                        const float resultVar = reduceCustom2( sData, resultTemp*resultTemp, tid);
//                        if (resultVar-sData[0] != 0)printf("resultVar tid: %d | %f-%f\n", tid,resultVar, sData[0]);
                        const float sumTargetResult = reduceCustom2( sData, (newTargetTemp)*(resultTemp), tid);
//                        if (sumTargetResult-sData[0] != 0)printf("sumTargetResult tid: %d | %f-%f\n", tid,sumTargetResult, sData[0]);
                        const float localCC = fabs((sumTargetResult) / sqrt(ttargetvar*resultVar));
                        
                        
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
           
            const unsigned int posIdx = 3 * atomic_add(definedBlock, 1);

            resultPosition += posIdx;
            targetPosition += posIdx;
            
//            printf("%f-%f-%f\n", bestDisplacement[0], bestDisplacement[1], bestDisplacement[2]);
            
            const float targetPosition_temp[3] = {get_group_id(0)*BLOCK_WIDTH,get_group_id(1)*BLOCK_WIDTH, get_group_id(2)*BLOCK_WIDTH };
            
            bestDisplacement[0] += targetPosition_temp[0];
            bestDisplacement[1] += targetPosition_temp[1];
            bestDisplacement[2] += targetPosition_temp[2];
            
            //float  tempPosition[3];
            reg_mat44_mul_cl(targetMatrix_xyz, targetPosition_temp, targetPosition);
            reg_mat44_mul_cl(targetMatrix_xyz, bestDisplacement, resultPosition);
            
        }
    }
    
}