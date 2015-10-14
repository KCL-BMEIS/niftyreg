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
    out[0] = mat[0 * 4 + 0] * in[0] +
        mat[0 * 4 + 1] * in[1] +
        mat[0 * 4 + 2] * 0 +
        mat[0 * 4 + 3];
    out[1] = mat[1 * 4 + 0] * in[0] +
        mat[1 * 4 + 1] * in[1] +
        mat[1 * 4 + 2] * 0 +
        mat[1 * 4 + 3];
}
/* *************************************************************** */
/* *************************************************************** */
__inline__
void reg_mat44_mul_cl(__global float* mat,
                        float const* in,
                      __global float *out)
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
/* *************************************************************** */
/* *************************************************************** */
__inline__ float reduce2DCustom(__local float* sData2,
                                        float data,
                                  const unsigned int tid)
{
    sData2[tid] = data;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 8; i > 0; i >>= 1){
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

    for (int i = 32; i > 0; i >>= 1){
        if (tid < i) sData2[tid] += sData2[tid + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const float temp = sData2[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    return temp;
}
/* *************************************************************** */
/* *************************************************************** */
__kernel void blockMatchingKernel2D(__local float *sResultValues,
                                    __global float* resultImageArray,
                                    __global float* targetImageArray,
                                    __global float *warpedPosition,
                                    __global float *referencePosition,
                                    __global int *totalBlock,
                                    __global int* mask,
                                    __global float* referenceMatrix_xyz,
                                    __global int* definedBlock,
                                    uint3 c_ImageSize,
                                    const int blocksRange,
                                    const unsigned int stepSize)
{

    const uint numBlocks = blocksRange * 2 + 1;

    __local float sData[16];

    const unsigned int idx = get_local_id(0);
    const unsigned int idy = get_local_id(1);

    const unsigned int bid = get_group_id(0) + get_num_groups(0) * get_group_id(1);

    const unsigned int xBaseImage = get_group_id(0) * 4;
    const unsigned int yBaseImage = get_group_id(1) * 4;

    const unsigned int tid = idy * 4 + idx;

    const unsigned int xImage = xBaseImage + idx;
    const unsigned int yImage = yBaseImage + idy;

    const unsigned long imgIdx = xImage + yImage *(c_ImageSize.x);
    const bool targetInBounds = xImage < c_ImageSize.x && yImage < c_ImageSize.y;

    const int currentBlockIndex = totalBlock[bid];

    __global float* start_warpedPosition = &warpedPosition[0];
    __global float* start_referencePosition = &referencePosition[0];

    if (currentBlockIndex > -1){

        float bestDisplacement[3] = { NAN, 0.0f, 0.0f };
        float bestCC = blocksRange > 1 ? 0.9f : 0.0f;

        //populate shared memory with resultImageArray's values
            for (int m = -1 * blocksRange; m <= blocksRange; m += 1) {
                for (int l = -1 * blocksRange; l <= blocksRange; l += 1) {
                    const int x = l * 4 + idx;
                    const int y = m * 4 + idy;

                    const unsigned int sIdx = (y + blocksRange * 4) * numBlocks * 4 + (x + blocksRange * 4);

                    const int xImageIn = xBaseImage + x;
                    const int yImageIn = yBaseImage + y;

                    const int indexXYZIn = xImageIn + yImageIn *(c_ImageSize.x);

                    const bool valid = (xImageIn >= 0 && xImageIn < c_ImageSize.x) && (yImageIn >= 0 && yImageIn < c_ImageSize.y);
                    sResultValues[sIdx] = (valid && mask[indexXYZIn] > -1) ? resultImageArray[indexXYZIn] : NAN;

                }
            }

        float rTargetValue = (targetInBounds && mask[imgIdx] > -1) ? targetImageArray[imgIdx] : NAN;
        const bool finiteTarget = isfinite(rTargetValue);
        rTargetValue = finiteTarget ? rTargetValue : 0.0f;

        const unsigned int targetSize = REDUCE2D(sData, finiteTarget ? 1.0f : 0.0f, tid);

        if (targetSize > 8){

            const float targetMean = REDUCE2D(sData, rTargetValue, tid) / targetSize;
            const float targetTemp = finiteTarget ? rTargetValue - targetMean : 0.0f;
            const float targetVar = REDUCE2D(sData, targetTemp*targetTemp, tid);

            // iteration over the result blocks (block matching part)
                for (unsigned int m = 1; m < blocksRange * 8 /*2*4*/; m += stepSize) {
                    for (unsigned int l = 1; l < blocksRange * 8 /*2*4*/; l += stepSize) {

                        const unsigned int sIdxIn = (idy + m) * numBlocks * 4 + idx + l;

                        const float rResultValue = sResultValues[sIdxIn];
                        const bool overlap = isfinite(rResultValue) && finiteTarget;
                        const unsigned int bSize = REDUCE2D(sData, overlap ? 1.0f : 0.0f, tid);

                        if (bSize > 8){

                            float newTargetTemp = targetTemp;
                            float newTargetvar = targetVar;
                            if (bSize != targetSize){

                                const float newTargetValue = overlap ? rTargetValue : 0.0f;
                                const float newargetMean = REDUCE2D(sData, newTargetValue, tid) / bSize;
                                newTargetTemp = overlap ? newTargetValue - newargetMean : 0.0f;
                                newTargetvar = REDUCE2D(sData, newTargetTemp*newTargetTemp, tid);
                            }

                            const float rChecked = overlap ? rResultValue : 0.0f;
                            const float resultMean = REDUCE2D(sData, rChecked, tid) / bSize;
                            const float resultTemp = overlap ? rResultValue - resultMean : 0.0f;
                            const float resultVar = REDUCE2D(sData, resultTemp*resultTemp, tid);

                            const float sumTargetResult = REDUCE2D(sData, (newTargetTemp)*(resultTemp), tid);
                            const float localCC = fabs((sumTargetResult) / sqrt(newTargetvar*resultVar));

                            if (tid == 0 && localCC > bestCC) {
                                bestCC = localCC;
                                bestDisplacement[0] = l - 4.0f;
                                bestDisplacement[1] = m - 4.0f;
                                bestDisplacement[2] = 0;
                            }
                        }
                    }
                }
        }

        if (tid == 0 /*&& isfinite(bestDisplacement[0])*/) {
            const unsigned int posIdx = 2 * currentBlockIndex;

            referencePosition = start_referencePosition + posIdx;
            warpedPosition = start_warpedPosition + posIdx;

            const float referencePosition_temp[3] = { (float)xBaseImage, (float)yBaseImage, 0 };

            bestDisplacement[0] += referencePosition_temp[0];
            bestDisplacement[1] += referencePosition_temp[1];
            bestDisplacement[2] += 0;

            reg2D_mat44_mul_cl(referenceMatrix_xyz, referencePosition_temp, referencePosition);
            reg2D_mat44_mul_cl(referenceMatrix_xyz, bestDisplacement, warpedPosition);
            if (isfinite(bestDisplacement[0])) {
                atomic_add(definedBlock, 1);
            }
        }
    }
}
/* *************************************************************** */
/* *************************************************************** */
__kernel void blockMatchingKernel3D(__local float *sResultValues,
                                            __global float* resultImageArray,
                                            __global float* targetImageArray,
                                            __global float *warpedPosition,
                                            __global float *referencePosition,
                                            __global int *totalBlock,
                                            __global int* mask,
                                            __global float* referenceMatrix_xyz,
                                            __global int* definedBlock,
                                            uint3 c_ImageSize,
                                            const int blocksRange,
                                            const unsigned int stepSize)
{

    const uint numBlocks = blocksRange * 2 + 1;

    __local float sData[64];

    const unsigned int idx = get_local_id(0);
    const unsigned int idy = get_local_id(1);
    const unsigned int idz = get_local_id(2);

    const unsigned int bid = get_group_id(0) + get_num_groups(0) * get_group_id(1) + (get_num_groups(0) * get_num_groups(1)) * get_group_id(2);

    const unsigned int xBaseImage = get_group_id(0) * 4;
    const unsigned int yBaseImage = get_group_id(1) * 4;
    const unsigned int zBaseImage = get_group_id(2) * 4;

    const unsigned int tid = idz * 16 + idy * 4 + idx;

    const unsigned int xImage = xBaseImage + idx;
    const unsigned int yImage = yBaseImage + idy;
    const unsigned int zImage = zBaseImage + idz;

    const unsigned long imgIdx = xImage + yImage *(c_ImageSize.x) + zImage * (c_ImageSize.x * c_ImageSize.y);
    const bool targetInBounds = xImage < c_ImageSize.x && yImage < c_ImageSize.y && zImage < c_ImageSize.z;

    const int currentBlockIndex = totalBlock[bid];

    __global float* start_warpedPosition = &warpedPosition[0];
    __global float* start_referencePosition = &referencePosition[0];

    if (currentBlockIndex > -1){

        float bestDisplacement[3] = { NAN, 0.0f, 0.0f };
        float bestCC = blocksRange > 1 ? 0.9f : 0.0f;

        //populate shared memory with resultImageArray's values
        for (int n = -1 * blocksRange; n <= blocksRange; n += 1) {
            for (int m = -1 * blocksRange; m <= blocksRange; m += 1) {
                for (int l = -1 * blocksRange; l <= blocksRange; l += 1) {
                    const int x = l * 4 + idx;
                    const int y = m * 4 + idy;
                    const int z = n * 4 + idz;

                    const unsigned int sIdx = (z + blocksRange * 4)* numBlocks * 4 * numBlocks * 4 + (y + blocksRange * 4) * numBlocks * 4 + (x + blocksRange * 4);

                    const int xImageIn = xBaseImage + x;
                    const int yImageIn = yBaseImage + y;
                    const int zImageIn = zBaseImage + z;

                    const int indexXYZIn = xImageIn + yImageIn *(c_ImageSize.x) + zImageIn * (c_ImageSize.x * c_ImageSize.y);

                    const bool valid = (xImageIn >= 0 && xImageIn < c_ImageSize.x) && (yImageIn >= 0 && yImageIn < c_ImageSize.y) && (zImageIn >= 0 && zImageIn < c_ImageSize.z);
                    sResultValues[sIdx] = (valid && mask[indexXYZIn] > -1) ? resultImageArray[indexXYZIn] : NAN;

                }
            }
        }

        float rTargetValue = (targetInBounds && mask[imgIdx] > -1) ? targetImageArray[imgIdx] : NAN;
        const bool finiteTarget = isfinite(rTargetValue);
        rTargetValue = finiteTarget ? rTargetValue : 0.0f;

        const unsigned int targetSize = REDUCE(sData, finiteTarget ? 1.0f : 0.0f, tid);

        if (targetSize > 32){

            const float targetMean = REDUCE(sData, rTargetValue, tid) / targetSize;
            const float targetTemp = finiteTarget ? rTargetValue - targetMean : 0.0f;
            const float targetVar = REDUCE(sData, targetTemp*targetTemp, tid);

            // iteration over the result blocks (block matching part)
            for (unsigned int n = 1; n < blocksRange * 8 /*2*4*/; n += stepSize) {
                for (unsigned int m = 1; m < blocksRange * 8 /*2*4*/; m += stepSize) {
                    for (unsigned int l = 1; l < blocksRange * 8 /*2*4*/; l += stepSize) {

                        const unsigned int sIdxIn = (idz + n) * numBlocks * 4 * numBlocks * 4 + (idy + m) * numBlocks * 4 + idx + l;

                        const float rResultValue = sResultValues[sIdxIn];
                        const bool overlap = isfinite(rResultValue) && finiteTarget;
                        const unsigned int bSize = REDUCE(sData, overlap ? 1.0f : 0.0f, tid);

                        if (bSize > 32){

                            float newTargetTemp = targetTemp;
                            float newTargetvar = targetVar;
                            if (bSize != targetSize){

                                const float newTargetValue = overlap ? rTargetValue : 0.0f;
                                const float newargetMean = REDUCE(sData, newTargetValue, tid) / bSize;
                                newTargetTemp = overlap ? newTargetValue - newargetMean : 0.0f;
                                newTargetvar = REDUCE(sData, newTargetTemp*newTargetTemp, tid);
                            }

                            const float rChecked = overlap ? rResultValue : 0.0f;
                            const float resultMean = REDUCE(sData, rChecked, tid) / bSize;
                            const float resultTemp = overlap ? rResultValue - resultMean : 0.0f;
                            const float resultVar = REDUCE(sData, resultTemp*resultTemp, tid);

                            const float sumTargetResult = REDUCE(sData, (newTargetTemp)*(resultTemp), tid);
                            const float localCC = fabs((sumTargetResult) / sqrt(newTargetvar*resultVar));

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
        }

        if (tid == 0 /*&& isfinite(bestDisplacement[0])*/) {
            const unsigned int posIdx = 3 * currentBlockIndex;

            referencePosition = start_referencePosition + posIdx;
            warpedPosition = start_warpedPosition + posIdx;

            const float referencePosition_temp[3] = { (float)xBaseImage, (float)yBaseImage, (float)zBaseImage };

            bestDisplacement[0] += referencePosition_temp[0];
            bestDisplacement[1] += referencePosition_temp[1];
            bestDisplacement[2] += referencePosition_temp[2];

            reg_mat44_mul_cl(referenceMatrix_xyz, referencePosition_temp, referencePosition);
            reg_mat44_mul_cl(referenceMatrix_xyz, bestDisplacement, warpedPosition);
            if (isfinite(bestDisplacement[0])) {
                atomic_add(definedBlock, 1);
            }
        }
    }
}
/* *************************************************************** */
/* *************************************************************** */
