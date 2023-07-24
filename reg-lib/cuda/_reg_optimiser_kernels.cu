/* *************************************************************** */
__global__ void reg_initialiseConjugateGradient_kernel(float4 *conjugateGCuda,
                                                       cudaTextureObject_t gradientImageTexture,
                                                       const unsigned nVoxels) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < nVoxels) {
        const float4 gradValue = tex1Dfetch<float4>(gradientImageTexture, tid);
        conjugateGCuda[tid] = make_float4(-gradValue.x, -gradValue.y, -gradValue.z, 0);
    }
}
/* *************************************************************** */
__global__ void reg_getConjugateGradient1_kernel(float2 *sums,
                                                 cudaTextureObject_t gradientImageTexture,
                                                 cudaTextureObject_t conjugateGTexture,
                                                 cudaTextureObject_t conjugateHTexture,
                                                 const unsigned nVoxels) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < nVoxels) {
        const float4 valueH = tex1Dfetch<float4>(conjugateHTexture, tid);
        const float4 valueG = tex1Dfetch<float4>(conjugateGTexture, tid);
        const float gg = valueG.x * valueH.x + valueG.y * valueH.y + valueG.z * valueH.z;

        const float4 grad = tex1Dfetch<float4>(gradientImageTexture, tid);
        const float dgg = (grad.x + valueG.x) * grad.x + (grad.y + valueG.y) * grad.y + (grad.z + valueG.z) * grad.z;

        sums[tid] = make_float2(dgg, gg);
    }
}
/* *************************************************************** */
__global__ void reg_getConjugateGradient2_kernel(float4 *gradientImageCuda,
                                                 float4 *conjugateGCuda,
                                                 float4 *conjugateHCuda,
                                                 const unsigned nVoxels,
                                                 const float scale) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < nVoxels) {
        // G = - grad
        float4 gradGValue = gradientImageCuda[tid];
        gradGValue = make_float4(-gradGValue.x, -gradGValue.y, -gradGValue.z, 0);
        conjugateGCuda[tid] = gradGValue;

        // H = G + gam * H
        float4 gradHValue = conjugateHCuda[tid];
        gradHValue = make_float4(gradGValue.x + scale * gradHValue.x,
                                 gradGValue.y + scale * gradHValue.y,
                                 gradGValue.z + scale * gradHValue.z,
                                 0);
        conjugateHCuda[tid] = gradHValue;

        gradientImageCuda[tid] = make_float4(-gradHValue.x, -gradHValue.y, -gradHValue.z, 0);
    }
}
/* *************************************************************** */
__global__ void reg_updateControlPointPosition_kernel(float4 *controlPointImageCuda,
                                                      cudaTextureObject_t bestControlPointTexture,
                                                      cudaTextureObject_t gradientImageTexture,
                                                      const unsigned nVoxels,
                                                      const float scale,
                                                      const bool optimiseX,
                                                      const bool optimiseY,
                                                      const bool optimiseZ) {
    const unsigned tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < nVoxels) {
        float4 value = controlPointImageCuda[tid];
        const float4 bestValue = tex1Dfetch<float4>(bestControlPointTexture, tid);
        const float4 gradValue = tex1Dfetch<float4>(gradientImageTexture, tid);
        if (optimiseX)
            value.x = bestValue.x + scale * gradValue.x;
        if (optimiseY)
            value.y = bestValue.y + scale * gradValue.y;
        if (optimiseZ)
            value.z = bestValue.z + scale * gradValue.z;
        value.w = 0;
        controlPointImageCuda[tid] = value;
    }
}
/* *************************************************************** */
