/*
 *  _reg_mutualinformation_kernels.cu
 *
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_MUTUALINFORMATION_kernels_CU
#define _REG_MUTUALINFORMATION_kernels_CU

__device__ __constant__ int c_VoxelNumber;
__device__ __constant__ int3 c_ImageSize;
__device__ __constant__ int c_Binning;
__device__ __constant__ float4 c_Entropies;
__device__ __constant__ float c_NMI;
__device__ __constant__ int c_ActiveVoxelNumber;

texture<float, 1, cudaReadModeElementType> targetImageTexture;
texture<float, 1, cudaReadModeElementType> resultImageTexture;
texture<float4, 1, cudaReadModeElementType> resultImageGradientTexture;
texture<float, 1, cudaReadModeElementType> histogramTexture;
texture<float4, 1, cudaReadModeElementType> gradientImageTexture;
texture<int, 1, cudaReadModeElementType> maskTexture;

__device__ float GetBasisSplineValue(float x)
{
	x=fabsf(x);
	float value=0.0f;
	if(x<2.0f)
		if(x<1.0f)
			value = 2.0f/3.0f + (0.5f*x-1.0f)*x*x;
		else{
			x-=2.0f;
			value = -x*x*x/6.0f;
	}
	return value;
}
__device__ float GetBasisSplineDerivativeValue(float ori)
{
	float x=fabsf(ori);
	float value=0.0f;
	if(x<2.0f)
		if(x<1.0f)
			value = (1.5f*x-2.0f)*ori;
		else{
			x-=2.0f;
			value = -0.5f * x * x;
			if(ori<0.0f)value =-value;
	}
	return value;
}

__global__ void reg_getVoxelBasedNMIGradientUsingPW_kernel(float4 *voxelNMIGradientArray_d)
{
	const int tid=blockIdx.x*blockDim.x+threadIdx.x;
	if(tid<c_ActiveVoxelNumber){

        const int targetIndex = tex1Dfetch(maskTexture,tid);
		float targetImageValue = tex1Dfetch(targetImageTexture,targetIndex);
		float resultImageValue = tex1Dfetch(resultImageTexture,targetIndex);
		float4 resultImageGradient = tex1Dfetch(resultImageGradientTexture,tid);
		
		float4 gradValue = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

		// No computation is performed if any of the point is part of the background
        // The two is added because the image is resample between 2 and bin +2
        // if 64 bins are used the histogram will have 68 bins et the image will be between 2 and 65
        if( targetImageValue>0.0f &&
            resultImageValue>0.0f &&
            targetImageValue<c_Binning &&
            resultImageValue<c_Binning &&
            targetImageValue==targetImageValue &&
            resultImageValue==resultImageValue){

            targetImageValue = floor(targetImageValue); // Parzen window filling of the joint histogram is approximated
            resultImageValue = floor(resultImageValue);

			float3 resDeriv = make_float3(
				resultImageGradient.x,
				resultImageGradient.y,
				resultImageGradient.z);
				
			if( resultImageGradient.x==resultImageGradient.x &&
				resultImageGradient.y==resultImageGradient.y &&
				resultImageGradient.z==resultImageGradient.z){
					
				float jointEntropyDerivative_X = 0.0f;
				float movingEntropyDerivative_X = 0.0f;
				float fixedEntropyDerivative_X = 0.0f;
						
				float jointEntropyDerivative_Y = 0.0f;
				float movingEntropyDerivative_Y = 0.0f;
				float fixedEntropyDerivative_Y = 0.0f;
						
				float jointEntropyDerivative_Z = 0.0f;
				float movingEntropyDerivative_Z = 0.0f;
				float fixedEntropyDerivative_Z = 0.0f;
						
				for(int t=(int)(targetImageValue-1.0f); t<(int)(targetImageValue+2.0f); t++){
					if(-1<t && t<c_Binning){
						for(int r=(int)(resultImageValue-1.0f); r<(int)(resultImageValue+2.0f); r++){
							if(-1<r && r<c_Binning){
								float commonValue = GetBasisSplineValue((float)t-targetImageValue) *
									GetBasisSplineDerivativeValue((float)r-resultImageValue);
		
								float jointLog = tex1Dfetch(histogramTexture, t*c_Binning+r);
								float targetLog = tex1Dfetch(histogramTexture, c_Binning*c_Binning+t);
								float resultLog = tex1Dfetch(histogramTexture, c_Binning*c_Binning+c_Binning+r);
		
								float temp = commonValue * resDeriv.x;
								jointEntropyDerivative_X -= temp * jointLog;
								fixedEntropyDerivative_X -= temp * targetLog;
								movingEntropyDerivative_X -= temp * resultLog;
		
								temp = commonValue * resDeriv.y;
								jointEntropyDerivative_Y -= temp * jointLog;
								fixedEntropyDerivative_Y -= temp * targetLog;
								movingEntropyDerivative_Y -= temp * resultLog;
		
								temp = commonValue * resDeriv.z;
								jointEntropyDerivative_Z -= temp * jointLog;
								fixedEntropyDerivative_Z -= temp * targetLog;
								movingEntropyDerivative_Z -= temp * resultLog;
							} // O<t<bin
						} // t
					} // 0<r<bin
				} // r
		
				float NMI= c_NMI;
		           float temp = c_Entropies.z;
		           // (Marc) I removed the normalisation by the voxel number as each gradient has to be normalised in the same way
				gradValue.x = (fixedEntropyDerivative_X + movingEntropyDerivative_X - NMI * jointEntropyDerivative_X) / temp;
				gradValue.y = (fixedEntropyDerivative_Y + movingEntropyDerivative_Y - NMI * jointEntropyDerivative_Y) / temp;
				gradValue.z = (fixedEntropyDerivative_Z + movingEntropyDerivative_Z - NMI * jointEntropyDerivative_Z) / temp;
		
			}
		}
		voxelNMIGradientArray_d[targetIndex]=gradValue;

	}
	return;
}

#endif
