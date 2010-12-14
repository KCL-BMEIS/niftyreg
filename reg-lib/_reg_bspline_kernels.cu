/*
 *  _reg_bspline_kernels.cu
 *
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_BSPLINE_KERNELS_CU
#define _REG_BSPLINE_KERNELS_CU

#include "_reg_blocksize_gpu.h"

__device__ __constant__ int c_VoxelNumber;
__device__ __constant__ int c_ControlPointNumber;
__device__ __constant__ int3 c_TargetImageDim;
__device__ __constant__ int3 c_ControlPointImageDim;
__device__ __constant__ float3 c_ControlPointVoxelSpacing;
__device__ __constant__ float3 c_ControlPointSpacing;
__device__ __constant__ float c_Weight;
__device__ __constant__ int c_ActiveVoxelNumber;
__device__ __constant__ bool c_Type;
__device__ __constant__ float3 c_AffineMatrix0;
__device__ __constant__ float3 c_AffineMatrix1;
__device__ __constant__ float3 c_AffineMatrix2;

/* *************************************************************** */
/* *************************************************************** */

texture<float4, 1, cudaReadModeElementType> controlPointTexture;
texture<float4, 1, cudaReadModeElementType> basisValueATexture;
texture<float2, 1, cudaReadModeElementType> basisValueBTexture;
texture<int, 1, cudaReadModeElementType> maskTexture;
texture<float4, 1, cudaReadModeElementType> txVoxelToRealMatrix;
texture<float4, 1, cudaReadModeElementType> txRealToVoxelMatrix;

texture<float,1, cudaReadModeElementType> xBasisTexture;
texture<float,1, cudaReadModeElementType> yBasisTexture;
texture<float,1, cudaReadModeElementType> zBasisTexture;

texture<float,1, cudaReadModeElementType> jacobianDeterminantTexture;
texture<float,1, cudaReadModeElementType> jacobianMatricesTexture;
texture<float4,1, cudaReadModeElementType> voxelDisplacementTexture;

/* *************************************************************** */
/* *************************************************************** */

__device__ float3 operator*(float a, float3 b){
    return make_float3(a*b.x, a*b.y, a*b.z);
}
__device__ float4 operator*(float a, float4 b){
    return make_float4(a*b.x, a*b.y, a*b.z, 0.0f);
}
__device__ float3 operator/(float3 b, float a){
    return make_float3(b.x/a, b.y/a, b.z/a);
}
__device__ float4 operator+(float4 a, float4 b){
    return make_float4(a.x+b.x, a.y+b.y, a.z+b.z, 0.0f);
}
__device__ float3 operator+(float3 a, float3 b){
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}
__device__ float3 operator-(float3 a, float3 b){
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}
__device__ float3 operator*(float3 a, float3 b){
    return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);
}

/* *************************************************************** */
/* *************************************************************** */

__device__ void bendingEnergyMult(	float3 *XX,
                    float3 *YY,
                    float3 *ZZ,
                    float3 *XY,
                    float3 *YZ,
                    float3 *XZ,
                    float basisXX,
                    float basisYY,
                    float basisZZ,
                    float basisXY,
                    float basisYZ,
                    float basisXZ,
                    int index)
{
    float4 position = tex1Dfetch(controlPointTexture,index);
    (*XX).x += basisXX * position.x;
    (*XX).y += basisXX * position.y;
    (*XX).z += basisXX * position.z;

    (*YY).x += basisYY * position.x;
    (*YY).y += basisYY * position.y;
    (*YY).z += basisYY * position.z;

    (*ZZ).x += basisZZ * position.x;
    (*ZZ).y += basisZZ * position.y;
    (*ZZ).z += basisZZ * position.z;

    (*XY).x += basisXY * position.x;
    (*XY).y += basisXY * position.y;
    (*XY).z += basisXY * position.z;

    (*YZ).x += basisYZ * position.x;
    (*YZ).y += basisYZ * position.y;
    (*YZ).z += basisYZ * position.z;

    (*XZ).x += basisXZ * position.x;
    (*XZ).y += basisXZ * position.y;
    (*XZ).z += basisXZ * position.z;

    return;
}

/* *************************************************************** */
/* *************************************************************** */

__global__ void reg_freeForm_deformationField_kernel(float4 *positionField)
{
    const unsigned int tid= blockIdx.x*blockDim.x + threadIdx.x;
	if(tid<c_ActiveVoxelNumber){

		int3 imageSize = c_TargetImageDim;

		unsigned int tempIndex=tex1Dfetch(maskTexture,tid);
		const unsigned short z =(unsigned short)(tempIndex/(imageSize.x*imageSize.y));
		tempIndex -= z*(imageSize.x)*(imageSize.y);
		const unsigned short y =(unsigned short)(tempIndex/(imageSize.x));
		const unsigned short x = tempIndex - y*(imageSize.x);
	
		// the "nearest previous" node is determined [0,0,0]
		short3 nodeAnte;
		float3 gridVoxelSpacing = c_ControlPointVoxelSpacing;
		nodeAnte.x = (short)floorf((float)x/gridVoxelSpacing.x);
		nodeAnte.y = (short)floorf((float)y/gridVoxelSpacing.y);
		nodeAnte.z = (short)floorf((float)z/gridVoxelSpacing.z);

        // Z basis values
        const unsigned short shareMemIndex = 4*threadIdx.x;
        __shared__ float zBasis[Block_reg_freeForm_deformationField*4];
        float relative = fabsf((float)z/gridVoxelSpacing.z-(float)nodeAnte.z);
        float FF= relative*relative;
        float FFF= FF*relative;
        float MF=1.0f-relative;
        zBasis[shareMemIndex] = MF*MF*MF/6.0f;
        zBasis[shareMemIndex+1] = (3.0f*FFF - 6.0f*FF +4.0f)/6.0f;
        zBasis[shareMemIndex+2] = (-3.0f*FFF + 3.0f*FF + 3.0f*relative + 1.0f)/6.0f;
        zBasis[shareMemIndex+3] = FFF/6.0f;

        // Y basis values
        __shared__ float yBasis[Block_reg_freeForm_deformationField*4];
        relative = fabsf((float)y/gridVoxelSpacing.y-(float)nodeAnte.y);
        FF= relative*relative;
        FFF= FF*relative;
        MF=1.0f-relative;
        yBasis[shareMemIndex] = MF*MF*MF/6.0f;
        yBasis[shareMemIndex+1] = (3.0f*FFF - 6.0f*FF +4.0f)/6.0f;
        yBasis[shareMemIndex+2] = (-3.0f*FFF + 3.0f*FF + 3.0f*relative + 1.0f)/6.0f;
        yBasis[shareMemIndex+3] = FFF/6.0f;

        // X basis values
        relative = fabsf((float)x/gridVoxelSpacing.x-(float)nodeAnte.x);
        float4 xBasis;
        xBasis.w= relative * relative * relative / 6.0f;
        xBasis.x= 1.0f/6.0f + relative*(relative-1.0f)/2.0f - xBasis.w;
        xBasis.z= relative + xBasis.x - 2.0f*xBasis.w;
        xBasis.y= 1.0f - xBasis.x - xBasis.z - xBasis.w;

        int3 controlPointImageDim = c_ControlPointImageDim;
        float4 displacement=make_float4(0.0f,0.0f,0.0f,0.0f);
        float basis;

        float3 tempDisplacement;

        int indexYZ, indexXYZ;
        for(short c=0; c<4; c++){
            tempDisplacement=make_float3(0.0f,0.0f,0.0f);
            indexYZ= ( (nodeAnte.z + c) * controlPointImageDim.y + nodeAnte.y) * controlPointImageDim.x;
            for(short b=0; b<4; b++){

                indexXYZ = indexYZ + nodeAnte.x;
                float4 nodeCoefficientA = tex1Dfetch(controlPointTexture,indexXYZ++);
                float4 nodeCoefficientB = tex1Dfetch(controlPointTexture,indexXYZ++);
                float4 nodeCoefficientC = tex1Dfetch(controlPointTexture,indexXYZ++);
                float4 nodeCoefficientD = tex1Dfetch(controlPointTexture,indexXYZ);

                basis=yBasis[shareMemIndex+b];
                tempDisplacement.x += (nodeCoefficientA.x * xBasis.x
                    + nodeCoefficientB.x * xBasis.y
                    + nodeCoefficientC.x * xBasis.z
                    + nodeCoefficientD.x * xBasis.w) * basis;

                tempDisplacement.y += (nodeCoefficientA.y * xBasis.x
                    + nodeCoefficientB.y * xBasis.y
                    + nodeCoefficientC.y * xBasis.z
                    + nodeCoefficientD.y * xBasis.w) * basis;

                tempDisplacement.z += (nodeCoefficientA.z * xBasis.x
                    + nodeCoefficientB.z * xBasis.y
                    + nodeCoefficientC.z * xBasis.z
                    + nodeCoefficientD.z * xBasis.w) * basis;

                indexYZ += controlPointImageDim.x;
            }

            basis =zBasis[shareMemIndex+c];
            displacement.x += tempDisplacement.x * basis;
            displacement.y += tempDisplacement.y * basis;
            displacement.z += tempDisplacement.z * basis;
        }
        positionField[tid] = displacement;
    }
    return;
}

/* *************************************************************** */
/* *************************************************************** */

__global__ void reg_bspline_ApproxBendingEnergy_kernel(float *penaltyTerm)
{
	const int tid= blockIdx.x*blockDim.x + threadIdx.x;
	if(tid<c_ControlPointNumber){

		int3 gridSize = c_ControlPointImageDim;

		int tempIndex=tid;
		const short z =(int)(tempIndex/(gridSize.x*gridSize.y));
		tempIndex -= z*(gridSize.x)*(gridSize.y);
		const short y =(int)(tempIndex/(gridSize.x));
		const short x = tempIndex - y*(gridSize.x);

		if(	0<x && x<gridSize.x-1 &&
			0<y && y<gridSize.y-1 &&
			0<z && z<gridSize.z-1){

			float3 XX = make_float3(0.0f,0.0f,0.0f);
			float3 YY = make_float3(0.0f,0.0f,0.0f);
			float3 ZZ = make_float3(0.0f,0.0f,0.0f);
			float3 XY = make_float3(0.0f,0.0f,0.0f);
			float3 YZ = make_float3(0.0f,0.0f,0.0f);
			float3 XZ = make_float3(0.0f,0.0f,0.0f);


			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,0.027778f,0.027778f,0.027778f,0.041667f,0.041667f,0.041667f,((z-1)*gridSize.y+y-1)*gridSize.x+x-1);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,-0.055556f,0.111111f,0.111111f,-0.000000f,0.166667f,-0.000000f,((z-1)*gridSize.y+y-1)*gridSize.x+x);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,0.027778f,0.027778f,0.027778f,-0.041667f,0.041667f,-0.041667f,((z-1)*gridSize.y+y-1)*gridSize.x+x+1);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,0.111111f,-0.055556f,0.111111f,-0.000000f,-0.000000f,0.166667f,((z-1)*gridSize.y+y)*gridSize.x+x-1);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,-0.222222f,-0.222222f,0.444444f,0.000000f,-0.000000f,-0.000000f,((z-1)*gridSize.y+y)*gridSize.x+x);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,0.111111f,-0.055556f,0.111111f,0.000000f,-0.000000f,-0.166667f,((z-1)*gridSize.y+y)*gridSize.x+x+1);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,0.027778f,0.027778f,0.027778f,-0.041667f,-0.041667f,0.041667f,((z-1)*gridSize.y+y+1)*gridSize.x+x-1);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,-0.055556f,0.111111f,0.111111f,0.000000f,-0.166667f,-0.000000f,((z-1)*gridSize.y+y+1)*gridSize.x+x);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,0.027778f,0.027778f,0.027778f,0.041667f,-0.041667f,-0.041667f,((z-1)*gridSize.y+y+1)*gridSize.x+x+1);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,0.111111f,0.111111f,-0.055556f,0.166667f,-0.000000f,-0.000000f,((z)*gridSize.y+y-1)*gridSize.x+x-1);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,-0.222222f,0.444444f,-0.222222f,-0.000000f,-0.000000f,0.000000f,((z)*gridSize.y+y-1)*gridSize.x+x);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,0.111111f,0.111111f,-0.055556f,-0.166667f,-0.000000f,0.000000f,((z)*gridSize.y+y-1)*gridSize.x+x+1);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,0.444444f,-0.222222f,-0.222222f,-0.000000f,0.000000f,-0.000000f,((z)*gridSize.y+y)*gridSize.x+x-1);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,-0.888889f,-0.888889f,-0.888889f,0.000000f,0.000000f,0.000000f,((z)*gridSize.y+y)*gridSize.x+x);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,0.444444f,-0.222222f,-0.222222f,0.000000f,0.000000f,0.000000f,((z)*gridSize.y+y)*gridSize.x+x+1);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,0.111111f,0.111111f,-0.055556f,-0.166667f,0.000000f,-0.000000f,((z)*gridSize.y+y+1)*gridSize.x+x-1);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,-0.222222f,0.444444f,-0.222222f,0.000000f,0.000000f,0.000000f,((z)*gridSize.y+y+1)*gridSize.x+x);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,0.111111f,0.111111f,-0.055556f,0.166667f,0.000000f,0.000000f,((z)*gridSize.y+y+1)*gridSize.x+x+1);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,0.027778f,0.027778f,0.027778f,0.041667f,-0.041667f,-0.041667f,((z+1)*gridSize.y+y-1)*gridSize.x+x-1);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,-0.055556f,0.111111f,0.111111f,-0.000000f,-0.166667f,0.000000f,((z+1)*gridSize.y+y-1)*gridSize.x+x);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,0.027778f,0.027778f,0.027778f,-0.041667f,-0.041667f,0.041667f,((z+1)*gridSize.y+y-1)*gridSize.x+x+1);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,0.111111f,-0.055556f,0.111111f,-0.000000f,0.000000f,-0.166667f,((z+1)*gridSize.y+y)*gridSize.x+x-1);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,-0.222222f,-0.222222f,0.444444f,0.000000f,0.000000f,0.000000f,((z+1)*gridSize.y+y)*gridSize.x+x);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,0.111111f,-0.055556f,0.111111f,0.000000f,0.000000f,0.166667f,((z+1)*gridSize.y+y)*gridSize.x+x+1);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,0.027778f,0.027778f,0.027778f,-0.041667f,0.041667f,-0.041667f,((z+1)*gridSize.y+y+1)*gridSize.x+x-1);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,-0.055556f,0.111111f,0.111111f,0.000000f,0.166667f,0.000000f,((z+1)*gridSize.y+y+1)*gridSize.x+x);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,0.027778f,0.027778f,0.027778f,0.041667f,0.041667f,0.041667f,((z+1)*gridSize.y+y+1)*gridSize.x+x+1);

			float penalty = XX.x*XX.x + YY.x*YY.x + ZZ.x*ZZ.x + 2.0f*(XY.x*XY.x + YZ.x*YZ.x + XZ.x*XZ.x);
			penalty += XX.y*XX.y + YY.y*YY.y + ZZ.y*ZZ.y + 2.0f*(XY.y*XY.y + YZ.y*YZ.y + XZ.y*XZ.y);
			penalty += XX.z*XX.z + YY.z*YY.z + ZZ.z*ZZ.z + 2.0f*(XY.z*XY.z + YZ.z*YZ.z + XZ.z*XZ.z);
			penaltyTerm[tid]=penalty;
		}
		else penaltyTerm[tid]=0.0f;
	}
	return;
}

/* *************************************************************** */
/* *************************************************************** */

__global__ void reg_bspline_JacDet_kernel(float *jacobianMap)
{
    const unsigned int tid= blockIdx.x*blockDim.x + threadIdx.x;
    if(tid<c_VoxelNumber){

        int3 imageSize = c_TargetImageDim;

        unsigned int tempIndex=tid;
        const unsigned short z =(unsigned short)(tempIndex/(imageSize.x*imageSize.y));
        tempIndex -= z*(imageSize.x)*(imageSize.y);
        const unsigned short y =(unsigned short)(tempIndex/(imageSize.x));
        const unsigned short x = tempIndex - y*(imageSize.x);

        // the "nearest previous" node is determined [0,0,0]
        short3 nodeAnte;
        float3 gridVoxelSpacing = c_ControlPointVoxelSpacing;
        nodeAnte.x = (short)floorf((float)x/gridVoxelSpacing.x);
        nodeAnte.y = (short)floorf((float)y/gridVoxelSpacing.y);
        nodeAnte.z = (short)floorf((float)z/gridVoxelSpacing.z);

        // Z basis values
        const unsigned short shareMemIndex = 4*threadIdx.x;
        __shared__ float zBasis[Block_reg_bspline_JacDet*4];
        __shared__ float zFirst[Block_reg_bspline_JacDet*4];
        float relative = fabsf((float)z/gridVoxelSpacing.z-(float)nodeAnte.z);
        float FF= relative*relative;
        float FFF= FF*relative;
        float MF=1.0f-relative;
        zBasis[shareMemIndex] = MF*MF*MF/6.0f;
        zBasis[shareMemIndex+1] = (3.0f*FFF - 6.0f*FF +4.0f)/6.0f;
        zBasis[shareMemIndex+2] = (-3.0f*FFF + 3.0f*FF + 3.0f*relative + 1.0f)/6.0f;
        zBasis[shareMemIndex+3] = FFF/6.0f;
        zFirst[shareMemIndex] = (2.0f*relative - FF - 1.0f)/2.0f;
        zFirst[shareMemIndex+1] = (3.0f*FF - 4.0f*relative)/2.0f;
        zFirst[shareMemIndex+2] = (2.0f*relative - 3.0f*FF + 1.0f)/2.0f;
        zFirst[shareMemIndex+3] = FF/2.0f;

        // Y basis values
        __shared__ float yBasis[Block_reg_bspline_JacDet*4];
        __shared__ float yFirst[Block_reg_bspline_JacDet*4];
        relative = fabsf((float)y/gridVoxelSpacing.y-(float)nodeAnte.y);
        FF= relative*relative;
        FFF= FF*relative;
        MF=1.0f-relative;
        yBasis[shareMemIndex] = MF*MF*MF/6.0f;
        yBasis[shareMemIndex+1] = (3.0f*FFF - 6.0f*FF +4.0f)/6.0f;
        yBasis[shareMemIndex+2] = (-3.0f*FFF + 3.0f*FF + 3.0f*relative + 1.0f)/6.0f;
        yBasis[shareMemIndex+3] = FFF/6.0f;
        yFirst[shareMemIndex] = (2.0f*relative - FF - 1.0f)/2.0f;
        yFirst[shareMemIndex+1] = (3.0f*FF - 4.0f*relative)/2.0f;
        yFirst[shareMemIndex+2] = (2.0f*relative - 3.0f*FF + 1.0f)/2.0f;
        yFirst[shareMemIndex+3] = FF/2.0f;

        // X basis values
        relative = fabsf((float)x/gridVoxelSpacing.x-(float)nodeAnte.x);
        float4 xBasis;
        float4 xFirst;
        xBasis.w= relative * relative * relative / 6.0f;
        xBasis.x= 1.0f/6.0f + relative*(relative-1.0f)/2.0f - xBasis.w;
        xBasis.z= relative + xBasis.x - 2.0f*xBasis.w;
        xBasis.y= 1.0f - xBasis.x - xBasis.z - xBasis.w;
        xFirst.w= relative * relative / 2.0f;
        xFirst.x= relative - 0.5f - xFirst.w;
        xFirst.z= 1.0f + xFirst.x - 2.0f*xFirst.w;
        xFirst.y= - xFirst.x - xFirst.z - xFirst.w;

        int3 controlPointImageDim = c_ControlPointImageDim;

        float Tx_x=0.0f;
        float Ty_x=0.0f;
        float Tz_x=0.0f;
        float Tx_y=0.0f;
        float Ty_y=0.0f;
        float Tz_y=0.0f;
        float Tx_z=0.0f;
        float Ty_z=0.0f;
        float Tz_z=0.0f;

        float4 nodeCoefficient;
        float3 tempBasis;
        float basis;

        int indexYZ, indexXYZ;
        for(short c=0; c<4; c++){
            indexYZ= ( (nodeAnte.z + c) * controlPointImageDim.y + nodeAnte.y) * controlPointImageDim.x;
            for(short b=0; b<4; b++){

                tempBasis.x = zBasis[shareMemIndex+c] * yBasis[shareMemIndex+b];
                tempBasis.y = zBasis[shareMemIndex+c] * yFirst[shareMemIndex+b];
                tempBasis.z = zFirst[shareMemIndex+c] * yBasis[shareMemIndex+b];

                indexXYZ = indexYZ + nodeAnte.x;
                nodeCoefficient = tex1Dfetch(controlPointTexture,indexXYZ++);
                basis = xFirst.x * tempBasis.x;
                Tx_x+=nodeCoefficient.x * basis;
                Ty_x+=nodeCoefficient.y * basis;
                Tz_x+=nodeCoefficient.z * basis;
                basis = xBasis.x * tempBasis.y;
                Tx_y+=nodeCoefficient.x * basis;
                Ty_y+=nodeCoefficient.y * basis;
                Tz_y+=nodeCoefficient.z * basis;
                basis = xBasis.x * tempBasis.z;
                Tx_z+=nodeCoefficient.x * basis;
                Ty_z+=nodeCoefficient.y * basis;
                Tz_z+=nodeCoefficient.z * basis;
                nodeCoefficient = tex1Dfetch(controlPointTexture,indexXYZ++);
                basis = xFirst.y * tempBasis.x;
                Tx_x+=nodeCoefficient.x * basis;
                Ty_x+=nodeCoefficient.y * basis;
                Tz_x+=nodeCoefficient.z * basis;
                basis = xBasis.y * tempBasis.y;
                Tx_y+=nodeCoefficient.x * basis;
                Ty_y+=nodeCoefficient.y * basis;
                Tz_y+=nodeCoefficient.z * basis;
                basis = xBasis.y * tempBasis.z;
                Tx_z+=nodeCoefficient.x * basis;
                Ty_z+=nodeCoefficient.y * basis;
                Tz_z+=nodeCoefficient.z * basis;
                nodeCoefficient = tex1Dfetch(controlPointTexture,indexXYZ++);
                basis = xFirst.z * tempBasis.x;
                Tx_x+=nodeCoefficient.x * basis;
                Ty_x+=nodeCoefficient.y * basis;
                Tz_x+=nodeCoefficient.z * basis;
                basis = xBasis.z * tempBasis.y;
                Tx_y+=nodeCoefficient.x * basis;
                Ty_y+=nodeCoefficient.y * basis;
                Tz_y+=nodeCoefficient.z * basis;
                basis = xBasis.z * tempBasis.z;
                Tx_z+=nodeCoefficient.x * basis;
                Ty_z+=nodeCoefficient.y * basis;
                Tz_z+=nodeCoefficient.z * basis;
                nodeCoefficient = tex1Dfetch(controlPointTexture,indexXYZ);
                basis = xFirst.w * tempBasis.x;
                Tx_x+=nodeCoefficient.x * basis;
                Ty_x+=nodeCoefficient.y * basis;
                Tz_x+=nodeCoefficient.z * basis;
                basis = xBasis.w * tempBasis.y;
                Tx_y+=nodeCoefficient.x * basis;
                Ty_y+=nodeCoefficient.y * basis;
                Tz_y+=nodeCoefficient.z * basis;
                basis = xBasis.w * tempBasis.z;
                Tx_z+=nodeCoefficient.x * basis;
                Ty_z+=nodeCoefficient.y * basis;
                Tz_z+=nodeCoefficient.z * basis;

                indexYZ += controlPointImageDim.x;
            }
        }

        Tx_x /= c_ControlPointSpacing.x;
        Ty_x /= c_ControlPointSpacing.x;
        Tz_x /= c_ControlPointSpacing.x;
        Tx_y /= c_ControlPointSpacing.y;
        Ty_y /= c_ControlPointSpacing.y;
        Tz_y /= c_ControlPointSpacing.y;
        Tx_z /= c_ControlPointSpacing.z;
        Ty_z /= c_ControlPointSpacing.z;
        Tz_z /= c_ControlPointSpacing.z;

        float Tx_x2=c_AffineMatrix0.x*Tx_x + c_AffineMatrix0.y*Ty_x + c_AffineMatrix0.z*Tz_x;
        float Ty_x2=c_AffineMatrix0.x*Tx_y + c_AffineMatrix0.y*Ty_y + c_AffineMatrix0.z*Tz_y;
        float Tz_x2=c_AffineMatrix0.x*Tx_z + c_AffineMatrix0.y*Ty_z + c_AffineMatrix0.z*Tz_z;

        float Tx_y2=c_AffineMatrix1.x*Tx_x + c_AffineMatrix1.y*Ty_x + c_AffineMatrix1.z*Tz_x;
        float Ty_y2=c_AffineMatrix1.x*Tx_y + c_AffineMatrix1.y*Ty_y + c_AffineMatrix1.z*Tz_y;
        float Tz_y2=c_AffineMatrix1.x*Tx_z + c_AffineMatrix1.y*Ty_z + c_AffineMatrix1.z*Tz_z;

        float Tx_z2=c_AffineMatrix2.x*Tx_x + c_AffineMatrix2.y*Ty_x + c_AffineMatrix2.z*Tz_x;
        float Ty_z2=c_AffineMatrix2.x*Tx_y + c_AffineMatrix2.y*Ty_y + c_AffineMatrix2.z*Tz_y;
        float Tz_z2=c_AffineMatrix2.x*Tx_z + c_AffineMatrix2.y*Ty_z + c_AffineMatrix2.z*Tz_z;


        /* The Jacobian determinant is computed and stored */
        jacobianMap[tid]= Tx_x2*Ty_y2*Tz_z2
                        + Tx_y2*Ty_z2*Tz_x2
                        + Tx_z2*Ty_x2*Tz_y2
                        - Tx_x2*Ty_z2*Tz_y2
                        - Tx_y2*Ty_x2*Tz_z2
                        - Tx_z2*Ty_y2*Tz_x2;
    }
    return;
}

/* *************************************************************** */
/* *************************************************************** */

__global__ void reg_bspline_ApproxJacDet_kernel(float *penaltyTerm)
{
    __shared__ float basisX[27];
    __shared__ float basisY[27];
    __shared__ float basisZ[27];
    if(threadIdx.x<27){
        basisX[threadIdx.x] = tex1Dfetch(xBasisTexture,threadIdx.x);
        basisY[threadIdx.x] = tex1Dfetch(yBasisTexture,threadIdx.x);
        basisZ[threadIdx.x] = tex1Dfetch(zBasisTexture,threadIdx.x);
    }
    __syncthreads();

    const int tid= blockIdx.x*blockDim.x + threadIdx.x;
    if(tid<c_ControlPointNumber){

        int3 gridSize = c_ControlPointImageDim;

        int tempIndex=tid;
        const int z =(int)(tempIndex/(gridSize.x*gridSize.y));
        tempIndex -= z*(gridSize.x)*(gridSize.y);
        const int y =(int)(tempIndex/(gridSize.x));
        const int x = tempIndex - y*(gridSize.x) ;

        if( 0<x && x<gridSize.x-1 &&
            0<y && y<gridSize.y-1 &&
            0<z && z<gridSize.z-1){

            /* The Jacobian matrix is computed */
            float Tx_x=0.0f;
            float Ty_x=0.0f;
            float Tz_x=0.0f;
            float Tx_y=0.0f;
            float Ty_y=0.0f;
            float Tz_y=0.0f;
            float Tx_z=0.0f;
            float Ty_z=0.0f;
            float Tz_z=0.0f;
            float4 controlPointPosition;
            float tempBasis;
            int index2=0;
            for(int c=z-1; c<z+2; c++){
                for(int b=y-1; b<y+2; b++){
                    int index = (c*gridSize.y+b)*gridSize.x+x-1;
                    for(int a=x-1; a<x+2; a++){
                        controlPointPosition = tex1Dfetch(controlPointTexture,index++);
                        tempBasis=basisX[index2];
                        Tx_x+=controlPointPosition.x * tempBasis;
                        Ty_x+=controlPointPosition.y * tempBasis;
                        Tz_x+=controlPointPosition.z * tempBasis;
                        tempBasis=basisY[index2];
                        Tx_y+=controlPointPosition.x * tempBasis;
                        Ty_y+=controlPointPosition.y * tempBasis;
                        Tz_y+=controlPointPosition.z * tempBasis;
                        tempBasis=basisZ[index2++];
                        Tx_z+=controlPointPosition.x * tempBasis;
                        Ty_z+=controlPointPosition.y * tempBasis;
                        Tz_z+=controlPointPosition.z * tempBasis;
                    }
                }
            }
            Tx_x /= c_ControlPointSpacing.x;
            Ty_x /= c_ControlPointSpacing.x;
            Tz_x /= c_ControlPointSpacing.x;
            Tx_y /= c_ControlPointSpacing.y;
            Ty_y /= c_ControlPointSpacing.y;
            Tz_y /= c_ControlPointSpacing.y;
            Tx_z /= c_ControlPointSpacing.z;
            Ty_z /= c_ControlPointSpacing.z;
            Tz_z /= c_ControlPointSpacing.z;

            /* The Jacobian matrix is reoriented */
            float Tx_x2=c_AffineMatrix0.x*Tx_x + c_AffineMatrix0.y*Ty_x + c_AffineMatrix0.z*Tz_x;
            float Ty_x2=c_AffineMatrix0.x*Tx_y + c_AffineMatrix0.y*Ty_y + c_AffineMatrix0.z*Tz_y;
            float Tz_x2=c_AffineMatrix0.x*Tx_z + c_AffineMatrix0.y*Ty_z + c_AffineMatrix0.z*Tz_z;

            float Tx_y2=c_AffineMatrix1.x*Tx_x + c_AffineMatrix1.y*Ty_x + c_AffineMatrix1.z*Tz_x;
            float Ty_y2=c_AffineMatrix1.x*Tx_y + c_AffineMatrix1.y*Ty_y + c_AffineMatrix1.z*Tz_y;
            float Tz_y2=c_AffineMatrix1.x*Tx_z + c_AffineMatrix1.y*Ty_z + c_AffineMatrix1.z*Tz_z;

            float Tx_z2=c_AffineMatrix2.x*Tx_x + c_AffineMatrix2.y*Ty_x + c_AffineMatrix2.z*Tz_x;
            float Ty_z2=c_AffineMatrix2.x*Tx_y + c_AffineMatrix2.y*Ty_y + c_AffineMatrix2.z*Tz_y;
            float Tz_z2=c_AffineMatrix2.x*Tx_z + c_AffineMatrix2.y*Ty_z + c_AffineMatrix2.z*Tz_z;

            /* The Jacobian determinant is computed and stored */
            penaltyTerm[tid]=Tx_x2*Ty_y2*Tz_z2 + Tx_y2*Ty_z2*Tz_x2 + Tx_z2*Ty_x2*Tz_y2
                - Tx_x2*Ty_z2*Tz_y2 - Tx_y2*Ty_x2*Tz_z2 - Tx_z2*Ty_y2*Tz_x2;
        }
        else penaltyTerm[tid]=1.0f;
    }
}

/* *************************************************************** */
/* *************************************************************** */

__global__ void reg_bspline_JacDetFromVelField_kernel( float *jacobianMap,
                                                        float4 *displacementField_d)
{
    const unsigned int tid= blockIdx.x*blockDim.x + threadIdx.x;
    if(tid<c_VoxelNumber){

        int3 targetImageDim = c_TargetImageDim;

        int tempIndex=tid;
        float z = floorf(tempIndex/(targetImageDim.x*targetImageDim.y));
        tempIndex -= (int)z*(targetImageDim.x)*(targetImageDim.y);
        float y = floorf(tempIndex/(targetImageDim.x));
        float x = tempIndex - y*(targetImageDim.x);

        // The displacement of the voxel is updated
        float4 position = displacementField_d[tid];
        float4 matrix = tex1Dfetch(txRealToVoxelMatrix,0);
        x +=    matrix.x*position.x + matrix.y*position.y  +
                matrix.z*position.z  +  matrix.w;
        matrix = tex1Dfetch(txRealToVoxelMatrix,1);
        y +=    matrix.x*position.x + matrix.y*position.y  +
                matrix.z*position.z  +  matrix.w;
        matrix = tex1Dfetch(txRealToVoxelMatrix,2);
        z +=    matrix.x*position.x + matrix.y*position.y  +
                matrix.z*position.z  +  matrix.w;

        // the "nearest previous" node is determined [0,0,0]
        short3 nodeAnte;
        float3 gridVoxelSpacing = c_ControlPointVoxelSpacing;
        nodeAnte.x = (short)floorf(x/gridVoxelSpacing.x);
        nodeAnte.y = (short)floorf(y/gridVoxelSpacing.y);
        nodeAnte.z = (short)floorf(z/gridVoxelSpacing.z);

        int3 controlPointImageDim = c_ControlPointImageDim;

        if(nodeAnte.x>=0 && nodeAnte.x<controlPointImageDim.x-3 &&
           nodeAnte.y>=0 && nodeAnte.y<controlPointImageDim.y-3 &&
           nodeAnte.z>=0 && nodeAnte.z<controlPointImageDim.z-3){

            // Z basis values
            const unsigned short shareMemIndex = 4*threadIdx.x;
            __shared__ float zBasis[Block_reg_bspline_JacDetFromVelField*4];
            __shared__ float zFirst[Block_reg_bspline_JacDetFromVelField*4];
            float relative = fabsf(z/gridVoxelSpacing.z-(float)nodeAnte.z);
            float FF= relative*relative;
            float FFF= FF*relative;
            float MF=1.0f-relative;
            zBasis[shareMemIndex] = MF*MF*MF/6.0f;
            zBasis[shareMemIndex+1] = (3.0f*FFF - 6.0f*FF +4.0f)/6.0f;
            zBasis[shareMemIndex+2] = (-3.0f*FFF + 3.0f*FF + 3.0f*relative + 1.0f)/6.0f;
            zBasis[shareMemIndex+3] = FFF/6.0f;
            zFirst[shareMemIndex] = (2.0f*relative - FF - 1.0f)/2.0f;
            zFirst[shareMemIndex+1] = (3.0f*FF - 4.0f*relative)/2.0f;
            zFirst[shareMemIndex+2] = (2.0f*relative - 3.0f*FF + 1.0f)/2.0f;
            zFirst[shareMemIndex+3] = FF/2.0f;

            // Y basis values
            __shared__ float yBasis[Block_reg_bspline_JacDetFromVelField*4];
            __shared__ float yFirst[Block_reg_bspline_JacDetFromVelField*4];
            relative = fabsf(y/gridVoxelSpacing.y-(float)nodeAnte.y);
            FF= relative*relative;
            FFF= FF*relative;
            MF=1.0f-relative;
            yBasis[shareMemIndex] = MF*MF*MF/6.0f;
            yBasis[shareMemIndex+1] = (3.0f*FFF - 6.0f*FF +4.0f)/6.0f;
            yBasis[shareMemIndex+2] = (-3.0f*FFF + 3.0f*FF + 3.0f*relative + 1.0f)/6.0f;
            yBasis[shareMemIndex+3] = FFF/6.0f;
            yFirst[shareMemIndex] = (2.0f*relative - FF - 1.0f)/2.0f;
            yFirst[shareMemIndex+1] = (3.0f*FF - 4.0f*relative)/2.0f;
            yFirst[shareMemIndex+2] = (2.0f*relative - 3.0f*FF + 1.0f)/2.0f;
            yFirst[shareMemIndex+3] = FF/2.0f;

            // X basis values
            relative = fabsf(x/gridVoxelSpacing.x-(float)nodeAnte.x);
            float4 xBasis;
            float4 xFirst;
            xBasis.w= relative * relative * relative / 6.0f;
            xBasis.x= 1.0f/6.0f + relative*(relative-1.0f)/2.0f - xBasis.w;
            xBasis.z= relative + xBasis.x - 2.0f*xBasis.w;
            xBasis.y= 1.0f - xBasis.x - xBasis.z - xBasis.w;
            xFirst.w= relative * relative / 2.0f;
            xFirst.x= relative - 0.5f - xFirst.w;
            xFirst.z= 1.0f + xFirst.x - 2.0f*xFirst.w;
            xFirst.y= - xFirst.x - xFirst.z - xFirst.w;

            float Tx_x=0.0f;
            float Ty_x=0.0f;
            float Tz_x=0.0f;
            float Tx_y=0.0f;
            float Ty_y=0.0f;
            float Tz_y=0.0f;
            float Tx_z=0.0f;
            float Ty_z=0.0f;
            float Tz_z=0.0f;
            float3 displacement=make_float3(.0f, .0f, .0f);

            float4 nodeCoefficient;
            float3 tempBasis;
            float basis;

            int indexYZ, indexXYZ;
            for(short c=0; c<4; c++){
                indexYZ= ( (nodeAnte.z + c) * controlPointImageDim.y + nodeAnte.y) * controlPointImageDim.x;
                for(short b=0; b<4; b++){

                    tempBasis.x = zBasis[shareMemIndex+c] * yBasis[shareMemIndex+b]; // x derivative and no derivative
                    tempBasis.y = zBasis[shareMemIndex+c] * yFirst[shareMemIndex+b]; // y derivative
                    tempBasis.z = zFirst[shareMemIndex+c] * yBasis[shareMemIndex+b]; // z derivative

                    indexXYZ = indexYZ + nodeAnte.x;
                    nodeCoefficient = tex1Dfetch(controlPointTexture,indexXYZ++);
                    basis = xBasis.x * tempBasis.x;
                    displacement.x+=nodeCoefficient.x * basis;
                    displacement.y+=nodeCoefficient.y * basis;
                    displacement.z+=nodeCoefficient.z * basis;
                    basis = xFirst.x * tempBasis.x;
                    Tx_x+=nodeCoefficient.x * basis;
                    Ty_x+=nodeCoefficient.y * basis;
                    Tz_x+=nodeCoefficient.z * basis;
                    basis = xBasis.x * tempBasis.y;
                    Tx_y+=nodeCoefficient.x * basis;
                    Ty_y+=nodeCoefficient.y * basis;
                    Tz_y+=nodeCoefficient.z * basis;
                    basis = xBasis.x * tempBasis.z;
                    Tx_z+=nodeCoefficient.x * basis;
                    Ty_z+=nodeCoefficient.y * basis;
                    Tz_z+=nodeCoefficient.z * basis;
                    nodeCoefficient = tex1Dfetch(controlPointTexture,indexXYZ++);
                    basis = xBasis.y * tempBasis.x;
                    displacement.x+=nodeCoefficient.x * basis;
                    displacement.y+=nodeCoefficient.y * basis;
                    displacement.z+=nodeCoefficient.z * basis;
                    basis = xFirst.y * tempBasis.x;
                    Tx_x+=nodeCoefficient.x * basis;
                    Ty_x+=nodeCoefficient.y * basis;
                    Tz_x+=nodeCoefficient.z * basis;
                    basis = xBasis.y * tempBasis.y;
                    Tx_y+=nodeCoefficient.x * basis;
                    Ty_y+=nodeCoefficient.y * basis;
                    Tz_y+=nodeCoefficient.z * basis;
                    basis = xBasis.y * tempBasis.z;
                    Tx_z+=nodeCoefficient.x * basis;
                    Ty_z+=nodeCoefficient.y * basis;
                    Tz_z+=nodeCoefficient.z * basis;
                    nodeCoefficient = tex1Dfetch(controlPointTexture,indexXYZ++);
                    basis = xBasis.z * tempBasis.x;
                    displacement.x+=nodeCoefficient.x * basis;
                    displacement.y+=nodeCoefficient.y * basis;
                    displacement.z+=nodeCoefficient.z * basis;
                    basis = xFirst.z * tempBasis.x;
                    Tx_x+=nodeCoefficient.x * basis;
                    Ty_x+=nodeCoefficient.y * basis;
                    Tz_x+=nodeCoefficient.z * basis;
                    basis = xBasis.z * tempBasis.y;
                    Tx_y+=nodeCoefficient.x * basis;
                    Ty_y+=nodeCoefficient.y * basis;
                    Tz_y+=nodeCoefficient.z * basis;
                    basis = xBasis.z * tempBasis.z;
                    Tx_z+=nodeCoefficient.x * basis;
                    Ty_z+=nodeCoefficient.y * basis;
                    Tz_z+=nodeCoefficient.z * basis;
                    nodeCoefficient = tex1Dfetch(controlPointTexture,indexXYZ);
                    basis = xBasis.w * tempBasis.x;
                    displacement.x+=nodeCoefficient.x * basis;
                    displacement.y+=nodeCoefficient.y * basis;
                    displacement.z+=nodeCoefficient.z * basis;
                    basis = xFirst.w * tempBasis.x;
                    Tx_x+=nodeCoefficient.x * basis;
                    Ty_x+=nodeCoefficient.y * basis;
                    Tz_x+=nodeCoefficient.z * basis;
                    basis = xBasis.w * tempBasis.y;
                    Tx_y+=nodeCoefficient.x * basis;
                    Ty_y+=nodeCoefficient.y * basis;
                    Tz_y+=nodeCoefficient.z * basis;
                    basis = xBasis.w * tempBasis.z;
                    Tx_z+=nodeCoefficient.x * basis;
                    Ty_z+=nodeCoefficient.y * basis;
                    Tz_z+=nodeCoefficient.z * basis;

                    indexYZ += controlPointImageDim.x;
                }
            }


            Tx_x /= c_ControlPointSpacing.x;
            Ty_x /= c_ControlPointSpacing.x;
            Tz_x /= c_ControlPointSpacing.x;
            Tx_y /= c_ControlPointSpacing.y;
            Ty_y /= c_ControlPointSpacing.y;
            Tz_y /= c_ControlPointSpacing.y;
            Tx_z /= c_ControlPointSpacing.z;
            Ty_z /= c_ControlPointSpacing.z;
            Tz_z /= c_ControlPointSpacing.z;

            float Tx_x2=c_AffineMatrix0.x*Tx_x + c_AffineMatrix0.y*Ty_x + c_AffineMatrix0.z*Tz_x + 1.f;
            float Ty_x2=c_AffineMatrix0.x*Tx_y + c_AffineMatrix0.y*Ty_y + c_AffineMatrix0.z*Tz_y;
            float Tz_x2=c_AffineMatrix0.x*Tx_z + c_AffineMatrix0.y*Ty_z + c_AffineMatrix0.z*Tz_z;

            float Tx_y2=c_AffineMatrix1.x*Tx_x + c_AffineMatrix1.y*Ty_x + c_AffineMatrix1.z*Tz_x;
            float Ty_y2=c_AffineMatrix1.x*Tx_y + c_AffineMatrix1.y*Ty_y + c_AffineMatrix1.z*Tz_y + 1.f;
            float Tz_y2=c_AffineMatrix1.x*Tx_z + c_AffineMatrix1.y*Ty_z + c_AffineMatrix1.z*Tz_z;

            float Tx_z2=c_AffineMatrix2.x*Tx_x + c_AffineMatrix2.y*Ty_x + c_AffineMatrix2.z*Tz_x;
            float Ty_z2=c_AffineMatrix2.x*Tx_y + c_AffineMatrix2.y*Ty_y + c_AffineMatrix2.z*Tz_y;
            float Tz_z2=c_AffineMatrix2.x*Tx_z + c_AffineMatrix2.y*Ty_z + c_AffineMatrix2.z*Tz_z + 1.f;


            /* The Jacobian determinant is computed and stored */
            jacobianMap[tid] *= Tx_x2*Ty_y2*Tz_z2
                                + Tx_y2*Ty_z2*Tz_x2
                                + Tx_z2*Ty_x2*Tz_y2
                                - Tx_x2*Ty_z2*Tz_y2
                                - Tx_y2*Ty_x2*Tz_z2
                                - Tx_z2*Ty_y2*Tz_x2;
            displacementField_d[tid] = make_float4(position.x+displacement.x,
                                                   position.y+displacement.y,
                                                   position.z+displacement.z,
                                                   0.f);
        }
    }
}

/* *************************************************************** */
/* *************************************************************** */

__global__ void reg_bspline_ApproxJacDetFromVelField_kernel(   float *jacobianMap,
                                                                float4 *displacementField_d)
{
    const unsigned int tid= blockIdx.x*blockDim.x + threadIdx.x;
    if(tid<c_ControlPointNumber){

        int3 controlPointImageDim = c_ControlPointImageDim;

        int tempIndex=tid;
        float z = floorf(tempIndex/(controlPointImageDim.x*controlPointImageDim.y));
        tempIndex -= (int)z*(controlPointImageDim.x)*(controlPointImageDim.y);
        float y = floorf(tempIndex/(controlPointImageDim.x));
        float x = tempIndex - y*(controlPointImageDim.x);

        // The position of the control point is updated
        float4 position = displacementField_d[tid];
        float4 matrix = tex1Dfetch(txRealToVoxelMatrix,0);
        x +=    matrix.x*position.x + matrix.y*position.y  +
                matrix.z*position.z  +  matrix.w;
        matrix = tex1Dfetch(txRealToVoxelMatrix,1);
        y +=    matrix.x*position.x + matrix.y*position.y  +
                matrix.z*position.z  +  matrix.w;
        matrix = tex1Dfetch(txRealToVoxelMatrix,2);
        z +=    matrix.x*position.x + matrix.y*position.y  +
                matrix.z*position.z  +  matrix.w;

        if(x>=1 && x <controlPointImageDim.x-2 &&
           y>=1 && y <controlPointImageDim.y-2 &&
           z>=1 && z <controlPointImageDim.z-2){

            // the "nearest previous" node is determined [0,0,0]
            short3 nodeAnte;
            nodeAnte.x = (short)floorf((float)x);
            nodeAnte.y = (short)floorf((float)y);
            nodeAnte.z = (short)floorf((float)z);

            // Z basis values
            const unsigned short shareMemIndex = 4*threadIdx.x;
            __shared__ float zBasis[Block_reg_bspline_ApproxJacDetFromVelField*4];
            __shared__ float zFirst[Block_reg_bspline_ApproxJacDetFromVelField*4];
            float relative = fabsf(z-(float)nodeAnte.z);
            float FF= relative*relative;
            float FFF= FF*relative;
            float MF=1.0f-relative;
            zBasis[shareMemIndex] = MF*MF*MF/6.0f;
            zBasis[shareMemIndex+1] = (3.0f*FFF - 6.0f*FF +4.0f)/6.0f;
            zBasis[shareMemIndex+2] = (-3.0f*FFF + 3.0f*FF + 3.0f*relative + 1.0f)/6.0f;
            zBasis[shareMemIndex+3] = FFF/6.0f;
            zFirst[shareMemIndex] = (2.0f*relative - FF - 1.0f)/2.0f;
            zFirst[shareMemIndex+1] = (3.0f*FF - 4.0f*relative)/2.0f;
            zFirst[shareMemIndex+2] = (2.0f*relative - 3.0f*FF + 1.0f)/2.0f;
            zFirst[shareMemIndex+3] = FF/2.0f;

            // Y basis values
            __shared__ float yBasis[Block_reg_bspline_ApproxJacDetFromVelField*4];
            __shared__ float yFirst[Block_reg_bspline_ApproxJacDetFromVelField*4];
            relative = fabsf(y-(float)nodeAnte.y);
            FF= relative*relative;
            FFF= FF*relative;
            MF=1.0f-relative;
            yBasis[shareMemIndex] = MF*MF*MF/6.0f;
            yBasis[shareMemIndex+1] = (3.0f*FFF - 6.0f*FF +4.0f)/6.0f;
            yBasis[shareMemIndex+2] = (-3.0f*FFF + 3.0f*FF + 3.0f*relative + 1.0f)/6.0f;
            yBasis[shareMemIndex+3] = FFF/6.0f;
            yFirst[shareMemIndex] = (2.0f*relative - FF - 1.0f)/2.0f;
            yFirst[shareMemIndex+1] = (3.0f*FF - 4.0f*relative)/2.0f;
            yFirst[shareMemIndex+2] = (2.0f*relative - 3.0f*FF + 1.0f)/2.0f;
            yFirst[shareMemIndex+3] = FF/2.0f;

            // X basis values
            relative = fabsf(x-(float)nodeAnte.x);
            float4 xBasis;
            float4 xFirst;
            xBasis.w= relative * relative * relative / 6.0f;
            xBasis.x= 1.0f/6.0f + relative*(relative-1.0f)/2.0f - xBasis.w;
            xBasis.z= relative + xBasis.x - 2.0f*xBasis.w;
            xBasis.y= 1.0f - xBasis.x - xBasis.z - xBasis.w;
            xFirst.w= relative * relative / 2.0f;
            xFirst.x= relative - 0.5f - xFirst.w;
            xFirst.z= 1.0f + xFirst.x - 2.0f*xFirst.w;
            xFirst.y= - xFirst.x - xFirst.z - xFirst.w;

            --nodeAnte.x;
            --nodeAnte.y;
            --nodeAnte.z;

            float Tx_x=0.0f;
            float Ty_x=0.0f;
            float Tz_x=0.0f;
            float Tx_y=0.0f;
            float Ty_y=0.0f;
            float Tz_y=0.0f;
            float Tx_z=0.0f;
            float Ty_z=0.0f;
            float Tz_z=0.0f;
            float3 displacement=make_float3(.0f, .0f, .0f);

            float4 nodeCoefficient;
            float3 tempBasis;
            float basis;

            int indexYZ, indexXYZ;
            for(short c=0; c<4; c++){
                indexYZ= ( (nodeAnte.z + c) * controlPointImageDim.y + nodeAnte.y) * controlPointImageDim.x;
                for(short b=0; b<4; b++){

                    tempBasis.x = zBasis[shareMemIndex+c] * yBasis[shareMemIndex+b]; // x derivative and no derivative
                    tempBasis.y = zBasis[shareMemIndex+c] * yFirst[shareMemIndex+b]; // y derivative
                    tempBasis.z = zFirst[shareMemIndex+c] * yBasis[shareMemIndex+b]; // z derivative

                    indexXYZ = indexYZ + nodeAnte.x;
                    nodeCoefficient = tex1Dfetch(controlPointTexture,indexXYZ++);
                    basis = xBasis.x * tempBasis.x;
                    displacement.x+=nodeCoefficient.x * basis;
                    displacement.y+=nodeCoefficient.y * basis;
                    displacement.z+=nodeCoefficient.z * basis;
                    basis = xFirst.x * tempBasis.x;
                    Tx_x+=nodeCoefficient.x * basis;
                    Ty_x+=nodeCoefficient.y * basis;
                    Tz_x+=nodeCoefficient.z * basis;
                    basis = xBasis.x * tempBasis.y;
                    Tx_y+=nodeCoefficient.x * basis;
                    Ty_y+=nodeCoefficient.y * basis;
                    Tz_y+=nodeCoefficient.z * basis;
                    basis = xBasis.x * tempBasis.z;
                    Tx_z+=nodeCoefficient.x * basis;
                    Ty_z+=nodeCoefficient.y * basis;
                    Tz_z+=nodeCoefficient.z * basis;
                    nodeCoefficient = tex1Dfetch(controlPointTexture,indexXYZ++);
                    basis = xBasis.y * tempBasis.x;
                    displacement.x+=nodeCoefficient.x * basis;
                    displacement.y+=nodeCoefficient.y * basis;
                    displacement.z+=nodeCoefficient.z * basis;
                    basis = xFirst.y * tempBasis.x;
                    Tx_x+=nodeCoefficient.x * basis;
                    Ty_x+=nodeCoefficient.y * basis;
                    Tz_x+=nodeCoefficient.z * basis;
                    basis = xBasis.y * tempBasis.y;
                    Tx_y+=nodeCoefficient.x * basis;
                    Ty_y+=nodeCoefficient.y * basis;
                    Tz_y+=nodeCoefficient.z * basis;
                    basis = xBasis.y * tempBasis.z;
                    Tx_z+=nodeCoefficient.x * basis;
                    Ty_z+=nodeCoefficient.y * basis;
                    Tz_z+=nodeCoefficient.z * basis;
                    nodeCoefficient = tex1Dfetch(controlPointTexture,indexXYZ++);
                    basis = xBasis.z * tempBasis.x;
                    displacement.x+=nodeCoefficient.x * basis;
                    displacement.y+=nodeCoefficient.y * basis;
                    displacement.z+=nodeCoefficient.z * basis;
                    basis = xFirst.z * tempBasis.x;
                    Tx_x+=nodeCoefficient.x * basis;
                    Ty_x+=nodeCoefficient.y * basis;
                    Tz_x+=nodeCoefficient.z * basis;
                    basis = xBasis.z * tempBasis.y;
                    Tx_y+=nodeCoefficient.x * basis;
                    Ty_y+=nodeCoefficient.y * basis;
                    Tz_y+=nodeCoefficient.z * basis;
                    basis = xBasis.z * tempBasis.z;
                    Tx_z+=nodeCoefficient.x * basis;
                    Ty_z+=nodeCoefficient.y * basis;
                    Tz_z+=nodeCoefficient.z * basis;
                    nodeCoefficient = tex1Dfetch(controlPointTexture,indexXYZ);
                    basis = xBasis.w * tempBasis.x;
                    displacement.x+=nodeCoefficient.x * basis;
                    displacement.y+=nodeCoefficient.y * basis;
                    displacement.z+=nodeCoefficient.z * basis;
                    basis = xFirst.w * tempBasis.x;
                    Tx_x+=nodeCoefficient.x * basis;
                    Ty_x+=nodeCoefficient.y * basis;
                    Tz_x+=nodeCoefficient.z * basis;
                    basis = xBasis.w * tempBasis.y;
                    Tx_y+=nodeCoefficient.x * basis;
                    Ty_y+=nodeCoefficient.y * basis;
                    Tz_y+=nodeCoefficient.z * basis;
                    basis = xBasis.w * tempBasis.z;
                    Tx_z+=nodeCoefficient.x * basis;
                    Ty_z+=nodeCoefficient.y * basis;
                    Tz_z+=nodeCoefficient.z * basis;

                    indexYZ += controlPointImageDim.x;
                }
            }

            Tx_x /= c_ControlPointSpacing.x;
            Ty_x /= c_ControlPointSpacing.x;
            Tz_x /= c_ControlPointSpacing.x;
            Tx_y /= c_ControlPointSpacing.y;
            Ty_y /= c_ControlPointSpacing.y;
            Tz_y /= c_ControlPointSpacing.y;
            Tx_z /= c_ControlPointSpacing.z;
            Ty_z /= c_ControlPointSpacing.z;
            Tz_z /= c_ControlPointSpacing.z;

            float Tx_x2=c_AffineMatrix0.x*Tx_x + c_AffineMatrix0.y*Ty_x + c_AffineMatrix0.z*Tz_x + 1.f;
            float Ty_x2=c_AffineMatrix0.x*Tx_y + c_AffineMatrix0.y*Ty_y + c_AffineMatrix0.z*Tz_y;
            float Tz_x2=c_AffineMatrix0.x*Tx_z + c_AffineMatrix0.y*Ty_z + c_AffineMatrix0.z*Tz_z;

            float Tx_y2=c_AffineMatrix1.x*Tx_x + c_AffineMatrix1.y*Ty_x + c_AffineMatrix1.z*Tz_x;
            float Ty_y2=c_AffineMatrix1.x*Tx_y + c_AffineMatrix1.y*Ty_y + c_AffineMatrix1.z*Tz_y + 1.f;
            float Tz_y2=c_AffineMatrix1.x*Tx_z + c_AffineMatrix1.y*Ty_z + c_AffineMatrix1.z*Tz_z;

            float Tx_z2=c_AffineMatrix2.x*Tx_x + c_AffineMatrix2.y*Ty_x + c_AffineMatrix2.z*Tz_x;
            float Ty_z2=c_AffineMatrix2.x*Tx_y + c_AffineMatrix2.y*Ty_y + c_AffineMatrix2.z*Tz_y;
            float Tz_z2=c_AffineMatrix2.x*Tx_z + c_AffineMatrix2.y*Ty_z + c_AffineMatrix2.z*Tz_z + 1.f;


            /* The Jacobian determinant is computed and stored */
            jacobianMap[tid] *= Tx_x2*Ty_y2*Tz_z2
                                + Tx_y2*Ty_z2*Tz_x2
                                + Tx_z2*Ty_x2*Tz_y2
                                - Tx_x2*Ty_z2*Tz_y2
                                - Tx_y2*Ty_x2*Tz_z2
                                - Tx_z2*Ty_y2*Tz_x2;
            displacementField_d[tid] = make_float4(position.x+displacement.x,
                                                   position.y+displacement.y,
                                                   position.z+displacement.z,
                                                   0.f);
        }
    }
}

/* *************************************************************** */
/* *************************************************************** */

__global__ void reg_bspline_JacobianMatrix_kernel(float *jacobianMatrices, float *jacobianDeterminant)
{
    const unsigned int tid= blockIdx.x*blockDim.x + threadIdx.x;
    if(tid<c_VoxelNumber){

        int3 imageSize = c_TargetImageDim;

        unsigned int tempIndex=tid;
        const unsigned short z =(unsigned short)(tempIndex/(imageSize.x*imageSize.y));
        tempIndex -= z*(imageSize.x)*(imageSize.y);
        const unsigned short y =(unsigned short)(tempIndex/(imageSize.x));
        const unsigned short x = tempIndex - y*(imageSize.x);

        // the "nearest previous" node is determined [0,0,0]
        short3 nodeAnte;
        float3 gridVoxelSpacing = c_ControlPointVoxelSpacing;
        nodeAnte.x = (short)floorf((float)x/gridVoxelSpacing.x);
        nodeAnte.y = (short)floorf((float)y/gridVoxelSpacing.y);
        nodeAnte.z = (short)floorf((float)z/gridVoxelSpacing.z);

        // Z basis values
        const unsigned short shareMemIndex = 4*threadIdx.x;
        __shared__ float zBasis[Block_reg_bspline_JacobianMatrix*4];
        __shared__ float zFirst[Block_reg_bspline_JacobianMatrix*4];
        float relative = fabsf((float)z/gridVoxelSpacing.z-(float)nodeAnte.z);
        float FF= relative*relative;
        float FFF= FF*relative;
        float MF=1.0f-relative;
        zBasis[shareMemIndex] = MF*MF*MF/6.0f;
        zBasis[shareMemIndex+1] = (3.0f*FFF - 6.0f*FF +4.0f)/6.0f;
        zBasis[shareMemIndex+2] = (-3.0f*FFF + 3.0f*FF + 3.0f*relative + 1.0f)/6.0f;
        zBasis[shareMemIndex+3] = FFF/6.0f;
        zFirst[shareMemIndex] = (2.0f*relative - FF - 1.0f)/2.0f;
        zFirst[shareMemIndex+1] = (3.0f*FF - 4.0f*relative)/2.0f;
        zFirst[shareMemIndex+2] = (2.0f*relative - 3.0f*FF + 1.0f)/2.0f;
        zFirst[shareMemIndex+3] = FF/2.0f;

        // Y basis values
        __shared__ float yBasis[Block_reg_bspline_JacobianMatrix*4];
        __shared__ float yFirst[Block_reg_bspline_JacobianMatrix*4];
        relative = fabsf((float)y/gridVoxelSpacing.y-(float)nodeAnte.y);
        FF= relative*relative;
        FFF= FF*relative;
        MF=1.0f-relative;
        yBasis[shareMemIndex] = MF*MF*MF/6.0f;
        yBasis[shareMemIndex+1] = (3.0f*FFF - 6.0f*FF +4.0f)/6.0f;
        yBasis[shareMemIndex+2] = (-3.0f*FFF + 3.0f*FF + 3.0f*relative + 1.0f)/6.0f;
        yBasis[shareMemIndex+3] = FFF/6.0f;
        yFirst[shareMemIndex] = (2.0f*relative - FF - 1.0f)/2.0f;
        yFirst[shareMemIndex+1] = (3.0f*FF - 4.0f*relative)/2.0f;
        yFirst[shareMemIndex+2] = (2.0f*relative - 3.0f*FF + 1.0f)/2.0f;
        yFirst[shareMemIndex+3] = FF/2.0f;

        // X basis values
        relative = fabsf((float)x/gridVoxelSpacing.x-(float)nodeAnte.x);
        float4 xBasis;
        float4 xFirst;
        xBasis.w= relative * relative * relative / 6.0f;
        xBasis.x= 1.0f/6.0f + relative*(relative-1.0f)/2.0f - xBasis.w;
        xBasis.z= relative + xBasis.x - 2.0f*xBasis.w;
        xBasis.y= 1.0f - xBasis.x - xBasis.z - xBasis.w;
        xFirst.w= relative * relative / 2.0f;
        xFirst.x= relative - 0.5f - xFirst.w;
        xFirst.z= 1.0f + xFirst.x - 2.0f*xFirst.w;
        xFirst.y= - xFirst.x - xFirst.z - xFirst.w;

        int3 controlPointImageDim = c_ControlPointImageDim;

        float Tx_x=0.0f;
        float Ty_x=0.0f;
        float Tz_x=0.0f;
        float Tx_y=0.0f;
        float Ty_y=0.0f;
        float Tz_y=0.0f;
        float Tx_z=0.0f;
        float Ty_z=0.0f;
        float Tz_z=0.0f;

        float4 nodeCoefficient;
        float3 tempBasis;
        float basis;

        int indexYZ, indexXYZ;
        for(short c=0; c<4; c++){
            indexYZ= ( (nodeAnte.z + c) * controlPointImageDim.y + nodeAnte.y) * controlPointImageDim.x;
            for(short b=0; b<4; b++){

                tempBasis.x = zBasis[shareMemIndex+c] * yBasis[shareMemIndex+b];
                tempBasis.y = zBasis[shareMemIndex+c] * yFirst[shareMemIndex+b];
                tempBasis.z = zFirst[shareMemIndex+c] * yBasis[shareMemIndex+b];

                indexXYZ = indexYZ + nodeAnte.x;
                nodeCoefficient = tex1Dfetch(controlPointTexture,indexXYZ++);
                basis = xFirst.x * tempBasis.x;
                Tx_x+=nodeCoefficient.x * basis;
                Ty_x+=nodeCoefficient.y * basis;
                Tz_x+=nodeCoefficient.z * basis;
                basis = xBasis.x * tempBasis.y;
                Tx_y+=nodeCoefficient.x * basis;
                Ty_y+=nodeCoefficient.y * basis;
                Tz_y+=nodeCoefficient.z * basis;
                basis = xBasis.x * tempBasis.z;
                Tx_z+=nodeCoefficient.x * basis;
                Ty_z+=nodeCoefficient.y * basis;
                Tz_z+=nodeCoefficient.z * basis;
                nodeCoefficient = tex1Dfetch(controlPointTexture,indexXYZ++);
                basis = xFirst.y * tempBasis.x;
                Tx_x+=nodeCoefficient.x * basis;
                Ty_x+=nodeCoefficient.y * basis;
                Tz_x+=nodeCoefficient.z * basis;
                basis = xBasis.y * tempBasis.y;
                Tx_y+=nodeCoefficient.x * basis;
                Ty_y+=nodeCoefficient.y * basis;
                Tz_y+=nodeCoefficient.z * basis;
                basis = xBasis.y * tempBasis.z;
                Tx_z+=nodeCoefficient.x * basis;
                Ty_z+=nodeCoefficient.y * basis;
                Tz_z+=nodeCoefficient.z * basis;
                nodeCoefficient = tex1Dfetch(controlPointTexture,indexXYZ++);
                basis = xFirst.z * tempBasis.x;
                Tx_x+=nodeCoefficient.x * basis;
                Ty_x+=nodeCoefficient.y * basis;
                Tz_x+=nodeCoefficient.z * basis;
                basis = xBasis.z * tempBasis.y;
                Tx_y+=nodeCoefficient.x * basis;
                Ty_y+=nodeCoefficient.y * basis;
                Tz_y+=nodeCoefficient.z * basis;
                basis = xBasis.z * tempBasis.z;
                Tx_z+=nodeCoefficient.x * basis;
                Ty_z+=nodeCoefficient.y * basis;
                Tz_z+=nodeCoefficient.z * basis;
                nodeCoefficient = tex1Dfetch(controlPointTexture,indexXYZ);
                basis = xFirst.w * tempBasis.x;
                Tx_x+=nodeCoefficient.x * basis;
                Ty_x+=nodeCoefficient.y * basis;
                Tz_x+=nodeCoefficient.z * basis;
                basis = xBasis.w * tempBasis.y;
                Tx_y+=nodeCoefficient.x * basis;
                Ty_y+=nodeCoefficient.y * basis;
                Tz_y+=nodeCoefficient.z * basis;
                basis = xBasis.w * tempBasis.z;
                Tx_z+=nodeCoefficient.x * basis;
                Ty_z+=nodeCoefficient.y * basis;
                Tz_z+=nodeCoefficient.z * basis;

                indexYZ += controlPointImageDim.x;
            }
        }

        Tx_x /= c_ControlPointSpacing.x;
        Ty_x /= c_ControlPointSpacing.x;
        Tz_x /= c_ControlPointSpacing.x;
        Tx_y /= c_ControlPointSpacing.y;
        Ty_y /= c_ControlPointSpacing.y;
        Tz_y /= c_ControlPointSpacing.y;
        Tx_z /= c_ControlPointSpacing.z;
        Ty_z /= c_ControlPointSpacing.z;
        Tz_z /= c_ControlPointSpacing.z;

        float Tx_x2=c_AffineMatrix0.x*Tx_x + c_AffineMatrix0.y*Ty_x + c_AffineMatrix0.z*Tz_x;
        float Ty_x2=c_AffineMatrix0.x*Tx_y + c_AffineMatrix0.y*Ty_y + c_AffineMatrix0.z*Tz_y;
        float Tz_x2=c_AffineMatrix0.x*Tx_z + c_AffineMatrix0.y*Ty_z + c_AffineMatrix0.z*Tz_z;

        float Tx_y2=c_AffineMatrix1.x*Tx_x + c_AffineMatrix1.y*Ty_x + c_AffineMatrix1.z*Tz_x;
        float Ty_y2=c_AffineMatrix1.x*Tx_y + c_AffineMatrix1.y*Ty_y + c_AffineMatrix1.z*Tz_y;
        float Tz_y2=c_AffineMatrix1.x*Tx_z + c_AffineMatrix1.y*Ty_z + c_AffineMatrix1.z*Tz_z;

        float Tx_z2=c_AffineMatrix2.x*Tx_x + c_AffineMatrix2.y*Ty_x + c_AffineMatrix2.z*Tz_x;
        float Ty_z2=c_AffineMatrix2.x*Tx_y + c_AffineMatrix2.y*Ty_y + c_AffineMatrix2.z*Tz_y;
        float Tz_z2=c_AffineMatrix2.x*Tx_z + c_AffineMatrix2.y*Ty_z + c_AffineMatrix2.z*Tz_z;

        /* The Jacobian matrix is inverted and stored */
        float det= Tx_x2*Ty_y2*Tz_z2
                + Tx_y2*Ty_z2*Tz_x2
                + Tx_z2*Ty_x2*Tz_y2
                - Tx_x2*Ty_z2*Tz_y2
                - Tx_y2*Ty_x2*Tz_z2
                - Tx_z2*Ty_y2*Tz_x2;

        jacobianDeterminant[tid]= det;

        det = 1.f / det;
        int id = 9*tid;
        jacobianMatrices[id++] = det * (Ty_y2* Tz_z2 - Tz_y2*Ty_z2);
        jacobianMatrices[id++] = det * (Tz_y2* Tx_z2 - Tx_y2*Tz_z2);
        jacobianMatrices[id++] = det * (Tx_y2* Ty_z2 - Ty_y2*Tx_z2);

        jacobianMatrices[id++] = det * (Tz_x2* Ty_z2 - Ty_x2*Tz_z2);
        jacobianMatrices[id++] = det * (Tx_x2* Tz_z2 - Tz_x2*Tx_z2);
        jacobianMatrices[id++] = det * (Ty_x2* Tx_z2 - Tx_x2*Ty_z2);

        jacobianMatrices[id++] = det * (Ty_x2* Tz_y2 - Tz_x2*Ty_y2);
        jacobianMatrices[id++] = det * (Tz_x2* Tx_y2 - Tx_x2*Tz_y2);
        jacobianMatrices[id]  =  det * (Tx_x2* Ty_y2 - Ty_x2*Tx_y2);

    }
    return;
}

/* *************************************************************** */
/* *************************************************************** */

__global__ void reg_bspline_ApproxJacobianMatrix_kernel(float *matrices, float *determinant)
{
    __shared__ float basisX[27];
    __shared__ float basisY[27];
    __shared__ float basisZ[27];
    if(threadIdx.x<27){
        basisX[threadIdx.x] = tex1Dfetch(xBasisTexture,threadIdx.x);
        basisY[threadIdx.x] = tex1Dfetch(yBasisTexture,threadIdx.x);
        basisZ[threadIdx.x] = tex1Dfetch(zBasisTexture,threadIdx.x);
    }
    __syncthreads();

    const int3 gridSize = c_ControlPointImageDim;

    const unsigned int tid= blockIdx.x*blockDim.x + threadIdx.x;
    int tempIndex=tid;
    const int z =(int)(tempIndex/(gridSize.x*gridSize.y));
    tempIndex -= z*(gridSize.x)*(gridSize.y);
    const int y =(int)(tempIndex/(gridSize.x));
    const int x = tempIndex - y*(gridSize.x) ;

    if( 0<x && x<gridSize.x-1 &&
        0<y && y<gridSize.y-1 &&
        0<z && z<gridSize.z-1){

        /* The Jacobian matrix is computed */
        float Tx_x=0.0f;
        float Ty_x=0.0f;
        float Tz_x=0.0f;
        float Tx_y=0.0f;
        float Ty_y=0.0f;
        float Tz_y=0.0f;
        float Tx_z=0.0f;
        float Ty_z=0.0f;
        float Tz_z=0.0f;
        float4 controlPointPosition;
        float tempBasis;
        int index2=0;
        for(int c=z-1; c<z+2; c++){
            for(int b=y-1; b<y+2; b++){
                int index = (c*gridSize.y+b)*gridSize.x+x-1;
                for(int a=x-1; a<x+2; a++){
                    controlPointPosition = tex1Dfetch(controlPointTexture,index++);
                    tempBasis=basisX[index2];
                    Tx_x+=controlPointPosition.x * tempBasis;
                    Ty_x+=controlPointPosition.y * tempBasis;
                    Tz_x+=controlPointPosition.z * tempBasis;
                    tempBasis=basisY[index2];
                    Tx_y+=controlPointPosition.x * tempBasis;
                    Ty_y+=controlPointPosition.y * tempBasis;
                    Tz_y+=controlPointPosition.z * tempBasis;
                    tempBasis=basisZ[index2++];
                    Tx_z+=controlPointPosition.x * tempBasis;
                    Ty_z+=controlPointPosition.y * tempBasis;
                    Tz_z+=controlPointPosition.z * tempBasis;
                }
            }
        }
        Tx_x /= c_ControlPointSpacing.x;
        Ty_x /= c_ControlPointSpacing.x;
        Tz_x /= c_ControlPointSpacing.x;
        Tx_y /= c_ControlPointSpacing.y;
        Ty_y /= c_ControlPointSpacing.y;
        Tz_y /= c_ControlPointSpacing.y;
        Tx_z /= c_ControlPointSpacing.z;
        Ty_z /= c_ControlPointSpacing.z;
        Tz_z /= c_ControlPointSpacing.z;

        /* The Jacobian matrix is reoriented */
        float Tx_x2=c_AffineMatrix0.x*Tx_x + c_AffineMatrix0.y*Ty_x + c_AffineMatrix0.z*Tz_x;
        float Ty_x2=c_AffineMatrix0.x*Tx_y + c_AffineMatrix0.y*Ty_y + c_AffineMatrix0.z*Tz_y;
        float Tz_x2=c_AffineMatrix0.x*Tx_z + c_AffineMatrix0.y*Ty_z + c_AffineMatrix0.z*Tz_z;

        float Tx_y2=c_AffineMatrix1.x*Tx_x + c_AffineMatrix1.y*Ty_x + c_AffineMatrix1.z*Tz_x;
        float Ty_y2=c_AffineMatrix1.x*Tx_y + c_AffineMatrix1.y*Ty_y + c_AffineMatrix1.z*Tz_y;
        float Tz_y2=c_AffineMatrix1.x*Tx_z + c_AffineMatrix1.y*Ty_z + c_AffineMatrix1.z*Tz_z;

        float Tx_z2=c_AffineMatrix2.x*Tx_x + c_AffineMatrix2.y*Ty_x + c_AffineMatrix2.z*Tz_x;
        float Ty_z2=c_AffineMatrix2.x*Tx_y + c_AffineMatrix2.y*Ty_y + c_AffineMatrix2.z*Tz_y;
        float Tz_z2=c_AffineMatrix2.x*Tx_z + c_AffineMatrix2.y*Ty_z + c_AffineMatrix2.z*Tz_z;

        /* The Jacobian determinant is computed and stored */

        float det= Tx_x2*Ty_y2*Tz_z2
                + Tx_y2*Ty_z2*Tz_x2
                + Tx_z2*Ty_x2*Tz_y2
                - Tx_x2*Ty_z2*Tz_y2
                - Tx_y2*Ty_x2*Tz_z2
                - Tx_z2*Ty_y2*Tz_x2;

        int id = ((z-1)*(gridSize.y-2)+y-1)*(gridSize.x-2)+x-1;
        determinant[id]= det ;
        id *= 9;

        det = 1.f / det;
        matrices[id++] = det * (Ty_y2*Tz_z2 - Tz_y2*Ty_z2);
        matrices[id++] = det * (Tz_y2*Tx_z2 - Tx_y2*Tz_z2);
        matrices[id++] = det * (Tx_y2*Ty_z2 - Ty_y2*Tx_z2);

        matrices[id++] = det * (Tz_x2*Ty_z2 - Ty_x2*Tz_z2);
        matrices[id++] = det * (Tx_x2*Tz_z2 - Tz_x2*Tx_z2);
        matrices[id++] = det * (Ty_x2*Tx_z2 - Tx_x2*Ty_z2);

        matrices[id++] = det * (Ty_x2*Tz_y2 - Tz_x2*Ty_y2);
        matrices[id++] = det * (Tz_x2*Tx_y2 - Tx_x2*Tz_y2);
        matrices[id]  =  det * (Tx_x2*Ty_y2 - Ty_x2*Tx_y2);

    }
}

/* *************************************************************** */
/* *************************************************************** */

__global__ void reg_bspline_JacobianMatrixFromVel_kernel(  float *jacobianMatrices,
                                                            float *jacobianDeterminant,
                                                            float4 *voxelPosition_array)
{
    const unsigned int tid= blockIdx.x*blockDim.x + threadIdx.x;
    if(tid<c_VoxelNumber){

        float4 voxelPosition = voxelPosition_array[tid];

        // the "nearest previous" node is determined [0,0,0]
        short3 nodeAnte;
        float3 gridVoxelSpacing = c_ControlPointVoxelSpacing;
        nodeAnte.x = (short)floorf((float)voxelPosition.x/gridVoxelSpacing.x);
        nodeAnte.y = (short)floorf((float)voxelPosition.y/gridVoxelSpacing.y);
        nodeAnte.z = (short)floorf((float)voxelPosition.z/gridVoxelSpacing.z);

        int3 controlPointImageDim = c_ControlPointImageDim;

        if(nodeAnte.x<0 || nodeAnte.y<0 || nodeAnte.z<0 ||
           nodeAnte.x>controlPointImageDim.x-4 ||
           nodeAnte.y>controlPointImageDim.y-4 ||
           nodeAnte.z>controlPointImageDim.z-4){
           return;
        }

        // Z basis values
        const unsigned short shareMemIndex = 4*threadIdx.x;
        __shared__ float zBasis[Block_reg_bspline_JacobianMatrixFromVel*4];
        __shared__ float zFirst[Block_reg_bspline_JacobianMatrixFromVel*4];
        float relative = fabsf(voxelPosition.z/gridVoxelSpacing.z-(float)nodeAnte.z);
        float FF= relative*relative;
        float FFF= FF*relative;
        float MF=1.0f-relative;
        zBasis[shareMemIndex] = MF*MF*MF/6.0f;
        zBasis[shareMemIndex+1] = (3.0f*FFF - 6.0f*FF +4.0f)/6.0f;
        zBasis[shareMemIndex+2] = (-3.0f*FFF + 3.0f*FF + 3.0f*relative + 1.0f)/6.0f;
        zBasis[shareMemIndex+3] = FFF/6.0f;
        zFirst[shareMemIndex] = (2.0f*relative - FF - 1.0f)/2.0f;
        zFirst[shareMemIndex+1] = (3.0f*FF - 4.0f*relative)/2.0f;
        zFirst[shareMemIndex+2] = (2.0f*relative - 3.0f*FF + 1.0f)/2.0f;
        zFirst[shareMemIndex+3] = FF/2.0f;

        // Y basis values
        __shared__ float yBasis[Block_reg_bspline_JacobianMatrixFromVel*4];
        __shared__ float yFirst[Block_reg_bspline_JacobianMatrixFromVel*4];
        relative = fabsf(voxelPosition.y/gridVoxelSpacing.y-(float)nodeAnte.y);
        FF= relative*relative;
        FFF= FF*relative;
        MF=1.0f-relative;
        yBasis[shareMemIndex] = MF*MF*MF/6.0f;
        yBasis[shareMemIndex+1] = (3.0f*FFF - 6.0f*FF +4.0f)/6.0f;
        yBasis[shareMemIndex+2] = (-3.0f*FFF + 3.0f*FF + 3.0f*relative + 1.0f)/6.0f;
        yBasis[shareMemIndex+3] = FFF/6.0f;
        yFirst[shareMemIndex] = (2.0f*relative - FF - 1.0f)/2.0f;
        yFirst[shareMemIndex+1] = (3.0f*FF - 4.0f*relative)/2.0f;
        yFirst[shareMemIndex+2] = (2.0f*relative - 3.0f*FF + 1.0f)/2.0f;
        yFirst[shareMemIndex+3] = FF/2.0f;

        // X basis values
        relative = fabsf(voxelPosition.x/gridVoxelSpacing.x-(float)nodeAnte.x);
        float4 xBasis;
        float4 xFirst;
        xBasis.w= relative * relative * relative / 6.0f;
        xBasis.x= 1.0f/6.0f + relative*(relative-1.0f)/2.0f - xBasis.w;
        xBasis.z= relative + xBasis.x - 2.0f*xBasis.w;
        xBasis.y= 1.0f - xBasis.x - xBasis.z - xBasis.w;
        xFirst.w= relative * relative / 2.0f;
        xFirst.x= relative - 0.5f - xFirst.w;
        xFirst.z= 1.0f + xFirst.x - 2.0f*xFirst.w;
        xFirst.y= - xFirst.x - xFirst.z - xFirst.w;

        float4 position=make_float4(0.f,0.f,0.f,0.f);
        float Tx_x=0.0f;
        float Ty_x=0.0f;
        float Tz_x=0.0f;
        float Tx_y=0.0f;
        float Ty_y=0.0f;
        float Tz_y=0.0f;
        float Tx_z=0.0f;
        float Ty_z=0.0f;
        float Tz_z=0.0f;

        float4 nodeCoefficient;
        float3 tempBasis;
        float basis;

        int indexYZ, indexXYZ;
        for(short c=0; c<4; c++){
            indexYZ= ( (nodeAnte.z + c) * controlPointImageDim.y + nodeAnte.y) * controlPointImageDim.x;
            for(short b=0; b<4; b++){

                tempBasis.x = zBasis[shareMemIndex+c] * yBasis[shareMemIndex+b];
                tempBasis.y = zBasis[shareMemIndex+c] * yFirst[shareMemIndex+b];
                tempBasis.z = zFirst[shareMemIndex+c] * yBasis[shareMemIndex+b];

                indexXYZ = indexYZ + nodeAnte.x;
                nodeCoefficient = tex1Dfetch(controlPointTexture,indexXYZ++);
                basis = xBasis.x * tempBasis.x;
                position.x += nodeCoefficient.x * basis;
                position.y += nodeCoefficient.y * basis;
                position.z += nodeCoefficient.z * basis;
                basis = xFirst.x * tempBasis.x;
                Tx_x+=nodeCoefficient.x * basis;
                Ty_x+=nodeCoefficient.y * basis;
                Tz_x+=nodeCoefficient.z * basis;
                basis = xBasis.x * tempBasis.y;
                Tx_y+=nodeCoefficient.x * basis;
                Ty_y+=nodeCoefficient.y * basis;
                Tz_y+=nodeCoefficient.z * basis;
                basis = xBasis.x * tempBasis.z;
                Tx_z+=nodeCoefficient.x * basis;
                Ty_z+=nodeCoefficient.y * basis;
                Tz_z+=nodeCoefficient.z * basis;
                nodeCoefficient = tex1Dfetch(controlPointTexture,indexXYZ++);
                basis = xBasis.y * tempBasis.x;
                position.x += nodeCoefficient.x * basis;
                position.y += nodeCoefficient.y * basis;
                position.z += nodeCoefficient.z * basis;
                basis = xFirst.y * tempBasis.x;
                Tx_x+=nodeCoefficient.x * basis;
                Ty_x+=nodeCoefficient.y * basis;
                Tz_x+=nodeCoefficient.z * basis;
                basis = xBasis.y * tempBasis.y;
                Tx_y+=nodeCoefficient.x * basis;
                Ty_y+=nodeCoefficient.y * basis;
                Tz_y+=nodeCoefficient.z * basis;
                basis = xBasis.y * tempBasis.z;
                Tx_z+=nodeCoefficient.x * basis;
                Ty_z+=nodeCoefficient.y * basis;
                Tz_z+=nodeCoefficient.z * basis;
                nodeCoefficient = tex1Dfetch(controlPointTexture,indexXYZ++);
                basis = xBasis.z * tempBasis.x;
                position.x += nodeCoefficient.x * basis;
                position.y += nodeCoefficient.y * basis;
                position.z += nodeCoefficient.z * basis;
                basis = xFirst.z * tempBasis.x;
                Tx_x+=nodeCoefficient.x * basis;
                Ty_x+=nodeCoefficient.y * basis;
                Tz_x+=nodeCoefficient.z * basis;
                basis = xBasis.z * tempBasis.y;
                Tx_y+=nodeCoefficient.x * basis;
                Ty_y+=nodeCoefficient.y * basis;
                Tz_y+=nodeCoefficient.z * basis;
                basis = xBasis.z * tempBasis.z;
                Tx_z+=nodeCoefficient.x * basis;
                Ty_z+=nodeCoefficient.y * basis;
                Tz_z+=nodeCoefficient.z * basis;
                nodeCoefficient = tex1Dfetch(controlPointTexture,indexXYZ);
                basis = xBasis.w * tempBasis.x;
                position.x += nodeCoefficient.x * basis;
                position.y += nodeCoefficient.y * basis;
                position.z += nodeCoefficient.z * basis;
                basis = xFirst.w * tempBasis.x;
                Tx_x+=nodeCoefficient.x * basis;
                Ty_x+=nodeCoefficient.y * basis;
                Tz_x+=nodeCoefficient.z * basis;
                basis = xBasis.w * tempBasis.y;
                Tx_y+=nodeCoefficient.x * basis;
                Ty_y+=nodeCoefficient.y * basis;
                Tz_y+=nodeCoefficient.z * basis;
                basis = xBasis.w * tempBasis.z;
                Tx_z+=nodeCoefficient.x * basis;
                Ty_z+=nodeCoefficient.y * basis;
                Tz_z+=nodeCoefficient.z * basis;

                indexYZ += controlPointImageDim.x;
            }
        }
        voxelPosition_array[tid]=position;

        Tx_x /= c_ControlPointSpacing.x;
        Ty_x /= c_ControlPointSpacing.x;
        Tz_x /= c_ControlPointSpacing.x;
        Tx_y /= c_ControlPointSpacing.y;
        Ty_y /= c_ControlPointSpacing.y;
        Tz_y /= c_ControlPointSpacing.y;
        Tx_z /= c_ControlPointSpacing.z;
        Ty_z /= c_ControlPointSpacing.z;
        Tz_z /= c_ControlPointSpacing.z;

        float Tx_x2=c_AffineMatrix0.x*Tx_x + c_AffineMatrix0.y*Ty_x + c_AffineMatrix0.z*Tz_x;
        float Ty_x2=c_AffineMatrix0.x*Tx_y + c_AffineMatrix0.y*Ty_y + c_AffineMatrix0.z*Tz_y;
        float Tz_x2=c_AffineMatrix0.x*Tx_z + c_AffineMatrix0.y*Ty_z + c_AffineMatrix0.z*Tz_z;

        float Tx_y2=c_AffineMatrix1.x*Tx_x + c_AffineMatrix1.y*Ty_x + c_AffineMatrix1.z*Tz_x;
        float Ty_y2=c_AffineMatrix1.x*Tx_y + c_AffineMatrix1.y*Ty_y + c_AffineMatrix1.z*Tz_y;
        float Tz_y2=c_AffineMatrix1.x*Tx_z + c_AffineMatrix1.y*Ty_z + c_AffineMatrix1.z*Tz_z;

        float Tx_z2=c_AffineMatrix2.x*Tx_x + c_AffineMatrix2.y*Ty_x + c_AffineMatrix2.z*Tz_x;
        float Ty_z2=c_AffineMatrix2.x*Tx_y + c_AffineMatrix2.y*Ty_y + c_AffineMatrix2.z*Tz_y;
        float Tz_z2=c_AffineMatrix2.x*Tx_z + c_AffineMatrix2.y*Ty_z + c_AffineMatrix2.z*Tz_z;

        /* The Jacobian matrix is inverted and stored */
        float det= Tx_x2*Ty_y2*Tz_z2
                + Tx_y2*Ty_z2*Tz_x2
                + Tx_z2*Ty_x2*Tz_y2
                - Tx_x2*Ty_z2*Tz_y2
                - Tx_y2*Ty_x2*Tz_z2
                - Tx_z2*Ty_y2*Tz_x2;

        jacobianDeterminant[tid]= det;

        det = 1.f / det;
        int id = 9*tid;
        jacobianMatrices[id++] = det * (Ty_y2* Tz_z2 - Tz_y2*Ty_z2);
        jacobianMatrices[id++] = det * (Tz_y2* Tx_z2 - Tx_y2*Tz_z2);
        jacobianMatrices[id++] = det * (Tx_y2* Ty_z2 - Ty_y2*Tx_z2);

        jacobianMatrices[id++] = det * (Tz_x2* Ty_z2 - Ty_x2*Tz_z2);
        jacobianMatrices[id++] = det * (Tx_x2* Tz_z2 - Tz_x2*Tx_z2);
        jacobianMatrices[id++] = det * (Ty_x2* Tx_z2 - Tx_x2*Ty_z2);

        jacobianMatrices[id++] = det * (Ty_x2* Tz_y2 - Tz_x2*Ty_y2);
        jacobianMatrices[id++] = det * (Tz_x2* Tx_y2 - Tx_x2*Tz_y2);
        jacobianMatrices[id]  =  det * (Tx_x2* Ty_y2 - Ty_x2*Tx_y2);

    }
    return;
}

/* *************************************************************** */
/* *************************************************************** */

__global__ void reg_bspline_JacobianGradient_kernel(float4 *gradient)
{
    const int tid= blockIdx.x*blockDim.x + threadIdx.x;
    if(tid<c_ControlPointNumber){

        int3 gridSize = c_ControlPointImageDim;

        int tempIndex=tid;
        const int z =(int)(tempIndex/(gridSize.x*gridSize.y));
        tempIndex -= z*(gridSize.x)*(gridSize.y);
        const int y =(int)(tempIndex/(gridSize.x));
        const int x = tempIndex - y*(gridSize.x) ;

        float3 jacobianConstraint=make_float3(0.0f,0.0f,0.0f);
        float3 basisValues;
        float3 basis;
        float3 first;
        float relative;
        int pre;

        int3 targetSize = c_TargetImageDim;
        float3 gridVoxelSpacing = c_ControlPointVoxelSpacing;

        // Loop over all the control points in the surrounding area
        for(int pixelZ=(int)((z-3)*gridVoxelSpacing.z);pixelZ<(int)((z+1)*gridVoxelSpacing.z); pixelZ++){
            if(pixelZ>-1 && pixelZ<targetSize.z){

                pre=(int)((float)pixelZ/gridVoxelSpacing.z);
                relative =(float)pixelZ/gridVoxelSpacing.z-(float)pre;
                switch(z-pre){
                    case 0:
                        basis.z=(relative-1.0f)*(relative-1.0f)*(relative-1.0f)/6.0f;
                        first.z=(2.0f*relative - relative*relative - 1.0f) / 2.0f;
                        break;
                    case 1:
                        basis.z=(3.0f*relative*relative*relative - 6.0f*relative*relative + 4.0f)/6.0f;
                        first.z=(3.0f*relative*relative - 4.0f*relative) / 2.0f;
                        break;
                    case 2:
                        basis.z=(-3.0f*relative*relative*relative + 3.0f*relative*relative + 3.0f*relative + 1.0f)/6.0f;
                        first.z=(-3.0f*relative*relative + 2.0f*relative + 1.0f) / 2.0f;
                        break;
                    case 3:
                        basis.z=relative*relative*relative/6.0f;
                        first.z=relative*relative/2.0f;
                        break;
                    default:
                        basis.z=0.0f;
                        first.z=0.0f;
                        break;
                }
                for(int pixelY=(int)((y-3)*gridVoxelSpacing.y);pixelY<(int)((y+1)*gridVoxelSpacing.y); pixelY++){
                    if(pixelY>-1 && pixelY<targetSize.y){

                        pre=(int)((float)pixelY/gridVoxelSpacing.y);
                        relative =(float)pixelY/gridVoxelSpacing.y-(float)pre;
                        switch(y-pre){
                            case 0:
                                basis.y=(relative-1.0f)*(relative-1.0f)*(relative-1.0f)/6.0f;
                                first.y=(2.0f*relative - relative*relative - 1.0f) / 2.0f;
                                break;
                            case 1:
                                basis.y=(3.0f*relative*relative*relative - 6.0f*relative*relative + 4.0f)/6.0f;
                                first.y=(3.0f*relative*relative - 4.0f*relative) / 2.0f;
                                break;
                            case 2:
                                basis.y=(-3.0f*relative*relative*relative + 3.0f*relative*relative + 3.0f*relative + 1.0f)/6.0f;
                                first.y=(-3.0f*relative*relative + 2.0f*relative + 1.0f) / 2.0f;
                                break;
                            case 3:
                                basis.y=relative*relative*relative/6.0f;
                                first.y=relative*relative/2.0f;
                                break;
                            default:
                                basis.y=0.0f;
                                first.y=0.0f;
                                break;
                        }
                        for(int pixelX=(int)((x-3)*gridVoxelSpacing.x);pixelX<(int)((x+1)*gridVoxelSpacing.x); pixelX++){
                            if(pixelX>-1 && pixelX<targetSize.x){

                                pre=(int)((float)pixelX/gridVoxelSpacing.x);
                                relative =(float)pixelX/gridVoxelSpacing.x-(float)pre;
                                switch(x-pre){
                                    case 0:
                                        basis.x=(relative-1.0f)*(relative-1.0f)*(relative-1.0f)/6.0f;
                                        first.x=(2.0f*relative - relative*relative - 1.0f) / 2.0f;
                                        break;
                                    case 1:
                                        basis.x=(3.0f*relative*relative*relative - 6.0f*relative*relative + 4.0f)/6.0f;
                                        first.x=(3.0f*relative*relative - 4.0f*relative) / 2.0f;
                                        break;
                                    case 2:
                                        basis.x=(-3.0f*relative*relative*relative + 3.0f*relative*relative + 3.0f*relative + 1.0f)/6.0f;
                                        first.x=(-3.0f*relative*relative + 2.0f*relative + 1.0f) / 2.0f;
                                        break;
                                    case 3:
                                        basis.x=relative*relative*relative/6.0f;
                                        first.x=relative*relative/2.0f;
                                        break;
                                    default:
                                        basis.x=0.0f;
                                        first.x=0.0f;
                                        break;
                                }
                                basisValues.x = first.x * basis.y * basis.z;
                                basisValues.y = basis.x * first.y * basis.z;
                                basisValues.z = basis.x * basis.y * first.z;

                                int storageIndex =(pixelZ*targetSize.y+pixelY)*targetSize.x+pixelX;
                                float jacDeterminant = tex1Dfetch(jacobianDeterminantTexture,storageIndex);
                                if(jacDeterminant>0) jacDeterminant = 2.f * log(jacDeterminant);
                                storageIndex *= 9;
                                jacobianConstraint.x += jacDeterminant
                                    * (tex1Dfetch(jacobianMatricesTexture,storageIndex++)*basisValues.x
                                    +  tex1Dfetch(jacobianMatricesTexture,storageIndex++)*basisValues.y
                                    +  tex1Dfetch(jacobianMatricesTexture,storageIndex++)*basisValues.z);
                                jacobianConstraint.y += jacDeterminant
                                    * (tex1Dfetch(jacobianMatricesTexture,storageIndex++)*basisValues.x
                                    +  tex1Dfetch(jacobianMatricesTexture,storageIndex++)*basisValues.y
                                    +  tex1Dfetch(jacobianMatricesTexture,storageIndex++)*basisValues.z);
                                jacobianConstraint.z += jacDeterminant
                                    * (tex1Dfetch(jacobianMatricesTexture,storageIndex++)*basisValues.x
                                    +  tex1Dfetch(jacobianMatricesTexture,storageIndex++)*basisValues.y
                                    +  tex1Dfetch(jacobianMatricesTexture,storageIndex)*basisValues.z);
                            }
                        }
                    }
                }
            }
        }
        gradient[tid] = gradient[tid] + make_float4(c_Weight
                                                    * (c_AffineMatrix0.x *jacobianConstraint.x
                                                    + c_AffineMatrix0.y *jacobianConstraint.y
                                                    + c_AffineMatrix0.z *jacobianConstraint.z),
                                                    c_Weight
                                                    * (c_AffineMatrix1.x *jacobianConstraint.x
                                                    + c_AffineMatrix1.y *jacobianConstraint.y
                                                    + c_AffineMatrix1.z *jacobianConstraint.z),
                                                    c_Weight
                                                    * (c_AffineMatrix2.x *jacobianConstraint.x
                                                    + c_AffineMatrix2.y *jacobianConstraint.y
                                                    + c_AffineMatrix2.z *jacobianConstraint.z),
                                                    0.0f);
    }
}

/* *************************************************************** */
/* *************************************************************** */

__global__ void reg_bspline_ApproxJacobianGradient_kernel(float4 *gradient)
{
    const int tid= blockIdx.x*blockDim.x + threadIdx.x;
    if(tid<c_ControlPointNumber){

        int3 gridSize = c_ControlPointImageDim;

        int tempIndex=tid;
        const int z =(int)(tempIndex/(gridSize.x*gridSize.y));
        tempIndex -= z*(gridSize.x)*(gridSize.y);
        const int y =(int)(tempIndex/(gridSize.x));
        const int x = tempIndex - y*(gridSize.x) ;

        float3 jacobianConstraint=make_float3(0.0f,0.0f,0.0f);
        float3 basisValues;
        float3 basis;
        float3 first;

        // Loop over all the control points in the surrounding area
        for(int pixelZ=(z-1);pixelZ<(z+2); pixelZ++){
            if(pixelZ>0 && pixelZ<gridSize.z-1){

                switch(pixelZ-z){
                    case -1:
                        basis.z=0.1666667f;
                        first.z=0.5f;
                        break;
                    case 0:
                        basis.z=0.6666667f;
                        first.z=0.0f;
                        break;
                    case 1:
                        basis.z=0.1666667f;
                        first.z=-0.5f;
                        break;
                    default:
                        basis.z=0.0f;
                        first.z=0.0f;
                        break;
                }
                for(int pixelY=(y-1);pixelY<(y+2); pixelY++){
                    if(pixelY>0 && pixelY<gridSize.y-1){

                        switch(pixelY-y){
                            case -1:
                                basis.y=0.1666667f;
                                first.y=0.5f;
                                break;
                            case 0:
                                basis.y=0.6666667f;
                                first.y=0.0f;
                                break;
                            case 1:
                                basis.y=0.1666667f;
                                first.y=-0.5f;
                                break;
                            default:
                                basis.y=0.0f;
                                first.y=0.0f;
                                break;
                        }
                        for(int pixelX=(x-1);pixelX<(x+2); pixelX++){
                            if(pixelX>0 && pixelX<gridSize.x-1){

                                switch(pixelX-x){
                                    case -1:
                                        basis.x=0.1666667f;
                                        first.x=0.5f;
                                        break;
                                    case 0:
                                        basis.x=0.6666667f;
                                        first.x=0.0f;
                                        break;
                                    case 1:
                                        basis.x=0.1666667f;
                                        first.x=-0.5f;
                                        break;
                                    default:
                                        basis.x=0.0f;
                                        first.x=0.0f;
                                        break;
                                }
                                basisValues.x = first.x * basis.y * basis.z;
                                basisValues.y = basis.x * first.y * basis.z;
                                basisValues.z = basis.x * basis.y * first.z;

                                int storageIndex = ((pixelZ-1)*(gridSize.y-2)+pixelY-1)*(gridSize.x-2)+pixelX-1;
                                float jacDeterminant = tex1Dfetch(jacobianDeterminantTexture,storageIndex);
                                if(jacDeterminant>0) jacDeterminant = 2.f * log(jacDeterminant);
                                storageIndex *= 9;
                                jacobianConstraint.x += jacDeterminant
                                    * (tex1Dfetch(jacobianMatricesTexture,storageIndex)*basisValues.x
                                    +  tex1Dfetch(jacobianMatricesTexture,storageIndex+1)*basisValues.y
                                    +  tex1Dfetch(jacobianMatricesTexture,storageIndex+2)*basisValues.z);
                                jacobianConstraint.y += jacDeterminant
                                    * (tex1Dfetch(jacobianMatricesTexture,storageIndex+3)*basisValues.x
                                    +  tex1Dfetch(jacobianMatricesTexture,storageIndex+4)*basisValues.y
                                    +  tex1Dfetch(jacobianMatricesTexture,storageIndex+5)*basisValues.z);
                                jacobianConstraint.z += jacDeterminant
                                    * (tex1Dfetch(jacobianMatricesTexture,storageIndex+6)*basisValues.x
                                    +  tex1Dfetch(jacobianMatricesTexture,storageIndex+7)*basisValues.y
                                    +  tex1Dfetch(jacobianMatricesTexture,storageIndex+8)*basisValues.z);
                            }
                        }
                    }
                }
            }
        }
        gradient[tid] = gradient[tid]
            + make_float4(c_Weight
            * (c_AffineMatrix0.x *jacobianConstraint.x
            + c_AffineMatrix0.y *jacobianConstraint.y
            + c_AffineMatrix0.z *jacobianConstraint.z),
            c_Weight
            * (c_AffineMatrix1.x *jacobianConstraint.x
            + c_AffineMatrix1.y *jacobianConstraint.y
            + c_AffineMatrix1.z *jacobianConstraint.z),
            c_Weight
            * (c_AffineMatrix2.x *jacobianConstraint.x
            + c_AffineMatrix2.y *jacobianConstraint.y
            + c_AffineMatrix2.z *jacobianConstraint.z),
            0.0f);
    }
}
/* *************************************************************** */
/* *************************************************************** */

__global__ void reg_bspline_CorrectFolding_kernel(float4 *controlPointArray)
{
    const int tid= blockIdx.x*blockDim.x + threadIdx.x;
    if(tid<c_ControlPointNumber){

        int3 gridSize = c_ControlPointImageDim;

        int tempIndex=tid;
        const int z =(int)(tempIndex/(gridSize.x*gridSize.y));
        tempIndex -= z*(gridSize.x)*(gridSize.y);
        const int y =(int)(tempIndex/(gridSize.x));
        const int x = tempIndex - y*(gridSize.x) ;

        float3 jacobianConstraint=make_float3(0.0f,0.0f,0.0f);
        float3 basisValues;
        float3 basis;
        float3 first;
        float relative;
        int pre;

        int3 targetSize = c_TargetImageDim;
        float3 gridVoxelSpacing = c_ControlPointVoxelSpacing;

        // Loop over all the control points in the surrounding area
        for(int pixelZ=(int)((z-2)*gridVoxelSpacing.z);pixelZ<(int)((z)*gridVoxelSpacing.z); pixelZ++){
            if(pixelZ>-1 && pixelZ<targetSize.z){

                pre=(int)((float)pixelZ/gridVoxelSpacing.z);
                relative =(float)pixelZ/gridVoxelSpacing.z-(float)pre;
                switch(z-pre){
                    case 0:
                        basis.z=(relative-1.0f)*(relative-1.0f)*(relative-1.0f)/6.0f;
                        first.z=(2.0f*relative - relative*relative - 1.0f) / 2.0f;
                        break;
                    case 1:
                        basis.z=(3.0f*relative*relative*relative - 6.0f*relative*relative + 4.0f)/6.0f;
                        first.z=(3.0f*relative*relative - 4.0f*relative) / 2.0f;
                        break;
                    case 2:
                        basis.z=(-3.0f*relative*relative*relative + 3.0f*relative*relative + 3.0f*relative + 1.0f)/6.0f;
                        first.z=(-3.0f*relative*relative + 2.0f*relative + 1.0f) / 2.0f;
                        break;
                    case 3:
                        basis.z=relative*relative*relative/6.0f;
                        first.z=relative*relative/2.0f;
                        break;
                    default:
                        basis.z=0.0f;
                        first.z=0.0f;
                        break;
                }
                for(int pixelY=(int)((y-2)*gridVoxelSpacing.y);pixelY<(int)((y)*gridVoxelSpacing.y); pixelY++){
                    if(pixelY>-1 && pixelY<targetSize.y){

                        pre=(int)((float)pixelY/gridVoxelSpacing.y);
                        relative =(float)pixelY/gridVoxelSpacing.y-(float)pre;
                        switch(y-pre){
                            case 0:
                                basis.y=(relative-1.0f)*(relative-1.0f)*(relative-1.0f)/6.0f;
                                first.y=(2.0f*relative - relative*relative - 1.0f) / 2.0f;
                                break;
                            case 1:
                                basis.y=(3.0f*relative*relative*relative - 6.0f*relative*relative + 4.0f)/6.0f;
                                first.y=(3.0f*relative*relative - 4.0f*relative) / 2.0f;
                                break;
                            case 2:
                                basis.y=(-3.0f*relative*relative*relative + 3.0f*relative*relative + 3.0f*relative + 1.0f)/6.0f;
                                first.y=(-3.0f*relative*relative + 2.0f*relative + 1.0f) / 2.0f;
                                break;
                            case 3:
                                basis.y=relative*relative*relative/6.0f;
                                first.y=relative*relative/2.0f;
                                break;
                            default:
                                basis.y=0.0f;
                                first.y=0.0f;
                                break;
                        }
                        for(int pixelX=(int)((x-2)*gridVoxelSpacing.x);pixelX<(int)((x)*gridVoxelSpacing.x); pixelX++){
                            if(pixelX>-1 && pixelX<targetSize.x){

                                pre=(int)((float)pixelX/gridVoxelSpacing.x);
                                relative =(float)pixelX/gridVoxelSpacing.x-(float)pre;
                                switch(x-pre){
                                    case 0:
                                        basis.x=(relative-1.0f)*(relative-1.0f)*(relative-1.0f)/6.0f;
                                        first.x=(2.0f*relative - relative*relative - 1.0f) / 2.0f;
                                        break;
                                    case 1:
                                        basis.x=(3.0f*relative*relative*relative - 6.0f*relative*relative + 4.0f)/6.0f;
                                        first.x=(3.0f*relative*relative - 4.0f*relative) / 2.0f;
                                        break;
                                    case 2:
                                        basis.x=(-3.0f*relative*relative*relative + 3.0f*relative*relative + 3.0f*relative + 1.0f)/6.0f;
                                        first.x=(-3.0f*relative*relative + 2.0f*relative + 1.0f) / 2.0f;
                                        break;
                                    case 3:
                                        basis.x=relative*relative*relative/6.0f;
                                        first.x=relative*relative/2.0f;
                                        break;
                                    default:
                                        basis.x=0.0f;
                                        first.x=0.0f;
                                        break;
                                }
                                basisValues.x = first.x * basis.y * basis.z;
                                basisValues.y = basis.x * first.y * basis.z;
                                basisValues.z = basis.x * basis.y * first.z;

                                int storageIndex =(pixelZ*targetSize.y+pixelY)*targetSize.x+pixelX;
                                float jacDeterminant = tex1Dfetch(jacobianDeterminantTexture,storageIndex);
                                if(jacDeterminant<=0){
                                    storageIndex *= 9;
                                    jacobianConstraint.x += jacDeterminant
                                        * (tex1Dfetch(jacobianMatricesTexture,storageIndex++)*basisValues.x
                                        +  tex1Dfetch(jacobianMatricesTexture,storageIndex++)*basisValues.y
                                        +  tex1Dfetch(jacobianMatricesTexture,storageIndex++)*basisValues.z);
                                    jacobianConstraint.y += jacDeterminant
                                        * (tex1Dfetch(jacobianMatricesTexture,storageIndex++)*basisValues.x
                                        +  tex1Dfetch(jacobianMatricesTexture,storageIndex++)*basisValues.y
                                        +  tex1Dfetch(jacobianMatricesTexture,storageIndex++)*basisValues.z);
                                    jacobianConstraint.z += jacDeterminant
                                        * (tex1Dfetch(jacobianMatricesTexture,storageIndex++)*basisValues.x
                                        +  tex1Dfetch(jacobianMatricesTexture,storageIndex++)*basisValues.y
                                        +  tex1Dfetch(jacobianMatricesTexture,storageIndex)*basisValues.z);
                                }
                            }
                        }
                    }
                }
            }
        }
        float norm = sqrt(  jacobianConstraint.x*jacobianConstraint.x
                          + jacobianConstraint.y*jacobianConstraint.y
                          + jacobianConstraint.z*jacobianConstraint.z);
        if(norm>0.f){
            controlPointArray[tid] = controlPointArray[tid] + make_float4(
                                                            (c_ControlPointSpacing.x
                                                            * c_AffineMatrix0.x * jacobianConstraint.x
                                                            + c_AffineMatrix0.y * jacobianConstraint.y
                                                            + c_AffineMatrix0.z * jacobianConstraint.z)
                                                            / norm,
                                                            (c_ControlPointSpacing.y
                                                            * c_AffineMatrix1.x * jacobianConstraint.x
                                                            + c_AffineMatrix1.y * jacobianConstraint.y
                                                            + c_AffineMatrix1.z * jacobianConstraint.z)
                                                            / norm,
                                                            (c_ControlPointSpacing.z
                                                            * c_AffineMatrix2.x * jacobianConstraint.x
                                                            + c_AffineMatrix2.y * jacobianConstraint.y
                                                            + c_AffineMatrix2.z * jacobianConstraint.z)
                                                            / norm,
                                                            0.0f);
        }
    }
}

/* *************************************************************** */
/* *************************************************************** */

__global__ void reg_bspline_ApproxCorrectFolding_kernel(float4 *controlPointArray)
{
    const int tid= blockIdx.x*blockDim.x + threadIdx.x;
    if(tid<c_ControlPointNumber){

        int3 gridSize = c_ControlPointImageDim;

        int tempIndex=tid;
        const int z =(int)(tempIndex/(gridSize.x*gridSize.y));
        tempIndex -= z*(gridSize.x)*(gridSize.y);
        const int y =(int)(tempIndex/(gridSize.x));
        const int x = tempIndex - y*(gridSize.x) ;

        float3 jacobianConstraint=make_float3(0.0f,0.0f,0.0f);
        float3 basisValues;
        float3 basis;
        float3 first;

        // Loop over all the control points in the surrounding area
        for(int pixelZ=(z-1);pixelZ<(z+2); pixelZ++){
            if(pixelZ>0 && pixelZ<gridSize.z-1){

                switch(pixelZ-z){
                    case -1:
                        basis.z=0.1666667f;
                        first.z=0.5f;
                        break;
                    case 0:
                        basis.z=0.6666667f;
                        first.z=0.0f;
                        break;
                    case 1:
                        basis.z=0.1666667f;
                        first.z=-0.5f;
                        break;
                    default:
                        basis.z=0.0f;
                        first.z=0.0f;
                        break;
                }
                for(int pixelY=(y-1);pixelY<(y+2); pixelY++){
                    if(pixelY>0 && pixelY<gridSize.y-1){

                        switch(pixelY-y){
                            case -1:
                                basis.y=0.1666667f;
                                first.y=0.5f;
                                break;
                            case 0:
                                basis.y=0.6666667f;
                                first.y=0.0f;
                                break;
                            case 1:
                                basis.y=0.1666667f;
                                first.y=-0.5f;
                                break;
                            default:
                                basis.y=0.0f;
                                first.y=0.0f;
                                break;
                        }
                        for(int pixelX=(x-1);pixelX<(x+2); pixelX++){
                            if(pixelX>0 && pixelX<gridSize.x-1){

                                switch(pixelX-x){
                                    case -1:
                                        basis.x=0.1666667f;
                                        first.x=0.5f;
                                        break;
                                    case 0:
                                        basis.x=0.6666667f;
                                        first.x=0.0f;
                                        break;
                                    case 1:
                                        basis.x=0.1666667f;
                                        first.x=-0.5f;
                                        break;
                                    default:
                                        basis.x=0.0f;
                                        first.x=0.0f;
                                        break;
                                }
                                basisValues.x = first.x * basis.y * basis.z;
                                basisValues.y = basis.x * first.y * basis.z;
                                basisValues.z = basis.x * basis.y * first.z;

                                int storageIndex = ((pixelZ-1)*(gridSize.y-2)+pixelY-1)*(gridSize.x-2)+pixelX-1;
                                float jacDeterminant = tex1Dfetch(jacobianDeterminantTexture,storageIndex);
                                if(jacDeterminant<=0){
                                    storageIndex *= 9;
                                    jacobianConstraint.x += jacDeterminant
                                        * (tex1Dfetch(jacobianMatricesTexture,storageIndex++)*basisValues.x
                                        +  tex1Dfetch(jacobianMatricesTexture,storageIndex++)*basisValues.y
                                        +  tex1Dfetch(jacobianMatricesTexture,storageIndex++)*basisValues.z);
                                    jacobianConstraint.y += jacDeterminant
                                        * (tex1Dfetch(jacobianMatricesTexture,storageIndex++)*basisValues.x
                                        +  tex1Dfetch(jacobianMatricesTexture,storageIndex++)*basisValues.y
                                        +  tex1Dfetch(jacobianMatricesTexture,storageIndex+5)*basisValues.z);
                                    jacobianConstraint.z += jacDeterminant
                                        * (tex1Dfetch(jacobianMatricesTexture,storageIndex++)*basisValues.x
                                        +  tex1Dfetch(jacobianMatricesTexture,storageIndex++)*basisValues.y
                                        +  tex1Dfetch(jacobianMatricesTexture,storageIndex++)*basisValues.z);
                                }
                            }
                        }
                    }
                }
            }
        }
        float norm = sqrt(  jacobianConstraint.x*jacobianConstraint.x
                          + jacobianConstraint.y*jacobianConstraint.y
                          + jacobianConstraint.z*jacobianConstraint.z);
        if(norm>0){
            controlPointArray[tid] = controlPointArray[tid] + make_float4(
                                                            (c_ControlPointSpacing.x
                                                            * c_AffineMatrix0.x *jacobianConstraint.x
                                                            + c_AffineMatrix0.y *jacobianConstraint.y
                                                            + c_AffineMatrix0.z *jacobianConstraint.z)
                                                            / norm,
                                                            (c_ControlPointSpacing.y
                                                            * c_AffineMatrix1.x *jacobianConstraint.x
                                                            + c_AffineMatrix1.y *jacobianConstraint.y
                                                            + c_AffineMatrix1.z *jacobianConstraint.z)
                                                            / norm,
                                                            (c_ControlPointSpacing.z
                                                            * c_AffineMatrix2.x *jacobianConstraint.x
                                                            + c_AffineMatrix2.y *jacobianConstraint.y
                                                            + c_AffineMatrix2.z *jacobianConstraint.z)
                                                            / norm,
                                                            0.0f);
        }
    }
}

/* *************************************************************** */
/* *************************************************************** */

__global__ void reg_bspline_storeApproxBendingEnergy_kernel(float3 *beValues)
{
	const int tid= blockIdx.x*blockDim.x + threadIdx.x;
	if(tid<c_ControlPointNumber){

		int3 gridSize = c_ControlPointImageDim;

		int tempIndex=tid;
		const short z =(int)(tempIndex/(gridSize.x*gridSize.y));
		tempIndex -= z*(gridSize.x)*(gridSize.y);
		const short y =(int)(tempIndex/(gridSize.x));
		const short x = tempIndex - y*(gridSize.x);

		if(	0<x && x<gridSize.x-1 &&
			0<y && y<gridSize.y-1 &&
			0<z && z<gridSize.z-1){

			float3 XX = make_float3(0.0f,0.0f,0.0f);
			float3 YY = make_float3(0.0f,0.0f,0.0f);
			float3 ZZ = make_float3(0.0f,0.0f,0.0f);
			float3 XY = make_float3(0.0f,0.0f,0.0f);
			float3 YZ = make_float3(0.0f,0.0f,0.0f);
			float3 XZ = make_float3(0.0f,0.0f,0.0f);

			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,0.027778f,0.027778f,0.027778f,0.041667f,0.041667f,0.041667f,((z-1)*gridSize.y+y-1)*gridSize.x+x-1);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,-0.055556f,0.111111f,0.111111f,-0.000000f,0.166667f,-0.000000f,((z-1)*gridSize.y+y-1)*gridSize.x+x);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,0.027778f,0.027778f,0.027778f,-0.041667f,0.041667f,-0.041667f,((z-1)*gridSize.y+y-1)*gridSize.x+x+1);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,0.111111f,-0.055556f,0.111111f,-0.000000f,-0.000000f,0.166667f,((z-1)*gridSize.y+y)*gridSize.x+x-1);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,-0.222222f,-0.222222f,0.444444f,0.000000f,-0.000000f,-0.000000f,((z-1)*gridSize.y+y)*gridSize.x+x);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,0.111111f,-0.055556f,0.111111f,0.000000f,-0.000000f,-0.166667f,((z-1)*gridSize.y+y)*gridSize.x+x+1);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,0.027778f,0.027778f,0.027778f,-0.041667f,-0.041667f,0.041667f,((z-1)*gridSize.y+y+1)*gridSize.x+x-1);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,-0.055556f,0.111111f,0.111111f,0.000000f,-0.166667f,-0.000000f,((z-1)*gridSize.y+y+1)*gridSize.x+x);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,0.027778f,0.027778f,0.027778f,0.041667f,-0.041667f,-0.041667f,((z-1)*gridSize.y+y+1)*gridSize.x+x+1);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,0.111111f,0.111111f,-0.055556f,0.166667f,-0.000000f,-0.000000f,((z)*gridSize.y+y-1)*gridSize.x+x-1);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,-0.222222f,0.444444f,-0.222222f,-0.000000f,-0.000000f,0.000000f,((z)*gridSize.y+y-1)*gridSize.x+x);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,0.111111f,0.111111f,-0.055556f,-0.166667f,-0.000000f,0.000000f,((z)*gridSize.y+y-1)*gridSize.x+x+1);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,0.444444f,-0.222222f,-0.222222f,-0.000000f,0.000000f,-0.000000f,((z)*gridSize.y+y)*gridSize.x+x-1);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,-0.888889f,-0.888889f,-0.888889f,0.000000f,0.000000f,0.000000f,((z)*gridSize.y+y)*gridSize.x+x);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,0.444444f,-0.222222f,-0.222222f,0.000000f,0.000000f,0.000000f,((z)*gridSize.y+y)*gridSize.x+x+1);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,0.111111f,0.111111f,-0.055556f,-0.166667f,0.000000f,-0.000000f,((z)*gridSize.y+y+1)*gridSize.x+x-1);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,-0.222222f,0.444444f,-0.222222f,0.000000f,0.000000f,0.000000f,((z)*gridSize.y+y+1)*gridSize.x+x);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,0.111111f,0.111111f,-0.055556f,0.166667f,0.000000f,0.000000f,((z)*gridSize.y+y+1)*gridSize.x+x+1);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,0.027778f,0.027778f,0.027778f,0.041667f,-0.041667f,-0.041667f,((z+1)*gridSize.y+y-1)*gridSize.x+x-1);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,-0.055556f,0.111111f,0.111111f,-0.000000f,-0.166667f,0.000000f,((z+1)*gridSize.y+y-1)*gridSize.x+x);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,0.027778f,0.027778f,0.027778f,-0.041667f,-0.041667f,0.041667f,((z+1)*gridSize.y+y-1)*gridSize.x+x+1);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,0.111111f,-0.055556f,0.111111f,-0.000000f,0.000000f,-0.166667f,((z+1)*gridSize.y+y)*gridSize.x+x-1);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,-0.222222f,-0.222222f,0.444444f,0.000000f,0.000000f,0.000000f,((z+1)*gridSize.y+y)*gridSize.x+x);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,0.111111f,-0.055556f,0.111111f,0.000000f,0.000000f,0.166667f,((z+1)*gridSize.y+y)*gridSize.x+x+1);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,0.027778f,0.027778f,0.027778f,-0.041667f,0.041667f,-0.041667f,((z+1)*gridSize.y+y+1)*gridSize.x+x-1);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,-0.055556f,0.111111f,0.111111f,0.000000f,0.166667f,0.000000f,((z+1)*gridSize.y+y+1)*gridSize.x+x);
			bendingEnergyMult(&XX,&YY,&ZZ,&XY,&YZ,&XZ,0.027778f,0.027778f,0.027778f,0.041667f,0.041667f,0.041667f,((z+1)*gridSize.y+y+1)*gridSize.x+x+1);

			int index=6*tid;
			beValues[index++]=2.0f*XX;
			beValues[index++]=2.0f*YY;
			beValues[index++]=2.0f*ZZ;
			beValues[index++]=4.0f*XY;
			beValues[index++]=4.0f*YZ;
			beValues[index]=4.0f*XZ;
		}
	}
	return;
}

/* *************************************************************** */
/* *************************************************************** */

__global__ void reg_bspline_getApproxBendingEnergyGradient_kernel(  float3 *bendingEnergyValue,
                                                                    float4 *nodeNMIGradientArray_d)
{
    __shared__ float basisXX[27];
    __shared__ float basisYY[27];
    __shared__ float basisZZ[27];
    __shared__ float basisXY[27];
    __shared__ float basisYZ[27];
    __shared__ float basisXZ[27];

    if(threadIdx.x<27){
        float4 tempA = tex1Dfetch(basisValueATexture,threadIdx.x);
        basisXX[threadIdx.x] = tempA.x;
        basisYY[threadIdx.x] = tempA.y;
        basisZZ[threadIdx.x] = tempA.z;
        basisXY[threadIdx.x] = tempA.w;
        float2 tempB = tex1Dfetch(basisValueBTexture,threadIdx.x);
        basisYZ[threadIdx.x] = tempB.x;
        basisXZ[threadIdx.x] = tempB.y;
    }
    __syncthreads();

	const int tid= blockIdx.x*blockDim.x + threadIdx.x;
	if(tid<c_ControlPointNumber){

		int3 gridSize = c_ControlPointImageDim;

		int tempIndex=tid;
		const short z =(int)(tempIndex/(gridSize.x*gridSize.y));
		tempIndex -= z*(gridSize.x)*(gridSize.y);
		const short y =(int)(tempIndex/(gridSize.x));
		const short x = tempIndex - y*(gridSize.x);

		float3 gradientValue=make_float3(0.0f,0.0f,0.0f);

		float3 *bendingEnergyValuePtr;
		float3 bendingEnergyCurrentValue;

		short coord=0;
		for(short Z=z-1; Z<z+2; Z++){
			for(short Y=y-1; Y<y+2; Y++){
				for(short X=x-1; X<x+2; X++){
					if(-1<X && -1<Y && -1<Z && X<gridSize.x && Y<gridSize.y && Z<gridSize.z){
						bendingEnergyValuePtr = &bendingEnergyValue[6 * ((Z*gridSize.y + Y)*gridSize.x + X)];

						bendingEnergyCurrentValue =  *bendingEnergyValuePtr++;
						gradientValue.x += bendingEnergyCurrentValue.x * basisXX[coord];
						gradientValue.y += bendingEnergyCurrentValue.y * basisXX[coord];
						gradientValue.z += bendingEnergyCurrentValue.z * basisXX[coord];
			
						bendingEnergyCurrentValue =  *bendingEnergyValuePtr++;
						gradientValue.x += bendingEnergyCurrentValue.x * basisYY[coord];
						gradientValue.y += bendingEnergyCurrentValue.y * basisYY[coord];
						gradientValue.z += bendingEnergyCurrentValue.z * basisYY[coord];
			
						bendingEnergyCurrentValue =  *bendingEnergyValuePtr++;
						gradientValue.x += bendingEnergyCurrentValue.x * basisZZ[coord];
						gradientValue.y += bendingEnergyCurrentValue.y * basisZZ[coord];
						gradientValue.z += bendingEnergyCurrentValue.z * basisZZ[coord];
			
						bendingEnergyCurrentValue =  *bendingEnergyValuePtr++;
						gradientValue.x += bendingEnergyCurrentValue.x * basisXY[coord];
						gradientValue.y += bendingEnergyCurrentValue.y * basisXY[coord];
						gradientValue.z += bendingEnergyCurrentValue.z * basisXY[coord];
			
						bendingEnergyCurrentValue =  *bendingEnergyValuePtr++;
						gradientValue.x += bendingEnergyCurrentValue.x * basisYZ[coord];
						gradientValue.y += bendingEnergyCurrentValue.y * basisYZ[coord];
						gradientValue.z += bendingEnergyCurrentValue.z * basisYZ[coord];
			
						bendingEnergyCurrentValue =  *bendingEnergyValuePtr;
						gradientValue.x += bendingEnergyCurrentValue.x * basisXZ[coord];
						gradientValue.y += bendingEnergyCurrentValue.y * basisXZ[coord];
						gradientValue.z += bendingEnergyCurrentValue.z * basisXZ[coord];
					}
					coord++;
				}
			}
		}
		float4 metricGradientValue;
		metricGradientValue = nodeNMIGradientArray_d[tid];
		float weight = c_Weight;
		// (Marc) I removed the normalisation by the voxel number as each gradient has to be normalised in the same way
		metricGradientValue.x += weight*gradientValue.x;
		metricGradientValue.y += weight*gradientValue.y;
		metricGradientValue.z += weight*gradientValue.z;
		nodeNMIGradientArray_d[tid]=metricGradientValue;
	}
}

/* *************************************************************** */
/* *************************************************************** */

__global__ void reg_spline_cppComposition_kernel(float4 *toUpdateArray)
{
    const int tid= blockIdx.x*blockDim.x + threadIdx.x;
    if(tid<c_ControlPointNumber){

        int3 controlPointImageDim = c_ControlPointImageDim;

        // The current position is extracted
        float4 matrix;
        float3 voxel;
        float4 position=toUpdateArray[tid];
        if(c_Type==0){ // input node contains displacement
            int tempIndex=tid;
            voxel.z =(int)(tempIndex/(controlPointImageDim.x*controlPointImageDim.y));
            tempIndex -= (int)voxel.z*(controlPointImageDim.x)*(controlPointImageDim.y);
            voxel.y =(int)(tempIndex/(controlPointImageDim.x));
            voxel.x = tempIndex - (int)voxel.y*(controlPointImageDim.x);

            matrix = tex1Dfetch(txVoxelToRealMatrix,0);
            position.x +=   matrix.x*voxel.x + matrix.y*voxel.y  +
                            matrix.z*voxel.z  +  matrix.w;
            matrix = tex1Dfetch(txVoxelToRealMatrix,1);
            position.y +=   matrix.x*voxel.x + matrix.y*voxel.y  +
                            matrix.z*voxel.z  +  matrix.w;
            matrix = tex1Dfetch(txVoxelToRealMatrix,2);
            position.z +=   matrix.x*voxel.x + matrix.y*voxel.y  +
                            matrix.z*voxel.z  +  matrix.w;
        }

        // The voxel position is computed
        matrix = tex1Dfetch(txRealToVoxelMatrix,0);
        voxel.x =   matrix.x*position.x + matrix.y*position.y  +
                    matrix.z*position.z  +  matrix.w;
        matrix = tex1Dfetch(txRealToVoxelMatrix,1);
        voxel.y =   matrix.x*position.x + matrix.y*position.y  +
                    matrix.z*position.z  +  matrix.w;
        matrix = tex1Dfetch(txRealToVoxelMatrix,2);
        voxel.z =   matrix.x*position.x + matrix.y*position.y  +
                    matrix.z*position.z  +  matrix.w;

        // the "nearest previous" node is determined [0,0,0]
        int3 nodeAnte;
        nodeAnte.x = (int)floorf(voxel.x);
        nodeAnte.y = (int)floorf(voxel.y);
        nodeAnte.z = (int)floorf(voxel.z);

        float relative = fabsf(voxel.x - (float)nodeAnte.x);
        float xBasis[4];
        xBasis[3]= relative * relative * relative / 6.0f;
        xBasis[0]= 1.0f/6.0f + relative*(relative-1.0f)/2.0f - xBasis[3];
        xBasis[2]= relative + xBasis[0] - 2.0f*xBasis[3];
        xBasis[1]= 1.0f - xBasis[0] - xBasis[2] - xBasis[3];

        relative = fabsf((float)voxel.y-(float)nodeAnte.y);
        float yBasis[4];
        yBasis[3]= relative * relative * relative / 6.0f;
        yBasis[0]= 1.0f/6.0f + relative*(relative-1.0f)/2.0f - yBasis[3];
        yBasis[2]= relative + yBasis[0] - 2.0f*yBasis[3];
        yBasis[1]= 1.0f - yBasis[0] - yBasis[2] - yBasis[3];

        relative = fabsf((float)voxel.z-(float)nodeAnte.z);
        float zBasis[4];
        zBasis[3]= relative * relative * relative / 6.0f;
        zBasis[0]= 1.0f/6.0f + relative*(relative-1.0f)/2.0f - zBasis[3];
        zBasis[2]= relative + zBasis[0] - 2.0f*zBasis[3];
        zBasis[1]= 1.0f - zBasis[0] - zBasis[2] - zBasis[3];

        float4 displacement=make_float4(0.0f,0.0f,0.0f,0.0f);

        nodeAnte.x--;
        nodeAnte.y--;
        nodeAnte.z--;

        int indexYZ, indexXYZ;
        for(short c=0; c<4; c++){
            float4 tempValueY=make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            indexYZ= ( (nodeAnte.z + c) * controlPointImageDim.y + nodeAnte.y ) * controlPointImageDim.x;
            for(short b=0; b<4; b++){
                float4 tempValueX=make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                indexXYZ= indexYZ + nodeAnte.x;
                for(short a=0; a<4; a++){
                    float4 nodeCoefficient = tex1Dfetch(controlPointTexture, indexXYZ);
                    tempValueX.x +=  nodeCoefficient.x* xBasis[a];
                    tempValueX.y +=  nodeCoefficient.y* xBasis[a];
                    tempValueX.z +=  nodeCoefficient.z* xBasis[a];
                    indexXYZ++;
                }
                tempValueY.x += tempValueX.x * yBasis[b];
                tempValueY.y += tempValueX.y * yBasis[b];
                tempValueY.z += tempValueX.z * yBasis[b];
                indexYZ += controlPointImageDim.x;
            }
            displacement.x += tempValueY.x * zBasis[c];
            displacement.y += tempValueY.y * zBasis[c];
            displacement.z += tempValueY.z * zBasis[c];
        }
        toUpdateArray[tid] = toUpdateArray[tid] + c_Weight * displacement;
    }
    return;
}

/* *************************************************************** */
/* *************************************************************** */
__global__ void reg_spline_cppDeconvolve_kernel(   float3 *temporaryGridImage_d,
                                                    float4 *outputControlPointArray_d,
                                                    int axis)
{

    const int tid= blockIdx.x*blockDim.x + threadIdx.x;
    const int3 gridSize = c_ControlPointImageDim;

    int controlPointNumber;
    int start;
    int last;
    int number;
    int increment;
    switch(axis){
        case 0: // X axis is assumed
            start = tid * gridSize.x;
            last = start + gridSize.x;
            number=gridSize.x;
            increment=1;
            controlPointNumber = gridSize.y * gridSize.z;
            break;
        case 1: // y axis is assumed
            start = tid + tid/gridSize.x * gridSize.x * (gridSize.y - 1);
            last = start + gridSize.x*gridSize.y;
            number=gridSize.y;
            increment=gridSize.x;
            controlPointNumber = gridSize.x* gridSize.z;
            break;
        case 2: // X axis is assumed
            start = tid;
            last = start + gridSize.x*gridSize.y*gridSize.z;
            number=gridSize.z;
            increment=gridSize.x*gridSize.y;
            controlPointNumber = gridSize.x * gridSize.y;
            break;
    }
    if(tid<controlPointNumber){
        const int newStart = tid * number;
        const float pole = -0.26794919f;//sqrt(3.0) - 2.0

        unsigned int coord=newStart;
        for(unsigned int i=start; i<last; i+=increment){
            memcpy(&temporaryGridImage_d[coord++], &outputControlPointArray_d[i], sizeof(float3));
        }

        float3 currentPole=make_float3(pole,pole,pole);
        float3 currentOpposite = make_float3(pow(pole,(2.f*(float)(number)-1.f)),0.f,0.f);
        currentOpposite.z=currentOpposite.y=currentOpposite.x;
        float3 sum=make_float3(0.f,0.f,0.f);
        for(int i=newStart+1; i<newStart+number; i++){
            sum = sum + (currentPole - currentOpposite) * temporaryGridImage_d[i];
            currentPole = pole *currentPole;
            currentOpposite = currentOpposite / pole;
        }

        temporaryGridImage_d[newStart] = (temporaryGridImage_d[newStart] - pole*pole*(temporaryGridImage_d[newStart] + sum)) / (1.f - pow(pole,(2.f*number+2.f)));

        //other values forward
        for(int i=newStart+1; i<newStart+number; i++){
            temporaryGridImage_d[i] = temporaryGridImage_d[i] + pole * temporaryGridImage_d[i-1];
        }

        float ipp=(1.f-pole);
        ipp*=ipp;

        //last value
        temporaryGridImage_d[newStart+number-1] = ipp * temporaryGridImage_d[newStart+number-1];

        //other values backward
        for(int i=newStart+number-2; newStart<=i; i--){
            temporaryGridImage_d[i] = pole * temporaryGridImage_d[i+1] + ipp*temporaryGridImage_d[i];
        }

        // Data are transfered back in the original image
        coord=newStart;
        for(int i=start; i<last; i+=increment){
            memcpy(&outputControlPointArray_d[i], &temporaryGridImage_d[coord++], sizeof(float3));
        }
    }
    return;
}
/* *************************************************************** */
/* *************************************************************** */
__global__ void reg_spline_getDeformationFromDisplacement_kernel(float4 *imageArray_d)
{
    const int tid= blockIdx.x*blockDim.x + threadIdx.x;
    if(tid<c_ControlPointNumber){

        // The current displacement is extracted
        float3 position;
        memcpy(&position, &imageArray_d[tid], sizeof(float3));

        // The voxel index is computed
        float3 voxel;
        int tempIndex=tid;
        int3 imageDim = c_ControlPointImageDim;
        voxel.z =tempIndex/(imageDim.x*imageDim.y);
        tempIndex -= (int)voxel.z*(imageDim.x)*(imageDim.y);
        voxel.y = tempIndex/imageDim.x;
        voxel.x = tempIndex - (int)voxel.y*(imageDim.x);

        // The initial voxel position is added to the displacmeent
        float4 matrix = tex1Dfetch(txVoxelToRealMatrix,0);
        position.x +=   matrix.x*voxel.x + matrix.y*voxel.y  +
                        matrix.z*voxel.z  +  matrix.w;
        matrix = tex1Dfetch(txVoxelToRealMatrix,1);
        position.y +=   matrix.x*voxel.x + matrix.y*voxel.y  +
                        matrix.z*voxel.z  +  matrix.w;
        matrix = tex1Dfetch(txVoxelToRealMatrix,2);
        position.z +=   matrix.x*voxel.x + matrix.y*voxel.y  +
                        matrix.z*voxel.z  +  matrix.w;

        imageArray_d[tid] = make_float4(position.x, position.y, position.z, 0.f);
    }
}
/* *************************************************************** */
/* *************************************************************** */
__global__ void reg_bspline_SetJacDetToOne_kernel(float *array)
{
    const int tid= blockIdx.x*blockDim.x + threadIdx.x;
    if(tid<c_VoxelNumber){
        array[tid]=1.f;
    }
}
/* *************************************************************** */

__global__ void reg_bspline_GetSquaredLogJacDet_kernel(float *array)
{
    const int tid= blockIdx.x*blockDim.x + threadIdx.x;
    if(tid<c_VoxelNumber){
        float val = log(array[tid]);
        array[tid]=val*val;
    }
}

/* *************************************************************** */
/* *************************************************************** */

__global__ void reg_bspline_JacobianGradFromVel_kernel(float4 *gradientImageArray_d)
{
    const int tid= blockIdx.x*blockDim.x + threadIdx.x;
    if(tid<c_ControlPointNumber){

        int3 controlPointIndex;
        int3 controlPointImageDim = c_ControlPointImageDim;
        int tempIndex=tid;
        controlPointIndex.z =tempIndex/(controlPointImageDim.x*controlPointImageDim.y);
        tempIndex -= (int)controlPointIndex.z*(controlPointImageDim.x)*(controlPointImageDim.y);
        controlPointIndex.y = tempIndex/controlPointImageDim.x;
        controlPointIndex.x = tempIndex - (int)controlPointIndex.y*(controlPointImageDim.x);

        /* For each control point (thread), I need to loop over every voxel in order to
           know which one are in the area of interest */

        float4 JacobianGradient=make_float4(0.f,0.f,0.f,0.f);

        for(int i=0;i<c_VoxelNumber;i++){
            float4 voxelIndex = tex1Dfetch(voxelDisplacementTexture,i);

            int zPre = (int)floorf((float)voxelIndex.z/(float)c_ControlPointVoxelSpacing.z);
            if(controlPointIndex.z>=zPre && controlPointIndex.z<zPre+4){

                int yPre = (int)floorf((float)voxelIndex.y/(float)c_ControlPointVoxelSpacing.y);
                if(controlPointIndex.y>=yPre && controlPointIndex.y<yPre+4){

                    int xPre = (int)floorf((float)voxelIndex.x/(float)c_ControlPointVoxelSpacing.x);
                    if(controlPointIndex.x>=xPre && controlPointIndex.x<xPre+4){

                        float3 basisValues, firstValues;
                        float basis=(float)voxelIndex.x/c_ControlPointVoxelSpacing.x-(float)xPre;
                        switch(controlPointIndex.x-xPre){
                            case 0:
                                basisValues.x=(basis-1.f)*(basis-1.f)*(basis-1.f)/6.f;
                                firstValues.x=(-basis*basis + 2.f*basis - 1.f) / 2.f;
                                break;
                            case 1:
                                basisValues.x=(3.f*basis*basis*basis - 6.f*basis*basis + 4.f)/6.f;
                                firstValues.x=(3.f*basis*basis - 4.f*basis) / 2.f;
                                break;
                            case 2:
                                basisValues.x=(-3.f*basis*basis*basis + 3.f*basis*basis + 3.f*basis + 1.f)/6.f;
                                firstValues.x=(-3.f*basis*basis + 2.f*basis + 1.f) / 2.f;
                                break;
                            case 3:
                                basisValues.x=basis*basis*basis/6.f;
                                firstValues.x=basis*basis/2.f;
                                break;
                            default:
                                basisValues.x=0.f;
                                firstValues.x=0.f;
                                break;
                        }

                        basis=(float)voxelIndex.y/c_ControlPointVoxelSpacing.y-(float)yPre;
                        switch(controlPointIndex.y-yPre){
                            case 0:
                                basisValues.y=(basis-1.f)*(basis-1.f)*(basis-1.f)/6.f;
                                firstValues.y=(-basis*basis + 2.f*basis - 1.f) / 2.f;
                                break;
                            case 1:
                                basisValues.y=(3.f*basis*basis*basis - 6.f*basis*basis + 4.f)/6.f;
                                firstValues.y=(3.f*basis*basis - 4.f*basis) / 2.f;
                                break;
                            case 2:
                                basisValues.y=(-3.f*basis*basis*basis + 3.f*basis*basis + 3.f*basis + 1.f)/6.f;
                                firstValues.y=(-3.f*basis*basis + 2.f*basis + 1.f) / 2.f;
                                break;
                            case 3:
                                basisValues.y=basis*basis*basis/6.f;
                                firstValues.y=basis*basis/2.f;
                                break;
                            default:
                                basisValues.y=0.f;
                                firstValues.y=0.f;
                                break;
                        }

                        basis=(float)voxelIndex.z/c_ControlPointVoxelSpacing.z-(float)zPre;
                        switch(controlPointIndex.z-zPre){
                            case 0:
                                basisValues.z=(basis-1.f)*(basis-1.f)*(basis-1.f)/6.f;
                                firstValues.z=(-basis*basis + 2.f*basis - 1.f) / 2.f;
                                break;
                            case 1:
                                basisValues.z=(3.f*basis*basis*basis - 6.f*basis*basis + 4.f)/6.f;
                                firstValues.z=(3.f*basis*basis - 4.f*basis) / 2.f;
                                break;
                            case 2:
                                basisValues.z=(-3.f*basis*basis*basis + 3.f*basis*basis + 3.f*basis + 1.f)/6.f;
                                firstValues.z=(-3.f*basis*basis + 2.f*basis + 1.f) / 2.f;
                                break;
                            case 3:
                                basisValues.z=basis*basis*basis/6.f;
                                firstValues.z=basis*basis/2.f;
                                break;
                            default:
                                basisValues.z=0.f;
                                firstValues.z=0.f;
                                break;
                        }

                        float3 currentBasisValues = make_float3(
                            firstValues.x * basisValues.y * basisValues.z,
                            basisValues.x * firstValues.y * basisValues.z,
                            basisValues.x * basisValues.y * firstValues.z);

                        int storageIndex = i;
                        float jacDeterminant = tex1Dfetch(jacobianDeterminantTexture,i);
                        if(jacDeterminant>0) jacDeterminant = 2.f * log(jacDeterminant);
                        storageIndex *= 9;
                        JacobianGradient.x += jacDeterminant
                            * (tex1Dfetch(jacobianMatricesTexture,storageIndex++)*currentBasisValues.x
                            +  tex1Dfetch(jacobianMatricesTexture,storageIndex++)*currentBasisValues.y
                            +  tex1Dfetch(jacobianMatricesTexture,storageIndex++)*currentBasisValues.z);
                        JacobianGradient.y += jacDeterminant
                            * (tex1Dfetch(jacobianMatricesTexture,storageIndex++)*currentBasisValues.x
                            +  tex1Dfetch(jacobianMatricesTexture,storageIndex++)*currentBasisValues.y
                            +  tex1Dfetch(jacobianMatricesTexture,storageIndex++)*currentBasisValues.z);
                        JacobianGradient.z += jacDeterminant
                            * (tex1Dfetch(jacobianMatricesTexture,storageIndex++)*currentBasisValues.x
                            +  tex1Dfetch(jacobianMatricesTexture,storageIndex++)*currentBasisValues.y
                            +  tex1Dfetch(jacobianMatricesTexture,storageIndex)*currentBasisValues.z);

                    } //z in the range
                } //y in the range
            } //x in the range
        } //i

        gradientImageArray_d[tid] = gradientImageArray_d[tid]
                                    + make_float4(c_Weight
                                    * (c_AffineMatrix0.x *JacobianGradient.x
                                    + c_AffineMatrix0.y *JacobianGradient.y
                                    + c_AffineMatrix0.z *JacobianGradient.z),
                                    c_Weight
                                    * (c_AffineMatrix1.x *JacobianGradient.x
                                    + c_AffineMatrix1.y *JacobianGradient.y
                                    + c_AffineMatrix1.z *JacobianGradient.z),
                                    c_Weight
                                    * (c_AffineMatrix2.x *JacobianGradient.x
                                    + c_AffineMatrix2.y *JacobianGradient.y
                                    + c_AffineMatrix2.z *JacobianGradient.z),
                                    0.0f);
    } //tid
}

/* *************************************************************** */
/* *************************************************************** */
__global__ void reg_bspline_PositionToIndices_kernel(float4 *position)
{
    const int tid= blockIdx.x*blockDim.x + threadIdx.x;
    if(tid<c_VoxelNumber){

        float4 voxelPosition = position[tid];
        float3 voxelIndices;

        float4 matrix = tex1Dfetch(txRealToVoxelMatrix,0);
        voxelIndices.x =   matrix.x*voxelPosition.x + matrix.y*voxelPosition.y  +
                        matrix.z*voxelPosition.z  +  matrix.w;

        matrix = tex1Dfetch(txRealToVoxelMatrix,1);
        voxelIndices.y =   matrix.x*voxelPosition.x + matrix.y*voxelPosition.y  +
                        matrix.z*voxelPosition.z  +  matrix.w;

        matrix = tex1Dfetch(txRealToVoxelMatrix,2);
        voxelIndices.z =   matrix.x*voxelPosition.x + matrix.y*voxelPosition.y  +
                        matrix.z*voxelPosition.z  +  matrix.w;

        position[tid] = make_float4(voxelIndices.x,
                                    voxelIndices.y,
                                    voxelIndices.z,
                                    0.f);
    }
}

/* *************************************************************** */
/* *************************************************************** */
#endif

