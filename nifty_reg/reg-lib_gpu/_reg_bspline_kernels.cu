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

__device__ __constant__ int c_VoxelNumber;
__device__ __constant__ int c_ControlPointNumber;
__device__ __constant__ int3 c_TargetImageDim;
__device__ __constant__ int3 c_ControlPointImageDim;
__device__ __constant__ float3 c_ControlPointVoxelSpacing;
__device__ __constant__ float c_BendingEnergyWeight;

texture<float4, 1, cudaReadModeElementType> controlPointTexture;


__device__ float3 operator*(float a, float3 b){
    return make_float3(a*b.x, a*b.y, a*b.z);
}

__global__ void _reg_freeForm_interpolatePosition(float4 *positionField)
{
	const int tid= blockIdx.x*blockDim.x + threadIdx.x;
	if(tid<c_VoxelNumber){

		int3 imageSize = c_TargetImageDim;

		int tempIndex=tid;
		const short z =(int)(tempIndex/(imageSize.x*imageSize.y));
		tempIndex -= z*(imageSize.x)*(imageSize.y);
		const short y =(int)(tempIndex/(imageSize.x));
		const short x = tempIndex - y*(imageSize.x);
	
		// the "nearest previous" node is determined [0,0,0]
		int3 nodeAnte;
		float3 gridVoxelSpacing = c_ControlPointVoxelSpacing;
		nodeAnte.x = (short)floorf((float)x/gridVoxelSpacing.x);
		nodeAnte.y = (short)floorf((float)y/gridVoxelSpacing.y);
		nodeAnte.z = (short)floorf((float)z/gridVoxelSpacing.z);
		
		float relative = fabsf((float)x/gridVoxelSpacing.x - (float)nodeAnte.x);
		float xBasis[4];
		xBasis[3]= relative * relative * relative / 6.0f;
		xBasis[0]= 1.0f/6.0f + relative*(relative-1.0f)/2.0f - xBasis[3];
		xBasis[2]= relative + xBasis[0] - 2.0f*xBasis[3];
		xBasis[1]= 1.0f - xBasis[0] - xBasis[2] - xBasis[3];
		
		relative = fabsf((float)y/gridVoxelSpacing.y-(float)nodeAnte.y);
		float yBasis[4];
		yBasis[3]= relative * relative * relative / 6.0f;
		yBasis[0]= 1.0f/6.0f + relative*(relative-1.0f)/2.0f - yBasis[3];
		yBasis[2]= relative + yBasis[0] - 2.0f*yBasis[3];
		yBasis[1]= 1.0f - yBasis[0] - yBasis[2] - yBasis[3];
		
		relative = fabsf((float)z/gridVoxelSpacing.z-(float)nodeAnte.z);
		float zBasis[4];
		zBasis[3]= relative * relative * relative / 6.0f;
		zBasis[0]= 1.0f/6.0f + relative*(relative-1.0f)/2.0f - zBasis[3];
		zBasis[2]= relative + zBasis[0] - 2.0f*zBasis[3];
		zBasis[1]= 1.0f - zBasis[0] - zBasis[2] - zBasis[3];

		float4 displacement=make_float4(0.0f,0.0f,0.0f,0.0f);

		int3 controlPointImageDim = c_ControlPointImageDim;

		int indexYZ, indexXYZ;
		for(short c=0; c<4; c++){
			float4 tempValueY=make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			indexYZ= ( (nodeAnte.z + c) * controlPointImageDim.y + nodeAnte.y ) * controlPointImageDim.x;
			for(short b=0; b<4; b++){
				float4 tempValueX=make_float4(0.0f, 0.0f, 0.0f, 0.0f);
				indexXYZ= indexYZ + nodeAnte.x;
				for(short a=0; a<4; a++){
					float4 nodeCoefficient = tex1Dfetch(controlPointTexture,indexXYZ);
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
		positionField[tid] = displacement;
	}
	return;
}

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

__global__ void reg_bspline_ApproxBendingEnergy_kernel( float *penaltyTerm)
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


__global__ void reg_bspline_getApproxBendingEnergyGradient_kernel(  float3 *bendingEnergyValue,
                                                                    float4 *nodeNMIGradientArray_d,
                                                                    float4 *basis_a_d,
                                                                    float2 *basis_b_d)
{
    __shared__ float basisXX[27];
    __shared__ float basisYY[27];
    __shared__ float basisZZ[27];
    __shared__ float basisXY[27];
    __shared__ float basisYZ[27];
    __shared__ float basisXZ[27];

    if(threadIdx.x<28){
        basisXX[threadIdx.x] = basis_a_d[threadIdx.x].x;
        basisYY[threadIdx.x] = basis_a_d[threadIdx.x].y;
        basisZZ[threadIdx.x] = basis_a_d[threadIdx.x].z;
        basisXY[threadIdx.x] = basis_a_d[threadIdx.x].w;
        basisYZ[threadIdx.x] = basis_b_d[threadIdx.x].x;
        basisXZ[threadIdx.x] = basis_b_d[threadIdx.x].y;
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

		short coord=0;
		for(short Z=z-1; Z<z+2; Z++){
			for(short Y=y-1; Y<y+2; Y++){
				for(short X=x-1; X<x+2; X++){
					if(-1<X && -1<Y && -1<Z && X<gridSize.x && Y<gridSize.y && Z<gridSize.z){
						bendingEnergyValuePtr = &(bendingEnergyValue[6 * ((Z*gridSize.y + Y)*gridSize.x + X)]);

                        gradientValue.x += (*bendingEnergyValuePtr).x * basisXX[coord];
                        gradientValue.y += (*bendingEnergyValuePtr).y * basisXX[coord];
                        gradientValue.z += (*bendingEnergyValuePtr).z * basisXX[coord];
                        bendingEnergyValuePtr++;

                        gradientValue.x += (*bendingEnergyValuePtr).x * basisYY[coord];
                        gradientValue.y += (*bendingEnergyValuePtr).y * basisYY[coord];
                        gradientValue.z += (*bendingEnergyValuePtr).z * basisYY[coord];
                        bendingEnergyValuePtr++;

                        gradientValue.x += (*bendingEnergyValuePtr).x * basisZZ[coord];
                        gradientValue.y += (*bendingEnergyValuePtr).y * basisZZ[coord];
                        gradientValue.z += (*bendingEnergyValuePtr).z * basisZZ[coord];
                        bendingEnergyValuePtr++;

                        gradientValue.x += (*bendingEnergyValuePtr).x * basisXY[coord];
                        gradientValue.y += (*bendingEnergyValuePtr).y * basisXY[coord];
                        gradientValue.z += (*bendingEnergyValuePtr).z * basisXY[coord];
                        bendingEnergyValuePtr++;

                        gradientValue.x += (*bendingEnergyValuePtr).x * basisYZ[coord];
                        gradientValue.y += (*bendingEnergyValuePtr).y * basisYZ[coord];
                        gradientValue.z += (*bendingEnergyValuePtr).z * basisYZ[coord];
                        bendingEnergyValuePtr++;

                        gradientValue.x += (*bendingEnergyValuePtr).x * basisXZ[coord];
                        gradientValue.y += (*bendingEnergyValuePtr).y * basisXZ[coord];
                        gradientValue.z += (*bendingEnergyValuePtr).z * basisXZ[coord];
					}
					coord++;
				}
			}
		}
		float4 metricGradientValue;
		metricGradientValue = nodeNMIGradientArray_d[tid];
		metricGradientValue.x = (1.0f-c_BendingEnergyWeight)*metricGradientValue.x + c_BendingEnergyWeight*gradientValue.x/(float)c_ControlPointNumber;
		metricGradientValue.y = (1.0f-c_BendingEnergyWeight)*metricGradientValue.y + c_BendingEnergyWeight*gradientValue.y/(float)c_ControlPointNumber;
		metricGradientValue.z = (1.0f-c_BendingEnergyWeight)*metricGradientValue.z + c_BendingEnergyWeight*gradientValue.z/(float)c_ControlPointNumber;
		nodeNMIGradientArray_d[tid]=metricGradientValue;

	}
}

#endif

