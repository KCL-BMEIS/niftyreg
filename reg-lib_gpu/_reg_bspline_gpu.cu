/*
 *  _reg_bspline_gpu.cu
 *  
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_BSPLINE_GPU_CU
#define _REG_BSPLINE_GPU_CU

#include "_reg_bspline_gpu.h"
#include "_reg_bspline_kernels.cu"

/* *************************************************************** */
/* *************************************************************** */

void reg_bspline_gpu(   nifti_image *controlPointImage,
                        nifti_image *targetImage,
                        float4 **controlPointImageArray_d,
                        float4 **positionFieldImageArray_d,
                        int **mask_d,
                        int activeVoxelNumber)
{
	const int voxelNumber = targetImage->nx * targetImage->ny * targetImage->nz;
	const int controlPointNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
	const int3 targetImageDim = make_int3(targetImage->nx, targetImage->ny, targetImage->nz);
	const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);

	const int controlPointGridMem = controlPointNumber*sizeof(float4);

	const float3 controlPointVoxelSpacing = make_float3(
		controlPointImage->dx / targetImage->dx,
		controlPointImage->dy / targetImage->dy,
		controlPointImage->dz / targetImage->dz);

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&voxelNumber,sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_TargetImageDim,&targetImageDim,sizeof(int3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim,sizeof(int3)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointVoxelSpacing,&controlPointVoxelSpacing,sizeof(float3)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ActiveVoxelNumber,&activeVoxelNumber,sizeof(int)));

    CUDA_SAFE_CALL(cudaBindTexture(0, controlPointTexture, *controlPointImageArray_d, controlPointGridMem));
    CUDA_SAFE_CALL(cudaBindTexture(0, maskTexture, *mask_d, activeVoxelNumber*sizeof(int)));

	const unsigned int Grid_reg_freeForm_interpolatePosition = 
		(unsigned int)ceil((float)activeVoxelNumber/(float)(Block_reg_freeForm_interpolatePosition));
	dim3 BlockP1(Block_reg_freeForm_interpolatePosition,1,1);
	dim3 GridP1(Grid_reg_freeForm_interpolatePosition,1,1);

	_reg_freeForm_interpolatePosition <<< GridP1, BlockP1 >>>(*positionFieldImageArray_d);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
	printf("[DEBUG] reg_freeForm_interpolatePosition kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
	       cudaGetErrorString(cudaGetLastError()),GridP1.x,GridP1.y,GridP1.z,BlockP1.x,BlockP1.y,BlockP1.z);
#endif
	return;
}

/* *************************************************************** */
/* *************************************************************** */

float reg_bspline_ApproxBendingEnergy_gpu(	nifti_image *controlPointImage,
						float4 **controlPointImageArray_d)
{
	const int controlPointNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
	const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
	const int controlPointGridMem = controlPointNumber*sizeof(float4);

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointNumber,&controlPointNumber,sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim,sizeof(int3)));
	CUDA_SAFE_CALL(cudaBindTexture(0,controlPointTexture, *controlPointImageArray_d, controlPointGridMem));

	float *penaltyTerm_d;
	CUDA_SAFE_CALL(cudaMalloc((void **)&penaltyTerm_d, controlPointNumber*sizeof(float)));

	const unsigned int Grid_reg_bspline_ApproxBendingEnergy = 
		(unsigned int)ceil((float)controlPointNumber/(float)(Block_reg_bspline_ApproxBendingEnergy));
	dim3 B1(Block_reg_bspline_ApproxBendingEnergy,1,1);
	dim3 G1(Grid_reg_bspline_ApproxBendingEnergy,1,1);

	reg_bspline_ApproxBendingEnergy_kernel <<< G1, B1 >>>(penaltyTerm_d);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
	printf("[DEBUG] reg_bspline_ApproxBendingEnergy kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
	       cudaGetErrorString(cudaGetLastError()),G1.x,G1.y,G1.z,B1.x,B1.y,B1.z);
#endif

	float *penaltyTerm_h;
	CUDA_SAFE_CALL(cudaMallocHost((void **)&penaltyTerm_h, controlPointNumber*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpy(penaltyTerm_h, penaltyTerm_d, controlPointNumber*sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(penaltyTerm_d));

	double penaltyValue=0.0;
	for(int i=0;i<controlPointNumber;i++)
		penaltyValue += penaltyTerm_h[i];
	CUDA_SAFE_CALL(cudaFreeHost((void *)penaltyTerm_h));

	return (float)(penaltyValue/(3.0*(double)controlPointNumber));
}

/* *************************************************************** */
/* *************************************************************** */

void reg_bspline_ApproxBendingEnergyGradient_gpu(   nifti_image *targetImage,
                                                    nifti_image *controlPointImage,
                                                    float4 **controlPointImageArray_d,
                                                    float4 **nodeNMIGradientArray_d,
                                                    float bendingEnergyWeight)
{
	const int controlPointNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
	const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
	const int controlPointGridMem = controlPointNumber*sizeof(float4);

	bendingEnergyWeight *= targetImage->nx*targetImage->ny*targetImage->nz
    		/ ( controlPointImage->nx*controlPointImage->ny*controlPointImage->nz );

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointNumber,&controlPointNumber,sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim,sizeof(int3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_Weight,&bendingEnergyWeight,sizeof(float)));
	CUDA_SAFE_CALL(cudaBindTexture(0,controlPointTexture, *controlPointImageArray_d, controlPointGridMem));

	float3 *bendingEnergyValue_d;
	CUDA_SAFE_CALL(cudaMalloc((void **)&bendingEnergyValue_d, 6*controlPointNumber*sizeof(float3)));
	CUDA_SAFE_CALL(cudaMemset(bendingEnergyValue_d, 0, 6*controlPointNumber*sizeof(float3)));

	const unsigned int Grid_reg_bspline_storeApproxBendingEnergy =
		(unsigned int)ceil((float)controlPointNumber/(float)(Block_reg_bspline_storeApproxBendingEnergy));
	dim3 B1(Block_reg_bspline_storeApproxBendingEnergy,1,1);
	dim3 G1(Grid_reg_bspline_storeApproxBendingEnergy,1,1);

	reg_bspline_storeApproxBendingEnergy_kernel <<< G1, B1 >>>(bendingEnergyValue_d);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
	printf("[DEBUG] reg_bspline_storeApproxBendingEnergy kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
	       cudaGetErrorString(cudaGetLastError()),G1.x,G1.y,G1.z,B1.x,B1.y,B1.z);
#endif

	float normal[3],first[3],second[3];
	normal[0] = 1.0f/6.0f;normal[1] = 2.0f/3.0f;normal[2] = 1.0f/6.0f;
	first[0] = -0.5f;first[1] = 0.0f;first[2] = 0.5f;
	second[0] = 1.0f;second[1] = -2.0f;second[2] = 1.0f;
	
	float4 *basis_a;CUDA_SAFE_CALL(cudaMallocHost((void **)&basis_a, 27*sizeof(float4)));
	float2 *basis_b;CUDA_SAFE_CALL(cudaMallocHost((void **)&basis_b, 27*sizeof(float2)));
	short coord=0;
	for(int c=0; c<3; c++){
		for(int b=0; b<3; b++){
			for(int a=0; a<3; a++){
				basis_a[coord].x=second[a]*normal[b]*normal[c];	// z * y * x"
				basis_a[coord].y=normal[a]*second[b]*normal[c];	// z * y"* x
				basis_a[coord].z=normal[a]*normal[b]*second[c];	// z"* y * x
				basis_a[coord].w=first[a]*first[b]*normal[c];	// z * y'* x'
				basis_b[coord].x=normal[a]*first[b]*first[c];	// z'* y'* x
				basis_b[coord].y=first[a]*normal[b]*first[c];	// z'* y * x'
				coord++;
			}
		}
	}
	float4 *basis_a_d;CUDA_SAFE_CALL(cudaMalloc((void **)&basis_a_d,27*sizeof(float4)));
	float2 *basis_b_d;CUDA_SAFE_CALL(cudaMalloc((void **)&basis_b_d,27*sizeof(float2)));
	CUDA_SAFE_CALL(cudaMemcpy(basis_a_d, basis_a, 27*sizeof(float4), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(basis_b_d, basis_b, 27*sizeof(float2), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaFreeHost((void *)basis_a));
	CUDA_SAFE_CALL(cudaFreeHost((void *)basis_b));
	CUDA_SAFE_CALL(cudaBindTexture(0, basisValueATexture, basis_a_d, 27*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, basisValueBTexture, basis_b_d, 27*sizeof(float2)));

	const unsigned int Grid_reg_bspline_getApproxBendingEnergyGradient =
		(unsigned int)ceil((float)controlPointNumber/(float)(Block_reg_bspline_getApproxBendingEnergyGradient));
	dim3 B2(Block_reg_bspline_getApproxBendingEnergyGradient,1,1);
	dim3 G2(Grid_reg_bspline_getApproxBendingEnergyGradient,1,1);

	reg_bspline_getApproxBendingEnergyGradient_kernel <<< G2, B2 >>>(	bendingEnergyValue_d,
										                                *nodeNMIGradientArray_d);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
	printf("[DEBUG] reg_bspline_getApproxBendingEnergyGradient kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
	       cudaGetErrorString(cudaGetLastError()),G2.x,G2.y,G2.z,B2.x,B2.y,B2.z);
#endif

	CUDA_SAFE_CALL(cudaFree((void *)basis_a_d));
	CUDA_SAFE_CALL(cudaFree((void *)basis_b_d));
	CUDA_SAFE_CALL(cudaFree((void *)bendingEnergyValue_d));

	return;
}

/* *************************************************************** */
/* *************************************************************** */

void reg_spline_cppComposition_gpu( nifti_image *toUpdate,
                                    nifti_image *toCompose,
                                    float4 **toUpdateArray_d,
                                    float4 **toComposeArray_d,
                                    float ratio,
                                    bool type)
{
    if(toUpdate->nvox != toCompose->nvox){
        fprintf(stderr,"ERROR:\treg_spline_cppComposition_gpu\n");
        fprintf(stderr,"ERROR:\tBoth image are expected to have the same size ... Exit\n");
        exit(1);
    }

    const int controlPointNumber = toCompose->nx*toCompose->ny*toCompose->nz;
    const int3 controlPointImageDim = make_int3(toCompose->nx, toCompose->ny, toCompose->nz);

    const int controlPointGridMem = controlPointNumber*sizeof(float4);

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_Type,&type,sizeof(bool)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointNumber,&controlPointNumber,sizeof(int3)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim,sizeof(int3)));

    // The transformation matrix is binded to a texture
    float4 *transformationMatrix_h;
    float4 *voxelToRealMatrix_d;
    float4 *realToVoxelMatrix_d;
    CUDA_SAFE_CALL(cudaMallocHost((void **)&transformationMatrix_h, 3*sizeof(float4)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&voxelToRealMatrix_d, 3*sizeof(float4)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&realToVoxelMatrix_d, 3*sizeof(float4)));
    mat44 *voxelToReal=NULL;
    mat44 *realToVoxel=NULL;
    if(toUpdate->sform_code>0){
        voxelToReal=&(toUpdate->sto_xyz);
        realToVoxel=&(toUpdate->sto_ijk);
    }
    else{
        voxelToReal=&(toUpdate->qto_xyz);
        realToVoxel=&(toUpdate->qto_ijk);
    }
    for(int i=0; i<3; i++){
        transformationMatrix_h[i].x=voxelToReal->m[i][0];
        transformationMatrix_h[i].y=voxelToReal->m[i][1];
        transformationMatrix_h[i].z=voxelToReal->m[i][2];
        transformationMatrix_h[i].w=voxelToReal->m[i][3];
    }
    CUDA_SAFE_CALL(cudaMemcpy(voxelToRealMatrix_d, transformationMatrix_h, 3*sizeof(float4), cudaMemcpyHostToDevice));
    cudaBindTexture(0,txVoxelToRealMatrix,voxelToRealMatrix_d,3*sizeof(float4));
    for(int i=0; i<3; i++){
        transformationMatrix_h[i].x=realToVoxel->m[i][0];
        transformationMatrix_h[i].y=realToVoxel->m[i][1];
        transformationMatrix_h[i].z=realToVoxel->m[i][2];
        transformationMatrix_h[i].w=realToVoxel->m[i][3];
    }
    CUDA_SAFE_CALL(cudaMemcpy(realToVoxelMatrix_d, transformationMatrix_h, 3*sizeof(float4), cudaMemcpyHostToDevice));
    cudaBindTexture(0,txRealToVoxelMatrix,realToVoxelMatrix_d,3*sizeof(float4));
    CUDA_SAFE_CALL(cudaFreeHost((void *)transformationMatrix_h));

    // The control point grid is binded to a texture
    CUDA_SAFE_CALL(cudaBindTexture(0, controlPointTexture, *toComposeArray_d, controlPointGridMem));

    const unsigned int Grid_reg_freeForm_interpolatePosition = 
        (unsigned int)ceil((float)controlPointNumber/(float)(Block_reg_spline_cppComposition));
    dim3 BlockP1(Block_reg_freeForm_interpolatePosition,1,1);
    dim3 GridP1(Grid_reg_freeForm_interpolatePosition,1,1);

    _reg_spline_cppComposition_kernel <<< GridP1, BlockP1 >>>(*toUpdateArray_d);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
    printf("[DEBUG] _reg_spline_cppComposition_kernel kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
           cudaGetErrorString(cudaGetLastError()),GridP1.x,GridP1.y,GridP1.z,BlockP1.x,BlockP1.y,BlockP1.z);
#endif
    return;
}

/* *************************************************************** */
/* *************************************************************** */


#endif
