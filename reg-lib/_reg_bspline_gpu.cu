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

    const unsigned int Grid_reg_freeForm_deformationField =
        (unsigned int)ceil((float)activeVoxelNumber/(float)(Block_reg_freeForm_deformationField));
    dim3 BlockP1(Block_reg_freeForm_deformationField,1,1);
    dim3 GridP1(Grid_reg_freeForm_deformationField,1,1);

    reg_freeForm_deformationField_kernel <<< GridP1, BlockP1 >>>(*positionFieldImageArray_d);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
    printf("[NiftyReg CUDA DEBUG] reg_freeForm_deformationField_kernel kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
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
    CUDA_SAFE_CALL(cudaMalloc(&penaltyTerm_d, controlPointNumber*sizeof(float)));

    const unsigned int Grid_reg_bspline_ApproxBendingEnergy =
        (unsigned int)ceil((float)controlPointNumber/(float)(Block_reg_bspline_ApproxBendingEnergy));
    dim3 B1(Block_reg_bspline_ApproxBendingEnergy,1,1);
    dim3 G1(Grid_reg_bspline_ApproxBendingEnergy,1,1);

    reg_bspline_ApproxBendingEnergy_kernel <<< G1, B1 >>>(penaltyTerm_d);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
    printf("[NiftyReg CUDA DEBUG] reg_bspline_ApproxBendingEnergy_kernel kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
	       cudaGetErrorString(cudaGetLastError()),G1.x,G1.y,G1.z,B1.x,B1.y,B1.z);
#endif

	float *penaltyTerm_h;
    CUDA_SAFE_CALL(cudaMallocHost(&penaltyTerm_h, controlPointNumber*sizeof(float)));
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
    CUDA_SAFE_CALL(cudaMalloc(&bendingEnergyValue_d, 6*controlPointNumber*sizeof(float3)));
    CUDA_SAFE_CALL(cudaMemset(bendingEnergyValue_d, 0, 6*controlPointNumber*sizeof(float3)));

    const unsigned int Grid_reg_bspline_storeApproxBendingEnergy =
        (unsigned int)ceil((float)controlPointNumber/(float)(Block_reg_bspline_storeApproxBendingEnergy));
    dim3 B1(Block_reg_bspline_storeApproxBendingEnergy,1,1);
    dim3 G1(Grid_reg_bspline_storeApproxBendingEnergy,1,1);

    reg_bspline_storeApproxBendingEnergy_kernel <<< G1, B1 >>>(bendingEnergyValue_d);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
    printf("[NiftyReg CUDA DEBUG] reg_bspline_storeApproxBendingEnergy_kernel kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
           cudaGetErrorString(cudaGetLastError()),G1.x,G1.y,G1.z,B1.x,B1.y,B1.z);
#endif

    float normal[3],first[3],second[3];
    normal[0] = 1.0f/6.0f;normal[1] = 2.0f/3.0f;normal[2] = 1.0f/6.0f;
    first[0] = -0.5f;first[1] = 0.0f;first[2] = 0.5f;
    second[0] = 1.0f;second[1] = -2.0f;second[2] = 1.0f;

    float4 *basis_a;CUDA_SAFE_CALL(cudaMallocHost(&basis_a, 27*sizeof(float4)));
    float2 *basis_b;CUDA_SAFE_CALL(cudaMallocHost(&basis_b, 27*sizeof(float2)));
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
    float4 *basis_a_d;CUDA_SAFE_CALL(cudaMalloc(&basis_a_d,27*sizeof(float4)));
    float2 *basis_b_d;CUDA_SAFE_CALL(cudaMalloc(&basis_b_d,27*sizeof(float2)));
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
    printf("[NiftyReg CUDA DEBUG] reg_bspline_getApproxBendingEnergyGradient_kernel kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
           cudaGetErrorString(cudaGetLastError()),G2.x,G2.y,G2.z,B2.x,B2.y,B2.z);
#endif

    CUDA_SAFE_CALL(cudaFree((void *)basis_a_d));
    CUDA_SAFE_CALL(cudaFree((void *)basis_b_d));
    CUDA_SAFE_CALL(cudaFree((void *)bendingEnergyValue_d));

    return;
}

/* *************************************************************** */
/* *************************************************************** */

void reg_bspline_ComputeApproximatedJacobianMap(   nifti_image *controlPointImage,
                                                    float4 **controlPointImageArray_d,
                                                    float **jacobianMap)
{
    /* Since we are using an approximation, only 27 basis values are used
        and they can be precomputed. We will store then in constant memory */
    float xBasisValues_h[27] = {-0.0138889f, 0.0000000f, 0.0138889f, -0.0555556f, 0.0000000f, 0.0555556f, -0.0138889f, 0.0000000f, 0.0138889f, 
                                -0.0555556f, 0.0000000f, 0.0555556f, -0.2222222f, 0.0000000f, 0.2222222f, -0.0555556f, 0.0000000f, 0.0555556f, 
                                -0.0138889f, 0.0000000f, 0.0138889f, -0.0555556f, 0.0000000f, 0.0555556f, -0.0138889f, 0.0000000f, 0.0138889f};
    float yBasisValues_h[27] = {-0.0138889f, -0.0555556f, -0.0138889f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0138889f, 0.0555556f, 0.0138889f, 
                                -0.0555556f, -0.2222222f, -0.0555556f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0555556f, 0.2222222f, 0.0555556f, 
                                -0.0138889f, -0.0555556f, -0.0138889f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0138889f, 0.0555556f, 0.0138889f};
    float zBasisValues_h[27] = {-0.0138889f, -0.0555556f, -0.0138889f, -0.0555556f, -0.2222222f, -0.0555556f, -0.0138889f, -0.0555556f, -0.0138889f, 
                                0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 
                                0.0138889f, 0.0555556f, 0.0138889f, 0.0555556f, 0.2222222f, 0.0555556f, 0.0138889f, 0.0555556f, 0.0138889f};
    float *xBasisValues_d, *yBasisValues_d, *zBasisValues_d;
    CUDA_SAFE_CALL(cudaMalloc(&xBasisValues_d, 27*sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc(&yBasisValues_d, 27*sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc(&zBasisValues_d, 27*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpy(xBasisValues_d, xBasisValues_h, 27*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(yBasisValues_d, yBasisValues_h, 27*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(zBasisValues_d, zBasisValues_h, 27*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaBindTexture(0, xBasisTexture, xBasisValues_d, 27*sizeof(float)));
    CUDA_SAFE_CALL(cudaBindTexture(0, yBasisTexture, yBasisValues_d, 27*sizeof(float)));
    CUDA_SAFE_CALL(cudaBindTexture(0, zBasisTexture, zBasisValues_d, 27*sizeof(float)));

    // Other constant memory and texture are binded
    const int controlPointNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointNumber,&controlPointNumber,sizeof(int)));
    const float3 controlPointSpacing = make_float3(controlPointImage->dx, controlPointImage->dy, controlPointImage->dz);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointSpacing,&controlPointSpacing, sizeof(float3)));
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim, sizeof(int3)));
    CUDA_SAFE_CALL(cudaBindTexture(0, controlPointTexture, *controlPointImageArray_d, controlPointNumber*sizeof(float4)));

    // The Jacobian matrices need to be reoriented
    mat33 reorient;
    reorient.m[0][0]=controlPointImage->dx; reorient.m[0][1]=0.0f; reorient.m[0][2]=0.0f;
    reorient.m[1][0]=0.0f; reorient.m[1][1]=controlPointImage->dy; reorient.m[1][2]=0.0f;
    reorient.m[2][0]=0.0f; reorient.m[2][1]=0.0f; reorient.m[2][2]=controlPointImage->dz;
    mat33 spline_ijk;
    if(controlPointImage->sform_code>0){
        spline_ijk.m[0][0]=controlPointImage->sto_ijk.m[0][0];
        spline_ijk.m[0][1]=controlPointImage->sto_ijk.m[0][1];
        spline_ijk.m[0][2]=controlPointImage->sto_ijk.m[0][2];
        spline_ijk.m[1][0]=controlPointImage->sto_ijk.m[1][0];
        spline_ijk.m[1][1]=controlPointImage->sto_ijk.m[1][1];
        spline_ijk.m[1][2]=controlPointImage->sto_ijk.m[1][2];
        spline_ijk.m[2][0]=controlPointImage->sto_ijk.m[2][0];
        spline_ijk.m[2][1]=controlPointImage->sto_ijk.m[2][1];
        spline_ijk.m[2][2]=controlPointImage->sto_ijk.m[2][2];
    }
    else{
        spline_ijk.m[0][0]=controlPointImage->qto_ijk.m[0][0];
        spline_ijk.m[0][1]=controlPointImage->qto_ijk.m[0][1];
        spline_ijk.m[0][2]=controlPointImage->qto_ijk.m[0][2];
        spline_ijk.m[1][0]=controlPointImage->qto_ijk.m[1][0];
        spline_ijk.m[1][1]=controlPointImage->qto_ijk.m[1][1];
        spline_ijk.m[1][2]=controlPointImage->qto_ijk.m[1][2];
        spline_ijk.m[2][0]=controlPointImage->qto_ijk.m[2][0];
        spline_ijk.m[2][1]=controlPointImage->qto_ijk.m[2][1];
        spline_ijk.m[2][2]=controlPointImage->qto_ijk.m[2][2];
    }
    reorient=nifti_mat33_inverse(nifti_mat33_mul(spline_ijk, reorient));
    float3 temp=make_float3(reorient.m[0][0],reorient.m[0][1],reorient.m[0][2]);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix0,&temp,sizeof(float3)));
    temp=make_float3(reorient.m[1][0],reorient.m[1][1],reorient.m[1][2]);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix1,&temp,sizeof(float3)));
    temp=make_float3(reorient.m[2][0],reorient.m[2][1],reorient.m[2][2]);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix2,&temp,sizeof(float3)));

    // The kernel is ran
    const unsigned int Grid_reg_bspline_ApproxJacDet =
        (unsigned int)ceil((float)controlPointNumber/(float)(Block_reg_bspline_ApproxJacDet));
    dim3 B1(Block_reg_bspline_ApproxJacDet,1,1);
    dim3 G1(Grid_reg_bspline_ApproxJacDet,1,1);

    reg_bspline_ApproxJacDet_kernel <<< G1, B1 >>>(*jacobianMap);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
    printf("[NiftyReg CUDA DEBUG] reg_bspline_ApproxJacDet_kernel kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
           cudaGetErrorString(cudaGetLastError()),G1.x,G1.y,G1.z,B1.x,B1.y,B1.z);
#endif

    CUDA_SAFE_CALL(cudaFree(xBasisValues_d));
    CUDA_SAFE_CALL(cudaFree(yBasisValues_d));
    CUDA_SAFE_CALL(cudaFree(zBasisValues_d));
}

/* *************************************************************** */
/* *************************************************************** */

void reg_bspline_ComputeJacobianMap(nifti_image *targetImage,
                                    nifti_image *controlPointImage,
                                    float4 **controlPointImageArray_d,
                                    float **jacobianMap)
{
    // Some constant memory variable are computed and allocated
    const int voxelNumber = targetImage->nx * targetImage->ny * targetImage->nz;
    const int controlPointNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
    const int3 targetImageDim = make_int3(targetImage->nx, targetImage->ny, targetImage->nz);
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    const float3 controlPointSpacing = make_float3(controlPointImage->dx, controlPointImage->dy, controlPointImage->dz);

    const int controlPointGridMem = controlPointNumber*sizeof(float4);

    const float3 controlPointVoxelSpacing = make_float3(
        controlPointImage->dx / targetImage->dx,
        controlPointImage->dy / targetImage->dy,
        controlPointImage->dz / targetImage->dz);

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&voxelNumber,sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_TargetImageDim,&targetImageDim,sizeof(int3)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim,sizeof(int3)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointVoxelSpacing,&controlPointVoxelSpacing,sizeof(float3)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointSpacing,&controlPointSpacing, sizeof(float3)));

    // Texture binding: control point position
    CUDA_SAFE_CALL(cudaBindTexture(0, controlPointTexture, *controlPointImageArray_d, controlPointGridMem));

    // The Jacobian matrices need to be reoriented, the affine matrix is store in constant memory
    mat33 reorient;
    reorient.m[0][0]=controlPointImage->dx; reorient.m[0][1]=0.0f; reorient.m[0][2]=0.0f;
    reorient.m[1][0]=0.0f; reorient.m[1][1]=controlPointImage->dy; reorient.m[1][2]=0.0f;
    reorient.m[2][0]=0.0f; reorient.m[2][1]=0.0f; reorient.m[2][2]=controlPointImage->dz;
    mat33 spline_ijk;
    if(controlPointImage->sform_code>0){
        spline_ijk.m[0][0]=controlPointImage->sto_ijk.m[0][0];
        spline_ijk.m[0][1]=controlPointImage->sto_ijk.m[0][1];
        spline_ijk.m[0][2]=controlPointImage->sto_ijk.m[0][2];
        spline_ijk.m[1][0]=controlPointImage->sto_ijk.m[1][0];
        spline_ijk.m[1][1]=controlPointImage->sto_ijk.m[1][1];
        spline_ijk.m[1][2]=controlPointImage->sto_ijk.m[1][2];
        spline_ijk.m[2][0]=controlPointImage->sto_ijk.m[2][0];
        spline_ijk.m[2][1]=controlPointImage->sto_ijk.m[2][1];
        spline_ijk.m[2][2]=controlPointImage->sto_ijk.m[2][2];
    }
    else{
        spline_ijk.m[0][0]=controlPointImage->qto_ijk.m[0][0];
        spline_ijk.m[0][1]=controlPointImage->qto_ijk.m[0][1];
        spline_ijk.m[0][2]=controlPointImage->qto_ijk.m[0][2];
        spline_ijk.m[1][0]=controlPointImage->qto_ijk.m[1][0];
        spline_ijk.m[1][1]=controlPointImage->qto_ijk.m[1][1];
        spline_ijk.m[1][2]=controlPointImage->qto_ijk.m[1][2];
        spline_ijk.m[2][0]=controlPointImage->qto_ijk.m[2][0];
        spline_ijk.m[2][1]=controlPointImage->qto_ijk.m[2][1];
        spline_ijk.m[2][2]=controlPointImage->qto_ijk.m[2][2];
    }
    reorient=nifti_mat33_inverse(nifti_mat33_mul(spline_ijk, reorient));
    float3 temp=make_float3(reorient.m[0][0],reorient.m[0][1],reorient.m[0][2]);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix0,&temp,sizeof(float3)));
    temp=make_float3(reorient.m[1][0],reorient.m[1][1],reorient.m[1][2]);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix1,&temp,sizeof(float3)));
    temp=make_float3(reorient.m[2][0],reorient.m[2][1],reorient.m[2][2]);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix2,&temp,sizeof(float3)));

    // The kernel is ran
    const unsigned int Grid_reg_bspline_JacDet =
        (unsigned int)ceil((float)voxelNumber/(float)(Block_reg_bspline_JacDet));
    dim3 B1(Block_reg_bspline_JacDet,1,1);
    dim3 G1(Grid_reg_bspline_JacDet,1,1);

    reg_bspline_JacDet_kernel <<< G1, B1 >>>(*jacobianMap);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
    printf("[NiftyReg CUDA DEBUG] reg_bspline_JacDet_kernel kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
           cudaGetErrorString(cudaGetLastError()),G1.x,G1.y,G1.z,B1.x,B1.y,B1.z);
#endif
}

/* *************************************************************** */
/* *************************************************************** */

double reg_bspline_ComputeJacobianPenaltyTerm_gpu(  nifti_image *targetImage,
                                                    nifti_image *controlPointImage,
                                                    float4 **controlPointImageArray_d,
                                                    bool approximate)
{
    // The Jacobian determinant will be stored into one array
    unsigned int pointNumber;
    if(approximate)
        pointNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
    else pointNumber = targetImage->nvox;


    float *jacobianMap_d;
    CUDA_SAFE_CALL(cudaMalloc(&jacobianMap_d, pointNumber*sizeof(float)));

    // The Jacobian map is computed
    if(approximate){
        reg_bspline_ComputeApproximatedJacobianMap( controlPointImage,
                                                    controlPointImageArray_d,
                                                    &jacobianMap_d);
    }
    else{
        reg_bspline_ComputeJacobianMap( targetImage,
                                        controlPointImage,
                                        controlPointImageArray_d,
                                        &jacobianMap_d);
    }

    // The Jacobian map is transfered back to the CPU and summed over
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&pointNumber,sizeof(int)));
    dim3 B1(Block_reg_bspline_GetSquaredLogJacDet,1,1);
    dim3 G1((unsigned int)ceil((float)pointNumber/(float)(Block_reg_bspline_GetSquaredLogJacDet)),1,1);
    reg_bspline_GetSquaredLogJacDet_kernel <<< G1, B1 >>>(jacobianMap_d);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
    printf("[NiftyReg CUDA DEBUG] reg_bspline_GetSquaredLogJacDet_kernel kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
           cudaGetErrorString(cudaGetLastError()),G1.x,G1.y,G1.z,B1.x,B1.y,B1.z);
#endif

    float *jacobianMap_h;
    CUDA_SAFE_CALL(cudaMallocHost(&jacobianMap_h, pointNumber*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpy(jacobianMap_h, jacobianMap_d, pointNumber*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(jacobianMap_d));
    double penaltyValue=0.0;
    for(unsigned i=0;i<pointNumber;i++){
        penaltyValue += (double)jacobianMap_h[i];
    }
    CUDA_SAFE_CALL(cudaFreeHost((void *)jacobianMap_h));
    if(approximate)
        penaltyValue /= (double)((controlPointImage->nx-2)*(controlPointImage->ny-2)*(controlPointImage->nz-2));
    else penaltyValue /= (double)(targetImage->nvox);
    return penaltyValue;
}

/* *************************************************************** */
/* *************************************************************** */

void reg_bspline_ComputeJacobianGradient_gpu(   nifti_image *targetImage,
                                                nifti_image *controlPointImage,
                                                float4 **controlPointImageArray_d,
                                                float4 **nodeNMIGradientArray_d,
                                                float jacobianWeight,
                                                bool approximate)
{
    // The Jacobian matrices need to be reoriented
    mat33 reorient;
    reorient.m[0][0]=controlPointImage->dx; reorient.m[0][1]=0.0f; reorient.m[0][2]=0.0f;
    reorient.m[1][0]=0.0f; reorient.m[1][1]=controlPointImage->dy; reorient.m[1][2]=0.0f;
    reorient.m[2][0]=0.0f; reorient.m[2][1]=0.0f; reorient.m[2][2]=controlPointImage->dz;
    mat33 spline_ijk;
    if(controlPointImage->sform_code>0){
        spline_ijk.m[0][0]=controlPointImage->sto_ijk.m[0][0];
        spline_ijk.m[0][1]=controlPointImage->sto_ijk.m[0][1];
        spline_ijk.m[0][2]=controlPointImage->sto_ijk.m[0][2];
        spline_ijk.m[1][0]=controlPointImage->sto_ijk.m[1][0];
        spline_ijk.m[1][1]=controlPointImage->sto_ijk.m[1][1];
        spline_ijk.m[1][2]=controlPointImage->sto_ijk.m[1][2];
        spline_ijk.m[2][0]=controlPointImage->sto_ijk.m[2][0];
        spline_ijk.m[2][1]=controlPointImage->sto_ijk.m[2][1];
        spline_ijk.m[2][2]=controlPointImage->sto_ijk.m[2][2];
    }
    else{
        spline_ijk.m[0][0]=controlPointImage->qto_ijk.m[0][0];
        spline_ijk.m[0][1]=controlPointImage->qto_ijk.m[0][1];
        spline_ijk.m[0][2]=controlPointImage->qto_ijk.m[0][2];
        spline_ijk.m[1][0]=controlPointImage->qto_ijk.m[1][0];
        spline_ijk.m[1][1]=controlPointImage->qto_ijk.m[1][1];
        spline_ijk.m[1][2]=controlPointImage->qto_ijk.m[1][2];
        spline_ijk.m[2][0]=controlPointImage->qto_ijk.m[2][0];
        spline_ijk.m[2][1]=controlPointImage->qto_ijk.m[2][1];
        spline_ijk.m[2][2]=controlPointImage->qto_ijk.m[2][2];
    }
    mat33 desorient = nifti_mat33_mul(spline_ijk, reorient);
    reorient=nifti_mat33_inverse(desorient);
    float3 temp=make_float3(reorient.m[0][0],reorient.m[0][1],reorient.m[0][2]);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix0,&temp,sizeof(float3)));
    temp=make_float3(reorient.m[1][0],reorient.m[1][1],reorient.m[1][2]);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix1,&temp,sizeof(float3)));
    temp=make_float3(reorient.m[2][0],reorient.m[2][1],reorient.m[2][2]);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix2,&temp,sizeof(float3)));

    // Constant memory allocation
    const int voxelNumber = targetImage->nx * targetImage->ny * targetImage->nz;
    const int controlPointNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
    const int3 targetImageDim = make_int3(targetImage->nx, targetImage->ny, targetImage->nz);
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    const float3 controlPointSpacing = make_float3(controlPointImage->dx, controlPointImage->dy, controlPointImage->dz);

    const float3 controlPointVoxelSpacing = make_float3(
        controlPointImage->dx / targetImage->dx,
        controlPointImage->dy / targetImage->dy,
        controlPointImage->dz / targetImage->dz);

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&voxelNumber,sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointNumber,&controlPointNumber,sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_TargetImageDim,&targetImageDim,sizeof(int3)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim,sizeof(int3)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointVoxelSpacing,&controlPointVoxelSpacing,sizeof(float3)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointSpacing,&controlPointSpacing, sizeof(float3)));

    // Texture binding: control point position
    CUDA_SAFE_CALL(cudaBindTexture(0, controlPointTexture, *controlPointImageArray_d, controlPointNumber*sizeof(float4)));

    // All the values will be store in two arrays
    float *jacobianMatrices_d;
    float *jacobianDeterminant_d;

    if(approximate){
            /* Since we are using an approximation, only 27 basis values are used
            and they can be precomputed. We will store then in constant memory */
        float xBasisValues_h[27] = {-0.0138889f, 0.0000000f, 0.0138889f, -0.0555556f, 0.0000000f, 0.0555556f, -0.0138889f, 0.0000000f, 0.0138889f, 
                                    -0.0555556f, 0.0000000f, 0.0555556f, -0.2222222f, 0.0000000f, 0.2222222f, -0.0555556f, 0.0000000f, 0.0555556f, 
                                    -0.0138889f, 0.0000000f, 0.0138889f, -0.0555556f, 0.0000000f, 0.0555556f, -0.0138889f, 0.0000000f, 0.0138889f};
        float yBasisValues_h[27] = {-0.0138889f, -0.0555556f, -0.0138889f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0138889f, 0.0555556f, 0.0138889f, 
                                    -0.0555556f, -0.2222222f, -0.0555556f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0555556f, 0.2222222f, 0.0555556f, 
                                    -0.0138889f, -0.0555556f, -0.0138889f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0138889f, 0.0555556f, 0.0138889f};
        float zBasisValues_h[27] = {-0.0138889f, -0.0555556f, -0.0138889f, -0.0555556f, -0.2222222f, -0.0555556f, -0.0138889f, -0.0555556f, -0.0138889f, 
                                    0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 
                                    0.0138889f, 0.0555556f, 0.0138889f, 0.0555556f, 0.2222222f, 0.0555556f, 0.0138889f, 0.0555556f, 0.0138889f};
        float *xBasisValues_d, *yBasisValues_d, *zBasisValues_d;
        CUDA_SAFE_CALL(cudaMalloc(&xBasisValues_d, 27*sizeof(float)));
        CUDA_SAFE_CALL(cudaMalloc(&yBasisValues_d, 27*sizeof(float)));
        CUDA_SAFE_CALL(cudaMalloc(&zBasisValues_d, 27*sizeof(float)));
        CUDA_SAFE_CALL(cudaMemcpy(xBasisValues_d, xBasisValues_h, 27*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(yBasisValues_d, yBasisValues_h, 27*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(zBasisValues_d, zBasisValues_h, 27*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaBindTexture(0, xBasisTexture, xBasisValues_d, 27*sizeof(float)));
        CUDA_SAFE_CALL(cudaBindTexture(0, yBasisTexture, yBasisValues_d, 27*sizeof(float)));
        CUDA_SAFE_CALL(cudaBindTexture(0, zBasisTexture, zBasisValues_d, 27*sizeof(float)));

        CUDA_SAFE_CALL(cudaMalloc(&jacobianMatrices_d,
            9*(controlPointImage->nx-2)*(controlPointImage->ny-2)*(controlPointImage->nz-2)*sizeof(float)));
        CUDA_SAFE_CALL(cudaMalloc(&jacobianDeterminant_d,
            (controlPointImage->nx-2)*(controlPointImage->ny-2)*(controlPointImage->nz-2)*sizeof(float)));

        // The Jacobian matrices array is filled
        const unsigned int Grid_reg_bspline_ApproxJacobianMatrix =
            (unsigned int)ceil((float)controlPointNumber/(float)(Block_reg_bspline_ApproxJacobianMatrix));
        dim3 B1(Block_reg_bspline_ApproxJacobianMatrix,1,1);
        dim3 G1(Grid_reg_bspline_ApproxJacobianMatrix,1,1);

        reg_bspline_ApproxJacobianMatrix_kernel <<< G1, B1 >>>(jacobianMatrices_d, jacobianDeterminant_d);
        CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
        printf("[NiftyReg CUDA DEBUG] _reg_bspline_ApproxJacobianMatrix_kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
            cudaGetErrorString(cudaGetLastError()),G1.x,G1.y,G1.z,B1.x,B1.y,B1.z);
#endif

        CUDA_SAFE_CALL(cudaFree(xBasisValues_d));
        CUDA_SAFE_CALL(cudaFree(yBasisValues_d));
        CUDA_SAFE_CALL(cudaFree(zBasisValues_d));
    }
    else{
        CUDA_SAFE_CALL(cudaMalloc(&jacobianDeterminant_d,
            targetImage->nvox*sizeof(float)));
        CUDA_SAFE_CALL(cudaMalloc(&jacobianMatrices_d,
            9*targetImage->nvox*sizeof(float)));

        // The Jacobian matrices array is filled
        const unsigned int Grid_reg_bspline_JacobianMatrix =
            (unsigned int)ceil((float)targetImage->nvox/(float)(Block_reg_bspline_JacobianMatrix));
        dim3 B1(Block_reg_bspline_JacobianMatrix,1,1);
        dim3 G1(Grid_reg_bspline_JacobianMatrix,1,1);

        reg_bspline_JacobianMatrix_kernel <<< G1, B1 >>>(jacobianMatrices_d, jacobianDeterminant_d);
        CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
        printf("[NiftyReg CUDA DEBUG] reg_bspline_JacobianMatrix_kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
            cudaGetErrorString(cudaGetLastError()),G1.x,G1.y,G1.z,B1.x,B1.y,B1.z);
#endif
    }
    // The gradient computed on a node basis
    temp=make_float3(desorient.m[0][0],desorient.m[0][1],desorient.m[0][2]);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix0,&temp,sizeof(float3)));
    temp=make_float3(desorient.m[1][0],desorient.m[1][1],desorient.m[1][2]);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix1,&temp,sizeof(float3)));
    temp=make_float3(desorient.m[2][0],desorient.m[2][1],desorient.m[2][2]);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix2,&temp,sizeof(float3)));

    if(approximate){
        // The weight is transfered to constant memory
        float weight=jacobianWeight;
        weight = jacobianWeight * targetImage->nvox
            / ( controlPointImage->nx*controlPointImage->ny*controlPointImage->nz);
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_Weight,&weight,sizeof(float)));

        // The jacobian matrices are binded to a texture
        CUDA_SAFE_CALL(cudaBindTexture(0, jacobianMatricesTexture, jacobianMatrices_d,
            9*(controlPointImage->nx-2)*(controlPointImage->ny-2)*(controlPointImage->nz-2)*sizeof(float)));
        CUDA_SAFE_CALL(cudaBindTexture(0, jacobianDeterminantTexture, jacobianDeterminant_d,
            (controlPointImage->nx-2)*(controlPointImage->ny-2)*(controlPointImage->nz-2)*sizeof(float)));

        const unsigned int Grid_reg_bspline_ApproxJacobianGradient =
            (unsigned int)ceil((float)controlPointNumber/(float)(Block_reg_bspline_ApproxJacobianGradient));
        dim3 B2(Block_reg_bspline_ApproxJacobianGradient,1,1);
        dim3 G2(Grid_reg_bspline_ApproxJacobianGradient,1,1);

        reg_bspline_ApproxJacobianGradient_kernel <<< G2, B2 >>>(*nodeNMIGradientArray_d);
        CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
        printf("[NiftyReg CUDA DEBUG] _reg_bspline_ApproxJacobianGradient_kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
            cudaGetErrorString(cudaGetLastError()),G2.x,G2.y,G2.z,B2.x,B2.y,B2.z);
#endif
    }
    else{
        // The weight is transfered to constant memory
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_Weight,&jacobianWeight,sizeof(float)));

        // The jacobian matrices are binded to a texture
        CUDA_SAFE_CALL(cudaBindTexture(0, jacobianDeterminantTexture, jacobianDeterminant_d,
            targetImage->nvox*sizeof(float)));
        CUDA_SAFE_CALL(cudaBindTexture(0, jacobianMatricesTexture, jacobianMatrices_d,
            9*targetImage->nvox*sizeof(float)));

        const unsigned int Grid_reg_bspline_JacobianGradient =
            (unsigned int)ceil((float)controlPointNumber/(float)(Block_reg_bspline_JacobianGradient));
        dim3 B2(Block_reg_bspline_JacobianGradient,1,1);
        dim3 G2(Grid_reg_bspline_JacobianGradient,1,1);

        reg_bspline_JacobianGradient_kernel <<< G2, B2 >>>(*nodeNMIGradientArray_d);
        CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
        printf("[NiftyReg CUDA DEBUG] _reg_bspline_JacobianGradient_kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
            cudaGetErrorString(cudaGetLastError()),G2.x,G2.y,G2.z,B2.x,B2.y,B2.z);
#endif
    }

    CUDA_SAFE_CALL(cudaFree(jacobianMatrices_d));
    CUDA_SAFE_CALL(cudaFree(jacobianDeterminant_d));
}

/* *************************************************************** */
/* *************************************************************** */

double reg_bspline_correctFolding_gpu(  nifti_image *targetImage,
                                        nifti_image *controlPointImage,
                                        float4 **controlPointImageArray_d,
                                        bool approximate)
{
    // The Jacobian matrices need to be reoriented
    mat33 reorient;
    reorient.m[0][0]=controlPointImage->dx; reorient.m[0][1]=0.0f; reorient.m[0][2]=0.0f;
    reorient.m[1][0]=0.0f; reorient.m[1][1]=controlPointImage->dy; reorient.m[1][2]=0.0f;
    reorient.m[2][0]=0.0f; reorient.m[2][1]=0.0f; reorient.m[2][2]=controlPointImage->dz;
    mat33 spline_ijk;
    if(controlPointImage->sform_code>0){
        spline_ijk.m[0][0]=controlPointImage->sto_ijk.m[0][0];
        spline_ijk.m[0][1]=controlPointImage->sto_ijk.m[0][1];
        spline_ijk.m[0][2]=controlPointImage->sto_ijk.m[0][2];
        spline_ijk.m[1][0]=controlPointImage->sto_ijk.m[1][0];
        spline_ijk.m[1][1]=controlPointImage->sto_ijk.m[1][1];
        spline_ijk.m[1][2]=controlPointImage->sto_ijk.m[1][2];
        spline_ijk.m[2][0]=controlPointImage->sto_ijk.m[2][0];
        spline_ijk.m[2][1]=controlPointImage->sto_ijk.m[2][1];
        spline_ijk.m[2][2]=controlPointImage->sto_ijk.m[2][2];
    }
    else{
        spline_ijk.m[0][0]=controlPointImage->qto_ijk.m[0][0];
        spline_ijk.m[0][1]=controlPointImage->qto_ijk.m[0][1];
        spline_ijk.m[0][2]=controlPointImage->qto_ijk.m[0][2];
        spline_ijk.m[1][0]=controlPointImage->qto_ijk.m[1][0];
        spline_ijk.m[1][1]=controlPointImage->qto_ijk.m[1][1];
        spline_ijk.m[1][2]=controlPointImage->qto_ijk.m[1][2];
        spline_ijk.m[2][0]=controlPointImage->qto_ijk.m[2][0];
        spline_ijk.m[2][1]=controlPointImage->qto_ijk.m[2][1];
        spline_ijk.m[2][2]=controlPointImage->qto_ijk.m[2][2];
    }
    mat33 desorient = nifti_mat33_mul(spline_ijk, reorient);
    reorient=nifti_mat33_inverse(desorient);
    float3 temp=make_float3(reorient.m[0][0],reorient.m[0][1],reorient.m[0][2]);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix0,&temp,sizeof(float3)));
    temp=make_float3(reorient.m[1][0],reorient.m[1][1],reorient.m[1][2]);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix1,&temp,sizeof(float3)));
    temp=make_float3(reorient.m[2][0],reorient.m[2][1],reorient.m[2][2]);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix2,&temp,sizeof(float3)));

    // Constant memory allocation
    const int voxelNumber = targetImage->nx * targetImage->ny * targetImage->nz;
    const int controlPointNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
    const int3 targetImageDim = make_int3(targetImage->nx, targetImage->ny, targetImage->nz);
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    const float3 controlPointSpacing = make_float3(controlPointImage->dx, controlPointImage->dy, controlPointImage->dz);

    const float3 controlPointVoxelSpacing = make_float3(
        controlPointImage->dx / targetImage->dx,
        controlPointImage->dy / targetImage->dy,
        controlPointImage->dz / targetImage->dz);

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&voxelNumber,sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointNumber,&controlPointNumber,sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_TargetImageDim,&targetImageDim,sizeof(int3)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim,sizeof(int3)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointVoxelSpacing,&controlPointVoxelSpacing,sizeof(float3)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointSpacing,&controlPointSpacing, sizeof(float3)));

    // Texture binding: control point position
    CUDA_SAFE_CALL(cudaBindTexture(0, controlPointTexture, *controlPointImageArray_d, controlPointNumber*sizeof(float4)));

    // All the values will be store in an array
    float *jacobianMatrices_d;
    float *jacobianDeterminant_d;

    if(approximate){
            /* Since we are using an approximation, only 27 basis values are used
            and they can be precomputed. We will store then in constant memory */
        float xBasisValues_h[27] = {-0.0138889f, 0.0000000f, 0.0138889f, -0.0555556f, 0.0000000f, 0.0555556f, -0.0138889f, 0.0000000f, 0.0138889f, 
                                    -0.0555556f, 0.0000000f, 0.0555556f, -0.2222222f, 0.0000000f, 0.2222222f, -0.0555556f, 0.0000000f, 0.0555556f, 
                                    -0.0138889f, 0.0000000f, 0.0138889f, -0.0555556f, 0.0000000f, 0.0555556f, -0.0138889f, 0.0000000f, 0.0138889f};
        float yBasisValues_h[27] = {-0.0138889f, -0.0555556f, -0.0138889f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0138889f, 0.0555556f, 0.0138889f, 
                                    -0.0555556f, -0.2222222f, -0.0555556f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0555556f, 0.2222222f, 0.0555556f, 
                                    -0.0138889f, -0.0555556f, -0.0138889f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0138889f, 0.0555556f, 0.0138889f};
        float zBasisValues_h[27] = {-0.0138889f, -0.0555556f, -0.0138889f, -0.0555556f, -0.2222222f, -0.0555556f, -0.0138889f, -0.0555556f, -0.0138889f, 
                                    0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 
                                    0.0138889f, 0.0555556f, 0.0138889f, 0.0555556f, 0.2222222f, 0.0555556f, 0.0138889f, 0.0555556f, 0.0138889f};
        float *xBasisValues_d, *yBasisValues_d, *zBasisValues_d;
        CUDA_SAFE_CALL(cudaMalloc(&xBasisValues_d, 27*sizeof(float)));
        CUDA_SAFE_CALL(cudaMalloc(&yBasisValues_d, 27*sizeof(float)));
        CUDA_SAFE_CALL(cudaMalloc(&zBasisValues_d, 27*sizeof(float)));
        CUDA_SAFE_CALL(cudaMemcpy(xBasisValues_d, xBasisValues_h, 27*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(yBasisValues_d, yBasisValues_h, 27*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(zBasisValues_d, zBasisValues_h, 27*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaBindTexture(0, xBasisTexture, xBasisValues_d, 27*sizeof(float)));
        CUDA_SAFE_CALL(cudaBindTexture(0, yBasisTexture, yBasisValues_d, 27*sizeof(float)));
        CUDA_SAFE_CALL(cudaBindTexture(0, zBasisTexture, zBasisValues_d, 27*sizeof(float)));

        CUDA_SAFE_CALL(cudaMalloc(&jacobianMatrices_d,
            9*(controlPointImage->nx-2)*(controlPointImage->ny-2)*(controlPointImage->nz-2)*sizeof(float)));
        CUDA_SAFE_CALL(cudaMalloc(&jacobianDeterminant_d,
            (controlPointImage->nx-2)*(controlPointImage->ny-2)*(controlPointImage->nz-2)*sizeof(float)));

        // The Jacobian matrices array is filled
        const unsigned int Grid_reg_bspline_ApproxJacobianMatrix =
            (unsigned int)ceil((float)controlPointNumber/(float)(Block_reg_bspline_ApproxJacobianMatrix));
        dim3 B1(Block_reg_bspline_ApproxJacobianMatrix,1,1);
        dim3 G1(Grid_reg_bspline_ApproxJacobianMatrix,1,1);

        reg_bspline_ApproxJacobianMatrix_kernel <<< G1, B1 >>>(jacobianMatrices_d, jacobianDeterminant_d);
        CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
        printf("[NiftyReg CUDA DEBUG] _reg_bspline_ApproxJacobianMatrix_kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
            cudaGetErrorString(cudaGetLastError()),G1.x,G1.y,G1.z,B1.x,B1.y,B1.z);
#endif

        CUDA_SAFE_CALL(cudaFree(xBasisValues_d));
        CUDA_SAFE_CALL(cudaFree(yBasisValues_d));
        CUDA_SAFE_CALL(cudaFree(zBasisValues_d));
    }
    else{
        CUDA_SAFE_CALL(cudaMalloc(&jacobianMatrices_d,
            9*targetImage->nvox*sizeof(float)));
        CUDA_SAFE_CALL(cudaMalloc(&jacobianDeterminant_d,
            targetImage->nvox*sizeof(float)));

        // The Jacobian matrices array is filled
        const unsigned int Grid_reg_bspline_JacobianMatrix =
            (unsigned int)ceil((float)targetImage->nvox/(float)(Block_reg_bspline_JacobianMatrix));
        dim3 B1(Block_reg_bspline_JacobianMatrix,1,1);
        dim3 G1(Grid_reg_bspline_JacobianMatrix,1,1);

        reg_bspline_JacobianMatrix_kernel <<< G1, B1 >>>(jacobianMatrices_d, jacobianDeterminant_d);
        CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
        printf("[NiftyReg CUDA DEBUG] _reg_bspline_JacobianMatrix_kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
            cudaGetErrorString(cudaGetLastError()),G1.x,G1.y,G1.z,B1.x,B1.y,B1.z);
#endif
    }

    double JacDetpenaltyTerm=0.;
    float *jacobianDeterminant_h;
    if(approximate){
        CUDA_SAFE_CALL(cudaMallocHost(&jacobianDeterminant_h,(controlPointImage->nx-2)*(controlPointImage->ny-2)*(controlPointImage->nz-2)*sizeof(float)));
        CUDA_SAFE_CALL(cudaMemcpy(jacobianDeterminant_h, jacobianDeterminant_d,
                                  (controlPointImage->nx-2)*(controlPointImage->ny-2)*(controlPointImage->nz-2)*sizeof(float),
                                  cudaMemcpyDeviceToHost));
        for(int i=0; i<(controlPointImage->nx-2)*(controlPointImage->ny-2)*(controlPointImage->nz-2); i++){
            double logDet = log(jacobianDeterminant_h[i]);
            JacDetpenaltyTerm += logDet * logDet;
        }
        JacDetpenaltyTerm /= (double)((controlPointImage->nx-2)*(controlPointImage->ny-2)*(controlPointImage->nz-2));
    }
    else{
        CUDA_SAFE_CALL(cudaMallocHost(&jacobianDeterminant_h,targetImage->nvox*sizeof(float)));
        CUDA_SAFE_CALL(cudaMemcpy(jacobianDeterminant_h, jacobianDeterminant_d,
                                  targetImage->nvox*sizeof(float),
                                  cudaMemcpyDeviceToHost));
        for(unsigned i=0; i<targetImage->nvox; i++){
            double logDet = log((double)jacobianDeterminant_h[i]);
            JacDetpenaltyTerm += logDet * logDet;
        }
        JacDetpenaltyTerm /= (double)(targetImage->nvox);

    }
    CUDA_SAFE_CALL(cudaFreeHost(jacobianDeterminant_h));

    if(JacDetpenaltyTerm==JacDetpenaltyTerm){
        CUDA_SAFE_CALL(cudaFree(jacobianMatrices_d));
        CUDA_SAFE_CALL(cudaFree(jacobianDeterminant_d));
        return JacDetpenaltyTerm;
    }

    temp=make_float3(desorient.m[0][0],desorient.m[0][1],desorient.m[0][2]);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix0,&temp,sizeof(float3)));
    temp=make_float3(desorient.m[1][0],desorient.m[1][1],desorient.m[1][2]);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix1,&temp,sizeof(float3)));
    temp=make_float3(desorient.m[2][0],desorient.m[2][1],desorient.m[2][2]);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix2,&temp,sizeof(float3)));

    // The folded voxel impact on the node position
    if(approximate){
        // The jacobian matrices are binded to a texture
        CUDA_SAFE_CALL(cudaBindTexture(0, jacobianMatricesTexture, jacobianMatrices_d,
            9*(controlPointImage->nx-2)*(controlPointImage->ny-2)*(controlPointImage->nz-2)*sizeof(float)));
        CUDA_SAFE_CALL(cudaBindTexture(0, jacobianDeterminantTexture, jacobianDeterminant_d,
            (controlPointImage->nx-2)*(controlPointImage->ny-2)*(controlPointImage->nz-2)*sizeof(float)));

        const unsigned int Grid_reg_bspline_ApproxCorrectFolding =
            (unsigned int)ceil((float)controlPointNumber/(float)(Block_reg_bspline_ApproxCorrectFolding));
        dim3 B2(Block_reg_bspline_ApproxCorrectFolding,1,1);
        dim3 G2(Grid_reg_bspline_ApproxCorrectFolding,1,1);

        reg_bspline_ApproxCorrectFolding_kernel <<< G2, B2 >>>(*controlPointImageArray_d);
        CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
        printf("[NiftyReg CUDA DEBUG] _reg_bspline_ApproxCorrectFolding_kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
            cudaGetErrorString(cudaGetLastError()),G2.x,G2.y,G2.z,B2.x,B2.y,B2.z);
#endif
    }
    else{
        // The jacobian matrices are binded to a texture
        CUDA_SAFE_CALL(cudaBindTexture(0, jacobianMatricesTexture, jacobianMatrices_d,
            9*targetImage->nvox*sizeof(float)));
        CUDA_SAFE_CALL(cudaBindTexture(0, jacobianDeterminantTexture, jacobianDeterminant_d,
            targetImage->nvox*sizeof(float)));

        const unsigned int Grid_reg_bspline_CorrectFolding =
            (unsigned int)ceil((float)controlPointNumber/(float)(Block_reg_bspline_CorrectFolding));
        dim3 B2(Block_reg_bspline_CorrectFolding,1,1);
        dim3 G2(Grid_reg_bspline_CorrectFolding,1,1);

        reg_bspline_CorrectFolding_kernel <<< G2, B2 >>>(*controlPointImageArray_d);
        CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
        printf("[NiftyReg CUDA DEBUG] _reg_bspline_CorrectFolding_kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
            cudaGetErrorString(cudaGetLastError()),G2.x,G2.y,G2.z,B2.x,B2.y,B2.z);
#endif
    }
    CUDA_SAFE_CALL(cudaFree(jacobianMatrices_d));
    CUDA_SAFE_CALL(cudaFree(jacobianDeterminant_d));

    return std::numeric_limits<double>::quiet_NaN();
}

/* *************************************************************** */
/* *************************************************************** */
void reg_spline_cppComposition_gpu( nifti_image *toUpdate,
                                    nifti_image *toCompose,
                                    float4 **toUpdateArray_d,
                                    float4 **toComposeArray_d, // displacement
                                    float length,
                                    bool type)
{
    if(toUpdate->nvox != toCompose->nvox){
        fprintf(stderr,"ERROR:\treg_spline_cppComposition_gpu\n");
        fprintf(stderr,"ERROR:\tBoths image are expected to have the same size ... Exit\n");
        exit(1);
    }

    const int controlPointNumber = toCompose->nx*toCompose->ny*toCompose->nz;
    const int3 controlPointImageDim = make_int3(toCompose->nx, toCompose->ny, toCompose->nz);

    const int controlPointGridMem = controlPointNumber*sizeof(float4);

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_Weight,&length,sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_Type,&type,sizeof(bool)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointNumber,&controlPointNumber,sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim,sizeof(int3)));

    // The transformation matrix is binded to a texture
    float4 *transformationMatrix_h;
    float4 *voxelToRealMatrix_d;
    float4 *realToVoxelMatrix_d;
    CUDA_SAFE_CALL(cudaMallocHost(&transformationMatrix_h, 3*sizeof(float4)));
    CUDA_SAFE_CALL(cudaMalloc(&voxelToRealMatrix_d, 3*sizeof(float4)));
    CUDA_SAFE_CALL(cudaMalloc(&realToVoxelMatrix_d, 3*sizeof(float4)));
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

    const unsigned int Grid_reg_spline_cppComposition =
        (unsigned int)ceil((float)controlPointNumber/(float)(Block_reg_spline_cppComposition));
    dim3 BlockP1(Block_reg_spline_cppComposition,1,1);
    dim3 GridP1(Grid_reg_spline_cppComposition,1,1);

    reg_spline_cppComposition_kernel <<< GridP1, BlockP1 >>>(*toUpdateArray_d);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
    printf("[NiftyReg CUDA DEBUG] _reg_spline_cppComposition_kernel kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
           cudaGetErrorString(cudaGetLastError()),GridP1.x,GridP1.y,GridP1.z,BlockP1.x,BlockP1.y,BlockP1.z);
#endif
    CUDA_SAFE_CALL(cudaFree(realToVoxelMatrix_d));
    CUDA_SAFE_CALL(cudaFree(voxelToRealMatrix_d));
    return;
}

/* *************************************************************** */
/* *************************************************************** */
void reg_spline_cppDeconvolve_gpu(  nifti_image *inputControlPointImage,
                                    nifti_image *outputControlPointImage,
                                    float4 **inputControlPointArray_d,
                                    float4 **outputControlPointArray_d)
{
    if(inputControlPointImage->nvox != outputControlPointImage->nvox){
        fprintf(stderr,"ERROR:\treg_spline_deconvolve_gpu\n");
        fprintf(stderr,"ERROR:\tBoth images are expected to have the same size ... Exit\n");
        exit(1);
    }

    // Some useful variables are transfered to the card
    const int controlPointNumber = inputControlPointImage->nx*inputControlPointImage->ny*inputControlPointImage->nz;
    const int3 controlPointImageDim = make_int3(inputControlPointImage->nx, inputControlPointImage->ny, inputControlPointImage->nz);
    const int controlPointGridMem = controlPointNumber*sizeof(float4);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointNumber,&controlPointNumber,sizeof(int3)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim,sizeof(int3)));

    // The control point grid is copied to the output grid
    CUDA_SAFE_CALL(cudaMemcpy(*outputControlPointArray_d, *inputControlPointArray_d, controlPointGridMem, cudaMemcpyDeviceToDevice));

    // An image in the global memory is allocated to reduce the occupancy (not sure that a good idea)
    float3 *temporaryGridImage_d;
    CUDA_SAFE_CALL(cudaMalloc(&temporaryGridImage_d, controlPointNumber*sizeof(float3)));

    // A first kernel along the X axis is ran
    unsigned int Grid_reg_spline_cppDeconvolve =
        (unsigned int)ceil((float)(inputControlPointImage->ny*inputControlPointImage->nz)/(float)(Block_reg_spline_cppDeconvolve));
    dim3 BlockP1(Block_reg_spline_cppDeconvolve,1,1);
    dim3 GridP1(Grid_reg_spline_cppDeconvolve,1,1);
    reg_spline_cppDeconvolve_kernel <<< GridP1, BlockP1 >>>(temporaryGridImage_d, *outputControlPointArray_d, 0);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
    printf("[NiftyReg CUDA DEBUG] _reg_spline_cppDeconvolve_kernel X axis: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
           cudaGetErrorString(cudaGetLastError()),GridP1.x,GridP1.y,GridP1.z,BlockP1.x,BlockP1.y,BlockP1.z);
#endif

    // A second kernel along the Y axis is ran
    Grid_reg_spline_cppDeconvolve =
        (unsigned int)ceil((float)(inputControlPointImage->nx*inputControlPointImage->nz)/(float)(Block_reg_spline_cppDeconvolve));
    dim3 GridP2(Grid_reg_spline_cppDeconvolve,1,1);
    reg_spline_cppDeconvolve_kernel <<< GridP2, BlockP1 >>>(temporaryGridImage_d, *outputControlPointArray_d,1);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
    printf("[NiftyReg CUDA DEBUG] _reg_spline_cppDeconvolve_kernel Y axis: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
           cudaGetErrorString(cudaGetLastError()),GridP2.x,GridP2.y,GridP2.z,BlockP1.x,BlockP1.y,BlockP1.z);
#endif

    // A third kernel along the Z axis is ran
    Grid_reg_spline_cppDeconvolve =
        (unsigned int)ceil((float)(inputControlPointImage->nx*inputControlPointImage->ny)/(float)(Block_reg_spline_cppDeconvolve));
    dim3 GridP3(Grid_reg_spline_cppDeconvolve,1,1);
    reg_spline_cppDeconvolve_kernel <<< GridP3, BlockP1 >>>(temporaryGridImage_d, *outputControlPointArray_d,2);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
    printf("[NiftyReg CUDA DEBUG] _reg_spline_cppDeconvolve_kernel Z axis: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
           cudaGetErrorString(cudaGetLastError()),GridP3.x,GridP3.y,GridP3.z,BlockP1.x,BlockP1.y,BlockP1.z);
#endif

    CUDA_SAFE_CALL(cudaFree(temporaryGridImage_d));
}

/* *************************************************************** */
/* *************************************************************** */
void reg_spline_getDeformationFromDisplacement_gpu( nifti_image *image,
                                                    float4 **imageArray_d)
{
    // Some useful variables are transfered to the card
    const int controlPointNumber = image->nx*image->ny*image->nz;
    const int3 controlPointImageDim = make_int3(image->nx, image->ny, image->nz);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointNumber,&controlPointNumber,sizeof(int3)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim,sizeof(int3)));

    // The transformation matrix is binded to a texture
    float4 *transformationMatrix_h;
    float4 *voxelToRealMatrix_d;
    CUDA_SAFE_CALL(cudaMallocHost(&transformationMatrix_h, 3*sizeof(float4)));
    CUDA_SAFE_CALL(cudaMalloc(&voxelToRealMatrix_d, 3*sizeof(float4)));
    mat44 *voxelToReal=NULL;
    if(image->sform_code>0)
        voxelToReal=&(image->sto_xyz);
    else
        voxelToReal=&(image->qto_xyz);
    for(int i=0; i<3; i++){
        transformationMatrix_h[i].x=voxelToReal->m[i][0];
        transformationMatrix_h[i].y=voxelToReal->m[i][1];
        transformationMatrix_h[i].z=voxelToReal->m[i][2];
        transformationMatrix_h[i].w=voxelToReal->m[i][3];
    }
    CUDA_SAFE_CALL(cudaMemcpy(voxelToRealMatrix_d, transformationMatrix_h, 3*sizeof(float4), cudaMemcpyHostToDevice));
    cudaBindTexture(0,txVoxelToRealMatrix,voxelToRealMatrix_d,3*sizeof(float4));
    CUDA_SAFE_CALL(cudaFreeHost((void *)transformationMatrix_h));

    // A first kernel along the X axis is ran
    unsigned int Grid_reg_spline_getDeformationFromDisplacement =
        (unsigned int)ceil((float)(controlPointNumber)/(float)(Block_reg_spline_getDeformationFromDisplacement));
    dim3 BlockP1(Block_reg_spline_getDeformationFromDisplacement,1,1);
    dim3 GridP1(Grid_reg_spline_getDeformationFromDisplacement,1,1);
    reg_spline_getDeformationFromDisplacement_kernel <<< GridP1, BlockP1 >>>(*imageArray_d);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
    printf("[NiftyReg CUDA DEBUG] _reg_spline_getDeformationFromDisplacement kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
           cudaGetErrorString(cudaGetLastError()),GridP1.x,GridP1.y,GridP1.z,BlockP1.x,BlockP1.y,BlockP1.z);
#endif
    CUDA_SAFE_CALL(cudaFree(voxelToRealMatrix_d));
}
/* *************************************************************** */
/* *************************************************************** */
void reg_spline_scaling_squaring_gpu(   nifti_image *velocityFieldImage,
                                        nifti_image *controlPointImage,
                                        float4 **velocityArray_d,
                                        float4 **controlPointArray_d)
{
    if(velocityFieldImage->nvox != controlPointImage->nvox){
        fprintf(stderr,"ERROR:\treg_spline_scaling_squaring_gpu\n");
        fprintf(stderr,"ERROR:\tBoth images are expected to have the same size ... Exit\n");
        exit(1);
    }

    const int controlPointNumber = velocityFieldImage->nx*velocityFieldImage->ny*velocityFieldImage->nz;

    // A temporary image is allocated on the card
    float4 *nodePositionArray_d;
    CUDA_SAFE_CALL(cudaMalloc(&nodePositionArray_d, controlPointNumber*sizeof(float4)));
    CUDA_SAFE_CALL(cudaMemcpy(nodePositionArray_d, *velocityArray_d, controlPointNumber*sizeof(float4) , cudaMemcpyDeviceToDevice));


    reg_spline_cppDeconvolve_gpu(   controlPointImage,
                                    controlPointImage,
                                    &nodePositionArray_d,   // input
                                    controlPointArray_d);   // output

    for(unsigned int i=0; i<SQUARING_VALUE; i++){
        reg_spline_cppComposition_gpu(  controlPointImage,
                                        controlPointImage,
                                        &nodePositionArray_d,   // deformed (output)
                                        controlPointArray_d,    // used as a grid
                                        1.0,
                                        0);
        reg_spline_cppDeconvolve_gpu(   controlPointImage,
                                        controlPointImage,
                                        &nodePositionArray_d,
                                        controlPointArray_d);
    }
    CUDA_SAFE_CALL(cudaFree(nodePositionArray_d));

    reg_spline_getDeformationFromDisplacement_gpu(  controlPointImage,
                                                    controlPointArray_d);
}
/* *************************************************************** */
/* *************************************************************** */
void reg_bspline_ComputeApproximatedJacobianMapFromVelocityField(   nifti_image *controlPointImage,
                                                                    float4 **controlPointImageArray_d,
                                                                    float **jacobianMap_d,
                                                                    float4 **voxelDisplacementField_d)
{

    // Other constant memory and texture are binded
    const int controlPointNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointNumber,&controlPointNumber,sizeof(int)));
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim, sizeof(int3)));
    CUDA_SAFE_CALL(cudaBindTexture(0, controlPointTexture, *controlPointImageArray_d, controlPointNumber*sizeof(float4)));
    const float3 controlPointSpacing = make_float3(controlPointImage->dx, controlPointImage->dy, controlPointImage->dz);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointSpacing,&controlPointSpacing, sizeof(float3)));

    // The Jacobian matrices need to be reoriented
    mat33 reorient;
    reorient.m[0][0]=controlPointImage->dx; reorient.m[0][1]=0.0f; reorient.m[0][2]=0.0f;
    reorient.m[1][0]=0.0f; reorient.m[1][1]=controlPointImage->dy; reorient.m[1][2]=0.0f;
    reorient.m[2][0]=0.0f; reorient.m[2][1]=0.0f; reorient.m[2][2]=controlPointImage->dz;
    mat33 spline_ijk;
    if(controlPointImage->sform_code>0){
        spline_ijk.m[0][0]=controlPointImage->sto_ijk.m[0][0];
        spline_ijk.m[0][1]=controlPointImage->sto_ijk.m[0][1];
        spline_ijk.m[0][2]=controlPointImage->sto_ijk.m[0][2];
        spline_ijk.m[1][0]=controlPointImage->sto_ijk.m[1][0];
        spline_ijk.m[1][1]=controlPointImage->sto_ijk.m[1][1];
        spline_ijk.m[1][2]=controlPointImage->sto_ijk.m[1][2];
        spline_ijk.m[2][0]=controlPointImage->sto_ijk.m[2][0];
        spline_ijk.m[2][1]=controlPointImage->sto_ijk.m[2][1];
        spline_ijk.m[2][2]=controlPointImage->sto_ijk.m[2][2];
    }
    else{
        spline_ijk.m[0][0]=controlPointImage->qto_ijk.m[0][0];
        spline_ijk.m[0][1]=controlPointImage->qto_ijk.m[0][1];
        spline_ijk.m[0][2]=controlPointImage->qto_ijk.m[0][2];
        spline_ijk.m[1][0]=controlPointImage->qto_ijk.m[1][0];
        spline_ijk.m[1][1]=controlPointImage->qto_ijk.m[1][1];
        spline_ijk.m[1][2]=controlPointImage->qto_ijk.m[1][2];
        spline_ijk.m[2][0]=controlPointImage->qto_ijk.m[2][0];
        spline_ijk.m[2][1]=controlPointImage->qto_ijk.m[2][1];
        spline_ijk.m[2][2]=controlPointImage->qto_ijk.m[2][2];
    }
    reorient=nifti_mat33_inverse(nifti_mat33_mul(spline_ijk, reorient));
    float3 temp=make_float3(reorient.m[0][0],reorient.m[0][1],reorient.m[0][2]);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix0,&temp,sizeof(float3)));
    temp=make_float3(reorient.m[1][0],reorient.m[1][1],reorient.m[1][2]);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix1,&temp,sizeof(float3)));
    temp=make_float3(reorient.m[2][0],reorient.m[2][1],reorient.m[2][2]);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix2,&temp,sizeof(float3)));

    // Since the control point grid and the jacobian map are in the same space,
    // Only the real2voxel matrix is need to convert the displacement in voxel
    float4 *transformationMatrix_h;
    float4 *realToVoxelMatrix_d;
    CUDA_SAFE_CALL(cudaMallocHost(&transformationMatrix_h, 3*sizeof(float4)));
    CUDA_SAFE_CALL(cudaMalloc(&realToVoxelMatrix_d, 3*sizeof(float4)));
    mat44 *realToVoxel=NULL;
    if(controlPointImage->sform_code>0)
        realToVoxel=&(controlPointImage->sto_ijk);
    else realToVoxel=&(controlPointImage->qto_ijk);
    for(int i=0; i<3; i++){
        transformationMatrix_h[i].x=realToVoxel->m[i][0];
        transformationMatrix_h[i].y=realToVoxel->m[i][1];
        transformationMatrix_h[i].z=realToVoxel->m[i][2];
        transformationMatrix_h[i].w=realToVoxel->m[i][3];
    }// 41 regs - 025% occupancy
    CUDA_SAFE_CALL(cudaMemcpy(realToVoxelMatrix_d, transformationMatrix_h, 3*sizeof(float4), cudaMemcpyHostToDevice));
    cudaBindTexture(0,txRealToVoxelMatrix,realToVoxelMatrix_d,3*sizeof(float4));
    CUDA_SAFE_CALL(cudaFreeHost((void *)transformationMatrix_h));

    // The kernel is ran
    const unsigned int Grid_reg_bspline_ApproxJacDetFromVelField =
        (unsigned int)ceil((float)controlPointNumber/(float)(Block_reg_bspline_ApproxJacDetFromVelField));
    dim3 B1(Block_reg_bspline_ApproxJacDetFromVelField,1,1);
    dim3 G1(Grid_reg_bspline_ApproxJacDetFromVelField,1,1);

    reg_bspline_ApproxJacDetFromVelField_kernel <<< G1, B1 >>>(*jacobianMap_d, *voxelDisplacementField_d);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
    printf("[NiftyReg CUDA DEBUG] _reg_bspline_ApproxJacobianDeterminantFromVelocityField_kernel kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
           cudaGetErrorString(cudaGetLastError()),G1.x,G1.y,G1.z,B1.x,B1.y,B1.z);
#endif

    CUDA_SAFE_CALL(cudaFree(realToVoxelMatrix_d));
}
/* *************************************************************** */
/* *************************************************************** */
void reg_bspline_ComputeJacobianMapFromVelocityField(   nifti_image *targetImage,
                                                        nifti_image *controlPointImage,
                                                        float4 **controlPointImageArray_d,
                                                        float **jacobianMap_d,
                                                        float4 **voxelDisplacementField_d)
{

    // Other constant memory and texture are binded
    const int controlPointNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
    CUDA_SAFE_CALL(cudaBindTexture(0, controlPointTexture, *controlPointImageArray_d, controlPointNumber*sizeof(float4)));

    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim, sizeof(int3)));

    const int3 targetImageDim = make_int3(targetImage->nx, targetImage->ny, targetImage->nz);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_TargetImageDim,&targetImageDim, sizeof(int3)));

    const int voxelNumber = targetImage->nvox;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&voxelNumber,sizeof(int)));

    const float3 controlPointSpacing = make_float3(controlPointImage->dx, controlPointImage->dy, controlPointImage->dz);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointSpacing,&controlPointSpacing, sizeof(float3)));

    const float3 controlPointVoxelSpacing = make_float3(
        controlPointImage->dx / targetImage->dx,
        controlPointImage->dy / targetImage->dy,
        controlPointImage->dz / targetImage->dz);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointVoxelSpacing,&controlPointVoxelSpacing,sizeof(float3)));

    // The Jacobian matrices need to be reoriented
    mat33 reorient;
    reorient.m[0][0]=controlPointImage->dx; reorient.m[0][1]=0.0f; reorient.m[0][2]=0.0f;
    reorient.m[1][0]=0.0f; reorient.m[1][1]=controlPointImage->dy; reorient.m[1][2]=0.0f;
    reorient.m[2][0]=0.0f; reorient.m[2][1]=0.0f; reorient.m[2][2]=controlPointImage->dz;
    mat33 spline_ijk;
    if(controlPointImage->sform_code>0){
        spline_ijk.m[0][0]=controlPointImage->sto_ijk.m[0][0];
        spline_ijk.m[0][1]=controlPointImage->sto_ijk.m[0][1];
        spline_ijk.m[0][2]=controlPointImage->sto_ijk.m[0][2];
        spline_ijk.m[1][0]=controlPointImage->sto_ijk.m[1][0];
        spline_ijk.m[1][1]=controlPointImage->sto_ijk.m[1][1];
        spline_ijk.m[1][2]=controlPointImage->sto_ijk.m[1][2];
        spline_ijk.m[2][0]=controlPointImage->sto_ijk.m[2][0];
        spline_ijk.m[2][1]=controlPointImage->sto_ijk.m[2][1];
        spline_ijk.m[2][2]=controlPointImage->sto_ijk.m[2][2];
    }
    else{
        spline_ijk.m[0][0]=controlPointImage->qto_ijk.m[0][0];
        spline_ijk.m[0][1]=controlPointImage->qto_ijk.m[0][1];
        spline_ijk.m[0][2]=controlPointImage->qto_ijk.m[0][2];
        spline_ijk.m[1][0]=controlPointImage->qto_ijk.m[1][0];
        spline_ijk.m[1][1]=controlPointImage->qto_ijk.m[1][1];
        spline_ijk.m[1][2]=controlPointImage->qto_ijk.m[1][2];
        spline_ijk.m[2][0]=controlPointImage->qto_ijk.m[2][0];
        spline_ijk.m[2][1]=controlPointImage->qto_ijk.m[2][1];
        spline_ijk.m[2][2]=controlPointImage->qto_ijk.m[2][2];
    }
    reorient=nifti_mat33_inverse(nifti_mat33_mul(spline_ijk, reorient));
    float3 temp=make_float3(reorient.m[0][0],reorient.m[0][1],reorient.m[0][2]);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix0,&temp,sizeof(float3)));
    temp=make_float3(reorient.m[1][0],reorient.m[1][1],reorient.m[1][2]);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix1,&temp,sizeof(float3)));
    temp=make_float3(reorient.m[2][0],reorient.m[2][1],reorient.m[2][2]);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix2,&temp,sizeof(float3)));

    // Since the control point grid and the jacobian map are in the same space,
    // Only the real2voxel matrix is need to convert the displacement in voxel
    float4 *transformationMatrix_h;
    float4 *realToVoxelMatrix_d;
    CUDA_SAFE_CALL(cudaMallocHost(&transformationMatrix_h, 3*sizeof(float4)));
    CUDA_SAFE_CALL(cudaMalloc(&realToVoxelMatrix_d, 3*sizeof(float4)));
    mat44 *realToVoxel=NULL;
    if(targetImage->sform_code>0) realToVoxel=&(targetImage->sto_ijk);
    else realToVoxel=&(targetImage->qto_ijk);
    for(int i=0; i<3; i++){
        transformationMatrix_h[i].x=realToVoxel->m[i][0];
        transformationMatrix_h[i].y=realToVoxel->m[i][1];
        transformationMatrix_h[i].z=realToVoxel->m[i][2];
        transformationMatrix_h[i].w=realToVoxel->m[i][3];
    }
    CUDA_SAFE_CALL(cudaMemcpy(realToVoxelMatrix_d, transformationMatrix_h, 3*sizeof(float4), cudaMemcpyHostToDevice));
    cudaBindTexture(0,txRealToVoxelMatrix,realToVoxelMatrix_d,3*sizeof(float4));
    CUDA_SAFE_CALL(cudaFreeHost((void *)transformationMatrix_h));

    // The kernel is ran
    const unsigned int Grid_reg_bspline_JacDetFromVelField =
        (unsigned int)ceil((float)voxelNumber/(float)(Block_reg_bspline_JacDetFromVelField));
    dim3 B1(Block_reg_bspline_JacDetFromVelField,1,1);
    dim3 G1(Grid_reg_bspline_JacDetFromVelField,1,1);

    reg_bspline_JacDetFromVelField_kernel <<< G1, B1 >>>(*jacobianMap_d, *voxelDisplacementField_d);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
    printf("[NiftyReg CUDA DEBUG] _reg_bspline_JacobianDeterminantFromVelocityField_kernel kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
           cudaGetErrorString(cudaGetLastError()),G1.x,G1.y,G1.z,B1.x,B1.y,B1.z);
#endif

    CUDA_SAFE_CALL(cudaFree(realToVoxelMatrix_d));
}

/* *************************************************************** */
/* *************************************************************** */

void reg_bspline_ComputeJacobianGradientFromVelocityField(  nifti_image *targetImage,
                                                            nifti_image *controlPointImage,
                                                            float4 **controlPointImageArray_d,
                                                            float4 **voxelDisplacementField_d,
                                                            float **JacobianMatricesArray_d,
                                                            float **jacobianDetArray_d,
                                                            float4 **gradientImageArray_d,
                                                            float weight,
                                                            bool approx)
{

    // Other constant memory and texture are binded
    const int controlPointNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
    CUDA_SAFE_CALL(cudaBindTexture(0, controlPointTexture, *controlPointImageArray_d, controlPointNumber*sizeof(float4)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointNumber,&controlPointNumber, sizeof(int)));

    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim, sizeof(int3)));

    const int3 targetImageDim = make_int3(targetImage->nx, targetImage->ny, targetImage->nz);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_TargetImageDim,&targetImageDim, sizeof(int3)));

    int voxelNumber = targetImage->nvox;
    if(approx) voxelNumber = controlPointImage->nvox/3;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&voxelNumber,sizeof(int)));

    const float3 controlPointSpacing = make_float3(controlPointImage->dx, controlPointImage->dy, controlPointImage->dz);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointSpacing,&controlPointSpacing, sizeof(float3)));

    float3 controlPointVoxelSpacing=make_float3(1.f,1.f,1.f);
    if(!approx)
        controlPointVoxelSpacing = make_float3(
            controlPointImage->dx / targetImage->dx,
            controlPointImage->dy / targetImage->dy,
            controlPointImage->dz / targetImage->dz);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointVoxelSpacing,&controlPointVoxelSpacing,sizeof(float3)));

    // The Jacobian matrices need to be reoriented
    mat33 desorient;
    desorient.m[0][0]=controlPointImage->dx; desorient.m[0][1]=0.0f; desorient.m[0][2]=0.0f;
    desorient.m[1][0]=0.0f; desorient.m[1][1]=controlPointImage->dy; desorient.m[1][2]=0.0f;
    desorient.m[2][0]=0.0f; desorient.m[2][1]=0.0f; desorient.m[2][2]=controlPointImage->dz;
    mat33 spline_ijk;
    if(controlPointImage->sform_code>0){
        spline_ijk.m[0][0]=controlPointImage->sto_ijk.m[0][0];
        spline_ijk.m[0][1]=controlPointImage->sto_ijk.m[0][1];
        spline_ijk.m[0][2]=controlPointImage->sto_ijk.m[0][2];
        spline_ijk.m[1][0]=controlPointImage->sto_ijk.m[1][0];
        spline_ijk.m[1][1]=controlPointImage->sto_ijk.m[1][1];
        spline_ijk.m[1][2]=controlPointImage->sto_ijk.m[1][2];
        spline_ijk.m[2][0]=controlPointImage->sto_ijk.m[2][0];
        spline_ijk.m[2][1]=controlPointImage->sto_ijk.m[2][1];
        spline_ijk.m[2][2]=controlPointImage->sto_ijk.m[2][2];
    }
    else{
        spline_ijk.m[0][0]=controlPointImage->qto_ijk.m[0][0];
        spline_ijk.m[0][1]=controlPointImage->qto_ijk.m[0][1];
        spline_ijk.m[0][2]=controlPointImage->qto_ijk.m[0][2];
        spline_ijk.m[1][0]=controlPointImage->qto_ijk.m[1][0];
        spline_ijk.m[1][1]=controlPointImage->qto_ijk.m[1][1];
        spline_ijk.m[1][2]=controlPointImage->qto_ijk.m[1][2];
        spline_ijk.m[2][0]=controlPointImage->qto_ijk.m[2][0];
        spline_ijk.m[2][1]=controlPointImage->qto_ijk.m[2][1];
        spline_ijk.m[2][2]=controlPointImage->qto_ijk.m[2][2];
    }
    mat33 reorient=nifti_mat33_inverse(nifti_mat33_mul(spline_ijk, desorient));
    float3 temp=make_float3(reorient.m[0][0],reorient.m[0][1],reorient.m[0][2]);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix0,&temp,sizeof(float3)));
    temp=make_float3(reorient.m[1][0],reorient.m[1][1],reorient.m[1][2]);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix1,&temp,sizeof(float3)));
    temp=make_float3(reorient.m[2][0],reorient.m[2][1],reorient.m[2][2]);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix2,&temp,sizeof(float3)));

    float4 *transformationMatrix_h;
    float4 *realToVoxelMatrix_d;
    CUDA_SAFE_CALL(cudaMallocHost(&transformationMatrix_h, 3*sizeof(float4)));
    CUDA_SAFE_CALL(cudaMalloc(&realToVoxelMatrix_d, 3*sizeof(float4)));
    mat44 *realToVoxel=NULL;
    if(approx){
        if(controlPointImage->sform_code>0) realToVoxel=&(controlPointImage->sto_ijk);
        else realToVoxel=&(controlPointImage->qto_ijk);
    }
    else{
        if(targetImage->sform_code>0) realToVoxel=&(targetImage->sto_ijk);
        else realToVoxel=&(targetImage->qto_ijk);
    }
    for(int i=0; i<3; i++){
        transformationMatrix_h[i].x=realToVoxel->m[i][0];
        transformationMatrix_h[i].y=realToVoxel->m[i][1];
        transformationMatrix_h[i].z=realToVoxel->m[i][2];
        transformationMatrix_h[i].w=realToVoxel->m[i][3];
    }
    CUDA_SAFE_CALL(cudaMemcpy(realToVoxelMatrix_d, transformationMatrix_h, 3*sizeof(float4), cudaMemcpyHostToDevice));
    cudaBindTexture(0,txRealToVoxelMatrix,realToVoxelMatrix_d,3*sizeof(float4));
    CUDA_SAFE_CALL(cudaFreeHost((void *)transformationMatrix_h));

    // The current voxel position is saved since it is need for the gradient computation
    float4 *oldVoxelDisplacementField_d;
    CUDA_SAFE_CALL(cudaMalloc(&oldVoxelDisplacementField_d, voxelNumber*sizeof(float4)));
    CUDA_SAFE_CALL(cudaMemcpy(oldVoxelDisplacementField_d, *voxelDisplacementField_d,
          voxelNumber*sizeof(float4) , cudaMemcpyDeviceToDevice));

    // The Jacobian matrice array is filled
    const unsigned int Grid_reg_bspline_JacobianMatrixFromVel =
        (unsigned int)ceil((float)voxelNumber/(float)(Block_reg_bspline_JacobianMatrixFromVel));
    dim3 B1(Block_reg_bspline_JacobianMatrixFromVel,1,1);
    dim3 G1(Grid_reg_bspline_JacobianMatrixFromVel,1,1);

    reg_bspline_JacobianMatrixFromVel_kernel <<< G1, B1 >>>(*JacobianMatricesArray_d, *jacobianDetArray_d, *voxelDisplacementField_d);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
    printf("[NiftyReg CUDA DEBUG] _reg_bspline_JacobianMatrix_kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
        cudaGetErrorString(cudaGetLastError()),G1.x,G1.y,G1.z,B1.x,B1.y,B1.z);
#endif

    // The gradient computed on a node basis and need to be oriented in the control point space
    temp=make_float3(desorient.m[0][0],desorient.m[0][1],desorient.m[0][2]);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix0,&temp,sizeof(float3)));
    temp=make_float3(desorient.m[1][0],desorient.m[1][1],desorient.m[1][2]);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix1,&temp,sizeof(float3)));
    temp=make_float3(desorient.m[2][0],desorient.m[2][1],desorient.m[2][2]);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix2,&temp,sizeof(float3)));

    // The weight is transfered to constant memory
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_Weight,&weight,sizeof(float)));

    // The jacobian matrices are binded to a texture
    CUDA_SAFE_CALL(cudaBindTexture(0, jacobianDeterminantTexture, *jacobianDetArray_d,
        targetImage->nvox*sizeof(float)));
    CUDA_SAFE_CALL(cudaBindTexture(0, jacobianMatricesTexture, *JacobianMatricesArray_d,
        9*targetImage->nvox*sizeof(float)));

    // The voxel position are converted into indices
    if(approx)
        reg_spline_getDeformationFromDisplacement_gpu( controlPointImage,
                                                       &oldVoxelDisplacementField_d);
    else reg_spline_getDeformationFromDisplacement_gpu( targetImage,
                                                        &oldVoxelDisplacementField_d);

    const unsigned int Grid_reg_bspline_PositionToIndices =
        (unsigned int)ceil((float)voxelNumber/(float)(Block_reg_bspline_PositionToIndices));
    dim3 B2(Block_reg_bspline_PositionToIndices,1,1);
    dim3 G2(Grid_reg_bspline_PositionToIndices,1,1);
    reg_bspline_PositionToIndices_kernel <<< G2, B2 >>>(oldVoxelDisplacementField_d);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
    printf("[NiftyReg CUDA DEBUG] _reg_bspline_PositionToIndices_kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
        cudaGetErrorString(cudaGetLastError()),G2.x,G2.y,G2.z,B2.x,B2.y,B2.z);
#endif

    // The voxel indices are binded to a texture
    CUDA_SAFE_CALL(cudaBindTexture(0, voxelDisplacementTexture, oldVoxelDisplacementField_d,
        voxelNumber*sizeof(float4)));


    const unsigned int Grid_reg_bspline_JacobianGradFromVel =
        (unsigned int)ceil((float)controlPointNumber/(float)(Block_reg_bspline_JacobianGradFromVel));
    dim3 B3(Block_reg_bspline_JacobianGradFromVel,1,1);
    dim3 G3(Grid_reg_bspline_JacobianGradFromVel,1,1);
    reg_bspline_JacobianGradFromVel_kernel <<< G3, B3 >>>(*gradientImageArray_d);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
    printf("[NiftyReg CUDA DEBUG] _reg_bspline_JacobianGradFromVel_kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
        cudaGetErrorString(cudaGetLastError()),G3.x,G3.y,G3.z,B3.x,B3.y,B3.z);
#endif

    CUDA_SAFE_CALL(cudaFree(oldVoxelDisplacementField_d));
    CUDA_SAFE_CALL(cudaFree(realToVoxelMatrix_d));
}
/* *************************************************************** */
/* *************************************************************** */
double reg_bspline_ComputeJacobianPenaltyTermFromVelocity_gpu(  nifti_image *targetImage,
                                                                nifti_image *velocityFieldImage,
                                                                float4 **velocityFieldImageArray_d,
                                                                bool approximate)
{
    const int controlPointNumber = velocityFieldImage->nx*velocityFieldImage->ny*velocityFieldImage->nz;

    // A Jacobian array is allocated
    int pointNumber=0;
    if(approximate) pointNumber = controlPointNumber;
    else pointNumber = (int)targetImage->nvox;
    float *jacobianDeterminantArray_d;
    CUDA_SAFE_CALL(cudaMalloc(&jacobianDeterminantArray_d, pointNumber*sizeof(float)));

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&pointNumber,sizeof(int)));
    dim3 B1(Block_reg_bspline_SetJacDetToOne,1,1);
    dim3 G1((unsigned int)ceil((float)pointNumber/(float)(Block_reg_bspline_SetJacDetToOne)),1,1);
    reg_bspline_SetJacDetToOne_kernel <<< G1, B1 >>>(jacobianDeterminantArray_d);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
    printf("[NiftyReg CUDA DEBUG] _reg_bspline_SetJacobianDeterminantToOne_kernel kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
           cudaGetErrorString(cudaGetLastError()),G1.x,G1.y,G1.z,B1.x,B1.y,B1.z);
#endif

    // A displacement field array is allocated since the voxel position as to be always updated
    float4 *voxelDisplacementField_d;
    CUDA_SAFE_CALL(cudaMalloc(&voxelDisplacementField_d, pointNumber*sizeof(float4)));
    CUDA_SAFE_CALL(cudaMemset(voxelDisplacementField_d, 0, pointNumber*sizeof(float4)));

    // Two temporary images are allocated on the card
    float4 *controlPointArray_d;
    float4 *nodePositionArray_d;
    CUDA_SAFE_CALL(cudaMalloc(&controlPointArray_d, controlPointNumber*sizeof(float4)));
    CUDA_SAFE_CALL(cudaMalloc(&nodePositionArray_d, controlPointNumber*sizeof(float4)));
    CUDA_SAFE_CALL(cudaMemcpy(nodePositionArray_d, *velocityFieldImageArray_d,
          controlPointNumber*sizeof(float4) , cudaMemcpyDeviceToDevice));

    // The velocity field is first deconvolved
    reg_spline_cppDeconvolve_gpu(   velocityFieldImage,
                                    velocityFieldImage,
                                    &nodePositionArray_d,   // input
                                    &controlPointArray_d);  // output


    // The Jacobian map is computed using assuming we deal with a control point position
    if(approximate){
        reg_bspline_ComputeApproximatedJacobianMapFromVelocityField(velocityFieldImage,
                                                                    &controlPointArray_d,
                                                                    &jacobianDeterminantArray_d,
                                                                    &voxelDisplacementField_d);
    }
    else{
        reg_bspline_ComputeJacobianMapFromVelocityField(targetImage,
                                                        velocityFieldImage,
                                                        &controlPointArray_d,
                                                        &jacobianDeterminantArray_d,
                                                        &voxelDisplacementField_d);
    }

    for(unsigned int i=1; i<SQUARING_VALUE; i++){
        reg_spline_cppComposition_gpu(  velocityFieldImage,
                                        velocityFieldImage,
                                        &nodePositionArray_d,   // deformed (output)
                                        &controlPointArray_d,    // used as a grid
                                        1.0,
                                        0);
        reg_spline_cppDeconvolve_gpu(   velocityFieldImage,
                                        velocityFieldImage,
                                        &nodePositionArray_d,
                                        &controlPointArray_d);
        if(approximate){
            reg_bspline_ComputeApproximatedJacobianMapFromVelocityField(velocityFieldImage,
                                                                        &controlPointArray_d,
                                                                        &jacobianDeterminantArray_d,
                                                                        &voxelDisplacementField_d);
        }
        else{
            reg_bspline_ComputeJacobianMapFromVelocityField(targetImage,
                                                            velocityFieldImage,
                                                            &controlPointArray_d,
                                                            &jacobianDeterminantArray_d,
                                                            &voxelDisplacementField_d);
        }
    }


    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&pointNumber,sizeof(int)));
    dim3 B2(Block_reg_bspline_GetSquaredLogJacDet,1,1);
    dim3 G2((unsigned int)ceil((float)pointNumber/(float)(Block_reg_bspline_GetSquaredLogJacDet)),1,1);
    reg_bspline_GetSquaredLogJacDet_kernel <<< G2, B2 >>>(jacobianDeterminantArray_d);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
    printf("[NiftyReg CUDA DEBUG] _reg_bspline_GetSquaredLogJacobianDeterminant_kernel kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
           cudaGetErrorString(cudaGetLastError()),G2.x,G2.y,G2.z,B2.x,B2.y,B2.z);
#endif
    // The Jacobian array is computed
    float *array_h = (float *)calloc(pointNumber, sizeof(float));
    double jacPenaltyTerm = 0.0;
    CUDA_SAFE_CALL(cudaMemcpy(array_h, jacobianDeterminantArray_d, pointNumber*sizeof(float), cudaMemcpyDeviceToHost));
    for(int i=0; i<pointNumber; i++)
        jacPenaltyTerm += array_h[i];
    free(array_h);
    if(approximate)
        jacPenaltyTerm /= (double)((velocityFieldImage->nx-2)*(velocityFieldImage->ny-2)*(velocityFieldImage->nz-2));
    else jacPenaltyTerm /= (double)pointNumber;


    CUDA_SAFE_CALL(cudaFree(voxelDisplacementField_d));
    CUDA_SAFE_CALL(cudaFree(controlPointArray_d));
    CUDA_SAFE_CALL(cudaFree(nodePositionArray_d));
    CUDA_SAFE_CALL(cudaFree(jacobianDeterminantArray_d));

    return jacPenaltyTerm;
}
/* *************************************************************** */
/* *************************************************************** */
void reg_bspline_ComputeJacGradientFromVelocity_gpu(nifti_image *targetImage,
                                                    nifti_image *velocityFieldImage,
                                                    float4 **velocityFieldImageArray_d,
                                                    float4 **gradientImageArray_d,
                                                    float jacobianWeight,
                                                    bool approximate)
{
    const int controlPointNumber = velocityFieldImage->nx*velocityFieldImage->ny*velocityFieldImage->nz;

    // A Jacobian array is allocated
    int pointNumber=0;
    if(approximate) pointNumber = controlPointNumber;
    else pointNumber = (int)targetImage->nvox;
    float *jacobianDeterminantArray_d;
    CUDA_SAFE_CALL(cudaMalloc(&jacobianDeterminantArray_d, pointNumber*sizeof(float)));

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&pointNumber,sizeof(int)));
    dim3 B1(Block_reg_bspline_SetJacDetToOne,1,1);
    dim3 G1((unsigned int)ceil((float)pointNumber/(float)(Block_reg_bspline_SetJacDetToOne)),1,1);
    reg_bspline_SetJacDetToOne_kernel <<< G1, B1 >>>(jacobianDeterminantArray_d);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
    printf("[NiftyReg CUDA DEBUG] _reg_bspline_SetJacobianDeterminantToOne_kernel kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
           cudaGetErrorString(cudaGetLastError()),G1.x,G1.y,G1.z,B1.x,B1.y,B1.z);
#endif

    // A displacement field array is allocated since the voxel position as to be always updated
    float4 *voxelDisplacementField_d;
    CUDA_SAFE_CALL(cudaMalloc(&voxelDisplacementField_d, pointNumber*sizeof(float4)));
    CUDA_SAFE_CALL(cudaMemset(voxelDisplacementField_d, 0, pointNumber*sizeof(float4)));

    // An array to contain the Jacobian matrix is allocated
    float *JacobianMatricesArray_d;
    CUDA_SAFE_CALL(cudaMalloc(&JacobianMatricesArray_d, 9*pointNumber*sizeof(float)));

    // Two temporary images are allocated on the card
    float4 *controlPointArray_d;
    float4 *nodePositionArray_d;
    CUDA_SAFE_CALL(cudaMalloc(&controlPointArray_d, controlPointNumber*sizeof(float4)));
    CUDA_SAFE_CALL(cudaMalloc(&nodePositionArray_d, controlPointNumber*sizeof(float4)));
    CUDA_SAFE_CALL(cudaMemcpy(nodePositionArray_d, *velocityFieldImageArray_d,
          controlPointNumber*sizeof(float4) , cudaMemcpyDeviceToDevice));

    // The velocity field is first deconvolved
    reg_spline_cppDeconvolve_gpu(   velocityFieldImage,
                                    velocityFieldImage,
                                    &nodePositionArray_d,   // input
                                    &controlPointArray_d);  // output


    // The Jacobian map is computed using assuming we deal with a control point position
    reg_bspline_ComputeJacobianGradientFromVelocityField(targetImage,
                                                        velocityFieldImage,
                                                        &controlPointArray_d,
                                                        &voxelDisplacementField_d,
                                                        &JacobianMatricesArray_d,
                                                        &jacobianDeterminantArray_d,
                                                        gradientImageArray_d,
                                                        jacobianWeight,
                                                        approximate);

    for(unsigned int i=1; i<SQUARING_VALUE; i++){
        reg_spline_cppComposition_gpu(  velocityFieldImage,
                                        velocityFieldImage,
                                        &nodePositionArray_d,   // deformed (output)
                                        &controlPointArray_d,    // used as a grid
                                        1.0,
                                        0);
        reg_spline_cppDeconvolve_gpu(   velocityFieldImage,
                                        velocityFieldImage,
                                        &nodePositionArray_d,
                                        &controlPointArray_d);
        reg_bspline_ComputeJacobianGradientFromVelocityField(targetImage,
                                                            velocityFieldImage,
                                                            &controlPointArray_d,
                                                            &voxelDisplacementField_d,
                                                            &JacobianMatricesArray_d,
                                                            &jacobianDeterminantArray_d,
                                                            gradientImageArray_d,
                                                            jacobianWeight,
                                                            approximate);
    }


    CUDA_SAFE_CALL(cudaFree(JacobianMatricesArray_d));
    CUDA_SAFE_CALL(cudaFree(voxelDisplacementField_d));
    CUDA_SAFE_CALL(cudaFree(controlPointArray_d));
    CUDA_SAFE_CALL(cudaFree(nodePositionArray_d));
    CUDA_SAFE_CALL(cudaFree(jacobianDeterminantArray_d));

    return;
}
/* *************************************************************** */
/* *************************************************************** */


#endif
