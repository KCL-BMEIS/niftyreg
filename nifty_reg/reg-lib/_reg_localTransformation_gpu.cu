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

#include "_reg_localTransformation_gpu.h"
#include "_reg_localTransformation_kernels.cu"

/* *************************************************************** */
/* *************************************************************** */
void reg_bspline_gpu(nifti_image *controlPointImage,
                     nifti_image *reference,
                     float4 **controlPointImageArray_d,
                     float4 **positionFieldImageArray_d,
                     int **mask_d,
                     int activeVoxelNumber,
                     bool bspline)
{
    const int voxelNumber = reference->nx * reference->ny * reference->nz;
    const int controlPointNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
    const int3 referenceImageDim = make_int3(reference->nx, reference->ny, reference->nz);
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    const int useBSpline = bspline;

    const float3 controlPointVoxelSpacing = make_float3(
        controlPointImage->dx / reference->dx,
        controlPointImage->dy / reference->dy,
        controlPointImage->dz / reference->dz);

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_UseBSpline,&useBSpline,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&voxelNumber,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ReferenceImageDim,&referenceImageDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointVoxelSpacing,&controlPointVoxelSpacing,sizeof(float3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ActiveVoxelNumber,&activeVoxelNumber,sizeof(int)))

    NR_CUDA_SAFE_CALL(cudaBindTexture(0, controlPointTexture, *controlPointImageArray_d, controlPointNumber*sizeof(float4)))
    NR_CUDA_SAFE_CALL(cudaBindTexture(0, maskTexture, *mask_d, activeVoxelNumber*sizeof(int)))

    const unsigned int Grid_reg_bspline_getDeformationField =
        (unsigned int)ceilf((float)activeVoxelNumber/(float)(Block_reg_bspline_getDeformationField));
    dim3 G1(Grid_reg_bspline_getDeformationField,1,1);
    dim3 B1(Block_reg_bspline_getDeformationField,1,1);
    reg_bspline_getDeformationField <<< G1, B1 >>>(*positionFieldImageArray_d);
    NR_CUDA_CHECK_KERNEL(G1,B1)

    NR_CUDA_SAFE_CALL(cudaUnbindTexture(controlPointTexture))
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(maskTexture))
    return;
}
/* *************************************************************** */
/* *************************************************************** */
float reg_bspline_ApproxBendingEnergy_gpu(nifti_image *controlPointImage,
                                          float4 **controlPointImageArray_d)
{
    const int controlPointNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    const int controlPointGridMem = controlPointNumber*sizeof(float4);

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointNumber,&controlPointNumber,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaBindTexture(0,controlPointTexture, *controlPointImageArray_d, controlPointGridMem))

    // First compute all the second derivatives
    float4 *secondDerivativeValues_d;
    NR_CUDA_SAFE_CALL(cudaMalloc(&secondDerivativeValues_d, 6*controlPointGridMem))
    const unsigned int Grid_bspline_getApproxSecondDerivatives =
        (unsigned int)ceilf((float)controlPointNumber/(float)(Block_reg_bspline_getApproxSecondDerivatives));
    dim3 G1(Grid_bspline_getApproxSecondDerivatives,1,1);
    dim3 B1(Block_reg_bspline_getApproxSecondDerivatives,1,1);
    reg_bspline_getApproxSecondDerivatives <<< G1, B1 >>>(secondDerivativeValues_d);
    NR_CUDA_CHECK_KERNEL(G1,B1)
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(controlPointTexture))

    // Compute the bending energy from the second derivatives
    float *penaltyTerm_d;
    NR_CUDA_SAFE_CALL(cudaMalloc(&penaltyTerm_d, controlPointNumber*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaBindTexture(0,secondDerivativesTexture,
                                      secondDerivativeValues_d,
                                      6*controlPointGridMem))
    const unsigned int Grid_reg_bspline_ApproxBendingEnergy =
        (unsigned int)ceilf((float)controlPointNumber/(float)(Block_reg_bspline_getApproxBendingEnergy));
    dim3 G2(Grid_reg_bspline_ApproxBendingEnergy,1,1);
    dim3 B2(Block_reg_bspline_getApproxBendingEnergy,1,1);
    reg_bspline_getApproxBendingEnergy_kernel <<< G2, B2 >>>(penaltyTerm_d);
    NR_CUDA_CHECK_KERNEL(G2,B2)
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(secondDerivativesTexture))
    NR_CUDA_SAFE_CALL(cudaFree(secondDerivativeValues_d))

    // Transfert the vales back to the CPU and average them
    float *penaltyTerm_h;
    NR_CUDA_SAFE_CALL(cudaMallocHost(&penaltyTerm_h, controlPointNumber*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaMemcpy(penaltyTerm_h, penaltyTerm_d, controlPointNumber*sizeof(float), cudaMemcpyDeviceToHost))
    NR_CUDA_SAFE_CALL(cudaFree(penaltyTerm_d))

    double penaltyValue=0.0;
    for(int i=0;i<controlPointNumber;i++)
            penaltyValue += penaltyTerm_h[i];
    NR_CUDA_SAFE_CALL(cudaFreeHost((void *)penaltyTerm_h))
    return (float)(penaltyValue/(3.0*(double)controlPointNumber));
}
/* *************************************************************** */
/* *************************************************************** */
void reg_bspline_ApproxBendingEnergyGradient_gpu(nifti_image *referenceImage,
                                                 nifti_image *controlPointImage,
                                                 float4 **controlPointImageArray_d,
                                                 float4 **nodeNMIGradientArray_d,
                                                 float bendingEnergyWeight)
{
    const int controlPointNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    const int controlPointGridMem = controlPointNumber*sizeof(float4);

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointNumber,&controlPointNumber,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaBindTexture(0,controlPointTexture, *controlPointImageArray_d, controlPointGridMem))

    // First compute all the second derivatives
    float4 *secondDerivativeValues_d;
    NR_CUDA_SAFE_CALL(cudaMalloc(&secondDerivativeValues_d, 6*controlPointNumber*sizeof(float4)))
    const unsigned int Grid_bspline_getApproxSecondDerivatives =
        (unsigned int)ceilf((float)controlPointNumber/(float)(Block_reg_bspline_getApproxSecondDerivatives));
    dim3 G1(Grid_bspline_getApproxSecondDerivatives,1,1);
    dim3 B1(Block_reg_bspline_getApproxSecondDerivatives,1,1);
    reg_bspline_getApproxSecondDerivatives <<< G1, B1 >>>(secondDerivativeValues_d);
    NR_CUDA_CHECK_KERNEL(G1,B1)
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(controlPointTexture))

    // Compute the gradient
    bendingEnergyWeight *= referenceImage->nx*referenceImage->ny*referenceImage->nz /
                           (controlPointImage->nx*controlPointImage->ny*controlPointImage->nz);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_Weight,&bendingEnergyWeight,sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaBindTexture(0,secondDerivativesTexture,
                                      secondDerivativeValues_d,
                                      6*controlPointNumber*sizeof(float4)))
    const unsigned int Grid_reg_bspline_getApproxBendingEnergyGradient =
        (unsigned int)ceilf((float)controlPointNumber/(float)(Block_reg_bspline_getApproxBendingEnergyGradient));
    dim3 G2(Grid_reg_bspline_getApproxBendingEnergyGradient,1,1);
    dim3 B2(Block_reg_bspline_getApproxBendingEnergyGradient,1,1);
    reg_bspline_getApproxBendingEnergyGradient_kernel <<< G2, B2 >>>(*nodeNMIGradientArray_d);
    NR_CUDA_CHECK_KERNEL(G2,B2)
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(secondDerivativesTexture))

    return;
}
/* *************************************************************** */
/* *************************************************************** */
void reg_bspline_ComputeApproxJacobianValues(nifti_image *controlPointImage,
                                             float4 **controlPointImageArray_d,
                                             float **jacobianMatrices_d,
                                             float **jacobianDet_d)
{
    // Need to reorient the Jacobian matrix using the header information - real to voxel conversion
    mat33 reorient, desorient;
    reg_getReorientationMatrix(controlPointImage, &desorient, &reorient);
    float3 temp=make_float3(reorient.m[0][0],reorient.m[0][1],reorient.m[0][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix0,&temp,sizeof(float3)))
    temp=make_float3(reorient.m[1][0],reorient.m[1][1],reorient.m[1][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix1,&temp,sizeof(float3)))
    temp=make_float3(reorient.m[2][0],reorient.m[2][1],reorient.m[2][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix2,&temp,sizeof(float3)))

    // Bind some variables
    const int controlPointNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    const float3 controlPointSpacing = make_float3(controlPointImage->dx,controlPointImage->dy,controlPointImage->dz);
    const int controlPointGridMem = controlPointNumber*sizeof(float4);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointNumber,&controlPointNumber,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointSpacing,&controlPointSpacing,sizeof(float3)))
    NR_CUDA_SAFE_CALL(cudaBindTexture(0,controlPointTexture, *controlPointImageArray_d, controlPointGridMem))

    // The Jacobian matrix is computed for every control point
    const unsigned int Grid_reg_bspline_getApproxJacobianValues =
        (unsigned int)ceilf((float)controlPointNumber/(float)(Block_reg_bspline_getApproxJacobianValues));
    dim3 G1(Grid_reg_bspline_getApproxJacobianValues,1,1);
    dim3 B1(Block_reg_bspline_getApproxJacobianValues,1,1);
    reg_bspline_getApproxJacobianValues_kernel<<< G1, B1>>>(*jacobianMatrices_d, *jacobianDet_d);
    NR_CUDA_CHECK_KERNEL(G1,B1)
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(controlPointTexture))
}
/* *************************************************************** */
void reg_bspline_ComputeJacobianValues(nifti_image *controlPointImage,
                                       nifti_image *referenceImage,
                                       float4 **controlPointImageArray_d,
                                       float **jacobianMatrices_d,
                                       float **jacobianDet_d)
{
    // Need to reorient the Jacobian matrix using the header information - real to voxel conversion
    mat33 reorient, desorient;
    reg_getReorientationMatrix(controlPointImage, &desorient, &reorient);
    float3 temp=make_float3(reorient.m[0][0],reorient.m[0][1],reorient.m[0][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix0,&temp,sizeof(float3)))
    temp=make_float3(reorient.m[1][0],reorient.m[1][1],reorient.m[1][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix1,&temp,sizeof(float3)))
    temp=make_float3(reorient.m[2][0],reorient.m[2][1],reorient.m[2][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix2,&temp,sizeof(float3)))

    // Bind some variables
    const int voxelNumber = referenceImage->nx*referenceImage->ny*referenceImage->nz;
    const int controlPointNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
    const int3 referenceImageDim = make_int3(referenceImage->nx, referenceImage->ny, referenceImage->nz);
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    const float3 controlPointSpacing = make_float3(controlPointImage->dx,controlPointImage->dy,controlPointImage->dz);
    const float3 controlPointVoxelSpacing = make_float3(
            controlPointImage->dx / referenceImage->dx,
            controlPointImage->dy / referenceImage->dy,
            controlPointImage->dz / referenceImage->dz);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&voxelNumber,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointNumber,&controlPointNumber,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ReferenceImageDim,&referenceImageDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointSpacing,&controlPointSpacing,sizeof(float3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointVoxelSpacing,&controlPointVoxelSpacing,sizeof(float3)))
    NR_CUDA_SAFE_CALL(cudaBindTexture(0,controlPointTexture, *controlPointImageArray_d, controlPointNumber*sizeof(float4)))

    // The Jacobian matrix is computed for every voxel
    const unsigned int Grid_reg_bspline_getJacobianValues =
        (unsigned int)ceilf((float)voxelNumber/(float)(Block_reg_bspline_getJacobianValues));
    dim3 G1(Grid_reg_bspline_getJacobianValues,1,1);
    dim3 B1(Block_reg_bspline_getJacobianValues,1,1);
    reg_bspline_getJacobianValues_kernel<<< G1, B1>>>(*jacobianMatrices_d, *jacobianDet_d);
    NR_CUDA_CHECK_KERNEL(G1,B1)
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(controlPointTexture))
}
/* *************************************************************** */
/* *************************************************************** */
double reg_bspline_ComputeJacobianPenaltyTerm_gpu(nifti_image *referenceImage,
                                                  nifti_image *controlPointImage,
                                                  float4 **controlPointImageArray_d,
                                                  bool approx
                                                  )
{
    // The Jacobian matrices and determinants are computed
    float *jacobianMatrices_d;
    float *jacobianDet_d;
    int jacNumber;
    double jacSum;
    if(approx){
        jacNumber=controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
        jacSum=(controlPointImage->nx-2)*(controlPointImage->ny-2)*(controlPointImage->nz-2);
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatrices_d,9*jacNumber*sizeof(float)))
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDet_d,jacNumber*sizeof(float)))
        reg_bspline_ComputeApproxJacobianValues(controlPointImage,
                                                controlPointImageArray_d,
                                                &jacobianMatrices_d,
                                                &jacobianDet_d);
    }
    else{
        jacNumber=referenceImage->nx*referenceImage->ny*referenceImage->nz;
        jacSum=jacNumber;
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatrices_d,9*jacNumber*sizeof(float)))
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDet_d,jacNumber*sizeof(float)))
        reg_bspline_ComputeJacobianValues(controlPointImage,
                                          referenceImage,
                                          controlPointImageArray_d,
                                          &jacobianMatrices_d,
                                          &jacobianDet_d);
    }
    NR_CUDA_SAFE_CALL(cudaFree(jacobianMatrices_d))

    // The Jacobian determinant are squared and logged (might not be english but will do)
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&jacNumber,sizeof(int)))
    const unsigned int Grid_reg_bspline_logSquaredValues =
        (unsigned int)ceilf((float)jacNumber/(float)(Block_reg_bspline_logSquaredValues));
    dim3 G1(Grid_reg_bspline_logSquaredValues,1,1);
    dim3 B1(Block_reg_bspline_logSquaredValues,1,1);
    reg_bspline_logSquaredValues_kernel<<< G1, B1>>>(jacobianDet_d);
    NR_CUDA_CHECK_KERNEL(G1,B1)
    // Transfert the data back to the CPU
    float *jacobianDet_h;
    NR_CUDA_SAFE_CALL(cudaMallocHost(&jacobianDet_h,jacNumber*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaMemcpy(jacobianDet_h,jacobianDet_d,
                                 jacNumber*sizeof(float),
                                 cudaMemcpyDeviceToHost))
    NR_CUDA_SAFE_CALL(cudaFree(jacobianDet_d))
    double penaltyTermValue=0.;
    for(int i=0;i<jacNumber;++i)
        penaltyTermValue += jacobianDet_h[i];
    NR_CUDA_SAFE_CALL(cudaFreeHost(jacobianDet_h))
    return penaltyTermValue/jacSum;
}
/* *************************************************************** */
void reg_bspline_ComputeJacobianPenaltyTermGradient_gpu(nifti_image *referenceImage,
                                                        nifti_image *controlPointImage,
                                                        float4 **controlPointImageArray_d,
                                                        float4 **nodeNMIGradientArray_d,
                                                        float jacobianWeight,
                                                        bool approx)
{
    // The Jacobian matrices and determinants are computed
    float *jacobianMatrices_d;
    float *jacobianDet_d;
    int jacNumber;
    if(approx){
        jacNumber=controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatrices_d,9*jacNumber*sizeof(float)))
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDet_d,jacNumber*sizeof(float)))
        reg_bspline_ComputeApproxJacobianValues(controlPointImage,
                                                controlPointImageArray_d,
                                                &jacobianMatrices_d,
                                                &jacobianDet_d);
    }
    else{
        jacNumber=referenceImage->nx*referenceImage->ny*referenceImage->nz;
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatrices_d,9*jacNumber*sizeof(float)))
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDet_d,jacNumber*sizeof(float)))
        reg_bspline_ComputeJacobianValues(controlPointImage,
                                          referenceImage,
                                          controlPointImageArray_d,
                                          &jacobianMatrices_d,
                                          &jacobianDet_d);
    }

    // Need to desorient the Jacobian matrix using the header information - voxel to real conversion
    mat33 reorient, desorient;
    reg_getReorientationMatrix(controlPointImage, &desorient, &reorient);
    float3 temp=make_float3(desorient.m[0][0],desorient.m[0][1],desorient.m[0][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix0,&temp,sizeof(float3)))
    temp=make_float3(desorient.m[1][0],desorient.m[1][1],desorient.m[1][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix1,&temp,sizeof(float3)))
    temp=make_float3(desorient.m[2][0],desorient.m[2][1],desorient.m[2][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix2,&temp,sizeof(float3)))

    NR_CUDA_SAFE_CALL(cudaBindTexture(0,jacobianDeterminantTexture, jacobianDet_d,
                                      jacNumber*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaBindTexture(0,jacobianMatricesTexture, jacobianMatrices_d,
                                      9*jacNumber*sizeof(float)))

    // Bind some variables
    const int controlPointNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    const float3 controlPointSpacing = make_float3(controlPointImage->dx,controlPointImage->dy,controlPointImage->dz);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointNumber,&controlPointNumber,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointSpacing,&controlPointSpacing,sizeof(float3)))
    if(approx){
        float weight=jacobianWeight;
        weight = jacobianWeight * (float)(referenceImage->nx * referenceImage->ny * referenceImage->nz)
                 / (float)( controlPointImage->nx*controlPointImage->ny*controlPointImage->nz);
        NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_Weight,&weight,sizeof(float)))
        const unsigned int Grid_reg_bspline_computeApproxJacGradient =
            (unsigned int)ceilf((float)controlPointNumber/(float)(Block_reg_bspline_computeApproxJacGradient));
        dim3 G1(Grid_reg_bspline_computeApproxJacGradient,1,1);
        dim3 B1(Block_reg_bspline_computeApproxJacGradient,1,1);
        reg_bspline_computeApproxJacGradient_kernel<<< G1, B1>>>(*nodeNMIGradientArray_d);
        NR_CUDA_CHECK_KERNEL(G1,B1)
    }
    else{
        const int voxelNumber = referenceImage->nx*referenceImage->ny*referenceImage->nz;
        const int3 referenceImageDim = make_int3(referenceImage->nx, referenceImage->ny, referenceImage->nz);
        const float3 controlPointVoxelSpacing = make_float3(
                controlPointImage->dx / referenceImage->dx,
                controlPointImage->dy / referenceImage->dy,
                controlPointImage->dz / referenceImage->dz);
        NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&voxelNumber,sizeof(int)))
        NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ReferenceImageDim,&referenceImageDim,sizeof(int3)))
        NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointVoxelSpacing,&controlPointVoxelSpacing,sizeof(float3)))
        NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_Weight,&jacobianWeight,sizeof(float)))
        const unsigned int Grid_reg_bspline_computeJacGradient =
            (unsigned int)ceilf((float)controlPointNumber/(float)(Block_reg_bspline_computeJacGradient));
        dim3 G1(Grid_reg_bspline_computeJacGradient,1,1);
        dim3 B1(Block_reg_bspline_computeJacGradient,1,1);
        reg_bspline_computeJacGradient_kernel<<< G1, B1>>>(*nodeNMIGradientArray_d);
        NR_CUDA_CHECK_KERNEL(G1,B1)
    }
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(jacobianDeterminantTexture))
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(jacobianMatricesTexture))
    NR_CUDA_SAFE_CALL(cudaFree(jacobianDet_d))
    NR_CUDA_SAFE_CALL(cudaFree(jacobianMatrices_d))
}
/* *************************************************************** */
double reg_bspline_correctFolding_gpu(nifti_image *referenceImage,
                                      nifti_image *controlPointImage,
                                      float4 **controlPointImageArray_d,
                                      bool approx)
{
    // The Jacobian matrices and determinants are computed
    float *jacobianMatrices_d;
    float *jacobianDet_d;
    int jacNumber;
    double jacSum;
    if(approx){
        jacNumber=controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
        jacSum = (controlPointImage->nx-2)*(controlPointImage->ny-2)*(controlPointImage->nz-2);
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatrices_d,9*jacNumber*sizeof(float)))
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDet_d,jacNumber*sizeof(float)))
        reg_bspline_ComputeApproxJacobianValues(controlPointImage,
                                                controlPointImageArray_d,
                                                &jacobianMatrices_d,
                                                &jacobianDet_d);
    }
    else{
        jacSum=jacNumber=referenceImage->nx*referenceImage->ny*referenceImage->nz;
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatrices_d,9*jacNumber*sizeof(float)))
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDet_d,jacNumber*sizeof(float)))
        reg_bspline_ComputeJacobianValues(controlPointImage,
                                          referenceImage,
                                          controlPointImageArray_d,
                                          &jacobianMatrices_d,
                                          &jacobianDet_d);
    }

    // Check if the Jacobian determinant average
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&jacNumber,sizeof(int)))
    float *jacobianDet2_d;
    NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDet2_d,jacNumber*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaMemcpy(jacobianDet2_d,jacobianDet_d,jacNumber*sizeof(float),cudaMemcpyDeviceToDevice))
    const unsigned int Grid_reg_bspline_logSquaredValues =
        (unsigned int)ceilf((float)jacNumber/(float)(Block_reg_bspline_logSquaredValues));
    dim3 G1(Grid_reg_bspline_logSquaredValues,1,1);
    dim3 B1(Block_reg_bspline_logSquaredValues,1,1);
    reg_bspline_logSquaredValues_kernel<<< G1, B1>>>(jacobianDet2_d);
    NR_CUDA_CHECK_KERNEL(G1,B1)
    float *jacobianDet_h;
    NR_CUDA_SAFE_CALL(cudaMallocHost(&jacobianDet_h,jacNumber*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaMemcpy(jacobianDet_h,jacobianDet2_d,
                                 jacNumber*sizeof(float),
                                 cudaMemcpyDeviceToHost))
    NR_CUDA_SAFE_CALL(cudaFree(jacobianDet2_d))
    double penaltyTermValue=0.;
    for(int i=0;i<jacNumber;++i) penaltyTermValue += jacobianDet_h[i];
    NR_CUDA_SAFE_CALL(cudaFreeHost(jacobianDet_h))
    penaltyTermValue /= jacSum;
    printf("value %g\n", penaltyTermValue);
    if(penaltyTermValue==penaltyTermValue){
        NR_CUDA_SAFE_CALL(cudaFree(jacobianDet_d))
        NR_CUDA_SAFE_CALL(cudaFree(jacobianMatrices_d))
        return penaltyTermValue;
    }

    // Need to desorient the Jacobian matrix using the header information - voxel to real conversion
    mat33 reorient, desorient;
    reg_getReorientationMatrix(controlPointImage, &desorient, &reorient);
    float3 temp=make_float3(desorient.m[0][0],desorient.m[0][1],desorient.m[0][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix0,&temp,sizeof(float3)))
    temp=make_float3(desorient.m[1][0],desorient.m[1][1],desorient.m[1][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix1,&temp,sizeof(float3)))
    temp=make_float3(desorient.m[2][0],desorient.m[2][1],desorient.m[2][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix2,&temp,sizeof(float3)))

    NR_CUDA_SAFE_CALL(cudaBindTexture(0,jacobianDeterminantTexture, jacobianDet_d,
                                      jacNumber*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaBindTexture(0,jacobianMatricesTexture, jacobianMatrices_d,
                                      9*jacNumber*sizeof(float)))

    // Bind some variables
    const int controlPointNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    const float3 controlPointSpacing = make_float3(controlPointImage->dx,controlPointImage->dy,controlPointImage->dz);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointNumber,&controlPointNumber,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointSpacing,&controlPointSpacing,sizeof(float3)))
    if(approx){
        const unsigned int Grid_reg_bspline_approxCorrectFolding =
            (unsigned int)ceilf((float)controlPointNumber/(float)(Block_reg_bspline_approxCorrectFolding));
        dim3 G1(Grid_reg_bspline_approxCorrectFolding,1,1);
        dim3 B1(Block_reg_bspline_approxCorrectFolding,1,1);
        reg_bspline_approxCorrectFolding_kernel<<< G1, B1>>>(*controlPointImageArray_d);
        NR_CUDA_CHECK_KERNEL(G1,B1)
    }
    else{
        const int voxelNumber = referenceImage->nx*referenceImage->ny*referenceImage->nz;
        const int3 referenceImageDim = make_int3(referenceImage->nx, referenceImage->ny, referenceImage->nz);
        const float3 controlPointVoxelSpacing = make_float3(
                controlPointImage->dx / referenceImage->dx,
                controlPointImage->dy / referenceImage->dy,
                controlPointImage->dz / referenceImage->dz);
        NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&voxelNumber,sizeof(int)))
        NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ReferenceImageDim,&referenceImageDim,sizeof(int3)))
        NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointVoxelSpacing,&controlPointVoxelSpacing,sizeof(float3)))
        const unsigned int Grid_reg_bspline_correctFolding =
        (unsigned int)ceilf((float)controlPointNumber/(float)(Block_reg_bspline_correctFolding));
        dim3 G1(Grid_reg_bspline_correctFolding,1,1);
        dim3 B1(Block_reg_bspline_correctFolding,1,1);
        reg_bspline_correctFolding_kernel<<< G1, B1>>>(*controlPointImageArray_d);
        NR_CUDA_CHECK_KERNEL(G1,B1)
    }
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(jacobianDeterminantTexture))
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(jacobianMatricesTexture))
    NR_CUDA_SAFE_CALL(cudaFree(jacobianDet_d))
    NR_CUDA_SAFE_CALL(cudaFree(jacobianMatrices_d))
    return std::numeric_limits<double>::quiet_NaN();
}
/* *************************************************************** */

#endif
