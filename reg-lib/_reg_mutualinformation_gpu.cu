/*
 *  _reg_mutualinformation_gpu.cu
 *
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_MUTUALINFORMATION_GPU_CU
#define _REG_MUTUALINFORMATION_GPU_CU

#include "_reg_blocksize_gpu.h"
#include "_reg_mutualinformation_gpu.h"
#include "_reg_mutualinformation_kernels.cu"

/// Called when we only have one target and one source image
void reg_getVoxelBasedNMIGradientUsingPW_gpu(   nifti_image *targetImage,
                                                nifti_image *resultImage,
                                                float **targetImageArray_d,
                                                float **resultImageArray_d,
                                                float4 **resultGradientArray_d,
                                                float **logJointHistogram_d,
                                                float4 **voxelNMIGradientArray_d,
                                                int **mask_d,
                                                int activeVoxelNumber,
                                                double *entropies,
                                                int refBinning,
                                                int floBinning)
{
    const int voxelNumber = targetImage->nx*targetImage->ny*targetImage->nz;
    const int binNumber = refBinning*floBinning+refBinning+floBinning;
    const float4 entropies_h=make_float4((float)entropies[0],(float)entropies[1],(float)entropies[2],(float)entropies[3]);
    const float NMI = (float)((entropies[0]+entropies[1])/entropies[2]);

    // Bind Symbols
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&voxelNumber,sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_firstTargetBin,&refBinning,sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_firstResultBin,&floBinning,sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_Entropies,&entropies_h,sizeof(float4)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_NMI,&NMI,sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ActiveVoxelNumber,&activeVoxelNumber,sizeof(int)));

    // Texture bindingcurrentFloating
    CUDA_SAFE_CALL(cudaBindTexture(0, firstTargetImageTexture, *targetImageArray_d, voxelNumber*sizeof(float)));
    CUDA_SAFE_CALL(cudaBindTexture(0, firstResultImageTexture, *resultImageArray_d, voxelNumber*sizeof(float)));
    CUDA_SAFE_CALL(cudaBindTexture(0, firstResultImageGradientTexture, *resultGradientArray_d, voxelNumber*sizeof(float4)));
    CUDA_SAFE_CALL(cudaBindTexture(0, histogramTexture, *logJointHistogram_d, binNumber*sizeof(float)));
    CUDA_SAFE_CALL(cudaBindTexture(0, maskTexture, *mask_d, activeVoxelNumber*sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(*voxelNMIGradientArray_d, 0, voxelNumber*sizeof(float4)));

    const unsigned int Grid_reg_getVoxelBasedNMIGradientUsingPW =
        (unsigned int)ceil((float)activeVoxelNumber/(float)Block_reg_getVoxelBasedNMIGradientUsingPW);
    dim3 B1(Block_reg_getVoxelBasedNMIGradientUsingPW,1,1);
    dim3 G1(Grid_reg_getVoxelBasedNMIGradientUsingPW,1,1);

    reg_getVoxelBasedNMIGradientUsingPW_kernel <<< G1, B1 >>> (*voxelNMIGradientArray_d);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
    printf("[NiftyReg CUDA DEBUG] reg_getVoxelBasedNMIGradientUsingPW_kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
           cudaGetErrorString(cudaGetLastError()),G1.x,G1.y,G1.z,B1.x,B1.y,B1.z);
#endif
}
void reg_getVoxelBasedNMIGradientUsingPW2x2_gpu(nifti_image *targetImage,
                                                nifti_image *resultImage,
                                                float **targetImageArray1_d,
                                                float **targetImageArray2_d,
                                                float **resultImageArray1_d,
                                                float **resultImageArray2_d,
                                                float4 **resultGradientArray1_d,
                                                float4 **resultGradientArray2_d,
                                                float **logJointHistogram_d,
                                                float4 **voxelNMIGradientArray_d,
                                                int **mask_d,
                                                int activeVoxelNumber,
                                                double *entropies,
                                                unsigned int *targetBinning,
                                                unsigned int *resultBinning)
{
    if (targetImage->nt != 2 || resultImage->nt != 2) {
        printf("[NiftyReg CUDA] reg_getVoxelBasedNMIGradientUsingPW2x2_gpu: This kernel should only be used with two target and source images\n");
        return;
    }
    const int voxelNumber = targetImage->nx*targetImage->ny*targetImage->nz;
    const float4 entropies_h=make_float4((float)entropies[0],(float)entropies[1],(float)entropies[2],(float)entropies[3]);
    const float NMI = (float)((entropies[0]+entropies[1])/entropies[2]);
    const int binNumber = targetBinning[0]*targetBinning[1]*resultBinning[0]*resultBinning[1] + (targetBinning[0]*targetBinning[1]) + (resultBinning[0]*resultBinning[1]);

    // Bind Symbols
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&voxelNumber,sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_firstTargetBin,&targetBinning[0],sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_secondTargetBin,&targetBinning[1],sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_firstResultBin,&resultBinning[0],sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_secondResultBin,&resultBinning[1],sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_Entropies,&entropies_h,sizeof(float4)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_NMI,&NMI,sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ActiveVoxelNumber,&activeVoxelNumber,sizeof(int)));

    // Texture binding
    CUDA_SAFE_CALL(cudaBindTexture(0, firstTargetImageTexture, *targetImageArray1_d, voxelNumber*sizeof(float)));
    CUDA_SAFE_CALL(cudaBindTexture(0, secondTargetImageTexture, *targetImageArray2_d, voxelNumber*sizeof(float)));
    CUDA_SAFE_CALL(cudaBindTexture(0, firstResultImageTexture, *resultImageArray1_d, voxelNumber*sizeof(float)));
    CUDA_SAFE_CALL(cudaBindTexture(0, secondResultImageTexture, *resultImageArray2_d, voxelNumber*sizeof(float)));
    CUDA_SAFE_CALL(cudaBindTexture(0, firstResultImageGradientTexture, *resultGradientArray1_d, voxelNumber*sizeof(float4)));
    CUDA_SAFE_CALL(cudaBindTexture(0, secondResultImageGradientTexture, *resultGradientArray2_d, voxelNumber*sizeof(float4)));
    CUDA_SAFE_CALL(cudaBindTexture(0, histogramTexture, *logJointHistogram_d, binNumber*sizeof(float)));
    CUDA_SAFE_CALL(cudaBindTexture(0, maskTexture, *mask_d, activeVoxelNumber*sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(*voxelNMIGradientArray_d, 0, voxelNumber*sizeof(float4)));

    const unsigned int Grid_reg_getVoxelBasedNMIGradientUsingPW2x2 =
        (unsigned int)ceil((float)activeVoxelNumber/(float)Block_reg_getVoxelBasedNMIGradientUsingPW2x2);
    dim3 B1(Block_reg_getVoxelBasedNMIGradientUsingPW2x2,1,1);
    dim3 G1(Grid_reg_getVoxelBasedNMIGradientUsingPW2x2,1,1);

    reg_getVoxelBasedNMIGradientUsingPW_kernel2x2 <<< G1, B1 >>> (*voxelNMIGradientArray_d);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
    printf("[NiftyReg CUDA DEBUG] reg_getVoxelBasedNMIGradientUsingPW2x2_gpu: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
           cudaGetErrorString(cudaGetLastError()),G1.x,G1.y,G1.z,B1.x,B1.y,B1.z);
#endif
}

#endif
