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

#include <iostream>


/// Called when we have two target and two source image
void reg_getEntropies2x2_gpu(nifti_image *targetImages,
                             nifti_image *resultImages,
                             int type,
                             unsigned int *target_bins, // should be an array of size num_target_volumes
                             unsigned int *result_bins, // should be an array of size num_result_volumes
                             double *probaJointHistogram,
                             double *logJointHistogram,
                             float  **logJointHistogram_d,
                             double *entropies,
                             int *mask)
{
    // The joint histogram is filled using the CPU arrays
    //Check the type of the target and source images
    if(targetImages->datatype!=NIFTI_TYPE_FLOAT32 || resultImages->datatype!=NIFTI_TYPE_FLOAT32){
        printf("[NiftyReg CUDA] reg_getEntropies2x2_gpu: This kernel should only be used floating images.\n");
        exit(1);
    }
    unsigned int voxelNumber = targetImages->nx*targetImages->ny*targetImages->nz;
    unsigned int binNumber = target_bins[0]*target_bins[1]*result_bins[0]*result_bins[1]+
                             target_bins[0]*target_bins[1]+result_bins[0]*result_bins[1];
    float *ref1Ptr = static_cast<float *>(targetImages->data);
    float *ref2Ptr = &ref1Ptr[voxelNumber];
    float *res1Ptr = static_cast<float *>(resultImages->data);
    float *res2Ptr = &res1Ptr[voxelNumber];
    int *maskPtr = &mask[0];
    memset(probaJointHistogram, 0, binNumber*sizeof(double));
    double voxelSum=0.;
    for(unsigned int i=0;i<voxelNumber;++i){
        if(*maskPtr++>-1){
            int val1 = static_cast<int>(*ref1Ptr);
            int val2 = static_cast<int>(*ref2Ptr);
            int val3 = static_cast<int>(*res1Ptr);
            int val4 = static_cast<int>(*res2Ptr);
            if(val1==val1 && val2==val2 && val3==val3 && val4==val4 &&
               val1>-1 && val1<(int)target_bins[0] && val2>-1 && val2<(int)target_bins[1] &&
               val3>-1 && val3<(int)result_bins[0] && val4>-1 && val4<(int)result_bins[1]){
                unsigned int index = ((val4*result_bins[0]+val3)*target_bins[1]+val2)*target_bins[0]+val1;
                probaJointHistogram[index]++;
                voxelSum++;
            }
        }
        ref1Ptr++;
        ref2Ptr++;
        res1Ptr++;
        res2Ptr++;
    }

    // The joint histogram is normalised and tranfered to the device
    float *logJointHistogram_float=NULL;
    CUDA_SAFE_CALL(cudaMallocHost(&logJointHistogram_float,binNumber*sizeof(float)));
    for(unsigned int i=0;i<target_bins[0]*target_bins[1]*result_bins[0]*result_bins[1];++i)
        logJointHistogram_float[i]=float(probaJointHistogram[i]/voxelSum);

    CUDA_SAFE_CALL(cudaMemcpy(*logJointHistogram_d,logJointHistogram_float,binNumber*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaFreeHost(logJointHistogram_float));

    float *tempHistogram=NULL;
    CUDA_SAFE_CALL(cudaMalloc(&tempHistogram,binNumber*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_firstTargetBin,&target_bins[0],sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_secondTargetBin,&target_bins[1],sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_firstResultBin,&result_bins[0],sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_secondResultBin,&result_bins[1],sizeof(int)));


    // The joint histogram is smoothed along the x axis
    CUDA_SAFE_CALL(cudaBindTexture(0, histogramTexture, *logJointHistogram_d, binNumber*sizeof(float)));
    dim3 B1(Block_reg_smoothJointHistogramX,1,1);
    dim3 G1((int)ceil((float)(target_bins[1]*result_bins[0]*result_bins[1])/(float)B1.x),1,1);
    reg_smoothJointHistogramX_kernel <<< G1, B1 >>> (tempHistogram);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
    printf("[NiftyReg CUDA DEBUG] reg_smoothJointHistogramX_kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
           cudaGetErrorString(cudaGetLastError()),G1.x,G1.y,G1.z,B1.x,B1.y,B1.z);
#endif

    // The joint histogram is smoothed along the y axis
    CUDA_SAFE_CALL(cudaBindTexture(0, histogramTexture, tempHistogram, binNumber*sizeof(float)));
    dim3 B2(Block_reg_smoothJointHistogramY,1,1);
    dim3 G2((int)ceil((float)(target_bins[0]*result_bins[0]*result_bins[1])/(float)B2.x),1,1);
    reg_smoothJointHistogramY_kernel <<< G2, B2 >>> (*logJointHistogram_d);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
    printf("[NiftyReg CUDA DEBUG] reg_smoothJointHistogramY_kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
           cudaGetErrorString(cudaGetLastError()),G2.x,G2.y,G2.z,B2.x,B2.y,B2.z);
#endif

    // The joint histogram is smoothed along the z axis
    CUDA_SAFE_CALL(cudaBindTexture(0, histogramTexture, *logJointHistogram_d, binNumber*sizeof(float)));
    dim3 B3(Block_reg_smoothJointHistogramZ,1,1);
    dim3 G3((int)ceil((float)(target_bins[0]*target_bins[1]*result_bins[1])/(float)B3.x),1,1);
    reg_smoothJointHistogramZ_kernel <<< G3, B3 >>> (tempHistogram);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
    printf("[NiftyReg CUDA DEBUG] reg_smoothJointHistogramZ_kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
           cudaGetErrorString(cudaGetLastError()),G3.x,G3.y,G3.z,B3.x,B3.y,B3.z);
#endif

    // The joint histogram is smoothed along the w axis
    CUDA_SAFE_CALL(cudaBindTexture(0, histogramTexture, tempHistogram, binNumber*sizeof(float)));
    dim3 B4(Block_reg_smoothJointHistogramW,1,1);
    dim3 G4((int)ceil((float)(target_bins[0]*target_bins[1]*result_bins[0])/(float)B4.x),1,1);
    reg_smoothJointHistogramW_kernel <<< G4, B4 >>> (*logJointHistogram_d);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
    printf("[NiftyReg CUDA DEBUG] reg_smoothJointHistogramW_kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
           cudaGetErrorString(cudaGetLastError()),G4.x,G4.y,G4.z,B4.x,B4.y,B4.z);
#endif
    CUDA_SAFE_CALL(cudaFree(tempHistogram));
    CUDA_SAFE_CALL(cudaMallocHost(&logJointHistogram_float,binNumber*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpy(logJointHistogram_float,*logJointHistogram_d,binNumber*sizeof(float),cudaMemcpyDeviceToHost));
    for(unsigned int i=0;i<target_bins[0]*target_bins[1]*result_bins[0]*result_bins[1];++i)
        probaJointHistogram[i]=logJointHistogram_float[i];
    CUDA_SAFE_CALL(cudaFreeHost(logJointHistogram_float));

    // The 4D joint histogram is first marginalised along the x axis (target_bins[0])
    float *temp3DHistogram=NULL;
    CUDA_SAFE_CALL(cudaMalloc(&temp3DHistogram,target_bins[1]*result_bins[0]*result_bins[1]*sizeof(float)));
    CUDA_SAFE_CALL(cudaBindTexture(0, histogramTexture, *logJointHistogram_d, binNumber*sizeof(float)));
    dim3 B5(Block_reg_marginaliseTargetX,1,1);
    dim3 G5((int)ceil((float)(target_bins[1]*result_bins[0]*result_bins[1])/(float)B5.x),1,1);
    reg_marginaliseTargetX_kernel <<< G5, B5 >>> (temp3DHistogram);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
    printf("[NiftyReg CUDA DEBUG] reg_marginaliseTargetX_kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
           cudaGetErrorString(cudaGetLastError()),G5.x,G5.y,G5.z,B5.x,B5.y,B5.z);
#endif

    // The 3D joint histogram is then marginalised along the y axis (target_bins[1])
    float *temp2DHistogram=NULL;
    CUDA_SAFE_CALL(cudaMalloc(&temp2DHistogram,result_bins[0]*result_bins[1]*sizeof(float)));
    CUDA_SAFE_CALL(cudaBindTexture(0, histogramTexture, temp3DHistogram, target_bins[1]*result_bins[0]*result_bins[1]*sizeof(float)));
    dim3 B6(Block_reg_marginaliseTargetXY,1,1);
    dim3 G6((int)ceil((float)(result_bins[0]*result_bins[1])/(float)B6.x),1,1);
    reg_marginaliseTargetXY_kernel <<< G6, B6 >>> (temp2DHistogram);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
    printf("[NiftyReg CUDA DEBUG] reg_marginaliseTargetXY_kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
           cudaGetErrorString(cudaGetLastError()),G6.x,G6.y,G6.z,B6.x,B6.y,B6.z);
#endif    
    CUDA_SAFE_CALL(cudaFree(temp3DHistogram));

    // We need to transfer it to an array of floats (cannot directly copy it to probaJointHistogram
    // as that is an array of doubles) and cudaMemcpy will produce unpredictable results
    const int total_target_entries = target_bins[0] * target_bins[1];
    const int total_result_entries = result_bins[0] * result_bins[1];
    const int num_probabilities =  total_target_entries * total_result_entries;
    int offset = num_probabilities + total_target_entries;    
    float *temp2DHistogram_h = new float[total_result_entries];
    cudaMemcpy(temp2DHistogram_h,temp2DHistogram,total_result_entries*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < total_result_entries; ++i) {
        probaJointHistogram[offset + i] = temp2DHistogram_h[i];
    }
    delete[] temp2DHistogram_h;
    CUDA_SAFE_CALL(cudaFree(temp2DHistogram));


    // Now marginalise over the result axes.
    // First over W axes. (result_bins[1])
    temp3DHistogram=NULL;
    CUDA_SAFE_CALL(cudaMalloc(&temp3DHistogram, target_bins[0]*target_bins[1]*result_bins[0]*sizeof(float)));
    CUDA_SAFE_CALL(cudaBindTexture(0, histogramTexture, *logJointHistogram_d, binNumber*sizeof(float)));
    dim3 B7(Block_reg_marginaliseResultX,1,1);
    dim3 G7((int)ceil((float)(target_bins[0]*target_bins[1]*result_bins[0])/(float)B7.x),1,1);
    reg_marginaliseResultX_kernel <<< G7, B7 >>> (temp3DHistogram);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
    printf("[NiftyReg CUDA DEBUG] reg_marginaliseResultX_kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
           cudaGetErrorString(cudaGetLastError()),G7.x,G7.y,G7.z,B7.x,B7.y,B7.z);
#endif

    // Now over Z axes. (result_bins[0])
    temp2DHistogram=NULL;
    CUDA_SAFE_CALL(cudaMalloc(&temp2DHistogram,target_bins[0]*target_bins[1]*sizeof(float)));
    CUDA_SAFE_CALL(cudaBindTexture(0, histogramTexture, temp3DHistogram, target_bins[0]*target_bins[1]*result_bins[0]*sizeof(float)));
    dim3 B8(Block_reg_marginaliseResultXY,1,1);
    dim3 G8((int)ceil((float)(target_bins[0]*target_bins[1])/(float)B8.x),1,1);
    reg_marginaliseResultXY_kernel <<< G8, B8 >>> (temp2DHistogram);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
    printf("[NiftyReg CUDA DEBUG] reg_marginaliseResultXY_kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
           cudaGetErrorString(cudaGetLastError()),G8.x,G8.y,G8.z,B8.x,B8.y,B8.z);
#endif

    cudaFree(temp3DHistogram);
    // Transfer the data to CPU
    temp2DHistogram_h = new float[total_target_entries];
    cudaMemcpy(temp2DHistogram_h,temp2DHistogram,total_target_entries*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < total_target_entries; ++i) {
        probaJointHistogram[num_probabilities + i] = temp2DHistogram_h[i];
    }    
    delete[] temp2DHistogram_h;
    cudaFree(temp2DHistogram);

    // The next bits can be put on the GPU but there is not much performance gain and it is
    // better to go the log and accumulation using double precision.

    // Generate joint entropy
    float current_value, current_log;
    double joint_entropy = 0.0;
    for (int i = 0; i < num_probabilities; ++i)
    {
        current_value = probaJointHistogram[i];
        current_log = 0.0;
        if (current_value) current_log = log(current_value);
        joint_entropy -= current_value * current_log;
        logJointHistogram[i] = current_log;
    }

    // Generate target entropy
    double *log_joint_target = &logJointHistogram[num_probabilities];
    double target_entropy = 0.0;
    for (int i = 0; i < total_target_entries; ++i)
    {
        current_value = probaJointHistogram[num_probabilities + i];
        current_log = 0.0;
        if (current_value) current_log = log(current_value);
        target_entropy -= current_value * current_log;
        log_joint_target[i] = current_log;
    }

    // Generate result entropy
    double *log_joint_result = &logJointHistogram[num_probabilities+total_target_entries];
    double result_entropy = 0.0;
    for (int i = 0; i < total_result_entries; ++i)
    {
        current_value = probaJointHistogram[num_probabilities + total_target_entries + i];
        current_log = 0.0;
        if (current_value) current_log = log(current_value);
        result_entropy -= current_value * current_log;
        log_joint_result[i] = current_log;
    }

    entropies[0] = target_entropy;
    entropies[1] = result_entropy;
    entropies[2] = joint_entropy;
    entropies[3] = voxelSum;
}

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
/// Called when we have two target and two source image
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

    reg_getVoxelBasedNMIGradientUsingPW2x2_kernel <<< G1, B1 >>> (*voxelNMIGradientArray_d);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
    printf("[NiftyReg CUDA DEBUG] reg_getVoxelBasedNMIGradientUsingPW2x2_gpu: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
           cudaGetErrorString(cudaGetLastError()),G1.x,G1.y,G1.z,B1.x,B1.y,B1.z);
#endif
}

#endif
