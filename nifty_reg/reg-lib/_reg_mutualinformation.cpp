/*
 *  _reg_mutualinformation.cpp
 *
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_MUTUALINFORMATION_CPP
#define _REG_MUTUALINFORMATION_CPP

#include "_reg_mutualinformation.h"
#include "_reg_tools.h"
#include <iostream>

/// Smooth the histogram along the given axes. Uses recursion
template<class PrecisionTYPE>
void smooth_axes(int axes, int current, PrecisionTYPE *histogram,
                 PrecisionTYPE *result, PrecisionTYPE *window,
                 int num_dims, int *dimensions, int *indices)
{
    int temp, index;
    PrecisionTYPE value;
    for(indices[current] = 0; indices[current] < dimensions[current]; ++indices[current])
    {
        if(axes == current) {
            temp = indices[current];
            indices[current]--;
            value = (PrecisionTYPE)0;
            for(int it=0; it<3; it++) {
                if(-1<indices[current] && indices[current]<dimensions[current]) {
                    index = calculate_index(num_dims, dimensions, indices);
                    value += histogram[index] * window[it];
                }
                indices[current]++;
            }
            indices[current] = temp;
            index = calculate_index(num_dims, dimensions, indices);
            result[index] = value;
        }
        else {
            smooth_axes<PrecisionTYPE>(axes, previous(current, num_dims), histogram,
                                       result, window, num_dims, dimensions, indices);
        }
    }
}

/// Traverse the histogram along the specified axes and smooth along it
template<class PrecisionTYPE>
void traverse_and_smooth_axes(int axes, PrecisionTYPE *histogram,
                              PrecisionTYPE *result, PrecisionTYPE *window,
                              int num_dims, int *dimensions)
{
    SafeArray<int> indices(num_dims);
    for(int dim = 0; dim < num_dims; ++dim) indices[dim] = 0;

    smooth_axes<PrecisionTYPE>(axes, previous(axes, num_dims), histogram,
                               result, window, num_dims, dimensions, indices);
}

/// Sum along the specified axes. Uses recursion
template<class PrecisionTYPE>
void sum_axes(int axes, int current, PrecisionTYPE *histogram, PrecisionTYPE *&sums,
              int num_dims, int *dimensions, int *indices)
{
    int index;
    PrecisionTYPE value = (PrecisionTYPE)0;

    for(indices[current] = 0; indices[current] < dimensions[current]; ++indices[current])
    {
        if(axes == current) {
            index = calculate_index(num_dims, dimensions, indices);
            value += histogram[index];
        }
        else {
            sum_axes<PrecisionTYPE>(axes, previous(current, num_dims), histogram,
                                    sums, num_dims, dimensions, indices);
        }
    }
    // Store the sum along the current line and increment the storage pointer
    if (axes == current)
    {
        *(sums) = value;
        ++sums;
    }
}

/// Traverse and sum along an axes
template<class PrecisionTYPE>
void traverse_and_sum_axes(int axes, PrecisionTYPE *histogram, PrecisionTYPE *&sums,
                           int num_dims, int *dimensions)
{
    SafeArray<int> indices(num_dims);
    for(int dim = 0; dim < num_dims; ++dim) indices[dim] = 0;
    sum_axes<PrecisionTYPE>(axes, previous(axes, num_dims), histogram, sums,
                            num_dims, dimensions, indices);
}


/* *************************************************************** */
template<class PrecisionTYPE>
PrecisionTYPE GetBasisSplineValue(PrecisionTYPE x)
{
    x=fabs(x);
    PrecisionTYPE value=0.0;
    if(x<2.0)
        if(x<1.0)
            value = (PrecisionTYPE)(2.0f/3.0f + (0.5f*x-1.0)*x*x);
        else{
            x-=2.0f;
            value = -x*x*x/6.0f;
    }
    return value;
}
/* *************************************************************** */
template<class PrecisionTYPE>
PrecisionTYPE GetBasisSplineDerivativeValue(PrecisionTYPE ori)
{
    PrecisionTYPE x=fabs(ori);
    PrecisionTYPE value=0.0;
    if(x<2.0)
        if(x<1.0)
            value = (PrecisionTYPE)((1.5f*x-2.0)*ori);
        else{
            x-=2.0f;
            value = -0.5f * x * x;
            if(ori<0.0f)value =-value;
    }
    return value;
}

/* *************************************************************** */
/* *************************************************************** */



/// Multi channel NMI joint histogram and entropy calculation
template<class PrecisionTYPE, class TargetTYPE, class ResultTYPE>
void reg_getEntropies3(nifti_image *targetImage,
                      nifti_image *resultImage,
                      int type, //! Not used at the moment
                      unsigned int *target_bins,
                      unsigned int *result_bins,
                      PrecisionTYPE *probaJointHistogram,
                      PrecisionTYPE *logJointHistogram,
                      PrecisionTYPE *entropies,
                      int *mask)
{
    unsigned int num_target_volumes = targetImage->nt;
    unsigned int num_result_volumes = resultImage->nt;

    unsigned targetVoxelNumber = targetImage->nx * targetImage->ny * targetImage->nz;

    TargetTYPE *targetImagePtr = static_cast<TargetTYPE *>(targetImage->data);
    ResultTYPE *resultImagePtr = static_cast<ResultTYPE *>(resultImage->data);

    // Build up this arrays of offsets that will help us index the histogram entries
    SafeArray<int> target_offsets(num_target_volumes);
    SafeArray<int> result_offsets(num_result_volumes);

    int num_histogram_entries = 1;
    int total_target_entries = 1;
    int total_result_entries = 1;

    // Data pointers
    SafeArray<int> histogram_dimensions(num_target_volumes + num_result_volumes);

    // Calculate some constants and initialize the data pointers
    for (unsigned int i = 0; i < num_target_volumes; ++i) {
        num_histogram_entries *= target_bins[i];
        total_target_entries *= target_bins[i];
        histogram_dimensions[i] = target_bins[i];

        target_offsets[i] = 1;
        for (int j = i; j > 0; --j) target_offsets[i] *= target_bins[j - 1];
    }

    for (unsigned int i = 0; i < num_result_volumes; ++i) {
        num_histogram_entries *= result_bins[i];
        total_result_entries *= result_bins[i];
        histogram_dimensions[num_target_volumes + i] = result_bins[i];

        result_offsets[i] = 1;
        for (int j = i; j > 0; --j) result_offsets[i] *= result_bins[j-1];
    }

    int num_probabilities = num_histogram_entries;

    // Space for storing the marginal entropies.
    num_histogram_entries += total_target_entries + total_result_entries;

    memset(probaJointHistogram, 0, num_histogram_entries * sizeof(PrecisionTYPE));
    memset(logJointHistogram, 0, num_histogram_entries * sizeof(PrecisionTYPE));

    int *mask_ptr = &mask[0];

    // These hold the current target and result values
    SafeArray<TargetTYPE> target_values(num_target_volumes);
    SafeArray<ResultTYPE> result_values(num_result_volumes);

    bool valid_values;

    unsigned int target_flat_index, result_flat_index;
    PrecisionTYPE voxel_number = (PrecisionTYPE)0;

    // For now we only use the approximate PW approach for filling the joint histogram.
    // Fill the joint histogram using the classical approach

    for (unsigned int index = 0; index < targetVoxelNumber; ++index){
        if (*mask_ptr++ > -1) {
            valid_values = true;
            target_flat_index = 0;
            // Get the target values
            for (unsigned int i = 0; i < num_target_volumes; ++i) {
                target_values[i] = targetImagePtr[index+i*targetVoxelNumber];
                if (target_values[i] <= (TargetTYPE)0 ||
                    target_values[i] >= (TargetTYPE)target_bins[i] ||
                    target_values[i] != target_values[i]) {
                    valid_values = false;
                    break;
                }
                target_flat_index += static_cast<int>(target_values[i]) * target_offsets[i];
            }

            if (valid_values) {
                result_flat_index = 0;
                // Get the result values
                for (unsigned int i = 0; i < num_result_volumes; ++i){
                    result_values[i] = resultImagePtr[index+i*targetVoxelNumber];
                    if (result_values[i] <= (ResultTYPE)0 ||
                        result_values[i] >= (ResultTYPE)result_bins[i] ||
                        result_values[i] != result_values[i]) {
                        valid_values = false;
                        break;
                    }
                    result_flat_index += static_cast<int>(result_values[i]) * result_offsets[i];
                }
            }
            if (valid_values) {
                probaJointHistogram[target_flat_index + (result_flat_index * total_target_entries)]++;
                ++voxel_number;
            }
        }        
    }

    PrecisionTYPE window[3];
    window[0] = window[2] = GetBasisSplineValue((PrecisionTYPE)(-1.0));
    window[1] = GetBasisSplineValue((PrecisionTYPE)(0.0));

    PrecisionTYPE *histogram=NULL;
    PrecisionTYPE *result=NULL;
    int num_axes = num_target_volumes + num_result_volumes;

    // Smooth along each of the axes
    for (int i = 0; i < num_axes; ++i)
    {
        // Use the arrays for storage of results
        if (i % 2 == 0) {
            result = logJointHistogram;
            histogram = probaJointHistogram;
        }
        else {
            result = probaJointHistogram;
            histogram = logJointHistogram;
        }
        traverse_and_smooth_axes<PrecisionTYPE>(i, histogram, result, window,
                                                num_axes, histogram_dimensions);
    }

    // We may need to transfer the result
    if (result == logJointHistogram) memcpy(probaJointHistogram, logJointHistogram,
                                            sizeof(PrecisionTYPE)*num_probabilities);

    memset(logJointHistogram, 0, num_histogram_entries * sizeof(PrecisionTYPE));

    // Convert to probabilities
    for(int i = 0; i < num_probabilities; ++i) {
        if (probaJointHistogram[i]) probaJointHistogram[i] /= voxel_number;
    }

    // Marginalise over all the result axes to generate the target entropy
    PrecisionTYPE *data = probaJointHistogram;
    PrecisionTYPE *store = logJointHistogram;
    PrecisionTYPE current_value, current_log;

    PrecisionTYPE target_entropy = (PrecisionTYPE)0;
    {
        SafeArray<PrecisionTYPE> scratch (num_probabilities/histogram_dimensions[num_axes - 1]);
        // marginalise over the result axes
        for (int i = num_result_volumes-1, count = 0; i >= 0; --i, ++count)
        {
            traverse_and_sum_axes<PrecisionTYPE>(num_axes - count - 1,
                                                 data, store, num_axes - count,
                                                 histogram_dimensions);

            if (count % 2 == 0) {
                data = logJointHistogram;
                store = scratch;
            }
            else {
                data = scratch;
                store = logJointHistogram;
            }
        }

        // Generate target entropy
        PrecisionTYPE *log_joint_target = &logJointHistogram[num_probabilities];

        for (int i = 0; i < total_target_entries; ++i)
        {
            current_value = data[i];            
            current_log = (PrecisionTYPE)0;
            if (current_value) current_log = log(current_value);
            target_entropy -= current_value * current_log;
            log_joint_target[i] = current_log;
        }
    }
    memset(logJointHistogram, 0, num_probabilities * sizeof(PrecisionTYPE));
    data = probaJointHistogram;
    store = logJointHistogram;

    // Marginalise over the target axes
    PrecisionTYPE result_entropy = (PrecisionTYPE)0;
    {
        SafeArray<PrecisionTYPE> scratch (num_probabilities / histogram_dimensions[0]);
        for (unsigned int i = 0; i < num_target_volumes; ++i)
        {
            traverse_and_sum_axes<PrecisionTYPE>(0, data, store, num_axes - i, &histogram_dimensions[i]);
            if (i % 2 == 0) {
                data = logJointHistogram;
                store = scratch;
            }
            else {
                data = scratch;
                store = logJointHistogram;
            }
        }
        // Generate result entropy
        PrecisionTYPE *log_joint_result = &logJointHistogram[num_probabilities+total_target_entries];

        for (int i = 0; i < total_result_entries; ++i)
        {
            current_value = data[i];            
            current_log = (PrecisionTYPE)0;
            if (current_value) current_log = log(current_value);
            result_entropy -= current_value * current_log;
            log_joint_result[i] = current_log;
        }
    }

    // Generate joint entropy
    PrecisionTYPE joint_entropy = (PrecisionTYPE)0;
    for (int i = 0; i < num_probabilities; ++i)
    {
        current_value = probaJointHistogram[i];        
        current_log = (PrecisionTYPE)0;
        if (current_value) current_log = log(current_value);
        joint_entropy -= current_value * current_log;
        logJointHistogram[i] = current_log;
    }

    entropies[0] = target_entropy;
    entropies[1] = result_entropy;
    entropies[2] = joint_entropy;
    entropies[3] = voxel_number;
}
/***************************************************************** */
extern "C++" template<class PrecisionTYPE, class TargetTYPE>
void reg_getEntropies2(nifti_image *targetImage,
                       nifti_image *resultImage,
                       int type, //! Not used at the moment
                       unsigned int *target_bins, // should be an array of size num_target_volumes
                       unsigned int *result_bins, // should be an array of size num_result_volumes
                       PrecisionTYPE *probaJointHistogram,
                       PrecisionTYPE *logJointHistogram,
                       PrecisionTYPE *entropies,
                       int *mask)
{
    switch(resultImage->datatype){
#ifdef _NR_DEV
    case NIFTI_TYPE_UINT8:
        reg_getEntropies3<PrecisionTYPE,TargetTYPE,unsigned char>
                (targetImage, resultImage, type, target_bins, result_bins, probaJointHistogram,
                 logJointHistogram, entropies, mask);
        break;
    case NIFTI_TYPE_INT8:
        reg_getEntropies3<PrecisionTYPE,TargetTYPE,char>
                (targetImage, resultImage, type, target_bins, result_bins, probaJointHistogram,
                 logJointHistogram, entropies, mask);
        break;
    case NIFTI_TYPE_UINT16:
        reg_getEntropies3<PrecisionTYPE,TargetTYPE,unsigned short>
                (targetImage, resultImage, type, target_bins, result_bins, probaJointHistogram,
                 logJointHistogram, entropies, mask);
        break;
    case NIFTI_TYPE_INT16:
        reg_getEntropies3<PrecisionTYPE,TargetTYPE,short>
                (targetImage, resultImage, type, target_bins, result_bins, probaJointHistogram,
                 logJointHistogram, entropies, mask);
        break;
    case NIFTI_TYPE_UINT32:
        reg_getEntropies3<PrecisionTYPE,TargetTYPE,int>
                (targetImage, resultImage, type, target_bins, result_bins, probaJointHistogram,
                 logJointHistogram, entropies, mask);
        break;
    case NIFTI_TYPE_INT32:
        reg_getEntropies3<PrecisionTYPE,TargetTYPE,unsigned int>
                (targetImage, resultImage, type, target_bins, result_bins, probaJointHistogram,
                 logJointHistogram, entropies, mask);
        break;
        case NIFTI_TYPE_FLOAT64:
            reg_getEntropies3<PrecisionTYPE,TargetTYPE,double>
                    (targetImage, resultImage, type, target_bins, result_bins, probaJointHistogram,
                     logJointHistogram, entropies, mask);
            break;
#endif
        case NIFTI_TYPE_FLOAT32:
            reg_getEntropies3<PrecisionTYPE,TargetTYPE,float>
                    (targetImage, resultImage, type, target_bins, result_bins, probaJointHistogram,
                     logJointHistogram, entropies, mask);
            break;
        default:
            fprintf(stderr,"[NiftyReg ERROR] reg_getEntropies\tThe result image data type is not supported\n");
            exit(1);
    }
    return;

}

/***************************************************************** */
extern "C++" template<class PrecisionTYPE>
void reg_getEntropies(nifti_image *targetImage,
                      nifti_image *resultImage,
                      int type, //! Not used at the moment
                      unsigned int *target_bins, // should be an array of size num_target_volumes
                      unsigned int *result_bins, // should be an array of size num_result_volumes
                      PrecisionTYPE *probaJointHistogram,
                      PrecisionTYPE *logJointHistogram,
                      PrecisionTYPE *entropies,
                      int *mask)
{
    switch(targetImage->datatype){
#ifdef _NR_DEV
        case NIFTI_TYPE_UINT8:
            reg_getEntropies2<PrecisionTYPE,unsigned char>
                    (targetImage, resultImage, type, target_bins, result_bins, probaJointHistogram,
                     logJointHistogram, entropies, mask);
            break;
        case NIFTI_TYPE_INT8:
            reg_getEntropies2<PrecisionTYPE,char>
                    (targetImage, resultImage, type, target_bins, result_bins, probaJointHistogram,
                     logJointHistogram, entropies, mask);
            break;
        case NIFTI_TYPE_UINT16:
            reg_getEntropies2<PrecisionTYPE,unsigned short>
                    (targetImage, resultImage, type, target_bins, result_bins, probaJointHistogram,
                     logJointHistogram, entropies, mask);
            break;
        case NIFTI_TYPE_INT16:
            reg_getEntropies2<PrecisionTYPE,short>
                    (targetImage, resultImage, type, target_bins, result_bins, probaJointHistogram,
                     logJointHistogram, entropies, mask);
            break;
        case NIFTI_TYPE_UINT32:
            reg_getEntropies2<PrecisionTYPE,int>
                    (targetImage, resultImage, type, target_bins, result_bins, probaJointHistogram,
                     logJointHistogram, entropies, mask);
            break;
        case NIFTI_TYPE_INT32:
            reg_getEntropies2<PrecisionTYPE,unsigned int>
                    (targetImage, resultImage, type, target_bins, result_bins, probaJointHistogram,
                     logJointHistogram, entropies, mask);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_getEntropies2<PrecisionTYPE,double>
                    (targetImage, resultImage, type, target_bins, result_bins, probaJointHistogram,
                     logJointHistogram, entropies, mask);
            break;
#endif
    case NIFTI_TYPE_FLOAT32:
        reg_getEntropies2<PrecisionTYPE,float>
                (targetImage, resultImage, type, target_bins, result_bins, probaJointHistogram,
                 logJointHistogram, entropies, mask);
        break;
        default:
            fprintf(stderr,"[NiftyReg ERROR] reg_getEntropies\tThe target image data type is not supported\n");
            exit(1);
    }
    return;
}

/* *************************************************************** */
template void reg_getEntropies<double>(nifti_image *, nifti_image *,
                                       int, unsigned int *, unsigned int *, double *, double *, double *, int *);
template void reg_getEntropies<float>(nifti_image *, nifti_image *,
                                      int, unsigned int *, unsigned int *, float *, float *, float *, int *);
/* *************************************************************** */
/* *************************************************************** */
/// Voxel based multichannel gradient computation
template<class PrecisionTYPE,class TargetTYPE,class ResultTYPE,class ResultGradientTYPE,class NMIGradientTYPE>
void reg_getVoxelBasedNMIGradientUsingPW3D(nifti_image *targetImage,
                                           nifti_image *resultImage,
                                           int type, //! Not used at the moment
                                           nifti_image *resultImageGradient,
                                           unsigned int *target_bins,
                                           unsigned int *result_bins,
                                           PrecisionTYPE *logJointHistogram,
                                           PrecisionTYPE *entropies,
                                           nifti_image *nmiGradientImage,
                                           int *mask)
{
    unsigned int num_target_volumes=targetImage->nt;
    unsigned int num_result_volumes=resultImage->nt;
    unsigned int num_loops = num_target_volumes + num_result_volumes;

    unsigned targetVoxelNumber = targetImage->nx * targetImage->ny * targetImage->nz;

    TargetTYPE *targetImagePtr = static_cast<TargetTYPE *>(targetImage->data);
    ResultTYPE *resultImagePtr = static_cast<ResultTYPE *>(resultImage->data);
    ResultGradientTYPE *resulImageGradientPtrX = static_cast<ResultGradientTYPE *>(resultImageGradient->data);
    ResultGradientTYPE *resulImageGradientPtrY = &resulImageGradientPtrX[targetVoxelNumber*num_result_volumes];
    ResultGradientTYPE *resulImageGradientPtrZ = &resulImageGradientPtrY[targetVoxelNumber*num_result_volumes];

    // Build up this arrays of offsets that will help us index the histogram entries
    int target_offsets[10];
    int result_offsets[10];

    int total_target_entries = 1;
    int total_result_entries = 1;

    // The 4D
    for (unsigned int i = 0; i < num_target_volumes; ++i) {
        total_target_entries *= target_bins[i];
        target_offsets[i] = 1;
        for (int j = i; j > 0; --j) target_offsets[i] *= target_bins[j - 1];
    }

    for (unsigned int i = 0; i < num_result_volumes; ++i) {
        total_result_entries *= result_bins[i];
        result_offsets[i] = 1;
        for (int j = i; j > 0; --j) result_offsets[i] *= result_bins[j - 1];
    }

    int num_probabilities = total_target_entries * total_result_entries;

    int *maskPtr = &mask[0];
    PrecisionTYPE NMI = (entropies[0] + entropies[1]) / entropies[2];

    // Hold current values.
    // target and result images limited to 10 max for speed.
    TargetTYPE voxel_values[20];
    ResultGradientTYPE result_gradient_x_values[10];
    ResultGradientTYPE result_gradient_y_values[10];
    ResultGradientTYPE result_gradient_z_values[10];

    bool valid_values;
    PrecisionTYPE common_target_value;

    PrecisionTYPE jointEntropyDerivative_X;
    PrecisionTYPE movingEntropyDerivative_X;
    PrecisionTYPE fixedEntropyDerivative_X;

    PrecisionTYPE jointEntropyDerivative_Y;
    PrecisionTYPE movingEntropyDerivative_Y;
    PrecisionTYPE fixedEntropyDerivative_Y;

    PrecisionTYPE jointEntropyDerivative_Z;
    PrecisionTYPE movingEntropyDerivative_Z;
    PrecisionTYPE fixedEntropyDerivative_Z;

    PrecisionTYPE jointLog, targetLog, resultLog;
    PrecisionTYPE joint_entropy = (PrecisionTYPE)(entropies[2]);

    NMIGradientTYPE *nmiGradientPtrX = static_cast<NMIGradientTYPE *>(nmiGradientImage->data);
    NMIGradientTYPE *nmiGradientPtrY = &nmiGradientPtrX[targetVoxelNumber];
    NMIGradientTYPE *nmiGradientPtrZ = &nmiGradientPtrY[targetVoxelNumber];
    memset(nmiGradientPtrX,0,nmiGradientImage->nvox*nmiGradientImage->nbyper);

    // Set up the multi loop
    Multi_Loop<int> loop;
    for (unsigned int i = 0; i < num_loops; ++i) loop.Add(-1, 2);

    SafeArray<int> bins(num_loops);
    for (unsigned int i = 0; i < num_target_volumes; ++i) bins[i] = target_bins[i];
    for (unsigned int i = 0; i < num_result_volumes; ++i) bins[i + num_target_volumes] = result_bins[i];

    PrecisionTYPE coefficients[20];
    PrecisionTYPE positions[20];
    int relative_positions[20];

    PrecisionTYPE result_common[3];
    PrecisionTYPE der_term[3];

    // Loop over all the voxels
    for (unsigned int index = 0; index < targetVoxelNumber; ++index) {
        if(*maskPtr++>-1){
            valid_values = true;
            // Collect the target intensities and do some sanity checking
            for (unsigned int i = 0; i < num_target_volumes; ++i) {
                voxel_values[i] = targetImagePtr[index+i*targetVoxelNumber];
                if (voxel_values[i] <= (TargetTYPE)0 ||
                    voxel_values[i] >= (TargetTYPE)target_bins[i] ||
                    voxel_values[i] != voxel_values[i]) {
                    valid_values = false;
                    break;
                }
                voxel_values[i] = (TargetTYPE)static_cast<int>((double)voxel_values[i]);
            }

            // Collect the result intensities and do some sanity checking
            if (valid_values) {
                for (unsigned int i = 0; i < num_result_volumes; ++i) {
                    unsigned int currentIndex = index+i*targetVoxelNumber;
                    ResultTYPE temp = resultImagePtr[currentIndex];
                    result_gradient_x_values[i] = resulImageGradientPtrX[currentIndex];
                    result_gradient_y_values[i] = resulImageGradientPtrY[currentIndex];
                    result_gradient_z_values[i] = resulImageGradientPtrZ[currentIndex];

                    if (temp <= (ResultTYPE)0 ||
                        temp >= (ResultTYPE)result_bins[i] ||
                        temp != temp ||
                        result_gradient_x_values[i] != result_gradient_x_values[i] ||
                        result_gradient_y_values[i] != result_gradient_y_values[i] ||
                        result_gradient_z_values[i] != result_gradient_z_values[i]) {
                        valid_values = false;
                        break;
                    }
                    voxel_values[num_target_volumes + i] = (TargetTYPE)static_cast<int>((double)temp);
                }
            }
            if (valid_values) {
                jointEntropyDerivative_X = 0.0;
                movingEntropyDerivative_X = 0.0;
                fixedEntropyDerivative_X = 0.0;

                jointEntropyDerivative_Y = 0.0;
                movingEntropyDerivative_Y = 0.0;
                fixedEntropyDerivative_Y = 0.0;

                jointEntropyDerivative_Z = 0.0;
                movingEntropyDerivative_Z = 0.0;
                fixedEntropyDerivative_Z = 0.0;

                int target_flat_index, result_flat_index;

                for (loop.Initialise(); loop.Continue(); loop.Next()) {
                    target_flat_index = result_flat_index = 0;
                    valid_values = true;

                    for(unsigned int lc = 0; lc < num_target_volumes; ++lc){
                        int relative_pos = int(voxel_values[lc] + loop.Index(lc));
                        if(relative_pos< 0 || relative_pos >= bins[lc]){
                            valid_values = false; break;
                        }                        
                        PrecisionTYPE common_value = GetBasisSplineValue<PrecisionTYPE>((PrecisionTYPE)relative_pos-(PrecisionTYPE)voxel_values[lc]);
                        coefficients[lc] = common_value;
                        positions[lc] = (PrecisionTYPE)relative_pos-(PrecisionTYPE)voxel_values[lc];
                        relative_positions[lc] = relative_pos;
                    }

                    for(unsigned int jc = num_target_volumes; jc < num_loops; ++jc){
                        int relative_pos = int(voxel_values[jc] + loop.Index(jc));
                        if(relative_pos< 0 || relative_pos >= bins[jc]){
                            valid_values = false; break;
                        }
                        if (num_result_volumes > 1) {
                            PrecisionTYPE common_value = GetBasisSplineValue<PrecisionTYPE>((PrecisionTYPE)relative_pos-(PrecisionTYPE)voxel_values[jc]);
                            coefficients[jc] = common_value;
                        }
                        positions[jc] = (PrecisionTYPE)relative_pos-(PrecisionTYPE)voxel_values[jc];
                        relative_positions[jc] = relative_pos;
                    }

                    if(valid_values) {
                        common_target_value = (PrecisionTYPE)1.0;
                        for (unsigned int i = 0; i < num_target_volumes; ++i) common_target_value *= coefficients[i];

                        result_common[0] = result_common[1] = result_common[2] = (PrecisionTYPE)0.0;

                        for (unsigned int i = 0; i < num_result_volumes; ++i)
                        {
                            der_term[0] = der_term[1] = der_term[2] = (PrecisionTYPE)1.0;
                            for (unsigned int j = 0; j < num_result_volumes; ++j)
                            {
                                if (i == j) {
                                    PrecisionTYPE reg = GetBasisSplineDerivativeValue<PrecisionTYPE>
                                                        ((PrecisionTYPE)positions[j + num_target_volumes]);
                                    der_term[0] *= reg * (PrecisionTYPE)result_gradient_x_values[j];
                                    der_term[1] *= reg * (PrecisionTYPE)result_gradient_y_values[j];
                                    der_term[2] *= reg * (PrecisionTYPE)result_gradient_z_values[j];
                                }
                                else {
                                    der_term[0] *= coefficients[j+num_target_volumes];
                                    der_term[1] *= coefficients[j+num_target_volumes];
                                    der_term[2] *= coefficients[j+num_target_volumes];
                                }
                            }
                            result_common[0] += der_term[0];
                            result_common[1] += der_term[1];
                            result_common[2] += der_term[2];
                        }

                        result_common[0] *= common_target_value;
                        result_common[1] *= common_target_value;
                        result_common[2] *= common_target_value;

                        for (unsigned int i = 0; i < num_target_volumes; ++i) target_flat_index += relative_positions[i] * target_offsets[i];
                        for (unsigned int i = 0; i < num_result_volumes; ++i) result_flat_index += relative_positions[i + num_target_volumes] * result_offsets[i];

                        jointLog = logJointHistogram[target_flat_index + (result_flat_index * total_target_entries)];
                        targetLog = logJointHistogram[num_probabilities + target_flat_index];
                        resultLog = logJointHistogram[num_probabilities + total_target_entries + result_flat_index];

                        jointEntropyDerivative_X -= result_common[0] * jointLog;
                        fixedEntropyDerivative_X -= result_common[0] * targetLog;
                        movingEntropyDerivative_X -= result_common[0] * resultLog;

                        jointEntropyDerivative_Y -= result_common[1] * jointLog;
                        fixedEntropyDerivative_Y -= result_common[1] * targetLog;
                        movingEntropyDerivative_Y -= result_common[1] * resultLog;

                        jointEntropyDerivative_Z -= result_common[2] * jointLog;
                        fixedEntropyDerivative_Z -= result_common[2] * targetLog;
                        movingEntropyDerivative_Z -= result_common[2] * resultLog;
                    }

                    *nmiGradientPtrX = (NMIGradientTYPE)((fixedEntropyDerivative_X + movingEntropyDerivative_X - NMI * jointEntropyDerivative_X) / joint_entropy);
                    *nmiGradientPtrY = (NMIGradientTYPE)((fixedEntropyDerivative_Y + movingEntropyDerivative_Y - NMI * jointEntropyDerivative_Y) / joint_entropy);
                    *nmiGradientPtrZ = (NMIGradientTYPE)((fixedEntropyDerivative_Z + movingEntropyDerivative_Z - NMI * jointEntropyDerivative_Z) / joint_entropy);
                }
            }
        }

        nmiGradientPtrX++; nmiGradientPtrY++; nmiGradientPtrZ++;
    }
}
/* *************************************************************** */
template<class PrecisionTYPE,class TargetTYPE,class ResultTYPE,class ResultGradientTYPE>
void reg_getVoxelBasedNMIGradientUsingPW3(nifti_image *targetImage,
                                          nifti_image *resultImage,
                                          int type,
                                          nifti_image *resultImageGradient,
                                          unsigned int *target_bins,
                                          unsigned int *result_bins,
                                          PrecisionTYPE *logJointHistogram,
                                          PrecisionTYPE *entropies,
                                          nifti_image *nmiGradientImage,
                                          int *mask)
{
    if(nmiGradientImage->nz>1){
        switch(nmiGradientImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_getVoxelBasedNMIGradientUsingPW3D<PrecisionTYPE,TargetTYPE,ResultTYPE,ResultGradientTYPE,float>
                        (targetImage, resultImage, type, resultImageGradient, target_bins, result_bins, logJointHistogram,
                         entropies, nmiGradientImage, mask);
                break;
#ifdef _NR_DEV
            case NIFTI_TYPE_FLOAT64:
                reg_getVoxelBasedNMIGradientUsingPW3D<PrecisionTYPE,TargetTYPE,ResultTYPE,ResultGradientTYPE,double>
                        (targetImage, resultImage, type, resultImageGradient, target_bins, result_bins, logJointHistogram,
                         entropies, nmiGradientImage, mask);
                break;
#endif
            default:
                fprintf(stderr,"[NiftyReg ERROR] reg_getVoxelBasedNMIGradientUsingPW\tThe result image gradient data type is not supported\n");
                exit(1);
        }
    }else{/*
        switch(nmiGradientImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_getVoxelBasedNMIGradientUsingPW2D<PrecisionTYPE,TargetTYPE,ResultTYPE,ResultGradientTYPE,float>
                        (targetImage, resultImage, type, resultImageGradient, target_bins, result_bins, logJointHistogram,
                         entropies, nmiGradientImage, mask, num_target_volumes, num_result_volumes);
                break;
            case NIFTI_TYPE_FLOAT64:
                reg_getVoxelBasedNMIGradientUsingPW2D<PrecisionTYPE,TargetTYPE,ResultTYPE,ResultGradientTYPE,double>
                        (targetImage, resultImage, type, resultImageGradient, target_bins, result_bins, logJointHistogram,
                         entropies, nmiGradientImage, mask, num_target_volumes, num_result_volumes);
                break;
            default:
                fprintf(stderr,"[NiftyReg ERROR] reg_getVoxelBasedNMIGradientUsingPW\tThe result image gradient data type is not supported\n");
                exit(1);
        }*/
    }
}
/* *************************************************************** */
template<class PrecisionTYPE,class TargetTYPE,class ResultTYPE>
void reg_getVoxelBasedNMIGradientUsingPW2(nifti_image *targetImage,
                                          nifti_image *resultImage,
                                          int type,
                                          nifti_image *resultImageGradient,
                                          unsigned int *target_bins,
                                          unsigned int *result_bins,
                                          PrecisionTYPE *logJointHistogram,
                                          PrecisionTYPE *entropies,
                                          nifti_image *nmiGradientImage,
                                          int *mask)
{
    switch(resultImageGradient->datatype){
#ifdef _NR_DEV
        case NIFTI_TYPE_UINT8:
            reg_getVoxelBasedNMIGradientUsingPW3<PrecisionTYPE,TargetTYPE,ResultTYPE,unsigned char>
                    (targetImage, resultImage, type, resultImageGradient, target_bins, result_bins, logJointHistogram,
                     entropies, nmiGradientImage, mask);
            break;
        case NIFTI_TYPE_INT8:
            reg_getVoxelBasedNMIGradientUsingPW3<PrecisionTYPE,TargetTYPE,ResultTYPE,char>
                    (targetImage, resultImage, type, resultImageGradient, target_bins, result_bins, logJointHistogram,
                     entropies, nmiGradientImage, mask);
            break;
        case NIFTI_TYPE_UINT16:
            reg_getVoxelBasedNMIGradientUsingPW3<PrecisionTYPE,TargetTYPE,ResultTYPE,unsigned short>
                    (targetImage, resultImage, type, resultImageGradient, target_bins, result_bins, logJointHistogram,
                     entropies, nmiGradientImage, mask);
            break;
        case NIFTI_TYPE_INT16:
            reg_getVoxelBasedNMIGradientUsingPW3<PrecisionTYPE,TargetTYPE,ResultTYPE,short>
                    (targetImage, resultImage, type, resultImageGradient, target_bins, result_bins, logJointHistogram,
                     entropies, nmiGradientImage, mask);
            break;
        case NIFTI_TYPE_UINT32:
            reg_getVoxelBasedNMIGradientUsingPW3<PrecisionTYPE,TargetTYPE,ResultTYPE,unsigned int>
                    (targetImage, resultImage, type, resultImageGradient, target_bins, result_bins, logJointHistogram,
                     entropies, nmiGradientImage, mask);
            break;
        case NIFTI_TYPE_INT32:
            reg_getVoxelBasedNMIGradientUsingPW3<PrecisionTYPE,TargetTYPE,ResultTYPE,int>
                    (targetImage, resultImage, type, resultImageGradient, target_bins, result_bins, logJointHistogram,
                     entropies, nmiGradientImage, mask);
            break;
            case NIFTI_TYPE_FLOAT64:
            reg_getVoxelBasedNMIGradientUsingPW3<PrecisionTYPE,TargetTYPE,ResultTYPE,double>
                    (targetImage, resultImage, type, resultImageGradient, target_bins, result_bins, logJointHistogram,
                     entropies, nmiGradientImage, mask);
            break;
#endif
        case NIFTI_TYPE_FLOAT32:
            reg_getVoxelBasedNMIGradientUsingPW3<PrecisionTYPE,TargetTYPE,ResultTYPE,float>
                    (targetImage, resultImage, type, resultImageGradient, target_bins, result_bins, logJointHistogram,
                     entropies, nmiGradientImage, mask);
            break;
        default:
            fprintf(stderr,"[NiftyReg ERROR] reg_getVoxelBasedNMIGradientUsingPW\tThe result image gradient data type is not supported\n");
            exit(1);
    }
}
/* *************************************************************** */
template<class PrecisionTYPE,class TargetTYPE>
void reg_getVoxelBasedNMIGradientUsingPW1(nifti_image *targetImage,
                                          nifti_image *resultImage,
                                          int type,
                                          nifti_image *resultImageGradient,
                                          unsigned int *target_bins,
                                          unsigned int *result_bins,
                                          PrecisionTYPE *logJointHistogram,
                                          PrecisionTYPE *entropies,
                                          nifti_image *nmiGradientImage,
                                          int *mask)
{

    switch(resultImage->datatype){
#ifdef _NR_DEV
        case NIFTI_TYPE_UINT8:
            reg_getVoxelBasedNMIGradientUsingPW2<PrecisionTYPE,TargetTYPE,unsigned char>
                    (targetImage, resultImage, type, resultImageGradient, target_bins, result_bins, logJointHistogram,
                     entropies, nmiGradientImage, mask);
            break;
        case NIFTI_TYPE_INT8:
            reg_getVoxelBasedNMIGradientUsingPW2<PrecisionTYPE,TargetTYPE,char>
                    (targetImage, resultImage, type, resultImageGradient, target_bins, result_bins, logJointHistogram,
                     entropies, nmiGradientImage, mask);
            break;
        case NIFTI_TYPE_UINT16:
            reg_getVoxelBasedNMIGradientUsingPW2<PrecisionTYPE,TargetTYPE,unsigned short>
                    (targetImage, resultImage, type, resultImageGradient, target_bins, result_bins, logJointHistogram,
                     entropies, nmiGradientImage, mask);
            break;
        case NIFTI_TYPE_INT16:
            reg_getVoxelBasedNMIGradientUsingPW2<PrecisionTYPE,TargetTYPE,short>
                    (targetImage, resultImage, type, resultImageGradient, target_bins, result_bins, logJointHistogram,
                     entropies, nmiGradientImage, mask);
            break;
        case NIFTI_TYPE_UINT32:
            reg_getVoxelBasedNMIGradientUsingPW2<PrecisionTYPE,TargetTYPE,unsigned int>
                    (targetImage, resultImage, type, resultImageGradient, target_bins, result_bins, logJointHistogram,
                     entropies, nmiGradientImage, mask);
            break;
        case NIFTI_TYPE_INT32:
            reg_getVoxelBasedNMIGradientUsingPW2<PrecisionTYPE,TargetTYPE,int>
                    (targetImage, resultImage, type, resultImageGradient, target_bins, result_bins, logJointHistogram,
                     entropies, nmiGradientImage, mask);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_getVoxelBasedNMIGradientUsingPW2<PrecisionTYPE,TargetTYPE,double>
                    (targetImage, resultImage, type, resultImageGradient, target_bins, result_bins, logJointHistogram,
                     entropies, nmiGradientImage, mask);
            break;
#endif
        case NIFTI_TYPE_FLOAT32:
            reg_getVoxelBasedNMIGradientUsingPW2<PrecisionTYPE,TargetTYPE,float>
                    (targetImage, resultImage, type, resultImageGradient, target_bins, result_bins, logJointHistogram,
                     entropies, nmiGradientImage, mask);
            break;
        default:
            fprintf(stderr,"[NiftyReg ERROR] reg_getVoxelBasedNMIGradientUsingPW\tThe result image data type is not supported\n");
            exit(1);
    }
}

/* *************************************************************** */
template<class PrecisionTYPE>
void reg_getVoxelBasedNMIGradientUsingPW(nifti_image *targetImage,
                                         nifti_image *resultImage,
                                         int type,
                                         nifti_image *resultImageGradient,
                                         unsigned int *target_bins,
                                         unsigned int *result_bins,
                                         PrecisionTYPE *logJointHistogram,
                                         PrecisionTYPE *entropies,
                                         nifti_image *nmiGradientImage,
                                         int *mask)
{
    switch(targetImage->datatype){
        case NIFTI_TYPE_FLOAT32:
            reg_getVoxelBasedNMIGradientUsingPW1<PrecisionTYPE,float>
                    (targetImage, resultImage, type, resultImageGradient, target_bins, result_bins, logJointHistogram,
                     entropies, nmiGradientImage, mask);
            break;
#ifdef _NR_DEV
        case NIFTI_TYPE_FLOAT64:
            reg_getVoxelBasedNMIGradientUsingPW1<PrecisionTYPE,double>
                    (targetImage, resultImage, type, resultImageGradient, target_bins, result_bins, logJointHistogram,
                     entropies, nmiGradientImage, mask);
            break;
#endif
        default:
            fprintf(stderr,"[NiftyReg ERROR] reg_getVoxelBasedNMIGradientUsingPW\tThe target image data type is not supported\n");
            exit(1);
    }
}
/* *************************************************************** */
template void reg_getVoxelBasedNMIGradientUsingPW<float>(nifti_image*, nifti_image*, int, nifti_image*, unsigned int*,
                                                          unsigned int*, float*, float*, nifti_image*, int*);
template void reg_getVoxelBasedNMIGradientUsingPW<double>(nifti_image*, nifti_image*, int, nifti_image*, unsigned int*,
                                                          unsigned int*, double*, double*, nifti_image*, int*);
/* *************************************************************** */
/* *************************************************************** */

#endif
