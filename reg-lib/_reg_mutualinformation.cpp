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
                 PrecisionTYPE *warped, PrecisionTYPE *window,
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
            warped[index] = value;
        }
        else {
            smooth_axes<PrecisionTYPE>(axes, previous(current, num_dims), histogram,
                                       warped, window, num_dims, dimensions, indices);
        }
    }
}

/// Traverse the histogram along the specified axes and smooth along it
template<class PrecisionTYPE>
void traverse_and_smooth_axes(int axes, PrecisionTYPE *histogram,
                              PrecisionTYPE *warped, PrecisionTYPE *window,
                              int num_dims, int *dimensions)
{
    SafeArray<int> indices(num_dims);
    for(int dim = 0; dim < num_dims; ++dim) indices[dim] = 0;

    smooth_axes<PrecisionTYPE>(axes, previous(axes, num_dims), histogram,
                               warped, window, num_dims, dimensions, indices);
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
    if(x<2.0){
        if(x<1.0)
            value = (PrecisionTYPE)(2.0f/3.0f + (0.5f*x-1.0)*x*x);
        else{
            x-=2.0f;
            value = -x*x*x/6.0f;
        }
    }
    return value;
}
/* *************************************************************** */
template<class PrecisionTYPE>
PrecisionTYPE GetBasisSplineDerivativeValue(PrecisionTYPE ori)
{
    PrecisionTYPE x=fabs(ori);
    PrecisionTYPE value=0.0;
    if(x<2.0){
        if(x<1.0)
            value = (PrecisionTYPE)((1.5f*x-2.0)*ori);
        else{
            x-=2.0f;
            value = -0.5f * x * x;
            if(ori<0.0f)value =-value;
        }
    }
    return value;
}

/* *************************************************************** */
/* *************************************************************** */
/// Multi channel NMI joint histogram and entropy calculation
template<class DTYPE>
void reg_getEntropies1(nifti_image *referenceImage,
                       nifti_image *warpedImage,
                       unsigned int *fixed_bins,
                       unsigned int *warped_bins,
                       double *probaJointHistogram,
                       double *logJointHistogram,
                       double *entropies,
                       int *mask,
                       bool approx)
{

    int num_fixed_volumes = referenceImage->nt;
    int num_warped_volumes = warpedImage->nt;
    int i, j, index;

    if(num_fixed_volumes>1 || num_warped_volumes>1) approx=true;

    int fixedVoxelNumber = referenceImage->nx * referenceImage->ny * referenceImage->nz;

    DTYPE *referenceImagePtr = static_cast<DTYPE *>(referenceImage->data);
    DTYPE *warpedImagePtr = static_cast<DTYPE *>(warpedImage->data);

    // Build up this arrays of offsets that will help us index the histogram entries
    SafeArray<int> fixed_offsets(num_fixed_volumes);
    SafeArray<int> warped_offsets(num_warped_volumes);

    int num_histogram_entries = 1;
    int total_fixed_entries = 1;
    int total_warped_entries = 1;

    // Data pointers
    SafeArray<int> histogram_dimensions(num_fixed_volumes + num_warped_volumes);

    // Calculate some constants and initialize the data pointers
    for (i = 0; i < num_fixed_volumes; ++i) {
        num_histogram_entries *= fixed_bins[i];
        total_fixed_entries *= fixed_bins[i];
        histogram_dimensions[i] = fixed_bins[i];

        fixed_offsets[i] = 1;
        for (j = i; j > 0; --j) fixed_offsets[i] *= fixed_bins[j - 1];
    }

    for (i = 0; i < num_warped_volumes; ++i) {
        num_histogram_entries *= warped_bins[i];
        total_warped_entries *= warped_bins[i];
        histogram_dimensions[num_fixed_volumes + i] = warped_bins[i];

        warped_offsets[i] = 1;
        for (j = i; j > 0; --j) warped_offsets[i] *= warped_bins[j-1];
    }

    int num_probabilities = num_histogram_entries;

    // Space for storing the marginal entropies.
    num_histogram_entries += total_fixed_entries + total_warped_entries;

    memset(probaJointHistogram, 0, num_histogram_entries * sizeof(double));
    memset(logJointHistogram, 0, num_histogram_entries * sizeof(double));

    // These hold the current fixed and warped values
    // No more than 10 timepoints are assumed
    DTYPE fixed_values[10];
    DTYPE warped_values[10];

    bool valid_values;

    DTYPE fixed_flat_index, warped_flat_index;
    double voxel_number = 0., added_value;

    // For now we only use the approximate PW approach for filling the joint histogram.
    // Fill the joint histogram using the classical approach
#ifdef _OPENMP
    int maxThreadNumber = omp_get_max_threads(), tid;
    double **tempHistogram=(double **)malloc(maxThreadNumber*sizeof(double *));
    for(i=0;i<maxThreadNumber;++i)
        tempHistogram[i]=(double *)calloc(num_histogram_entries,sizeof(double));
#pragma omp parallel for default(none) \
    shared(tempHistogram, num_fixed_volumes, num_warped_volumes, mask, \
    referenceImagePtr, warpedImagePtr, fixedVoxelNumber, fixed_bins, warped_bins, \
    fixed_offsets, warped_offsets, total_fixed_entries, approx) \
    private(index, i, valid_values, fixed_flat_index, tid, \
    warped_flat_index, fixed_values, warped_values, added_value) \
    reduction(+:voxel_number)
#endif
    for (index=0; index<fixedVoxelNumber; ++index){
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        if (mask[index] > -1) {
            added_value=0.;
            valid_values = true;
            fixed_flat_index = 0;

            // Get the fixed values
            for (i = 0; i < num_fixed_volumes; ++i) {
                fixed_values[i] = referenceImagePtr[index+i*fixedVoxelNumber];
                if (fixed_values[i] < (DTYPE)0 ||
                    fixed_values[i] >= (DTYPE)fixed_bins[i] ||
                    fixed_values[i] != fixed_values[i]) {
                    valid_values = false;
                    break;
                }
                // This needs to be cast to a valid int position!
                fixed_flat_index += static_cast<int>(fixed_values[i]) * (fixed_offsets[i]);
            }

            if (valid_values) {
                warped_flat_index = 0;
                // Get the warped values
                for (i = 0; i < num_warped_volumes; ++i){
                    warped_values[i] = warpedImagePtr[index+i*fixedVoxelNumber];
                    if (warped_values[i] < (DTYPE)0 ||
                        warped_values[i] >= (DTYPE)warped_bins[i] ||
                        warped_values[i] != warped_values[i]) {
                        valid_values = false;
                        break;
                    }
                    // This needs to be cast to a valid int position!
                    warped_flat_index += static_cast<int>(warped_values[i]) * (warped_offsets[i]);
                }
            }
            if (valid_values) {
                if(approx){ // standard joint histogram filling
#ifdef _OPENMP
                    tempHistogram[tid][static_cast<int>(fixed_flat_index) +
                            (static_cast<int>(warped_flat_index) * total_fixed_entries)]++;
#else
                    probaJointHistogram[static_cast<int>(fixed_flat_index) +
                            (static_cast<int>(warped_flat_index) * total_fixed_entries)]++;
#endif
                    ++voxel_number;
                }
                else{ // Parzen window joint histogram filling
                    for(int t=static_cast<int>(fixed_values[0]-1.); t<static_cast<int>(fixed_values[0]+2.); ++t){
                        if(t>=0 || t<static_cast<int>(fixed_bins[0])){
                            double fixed_weight = GetBasisSplineValue<double>(double(fixed_values[0])-double(t));
                            for(int r=static_cast<int>(warped_values[0]-1.); r<static_cast<int>(warped_values[0]+2.); ++r){
                                if(r>=0 || r<static_cast<int>(warped_bins[0])){
                                    double weight = fixed_weight * GetBasisSplineValue<double>(double(warped_values[0])-double(r));
                                    added_value+= weight;
#ifdef _OPENMP
                                    tempHistogram[tid][t + r * total_fixed_entries]  += weight;
#else
                                    probaJointHistogram[t + r * total_fixed_entries] += weight;
                                    voxel_number+=added_value;
#endif
                                }
                            }
                        }
                    }
                }
            }
        } //mask
    }
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(maxThreadNumber, num_histogram_entries, probaJointHistogram, tempHistogram) \
    private(i,j)
    for(i=0;i<num_histogram_entries;++i){
        for(j=0;j<maxThreadNumber;++j){
            probaJointHistogram[i] += tempHistogram[j][i];
        }
    }
    for(j=0;j<maxThreadNumber;++j)
        free(tempHistogram[j]);
    free(tempHistogram);
#endif

    int num_axes = num_fixed_volumes + num_warped_volumes;
    if(approx || referenceImage->nt>1 || warpedImage->nt>1){
    // standard joint histogram filling has been used
    // Joint histogram has to be smoothed
        double window[3];
        window[0] = window[2] = GetBasisSplineValue((double)(-1.0));
        window[1] = GetBasisSplineValue((double)(0.0));

        double *histogram=NULL;
        double *warped=NULL;

        // Smooth along each of the axes
        for (i = 0; i < num_axes; ++i)
        {
            // Use the arrays for storage of warpeds
            if (i % 2 == 0) {
                warped = logJointHistogram;
                histogram = probaJointHistogram;
            }
            else {
                warped = probaJointHistogram;
                histogram = logJointHistogram;
            }
            traverse_and_smooth_axes<double>(i, histogram, warped, window,
                                             num_axes, histogram_dimensions);
        }

        // We may need to transfer the warped
        if (warped == logJointHistogram) memcpy(probaJointHistogram, logJointHistogram,
                                                sizeof(double)*num_probabilities);
    }// approx
    memset(logJointHistogram, 0, num_histogram_entries * sizeof(double));

    // Convert to probabilities
    for(i = 0; i < num_probabilities; ++i) {
        if (probaJointHistogram[i]) probaJointHistogram[i] /= voxel_number;
    }

    // Marginalise over all the warped axes to generate the fixed entropy
    double *data = probaJointHistogram;
    double *store = logJointHistogram;
    double current_value, current_log;

    int count;
    double fixed_entropy = 0;
    {
        SafeArray<double> scratch (num_probabilities/histogram_dimensions[num_axes - 1]);
        // marginalise over the warped axes
        for (i = num_warped_volumes-1, count = 0; i >= 0; --i, ++count)
        {
            traverse_and_sum_axes<double>(num_axes - count - 1,
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

        // Generate fixed entropy
        double *log_joint_fixed = &logJointHistogram[num_probabilities];

        for (i = 0; i < total_fixed_entries; ++i)
        {
            current_value = data[i];
            current_log = 0;
            if (current_value) current_log = log(current_value);
            fixed_entropy -= current_value * current_log;
            log_joint_fixed[i] = current_log;
        }
    }
    memset(logJointHistogram, 0, num_probabilities * sizeof(double));
    data = probaJointHistogram;
    store = logJointHistogram;

    // Marginalise over the fixed axes
    double warped_entropy = 0;
    {
        SafeArray<double> scratch (num_probabilities / histogram_dimensions[0]);
        for (i = 0; i < num_fixed_volumes; ++i)
        {
            traverse_and_sum_axes<double>(0, data, store, num_axes - i, &histogram_dimensions[i]);
            if (i % 2 == 0) {
                data = logJointHistogram;
                store = scratch;
            }
            else {
                data = scratch;
                store = logJointHistogram;
            }
        }
        // Generate warped entropy
        double *log_joint_warped = &logJointHistogram[num_probabilities+total_fixed_entries];

        for (i = 0; i < total_warped_entries; ++i)
        {
            current_value = data[i];
            current_log = 0;
            if (current_value) current_log = log(current_value);
            warped_entropy -= current_value * current_log;
            log_joint_warped[i] = current_log;
        }
    }

    // Generate joint entropy
    double joint_entropy = 0;
    for (i = 0; i < num_probabilities; ++i)
    {
        current_value = probaJointHistogram[i];
        current_log = 0;
        if (current_value) current_log = log(current_value);
        joint_entropy -= current_value * current_log;
        logJointHistogram[i] = current_log;
    }

    entropies[0] = fixed_entropy;
    entropies[1] = warped_entropy;
    entropies[2] = joint_entropy;
    entropies[3] = voxel_number;

    return;
}
/***************************************************************** */
extern "C++"
void reg_getEntropies(nifti_image *referenceImage,
                      nifti_image *warpedImage,
                      unsigned int *fixed_bins, // should be an array of size num_fixed_volumes
                      unsigned int *warped_bins, // should be an array of size num_warped_volumes
                      double *probaJointHistogram,
                      double *logJointHistogram,
                      double *entropies,
                      int *mask,
                      bool approx)
{
    if(referenceImage->datatype != warpedImage->datatype){
        fprintf(stderr, "[NiftyReg ERROR] reg_getEntropies\n");
        fprintf(stderr, "[NiftyReg ERROR] Both input images are exepected to have the same type\n");
        exit(1);
    }

    switch(referenceImage->datatype){
    case NIFTI_TYPE_FLOAT32:
        reg_getEntropies1<float>
                (referenceImage, warpedImage, /*type,*/ fixed_bins, warped_bins, probaJointHistogram,
                 logJointHistogram, entropies, mask, approx);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_getEntropies1<double>
                (referenceImage, warpedImage, /*type,*/ fixed_bins, warped_bins, probaJointHistogram,
                 logJointHistogram, entropies, mask, approx);
        break;
    default:
        fprintf(stderr,"[NiftyReg ERROR] reg_getEntropies\tThe fixed image data type is not supported\n");
        exit(1);
    }
    return;
}
/* *************************************************************** */
/* *************************************************************** */
/// Voxel based multichannel gradient computation
template<class DTYPE,class GradTYPE>
void reg_getVoxelBasedNMIGradientUsingPW2D(nifti_image *referenceImage,
                                           nifti_image *warpedImage,
                                           //int type, //! Not used at the moment
                                           nifti_image *warpedImageGradient,
                                           unsigned int *fixed_bins,
                                           unsigned int *warped_bins,
                                           double *logJointHistogram,
                                           double *entropies,
                                           nifti_image *nmiGradientImage,
                                           int *mask,
                                           bool approx)
{
    unsigned int num_fixed_volumes=referenceImage->nt;
    unsigned int num_warped_volumes=warpedImage->nt;
    unsigned int num_loops = num_fixed_volumes + num_warped_volumes;

    if(num_fixed_volumes>1 || num_warped_volumes>1) approx=true;

    unsigned fixedVoxelNumber = referenceImage->nx * referenceImage->ny;

    DTYPE *referenceImagePtr = static_cast<DTYPE *>(referenceImage->data);
    DTYPE *warpedImagePtr = static_cast<DTYPE *>(warpedImage->data);
    GradTYPE *resulImageGradientPtrX = static_cast<GradTYPE *>(warpedImageGradient->data);
    GradTYPE *resulImageGradientPtrY = &resulImageGradientPtrX[fixedVoxelNumber*num_warped_volumes];

    // Build up this arrays of offsets that will help us index the histogram entries
    int fixed_offsets[10];
    int warped_offsets[10];

    int total_fixed_entries = 1;
    int total_warped_entries = 1;

    // The 4D
    for (unsigned int i = 0; i < num_fixed_volumes; ++i) {
        total_fixed_entries *= fixed_bins[i];
        fixed_offsets[i] = 1;
        for (int j = i; j > 0; --j) fixed_offsets[i] *= fixed_bins[j - 1];
    }

    for (unsigned int i = 0; i < num_warped_volumes; ++i) {
        total_warped_entries *= warped_bins[i];
        warped_offsets[i] = 1;
        for (int j = i; j > 0; --j) warped_offsets[i] *= warped_bins[j - 1];
    }

    int num_probabilities = total_fixed_entries * total_warped_entries;

    int *maskPtr = &mask[0];
    double NMI = (entropies[0] + entropies[1]) / entropies[2];

    // Hold current values.
    // fixed and warped images limited to 10 max for speed.
    DTYPE voxel_values[20];
    GradTYPE warped_gradient_x_values[10];
    GradTYPE warped_gradient_y_values[10];

    bool valid_values;
    double common_fixed_value;

    double jointEntropyDerivative_X;
    double warpedEntropyDerivative_X;
    double fixedEntropyDerivative_X;

    double jointEntropyDerivative_Y;
    double warpedEntropyDerivative_Y;
    double fixedEntropyDerivative_Y;

    double jointLog, fixedLog, warpedLog;
    double normalised_joint_entropy = entropies[2]*entropies[3];

    GradTYPE *nmiGradientPtrX = static_cast<GradTYPE *>(nmiGradientImage->data);
    GradTYPE *nmiGradientPtrY = &nmiGradientPtrX[fixedVoxelNumber];
    memset(nmiGradientPtrX,0,nmiGradientImage->nvox*nmiGradientImage->nbyper);

    // Set up the multi loop
    Multi_Loop<int> loop;
    for (unsigned int i = 0; i < num_loops; ++i) loop.Add(-1, 3);

    SafeArray<int> bins(num_loops);
    for (unsigned int i = 0; i < num_fixed_volumes; ++i) bins[i] = fixed_bins[i];
    for (unsigned int i = 0; i < num_warped_volumes; ++i) bins[i + num_fixed_volumes] = warped_bins[i];

    double coefficients[20];
    double positions[20];
    int relative_positions[20];

    double warped_common[2];
    double der_term[3];

    // Loop over all the voxels
    for (unsigned int index = 0; index < fixedVoxelNumber; ++index) {
        if(*maskPtr++>-1){
            valid_values = true;
            // Collect the fixed intensities and do some sanity checking
            for (unsigned int i = 0; i < num_fixed_volumes; ++i) {
                voxel_values[i] = referenceImagePtr[index+i*fixedVoxelNumber];
                if (voxel_values[i] <= (DTYPE)0 ||
                    voxel_values[i] >= (DTYPE)fixed_bins[i] ||
                    voxel_values[i] != voxel_values[i]) {
                    valid_values = false;
                    break;
                }
            }

            // Collect the warped intensities and do some sanity checking
            if (valid_values) {
                for (unsigned int i = 0; i < num_warped_volumes; ++i) {
                    unsigned int currentIndex = index+i*fixedVoxelNumber;
                    DTYPE temp = warpedImagePtr[currentIndex];
                    warped_gradient_x_values[i] = resulImageGradientPtrX[currentIndex];
                    warped_gradient_y_values[i] = resulImageGradientPtrY[currentIndex];

                    if (temp <= (DTYPE)0 ||
                        temp >= (DTYPE)warped_bins[i] ||
                        temp != temp ||
                        warped_gradient_x_values[i] != warped_gradient_x_values[i] ||
                        warped_gradient_y_values[i] != warped_gradient_y_values[i]) {
                        valid_values = false;
                        break;
                    }
                    voxel_values[num_fixed_volumes + i] = temp;
                }
            }
            if (valid_values) {
                jointEntropyDerivative_X = 0.0;
                warpedEntropyDerivative_X = 0.0;
                fixedEntropyDerivative_X = 0.0;

                jointEntropyDerivative_Y = 0.0;
                warpedEntropyDerivative_Y = 0.0;
                fixedEntropyDerivative_Y = 0.0;

                int fixed_flat_index, warped_flat_index;

                for (loop.Initialise(); loop.Continue(); loop.Next()) {
                    fixed_flat_index = warped_flat_index = 0;
                    valid_values = true;

                    for(unsigned int lc = 0; lc < num_fixed_volumes; ++lc){
                        int relative_pos = int(voxel_values[lc] + loop.Index(lc));
                        if(relative_pos< 0 || relative_pos >= bins[lc]){
                            valid_values = false; break;
                        }
                        double common_value = GetBasisSplineValue<double>((double)voxel_values[lc]-(double)relative_pos);
                        coefficients[lc] = common_value;
                        positions[lc] = (double)voxel_values[lc]-(double)relative_pos;
                        relative_positions[lc] = relative_pos;
                    }

                    for(unsigned int jc = num_fixed_volumes; jc < num_loops; ++jc){
                        int relative_pos = int(voxel_values[jc] + loop.Index(jc));
                        if(relative_pos< 0 || relative_pos >= bins[jc]){
                            valid_values = false; break;
                        }
                        if (num_warped_volumes > 1) {
                            double common_value = GetBasisSplineValue<double>((double)voxel_values[jc]-(double)relative_pos);
                            coefficients[jc] = common_value;
                        }
                        positions[jc] = (double)voxel_values[jc]-(double)relative_pos;
                        relative_positions[jc] = relative_pos;
                    }

                    if(valid_values) {
                        common_fixed_value = (double)1.0;
                        for (unsigned int i = 0; i < num_fixed_volumes; ++i)
                            common_fixed_value *= coefficients[i];

                        warped_common[0] = warped_common[1] = (double)0.0;

                        for (unsigned int i = 0; i < num_warped_volumes; ++i)
                        {
                            der_term[0] = der_term[1] = der_term[2] = (double)1.0;
                            for (unsigned int j = 0; j < num_warped_volumes; ++j)
                            {
                                if (i == j) {
                                    double reg = GetBasisSplineDerivativeValue<double>
                                            ((double)positions[j + num_fixed_volumes]);
                                    der_term[0] *= reg * (double)warped_gradient_x_values[j];
                                    der_term[1] *= reg * (double)warped_gradient_y_values[j];
                                }
                                else {
                                    der_term[0] *= coefficients[j+num_fixed_volumes];
                                    der_term[1] *= coefficients[j+num_fixed_volumes];
                                }
                            }
                            warped_common[0] += der_term[0];
                            warped_common[1] += der_term[1];
                        }

                        warped_common[0] *= common_fixed_value;
                        warped_common[1] *= common_fixed_value;

                        for (unsigned int i = 0; i < num_fixed_volumes; ++i) fixed_flat_index += relative_positions[i] * fixed_offsets[i];
                        for (unsigned int i = 0; i < num_warped_volumes; ++i) warped_flat_index += relative_positions[i + num_fixed_volumes] * warped_offsets[i];

                        jointLog = logJointHistogram[fixed_flat_index * total_warped_entries + warped_flat_index];
                        fixedLog = logJointHistogram[num_probabilities + fixed_flat_index];
                        warpedLog = logJointHistogram[num_probabilities + total_fixed_entries + warped_flat_index];

                        jointEntropyDerivative_X -= warped_common[0] * (jointLog + 1.0);
                        fixedEntropyDerivative_X -= warped_common[0] * (fixedLog + 1.0);
                        warpedEntropyDerivative_X -= warped_common[0] * (warpedLog + 1.0);

                        jointEntropyDerivative_Y -= warped_common[1] * (jointLog + 1.0);
                        fixedEntropyDerivative_Y -= warped_common[1] * (fixedLog + 1.0);
                        warpedEntropyDerivative_Y -= warped_common[1] * (warpedLog + 1.0);
                    }
                }
                *nmiGradientPtrX = (GradTYPE)((fixedEntropyDerivative_X + warpedEntropyDerivative_X - NMI * jointEntropyDerivative_X) / normalised_joint_entropy);
                *nmiGradientPtrY = (GradTYPE)((fixedEntropyDerivative_Y + warpedEntropyDerivative_Y - NMI * jointEntropyDerivative_Y) / normalised_joint_entropy);
            }
        }

        nmiGradientPtrX++; nmiGradientPtrY++;
    }
}
/* *************************************************************** */
/* *************************************************************** */
/// Voxel based multichannel gradient computation
template<class DTYPE,class GradTYPE>
void reg_getVoxelBasedNMIGradientUsingPW3D(nifti_image *referenceImage,
                                           nifti_image *warpedImage,
                                           nifti_image *warpedImageGradient,
                                           unsigned int *fixed_bins,
                                           unsigned int *warped_bins,
                                           double *logJointHistogram,
                                           double *entropies,
                                           nifti_image *nmiGradientImage,
                                           int *mask,
                                           bool approx)
{
    int num_fixed_volumes=referenceImage->nt;
    int num_warped_volumes=warpedImage->nt;
    int num_loops = num_fixed_volumes + num_warped_volumes;

    if(num_fixed_volumes>1 || num_warped_volumes>1) approx=true;

    int fixedVoxelNumber = referenceImage->nx * referenceImage->ny * referenceImage->nz;

    DTYPE *referenceImagePtr = static_cast<DTYPE *>(referenceImage->data);
    DTYPE *warpedImagePtr = static_cast<DTYPE *>(warpedImage->data);
    GradTYPE *warpedImageGradientPtrX = static_cast<GradTYPE *>(warpedImageGradient->data);
    GradTYPE *warpedImageGradientPtrY = &warpedImageGradientPtrX[fixedVoxelNumber*num_warped_volumes];
    GradTYPE *warpedImageGradientPtrZ = &warpedImageGradientPtrY[fixedVoxelNumber*num_warped_volumes];

    // Build up this arrays of offsets that will help us index the histogram entries
    int fixed_offsets[10];
    int warped_offsets[10];

    int total_fixed_entries = 1;
    int total_warped_entries = 1;

    // The 4D
    for (int i = 0; i < num_fixed_volumes; ++i) {
        total_fixed_entries *= fixed_bins[i];
        fixed_offsets[i] = 1;
        for (int j = i; j > 0; --j) fixed_offsets[i] *= fixed_bins[j - 1];
    }

    for (int i = 0; i < num_warped_volumes; ++i) {
        total_warped_entries *= warped_bins[i];
        warped_offsets[i] = 1;
        for (int j = i; j > 0; --j) warped_offsets[i] *= warped_bins[j - 1];
    }

    int num_probabilities = total_fixed_entries * total_warped_entries;

    double NMI = (entropies[0] + entropies[1]) / entropies[2];

    // Hold current values.
    // fixed and warped images limited to 10 max for speed.
    DTYPE voxel_values[20];
    GradTYPE warped_gradient_x_values[10];
    GradTYPE warped_gradient_y_values[10];
    GradTYPE warped_gradient_z_values[10];

    bool valid_values;
    double common_fixed_value;

    double jointEntropyDerivative_X;
    double warpedEntropyDerivative_X;
    double fixedEntropyDerivative_X;

    double jointEntropyDerivative_Y;
    double warpedEntropyDerivative_Y;
    double fixedEntropyDerivative_Y;

    double jointEntropyDerivative_Z;
    double warpedEntropyDerivative_Z;
    double fixedEntropyDerivative_Z;

    double jointLog, fixedLog, warpedLog;
    double joint_entropy = entropies[2]*entropies[3];

    GradTYPE *nmiGradientPtrX = static_cast<GradTYPE *>(nmiGradientImage->data);
    GradTYPE *nmiGradientPtrY = &nmiGradientPtrX[fixedVoxelNumber];
    GradTYPE *nmiGradientPtrZ = &nmiGradientPtrY[fixedVoxelNumber];
    memset(nmiGradientPtrX,0,nmiGradientImage->nvox*nmiGradientImage->nbyper);

    // Set up the multi loop
    Multi_Loop<int> loop;
    for (int i = 0; i < num_loops; ++i) loop.Add(-1, 3);

    SafeArray<int> bins(num_loops);
    for (int i = 0; i < num_fixed_volumes; ++i) bins[i] = fixed_bins[i];
    for (int i = 0; i < num_warped_volumes; ++i) bins[i + num_fixed_volumes] = warped_bins[i];

    GradTYPE coefficients[20];
    GradTYPE positions[20];
    int relative_positions[20];

    GradTYPE warped_common[3];
    GradTYPE der_term[3];

    int index, currentIndex, relative_pos, i, j, lc, jc;
    int fixed_flat_index, warped_flat_index;
    DTYPE temp;
    GradTYPE reg;

    // Loop over all the voxels
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    firstprivate(loop) \
    private(index, valid_values, i, j, lc, jc, voxel_values, currentIndex, temp, \
    warped_gradient_x_values, warped_gradient_y_values, warped_gradient_z_values, \
    jointEntropyDerivative_X, jointEntropyDerivative_Y, jointEntropyDerivative_Z, \
    warpedEntropyDerivative_X, warpedEntropyDerivative_Y, warpedEntropyDerivative_Z, \
    fixedEntropyDerivative_X, fixedEntropyDerivative_Y, fixedEntropyDerivative_Z, \
    fixed_flat_index, warped_flat_index, relative_pos, coefficients, reg, \
    positions, relative_positions, common_fixed_value, warped_common, der_term, \
    jointLog, fixedLog, warpedLog) \
    shared(referenceImagePtr, fixedVoxelNumber, mask, num_fixed_volumes, fixed_bins, \
    num_warped_volumes, warped_bins, warpedImagePtr, bins, num_loops, num_probabilities, \
    warped_offsets, fixed_offsets, total_fixed_entries, logJointHistogram, \
    warpedImageGradientPtrX, warpedImageGradientPtrY, warpedImageGradientPtrZ, \
    nmiGradientPtrX, nmiGradientPtrY, nmiGradientPtrZ, NMI, joint_entropy, approx)
#endif // _OPENMP
    for (index = 0; index < fixedVoxelNumber; ++index){
        if(mask[index]>-1){
            valid_values = true;
            // Collect the fixed intensities and do some sanity checking
            for (i = 0; i < num_fixed_volumes; ++i) {
                voxel_values[i] = referenceImagePtr[index+i*fixedVoxelNumber];
                if (voxel_values[i] <= (DTYPE)0 ||
                    voxel_values[i] >= (DTYPE)fixed_bins[i] ||
                    voxel_values[i] != voxel_values[i]) {
                    valid_values = false;
                    break;
                }
            }

            // Collect the warped intensities and do some sanity checking
            if (valid_values) {
                for (i = 0; i < num_warped_volumes; ++i) {
                    currentIndex = index+i*fixedVoxelNumber;
                    temp = warpedImagePtr[currentIndex];
                    warped_gradient_x_values[i] = warpedImageGradientPtrX[currentIndex];
                    warped_gradient_y_values[i] = warpedImageGradientPtrY[currentIndex];
                    warped_gradient_z_values[i] = warpedImageGradientPtrZ[currentIndex];

                    if (temp <= (DTYPE)0 ||
                        temp >= (DTYPE)warped_bins[i] ||
                        temp != temp ||
                        warped_gradient_x_values[i] != warped_gradient_x_values[i] ||
                        warped_gradient_y_values[i] != warped_gradient_y_values[i] ||
                        warped_gradient_z_values[i] != warped_gradient_z_values[i]) {
                        valid_values = false;
                        break;
                    }
                    voxel_values[num_fixed_volumes + i] = temp;
                }
            }
            if (valid_values) {
                jointEntropyDerivative_X = 0.0;
                warpedEntropyDerivative_X = 0.0;
                fixedEntropyDerivative_X = 0.0;

                jointEntropyDerivative_Y = 0.0;
                warpedEntropyDerivative_Y = 0.0;
                fixedEntropyDerivative_Y = 0.0;

                jointEntropyDerivative_Z = 0.0;
                warpedEntropyDerivative_Z = 0.0;
                fixedEntropyDerivative_Z = 0.0;

                for (loop.Initialise(); loop.Continue(); loop.Next()) {
                    fixed_flat_index = warped_flat_index = 0;
                    valid_values = true;

                    for(lc = 0; lc < num_fixed_volumes; ++lc){
                        relative_pos = static_cast<int>(voxel_values[lc] + loop.Index(lc));
                        if(relative_pos< 0 || relative_pos >= bins[lc]){
                            valid_values = false; break;
                        }
                        double common_value = GetBasisSplineValue<double>((double)voxel_values[lc]-(double)relative_pos);
                        coefficients[lc] = common_value;
                        positions[lc] = (GradTYPE)voxel_values[lc]-(GradTYPE)relative_pos;
                        relative_positions[lc] = relative_pos;
                    }

                    for(jc = num_fixed_volumes; jc < num_loops; ++jc){
                        relative_pos = static_cast<int>(voxel_values[jc] + loop.Index(jc));
                        if(relative_pos< 0 || relative_pos >= bins[jc]){
                            valid_values = false; break;
                        }
                        if (num_warped_volumes > 1) {
                            double common_value = GetBasisSplineValue<double>((double)voxel_values[jc]-(double)relative_pos);
                            coefficients[jc] = common_value;
                        }
                        positions[jc] = (GradTYPE)voxel_values[jc]-(GradTYPE)relative_pos;
                        relative_positions[jc] = relative_pos;
                    }

                    if(valid_values) {
                        common_fixed_value = (GradTYPE)1.0;
                        for (i = 0; i < num_fixed_volumes; ++i)
                            common_fixed_value *= coefficients[i];

                        warped_common[0] = warped_common[1] = warped_common[2] = (GradTYPE)0.0;

                        for (i = 0; i < num_warped_volumes; ++i){
                            der_term[0] = der_term[1] = der_term[2] = (GradTYPE)1.0;
                            for (j = 0; j < num_warped_volumes; ++j){
                                if (i == j) {
                                    reg = GetBasisSplineDerivativeValue<double>
                                            ((double)positions[j + num_fixed_volumes]);
                                    der_term[0] *= reg * (GradTYPE)warped_gradient_x_values[j];
                                    der_term[1] *= reg * (GradTYPE)warped_gradient_y_values[j];
                                    der_term[2] *= reg * (GradTYPE)warped_gradient_z_values[j];
                                }
                                else {
                                    der_term[0] *= coefficients[j+num_fixed_volumes];
                                    der_term[1] *= coefficients[j+num_fixed_volumes];
                                    der_term[2] *= coefficients[j+num_fixed_volumes];
                                }
                            }
                            warped_common[0] += der_term[0];
                            warped_common[1] += der_term[1];
                            warped_common[2] += der_term[2];
                        }

                        warped_common[0] *= common_fixed_value;
                        warped_common[1] *= common_fixed_value;
                        warped_common[2] *= common_fixed_value;

                        for (i = 0; i < num_fixed_volumes; ++i)
                            fixed_flat_index += relative_positions[i] * fixed_offsets[i];
                        for (i = 0; i < num_warped_volumes; ++i)
                            warped_flat_index += relative_positions[i + num_fixed_volumes] * warped_offsets[i];

                        jointLog = logJointHistogram[fixed_flat_index * total_warped_entries + warped_flat_index];
                        fixedLog = logJointHistogram[num_probabilities + fixed_flat_index];
                        warpedLog = logJointHistogram[num_probabilities + total_fixed_entries + warped_flat_index];

                        jointEntropyDerivative_X -= warped_common[0] * jointLog;
                        fixedEntropyDerivative_X -= warped_common[0] * fixedLog;
                        warpedEntropyDerivative_X -= warped_common[0] * warpedLog;

                        jointEntropyDerivative_Y -= warped_common[1] * jointLog;
                        fixedEntropyDerivative_Y -= warped_common[1] * fixedLog;
                        warpedEntropyDerivative_Y -= warped_common[1] * warpedLog;

                        jointEntropyDerivative_Z -= warped_common[2] * jointLog;
                        fixedEntropyDerivative_Z -= warped_common[2] * fixedLog;
                        warpedEntropyDerivative_Z -= warped_common[2] * warpedLog;
                    }

                    nmiGradientPtrX[index] = (GradTYPE)((fixedEntropyDerivative_X + warpedEntropyDerivative_X - NMI * jointEntropyDerivative_X) / joint_entropy);
                    nmiGradientPtrY[index] = (GradTYPE)((fixedEntropyDerivative_Y + warpedEntropyDerivative_Y - NMI * jointEntropyDerivative_Y) / joint_entropy);
                    nmiGradientPtrZ[index] = (GradTYPE)((fixedEntropyDerivative_Z + warpedEntropyDerivative_Z - NMI * jointEntropyDerivative_Z) / joint_entropy);
                }
            }
        }
    }
}
/* *************************************************************** */
template<class DTYPE>
void reg_getVoxelBasedNMIGradientUsingPW1(nifti_image *referenceImage,
                                          nifti_image *warpedImage,
                                          nifti_image *warpedImageGradient,
                                          unsigned int *fixed_bins,
                                          unsigned int *warped_bins,
                                          double *logJointHistogram,
                                          double *entropies,
                                          nifti_image *nmiGradientImage,
                                          int *mask,
                                          bool approx)
{
    if(warpedImageGradient->datatype != nmiGradientImage->datatype){
        fprintf(stderr, "[NiftyReg ERROR] reg_getVoxelBasedNMIGradientUsingPW\n");
        fprintf(stderr, "[NiftyReg ERROR] Both gradient images are exepected to have the same type\n");
        exit(1);
    }

    if(nmiGradientImage->nz==1){
        switch(warpedImageGradient->datatype){
        case NIFTI_TYPE_FLOAT32:
            reg_getVoxelBasedNMIGradientUsingPW2D<DTYPE,float>
                    (referenceImage, warpedImage, warpedImageGradient, fixed_bins, warped_bins, logJointHistogram,
                     entropies, nmiGradientImage, mask, approx);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_getVoxelBasedNMIGradientUsingPW2D<DTYPE,double>
                    (referenceImage, warpedImage, warpedImageGradient, fixed_bins, warped_bins, logJointHistogram,
                     entropies, nmiGradientImage, mask, approx);
            break;
        default:
            fprintf(stderr,"[NiftyReg ERROR] reg_getVoxelBasedNMIGradientUsingPW\tThe gradient images data type is not supported\n");
            exit(1);
        }
    }
    else{
        switch(warpedImageGradient->datatype){
        case NIFTI_TYPE_FLOAT32:
            reg_getVoxelBasedNMIGradientUsingPW3D<DTYPE,float>
                    (referenceImage, warpedImage, warpedImageGradient, fixed_bins, warped_bins, logJointHistogram,
                     entropies, nmiGradientImage, mask, approx);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_getVoxelBasedNMIGradientUsingPW3D<DTYPE,double>
                    (referenceImage, warpedImage, warpedImageGradient, fixed_bins, warped_bins, logJointHistogram,
                     entropies, nmiGradientImage, mask, approx);
            break;
        default:
            fprintf(stderr,"[NiftyReg ERROR] reg_getVoxelBasedNMIGradientUsingPW\tThe gradient images data type is not supported\n");
            exit(1);
        }

    }
}
/* *************************************************************** */
void reg_getVoxelBasedNMIGradientUsingPW(nifti_image *referenceImage,
                                         nifti_image *warpedImage,
                                         nifti_image *warpedImageGradient,
                                         unsigned int *fixed_bins,
                                         unsigned int *warped_bins,
                                         double *logJointHistogram,
                                         double *entropies,
                                         nifti_image *nmiGradientImage,
                                         int *mask,
                                         bool approx)
{
    if(referenceImage->datatype != warpedImage->datatype){
        fprintf(stderr, "[NiftyReg ERROR] reg_getVoxelBasedNMIGradientUsingPW\n");
        fprintf(stderr, "[NiftyReg ERROR] Both input images are exepected to have the same type\n");
        exit(1);
    }

    switch(referenceImage->datatype){
    case NIFTI_TYPE_FLOAT32:
        reg_getVoxelBasedNMIGradientUsingPW1<float>
                (referenceImage, warpedImage, warpedImageGradient, fixed_bins, warped_bins, logJointHistogram,
                 entropies, nmiGradientImage, mask, approx);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_getVoxelBasedNMIGradientUsingPW1<double>
                (referenceImage, warpedImage, warpedImageGradient, fixed_bins, warped_bins, logJointHistogram,
                 entropies, nmiGradientImage, mask, approx);
        break;
    default:
        fprintf(stderr,"[NiftyReg ERROR] reg_getVoxelBasedNMIGradientUsingPW\tThe input image data type is not supported\n");
        exit(1);
    }
}
/* *************************************************************** */
/* *************************************************************** */

#endif
