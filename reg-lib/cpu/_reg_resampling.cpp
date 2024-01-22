/*
 *  _reg_resampling.cpp
 *
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_resampling.h"
#include "_reg_maths.h"
#include "_reg_maths_eigen.h"
#include "_reg_tools.h"

#define SINC_KERNEL_RADIUS 3
#define SINC_KERNEL_SIZE SINC_KERNEL_RADIUS*2

/* *************************************************************** */
void interpWindowedSincKernel(double relative, double *basis) {
    if (relative < 0) relative = 0; //reg_rounding error
    int j = 0;
    double sum = 0.;
    for (int i = -SINC_KERNEL_RADIUS; i < SINC_KERNEL_RADIUS; ++i) {
        double x = relative - static_cast<double>(i);
        if (x == 0)
            basis[j] = 1.0;
        else if (fabs(x) >= static_cast<double>(SINC_KERNEL_RADIUS))
            basis[j] = 0;
        else {
            double pi_x = M_PI * x;
            basis[j] = static_cast<double>(SINC_KERNEL_RADIUS) *
                sin(pi_x) *
                sin(pi_x / static_cast<double>(SINC_KERNEL_RADIUS)) /
                (pi_x * pi_x);
        }
        sum += basis[j];
        j++;
    }
    for (int i = 0; i < SINC_KERNEL_SIZE; ++i)
        basis[i] /= sum;
}
/* *************************************************************** */
double interpWindowedSincKernel_Samp(double x, double kernelsize) {
    if (x == 0)
        return 1.0;
    else if (fabs(x) >= static_cast<double>(kernelsize))
        return 0;
    else {
        double pi_x = M_PI * fabs(x);
        return static_cast<double>(kernelsize) *
            sin(pi_x) *
            sin(pi_x / static_cast<double>(kernelsize)) /
            (pi_x * pi_x);
    }
}
/* *************************************************************** */
void interpCubicSplineKernel(double relative, double *basis) {
    if (relative < 0) relative = 0; //reg_rounding error
    double FF = relative * relative;
    basis[0] = (relative * ((2.0 - relative) * relative - 1.0)) / 2.0;
    basis[1] = (FF * (3.0 * relative - 5.0) + 2.0) / 2.0;
    basis[2] = (relative * ((4.0 - 3.0 * relative) * relative + 1.0)) / 2.0;
    basis[3] = (relative - 1.0) * FF / 2.0;
}
/* *************************************************************** */
void interpCubicSplineKernel(double relative, double *basis, double *derivative) {
    interpCubicSplineKernel(relative, basis);
    if (relative < 0) relative = 0; //reg_rounding error
    double FF = relative * relative;
    derivative[0] = (4.0 * relative - 3.0 * FF - 1.0) / 2.0;
    derivative[1] = (9.0 * relative - 10.0) * relative / 2.0;
    derivative[2] = (8.0 * relative - 9.0 * FF + 1.0) / 2.0;
    derivative[3] = (3.0 * relative - 2.0) * relative / 2.0;
}
/* *************************************************************** */
void interpLinearKernel(double relative, double *basis) {
    if (relative < 0) relative = 0; //reg_rounding error
    basis[1] = relative;
    basis[0] = 1.0 - relative;
}
/* *************************************************************** */
void interpLinearKernel(double relative, double *basis, double *derivative) {
    interpLinearKernel(relative, basis);
    derivative[1] = 1;
    derivative[0] = 0;
}
/* *************************************************************** */
void interpNearestNeighKernel(double relative, double *basis) {
    if (relative < 0) relative = 0; //reg_rounding error
    basis[0] = basis[1] = 0;
    if (relative >= 0.5)
        basis[1] = 1;
    else basis[0] = 1;
}
/* *************************************************************** */
template <class DataType>
void reg_dti_resampling_preprocessing(nifti_image *floatingImage,
                                      void **originalFloatingData,
                                      const int *dtIndicies) {
    // If we have some valid diffusion tensor indicies, we need to replace the tensor components
    // by the the log tensor components
    if (dtIndicies[0] != -1) {
#ifndef NDEBUG
        std::string msg = "DTI indices: Active time point:";
        for (unsigned i = 0; i < 6; i++)
            msg += " " + std::to_string(dtIndicies[i]);
        NR_DEBUG(msg);
#endif

#ifdef WIN32
        long floatingIndex;
        const long floatingVoxelNumber = (long)NiftiImage::calcVoxelNumber(floatingImage, 3);
#else
        size_t floatingIndex;
        const size_t floatingVoxelNumber = NiftiImage::calcVoxelNumber(floatingImage, 3);
#endif

        *originalFloatingData = malloc(floatingImage->nvox * sizeof(DataType));
        memcpy(*originalFloatingData, floatingImage->data, floatingImage->nvox * sizeof(DataType));
        NR_DEBUG("The floating image data has been copied");

        // As the tensor has 6 unique components that we need to worry about, read them out
        // for the floating image.
        DataType *firstVox = static_cast<DataType*>(floatingImage->data);
        // CAUTION: Here the tensor is assumed to be encoding in lower triangular order
        DataType *floatingIntensityXX = &firstVox[floatingVoxelNumber * dtIndicies[0]];
        DataType *floatingIntensityXY = &firstVox[floatingVoxelNumber * dtIndicies[1]];
        DataType *floatingIntensityYY = &firstVox[floatingVoxelNumber * dtIndicies[2]];
        DataType *floatingIntensityXZ = &firstVox[floatingVoxelNumber * dtIndicies[3]];
        DataType *floatingIntensityYZ = &firstVox[floatingVoxelNumber * dtIndicies[4]];
        DataType *floatingIntensityZZ = &firstVox[floatingVoxelNumber * dtIndicies[5]];

        // Should log the tensor up front
        // We need to take the logarithm of the tensor for each voxel in the floating intensity
        // image, and replace the warped
        int tid = 0;
#ifdef _OPENMP
        mat33 diffTensor[16];
        int max_thread_number = omp_get_max_threads();
        if (max_thread_number > 16) omp_set_num_threads(16);
#pragma omp parallel for default(none) \
    private(tid) \
    shared(floatingVoxelNumber,floatingIntensityXX,floatingIntensityYY, \
    floatingIntensityZZ,floatingIntensityXY,floatingIntensityXZ, \
    floatingIntensityYZ, diffTensor)
#else
        mat33 diffTensor[1];
#endif
        for (floatingIndex = 0; floatingIndex < floatingVoxelNumber; ++floatingIndex) {
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            // Fill a mat44 with the tensor components
            diffTensor[tid].m[0][0] = static_cast<float>(floatingIntensityXX[floatingIndex]);
            diffTensor[tid].m[0][1] = static_cast<float>(floatingIntensityXY[floatingIndex]);
            diffTensor[tid].m[1][0] = diffTensor[tid].m[0][1];
            diffTensor[tid].m[1][1] = static_cast<float>(floatingIntensityYY[floatingIndex]);
            diffTensor[tid].m[0][2] = static_cast<float>(floatingIntensityXZ[floatingIndex]);
            diffTensor[tid].m[2][0] = diffTensor[tid].m[0][2];
            diffTensor[tid].m[1][2] = static_cast<float>(floatingIntensityYZ[floatingIndex]);
            diffTensor[tid].m[2][1] = diffTensor[tid].m[1][2];
            diffTensor[tid].m[2][2] = static_cast<float>(floatingIntensityZZ[floatingIndex]);

            // Compute the log of the diffusion tensor.
            reg_mat33_logm(&diffTensor[tid]);

            // Write this out as a new image
            floatingIntensityXX[floatingIndex] = static_cast<DataType>(diffTensor[tid].m[0][0]);
            floatingIntensityXY[floatingIndex] = static_cast<DataType>(diffTensor[tid].m[0][1]);
            floatingIntensityYY[floatingIndex] = static_cast<DataType>(diffTensor[tid].m[1][1]);
            floatingIntensityXZ[floatingIndex] = static_cast<DataType>(diffTensor[tid].m[0][2]);
            floatingIntensityYZ[floatingIndex] = static_cast<DataType>(diffTensor[tid].m[1][2]);
            floatingIntensityZZ[floatingIndex] = static_cast<DataType>(diffTensor[tid].m[2][2]);
        }
#ifdef _OPENMP
        omp_set_num_threads(max_thread_number);
#endif
        NR_DEBUG("Tensors have been logged");
    }
}
/* *************************************************************** */
template <class DataType>
void reg_dti_resampling_postprocessing(nifti_image *inputImage,
                                       const int *mask,
                                       const mat33 *jacMat,
                                       const int *dtIndicies,
                                       const nifti_image *warpedImage = nullptr) {
    // If we have some valid diffusion tensor indicies, we need to exponentiate the previously logged tensor components
    // we also need to reorient the tensors based on the local transformation Jacobians
    if (dtIndicies[0] != -1) {
#ifdef WIN32
        long warpedIndex;
        const long voxelNumber = (long)NiftiImage::calcVoxelNumber(inputImage, 3);
#else
        size_t warpedIndex;
        const size_t voxelNumber = NiftiImage::calcVoxelNumber(inputImage, 3);
#endif
        const DataType *warpVox, *warpedXX, *warpedXY, *warpedXZ, *warpedYY, *warpedYZ, *warpedZZ;
        if (warpedImage != nullptr) {
            warpVox = static_cast<DataType*>(warpedImage->data);
            // CAUTION: Here the tensor is assumed to be encoding in lower triangular order
            warpedXX = &warpVox[voxelNumber * dtIndicies[0]];
            warpedXY = &warpVox[voxelNumber * dtIndicies[1]];
            warpedYY = &warpVox[voxelNumber * dtIndicies[2]];
            warpedXZ = &warpVox[voxelNumber * dtIndicies[3]];
            warpedYZ = &warpVox[voxelNumber * dtIndicies[4]];
            warpedZZ = &warpVox[voxelNumber * dtIndicies[5]];
        }
        for (int u = 0; u < inputImage->nu; ++u) {
            // Now, we need to exponentiate the warped intensities back to give us a regular tensor
            // let's reorient each tensor based on the rigid component of the local warping
            /* As the tensor has 6 unique components that we need to worry about, read them out
         for the warped image. */
         // CAUTION: Here the tensor is assumed to be encoding in lower triangular order
            DataType *firstWarpVox = static_cast<DataType*>(inputImage->data);
            DataType *inputIntensityXX = &firstWarpVox[voxelNumber * (dtIndicies[0] + inputImage->nt * u)];
            DataType *inputIntensityXY = &firstWarpVox[voxelNumber * (dtIndicies[1] + inputImage->nt * u)];
            DataType *inputIntensityYY = &firstWarpVox[voxelNumber * (dtIndicies[2] + inputImage->nt * u)];
            DataType *inputIntensityXZ = &firstWarpVox[voxelNumber * (dtIndicies[3] + inputImage->nt * u)];
            DataType *inputIntensityYZ = &firstWarpVox[voxelNumber * (dtIndicies[4] + inputImage->nt * u)];
            DataType *inputIntensityZZ = &firstWarpVox[voxelNumber * (dtIndicies[5] + inputImage->nt * u)];

            // Step through each voxel in the warped image
            double testSum = 0;
            int col, row;
            int tid = 0;
#ifdef _OPENMP
            mat33 inputTensor[16], warpedTensor[16], RotMat[16], RotMatT[16];
            int max_thread_number = omp_get_max_threads();
            if (max_thread_number > 16) omp_set_num_threads(16);
#pragma omp parallel for default(none) \
    private(testSum, col, row, tid) \
    shared(voxelNumber,inputIntensityXX,inputIntensityYY,inputIntensityZZ, \
    warpedXX, warpedXY, warpedXZ, warpedYY, warpedYZ, warpedZZ, warpedImage, \
    inputIntensityXY,inputIntensityXZ,inputIntensityYZ, jacMat, mask, \
    inputTensor, warpedTensor,RotMat,RotMatT)
#else
            mat33 inputTensor[1], warpedTensor[1], RotMat[1], RotMatT[1];
#endif
            for (warpedIndex = 0; warpedIndex < voxelNumber; ++warpedIndex) {
#ifdef _OPENMP
                tid = omp_get_thread_num();
#endif
                if (mask[warpedIndex] > -1) {
                    // Fill the rest of the mat44 with the tensor components
                    inputTensor[tid].m[0][0] = static_cast<float>(inputIntensityXX[warpedIndex]);
                    inputTensor[tid].m[0][1] = static_cast<float>(inputIntensityXY[warpedIndex]);
                    inputTensor[tid].m[1][0] = inputTensor[tid].m[0][1];
                    inputTensor[tid].m[1][1] = static_cast<float>(inputIntensityYY[warpedIndex]);
                    inputTensor[tid].m[0][2] = static_cast<float>(inputIntensityXZ[warpedIndex]);
                    inputTensor[tid].m[2][0] = inputTensor[tid].m[0][2];
                    inputTensor[tid].m[1][2] = static_cast<float>(inputIntensityYZ[warpedIndex]);
                    inputTensor[tid].m[2][1] = inputTensor[tid].m[1][2];
                    inputTensor[tid].m[2][2] = static_cast<float>(inputIntensityZZ[warpedIndex]);
                    // Exponentiate the warped tensor
                    if (warpedImage == nullptr) {
                        reg_mat33_expm(&inputTensor[tid]);
                        testSum = 0;
                    } else {
                        reg_mat33_eye(&warpedTensor[tid]);
                        warpedTensor[tid].m[0][0] = static_cast<float>(warpedXX[warpedIndex]);
                        warpedTensor[tid].m[0][1] = static_cast<float>(warpedXY[warpedIndex]);
                        warpedTensor[tid].m[1][0] = warpedTensor[tid].m[0][1];
                        warpedTensor[tid].m[1][1] = static_cast<float>(warpedYY[warpedIndex]);
                        warpedTensor[tid].m[0][2] = static_cast<float>(warpedXZ[warpedIndex]);
                        warpedTensor[tid].m[2][0] = warpedTensor[tid].m[0][2];
                        warpedTensor[tid].m[1][2] = static_cast<float>(warpedYZ[warpedIndex]);
                        warpedTensor[tid].m[2][1] = warpedTensor[tid].m[1][2];
                        warpedTensor[tid].m[2][2] = static_cast<float>(warpedZZ[warpedIndex]);
                        inputTensor[tid] = nifti_mat33_mul(warpedTensor[tid], inputTensor[tid]);
                        testSum = static_cast<double>(warpedTensor[tid].m[0][0] + warpedTensor[tid].m[0][1] +
                                                      warpedTensor[tid].m[0][2] + warpedTensor[tid].m[1][0] + warpedTensor[tid].m[1][1] +
                                                      warpedTensor[tid].m[1][2] + warpedTensor[tid].m[2][0] + warpedTensor[tid].m[2][1] +
                                                      warpedTensor[tid].m[2][2]);
                    }

                    if (testSum == testSum) {
                        // Calculate the polar decomposition of the local Jacobian matrix, which
                        // tells us how to rotate the local tensor information
                        RotMat[tid] = nifti_mat33_polar(jacMat[warpedIndex]);
                        // We need both the rotation matrix, and it's transpose
                        for (col = 0; col < 3; col++)
                            for (row = 0; row < 3; row++)
                                RotMatT[tid].m[col][row] = RotMat[tid].m[row][col];
                        // As the mat44 multiplication uses pointers, do the multiplications separately
                        inputTensor[tid] = nifti_mat33_mul(nifti_mat33_mul(RotMatT[tid], inputTensor[tid]), RotMat[tid]);

                        // Finally, read the tensor back out as a warped image
                        inputIntensityXX[warpedIndex] = static_cast<DataType>(inputTensor[tid].m[0][0]);
                        inputIntensityYY[warpedIndex] = static_cast<DataType>(inputTensor[tid].m[1][1]);
                        inputIntensityZZ[warpedIndex] = static_cast<DataType>(inputTensor[tid].m[2][2]);
                        inputIntensityXY[warpedIndex] = static_cast<DataType>(inputTensor[tid].m[0][1]);
                        inputIntensityXZ[warpedIndex] = static_cast<DataType>(inputTensor[tid].m[0][2]);
                        inputIntensityYZ[warpedIndex] = static_cast<DataType>(inputTensor[tid].m[1][2]);
                    } else {
                        inputIntensityXX[warpedIndex] = std::numeric_limits<DataType>::quiet_NaN();
                        inputIntensityYY[warpedIndex] = std::numeric_limits<DataType>::quiet_NaN();
                        inputIntensityZZ[warpedIndex] = std::numeric_limits<DataType>::quiet_NaN();
                        inputIntensityXY[warpedIndex] = std::numeric_limits<DataType>::quiet_NaN();
                        inputIntensityXZ[warpedIndex] = std::numeric_limits<DataType>::quiet_NaN();
                        inputIntensityYZ[warpedIndex] = std::numeric_limits<DataType>::quiet_NaN();
                    }
                }
            }
#ifdef _OPENMP
            omp_set_num_threads(max_thread_number);
#endif
        }
        NR_DEBUG("Exponentiated and rotated all voxels");
    }
}
/* *************************************************************** */
template<class FloatingType, class FieldType>
void ResampleImage3D(const nifti_image *floatingImage,
                     const nifti_image *deformationField,
                     nifti_image *warpedImage,
                     const int *mask,
                     const FieldType paddingValue,
                     const int kernel) {
#ifdef _WIN32
    long  index;
    const long warpedVoxelNumber = (long)NiftiImage::calcVoxelNumber(warpedImage, 3);
    const long floatingVoxelNumber = (long)NiftiImage::calcVoxelNumber(floatingImage, 3);
#else
    size_t  index;
    const size_t warpedVoxelNumber = NiftiImage::calcVoxelNumber(warpedImage, 3);
    const size_t floatingVoxelNumber = NiftiImage::calcVoxelNumber(floatingImage, 3);
#endif
    const FloatingType *floatingIntensityPtr = static_cast<FloatingType*>(floatingImage->data);
    FloatingType *warpedIntensityPtr = static_cast<FloatingType*>(warpedImage->data);
    const FieldType *deformationFieldPtrX = static_cast<FieldType*>(deformationField->data);
    const FieldType *deformationFieldPtrY = &deformationFieldPtrX[warpedVoxelNumber];
    const FieldType *deformationFieldPtrZ = &deformationFieldPtrY[warpedVoxelNumber];

    const mat44 *floatingIJKMatrix;
    if (floatingImage->sform_code > 0)
        floatingIJKMatrix = &floatingImage->sto_ijk;
    else floatingIJKMatrix = &floatingImage->qto_ijk;

    // Define the kernel to use
    int kernel_size;
    int kernel_offset = 0;
    void (*kernelCompFctPtr)(double, double *);
    switch (kernel) {
    case 0:
        kernel_size = 2;
        kernelCompFctPtr = &interpNearestNeighKernel;
        kernel_offset = 0;
        break; // nearest-neighbour interpolation
    case 1:
        kernel_size = 2;
        kernelCompFctPtr = &interpLinearKernel;
        kernel_offset = 0;
        break; // linear interpolation
    case 4:
        kernel_size = SINC_KERNEL_SIZE;
        kernelCompFctPtr = &interpWindowedSincKernel;
        kernel_offset = SINC_KERNEL_RADIUS;
        break; // sinc interpolation
    default:
        kernel_size = 4;
        kernelCompFctPtr = &interpCubicSplineKernel;
        kernel_offset = 1;
        break; // cubic spline interpolation
    }

    // Iteration over the different volume along the 4th axis
    for (int t = 0; t < warpedImage->nt * warpedImage->nu; t++) {
        NR_DEBUG("3D resampling of volume number " << t);

        FloatingType *warpedIntensity = &warpedIntensityPtr[t * warpedVoxelNumber];
        const FloatingType *floatingIntensity = &floatingIntensityPtr[t * floatingVoxelNumber];

        int a, b, c, Y, Z, previous[3];
        const FloatingType *zPointer, *xyzPointer;
        double xBasis[SINC_KERNEL_SIZE], yBasis[SINC_KERNEL_SIZE], zBasis[SINC_KERNEL_SIZE], relative[3];
        double xTempNewValue, yTempNewValue, intensity;
        float world[3], position[3];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(intensity, world, position, previous, xBasis, yBasis, zBasis, relative, \
    a, b, c, Y, Z, zPointer, xyzPointer, xTempNewValue, yTempNewValue) \
    shared(floatingIntensity, warpedIntensity, warpedVoxelNumber, floatingVoxelNumber, \
    deformationFieldPtrX, deformationFieldPtrY, deformationFieldPtrZ, mask, \
    floatingIJKMatrix, floatingImage, paddingValue, kernel_size, kernel_offset, kernelCompFctPtr)
#endif // _OPENMP
        for (index = 0; index < warpedVoxelNumber; index++) {
            intensity = paddingValue;

            if (mask[index] > -1) {
                world[0] = static_cast<float>(deformationFieldPtrX[index]);
                world[1] = static_cast<float>(deformationFieldPtrY[index]);
                world[2] = static_cast<float>(deformationFieldPtrZ[index]);

                // real -> voxel; floating space
                reg_mat44_mul(floatingIJKMatrix, world, position);

                previous[0] = Floor(position[0]);
                previous[1] = Floor(position[1]);
                previous[2] = Floor(position[2]);

                relative[0] = static_cast<double>(position[0]) - static_cast<double>(previous[0]);
                relative[1] = static_cast<double>(position[1]) - static_cast<double>(previous[1]);
                relative[2] = static_cast<double>(position[2]) - static_cast<double>(previous[2]);

                (*kernelCompFctPtr)(relative[0], xBasis);
                (*kernelCompFctPtr)(relative[1], yBasis);
                (*kernelCompFctPtr)(relative[2], zBasis);
                previous[0] -= kernel_offset;
                previous[1] -= kernel_offset;
                previous[2] -= kernel_offset;

                intensity = 0;
                if (-1 < (previous[0]) && (previous[0] + kernel_size - 1) < floatingImage->nx &&
                    -1 < (previous[1]) && (previous[1] + kernel_size - 1) < floatingImage->ny &&
                    -1 < (previous[2]) && (previous[2] + kernel_size - 1) < floatingImage->nz) {
                    for (c = 0; c < kernel_size; c++) {
                        Z = previous[2] + c;
                        zPointer = &floatingIntensity[Z * floatingImage->nx * floatingImage->ny];
                        yTempNewValue = 0;
                        for (b = 0; b < kernel_size; b++) {
                            Y = previous[1] + b;
                            xyzPointer = &zPointer[Y * floatingImage->nx + previous[0]];
                            xTempNewValue = 0;
                            for (a = 0; a < kernel_size; a++) {
                                xTempNewValue += *xyzPointer++ * xBasis[a];
                            }
                            yTempNewValue += xTempNewValue * yBasis[b];
                        }
                        intensity += yTempNewValue * zBasis[c];
                    }
                } else {
                    for (c = 0; c < kernel_size; c++) {
                        Z = previous[2] + c;
                        zPointer = &floatingIntensity[Z * floatingImage->nx * floatingImage->ny];
                        yTempNewValue = 0;
                        for (b = 0; b < kernel_size; b++) {
                            Y = previous[1] + b;
                            xyzPointer = &zPointer[Y * floatingImage->nx + previous[0]];
                            xTempNewValue = 0;
                            for (a = 0; a < kernel_size; a++) {
                                if (-1 < (previous[0] + a) && (previous[0] + a) < floatingImage->nx &&
                                    -1 < Z && Z < floatingImage->nz &&
                                    -1 < Y && Y < floatingImage->ny) {
                                    xTempNewValue += *xyzPointer * xBasis[a];
                                } else {
                                    // paddingValue
                                    xTempNewValue += paddingValue * xBasis[a];
                                }
                                xyzPointer++;
                            }
                            yTempNewValue += xTempNewValue * yBasis[b];
                        }
                        intensity += yTempNewValue * zBasis[c];
                    }
                }
            }

            switch (floatingImage->datatype) {
            case NIFTI_TYPE_FLOAT32:
                warpedIntensity[index] = static_cast<FloatingType>(intensity);
                break;
            case NIFTI_TYPE_FLOAT64:
                warpedIntensity[index] = static_cast<FloatingType>(intensity);
                break;
            case NIFTI_TYPE_UINT8:
                if (intensity != intensity)
                    intensity = 0;
                intensity = (intensity <= 255 ? Round(intensity) : 255); // 255=2^8-1
                warpedIntensity[index] = static_cast<FloatingType>(intensity > 0 ? Round(intensity) : 0);
                break;
            case NIFTI_TYPE_UINT16:
                if (intensity != intensity)
                    intensity = 0;
                intensity = (intensity <= 65535 ? Round(intensity) : 65535); // 65535=2^16-1
                warpedIntensity[index] = static_cast<FloatingType>(intensity > 0 ? Round(intensity) : 0);
                break;
            case NIFTI_TYPE_UINT32:
                if (intensity != intensity)
                    intensity = 0;
                intensity = (intensity <= 4294967295 ? Round(intensity) : 4294967295); // 4294967295=2^32-1
                warpedIntensity[index] = static_cast<FloatingType>(intensity > 0 ? Round(intensity) : 0);
                break;
            default:
                if (intensity != intensity)
                    intensity = 0;
                warpedIntensity[index] = static_cast<FloatingType>(Round(intensity));
                break;
            }
        }
    }
}
/* *************************************************************** */
template<class FloatingType, class FieldType>
void ResampleImage2D(const nifti_image *floatingImage,
                     const nifti_image *deformationField,
                     nifti_image *warpedImage,
                     const int *mask,
                     const FieldType paddingValue,
                     const int kernel) {
#ifdef _WIN32
    long  index;
    const long warpedVoxelNumber = (long)NiftiImage::calcVoxelNumber(warpedImage, 2);
    const long floatingVoxelNumber = (long)NiftiImage::calcVoxelNumber(floatingImage, 2);
#else
    size_t  index;
    const size_t warpedVoxelNumber = NiftiImage::calcVoxelNumber(warpedImage, 2);
    const size_t floatingVoxelNumber = NiftiImage::calcVoxelNumber(floatingImage, 2);
#endif
    const FloatingType *floatingIntensityPtr = static_cast<FloatingType*>(floatingImage->data);
    FloatingType *warpedIntensityPtr = static_cast<FloatingType*>(warpedImage->data);
    const FieldType *deformationFieldPtrX = static_cast<FieldType*>(deformationField->data);
    const FieldType *deformationFieldPtrY = &deformationFieldPtrX[warpedVoxelNumber];

    const mat44 *floatingIJKMatrix;
    if (floatingImage->sform_code > 0)
        floatingIJKMatrix = &floatingImage->sto_ijk;
    else floatingIJKMatrix = &floatingImage->qto_ijk;

    int kernel_size;
    int kernel_offset = 0;
    void (*kernelCompFctPtr)(double, double *);
    switch (kernel) {
    case 0:
        kernel_size = 2;
        kernelCompFctPtr = &interpNearestNeighKernel;
        kernel_offset = 0;
        break; // nearest-neighbour interpolation
    case 1:
        kernel_size = 2;
        kernelCompFctPtr = &interpLinearKernel;
        kernel_offset = 0;
        break; // linear interpolation
    case 4:
        kernel_size = SINC_KERNEL_SIZE;
        kernelCompFctPtr = &interpWindowedSincKernel;
        kernel_offset = SINC_KERNEL_RADIUS;
        break; // sinc interpolation
    default:
        kernel_size = 4;
        kernelCompFctPtr = &interpCubicSplineKernel;
        kernel_offset = 1;
        break; // cubic spline interpolation
    }

    // Iteration over the different volume along the 4th axis
    for (int t = 0; t < warpedImage->nt * warpedImage->nu; t++) {
        NR_DEBUG("2D resampling of volume number " << t);

        FloatingType *warpedIntensity = &warpedIntensityPtr[t * warpedVoxelNumber];
        const FloatingType *floatingIntensity = &floatingIntensityPtr[t * floatingVoxelNumber];

        int a, b, Y, previous[2];
        const FloatingType *xyzPointer;
        double xBasis[SINC_KERNEL_SIZE], yBasis[SINC_KERNEL_SIZE], relative[2];
        double xTempNewValue, intensity;
        float world[3] = { 0, 0, 0 };
        float position[3] = { 0, 0, 0 };
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(intensity, world, position, previous, xBasis, yBasis, relative, \
    a, b, Y, xyzPointer, xTempNewValue) \
    shared(floatingIntensity, warpedIntensity, warpedVoxelNumber, floatingVoxelNumber, \
    deformationFieldPtrX, deformationFieldPtrY, mask, \
    floatingIJKMatrix, floatingImage, paddingValue, kernel_size, kernel_offset, kernelCompFctPtr)
#endif // _OPENMP
        for (index = 0; index < warpedVoxelNumber; index++) {
            intensity = paddingValue;

            if (mask[index] > -1) {
                world[0] = static_cast<float>(deformationFieldPtrX[index]);
                world[1] = static_cast<float>(deformationFieldPtrY[index]);
                world[2] = 0;

                // real -> voxel; floating space
                reg_mat44_mul(floatingIJKMatrix, world, position);

                previous[0] = Floor(position[0]);
                previous[1] = Floor(position[1]);

                relative[0] = static_cast<double>(position[0]) - static_cast<double>(previous[0]);
                relative[1] = static_cast<double>(position[1]) - static_cast<double>(previous[1]);

                (*kernelCompFctPtr)(relative[0], xBasis);
                (*kernelCompFctPtr)(relative[1], yBasis);
                previous[0] -= kernel_offset;
                previous[1] -= kernel_offset;

                intensity = 0;
                for (b = 0; b < kernel_size; b++) {
                    Y = previous[1] + b;
                    xyzPointer = &floatingIntensity[Y * floatingImage->nx + previous[0]];
                    xTempNewValue = 0;
                    for (a = 0; a < kernel_size; a++) {
                        if (-1 < (previous[0] + a) && (previous[0] + a) < floatingImage->nx &&
                            -1 < Y && Y < floatingImage->ny) {
                            xTempNewValue += *xyzPointer * xBasis[a];
                        } else {
                            // paddingValue
                            xTempNewValue += paddingValue * xBasis[a];
                        }
                        xyzPointer++;
                    }
                    intensity += xTempNewValue * yBasis[b];
                }

                switch (floatingImage->datatype) {
                case NIFTI_TYPE_FLOAT32:
                    warpedIntensity[index] = static_cast<FloatingType>(intensity);
                    break;
                case NIFTI_TYPE_FLOAT64:
                    warpedIntensity[index] = static_cast<FloatingType>(intensity);
                    break;
                case NIFTI_TYPE_UINT8:
                    intensity = (intensity <= 255 ? Round(intensity) : 255); // 255=2^8-1
                    warpedIntensity[index] = static_cast<FloatingType>(intensity > 0 ? Round(intensity) : 0);
                    break;
                case NIFTI_TYPE_UINT16:
                    intensity = (intensity <= 65535 ? Round(intensity) : 65535); // 65535=2^16-1
                    warpedIntensity[index] = static_cast<FloatingType>(intensity > 0 ? Round(intensity) : 0);
                    break;
                case NIFTI_TYPE_UINT32:
                    intensity = (intensity <= 4294967295 ? Round(intensity) : 4294967295); // 4294967295=2^32-1
                    warpedIntensity[index] = static_cast<FloatingType>(intensity > 0 ? Round(intensity) : 0);
                    break;
                default:
                    warpedIntensity[index] = static_cast<FloatingType>(Round(intensity));
                    break;
                }
            }
        }
    }
}
/* *************************************************************** */
/** This function resample a floating image into the referential
 * of a reference image by applying an affine transformation and
 * a deformation field. The affine transformation has to be in
 * real coordinate and the deformation field is in mm in the space
 * of the reference image.
 * interpolation can be either 0, 1 or 3 meaning nearest neighbor, linear
 * or cubic spline interpolation.
 * every voxel which is not fully in the floating image takes the
 * backgreg_round value. The dtIndicies are an array of size 6
 * that provides the position of the DT components (if there are any)
 * these values are set to -1 if there are not
 */
template <class FieldType, class FloatingType>
void reg_resampleImage(nifti_image *floatingImage,
                       nifti_image *warpedImage,
                       const nifti_image *deformationFieldImage,
                       const int *mask,
                       const int interpolation,
                       const FieldType paddingValue,
                       const int *dtIndicies,
                       const mat33 *jacMat) {
    // The floating image data is copied in case one deal with DTI
    void *originalFloatingData = nullptr;
    // The DTI are logged
    reg_dti_resampling_preprocessing<FloatingType>(floatingImage, &originalFloatingData, dtIndicies);

    // The deformation field contains the position in the real world
    if (deformationFieldImage->nu > 2) {
        ResampleImage3D<FloatingType, FieldType>(floatingImage,
                                                 deformationFieldImage,
                                                 warpedImage,
                                                 mask,
                                                 paddingValue,
                                                 interpolation);
    } else {
        ResampleImage2D<FloatingType, FieldType>(floatingImage,
                                                 deformationFieldImage,
                                                 warpedImage,
                                                 mask,
                                                 paddingValue,
                                                 interpolation);
    }
    // The temporary logged floating array is deleted and the original restored
    if (originalFloatingData != nullptr) {
        free(floatingImage->data);
        floatingImage->data = originalFloatingData;
        originalFloatingData = nullptr;
    }

    // The interpolated tensors are reoriented and exponentiated
    reg_dti_resampling_postprocessing<FloatingType>(warpedImage, mask, jacMat, dtIndicies);
}
/* *************************************************************** */
void reg_resampleImage(nifti_image *floatingImage,
                       nifti_image *warpedImage,
                       const nifti_image *deformationField,
                       const int *mask,
                       const int interpolation,
                       const float paddingValue,
                       const bool *dtiTimePoint,
                       const mat33 *jacMat) {
    if (floatingImage->datatype != warpedImage->datatype)
        NR_FATAL_ERROR("The floating and warped image should have the same data type");
    if (floatingImage->nt != warpedImage->nt)
        NR_FATAL_ERROR("The floating and warped images have different dimensions along the time axis");
    if (deformationField->datatype != NIFTI_TYPE_FLOAT32 && deformationField->datatype != NIFTI_TYPE_FLOAT64)
        NR_FATAL_ERROR("The deformation field image is expected to be of type float or double");

    // Define the DTI indices if required
    int dtIndicies[6];
    for (int i = 0; i < 6; ++i) dtIndicies[i] = -1;
    if (dtiTimePoint != nullptr) {
        if (jacMat == nullptr)
            NR_FATAL_ERROR("DTI resampling: No Jacobian matrix array has been provided");
        int j = 0;
        for (int i = 0; i < floatingImage->nt; ++i) {
            if (dtiTimePoint[i])
                dtIndicies[j++] = i;
        }
        if ((floatingImage->nz > 1 && j != 6) && (floatingImage->nz == 1 && j != 3))
            NR_FATAL_ERROR("DTI resampling: Unexpected number of DTI components");
    }

    // a mask array is created if no mask is specified
    bool MrPropreRules = false;
    if (mask == nullptr) {
        // voxels in the background are set to negative value so 0 corresponds to active voxel
        mask = (int*)calloc(NiftiImage::calcVoxelNumber(warpedImage, 3), sizeof(int));
        MrPropreRules = true;
    }

    std::visit([&](auto&& defFieldDataType, auto&& floImgDataType) {
        using DefFieldDataType = std::decay_t<decltype(defFieldDataType)>;
        using FloImgDataType = std::decay_t<decltype(floImgDataType)>;
        reg_resampleImage<DefFieldDataType, FloImgDataType>(floatingImage,
                                                            warpedImage,
                                                            deformationField,
                                                            mask,
                                                            interpolation,
                                                            paddingValue,
                                                            dtIndicies,
                                                            jacMat);
    }, NiftiImage::getFloatingDataType(deformationField), NiftiImage::getDataType(floatingImage));

    if (MrPropreRules)
        free(const_cast<int*>(mask));
}
/* *************************************************************** */
template<class FloatingType, class FieldType>
void ResampleImage3D_PSF_Sinc(const nifti_image *floatingImage,
                              const nifti_image *deformationField,
                              nifti_image *warpedImage,
                              const int *mask,
                              const FieldType paddingValue,
                              const int kernel) {
#ifdef _WIN32
    long index;
    const long warpedVoxelNumber = (long)NiftiImage::calcVoxelNumber(warpedImage, 3);
    const long warpedPlaneNumber = (long)NiftiImage::calcVoxelNumber(warpedImage, 2);
    const long warpedLineNumber = (long)warpedImage->nx;
    const long floatingVoxelNumber = (long)NiftiImage::calcVoxelNumber(floatingImage, 3);
#else
    size_t index;
    const size_t warpedVoxelNumber = NiftiImage::calcVoxelNumber(warpedImage, 3);
    const size_t warpedPlaneNumber = NiftiImage::calcVoxelNumber(warpedImage, 2);
    const size_t warpedLineNumber = (size_t)warpedImage->nx;
    const size_t floatingVoxelNumber = NiftiImage::calcVoxelNumber(floatingImage, 3);
#endif
    const FloatingType *floatingIntensityPtr = static_cast<FloatingType*>(floatingImage->data);
    FloatingType *warpedIntensityPtr = static_cast<FloatingType*>(warpedImage->data);
    const FieldType *deformationFieldPtrX = static_cast<FieldType*>(deformationField->data);
    const FieldType *deformationFieldPtrY = &deformationFieldPtrX[warpedVoxelNumber];
    const FieldType *deformationFieldPtrZ = &deformationFieldPtrY[warpedVoxelNumber];

    const mat44 *floatingIJKMatrix;
    if (floatingImage->sform_code > 0)
        floatingIJKMatrix = &floatingImage->sto_ijk;
    else floatingIJKMatrix = &floatingImage->qto_ijk;

    // Define the kernel to use
    int kernel_size;
    int kernel_offset = 0;
    void (*kernelCompFctPtr)(double, double *);
    switch (kernel) {
    case 0:
        NR_FATAL_ERROR("Not implemented for NN interpolation yet");
        kernel_size = 2;
        kernelCompFctPtr = &interpNearestNeighKernel;
        kernel_offset = 0;
        break; // nearest-neighbour interpolation
    case 1:
        kernel_size = 2;
        kernelCompFctPtr = &interpLinearKernel;
        kernel_offset = 0;
        break; // linear interpolation
    case 4:
        kernel_size = SINC_KERNEL_SIZE;
        kernelCompFctPtr = &interpWindowedSincKernel;
        kernel_offset = SINC_KERNEL_RADIUS;
        break; // sinc interpolation
    default:
        kernel_size = 4;
        kernelCompFctPtr = &interpCubicSplineKernel;
        kernel_offset = 1;
        break; // cubic spline interpolation
    }

    // Iteration over the different volume along the 4th axis
    for (size_t t = 0; t < (size_t)warpedImage->nt * warpedImage->nu; t++) {
        NR_DEBUG("3D resampling of volume number " << t);

        FloatingType *warpedIntensity = &warpedIntensityPtr[t * warpedVoxelNumber];
        const FloatingType *floatingIntensity = &floatingIntensityPtr[t * floatingVoxelNumber];

        double xBasis[SINC_KERNEL_SIZE], yBasis[SINC_KERNEL_SIZE], zBasis[SINC_KERNEL_SIZE], relative[3];
        double xBasisSamp[SINC_KERNEL_SIZE], yBasisSamp[SINC_KERNEL_SIZE], zBasisSamp[SINC_KERNEL_SIZE];
        int a, b, c, Y, Z, previous[3];

        interpWindowedSincKernel(0.00001, xBasisSamp);
        interpWindowedSincKernel(0.00001, yBasisSamp);
        interpWindowedSincKernel(0.00001, zBasisSamp);

        float psfWeightSum;
        const FloatingType *zPointer, *xyzPointer;
        double xTempNewValue, yTempNewValue, intensity, psfIntensity, psfWorld[3], position[3];
        float currentA, currentB, currentC, psfWeight;
        float shiftSamp[3];
        float currentAPre, currentARel, currentBPre, currentBRel, currentCPre, currentCRel, resamplingWeightSum, resamplingWeight;
        size_t currentIndex;

/*
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(intensity, psfWeightSum, psfWeight, \
    currentA, currentB, currentC, psfWorld, position,  shiftSamp,\
    psf_xyz, currentAPre, currentARel, currentBPre, currentBRel, currentCPre, currentCRel,\
    resamplingWeightSum, resamplingWeight, currentIndex, previous, relative,\
    xBasis, yBasis, zBasis, xBasisSamp, yBasisSamp, zBasisSamp, relativeSamp, Y, Z, psfIntensity, yTempNewValue, xTempNewValue,\
    xyzPointer, zPointer) \
    shared(warpedVoxelNumber, mask, paddingValue,\
    a, b, c , warpedPlaneNumber, warpedLineNumber, floatingIntensity,\
    deformationFieldPtrX, deformationFieldPtrY, deformationFieldPtrZ, floatingIJKMatrix,\
    floatingImage, warpedImage, kernelCompFctPtr, kernel_offset, kernel_size, warpedIntensity)
#endif // _OPENMP
*/
        for (index = 0; index < warpedVoxelNumber; index++) {
            intensity = paddingValue;

            if (mask[index] > -1) {
                //initialise weights
                psfWeightSum = 0.0f;
                intensity = 0.0f;
                currentC = static_cast<float>(index / warpedPlaneNumber);
                currentB = (index - currentC * warpedPlaneNumber) / warpedLineNumber;
                currentA = (index - currentB * warpedLineNumber - currentC * warpedPlaneNumber);

                // coordinates in eigen space
                float shiftall = SINC_KERNEL_RADIUS;
                float spacing = 1.0f;
                spacing = 0.3f;
                for (shiftSamp[0] = -shiftall; shiftSamp[0] <= shiftall; shiftSamp[0] += spacing) {
                    for (shiftSamp[1] = -shiftall; shiftSamp[1] <= shiftall; shiftSamp[1] += spacing) {
                        for (shiftSamp[2] = -shiftall; shiftSamp[2] <= shiftall; shiftSamp[2] += spacing) {
                            // Distance threshold (only interpolate if distance is below 3 std)

                            // Use the Eigen coordinates and convert them to XYZ
                            // The new lambda per coordinate is eige_coordinate*sqrt(eigenVal)
                            // as the sqrt(eigenVal) is equivalent to the STD

                            psfWeight = static_cast<float>(interpWindowedSincKernel_Samp(shiftSamp[0], shiftall) *
                                                           interpWindowedSincKernel_Samp(shiftSamp[1], shiftall) *
                                                           interpWindowedSincKernel_Samp(shiftSamp[2], shiftall));
                            //  NR_COUT<<shiftSamp[0]<<", "<<shiftSamp[1]<<", "<<shiftSamp[2]<<", "<<psfWeight<<std::endl;

                            // Interpolate (trilinearly) the deformation field for non-integer positions
                            float scalling = 1.0f;
                            currentAPre = (float)Floor(currentA + (shiftSamp[0] / warpedImage->pixdim[1]) * scalling);
                            currentARel = currentA + (shiftSamp[0] / warpedImage->pixdim[1] * scalling) - (float)(currentAPre);

                            currentBPre = (float)Floor(currentB + (shiftSamp[1] / warpedImage->pixdim[2]));
                            currentBRel = currentB + (shiftSamp[1] / warpedImage->pixdim[2] * scalling) - (float)(currentBPre);

                            currentCPre = (float)Floor(currentC + (shiftSamp[2] / warpedImage->pixdim[3] * scalling));
                            currentCRel = currentC + (shiftSamp[2] / warpedImage->pixdim[3] * scalling) - (float)(currentCPre);

                            // Interpolate the PSF world coordinates
                            psfWorld[0] = 0.0f;
                            psfWorld[1] = 0.0f;
                            psfWorld[2] = 0.0f;
                            if (psfWeight > 0) {
                                resamplingWeightSum = 0.0f;
                                for (a = 0; a <= 1; a++) {
                                    for (b = 0; b <= 1; b++) {
                                        for (c = 0; c <= 1; c++) {

                                            if ((currentAPre + a) >= 0
                                                && (currentBPre + b) >= 0
                                                && (currentCPre + c) >= 0
                                                && (currentAPre + a) < warpedImage->nx
                                                && (currentBPre + b) < warpedImage->ny
                                                && (currentCPre + c) < warpedImage->nz) {

                                                currentIndex = static_cast<size_t>((currentAPre + a) +
                                                                                   (currentBPre + b) * warpedLineNumber +
                                                                                   (currentCPre + c) * warpedPlaneNumber);

                                                resamplingWeight = fabs((float)(1 - a) - currentARel) *
                                                    fabs((float)(1 - b) - currentBRel) *
                                                    fabs((float)(1 - c) - currentCRel);

                                                resamplingWeightSum += resamplingWeight;

                                                psfWorld[0] += static_cast<double>(resamplingWeight * deformationFieldPtrX[currentIndex]);
                                                psfWorld[1] += static_cast<double>(resamplingWeight * deformationFieldPtrY[currentIndex]);
                                                psfWorld[2] += static_cast<double>(resamplingWeight * deformationFieldPtrZ[currentIndex]);
                                            }
                                        }
                                    }
                                }

                                if (resamplingWeightSum > 0) {
                                    psfWorld[0] /= resamplingWeightSum;
                                    psfWorld[1] /= resamplingWeightSum;
                                    psfWorld[2] /= resamplingWeightSum;

                                    // real -> voxel; floating space
                                    reg_mat44_mul(floatingIJKMatrix, psfWorld, position);

                                    previous[0] = Floor(position[0]);
                                    previous[1] = Floor(position[1]);
                                    previous[2] = Floor(position[2]);

                                    relative[0] = position[0] - static_cast<double>(previous[0]);
                                    relative[1] = position[1] - static_cast<double>(previous[1]);
                                    relative[2] = position[2] - static_cast<double>(previous[2]);

                                    (*kernelCompFctPtr)(relative[0], xBasis);
                                    (*kernelCompFctPtr)(relative[1], yBasis);
                                    (*kernelCompFctPtr)(relative[2], zBasis);
                                    previous[0] -= kernel_offset;
                                    previous[1] -= kernel_offset;
                                    previous[2] -= kernel_offset;

                                    psfIntensity = 0;
                                    for (c = 0; c < kernel_size; c++) {
                                        Z = previous[2] + c;
                                        zPointer = &floatingIntensity[Z * floatingImage->nx * floatingImage->ny];
                                        yTempNewValue = 0;
                                        for (b = 0; b < kernel_size; b++) {
                                            Y = previous[1] + b;
                                            xyzPointer = &zPointer[Y * floatingImage->nx + previous[0]];
                                            xTempNewValue = 0;
                                            for (a = 0; a < kernel_size; a++) {
                                                if (-1 < (previous[0] + a) && (previous[0] + a) < floatingImage->nx &&
                                                    -1 < Z && Z < floatingImage->nz &&
                                                    -1 < Y && Y < floatingImage->ny) {
                                                    xTempNewValue += *xyzPointer * xBasis[a];
                                                } else {
                                                    if (!(paddingValue != paddingValue))// paddingValue
                                                        xTempNewValue += paddingValue * xBasis[a];
                                                }
                                                xyzPointer++;
                                            }
                                            yTempNewValue += xTempNewValue * yBasis[b];
                                        }
                                        psfIntensity += yTempNewValue * zBasis[c];
                                    }
                                    if (!(psfIntensity != psfIntensity)) {
                                        intensity += psfWeight * psfIntensity;
                                        psfWeightSum += psfWeight;
                                    }
                                }
                            }
                        }
                    }
                }
                if (psfWeightSum > 0) {
                    intensity /= psfWeightSum;
                } else {
                    intensity = paddingValue;
                }
            } // if in mask
            switch (floatingImage->datatype) {
            case NIFTI_TYPE_FLOAT32:
                warpedIntensity[index] = static_cast<FloatingType>(intensity);
                break;
            case NIFTI_TYPE_FLOAT64:
                warpedIntensity[index] = static_cast<FloatingType>(intensity);
                break;
            case NIFTI_TYPE_UINT8:
                if (intensity != intensity)
                    intensity = 0;
                intensity = (intensity <= 255 ? Round(intensity) : 255); // 255=2^8-1
                warpedIntensity[index] = static_cast<FloatingType>(intensity > 0 ? Round(intensity) : 0);
                break;
            case NIFTI_TYPE_UINT16:
                if (intensity != intensity)
                    intensity = 0;
                intensity = (intensity <= 65535 ? Round(intensity) : 65535); // 65535=2^16-1
                warpedIntensity[index] = static_cast<FloatingType>(intensity > 0 ? Round(intensity) : 0);
                break;
            case NIFTI_TYPE_UINT32:
                if (intensity != intensity)
                    intensity = 0;
                intensity = (intensity <= 4294967295 ? Round(intensity) : 4294967295); // 4294967295=2^32-1
                warpedIntensity[index] = static_cast<FloatingType>(intensity > 0 ? Round(intensity) : 0);
                break;
            default:
                if (intensity != intensity)
                    intensity = 0;
                warpedIntensity[index] = static_cast<FloatingType>(Round(intensity));
                break;
            }
        }
    }
}
/* *************************************************************** */
template<class FloatingType, class FieldType>
void ResampleImage3D_PSF(const nifti_image *floatingImage,
                         const nifti_image *deformationField,
                         nifti_image *warpedImage,
                         const int *mask,
                         const FieldType paddingValue,
                         const int kernel,
                         const mat33 *jacMat,
                         const char algorithm) {
#ifdef _WIN32
    long index;
    const long warpedVoxelNumber = (long)NiftiImage::calcVoxelNumber(warpedImage, 3);
    const long warpedPlaneNumber = (long)NiftiImage::calcVoxelNumber(warpedImage, 2);
    const long warpedLineNumber = (long)warpedImage->nx;
    const long floatingVoxelNumber = (long)NiftiImage::calcVoxelNumber(floatingImage, 3);
#else
    size_t index;
    const size_t warpedVoxelNumber = NiftiImage::calcVoxelNumber(warpedImage, 3);
    const size_t warpedPlaneNumber = NiftiImage::calcVoxelNumber(warpedImage, 2);
    const size_t warpedLineNumber = (size_t)warpedImage->nx;
    const size_t floatingVoxelNumber = NiftiImage::calcVoxelNumber(floatingImage, 3);
#endif
    const FloatingType *floatingIntensityPtr = static_cast<FloatingType*>(floatingImage->data);
    FloatingType *warpedIntensityPtr = static_cast<FloatingType*>(warpedImage->data);
    const FieldType *deformationFieldPtrX = static_cast<FieldType*>(deformationField->data);
    const FieldType *deformationFieldPtrY = &deformationFieldPtrX[warpedVoxelNumber];
    const FieldType *deformationFieldPtrZ = &deformationFieldPtrY[warpedVoxelNumber];

    const mat44 *floatingIJKMatrix;
    if (floatingImage->sform_code > 0)
        floatingIJKMatrix = &floatingImage->sto_ijk;
    else floatingIJKMatrix = &floatingImage->qto_ijk;
    mat44 *warpedMatrix = &warpedImage->qto_xyz;
    if (warpedImage->sform_code > 0)
        warpedMatrix = &warpedImage->sto_xyz;
    const mat44 *floatingMatrix = &floatingImage->qto_xyz;
    if (floatingImage->sform_code > 0)
        floatingMatrix = &floatingImage->sto_xyz;

    float fwhmToStd = 2.355f;
    // T is the reference PSF and S is the floating PSF
    mat33 T, S;
    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 3; i++) {
            T.m[i][j] = 0;
            S.m[i][j] = 0;
        }
    }
    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 3; i++) {
            T.m[j][j] += Square(warpedMatrix->m[i][j]);
            S.m[j][j] += Square(floatingMatrix->m[i][j]);
        }
        T.m[j][j] = Square(sqrtf(T.m[j][j]) / fwhmToStd) / 2.0f;
        S.m[j][j] = Square(sqrtf(S.m[j][j]) / fwhmToStd) / 2.0f;
    }

    // Define the kernel to use
    int kernel_size;
    int kernel_offset = 0;
    void (*kernelCompFctPtr)(double, double *);
    switch (kernel) {
    case 0:
        NR_FATAL_ERROR("Not implemented for NN interpolation yet");
        kernel_size = 2;
        kernelCompFctPtr = &interpNearestNeighKernel;
        kernel_offset = 0;
        break; // nearest-neighbour interpolation
    case 1:
        kernel_size = 2;
        kernelCompFctPtr = &interpLinearKernel;
        kernel_offset = 0;
        break; // linear interpolation
    case 4:
        kernel_size = SINC_KERNEL_SIZE;
        kernelCompFctPtr = &interpWindowedSincKernel;
        kernel_offset = SINC_KERNEL_RADIUS;
        break; // sinc interpolation
    default:
        kernel_size = 4;
        kernelCompFctPtr = &interpCubicSplineKernel;
        kernel_offset = 1;
        break; // cubic spline interpolation
    }

    // Iteration over the different volume along the 4th axis
    for (size_t t = 0; t < (size_t)warpedImage->nt * warpedImage->nu; t++) {
        NR_DEBUG("PSF 3D resampling of volume number " << t);

        FloatingType *warpedIntensity = &warpedIntensityPtr[t * warpedVoxelNumber];
        const FloatingType *floatingIntensity = &floatingIntensityPtr[t * floatingVoxelNumber];

        double xBasis[SINC_KERNEL_SIZE], yBasis[SINC_KERNEL_SIZE], zBasis[SINC_KERNEL_SIZE], relative[3];
        int Y, Z, previous[3];

        float psf_xyz[3];

        mat33 P, invP, ASAt, A, TmS, TmS_EigVec, TmS_EigVec_trans, TmS_EigVal, TmS_EigVal_inv;
        float currentDeterminant, psfKernelShift[3], psfSampleSpacing, psfWeightSum, curLambda;
        float psfNumbSamples;

        const FloatingType *zPointer, *xyzPointer;
        double xTempNewValue, yTempNewValue, intensity, psfIntensity, psfWorld[3], position[3];
        size_t currentA, currentB, currentC, currentAPre, currentBPre, currentCPre;
        float  psf_eig[3], mahal, psfWeight;
        float currentARel, currentBRel, currentCRel, resamplingWeightSum, resamplingWeight;
        size_t currentIndex;

        for (index = 0; index < warpedVoxelNumber; index++) {
            intensity = paddingValue;

            if (mask[index] > -1) {
                if (algorithm == 0) {
                    // T=P+A*S*At
                    A = nifti_mat33_inverse(jacMat[index]);

                    ASAt = A * S * reg_mat33_trans(A);

                    TmS = T - ASAt;
                    //reg_mat33_disp(&TmS, "matTmS");

                    reg_mat33_diagonalize(&TmS, &TmS_EigVec, &TmS_EigVal);

                    // If eigen values are less than 0, set them to 0.
                    // Also, invert the eigenvalues to estimate the inverse.
                    for (int m = 0; m < 3; m++) {
                        for (int n = 0; n < 3; n++) {
                            if (m == n) { // Set diagonals to max(val,0)
                                TmS_EigVal.m[m][n] = TmS_EigVal.m[m][n] > 0.000001f ? TmS_EigVal.m[m][n] : 0.000001f;
                                TmS_EigVal_inv.m[m][n] = 1.0f / TmS_EigVal.m[m][n];
                            } else { // Set off-diagonal residuals to 0
                                TmS_EigVal.m[m][n] = 0;
                                TmS_EigVal_inv.m[m][n] = 0;
                            }
                        }
                    }

                    TmS_EigVec_trans = reg_mat33_trans(TmS_EigVec);
                    P = TmS_EigVec * TmS_EigVal * TmS_EigVec_trans;
                    invP = TmS_EigVec * TmS_EigVal_inv * TmS_EigVec_trans;
                    currentDeterminant = TmS_EigVal.m[0][0] * TmS_EigVal.m[1][1] * TmS_EigVal.m[2][2];
                    currentDeterminant = currentDeterminant < 0.000001f ? 0.000001f : currentDeterminant;
                } else {

                    A = nifti_mat33_inverse(jacMat[index]);

                    ASAt = A * S * reg_mat33_trans(A);

                    mat33 S_EigVec, S_EigVal;

                    //                % rotate S
                    //                [ZS, DS] = eig(S);
                    reg_mat33_diagonalize(&ASAt, &S_EigVec, &S_EigVal);

                    //                T1 = ZS'*T*ZS;
                    mat33 T1 = reg_mat33_trans(S_EigVec) * T * S_EigVec;

                    //                % Volume-preserving scale of S to make it isotropic
                    //                detS = prod(diag(DS));
                    float detASAt = S_EigVal.m[0][0] * S_EigVal.m[1][1] * S_EigVal.m[2][2];

                    //                factDetS = detS^(1/4);
                    float factDetS = powf(detASAt, 0.25);

                    //                LambdaN = factDetS*diag(diag(DS).^(-1/2));
                    //                invLambdaN = diag(1./diag(LambdaN))
                    mat33 LambdaN, invLambdaN;
                    for (int m = 0; m < 3; m++) {
                        for (int n = 0; n < 3; n++) {
                            if (m == n) {
                                LambdaN.m[m][n] = factDetS * powf(S_EigVal.m[m][n], -0.5);
                                invLambdaN.m[m][n] = 1.0f / LambdaN.m[m][n];
                            } else { // Set off-diagonal to 0
                                LambdaN.m[m][n] = 0;
                                invLambdaN.m[m][n] = 0;
                            }
                        }
                    }

                    //                T2 = LambdaN*T1*LambdaN';
                    mat33 T2 = LambdaN * T1 * reg_mat33_trans(LambdaN);

                    //                % Rotate to make thing axis-aligned
                    //                [ZT2, DT2] = eig(T2);
                    mat33 T2_EigVec, T2_EigVal;
                    reg_mat33_diagonalize(&T2, &T2_EigVec, &T2_EigVal);

                    //                % Optimal solution in the transformed axis-aligned space
                    //                DP2 = diag(max(sqrt(detS),diag(DT2)));
                    mat33 DP2;
                    for (int m = 0; m < 3; m++) {
                        for (int n = 0; n < 3; n++) {
                            if (m == n) {
                                DP2.m[m][n] = powf(factDetS, 0.5) > (T2_EigVal.m[m][n]) ? powf(factDetS, 0.5) : (T2_EigVal.m[m][n]);
                            } else { // Set off-diagonal to 0
                                DP2.m[m][n] = 0;
                            }
                        }
                    }

                    //                % Roll back the transforms
                    //                Q = ZS*invLambdaN*ZT2*DQ2*ZT2'*invLambdaN*ZS'
                    mat33 Q = S_EigVec * invLambdaN * T2_EigVec * DP2 * reg_mat33_trans(T2_EigVec) * invLambdaN * reg_mat33_trans(S_EigVec);
                    //                P=Q-S
                    TmS = Q - S;
                    invP = nifti_mat33_inverse(TmS);
                    reg_mat33_diagonalize(&TmS, &TmS_EigVec, &TmS_EigVal);

                    currentDeterminant = TmS_EigVal.m[0][0] * TmS_EigVal.m[1][1] * TmS_EigVal.m[2][2];
                    currentDeterminant = currentDeterminant < 0.000001f ? 0.000001f : currentDeterminant;
                }

                // set sampling rate
                psfNumbSamples = 3; // in standard deviations mm
                psfSampleSpacing = 0.75; // in standard deviations mm
                psfKernelShift[0] = TmS_EigVal.m[0][0] < 0.01f ? 0.0f : (float)(psfNumbSamples)*psfSampleSpacing;
                psfKernelShift[1] = TmS_EigVal.m[1][1] < 0.01f ? 0.0f : (float)(psfNumbSamples)*psfSampleSpacing;
                psfKernelShift[2] = TmS_EigVal.m[2][2] < 0.01f ? 0.0f : (float)(psfNumbSamples)*psfSampleSpacing;

                // Get image coordinates of the centre
                currentC = index / warpedPlaneNumber;
                currentB = (index - currentC * warpedPlaneNumber) / warpedLineNumber;
                currentA = (index - currentB * warpedLineNumber - currentC * warpedPlaneNumber);

                //initialise weights
                psfWeightSum = 0.0f;
                intensity = 0.0f;

                // coordinates in eigen space
                for (psf_eig[0] = -psfKernelShift[0]; psf_eig[0] <= (psfKernelShift[0]); psf_eig[0] += psfSampleSpacing) {
                    for (psf_eig[1] = -psfKernelShift[1]; psf_eig[1] <= (psfKernelShift[1]); psf_eig[1] += psfSampleSpacing) {
                        for (psf_eig[2] = -psfKernelShift[2]; psf_eig[2] <= (psfKernelShift[2]); psf_eig[2] += psfSampleSpacing) {
                            // Distance threshold (only interpolate if distance is below 3 std)
                            if (sqrtf(psf_eig[0] * psf_eig[0] + psf_eig[1] * psf_eig[1] + psf_eig[2] * psf_eig[2]) <= 3) {
                                // Use the Eigen coordinates and convert them to XYZ
                                // The new lambda per coordinate is eige_coordinate*sqrt(eigenVal)
                                // as the sqrt(eigenVal) is equivalent to the STD
                                psf_xyz[0] = 0;
                                psf_xyz[1] = 0;
                                psf_xyz[2] = 0;
                                for (int m = 0; m < 3; m++) {
                                    curLambda = (float)(psf_eig[m]) * sqrt(TmS_EigVal.m[m][m]);
                                    psf_xyz[0] += curLambda * TmS_EigVec.m[0][m];
                                    psf_xyz[1] += curLambda * TmS_EigVec.m[1][m];
                                    psf_xyz[2] += curLambda * TmS_EigVec.m[2][m];
                                }

                                //mahal=0;
                                mahal = psf_xyz[0] * invP.m[0][0] * psf_xyz[0] +
                                    psf_xyz[0] * invP.m[1][0] * psf_xyz[1] +
                                    psf_xyz[0] * invP.m[2][0] * psf_xyz[2] +
                                    psf_xyz[1] * invP.m[0][1] * psf_xyz[0] +
                                    psf_xyz[1] * invP.m[1][1] * psf_xyz[1] +
                                    psf_xyz[1] * invP.m[2][1] * psf_xyz[2] +
                                    psf_xyz[2] * invP.m[0][2] * psf_xyz[0] +
                                    psf_xyz[2] * invP.m[1][2] * psf_xyz[1] +
                                    psf_xyz[2] * invP.m[2][2] * psf_xyz[2];

                                psfWeight = powf(2.f * (float)M_PI, -3.f / 2.f) * powf(currentDeterminant, -0.5f) * expf(-0.5f * mahal);

                                if (psfWeight != 0.f) { // If the relative weight is above 0
                                    // Interpolate (trilinearly) the deformation field for non-integer positions
                                    currentAPre = (size_t)(currentA + (size_t)Floor(psf_xyz[0] / (float)warpedImage->pixdim[1]));
                                    currentARel = (float)currentA + (float)(psf_xyz[0] / (float)warpedImage->pixdim[1]) - (float)(currentAPre);

                                    currentBPre = (size_t)(currentB + (size_t)Floor(psf_xyz[1] / (float)warpedImage->pixdim[2]));
                                    currentBRel = (float)currentB + (float)(psf_xyz[1] / (float)warpedImage->pixdim[2]) - (float)(currentBPre);

                                    currentCPre = (size_t)(currentC + (size_t)Floor(psf_xyz[2] / (float)warpedImage->pixdim[3]));
                                    currentCRel = (float)currentC + (float)(psf_xyz[2] / (float)warpedImage->pixdim[3]) - (float)(currentCPre);

                                    // Interpolate the PSF world coordinates
                                    psfWorld[0] = 0.0f;
                                    psfWorld[1] = 0.0f;
                                    psfWorld[2] = 0.0f;
                                    resamplingWeightSum = 0.0f;
                                    for (int a = 0; a <= 1; a++) {
                                        for (int b = 0; b <= 1; b++) {
                                            for (int c = 0; c <= 1; c++) {

                                                if (((int)currentAPre + a) >= 0
                                                    && ((int)currentBPre + b) >= 0
                                                    && ((int)currentCPre + c) >= 0
                                                    && ((int)currentAPre + a) < warpedImage->nx
                                                    && ((int)currentBPre + b) < warpedImage->ny
                                                    && ((int)currentCPre + c) < warpedImage->nz) {

                                                    currentIndex = ((size_t)currentAPre + (size_t)a) +
                                                        ((size_t)currentBPre + (size_t)b) * warpedLineNumber +
                                                        ((size_t)currentCPre + (size_t)c) * warpedPlaneNumber;

                                                    resamplingWeight = fabs((float)(1 - a) - currentARel) *
                                                        fabs((float)(1 - b) - currentBRel) *
                                                        fabs((float)(1 - c) - currentCRel);

                                                    resamplingWeightSum += resamplingWeight;

                                                    psfWorld[0] += static_cast<double>(resamplingWeight * deformationFieldPtrX[currentIndex]);
                                                    psfWorld[1] += static_cast<double>(resamplingWeight * deformationFieldPtrY[currentIndex]);
                                                    psfWorld[2] += static_cast<double>(resamplingWeight * deformationFieldPtrZ[currentIndex]);
                                                }
                                            }
                                        }
                                    }

                                    if (resamplingWeightSum > 0.0f) {
                                        psfWorld[0] /= resamplingWeightSum;
                                        psfWorld[1] /= resamplingWeightSum;
                                        psfWorld[2] /= resamplingWeightSum;

                                        // real -> voxel; floating space
                                        reg_mat44_mul(floatingIJKMatrix, psfWorld, position);

                                        previous[0] = Floor(position[0]);
                                        previous[1] = Floor(position[1]);
                                        previous[2] = Floor(position[2]);

                                        relative[0] = position[0] - static_cast<double>(previous[0]);
                                        relative[1] = position[1] - static_cast<double>(previous[1]);
                                        relative[2] = position[2] - static_cast<double>(previous[2]);

                                        (*kernelCompFctPtr)(relative[0], xBasis);
                                        (*kernelCompFctPtr)(relative[1], yBasis);
                                        (*kernelCompFctPtr)(relative[2], zBasis);
                                        previous[0] -= kernel_offset;
                                        previous[1] -= kernel_offset;
                                        previous[2] -= kernel_offset;

                                        psfIntensity = 0;
                                        for (int c = 0; c < kernel_size; c++) {
                                            Z = previous[2] + c;
                                            zPointer = &floatingIntensity[Z * floatingImage->nx * floatingImage->ny];
                                            yTempNewValue = 0;
                                            for (int b = 0; b < kernel_size; b++) {
                                                Y = previous[1] + b;
                                                xyzPointer = &zPointer[Y * floatingImage->nx + previous[0]];
                                                xTempNewValue = 0;
                                                for (int a = 0; a < kernel_size; a++) {
                                                    if (-1 < (previous[0] + a) && (previous[0] + a) < floatingImage->nx &&
                                                        -1 < Z && Z < floatingImage->nz &&
                                                        -1 < Y && Y < floatingImage->ny) {
                                                        xTempNewValue += *xyzPointer * xBasis[a];
                                                    } else {
                                                        // paddingValue
                                                        if (!(paddingValue != paddingValue))// paddingValue
                                                            xTempNewValue += paddingValue * xBasis[a];
                                                    }
                                                    xyzPointer++;
                                                }
                                                yTempNewValue += xTempNewValue * yBasis[b];
                                            }
                                            psfIntensity += yTempNewValue * zBasis[c];
                                        }
                                        if (!(psfIntensity != psfIntensity)) {
                                            intensity += psfWeight * psfIntensity;
                                            psfWeightSum += psfWeight;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if (psfWeightSum > 0) {
                    intensity /= psfWeightSum;
                } else {
                    intensity = paddingValue;
                }
            } // if in mask
            switch (floatingImage->datatype) {
            case NIFTI_TYPE_FLOAT32:
                warpedIntensity[index] = static_cast<FloatingType>(intensity);
                break;
            case NIFTI_TYPE_FLOAT64:
                warpedIntensity[index] = static_cast<FloatingType>(intensity);
                break;
            case NIFTI_TYPE_UINT8:
                if (intensity != intensity)
                    intensity = 0;
                intensity = (intensity <= 255 ? Round(intensity) : 255); // 255=2^8-1
                warpedIntensity[index] = static_cast<FloatingType>(intensity > 0 ? Round(intensity) : 0);
                break;
            case NIFTI_TYPE_UINT16:
                if (intensity != intensity)
                    intensity = 0;
                intensity = (intensity <= 65535 ? Round(intensity) : 65535); // 65535=2^16-1
                warpedIntensity[index] = static_cast<FloatingType>(intensity > 0 ? Round(intensity) : 0);
                break;
            case NIFTI_TYPE_UINT32:
                if (intensity != intensity)
                    intensity = 0;
                intensity = (intensity <= 4294967295 ? Round(intensity) : 4294967295); // 4294967295=2^32-1
                warpedIntensity[index] = static_cast<FloatingType>(intensity > 0 ? Round(intensity) : 0);
                break;
            case NIFTI_TYPE_INT16:
                if (intensity != intensity)
                    intensity = 0;
                intensity = (intensity <= 32767 ? Round(intensity) : 32767); // 32767=2^15-1
                warpedIntensity[index] = static_cast<FloatingType>(intensity);
                break;
            case NIFTI_TYPE_INT32:
                if (intensity != intensity)
                    intensity = 0;
                intensity = (intensity <= 2147483647 ? Round(intensity) : 2147483647); // 2147483647=2^31-1
                warpedIntensity[index] = static_cast<FloatingType>(intensity);
                break;
            default:
                if (intensity != intensity)
                    intensity = 0;
                warpedIntensity[index] = static_cast<FloatingType>(Round(intensity));
                break;
            }
        }
    }
}
/* *************************************************************** */
template <class FieldType, class FloatingType>
void reg_resampleImage_PSF(const nifti_image *floatingImage,
                            nifti_image *warpedImage,
                            const nifti_image *deformationFieldImage,
                            const int *mask,
                            const int interpolation,
                            const FieldType paddingValue,
                            const mat33 *jacMat,
                            const char algorithm) {
    // The deformation field contains the position in the real world
    if (deformationFieldImage->nu > 2) {
        if (algorithm == 2) {
            NR_DEBUG("Running ResampleImage3D_PSF_Sinc 1");
            ResampleImage3D_PSF_Sinc<FloatingType, FieldType>(floatingImage,
                                                              deformationFieldImage,
                                                              warpedImage,
                                                              mask,
                                                              paddingValue,
                                                              interpolation);
        } else {
            NR_DEBUG("Running ResampleImage3D_PSF");
            ResampleImage3D_PSF<FloatingType, FieldType>(floatingImage,
                                                         deformationFieldImage,
                                                         warpedImage,
                                                         mask,
                                                         paddingValue,
                                                         interpolation,
                                                         jacMat,
                                                         algorithm);
        }
    } else {
        NR_FATAL_ERROR("Not implemented for 2D images yet");
    }
}
/* *************************************************************** */
void reg_resampleImage_PSF(const nifti_image *floatingImage,
                           nifti_image *warpedImage,
                           const nifti_image *deformationField,
                           const int *mask,
                           const int interpolation,
                           const float paddingValue,
                           const mat33 *jacMat,
                           const char algorithm) {
    if (floatingImage->datatype != warpedImage->datatype)
        NR_FATAL_ERROR("The floating and warped image should have the same data type");
    if (floatingImage->nt != warpedImage->nt)
        NR_FATAL_ERROR("The floating and warped images have different dimension along the time axis");
    if (deformationField->datatype != NIFTI_TYPE_FLOAT32 && deformationField->datatype != NIFTI_TYPE_FLOAT64)
        NR_FATAL_ERROR("The deformation field image is expected to be of type float or double");

    // a mask array is created if no mask is specified
    bool MrPropreRules = false;
    if (mask == nullptr) {
        // voxels in the background are set to negative value so 0 corresponds to active voxel
        mask = (int*)calloc(NiftiImage::calcVoxelNumber(warpedImage, 3), sizeof(int));
        MrPropreRules = true;
    }

    std::visit([&](auto&& defFieldDataType, auto&& floImgDataType) {
        using DefFieldDataType = std::decay_t<decltype(defFieldDataType)>;
        using FloImgDataType = std::decay_t<decltype(floImgDataType)>;
        reg_resampleImage_PSF<DefFieldDataType, FloImgDataType>(floatingImage,
                                                                warpedImage,
                                                                deformationField,
                                                                mask,
                                                                interpolation,
                                                                paddingValue,
                                                                jacMat,
                                                                algorithm);
    }, NiftiImage::getFloatingDataType(deformationField), NiftiImage::getDataType(floatingImage));

    if (MrPropreRules)
        free(const_cast<int*>(mask));
}
/* *************************************************************** */
template <class DataType>
void reg_bilinearResampleGradient(const nifti_image *floatingImage,
                                  nifti_image *warpedImage,
                                  const nifti_image *deformationField,
                                  const float paddingValue) {
    const size_t floatingVoxelNumber = NiftiImage::calcVoxelNumber(floatingImage, 3);
    const size_t warpedVoxelNumber = NiftiImage::calcVoxelNumber(warpedImage, 3);
    const DataType *floatingIntensityX = static_cast<DataType*>(floatingImage->data);
    const DataType *floatingIntensityY = &floatingIntensityX[floatingVoxelNumber];
    DataType *warpedIntensityX = static_cast<DataType*>(warpedImage->data);
    DataType *warpedIntensityY = &warpedIntensityX[warpedVoxelNumber];
    const DataType *deformationFieldPtrX = static_cast<DataType*>(deformationField->data);
    const DataType *deformationFieldPtrY = &deformationFieldPtrX[NiftiImage::calcVoxelNumber(deformationField, 3)];

    // Extract the relevant affine matrix
    const mat44 *floating_mm_to_voxel = &floatingImage->qto_ijk;
    if (floatingImage->sform_code != 0)
        floating_mm_to_voxel = &floatingImage->sto_ijk;

    // The spacing is computed in case the sform if defined
    float realSpacing[2];
    if (warpedImage->sform_code > 0) {
        reg_getRealImageSpacing(warpedImage, realSpacing);
    } else {
        realSpacing[0] = warpedImage->dx;
        realSpacing[1] = warpedImage->dy;
    }

    // Reorientation matrix is assessed in order to remove the rigid component
    mat33 reorient = nifti_mat33_inverse(nifti_mat33_polar(reg_mat44_to_mat33(&deformationField->sto_xyz)));

    // Some useful variables
    mat33 jacMat;
    DataType defX, defY;
    DataType basisX[2], basisY[2], deriv[2], basis[2];
    DataType xFloCoord, yFloCoord;
    int anteIntX[2], anteIntY[2];
    int x, y, a, b, defIndex, floIndex, warpedIndex;
    DataType val_x, val_y, weight[2];

    // Loop over all voxel
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(x,a,b,val_x,val_y,defIndex,floIndex,warpedIndex, \
    anteIntX,anteIntY,xFloCoord,yFloCoord, \
    basisX,basisY,deriv,basis,defX,defY,jacMat,weight) \
    shared(warpedImage,warpedIntensityX,warpedIntensityY, \
    deformationField,deformationFieldPtrX,deformationFieldPtrY, \
    floatingImage,floatingIntensityX,floatingIntensityY,floating_mm_to_voxel, \
    paddingValue, reorient,realSpacing)
#endif // _OPENMP
    for (y = 0; y < warpedImage->ny; ++y) {
        warpedIndex = y * warpedImage->nx;
        deriv[0] = -1;
        deriv[1] = 1;
        basis[0] = 1;
        basis[1] = 0;
        for (x = 0; x < warpedImage->nx; ++x) {
            warpedIntensityX[warpedIndex] = paddingValue;
            warpedIntensityY[warpedIndex] = paddingValue;

            // Compute the index in the floating image
            defX = deformationFieldPtrX[warpedIndex];
            defY = deformationFieldPtrY[warpedIndex];
            xFloCoord =
                floating_mm_to_voxel->m[0][0] * defX +
                floating_mm_to_voxel->m[0][1] * defY +
                floating_mm_to_voxel->m[0][3];
            yFloCoord =
                floating_mm_to_voxel->m[1][0] * defX +
                floating_mm_to_voxel->m[1][1] * defY +
                floating_mm_to_voxel->m[1][3];

            // Extract the floating value using bilinear interpolation
            anteIntX[0] = Floor(xFloCoord);
            anteIntX[1] = Ceil(xFloCoord);
            anteIntY[0] = Floor(yFloCoord);
            anteIntY[1] = Ceil(yFloCoord);
            val_x = 0;
            val_y = 0;
            basisX[1] = fabs(xFloCoord - (DataType)anteIntX[0]);
            basisY[1] = fabs(yFloCoord - (DataType)anteIntY[0]);
            basisX[0] = 1 - basisX[1];
            basisY[0] = 1 - basisY[1];
            for (b = 0; b < 2; ++b) {
                if (anteIntY[b] > -1 && anteIntY[b] < floatingImage->ny) {
                    for (a = 0; a < 2; ++a) {
                        weight[0] = basisX[a] * basisY[b];
                        if (anteIntX[a] > -1 && anteIntX[a] < floatingImage->nx) {
                            floIndex = anteIntY[b] * floatingImage->nx + anteIntX[a];
                            val_x += floatingIntensityX[floIndex] * weight[0];
                            val_y += floatingIntensityY[floIndex] * weight[0];
                        } // anteIntX not in the floating image space
                        else {
                            val_x += paddingValue * weight[0];
                            val_y += paddingValue * weight[0];
                        }
                    } // a
                } // anteIntY not in the floating image space
                else {
                    val_x += paddingValue * basisY[b];
                    val_y += paddingValue * basisY[b];
                }
            } // b

            // Compute the Jacobian matrix
            memset(&jacMat, 0, sizeof(mat33));
            jacMat.m[2][2] = 1.;
            for (b = 0; b < 2; ++b) {
                anteIntY[0] = y + b;
                basisY[0] = basis[b];
                basisY[1] = deriv[b];
                // Boundary conditions along y - slidding
                if (y == deformationField->ny - 1) {
                    if (b == 1)
                        anteIntY[0] -= 2;
                    basisY[0] = fabs(basisY[0] - 1);
                    basisY[1] *= -1;
                }
                for (a = 0; a < 2; ++a) {
                    anteIntX[0] = x + a;
                    basisX[0] = basis[a];
                    basisX[1] = deriv[a];
                    // Boundary conditions along x - slidding
                    if (x == deformationField->nx - 1) {
                        if (a == 1)
                            anteIntX[0] -= 2;
                        basisX[0] = fabs(basisX[0] - 1);
                        basisX[1] *= -1;
                    }

                    // Compute the basis function values
                    weight[0] = basisX[1] * basisY[0];
                    weight[1] = basisX[0] * basisY[1];

                    // Get the deformation field index
                    defIndex = anteIntY[0] * deformationField->nx + anteIntX[0];

                    // Get the deformation field values
                    defX = deformationFieldPtrX[defIndex];
                    defY = deformationFieldPtrY[defIndex];

                    // Symmetric difference to compute the derivatives
                    jacMat.m[0][0] += static_cast<float>(weight[0] * defX);
                    jacMat.m[0][1] += static_cast<float>(weight[1] * defX);
                    jacMat.m[1][0] += static_cast<float>(weight[0] * defY);
                    jacMat.m[1][1] += static_cast<float>(weight[1] * defY);
                }
            }
            // reorient and scale the Jacobian matrix
            jacMat = nifti_mat33_mul(reorient, jacMat);
            jacMat.m[0][0] /= realSpacing[0];
            jacMat.m[0][1] /= realSpacing[1];
            jacMat.m[1][0] /= realSpacing[0];
            jacMat.m[1][1] /= realSpacing[1];

            // Modulate the gradient scalar values
            warpedIntensityX[warpedIndex] = jacMat.m[0][0] * val_x + jacMat.m[0][1] * val_y;
            warpedIntensityY[warpedIndex] = jacMat.m[1][0] * val_x + jacMat.m[1][1] * val_y;

            ++warpedIndex;
        } // x
    } // y
}
/* *************************************************************** */
template <class DataType>
void reg_trilinearResampleGradient(const nifti_image *floatingImage,
                                   nifti_image *warpedImage,
                                   const nifti_image *deformationField,
                                   const float paddingValue) {
    const size_t floatingVoxelNumber = NiftiImage::calcVoxelNumber(floatingImage, 3);
    const size_t warpedVoxelNumber = NiftiImage::calcVoxelNumber(warpedImage, 3);
    const size_t deformationFieldVoxelNumber = NiftiImage::calcVoxelNumber(deformationField, 3);
    const DataType *floatingIntensityX = static_cast<DataType*>(floatingImage->data);
    const DataType *floatingIntensityY = &floatingIntensityX[floatingVoxelNumber];
    const DataType *floatingIntensityZ = &floatingIntensityY[floatingVoxelNumber];
    DataType *warpedIntensityX = static_cast<DataType*>(warpedImage->data);
    DataType *warpedIntensityY = &warpedIntensityX[warpedVoxelNumber];
    DataType *warpedIntensityZ = &warpedIntensityY[warpedVoxelNumber];
    const DataType *deformationFieldPtrX = static_cast<DataType*>(deformationField->data);
    const DataType *deformationFieldPtrY = &deformationFieldPtrX[deformationFieldVoxelNumber];
    const DataType *deformationFieldPtrZ = &deformationFieldPtrY[deformationFieldVoxelNumber];

    // Extract the relevant affine matrix
    const mat44 *floating_mm_to_voxel = &floatingImage->qto_ijk;
    if (floatingImage->sform_code != 0)
        floating_mm_to_voxel = &floatingImage->sto_ijk;

    // The spacing is computed if the sform is defined
    float realSpacing[3];
    if (warpedImage->sform_code > 0) {
        reg_getRealImageSpacing(warpedImage, realSpacing);
    } else {
        realSpacing[0] = warpedImage->dx;
        realSpacing[1] = warpedImage->dy;
        realSpacing[2] = warpedImage->dz;
    }

    // Reorientation matrix is assessed in order to remove the rigid component
    mat33 reorient = nifti_mat33_inverse(nifti_mat33_polar(reg_mat44_to_mat33(&deformationField->sto_xyz)));

    // Some useful variables
    mat33 jacMat;
    DataType defX, defY, defZ;
    DataType basisX[2], basisY[2], basisZ[2], deriv[2], basis[2];
    DataType xFloCoord, yFloCoord, zFloCoord;
    int anteIntX[2], anteIntY[2], anteIntZ[2];
    int x, y, z, a, b, c, defIndex, floIndex, warpedIndex;
    DataType val_x, val_y, val_z, weight[3];

    // Loop over all voxel
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(x,y,a,b,c,val_x,val_y,val_z,defIndex,floIndex,warpedIndex, \
    anteIntX,anteIntY,anteIntZ,xFloCoord,yFloCoord,zFloCoord, \
    basisX,basisY,basisZ,deriv,basis,defX,defY,defZ,jacMat,weight) \
    shared(warpedImage,warpedIntensityX,warpedIntensityY,warpedIntensityZ, \
    deformationField,deformationFieldPtrX,deformationFieldPtrY,deformationFieldPtrZ, \
    floatingImage,floatingIntensityX,floatingIntensityY,floatingIntensityZ,floating_mm_to_voxel, \
    paddingValue, reorient, realSpacing)
#endif // _OPENMP
    for (z = 0; z < warpedImage->nz; ++z) {
        warpedIndex = z * warpedImage->nx * warpedImage->ny;
        deriv[0] = -1;
        deriv[1] = 1;
        basis[0] = 1;
        basis[1] = 0;
        for (y = 0; y < warpedImage->ny; ++y) {
            for (x = 0; x < warpedImage->nx; ++x) {
                warpedIntensityX[warpedIndex] = paddingValue;
                warpedIntensityY[warpedIndex] = paddingValue;
                warpedIntensityZ[warpedIndex] = paddingValue;

                // Compute the index in the floating image
                defX = deformationFieldPtrX[warpedIndex];
                defY = deformationFieldPtrY[warpedIndex];
                defZ = deformationFieldPtrZ[warpedIndex];
                xFloCoord =
                    floating_mm_to_voxel->m[0][0] * defX +
                    floating_mm_to_voxel->m[0][1] * defY +
                    floating_mm_to_voxel->m[0][2] * defZ +
                    floating_mm_to_voxel->m[0][3];
                yFloCoord =
                    floating_mm_to_voxel->m[1][0] * defX +
                    floating_mm_to_voxel->m[1][1] * defY +
                    floating_mm_to_voxel->m[1][2] * defZ +
                    floating_mm_to_voxel->m[1][3];
                zFloCoord =
                    floating_mm_to_voxel->m[2][0] * defX +
                    floating_mm_to_voxel->m[2][1] * defY +
                    floating_mm_to_voxel->m[2][2] * defZ +
                    floating_mm_to_voxel->m[2][3];

                // Extract the floating value using bilinear interpolation
                anteIntX[0] = Floor(xFloCoord);
                anteIntX[1] = Ceil(xFloCoord);
                anteIntY[0] = Floor(yFloCoord);
                anteIntY[1] = Ceil(yFloCoord);
                anteIntZ[0] = Floor(zFloCoord);
                anteIntZ[1] = Ceil(zFloCoord);
                val_x = 0;
                val_y = 0;
                val_z = 0;
                basisX[1] = fabs(xFloCoord - (DataType)anteIntX[0]);
                basisY[1] = fabs(yFloCoord - (DataType)anteIntY[0]);
                basisZ[1] = fabs(zFloCoord - (DataType)anteIntZ[0]);
                basisX[0] = 1 - basisX[1];
                basisY[0] = 1 - basisY[1];
                basisZ[0] = 1 - basisZ[1];
                for (c = 0; c < 2; ++c) {
                    if (anteIntZ[c] > -1 && anteIntZ[c] < floatingImage->nz) {
                        for (b = 0; b < 2; ++b) {
                            if (anteIntY[b] > -1 && anteIntY[b] < floatingImage->ny) {
                                for (a = 0; a < 2; ++a) {
                                    weight[0] = basisX[a] * basisY[b] * basisZ[c];
                                    if (anteIntX[a] > -1 && anteIntX[a] < floatingImage->nx) {
                                        floIndex = (anteIntZ[c] * floatingImage->ny + anteIntY[b]) * floatingImage->nx + anteIntX[a];
                                        val_x += floatingIntensityX[floIndex] * weight[0];
                                        val_y += floatingIntensityY[floIndex] * weight[0];
                                        val_z += floatingIntensityZ[floIndex] * weight[0];
                                    } // anteIntX not in the floating image space
                                    else {
                                        val_x += paddingValue * weight[0];
                                        val_y += paddingValue * weight[0];
                                        val_z += paddingValue * weight[0];
                                    }
                                } // a
                            } // anteIntY not in the floating image space
                            else {
                                val_x += paddingValue * basisY[b] * basisZ[c];
                                val_y += paddingValue * basisY[b] * basisZ[c];
                                val_z += paddingValue * basisY[b] * basisZ[c];
                            }
                        } // b
                    } // anteIntZ not in the floating image space
                    else {
                        val_x += paddingValue * basisZ[c];
                        val_y += paddingValue * basisZ[c];
                        val_z += paddingValue * basisZ[c];
                    }
                } // c

                // Compute the Jacobian matrix
                memset(&jacMat, 0, sizeof(mat33));
                for (c = 0; c < 2; ++c) {
                    anteIntZ[0] = z + c;
                    basisZ[0] = basis[c];
                    basisZ[1] = deriv[c];
                    // Boundary conditions along z - slidding
                    if (z == deformationField->nz - 1) {
                        if (c == 1)
                            anteIntZ[0] -= 2;
                        basisZ[0] = fabs(basisZ[0] - 1);
                        basisZ[1] *= -1;
                    }
                    for (b = 0; b < 2; ++b) {
                        anteIntY[0] = y + b;
                        basisY[0] = basis[b];
                        basisY[1] = deriv[b];
                        // Boundary conditions along y - slidding
                        if (y == deformationField->ny - 1) {
                            if (b == 1)
                                anteIntY[0] -= 2;
                            basisY[0] = fabs(basisY[0] - 1);
                            basisY[1] *= -1;
                        }
                        for (a = 0; a < 2; ++a) {
                            anteIntX[0] = x + a;
                            basisX[0] = basis[a];
                            basisX[1] = deriv[a];
                            // Boundary conditions along x - slidding
                            if (x == deformationField->nx - 1) {
                                if (a == 1)
                                    anteIntX[0] -= 2;
                                basisX[0] = fabs(basisX[0] - 1);
                                basisX[1] *= -1;
                            }

                            // Compute the basis function values
                            weight[0] = basisX[1] * basisY[0] * basisZ[0];
                            weight[1] = basisX[0] * basisY[1] * basisZ[0];
                            weight[2] = basisX[0] * basisY[0] * basisZ[1];

                            // Get the deformation field index
                            defIndex = (anteIntZ[0] * deformationField->ny + anteIntY[0]) *
                                deformationField->nx + anteIntX[0];

                            // Get the deformation field values
                            defX = deformationFieldPtrX[defIndex];
                            defY = deformationFieldPtrY[defIndex];
                            defZ = deformationFieldPtrZ[defIndex];

                            // Symmetric difference to compute the derivatives
                            jacMat.m[0][0] += static_cast<float>(weight[0] * defX);
                            jacMat.m[0][1] += static_cast<float>(weight[1] * defX);
                            jacMat.m[0][2] += static_cast<float>(weight[2] * defX);
                            jacMat.m[1][0] += static_cast<float>(weight[0] * defY);
                            jacMat.m[1][1] += static_cast<float>(weight[1] * defY);
                            jacMat.m[1][2] += static_cast<float>(weight[2] * defY);
                            jacMat.m[2][0] += static_cast<float>(weight[0] * defZ);
                            jacMat.m[2][1] += static_cast<float>(weight[1] * defZ);
                            jacMat.m[2][2] += static_cast<float>(weight[2] * defZ);
                        }
                    }
                }
                // reorient and scale the Jacobian matrix
                jacMat = nifti_mat33_mul(reorient, jacMat);
                jacMat.m[0][0] /= realSpacing[0];
                jacMat.m[0][1] /= realSpacing[1];
                jacMat.m[0][2] /= realSpacing[2];
                jacMat.m[1][0] /= realSpacing[0];
                jacMat.m[1][1] /= realSpacing[1];
                jacMat.m[1][2] /= realSpacing[2];
                jacMat.m[2][0] /= realSpacing[0];
                jacMat.m[2][1] /= realSpacing[1];
                jacMat.m[2][2] /= realSpacing[2];

                // Modulate the gradient scalar values
                warpedIntensityX[warpedIndex] = jacMat.m[0][0] * val_x + jacMat.m[0][1] * val_y + jacMat.m[0][2] * val_z;
                warpedIntensityY[warpedIndex] = jacMat.m[1][0] * val_x + jacMat.m[1][1] * val_y + jacMat.m[1][2] * val_z;
                warpedIntensityZ[warpedIndex] = jacMat.m[2][0] * val_x + jacMat.m[2][1] * val_y + jacMat.m[2][2] * val_z;
                ++warpedIndex;
            } // x
        } // y
    } // z
}
/* *************************************************************** */
void reg_resampleGradient(const nifti_image *floatingImage,
                          nifti_image *warpedImage,
                          const nifti_image *deformationField,
                          const int interpolation,
                          const float paddingValue) {
    if (interpolation != 1)
        NR_FATAL_ERROR("Only linear interpolation is supported");
    if (floatingImage->datatype != warpedImage->datatype || floatingImage->datatype != deformationField->datatype)
        NR_FATAL_ERROR("Input images are expected to have the same type");
    if (floatingImage->datatype != NIFTI_TYPE_FLOAT32 && floatingImage->datatype != NIFTI_TYPE_FLOAT64)
        NR_FATAL_ERROR("Input images are expected to be of type float or double");

    std::visit([&](auto&& floImgDataType) {
        using FloImgDataType = std::decay_t<decltype(floImgDataType)>;
        if (warpedImage->nz > 1) {
            reg_trilinearResampleGradient<FloImgDataType>(floatingImage,
                                                          warpedImage,
                                                          deformationField,
                                                          paddingValue);
        } else {
            reg_bilinearResampleGradient<FloImgDataType>(floatingImage,
                                                         warpedImage,
                                                         deformationField,
                                                         paddingValue);
        }
    }, NiftiImage::getFloatingDataType(floatingImage));
}
/* *************************************************************** */
template<class FloatingType, class GradientType, class FieldType>
void TrilinearImageGradient(const nifti_image *floatingImage,
                            const nifti_image *deformationField,
                            nifti_image *warpedGradient,
                            const int *mask,
                            const float paddingValue,
                            const int activeTimePoint) {
    if (activeTimePoint < 0 || activeTimePoint >= floatingImage->nt)
        NR_FATAL_ERROR("The specified active time point is not defined in the floating image");
#ifdef _WIN32
    long index;
    const long referenceVoxelNumber = (long)NiftiImage::calcVoxelNumber(warpedGradient, 3);
    const long floatingVoxelNumber = (long)NiftiImage::calcVoxelNumber(floatingImage, 3);
#else
    size_t index;
    const size_t referenceVoxelNumber = NiftiImage::calcVoxelNumber(warpedGradient, 3);
    const size_t floatingVoxelNumber = NiftiImage::calcVoxelNumber(floatingImage, 3);
#endif
    const FloatingType *floatingIntensityPtr = static_cast<FloatingType*>(floatingImage->data);
    const FloatingType *floatingIntensity = &floatingIntensityPtr[activeTimePoint * floatingVoxelNumber];

    const FieldType *deformationFieldPtrX = static_cast<FieldType*>(deformationField->data);
    const FieldType *deformationFieldPtrY = &deformationFieldPtrX[referenceVoxelNumber];
    const FieldType *deformationFieldPtrZ = &deformationFieldPtrY[referenceVoxelNumber];

    GradientType *warpedGradientPtrX = static_cast<GradientType*>(warpedGradient->data);
    GradientType *warpedGradientPtrY = &warpedGradientPtrX[referenceVoxelNumber];
    GradientType *warpedGradientPtrZ = &warpedGradientPtrY[referenceVoxelNumber];

    const mat44 *floatingIJKMatrix;
    if (floatingImage->sform_code > 0)
        floatingIJKMatrix = &floatingImage->sto_ijk;
    else floatingIJKMatrix = &floatingImage->qto_ijk;

    NR_DEBUG("3D linear gradient computation of volume number " << activeTimePoint);

    int previous[3], a, b, c, X, Y, Z;
    FieldType position[3], xBasis[2], yBasis[2], zBasis[2];
    FieldType deriv[2];
    deriv[0] = -1;
    deriv[1] = 1;
    FieldType relative, world[3], grad[3], coeff;
    FieldType xxTempNewValue, yyTempNewValue, zzTempNewValue, xTempNewValue, yTempNewValue;
    const FloatingType *zPointer, *xyzPointer;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(world, position, previous, xBasis, yBasis, zBasis, relative, grad, coeff, \
    a, b, c, X, Y, Z, zPointer, xyzPointer, xTempNewValue, yTempNewValue, xxTempNewValue, yyTempNewValue, zzTempNewValue) \
    shared(floatingIntensity, referenceVoxelNumber, floatingVoxelNumber, deriv, paddingValue, \
    deformationFieldPtrX, deformationFieldPtrY, deformationFieldPtrZ, mask, \
    floatingIJKMatrix, floatingImage, warpedGradientPtrX, warpedGradientPtrY, warpedGradientPtrZ)
#endif // _OPENMP
    for (index = 0; index < referenceVoxelNumber; index++) {
        grad[0] = 0;
        grad[1] = 0;
        grad[2] = 0;

        if (mask[index] > -1) {
            world[0] = (FieldType)deformationFieldPtrX[index];
            world[1] = (FieldType)deformationFieldPtrY[index];
            world[2] = (FieldType)deformationFieldPtrZ[index];

            /* real -> voxel; floating space */
            reg_mat44_mul(floatingIJKMatrix, world, position);

            previous[0] = Floor(position[0]);
            previous[1] = Floor(position[1]);
            previous[2] = Floor(position[2]);
            // basis values along the x axis
            relative = position[0] - (FieldType)previous[0];
            xBasis[0] = (FieldType)(1.0 - relative);
            xBasis[1] = relative;
            // basis values along the y axis
            relative = position[1] - (FieldType)previous[1];
            yBasis[0] = (FieldType)(1.0 - relative);
            yBasis[1] = relative;
            // basis values along the z axis
            relative = position[2] - (FieldType)previous[2];
            zBasis[0] = (FieldType)(1.0 - relative);
            zBasis[1] = relative;

            // The padding value is used for interpolation if it is different from NaN
            if (paddingValue == paddingValue) {
                for (c = 0; c < 2; c++) {
                    Z = previous[2] + c;
                    if (Z > -1 && Z < floatingImage->nz) {
                        zPointer = &floatingIntensity[Z * floatingImage->nx * floatingImage->ny];
                        xxTempNewValue = 0;
                        yyTempNewValue = 0;
                        zzTempNewValue = 0;
                        for (b = 0; b < 2; b++) {
                            Y = previous[1] + b;
                            if (Y > -1 && Y < floatingImage->ny) {
                                xyzPointer = &zPointer[Y * floatingImage->nx + previous[0]];
                                xTempNewValue = 0;
                                yTempNewValue = 0;
                                for (a = 0; a < 2; a++) {
                                    X = previous[0] + a;
                                    if (X > -1 && X < floatingImage->nx) {
                                        coeff = static_cast<FieldType>(*xyzPointer);
                                        xTempNewValue += coeff * deriv[a];
                                        yTempNewValue += coeff * xBasis[a];
                                    } // end X in range
                                    else {
                                        xTempNewValue += paddingValue * deriv[a];
                                        yTempNewValue += paddingValue * xBasis[a];
                                    }
                                    xyzPointer++;
                                } // end a
                                xxTempNewValue += xTempNewValue * yBasis[b];
                                yyTempNewValue += yTempNewValue * deriv[b];
                                zzTempNewValue += yTempNewValue * yBasis[b];
                            } // end Y in range
                            else {
                                xxTempNewValue += paddingValue * yBasis[b];
                                yyTempNewValue += paddingValue * deriv[b];
                                zzTempNewValue += paddingValue * yBasis[b];
                            }
                        } // end b
                        grad[0] += xxTempNewValue * zBasis[c];
                        grad[1] += yyTempNewValue * zBasis[c];
                        grad[2] += zzTempNewValue * deriv[c];
                    } // end Z in range
                    else {
                        grad[0] += paddingValue * zBasis[c];
                        grad[1] += paddingValue * zBasis[c];
                        grad[2] += paddingValue * deriv[c];
                    }
                } // end c
            } // end padding value is different from NaN
            else if (previous[0] >= 0.f && previous[0] < (floatingImage->nx - 1) &&
                     previous[1] >= 0.f && previous[1] < (floatingImage->ny - 1) &&
                     previous[2] >= 0.f && previous[2] < (floatingImage->nz - 1)) {
                for (c = 0; c < 2; c++) {
                    Z = previous[2] + c;
                    zPointer = &floatingIntensity[Z * floatingImage->nx * floatingImage->ny];
                    xxTempNewValue = 0;
                    yyTempNewValue = 0;
                    zzTempNewValue = 0;
                    for (b = 0; b < 2; b++) {
                        Y = previous[1] + b;
                        xyzPointer = &zPointer[Y * floatingImage->nx + previous[0]];
                        xTempNewValue = 0;
                        yTempNewValue = 0;
                        for (a = 0; a < 2; a++) {
                            X = previous[0] + a;
                            coeff = static_cast<FieldType>(*xyzPointer);
                            xTempNewValue += coeff * deriv[a];
                            yTempNewValue += coeff * xBasis[a];
                            xyzPointer++;
                        } // end a
                        xxTempNewValue += xTempNewValue * yBasis[b];
                        yyTempNewValue += yTempNewValue * deriv[b];
                        zzTempNewValue += yTempNewValue * yBasis[b];
                    } // end b
                    grad[0] += xxTempNewValue * zBasis[c];
                    grad[1] += yyTempNewValue * zBasis[c];
                    grad[2] += zzTempNewValue * deriv[c];
                } // end c
            } // end padding value is NaN
            else grad[0] = grad[1] = grad[2] = 0;
        } // end mask

        warpedGradientPtrX[index] = static_cast<GradientType>(grad[0]);
        warpedGradientPtrY[index] = static_cast<GradientType>(grad[1]);
        warpedGradientPtrZ[index] = static_cast<GradientType>(grad[2]);
    }
}
/* *************************************************************** */
template<class FloatingType, class GradientType, class FieldType>
void BilinearImageGradient(const nifti_image *floatingImage,
                           const nifti_image *deformationField,
                           nifti_image *warpedGradient,
                           const int *mask,
                           const float paddingValue,
                           const int activeTimePoint) {
    if (activeTimePoint < 0 || activeTimePoint >= floatingImage->nt)
        NR_FATAL_ERROR("The specified active time point is not defined in the floating image");
#ifdef _WIN32
    long index;
    const long referenceVoxelNumber = (long)NiftiImage::calcVoxelNumber(warpedGradient, 2);
    const long floatingVoxelNumber = (long)NiftiImage::calcVoxelNumber(floatingImage, 2);
#else
    size_t index;
    const size_t referenceVoxelNumber = NiftiImage::calcVoxelNumber(warpedGradient, 2);
    const size_t floatingVoxelNumber = NiftiImage::calcVoxelNumber(floatingImage, 2);
#endif
    const FloatingType *floatingIntensityPtr = static_cast<FloatingType*>(floatingImage->data);
    const FloatingType *floatingIntensity = &floatingIntensityPtr[activeTimePoint * floatingVoxelNumber];

    const FieldType *deformationFieldPtrX = static_cast<FieldType*>(deformationField->data);
    const FieldType *deformationFieldPtrY = &deformationFieldPtrX[referenceVoxelNumber];

    GradientType *warpedGradientPtrX = static_cast<GradientType*>(warpedGradient->data);
    GradientType *warpedGradientPtrY = &warpedGradientPtrX[referenceVoxelNumber];

    const mat44 *floatingIJKMatrix;
    if (floatingImage->sform_code > 0)
        floatingIJKMatrix = &floatingImage->sto_ijk;
    else floatingIJKMatrix = &floatingImage->qto_ijk;

    NR_DEBUG("2D linear gradient computation of volume number " << activeTimePoint);

    FieldType position[3], xBasis[2], yBasis[2], relative, world[2], grad[2];
    FieldType deriv[2];
    deriv[0] = -1;
    deriv[1] = 1;
    FieldType coeff, xTempNewValue, yTempNewValue;

    int previous[3], a, b, X, Y;
    const FloatingType *xyPointer;

#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(world, position, previous, xBasis, yBasis, relative, grad, coeff, \
    a, b, X, Y, xyPointer, xTempNewValue, yTempNewValue) \
    shared(floatingIntensity, referenceVoxelNumber, floatingVoxelNumber, deriv, \
    deformationFieldPtrX, deformationFieldPtrY, mask, paddingValue, \
    floatingIJKMatrix, floatingImage, warpedGradientPtrX, warpedGradientPtrY)
#endif // _OPENMP
    for (index = 0; index < referenceVoxelNumber; index++) {
        grad[0] = 0;
        grad[1] = 0;

        if (mask[index] > -1) {
            world[0] = (FieldType)deformationFieldPtrX[index];
            world[1] = (FieldType)deformationFieldPtrY[index];

            /* real -> voxel; floating space */
            position[0] = world[0] * floatingIJKMatrix->m[0][0] + world[1] * floatingIJKMatrix->m[0][1] + floatingIJKMatrix->m[0][3];
            position[1] = world[0] * floatingIJKMatrix->m[1][0] + world[1] * floatingIJKMatrix->m[1][1] + floatingIJKMatrix->m[1][3];

            previous[0] = Floor(position[0]);
            previous[1] = Floor(position[1]);
            // basis values along the x axis
            relative = position[0] - (FieldType)previous[0];
            relative = relative > 0 ? relative : 0;
            xBasis[0] = (FieldType)(1.0 - relative);
            xBasis[1] = relative;
            // basis values along the y axis
            relative = position[1] - (FieldType)previous[1];
            relative = relative > 0 ? relative : 0;
            yBasis[0] = (FieldType)(1.0 - relative);
            yBasis[1] = relative;

            for (b = 0; b < 2; b++) {
                Y = previous[1] + b;
                if (Y > -1 && Y < floatingImage->ny) {
                    xyPointer = &floatingIntensity[Y * floatingImage->nx + previous[0]];
                    xTempNewValue = 0;
                    yTempNewValue = 0;
                    for (a = 0; a < 2; a++) {
                        X = previous[0] + a;
                        if (X > -1 && X < floatingImage->nx) {
                            coeff = static_cast<FieldType>(*xyPointer);
                            xTempNewValue += coeff * deriv[a];
                            yTempNewValue += coeff * xBasis[a];
                        } else {
                            xTempNewValue += paddingValue * deriv[a];
                            yTempNewValue += paddingValue * xBasis[a];
                        }
                        xyPointer++;
                    }
                    grad[0] += xTempNewValue * yBasis[b];
                    grad[1] += yTempNewValue * deriv[b];
                } else {
                    grad[0] += paddingValue * yBasis[b];
                    grad[1] += paddingValue * deriv[b];
                }
            }
            if (grad[0] != grad[0]) grad[0] = 0;
            if (grad[1] != grad[1]) grad[1] = 0;
        }// mask

        warpedGradientPtrX[index] = static_cast<GradientType>(grad[0]);
        warpedGradientPtrY[index] = static_cast<GradientType>(grad[1]);
    }
}
/* *************************************************************** */
template<class FloatingType, class GradientType, class FieldType>
void CubicSplineImageGradient3D(const nifti_image *floatingImage,
                                const nifti_image *deformationField,
                                nifti_image *warpedGradient,
                                const int *mask,
                                const float paddingValue,
                                const int activeTimePoint) {
    if (activeTimePoint < 0 || activeTimePoint >= floatingImage->nt)
        NR_FATAL_ERROR("The specified active time point is not defined in the floating image");
#ifdef _WIN32
    long index;
    const long referenceVoxelNumber = (long)NiftiImage::calcVoxelNumber(warpedGradient, 3);
    const long floatingVoxelNumber = (long)NiftiImage::calcVoxelNumber(floatingImage, 3);
#else
    size_t index;
    const size_t referenceVoxelNumber = NiftiImage::calcVoxelNumber(warpedGradient, 3);
    const size_t floatingVoxelNumber = NiftiImage::calcVoxelNumber(floatingImage, 3);
#endif
    const FloatingType *floatingIntensityPtr = static_cast<FloatingType*>(floatingImage->data);
    const FloatingType *floatingIntensity = &floatingIntensityPtr[activeTimePoint * floatingVoxelNumber];

    const FieldType *deformationFieldPtrX = static_cast<FieldType*>(deformationField->data);
    const FieldType *deformationFieldPtrY = &deformationFieldPtrX[referenceVoxelNumber];
    const FieldType *deformationFieldPtrZ = &deformationFieldPtrY[referenceVoxelNumber];

    GradientType *warpedGradientPtrX = static_cast<GradientType*>(warpedGradient->data);
    GradientType *warpedGradientPtrY = &warpedGradientPtrX[referenceVoxelNumber];
    GradientType *warpedGradientPtrZ = &warpedGradientPtrY[referenceVoxelNumber];

    const mat44 *floatingIJKMatrix;
    if (floatingImage->sform_code > 0)
        floatingIJKMatrix = &floatingImage->sto_ijk;
    else floatingIJKMatrix = &floatingImage->qto_ijk;

    NR_DEBUG("3D cubic spline gradient computation of volume number " << activeTimePoint);

    int previous[3], c, Z, b, Y, a;

    double xBasis[4], yBasis[4], zBasis[4], xDeriv[4], yDeriv[4], zDeriv[4], relative;
    FieldType coeff, position[3], world[3], grad[3];
    FieldType xxTempNewValue, yyTempNewValue, zzTempNewValue, xTempNewValue, yTempNewValue;
    const FloatingType *zPointer, *yzPointer, *xyzPointer;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(world, position, previous, xBasis, yBasis, zBasis, xDeriv, yDeriv, zDeriv, relative, grad, coeff, \
    a, b, c, Y, Z, zPointer, yzPointer, xyzPointer, xTempNewValue, yTempNewValue, xxTempNewValue, yyTempNewValue, zzTempNewValue) \
    shared(floatingIntensity, referenceVoxelNumber, floatingVoxelNumber, paddingValue, \
    deformationFieldPtrX, deformationFieldPtrY, deformationFieldPtrZ, mask, \
    floatingIJKMatrix, floatingImage, warpedGradientPtrX, warpedGradientPtrY, warpedGradientPtrZ)
#endif // _OPENMP
    for (index = 0; index < referenceVoxelNumber; index++) {
        grad[0] = 0;
        grad[1] = 0;
        grad[2] = 0;

        if (mask[index] > -1) {
            world[0] = (FieldType)deformationFieldPtrX[index];
            world[1] = (FieldType)deformationFieldPtrY[index];
            world[2] = (FieldType)deformationFieldPtrZ[index];

            /* real -> voxel; floating space */
            reg_mat44_mul(floatingIJKMatrix, world, position);

            previous[0] = Floor(position[0]);
            previous[1] = Floor(position[1]);
            previous[2] = Floor(position[2]);

            // basis values along the x axis
            relative = position[0] - (FieldType)previous[0];
            interpCubicSplineKernel(relative, xBasis, xDeriv);

            // basis values along the y axis
            relative = position[1] - (FieldType)previous[1];
            interpCubicSplineKernel(relative, yBasis, yDeriv);

            // basis values along the z axis
            relative = position[2] - (FieldType)previous[2];
            interpCubicSplineKernel(relative, zBasis, zDeriv);

            previous[0]--;
            previous[1]--;
            previous[2]--;

            for (c = 0; c < 4; c++) {
                Z = previous[2] + c;
                if (-1 < Z && Z < floatingImage->nz) {
                    zPointer = &floatingIntensity[Z * floatingImage->nx * floatingImage->ny];
                    xxTempNewValue = 0;
                    yyTempNewValue = 0;
                    zzTempNewValue = 0;
                    for (b = 0; b < 4; b++) {
                        Y = previous[1] + b;
                        yzPointer = &zPointer[Y * floatingImage->nx];
                        if (-1 < Y && Y < floatingImage->ny) {
                            xyzPointer = &yzPointer[previous[0]];
                            xTempNewValue = 0;
                            yTempNewValue = 0;
                            for (a = 0; a < 4; a++) {
                                if (-1 < (previous[0] + a) && (previous[0] + a) < floatingImage->nx) {
                                    coeff = static_cast<FieldType>(*xyzPointer);
                                    xTempNewValue += coeff * static_cast<FieldType>(xDeriv[a]);
                                    yTempNewValue += coeff * static_cast<FieldType>(xBasis[a]);
                                } // previous[0]+a in range
                                else {
                                    xTempNewValue += static_cast<FieldType>(paddingValue * xDeriv[a]);
                                    yTempNewValue += static_cast<FieldType>(paddingValue * xBasis[a]);
                                }
                                xyzPointer++;
                            } // a
                            xxTempNewValue += static_cast<FieldType>(xTempNewValue * yBasis[b]);
                            yyTempNewValue += static_cast<FieldType>(yTempNewValue * yDeriv[b]);
                            zzTempNewValue += static_cast<FieldType>(yTempNewValue * yBasis[b]);
                        } // Y in range
                        else {
                            xxTempNewValue += static_cast<FieldType>(paddingValue * yBasis[b]);
                            yyTempNewValue += static_cast<FieldType>(paddingValue * yDeriv[b]);
                            zzTempNewValue += static_cast<FieldType>(paddingValue * yBasis[b]);
                        }
                    } // b
                    grad[0] += static_cast<FieldType>(xxTempNewValue * zBasis[c]);
                    grad[1] += static_cast<FieldType>(yyTempNewValue * zBasis[c]);
                    grad[2] += static_cast<FieldType>(zzTempNewValue * zDeriv[c]);
                } // Z in range
                else {
                    grad[0] += static_cast<FieldType>(paddingValue * zBasis[c]);
                    grad[1] += static_cast<FieldType>(paddingValue * zBasis[c]);
                    grad[2] += static_cast<FieldType>(paddingValue * zDeriv[c]);
                }
            } // c

            grad[0] = grad[0] == grad[0] ? grad[0] : 0;
            grad[1] = grad[1] == grad[1] ? grad[1] : 0;
            grad[2] = grad[2] == grad[2] ? grad[2] : 0;
        } // outside of the mask

        warpedGradientPtrX[index] = static_cast<GradientType>(grad[0]);
        warpedGradientPtrY[index] = static_cast<GradientType>(grad[1]);
        warpedGradientPtrZ[index] = static_cast<GradientType>(grad[2]);
    }
}
/* *************************************************************** */
template<class FloatingType, class GradientType, class FieldType>
void CubicSplineImageGradient2D(const nifti_image *floatingImage,
                                const nifti_image *deformationField,
                                nifti_image *warpedGradient,
                                const int *mask,
                                const float paddingValue,
                                const int activeTimePoint) {
    if (activeTimePoint < 0 || activeTimePoint >= floatingImage->nt)
        NR_FATAL_ERROR("The specified active time point is not defined in the floating image");
#ifdef _WIN32
    long index;
    const long referenceVoxelNumber = (long)NiftiImage::calcVoxelNumber(warpedGradient, 2);
    const long floatingVoxelNumber = (long)NiftiImage::calcVoxelNumber(floatingImage, 2);
#else
    size_t index;
    const size_t referenceVoxelNumber = NiftiImage::calcVoxelNumber(warpedGradient, 2);
    const size_t floatingVoxelNumber = NiftiImage::calcVoxelNumber(floatingImage, 2);
#endif
    const FloatingType *floatingIntensityPtr = static_cast<FloatingType*>(floatingImage->data);
    const FloatingType *floatingIntensity = &floatingIntensityPtr[activeTimePoint * floatingVoxelNumber];

    const FieldType *deformationFieldPtrX = static_cast<FieldType*>(deformationField->data);
    const FieldType *deformationFieldPtrY = &deformationFieldPtrX[referenceVoxelNumber];

    GradientType *warpedGradientPtrX = static_cast<GradientType*>(warpedGradient->data);
    GradientType *warpedGradientPtrY = &warpedGradientPtrX[referenceVoxelNumber];

    const mat44 *floatingIJKMatrix;
    if (floatingImage->sform_code > 0)
        floatingIJKMatrix = &floatingImage->sto_ijk;
    else floatingIJKMatrix = &floatingImage->qto_ijk;

    NR_DEBUG("2D cubic spline gradient computation of volume number " << activeTimePoint);

    int previous[2], b, Y, a;
    double xBasis[4], yBasis[4], xDeriv[4], yDeriv[4], relative;
    FieldType coeff, position[3], world[3], grad[2];
    FieldType xTempNewValue, yTempNewValue;
    const FloatingType *yPointer, *xyPointer;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(world, position, previous, xBasis, yBasis, xDeriv, yDeriv, relative, grad, coeff, \
    a, b, Y, yPointer, xyPointer, xTempNewValue, yTempNewValue) \
    shared(floatingIntensity, referenceVoxelNumber, floatingVoxelNumber, \
    deformationFieldPtrX, deformationFieldPtrY, mask, paddingValue, \
    floatingIJKMatrix, floatingImage, warpedGradientPtrX, warpedGradientPtrY)
#endif // _OPENMP
    for (index = 0; index < referenceVoxelNumber; index++) {
        grad[0] = 0;
        grad[1] = 0;

        if (mask[index] > -1) {
            world[0] = (FieldType)deformationFieldPtrX[index];
            world[1] = (FieldType)deformationFieldPtrY[index];

            /* real -> voxel; floating space */
            position[0] = world[0] * floatingIJKMatrix->m[0][0] + world[1] * floatingIJKMatrix->m[0][1] + floatingIJKMatrix->m[0][3];
            position[1] = world[0] * floatingIJKMatrix->m[1][0] + world[1] * floatingIJKMatrix->m[1][1] + floatingIJKMatrix->m[1][3];

            previous[0] = Floor(position[0]);
            previous[1] = Floor(position[1]);
            // basis values along the x axis
            relative = position[0] - (FieldType)previous[0];
            relative = relative > 0 ? relative : 0;
            interpCubicSplineKernel(relative, xBasis, xDeriv);
            // basis values along the y axis
            relative = position[1] - (FieldType)previous[1];
            relative = relative > 0 ? relative : 0;
            interpCubicSplineKernel(relative, yBasis, yDeriv);

            previous[0]--;
            previous[1]--;

            for (b = 0; b < 4; b++) {
                Y = previous[1] + b;
                yPointer = &floatingIntensity[Y * floatingImage->nx];
                if (-1 < Y && Y < floatingImage->ny) {
                    xyPointer = &yPointer[previous[0]];
                    xTempNewValue = 0;
                    yTempNewValue = 0;
                    for (a = 0; a < 4; a++) {
                        if (-1 < (previous[0] + a) && (previous[0] + a) < floatingImage->nx) {
                            coeff = static_cast<FieldType>(*xyPointer);
                            xTempNewValue += static_cast<FieldType>(coeff * xDeriv[a]);
                            yTempNewValue += static_cast<FieldType>(coeff * xBasis[a]);
                        } // previous[0]+a in range
                        else {
                            xTempNewValue += static_cast<FieldType>(paddingValue * xDeriv[a]);
                            yTempNewValue += static_cast<FieldType>(paddingValue * xBasis[a]);
                        }
                        xyPointer++;
                    } // a
                    grad[0] += static_cast<FieldType>(xTempNewValue * yBasis[b]);
                    grad[1] += static_cast<FieldType>(yTempNewValue * yDeriv[b]);
                } // Y in range
                else {
                    grad[0] += static_cast<FieldType>(paddingValue * yBasis[b]);
                    grad[1] += static_cast<FieldType>(paddingValue * yDeriv[b]);
                }
            } // b

            grad[0] = grad[0] == grad[0] ? grad[0] : 0;
            grad[1] = grad[1] == grad[1] ? grad[1] : 0;
        } // outside of the mask

        warpedGradientPtrX[index] = static_cast<GradientType>(grad[0]);
        warpedGradientPtrY[index] = static_cast<GradientType>(grad[1]);
    }
}
/* *************************************************************** */
template <class FieldType, class FloatingType, class GradientType>
void reg_getImageGradient(nifti_image *floatingImage,
                          nifti_image *warpedGradient,
                          const nifti_image *deformationField,
                          const int *mask,
                          const int interpolation,
                          const float paddingValue,
                          const int activeTimePoint,
                          const int *dtIndicies,
                          const mat33 *jacMat,
                          const nifti_image *warpedImage = nullptr) {
    // The floating image data is copied in case one deal with DTI
    void *originalFloatingData = nullptr;
    // The DTI are logged
    reg_dti_resampling_preprocessing<FloatingType>(floatingImage, &originalFloatingData, dtIndicies);
    /* The deformation field contains the position in the real world */
    if (interpolation == 3) {
        if (deformationField->nu > 2) {
            CubicSplineImageGradient3D<FloatingType, GradientType, FieldType>(floatingImage,
                                                                              deformationField,
                                                                              warpedGradient,
                                                                              mask,
                                                                              paddingValue,
                                                                              activeTimePoint);
        } else {
            CubicSplineImageGradient2D<FloatingType, GradientType, FieldType>(floatingImage,
                                                                              deformationField,
                                                                              warpedGradient,
                                                                              mask,
                                                                              paddingValue,
                                                                              activeTimePoint);
        }
    } else { // trilinear interpolation [ by default ]
        if (deformationField->nu > 2) {
            TrilinearImageGradient<FloatingType, GradientType, FieldType>(floatingImage,
                                                                          deformationField,
                                                                          warpedGradient,
                                                                          mask,
                                                                          paddingValue,
                                                                          activeTimePoint);
        } else {
            BilinearImageGradient<FloatingType, GradientType, FieldType>(floatingImage,
                                                                         deformationField,
                                                                         warpedGradient,
                                                                         mask,
                                                                         paddingValue,
                                                                         activeTimePoint);
        }
    }
    // The temporary logged floating array is deleted
    if (originalFloatingData != nullptr) {
        free(floatingImage->data);
        floatingImage->data = originalFloatingData;
        originalFloatingData = nullptr;
    }
    // The interpolated tensors are reoriented and exponentiated
    reg_dti_resampling_postprocessing<FloatingType>(warpedGradient, mask, jacMat, dtIndicies, warpedImage);
}
/* *************************************************************** */
void reg_getImageGradient(nifti_image *floatingImage,
                          nifti_image *warpedGradient,
                          const nifti_image *deformationField,
                          const int *mask,
                          const int interpolation,
                          const float paddingValue,
                          const int activeTimePoint,
                          const bool *dtiTimePoint,
                          const mat33 *jacMat,
                          const nifti_image *warpedImage) {
    if (deformationField->datatype != NIFTI_TYPE_FLOAT32 && deformationField->datatype != NIFTI_TYPE_FLOAT64)
        NR_FATAL_ERROR("The deformation field image is expected to be of type float or double");
    if (warpedGradient->datatype != NIFTI_TYPE_FLOAT32 && warpedGradient->datatype != NIFTI_TYPE_FLOAT64)
        NR_FATAL_ERROR("The warped gradient image is expected to be of type float or double");

    // a mask array is created if no mask is specified
    bool MrPropreRule = false;
    if (mask == nullptr) {
        // voxels in the backgreg_round are set to -1 so 0 will do the job here
        mask = (int*)calloc(NiftiImage::calcVoxelNumber(deformationField, 3), sizeof(int));
        MrPropreRule = true;
    }

    // Define the DTI indices if required
    int dtIndicies[6];
    for (int i = 0; i < 6; ++i) dtIndicies[i] = -1;
    if (dtiTimePoint != nullptr) {
        if (jacMat == nullptr)
            NR_FATAL_ERROR("DTI resampling: No Jacobian matrix array has been provided");
        int j = 0;
        for (int i = 0; i < floatingImage->nt; ++i) {
            if (dtiTimePoint[i])
                dtIndicies[j++] = i;
        }
        if ((floatingImage->nz > 1 && j != 6) && (floatingImage->nz == 1 && j != 3))
            NR_FATAL_ERROR("DTI resampling: Unexpected number of DTI components");
    }

    std::visit([&](auto&& defFieldDataType, auto&& floImgDataType, auto&& warpedGradDataType) {
        using DefFieldDataType = std::decay_t<decltype(defFieldDataType)>;
        using FloImgDataType = std::decay_t<decltype(floImgDataType)>;
        using WarpedGradDataType = std::decay_t<decltype(warpedGradDataType)>;
        reg_getImageGradient<DefFieldDataType, FloImgDataType, WarpedGradDataType>(floatingImage,
                                                                                   warpedGradient,
                                                                                   deformationField,
                                                                                   mask,
                                                                                   interpolation,
                                                                                   paddingValue,
                                                                                   activeTimePoint,
                                                                                   dtIndicies,
                                                                                   jacMat,
                                                                                   warpedImage);
    }, NiftiImage::getFloatingDataType(deformationField), NiftiImage::getDataType(floatingImage), NiftiImage::getFloatingDataType(warpedGradient));

    if (MrPropreRule)
        free(const_cast<int*>(mask));
}
/* *************************************************************** */
template<class DataType>
void reg_getImageGradient_symDiff(const nifti_image *img,
                                  nifti_image *gradImg,
                                  const int *mask,
                                  const float paddingValue,
                                  const int timePoint) {
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(img, 3);

    int dimImg = img->nz > 1 ? 3 : 2;
    int x, y, z;

    const DataType *imgPtr = static_cast<DataType*>(img->data);
    const DataType *currentImgPtr = &imgPtr[timePoint * voxelNumber];

    DataType *gradPtrX = static_cast<DataType*>(gradImg->data);
    DataType *gradPtrY = &gradPtrX[voxelNumber];
    DataType *gradPtrZ = nullptr;
    if (dimImg == 3)
        gradPtrZ = &gradPtrY[voxelNumber];

    DataType valX, valY, valZ, pre, post;

#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(img, currentImgPtr, mask, \
    gradPtrX, gradPtrY, gradPtrZ, paddingValue) \
    private(x, y, pre, post, valX, valY, valZ)
#endif
    for (z = 0; z < img->nz; ++z) {
        size_t voxIndex = z * img->nx * img->ny;
        for (y = 0; y < img->ny; ++y) {
            for (x = 0; x < img->nx; ++x) {
                valX = valY = valZ = 0;
                if (mask[voxIndex] > -1) {

                    pre = post = paddingValue;
                    if (x < img->nx - 1) post = currentImgPtr[voxIndex + 1];
                    if (x > 0) pre = currentImgPtr[voxIndex - 1];
                    valX = (post - pre) / 2.f;

                    pre = post = paddingValue;
                    if (y < img->ny - 1) post = currentImgPtr[voxIndex + img->nx];
                    if (y > 0) pre = currentImgPtr[voxIndex - img->nx];
                    valY = (post - pre) / 2.f;

                    if (gradPtrZ != nullptr) {
                        pre = post = paddingValue;
                        if (z < img->nz - 1) post = currentImgPtr[voxIndex + img->nx * img->ny];
                        if (z > 0) pre = currentImgPtr[voxIndex - img->nx * img->ny];
                        valZ = (post - pre) / 2.f;
                    }
                }
                gradPtrX[voxIndex] = valX == valX ? valX : 0;
                gradPtrY[voxIndex] = valY == valY ? valY : 0;
                if (gradPtrZ != nullptr)
                    gradPtrZ[voxIndex] = valZ == valZ ? valZ : 0;
                ++voxIndex;
            } // x
        } // y
    } // z
}
/* *************************************************************** */
void reg_getImageGradient_symDiff(const nifti_image *img,
                                  nifti_image *gradImg,
                                  const int *mask,
                                  const float paddingValue,
                                  const int timePoint) {
    if (img->datatype != gradImg->datatype)
        NR_FATAL_ERROR("Input images are expected to be of the same type");
    if (img->datatype != NIFTI_TYPE_FLOAT32 && img->datatype != NIFTI_TYPE_FLOAT64)
        NR_FATAL_ERROR("Input images are expected to be of floating precision type");

    std::visit([&](auto&& imgDataType) {
        using ImgDataType = std::decay_t<decltype(imgDataType)>;
        reg_getImageGradient_symDiff<ImgDataType>(img, gradImg, mask, paddingValue, timePoint);
    }, NiftiImage::getFloatingDataType(img));
}
/* *************************************************************** */
nifti_image* reg_makeIsotropic(nifti_image *img, int inter) {
    // Get the smallest voxel size
    float smallestPixDim = img->pixdim[1];
    for (size_t i = 2; i < 4; ++i)
        if (i < static_cast<size_t>(img->dim[0] + 2))
            smallestPixDim = img->pixdim[i] < smallestPixDim ? img->pixdim[i] : smallestPixDim;
    // Define the size of the new image
    int newDim[8];
    for (size_t i = 0; i < 8; ++i) newDim[i] = img->dim[i];
    for (size_t i = 1; i < 4; ++i) {
        if (i < static_cast<size_t>(img->dim[0] + 1))
            newDim[i] = Ceil(img->dim[i] * img->pixdim[i] / smallestPixDim);
    }
    // Create the new image
    nifti_image *newImg = nifti_make_new_nim(newDim, img->datatype, true);
    newImg->pixdim[1] = newImg->dx = smallestPixDim;
    newImg->pixdim[2] = newImg->dy = smallestPixDim;
    newImg->pixdim[3] = newImg->dz = smallestPixDim;
    newImg->qform_code = img->qform_code;
    newImg->sform_code = img->sform_code;
    // Update the qform matrix
    newImg->qfac = img->qfac;
    newImg->quatern_b = img->quatern_b;
    newImg->quatern_c = img->quatern_c;
    newImg->quatern_d = img->quatern_d;
    newImg->qoffset_x = img->qoffset_x + smallestPixDim / 2.f - img->dx / 2.f;
    newImg->qoffset_y = img->qoffset_y + smallestPixDim / 2.f - img->dy / 2.f;
    newImg->qoffset_z = img->qoffset_z + smallestPixDim / 2.f - img->dz / 2.f;
    newImg->qto_xyz = nifti_quatern_to_mat44(newImg->quatern_b,
                                             newImg->quatern_c,
                                             newImg->quatern_d,
                                             newImg->qoffset_x,
                                             newImg->qoffset_y,
                                             newImg->qoffset_z,
                                             smallestPixDim,
                                             smallestPixDim,
                                             smallestPixDim,
                                             newImg->qfac);
    newImg->qto_ijk = nifti_mat44_inverse(newImg->qto_xyz);
    if (newImg->sform_code > 0) {
        // Compute the new sform
        float scalingRatio[3];
        scalingRatio[0] = newImg->dx / img->dx;
        scalingRatio[1] = newImg->dy / img->dy;
        scalingRatio[2] = newImg->dz / img->dz;
        newImg->sto_xyz.m[0][0] = img->sto_xyz.m[0][0] * scalingRatio[0];
        newImg->sto_xyz.m[1][0] = img->sto_xyz.m[1][0] * scalingRatio[0];
        newImg->sto_xyz.m[2][0] = img->sto_xyz.m[2][0] * scalingRatio[0];
        newImg->sto_xyz.m[3][0] = img->sto_xyz.m[3][0];
        newImg->sto_xyz.m[0][1] = img->sto_xyz.m[0][1] * scalingRatio[1];
        newImg->sto_xyz.m[1][1] = img->sto_xyz.m[1][1] * scalingRatio[1];
        newImg->sto_xyz.m[2][1] = img->sto_xyz.m[2][1] * scalingRatio[1];
        newImg->sto_xyz.m[3][1] = img->sto_xyz.m[3][1];
        newImg->sto_xyz.m[0][2] = img->sto_xyz.m[0][2] * scalingRatio[2];
        newImg->sto_xyz.m[1][2] = img->sto_xyz.m[1][2] * scalingRatio[2];
        newImg->sto_xyz.m[2][2] = img->sto_xyz.m[2][2] * scalingRatio[2];
        newImg->sto_xyz.m[3][2] = img->sto_xyz.m[3][2];
        newImg->sto_xyz.m[0][3] = img->sto_xyz.m[0][3] + smallestPixDim / 2.f - img->dx / 2.f;
        newImg->sto_xyz.m[1][3] = img->sto_xyz.m[1][3] + smallestPixDim / 2.f - img->dy / 2.f;
        newImg->sto_xyz.m[2][3] = img->sto_xyz.m[2][3] + smallestPixDim / 2.f - img->dz / 2.f;
        newImg->sto_xyz.m[3][3] = img->sto_xyz.m[3][3];
        newImg->sto_ijk = nifti_mat44_inverse(newImg->sto_xyz);
    }
    reg_checkAndCorrectDimension(newImg);
    // Create a deformation field
    nifti_image *def = nifti_copy_nim_info(newImg);
    def->dim[0] = def->ndim = 5;
    def->dim[4] = def->nt = 1;
    def->pixdim[4] = def->dt = 1.0;
    def->dim[5] = def->nu = newImg->nz > 1 ? 3 : 2;
    def->pixdim[5] = def->du = 1.0;
    def->dim[6] = def->nv = 1;
    def->pixdim[6] = def->dv = 1.0;
    def->dim[7] = def->nw = 1;
    def->pixdim[7] = def->dw = 1.0;
    def->nvox = NiftiImage::calcVoxelNumber(def, def->ndim);
    def->nbyper = sizeof(float);
    def->datatype = NIFTI_TYPE_FLOAT32;
    def->data = calloc(def->nvox, def->nbyper);
    // Fill the deformation field with an identity transformation
    reg_getDeformationFromDisplacement(def);
    // resample the original image into the space of the new image
    reg_resampleImage(img, newImg, def, nullptr, inter, 0.f);
    nifti_set_filenames(newImg, "tempIsotropicImage", 0, 0);
    nifti_image_free(def);
    return newImg;
}
/* *************************************************************** */
