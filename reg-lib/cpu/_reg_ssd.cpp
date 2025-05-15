/*
 *  _reg_ssd.cpp
 *
 *
 *  Created by Marc Modat on 19/05/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_ssd.h"

// #define MRF_USE_SAD

/* *************************************************************** */
void reg_ssd::InitialiseMeasure(nifti_image *refImg,
                                nifti_image *floImg,
                                int *refMask,
                                nifti_image *warpedImg,
                                nifti_image *warpedGrad,
                                nifti_image *voxelBasedGrad,
                                nifti_image *localWeightSim,
                                int *floMask,
                                nifti_image *warpedImgBw,
                                nifti_image *warpedGradBw,
                                nifti_image *voxelBasedGradBw) {
    // Set the pointers using the parent class function
    reg_measure::InitialiseMeasure(refImg,
                                   floImg,
                                   refMask,
                                   warpedImg,
                                   warpedGrad,
                                   voxelBasedGrad,
                                   localWeightSim,
                                   floMask,
                                   warpedImgBw,
                                   warpedGradBw,
                                   voxelBasedGradBw);

    // Check that the input images have the same number of time point
    if (this->referenceImage->nt != this->floatingImage->nt)
        NR_FATAL_ERROR("This number of time point should be the same for both input images");
    // Input images are normalised between 0 and 1
    for (int i = 0; i < this->referenceTimePoints; ++i) {
        if (this->timePointWeights[i] > 0 && normaliseTimePoint[i]) {
            //sets max value over both images to be 1 and min value over both images to be 0
            //scales values such that identical values in the images are still identical after scaling
            const auto [minR, maxR] = NiftiImage(this->referenceImage).data(i).minmax();
            const auto [minF, maxF] = NiftiImage(this->floatingImage).data(i).minmax();
            const auto maxFR = std::max(maxF, maxR);
            const auto minFR = std::min(minF, minR);
            const auto rangeFR = maxFR - minFR;
            reg_intensityRescale(this->referenceImage,
                                 i,
                                 (minR - minFR) / rangeFR,
                                 1 - ((maxFR - maxR) / rangeFR));
            reg_intensityRescale(this->floatingImage,
                                 i,
                                 (minF - minFR) / rangeFR,
                                 1 - ((maxFR - maxF) / rangeFR));
        }
    }
#ifdef MRF_USE_SAD
    NR_WARN("SAD is used instead of SSD");
#endif
#ifndef NDEBUG
    for (int i = 0; i < this->referenceTimePoints; ++i)
        NR_DEBUG("Weight for time point " << i << ": " << this->timePointWeights[i]);
    std::string msg = "Normalize time point:";
    for (int i = 0; i < this->referenceTimePoints; ++i)
        if (this->normaliseTimePoint[i])
            msg += " " + std::to_string(i);
    NR_DEBUG(msg);
    NR_FUNC_CALLED();
#endif
}
/* *************************************************************** */
void reg_ssd::SetNormaliseTimePoint(int timePoint, bool normalise) {
    this->normaliseTimePoint[timePoint] = normalise;
}
/* *************************************************************** */
template<class DataType>
double reg_getSsdValue(const nifti_image *referenceImage,
                       const nifti_image *warpedImage,
                       const double *timePointWeights,
                       const int referenceTimePoints,
                       const nifti_image *jacobianDetImage,
                       const int *mask,
                       const nifti_image *localWeightSim) {
#ifdef _WIN32
    long voxel;
    const long voxelNumber = (long)NiftiImage::calcVoxelNumber(referenceImage, 3);
#else
    size_t voxel;
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);
#endif
    // Create pointers to the reference and warped image data
    const DataType *referencePtr = static_cast<DataType*>(referenceImage->data);
    const DataType *warpedPtr = static_cast<DataType*>(warpedImage->data);
    // Create a pointer to the Jacobian determinant image if defined
    const DataType *jacDetPtr = jacobianDetImage ? static_cast<DataType*>(jacobianDetImage->data) : nullptr;
    // Create a pointer to the local weight image if defined
    const DataType *localWeightPtr = localWeightSim ? static_cast<DataType*>(localWeightSim->data) : nullptr;

    double ssdGlobal = 0;

    // Loop over the different time points
    for (int time = 0; time < referenceTimePoints; ++time) {
        if (timePointWeights[time] > 0) {
            // Create pointers to the current time point of the reference and warped images
            const DataType *currentRefPtr = &referencePtr[time * voxelNumber];
            const DataType *currentWarPtr = &warpedPtr[time * voxelNumber];
            double ssdLocal = 0, n = 0;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(referenceImage, warpedImage, currentRefPtr, currentWarPtr, mask, \
    jacobianDetImage, jacDetPtr, voxelNumber, localWeightPtr) \
    reduction(+:ssdLocal, n)
#endif
            for (voxel = 0; voxel < voxelNumber; ++voxel) {
                // Check if the current voxel belongs to the mask
                if (mask[voxel] > -1) {
                    // Ensure that both ref and warped values are defined
                    const double refValue = currentRefPtr[voxel] * referenceImage->scl_slope + referenceImage->scl_inter;
                    const double warValue = currentWarPtr[voxel] * warpedImage->scl_slope + warpedImage->scl_inter;
                    if (refValue == refValue && warValue == warValue) {
#ifdef MRF_USE_SAD
                        const double diff = fabs(refValue - warValue);
#else
                        const double diff = Square(refValue - warValue);
#endif
                        // Jacobian determinant modulation of the ssd if required
                        const DataType val = jacDetPtr ? jacDetPtr[voxel] : (localWeightPtr ? localWeightPtr[voxel] : 1);
                        ssdLocal += diff * val;
                        n += val;
                    }
                }
            }

            ssdLocal *= timePointWeights[time];
            ssdGlobal -= ssdLocal / n;
        }
    }
    return ssdGlobal;
}
template double reg_getSsdValue<float>(const nifti_image*, const nifti_image*, const double*, const int, const nifti_image*, const int*, const nifti_image*);
template double reg_getSsdValue<double>(const nifti_image*, const nifti_image*, const double*, const int, const nifti_image*, const int*, const nifti_image*);
/* *************************************************************** */
double GetSimilarityMeasureValue(const nifti_image *referenceImage,
                                 const nifti_image *warpedImage,
                                 const double *timePointWeights,
                                 const int referenceTimePoints,
                                 const nifti_image *jacobianDetImage,
                                 const int *mask,
                                 const nifti_image *localWeightSim) {
    return std::visit([&](auto&& refImgDataType) {
        using RefImgDataType = std::decay_t<decltype(refImgDataType)>;
        return reg_getSsdValue<RefImgDataType>(referenceImage,
                                               warpedImage,
                                               timePointWeights,
                                               referenceTimePoints,
                                               jacobianDetImage,
                                               mask,
                                               localWeightSim);
    }, NiftiImage::getFloatingDataType(referenceImage));
}
/* *************************************************************** */
double reg_ssd::GetSimilarityMeasureValueFw() {
    return ::GetSimilarityMeasureValue(this->referenceImage,
                                       this->warpedImage,
                                       this->timePointWeights,
                                       this->referenceTimePoints,
                                       nullptr, // TODO this->forwardJacDetImagePointer,
                                       this->referenceMask,
                                       this->localWeightSim);
}
/* *************************************************************** */
double reg_ssd::GetSimilarityMeasureValueBw() {
    return ::GetSimilarityMeasureValue(this->floatingImage,
                                       this->warpedImageBw,
                                       this->timePointWeights,
                                       this->referenceTimePoints,
                                       nullptr, // TODO this->backwardJacDetImagePointer,
                                       this->floatingMask,
                                       nullptr);
}
/* *************************************************************** */
template <class DataType>
void reg_getVoxelBasedSsdGradient(const nifti_image *referenceImage,
                                  const nifti_image *warpedImage,
                                  const nifti_image *warpedGradient,
                                  nifti_image *measureGradientImage,
                                  const nifti_image *jacobianDetImage,
                                  const int *mask,
                                  const int currentTimePoint,
                                  const double timePointWeight,
                                  const nifti_image *localWeightSim) {
    // Create pointers to the reference and warped images
#ifdef _WIN32
    long voxel;
    const long voxelNumber = (long)NiftiImage::calcVoxelNumber(referenceImage, 3);
#else
    size_t voxel;
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);
#endif
    // Pointers to the image data
    const DataType *refImagePtr = static_cast<DataType*>(referenceImage->data);
    const DataType *currentRefPtr = &refImagePtr[currentTimePoint * voxelNumber];
    const DataType *warImagePtr = static_cast<DataType*>(warpedImage->data);
    const DataType *currentWarPtr = &warImagePtr[currentTimePoint * voxelNumber];

    // Pointers to the spatial gradient of the warped image
    const DataType *spatialGradPtrX = static_cast<DataType*>(warpedGradient->data);
    const DataType *spatialGradPtrY = &spatialGradPtrX[voxelNumber];
    const DataType *spatialGradPtrZ = referenceImage->nz > 1 ? &spatialGradPtrY[voxelNumber] : nullptr;

    // Pointers to the measure of similarity gradient
    DataType *measureGradPtrX = static_cast<DataType*>(measureGradientImage->data);
    DataType *measureGradPtrY = &measureGradPtrX[voxelNumber];
    DataType *measureGradPtrZ = referenceImage->nz > 1 ? &measureGradPtrY[voxelNumber] : nullptr;

    // Create a pointer to the Jacobian determinant values if defined
    const DataType *jacDetPtr = jacobianDetImage ? static_cast<DataType*>(jacobianDetImage->data) : nullptr;
    // Create a pointer to the local weight image if defined
    const DataType *localWeightPtr = localWeightSim ? static_cast<DataType*>(localWeightSim->data) : nullptr;

    // Find number of active voxels and correct weight
    size_t activeVoxelNumber = 0;
    for (voxel = 0; voxel < voxelNumber; voxel++) {
        if (mask[voxel] > -1) {
            if (currentRefPtr[voxel] == currentRefPtr[voxel] && currentWarPtr[voxel] == currentWarPtr[voxel])
                activeVoxelNumber++;
        }
    }
    const double adjustedWeight = timePointWeight / activeVoxelNumber;

#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(referenceImage, warpedImage, currentRefPtr, currentWarPtr, \
    mask, jacDetPtr, spatialGradPtrX, spatialGradPtrY, spatialGradPtrZ, \
    measureGradPtrX, measureGradPtrY, measureGradPtrZ, voxelNumber, \
    localWeightPtr, adjustedWeight)
#endif
    for (voxel = 0; voxel < voxelNumber; voxel++) {
        if (mask[voxel] > -1) {
            const double refValue = currentRefPtr[voxel] * referenceImage->scl_slope + referenceImage->scl_inter;
            const double warValue = currentWarPtr[voxel] * warpedImage->scl_slope + warpedImage->scl_inter;
            if (refValue == refValue && warValue == warValue) {
#ifdef MRF_USE_SAD
                double common = refValue > warValue ? -1.f : 1.f;
                common *= (refValue - warValue);
#else
                double common = -2.0 * (refValue - warValue);
#endif
                if (jacDetPtr != nullptr)
                    common *= jacDetPtr[voxel];
                else if (localWeightPtr != nullptr)
                    common *= localWeightPtr[voxel];

                common *= adjustedWeight;

                if (spatialGradPtrX[voxel] == spatialGradPtrX[voxel])
                    measureGradPtrX[voxel] += static_cast<DataType>(common * spatialGradPtrX[voxel]);
                if (spatialGradPtrY[voxel] == spatialGradPtrY[voxel])
                    measureGradPtrY[voxel] += static_cast<DataType>(common * spatialGradPtrY[voxel]);
                if (measureGradPtrZ) {
                    if (spatialGradPtrZ[voxel] == spatialGradPtrZ[voxel])
                        measureGradPtrZ[voxel] += static_cast<DataType>(common * spatialGradPtrZ[voxel]);
                }
            }
        }
    }
}
template void reg_getVoxelBasedSsdGradient<float>(const nifti_image*, const nifti_image*, const nifti_image*, nifti_image*, const nifti_image*, const int*, const int, const double, const nifti_image*);
template void reg_getVoxelBasedSsdGradient<double>(const nifti_image*, const nifti_image*, const nifti_image*, nifti_image*, const nifti_image*, const int*, const int, const double, const nifti_image*);
/* *************************************************************** */
void GetVoxelBasedSimilarityMeasureGradient(const nifti_image *referenceImage,
                                            const nifti_image *warpedImage,
                                            const nifti_image *warpedGradient,
                                            nifti_image *voxelBasedGradient,
                                            const nifti_image *jacobianDetImage,
                                            const int *mask,
                                            const int currentTimePoint,
                                            const double timePointWeight,
                                            const nifti_image *localWeightSim) {
    std::visit([&](auto&& refImgDataType) {
        using RefImgDataType = std::decay_t<decltype(refImgDataType)>;
        reg_getVoxelBasedSsdGradient<RefImgDataType>(referenceImage,
                                                     warpedImage,
                                                     warpedGradient,
                                                     voxelBasedGradient,
                                                     jacobianDetImage,
                                                     mask,
                                                     currentTimePoint,
                                                     timePointWeight,
                                                     localWeightSim);
    }, NiftiImage::getFloatingDataType(referenceImage));
}
/* *************************************************************** */
void reg_ssd::GetVoxelBasedSimilarityMeasureGradientFw(int currentTimePoint) {
    ::GetVoxelBasedSimilarityMeasureGradient(this->referenceImage,
                                             this->warpedImage,
                                             this->warpedGradient,
                                             this->voxelBasedGradient,
                                             nullptr, // TODO this->forwardJacDetImagePointer,
                                             this->referenceMask,
                                             currentTimePoint,
                                             this->timePointWeights[currentTimePoint],
                                             this->localWeightSim);
}
/* *************************************************************** */
void reg_ssd::GetVoxelBasedSimilarityMeasureGradientBw(int currentTimePoint) {
    ::GetVoxelBasedSimilarityMeasureGradient(this->floatingImage,
                                             this->warpedImageBw,
                                             this->warpedGradientBw,
                                             this->voxelBasedGradientBw,
                                             nullptr, // TODO this->backwardJacDetImagePointer,
                                             this->floatingMask,
                                             currentTimePoint,
                                             this->timePointWeights[currentTimePoint],
                                             nullptr);
}
/* *************************************************************** */
template <class DataType>
void GetDiscretisedValueSSD_core3D(nifti_image *controlPointGridImage,
                                   float *discretisedValue,
                                   int discretiseRadius,
                                   int discretiseStep,
                                   nifti_image *refImage,
                                   nifti_image *warImage,
                                   int *mask) {
    int cpx, cpy, cpz, t, x, y, z, a, b, c, blockIndex, blockIndex_t, discretisedIndex;
    size_t voxIndex, voxIndex_t;
    const int label_1D_number = (discretiseRadius / discretiseStep) * 2 + 1;
    const int label_2D_number = label_1D_number * label_1D_number;
    int label_nD_number = label_2D_number * label_1D_number;
    //output matrix = discretisedValue (first dimension displacement label, second dim. control point)
    float gridVox[3], imageVox[3];
    float currentValue;
    double currentSum;
    // Define the transformation matrices
    mat44 *grid_vox2mm = &controlPointGridImage->qto_xyz;
    if (controlPointGridImage->sform_code > 0)
        grid_vox2mm = &controlPointGridImage->sto_xyz;
    mat44 *image_mm2vox = &refImage->qto_ijk;
    if (refImage->sform_code > 0)
        image_mm2vox = &refImage->sto_ijk;
    mat44 grid2img_vox = *image_mm2vox * *grid_vox2mm;

    // Compute the block size
    const int blockSize[3] = {
        Ceil<int>(controlPointGridImage->dx / refImage->dx),
        Ceil<int>(controlPointGridImage->dy / refImage->dy),
        Ceil<int>(controlPointGridImage->dz / refImage->dz),
    };
    int voxelBlockNumber = blockSize[0] * blockSize[1] * blockSize[2];
    int voxelBlockNumber_t = blockSize[0] * blockSize[1] * blockSize[2] * refImage->nt;
    int currentControlPoint = 0;

    // Pointers to the input image
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(refImage, 3);
    DataType *refImgPtr = static_cast<DataType*>(refImage->data);
    DataType *warImgPtr = static_cast<DataType*>(warImage->data);

    DataType padding_value = 0;

    int definedValueNumber, idBlock, timeV;

    int threadNumber = 1;
    int tid = 0;
#ifdef _OPENMP
    threadNumber = omp_get_max_threads();
#endif

    // Allocate some static memory
    float **refBlockValue = (float**)malloc(threadNumber * sizeof(float*));
    for (a = 0; a < threadNumber; ++a)
        refBlockValue[a] = (float*)malloc(voxelBlockNumber_t * sizeof(float));

    // Loop over all control points
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(voxelBlockNumber_t, voxelNumber, voxelBlockNumber, label_nD_number, controlPointGridImage, refImage, warImage, grid2img_vox, blockSize, \
    padding_value, refBlockValue, mask, refImgPtr, warImgPtr, discretiseRadius, \
    discretiseStep, discretisedValue) \
    private(cpx, cpy, x, y, z, a, b, c, t, currentControlPoint, gridVox, imageVox, \
    voxIndex, idBlock, blockIndex, definedValueNumber, tid, \
    timeV, voxIndex_t, blockIndex_t, discretisedIndex, currentSum, currentValue)
#endif
    for (cpz = 0; cpz < controlPointGridImage->nz; ++cpz) {
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        gridVox[2] = static_cast<float>(cpz);
        for (cpy = 0; cpy < controlPointGridImage->ny; ++cpy) {
            gridVox[1] = static_cast<float>(cpy);
            for (cpx = 0; cpx < controlPointGridImage->nx; ++cpx) {
                gridVox[0] = static_cast<float>(cpx);
                currentControlPoint = controlPointGridImage->ny * controlPointGridImage->nx * cpz +
                    controlPointGridImage->nx * cpy + cpx;

                // Compute the corresponding image voxel position
                Mat44Mul(grid2img_vox, gridVox, imageVox);
                imageVox[0] = Round<float>(imageVox[0]);
                imageVox[1] = Round<float>(imageVox[1]);
                imageVox[2] = Round<float>(imageVox[2]);

                //INIT
                for (idBlock = 0; idBlock < voxelBlockNumber_t; idBlock++) {
                    refBlockValue[tid][idBlock] = static_cast<float>(padding_value);
                }

                // Extract the block in the reference image
                blockIndex = 0;
                definedValueNumber = 0;
                for (z = int(imageVox[2] - blockSize[2] / 2); z < imageVox[2] + blockSize[2] / 2; ++z) {
                    for (y = int(imageVox[1] - blockSize[1] / 2); y < imageVox[1] + blockSize[1] / 2; ++y) {
                        for (x = int(imageVox[0] - blockSize[0] / 2); x < imageVox[0] + blockSize[0] / 2; ++x) {
                            if (x > -1 && x<refImage->nx && y>-1 && y<refImage->ny && z>-1 && z < refImage->nz) {
                                voxIndex = refImage->ny * refImage->nx * z + refImage->nx * y + x;
                                if (mask[voxIndex] > -1) {
                                    for (timeV = 0; timeV < refImage->nt; ++timeV) {
                                        voxIndex_t = timeV * voxelNumber + voxIndex;
                                        blockIndex_t = timeV * voxelBlockNumber + blockIndex;
                                        refBlockValue[tid][blockIndex_t] = static_cast<float>(refImgPtr[voxIndex_t]);
                                        if (refBlockValue[tid][blockIndex_t] == refBlockValue[tid][blockIndex_t]) {
                                            ++definedValueNumber;
                                        } else refBlockValue[tid][blockIndex_t] = 0;
                                    } // timeV
                                } //inside mask
                            } //inside image
                            blockIndex++;
                        } // x
                    } // y
                } // z
                // Loop over the discretised value
                if (definedValueNumber > 0) {
                    discretisedIndex = 0;
                    for (c = int(imageVox[2] - discretiseRadius); c <= imageVox[2] + discretiseRadius; c += discretiseStep) {
                        for (b = int(imageVox[1] - discretiseRadius); b <= imageVox[1] + discretiseRadius; b += discretiseStep) {
                            for (a = int(imageVox[0] - discretiseRadius); a <= imageVox[0] + discretiseRadius; a += discretiseStep) {

                                blockIndex = 0;
                                currentSum = 0.;
                                definedValueNumber = 0;

                                for (z = c - blockSize[2] / 2; z < c + blockSize[2] / 2; ++z) {
                                    for (y = b - blockSize[1] / 2; y < b + blockSize[1] / 2; ++y) {
                                        for (x = a - blockSize[0] / 2; x < a + blockSize[0] / 2; ++x) {

                                            if (x > -1 && x<warImage->nx && y>-1 && y<warImage->ny && z>-1 && z < warImage->nz) {
                                                voxIndex = warImage->ny * warImage->nx * z + warImage->nx * y + x;
                                                for (t = 0; t < warImage->nt; ++t) {
                                                    voxIndex_t = t * voxelNumber + voxIndex;
                                                    blockIndex_t = t * voxelBlockNumber + blockIndex;
                                                    if (warImgPtr[voxIndex_t] == warImgPtr[voxIndex_t]) {
#ifdef MRF_USE_SAD
                                                        currentValue = fabs(warImgPtr[voxIndex_t] - refBlockValue[tid][blockIndex_t]);
#else
                                                        currentValue = static_cast<float>(Square(warImgPtr[voxIndex_t] - refBlockValue[tid][blockIndex_t]));
#endif
                                                    } else {
#ifdef MRF_USE_SAD
                                                        currentValue = fabs(0 - refBlockValue[tid][blockIndex_t]);
#else
                                                        currentValue = Square(0 - refBlockValue[tid][blockIndex_t]);
#endif
                                                    }

                                                    if (currentValue == currentValue) {
                                                        currentSum -= currentValue;
                                                        ++definedValueNumber;
                                                    }
                                                }
                                            } //inside image
                                            else {
                                                for (t = 0; t < warImage->nt; ++t) {
                                                    blockIndex_t = t * voxelBlockNumber + blockIndex;
#ifdef MRF_USE_SAD
                                                    currentValue = fabs(0 - refBlockValue[tid][blockIndex_t]);
#else
                                                    currentValue = Square(0 - refBlockValue[tid][blockIndex_t]);
#endif
                                                    if (currentValue == currentValue) {
                                                        currentSum -= currentValue;
                                                        ++definedValueNumber;
                                                    }
                                                }
                                            }
                                            blockIndex++;
                                        } // x
                                    } // y
                                } // z
                                discretisedValue[currentControlPoint * label_nD_number + discretisedIndex] = static_cast<float>(currentSum);
                                ++discretisedIndex;
                            } // a
                        } // b
                    } // cc
                } // defined value in the reference block
                ++currentControlPoint;
            } // cpx
        } // cpy
    } // cpz
    for (a = 0; a < threadNumber; ++a)
        free(refBlockValue[a]);
    free(refBlockValue);

    // Deal with the labels that contains NaN values
    for (size_t node = 0; node < NiftiImage::calcVoxelNumber(controlPointGridImage, 3); ++node) {
        int definedValueNumber = 0;
        float *discretisedValuePtr = &discretisedValue[node * label_nD_number];
        float meanValue = 0;
        for (int label = 0; label < label_nD_number; ++label) {
            if (discretisedValuePtr[label] == discretisedValuePtr[label]) {
                ++definedValueNumber;
                meanValue += discretisedValuePtr[label];
            }
        }
        if (definedValueNumber == 0) {
            for (int label = 0; label < label_nD_number; ++label) {
                discretisedValuePtr[label] = 0;
            }
        } else if (definedValueNumber < label_nD_number) {
            // Needs to be altered for efficiency
            int label = 0;
            // Loop over all labels
            int label_x, label2_x, label_y, label2_y, label_z, label2_z, label2;
            float min_distance, current_distance;
            for (label_z = 0; label_z < label_1D_number; ++label_z) {
                for (label_y = 0; label_y < label_1D_number; ++label_y) {
                    for (label_x = 0; label_x < label_1D_number; ++label_x) {
                        // check if the current label is defined
                        if (discretisedValuePtr[label] != discretisedValuePtr[label]) {
                            label2 = 0;
                            min_distance = std::numeric_limits<float>::max();
                            // Loop again over all label to detect the defined values
                            for (label2_z = 0; label2_z < label_1D_number; ++label2_z) {
                                for (label2_y = 0; label2_y < label_1D_number; ++label2_y) {
                                    for (label2_x = 0; label2_x < label_1D_number; ++label2_x) {
                                        // Check if the value is defined
                                        if (discretisedValuePtr[label2] == discretisedValuePtr[label2]) {
                                            // compute the distance between label and label2
                                            current_distance = static_cast<float>(Square(label_x - label2_x) + Square(label_y - label2_y) + Square(label_z - label2_z));
                                            if (current_distance < min_distance) {
                                                min_distance = current_distance;
                                                discretisedValuePtr[label] = discretisedValuePtr[label2];
                                            }
                                        } // Check if label2 is defined
                                        ++label2;
                                    } // x
                                } // y
                            } // z
                        } // check if undefined label
                        ++label;
                    } //x
                } // y
            } // z

        } // node with undefined label
    } // node
}
/* *************************************************************** */
void reg_ssd::GetDiscretisedValue(nifti_image *controlPointGridImage,
                                  float *discretisedValue,
                                  int discretiseRadius,
                                  int discretiseStep) {
    std::visit([&](auto&& refImgDataType) {
        using RefImgDataType = std::decay_t<decltype(refImgDataType)>;
        if (referenceImage->nz > 1) {
            GetDiscretisedValueSSD_core3D<RefImgDataType>(controlPointGridImage,
                                                          discretisedValue,
                                                          discretiseRadius,
                                                          discretiseStep,
                                                          this->referenceImage,
                                                          this->warpedImage,
                                                          this->referenceMask);
        } else {
            NR_FATAL_ERROR("Not implemented in 2D yet");
        }
    }, NiftiImage::getFloatingDataType(this->referenceImage));
}
/* *************************************************************** */
