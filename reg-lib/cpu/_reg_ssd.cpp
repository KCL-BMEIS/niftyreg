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
