/*
 *  _reg_kld.cpp
 *
 *
 *  Created by Marc Modat on 14/05/2012.
 *  Copyright (c) 2012-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_kld.h"

/* *************************************************************** */
reg_kld::reg_kld(): reg_measure() {
    NR_FUNC_CALLED();
}
/* *************************************************************** */
void reg_kld::InitialiseMeasure(nifti_image *refImg,
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

    // Input images are expected to be bounded between 0 and 1 as they are meant to be probabilities
    for (int t = 0; t < this->referenceTimePoints; ++t) {
        if (this->timePointWeights[t] > 0) {
            const float minRef = reg_tools_getMinValue(this->referenceImage, t);
            const float maxRef = reg_tools_getMaxValue(this->referenceImage, t);
            const float minFlo = reg_tools_getMinValue(this->floatingImage, t);
            const float maxFlo = reg_tools_getMaxValue(this->floatingImage, t);
            if (minRef < 0.f || minFlo < 0.f || maxRef > 1.f || maxFlo > 1.f)
                NR_FATAL_ERROR("The input images are expected to be probabilities to use the kld measure");
        }
    }

    for (int i = 0; i < this->referenceTimePoints; ++i)
        NR_DEBUG("Weight for time point " << i << ": " << this->timePointWeights[i]);
    NR_FUNC_CALLED();
}
/* *************************************************************** */
/** @brief Computes and returns the KLD between two input image
 * @param referenceImage First input image to use to compute the metric
 * @param warpedImage Second input image to use to compute the metric
 * @param timePointWeights Array that contains the weight of each time point
 * @param jacobianDetImg Image that contains the Jacobian
 * determinant of a transformation at every voxel position. This
 * image is used to modulate the KLD. The argument is ignored if the
 * pointer is set to nullptr
 * @param mask Array that contains a mask to specify which voxel
 * should be considered
 * @return Returns the computed sum squared difference
 */
template <class DataType>
double reg_getKLDivergence(const nifti_image *referenceImage,
                           const nifti_image *warpedImage,
                           const double *timePointWeights,
                           const nifti_image *jacobianDetImg,
                           const int *mask) {
#ifdef _WIN32
    long voxel;
    const long voxelNumber = (long)NiftiImage::calcVoxelNumber(referenceImage, 3);
#else
    size_t voxel;
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);
#endif
    const DataType *refPtr = static_cast<DataType*>(referenceImage->data);
    const DataType *warPtr = static_cast<DataType*>(warpedImage->data);
    const DataType *jacPtr = jacobianDetImg ? static_cast<DataType*>(jacobianDetImg->data) : nullptr;

    double measure = 0, measureTp = 0, num = 0;

    for (int time = 0; time < referenceImage->nt; ++time) {
        if (timePointWeights[time] > 0) {
            const DataType *currentRefPtr = &refPtr[time * voxelNumber];
            const DataType *currentWarPtr = &warPtr[time * voxelNumber];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(voxelNumber,currentRefPtr, currentWarPtr, mask, jacobianDetImg, jacPtr) \
    reduction(+:measureTp, num)
#endif
            for (voxel = 0; voxel < voxelNumber; ++voxel) {
                if (mask[voxel] > -1) {
                    const double tempRefValue = currentRefPtr[voxel] + 1e-16;
                    const double tempWarValue = currentWarPtr[voxel] + 1e-16;
                    const double tempValue = tempRefValue * log(tempRefValue / tempWarValue);
                    if (tempValue == tempValue && tempValue != std::numeric_limits<double>::infinity()) {
                        const DataType jacValue = jacPtr ? jacPtr[voxel] : 1;
                        measureTp -= tempValue * jacValue;
                        num += jacValue;
                    }
                }
            }
            measure += measureTp * timePointWeights[time] / num;
        }
    }
    return measure;
}
/* *************************************************************** */
double GetSimilarityMeasureValue(const nifti_image *referenceImage,
                                 const nifti_image *warpedImage,
                                 const double *timePointWeights,
                                 const nifti_image *jacobianDetImg,
                                 const int *mask) {
    return std::visit([&](auto&& refImgDataType) {
        using RefImgDataType = std::decay_t<decltype(refImgDataType)>;
        return reg_getKLDivergence<RefImgDataType>(referenceImage,
                                                   warpedImage,
                                                   timePointWeights,
                                                   jacobianDetImg,
                                                   mask);
    }, NiftiImage::getFloatingDataType(referenceImage));
}
/* *************************************************************** */
double reg_kld::GetSimilarityMeasureValueFw() {
    return ::GetSimilarityMeasureValue(this->referenceImage,
                                       this->warpedImage,
                                       this->timePointWeights,
                                       nullptr, // TODO this->forwardJacDetImagePointer,
                                       this->referenceMask);
}
/* *************************************************************** */
double reg_kld::GetSimilarityMeasureValueBw() {
    return ::GetSimilarityMeasureValue(this->floatingImage,
                                       this->warpedImageBw,
                                       this->timePointWeights,
                                       nullptr, // TODO this->backwardJacDetImagePointer,
                                       this->floatingMask);
}
/* *************************************************************** */
/** @brief Compute a voxel based gradient of the sum squared difference.
 * @param referenceImage First input image to use to compute the metric
 * @param warpedImage Second input image to use to compute the metric
 * @param warpedGradient Spatial gradient of the input result image
 * @param measureGradient Output image that will be updated with the
 * value of the KLD gradient
 * @param jacobianDetImg Image that contains the Jacobian
 * determinant of a transformation at every voxel position. This
 * image is used to modulate the KLD. The argument is ignored if the
 * pointer is set to nullptr
 * @param mask Array that contains a mask to specify which voxel
 * should be considered
 * @param currentTimePoint Specified which time point volumes have to be considered
 * @param timePointWeight Weight of the current time point
 */
template <class DataType>
void reg_getKLDivergenceVoxelBasedGradient(const nifti_image *referenceImage,
                                           const nifti_image *warpedImage,
                                           const nifti_image *warpedGradient,
                                           nifti_image *measureGradient,
                                           const nifti_image *jacobianDetImg,
                                           const int *mask,
                                           const int currentTimePoint,
                                           const double timePointWeight) {
#ifdef _WIN32
    long voxel;
    const long voxelNumber = (long)NiftiImage::calcVoxelNumber(referenceImage, 3);
#else
    size_t voxel;
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);
#endif
    const DataType *refImagePtr = static_cast<DataType*>(referenceImage->data);
    const DataType *warImagePtr = static_cast<DataType*>(warpedImage->data);
    const DataType *currentRefPtr = &refImagePtr[currentTimePoint * voxelNumber];
    const DataType *currentWarPtr = &warImagePtr[currentTimePoint * voxelNumber];
    const DataType *jacPtr = jacobianDetImg ? static_cast<DataType*>(jacobianDetImg->data) : nullptr;

    // Create pointers to the spatial gradient of the current warped volume
    const DataType *currentGradPtrX = static_cast<DataType*>(warpedGradient->data);
    const DataType *currentGradPtrY = &currentGradPtrX[voxelNumber];
    const DataType *currentGradPtrZ = referenceImage->nz > 1 ? &currentGradPtrY[voxelNumber] : nullptr;

    // Create pointers to the kld gradient image
    DataType *measureGradPtrX = static_cast<DataType*>(measureGradient->data);
    DataType *measureGradPtrY = &measureGradPtrX[voxelNumber];
    DataType *measureGradPtrZ = referenceImage->nz > 1 ? &measureGradPtrY[voxelNumber] : nullptr;

    // find number of active voxels and correct weight
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
    shared(voxelNumber,currentRefPtr, currentWarPtr, mask, jacobianDetImg, \
    jacPtr, referenceImage, measureGradPtrX, measureGradPtrY, measureGradPtrZ, \
    currentGradPtrX, currentGradPtrY, currentGradPtrZ, adjustedWeight)
#endif
    for (voxel = 0; voxel < voxelNumber; ++voxel) {
        // Check if the current voxel is in the mask
        if (mask[voxel] > -1) {
            // Read referenceImage and warpedImage probabilities and compute the ratio
            const double tempRefValue = currentRefPtr[voxel] + 1e-16;
            const double tempWarValue = currentWarPtr[voxel] + 1e-16;
            double tempValue = (currentRefPtr[voxel] + 1e-16) / (currentWarPtr[voxel] + 1e-16);
            // Check if the intensity ratio is defined and different from zero
            if (tempValue == tempValue &&
                tempValue != std::numeric_limits<double>::infinity() &&
                tempValue > 0) {
                tempValue = (tempRefValue / tempWarValue) * adjustedWeight;

                // Jacobian modulation if the Jacobian determinant image is defined
                if (jacPtr)
                    tempValue *= jacPtr[voxel];

                // Ensure that gradient of the warpedImage image along x-axis is not NaN
                const double tempGradX = currentGradPtrX[voxel];
                if (tempGradX == tempGradX)
                    // Update the gradient along the x-axis
                    measureGradPtrX[voxel] -= static_cast<DataType>(tempValue * tempGradX);

                // Ensure that gradient of the warpedImage image along y-axis is not NaN
                const double tempGradY = currentGradPtrY[voxel];
                if (tempGradY == tempGradY)
                    // Update the gradient along the y-axis
                    measureGradPtrY[voxel] -= static_cast<DataType>(tempValue * tempGradY);

                // Check if the current images are 3D
                if (referenceImage->nz > 1) {
                    // Ensure that gradient of the warpedImage image along z-axis is not NaN
                    const double tempGradZ = currentGradPtrZ[voxel];
                    if (tempGradZ == tempGradZ)
                        // Update the gradient along the z-axis
                        measureGradPtrZ[voxel] -= static_cast<DataType>(tempValue * tempGradZ);
                }
            }
        }
    }
}
/* *************************************************************** */
void GetVoxelBasedSimilarityMeasureGradient(nifti_image *referenceImage,
                                            nifti_image *warpedImage,
                                            nifti_image *warpedGradient,
                                            nifti_image *voxelBasedGradient,
                                            nifti_image *jacobianDetImg,
                                            int *mask,
                                            int currentTimePoint,
                                            double timePointWeight) {
    std::visit([&](auto&& refImgDataType) {
        using RefImgDataType = std::decay_t<decltype(refImgDataType)>;
        reg_getKLDivergenceVoxelBasedGradient<RefImgDataType>(referenceImage,
                                                              warpedImage,
                                                              warpedGradient,
                                                              voxelBasedGradient,
                                                              jacobianDetImg,
                                                              mask,
                                                              currentTimePoint,
                                                              timePointWeight);
    }, NiftiImage::getFloatingDataType(referenceImage));
}
/* *************************************************************** */
void reg_kld::GetVoxelBasedSimilarityMeasureGradientFw(int currentTimePoint) {
    ::GetVoxelBasedSimilarityMeasureGradient(this->referenceImage,
                                             this->warpedImage,
                                             this->warpedGradient,
                                             this->voxelBasedGradient,
                                             nullptr, // TODO this->forwardJacDetImagePointer,
                                             this->referenceMask,
                                             currentTimePoint,
                                             this->timePointWeights[currentTimePoint]);
}
/* *************************************************************** */
void reg_kld::GetVoxelBasedSimilarityMeasureGradientBw(int currentTimePoint) {
    ::GetVoxelBasedSimilarityMeasureGradient(this->floatingImage,
                                             this->warpedImageBw,
                                             this->warpedGradientBw,
                                             this->voxelBasedGradientBw,
                                             nullptr, // TODO this->backwardJacDetImagePointer,
                                             this->floatingMask,
                                             currentTimePoint,
                                             this->timePointWeights[currentTimePoint]);
}
/* *************************************************************** */
