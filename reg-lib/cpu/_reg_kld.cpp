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
#ifndef NDEBUG
    reg_print_msg_debug("reg_kld constructor called");
#endif
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
    if (this->referenceImage->nt != this->floatingImage->nt) {
        reg_print_fct_error("reg_kld::InitialiseMeasure");
        reg_print_msg_error("This number of time point should be the same for both input images");
        reg_exit();
    }
    // Input images are expected to be bounded between 0 and 1 as they
    // are meant to be probabilities
    for (int t = 0; t < this->referenceImage->nt; ++t) {
        if (this->timePointWeight[t] > 0) {
            const float minRef = reg_tools_getMinValue(this->referenceImage, t);
            const float maxRef = reg_tools_getMaxValue(this->referenceImage, t);
            const float minFlo = reg_tools_getMinValue(this->floatingImage, t);
            const float maxFlo = reg_tools_getMaxValue(this->floatingImage, t);
            if (minRef < 0.f || minFlo < 0.f || maxRef > 1.f || maxFlo > 1.f) {
                reg_print_fct_error("reg_kld::InitialiseMeasure");
                reg_print_msg_error("The input images are expected to be probabilities to use the kld measure");
                reg_exit();
            }
        }
    }
#ifndef NDEBUG
    char text[255];
    reg_print_msg_debug("reg_kld::InitialiseMeasure()");
    for (int i = 0; i < this->referenceImage->nt; ++i) {
        sprintf(text, "Weight for timepoint %i: %f", i, this->timePointWeight[i]);
        reg_print_msg_debug(text);
    }
#endif
}
/* *************************************************************** */
template <class DataType>
double reg_getKLDivergence(const nifti_image *referenceImage,
                           const nifti_image *warpedImage,
                           const double *timePointWeight,
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
    const DataType *jacPtr = nullptr;
    if (jacobianDetImg != nullptr)
        jacPtr = static_cast<DataType*>(jacobianDetImg->data);

    double measure = 0, measureTp = 0, num = 0, tempRefValue, tempWarValue, tempValue;

    for (int time = 0; time < referenceImage->nt; ++time) {
        if (timePointWeight[time] > 0) {
            const DataType *currentRefPtr = &refPtr[time * voxelNumber];
            const DataType *currentWarPtr = &warPtr[time * voxelNumber];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(voxelNumber,currentRefPtr, currentWarPtr, mask, jacobianDetImg, jacPtr) \
    private(tempRefValue, tempWarValue, tempValue) \
    reduction(+:measureTp, num)
#endif
            for (voxel = 0; voxel < voxelNumber; ++voxel) {
                if (mask[voxel] > -1) {
                    tempRefValue = currentRefPtr[voxel] + 1e-16;
                    tempWarValue = currentWarPtr[voxel] + 1e-16;
                    tempValue = tempRefValue * log(tempRefValue / tempWarValue);
                    if (tempValue == tempValue &&
                        tempValue != std::numeric_limits<double>::infinity()) {
                        if (jacobianDetImg == nullptr) {
                            measureTp -= tempValue;
                            num++;
                        } else {
                            measureTp -= tempValue * jacPtr[voxel];
                            num += jacPtr[voxel];
                        }
                    }
                }
            }
            measure += measureTp * timePointWeight[time] / num;
        }
    }
    return measure;
}
/* *************************************************************** */
double GetSimilarityMeasureValue(const nifti_image *referenceImage,
                                 const nifti_image *warpedImage,
                                 const double *timePointWeight,
                                 const nifti_image *jacobianDetImg,
                                 const int *mask) {
    return std::visit([&](auto&& refImgDataType) {
        using RefImgDataType = std::decay_t<decltype(refImgDataType)>;
        return reg_getKLDivergence<RefImgDataType>(referenceImage,
                                                   warpedImage,
                                                   timePointWeight,
                                                   jacobianDetImg,
                                                   mask);
    }, NiftiImage::getFloatingDataType(referenceImage));
}
/* *************************************************************** */
double reg_kld::GetSimilarityMeasureValueFw() {
    return ::GetSimilarityMeasureValue(this->referenceImage,
                                       this->warpedImage,
                                       this->timePointWeight,
                                       nullptr, // TODO this->forwardJacDetImagePointer,
                                       this->referenceMask);
}
/* *************************************************************** */
double reg_kld::GetSimilarityMeasureValueBw() {
    return ::GetSimilarityMeasureValue(this->floatingImage,
                                       this->warpedImageBw,
                                       this->timePointWeight,
                                       nullptr, // TODO this->backwardJacDetImagePointer,
                                       this->floatingMask);
}
/* *************************************************************** */
template <class DataType>
void reg_getKLDivergenceVoxelBasedGradient(nifti_image *referenceImage,
                                           nifti_image *warpedImage,
                                           nifti_image *warpedImageGradient,
                                           nifti_image *measureGradient,
                                           nifti_image *jacobianDetImg,
                                           int *mask,
                                           int currentTimepoint,
                                           double timepointWeight) {
#ifdef _WIN32
    long voxel;
    const long voxelNumber = (long)NiftiImage::calcVoxelNumber(referenceImage, 3);
#else
    size_t voxel;
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);
#endif

    DataType *refImagePtr = static_cast<DataType*>(referenceImage->data);
    DataType *warImagePtr = static_cast<DataType*>(warpedImage->data);
    DataType *currentRefPtr = &refImagePtr[currentTimepoint * voxelNumber];
    DataType *currentWarPtr = &warImagePtr[currentTimepoint * voxelNumber];
    int *maskPtr = nullptr;
    bool MrClean = false;
    if (mask == nullptr) {
        maskPtr = (int*)calloc(voxelNumber, sizeof(int));
        MrClean = true;
    } else maskPtr = &mask[0];

    DataType *jacPtr = nullptr;
    if (jacobianDetImg != nullptr)
        jacPtr = static_cast<DataType*>(jacobianDetImg->data);
    double tempValue, tempGradX, tempGradY, tempGradZ, tempRefValue, tempWarValue;

    // Create pointers to the spatial gradient of the current warped volume
    DataType *currentGradPtrX = static_cast<DataType*>(warpedImageGradient->data);
    DataType *currentGradPtrY = &currentGradPtrX[voxelNumber];
    DataType *currentGradPtrZ = nullptr;
    if (referenceImage->nz > 1)
        currentGradPtrZ = &currentGradPtrY[voxelNumber];

    // Create pointers to the kld gradient image
    DataType *measureGradPtrX = static_cast<DataType*>(measureGradient->data);
    DataType *measureGradPtrY = &measureGradPtrX[voxelNumber];
    DataType *measureGradPtrZ = nullptr;
    if (referenceImage->nz > 1)
        measureGradPtrZ = &measureGradPtrY[voxelNumber];

    // find number of active voxels and correct weight
    double activeVoxel_num = 0;
    for (voxel = 0; voxel < voxelNumber; voxel++) {
        if (mask[voxel] > -1) {
            if (currentRefPtr[voxel] == currentRefPtr[voxel] && currentWarPtr[voxel] == currentWarPtr[voxel])
                activeVoxel_num += 1.0;
        }
    }
    double adjusted_weight = timepointWeight / activeVoxel_num;

#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(voxelNumber,currentRefPtr, currentWarPtr, \
    maskPtr, jacobianDetImg, jacPtr, referenceImage, \
    measureGradPtrX, measureGradPtrY, measureGradPtrZ, \
    currentGradPtrX, currentGradPtrY, currentGradPtrZ, adjusted_weight) \
    private(tempValue, tempGradX, tempGradY, tempGradZ, \
    tempRefValue, tempWarValue)
#endif
    for (voxel = 0; voxel < voxelNumber; ++voxel) {
        // Check if the current voxel is in the mask
        if (maskPtr[voxel] > -1) {
            // Read referenceImage and warpedImage probabilities and compute the ratio
            tempRefValue = currentRefPtr[voxel] + 1e-16;
            tempWarValue = currentWarPtr[voxel] + 1e-16;
            tempValue = (currentRefPtr[voxel] + 1e-16) / (currentWarPtr[voxel] + 1e-16);
            // Check if the intensity ratio is defined and different from zero
            if (tempValue == tempValue &&
                tempValue != std::numeric_limits<double>::infinity() &&
                tempValue > 0) {
                tempValue = tempRefValue / tempWarValue;
                tempValue *= adjusted_weight;

                // Jacobian modulation if the Jacobian determinant image is defined
                if (jacobianDetImg != nullptr)
                    tempValue *= jacPtr[voxel];

                // Ensure that gradient of the warpedImage image along x-axis is not NaN
                tempGradX = currentGradPtrX[voxel];
                if (tempGradX == tempGradX)
                    // Update the gradient along the x-axis
                    measureGradPtrX[voxel] -= (DataType)(tempValue * tempGradX);

                // Ensure that gradient of the warpedImage image along y-axis is not NaN
                tempGradY = currentGradPtrY[voxel];
                if (tempGradY == tempGradY)
                    // Update the gradient along the y-axis
                    measureGradPtrY[voxel] -= (DataType)(tempValue * tempGradY);

                // Check if the current images are 3D
                if (referenceImage->nz > 1) {
                    // Ensure that gradient of the warpedImage image along z-axis is not NaN
                    tempGradZ = currentGradPtrZ[voxel];
                    if (tempGradZ == tempGradZ)
                        // Update the gradient along the z-axis
                        measureGradPtrZ[voxel] -= (DataType)(tempValue * tempGradZ);
                }
            }
        }
    }
    if (MrClean) free(maskPtr);
}
/* *************************************************************** */
void reg_kld::GetVoxelBasedSimilarityMeasureGradient(int currentTimepoint) {
    // Check if the specified time point exists and is active
    reg_measure::GetVoxelBasedSimilarityMeasureGradient(currentTimepoint);
    if (this->timePointWeight[currentTimepoint] == 0)
        return;

    // Check if all required input images are of the same data type
    int dtype = this->referenceImage->datatype;
    if (this->warpedImage->datatype != dtype ||
        this->warpedGradient->datatype != dtype ||
        this->voxelBasedGradient->datatype != dtype) {
        reg_print_fct_error("reg_kld::GetVoxelBasedSimilarityMeasureGradient");
        reg_print_msg_error("Input images are expected to be of the same type");
        reg_exit();
    }
    // Compute the gradient of the kld for the forward transformation
    switch (dtype) {
    case NIFTI_TYPE_FLOAT32:
        reg_getKLDivergenceVoxelBasedGradient<float>(this->referenceImage,
                                                     this->warpedImage,
                                                     this->warpedGradient,
                                                     this->voxelBasedGradient,
                                                     nullptr, // TODO this->forwardJacDetImagePointer,
                                                     this->referenceMask,
                                                     currentTimepoint,
                                                     this->timePointWeight[currentTimepoint]);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_getKLDivergenceVoxelBasedGradient<double>(this->referenceImage,
                                                      this->warpedImage,
                                                      this->warpedGradient,
                                                      this->voxelBasedGradient,
                                                      nullptr, // TODO this->forwardJacDetImagePointer,
                                                      this->referenceMask,
                                                      currentTimepoint,
                                                      this->timePointWeight[currentTimepoint]);
        break;
    default:
        reg_print_fct_error("reg_kld::GetVoxelBasedSimilarityMeasureGradient");
        reg_print_msg_error("Unsupported datatype");
        reg_exit();
    }
    // Compute the gradient of the kld for the backward transformation
    if (this->isSymmetric) {
        dtype = this->floatingImage->datatype;
        if (this->warpedImageBw->datatype != dtype ||
            this->warpedGradientBw->datatype != dtype ||
            this->voxelBasedGradientBw->datatype != dtype) {
            reg_print_fct_error("reg_kld::GetVoxelBasedSimilarityMeasureGradient");
            reg_print_msg_error("Input images are expected to be of the same type");
            reg_exit();
        }
        // Compute the gradient of the nmi for the backward transformation
        switch (dtype) {
        case NIFTI_TYPE_FLOAT32:
            reg_getKLDivergenceVoxelBasedGradient<float>(this->floatingImage,
                                                         this->warpedImageBw,
                                                         this->warpedGradientBw,
                                                         this->voxelBasedGradientBw,
                                                         nullptr, // TODO this->backwardJacDetImagePointer,
                                                         this->floatingMask,
                                                         currentTimepoint,
                                                         this->timePointWeight[currentTimepoint]);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_getKLDivergenceVoxelBasedGradient<double>(this->floatingImage,
                                                          this->warpedImageBw,
                                                          this->warpedGradientBw,
                                                          this->voxelBasedGradientBw,
                                                          nullptr, // TODO this->backwardJacDetImagePointer,
                                                          this->floatingMask,
                                                          currentTimepoint,
                                                          this->timePointWeight[currentTimepoint]);
            break;
        default:
            reg_print_fct_error("reg_kld::GetVoxelBasedSimilarityMeasureGradient");
            reg_print_msg_error("Unsupported datatype");
            reg_exit();
        }
    }
}
/* *************************************************************** */
