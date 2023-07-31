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

// #define USE_LOG_SSD
// #define MRF_USE_SAD

/* *************************************************************** */
reg_ssd::reg_ssd(): reg_measure() {
    memset(this->normaliseTimePoint, 0, 255 * sizeof(bool));
#ifndef NDEBUG
    reg_print_msg_debug("reg_ssd constructor called");
#endif
}
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
    if (this->referenceImage->nt != this->floatingImage->nt) {
        reg_print_fct_error("reg_ssd::InitialiseMeasure");
        reg_print_msg_error("This number of time point should be the same for both input images");
        reg_exit();
    }
    // Input images are normalised between 0 and 1
    for (int i = 0; i < this->referenceImage->nt; ++i) {
        if (this->timePointWeight[i] > 0 && normaliseTimePoint[i]) {
            //sets max value over both images to be 1 and min value over both images to be 0
            //scales values such that identical values in the images are still identical after scaling
            float maxF = reg_tools_getMaxValue(this->floatingImage, i);
            float maxR = reg_tools_getMaxValue(this->referenceImage, i);
            float minF = reg_tools_getMinValue(this->floatingImage, i);
            float minR = reg_tools_getMinValue(this->referenceImage, i);
            float maxFR = fmax(maxF, maxR);
            float minFR = fmin(minF, minR);
            float rangeFR = maxFR - minFR;
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
    reg_print_msg_warn("SAD is used instead of SSD");
#endif
#ifndef NDEBUG
    char text[255];
    reg_print_msg_debug("reg_ssd::InitialiseMeasure()");
    for (int i = 0; i < this->referenceImage->nt; ++i) {
        sprintf(text, "Weight for timepoint %i: %f", i, this->timePointWeight[i]);
        reg_print_msg_debug(text);
    }
    sprintf(text, "Normalize time point:");
    for (int i = 0; i < this->referenceImage->nt; ++i)
        if (this->normaliseTimePoint[i])
            sprintf(text, "%s %i", text, i);
    reg_print_msg_debug(text);
#endif
}
/* *************************************************************** */
void reg_ssd::SetNormaliseTimepoint(int timepoint, bool normalise) {
    this->normaliseTimePoint[timepoint] = normalise;
}
/* *************************************************************** */
template<class DataType>
double reg_getSsdValue(const nifti_image *referenceImage,
                       const nifti_image *warpedImage,
                       const double *timePointWeight,
                       const nifti_image *jacobianDetImage,
                       const int *mask,
                       float *currentValue,
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
    const DataType *jacDetPtr = nullptr;
    if (jacobianDetImage != nullptr)
        jacDetPtr = static_cast<DataType*>(jacobianDetImage->data);
    // Create a pointer to the local weight image if defined
    const DataType *localWeightPtr = nullptr;
    if (localWeightSim != nullptr)
        localWeightPtr = static_cast<DataType*>(localWeightSim->data);

    double ssdGlobal = 0;
    double refValue, warValue, diff;

    // Loop over the different time points
    for (int time = 0; time < referenceImage->nt; ++time) {
        if (timePointWeight[time] > 0) {
            // Create pointers to the current time point of the reference and warped images
            const DataType *currentRefPtr = &referencePtr[time * voxelNumber];
            const DataType *currentWarPtr = &warpedPtr[time * voxelNumber];

            double ssdLocal = 0, n = 0;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(referenceImage, warpedImage, currentRefPtr, currentWarPtr, mask, \
    jacobianDetImage, jacDetPtr, voxelNumber, localWeightPtr) \
    private(refValue, warValue, diff) \
    reduction(+:ssdLocal) \
    reduction(+:n)
#endif
            for (voxel = 0; voxel < voxelNumber; ++voxel) {
                // Check if the current voxel belongs to the mask
                if (mask[voxel] > -1) {
                    // Ensure that both ref and warped values are defined
                    refValue = (double)(currentRefPtr[voxel] * referenceImage->scl_slope + referenceImage->scl_inter);
                    warValue = (double)(currentWarPtr[voxel] * warpedImage->scl_slope + warpedImage->scl_inter);

                    if (refValue == refValue && warValue == warValue) {
#ifdef MRF_USE_SAD
                        diff = fabs(refValue - warValue);
#else
                        diff = reg_pow2(refValue - warValue);
#endif
                        // Jacobian determinant modulation of the ssd if required
                        if (jacDetPtr != nullptr) {
                            ssdLocal += diff * jacDetPtr[voxel];
                            n += jacDetPtr[voxel];
                        } else if (localWeightPtr != nullptr) {
                            ssdLocal += diff * localWeightPtr[voxel];
                            n += localWeightPtr[voxel];
                        } else {
                            ssdLocal += diff;
                            n += 1.0;
                        }
                    }
                }
            }

            ssdLocal *= timePointWeight[time];
            currentValue[time] = static_cast<float>(-ssdLocal);
            ssdGlobal -= ssdLocal / n;
        }
    }
    return ssdGlobal;
}
template double reg_getSsdValue<float>(const nifti_image*, const nifti_image*, const double*, const nifti_image*, const int*, float*, const nifti_image*);
template double reg_getSsdValue<double>(const nifti_image*, const nifti_image*, const double*, const nifti_image*, const int*, float*, const nifti_image*);
/* *************************************************************** */
double GetSimilarityMeasureValue(const nifti_image *referenceImage,
                                 const nifti_image *warpedImage,
                                 const double *timePointWeight,
                                 const nifti_image *jacobianDetImage,
                                 const int *mask,
                                 float *currentValue,
                                 const nifti_image *localWeightSim) {
    return std::visit([&](auto&& refImgDataType) {
        using RefImgDataType = std::decay_t<decltype(refImgDataType)>;
        return reg_getSsdValue<RefImgDataType>(referenceImage,
                                               warpedImage,
                                               timePointWeight,
                                               jacobianDetImage,
                                               mask,
                                               currentValue,
                                               localWeightSim);
    }, NiftiImage::getFloatingDataType(referenceImage));
}
/* *************************************************************** */
double reg_ssd::GetSimilarityMeasureValueFw() {
    return ::GetSimilarityMeasureValue(this->referenceImage,
                                       this->warpedImage,
                                       this->timePointWeight,
                                       nullptr, // TODO this->forwardJacDetImagePointer,
                                       this->referenceMask,
                                       this->currentValue,
                                       this->localWeightSim);
}
/* *************************************************************** */
double reg_ssd::GetSimilarityMeasureValueBw() {
    return ::GetSimilarityMeasureValue(this->floatingImage,
                                       this->warpedImageBw,
                                       this->timePointWeight,
                                       nullptr, // TODO this->backwardJacDetImagePointer,
                                       this->floatingMask,
                                       this->currentValue,
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
                                  const int& currentTimepoint,
                                  const double& timepointWeight,
                                  const nifti_image *localWeightSim) {
    if (currentTimepoint < 0 || currentTimepoint >= referenceImage->nt) {
        reg_print_fct_error("reg_getVoxelBasedSSDGradient");
        reg_print_msg_error("The specified active timepoint is not defined in the ref/war images");
        reg_exit();
    }
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
    const DataType *currentRefPtr = &refImagePtr[currentTimepoint * voxelNumber];
    const DataType *warImagePtr = static_cast<DataType*>(warpedImage->data);
    const DataType *currentWarPtr = &warImagePtr[currentTimepoint * voxelNumber];

    // Pointers to the spatial gradient of the warped image
    const DataType *spatialGradPtrX = static_cast<DataType*>(warpedGradient->data);
    const DataType *spatialGradPtrY = &spatialGradPtrX[voxelNumber];
    const DataType *spatialGradPtrZ = nullptr;
    if (referenceImage->nz > 1)
        spatialGradPtrZ = &spatialGradPtrY[voxelNumber];

    // Pointers to the measure of similarity gradient
    DataType *measureGradPtrX = static_cast<DataType*>(measureGradientImage->data);
    DataType *measureGradPtrY = &measureGradPtrX[voxelNumber];
    DataType *measureGradPtrZ = nullptr;
    if (referenceImage->nz > 1)
        measureGradPtrZ = &measureGradPtrY[voxelNumber];

    // Create a pointer to the Jacobian determinant values if defined
    const DataType *jacDetPtr = nullptr;
    if (jacobianDetImage != nullptr)
        jacDetPtr = static_cast<DataType*>(jacobianDetImage->data);
    // Create a pointer to the local weight image if defined
    const DataType *localWeightPtr = nullptr;
    if (localWeightSim != nullptr)
        localWeightPtr = static_cast<DataType*>(localWeightSim->data);

    // find number of active voxels and correct weight
    double activeVoxel_num = 0;
    for (voxel = 0; voxel < voxelNumber; voxel++) {
        if (mask[voxel] > -1) {
            if (currentRefPtr[voxel] == currentRefPtr[voxel] && currentWarPtr[voxel] == currentWarPtr[voxel])
                activeVoxel_num += 1.0;
        }
    }
    double adjustedWeight = timepointWeight / activeVoxel_num;

    double refValue, warValue, common;

#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(referenceImage, warpedImage, currentRefPtr, currentWarPtr, \
    mask, jacDetPtr, spatialGradPtrX, spatialGradPtrY, spatialGradPtrZ, \
    measureGradPtrX, measureGradPtrY, measureGradPtrZ, voxelNumber, \
    localWeightPtr, adjustedWeight) \
    private(refValue, warValue, common)
#endif
    for (voxel = 0; voxel < voxelNumber; voxel++) {
        if (mask[voxel] > -1) {
            refValue = currentRefPtr[voxel] * referenceImage->scl_slope + referenceImage->scl_inter;
            warValue = currentWarPtr[voxel] * warpedImage->scl_slope + warpedImage->scl_inter;
            if (refValue == refValue && warValue == warValue) {
#ifdef MRF_USE_SAD
                common = refValue > warValue ? -1.f : 1.f;
                common *= (refValue - warValue);
#else
                common = -2.0 * (refValue - warValue);
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

                if (measureGradPtrZ != nullptr) {
                    if (spatialGradPtrZ[voxel] == spatialGradPtrZ[voxel])
                        measureGradPtrZ[voxel] += static_cast<DataType>(common * spatialGradPtrZ[voxel]);
                }
            }
        }
    }
}
template void reg_getVoxelBasedSsdGradient<float>(const nifti_image*, const nifti_image*, const nifti_image*, nifti_image*, const nifti_image*, const int*, const int&, const double&, const nifti_image*);
template void reg_getVoxelBasedSsdGradient<double>(const nifti_image*, const nifti_image*, const nifti_image*, nifti_image*, const nifti_image*, const int*, const int&, const double&, const nifti_image*);
/* *************************************************************** */
void reg_ssd::GetVoxelBasedSimilarityMeasureGradient(int currentTimepoint) {
    // Check if the specified time point exists and is active
    reg_measure::GetVoxelBasedSimilarityMeasureGradient(currentTimepoint);
    if (this->timePointWeight[currentTimepoint] == 0)
        return;

    // Check if all required input images are of the same data type
    int dtype = this->referenceImage->datatype;
    if (this->warpedImage->datatype != dtype ||
        this->warpedGradient->datatype != dtype ||
        this->voxelBasedGradient->datatype != dtype) {
        reg_print_fct_error("reg_ssd::GetVoxelBasedSimilarityMeasureGradient");
        reg_print_msg_error("Input images are expected to be of the same type");
        reg_exit();
    }
    // Compute the gradient of the ssd for the forward transformation
    switch (dtype) {
    case NIFTI_TYPE_FLOAT32:
        reg_getVoxelBasedSsdGradient<float>(this->referenceImage,
                                            this->warpedImage,
                                            this->warpedGradient,
                                            this->voxelBasedGradient,
                                            nullptr, // TODO this->forwardJacDetImagePointer,
                                            this->referenceMask,
                                            currentTimepoint,
                                            this->timePointWeight[currentTimepoint],
                                            this->localWeightSim);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_getVoxelBasedSsdGradient<double>(this->referenceImage,
                                             this->warpedImage,
                                             this->warpedGradient,
                                             this->voxelBasedGradient,
                                             nullptr, // TODO this->forwardJacDetImagePointer,
                                             this->referenceMask,
                                             currentTimepoint,
                                             this->timePointWeight[currentTimepoint],
                                             this->localWeightSim);
        break;
    default:
        reg_print_fct_error("reg_ssd::GetVoxelBasedSimilarityMeasureGradient");
        reg_print_msg_error("Unsupported datatype");
        reg_exit();
    }
    // Compute the gradient of the ssd for the backward transformation
    if (this->isSymmetric) {
        dtype = this->floatingImage->datatype;
        if (this->warpedImageBw->datatype != dtype ||
            this->warpedGradientBw->datatype != dtype ||
            this->voxelBasedGradientBw->datatype != dtype) {
            reg_print_fct_error("reg_ssd::GetVoxelBasedSimilarityMeasureGradient");
            reg_print_msg_error("Input images are expected to be of the same type");
            reg_exit();
        }
        // Compute the gradient of the nmi for the backward transformation
        switch (dtype) {
        case NIFTI_TYPE_FLOAT32:
            reg_getVoxelBasedSsdGradient<float>(this->floatingImage,
                                                this->warpedImageBw,
                                                this->warpedGradientBw,
                                                this->voxelBasedGradientBw,
                                                nullptr, // TODO this->backwardJacDetImagePointer,
                                                this->floatingMask,
                                                currentTimepoint,
                                                this->timePointWeight[currentTimepoint],
                                                nullptr);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_getVoxelBasedSsdGradient<double>(this->floatingImage,
                                                 this->warpedImageBw,
                                                 this->warpedGradientBw,
                                                 this->voxelBasedGradientBw,
                                                 nullptr, // TODO this->backwardJacDetImagePointer,
                                                 this->floatingMask,
                                                 currentTimepoint,
                                                 this->timePointWeight[currentTimepoint],
                                                 nullptr);
            break;
        default:
            reg_print_fct_error("reg_ssd::GetVoxelBasedSimilarityMeasureGradient");
            reg_print_msg_error("Unsupported datatype");
            reg_exit();
        }
    }
}
/* *************************************************************** */
template <class DataType>
void GetDiscretisedValueSSD_core3D(nifti_image *controlPointGridImage,
                                   float *discretisedValue,
                                   int discretise_radius,
                                   int discretise_step,
                                   nifti_image *refImage,
                                   nifti_image *warImage,
                                   int *mask) {
    int cpx, cpy, cpz, t, x, y, z, a, b, c, blockIndex, discretisedIndex;
    size_t voxIndex, voxIndex_t;
    int label_1D_number = (discretise_radius / discretise_step) * 2 + 1;
    int label_2D_number = label_1D_number * label_1D_number;
    int label_nD_number = label_2D_number * label_1D_number;
    //output matrix = discretisedValue (first dimension displacement label, second dim. control point)
    float gridVox[3], imageVox[3];
    float currentValue;
    // Define the transformation matrices
    mat44 *grid_vox2mm = &controlPointGridImage->qto_xyz;
    if (controlPointGridImage->sform_code > 0)
        grid_vox2mm = &controlPointGridImage->sto_xyz;
    mat44 *image_mm2vox = &refImage->qto_ijk;
    if (refImage->sform_code > 0)
        image_mm2vox = &refImage->sto_ijk;
    mat44 grid2img_vox = reg_mat44_mul(image_mm2vox, grid_vox2mm);

    // Compute the block size
    int blockSize[3] = {
        (int)reg_ceil(controlPointGridImage->dx / refImage->dx),
        (int)reg_ceil(controlPointGridImage->dy / refImage->dy),
        (int)reg_ceil(controlPointGridImage->dz / refImage->dz),
    };
    int voxelBlockNumber = blockSize[0] * blockSize[1] * blockSize[2] * refImage->nt;
    int currentControlPoint = 0;

    // Allocate some static memory
    float *refBlockValue = (float*)malloc(voxelBlockNumber * sizeof(float));

    // Pointers to the input image
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(refImage, 3);
    DataType *refImgPtr = static_cast<DataType*>(refImage->data);
    DataType *warImgPtr = static_cast<DataType*>(warImage->data);

    // Create a padded version of the warped image to avoid boundary condition check
    int warPaddedOffset[3] = {
        discretise_radius + blockSize[0],
        discretise_radius + blockSize[1],
        discretise_radius + blockSize[2],
    };
    int warPaddedDim[4] = {
        warImage->nx + 2 * warPaddedOffset[0] + blockSize[0],
        warImage->ny + 2 * warPaddedOffset[1] + blockSize[1],
        warImage->nz + 2 * warPaddedOffset[2] + blockSize[2],
        warImage->nt
    };

    //DataType padding_value = std::numeric_limits<DataType>::quiet_NaN();
    DataType padding_value = 0;

    size_t warPaddedVoxelNumber = (size_t)warPaddedDim[0] *
        warPaddedDim[1] * warPaddedDim[2];
    DataType *paddedWarImgPtr = (DataType*)calloc(warPaddedVoxelNumber * warPaddedDim[3], sizeof(DataType));
    for (voxIndex = 0; voxIndex < warPaddedVoxelNumber * warPaddedDim[3]; ++voxIndex)
        paddedWarImgPtr[voxIndex] = padding_value;
    voxIndex = 0;
    voxIndex_t = 0;
    for (t = 0; t < warImage->nt; ++t) {
        for (z = warPaddedOffset[2]; z < warPaddedDim[2] - warPaddedOffset[2] - blockSize[2]; ++z) {
            for (y = warPaddedOffset[1]; y < warPaddedDim[1] - warPaddedOffset[1] - blockSize[1]; ++y) {
                voxIndex = t * warPaddedVoxelNumber + (z * warPaddedDim[1] + y) * warPaddedDim[0] + warPaddedOffset[0];
                for (x = warPaddedOffset[0]; x < warPaddedDim[0] - warPaddedOffset[0] - blockSize[0]; ++x) {
                    paddedWarImgPtr[voxIndex] = warImgPtr[voxIndex_t];
                    ++voxIndex;
                    ++voxIndex_t;
                }
            }
        }
    }

    int definedValueNumber;

    // Loop over all control points
    for (cpz = 1; cpz < controlPointGridImage->nz - 1; ++cpz) {
        gridVox[2] = cpz;
        for (cpy = 1; cpy < controlPointGridImage->ny - 1; ++cpy) {
            gridVox[1] = cpy;
            currentControlPoint = (cpz * controlPointGridImage->ny + cpy) * controlPointGridImage->nx + 1;
            for (cpx = 1; cpx < controlPointGridImage->nx - 1; ++cpx) {
                gridVox[0] = cpx;
                // Compute the corresponding image voxel position
                reg_mat44_mul(&grid2img_vox, gridVox, imageVox);
                imageVox[0] = reg_round(imageVox[0]);
                imageVox[1] = reg_round(imageVox[1]);
                imageVox[2] = reg_round(imageVox[2]);

                // Extract the block in the reference image
                blockIndex = 0;
                definedValueNumber = 0;
                for (z = imageVox[2] - blockSize[2] / 2; z < imageVox[2] + blockSize[2] / 2; ++z) {
                    for (y = imageVox[1] - blockSize[1] / 2; y < imageVox[1] + blockSize[1] / 2; ++y) {
                        for (x = imageVox[0] - blockSize[0] / 2; x < imageVox[0] + blockSize[0] / 2; ++x) {
                            if (x > -1 && x<refImage->nx && y>-1 && y<refImage->ny && z>-1 && z < refImage->nz) {
                                voxIndex = (z * refImage->ny + y) * refImage->nx + x;
                                if (mask[voxIndex] > -1) {
                                    for (t = 0; t < refImage->nt; ++t) {
                                        voxIndex_t = t * voxelNumber + voxIndex;
                                        refBlockValue[blockIndex] = refImgPtr[voxIndex_t];
                                        if (refBlockValue[blockIndex] == refBlockValue[blockIndex])
                                            ++definedValueNumber;
                                        blockIndex++;
                                    } //t
                                } else {
                                    for (t = 0; t < refImage->nt; ++t) {
                                        refBlockValue[blockIndex] = padding_value;
                                        blockIndex++;
                                    } // t
                                }
                            } else {
                                for (t = 0; t < refImage->nt; ++t) {
                                    refBlockValue[blockIndex] = padding_value;
                                    blockIndex++;
                                } // t
                            } // mask
                        } // x
                    } // y
                } // z
                // Loop over the discretised value
                if (definedValueNumber > 0) {

                    DataType warpedValue;
                    int paddedImageVox[3] = {
                        static_cast<int>(imageVox[0] + warPaddedOffset[0]),
                        static_cast<int>(imageVox[1] + warPaddedOffset[1]),
                        static_cast<int>(imageVox[2] + warPaddedOffset[2])
                    };
                    int cc;
                    double currentSum;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(label_1D_number, label_2D_number, label_nD_number, discretise_step, discretise_radius, \
    paddedImageVox, blockSize, warPaddedDim, paddedWarImgPtr, refBlockValue, warPaddedVoxelNumber, \
    discretisedValue, currentControlPoint, voxelBlockNumber) \
    private(a, b, c, cc, x, y, z, t, discretisedIndex, blockIndex, \
    currentValue, warpedValue, voxIndex, voxIndex_t, definedValueNumber, currentSum)
#endif
                    for (cc = 0; cc < label_1D_number; ++cc) {
                        discretisedIndex = cc * label_2D_number;
                        c = paddedImageVox[2] - discretise_radius + cc * discretise_step;
                        for (b = paddedImageVox[1] - discretise_radius; b <= paddedImageVox[1] + discretise_radius; b += discretise_step) {
                            for (a = paddedImageVox[0] - discretise_radius; a <= paddedImageVox[0] + discretise_radius; a += discretise_step) {

                                blockIndex = 0;
                                currentSum = 0.;
                                definedValueNumber = 0;

                                for (z = c - blockSize[2] / 2; z < c + blockSize[2] / 2; ++z) {
                                    for (y = b - blockSize[1] / 2; y < b + blockSize[1] / 2; ++y) {
                                        for (x = a - blockSize[0] / 2; x < a + blockSize[0] / 2; ++x) {
                                            voxIndex = (z * warPaddedDim[1] + y) * warPaddedDim[0] + x;
                                            for (t = 0; t < warPaddedDim[3]; ++t) {
                                                voxIndex_t = t * warPaddedVoxelNumber + voxIndex;
                                                warpedValue = paddedWarImgPtr[voxIndex_t];
#ifdef MRF_USE_SAD
                                                currentValue = fabs(warpedValue - refBlockValue[blockIndex]);
#else
                                                currentValue = reg_pow2(warpedValue - refBlockValue[blockIndex]);
#endif
                                                if (currentValue == currentValue) {
                                                    currentSum -= currentValue;
                                                    ++definedValueNumber;
                                                }
                                                blockIndex++;
                                            }
                                        } // x
                                    } // y
                                } // z
                                discretisedValue[currentControlPoint * label_nD_number + discretisedIndex] =
                                    currentSum / static_cast<float>(definedValueNumber);
                                ++discretisedIndex;
                            } // a
                        } // b
                    } // cc
                } // defined value in the reference block
                ++currentControlPoint;
            } // cpx
        } // cpy
    } // cpz
    free(paddedWarImgPtr);
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
                                            current_distance = reg_pow2(label_x - label2_x) + reg_pow2(label_y - label2_y) + reg_pow2(label_z - label2_z);
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
template <class DataType>
void GetDiscretisedValueSSD_core3D_2(nifti_image *controlPointGridImage,
                                     float *discretisedValue,
                                     int discretise_radius,
                                     int discretise_step,
                                     nifti_image *refImage,
                                     nifti_image *warImage,
                                     int *mask) {
    int cpx, cpy, cpz, t, x, y, z, a, b, c, blockIndex, blockIndex_t, discretisedIndex;
    size_t voxIndex, voxIndex_t;
    const int label_1D_number = (discretise_radius / discretise_step) * 2 + 1;
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
    mat44 grid2img_vox = reg_mat44_mul(image_mm2vox, grid_vox2mm);

    // Compute the block size
    const int blockSize[3] = {
        (int)reg_ceil(controlPointGridImage->dx / refImage->dx),
        (int)reg_ceil(controlPointGridImage->dy / refImage->dy),
        (int)reg_ceil(controlPointGridImage->dz / refImage->dz),
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
    padding_value, refBlockValue, mask, refImgPtr, warImgPtr, discretise_radius, \
    discretise_step, discretisedValue) \
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
                reg_mat44_mul(&grid2img_vox, gridVox, imageVox);
                imageVox[0] = static_cast<float>(reg_round(imageVox[0]));
                imageVox[1] = static_cast<float>(reg_round(imageVox[1]));
                imageVox[2] = static_cast<float>(reg_round(imageVox[2]));

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
                    for (c = int(imageVox[2] - discretise_radius); c <= imageVox[2] + discretise_radius; c += discretise_step) {
                        for (b = int(imageVox[1] - discretise_radius); b <= imageVox[1] + discretise_radius; b += discretise_step) {
                            for (a = int(imageVox[0] - discretise_radius); a <= imageVox[0] + discretise_radius; a += discretise_step) {

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
                                                        currentValue = static_cast<float>(reg_pow2(warImgPtr[voxIndex_t] - refBlockValue[tid][blockIndex_t]));
#endif
                                                    } else {
#ifdef MRF_USE_SAD
                                                        currentValue = fabs(0 - refBlockValue[tid][blockIndex_t]);
#else
                                                        currentValue = reg_pow2(0 - refBlockValue[tid][blockIndex_t]);
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
                                                    currentValue = reg_pow2(0 - refBlockValue[tid][blockIndex_t]);
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
                                            current_distance = static_cast<float>(reg_pow2(label_x - label2_x) + reg_pow2(label_y - label2_y) + reg_pow2(label_z - label2_z));
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
                                  int discretise_radius,
                                  int discretise_step) {
    if (referenceImage->nz > 1) {
        switch (this->referenceImage->datatype) {
        case NIFTI_TYPE_FLOAT32:
            GetDiscretisedValueSSD_core3D_2<float>(controlPointGridImage,
                                                   discretisedValue,
                                                   discretise_radius,
                                                   discretise_step,
                                                   this->referenceImage,
                                                   this->warpedImage,
                                                   this->referenceMask);
            break;
        case NIFTI_TYPE_FLOAT64:
            GetDiscretisedValueSSD_core3D_2<double>(controlPointGridImage,
                                                    discretisedValue,
                                                    discretise_radius,
                                                    discretise_step,
                                                    this->referenceImage,
                                                    this->warpedImage,
                                                    this->referenceMask);
            break;
        default:
            reg_print_fct_error("reg_ssd::GetDiscretisedValue");
            reg_print_msg_error("Unsupported datatype");
            reg_exit();
        }
    } else {
        reg_print_fct_error("reg_ssd::GetDiscretisedValue");
        reg_print_msg_error("Not implemented in 2D yet");
        reg_exit();
    }
}
/* *************************************************************** */
