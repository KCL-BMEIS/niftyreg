/**
 * @file  _reg_lncc.cpp
 * @author Aileen Corder
 * @author Marc Modat
 * @date 10/11/2012.
 * @brief CPP file for the LNCC related class and functions
 * Copyright (c) 2012-2018, University College London
 * Copyright (c) 2018, NiftyReg Developers.
 * All rights reserved.
 * See the LICENSE.txt file in the nifty_reg root folder
 */

#include "_reg_lncc.h"

/* *************************************************************** */
reg_lncc::reg_lncc(): reg_measure() {
    this->correlationImage = nullptr;
    this->meanImage = nullptr;
    this->sdevImage = nullptr;
    this->warpedMeanImage = nullptr;
    this->warpedSdevImage = nullptr;
    this->forwardMask = nullptr;

    this->correlationImageBw = nullptr;
    this->meanImageBw = nullptr;
    this->sdevImageBw = nullptr;
    this->warpedMeanImageBw = nullptr;
    this->warpedSdevImageBw = nullptr;
    this->backwardMask = nullptr;

    // Gaussian kernel is used by default
    this->kernelType = GAUSSIAN_KERNEL;

    for (int i = 0; i < 255; ++i)
        kernelStandardDeviation[i] = -5.f;
#ifndef NDEBUG
    reg_print_msg_debug("reg_lncc constructor called");
#endif
}
/* *************************************************************** */
reg_lncc::~reg_lncc() {
    if (this->correlationImage != nullptr)
        nifti_image_free(this->correlationImage);
    this->correlationImage = nullptr;
    if (this->meanImage != nullptr)
        nifti_image_free(this->meanImage);
    this->meanImage = nullptr;
    if (this->sdevImage != nullptr)
        nifti_image_free(this->sdevImage);
    this->sdevImage = nullptr;
    if (this->warpedMeanImage != nullptr)
        nifti_image_free(this->warpedMeanImage);
    this->warpedMeanImage = nullptr;
    if (this->warpedSdevImage != nullptr)
        nifti_image_free(this->warpedSdevImage);
    this->warpedSdevImage = nullptr;
    if (this->forwardMask != nullptr)
        free(this->forwardMask);
    this->forwardMask = nullptr;

    if (this->correlationImageBw != nullptr)
        nifti_image_free(this->correlationImageBw);
    this->correlationImageBw = nullptr;
    if (this->meanImageBw != nullptr)
        nifti_image_free(this->meanImageBw);
    this->meanImageBw = nullptr;
    if (this->sdevImageBw != nullptr)
        nifti_image_free(this->sdevImageBw);
    this->sdevImageBw = nullptr;
    if (this->warpedMeanImageBw != nullptr)
        nifti_image_free(this->warpedMeanImageBw);
    this->warpedMeanImageBw = nullptr;
    if (this->warpedSdevImageBw != nullptr)
        nifti_image_free(this->warpedSdevImageBw);
    this->warpedSdevImageBw = nullptr;
    if (this->backwardMask != nullptr)
        free(this->backwardMask);
    this->backwardMask = nullptr;
}
/* *************************************************************** */
template <class DataType>
void reg_lncc::UpdateLocalStatImages(nifti_image *refImage,
                                     nifti_image *warImage,
                                     nifti_image *meanImage,
                                     nifti_image *warpedMeanImage,
                                     nifti_image *stdDevImage,
                                     nifti_image *warpedSdevImage,
                                     int *refMask,
                                     int *combinedMask,
                                     int currentTimepoint) {
    // Generate the forward mask to ignore all NaN values
#ifdef _WIN32
    long voxel;
    const long voxelNumber = (long)NiftiImage::calcVoxelNumber(refImage, 3);
#else
    size_t voxel;
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(refImage, 3);
#endif
    memcpy(combinedMask, refMask, voxelNumber * sizeof(int));
    reg_tools_removeNanFromMask(refImage, combinedMask);
    reg_tools_removeNanFromMask(warImage, combinedMask);

    DataType *origRefPtr = static_cast<DataType*>(refImage->data);
    DataType *meanImgPtr = static_cast<DataType*>(meanImage->data);
    DataType *sdevImgPtr = static_cast<DataType*>(stdDevImage->data);
    memcpy(meanImgPtr, &origRefPtr[currentTimepoint * voxelNumber], voxelNumber * refImage->nbyper);
    memcpy(sdevImgPtr, &origRefPtr[currentTimepoint * voxelNumber], voxelNumber * refImage->nbyper);

    reg_tools_multiplyImageToImage(stdDevImage, stdDevImage, stdDevImage);
    reg_tools_kernelConvolution(meanImage, this->kernelStandardDeviation, this->kernelType, combinedMask);
    reg_tools_kernelConvolution(stdDevImage, this->kernelStandardDeviation, this->kernelType, combinedMask);

    DataType *origWarPtr = static_cast<DataType*>(warImage->data);
    DataType *warMeanPtr = static_cast<DataType*>(warpedMeanImage->data);
    DataType *warSdevPtr = static_cast<DataType*>(warpedSdevImage->data);
    memcpy(warMeanPtr, &origWarPtr[currentTimepoint * voxelNumber], voxelNumber * warImage->nbyper);
    memcpy(warSdevPtr, &origWarPtr[currentTimepoint * voxelNumber], voxelNumber * warImage->nbyper);

    reg_tools_multiplyImageToImage(warpedSdevImage, warpedSdevImage, warpedSdevImage);
    reg_tools_kernelConvolution(warpedMeanImage, this->kernelStandardDeviation, this->kernelType, combinedMask);
    reg_tools_kernelConvolution(warpedSdevImage, this->kernelStandardDeviation, this->kernelType, combinedMask);
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(voxelNumber, sdevImgPtr, meanImgPtr, warSdevPtr, warMeanPtr)
#endif
    for (voxel = 0; voxel < voxelNumber; ++voxel) {
        // G*(I^2) - (G*I)^2
        sdevImgPtr[voxel] = sqrt(sdevImgPtr[voxel] - reg_pow2(meanImgPtr[voxel]));
        warSdevPtr[voxel] = sqrt(warSdevPtr[voxel] - reg_pow2(warMeanPtr[voxel]));
        // Stabilise the computation
        if (sdevImgPtr[voxel] < 1.e-06) sdevImgPtr[voxel] = 0;
        if (warSdevPtr[voxel] < 1.e-06) warSdevPtr[voxel] = 0;
    }
}
/* *************************************************************** */
void reg_lncc::InitialiseMeasure(nifti_image *refImg,
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

    for (int i = 0; i < this->referenceImage->nt; ++i) {
        if (this->timePointWeight[i] > 0) {
            reg_intensityRescale(this->referenceImage, i, 0.f, 1.f);
            reg_intensityRescale(this->floatingImage, i, 0.f, 1.f);
        }
    }

    // Check that no images are already allocated
    if (this->correlationImage != nullptr)
        nifti_image_free(this->correlationImage);
    this->correlationImage = nullptr;
    if (this->meanImage != nullptr)
        nifti_image_free(this->meanImage);
    this->meanImage = nullptr;
    if (this->sdevImage != nullptr)
        nifti_image_free(this->sdevImage);
    this->sdevImage = nullptr;
    if (this->warpedMeanImage != nullptr)
        nifti_image_free(this->warpedMeanImage);
    this->warpedMeanImage = nullptr;
    if (this->warpedSdevImage != nullptr)
        nifti_image_free(this->warpedSdevImage);
    this->warpedSdevImage = nullptr;
    if (this->correlationImageBw != nullptr)
        nifti_image_free(this->correlationImageBw);
    this->correlationImageBw = nullptr;
    if (this->meanImageBw != nullptr)
        nifti_image_free(this->meanImageBw);
    this->meanImageBw = nullptr;
    if (this->sdevImageBw != nullptr)
        nifti_image_free(this->sdevImageBw);
    this->sdevImageBw = nullptr;
    if (this->warpedMeanImageBw != nullptr)
        nifti_image_free(this->warpedMeanImageBw);
    this->warpedMeanImageBw = nullptr;
    if (this->warpedSdevImageBw != nullptr)
        nifti_image_free(this->warpedSdevImageBw);
    this->warpedSdevImageBw = nullptr;
    if (this->forwardMask != nullptr)
        free(this->forwardMask);
    this->forwardMask = nullptr;
    if (this->backwardMask != nullptr)
        free(this->backwardMask);
    this->backwardMask = nullptr;

    size_t voxelNumber = NiftiImage::calcVoxelNumber(this->referenceImage, 3);

    // Allocate the required image to store the correlation of the forward transformation
    this->correlationImage = nifti_copy_nim_info(this->referenceImage);
    this->correlationImage->ndim = this->correlationImage->dim[0] = this->referenceImage->nz > 1 ? 3 : 2;
    this->correlationImage->nt = this->correlationImage->dim[4] = 1;
    this->correlationImage->nvox = voxelNumber;
    this->correlationImage->data = malloc(voxelNumber * this->correlationImage->nbyper);

    // Allocate the required images to store mean and stdev of the reference image
    this->meanImage = nifti_dup(*this->correlationImage, false);
    this->sdevImage = nifti_dup(*this->correlationImage, false);

    // Allocate the required images to store mean and stdev of the warped floating image
    this->warpedMeanImage = nifti_dup(*this->correlationImage, false);
    this->warpedSdevImage = nifti_dup(*this->correlationImage, false);

    // Allocate the array to store the mask of the forward image
    this->forwardMask = (int*)malloc(voxelNumber * sizeof(int));
    if (this->isSymmetric) {
        voxelNumber = NiftiImage::calcVoxelNumber(floatingImage, 3);

        // Allocate the required image to store the correlation of the backward transformation
        this->correlationImageBw = nifti_copy_nim_info(this->floatingImage);
        this->correlationImageBw->ndim = this->correlationImageBw->dim[0] = this->floatingImage->nz > 1 ? 3 : 2;
        this->correlationImageBw->nt = this->correlationImageBw->dim[4] = 1;
        this->correlationImageBw->nvox = voxelNumber;
        this->correlationImageBw->data = malloc(voxelNumber * this->correlationImageBw->nbyper);

        // Allocate the required images to store mean and stdev of the floating image
        this->meanImageBw = nifti_dup(*this->correlationImageBw, false);
        this->sdevImageBw = nifti_dup(*this->correlationImageBw, false);

        // Allocate the required images to store mean and stdev of the warped reference image
        this->warpedMeanImageBw = nifti_dup(*this->correlationImageBw, false);
        this->warpedSdevImageBw = nifti_dup(*this->correlationImageBw, false);

        // Allocate the array to store the mask of the backward image
        this->backwardMask = (int*)malloc(voxelNumber * sizeof(int));
    }
#ifndef NDEBUG
    char text[255];
    reg_print_msg_debug("reg_lncc::InitialiseMeasure().");
    for (int i = 0; i < this->referenceImage->nt; ++i) {
        sprintf(text, "Weight for timepoint %i: %f", i, this->timePointWeight[i]);
        reg_print_msg_debug(text);
    }
#endif
}
/* *************************************************************** */
template<class DataType>
double reg_getLNCCValue(nifti_image *referenceImage,
                        nifti_image *meanImage,
                        nifti_image *sdevImage,
                        nifti_image *warpedImage,
                        nifti_image *warpedMeanImage,
                        nifti_image *warpedSdevImage,
                        int *combinedMask,
                        float *kernelStandardDeviation,
                        nifti_image *correlationImage,
                        int kernelType,
                        int currentTimepoint) {
#ifdef _WIN32
    long voxel;
    const long voxelNumber = (long)NiftiImage::calcVoxelNumber(referenceImage, 3);
#else
    size_t voxel;
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);
#endif

    // Compute the local correlation
    DataType *refImagePtr = static_cast<DataType*>(referenceImage->data);
    DataType *currentRefPtr = &refImagePtr[currentTimepoint * voxelNumber];

    DataType *warImagePtr = static_cast<DataType*>(warpedImage->data);
    DataType *currentWarPtr = &warImagePtr[currentTimepoint * voxelNumber];

    DataType *meanImgPtr = static_cast<DataType*>(meanImage->data);
    DataType *warMeanPtr = static_cast<DataType*>(warpedMeanImage->data);
    DataType *sdevImgPtr = static_cast<DataType*>(sdevImage->data);
    DataType *warSdevPtr = static_cast<DataType*>(warpedSdevImage->data);
    DataType *correlationPtr = static_cast<DataType*>(correlationImage->data);

    for (size_t i = 0; i < voxelNumber; ++i)
        correlationPtr[i] = currentRefPtr[i] * currentWarPtr[i];

    reg_tools_kernelConvolution(correlationImage, kernelStandardDeviation, kernelType, combinedMask);

    double lncc_value_sum = 0., lncc_value;
    double activeVoxel_num = 0.;

    // Iteration over all voxels
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(voxelNumber,combinedMask,meanImgPtr,warMeanPtr, \
    sdevImgPtr,warSdevPtr,correlationPtr) \
    private(lncc_value) \
    reduction(+:lncc_value_sum) \
    reduction(+:activeVoxel_num)
#endif
    for (voxel = 0; voxel < voxelNumber; ++voxel) {
        // Check if the current voxel belongs to the mask
        if (combinedMask[voxel] > -1) {
            lncc_value = (correlationPtr[voxel] - (meanImgPtr[voxel] * warMeanPtr[voxel])) / (sdevImgPtr[voxel] * warSdevPtr[voxel]);
            if (lncc_value == lncc_value && isinf(lncc_value) == 0) {
                lncc_value_sum += fabs(lncc_value);
                ++activeVoxel_num;
            }
        }
    }
    return lncc_value_sum / activeVoxel_num;
}
/* *************************************************************** */
double reg_lncc::GetSimilarityMeasureValue() {
    double lncc_value = 0;

    for (int currentTimepoint = 0; currentTimepoint < this->referenceImage->nt; ++currentTimepoint) {
        if (this->timePointWeight[currentTimepoint] > 0) {
            double tp_value = 0;
            // Compute the mean and variance of the reference and warped floating
            switch (this->referenceImage->datatype) {
            case NIFTI_TYPE_FLOAT32:
                this->UpdateLocalStatImages<float>(this->referenceImage,
                                                   this->warpedImage,
                                                   this->meanImage,
                                                   this->warpedMeanImage,
                                                   this->sdevImage,
                                                   this->warpedSdevImage,
                                                   this->referenceMask,
                                                   this->forwardMask,
                                                   currentTimepoint);
                break;
            case NIFTI_TYPE_FLOAT64:
                this->UpdateLocalStatImages<double>(this->referenceImage,
                                                    this->warpedImage,
                                                    this->meanImage,
                                                    this->warpedMeanImage,
                                                    this->sdevImage,
                                                    this->warpedSdevImage,
                                                    this->referenceMask,
                                                    this->forwardMask,
                                                    currentTimepoint);
                break;
            }

            // Compute the LNCC - Forward
            switch (this->referenceImage->datatype) {
            case NIFTI_TYPE_FLOAT32:
                tp_value += reg_getLNCCValue<float>(this->referenceImage,
                                                    this->meanImage,
                                                    this->sdevImage,
                                                    this->warpedImage,
                                                    this->warpedMeanImage,
                                                    this->warpedSdevImage,
                                                    this->forwardMask,
                                                    this->kernelStandardDeviation,
                                                    this->correlationImage,
                                                    this->kernelType,
                                                    currentTimepoint);
                break;
            case NIFTI_TYPE_FLOAT64:
                tp_value += reg_getLNCCValue<double>(this->referenceImage,
                                                     this->meanImage,
                                                     this->sdevImage,
                                                     this->warpedImage,
                                                     this->warpedMeanImage,
                                                     this->warpedSdevImage,
                                                     this->forwardMask,
                                                     this->kernelStandardDeviation,
                                                     this->correlationImage,
                                                     this->kernelType,
                                                     currentTimepoint);
                break;
            }
            if (this->isSymmetric) {
                // Compute the mean and variance of the floating and warped reference
                switch (this->floatingImage->datatype) {
                case NIFTI_TYPE_FLOAT32:
                    this->UpdateLocalStatImages<float>(this->floatingImage,
                                                       this->warpedImageBw,
                                                       this->meanImageBw,
                                                       this->warpedMeanImageBw,
                                                       this->sdevImageBw,
                                                       this->warpedSdevImageBw,
                                                       this->floatingMask,
                                                       this->backwardMask,
                                                       currentTimepoint);
                    break;
                case NIFTI_TYPE_FLOAT64:
                    this->UpdateLocalStatImages<double>(this->floatingImage,
                                                        this->warpedImageBw,
                                                        this->meanImageBw,
                                                        this->warpedMeanImageBw,
                                                        this->sdevImageBw,
                                                        this->warpedSdevImageBw,
                                                        this->floatingMask,
                                                        this->backwardMask,
                                                        currentTimepoint);
                    break;
                }
                // Compute the LNCC - Backward
                switch (this->floatingImage->datatype) {
                case NIFTI_TYPE_FLOAT32:
                    tp_value += reg_getLNCCValue<float>(this->floatingImage,
                                                        this->meanImageBw,
                                                        this->sdevImageBw,
                                                        this->warpedImageBw,
                                                        this->warpedMeanImageBw,
                                                        this->warpedSdevImageBw,
                                                        this->backwardMask,
                                                        this->kernelStandardDeviation,
                                                        this->correlationImageBw,
                                                        this->kernelType,
                                                        currentTimepoint);
                    break;
                case NIFTI_TYPE_FLOAT64:
                    tp_value += reg_getLNCCValue<double>(this->floatingImage,
                                                         this->meanImageBw,
                                                         this->sdevImageBw,
                                                         this->warpedImageBw,
                                                         this->warpedMeanImageBw,
                                                         this->warpedSdevImageBw,
                                                         this->backwardMask,
                                                         this->kernelStandardDeviation,
                                                         this->correlationImageBw,
                                                         this->kernelType,
                                                         currentTimepoint);
                    break;
                }
            }
            lncc_value += tp_value * this->timePointWeight[currentTimepoint];
        }
    }
    return lncc_value;
}
/* *************************************************************** */
template <class DataType>
void reg_getVoxelBasedLNCCGradient(nifti_image *referenceImage,
                                   nifti_image *meanImage,
                                   nifti_image *sdevImage,
                                   nifti_image *warpedImage,
                                   nifti_image *warpedMeanImage,
                                   nifti_image *warpedSdevImage,
                                   int *combinedMask,
                                   float *kernelStandardDeviation,
                                   nifti_image *correlationImage,
                                   nifti_image *warpedGradient,
                                   nifti_image *measureGradientImage,
                                   int kernelType,
                                   int currentTimepoint,
                                   double timepointWeight) {
#ifdef _WIN32
    long voxel;
    long voxelNumber = (long)NiftiImage::calcVoxelNumber(referenceImage, 3);
#else
    size_t voxel;
    size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);
#endif

    // Compute the local correlation
    DataType *refImagePtr = static_cast<DataType*>(referenceImage->data);
    DataType *currentRefPtr = &refImagePtr[currentTimepoint * voxelNumber];

    DataType *warImagePtr = static_cast<DataType*>(warpedImage->data);
    DataType *currentWarPtr = &warImagePtr[currentTimepoint * voxelNumber];

    DataType *meanImgPtr = static_cast<DataType*>(meanImage->data);
    DataType *warMeanPtr = static_cast<DataType*>(warpedMeanImage->data);
    DataType *sdevImgPtr = static_cast<DataType*>(sdevImage->data);
    DataType *warSdevPtr = static_cast<DataType*>(warpedSdevImage->data);
    DataType *correlationPtr = static_cast<DataType*>(correlationImage->data);

    for (size_t i = 0; i < voxelNumber; ++i)
        correlationPtr[i] = currentRefPtr[i] * currentWarPtr[i];

    reg_tools_kernelConvolution(correlationImage, kernelStandardDeviation, kernelType, combinedMask);

    double refMeanValue, warMeanValue, refSdevValue, warSdevValue, correlaValue;
    double temp1, temp2, temp3;
    double activeVoxel_num = 0;

    // Iteration over all voxels
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(voxelNumber,combinedMask,meanImgPtr,warMeanPtr, \
    sdevImgPtr,warSdevPtr,correlationPtr) \
    private(refMeanValue,warMeanValue,refSdevValue, \
    warSdevValue, correlaValue, temp1, temp2, temp3) \
    reduction(+:activeVoxel_num)
#endif
    for (voxel = 0; voxel < voxelNumber; ++voxel) {
        // Check if the current voxel belongs to the mask
        if (combinedMask[voxel] > -1) {

            refMeanValue = meanImgPtr[voxel];
            warMeanValue = warMeanPtr[voxel];
            refSdevValue = sdevImgPtr[voxel];
            warSdevValue = warSdevPtr[voxel];
            correlaValue = correlationPtr[voxel] - (refMeanValue * warMeanValue);

            temp1 = 1.0 / (refSdevValue * warSdevValue);
            temp2 = correlaValue /
                (refSdevValue * warSdevValue * warSdevValue * warSdevValue);
            temp3 = (correlaValue * warMeanValue) /
                (refSdevValue * warSdevValue * warSdevValue * warSdevValue)
                -
                refMeanValue / (refSdevValue * warSdevValue);
            if (temp1 == temp1 && isinf(temp1) == 0 &&
                temp2 == temp2 && isinf(temp2) == 0 &&
                temp3 == temp3 && isinf(temp3) == 0) {
                // Derivative of the absolute function
                if (correlaValue < 0) {
                    temp1 *= -1;
                    temp2 *= -1;
                    temp3 *= -1;
                }
                warMeanPtr[voxel] = static_cast<DataType>(temp1);
                warSdevPtr[voxel] = static_cast<DataType>(temp2);
                correlationPtr[voxel] = static_cast<DataType>(temp3);
                activeVoxel_num++;
            } else warMeanPtr[voxel] = warSdevPtr[voxel] = correlationPtr[voxel] = 0;
        } else warMeanPtr[voxel] = warSdevPtr[voxel] = correlationPtr[voxel] = 0;
    }

    //adjust weight for number of voxels
    double adjusted_weight = timepointWeight / activeVoxel_num;

    // Smooth the newly computed values
    reg_tools_kernelConvolution(warpedMeanImage, kernelStandardDeviation, kernelType, combinedMask);
    reg_tools_kernelConvolution(warpedSdevImage, kernelStandardDeviation, kernelType, combinedMask);
    reg_tools_kernelConvolution(correlationImage, kernelStandardDeviation, kernelType, combinedMask);
    DataType *measureGradPtrX = static_cast<DataType*>(measureGradientImage->data);
    DataType *measureGradPtrY = &measureGradPtrX[voxelNumber];
    DataType *measureGradPtrZ = nullptr;
    if (referenceImage->nz > 1)
        measureGradPtrZ = &measureGradPtrY[voxelNumber];

    // Create pointers to the spatial gradient of the warped image
    DataType *warpGradPtrX = static_cast<DataType*>(warpedGradient->data);
    DataType *warpGradPtrY = &warpGradPtrX[voxelNumber];
    DataType *warpGradPtrZ = nullptr;
    if (referenceImage->nz > 1)
        warpGradPtrZ = &warpGradPtrY[voxelNumber];

    double common;
    // Iteration over all voxels
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(voxelNumber,combinedMask,currentRefPtr,currentWarPtr, \
    warMeanPtr,warSdevPtr,correlationPtr,measureGradPtrX,measureGradPtrY, \
    measureGradPtrZ, warpGradPtrX, warpGradPtrY, warpGradPtrZ, adjusted_weight) \
    private(common)
#endif
    for (voxel = 0; voxel < voxelNumber; ++voxel) {
        // Check if the current voxel belongs to the mask
        if (combinedMask[voxel] > -1) {
            common = warMeanPtr[voxel] * currentRefPtr[voxel] - warSdevPtr[voxel] * currentWarPtr[voxel] + correlationPtr[voxel];
            common *= adjusted_weight;
            measureGradPtrX[voxel] -= static_cast<DataType>(warpGradPtrX[voxel] * common);
            measureGradPtrY[voxel] -= static_cast<DataType>(warpGradPtrY[voxel] * common);
            if (warpGradPtrZ != nullptr)
                measureGradPtrZ[voxel] -= static_cast<DataType>(warpGradPtrZ[voxel] * common);
        }
    }
    // Check for NaN
    DataType val;
#ifdef _WIN32
    voxelNumber = (long)measureGradientImage->nvox;
#else
    voxelNumber = measureGradientImage->nvox;
#endif
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(voxelNumber,measureGradPtrX) \
    private(val)
#endif
    for (voxel = 0; voxel < voxelNumber; ++voxel) {
        val = measureGradPtrX[voxel];
        if (val != val || isinf(val) != 0)
            measureGradPtrX[voxel] = 0;
    }
}
/* *************************************************************** */
void reg_lncc::GetVoxelBasedSimilarityMeasureGradient(int currentTimepoint) {
    // Check if the specified time point exists and is active
    reg_measure::GetVoxelBasedSimilarityMeasureGradient(currentTimepoint);
    if (this->timePointWeight[currentTimepoint] == 0)
        return;

    // Compute the mean and variance of the reference and warped floating
    switch (this->referenceImage->datatype) {
    case NIFTI_TYPE_FLOAT32:
        this->UpdateLocalStatImages<float>(this->referenceImage,
                                           this->warpedImage,
                                           this->meanImage,
                                           this->warpedMeanImage,
                                           this->sdevImage,
                                           this->warpedSdevImage,
                                           this->referenceMask,
                                           this->forwardMask,
                                           currentTimepoint);
        break;
    case NIFTI_TYPE_FLOAT64:
        this->UpdateLocalStatImages<double>(this->referenceImage,
                                            this->warpedImage,
                                            this->meanImage,
                                            this->warpedMeanImage,
                                            this->sdevImage,
                                            this->warpedSdevImage,
                                            this->referenceMask,
                                            this->forwardMask,
                                            currentTimepoint);
        break;
    }

    // Compute the LNCC gradient - Forward
    switch (this->referenceImage->datatype) {
    case NIFTI_TYPE_FLOAT32:
        reg_getVoxelBasedLNCCGradient<float>(this->referenceImage,
                                             this->meanImage,
                                             this->sdevImage,
                                             this->warpedImage,
                                             this->warpedMeanImage,
                                             this->warpedSdevImage,
                                             this->forwardMask,
                                             this->kernelStandardDeviation,
                                             this->correlationImage,
                                             this->warpedGradient,
                                             this->voxelBasedGradient,
                                             this->kernelType,
                                             currentTimepoint,
                                             this->timePointWeight[currentTimepoint]);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_getVoxelBasedLNCCGradient<double>(this->referenceImage,
                                              this->meanImage,
                                              this->sdevImage,
                                              this->warpedImage,
                                              this->warpedMeanImage,
                                              this->warpedSdevImage,
                                              this->forwardMask,
                                              this->kernelStandardDeviation,
                                              this->correlationImage,
                                              this->warpedGradient,
                                              this->voxelBasedGradient,
                                              this->kernelType,
                                              currentTimepoint,
                                              this->timePointWeight[currentTimepoint]);
        break;
    }
    if (this->isSymmetric) {
        // Compute the mean and variance of the floating and warped reference
        switch (this->floatingImage->datatype) {
        case NIFTI_TYPE_FLOAT32:
            this->UpdateLocalStatImages<float>(this->floatingImage,
                                               this->warpedImageBw,
                                               this->meanImageBw,
                                               this->warpedMeanImageBw,
                                               this->sdevImageBw,
                                               this->warpedSdevImageBw,
                                               this->floatingMask,
                                               this->backwardMask,
                                               currentTimepoint);
            break;
        case NIFTI_TYPE_FLOAT64:
            this->UpdateLocalStatImages<double>(this->floatingImage,
                                                this->warpedImageBw,
                                                this->meanImageBw,
                                                this->warpedMeanImageBw,
                                                this->sdevImageBw,
                                                this->warpedSdevImageBw,
                                                this->floatingMask,
                                                this->backwardMask,
                                                currentTimepoint);
            break;
        }
        // Compute the LNCC gradient - Backward
        switch (this->floatingImage->datatype) {
        case NIFTI_TYPE_FLOAT32:
            reg_getVoxelBasedLNCCGradient<float>(this->floatingImage,
                                                 this->meanImageBw,
                                                 this->sdevImageBw,
                                                 this->warpedImageBw,
                                                 this->warpedMeanImageBw,
                                                 this->warpedSdevImageBw,
                                                 this->backwardMask,
                                                 this->kernelStandardDeviation,
                                                 this->correlationImageBw,
                                                 this->warpedGradientBw,
                                                 this->voxelBasedGradientBw,
                                                 this->kernelType,
                                                 currentTimepoint,
                                                 this->timePointWeight[currentTimepoint]);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_getVoxelBasedLNCCGradient<double>(this->floatingImage,
                                                  this->meanImageBw,
                                                  this->sdevImageBw,
                                                  this->warpedImageBw,
                                                  this->warpedMeanImageBw,
                                                  this->warpedSdevImageBw,
                                                  this->backwardMask,
                                                  this->kernelStandardDeviation,
                                                  this->correlationImageBw,
                                                  this->warpedGradientBw,
                                                  this->voxelBasedGradientBw,
                                                  this->kernelType,
                                                  currentTimepoint,
                                                  this->timePointWeight[currentTimepoint]);
            break;
        }
    }
}
/* *************************************************************** */
