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
    this->kernelType = ConvKernelType::Gaussian;

    for (int i = 0; i < 255; ++i)
        kernelStandardDeviation[i] = -5.f;

    NR_FUNC_CALLED();
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

    for (int i = 0; i < this->referenceTimePoints; ++i) {
        if (this->timePointWeights[i] > 0) {
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

    for (int i = 0; i < this->referenceTimePoints; ++i)
        NR_DEBUG("Weight for time point " << i << ": " << this->timePointWeights[i]);
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template <class DataType>
void UpdateLocalStatImages(const nifti_image *refImage,
                           const nifti_image *warImage,
                           nifti_image *meanImage,
                           nifti_image *warpedMeanImage,
                           nifti_image *sdevImage,
                           nifti_image *warpedSdevImage,
                           const int *refMask,
                           int *combinedMask,
                           const float *kernelStandardDeviation,
                           const ConvKernelType kernelType,
                           const int currentTimePoint) {
    // Generate the combined mask to ignore all NaN values
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

    const DataType *origRefPtr = static_cast<DataType*>(refImage->data);
    DataType *meanImgPtr = static_cast<DataType*>(meanImage->data);
    DataType *sdevImgPtr = static_cast<DataType*>(sdevImage->data);
    memcpy(meanImgPtr, &origRefPtr[currentTimePoint * voxelNumber], voxelNumber * refImage->nbyper);
    memcpy(sdevImgPtr, &origRefPtr[currentTimePoint * voxelNumber], voxelNumber * refImage->nbyper);

    reg_tools_multiplyImageToImage(sdevImage, sdevImage, sdevImage);
    reg_tools_kernelConvolution(meanImage, kernelStandardDeviation, kernelType, combinedMask);
    reg_tools_kernelConvolution(sdevImage, kernelStandardDeviation, kernelType, combinedMask);

    const DataType *origWarPtr = static_cast<DataType*>(warImage->data);
    DataType *warMeanPtr = static_cast<DataType*>(warpedMeanImage->data);
    DataType *warSdevPtr = static_cast<DataType*>(warpedSdevImage->data);
    memcpy(warMeanPtr, &origWarPtr[currentTimePoint * voxelNumber], voxelNumber * warImage->nbyper);
    memcpy(warSdevPtr, &origWarPtr[currentTimePoint * voxelNumber], voxelNumber * warImage->nbyper);

    reg_tools_multiplyImageToImage(warpedSdevImage, warpedSdevImage, warpedSdevImage);
    reg_tools_kernelConvolution(warpedMeanImage, kernelStandardDeviation, kernelType, combinedMask);
    reg_tools_kernelConvolution(warpedSdevImage, kernelStandardDeviation, kernelType, combinedMask);
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(voxelNumber, sdevImgPtr, meanImgPtr, warSdevPtr, warMeanPtr)
#endif
    for (voxel = 0; voxel < voxelNumber; ++voxel) {
        // G*(I^2) - (G*I)^2
        sdevImgPtr[voxel] = sqrt(sdevImgPtr[voxel] - Square(meanImgPtr[voxel]));
        warSdevPtr[voxel] = sqrt(warSdevPtr[voxel] - Square(warMeanPtr[voxel]));
        // Stabilise the computation
        if (sdevImgPtr[voxel] < 1.e-06) sdevImgPtr[voxel] = 0;
        if (warSdevPtr[voxel] < 1.e-06) warSdevPtr[voxel] = 0;
    }
}
/* *************************************************************** */
template<class DataType>
double reg_getLnccValue(const nifti_image *referenceImage,
                        const nifti_image *meanImage,
                        const nifti_image *sdevImage,
                        const nifti_image *warpedImage,
                        const nifti_image *warpedMeanImage,
                        const nifti_image *warpedSdevImage,
                        const int *combinedMask,
                        const float *kernelStandardDeviation,
                        nifti_image *correlationImage,
                        const ConvKernelType kernelType,
                        const int currentTimePoint) {
#ifdef _WIN32
    long voxel;
    const long voxelNumber = (long)NiftiImage::calcVoxelNumber(referenceImage, 3);
#else
    size_t voxel;
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);
#endif
    // Compute the local correlation
    const DataType *refImagePtr = static_cast<DataType*>(referenceImage->data);
    const DataType *currentRefPtr = &refImagePtr[currentTimePoint * voxelNumber];

    const DataType *warImagePtr = static_cast<DataType*>(warpedImage->data);
    const DataType *currentWarPtr = &warImagePtr[currentTimePoint * voxelNumber];

    const DataType *meanImgPtr = static_cast<DataType*>(meanImage->data);
    const DataType *warMeanPtr = static_cast<DataType*>(warpedMeanImage->data);
    const DataType *sdevImgPtr = static_cast<DataType*>(sdevImage->data);
    const DataType *warSdevPtr = static_cast<DataType*>(warpedSdevImage->data);
    DataType *correlationPtr = static_cast<DataType*>(correlationImage->data);

    for (size_t i = 0; i < voxelNumber; ++i)
        correlationPtr[i] = currentRefPtr[i] * currentWarPtr[i];

    reg_tools_kernelConvolution(correlationImage, kernelStandardDeviation, kernelType, combinedMask);

    double lnccSum = 0;
    size_t activeVoxelNumber = 0;

    // Iteration over all voxels
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(voxelNumber,combinedMask,meanImgPtr,warMeanPtr, \
    sdevImgPtr,warSdevPtr,correlationPtr) \
    reduction(+:lnccSum, activeVoxelNumber)
#endif
    for (voxel = 0; voxel < voxelNumber; ++voxel) {
        // Check if the current voxel belongs to the mask
        if (combinedMask[voxel] > -1) {
            const double lncc = (correlationPtr[voxel] - (meanImgPtr[voxel] * warMeanPtr[voxel])) / (sdevImgPtr[voxel] * warSdevPtr[voxel]);
            if (lncc == lncc && !isinf(lncc)) {
                lnccSum += fabs(lncc);
                ++activeVoxelNumber;
            }
        }
    }
    return lnccSum / activeVoxelNumber;
}
/* *************************************************************** */
double GetSimilarityMeasureValue(const nifti_image *referenceImage,
                                 nifti_image *meanImage,
                                 nifti_image *sdevImage,
                                 const nifti_image *warpedImage,
                                 nifti_image *warpedMeanImage,
                                 nifti_image *warpedSdevImage,
                                 const int *referenceMask,
                                 int *forwardMask,
                                 const float *kernelStandardDeviation,
                                 nifti_image *correlationImage,
                                 const ConvKernelType kernelType,
                                 const int referenceTimePoints,
                                 const double *timePointWeights) {
    double lncc = 0;
    for (int currentTimePoint = 0; currentTimePoint < referenceTimePoints; ++currentTimePoint) {
        if (timePointWeights[currentTimePoint] > 0) {
            const double tp = std::visit([&](auto&& refImgDataType) {
                using RefImgDataType = std::decay_t<decltype(refImgDataType)>;
                // Compute the mean and variance of the reference and warped floating
                UpdateLocalStatImages<RefImgDataType>(referenceImage,
                                                      warpedImage,
                                                      meanImage,
                                                      warpedMeanImage,
                                                      sdevImage,
                                                      warpedSdevImage,
                                                      referenceMask,
                                                      forwardMask,
                                                      kernelStandardDeviation,
                                                      kernelType,
                                                      currentTimePoint);
                // Compute the LNCC value
                return reg_getLnccValue<RefImgDataType>(referenceImage,
                                                        meanImage,
                                                        sdevImage,
                                                        warpedImage,
                                                        warpedMeanImage,
                                                        warpedSdevImage,
                                                        forwardMask,
                                                        kernelStandardDeviation,
                                                        correlationImage,
                                                        kernelType,
                                                        currentTimePoint);
            }, NiftiImage::getFloatingDataType(referenceImage));
            lncc += tp * timePointWeights[currentTimePoint];
        }
    }
    return lncc;
}
/* *************************************************************** */
double reg_lncc::GetSimilarityMeasureValueFw() {
    return ::GetSimilarityMeasureValue(this->referenceImage,
                                       this->meanImage,
                                       this->sdevImage,
                                       this->warpedImage,
                                       this->warpedMeanImage,
                                       this->warpedSdevImage,
                                       this->referenceMask,
                                       this->forwardMask,
                                       this->kernelStandardDeviation,
                                       this->correlationImage,
                                       this->kernelType,
                                       this->referenceTimePoints,
                                       this->timePointWeights);
}
/* *************************************************************** */
double reg_lncc::GetSimilarityMeasureValueBw() {
    return ::GetSimilarityMeasureValue(this->floatingImage,
                                       this->meanImageBw,
                                       this->sdevImageBw,
                                       this->warpedImageBw,
                                       this->warpedMeanImageBw,
                                       this->warpedSdevImageBw,
                                       this->floatingMask,
                                       this->backwardMask,
                                       this->kernelStandardDeviation,
                                       this->correlationImageBw,
                                       this->kernelType,
                                       this->referenceTimePoints,
                                       this->timePointWeights);
}
/* *************************************************************** */
template <class DataType>
void reg_getVoxelBasedLnccGradient(const nifti_image *referenceImage,
                                   const nifti_image *meanImage,
                                   const nifti_image *sdevImage,
                                   const nifti_image *warpedImage,
                                   nifti_image *warpedMeanImage,
                                   nifti_image *warpedSdevImage,
                                   const int *combinedMask,
                                   const float *kernelStandardDeviation,
                                   nifti_image *correlationImage,
                                   const nifti_image *warpedGradient,
                                   nifti_image *measureGradient,
                                   const ConvKernelType kernelType,
                                   const int currentTimePoint,
                                   const double timePointWeight) {
#ifdef _WIN32
    long voxel;
    long voxelNumber = (long)NiftiImage::calcVoxelNumber(referenceImage, 3);
#else
    size_t voxel;
    size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);
#endif
    // Compute the local correlation
    const DataType *refImagePtr = static_cast<DataType*>(referenceImage->data);
    const DataType *currentRefPtr = &refImagePtr[currentTimePoint * voxelNumber];

    const DataType *warImagePtr = static_cast<DataType*>(warpedImage->data);
    const DataType *currentWarPtr = &warImagePtr[currentTimePoint * voxelNumber];

    const DataType *meanImgPtr = static_cast<DataType*>(meanImage->data);
    DataType *warMeanPtr = static_cast<DataType*>(warpedMeanImage->data);
    const DataType *sdevImgPtr = static_cast<DataType*>(sdevImage->data);
    DataType *warSdevPtr = static_cast<DataType*>(warpedSdevImage->data);
    DataType *correlationPtr = static_cast<DataType*>(correlationImage->data);

    for (size_t i = 0; i < voxelNumber; ++i)
        correlationPtr[i] = currentRefPtr[i] * currentWarPtr[i];

    reg_tools_kernelConvolution(correlationImage, kernelStandardDeviation, kernelType, combinedMask);

    size_t activeVoxelNumber = 0;

    // Iteration over all voxels
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(voxelNumber,combinedMask,meanImgPtr,warMeanPtr, \
    sdevImgPtr,warSdevPtr,correlationPtr) \
    reduction(+:activeVoxelNumber)
#endif
    for (voxel = 0; voxel < voxelNumber; ++voxel) {
        // Check if the current voxel belongs to the mask
        if (combinedMask[voxel] > -1) {
            const double refMeanValue = meanImgPtr[voxel];
            const double warMeanValue = warMeanPtr[voxel];
            const double refSdevValue = sdevImgPtr[voxel];
            const double warSdevValue = warSdevPtr[voxel];
            const double correlaValue = correlationPtr[voxel] - (refMeanValue * warMeanValue);
            double temp1 = 1.0 / (refSdevValue * warSdevValue);
            double temp2 = correlaValue / (refSdevValue * warSdevValue * warSdevValue * warSdevValue);
            double temp3 = (correlaValue * warMeanValue) / (refSdevValue * warSdevValue * warSdevValue * warSdevValue)
                - refMeanValue / (refSdevValue * warSdevValue);
            if (temp1 == temp1 && !isinf(temp1) &&
                temp2 == temp2 && !isinf(temp2) &&
                temp3 == temp3 && !isinf(temp3)) {
                // Derivative of the absolute function
                if (correlaValue < 0) {
                    temp1 *= -1;
                    temp2 *= -1;
                    temp3 *= -1;
                }
                warMeanPtr[voxel] = static_cast<DataType>(temp1);
                warSdevPtr[voxel] = static_cast<DataType>(temp2);
                correlationPtr[voxel] = static_cast<DataType>(temp3);
                activeVoxelNumber++;
            } else warMeanPtr[voxel] = warSdevPtr[voxel] = correlationPtr[voxel] = 0;
        } else warMeanPtr[voxel] = warSdevPtr[voxel] = correlationPtr[voxel] = 0;
    }

    //adjust weight for number of voxels
    const double adjustedWeight = timePointWeight / activeVoxelNumber;

    // Smooth the newly computed values
    reg_tools_kernelConvolution(warpedMeanImage, kernelStandardDeviation, kernelType, combinedMask);
    reg_tools_kernelConvolution(warpedSdevImage, kernelStandardDeviation, kernelType, combinedMask);
    reg_tools_kernelConvolution(correlationImage, kernelStandardDeviation, kernelType, combinedMask);
    DataType *measureGradPtrX = static_cast<DataType*>(measureGradient->data);
    DataType *measureGradPtrY = &measureGradPtrX[voxelNumber];
    DataType *measureGradPtrZ = referenceImage->nz > 1 ? &measureGradPtrY[voxelNumber] : nullptr;

    // Create pointers to the spatial gradient of the warped image
    const DataType *warpGradPtrX = static_cast<DataType*>(warpedGradient->data);
    const DataType *warpGradPtrY = &warpGradPtrX[voxelNumber];
    const DataType *warpGradPtrZ = referenceImage->nz > 1 ? &warpGradPtrY[voxelNumber] : nullptr;

    // Iteration over all voxels
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(voxelNumber,combinedMask,currentRefPtr,currentWarPtr, \
    warMeanPtr,warSdevPtr,correlationPtr,measureGradPtrX,measureGradPtrY, \
    measureGradPtrZ, warpGradPtrX, warpGradPtrY, warpGradPtrZ, adjustedWeight)
#endif
    for (voxel = 0; voxel < voxelNumber; ++voxel) {
        // Check if the current voxel belongs to the mask
        if (combinedMask[voxel] > -1) {
            const double common = (warMeanPtr[voxel] * currentRefPtr[voxel] - warSdevPtr[voxel] * currentWarPtr[voxel] + correlationPtr[voxel]) * adjustedWeight;
            measureGradPtrX[voxel] -= static_cast<DataType>(warpGradPtrX[voxel] * common);
            measureGradPtrY[voxel] -= static_cast<DataType>(warpGradPtrY[voxel] * common);
            if (warpGradPtrZ != nullptr)
                measureGradPtrZ[voxel] -= static_cast<DataType>(warpGradPtrZ[voxel] * common);
        }
    }
    // Check for NaN
#ifdef _WIN32
    voxelNumber = (long)measureGradient->nvox;
#else
    voxelNumber = measureGradient->nvox;
#endif
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(voxelNumber, measureGradPtrX)
#endif
    for (voxel = 0; voxel < voxelNumber; ++voxel) {
        const DataType val = measureGradPtrX[voxel];
        if (val != val || isinf(val))
            measureGradPtrX[voxel] = 0;
    }
}
/* *************************************************************** */
void GetVoxelBasedSimilarityMeasureGradient(const nifti_image *referenceImage,
                                            nifti_image *meanImage,
                                            nifti_image *sdevImage,
                                            const nifti_image *warpedImage,
                                            nifti_image *warpedMeanImage,
                                            nifti_image *warpedSdevImage,
                                            const int *referenceMask,
                                            int *forwardMask,
                                            const float *kernelStandardDeviation,
                                            nifti_image *correlationImage,
                                            const nifti_image *warpedGradient,
                                            nifti_image *measureGradient,
                                            const ConvKernelType kernelType,
                                            const int currentTimePoint,
                                            const double timePointWeight) {
    std::visit([&](auto&& refImgDataType) {
        using RefImgDataType = std::decay_t<decltype(refImgDataType)>;
        // Compute the mean and variance of the reference and warped floating
        UpdateLocalStatImages<RefImgDataType>(referenceImage,
                                              warpedImage,
                                              meanImage,
                                              warpedMeanImage,
                                              sdevImage,
                                              warpedSdevImage,
                                              referenceMask,
                                              forwardMask,
                                              kernelStandardDeviation,
                                              kernelType,
                                              currentTimePoint);
        // Compute the LNCC gradient
        reg_getVoxelBasedLnccGradient<RefImgDataType>(referenceImage,
                                                      meanImage,
                                                      sdevImage,
                                                      warpedImage,
                                                      warpedMeanImage,
                                                      warpedSdevImage,
                                                      forwardMask,
                                                      kernelStandardDeviation,
                                                      correlationImage,
                                                      warpedGradient,
                                                      measureGradient,
                                                      kernelType,
                                                      currentTimePoint,
                                                      timePointWeight);
    }, NiftiImage::getFloatingDataType(referenceImage));
}
/* *************************************************************** */
void reg_lncc::GetVoxelBasedSimilarityMeasureGradientFw(int currentTimePoint) {
    ::GetVoxelBasedSimilarityMeasureGradient(this->referenceImage,
                                             this->meanImage,
                                             this->sdevImage,
                                             this->warpedImage,
                                             this->warpedMeanImage,
                                             this->warpedSdevImage,
                                             this->referenceMask,
                                             this->forwardMask,
                                             this->kernelStandardDeviation,
                                             this->correlationImage,
                                             this->warpedGradient,
                                             this->voxelBasedGradient,
                                             this->kernelType,
                                             currentTimePoint,
                                             this->timePointWeights[currentTimePoint]);
}
/* *************************************************************** */
void reg_lncc::GetVoxelBasedSimilarityMeasureGradientBw(int currentTimePoint) {
    ::GetVoxelBasedSimilarityMeasureGradient(this->floatingImage,
                                             this->meanImageBw,
                                             this->sdevImageBw,
                                             this->warpedImageBw,
                                             this->warpedMeanImageBw,
                                             this->warpedSdevImageBw,
                                             this->floatingMask,
                                             this->backwardMask,
                                             this->kernelStandardDeviation,
                                             this->correlationImageBw,
                                             this->warpedGradientBw,
                                             this->voxelBasedGradientBw,
                                             this->kernelType,
                                             currentTimePoint,
                                             this->timePointWeights[currentTimePoint]);
}
/* *************************************************************** */
