/*
 *  _reg_mind.cpp
 *
 *
 *  Created by Benoit Presles on 01/12/2015.
 *  Copyright (c) 2015-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_mind.h"

/* *************************************************************** */
template <class DataType>
void ShiftImage(const nifti_image *inputImage,
                nifti_image *shiftedImage,
                const int *mask,
                const int tx,
                const int ty,
                const int tz) {
    const DataType* inputData = static_cast<DataType*>(inputImage->data);
    DataType* shiftImageData = static_cast<DataType*>(shiftedImage->data);
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(inputData, shiftImageData, shiftedImage, inputImage, mask, tx, ty, tz)
#endif
    for (int z = 0; z < shiftedImage->nz; z++) {
        int currentIndex = z * shiftedImage->nx * shiftedImage->ny;
        const int oldZ = z - tz;
        for (int y = 0; y < shiftedImage->ny; y++) {
            const int oldY = y - ty;
            for (int x = 0; x < shiftedImage->nx; x++) {
                const int oldX = x - tx;
                if (-1 < oldX && oldX < inputImage->nx &&
                    -1 < oldY && oldY < inputImage->ny &&
                    -1 < oldZ && oldZ < inputImage->nz) {
                    const int shiftedIndex = (oldZ * inputImage->ny + oldY) * inputImage->nx + oldX;
                    if (mask[shiftedIndex] > -1) {
                        shiftImageData[currentIndex] = inputData[shiftedIndex];
                    } // mask is not defined
                    else {
                        shiftImageData[currentIndex] = 0;
                    }
                } // outside of the image
                else {
                    shiftImageData[currentIndex] = 0;
                }
                currentIndex++;
            }
        }
    }
}
/* *************************************************************** */
template <class DataType>
void GetMindImageDescriptorCore(const nifti_image *inputImage,
                                nifti_image *mindImage,
                                const int *mask,
                                const int descriptorOffset,
                                const int currentTimePoint) {
#ifdef WIN32
    long voxelIndex;
    const long voxelNumber = (long)NiftiImage::calcVoxelNumber(inputImage, 3);
#else
    size_t voxelIndex;
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(inputImage, 3);
#endif
    // Create a pointer to the descriptor image
    DataType* mindImgDataPtr = static_cast<DataType*>(mindImage->data);

    // Allocate an image to store the current time point reference image
    nifti_image *currentInputImage = nifti_copy_nim_info(inputImage);
    currentInputImage->ndim = currentInputImage->dim[0] = inputImage->nz > 1 ? 3 : 2;
    currentInputImage->nt = currentInputImage->dim[4] = 1;
    currentInputImage->nvox = voxelNumber;
    DataType *inputImagePtr = static_cast<DataType*>(inputImage->data);
    currentInputImage->data = &inputImagePtr[currentTimePoint * voxelNumber];

    // Allocate an image to store the mean image
    nifti_image *meanImage = nifti_dup(*currentInputImage, false);
    DataType* meanImgDataPtr = static_cast<DataType*>(meanImage->data);

    // Allocate an image to store the shifted image
    nifti_image *shiftedImage = nifti_dup(*currentInputImage, false);

    // Allocation of the difference image
    nifti_image *diffImage = nifti_dup(*currentInputImage, false);

    // Define the sigma for the convolution
    float sigma = -0.5;// negative value denotes voxel width

    //2D version
    int samplingNbr = (currentInputImage->nz > 1) ? 6 : 4;
    int rSamplingX[6] = { -descriptorOffset, descriptorOffset, 0, 0, 0, 0 };
    int rSamplingY[6] = { 0, 0, -descriptorOffset, descriptorOffset, 0, 0 };
    int rSamplingZ[6] = { 0, 0, 0, 0, -descriptorOffset, descriptorOffset };

    for (int i = 0; i < samplingNbr; i++) {
        ShiftImage<DataType>(currentInputImage, shiftedImage, mask, rSamplingX[i], rSamplingY[i], rSamplingZ[i]);
        reg_tools_subtractImageFromImage(currentInputImage, shiftedImage, diffImage);
        reg_tools_multiplyImageToImage(diffImage, diffImage, diffImage);
        reg_tools_kernelConvolution(diffImage, &sigma, ConvKernelType::Gaussian, mask);
        reg_tools_addImageToImage(meanImage, diffImage, meanImage);
        // Store the current descriptor
        const size_t index = i * diffImage->nvox;
        memcpy(&mindImgDataPtr[index], diffImage->data, diffImage->nbyper * diffImage->nvox);
    }
    // Compute the mean over the number of sample
    reg_tools_divideValueToImage(meanImage, meanImage, samplingNbr);

    // Compute the MIND descriptor
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(voxelNumber, samplingNbr, mask, meanImgDataPtr, mindImgDataPtr)
#endif
    for (voxelIndex = 0; voxelIndex < voxelNumber; voxelIndex++) {
        if (mask[voxelIndex] > -1) {
            // Get the mean value for the current voxel
            DataType meanValue = meanImgDataPtr[voxelIndex];
            if (meanValue == 0)
                meanValue = std::numeric_limits<DataType>::epsilon();
            DataType maxDesc = 0;
            int mindIndex = voxelIndex;
            for (int t = 0; t < samplingNbr; t++) {
                const DataType descValue = exp(-mindImgDataPtr[mindIndex] / meanValue);
                mindImgDataPtr[mindIndex] = descValue;
                maxDesc = std::max(maxDesc, descValue);
                mindIndex += voxelNumber;
            }

            mindIndex = voxelIndex;
            for (int t = 0; t < samplingNbr; t++) {
                const DataType descValue = mindImgDataPtr[mindIndex];
                mindImgDataPtr[mindIndex] = descValue / maxDesc;
                mindIndex += voxelNumber;
            }
        } // mask
    } // voxIndex
    nifti_image_free(diffImage);
    nifti_image_free(shiftedImage);
    nifti_image_free(meanImage);
    currentInputImage->data = nullptr;
    nifti_image_free(currentInputImage);
}
/* *************************************************************** */
void GetMindImageDescriptor(const nifti_image *inputImage,
                            nifti_image *mindImage,
                            const int *mask,
                            const int descriptorOffset,
                            const int currentTimePoint) {
    if (inputImage->datatype != mindImage->datatype)
        NR_FATAL_ERROR("The input image and the MIND image must have the same datatype");
    std::visit([&](auto&& imgType) {
        using ImgType = std::decay_t<decltype(imgType)>;
        GetMindImageDescriptorCore<ImgType>(inputImage, mindImage, mask, descriptorOffset, currentTimePoint);
    }, NiftiImage::getFloatingDataType(inputImage));
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template <class DataType>
void GetMindSscImageDescriptorCore(const nifti_image *inputImage,
                                   nifti_image *mindSscImage,
                                   const int *mask,
                                   const int descriptorOffset,
                                   const int currentTimePoint) {
#ifdef WIN32
    long voxelIndex;
    const long voxelNumber = (long)NiftiImage::calcVoxelNumber(inputImage, 3);
#else
    size_t voxelIndex;
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(inputImage, 3);
#endif
    // Create a pointer to the descriptor image
    DataType* mindSscImgDataPtr = static_cast<DataType*>(mindSscImage->data);

    // Allocate an image to store the current time point reference image
    nifti_image *currentInputImage = nifti_copy_nim_info(inputImage);
    currentInputImage->ndim = currentInputImage->dim[0] = inputImage->nz > 1 ? 3 : 2;
    currentInputImage->nt = currentInputImage->dim[4] = 1;
    currentInputImage->nvox = voxelNumber;
    DataType *inputImagePtr = static_cast<DataType*>(inputImage->data);
    currentInputImage->data = &inputImagePtr[currentTimePoint * voxelNumber];

    // Allocate an image to store the mean image
    nifti_image *meanImg = nifti_dup(*currentInputImage, false);
    DataType* meanImgDataPtr = static_cast<DataType*>(meanImg->data);

    // Allocate an image to store the warped image
    nifti_image *shiftedImage = nifti_dup(*currentInputImage, false);

    // Define the sigma for the convolution
    const float sigma = -0.5; // negative value denotes voxel width

    // 2D version
    const int samplingNbr = (currentInputImage->nz > 1) ? 6 : 2;
    const int lengthDescriptor = (currentInputImage->nz > 1) ? 12 : 4;

    // Allocation of the difference image
    //std::vector<nifti_image *> vectNiftiImage;
    //for(int i=0;i<samplingNbr;i++) {
    nifti_image *diffImage = nifti_dup(*currentInputImage, false);
    int *maskDiffImage = (int*)calloc(diffImage->nvox, sizeof(int));

    nifti_image *diffImageShifted = nifti_dup(*currentInputImage, false);

    int rSamplingX[6] = { +descriptorOffset, +descriptorOffset, -descriptorOffset, +0, +descriptorOffset, +0 };
    int rSamplingY[6] = { +descriptorOffset, -descriptorOffset, +0, -descriptorOffset, +0, +descriptorOffset };
    int rSamplingZ[6] = { +0, +0, +descriptorOffset, +descriptorOffset, +descriptorOffset, +descriptorOffset };

    int tx[12] = { -descriptorOffset, +0, -descriptorOffset, +0, +0, +descriptorOffset, +0, +0, +0, -descriptorOffset, +0, +0 };
    int ty[12] = { +0, -descriptorOffset, +0, +descriptorOffset, +0, +0, +0, +descriptorOffset, +0, +0, +0, -descriptorOffset };
    int tz[12] = { +0, +0, +0, +0, -descriptorOffset, +0, -descriptorOffset, +0, -descriptorOffset, +0, -descriptorOffset, +0 };
    int compteurId = 0;

    for (int i = 0; i < samplingNbr; i++) {
        ShiftImage<DataType>(currentInputImage, shiftedImage, mask, rSamplingX[i], rSamplingY[i], rSamplingZ[i]);
        reg_tools_subtractImageFromImage(currentInputImage, shiftedImage, diffImage);
        reg_tools_multiplyImageToImage(diffImage, diffImage, diffImage);
        reg_tools_kernelConvolution(diffImage, &sigma, ConvKernelType::Gaussian, mask);

        for (int j = 0; j < 2; j++) {
            ShiftImage<DataType>(diffImage, diffImageShifted, maskDiffImage, tx[compteurId], ty[compteurId], tz[compteurId]);
            reg_tools_addImageToImage(meanImg, diffImageShifted, meanImg);
            // Store the current descriptor
            const size_t index = compteurId * diffImageShifted->nvox;
            memcpy(&mindSscImgDataPtr[index], diffImageShifted->data, diffImageShifted->nbyper * diffImageShifted->nvox);
            compteurId++;
        }
    }
    // Compute the mean over the number of sample
    reg_tools_divideValueToImage(meanImg, meanImg, lengthDescriptor);

    // Compute the MIND-SSC descriptor
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(voxelNumber, lengthDescriptor, mask, meanImgDataPtr, mindSscImgDataPtr)
#endif
    for (voxelIndex = 0; voxelIndex < voxelNumber; voxelIndex++) {
        if (mask[voxelIndex] > -1) {
            // Get the mean value for the current voxel
            DataType meanValue = meanImgDataPtr[voxelIndex];
            if (meanValue == 0)
                meanValue = std::numeric_limits<DataType>::epsilon();
            DataType maxDesc = 0;
            int mindIndex = voxelIndex;
            for (int t = 0; t < lengthDescriptor; t++) {
                const DataType descValue = exp(-mindSscImgDataPtr[mindIndex] / meanValue);
                mindSscImgDataPtr[mindIndex] = descValue;
                maxDesc = std::max(maxDesc, descValue);
                mindIndex += voxelNumber;
            }

            mindIndex = voxelIndex;
            for (int t = 0; t < lengthDescriptor; t++) {
                const DataType descValue = mindSscImgDataPtr[mindIndex];
                mindSscImgDataPtr[mindIndex] = descValue / maxDesc;
                mindIndex += voxelNumber;
            }
        } // mask
    } // voxIndex
    nifti_image_free(diffImageShifted);
    free(maskDiffImage);
    nifti_image_free(diffImage);
    nifti_image_free(shiftedImage);
    nifti_image_free(meanImg);
    currentInputImage->data = nullptr;
    nifti_image_free(currentInputImage);
}
/* *************************************************************** */
void GetMindSscImageDescriptor(const nifti_image *inputImage,
                               nifti_image *mindSscImage,
                               const int *mask,
                               const int descriptorOffset,
                               const int currentTimePoint) {
    if (inputImage->datatype != mindSscImage->datatype)
        NR_FATAL_ERROR("The input image and the MINDSSC image must have the same datatype!");
    std::visit([&](auto&& imgType) {
        using ImgType = std::decay_t<decltype(imgType)>;
        GetMindSscImageDescriptorCore<ImgType>(inputImage, mindSscImage, mask, descriptorOffset, currentTimePoint);
    }, NiftiImage::getFloatingDataType(inputImage));
    NR_FUNC_CALLED();
}
/* *************************************************************** */
reg_mind::reg_mind(): reg_ssd() {
    this->referenceImageDescriptor = nullptr;
    this->floatingImageDescriptor = nullptr;
    this->warpedFloatingImageDescriptor = nullptr;
    this->warpedReferenceImageDescriptor = nullptr;
    this->mindType = MIND_TYPE;
    this->descriptorOffset = 1;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
reg_mind::~reg_mind() {
    if (this->referenceImageDescriptor != nullptr) {
        nifti_image_free(this->referenceImageDescriptor);
        this->referenceImageDescriptor = nullptr;
    }
    if (this->warpedFloatingImageDescriptor != nullptr) {
        nifti_image_free(this->warpedFloatingImageDescriptor);
        this->warpedFloatingImageDescriptor = nullptr;
    }
    if (this->floatingImageDescriptor != nullptr) {
        nifti_image_free(this->floatingImageDescriptor);
        this->floatingImageDescriptor = nullptr;
    }
    if (this->warpedReferenceImageDescriptor != nullptr) {
        nifti_image_free(this->warpedReferenceImageDescriptor);
        this->warpedReferenceImageDescriptor = nullptr;
    }
}
/* *************************************************************** */
void reg_mind::InitialiseMeasure(nifti_image *refImg,
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
    reg_ssd::InitialiseMeasure(refImg,
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

    this->descriptorNumber = 0;
    if (this->mindType == MIND_TYPE) {
        this->descriptorNumber = this->referenceImage->nz > 1 ? 6 : 4;
    } else if (this->mindType == MINDSSC_TYPE) {
        this->descriptorNumber = this->referenceImage->nz > 1 ? 12 : 4;
    }
    // Initialise the reference descriptor
    this->referenceImageDescriptor = nifti_copy_nim_info(this->referenceImage);
    this->referenceImageDescriptor->dim[0] = this->referenceImageDescriptor->ndim = 4;
    this->referenceImageDescriptor->dim[4] = this->referenceImageDescriptor->nt = this->descriptorNumber;
    this->referenceImageDescriptor->nvox = NiftiImage::calcVoxelNumber(this->referenceImageDescriptor, this->referenceImageDescriptor->ndim);
    this->referenceImageDescriptor->data = malloc(this->referenceImageDescriptor->nvox * this->referenceImageDescriptor->nbyper);
    // Initialise the warped floating descriptor
    this->warpedFloatingImageDescriptor = nifti_copy_nim_info(this->referenceImage);
    this->warpedFloatingImageDescriptor->dim[0] = this->warpedFloatingImageDescriptor->ndim = 4;
    this->warpedFloatingImageDescriptor->dim[4] = this->warpedFloatingImageDescriptor->nt = this->descriptorNumber;
    this->warpedFloatingImageDescriptor->nvox = NiftiImage::calcVoxelNumber(this->warpedFloatingImageDescriptor,
                                                                            this->warpedFloatingImageDescriptor->ndim);
    this->warpedFloatingImageDescriptor->data = malloc(this->warpedFloatingImageDescriptor->nvox *
                                                       this->warpedFloatingImageDescriptor->nbyper);

    if (this->isSymmetric) {
        if (this->floatingImage->nt > 1 || this->warpedImageBw->nt > 1)
            NR_FATAL_ERROR("reg_mind does not support multiple time point image");
        // Initialise the floating descriptor
        this->floatingImageDescriptor = nifti_copy_nim_info(this->floatingImage);
        this->floatingImageDescriptor->dim[0] = this->floatingImageDescriptor->ndim = 4;
        this->floatingImageDescriptor->dim[4] = this->floatingImageDescriptor->nt = this->descriptorNumber;
        this->floatingImageDescriptor->nvox = NiftiImage::calcVoxelNumber(this->floatingImageDescriptor,
                                                                          this->floatingImageDescriptor->ndim);
        this->floatingImageDescriptor->data = malloc(this->floatingImageDescriptor->nvox *
                                                     this->floatingImageDescriptor->nbyper);
        // Initialise the warped floating descriptor
        this->warpedReferenceImageDescriptor = nifti_copy_nim_info(this->floatingImage);
        this->warpedReferenceImageDescriptor->dim[0] = this->warpedReferenceImageDescriptor->ndim = 4;
        this->warpedReferenceImageDescriptor->dim[4] = this->warpedReferenceImageDescriptor->nt = this->descriptorNumber;
        this->warpedReferenceImageDescriptor->nvox = NiftiImage::calcVoxelNumber(this->warpedReferenceImageDescriptor,
                                                                                 this->warpedReferenceImageDescriptor->ndim);
        this->warpedReferenceImageDescriptor->data = malloc(this->warpedReferenceImageDescriptor->nvox *
                                                            this->warpedReferenceImageDescriptor->nbyper);
    }

    for (int i = 0; i < referenceImageDescriptor->nt; ++i) {
        this->timePointWeightsDescriptor[i] = 1.0;
    }

#ifndef NDEBUG
    std::string msg = "Active time point:";
    for (int i = 0; i < this->referenceImageDescriptor->nt; ++i)
        if (this->timePointWeightsDescriptor[i] > 0)
            msg += " " + std::to_string(i);
    NR_DEBUG(msg);
    NR_FUNC_CALLED();
#endif
}
/* *************************************************************** */
double GetSimilarityMeasureValue(nifti_image *referenceImage,
                                 nifti_image *referenceImageDescriptor,
                                 const int *referenceMask,
                                 nifti_image *warpedImage,
                                 nifti_image *warpedFloatingImageDescriptor,
                                 const double *timePointWeights,
                                 double *timePointWeightsDescriptor,
                                 nifti_image *jacobianDetImage,
                                 const int descriptorOffset,
                                 const int referenceTimePoints,
                                 const int mindType) {
    if (referenceImageDescriptor->datatype != NIFTI_TYPE_FLOAT32 &&
        referenceImageDescriptor->datatype != NIFTI_TYPE_FLOAT64)
        NR_FATAL_ERROR("The reference image descriptor is expected to be of floating precision type");

    double mind = 0;
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);
    unique_ptr<int[]> combinedMask(new int[voxelNumber]);
    auto GetMindImgDesc = mindType == MIND_TYPE ? GetMindImageDescriptor : GetMindSscImageDescriptor;

    for (int currentTimePoint = 0; currentTimePoint < referenceTimePoints; ++currentTimePoint) {
        if (timePointWeights[currentTimePoint] > 0) {
            memcpy(combinedMask.get(), referenceMask, voxelNumber * sizeof(int));
            reg_tools_removeNanFromMask(referenceImage, combinedMask.get());
            reg_tools_removeNanFromMask(warpedImage, combinedMask.get());

            GetMindImgDesc(referenceImage, referenceImageDescriptor, combinedMask.get(), descriptorOffset, currentTimePoint);
            GetMindImgDesc(warpedImage, warpedFloatingImageDescriptor, combinedMask.get(), descriptorOffset, currentTimePoint);

            std::visit([&](auto&& refImgDataType) {
                using RefImgDataType = std::decay_t<decltype(refImgDataType)>;
                mind += reg_getSsdValue<RefImgDataType>(referenceImageDescriptor,
                                                        warpedFloatingImageDescriptor,
                                                        timePointWeightsDescriptor,
                                                        referenceTimePoints,
                                                        jacobianDetImage,
                                                        combinedMask.get(),
                                                        nullptr);
            }, NiftiImage::getFloatingDataType(referenceImageDescriptor));
        }
    }
    return mind;
}
/* *************************************************************** */
double reg_mind::GetSimilarityMeasureValueFw() {
    return ::GetSimilarityMeasureValue(this->referenceImage,
                                       this->referenceImageDescriptor,
                                       this->referenceMask,
                                       this->warpedImage,
                                       this->warpedFloatingImageDescriptor,
                                       this->timePointWeights,
                                       this->timePointWeightsDescriptor,
                                       nullptr, // TODO this->forwardJacDetImagePointer,
                                       this->descriptorOffset,
                                       this->referenceTimePoints,
                                       this->mindType);
}
/* *************************************************************** */
double reg_mind::GetSimilarityMeasureValueBw() {
    return ::GetSimilarityMeasureValue(this->floatingImage,
                                       this->floatingImageDescriptor,
                                       this->floatingMask,
                                       this->warpedImageBw,
                                       this->warpedReferenceImageDescriptor,
                                       this->timePointWeights,
                                       this->timePointWeightsDescriptor,
                                       nullptr, // TODO this->backwardJacDetImagePointer,
                                       this->descriptorOffset,
                                       this->referenceTimePoints,
                                       this->mindType);
}
/* *************************************************************** */
void GetVoxelBasedSimilarityMeasureGradient(nifti_image *referenceImage,
                                            nifti_image *referenceImageDescriptor,
                                            const int *referenceMask,
                                            nifti_image *warpedImage,
                                            nifti_image *warpedGradient,
                                            nifti_image *warpedFloatingImageDescriptor,
                                            nifti_image *voxelBasedGradient,
                                            const int mindType,
                                            const int descriptorOffset,
                                            const int descriptorNumber,
                                            const int currentTimePoint) {
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);
    vector<int> combinedMask(referenceMask, referenceMask + voxelNumber);
    reg_tools_removeNanFromMask(referenceImage, combinedMask.data());
    reg_tools_removeNanFromMask(warpedImage, combinedMask.data());

    auto GetMindImgDesc = mindType == MIND_TYPE ? GetMindImageDescriptor : GetMindSscImageDescriptor;
    // Compute the reference image descriptors
    GetMindImgDesc(referenceImage, referenceImageDescriptor, combinedMask.data(), descriptorOffset, currentTimePoint);
    // Compute the warped floating image descriptors
    GetMindImgDesc(warpedImage, warpedFloatingImageDescriptor, combinedMask.data(), descriptorOffset, currentTimePoint);

    for (int descIndex = 0; descIndex < descriptorNumber; ++descIndex) {
        // Compute the warped image descriptors gradient
        reg_getImageGradient_symDiff(warpedFloatingImageDescriptor,
                                     warpedGradient,
                                     combinedMask.data(),
                                     std::numeric_limits<float>::quiet_NaN(),
                                     descIndex);

        // Compute the gradient of the ssd for the forward transformation
        std::visit([&](auto&& refDescDataType) {
            using RefDescDataType = std::decay_t<decltype(refDescDataType)>;
            reg_getVoxelBasedSsdGradient<RefDescDataType>(referenceImageDescriptor,
                                                          warpedFloatingImageDescriptor,
                                                          warpedGradient,
                                                          voxelBasedGradient,
                                                          nullptr, // no Jacobian required here
                                                          combinedMask.data(),
                                                          descIndex,
                                                          1.0,  // all descriptors given weight of 1
                                                          nullptr);
        }, NiftiImage::getFloatingDataType(referenceImageDescriptor));
    }
}
/* *************************************************************** */
void reg_mind::GetVoxelBasedSimilarityMeasureGradientFw(int currentTimePoint) {
    ::GetVoxelBasedSimilarityMeasureGradient(this->referenceImage,
                                             this->referenceImageDescriptor,
                                             this->referenceMask,
                                             this->warpedImage,
                                             this->warpedGradient,
                                             this->warpedFloatingImageDescriptor,
                                             this->voxelBasedGradient,
                                             this->mindType,
                                             this->descriptorOffset,
                                             this->descriptorNumber,
                                             currentTimePoint);
}
/* *************************************************************** */
void reg_mind::GetVoxelBasedSimilarityMeasureGradientBw(int currentTimePoint) {
    ::GetVoxelBasedSimilarityMeasureGradient(this->floatingImage,
                                             this->floatingImageDescriptor,
                                             this->floatingMask,
                                             this->warpedImageBw,
                                             this->warpedGradientBw,
                                             this->warpedReferenceImageDescriptor,
                                             this->voxelBasedGradientBw,
                                             this->mindType,
                                             this->descriptorOffset,
                                             this->descriptorNumber,
                                             currentTimePoint);
}
/* *************************************************************** */
reg_mindssc::reg_mindssc(): reg_mind() {
    this->mindType = MINDSSC_TYPE;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
reg_mindssc::~reg_mindssc() {
    NR_FUNC_CALLED();
}
/* *************************************************************** */
