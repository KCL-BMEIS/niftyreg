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
                const int& tx,
                const int& ty,
                const int& tz) {
    const DataType* inputData = static_cast<DataType*>(inputImage->data);
    DataType* shiftImageData = static_cast<DataType*>(shiftedImage->data);

    int currentIndex;
    int shiftedIndex;

    int x, y, z, old_x, old_y, old_z;

#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(inputData, shiftImageData, shiftedImage, inputImage, mask, tx, ty, tz) \
    private(x, y, old_x, old_y, old_z, shiftedIndex, currentIndex)
#endif
    for (z = 0; z < shiftedImage->nz; z++) {
        currentIndex = z * shiftedImage->nx * shiftedImage->ny;
        old_z = z - tz;
        for (y = 0; y < shiftedImage->ny; y++) {
            old_y = y - ty;
            for (x = 0; x < shiftedImage->nx; x++) {
                old_x = x - tx;
                if (old_x > -1 && old_x < inputImage->nx &&
                    old_y > -1 && old_y < inputImage->ny &&
                    old_z > -1 && old_z < inputImage->nz) {
                    shiftedIndex = (old_z * inputImage->ny + old_y) * inputImage->nx + old_x;
                    if (mask[shiftedIndex] > -1) {
                        shiftImageData[currentIndex] = inputData[shiftedIndex];
                    } // mask is not defined
                    else {
                        //shiftImageData[currentIndex]=std::numeric_limits<DataType>::quiet_NaN();
                        shiftImageData[currentIndex] = 0;
                    }
                } // outside of the image
                else {
                    //shiftImageData[currentIndex]=std::numeric_limits<DataType>::quiet_NaN();
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
                                const int& descriptorOffset,
                                const int& currentTimepoint) {
#ifdef WIN32
    long voxelIndex;
    const long voxelNumber = (long)NiftiImage::calcVoxelNumber(inputImage, 3);
#else
    size_t voxelIndex;
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(inputImage, 3);
#endif

    // Create a pointer to the descriptor image
    DataType* mindImgDataPtr = static_cast<DataType*>(mindImage->data);

    // Allocate an image to store the current timepoint reference image
    nifti_image *currentInputImage = nifti_copy_nim_info(inputImage);
    currentInputImage->ndim = currentInputImage->dim[0] = inputImage->nz > 1 ? 3 : 2;
    currentInputImage->nt = currentInputImage->dim[4] = 1;
    currentInputImage->nvox = voxelNumber;
    DataType *inputImagePtr = static_cast<DataType*>(inputImage->data);
    currentInputImage->data = &inputImagePtr[currentTimepoint * voxelNumber];

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
        reg_tools_kernelConvolution(diffImage, &sigma, GAUSSIAN_KERNEL, mask);
        reg_tools_addImageToImage(meanImage, diffImage, meanImage);

        // Store the current descriptor
        const size_t index = i * diffImage->nvox;
        memcpy(&mindImgDataPtr[index], diffImage->data, diffImage->nbyper * diffImage->nvox);
    }
    // Compute the mean over the number of sample
    reg_tools_divideValueToImage(meanImage, meanImage, samplingNbr);

    // Compute the MIND descriptor
    int mindIndex;
    DataType meanValue, maxDesc, descValue;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(voxelNumber, samplingNbr, mask, meanImgDataPtr, \
    mindImgDataPtr) \
    private(meanValue, maxDesc, descValue, mindIndex)
#endif
    for (voxelIndex = 0; voxelIndex < voxelNumber; voxelIndex++) {
        if (mask[voxelIndex] > -1) {
            // Get the mean value for the current voxel
            meanValue = meanImgDataPtr[voxelIndex];
            if (meanValue == 0) {
                meanValue = std::numeric_limits<DataType>::epsilon();
            }
            maxDesc = 0;
            mindIndex = voxelIndex;
            for (int t = 0; t < samplingNbr; t++) {
                descValue = (DataType)exp(-mindImgDataPtr[mindIndex] / meanValue);
                mindImgDataPtr[mindIndex] = descValue;
                maxDesc = std::max(maxDesc, descValue);
                mindIndex += voxelNumber;
            }

            mindIndex = voxelIndex;
            for (int t = 0; t < samplingNbr; t++) {
                descValue = mindImgDataPtr[mindIndex];
                mindImgDataPtr[mindIndex] = descValue / maxDesc;
                mindIndex += voxelNumber;
            }
        } // mask
    } // voxIndex
    // Mr Propre
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
                            const int& descriptorOffset,
                            const int& currentTimepoint) {
#ifndef NDEBUG
    reg_print_fct_debug("GetMindImageDescriptor()");
#endif
    if (inputImage->datatype != mindImage->datatype) {
        reg_print_fct_error("reg_mind::GetMindImageDescriptor");
        reg_print_msg_error("The input image and the MIND image must have the same datatype !");
        reg_exit();
    }

    switch (inputImage->datatype) {
    case NIFTI_TYPE_FLOAT32:
        GetMindImageDescriptorCore<float>(inputImage, mindImage, mask, descriptorOffset, currentTimepoint);
        break;
    case NIFTI_TYPE_FLOAT64:
        GetMindImageDescriptorCore<double>(inputImage, mindImage, mask, descriptorOffset, currentTimepoint);
        break;
    default:
        reg_print_fct_error("GetMindImageDescriptor");
        reg_print_msg_error("Input image datatype not supported");
        reg_exit();
        break;
    }
}
/* *************************************************************** */
template <class DataType>
void GetMindSscImageDescriptorCore(const nifti_image *inputImage,
                                   nifti_image *mindSscImage,
                                   const int *mask,
                                   const int& descriptorOffset,
                                   const int& currentTimepoint) {
#ifdef WIN32
    long voxelIndex;
    const long voxelNumber = (long)NiftiImage::calcVoxelNumber(inputImage, 3);
#else
    size_t voxelIndex;
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(inputImage, 3);
#endif

    // Create a pointer to the descriptor image
    DataType* mindSscImgDataPtr = static_cast<DataType*>(mindSscImage->data);

    // Allocate an image to store the current timepoint reference image
    nifti_image *currentInputImage = nifti_copy_nim_info(inputImage);
    currentInputImage->ndim = currentInputImage->dim[0] = inputImage->nz > 1 ? 3 : 2;
    currentInputImage->nt = currentInputImage->dim[4] = 1;
    currentInputImage->nvox = voxelNumber;
    DataType *inputImagePtr = static_cast<DataType*>(inputImage->data);
    currentInputImage->data = &inputImagePtr[currentTimepoint * voxelNumber];

    // Allocate an image to store the mean image
    nifti_image *meanImg = nifti_dup(*currentInputImage, false);
    DataType* meanImgDataPtr = static_cast<DataType*>(meanImg->data);

    // Allocate an image to store the warped image
    nifti_image *shiftedImage = nifti_dup(*currentInputImage, false);

    // Define the sigma for the convolution
    float sigma = -0.5; // negative value denotes voxel width

    //2D version
    int samplingNbr = (currentInputImage->nz > 1) ? 6 : 2;
    int lengthDescriptor = (currentInputImage->nz > 1) ? 12 : 4;

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
        reg_tools_kernelConvolution(diffImage, &sigma, GAUSSIAN_KERNEL, mask);

        for (int j = 0; j < 2; j++) {
            ShiftImage<DataType>(diffImage, diffImageShifted, maskDiffImage,
                                 tx[compteurId], ty[compteurId], tz[compteurId]);

            reg_tools_addImageToImage(meanImg, diffImageShifted, meanImg);
            // Store the current descriptor
            const size_t index = compteurId * diffImageShifted->nvox;
            memcpy(&mindSscImgDataPtr[index], diffImageShifted->data,
                   diffImageShifted->nbyper * diffImageShifted->nvox);
            compteurId++;
        }
    }
    // Compute the mean over the number of sample
    reg_tools_divideValueToImage(meanImg, meanImg, lengthDescriptor);

    // Compute the MIND-SSC descriptor
    int mindIndex;
    DataType meanValue, maxDesc, descValue;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(voxelNumber, lengthDescriptor, samplingNbr, mask, meanImgDataPtr, mindSscImgDataPtr) \
    private(meanValue, maxDesc, descValue, mindIndex)
#endif
    for (voxelIndex = 0; voxelIndex < voxelNumber; voxelIndex++) {
        if (mask[voxelIndex] > -1) {
            // Get the mean value for the current voxel
            meanValue = meanImgDataPtr[voxelIndex];
            if (meanValue == 0) {
                meanValue = std::numeric_limits<DataType>::epsilon();
            }
            maxDesc = 0;
            mindIndex = voxelIndex;
            for (int t = 0; t < lengthDescriptor; t++) {
                descValue = (DataType)exp(-mindSscImgDataPtr[mindIndex] / meanValue);
                mindSscImgDataPtr[mindIndex] = descValue;
                maxDesc = std::max(maxDesc, descValue);
                mindIndex += voxelNumber;
            }

            mindIndex = voxelIndex;
            for (int t = 0; t < lengthDescriptor; t++) {
                descValue = mindSscImgDataPtr[mindIndex];
                mindSscImgDataPtr[mindIndex] = descValue / maxDesc;
                mindIndex += voxelNumber;
            }
        } // mask
    } // voxIndex
    // Mr Propre
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
                               const int& descriptorOffset,
                               const int& currentTimepoint) {
#ifndef NDEBUG
    reg_print_fct_debug("GetMindSscImageDescriptor()");
#endif
    if (inputImage->datatype != mindSscImage->datatype) {
        reg_print_fct_error("reg_mindssc::GetMindSscImageDescriptor");
        reg_print_msg_error("The input image and the MINDSSC image must have the same datatype!");
        reg_exit();
    }

    switch (inputImage->datatype) {
    case NIFTI_TYPE_FLOAT32:
        GetMindSscImageDescriptorCore<float>(inputImage, mindSscImage, mask, descriptorOffset, currentTimepoint);
        break;
    case NIFTI_TYPE_FLOAT64:
        GetMindSscImageDescriptorCore<double>(inputImage, mindSscImage, mask, descriptorOffset, currentTimepoint);
        break;
    default:
        reg_print_fct_error("GetMindSscImageDescriptor");
        reg_print_msg_error("Input image datatype not supported");
        reg_exit();
        break;
    }
}
/* *************************************************************** */
reg_mind::reg_mind(): reg_ssd() {
    this->referenceImageDescriptor = nullptr;
    this->floatingImageDescriptor = nullptr;
    this->warpedFloatingImageDescriptor = nullptr;
    this->warpedReferenceImageDescriptor = nullptr;
    this->mindType = MIND_TYPE;
    this->descriptorOffset = 1;
#ifndef NDEBUG
    reg_print_msg_debug("reg_mind constructor called");
#endif
}
/* *************************************************************** */
void reg_mind::SetDescriptorOffset(int val) {
    this->descriptorOffset = val;
}
/* *************************************************************** */
int reg_mind::GetDescriptorOffset() {
    return this->descriptorOffset;
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
        if (this->floatingImage->nt > 1 || this->warpedImageBw->nt > 1) {
            reg_print_msg_error("reg_mind does not support multiple time point image");
            reg_exit();
        }
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
        this->timePointWeightDescriptor[i] = 1.0;
    }

#ifndef NDEBUG
    char text[255];
    reg_print_msg_debug("reg_mind::InitialiseMeasure()");
    sprintf(text, "Active time point:");
    for (int i = 0; i < this->referenceImageDescriptor->nt; ++i)
        if (this->timePointWeightDescriptor[i] > 0)
            sprintf(text, "%s %i", text, i);
    reg_print_msg_debug(text);
#endif
}
/* *************************************************************** */
double GetSimilarityMeasureValue(nifti_image *referenceImage,
                                 nifti_image *referenceImageDescriptor,
                                 const int *referenceMask,
                                 nifti_image *warpedImage,
                                 nifti_image *warpedFloatingImageDescriptor,
                                 const double *timePointWeight,
                                 double *timePointWeightDescriptor,
                                 nifti_image *jacobianDetImage,
                                 float *currentValue,
                                 int descriptorOffset,
                                 const int& referenceTimePoint,
                                 const int& mindType) {
    if (referenceImageDescriptor->datatype != NIFTI_TYPE_FLOAT32 &&
        referenceImageDescriptor->datatype != NIFTI_TYPE_FLOAT64) {
        reg_print_fct_error("reg_mind::GetSimilarityMeasureValue");
        reg_print_msg_error("The reference image descriptor is expected to be of floating precision type");
        reg_exit();
    }

    double mind = 0;
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);
    unique_ptr<int[]> combinedMask(new int[voxelNumber]);
    auto GetMindImgDesc = mindType == MIND_TYPE ? GetMindImageDescriptor : GetMindSscImageDescriptor;

    for (int currentTimepoint = 0; currentTimepoint < referenceTimePoint; ++currentTimepoint) {
        if (timePointWeight[currentTimepoint] > 0) {
            memcpy(combinedMask.get(), referenceMask, voxelNumber * sizeof(int));
            reg_tools_removeNanFromMask(referenceImage, combinedMask.get());
            reg_tools_removeNanFromMask(warpedImage, combinedMask.get());

            GetMindImgDesc(referenceImage, referenceImageDescriptor, combinedMask.get(), descriptorOffset, currentTimepoint);
            GetMindImgDesc(warpedImage, warpedFloatingImageDescriptor, combinedMask.get(), descriptorOffset, currentTimepoint);

            std::visit([&](auto&& refImgDataType) {
                using RefImgDataType = std::decay_t<decltype(refImgDataType)>;
                mind += reg_getSsdValue<RefImgDataType>(referenceImageDescriptor,
                                                        warpedFloatingImageDescriptor,
                                                        timePointWeightDescriptor,
                                                        jacobianDetImage,
                                                        combinedMask.get(),
                                                        currentValue,
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
                                       this->timePointWeight,
                                       this->timePointWeightDescriptor,
                                       nullptr, // TODO this->forwardJacDetImagePointer,
                                       this->currentValue,
                                       this->descriptorOffset,
                                       this->referenceTimePoint,
                                       this->mindType);
}
/* *************************************************************** */
double reg_mind::GetSimilarityMeasureValueBw() {
    return ::GetSimilarityMeasureValue(this->floatingImage,
                                       this->floatingImageDescriptor,
                                       this->floatingMask,
                                       this->warpedImageBw,
                                       this->warpedReferenceImageDescriptor,
                                       this->timePointWeight,
                                       this->timePointWeightDescriptor,
                                       nullptr, // TODO this->backwardJacDetImagePointer,
                                       this->currentValue,
                                       this->descriptorOffset,
                                       this->referenceTimePoint,
                                       this->mindType);
}
/* *************************************************************** */
void reg_mind::GetVoxelBasedSimilarityMeasureGradient(int currentTimepoint) {
    // Check if the specified time point exists and is active
    reg_measure::GetVoxelBasedSimilarityMeasureGradient(currentTimepoint);
    if (this->timePointWeight[currentTimepoint] == 0)
        return;

    // Create a combined mask to ignore masked and undefined values
    size_t voxelNumber = NiftiImage::calcVoxelNumber(this->referenceImage, 3);
    int *combinedMask = (int*)malloc(voxelNumber * sizeof(int));
    memcpy(combinedMask, this->referenceMask, voxelNumber * sizeof(int));
    reg_tools_removeNanFromMask(this->referenceImage, combinedMask);
    reg_tools_removeNanFromMask(this->warpedImage, combinedMask);

    if (this->mindType == MIND_TYPE) {
        // Compute the reference image descriptors
        GetMindImageDescriptor(this->referenceImage,
                               this->referenceImageDescriptor,
                               combinedMask,
                               this->descriptorOffset,
                               currentTimepoint);
        // Compute the warped floating image descriptors
        GetMindImageDescriptor(this->warpedImage,
                               this->warpedFloatingImageDescriptor,
                               combinedMask,
                               this->descriptorOffset,
                               currentTimepoint);
    } else if (this->mindType == MINDSSC_TYPE) {
        // Compute the reference image descriptors
        GetMindSscImageDescriptor(this->referenceImage,
                                  this->referenceImageDescriptor,
                                  combinedMask,
                                  this->descriptorOffset,
                                  currentTimepoint);
        // Compute the warped floating image descriptors
        GetMindSscImageDescriptor(this->warpedImage,
                                  this->warpedFloatingImageDescriptor,
                                  combinedMask,
                                  this->descriptorOffset,
                                  currentTimepoint);
    }


    for (int desc_index = 0; desc_index < this->descriptorNumber; ++desc_index) {
        // Compute the warped image descriptors gradient
        reg_getImageGradient_symDiff(this->warpedFloatingImageDescriptor,
                                     this->warpedGradient,
                                     combinedMask,
                                     std::numeric_limits<float>::quiet_NaN(),
                                     desc_index);

        // Compute the gradient of the ssd for the forward transformation
        switch (referenceImageDescriptor->datatype) {
        case NIFTI_TYPE_FLOAT32:
            reg_getVoxelBasedSsdGradient<float>(this->referenceImageDescriptor,
                                                this->warpedFloatingImageDescriptor,
                                                this->warpedGradient,
                                                this->voxelBasedGradient,
                                                nullptr, // no Jacobian required here,
                                                combinedMask,
                                                desc_index,
                                                1.0, //all descriptors given weight of 1
                                                nullptr);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_getVoxelBasedSsdGradient<double>(this->referenceImageDescriptor,
                                                 this->warpedFloatingImageDescriptor,
                                                 this->warpedGradient,
                                                 this->voxelBasedGradient,
                                                 nullptr, // no Jacobian required here,
                                                 combinedMask,
                                                 desc_index,
                                                 1.0, //all descriptors given weight of 1
                                                 nullptr);
            break;
        default:
            reg_print_fct_error("reg_mind::GetVoxelBasedSimilarityMeasureGradient");
            reg_print_msg_error("Unsupported datatype");
            reg_exit();
        }
    }
    free(combinedMask);

    // Compute the gradient of the ssd for the backward transformation
    if (this->isSymmetric) {
        voxelNumber = NiftiImage::calcVoxelNumber(floatingImage, 3);
        combinedMask = (int*)malloc(voxelNumber * sizeof(int));
        memcpy(combinedMask, this->floatingMask, voxelNumber * sizeof(int));
        reg_tools_removeNanFromMask(this->floatingImage, combinedMask);
        reg_tools_removeNanFromMask(this->warpedImageBw, combinedMask);

        if (this->mindType == MIND_TYPE) {
            GetMindImageDescriptor(this->floatingImage,
                                   this->floatingImageDescriptor,
                                   combinedMask,
                                   this->descriptorOffset,
                                   currentTimepoint);
            GetMindImageDescriptor(this->warpedImageBw,
                                   this->warpedReferenceImageDescriptor,
                                   combinedMask,
                                   this->descriptorOffset,
                                   currentTimepoint);
        } else if (this->mindType == MINDSSC_TYPE) {
            GetMindSscImageDescriptor(this->floatingImage,
                                      this->floatingImageDescriptor,
                                      combinedMask,
                                      this->descriptorOffset,
                                      currentTimepoint);
            GetMindSscImageDescriptor(this->warpedImageBw,
                                      this->warpedReferenceImageDescriptor,
                                      combinedMask,
                                      this->descriptorOffset,
                                      currentTimepoint);
        }

        for (int desc_index = 0; desc_index < this->descriptorNumber; ++desc_index) {
            reg_getImageGradient_symDiff(this->warpedReferenceImageDescriptor,
                                         this->warpedGradientBw,
                                         combinedMask,
                                         std::numeric_limits<float>::quiet_NaN(),
                                         desc_index);

            // Compute the gradient of the nmi for the backward transformation
            switch (floatingImage->datatype) {
            case NIFTI_TYPE_FLOAT32:
                reg_getVoxelBasedSsdGradient<float>(this->floatingImageDescriptor,
                                                    this->warpedReferenceImageDescriptor,
                                                    this->warpedGradientBw,
                                                    this->voxelBasedGradientBw,
                                                    nullptr, // no Jacobian required here,
                                                    combinedMask,
                                                    desc_index,
                                                    1.0, //all descriptors given weight of 1
                                                    nullptr);
                break;
            case NIFTI_TYPE_FLOAT64:
                reg_getVoxelBasedSsdGradient<double>(this->floatingImageDescriptor,
                                                     this->warpedReferenceImageDescriptor,
                                                     this->warpedGradientBw,
                                                     this->voxelBasedGradientBw,
                                                     nullptr, // no Jacobian required here,
                                                     combinedMask,
                                                     desc_index,
                                                     1.0, //all descriptors given weight of 1
                                                     nullptr);
                break;
            default:
                reg_print_fct_error("reg_mind::GetVoxelBasedSimilarityMeasureGradient");
                reg_print_msg_error("Unsupported datatype");
                reg_exit();
            }
        }
        free(combinedMask);
    }
}
/* *************************************************************** */
reg_mindssc::reg_mindssc(): reg_mind() {
    this->mindType = MINDSSC_TYPE;
#ifndef NDEBUG
    reg_print_msg_debug("reg_mindssc constructor called");
#endif
}
/* *************************************************************** */
reg_mindssc::~reg_mindssc() {
#ifndef NDEBUG
    reg_print_msg_debug("reg_mindssc destructor called");
#endif
}
/* *************************************************************** */
