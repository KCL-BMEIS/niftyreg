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
void ShiftImage(nifti_image* inputImgPtr,
                nifti_image* shiftedImgPtr,
                int *maskPtr,
                int tx,
                int ty,
                int tz) {
    DataType* inputData = static_cast<DataType*>(inputImgPtr->data);
    DataType* shiftImageData = static_cast<DataType*>(shiftedImgPtr->data);

    int currentIndex;
    int shiftedIndex;

    int x, y, z, old_x, old_y, old_z;

#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(inputData, shiftImageData, shiftedImgPtr, inputImgPtr, \
    maskPtr, tx, ty, tz) \
    private(x, y, old_x, old_y, old_z, shiftedIndex, \
    currentIndex)
#endif
    for (z = 0; z < shiftedImgPtr->nz; z++) {
        currentIndex = z * shiftedImgPtr->nx * shiftedImgPtr->ny;
        old_z = z - tz;
        for (y = 0; y < shiftedImgPtr->ny; y++) {
            old_y = y - ty;
            for (x = 0; x < shiftedImgPtr->nx; x++) {
                old_x = x - tx;
                if (old_x > -1 && old_x<inputImgPtr->nx &&
                    old_y>-1 && old_y<inputImgPtr->ny &&
                    old_z>-1 && old_z < inputImgPtr->nz) {
                    shiftedIndex = (old_z * inputImgPtr->ny + old_y) * inputImgPtr->nx + old_x;
                    if (maskPtr[shiftedIndex] > -1) {
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
void GetMINDImageDescriptor_core(nifti_image* inputImage,
                                 nifti_image* MINDImage,
                                 int *maskPtr,
                                 int descriptorOffset,
                                 int currentTimepoint) {
#ifdef WIN32
    long voxelIndex;
    const long voxelNumber = (long)NiftiImage::calcVoxelNumber(inputImage, 3);
#else
    size_t voxelIndex;
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(inputImage, 3);
#endif

    // Create a pointer to the descriptor image
    DataType* MINDImgDataPtr = static_cast<DataType*>(MINDImage->data);

    // Allocate an image to store the current timepoint reference image
    nifti_image *currentInputImage = nifti_copy_nim_info(inputImage);
    currentInputImage->ndim = currentInputImage->dim[0] = inputImage->nz > 1 ? 3 : 2;
    currentInputImage->nt = currentInputImage->dim[4] = 1;
    currentInputImage->nvox = voxelNumber;
    DataType *inputImagePtr = static_cast<DataType*>(inputImage->data);
    currentInputImage->data = static_cast<void*>(&inputImagePtr[currentTimepoint * voxelNumber]);

    // Allocate an image to store the mean image
    nifti_image *meanImage = nifti_dup(*currentInputImage, false);
    DataType* meanImgDataPtr = static_cast<DataType*>(meanImage->data);

    // Allocate an image to store the shifted image
    nifti_image *shiftedImage = nifti_dup(*currentInputImage, false);

    // Allocation of the difference image
    nifti_image *diff_image = nifti_dup(*currentInputImage, false);

    // Define the sigma for the convolution
    float sigma = -0.5;// negative value denotes voxel width

    //2D version
    int samplingNbr = (currentInputImage->nz > 1) ? 6 : 4;
    int RSampling3D_x[6] = { -descriptorOffset, descriptorOffset, 0, 0, 0, 0 };
    int RSampling3D_y[6] = { 0, 0, -descriptorOffset, descriptorOffset, 0, 0 };
    int RSampling3D_z[6] = { 0, 0, 0, 0, -descriptorOffset, descriptorOffset };

    for (int i = 0; i < samplingNbr; i++) {
        ShiftImage<DataType>(currentInputImage, shiftedImage, maskPtr,
                             RSampling3D_x[i], RSampling3D_y[i], RSampling3D_z[i]);
        reg_tools_subtractImageFromImage(currentInputImage, shiftedImage, diff_image);
        reg_tools_multiplyImageToImage(diff_image, diff_image, diff_image);
        reg_tools_kernelConvolution(diff_image, &sigma, GAUSSIAN_KERNEL, maskPtr);
        reg_tools_addImageToImage(meanImage, diff_image, meanImage);

        // Store the current descriptor
        const size_t index = i * diff_image->nvox;
        memcpy(&MINDImgDataPtr[index], diff_image->data, diff_image->nbyper * diff_image->nvox);
    }
    // Compute the mean over the number of sample
    reg_tools_divideValueToImage(meanImage, meanImage, samplingNbr);

    // Compute the MIND descriptor
    int mindIndex;
    DataType meanValue, max_desc, descValue;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(voxelNumber, samplingNbr, maskPtr, meanImgDataPtr, \
    MINDImgDataPtr) \
    private(meanValue, max_desc, descValue, mindIndex)
#endif
    for (voxelIndex = 0; voxelIndex < voxelNumber; voxelIndex++) {
        if (maskPtr[voxelIndex] > -1) {
            // Get the mean value for the current voxel
            meanValue = meanImgDataPtr[voxelIndex];
            if (meanValue == 0) {
                meanValue = std::numeric_limits<DataType>::epsilon();
            }
            max_desc = 0;
            mindIndex = voxelIndex;
            for (int t = 0; t < samplingNbr; t++) {
                descValue = (DataType)exp(-MINDImgDataPtr[mindIndex] / meanValue);
                MINDImgDataPtr[mindIndex] = descValue;
                max_desc = (std::max)(max_desc, descValue);
                mindIndex += voxelNumber;
            }

            mindIndex = voxelIndex;
            for (int t = 0; t < samplingNbr; t++) {
                descValue = MINDImgDataPtr[mindIndex];
                MINDImgDataPtr[mindIndex] = descValue / max_desc;
                mindIndex += voxelNumber;
            }
        } // mask
    } // voxIndex
    // Mr Propre
    nifti_image_free(diff_image);
    nifti_image_free(shiftedImage);
    nifti_image_free(meanImage);
    currentInputImage->data = nullptr;
    nifti_image_free(currentInputImage);
}
/* *************************************************************** */
void GetMINDImageDescriptor(nifti_image* inputImgPtr,
                            nifti_image* MINDImgPtr,
                            int *maskPtr,
                            int descriptorOffset,
                            int currentTimepoint) {
#ifndef NDEBUG
    reg_print_fct_debug("GetMINDImageDescriptor()");
#endif
    if (inputImgPtr->datatype != MINDImgPtr->datatype) {
        reg_print_fct_error("reg_mind -- GetMINDImageDescriptor");
        reg_print_msg_error("The input image and the MIND image must have the same datatype !");
        reg_exit();
    }

    switch (inputImgPtr->datatype) {
    case NIFTI_TYPE_FLOAT32:
        GetMINDImageDescriptor_core<float>(inputImgPtr, MINDImgPtr, maskPtr, descriptorOffset, currentTimepoint);
        break;
    case NIFTI_TYPE_FLOAT64:
        GetMINDImageDescriptor_core<double>(inputImgPtr, MINDImgPtr, maskPtr, descriptorOffset, currentTimepoint);
        break;
    default:
        reg_print_fct_error("GetMINDImageDescriptor");
        reg_print_msg_error("Input image datatype not supported");
        reg_exit();
        break;
    }
}
/* *************************************************************** */
template <class DataType>
void GetMINDSSCImageDescriptor_core(nifti_image* inputImage,
                                    nifti_image* MINDSSCImage,
                                    int *maskPtr,
                                    int descriptorOffset,
                                    int currentTimepoint) {
#ifdef WIN32
    long voxelIndex;
    const long voxelNumber = (long)NiftiImage::calcVoxelNumber(inputImage, 3);
#else
    size_t voxelIndex;
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(inputImage, 3);
#endif

    // Create a pointer to the descriptor image
    DataType* MINDSSCImgDataPtr = static_cast<DataType*>(MINDSSCImage->data);

    // Allocate an image to store the current timepoint reference image
    nifti_image *currentInputImage = nifti_copy_nim_info(inputImage);
    currentInputImage->ndim = currentInputImage->dim[0] = inputImage->nz > 1 ? 3 : 2;
    currentInputImage->nt = currentInputImage->dim[4] = 1;
    currentInputImage->nvox = voxelNumber;
    DataType *inputImagePtr = static_cast<DataType*>(inputImage->data);
    currentInputImage->data = static_cast<void*>(&inputImagePtr[currentTimepoint * voxelNumber]);

    // Allocate an image to store the mean image
    nifti_image *mean_img = nifti_dup(*currentInputImage, false);
    DataType* meanImgDataPtr = static_cast<DataType*>(mean_img->data);

    // Allocate an image to store the warped image
    nifti_image *shiftedImage = nifti_dup(*currentInputImage, false);

    // Define the sigma for the convolution
    float sigma = -0.5;// negative value denotes voxel width
    //float sigma = -1.0;// negative value denotes voxel width

    //2D version
    int samplingNbr = (currentInputImage->nz > 1) ? 6 : 2;
    int lengthDescriptor = (currentInputImage->nz > 1) ? 12 : 4;

    // Allocation of the difference image
    //std::vector<nifti_image *> vectNiftiImage;
    //for(int i=0;i<samplingNbr;i++) {
    nifti_image *diff_image = nifti_dup(*currentInputImage, false);
    int *mask_diff_image = (int*)calloc(diff_image->nvox, sizeof(int));

    nifti_image *diff_imageShifted = nifti_dup(*currentInputImage, false);

    int RSampling3D_x[6] = { +descriptorOffset, +descriptorOffset, -descriptorOffset, +0, +descriptorOffset, +0 };
    int RSampling3D_y[6] = { +descriptorOffset, -descriptorOffset, +0, -descriptorOffset, +0, +descriptorOffset };
    int RSampling3D_z[6] = { +0, +0, +descriptorOffset, +descriptorOffset, +descriptorOffset, +descriptorOffset };

    int tx[12] = { -descriptorOffset, +0, -descriptorOffset, +0, +0, +descriptorOffset, +0, +0, +0, -descriptorOffset, +0, +0 };
    int ty[12] = { +0, -descriptorOffset, +0, +descriptorOffset, +0, +0, +0, +descriptorOffset, +0, +0, +0, -descriptorOffset };
    int tz[12] = { +0, +0, +0, +0, -descriptorOffset, +0, -descriptorOffset, +0, -descriptorOffset, +0, -descriptorOffset, +0 };
    int compteurId = 0;

    for (int i = 0; i < samplingNbr; i++) {
        ShiftImage<DataType>(currentInputImage, shiftedImage, maskPtr,
                             RSampling3D_x[i], RSampling3D_y[i], RSampling3D_z[i]);
        reg_tools_subtractImageFromImage(currentInputImage, shiftedImage, diff_image);
        reg_tools_multiplyImageToImage(diff_image, diff_image, diff_image);
        reg_tools_kernelConvolution(diff_image, &sigma, GAUSSIAN_KERNEL, maskPtr);

        for (int j = 0; j < 2; j++) {

            ShiftImage<DataType>(diff_image, diff_imageShifted, mask_diff_image,
                                 tx[compteurId], ty[compteurId], tz[compteurId]);

            reg_tools_addImageToImage(mean_img, diff_imageShifted, mean_img);
            // Store the current descriptor
            const size_t index = compteurId * diff_imageShifted->nvox;
            memcpy(&MINDSSCImgDataPtr[index], diff_imageShifted->data,
                   diff_imageShifted->nbyper * diff_imageShifted->nvox);
            compteurId++;
        }
    }
    // Compute the mean over the number of sample
    reg_tools_divideValueToImage(mean_img, mean_img, lengthDescriptor);

    // Compute the MINDSSC descriptor
    int mindIndex;
    DataType meanValue, max_desc, descValue;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(voxelNumber, lengthDescriptor, samplingNbr, maskPtr, meanImgDataPtr, \
    MINDSSCImgDataPtr) \
    private(meanValue, max_desc, descValue, mindIndex)
#endif
    for (voxelIndex = 0; voxelIndex < voxelNumber; voxelIndex++) {
        if (maskPtr[voxelIndex] > -1) {
            // Get the mean value for the current voxel
            meanValue = meanImgDataPtr[voxelIndex];
            if (meanValue == 0) {
                meanValue = std::numeric_limits<DataType>::epsilon();
            }
            max_desc = 0;
            mindIndex = voxelIndex;
            for (int t = 0; t < lengthDescriptor; t++) {
                descValue = (DataType)exp(-MINDSSCImgDataPtr[mindIndex] / meanValue);
                MINDSSCImgDataPtr[mindIndex] = descValue;
                max_desc = std::max(max_desc, descValue);
                mindIndex += voxelNumber;
            }

            mindIndex = voxelIndex;
            for (int t = 0; t < lengthDescriptor; t++) {
                descValue = MINDSSCImgDataPtr[mindIndex];
                MINDSSCImgDataPtr[mindIndex] = descValue / max_desc;
                mindIndex += voxelNumber;
            }
        } // mask
    } // voxIndex
    // Mr Propre
    nifti_image_free(diff_imageShifted);
    free(mask_diff_image);
    nifti_image_free(diff_image);
    nifti_image_free(shiftedImage);
    nifti_image_free(mean_img);
    currentInputImage->data = nullptr;
    nifti_image_free(currentInputImage);
}
/* *************************************************************** */
void GetMINDSSCImageDescriptor(nifti_image* inputImgPtr,
                               nifti_image* MINDSSCImgPtr,
                               int *maskPtr,
                               int descriptorOffset,
                               int currentTimepoint) {
#ifndef NDEBUG
    reg_print_fct_debug("GetMINDSSCImageDescriptor()");
#endif
    if (inputImgPtr->datatype != MINDSSCImgPtr->datatype) {
        reg_print_fct_error("reg_mindssc -- GetMINDSSCImageDescriptor");
        reg_print_msg_error("The input image and the MINDSSC image must have the same datatype !");
        reg_exit();
    }

    switch (inputImgPtr->datatype) {
    case NIFTI_TYPE_FLOAT32:
        GetMINDSSCImageDescriptor_core<float>(inputImgPtr, MINDSSCImgPtr, maskPtr, descriptorOffset, currentTimepoint);
        break;
    case NIFTI_TYPE_FLOAT64:
        GetMINDSSCImageDescriptor_core<double>(inputImgPtr, MINDSSCImgPtr, maskPtr, descriptorOffset, currentTimepoint);
        break;
    default:
        reg_print_fct_error("GetMINDSSCImageDescriptor");
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
    this->mind_type = MIND_TYPE;
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

    this->descriptor_number = 0;
    if (this->mind_type == MIND_TYPE) {
        descriptor_number = this->referenceImage->nz > 1 ? 6 : 4;
    } else if (this->mind_type == MINDSSC_TYPE) {
        descriptor_number = this->referenceImage->nz > 1 ? 12 : 4;

    }
    // Initialise the reference descriptor
    this->referenceImageDescriptor = nifti_copy_nim_info(this->referenceImage);
    this->referenceImageDescriptor->dim[0] = this->referenceImageDescriptor->ndim = 4;
    this->referenceImageDescriptor->dim[4] = this->referenceImageDescriptor->nt = this->descriptor_number;
    this->referenceImageDescriptor->nvox = NiftiImage::calcVoxelNumber(this->referenceImageDescriptor, this->referenceImageDescriptor->ndim);
    this->referenceImageDescriptor->data = malloc(this->referenceImageDescriptor->nvox * this->referenceImageDescriptor->nbyper);
    // Initialise the warped floating descriptor
    this->warpedFloatingImageDescriptor = nifti_copy_nim_info(this->referenceImage);
    this->warpedFloatingImageDescriptor->dim[0] = this->warpedFloatingImageDescriptor->ndim = 4;
    this->warpedFloatingImageDescriptor->dim[4] = this->warpedFloatingImageDescriptor->nt = this->descriptor_number;
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
        this->floatingImageDescriptor->dim[4] = this->floatingImageDescriptor->nt = this->descriptor_number;
        this->floatingImageDescriptor->nvox = NiftiImage::calcVoxelNumber(this->floatingImageDescriptor,
                                                                          this->floatingImageDescriptor->ndim);
        this->floatingImageDescriptor->data = malloc(this->floatingImageDescriptor->nvox *
                                                     this->floatingImageDescriptor->nbyper);
        // Initialise the warped floating descriptor
        this->warpedReferenceImageDescriptor = nifti_copy_nim_info(this->floatingImage);
        this->warpedReferenceImageDescriptor->dim[0] = this->warpedReferenceImageDescriptor->ndim = 4;
        this->warpedReferenceImageDescriptor->dim[4] = this->warpedReferenceImageDescriptor->nt = this->descriptor_number;
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
    reg_print_msg_debug("reg_mind::InitialiseMeasure().");
    sprintf(text, "Active time point:");
    for (int i = 0; i < this->referenceImageDescriptor->nt; ++i)
        if (this->timePointWeightDescriptor[i] > 0)
            sprintf(text, "%s %i", text, i);
    reg_print_msg_debug(text);
#endif
}
/* *************************************************************** */
double reg_mind::GetSimilarityMeasureValue() {
    double MINDValue = 0.;
    for (int t = 0; t < this->referenceImage->nt; ++t) {
        if (this->timePointWeight[t] > 0) {
            size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);
            int *combinedMask = (int*)malloc(voxelNumber * sizeof(int));
            memcpy(combinedMask, this->referenceMask, voxelNumber * sizeof(int));
            reg_tools_removeNanFromMask(this->referenceImage, combinedMask);
            reg_tools_removeNanFromMask(this->warpedImage, combinedMask);

            if (this->mind_type == MIND_TYPE) {
                GetMINDImageDescriptor(this->referenceImage,
                                       this->referenceImageDescriptor,
                                       combinedMask,
                                       this->descriptorOffset,
                                       t);
                GetMINDImageDescriptor(this->warpedImage,
                                       this->warpedFloatingImageDescriptor,
                                       combinedMask,
                                       this->descriptorOffset,
                                       t);
            } else if (this->mind_type == MINDSSC_TYPE) {
                GetMINDSSCImageDescriptor(this->referenceImage,
                                          this->referenceImageDescriptor,
                                          combinedMask,
                                          this->descriptorOffset,
                                          t);
                GetMINDSSCImageDescriptor(this->warpedImage,
                                          this->warpedFloatingImageDescriptor,
                                          combinedMask,
                                          this->descriptorOffset,
                                          t);
            }

            switch (this->referenceImageDescriptor->datatype) {
            case NIFTI_TYPE_FLOAT32:
                MINDValue += reg_getSSDValue<float>(this->referenceImageDescriptor,
                                                    this->warpedFloatingImageDescriptor,
                                                    this->timePointWeightDescriptor,
                                                    nullptr, // TODO this->forwardJacDetImagePointer,
                                                    combinedMask,
                                                    this->currentValue,
                                                    nullptr);
                break;
            case NIFTI_TYPE_FLOAT64:
                MINDValue += reg_getSSDValue<double>(this->referenceImageDescriptor,
                                                     this->warpedFloatingImageDescriptor,
                                                     this->timePointWeightDescriptor,
                                                     nullptr, // TODO this->forwardJacDetImagePointer,
                                                     combinedMask,
                                                     this->currentValue,
                                                     nullptr);
                break;
            default:
                reg_print_fct_error("reg_mind::GetSimilarityMeasureValue");
                reg_print_msg_error("Warped pixel type unsupported");
                reg_exit();
            }
            free(combinedMask);

            // Backward computation
            if (this->isSymmetric) {
                voxelNumber = NiftiImage::calcVoxelNumber(floatingImage, 3);
                combinedMask = (int*)malloc(voxelNumber * sizeof(int));
                memcpy(combinedMask, this->floatingMask, voxelNumber * sizeof(int));
                reg_tools_removeNanFromMask(this->floatingImage, combinedMask);
                reg_tools_removeNanFromMask(this->warpedImageBw, combinedMask);

                if (this->mind_type == MIND_TYPE) {
                    GetMINDImageDescriptor(this->floatingImage,
                                           this->floatingImageDescriptor,
                                           combinedMask,
                                           this->descriptorOffset,
                                           t);
                    GetMINDImageDescriptor(this->warpedImageBw,
                                           this->warpedReferenceImageDescriptor,
                                           combinedMask,
                                           this->descriptorOffset,
                                           t);
                } else if (this->mind_type == MINDSSC_TYPE) {
                    GetMINDSSCImageDescriptor(this->floatingImage,
                                              this->floatingImageDescriptor,
                                              combinedMask,
                                              this->descriptorOffset,
                                              t);
                    GetMINDSSCImageDescriptor(this->warpedImageBw,
                                              this->warpedReferenceImageDescriptor,
                                              combinedMask,
                                              this->descriptorOffset,
                                              t);
                }

                switch (this->floatingImageDescriptor->datatype) {
                case NIFTI_TYPE_FLOAT32:
                    MINDValue += reg_getSSDValue<float>(this->floatingImageDescriptor,
                                                        this->warpedReferenceImageDescriptor,
                                                        this->timePointWeightDescriptor,
                                                        nullptr, // TODO this->backwardJacDetImagePointer,
                                                        combinedMask,
                                                        this->currentValue,
                                                        nullptr);
                    break;
                case NIFTI_TYPE_FLOAT64:
                    MINDValue += reg_getSSDValue<double>(this->floatingImageDescriptor,
                                                         this->warpedReferenceImageDescriptor,
                                                         this->timePointWeightDescriptor,
                                                         nullptr, // TODO this->backwardJacDetImagePointer,
                                                         combinedMask,
                                                         this->currentValue,
                                                         nullptr);
                    break;
                default:
                    reg_print_fct_error("reg_mind::GetSimilarityMeasureValue");
                    reg_print_msg_error("Warped pixel type unsupported");
                    reg_exit();
                }
                free(combinedMask);
            }
        }
    }
    return MINDValue;   // (double) this->referenceImageDescriptor->nt;
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

    if (this->mind_type == MIND_TYPE) {
        // Compute the reference image descriptors
        GetMINDImageDescriptor(this->referenceImage,
                               this->referenceImageDescriptor,
                               combinedMask,
                               this->descriptorOffset,
                               currentTimepoint);
        // Compute the warped floating image descriptors
        GetMINDImageDescriptor(this->warpedImage,
                               this->warpedFloatingImageDescriptor,
                               combinedMask,
                               this->descriptorOffset,
                               currentTimepoint);
    } else if (this->mind_type == MINDSSC_TYPE) {
        // Compute the reference image descriptors
        GetMINDSSCImageDescriptor(this->referenceImage,
                                  this->referenceImageDescriptor,
                                  combinedMask,
                                  this->descriptorOffset,
                                  currentTimepoint);
        // Compute the warped floating image descriptors
        GetMINDSSCImageDescriptor(this->warpedImage,
                                  this->warpedFloatingImageDescriptor,
                                  combinedMask,
                                  this->descriptorOffset,
                                  currentTimepoint);
    }


    for (int desc_index = 0; desc_index < this->descriptor_number; ++desc_index) {
        // Compute the warped image descriptors gradient
        reg_getImageGradient_symDiff(this->warpedFloatingImageDescriptor,
                                     this->warpedGradient,
                                     combinedMask,
                                     std::numeric_limits<float>::quiet_NaN(),
                                     desc_index);

        // Compute the gradient of the ssd for the forward transformation
        switch (referenceImageDescriptor->datatype) {
        case NIFTI_TYPE_FLOAT32:
            reg_getVoxelBasedSSDGradient<float>(this->referenceImageDescriptor,
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
            reg_getVoxelBasedSSDGradient<double>(this->referenceImageDescriptor,
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

        if (this->mind_type == MIND_TYPE) {
            GetMINDImageDescriptor(this->floatingImage,
                                   this->floatingImageDescriptor,
                                   combinedMask,
                                   this->descriptorOffset,
                                   currentTimepoint);
            GetMINDImageDescriptor(this->warpedImageBw,
                                   this->warpedReferenceImageDescriptor,
                                   combinedMask,
                                   this->descriptorOffset,
                                   currentTimepoint);
        } else if (this->mind_type == MINDSSC_TYPE) {
            GetMINDSSCImageDescriptor(this->floatingImage,
                                      this->floatingImageDescriptor,
                                      combinedMask,
                                      this->descriptorOffset,
                                      currentTimepoint);
            GetMINDSSCImageDescriptor(this->warpedImageBw,
                                      this->warpedReferenceImageDescriptor,
                                      combinedMask,
                                      this->descriptorOffset,
                                      currentTimepoint);
        }

        for (int desc_index = 0; desc_index < this->descriptor_number; ++desc_index) {
            reg_getImageGradient_symDiff(this->warpedReferenceImageDescriptor,
                                         this->warpedGradientBw,
                                         combinedMask,
                                         std::numeric_limits<float>::quiet_NaN(),
                                         desc_index);

            // Compute the gradient of the nmi for the backward transformation
            switch (floatingImage->datatype) {
            case NIFTI_TYPE_FLOAT32:
                reg_getVoxelBasedSSDGradient<float>(this->floatingImageDescriptor,
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
                reg_getVoxelBasedSSDGradient<double>(this->floatingImageDescriptor,
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
    this->mind_type = MINDSSC_TYPE;
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
