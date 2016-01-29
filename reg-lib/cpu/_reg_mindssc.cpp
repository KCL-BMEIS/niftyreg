/*
 *  _reg_mindssc.cpp
 *
 *
 *  Created by Benoit Presles on 01/12/2015.
 *  Copyright (c) 2015, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_mindssc.h"
#include "_reg_ReadWriteImage.h"

/* *************************************************************** */
reg_mindssc::reg_mindssc()
    : reg_ssd()
{
    memset(this->activeTimePointDescriptor,0,255*sizeof(bool) );
    //ATM
    //platform = new Platform(NR_PLATFORM_CPU);
    //convolutionKernel = this->platform->createKernel(ConvolutionKernel::getName(), NULL);
    this->referenceImageDescriptor=NULL;
    this->floatingImageDescriptor=NULL;
    this->warpedFloatingImageDescriptor=NULL;
    this->warpedReferenceImageDescriptor=NULL;
    this->warpedFloatingGradientImageDescriptor=NULL;
    this->warpedReferenceGradientImageDescriptor=NULL;
#ifndef NDEBUG
    reg_print_msg_debug("reg_mindssc constructor called");
#endif
}
/* *************************************************************** */
reg_mindssc::~reg_mindssc() {
    if(this->referenceImageDescriptor != NULL)
        nifti_image_free(this->referenceImageDescriptor);
    this->referenceImageDescriptor = NULL;

    if(this->warpedFloatingImageDescriptor != NULL)
        nifti_image_free(this->warpedFloatingImageDescriptor);
    this->warpedFloatingImageDescriptor = NULL;

    if(this->warpedFloatingGradientImageDescriptor != NULL)
        nifti_image_free(this->warpedFloatingGradientImageDescriptor);
    this->warpedFloatingGradientImageDescriptor = NULL;

    if(this->floatingImageDescriptor != NULL)
        nifti_image_free(this->floatingImageDescriptor);
    this->floatingImageDescriptor = NULL;

    if(this->warpedReferenceImageDescriptor != NULL)
        nifti_image_free(this->warpedReferenceImageDescriptor);
    this->warpedReferenceImageDescriptor = NULL;

    if(this->warpedReferenceGradientImageDescriptor != NULL)
        nifti_image_free(this->warpedReferenceGradientImageDescriptor);
    this->warpedReferenceGradientImageDescriptor = NULL;
}
/* *************************************************************** */
void reg_mindssc::InitialiseMeasure(nifti_image *refImgPtr,
                                 nifti_image *floImgPtr,
                                 int *maskRefPtr,
                                 nifti_image *warFloImgPtr,
                                 nifti_image *warFloGraPtr,
                                 nifti_image *forVoxBasedGraPtr,
                                 int *maskFloPtr,
                                 nifti_image *warRefImgPtr,
                                 nifti_image *warRefGraPtr,
                                 nifti_image *bckVoxBasedGraPtr)
{
    // Set the pointers using the parent class function
    reg_ssd::InitialiseMeasure(refImgPtr,
                               floImgPtr,
                               maskRefPtr,
                               warFloImgPtr,
                               warFloGraPtr,
                               forVoxBasedGraPtr,
                               maskFloPtr,
                               warRefImgPtr,
                               warRefGraPtr,
                               bckVoxBasedGraPtr);

    if(this->referenceImagePointer->nt>1 || this->warpedFloatingImagePointer->nt>1){
        reg_print_msg_error("reg_mindssc does not support multiple time point image");
        reg_exit();
    }

    // Initialise the reference descriptor
    int dim=this->referenceImagePointer->nz>1?3:2;
    int dimt=this->referenceImagePointer->nz>1?12:4;
    this->referenceImageDescriptor = nifti_copy_nim_info(this->referenceImagePointer);
    this->referenceImageDescriptor->dim[0]=this->referenceImageDescriptor->ndim=4;
    this->referenceImageDescriptor->dim[4]=this->referenceImageDescriptor->nt=dimt;
    this->referenceImageDescriptor->nvox = (size_t)this->referenceImageDescriptor->nx*
                                           this->referenceImageDescriptor->ny*
                                           this->referenceImageDescriptor->nz*
                                           this->referenceImageDescriptor->nt;
    this->referenceImageDescriptor->data=(void *)malloc(this->referenceImageDescriptor->nvox*
                                                        this->referenceImageDescriptor->nbyper);
    // Initialise the warped floating descriptor
    this->warpedFloatingImageDescriptor = nifti_copy_nim_info(this->referenceImagePointer);
    this->warpedFloatingImageDescriptor->dim[0]=this->warpedFloatingImageDescriptor->ndim=4;
    this->warpedFloatingImageDescriptor->dim[4]=this->warpedFloatingImageDescriptor->nt=dimt;
    this->warpedFloatingImageDescriptor->nvox = (size_t)this->warpedFloatingImageDescriptor->nx*
                                                this->warpedFloatingImageDescriptor->ny*
                                                this->warpedFloatingImageDescriptor->nz*
                                                this->warpedFloatingImageDescriptor->nt;
    this->warpedFloatingImageDescriptor->data=(void *)malloc(this->warpedFloatingImageDescriptor->nvox*
                                                             this->warpedFloatingImageDescriptor->nbyper);
    // Initialise the warped gradient descriptor
    this->warpedFloatingGradientImageDescriptor = nifti_copy_nim_info(this->referenceImagePointer);
    this->warpedFloatingGradientImageDescriptor->dim[0]=this->warpedFloatingGradientImageDescriptor->ndim=5;
    this->warpedFloatingGradientImageDescriptor->dim[4]=this->warpedFloatingGradientImageDescriptor->nt=dimt;
    this->warpedFloatingGradientImageDescriptor->dim[5]=this->warpedFloatingGradientImageDescriptor->nu=dim;
    this->warpedFloatingGradientImageDescriptor->nvox = (size_t)this->warpedFloatingGradientImageDescriptor->nx*
                                                        this->warpedFloatingGradientImageDescriptor->ny*
                                                        this->warpedFloatingGradientImageDescriptor->nz*
                                                        this->warpedFloatingGradientImageDescriptor->nt*
                                                        this->warpedFloatingGradientImageDescriptor->nu;
    this->warpedFloatingGradientImageDescriptor->data=(void *)malloc(this->warpedFloatingGradientImageDescriptor->nvox*
                                                                     this->warpedFloatingGradientImageDescriptor->nbyper);

    if(this->isSymmetric) {
        if(this->floatingImagePointer->nt>1 || this->warpedReferenceImagePointer->nt>1){
            reg_print_msg_error("reg_mindssc does not support multiple time point image");
            reg_exit();
        }
        // Initialise the floating descriptor
        int dim=this->floatingImagePointer->nz>1?3:2;
        int dimt=this->referenceImagePointer->nz>1?12:4;
        this->floatingImageDescriptor = nifti_copy_nim_info(this->floatingImagePointer);
        this->floatingImageDescriptor->dim[0]=this->floatingImageDescriptor->ndim=4;
        this->floatingImageDescriptor->dim[4]=this->floatingImageDescriptor->nt=dimt;
        this->floatingImageDescriptor->nvox = (size_t)this->floatingImageDescriptor->nx*
                                              this->floatingImageDescriptor->ny*
                                              this->floatingImageDescriptor->nz*
                                              this->floatingImageDescriptor->nt;
        this->floatingImageDescriptor->data=(void *)malloc(this->floatingImageDescriptor->nvox*
                                                           this->floatingImageDescriptor->nbyper);
        // Initialise the warped floating descriptor
        this->warpedReferenceImageDescriptor = nifti_copy_nim_info(this->floatingImagePointer);
        this->warpedReferenceImageDescriptor->dim[0]=this->warpedReferenceImageDescriptor->ndim=4;
        this->warpedReferenceImageDescriptor->dim[4]=this->warpedReferenceImageDescriptor->nt=dimt;
        this->warpedReferenceImageDescriptor->nvox = (size_t)this->warpedReferenceImageDescriptor->nx*
                                                     this->warpedReferenceImageDescriptor->ny*
                                                     this->warpedReferenceImageDescriptor->nz*
                                                     this->warpedReferenceImageDescriptor->nt;
        this->warpedReferenceImageDescriptor->data=(void *)malloc(this->warpedReferenceImageDescriptor->nvox*
                                                                  this->warpedReferenceImageDescriptor->nbyper);
        // Initialise the warped gradient descriptor
        this->warpedReferenceGradientImageDescriptor = nifti_copy_nim_info(this->floatingImagePointer);
        this->warpedReferenceGradientImageDescriptor->dim[0]=this->warpedReferenceGradientImageDescriptor->ndim=5;
        this->warpedReferenceGradientImageDescriptor->dim[4]=this->warpedReferenceGradientImageDescriptor->nt=dimt;
        this->warpedReferenceGradientImageDescriptor->dim[5]=this->warpedReferenceGradientImageDescriptor->nu=dim;
        this->warpedReferenceGradientImageDescriptor->nvox = (size_t)this->warpedReferenceGradientImageDescriptor->nx*
                                                             this->warpedReferenceGradientImageDescriptor->ny*
                                                             this->warpedReferenceGradientImageDescriptor->nz*
                                                             this->warpedReferenceGradientImageDescriptor->nt*
                                                             this->warpedReferenceGradientImageDescriptor->nu;
        this->warpedFloatingGradientImageDescriptor->data=(void *)malloc(this->warpedReferenceGradientImageDescriptor->nvox*
                                                                         this->warpedReferenceGradientImageDescriptor->nbyper);
    }

    for(int i=0;i<referenceImageDescriptor->nt;++i) {
        this->activeTimePointDescriptor[i]=true;
    }

#ifndef NDEBUG
    char text[255];
    reg_print_msg_debug("reg_mindssc::InitialiseMeasure().");
    sprintf(text, "Active time point:");
    for(int i=0; i<this->referenceImageDescriptor->nt; ++i)
        if(this->activeTimePointDescriptor[i])
            sprintf(text, "%s %i", text, i);
    reg_print_msg_debug(text);
#endif
}
/* *************************************************************** */
template <class DTYPE>
void ShiftImage(nifti_image* inputImgPtr,
                nifti_image* shiftedImgPtr,
                int *maskPtr,
                int tx,
                int ty,
                int tz)
{
    DTYPE* inputData = static_cast<DTYPE*> (inputImgPtr->data);
    DTYPE* shiftImageData = static_cast<DTYPE*> (shiftedImgPtr->data);

    int currentIndex = 0;
    int shiftedIndex;
    for (int z=0;z<shiftedImgPtr->nz;z++) {
        int old_z = z-tz;
        for (int y=0;y<shiftedImgPtr->ny;y++) {
            int old_y = y-ty;
            for (int x=0;x<shiftedImgPtr->nx;x++) {
                int old_x = x-tx;
                if(old_x>-1 && old_x<inputImgPtr->nx && old_y>-1 && old_y<inputImgPtr->ny && old_z>-1 && old_z<inputImgPtr->nz){
                    shiftedIndex = (old_z*inputImgPtr->ny+old_y)*inputImgPtr->nx+old_x;
                    if(maskPtr[shiftedIndex]>-1)
                        shiftImageData[currentIndex]=inputData[shiftedIndex];
                    else shiftImageData[currentIndex]=std::numeric_limits<DTYPE>::quiet_NaN();
                }
                else shiftImageData[currentIndex]=std::numeric_limits<DTYPE>::quiet_NaN();
                currentIndex++;
            }
        }
    }
}
/* *************************************************************** */
template <class DTYPE>
void GetMINDSSCImageDesciptor_core(nifti_image* inputImgPtr,
                                nifti_image* MINDSSCImgPtr,
                                int *maskPtr)
{
    const size_t voxNumber = (size_t)inputImgPtr->nx *
                             inputImgPtr->ny * inputImgPtr->nz;
    size_t voxIndex;

    // Create a pointer to the descriptor image
    DTYPE* MINDSSCImgDataPtr = static_cast<DTYPE *>(MINDSSCImgPtr->data);

    // Allocate an image to store the mean image
    nifti_image *mean_img = nifti_copy_nim_info(inputImgPtr);
    mean_img->data=(void *)calloc(inputImgPtr->nvox,inputImgPtr->nbyper);
    DTYPE* meanImgDataPtr = static_cast<DTYPE *>(mean_img->data);

    // Allocate an image to store the warped image
    nifti_image *warpedImage = nifti_copy_nim_info(inputImgPtr);
    warpedImage->data = (void *)malloc(warpedImage->nvox*warpedImage->nbyper);

    // Define the sigma for the convolution
    float sigma = -0.5;// negative value denotes voxel width

    //2D version
    int samplingNbr = (inputImgPtr->nz > 1) ? 6 : 2;
    int lengthDescriptor = (inputImgPtr->nz > 1) ? 12 : 4;

    // Allocation of the difference image
    //std::vector<nifti_image *> vectNiftiImage;
    //for(int i=0;i<samplingNbr;i++) {
    nifti_image *diff_image = nifti_copy_nim_info(inputImgPtr);
    diff_image->data = (void *) malloc(diff_image->nvox*diff_image->nbyper);
    int *mask_diff_image = (int *)calloc(diff_image->nvox, sizeof(int));
    //
    nifti_image *diff_imageShifted = nifti_copy_nim_info(inputImgPtr);
    diff_imageShifted->data = (void *) malloc(diff_imageShifted->nvox*diff_imageShifted->nbyper);

    //    vectNiftiImage.push_back(diff_image);
    //}

    //[+1,+1,-1,+0,+1,+0];
    //[+1,-1,+0,-1,+0,+1];
    //[+0,+0,+1,+1,+1,+1];
    int RSampling3D_x[6] = {+1,+1,-1,+0,+1,+0};
    int RSampling3D_y[6] = {+1,-1,+0,-1,+0,+1};
    int RSampling3D_z[6] = {+0,+0,+1,+1,+1,+1};

    int tx[12]={-1,+0,-1,+0,+0,+1,+0,+0,+0,-1,+0,+0};
    int ty[12]={+0,-1,+0,+1,+0,+0,+0,+1,+0,+0,+0,-1};
    int tz[12]={+0,+0,+0,+0,-1,+0,-1,+0,-1,+0,-1,+0};
    int compteurId = 0;

    for(int i=0;i<samplingNbr;i++) {
        ShiftImage<DTYPE>(inputImgPtr, warpedImage, maskPtr,
                          RSampling3D_x[i], RSampling3D_y[i], RSampling3D_z[i]);
        reg_tools_substractImageToImage(inputImgPtr, warpedImage, diff_image);
        reg_tools_multiplyImageToImage(diff_image, diff_image, diff_image);
        reg_tools_kernelConvolution(diff_image, &sigma, 0, maskPtr);

        for(int j=0;j<2;j++){

            ShiftImage<DTYPE>(diff_image, diff_imageShifted, mask_diff_image,
                              tx[compteurId], ty[compteurId], tz[compteurId]);

            reg_tools_addImageToImage(mean_img, diff_imageShifted, mean_img);
            // Store the current descriptor
            unsigned int index = compteurId * diff_imageShifted->nvox;
            memcpy(&MINDSSCImgDataPtr[index], diff_imageShifted->data,
                   diff_imageShifted->nbyper * diff_imageShifted->nvox);
            compteurId++;
        }
    }
    // Compute the mean over the number of sample
    reg_tools_divideValueToImage(mean_img, mean_img, lengthDescriptor);

    // Compute the MINDSSC desccriptor
    int mindIndex;
    DTYPE meanValue, max_desc, descValue;
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(lengthDescriptor, samplingNbr, maskPtr, meanImgDataPtr, \
   MINDSSCImgDataPtr) \
   private(voxIndex, meanValue, max_desc, descValue, mindIndex)
#endif
    for(voxIndex=0;voxIndex<voxNumber;voxIndex++) {

        if(maskPtr[voxIndex]>-1){
            // Get the mean value for the current voxel
            meanValue = meanImgDataPtr[voxIndex];
            if(meanValue == 0) {
                meanValue = std::numeric_limits<DTYPE>::epsilon();
            }
            max_desc = 0;
            mindIndex=voxIndex;
            for(int t=0;t<lengthDescriptor;t++) {
                descValue = (DTYPE)exp(-MINDSSCImgDataPtr[mindIndex]/meanValue);
                MINDSSCImgDataPtr[mindIndex] = descValue;
                max_desc = std::max(max_desc, descValue);
                mindIndex+=voxNumber;
            }

            mindIndex=voxIndex;
            for(int t=0;t<lengthDescriptor;t++) {
                descValue = MINDSSCImgDataPtr[mindIndex];
                MINDSSCImgDataPtr[mindIndex] = descValue/max_desc;
                mindIndex+=voxNumber;
            }
        } // mask
    } // voxIndex
    // Mr Propre
    nifti_image_free(diff_imageShifted);
    free(mask_diff_image);
    nifti_image_free(diff_image);
    nifti_image_free(warpedImage);
    nifti_image_free(mean_img);
}
/* *************************************************************** */
void GetMINDSSCImageDesciptor(nifti_image* inputImgPtr,
                           nifti_image* MINDSSCImgPtr,
                           int *maskPtr) {
#ifndef NDEBUG
    reg_print_fct_debug("GetMINDSSCImageDesciptor()");
#endif
    if(inputImgPtr->datatype != MINDSSCImgPtr->datatype) {
        reg_print_fct_error("reg_mindssc -- GetMINDSSCImageDesciptor");
        reg_print_msg_error("The input image and the MINDSSC image must have the same datatype !");
        reg_exit();
    }

    switch (inputImgPtr->datatype)
    {
    case NIFTI_TYPE_FLOAT32:
        GetMINDSSCImageDesciptor_core<float>(inputImgPtr, MINDSSCImgPtr, maskPtr);
        break;
    case NIFTI_TYPE_FLOAT64:
        GetMINDSSCImageDesciptor_core<double>(inputImgPtr, MINDSSCImgPtr, maskPtr);
        break;
    default:
        reg_print_fct_error("GetMINDSSCImageDesciptor");
        reg_print_msg_error("Input image datatype not supported");
        reg_exit();
        break;
    }
}
/* *************************************************************** */
double reg_mindssc::GetSimilarityMeasureValue()
{
    size_t voxelNumber = (size_t)referenceImagePointer->nx *
                         referenceImagePointer->ny * referenceImagePointer->nz;
    int *combinedMask = (int *)malloc(voxelNumber*sizeof(int));
    memcpy(combinedMask, this->referenceMaskPointer, voxelNumber*sizeof(int));
    reg_tools_removeNanFromMask(this->referenceImagePointer, combinedMask);
    reg_tools_removeNanFromMask(this->warpedFloatingImagePointer, combinedMask);

    GetMINDSSCImageDesciptor(this->referenceImagePointer,
                          this->referenceImageDescriptor,
                          combinedMask);
    GetMINDSSCImageDesciptor(this->warpedFloatingImagePointer,
                          this->warpedFloatingImageDescriptor,
                          combinedMask);

    double MINDSSCValue;
    switch(this->referenceImageDescriptor->datatype)
    {
    case NIFTI_TYPE_FLOAT32:
        MINDSSCValue = reg_getSSDValue<float>
                   (this->referenceImageDescriptor,
                    this->warpedFloatingImageDescriptor,
                    this->activeTimePointDescriptor,
                    NULL, // HERE TODO this->forwardJacDetImagePointer,
                    combinedMask,
                    this->currentValue
                    );
        break;
    case NIFTI_TYPE_FLOAT64:
        MINDSSCValue = reg_getSSDValue<double>
                   (this->referenceImageDescriptor,
                    this->warpedFloatingImageDescriptor,
                    this->activeTimePointDescriptor,
                    NULL, // HERE TODO this->forwardJacDetImagePointer,
                    combinedMask,
                    this->currentValue
                    );
        break;
    default:
        reg_print_fct_error("reg_mindssc::GetSimilarityMeasureValue");
        reg_print_msg_error("Warped pixel type unsupported");
        reg_exit();
    }
    free(combinedMask);

    // Backward computation
    if(this->isSymmetric)
    {
        voxelNumber = (size_t)floatingImagePointer->nx *
                      floatingImagePointer->ny * floatingImagePointer->nz;
        combinedMask = (int *)malloc(voxelNumber*sizeof(int));
        memcpy(combinedMask, this->referenceMaskPointer, voxelNumber*sizeof(int));
        reg_tools_removeNanFromMask(this->referenceImagePointer, combinedMask);
        reg_tools_removeNanFromMask(this->warpedFloatingImagePointer, combinedMask);
        GetMINDSSCImageDesciptor(this->floatingImagePointer,
                              this->floatingImageDescriptor,
                              combinedMask);
        GetMINDSSCImageDesciptor(this->warpedReferenceImagePointer,
                              this->warpedReferenceImageDescriptor,
                              combinedMask);

        switch(this->floatingImageDescriptor->datatype)
        {
        case NIFTI_TYPE_FLOAT32:
            MINDSSCValue += reg_getSSDValue<float>
                        (this->floatingImageDescriptor,
                         this->warpedReferenceImageDescriptor,
                         this->activeTimePointDescriptor,
                         NULL, // HERE TODO this->backwardJacDetImagePointer,
                         combinedMask,
                         this->currentValue
                         );
            break;
        case NIFTI_TYPE_FLOAT64:
            MINDSSCValue += reg_getSSDValue<double>
                        (this->floatingImageDescriptor,
                         this->warpedReferenceImageDescriptor,
                         this->activeTimePointDescriptor,
                         NULL, // HERE TODO this->backwardJacDetImagePointer,
                         combinedMask,
                         this->currentValue
                         );
            break;
        default:
            reg_print_fct_error("reg_mindssc::GetSimilarityMeasureValue");
            reg_print_msg_error("Warped pixel type unsupported");
            reg_exit();
        }
        free(combinedMask);
    }
    return MINDSSCValue;// /(double) this->referenceImageDescriptor->nt;
}
/* *************************************************************** */
template<class DTYPE>
void spatialGradient(nifti_image *img,
                     nifti_image *gradImg,
                     int *mask)
{
    size_t voxIndex, voxelNumber = (size_t)img->nx *
                         img->ny * img->nz;

    int dimImg = img->nz > 1 ? 3 : 2;
    int x, y, z;

    DTYPE *imgPtr = static_cast<DTYPE *>(img->data);
    DTYPE *gradPtr = static_cast<DTYPE *>(gradImg->data);
    for(int t=0; t<img->nt; ++t){
        DTYPE *currentImgPtr = &imgPtr[t*voxelNumber];
        DTYPE *gradPtrX = &gradPtr[t*voxelNumber];
        DTYPE *gradPtrY = &gradPtr[(img->nt+t)*voxelNumber];
        DTYPE *gradPtrZ = NULL;
        if(dimImg==3)
            gradPtrZ = &gradPtr[(2*img->nt+t)*voxelNumber];

#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(img, currentImgPtr, mask, \
   gradPtrX, gradPtrY, gradPtrZ) \
   private(x, y, z, voxIndex)
#endif
        for(z=0; z<img->nz; ++z){
           voxIndex=z*img->nx*img->ny;
            for(y=0; y<img->ny; ++y){
                for(x=0; x<img->nx; ++x){
                    if(mask[voxIndex]>-1){
                        if(x<img->nx-1)
                            gradPtrX[voxIndex] = currentImgPtr[voxIndex+1] - currentImgPtr[voxIndex];
                        else gradPtrX[voxIndex] = currentImgPtr[voxIndex] - currentImgPtr[voxIndex-1];
                        if(gradPtrX[voxIndex]!=gradPtrX[voxIndex]) gradPtrX[voxIndex]=0.;
                        if(y<img->ny-1)
                            gradPtrY[voxIndex] = currentImgPtr[voxIndex+img->nx] - currentImgPtr[voxIndex];
                        else gradPtrY[voxIndex] = currentImgPtr[voxIndex] - currentImgPtr[voxIndex-img->nx];
                        if(gradPtrY[voxIndex]!=gradPtrY[voxIndex]) gradPtrY[voxIndex]=0.;
                        if(gradPtrZ!=NULL){
                            if(z<img->nz-1)
                                gradPtrZ[voxIndex] = currentImgPtr[voxIndex+img->nx*img->ny] - currentImgPtr[voxIndex];
                            else gradPtrZ[voxIndex] = currentImgPtr[voxIndex] - currentImgPtr[voxIndex-img->nx*img->ny];
                            if(gradPtrZ[voxIndex]!=gradPtrZ[voxIndex]) gradPtrZ[voxIndex]=0.;
                        }
                    }
                    ++voxIndex;
                } // x
            } // y
        } // z
    } // t
}
template void spatialGradient<float>(nifti_image *img, nifti_image *gradImg, int *mask);
template void spatialGradient<double>(nifti_image *img, nifti_image *gradImg, int *mask);
/* *************************************************************** */
void reg_mindssc::GetVoxelBasedSimilarityMeasureGradient()
{
    // Compute the floating warped image descriptors
    size_t voxelNumber = (size_t)this->referenceImagePointer->nx *
                         this->referenceImagePointer->ny *
                         this->referenceImagePointer->nz;
    int *combinedMask = (int *)malloc(voxelNumber*sizeof(int));
    memcpy(combinedMask, this->referenceMaskPointer, voxelNumber*sizeof(int));
    reg_tools_removeNanFromMask(this->referenceImagePointer, combinedMask);
    reg_tools_removeNanFromMask(this->warpedFloatingImagePointer, combinedMask);

    GetMINDSSCImageDesciptor(this->referenceImagePointer,
                          this->referenceImageDescriptor,
                          combinedMask);
    GetMINDSSCImageDesciptor(this->warpedFloatingImagePointer,
                          this->warpedFloatingImageDescriptor,
                          combinedMask);

    spatialGradient<float>(this->warpedFloatingImageDescriptor,
                           this->warpedFloatingGradientImageDescriptor,
                           combinedMask);

    // Compute the gradient of the ssd for the forward transformation
    switch(referenceImageDescriptor->datatype)
    {
    case NIFTI_TYPE_FLOAT32:
        reg_getVoxelBasedSSDGradient<float>
                (this->referenceImageDescriptor,
                 this->warpedFloatingImageDescriptor,
                 this->activeTimePointDescriptor,
                 this->warpedFloatingGradientImageDescriptor,
                 this->forwardVoxelBasedGradientImagePointer,
                 NULL, // no Jacobian required here,
                 combinedMask
                 );
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_getVoxelBasedSSDGradient<double>
                (this->referenceImageDescriptor,
                 this->warpedFloatingImageDescriptor,
                 this->activeTimePointDescriptor,
                 this->warpedFloatingGradientImageDescriptor,
                 this->forwardVoxelBasedGradientImagePointer,
                 NULL, // no Jacobian required here,
                 combinedMask
                 );
        break;
    default:
        reg_print_fct_error("reg_mindssc::GetVoxelBasedSimilarityMeasureGradient");
        reg_print_msg_error("Unsupported datatype");
        reg_exit();
    }
    free(combinedMask);
    // Compute the gradient of the ssd for the backward transformation
    if(this->isSymmetric)
    {
        voxelNumber = (size_t)floatingImagePointer->nx *
                      floatingImagePointer->ny * floatingImagePointer->nz;
        combinedMask = (int *)malloc(voxelNumber*sizeof(int));
        memcpy(combinedMask, this->referenceMaskPointer, voxelNumber*sizeof(int));
        reg_tools_removeNanFromMask(this->referenceImagePointer, combinedMask);
        reg_tools_removeNanFromMask(this->warpedFloatingImagePointer, combinedMask);
        GetMINDSSCImageDesciptor(this->floatingImagePointer,
                              this->floatingImageDescriptor,
                              combinedMask);
        GetMINDSSCImageDesciptor(this->warpedReferenceImagePointer,
                              this->warpedReferenceImageDescriptor,
                              combinedMask);

        spatialGradient<float>(this->warpedReferenceImageDescriptor,
                                    this->warpedReferenceGradientImageDescriptor,
                                    combinedMask);

        // Compute the gradient of the nmi for the backward transformation
        switch(floatingImagePointer->datatype)
        {
        case NIFTI_TYPE_FLOAT32:
            reg_getVoxelBasedSSDGradient<float>
                    (this->floatingImageDescriptor,
                     this->warpedReferenceImageDescriptor,
                     this->activeTimePointDescriptor,
                     this->warpedReferenceGradientImageDescriptor,
                     this->backwardVoxelBasedGradientImagePointer,
                     NULL, // no Jacobian required here,
                     combinedMask
                     );
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_getVoxelBasedSSDGradient<double>
                    (this->floatingImageDescriptor,
                     this->warpedReferenceImageDescriptor,
                     this->activeTimePointDescriptor,
                     this->warpedReferenceGradientImageDescriptor,
                     this->backwardVoxelBasedGradientImagePointer,
                     NULL, // no Jacobian required here,
                     combinedMask
                     );
            break;
        default:
            reg_print_fct_error("reg_mindssc::GetVoxelBasedSimilarityMeasureGradient");
            reg_print_msg_error("Unsupported datatype");
            reg_exit();
        }
        free(combinedMask);
    }
}
/* *************************************************************** */

