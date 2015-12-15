/*
 *  _reg_mind.cpp
 *
 *
 *  Created by Benoit Presles on 01/12/2015.
 *  Copyright (c) 2015, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_mind.h"
#include "_reg_ReadWriteImage.h"

/* *************************************************************** */
reg_mind::reg_mind()
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
    reg_print_msg_debug("reg_mind constructor called");
#endif
}
/* *************************************************************** */
void reg_mind::InitialiseMeasure(nifti_image *refImgPtr,
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
        reg_print_msg_error("reg_mind does not support multiple time point image");
        reg_exit(1);
    }

    // Initialise the reference descriptor
    int dim=this->referenceImagePointer->nz>1?3:2;
    this->referenceImageDescriptor = nifti_copy_nim_info(this->referenceImagePointer);
    this->referenceImageDescriptor->dim[0]=this->referenceImageDescriptor->ndim=4;
    this->referenceImageDescriptor->dim[4]=this->referenceImageDescriptor->nt=dim*2;
    this->referenceImageDescriptor->nvox = (size_t)this->referenceImageDescriptor->nx*
                                           this->referenceImageDescriptor->ny*
                                           this->referenceImageDescriptor->nz*
                                           this->referenceImageDescriptor->nt;
    this->referenceImageDescriptor->data=(void *)malloc(this->referenceImageDescriptor->nvox*
                                                        this->referenceImageDescriptor->nbyper);
    // Initialise the warped floating descriptor
    this->warpedFloatingImageDescriptor = nifti_copy_nim_info(this->referenceImagePointer);
    this->warpedFloatingImageDescriptor->dim[0]=this->warpedFloatingImageDescriptor->ndim=4;
    this->warpedFloatingImageDescriptor->dim[4]=this->warpedFloatingImageDescriptor->nt=dim*2;
    this->warpedFloatingImageDescriptor->nvox = (size_t)this->warpedFloatingImageDescriptor->nx*
                                                this->warpedFloatingImageDescriptor->ny*
                                                this->warpedFloatingImageDescriptor->nz*
                                                this->warpedFloatingImageDescriptor->nt;
    this->warpedFloatingImageDescriptor->data=(void *)malloc(this->warpedFloatingImageDescriptor->nvox*
                                                             this->warpedFloatingImageDescriptor->nbyper);
    // Initialise the warped gradient descriptor
    this->warpedFloatingGradientImageDescriptor = nifti_copy_nim_info(this->referenceImagePointer);
    this->warpedFloatingGradientImageDescriptor->dim[0]=this->warpedFloatingGradientImageDescriptor->ndim=5;
    this->warpedFloatingGradientImageDescriptor->dim[4]=this->warpedFloatingGradientImageDescriptor->nt=dim*2;
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
            reg_print_msg_error("reg_mind does not support multiple time point image");
            reg_exit(1);
        }
        // Initialise the floating descriptor
        int dim=this->floatingImagePointer->nz>1?3:2;
        this->floatingImageDescriptor = nifti_copy_nim_info(this->floatingImagePointer);
        this->floatingImageDescriptor->dim[0]=this->floatingImageDescriptor->ndim=4;
        this->floatingImageDescriptor->dim[4]=this->floatingImageDescriptor->nt=dim*2;
        this->floatingImageDescriptor->nvox = (size_t)this->floatingImageDescriptor->nx*
                                              this->floatingImageDescriptor->ny*
                                              this->floatingImageDescriptor->nz*
                                              this->floatingImageDescriptor->nt;
        this->floatingImageDescriptor->data=(void *)malloc(this->floatingImageDescriptor->nvox*
                                                           this->floatingImageDescriptor->nbyper);
        // Initialise the warped floating descriptor
        this->warpedReferenceImageDescriptor = nifti_copy_nim_info(this->floatingImagePointer);
        this->warpedReferenceImageDescriptor->dim[0]=this->warpedReferenceImageDescriptor->ndim=4;
        this->warpedReferenceImageDescriptor->dim[4]=this->warpedReferenceImageDescriptor->nt=dim*2;
        this->warpedReferenceImageDescriptor->nvox = (size_t)this->warpedReferenceImageDescriptor->nx*
                                                     this->warpedReferenceImageDescriptor->ny*
                                                     this->warpedReferenceImageDescriptor->nz*
                                                     this->warpedReferenceImageDescriptor->nt;
        this->warpedReferenceImageDescriptor->data=(void *)malloc(this->warpedReferenceImageDescriptor->nvox*
                                                                  this->warpedReferenceImageDescriptor->nbyper);
        // Initialise the warped gradient descriptor
        this->warpedReferenceGradientImageDescriptor = nifti_copy_nim_info(this->floatingImagePointer);
        this->warpedReferenceGradientImageDescriptor->dim[0]=this->warpedReferenceGradientImageDescriptor->ndim=5;
        this->warpedReferenceGradientImageDescriptor->dim[4]=this->warpedReferenceGradientImageDescriptor->nt=dim*2;
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
    reg_print_msg_debug("reg_mind::InitialiseMeasure().");
    sprintf(text, "Active time point:");
    for(int i=0; i<this->referenceImageDescriptor->nt; ++i)
        if(this->activeTimePointDescriptor[i])
            sprintf(text, "%s %i", text, i);
    reg_print_msg_debug(text);
#endif

}
/* *************************************************************** */
reg_mind::~reg_mind() {
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
void reg_mind::GetVoxelBasedSimilarityMeasureGradient()
{
    // Compute the floating warped image descriptors
    size_t voxelNumber = (size_t)referenceImagePointer->nx *
                         referenceImagePointer->ny * referenceImagePointer->nz;
    int *combinedMask = (int *)malloc(voxelNumber*sizeof(int));
    memcpy(combinedMask, this->referenceMaskPointer, voxelNumber*sizeof(int));
    reg_tools_removeNanFromMask(this->referenceImagePointer, combinedMask);
    reg_tools_removeNanFromMask(this->warpedFloatingImagePointer, combinedMask);

    GetMINDImageDesciptor(this->referenceImagePointer,
                          this->referenceImageDescriptor,
                          combinedMask);
    GetMINDImageDesciptor(this->warpedFloatingImagePointer,
                          this->warpedFloatingImageDescriptor,
                          combinedMask);

    // Create an identity transformation
    nifti_image *identityDefField = nifti_copy_nim_info(this->referenceImagePointer);
    identityDefField->dim[0]=identityDefField->ndim=5;
    identityDefField->dim[4]=identityDefField->nt=1;
    identityDefField->dim[5]=identityDefField->nu=this->referenceImagePointer->nz>1?3:2;
    identityDefField->nvox = (size_t)identityDefField->nx *
                             identityDefField->ny *
                             identityDefField->nz *
                             identityDefField->nu;
    identityDefField->datatype=NIFTI_TYPE_FLOAT32;
    identityDefField->nbyper=sizeof(float);
    identityDefField->data = (void *)calloc(identityDefField->nvox,
                                            identityDefField->nbyper);
    identityDefField->intent_code=NIFTI_INTENT_VECTOR;
    memset(identityDefField->intent_name, 0, 16);
    strcpy(identityDefField->intent_name,"NREG_TRANS");
    identityDefField->intent_p1==DISP_FIELD;
    reg_getDeformationFromDisplacement(identityDefField);

    // Compute the gradient of the warped floating descriptor image
    reg_getImageGradient(this->warpedFloatingImageDescriptor,
                         this->warpedFloatingGradientImageDescriptor,
                         identityDefField,
                         combinedMask,
                         1,
                         std::numeric_limits<float>::quiet_NaN());
    nifti_image_free(identityDefField);identityDefField=NULL;

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
        reg_print_fct_error("reg_mind::GetVoxelBasedSimilarityMeasureGradient");
        reg_print_msg_error("Unsupported datatype");
        reg_exit(1);
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
        GetMINDImageDesciptor(this->floatingImagePointer,
                              this->floatingImageDescriptor,
                              combinedMask);
        GetMINDImageDesciptor(this->warpedReferenceImagePointer,
                              this->warpedReferenceImageDescriptor,
                              combinedMask);

        identityDefField = nifti_copy_nim_info(this->floatingImagePointer);
        identityDefField->dim[4]=identityDefField->nt=1;
        identityDefField->dim[5]=identityDefField->nu=this->floatingImagePointer->nz>1?3:2;
        identityDefField->dim[0]=identityDefField->ndim=5;
        identityDefField->nvox = (size_t)identityDefField->nx * identityDefField->ny *
                                       identityDefField->nz * identityDefField->nu;
        identityDefField->datatype=NIFTI_TYPE_FLOAT32;
        identityDefField->nbyper=sizeof(float);
        identityDefField->intent_code=NIFTI_INTENT_VECTOR;
        memset(identityDefField->intent_name, 0, 16);
        strcpy(identityDefField->intent_name,"NREG_TRANS");
        if(identityDefField->intent_p1==DISP_FIELD)
        identityDefField->data = (void *)calloc(identityDefField->nvox,
                                                      identityDefField->nbyper);
        reg_getDeformationFromDisplacement(identityDefField);
        reg_getImageGradient(this->warpedReferenceImageDescriptor,
                             this->warpedReferenceGradientImageDescriptor,
                             identityDefField,
                             combinedMask,
                             1,
                             std::numeric_limits<float>::quiet_NaN());
        nifti_image_free(identityDefField);identityDefField=NULL;

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
            reg_print_fct_error("reg_mind::GetVoxelBasedSimilarityMeasureGradient");
            reg_print_msg_error("Unsupported datatype");
            reg_exit(1);
        }
    }
    free(combinedMask);
}
/* *************************************************************** */
double reg_mind::GetSimilarityMeasureValue()
{
    size_t voxelNumber = (size_t)referenceImagePointer->nx *
                         referenceImagePointer->ny * referenceImagePointer->nz;
    int *combinedMask = (int *)malloc(voxelNumber*sizeof(int));
    memcpy(combinedMask, this->referenceMaskPointer, voxelNumber*sizeof(int));
    reg_tools_removeNanFromMask(this->referenceImagePointer, combinedMask);
    reg_tools_removeNanFromMask(this->warpedFloatingImagePointer, combinedMask);

    GetMINDImageDesciptor(this->referenceImagePointer,
                          this->referenceImageDescriptor,
                          combinedMask);
    GetMINDImageDesciptor(this->warpedFloatingImagePointer,
                          this->warpedFloatingImageDescriptor,
                          combinedMask);

    double MINDValue;
    switch(this->referenceImageDescriptor->datatype)
    {
    case NIFTI_TYPE_FLOAT32:
        MINDValue = reg_getSSDValue<float>
                   (this->referenceImageDescriptor,
                    this->warpedFloatingImageDescriptor,
                    this->activeTimePointDescriptor,
                    NULL, // HERE TODO this->forwardJacDetImagePointer,
                    combinedMask,
                    this->currentValue
                    );
        break;
    case NIFTI_TYPE_FLOAT64:
        MINDValue = reg_getSSDValue<double>
                   (this->referenceImageDescriptor,
                    this->warpedFloatingImageDescriptor,
                    this->activeTimePointDescriptor,
                    NULL, // HERE TODO this->forwardJacDetImagePointer,
                    combinedMask,
                    this->currentValue
                    );
        break;
    default:
        reg_print_fct_error("reg_ssd::GetSimilarityMeasureValue");
        reg_print_msg_error("Warped pixel type unsupported");
        reg_exit(1);
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
        GetMINDImageDesciptor(this->floatingImagePointer,
                              this->floatingImageDescriptor,
                              combinedMask);
        GetMINDImageDesciptor(this->warpedReferenceImagePointer,
                              this->warpedReferenceImageDescriptor,
                              combinedMask);

        switch(this->floatingImageDescriptor->datatype)
        {
        case NIFTI_TYPE_FLOAT32:
            MINDValue += reg_getSSDValue<float>
                        (this->floatingImageDescriptor,
                         this->warpedReferenceImageDescriptor,
                         this->activeTimePointDescriptor,
                         NULL, // HERE TODO this->backwardJacDetImagePointer,
                         combinedMask,
                         this->currentValue
                         );
            break;
        case NIFTI_TYPE_FLOAT64:
            MINDValue += reg_getSSDValue<double>
                        (this->floatingImageDescriptor,
                         this->warpedReferenceImageDescriptor,
                         this->activeTimePointDescriptor,
                         NULL, // HERE TODO this->backwardJacDetImagePointer,
                         combinedMask,
                         this->currentValue
                         );
            break;
        default:
            reg_print_fct_error("reg_ssd::GetSimilarityMeasureValue");
            reg_print_msg_error("Warped pixel type unsupported");
            reg_exit(1);
        }
    }
    free(combinedMask);
    return MINDValue;///(double) this->referenceImageDescriptor->nt;
}
/* *************************************************************** */
template <class DTYPE>
void ShiftImage(nifti_image* inputImgPtr,
                nifti_image* shiftedImgPtr,
                int tx,
                int ty,
                int tz)
{
    DTYPE* inputData = static_cast<DTYPE*> (inputImgPtr->data);
    DTYPE* shiftImageData = static_cast<DTYPE*> (shiftedImgPtr->data);

    int currentIndex = 0;
    int shiftedIndex;
    for (int z=0;z<shiftedImgPtr->nz;z++) {
        int new_z = z-tz;
        for (int y=0;y<shiftedImgPtr->ny;y++) {
            int new_y = y-ty;
            for (int x=0;x<shiftedImgPtr->nx;x++) {
                int new_x = x-tx;
                if(new_x>-1 && new_x<inputImgPtr->nx && new_y>-1 && new_y<inputImgPtr->ny && new_z>-1 && new_z<inputImgPtr->nz){
                    shiftedIndex = (new_z*inputImgPtr->ny+new_y)*inputImgPtr->nx+new_x;
                    shiftImageData[currentIndex]=inputData[shiftedIndex];
                }
                else {
                    shiftImageData[currentIndex]=std::numeric_limits<DTYPE>::quiet_NaN();
                }
                currentIndex++;
            }
        }
    }
}
/* *************************************************************** */
template <class DTYPE>
void GetMINDImageDesciptor_core(nifti_image* inputImgPtr,
                                nifti_image* MINDImgPtr,
                                int *maskPtr)
{
    const size_t voxNumber = (size_t)inputImgPtr->nx *
                             inputImgPtr->ny * inputImgPtr->nz;
    size_t voxIndex;

    // Create a pointer to the descriptor image
    DTYPE* MINDImgDataPtr = static_cast<DTYPE *>(MINDImgPtr->data);

    // Allocate an image to store the mean image
    nifti_image *mean_img = nifti_copy_nim_info(inputImgPtr);
    mean_img->data=(void *)calloc(inputImgPtr->nvox,inputImgPtr->nbyper);
    DTYPE* meanImgDataPtr = static_cast<DTYPE *>(mean_img->data);

    // Allocate an image to store the warped image
    nifti_image *warpedImage = nifti_copy_nim_info(inputImgPtr);
    warpedImage->data = (void *)malloc(warpedImage->nvox*warpedImage->nbyper);

    // Allocation of the difference image
    nifti_image *diff_image = nifti_copy_nim_info(inputImgPtr);
    diff_image->data = (void *) malloc(diff_image->nvox*diff_image->nbyper);

    // Define the sigma for the convolution
    float sigma = -0.5;// negative value denotes voxel width

    //2D version
    int samplingNbr = (inputImgPtr->nz > 1) ? 6 : 4;
    int RSampling3D_x[6] = {-1, 1,  0, 0,  0, 0};
    int RSampling3D_y[6] = {0,  0, -1, 1,  0, 0};
    int RSampling3D_z[6] = {0,  0,  0, 0, -1, 1};

    for(int i=0;i<samplingNbr;i++) {
        ShiftImage<DTYPE>(inputImgPtr, warpedImage,
                          RSampling3D_x[i], RSampling3D_y[i], RSampling3D_z[i]);
        reg_tools_substractImageToImage(inputImgPtr, warpedImage, diff_image);
        reg_tools_multiplyImageToImage(diff_image, diff_image, diff_image);
        reg_tools_kernelConvolution(diff_image, &sigma, 0, maskPtr, NULL, NULL);
        reg_tools_addImageToImage(mean_img, diff_image, mean_img);

        // Store the current descriptor
        unsigned int index = i * diff_image->nvox;
        memcpy(&MINDImgDataPtr[index], diff_image->data,
               diff_image->nbyper * diff_image->nvox);
    }
    // Compute the mean over the number of sample
    reg_tools_divideValueToImage(mean_img, mean_img, samplingNbr);

    // Compute the MIND desccriptor
    int mindIndex;
    for(voxIndex=0;voxIndex<voxNumber;voxIndex++) {

        if(maskPtr[voxIndex]!=0){
            // Get the mean value for the current voxel
            DTYPE meanValue = meanImgDataPtr[voxIndex];
            if(meanValue == 0) {
                meanValue = std::numeric_limits<DTYPE>::epsilon();
            }
            DTYPE max_desc = 0;
            mindIndex=voxIndex;
            for(int t=0;t<samplingNbr;t++) {
                DTYPE descValue = (DTYPE)exp(-MINDImgDataPtr[mindIndex]/meanValue);
                MINDImgDataPtr[mindIndex] = descValue;
                max_desc = std::max(max_desc, descValue);
                mindIndex+=voxNumber;
            }

            mindIndex=voxIndex;
            for(int t=0;t<samplingNbr;t++) {
                DTYPE descValue = MINDImgDataPtr[mindIndex];
                MINDImgDataPtr[mindIndex] = descValue/max_desc;
                mindIndex+=voxNumber;
            }
        } // mask
    } // voxIndex
    //FREE MEMORY
    nifti_image_free(diff_image);
    nifti_image_free(warpedImage);
    nifti_image_free(mean_img);
}
/* *************************************************************** */
void GetMINDImageDesciptor(nifti_image* inputImgPtr,
                           nifti_image* MINDImgPtr,
                           int *maskPtr) {
#ifndef NDEBUG
    reg_print_fct_debug("reg_mind -- GetMINDImageDesciptor()");
#endif
    if(inputImgPtr->datatype != MINDImgPtr->datatype) {
        reg_print_fct_error("reg_mind -- GetMINDImageDesciptor");
        reg_print_msg_error("The input image and the MIND image must have the same datatype !");
        reg_exit(EXIT_FAILURE);
    }

    switch (inputImgPtr->datatype)
    {
    case NIFTI_TYPE_FLOAT32:
        GetMINDImageDesciptor_core<float>(inputImgPtr, MINDImgPtr, maskPtr);
        break;
    case NIFTI_TYPE_FLOAT64:
        GetMINDImageDesciptor_core<double>(inputImgPtr, MINDImgPtr, maskPtr);
        break;
    default:
        reg_print_fct_error("reg_mind -- GetMINDImageDesciptor");
        reg_print_msg_error("Input image datatype not supported");
        reg_exit(EXIT_FAILURE);
        break;
    }
}
