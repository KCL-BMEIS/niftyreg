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

/* *************************************************************** */
reg_mind::reg_mind()
    : reg_ssd()
{
    memset(this->activeTimePointDescriptor,0,255*sizeof(bool) );
    //4TM
    //platform = new Platform(NR_PLATFORM_CPU);
    //convolutionKernel = this->platform->createKernel(ConvolutionKernel::getName(), NULL);

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

   // Check that the input images have the same number of time point
   if(this->referenceImagePointer->nt != this->floatingImagePointer->nt)
   {
      reg_print_fct_error("reg_mind::InitialiseMeasure");
      reg_print_msg_error("This number of time point should be the same for both input images");
      reg_exit(1);
   }

   int dim = (this->referenceImagePointer->nz > 1) ? 3 : 2;
   this->referenceImageDescriptor = nifti_copy_nim_info(this->referenceImagePointer);
   this->floatingImageDescriptor = nifti_copy_nim_info(this->floatingImagePointer);
   if(this->warpedFloatingImagePointer != NULL) {
       this->warpedFloatingImageDescriptor = nifti_copy_nim_info(this->warpedFloatingImagePointer);
   } else {
       this->warpedFloatingImageDescriptor = NULL;
   }
   if(this->warpedReferenceImagePointer != NULL) {
       this->warpedReferenceImageDescriptor = nifti_copy_nim_info(this->warpedReferenceImagePointer);
   } else {
       this->warpedReferenceImageDescriptor = NULL;
   }

   if(dim == 2) {
       this->referenceImageDescriptor->data=(void *)calloc(this->referenceImagePointer->nvox * 4,this->referenceImagePointer->nbyper);
       this->referenceImageDescriptor->dim[0] = 4;
       this->referenceImageDescriptor->dim[4] = 4;
       this->referenceImageDescriptor->nt = referenceImageDescriptor->dim[4];
       this->referenceImageDescriptor->nvox = this->referenceImageDescriptor->nvox*referenceImageDescriptor->dim[4];

       this->floatingImageDescriptor->data=(void *)calloc(this->floatingImagePointer->nvox * 4,this->floatingImagePointer->nbyper);
       this->floatingImageDescriptor->dim[0] = 4;
       this->floatingImageDescriptor->dim[4] = 4;
       this->floatingImageDescriptor->nt = floatingImageDescriptor->dim[4];
       this->floatingImageDescriptor->nvox = this->floatingImageDescriptor->nvox*floatingImageDescriptor->dim[4];

       if(this->warpedFloatingImageDescriptor != NULL) {
       this->warpedFloatingImageDescriptor->data=(void *)calloc(this->warpedFloatingImagePointer->nvox * 4,this->warpedFloatingImagePointer->nbyper);
       this->warpedFloatingImageDescriptor->dim[0] = 4;
       this->warpedFloatingImageDescriptor->dim[4] = 4;
       this->warpedFloatingImageDescriptor->nt = warpedFloatingImageDescriptor->dim[4];
       this->warpedFloatingImageDescriptor->nvox = this->warpedFloatingImageDescriptor->nvox*warpedFloatingImageDescriptor->dim[4];
       }

       if(this->warpedReferenceImageDescriptor != NULL) {
       this->warpedReferenceImageDescriptor->data=(void *)calloc(this->warpedReferenceImagePointer->nvox * 4,this->warpedReferenceImagePointer->nbyper);
       this->warpedReferenceImageDescriptor->dim[0] = 4;
       this->warpedReferenceImageDescriptor->dim[4] = 4;
       this->warpedReferenceImageDescriptor->nt = warpedReferenceImageDescriptor->dim[4];
       this->warpedReferenceImageDescriptor->nvox = this->warpedReferenceImageDescriptor->nvox*warpedReferenceImageDescriptor->dim[4];
       }
   }
   else if (dim == 3) {
       this->referenceImageDescriptor->data=(void *)calloc(this->referenceImagePointer->nvox * 6,this->referenceImagePointer->nbyper);
       this->referenceImageDescriptor->dim[0] = 4;
       this->referenceImageDescriptor->dim[4] = 6;
       this->referenceImageDescriptor->nt = referenceImageDescriptor->dim[4];
       this->referenceImageDescriptor->nvox = this->referenceImageDescriptor->nvox*referenceImageDescriptor->dim[4];

       this->floatingImageDescriptor->data=(void *)calloc(this->floatingImagePointer->nvox * 6,this->floatingImagePointer->nbyper);
       this->floatingImageDescriptor->dim[0] = 4;
       this->floatingImageDescriptor->dim[4] = 6;
       this->floatingImageDescriptor->nt = floatingImageDescriptor->dim[4];
       this->floatingImageDescriptor->nvox = this->floatingImageDescriptor->nvox*floatingImageDescriptor->dim[4];

       if(this->warpedFloatingImageDescriptor != NULL) {
       this->warpedFloatingImageDescriptor->data=(void *)calloc(this->warpedFloatingImagePointer->nvox * 6,this->warpedFloatingImagePointer->nbyper);
       this->warpedFloatingImageDescriptor->dim[0] = 4;
       this->warpedFloatingImageDescriptor->dim[4] = 6;
       this->warpedFloatingImageDescriptor->nt = warpedFloatingImageDescriptor->dim[4];
       this->warpedFloatingImageDescriptor->nvox = this->warpedFloatingImageDescriptor->nvox*warpedFloatingImageDescriptor->dim[4];
       }

       if(this->warpedReferenceImageDescriptor != NULL) {
       this->warpedReferenceImageDescriptor->data=(void *)calloc(this->warpedReferenceImagePointer->nvox * 6,this->warpedReferenceImagePointer->nbyper);
       this->warpedReferenceImageDescriptor->dim[0] = 4;
       this->warpedReferenceImageDescriptor->dim[4] = 6;
       this->warpedReferenceImageDescriptor->nt = warpedReferenceImageDescriptor->dim[4];
       this->warpedReferenceImageDescriptor->nvox = this->warpedReferenceImageDescriptor->nvox*warpedReferenceImageDescriptor->dim[4];
       }

   } else {
       reg_print_msg_error("dimension not supported");
       reg_exit(EXIT_FAILURE);
   }

   for(int i=0;i<referenceImageDescriptor->nt;++i) {
      this->activeTimePointDescriptor[i]=true;
   }

   //MIND
   GetMINDImageDesciptor(this->referenceImagePointer, this->referenceImageDescriptor);
   if(this->isSymmetric) {
      GetMINDImageDesciptor(this->floatingImagePointer, this->floatingImageDescriptor);
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
    if(this->referenceImageDescriptor != NULL) {
        nifti_image_free(this->referenceImageDescriptor);
        this->referenceImageDescriptor = NULL;
    }
    if(this->warpedFloatingImageDescriptor != NULL) {
        nifti_image_free(this->warpedFloatingImageDescriptor);
        this->warpedFloatingImageDescriptor = NULL;
    }
    if(this->floatingImageDescriptor != NULL) {
        nifti_image_free(this->floatingImageDescriptor);
        this->floatingImageDescriptor = NULL;
    }
    if(this->warpedReferenceImageDescriptor != NULL) {
        nifti_image_free(this->warpedReferenceImageDescriptor);
        this->warpedReferenceImageDescriptor = NULL;
    }
}
/* *************************************************************** */
void reg_mind::GetVoxelBasedSimilarityMeasureGradient()
{

}
/* *************************************************************** */
double reg_mind::GetSimilarityMeasureValue() {

    // Check that all the specified image are of the same datatype
    if(this->referenceImageDescriptor->datatype != this->referenceImageDescriptor->datatype)
    {
       reg_print_fct_error("reg_mind::GetSimilarityMeasureValue");
       reg_print_msg_error("Both input images are exepected to have the same type");
       reg_exit(1);
    }

    GetMINDImageDesciptor(this->warpedFloatingImagePointer, this->warpedFloatingImageDescriptor);

    double SSDValue;
    switch(this->referenceImageDescriptor->datatype)
    {
    case NIFTI_TYPE_FLOAT32:
       SSDValue = reg_getSSDValue<float>
                  (this->referenceImageDescriptor,
                   this->warpedFloatingImageDescriptor,
                   this->activeTimePointDescriptor,
                   NULL, // HERE TODO this->forwardJacDetImagePointer,
                   this->referenceMaskPointer,
                   this->currentValue
                  );
       break;
    case NIFTI_TYPE_FLOAT64:
       SSDValue = reg_getSSDValue<double>
                  (this->referenceImageDescriptor,
                   this->warpedFloatingImageDescriptor,
                   this->activeTimePointDescriptor,
                   NULL, // HERE TODO this->forwardJacDetImagePointer,
                   this->referenceMaskPointer,
                   this->currentValue
                  );
       break;
    default:
       reg_print_fct_error("reg_ssd::GetSimilarityMeasureValue");
       reg_print_msg_error("Warped pixel type unsupported");
       reg_exit(1);
    }

    // Backward computation
    if(this->isSymmetric)
    {
       // Check that all the specified image are of the same datatype
       if(this->warpedReferenceImageDescriptor->datatype != this->floatingImageDescriptor->datatype)
       {
          reg_print_fct_error("reg_ssd::GetSimilarityMeasureValue");
          reg_print_msg_error("Both input images are exepected to have the same type");
          reg_exit(1);
       }

       GetMINDImageDesciptor(this->warpedReferenceImagePointer, this->warpedReferenceImageDescriptor);

       switch(this->floatingImageDescriptor->datatype)
       {
       case NIFTI_TYPE_FLOAT32:
          SSDValue += reg_getSSDValue<float>
                      (this->floatingImageDescriptor,
                       this->warpedReferenceImageDescriptor,
                       this->activeTimePointDescriptor,
                       NULL, // HERE TODO this->backwardJacDetImagePointer,
                       this->floatingMaskPointer,
                       this->currentValue
                      );
          break;
       case NIFTI_TYPE_FLOAT64:
          SSDValue += reg_getSSDValue<double>
                      (this->floatingImageDescriptor,
                       this->warpedReferenceImageDescriptor,
                       this->activeTimePointDescriptor,
                       NULL, // HERE TODO this->backwardJacDetImagePointer,
                       this->floatingMaskPointer,
                       this->currentValue
                      );
          break;
       default:
          reg_print_fct_error("reg_ssd::GetSimilarityMeasureValue");
          reg_print_msg_error("Warped pixel type unsupported");
          reg_exit(1);
       }
    }
    return SSDValue;///(double) this->referenceImageDescriptor->nt;
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
    //we suppose that the allocation is already done
    //shiftedImgPtr->data = (void *)calloc(inputImgPtr->nvox, inputImgPtr->nbyper);

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
template <class InputTYPE>
void GetMINDImageDesciptor1(nifti_image* inputImgPtr, nifti_image* MINDImgPtr) {

    InputTYPE* MINDImgDataPtr = static_cast<InputTYPE *>(MINDImgPtr->data);
    //Mean image
    nifti_image *mean_img = nifti_copy_nim_info(inputImgPtr);
    mean_img->data=(void *)calloc(inputImgPtr->nvox,inputImgPtr->nbyper);
    InputTYPE* meanImgDataPtr = static_cast<InputTYPE *>(mean_img->data);

    //warpedImage
    nifti_image *warpedImage = nifti_copy_nim_info(inputImgPtr);
    warpedImage->data = (void *)malloc(warpedImage->nvox*warpedImage->nbyper);

    //Convolution
    nifti_image *diff_image = nifti_copy_nim_info(inputImgPtr);
    diff_image->data = (void *) malloc(diff_image->nvox*diff_image->nbyper);

    float sigma = -0.5;//voxel based

    int dim = (inputImgPtr->nz > 1) ? 3 : 2;

    if (dim == 2) {
        //2D version
        const int samplingNbr = 4;
        int RSampling2D_x[samplingNbr] = { -1, 1,  0, 0};
        int RSampling2D_y[samplingNbr] = { 0,  0, -1, 1};
        int RSampling2D_z[samplingNbr] = { 0,  0,  0, 0};

        for(int i=0;i<samplingNbr;i++) {

            ShiftImage<InputTYPE>(inputImgPtr, warpedImage, RSampling2D_x[i], RSampling2D_y[i], RSampling2D_z[i]);

            reg_tools_substractImageToImage(inputImgPtr, warpedImage, diff_image);

            reg_tools_multiplyImageToImage(diff_image, diff_image, diff_image);

            //this->convolutionKernel->castTo<ConvolutionKernel>()->calculate(diff_image, &sigma, 0, NULL, NULL, NULL);
            reg_tools_kernelConvolution(diff_image, &sigma, 0, NULL, NULL, NULL);

            //Mean
            reg_tools_addImageToImage(mean_img, diff_image, mean_img);

            //Let's store the result - we assume that MINDImgPtr has been well initialized
            //MIND datatype
            unsigned int index = i * diff_image->nvox;
            memcpy(&MINDImgDataPtr[index], diff_image->data,
                   diff_image->nbyper * diff_image->nvox);
        }
        //Let's calculate the mean over the t values
        reg_tools_divideValueToImage(mean_img, mean_img, samplingNbr);

        //Let's calculate the MIND desccriptor
        int currentMeanIndex=0;
        for(int y=0;y<MINDImgPtr->ny;y++) {
            for(int x=0;x<MINDImgPtr->nx;x++) {

                InputTYPE meanValue = meanImgDataPtr[currentMeanIndex];
                if(meanValue == 0) {
                    meanValue = std::numeric_limits<InputTYPE>::epsilon();
                }
                InputTYPE max_t = 0;
                for(int t=0;t<samplingNbr;t++) {
                    int currentMINDIndex = currentMeanIndex +
                                           diff_image->nvox * t;

                    MINDImgDataPtr[currentMINDIndex] =
                            (float) exp((double) -MINDImgDataPtr[currentMINDIndex]/(double) meanValue);
                    max_t = std::max(max_t,MINDImgDataPtr[currentMINDIndex]);
                }

                for(int t=0;t<samplingNbr;t++) {
                    int currentMINDIndex = currentMeanIndex +
                                           diff_image->nvox * t;

                    MINDImgDataPtr[currentMINDIndex] = (float)(MINDImgDataPtr[currentMINDIndex]/(double)max_t);
                }
                currentMeanIndex++;
            } // x
        } // y

    } else if (dim == 3) {
        //3D version
        const int samplingNbr = 6;
        int RSampling3D_x[samplingNbr] = {-1, 1,  0, 0,  0, 0};
        int RSampling3D_y[samplingNbr] = {0,  0, -1, 1,  0, 0};
        int RSampling3D_z[samplingNbr] = {0,  0,  0, 0, -1, 1};

        for(int i=0;i<samplingNbr;i++) {

            ShiftImage<InputTYPE>(inputImgPtr, warpedImage, RSampling3D_x[i], RSampling3D_y[i], RSampling3D_z[i]);

            //Convolution
            //I think I will have to make some padding - resampling before that...
            reg_tools_substractImageToImage(inputImgPtr, warpedImage, diff_image);
            reg_tools_multiplyImageToImage(diff_image, diff_image, diff_image);
            //this->convolutionKernel->castTo<ConvolutionKernel>()->calculate(diff_image, &sigma, 0, NULL, NULL, NULL);
            reg_tools_kernelConvolution(diff_image, &sigma, 0, NULL, NULL, NULL);
            //Mean
            reg_tools_addImageToImage(mean_img, diff_image, mean_img);

            //Let's store the result - we assume that MINDImgPtr has been well initialized
            //MIND datatype
            unsigned int index = i * diff_image->nvox;
            memcpy(&MINDImgDataPtr[index], diff_image->data, diff_image->nbyper * diff_image->nvox);
        }
        //Let's calculate the mean over the t values
        reg_tools_divideValueToImage(mean_img,mean_img,samplingNbr);
        //Let's calculate the MIND desccriptor
        for(int z=0;z<MINDImgPtr->nz;z++) {
            for(int y=0;y<MINDImgPtr->ny;y++) {
                for(int x=0;x<MINDImgPtr->nx;x++) {

                    int currentMeanIndex = x+
                                           mean_img->nx * y +
                                           mean_img->nx * mean_img->ny * z;
                    InputTYPE divValue = meanImgDataPtr[currentMeanIndex];
                    if(divValue == 0) {
                        divValue = std::numeric_limits<InputTYPE>::epsilon();
                    }
                    InputTYPE max_t = 0;
                    for(int t=0;t<MINDImgPtr->nt;t++) {
                            int currentMINDIndex = x+
                                                   MINDImgPtr->nx * y +
                                                   MINDImgPtr->nx * MINDImgPtr->ny * z +
                                                   MINDImgPtr->nx * MINDImgPtr->ny * MINDImgPtr->nz * t;

                            MINDImgDataPtr[currentMINDIndex] =
                                    (float) (exp((double)-MINDImgDataPtr[currentMINDIndex]/(double)divValue));
                            max_t = std::max(max_t,MINDImgDataPtr[currentMINDIndex]);
                    }

                    for(int t=0;t<MINDImgPtr->nt;t++) {
                        int currentMINDIndex = x+
                                               MINDImgPtr->nx * y +
                                               MINDImgPtr->nx * MINDImgPtr->ny * z +
                                               MINDImgPtr->nx * MINDImgPtr->ny * MINDImgPtr->nz * t;

                        MINDImgDataPtr[currentMINDIndex] = (float)((double)MINDImgDataPtr[currentMINDIndex] / max_t);

                    }
                }
            }
        }

    } else {
        //Error
        reg_print_fct_error("The input image has to be a 2D or a 3D image");
    }
    //FREE MEMORY
    nifti_image_free(diff_image);
    nifti_image_free(warpedImage);
    nifti_image_free(mean_img);
}
/* *************************************************************** */
void GetMINDImageDesciptor(nifti_image* inputImgPtr, nifti_image* MINDImgPtr) {
#ifndef NDEBUG
    reg_print_fct_debug("reg_mind -- GetMINDImageDesciptor()");
#endif
    //SECURITY
    if(inputImgPtr->datatype != MINDImgPtr->datatype) {
        reg_print_fct_error("reg_mind -- GetMINDImageDesciptor");
        reg_print_msg_error("The input image and the MIND image must have the same datatype !");
        reg_exit(EXIT_FAILURE);
    }

    switch (inputImgPtr->datatype)
    {
        case NIFTI_TYPE_FLOAT32:
            GetMINDImageDesciptor1<float>(inputImgPtr, MINDImgPtr);
            break;
        case NIFTI_TYPE_FLOAT64:
            GetMINDImageDesciptor1<double>(inputImgPtr, MINDImgPtr);
            break;
        default:
            reg_print_fct_error("reg_mind -- GetMINDImageDesciptor");
            reg_print_msg_error("Input image datatype not supported");
            reg_exit(EXIT_FAILURE);
            break;
    }
}
