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

   int currentIndex;
   int shiftedIndex;

   int x, y, z, old_x, old_y, old_z;

#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(inputData, shiftImageData, shiftedImgPtr, inputImgPtr, \
   maskPtr, tx, ty, tz) \
   private(x, y, z, old_x, old_y, old_z, shiftedIndex, \
   currentIndex)
#endif
   for (z=0;z<shiftedImgPtr->nz;z++) {
      currentIndex = z * shiftedImgPtr->nx * shiftedImgPtr->ny;
      old_z = z-tz;
      for (y=0;y<shiftedImgPtr->ny;y++) {
         old_y = y-ty;
         for (x=0;x<shiftedImgPtr->nx;x++) {
            old_x = x-tx;
            if(old_x>-1 && old_x<inputImgPtr->nx &&
                  old_y>-1 && old_y<inputImgPtr->ny &&
                  old_z>-1 && old_z<inputImgPtr->nz){
               shiftedIndex = (old_z*inputImgPtr->ny+old_y)*inputImgPtr->nx+old_x;
               if(maskPtr[shiftedIndex]>-1) {
                  shiftImageData[currentIndex]=inputData[shiftedIndex];
               } // mask is not defined
               else{
                  //shiftImageData[currentIndex]=std::numeric_limits<DTYPE>::quiet_NaN();
                   shiftImageData[currentIndex]=0.0;
               }
            } // outside of the image
            else{
               //shiftImageData[currentIndex]=std::numeric_limits<DTYPE>::quiet_NaN();
                shiftImageData[currentIndex]=0.0;
            }
            currentIndex++;
         }
      }
   }
}
/* *************************************************************** */
template<class DTYPE>
void spatialGradient(nifti_image *img,
                     nifti_image *gradImg,
                     int *mask,
                     int desc_index)
{
   size_t voxIndex, voxelNumber = (size_t)img->nx *
         img->ny * img->nz;

   int dimImg = img->nz > 1 ? 3 : 2;
   int x, y, z;

   DTYPE *imgPtr = static_cast<DTYPE *>(img->data);
   DTYPE *currentImgPtr = &imgPtr[desc_index*voxelNumber];

   DTYPE *gradPtrX = static_cast<DTYPE *>(gradImg->data);
   DTYPE *gradPtrY = &gradPtrX[voxelNumber];
   DTYPE *gradPtrZ = NULL;
   if(dimImg==3)
      gradPtrZ = &gradPtrY[voxelNumber];

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
               if(x<img->nx-1 && x>0)
                  gradPtrX[voxIndex] =  (currentImgPtr[voxIndex+1] - currentImgPtr[voxIndex-1]) / 2.f;
               else gradPtrX[voxIndex] = 0.f;
               if(gradPtrX[voxIndex]!=gradPtrX[voxIndex]) gradPtrX[voxIndex]=0.;
               if(y<img->ny-1 && y>0)
                  gradPtrY[voxIndex] = (currentImgPtr[voxIndex+img->nx] - currentImgPtr[voxIndex-img->nx]) / 2.f;
               else gradPtrY[voxIndex] = 0.f;
               if(gradPtrY[voxIndex]!=gradPtrY[voxIndex]) gradPtrY[voxIndex]=0.;
               if(gradPtrZ!=NULL){
                  if(z<img->nz-1 && z>0)
                     gradPtrZ[voxIndex] = (currentImgPtr[voxIndex+img->nx*img->ny] - currentImgPtr[voxIndex-img->nx*img->ny]) / 2.f;
                  else gradPtrZ[voxIndex] = 0.f;
                  if(gradPtrZ[voxIndex]!=gradPtrZ[voxIndex]) gradPtrZ[voxIndex]=0.;
               }
            }
            ++voxIndex;
         } // x
      } // y
   } // z
}
template void spatialGradient<float>(nifti_image *img, nifti_image *gradImg, int *mask, int);
template void spatialGradient<double>(nifti_image *img, nifti_image *gradImg, int *mask, int);
/* *************************************************************** */
template <class DTYPE>
void GetMINDImageDesciptor_core(nifti_image* inputImage,
                                nifti_image* MINDImage,
                                int *maskPtr,
                                int descriptorOffset,
                                int current_timepoint)
{
#ifdef WIN32
   const long voxelNumber = (long)inputImage->nx *
         inputImage->ny * inputImage->nz;
   long voxelIndex;
#else
   const size_t voxelNumber = (size_t)inputImage->nx *
         inputImage->ny * inputImage->nz;
   size_t voxelIndex;
#endif

   // Create a pointer to the descriptor image
   DTYPE* MINDImgDataPtr = static_cast<DTYPE *>(MINDImage->data);

   // Allocate an image to store the current timepoint reference image
   nifti_image *currentInputImage = nifti_copy_nim_info(inputImage);
   currentInputImage->ndim=currentInputImage->dim[0]=inputImage->nz>1?3:2;
   currentInputImage->nt=currentInputImage->dim[4]=1;
   currentInputImage->nvox=voxelNumber;
   DTYPE *inputImagePtr = static_cast<DTYPE *>(inputImage->data);
   currentInputImage->data = static_cast<void *>(&inputImagePtr[current_timepoint*voxelNumber]);

   // Allocate an image to store the mean image
   nifti_image *meanImage = nifti_copy_nim_info(currentInputImage);
   meanImage->data=(void *)calloc(meanImage->nvox,meanImage->nbyper);
   DTYPE* meanImgDataPtr = static_cast<DTYPE *>(meanImage->data);

   // Allocate an image to store the shifted image
   nifti_image *shiftedImage = nifti_copy_nim_info(currentInputImage);
   shiftedImage->data = (void *)malloc(shiftedImage->nvox*shiftedImage->nbyper);

   // Allocation of the difference image
   nifti_image *diff_image = nifti_copy_nim_info(currentInputImage);
   diff_image->data = (void *) malloc(diff_image->nvox*diff_image->nbyper);

   // Define the sigma for the convolution
   float sigma = -0.5;// negative value denotes voxel width

   //2D version
   int samplingNbr = (currentInputImage->nz > 1) ? 6 : 4;
   int RSampling3D_x[6] = {-descriptorOffset, descriptorOffset,  0, 0,  0, 0};
   int RSampling3D_y[6] = {0,  0, -descriptorOffset, descriptorOffset,  0, 0};
   int RSampling3D_z[6] = {0,  0,  0, 0, -descriptorOffset, descriptorOffset};

   for(int i=0;i<samplingNbr;i++) {
      ShiftImage<DTYPE>(currentInputImage, shiftedImage, maskPtr,
                        RSampling3D_x[i], RSampling3D_y[i], RSampling3D_z[i]);
      reg_tools_substractImageToImage(currentInputImage, shiftedImage, diff_image);
      reg_tools_multiplyImageToImage(diff_image, diff_image, diff_image);
      reg_tools_kernelConvolution(diff_image, &sigma, GAUSSIAN_KERNEL, maskPtr);
      reg_tools_addImageToImage(meanImage, diff_image, meanImage);

      // Store the current descriptor
      unsigned int index = i * diff_image->nvox;
      memcpy(&MINDImgDataPtr[index], diff_image->data,
             diff_image->nbyper * diff_image->nvox);
   }
   // Compute the mean over the number of sample
   reg_tools_divideValueToImage(meanImage, meanImage, samplingNbr);

   // Compute the MIND desccriptor
   int mindIndex;
   DTYPE meanValue, max_desc, descValue;
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(samplingNbr, maskPtr, meanImgDataPtr, \
   MINDImgDataPtr) \
   private(voxelIndex, meanValue, max_desc, descValue, mindIndex)
#endif
   for(voxelIndex=0;voxelIndex<voxelNumber;voxelIndex++) {

      if(maskPtr[voxelIndex]>-1){
         // Get the mean value for the current voxel
         meanValue = meanImgDataPtr[voxelIndex];
         if(meanValue == 0) {
            meanValue = std::numeric_limits<DTYPE>::epsilon();
         }
         max_desc = 0;
         mindIndex=voxelIndex;
         for(int t=0;t<samplingNbr;t++) {
            descValue = (DTYPE)exp(-MINDImgDataPtr[mindIndex]/meanValue);
            MINDImgDataPtr[mindIndex] = descValue;
            max_desc = (std::max)(max_desc, descValue);
            mindIndex+=voxelNumber;
         }

         mindIndex=voxelIndex;
         for(int t=0;t<samplingNbr;t++) {
            descValue = MINDImgDataPtr[mindIndex];
            MINDImgDataPtr[mindIndex] = descValue/max_desc;
            mindIndex+=voxelNumber;
         }
      } // mask
   } // voxIndex
   // Mr Propre
   nifti_image_free(diff_image);
   nifti_image_free(shiftedImage);
   nifti_image_free(meanImage);
   currentInputImage->data=NULL;
   nifti_image_free(currentInputImage);
}
/* *************************************************************** */
void GetMINDImageDesciptor(nifti_image* inputImgPtr,
                           nifti_image* MINDImgPtr,
                           int *maskPtr,
                           int descriptorOffset,
                           int current_timepoint) {
#ifndef NDEBUG
   reg_print_fct_debug("GetMINDImageDesciptor()");
#endif
   if(inputImgPtr->datatype != MINDImgPtr->datatype) {
      reg_print_fct_error("reg_mind -- GetMINDImageDesciptor");
      reg_print_msg_error("The input image and the MIND image must have the same datatype !");
      reg_exit();
   }

   switch (inputImgPtr->datatype)
   {
   case NIFTI_TYPE_FLOAT32:
      GetMINDImageDesciptor_core<float>(inputImgPtr, MINDImgPtr, maskPtr, descriptorOffset, current_timepoint);
      break;
   case NIFTI_TYPE_FLOAT64:
      GetMINDImageDesciptor_core<double>(inputImgPtr, MINDImgPtr, maskPtr, descriptorOffset, current_timepoint);
      break;
   default:
      reg_print_fct_error("GetMINDImageDesciptor");
      reg_print_msg_error("Input image datatype not supported");
      reg_exit();
      break;
   }
}
/* *************************************************************** */
template <class DTYPE>
void GetMINDSSCImageDesciptor_core(nifti_image* inputImage,
                                   nifti_image* MINDSSCImage,
                                   int *maskPtr,
                                   int descriptorOffset,
                                   int current_timepoint)
{

#ifdef WIN32
   const long voxelNumber = (long)inputImage->nx *
         inputImage->ny * inputImage->nz;
   long voxelIndex;
#else
   const size_t voxelNumber = (size_t)inputImage->nx *
         inputImage->ny * inputImage->nz;
   size_t voxelIndex;
#endif

   // Create a pointer to the descriptor image
   DTYPE* MINDSSCImgDataPtr = static_cast<DTYPE *>(MINDSSCImage->data);

   // Allocate an image to store the current timepoint reference image
   nifti_image *currentInputImage = nifti_copy_nim_info(inputImage);
   currentInputImage->ndim=currentInputImage->dim[0]=inputImage->nz>1?3:2;
   currentInputImage->nt=currentInputImage->dim[4]=1;
   currentInputImage->nvox=voxelNumber;
   DTYPE *inputImagePtr = static_cast<DTYPE *>(inputImage->data);
   currentInputImage->data = static_cast<void *>(&inputImagePtr[current_timepoint*voxelNumber]);

   // Allocate an image to store the mean image
   nifti_image *mean_img = nifti_copy_nim_info(currentInputImage);
   mean_img->data=(void *)calloc(mean_img->nvox,mean_img->nbyper);
   DTYPE* meanImgDataPtr = static_cast<DTYPE *>(mean_img->data);

   // Allocate an image to store the warped image
   nifti_image *shiftedImage = nifti_copy_nim_info(currentInputImage);
   shiftedImage->data = (void *)malloc(shiftedImage->nvox*shiftedImage->nbyper);

   // Define the sigma for the convolution
   float sigma = -0.5;// negative value denotes voxel width
   //float sigma = -1.0;// negative value denotes voxel width

   //2D version
   int samplingNbr = (currentInputImage->nz > 1) ? 6 : 2;
   int lengthDescriptor = (currentInputImage->nz > 1) ? 12 : 4;

   // Allocation of the difference image
   //std::vector<nifti_image *> vectNiftiImage;
   //for(int i=0;i<samplingNbr;i++) {
   nifti_image *diff_image = nifti_copy_nim_info(currentInputImage);
   diff_image->data = (void *) malloc(diff_image->nvox*diff_image->nbyper);
   int *mask_diff_image = (int *)calloc(diff_image->nvox, sizeof(int));

   nifti_image *diff_imageShifted = nifti_copy_nim_info(currentInputImage);
   diff_imageShifted->data = (void *) malloc(diff_imageShifted->nvox*diff_imageShifted->nbyper);

   int RSampling3D_x[6] = {+descriptorOffset,+descriptorOffset,-descriptorOffset,+0,+descriptorOffset,+0};
   int RSampling3D_y[6] = {+descriptorOffset,-descriptorOffset,+0,-descriptorOffset,+0,+descriptorOffset};
   int RSampling3D_z[6] = {+0,+0,+descriptorOffset,+descriptorOffset,+descriptorOffset,+descriptorOffset};

   int tx[12]={-descriptorOffset,+0,-descriptorOffset,+0,+0,+descriptorOffset,+0,+0,+0,-descriptorOffset,+0,+0};
   int ty[12]={+0,-descriptorOffset,+0,+descriptorOffset,+0,+0,+0,+descriptorOffset,+0,+0,+0,-descriptorOffset};
   int tz[12]={+0,+0,+0,+0,-descriptorOffset,+0,-descriptorOffset,+0,-descriptorOffset,+0,-descriptorOffset,+0};
   int compteurId = 0;

   for(int i=0;i<samplingNbr;i++) {
      ShiftImage<DTYPE>(currentInputImage, shiftedImage, maskPtr,
                        RSampling3D_x[i], RSampling3D_y[i], RSampling3D_z[i]);
      reg_tools_substractImageToImage(currentInputImage, shiftedImage, diff_image);
      reg_tools_multiplyImageToImage(diff_image, diff_image, diff_image);
      reg_tools_kernelConvolution(diff_image, &sigma, GAUSSIAN_KERNEL, maskPtr);

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
   private(voxelIndex, meanValue, max_desc, descValue, mindIndex)
#endif
   for(voxelIndex=0;voxelIndex<voxelNumber;voxelIndex++) {

      if(maskPtr[voxelIndex]>-1){
         // Get the mean value for the current voxel
         meanValue = meanImgDataPtr[voxelIndex];
         if(meanValue == 0) {
            meanValue = std::numeric_limits<DTYPE>::epsilon();
         }
         max_desc = 0;
         mindIndex=voxelIndex;
         for(int t=0;t<lengthDescriptor;t++) {
            descValue = (DTYPE)exp(-MINDSSCImgDataPtr[mindIndex]/meanValue);
            MINDSSCImgDataPtr[mindIndex] = descValue;
            max_desc = std::max(max_desc, descValue);
            mindIndex+=voxelNumber;
         }

         mindIndex=voxelIndex;
         for(int t=0;t<lengthDescriptor;t++) {
            descValue = MINDSSCImgDataPtr[mindIndex];
            MINDSSCImgDataPtr[mindIndex] = descValue/max_desc;
            mindIndex+=voxelNumber;
         }
      } // mask
   } // voxIndex
   // Mr Propre
   nifti_image_free(diff_imageShifted);
   free(mask_diff_image);
   nifti_image_free(diff_image);
   nifti_image_free(shiftedImage);
   nifti_image_free(mean_img);
   currentInputImage->data=NULL;
   nifti_image_free(currentInputImage);
}
/* *************************************************************** */
void GetMINDSSCImageDesciptor(nifti_image* inputImgPtr,
                              nifti_image* MINDSSCImgPtr,
                              int *maskPtr,
                              int descriptorOffset,
                              int current_timepoint) {
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
      GetMINDSSCImageDesciptor_core<float>(inputImgPtr, MINDSSCImgPtr, maskPtr, descriptorOffset, current_timepoint);
      break;
   case NIFTI_TYPE_FLOAT64:
      GetMINDSSCImageDesciptor_core<double>(inputImgPtr, MINDSSCImgPtr, maskPtr, descriptorOffset, current_timepoint);
      break;
   default:
      reg_print_fct_error("GetMINDSSCImageDesciptor");
      reg_print_msg_error("Input image datatype not supported");
      reg_exit();
      break;
   }
}
/* *************************************************************** */
reg_mind::reg_mind()
   : reg_ssd()
{
   memset(this->activeTimePointDescriptor,0,255*sizeof(bool) );
   this->referenceImageDescriptor=NULL;
   this->floatingImageDescriptor=NULL;
   this->warpedFloatingImageDescriptor=NULL;
   this->warpedReferenceImageDescriptor=NULL;
   this->mind_type=MIND_TYPE;
   this->descriptorOffset=1;
#ifndef NDEBUG
   reg_print_msg_debug("reg_mind constructor called");
#endif
}
/* *************************************************************** */
void reg_mind::SetDescriptorOffset(int val)
{
   this->descriptorOffset = val;
}
/* *************************************************************** */
int reg_mind::GetDescriptorOffset()
{
   return this->descriptorOffset;
}
/* *************************************************************** */
reg_mind::~reg_mind() {
   if(this->referenceImageDescriptor != NULL)
      nifti_image_free(this->referenceImageDescriptor);
   this->referenceImageDescriptor = NULL;

   if(this->warpedFloatingImageDescriptor != NULL)
      nifti_image_free(this->warpedFloatingImageDescriptor);
   this->warpedFloatingImageDescriptor = NULL;

   if(this->floatingImageDescriptor != NULL)
      nifti_image_free(this->floatingImageDescriptor);
   this->floatingImageDescriptor = NULL;

   if(this->warpedReferenceImageDescriptor != NULL)
      nifti_image_free(this->warpedReferenceImageDescriptor);
   this->warpedReferenceImageDescriptor = NULL;
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

   this->discriptor_number = 0;
   if(this->mind_type==MIND_TYPE){
      discriptor_number=this->referenceImagePointer->nz>1?6:4;
   }
   else if(this->mind_type==MINDSSC_TYPE){
      discriptor_number=this->referenceImagePointer->nz>1?12:4;

   }
   // Initialise the reference descriptor
   this->referenceImageDescriptor = nifti_copy_nim_info(this->referenceImagePointer);
   this->referenceImageDescriptor->dim[0]=this->referenceImageDescriptor->ndim=4;
   this->referenceImageDescriptor->dim[4]=this->referenceImageDescriptor->nt=this->discriptor_number;
   this->referenceImageDescriptor->nvox = (size_t)this->referenceImageDescriptor->nx*
         this->referenceImageDescriptor->ny*
         this->referenceImageDescriptor->nz*
         this->referenceImageDescriptor->nt;
   this->referenceImageDescriptor->data=(void *)malloc(this->referenceImageDescriptor->nvox*
                                                       this->referenceImageDescriptor->nbyper);
   // Initialise the warped floating descriptor
   this->warpedFloatingImageDescriptor = nifti_copy_nim_info(this->referenceImagePointer);
   this->warpedFloatingImageDescriptor->dim[0]=this->warpedFloatingImageDescriptor->ndim=4;
   this->warpedFloatingImageDescriptor->dim[4]=this->warpedFloatingImageDescriptor->nt=this->discriptor_number;
   this->warpedFloatingImageDescriptor->nvox = (size_t)this->warpedFloatingImageDescriptor->nx*
         this->warpedFloatingImageDescriptor->ny*
         this->warpedFloatingImageDescriptor->nz*
         this->warpedFloatingImageDescriptor->nt;
   this->warpedFloatingImageDescriptor->data=(void *)malloc(this->warpedFloatingImageDescriptor->nvox*
                                                            this->warpedFloatingImageDescriptor->nbyper);

   if(this->isSymmetric) {
      if(this->floatingImagePointer->nt>1 || this->warpedReferenceImagePointer->nt>1){
         reg_print_msg_error("reg_mind does not support multiple time point image");
         reg_exit();
      }
      // Initialise the floating descriptor
      this->floatingImageDescriptor = nifti_copy_nim_info(this->floatingImagePointer);
      this->floatingImageDescriptor->dim[0]=this->floatingImageDescriptor->ndim=4;
      this->floatingImageDescriptor->dim[4]=this->floatingImageDescriptor->nt=this->discriptor_number;
      this->floatingImageDescriptor->nvox = (size_t)this->floatingImageDescriptor->nx*
            this->floatingImageDescriptor->ny*
            this->floatingImageDescriptor->nz*
            this->floatingImageDescriptor->nt;
      this->floatingImageDescriptor->data=(void *)malloc(this->floatingImageDescriptor->nvox*
                                                         this->floatingImageDescriptor->nbyper);
      // Initialise the warped floating descriptor
      this->warpedReferenceImageDescriptor = nifti_copy_nim_info(this->floatingImagePointer);
      this->warpedReferenceImageDescriptor->dim[0]=this->warpedReferenceImageDescriptor->ndim=4;
      this->warpedReferenceImageDescriptor->dim[4]=this->warpedReferenceImageDescriptor->nt=this->discriptor_number;
      this->warpedReferenceImageDescriptor->nvox = (size_t)this->warpedReferenceImageDescriptor->nx*
            this->warpedReferenceImageDescriptor->ny*
            this->warpedReferenceImageDescriptor->nz*
            this->warpedReferenceImageDescriptor->nt;
      this->warpedReferenceImageDescriptor->data=(void *)malloc(this->warpedReferenceImageDescriptor->nvox*
                                                                this->warpedReferenceImageDescriptor->nbyper);
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
double reg_mind::GetSimilarityMeasureValue()
{
   double MINDValue=0.;
   for(int t=0; t<this->referenceImagePointer->nt; ++t){
      if(this->activeTimePoint[t]==true){
         size_t voxelNumber = (size_t)referenceImagePointer->nx *
               referenceImagePointer->ny * referenceImagePointer->nz;
         int *combinedMask = (int *)malloc(voxelNumber*sizeof(int));
         memcpy(combinedMask, this->referenceMaskPointer, voxelNumber*sizeof(int));
         reg_tools_removeNanFromMask(this->referenceImagePointer, combinedMask);
         reg_tools_removeNanFromMask(this->warpedFloatingImagePointer, combinedMask);

         if(this->mind_type==MIND_TYPE){
            GetMINDImageDesciptor(this->referenceImagePointer,
                                  this->referenceImageDescriptor,
                                  combinedMask,
                                  this->descriptorOffset,
                                  t);
            GetMINDImageDesciptor(this->warpedFloatingImagePointer,
                                  this->warpedFloatingImageDescriptor,
                                  combinedMask,
                                  this->descriptorOffset,
                                  t);
         }
         else if(this->mind_type==MINDSSC_TYPE){
            GetMINDSSCImageDesciptor(this->referenceImagePointer,
                                     this->referenceImageDescriptor,
                                     combinedMask,
                                     this->descriptorOffset,
                                     t);
            GetMINDSSCImageDesciptor(this->warpedFloatingImagePointer,
                                     this->warpedFloatingImageDescriptor,
                                     combinedMask,
                                     this->descriptorOffset,
                                     t);
         }

         switch(this->referenceImageDescriptor->datatype)
         {
         case NIFTI_TYPE_FLOAT32:
            MINDValue += reg_getSSDValue<float>
                  (this->referenceImageDescriptor,
                   this->warpedFloatingImageDescriptor,
                   this->activeTimePointDescriptor,
                   NULL, // HERE TODO this->forwardJacDetImagePointer,
                   combinedMask,
                   this->currentValue
                   );
            break;
         case NIFTI_TYPE_FLOAT64:
            MINDValue += reg_getSSDValue<double>
                  (this->referenceImageDescriptor,
                   this->warpedFloatingImageDescriptor,
                   this->activeTimePointDescriptor,
                   NULL, // HERE TODO this->forwardJacDetImagePointer,
                   combinedMask,
                   this->currentValue
                   );
            break;
         default:
            reg_print_fct_error("reg_mind::GetSimilarityMeasureValue");
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
            memcpy(combinedMask, this->floatingMaskPointer, voxelNumber*sizeof(int));
            reg_tools_removeNanFromMask(this->floatingImagePointer, combinedMask);
            reg_tools_removeNanFromMask(this->warpedReferenceImagePointer, combinedMask);

            if(this->mind_type==MIND_TYPE){
               GetMINDImageDesciptor(this->floatingImagePointer,
                                     this->floatingImageDescriptor,
                                     combinedMask,
                                     this->descriptorOffset,
                                     t);
               GetMINDImageDesciptor(this->warpedReferenceImagePointer,
                                     this->warpedReferenceImageDescriptor,
                                     combinedMask,
                                     this->descriptorOffset,
                                     t);
            }
            else if(this->mind_type==MINDSSC_TYPE){
               GetMINDSSCImageDesciptor(this->floatingImagePointer,
                                        this->floatingImageDescriptor,
                                        combinedMask,
                                        this->descriptorOffset,
                                        t);
               GetMINDSSCImageDesciptor(this->warpedReferenceImagePointer,
                                        this->warpedReferenceImageDescriptor,
                                        combinedMask,
                                        this->descriptorOffset,
                                        t);
            }

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
               reg_print_fct_error("reg_mind::GetSimilarityMeasureValue");
               reg_print_msg_error("Warped pixel type unsupported");
               reg_exit();
            }
            free(combinedMask);
         }
      }
   }
   return MINDValue;// /(double) this->referenceImageDescriptor->nt;
}
/* *************************************************************** */
void reg_mind::GetVoxelBasedSimilarityMeasureGradient(int current_timepoint)
{
   // Check if the specified time point exists and is active
   reg_measure::GetVoxelBasedSimilarityMeasureGradient(current_timepoint);
   if(this->activeTimePoint[current_timepoint]==false)
      return;

   // Create a combined mask to ignore masked and undefined values
   size_t voxelNumber = (size_t)this->referenceImagePointer->nx *
         this->referenceImagePointer->ny *
         this->referenceImagePointer->nz;
   int *combinedMask = (int *)malloc(voxelNumber*sizeof(int));
   memcpy(combinedMask, this->referenceMaskPointer, voxelNumber*sizeof(int));
   reg_tools_removeNanFromMask(this->referenceImagePointer, combinedMask);
   reg_tools_removeNanFromMask(this->warpedFloatingImagePointer, combinedMask);

   if(this->mind_type==MIND_TYPE){
      // Compute the reference image descriptors
      GetMINDImageDesciptor(this->referenceImagePointer,
                            this->referenceImageDescriptor,
                            combinedMask,
                            this->descriptorOffset,
                            current_timepoint);
      // Compute the warped floating image descriptors
      GetMINDImageDesciptor(this->warpedFloatingImagePointer,
                            this->warpedFloatingImageDescriptor,
                            combinedMask,
                            this->descriptorOffset,
                            current_timepoint);
   }
   else if(this->mind_type==MINDSSC_TYPE){
      // Compute the reference image descriptors
      GetMINDSSCImageDesciptor(this->referenceImagePointer,
                               this->referenceImageDescriptor,
                               combinedMask,
                               this->descriptorOffset,
                               current_timepoint);
      // Compute the warped floating image descriptors
      GetMINDSSCImageDesciptor(this->warpedFloatingImagePointer,
                               this->warpedFloatingImageDescriptor,
                               combinedMask,
                               this->descriptorOffset,
                               current_timepoint);
   }


   for(int desc_index=0; desc_index<this->discriptor_number; ++desc_index){
      // Compute the warped image descriptors gradient
      spatialGradient<float>(this->warpedFloatingImageDescriptor,
                             this->warpedFloatingGradientImagePointer,
                             combinedMask,
                             desc_index);

      // Compute the gradient of the ssd for the forward transformation
      switch(referenceImageDescriptor->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         reg_getVoxelBasedSSDGradient<float>
               (this->referenceImageDescriptor,
                this->warpedFloatingImageDescriptor,
                this->warpedFloatingGradientImagePointer,
                this->forwardVoxelBasedGradientImagePointer,
                NULL, // no Jacobian required here,
                combinedMask,
                desc_index
                );
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_getVoxelBasedSSDGradient<double>
               (this->referenceImageDescriptor,
                this->warpedFloatingImageDescriptor,
                this->warpedFloatingGradientImagePointer,
                this->forwardVoxelBasedGradientImagePointer,
                NULL, // no Jacobian required here,
                combinedMask,
                desc_index
                );
         break;
      default:
         reg_print_fct_error("reg_mind::GetVoxelBasedSimilarityMeasureGradient");
         reg_print_msg_error("Unsupported datatype");
         reg_exit();
      }
   }
   free(combinedMask);

   // Compute the gradient of the ssd for the backward transformation
   if(this->isSymmetric)
   {
      voxelNumber = (size_t)floatingImagePointer->nx *
            floatingImagePointer->ny * floatingImagePointer->nz;
      combinedMask = (int *)malloc(voxelNumber*sizeof(int));
      memcpy(combinedMask, this->floatingMaskPointer, voxelNumber*sizeof(int));
      reg_tools_removeNanFromMask(this->floatingImagePointer, combinedMask);
      reg_tools_removeNanFromMask(this->warpedReferenceImagePointer, combinedMask);

      if(this->mind_type==MIND_TYPE){
         GetMINDImageDesciptor(this->floatingImagePointer,
                               this->floatingImageDescriptor,
                               combinedMask,
                               this->descriptorOffset,
                               current_timepoint);
         GetMINDImageDesciptor(this->warpedReferenceImagePointer,
                               this->warpedReferenceImageDescriptor,
                               combinedMask,
                               this->descriptorOffset,
                               current_timepoint);
      }
      else if(this->mind_type==MINDSSC_TYPE){
         GetMINDSSCImageDesciptor(this->floatingImagePointer,
                                  this->floatingImageDescriptor,
                                  combinedMask,
                                  this->descriptorOffset,
                                  current_timepoint);
         GetMINDSSCImageDesciptor(this->warpedReferenceImagePointer,
                                  this->warpedReferenceImageDescriptor,
                                  combinedMask,
                                  this->descriptorOffset,
                                  current_timepoint);
      }

      for(int desc_index=0; desc_index<this->discriptor_number; ++desc_index){
         spatialGradient<float>(this->warpedReferenceImageDescriptor,
                                this->warpedReferenceGradientImagePointer,
                                combinedMask,
                                desc_index);

         // Compute the gradient of the nmi for the backward transformation
         switch(floatingImagePointer->datatype)
         {
         case NIFTI_TYPE_FLOAT32:
            reg_getVoxelBasedSSDGradient<float>
                  (this->floatingImageDescriptor,
                   this->warpedReferenceImageDescriptor,
                   this->warpedReferenceGradientImagePointer,
                   this->backwardVoxelBasedGradientImagePointer,
                   NULL, // no Jacobian required here,
                   combinedMask,
                   desc_index
                   );
            break;
         case NIFTI_TYPE_FLOAT64:
            reg_getVoxelBasedSSDGradient<double>
                  (this->floatingImageDescriptor,
                   this->warpedReferenceImageDescriptor,
                   this->warpedReferenceGradientImagePointer,
                   this->backwardVoxelBasedGradientImagePointer,
                   NULL, // no Jacobian required here,
                   combinedMask,
                   desc_index
                   );
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
/* *************************************************************** */
reg_mindssc::reg_mindssc()
   : reg_mind()
{
   this->mind_type=MINDSSC_TYPE;
#ifndef NDEBUG
   reg_print_msg_debug("reg_mindssc constructor called");
#endif
}
/* *************************************************************** */
reg_mindssc::~reg_mindssc()
{
#ifndef NDEBUG
   reg_print_msg_debug("reg_mindssc desctructor called");
#endif
}
/* *************************************************************** */
