/**
 * @file _reg_tools.cpp
 * @author Marc Modat
 * @date 25/03/2009
 * @brief Set of useful functions
 *
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_TOOLS_CPP
#define _REG_TOOLS_CPP

#include <cmath>
#include "_reg_tools.h"

/* *************************************************************** */
/* *************************************************************** */
void reg_checkAndCorrectDimension(nifti_image *image)
{
   // Ensure that no dimension is set to zero
   if(image->nx<1 || image->dim[1]<1) image->dim[1]=image->nx=1;
   if(image->ny<1 || image->dim[2]<1) image->dim[2]=image->ny=1;
   if(image->nz<1 || image->dim[3]<1) image->dim[3]=image->nz=1;
   if(image->nt<1 || image->dim[4]<1) image->dim[4]=image->nt=1;
   if(image->nu<1 || image->dim[5]<1) image->dim[5]=image->nu=1;
   if(image->nv<1 || image->dim[6]<1) image->dim[6]=image->nv=1;
   if(image->nw<1 || image->dim[7]<1) image->dim[7]=image->nw=1;
   //Correcting the dim of the images
   for(int i=1;i<8;++i) {
       if(image->dim[i]>1) {
            image->dim[0]=image->ndim=i;
       }
   }
   // Set the slope to 1 if undefined
   if(image->scl_slope==0) image->scl_slope=1.f;
   // Ensure that no spacing is set to zero
   if(image->ny==1 && (image->dy==0 || image->pixdim[2]==0))
      image->dy=image->pixdim[2]=1;
   if(image->nz==1 && (image->dz==0 || image->pixdim[3]==0))
      image->dz=image->pixdim[3]=1;
   // Create the qform matrix if required
   if(image->qform_code==0 && image->sform_code==0)
   {
      image->qto_xyz=nifti_quatern_to_mat44(image->quatern_b,
                                            image->quatern_c,
                                            image->quatern_d,
                                            image->qoffset_x,
                                            image->qoffset_y,
                                            image->qoffset_z,
                                            image->dx,
                                            image->dy,
                                            image->dz,
                                            image->qfac);
      image->qto_ijk=nifti_mat44_inverse(image->qto_xyz);
   }
   // Set the voxel spacing to millimeters
   if(image->xyz_units==NIFTI_UNITS_MICRON)
   {
      for(int d=1; d<=image->ndim; ++d)
         image->pixdim[d] /= 1000.f;
      image->xyz_units=NIFTI_UNITS_MM;
   }
   if(image->xyz_units==NIFTI_UNITS_METER)
   {
      for(int d=1; d<=image->ndim; ++d)
         image->pixdim[d] *= 1000.f;
      image->xyz_units=NIFTI_UNITS_MM;
   }
   image->dx=image->pixdim[1];
   image->dy=image->pixdim[2];
   image->dz=image->pixdim[3];
   image->dt=image->pixdim[4];
   image->du=image->pixdim[5];
   image->dv=image->pixdim[6];
   image->dw=image->pixdim[7];
}
/* *************************************************************** */
/* *************************************************************** */
bool reg_isAnImageFileName(char *name)
{
   std::string n(name);
   if(n.find( ".nii") != std::string::npos)
      return true;
   if(n.find( ".nii.gz") != std::string::npos)
      return true;
   if(n.find( ".hdr") != std::string::npos)
      return true;
   if(n.find( ".img") != std::string::npos)
      return true;
   if(n.find( ".img.gz") != std::string::npos)
      return true;
   if(n.find( ".nrrd") != std::string::npos)
      return true;
   if(n.find( ".png") != std::string::npos)
      return true;
   return false;
}
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
void reg_intensityRescale_core(nifti_image *image,
                               int timePoint,
                               float newMin,
                               float newMax
                               )
{
   DTYPE *imagePtr = static_cast<DTYPE *>(image->data);
   unsigned int voxelNumber = image->nx*image->ny*image->nz;

   // The rescasling is done for each volume independtly
   DTYPE *volumePtr = &imagePtr[timePoint*voxelNumber];
   DTYPE currentMin=0;
   DTYPE currentMax=0;
   switch(image->datatype)
   {
   case NIFTI_TYPE_UINT8:
      currentMin=(DTYPE)std::numeric_limits<unsigned char>::max();
      currentMax=0;
      break;
   case NIFTI_TYPE_INT8:
      currentMin=(DTYPE)std::numeric_limits<char>::max();
      currentMax=-(DTYPE)std::numeric_limits<char>::max();
      break;
   case NIFTI_TYPE_UINT16:
      currentMin=(DTYPE)std::numeric_limits<unsigned short>::max();
      currentMax=0;
      break;
   case NIFTI_TYPE_INT16:
      currentMin=(DTYPE)std::numeric_limits<short>::max();
      currentMax=-(DTYPE)std::numeric_limits<short>::max();
      break;
   case NIFTI_TYPE_UINT32:
      currentMin=(DTYPE)std::numeric_limits<unsigned int>::max();
      currentMax=0;
      break;
   case NIFTI_TYPE_INT32:
      currentMin=(DTYPE)std::numeric_limits<int>::max();
      currentMax=-(DTYPE)std::numeric_limits<int>::max();
      break;
   case NIFTI_TYPE_FLOAT32:
      currentMin=(DTYPE)std::numeric_limits<float>::max();
      currentMax=-(DTYPE)std::numeric_limits<float>::max();
      break;
   case NIFTI_TYPE_FLOAT64:
      currentMin=(DTYPE)std::numeric_limits<double>::max();
      currentMax=-(DTYPE)std::numeric_limits<double>::max();
      break;
   }

   // Extract the minimal and maximal values from the current volume
   if(image->scl_slope==0) image->scl_slope=1.0f;
   for(unsigned int index=0; index<voxelNumber; index++)
   {
      DTYPE value = (DTYPE)(*volumePtr++ * image->scl_slope + image->scl_inter);
      if(value==value)
      {
         currentMin=(currentMin<value)?currentMin:value;
         currentMax=(currentMax>value)?currentMax:value;
      }
   }

   // Compute constant values to rescale image intensities
   double currentDiff = (double)(currentMax-currentMin);
   double newDiff = (double)(newMax-newMin);

   // Set the image header information for appropriate display
   image->cal_min=newMin;
   image->cal_max=newMax;

   // Reset the volume pointer to the start of the current volume
   volumePtr = &imagePtr[timePoint*voxelNumber];

   // Iterates over all voxels in the current volume
   for(unsigned int index=0; index<voxelNumber; index++)
   {
      double value = (double)*volumePtr * image->scl_slope + image->scl_inter;
      // Check if the value is defined
      if(value==value)
      {
         // Normalise the value between 0 and 1
         value = (value-(double)currentMin)/currentDiff;
         // Rescale the value using the specified range
         value = value * newDiff + newMin;
      }
      *volumePtr++=(DTYPE)value;
   }
   image->scl_slope=1.f;
   image->scl_inter=0.f;
}
/* *************************************************************** */
void reg_intensityRescale(nifti_image *image,
                          int timepoint,
                          float newMin,
                          float newMax
                          )
{
   switch(image->datatype)
   {
   case NIFTI_TYPE_UINT8:
      reg_intensityRescale_core<unsigned char>(image, timepoint, newMin, newMax);
      break;
   case NIFTI_TYPE_INT8:
      reg_intensityRescale_core<char>(image, timepoint, newMin, newMax);
      break;
   case NIFTI_TYPE_UINT16:
      reg_intensityRescale_core<unsigned short>(image, timepoint, newMin, newMax);
      break;
   case NIFTI_TYPE_INT16:
      reg_intensityRescale_core<short>(image, timepoint, newMin, newMax);
      break;
   case NIFTI_TYPE_UINT32:
      reg_intensityRescale_core<unsigned int>(image, timepoint, newMin, newMax);
      break;
   case NIFTI_TYPE_INT32:
      reg_intensityRescale_core<int>(image, timepoint, newMin, newMax);
      break;
   case NIFTI_TYPE_FLOAT32:
      reg_intensityRescale_core<float>(image, timepoint, newMin, newMax);
      break;
   case NIFTI_TYPE_FLOAT64:
      reg_intensityRescale_core<double>(image, timepoint, newMin, newMax);
      break;
   default:
      reg_print_fct_error("reg_intensityRescale");
      reg_print_msg_error("The image data type is not supported");
      reg_exit();
   }
}
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
void reg_tools_removeSCLInfo_core(nifti_image *image)
{
   if(image->scl_slope==1.f && image->scl_inter==0.f)
      return;
   DTYPE *imgPtr = static_cast<DTYPE *>(image->data);
   for(size_t i=0;i<image->nvox; ++i){
      *imgPtr=*imgPtr*(DTYPE)image->scl_slope+(DTYPE)image->scl_inter;
      imgPtr++;
   }
   image->scl_slope=1.f;
   image->scl_inter=0.f;
}
/* *************************************************************** */
void reg_tools_removeSCLInfo(nifti_image *image)
{
   switch(image->datatype)
   {
   case NIFTI_TYPE_UINT8:
      reg_tools_removeSCLInfo_core<unsigned char>(image);
      break;
   case NIFTI_TYPE_INT8:
      reg_tools_removeSCLInfo_core<char>(image);
      break;
   case NIFTI_TYPE_UINT16:
      reg_tools_removeSCLInfo_core<unsigned short>(image);
      break;
   case NIFTI_TYPE_INT16:
      reg_tools_removeSCLInfo_core<short>(image);
      break;
   case NIFTI_TYPE_UINT32:
      reg_tools_removeSCLInfo_core<unsigned int>(image);
      break;
   case NIFTI_TYPE_INT32:
      reg_tools_removeSCLInfo_core<int>(image);
      break;
   case NIFTI_TYPE_FLOAT32:
      reg_tools_removeSCLInfo_core<float>(image);
      break;
   case NIFTI_TYPE_FLOAT64:
      reg_tools_removeSCLInfo_core<double>(image);
      break;
   default:
      reg_print_fct_error("reg_tools_removeSCLInfo");
      reg_print_msg_error("The image data type is not supported");
      reg_exit();
   }
   return;
}
/* *************************************************************** */
/* *************************************************************** */
void reg_getRealImageSpacing(nifti_image *image,
                             float *spacingValues)
{
   float indexVoxel1[3]= {0,0,0};
   float indexVoxel2[3], realVoxel1[3], realVoxel2[3];
   reg_mat44_mul(&(image->sto_xyz), indexVoxel1, realVoxel1);

   indexVoxel2[1]=indexVoxel2[2]=0;
   indexVoxel2[0]=1;
   reg_mat44_mul(&(image->sto_xyz), indexVoxel2, realVoxel2);
   spacingValues[0]=sqrtf(reg_pow2(realVoxel1[0]-realVoxel2[0])+reg_pow2(realVoxel1[1]-realVoxel2[1])+reg_pow2(realVoxel1[2]-realVoxel2[2]));

   indexVoxel2[0]=indexVoxel2[2]=0;
   indexVoxel2[1]=1;
   reg_mat44_mul(&(image->sto_xyz), indexVoxel2, realVoxel2);
   spacingValues[1]=sqrtf(reg_pow2(realVoxel1[0]-realVoxel2[0])+reg_pow2(realVoxel1[1]-realVoxel2[1])+reg_pow2(realVoxel1[2]-realVoxel2[2]));

   if(image->nz>1)
   {
      indexVoxel2[0]=indexVoxel2[1]=0;
      indexVoxel2[2]=1;
      reg_mat44_mul(&(image->sto_xyz), indexVoxel2, realVoxel2);
      spacingValues[2]=sqrtf(reg_pow2(realVoxel1[0]-realVoxel2[0])+reg_pow2(realVoxel1[1]-realVoxel2[1])+reg_pow2(realVoxel1[2]-realVoxel2[2]));
   }
}
/* *************************************************************** */
/* *************************************************************** */
//this function will threshold an image to the values provided,
//set the scl_slope and sct_inter of the image to 1 and 0 (SSD uses actual image data values),
//and sets cal_min and cal_max to have the min/max image data values
template<class T,class DTYPE>
void reg_thresholdImage2(nifti_image *image,
                         T lowThr,
                         T upThr
                         )
{
   DTYPE *imagePtr = static_cast<DTYPE *>(image->data);
   T currentMin=std::numeric_limits<T>::max();
   T currentMax=-std::numeric_limits<T>::max();

   if(image->scl_slope==0)image->scl_slope=1.0;

   for(unsigned int index=0; index<image->nvox; index++)
   {
      T value = (T)(*imagePtr * image->scl_slope + image->scl_inter);
      if(value==value)
      {
         if(value<lowThr)
         {
            value = lowThr;
         }
         else if(value>upThr)
         {
            value = upThr;
         }
         currentMin=(currentMin<value)?currentMin:value;
         currentMax=(currentMax>value)?currentMax:value;
      }
      *imagePtr++=(DTYPE)value;
   }

   image->cal_min = currentMin;
   image->cal_max = currentMax;
}
/* *************************************************************** */
template<class T>
void reg_thresholdImage(nifti_image *image,
                        T lowThr,
                        T upThr
                        )
{
   switch(image->datatype)
   {
   case NIFTI_TYPE_UINT8:
      reg_thresholdImage2<T,unsigned char>(image, lowThr, upThr);
      break;
   case NIFTI_TYPE_INT8:
      reg_thresholdImage2<T,char>(image, lowThr, upThr);
      break;
   case NIFTI_TYPE_UINT16:
      reg_thresholdImage2<T,unsigned short>(image, lowThr, upThr);
      break;
   case NIFTI_TYPE_INT16:
      reg_thresholdImage2<T,short>(image, lowThr, upThr);
      break;
   case NIFTI_TYPE_UINT32:
      reg_thresholdImage2<T,unsigned int>(image, lowThr, upThr);
      break;
   case NIFTI_TYPE_INT32:
      reg_thresholdImage2<T,int>(image, lowThr, upThr);
      break;
   case NIFTI_TYPE_FLOAT32:
      reg_thresholdImage2<T,float>(image, lowThr, upThr);
      break;
   case NIFTI_TYPE_FLOAT64:
      reg_thresholdImage2<T,double>(image, lowThr, upThr);
      break;
   default:
      reg_print_fct_error("reg_thresholdImage");
      reg_print_msg_error("The image data type is not supported");
      reg_exit();
   }
}
template void reg_thresholdImage<float>(nifti_image *, float, float);
template void reg_thresholdImage<double>(nifti_image *, double, double);
/* *************************************************************** */
/* *************************************************************** */
template <class PrecisionTYPE, class DTYPE>
PrecisionTYPE reg_getMaximalLength2D(nifti_image *image)
{
   DTYPE *dataPtrX = static_cast<DTYPE *>(image->data);
   DTYPE *dataPtrY = &dataPtrX[image->nx*image->ny*image->nz];

   PrecisionTYPE max=0.0;

   for(int i=0; i<image->nx*image->ny*image->nz; i++)
   {
      PrecisionTYPE valX = (PrecisionTYPE)(*dataPtrX++);
      PrecisionTYPE valY = (PrecisionTYPE)(*dataPtrY++);
      PrecisionTYPE length = (PrecisionTYPE)(sqrt(valX*valX + valY*valY));
      max = (length>max)?length:max;
   }
   return max;
}
/* *************************************************************** */
template <class PrecisionTYPE, class DTYPE>
PrecisionTYPE reg_getMaximalLength3D(nifti_image *image)
{
   DTYPE *dataPtrX = static_cast<DTYPE *>(image->data);
   DTYPE *dataPtrY = &dataPtrX[image->nx*image->ny*image->nz];
   DTYPE *dataPtrZ = &dataPtrY[image->nx*image->ny*image->nz];

   PrecisionTYPE max=0.0;

   for(int i=0; i<image->nx*image->ny*image->nz; i++)
   {
      PrecisionTYPE valX = (PrecisionTYPE)(*dataPtrX++);
      PrecisionTYPE valY = (PrecisionTYPE)(*dataPtrY++);
      PrecisionTYPE valZ = (PrecisionTYPE)(*dataPtrZ++);
      PrecisionTYPE length = (PrecisionTYPE)(sqrt(valX*valX + valY*valY + valZ*valZ));
      max = (length>max)?length:max;
   }
   return max;
}
/* *************************************************************** */
template <class PrecisionTYPE>
PrecisionTYPE reg_getMaximalLength(nifti_image *image)
{
   if(image->nz==1)
   {
      switch(image->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         return reg_getMaximalLength2D<PrecisionTYPE,float>(image);
         break;
      case NIFTI_TYPE_FLOAT64:
         return reg_getMaximalLength2D<PrecisionTYPE,double>(image);
         break;
      }
   }
   else
   {
      switch(image->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         return reg_getMaximalLength3D<PrecisionTYPE,float>(image);
         break;
      case NIFTI_TYPE_FLOAT64:
         return reg_getMaximalLength3D<PrecisionTYPE,double>(image);
         break;
      }
   }
   return EXIT_SUCCESS;
}
/* *************************************************************** */
template float reg_getMaximalLength<float>(nifti_image *);
template double reg_getMaximalLength<double>(nifti_image *);
/* *************************************************************** */
/* *************************************************************** */
template <class NewTYPE, class DTYPE>
void reg_tools_changeDatatype1(nifti_image *image,int type)
{
   // the initial array is saved and freeed
   DTYPE *initialValue = (DTYPE *)malloc(image->nvox*sizeof(DTYPE));
   memcpy(initialValue, image->data, image->nvox*sizeof(DTYPE));

   // the new array is allocated and then filled
   if(type>-1){
      image->datatype=type;
   }
   else{
      if(sizeof(NewTYPE)==sizeof(unsigned char)) {
          image->datatype = NIFTI_TYPE_UINT8;
#ifndef NDEBUG
    reg_print_msg_debug("new datatype is NIFTI_TYPE_UINT8");
#endif
      }
      else if(sizeof(NewTYPE)==sizeof(float)) {
          image->datatype = NIFTI_TYPE_FLOAT32;
#ifndef NDEBUG
    reg_print_msg_debug("new datatype is NIFTI_TYPE_FLOAT32");
#endif
      }
      else if(sizeof(NewTYPE)==sizeof(double)) {
          image->datatype = NIFTI_TYPE_FLOAT64;
#ifndef NDEBUG
    reg_print_msg_debug("new datatype is NIFTI_TYPE_FLOAT64");
#endif
      }
      else {
         reg_print_fct_error("reg_tools_changeDatatype1");
         reg_print_msg_error("Only change to unsigned char, float or double are supported");
         reg_exit();
      }
   }
   free(image->data);
   image->nbyper = sizeof(NewTYPE);
   image->data = (void *)calloc(image->nvox,sizeof(NewTYPE));
   NewTYPE *dataPtr = static_cast<NewTYPE *>(image->data);
   for (size_t i = 0; i < image->nvox; i++) {
       dataPtr[i] = (NewTYPE)(initialValue[i]);
   }

   free(initialValue);
   return;
}
/* *************************************************************** */
template <class NewTYPE>
void reg_tools_changeDatatype(nifti_image *image, int type)
{
   switch(image->datatype)
   {
   case NIFTI_TYPE_UINT8:
      reg_tools_changeDatatype1<NewTYPE,unsigned char>(image,type);
      break;
   case NIFTI_TYPE_INT8:
      reg_tools_changeDatatype1<NewTYPE,char>(image,type);
      break;
   case NIFTI_TYPE_UINT16:
      reg_tools_changeDatatype1<NewTYPE,unsigned short>(image,type);
      break;
   case NIFTI_TYPE_INT16:
      reg_tools_changeDatatype1<NewTYPE,short>(image,type);
      break;
   case NIFTI_TYPE_UINT32:
      reg_tools_changeDatatype1<NewTYPE,unsigned int>(image,type);
      break;
   case NIFTI_TYPE_INT32:
      reg_tools_changeDatatype1<NewTYPE,int>(image,type);
      break;
   case NIFTI_TYPE_FLOAT32:
      reg_tools_changeDatatype1<NewTYPE,float>(image,type);
      break;
   case NIFTI_TYPE_FLOAT64:
      reg_tools_changeDatatype1<NewTYPE,double>(image,type);
      break;
   default:
      reg_print_fct_error("reg_tools_changeDatatype");
      reg_print_msg_error("Unsupported datatype");
      reg_exit();
   }
}
/* *************************************************************** */
template void reg_tools_changeDatatype<unsigned char>(nifti_image *, int);
template void reg_tools_changeDatatype<unsigned short>(nifti_image *, int);
template void reg_tools_changeDatatype<unsigned int>(nifti_image *, int);
template void reg_tools_changeDatatype<char>(nifti_image *, int);
template void reg_tools_changeDatatype<short>(nifti_image *, int);
template void reg_tools_changeDatatype<int>(nifti_image *, int);
template void reg_tools_changeDatatype<float>(nifti_image *, int);
template void reg_tools_changeDatatype<double>(nifti_image *, int);
/* *************************************************************** */
/* *************************************************************** */
template <class TYPE1>
void reg_tools_operationImageToImage(nifti_image *img1,
                                     nifti_image *img2,
                                     nifti_image *res,
                                     int type)
{
   TYPE1 *img1Ptr = static_cast<TYPE1 *>(img1->data);
   TYPE1 *resPtr = static_cast<TYPE1 *>(res->data);
   TYPE1 *img2Ptr = static_cast<TYPE1 *>(img2->data);


   if(img1->scl_slope==0) {
      img1->scl_slope=1.f;
   }
   if(img2->scl_slope==0) {
       img2->scl_slope=1.f;
   }

   res->scl_slope=img1->scl_slope;
   res->scl_inter=img1->scl_inter;


#ifdef _WIN32
   long i;
   long voxelNumber=(long)res->nvox;
#else
   size_t i;
   size_t voxelNumber=res->nvox;
#endif

   switch(type)
   {
   case 0:
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   private(i) \
   shared(voxelNumber,resPtr,img1Ptr,img2Ptr,img1,img2)
#endif // _OPENMP
      for(i=0; i<voxelNumber; i++)
         resPtr[i] = (TYPE1)((((double)img1Ptr[i] * (double)img1->scl_slope + (double)img1->scl_inter) +
                              ((double)img2Ptr[i] * (double)img2->scl_slope + (double)img2->scl_inter) -
                              (double)img1->scl_inter)/(double)img1->scl_slope);
      break;
   case 1:
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   private(i) \
   shared(voxelNumber,resPtr,img1Ptr,img2Ptr,img1,img2)
#endif // _OPENMP
       for (i = 0; i < voxelNumber; i++) {
               resPtr[i] = (TYPE1)((((double)img1Ptr[i] * (double)img1->scl_slope + (double)img1->scl_inter) -
                                 ((double)img2Ptr[i] * (double)img2->scl_slope + (double)img2->scl_inter) -
                                   (double)img1->scl_inter) / (double)img1->scl_slope);
       }
      break;
   case 2:
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   private(i) \
   shared(voxelNumber,resPtr,img1Ptr,img2Ptr,img1,img2)
#endif // _OPENMP
       for (i = 0; i < voxelNumber; i++) {
           resPtr[i] = (TYPE1)((((double)img1Ptr[i] * (double)img1->scl_slope + (double)img1->scl_inter) *
               ((double)img2Ptr[i] * (double)img2->scl_slope + (double)img2->scl_inter) -
               (double)img1->scl_inter) / (double)img1->scl_slope);
       }
      break;
   case 3:
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   private(i) \
   shared(voxelNumber,resPtr,img1Ptr,img2Ptr,img1,img2)
#endif // _OPENMP
      for(i=0; i<voxelNumber; i++)
         resPtr[i] = (TYPE1)((((double)img1Ptr[i] * (double)img1->scl_slope + (double)img1->scl_inter) /
                              ((double)img2Ptr[i] * (double)img2->scl_slope + (double)img2->scl_inter) -
                              (double)img1->scl_inter)/(double)img1->scl_slope);
      break;
   }
}
/* *************************************************************** */
void reg_tools_addImageToImage(nifti_image *img1,
                               nifti_image *img2,
                               nifti_image *res)
{
   if(img1->datatype != res->datatype || img2->datatype != res->datatype)
   {
      reg_print_fct_error("reg_tools_addImageToImage");
      reg_print_msg_error("Input images are expected to be of the same type");
      reg_exit();
   }
   if(img1->nvox != res->nvox || img2->nvox != res->nvox)
   {
      reg_print_fct_error("reg_tools_addImageToImage");
      reg_print_msg_error("Input images are expected to have the same size");
      reg_exit();
   }
   switch(img1->datatype)
   {
   case NIFTI_TYPE_UINT8:
      reg_tools_operationImageToImage<unsigned char>(img1, img2, res, 0);
      break;
   case NIFTI_TYPE_INT8:
      reg_tools_operationImageToImage<char>(img1, img2, res, 0);
      break;
   case NIFTI_TYPE_UINT16:
      reg_tools_operationImageToImage<unsigned short>(img1, img2, res, 0);
      break;
   case NIFTI_TYPE_INT16:
      reg_tools_operationImageToImage<short>(img1, img2, res, 0);
      break;
   case NIFTI_TYPE_UINT32:
      reg_tools_operationImageToImage<unsigned int>(img1, img2, res, 0);
      break;
   case NIFTI_TYPE_INT32:
      reg_tools_operationImageToImage<int>(img1, img2, res, 0);
      break;
   case NIFTI_TYPE_FLOAT32:
      reg_tools_operationImageToImage<float>(img1, img2, res, 0);
      break;
   case NIFTI_TYPE_FLOAT64:
      reg_tools_operationImageToImage<double>(img1, img2, res, 0);
      break;
   default:
      reg_print_fct_error("reg_tools_addImageToImage");
      reg_print_msg_error("Unsupported datatype");
      reg_exit();
   }
}
/* *************************************************************** */
void reg_tools_substractImageToImage(nifti_image *img1,
                                     nifti_image *img2,
                                     nifti_image *res)
{
   if(img1->datatype != res->datatype || img2->datatype != res->datatype)
   {
      reg_print_fct_error("reg_tools_substractImageToImage");
      reg_print_msg_error("Input images are expected to be of the same type");
      reg_exit();
   }
   if(img1->nvox != res->nvox || img2->nvox != res->nvox)
   {
      reg_print_fct_error("reg_tools_substractImageToImage");
      reg_print_msg_error("Input images are expected to have the same size");
      reg_exit();
   }
   switch(img1->datatype)
   {
   case NIFTI_TYPE_UINT8:
      reg_tools_operationImageToImage<unsigned char>(img1, img2, res, 1);
      break;
   case NIFTI_TYPE_INT8:
      reg_tools_operationImageToImage<char>(img1, img2, res, 1);
      break;
   case NIFTI_TYPE_UINT16:
      reg_tools_operationImageToImage<unsigned short>(img1, img2, res, 1);
      break;
   case NIFTI_TYPE_INT16:
      reg_tools_operationImageToImage<short>(img1, img2, res, 1);
      break;
   case NIFTI_TYPE_UINT32:
      reg_tools_operationImageToImage<unsigned int>(img1, img2, res, 1);
      break;
   case NIFTI_TYPE_INT32:
      reg_tools_operationImageToImage<int>(img1, img2, res, 1);
      break;
   case NIFTI_TYPE_FLOAT32:
      reg_tools_operationImageToImage<float>(img1, img2, res, 1);
      break;
   case NIFTI_TYPE_FLOAT64:
      reg_tools_operationImageToImage<double>(img1, img2, res, 1);
      break;
   default:
      reg_print_fct_error("reg_tools_substractImageToImage");
      reg_print_msg_error("Unsupported datatype");
      reg_exit();
   }
}
/* *************************************************************** */
void reg_tools_multiplyImageToImage(nifti_image *img1,
                                    nifti_image *img2,
                                    nifti_image *res)
{
   if(img1->datatype != res->datatype || img2->datatype != res->datatype)
   {
      reg_print_fct_error("reg_tools_multiplyImageToImage");
      reg_print_msg_error("Input images are expected to have the same size");
      reg_exit();
   }
   if(img1->nvox != res->nvox || img2->nvox != res->nvox)
   {
      reg_print_fct_error("reg_tools_multiplyImageToImage");
      reg_print_msg_error("Input images are expected to have the same size");
      reg_exit();
   }
   switch(img1->datatype)
   {
   case NIFTI_TYPE_UINT8:
      reg_tools_operationImageToImage<unsigned char>(img1, img2, res, 2);
      break;
   case NIFTI_TYPE_INT8:
      reg_tools_operationImageToImage<char>(img1, img2, res, 2);
      break;
   case NIFTI_TYPE_UINT16:
      reg_tools_operationImageToImage<unsigned short>(img1, img2, res, 2);
      break;
   case NIFTI_TYPE_INT16:
      reg_tools_operationImageToImage<short>(img1, img2, res, 2);
      break;
   case NIFTI_TYPE_UINT32:
      reg_tools_operationImageToImage<unsigned int>(img1, img2, res, 2);
      break;
   case NIFTI_TYPE_INT32:
      reg_tools_operationImageToImage<int>(img1, img2, res, 2);
      break;
   case NIFTI_TYPE_FLOAT32:
      reg_tools_operationImageToImage<float>(img1, img2, res, 2);
      break;
   case NIFTI_TYPE_FLOAT64:
      reg_tools_operationImageToImage<double>(img1, img2, res, 2);
      break;
   default:
      reg_print_fct_error("reg_tools_multiplyImageToImage");
      reg_print_msg_error("Unsupported datatype");
      reg_exit();
   }
}
/* *************************************************************** */
void reg_tools_divideImageToImage(nifti_image *img1,
                                  nifti_image *img2,
                                  nifti_image *res)
{
   if(img1->datatype != res->datatype || img2->datatype != res->datatype)
   {
      reg_print_fct_error("reg_tools_divideImageToImage");
      reg_print_msg_error("Input images are expected to have the same size");
      reg_exit();
   }
   if(img1->nvox != res->nvox || img2->nvox != res->nvox)
   {
      reg_print_fct_error("reg_tools_divideImageToImage");
      reg_print_msg_error("Input images are expected to have the same size");
      reg_exit();
   }
   switch(img1->datatype)
   {
   case NIFTI_TYPE_UINT8:
      reg_tools_operationImageToImage<unsigned char>(img1, img2, res, 3);
      break;
   case NIFTI_TYPE_INT8:
      reg_tools_operationImageToImage<char>(img1, img2, res, 3);
      break;
   case NIFTI_TYPE_UINT16:
      reg_tools_operationImageToImage<unsigned short>(img1, img2, res, 3);
      break;
   case NIFTI_TYPE_INT16:
      reg_tools_operationImageToImage<short>(img1, img2, res, 3);
      break;
   case NIFTI_TYPE_UINT32:
      reg_tools_operationImageToImage<unsigned int>(img1, img2, res, 3);
      break;
   case NIFTI_TYPE_INT32:
      reg_tools_operationImageToImage<int>(img1, img2, res, 3);
      break;
   case NIFTI_TYPE_FLOAT32:
      reg_tools_operationImageToImage<float>(img1, img2, res, 3);
      break;
   case NIFTI_TYPE_FLOAT64:
      reg_tools_operationImageToImage<double>(img1, img2, res, 3);
      break;
   default:
      reg_print_fct_error("reg_tools_divideImageToImage");
      reg_print_msg_error("Unsupported datatype");
      reg_exit();
   }
}
/* *************************************************************** */
/* *************************************************************** */
template <class TYPE1>
void reg_tools_operationValueToImage(nifti_image *img1,
                                     nifti_image *res,
                                     float val,
                                     int type)
{
   TYPE1 *img1Ptr = static_cast<TYPE1 *>(img1->data);
   TYPE1 *resPtr = static_cast<TYPE1 *>(res->data);

   if(img1->scl_slope==0)
   {
      img1->scl_slope=1.f;
   }

   res->scl_slope=img1->scl_slope;
   res->scl_inter=img1->scl_inter;

#ifdef _WIN32
   long i;
   long voxelNumber=(long)res->nvox;
#else
   size_t i;
   size_t voxelNumber=res->nvox;
#endif

   switch(type)
   {
   case 0:
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   private(i) \
   shared(voxelNumber,resPtr,img1Ptr,img1,val)
#endif // _OPENMP
      for(i=0; i<voxelNumber; i++)
         resPtr[i] = (TYPE1)(((((double)img1Ptr[i] * (double)img1->scl_slope + (double)img1->scl_inter) +
                               (double)val) - (double)img1->scl_inter)/(double)img1->scl_slope);
      break;
   case 1:
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   private(i) \
   shared(voxelNumber,resPtr,img1Ptr,img1,val)
#endif // _OPENMP
      for(i=0; i<voxelNumber; i++)
         resPtr[i] = (TYPE1)(((((double)img1Ptr[i] * (double)img1->scl_slope + (double)img1->scl_inter) -
                               (double)val) - (double)img1->scl_inter)/(double)img1->scl_slope);
      break;
   case 2:
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   private(i) \
   shared(voxelNumber,resPtr,img1Ptr,img1,val)
#endif // _OPENMP
      for(i=0; i<voxelNumber; i++)
         resPtr[i] = (TYPE1)(((((double)img1Ptr[i] * (double)img1->scl_slope + (double)img1->scl_inter) *
                               (double)val) - (double)img1->scl_inter)/(double)img1->scl_slope);
      break;
   case 3:
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   private(i) \
   shared(voxelNumber,resPtr,img1Ptr,img1,val)
#endif // _OPENMP
      for(i=0; i<voxelNumber; i++)
         resPtr[i] = (TYPE1)(((((double)img1Ptr[i] * (double)img1->scl_slope + (double)img1->scl_inter) /
                               (double)val) - (double)img1->scl_inter)/(double)img1->scl_slope);
      break;
   }
}
/* *************************************************************** */
void reg_tools_addValueToImage(nifti_image *img1,
                               nifti_image *res,
                               float val)
{
   if(img1->datatype != res->datatype)
   {
      reg_print_fct_error("reg_tools_addValueToImage");
      reg_print_msg_error("Input and output image do not have the same data type");
      reg_exit();
   }
   if(img1->nvox != res->nvox)
   {
      reg_print_fct_error("reg_tools_addValueToImage");
      reg_print_msg_error("Input images are expected to have the same size");
      reg_exit();
   }
   switch(img1->datatype)
   {
   case NIFTI_TYPE_UINT8:
      reg_tools_operationValueToImage<unsigned char>(img1, res, val, 0);
      break;
   case NIFTI_TYPE_INT8:
      reg_tools_operationValueToImage<char>(img1, res, val, 0);
      break;
   case NIFTI_TYPE_UINT16:
      reg_tools_operationValueToImage<unsigned short>(img1, res, val, 0);
      break;
   case NIFTI_TYPE_INT16:
      reg_tools_operationValueToImage<short>(img1, res, val, 0);
      break;
   case NIFTI_TYPE_UINT32:
      reg_tools_operationValueToImage<unsigned int>(img1, res, val, 0);
      break;
   case NIFTI_TYPE_INT32:
      reg_tools_operationValueToImage<int>(img1, res, val, 0);
      break;
   case NIFTI_TYPE_FLOAT32:
      reg_tools_operationValueToImage<float>(img1, res, val, 0);
      break;
   case NIFTI_TYPE_FLOAT64:
      reg_tools_operationValueToImage<double>(img1, res, val, 0);
      break;
   default:
      reg_print_fct_error("reg_tools_addValueToImage");
      reg_print_msg_error("Image data type is not supported");
      reg_exit();
   }
}
/* *************************************************************** */
void reg_tools_substractValueToImage(nifti_image *img1,
                                     nifti_image *res,
                                     float val)
{
   if(img1->datatype != res->datatype)
   {
      reg_print_fct_error("reg_tools_substractValueToImage");
      reg_print_msg_error("Input and output image do not have the same data type");
      reg_exit();
   }
   if(img1->nvox != res->nvox)
   {
      reg_print_fct_error("reg_tools_substractValueToImage");
      reg_print_msg_error("Input images are expected to have the same size");
      reg_exit();
   }
   switch(img1->datatype)
   {
   case NIFTI_TYPE_UINT8:
      reg_tools_operationValueToImage<unsigned char>(img1, res, val, 1);
      break;
   case NIFTI_TYPE_INT8:
      reg_tools_operationValueToImage<char>(img1, res, val, 1);
      break;
   case NIFTI_TYPE_UINT16:
      reg_tools_operationValueToImage<unsigned short>(img1, res, val, 1);
      break;
   case NIFTI_TYPE_INT16:
      reg_tools_operationValueToImage<short>(img1, res, val, 1);
      break;
   case NIFTI_TYPE_UINT32:
      reg_tools_operationValueToImage<unsigned int>(img1, res, val, 1);
      break;
   case NIFTI_TYPE_INT32:
      reg_tools_operationValueToImage<int>(img1, res, val, 1);
      break;
   case NIFTI_TYPE_FLOAT32:
      reg_tools_operationValueToImage<float>(img1, res, val, 1);
      break;
   case NIFTI_TYPE_FLOAT64:
      reg_tools_operationValueToImage<double>(img1, res, val, 1);
      break;
   default:
      reg_print_fct_error("reg_tools_substractValueToImage");
      reg_print_msg_error("Image data type is not supported");
      reg_exit();
   }
}
/* *************************************************************** */
void reg_tools_multiplyValueToImage(nifti_image *img1,
                                    nifti_image *res,
                                    float val)
{
   if(img1->datatype != res->datatype)
   {
      reg_print_fct_error("reg_tools_multiplyValueToImage");
      reg_print_msg_error("Input and output image do not have the same data type");
      reg_exit();
   }
   if(img1->nvox != res->nvox)
   {
      reg_print_fct_error("reg_tools_multiplyValueToImage");
      reg_print_msg_error("Input images are expected to have the same size");
      reg_exit();
   }
   switch(img1->datatype)
   {
   case NIFTI_TYPE_UINT8:
      reg_tools_operationValueToImage<unsigned char>(img1, res, val, 2);
      break;
   case NIFTI_TYPE_INT8:
      reg_tools_operationValueToImage<char>(img1, res, val, 2);
      break;
   case NIFTI_TYPE_UINT16:
      reg_tools_operationValueToImage<unsigned short>(img1, res, val, 2);
      break;
   case NIFTI_TYPE_INT16:
      reg_tools_operationValueToImage<short>(img1, res, val, 2);
      break;
   case NIFTI_TYPE_UINT32:
      reg_tools_operationValueToImage<unsigned int>(img1, res, val, 2);
      break;
   case NIFTI_TYPE_INT32:
      reg_tools_operationValueToImage<int>(img1, res, val, 2);
      break;
   case NIFTI_TYPE_FLOAT32:
      reg_tools_operationValueToImage<float>(img1, res, val, 2);
      break;
   case NIFTI_TYPE_FLOAT64:
      reg_tools_operationValueToImage<double>(img1, res, val, 2);
      break;
   default:
      reg_print_fct_error("reg_tools_multiplyValueToImage");
      reg_print_msg_error("Image data type is not supported");
      reg_exit();
   }
}
/* *************************************************************** */
void reg_tools_divideValueToImage(nifti_image *img1,
                                  nifti_image *res,
                                  float val)
{
   if(img1->datatype != res->datatype)
   {
      reg_print_fct_error("reg_tools_divideValueToImage");
      reg_print_msg_error("Input and output image do not have the same data type");
      reg_exit();
   }
   if(img1->nvox != res->nvox)
   {
      reg_print_fct_error("reg_tools_divideValueToImage");
      reg_print_msg_error("Input images are expected to have the same size");
      reg_exit();
   }
   switch(img1->datatype)
   {
   case NIFTI_TYPE_UINT8:
      reg_tools_operationValueToImage<unsigned char>(img1, res, val, 3);
      break;
   case NIFTI_TYPE_INT8:
      reg_tools_operationValueToImage<char>(img1, res, val, 3);
      break;
   case NIFTI_TYPE_UINT16:
      reg_tools_operationValueToImage<unsigned short>(img1, res, val, 3);
      break;
   case NIFTI_TYPE_INT16:
      reg_tools_operationValueToImage<short>(img1, res, val, 3);
      break;
   case NIFTI_TYPE_UINT32:
      reg_tools_operationValueToImage<unsigned int>(img1, res, val, 3);
      break;
   case NIFTI_TYPE_INT32:
      reg_tools_operationValueToImage<int>(img1, res, val, 3);
      break;
   case NIFTI_TYPE_FLOAT32:
      reg_tools_operationValueToImage<float>(img1, res, val, 3);
      break;
   case NIFTI_TYPE_FLOAT64:
      reg_tools_operationValueToImage<double>(img1, res, val, 3);
      break;
   default:
      reg_print_fct_error("reg_tools_divideValueToImage");
      reg_print_msg_error("Image data type is not supported");
      reg_exit();
   }
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_tools_kernelConvolution_core(nifti_image *image,
                                      float *sigma,
                                      int kernelType,
                                      int *mask,
                                      bool *timePoint,
                                      bool *axis)
{
   if(image->nx>2048 || image->ny>2048 || image->nz>2048){
      reg_print_fct_error("reg_tools_kernelConvolution_core");
      reg_print_msg_error("This function does not support images with dimension > 2048");
      reg_exit();
   }
#ifdef WIN32
   long index;
   long voxelNumber = (long)image->nx*image->ny*image->nz;
#else
   size_t index;
   size_t voxelNumber = (size_t)image->nx*image->ny*image->nz;
#endif
   DTYPE *imagePtr = static_cast<DTYPE *>(image->data);
   int imageDim[3]= {image->nx,image->ny,image->nz};

   bool *nanImagePtr = (bool *)calloc(voxelNumber, sizeof(bool));
   float *densityPtr = (float *)calloc(voxelNumber, sizeof(float));

   // Loop over the dimension higher than 3
   for(int t=0; t<image->nt*image->nu; t++)
   {
      if(timePoint[t])
      {
         DTYPE *intensityPtr = &imagePtr[t * voxelNumber];
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(densityPtr, intensityPtr, mask, nanImagePtr, voxelNumber) \
   private(index)
#endif
         for(index=0; index<voxelNumber; index++)
         {
            densityPtr[index] = (intensityPtr[index]==intensityPtr[index])?1:0;
            densityPtr[index] *= (mask[index]>=0)?1:0;
            nanImagePtr[index] = static_cast<bool>(densityPtr[index]);
            if(nanImagePtr[index]==0)
               intensityPtr[index]=static_cast<DTYPE>(0);
         }
         // Loop over the x, y and z dimensions
         for(int n=0; n<3; n++)
         {
            if(axis[n] && image->dim[n]>1)
            {
               double temp;
               if(sigma[t]>0) temp=sigma[t]/image->pixdim[n+1]; // mm to voxel
               else temp=fabs(sigma[t]); // voxel based if negative value
               int radius=0;
               // Define the kernel size
               if(kernelType==MEAN_KERNEL || kernelType==LINEAR_KERNEL)
               {
                  // Mean  or linear filtering
                  radius = static_cast<int>(temp);
               }
               else if(kernelType==GAUSSIAN_KERNEL)
               {
                  // Gaussian kernel
                  radius=static_cast<int>(temp*3.0f);
               }
               else if(kernelType==CUBIC_SPLINE_KERNEL)
               {
                  // Spline kernel
                  radius=static_cast<int>(temp*2.0f);
               }
               else{
                  reg_print_fct_error("reg_tools_kernelConvolution_core");
                  reg_print_msg_error("Unknown kernel type");
                  reg_exit();
               }
               if(radius>0)
               {
                  // Allocate the kernel
                  float kernel[4096];
                  double kernelSum=0;
                  // Fill the kernel
                  if(kernelType==CUBIC_SPLINE_KERNEL)
                  {
                     // Compute the Cubic Spline kernel
                     for(int i=-radius; i<=radius; i++)
                     {
                        // temp contains the kernel node spacing
                        double relative = (double)(fabs((double)(double)i/(double)temp));
                        if(relative<1.0) kernel[i+radius] = (float)(2.0/3.0 - relative*relative + 0.5*relative*relative*relative);
                        else if (relative<2.0) kernel[i+radius] = (float)(-(relative-2.0)*(relative-2.0)*(relative-2.0)/6.0);
                        else kernel[i+radius]=0;
                        kernelSum += kernel[i+radius];
                     }
                  }
                  else if(kernelType==GAUSSIAN_KERNEL)
                  {
                     // Compute the Gaussian kernel
                     for(int i=-radius; i<=radius; i++)
                     {
                        // 2.506... = sqrt(2*pi)
                        // temp contains the sigma in voxel
                        kernel[radius+i]=static_cast<float>(exp(-(double)(i*i)/(2.0*reg_pow2(temp))) /
                                                            (temp*2.506628274631));
                        kernelSum += kernel[radius+i];
                     }
                  }
                  else if(kernelType==LINEAR_KERNEL)
                  {
                     // Compute the linear kernel
                     for(int i=-radius; i<=radius; i++)
                     {
                        kernel[radius+i]= 1.f - fabs(static_cast<float>(i)/static_cast<float>(radius));
                        kernelSum += kernel[radius+i];
                     }
                  }
                  else if(kernelType==MEAN_KERNEL && imageDim[2]==1)
                  {
                     // Compute the mean kernel
                     for(int i=-radius; i<=radius; i++)
                     {
                        kernel[radius+i]= 1.f;
                        kernelSum += kernel[radius+i];
                     }
                  }
                  // No kernel is required for the mean filtering
                  // No need for kernel normalisation as this is handle by the density function
#ifndef NDEBUG
                  char text[255];
                  sprintf(text, "Convolution type[%i] dim[%i] tp[%i] radius[%i] kernelSum[%g]", kernelType, n, t, radius, kernelSum);
                  reg_print_msg_debug(text);
#endif
                  int planeNumber, planeIndex, lineOffset;
                  int lineIndex, shiftPre, shiftPst, k;
                  switch(n)
                  {
                  case 0:
                     planeNumber=imageDim[1]*imageDim[2];
                     lineOffset  = 1;
                     break;
                  case 1:
                     planeNumber = imageDim[0]*imageDim[2];
                     lineOffset  = imageDim[0];
                     break;
                  case 2:
                     planeNumber = imageDim[0]*imageDim[1];
                     lineOffset  = planeNumber;
                     break;
                  }

                  size_t realIndex;
                  float *kernelPtr, kernelValue;
                  double densitySum, intensitySum;
                  DTYPE *currentIntensityPtr=NULL;
                  float *currentDensityPtr = NULL;
                  DTYPE bufferIntensity[2048];
                  float bufferDensity[2048];
                  double bufferIntensitycur=0;
                  double bufferDensitycur=0;

#ifdef _USE_SSE
                  union
                  {
                     __m128 m;
                     float f[4] ;
                  } intensity_sum_sse, density_sum_sse;
                  __m128 kernel_sse, intensity_sse, density_sse;
#endif

#if defined (_OPENMP)
#ifdef _USE_SSE
#pragma omp parallel for default(none) \
   shared(imageDim, intensityPtr, densityPtr, radius, kernel, lineOffset, n, \
   planeNumber,kernelSum) \
   private(realIndex,currentIntensityPtr,currentDensityPtr,lineIndex,bufferIntensity, \
   bufferDensity,shiftPre,shiftPst,kernelPtr,kernelValue,densitySum,intensitySum, \
   k, bufferIntensitycur,bufferDensitycur, planeIndex, \
   kernel_sse, intensity_sse, density_sse, intensity_sum_sse, density_sum_sse)
#else
#pragma omp parallel for default(none) \
   shared(imageDim, intensityPtr, densityPtr, radius, kernel, lineOffset, n, \
   planeNumber,kernelSum) \
   private(realIndex,currentIntensityPtr,currentDensityPtr,lineIndex,bufferIntensity, \
   bufferDensity,shiftPre,shiftPst,kernelPtr,kernelValue,densitySum,intensitySum, \
   k, bufferIntensitycur,bufferDensitycur, planeIndex)
#endif
#endif // _OPENMP
                  // Loop over the different voxel
                  for(planeIndex=0; planeIndex<planeNumber; ++planeIndex)
                  {

                     switch(n)
                     {
                     case 0:
                        realIndex = planeIndex * imageDim[0];
                        break;
                     case 1:
                        realIndex = (planeIndex/imageDim[0]) *
                              imageDim[0]*imageDim[1] +
                              planeIndex%imageDim[0];
                        break;
                     case 2:
                        realIndex = planeIndex;
                        break;
                     default:
                        realIndex=0;
                     }
                     // Fetch the current line into a stack buffer
                     currentIntensityPtr= &intensityPtr[realIndex];
                     currentDensityPtr  = &densityPtr[realIndex];
                     for(lineIndex=0; lineIndex<imageDim[n]; ++lineIndex)
                     {
                        bufferIntensity[lineIndex] = *currentIntensityPtr;
                        bufferDensity[lineIndex]   = *currentDensityPtr;
                        currentIntensityPtr       += lineOffset;
                        currentDensityPtr         += lineOffset;
                     }
                     if(kernelSum>0)
                     {
                        // Perform the kernel convolution along 1 line
                        for(lineIndex=0; lineIndex<imageDim[n]; ++lineIndex)
                        {
                           // Define the kernel boundaries
                           shiftPre = lineIndex - radius;
                           shiftPst = lineIndex + radius + 1;
                           if(shiftPre<0)
                           {
                              kernelPtr = &kernel[-shiftPre];
                              shiftPre=0;
                           }
                           else kernelPtr = &kernel[0];
                           if(shiftPst>imageDim[n]) shiftPst=imageDim[n];
                           // Set the current values to zero
                           // Increment the current value by performing the weighted sum
#ifdef _USE_SSE
                           intensity_sum_sse.m = _mm_set_ps1(0.0);
                           density_sum_sse.m = _mm_set_ps1(0.0);
                           k=shiftPre;
                           while(k<shiftPst-3)
                           {
                              kernel_sse = _mm_set_ps(kernelPtr[0], kernelPtr[1], kernelPtr[2], kernelPtr[3]);
                              kernelPtr+=4;
                              intensity_sse = _mm_set_ps(static_cast<float>(bufferIntensity[k]),
                                                         static_cast<float>(bufferIntensity[k+1]),
                                                         static_cast<float>(bufferIntensity[k+2]),
                                                         static_cast<float>(bufferIntensity[k+3]));
                              density_sse = _mm_set_ps(bufferDensity[k],
                                    bufferDensity[k+1],
                                    bufferDensity[k+2],
                                    bufferDensity[k+3]);
                              k+=4;
                              intensity_sum_sse.m = _mm_add_ps(_mm_mul_ps(kernel_sse, intensity_sse), intensity_sum_sse.m);
                              density_sum_sse.m = _mm_add_ps(_mm_mul_ps(kernel_sse, density_sse), density_sum_sse.m);
                           }
#ifdef __SSE3__
                           intensity_sum_sse.m = _mm_hadd_ps(intensity_sum_sse.m, density_sum_sse.m);
                           intensity_sum_sse.m = _mm_hadd_ps(intensity_sum_sse.m, intensity_sum_sse.m);
                           intensitySum = intensity_sum_sse.f[0];
                           densitySum = intensity_sum_sse.f[1];
#else
                           intensitySum = intensity_sum_sse.f[0] + intensity_sum_sse.f[1] + intensity_sum_sse.f[2] + intensity_sum_sse.f[3];
                           densitySum = density_sum_sse.f[0] + density_sum_sse.f[1] + density_sum_sse.f[2] + density_sum_sse.f[3];
#endif
                           while(k<shiftPst)
                           {
                              kernelValue   = *kernelPtr++;
                              intensitySum +=  kernelValue * bufferIntensity[k];
                              densitySum   +=  kernelValue * bufferDensity[k++];
                           }
#else
                           intensitySum=0;
                           densitySum=0;
                           for(k=shiftPre; k<shiftPst; ++k)
                           {
                              kernelValue   = *kernelPtr++;
                              intensitySum +=  kernelValue * bufferIntensity[k];
                              densitySum   +=  kernelValue * bufferDensity[k];
                           }
#endif
                           // Store the computed value inplace
                           intensityPtr[realIndex] = static_cast<DTYPE>(intensitySum);
                           densityPtr[realIndex] = static_cast<float>(densitySum);
                           realIndex += lineOffset;
                        } // line convolution
                     } // kernel sum
                     else
                     {
                        for(lineIndex=1; lineIndex<imageDim[n]; ++lineIndex)
                        {
                           bufferIntensity[lineIndex]+=bufferIntensity[lineIndex-1];
                           bufferDensity[lineIndex]+=bufferDensity[lineIndex-1];
                        }
                        shiftPre = -radius - 1;
                        shiftPst = radius;
                        for(lineIndex=0; lineIndex<imageDim[n]; ++lineIndex,++shiftPre,++shiftPst)
                        {
                           if(shiftPre>-1)
                           {
                              if(shiftPst<imageDim[n])
                              {
                                 bufferIntensitycur = bufferIntensity[shiftPre]-bufferIntensity[shiftPst];
                                 bufferDensitycur = bufferDensity[shiftPre]-bufferDensity[shiftPst];
                              }
                              else
                              {
                                 bufferIntensitycur = bufferIntensity[shiftPre]-bufferIntensity[imageDim[n]-1];
                                 bufferDensitycur = bufferDensity[shiftPre]-bufferDensity[imageDim[n]-1];
                              }
                           }
                           else
                           {
                              if(shiftPst<imageDim[n])
                              {
                                 bufferIntensitycur = -bufferIntensity[shiftPst];
                                 bufferDensitycur = -bufferDensity[shiftPst];
                              }
                              else{
                                 bufferIntensitycur = 0;
                                 bufferDensitycur = 0;
                              }
                           }
                           intensityPtr[realIndex]=static_cast<DTYPE>(bufferIntensitycur);
                           densityPtr[realIndex]=static_cast<float>(bufferDensitycur);

                           realIndex += lineOffset;
                        } // line convolution of mean filter
                     } // No kernel computation
                  } // pixel in starting plane
               } // radius > 0
            } // active axis
         } // axes
         // Normalise per timepoint
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(voxelNumber, intensityPtr, densityPtr, nanImagePtr) \
   private(index)
#endif
         for(index=0; index<voxelNumber; ++index)
         {
            if(nanImagePtr[index]!=0)
               intensityPtr[index] = static_cast<DTYPE>((float)intensityPtr[index]/densityPtr[index]);
            else intensityPtr[index] = std::numeric_limits<DTYPE>::quiet_NaN();
         }
      } // check if the time point is active
   } // loop over the time points
   free(nanImagePtr);
   free(densityPtr);
}


/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_tools_labelKernelConvolution_core(nifti_image *image,
                                           float varianceX,
                                           float varianceY,
                                           float varianceZ,
                                           int *mask,
                                           bool *timePoint)
{
   if(image->nx>2048 || image->ny>2048 || image->nz>2048){
      reg_print_fct_error("reg_tools_labelKernelConvolution_core");
      reg_print_msg_error("This function does not support images with dimension > 2048");
      reg_exit();
   }
#ifdef WIN32
   long index;
   long voxelNumber = (long)image->nx*image->ny*image->nz;
#else
   size_t index;
   size_t voxelNumber = (size_t)image->nx*image->ny*image->nz;
#endif
   DTYPE *imagePtr = static_cast<DTYPE *>(image->data);

   bool * activeTimePoint = (bool *)calloc(image->nt*image->nu,sizeof(bool));
   // Check if input time points and masks are NULL
   if(timePoint==NULL)
   {
      // All time points are considered as active
      for(int i=0; i<image->nt*image->nu; i++) activeTimePoint[i]=true;
   }
   else for(int i=0; i<image->nt*image->nu; i++) activeTimePoint[i]=timePoint[i];

   int *currentMask=NULL;
   if(mask==NULL)
   {
      currentMask=(int *)calloc(image->nx*image->ny*image->nz,sizeof(int));
   }
   else currentMask=mask;


   bool *nanImagePtr = (bool *)calloc(voxelNumber, sizeof(bool));
   DTYPE *tmpImagePtr = (DTYPE *)calloc(voxelNumber, sizeof(DTYPE));

   typedef std::map <DTYPE, float> DataPointMap;
   typedef std::pair <DTYPE, float> DataPointPair;
   typedef typename std::map<DTYPE,float>::iterator DataPointMapIt;

   // Loop over the dimension higher than 3
   for(int t=0; t<image->nt*image->nu; t++)
   {
      if(activeTimePoint[t])
      {
         DTYPE *intensityPtr = &imagePtr[t * voxelNumber];
         for(index=0; index<voxelNumber; index++)
         {
            nanImagePtr[index] = (intensityPtr[index]==intensityPtr[index])?true:false;
            nanImagePtr[index] = (currentMask[index]>=0)?nanImagePtr[index]:false;
         }
         float gaussX_var=varianceX;
         float gaussY_var=varianceY;
         float gaussZ_var=varianceZ;
         index=0;
         int currentXYZposition[3]={0};
         int dim_array[3]= {image->nx,image->ny,image->nz};
         int shiftdirection[3]= {1,image->nx,image->nx*image->ny};

         int kernelXsize, kernelXshift, shiftXstart, shiftXstop;
         int kernelYsize, kernelYshift, shiftYstart, shiftYstop;
         int kernelZsize, kernelZshift, shiftZstart, shiftZstop;
         int shiftx, shifty, shiftz;
         int indexNeighbour;
         float kernelval;
         DTYPE maxindex;
         double maxval;
         DataPointMapIt location, currIterator;
         DataPointMap tmp_lab;

         for(int currentZposition=0; currentZposition<dim_array[2]; currentZposition++)
         {
            currentXYZposition[2]=currentZposition;
            for(currentXYZposition[1]=0; currentXYZposition[1]<dim_array[1]; currentXYZposition[1]++)
            {
               for(currentXYZposition[0]=0; currentXYZposition[0]<dim_array[0]; currentXYZposition[0]++)
               {

                  tmp_lab.clear();
                  index=currentXYZposition[0]+(currentXYZposition[1]+currentXYZposition[2]*dim_array[1])*dim_array[0];

                  // Calculate allowed kernel shifts
                  kernelXsize=(int)(sqrtf(gaussX_var)*6.0f) % 2 != 0 ?
                           (int)(sqrtf(gaussX_var)*6.0f) : (int)(sqrtf(gaussX_var)*6.0f)+1;
                  kernelXshift=(int)(kernelXsize/2.0f);
                  shiftXstart=((currentXYZposition[0]<kernelXshift)?
                           -currentXYZposition[0]:-kernelXshift);
                  shiftXstop=((currentXYZposition[0]>=(dim_array[0]-kernelXshift))?
                           (int)dim_array[0]-currentXYZposition[0]-1:kernelXshift);

                  kernelYsize=(int)(sqrtf(gaussY_var)*6.0f) % 2 != 0 ?
                           (int)(sqrtf(gaussY_var)*6.0f) : (int)(sqrtf(gaussY_var)*6.0f)+1;
                  kernelYshift=(int)(kernelYsize/2.0f);
                  shiftYstart=((currentXYZposition[1]<kernelYshift)?
                           -currentXYZposition[1]:-kernelYshift);
                  shiftYstop=((currentXYZposition[1]>=(dim_array[1]-kernelYshift))?
                           (int)dim_array[1]-currentXYZposition[1]-1:kernelYshift);

                  kernelZsize=(int)(sqrtf(gaussZ_var)*6.0f) % 2 != 0 ?
                           (int)(sqrtf(gaussZ_var)*6.0f) : (int)(sqrtf(gaussZ_var)*6.0f)+1;
                  kernelZshift=(int)(kernelZsize/2.0f);
                  shiftZstart=((currentXYZposition[2]<kernelZshift)?
                           -currentXYZposition[2]:-kernelZshift);
                  shiftZstop=((currentXYZposition[2]>=(dim_array[2]-kernelZshift))?
                           (int)dim_array[2]-currentXYZposition[2]-1:kernelZshift);

                  if(nanImagePtr[index]!=0){
                     for(shiftx=shiftXstart; shiftx<=shiftXstop; shiftx++)
                     {
                        for(shifty=shiftYstart; shifty<=shiftYstop; shifty++)
                        {
                           for(shiftz=shiftZstart; shiftz<=shiftZstop; shiftz++)
                           {

                              // Data Blur
                              indexNeighbour=index+(shiftx*shiftdirection[0])+
                                    (shifty*shiftdirection[1])+(shiftz*shiftdirection[2]);
                              if(nanImagePtr[indexNeighbour]!=0){
                                 kernelval=expf((float)(-0.5f *(powf(shiftx,2)/gaussX_var
                                                                +powf(shifty,2)/gaussY_var
                                                                +powf(shiftz,2)/gaussZ_var
                                                                )))/
                                       (sqrtf(2.0f*3.14159265*powf(gaussX_var*gaussY_var*gaussZ_var, 2)));

                                 location=tmp_lab.find(intensityPtr[indexNeighbour]);
                                 if(location!=tmp_lab.end())
                                 {
                                    location->second=location->second+kernelval;
                                 }
                                 else
                                 {
                                    tmp_lab.insert(DataPointPair(intensityPtr[indexNeighbour],kernelval));
                                 }
                              }
                           }
                        }
                     }
                     currIterator = tmp_lab.begin();
                     maxindex=0;
                     maxval=-std::numeric_limits<float>::max();;
                     while(currIterator != tmp_lab.end())
                     {
                        if(currIterator->second>maxval)
                        {
                           maxindex=currIterator->first;
                           maxval=currIterator->second;
                        }
                        currIterator++;
                     }
                     tmpImagePtr[index]=maxindex;
                  }
                  else{
                     tmpImagePtr[index]=std::numeric_limits<DTYPE>::quiet_NaN();
                  }
               }
            }
         }
         // Normalise per timepoint
         for(index=0; index<voxelNumber; ++index)
         {
            if(nanImagePtr[index]==0)
               intensityPtr[index] = std::numeric_limits<DTYPE>::quiet_NaN();
            else
               intensityPtr[index]=tmpImagePtr[index];
         }
      } // check if the time point is active
   } // loop over the time points

   free(tmpImagePtr);
   free(currentMask);
   free(activeTimePoint);
   free(nanImagePtr);
}
/* *************************************************************** */

void reg_tools_labelKernelConvolution(nifti_image *image,
                                      float varianceX,
                                      float varianceY,
                                      float varianceZ,
                                      int *mask,
                                      bool *timePoint){
   switch(image->datatype)
   {
   case NIFTI_TYPE_UINT8:
      reg_tools_labelKernelConvolution_core<unsigned char>
            (image,varianceX,varianceY,varianceZ,mask,timePoint);
      break;
   case NIFTI_TYPE_INT8:
      reg_tools_labelKernelConvolution_core<char>
            (image,varianceX,varianceY,varianceZ,mask,timePoint);
      break;
   case NIFTI_TYPE_UINT16:
      reg_tools_labelKernelConvolution_core<unsigned short>
            (image,varianceX,varianceY,varianceZ,mask,timePoint);
      break;
   case NIFTI_TYPE_INT16:
      reg_tools_labelKernelConvolution_core<short>
            (image,varianceX,varianceY,varianceZ,mask,timePoint);
      break;
   case NIFTI_TYPE_UINT32:
      reg_tools_labelKernelConvolution_core<unsigned int>
            (image,varianceX,varianceY,varianceZ,mask,timePoint);
      break;
   case NIFTI_TYPE_INT32:
      reg_tools_labelKernelConvolution_core<int>
            (image,varianceX,varianceY,varianceZ,mask,timePoint);
      break;
   case NIFTI_TYPE_FLOAT32:
      reg_tools_labelKernelConvolution_core<float>
            (image,varianceX,varianceY,varianceZ,mask,timePoint);
      break;
   case NIFTI_TYPE_FLOAT64:
      reg_tools_labelKernelConvolution_core<double>
            (image,varianceX,varianceY,varianceZ,mask,timePoint);
      break;
   default:
      reg_print_fct_error("reg_tools_labelKernelConvolution");
      reg_print_msg_error("The image data type is not supported");
      reg_exit();
   }
   return;
}
/* *************************************************************** */

void reg_tools_kernelConvolution(nifti_image *image,
                                 float *sigma,
                                 int kernelType,
                                 int *mask,
                                 bool *timePoint,
                                 bool *axis)
{


   if(image->nt<=0) image->nt=image->dim[4]=1;
   if(image->nu<=0) image->nu=image->dim[5]=1;

   bool *axisToSmooth = new bool[3];
   bool *activeTimePoint = new bool[image->nt*image->nu];
   if(axis==NULL)
   {
      // All axis are smoothed by default
      for(int i=0; i<3; i++) axisToSmooth[i]=true;
   }
   else for(int i=0; i<3; i++) axisToSmooth[i]=axis[i];

   if(timePoint==NULL)
   {
      // All time points are considered as active
      for(int i=0; i<image->nt*image->nu; i++) activeTimePoint[i]=true;
   }
   else for(int i=0; i<image->nt*image->nu; i++) activeTimePoint[i]=timePoint[i];

   int *currentMask=NULL;
   if(mask==NULL)
   {
      currentMask=(int *)calloc(image->nx*image->ny*image->nz,sizeof(int));
   }
   else currentMask=mask;

   switch(image->datatype)
   {
   case NIFTI_TYPE_FLOAT32:
      reg_tools_kernelConvolution_core<float>(image, sigma, kernelType, currentMask, activeTimePoint, axisToSmooth);
      break;
   case NIFTI_TYPE_FLOAT64:
      reg_tools_kernelConvolution_core<double>(image, sigma, kernelType, currentMask, activeTimePoint, axisToSmooth);
      break;
   default:
      reg_print_fct_error("reg_tools_kernelConvolution");
      reg_print_msg_error("The image data type is not supported");
      reg_exit();
   }

   if(mask==NULL) free(currentMask);
   delete []axisToSmooth;
   delete []activeTimePoint;
}
/* *************************************************************** */
/* *************************************************************** */
template <class PrecisionTYPE, class ImageTYPE>
void reg_downsampleImage1(nifti_image *image, int type, bool *downsampleAxis)
{
   if(type==1)
   {
      /* the input image is first smooth */
      float *sigma=new float[image->nt];
      for(int i=0; i<image->nt; ++i) sigma[i]=-0.7355f;
      reg_tools_kernelConvolution(image,sigma,GAUSSIAN_KERNEL);
      delete []sigma;
   }

   /* the values are copied */
   ImageTYPE *oldValues = (ImageTYPE *)malloc(image->nvox * image->nbyper);
   ImageTYPE *imagePtr = static_cast<ImageTYPE *>(image->data);
   memcpy(oldValues, imagePtr, image->nvox*image->nbyper);
   free(image->data);

   // Keep the previous real to voxel qform
   mat44 real2Voxel_qform;
   for(int i=0; i<4; i++)
   {
      for(int j=0; j<4; j++)
      {
         real2Voxel_qform.m[i][j]=image->qto_ijk.m[i][j];
      }
   }

   // Update the axis dimension
   int oldDim[4];
   for(int i=1; i<4; i++)
   {
      oldDim[i]=image->dim[i];
      if(image->dim[i]>1 && downsampleAxis[i]==true) image->dim[i]=static_cast<int>(reg_ceil(image->dim[i]/2.0));
      if(image->pixdim[i]>0 && downsampleAxis[i]==true) image->pixdim[i]=image->pixdim[i]*2.0f;
   }
   image->nx=image->dim[1];
   image->ny=image->dim[2];
   image->nz=image->dim[3];
   image->dx=image->pixdim[1];
   image->dy=image->pixdim[2];
   image->dz=image->pixdim[3];
   if(image->nt<1 || image->dim[4]<1) image->nt=image->dim[4]=1;
   if(image->nu<1 || image->dim[5]<1) image->nu=image->dim[5]=1;
   if(image->nv<1 || image->dim[6]<1) image->nv=image->dim[6]=1;
   if(image->nw<1 || image->dim[7]<1) image->nw=image->dim[7]=1;

   // update the qform matrix
   image->qto_xyz=nifti_quatern_to_mat44(image->quatern_b,
                                         image->quatern_c,
                                         image->quatern_d,
                                         image->qoffset_x,
                                         image->qoffset_y,
                                         image->qoffset_z,
                                         image->dx,
                                         image->dy,
                                         image->dz,
                                         image->qfac);
   image->qto_ijk = nifti_mat44_inverse(image->qto_xyz);

   // update the sform matrix
   if(downsampleAxis[1])
   {
      image->sto_xyz.m[0][0] *= 2.f;
      image->sto_xyz.m[1][0] *= 2.f;
      image->sto_xyz.m[2][0] *= 2.f;
   }
   if(downsampleAxis[2])
   {
      image->sto_xyz.m[0][1] *= 2.f;
      image->sto_xyz.m[1][1] *= 2.f;
      image->sto_xyz.m[2][1] *= 2.f;
   }
   if(downsampleAxis[3])
   {
      image->sto_xyz.m[0][2] *= 2.f;
      image->sto_xyz.m[1][2] *= 2.f;
      image->sto_xyz.m[2][2] *= 2.f;
   }
   float origin_sform[3]= {image->sto_xyz.m[0][3], image->sto_xyz.m[1][3], image->sto_xyz.m[2][3]};
   image->sto_xyz.m[0][3]=origin_sform[0];
   image->sto_xyz.m[1][3]=origin_sform[1];
   image->sto_xyz.m[2][3]=origin_sform[2];
   image->sto_ijk = nifti_mat44_inverse(image->sto_xyz);

   // Reallocate the image
   image->nvox =
         (size_t)image->nx*
         (size_t)image->ny*
         (size_t)image->nz*
         (size_t)image->nt*
         (size_t)image->nu*
         (size_t)image->nv*
         (size_t)image->nw;
   image->data=(void *)calloc(image->nvox, image->nbyper);
   imagePtr = static_cast<ImageTYPE *>(image->data);

   PrecisionTYPE real[3];
   ImageTYPE intensity;
   int position[3];

   // qform is used for resampling
   for(size_t tuvw=0; tuvw<(size_t)image->nt*image->nu*image->nv*image->nw; tuvw++)
   {
      ImageTYPE *valuesPtrTUVW = &oldValues[tuvw*oldDim[1]*oldDim[2]*oldDim[3]];
      for(int z=0; z<image->nz; z++)
      {
         for(int y=0; y<image->ny; y++)
         {
            for(int x=0; x<image->nx; x++)
            {
               // Extract the voxel coordinate in mm
               real[0]=x*image->qto_xyz.m[0][0] +
                     y*image->qto_xyz.m[0][1] +
                     z*image->qto_xyz.m[0][2] +
                     image->qto_xyz.m[0][3];
               real[1]=x*image->qto_xyz.m[1][0] +
                     y*image->qto_xyz.m[1][1] +
                     z*image->qto_xyz.m[1][2] +
                     image->qto_xyz.m[1][3];
               real[2]=x*image->qto_xyz.m[2][0] +
                     y*image->qto_xyz.m[2][1] +
                     z*image->qto_xyz.m[2][2] +
                     image->qto_xyz.m[2][3];
               // Extract the position in voxel in the old image;
               position[0]=(int)reg_round(real[0]*real2Voxel_qform.m[0][0] + real[1]*real2Voxel_qform.m[0][1] + real[2]*real2Voxel_qform.m[0][2] + real2Voxel_qform.m[0][3]);
               position[1]=(int)reg_round(real[0]*real2Voxel_qform.m[1][0] + real[1]*real2Voxel_qform.m[1][1] + real[2]*real2Voxel_qform.m[1][2] + real2Voxel_qform.m[1][3]);
               position[2]=(int)reg_round(real[0]*real2Voxel_qform.m[2][0] + real[1]*real2Voxel_qform.m[2][1] + real[2]*real2Voxel_qform.m[2][2] + real2Voxel_qform.m[2][3]);
               if(oldDim[3]==1) position[2]=0;
               // Nearest neighboor is used as downsampling ratio is constant
               intensity=std::numeric_limits<ImageTYPE>::quiet_NaN();
               if(-1<position[0] && position[0]<oldDim[1] &&
                     -1<position[1] && position[1]<oldDim[2] &&
                     -1<position[2] && position[2]<oldDim[3])
               {
                  intensity = valuesPtrTUVW[(position[2]*oldDim[2]+position[1])*oldDim[1]+position[0]];
               }
               *imagePtr=intensity;
               imagePtr++;
            }
         }
      }
   }
   free(oldValues);
}
/* *************************************************************** */
template <class PrecisionTYPE>
void reg_downsampleImage(nifti_image *image, int type, bool *downsampleAxis)
{
   switch(image->datatype)
   {
   case NIFTI_TYPE_UINT8:
      reg_downsampleImage1<PrecisionTYPE,unsigned char>(image, type, downsampleAxis);
      break;
   case NIFTI_TYPE_INT8:
      reg_downsampleImage1<PrecisionTYPE,char>(image, type, downsampleAxis);
      break;
   case NIFTI_TYPE_UINT16:
      reg_downsampleImage1<PrecisionTYPE,unsigned short>(image, type, downsampleAxis);
      break;
   case NIFTI_TYPE_INT16:
      reg_downsampleImage1<PrecisionTYPE,short>(image, type, downsampleAxis);
      break;
   case NIFTI_TYPE_UINT32:
      reg_downsampleImage1<PrecisionTYPE,unsigned int>(image, type, downsampleAxis);
      break;
   case NIFTI_TYPE_INT32:
      reg_downsampleImage1<PrecisionTYPE,int>(image, type, downsampleAxis);
      break;
   case NIFTI_TYPE_FLOAT32:
      reg_downsampleImage1<PrecisionTYPE,float>(image, type, downsampleAxis);
      break;
   case NIFTI_TYPE_FLOAT64:
      reg_downsampleImage1<PrecisionTYPE,double>(image, type, downsampleAxis);
      break;
   default:
      reg_print_fct_error("reg_downsampleImage");
      reg_print_msg_error("The image data type is not supported");
      reg_exit();
   }
}
template void reg_downsampleImage<float>(nifti_image *, int, bool *);
template void reg_downsampleImage<double>(nifti_image *, int, bool *);
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_tools_binarise_image1(nifti_image *image)
{
   DTYPE *dataPtr=static_cast<DTYPE *>(image->data);
   image->scl_inter=0.f;
   image->scl_slope=1.f;
   for(size_t i=0; i<image->nvox; i++)
   {
      *dataPtr = (*dataPtr)!=0?(DTYPE)1:(DTYPE)0;
      dataPtr++;
   }
}
/* *************************************************************** */
void reg_tools_binarise_image(nifti_image *image)
{
   switch(image->datatype)
   {
   case NIFTI_TYPE_UINT8:
      reg_tools_binarise_image1<unsigned char>(image);
      break;
   case NIFTI_TYPE_INT8:
      reg_tools_binarise_image1<char>(image);
      break;
   case NIFTI_TYPE_UINT16:
      reg_tools_binarise_image1<unsigned short>(image);
      break;
   case NIFTI_TYPE_INT16:
      reg_tools_binarise_image1<short>(image);
      break;
   case NIFTI_TYPE_UINT32:
      reg_tools_binarise_image1<unsigned int>(image);
      break;
   case NIFTI_TYPE_INT32:
      reg_tools_binarise_image1<int>(image);
      break;
   case NIFTI_TYPE_FLOAT32:
      reg_tools_binarise_image1<float>(image);
      break;
   case NIFTI_TYPE_FLOAT64:
      reg_tools_binarise_image1<double>(image);
      break;
   default:
      reg_print_fct_error("reg_tools_binarise_image");
      reg_print_msg_error("The image data type is not supported");
      reg_exit();
   }
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_tools_binarise_image1(nifti_image *image, float threshold)
{
   DTYPE *dataPtr=static_cast<DTYPE *>(image->data);
   for(size_t i=0; i<image->nvox; i++)
   {
      *dataPtr = (*dataPtr)<threshold?(DTYPE)0:(DTYPE)1;
      dataPtr++;
   }
}
/* *************************************************************** */
void reg_tools_binarise_image(nifti_image *image, float threshold)
{
   switch(image->datatype)
   {
   case NIFTI_TYPE_UINT8:
      reg_tools_binarise_image1<unsigned char>(image, threshold);
      break;
   case NIFTI_TYPE_INT8:
      reg_tools_binarise_image1<char>(image, threshold);
      break;
   case NIFTI_TYPE_UINT16:
      reg_tools_binarise_image1<unsigned short>(image, threshold);
      break;
   case NIFTI_TYPE_INT16:
      reg_tools_binarise_image1<short>(image, threshold);
      break;
   case NIFTI_TYPE_UINT32:
      reg_tools_binarise_image1<unsigned int>(image, threshold);
      break;
   case NIFTI_TYPE_INT32:
      reg_tools_binarise_image1<int>(image, threshold);
      break;
   case NIFTI_TYPE_FLOAT32:
      reg_tools_binarise_image1<float>(image, threshold);
      break;
   case NIFTI_TYPE_FLOAT64:
      reg_tools_binarise_image1<double>(image, threshold);
      break;
   default:
      reg_print_fct_error("reg_tools_binarise_image");
      reg_print_msg_error("The image data type is not supported");
      reg_exit();
   }
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_tools_binaryImage2int1(nifti_image *image, int *array, int &activeVoxelNumber)
{
   // Active voxel are different from -1
   activeVoxelNumber=0;
   DTYPE *dataPtr=static_cast<DTYPE *>(image->data);
   for(int i=0; i<image->nx*image->ny*image->nz; i++)
   {
      if(*dataPtr++ != 0)
      {
         array[i]=1;
         activeVoxelNumber++;
      }
      else
      {
         array[i]=-1;
      }
   }
}
/* *************************************************************** */
void reg_tools_binaryImage2int(nifti_image *image, int *array, int &activeVoxelNumber)
{
   switch(image->datatype)
   {
   case NIFTI_TYPE_UINT8:
      reg_tools_binaryImage2int1<unsigned char>(image, array, activeVoxelNumber);
      break;
   case NIFTI_TYPE_INT8:
      reg_tools_binaryImage2int1<char>(image, array, activeVoxelNumber);
      break;
   case NIFTI_TYPE_UINT16:
      reg_tools_binaryImage2int1<unsigned short>(image, array, activeVoxelNumber);
      break;
   case NIFTI_TYPE_INT16:
      reg_tools_binaryImage2int1<short>(image, array, activeVoxelNumber);
      break;
   case NIFTI_TYPE_UINT32:
      reg_tools_binaryImage2int1<unsigned int>(image, array, activeVoxelNumber);
      break;
   case NIFTI_TYPE_INT32:
      reg_tools_binaryImage2int1<int>(image, array, activeVoxelNumber);
      break;
   case NIFTI_TYPE_FLOAT32:
      reg_tools_binaryImage2int1<float>(image, array, activeVoxelNumber);
      break;
   case NIFTI_TYPE_FLOAT64:
      reg_tools_binaryImage2int1<double>(image, array, activeVoxelNumber);
      break;
   default:
      reg_print_fct_error("reg_tools_binaryImage2int");
      reg_print_msg_error("The image data type is not supported");
      reg_exit();
   }
}
/* *************************************************************** */
/* *************************************************************** */
template <class ATYPE,class BTYPE>
double reg_tools_getMeanRMS2(nifti_image *imageA, nifti_image *imageB)
{
   ATYPE *imageAPtrX = static_cast<ATYPE *>(imageA->data);
   BTYPE *imageBPtrX = static_cast<BTYPE *>(imageB->data);
   ATYPE *imageAPtrY=NULL;
   BTYPE *imageBPtrY=NULL;
   ATYPE *imageAPtrZ=NULL;
   BTYPE *imageBPtrZ=NULL;
   if(imageA->dim[5]>1)
   {
      imageAPtrY = &imageAPtrX[imageA->nx*imageA->ny*imageA->nz];
      imageBPtrY = &imageBPtrX[imageA->nx*imageA->ny*imageA->nz];
   }
   if(imageA->dim[5]>2)
   {
      imageAPtrZ = &imageAPtrY[imageA->nx*imageA->ny*imageA->nz];
      imageBPtrZ = &imageBPtrY[imageA->nx*imageA->ny*imageA->nz];
   }
   double sum=0.0f;
   double rms;
   double diff;
   for(int i=0; i<imageA->nx*imageA->ny*imageA->nz; i++)
   {
      diff = (double)*imageAPtrX++ - (double)*imageBPtrX++;
      rms = diff * diff;
      if(imageA->dim[5]>1)
      {
         diff = (double)*imageAPtrY++ - (double)*imageBPtrY++;
         rms += diff * diff;
      }
      if(imageA->dim[5]>2)
      {
         diff = (double)*imageAPtrZ++ - (double)*imageBPtrZ++;
         rms += diff * diff;
      }
      if(rms==rms)
         sum += sqrt(rms);
   }
   return sum/(double)(imageA->nx*imageA->ny*imageA->nz);
}
/* *************************************************************** */
template <class ATYPE>
double reg_tools_getMeanRMS1(nifti_image *imageA, nifti_image *imageB)
{
   switch(imageB->datatype)
   {
   case NIFTI_TYPE_UINT8:
      return reg_tools_getMeanRMS2<ATYPE,unsigned char>(imageA, imageB);
   case NIFTI_TYPE_INT8:
      return reg_tools_getMeanRMS2<ATYPE,char>(imageA, imageB);
   case NIFTI_TYPE_UINT16:
      return reg_tools_getMeanRMS2<ATYPE,unsigned short>(imageA, imageB);
   case NIFTI_TYPE_INT16:
      return reg_tools_getMeanRMS2<ATYPE,short>(imageA, imageB);
   case NIFTI_TYPE_UINT32:
      return reg_tools_getMeanRMS2<ATYPE,unsigned int>(imageA, imageB);
   case NIFTI_TYPE_INT32:
      return reg_tools_getMeanRMS2<ATYPE,int>(imageA, imageB);
   case NIFTI_TYPE_FLOAT32:
      return reg_tools_getMeanRMS2<ATYPE,float>(imageA, imageB);
   case NIFTI_TYPE_FLOAT64:
      return reg_tools_getMeanRMS2<ATYPE,double>(imageA, imageB);
   default:
      reg_print_fct_error("reg_tools_getMeanRMS1");
      reg_print_msg_error("The image data type is not supported");
      reg_exit();
   }
}
/* *************************************************************** */
double reg_tools_getMeanRMS(nifti_image *imageA, nifti_image *imageB)
{
   switch(imageA->datatype)
   {
   case NIFTI_TYPE_UINT8:
      return reg_tools_getMeanRMS1<unsigned char>(imageA, imageB);
   case NIFTI_TYPE_INT8:
      return reg_tools_getMeanRMS1<char>(imageA, imageB);
   case NIFTI_TYPE_UINT16:
      return reg_tools_getMeanRMS1<unsigned short>(imageA, imageB);
   case NIFTI_TYPE_INT16:
      return reg_tools_getMeanRMS1<short>(imageA, imageB);
   case NIFTI_TYPE_UINT32:
      return reg_tools_getMeanRMS1<unsigned int>(imageA, imageB);
   case NIFTI_TYPE_INT32:
      return reg_tools_getMeanRMS1<int>(imageA, imageB);
   case NIFTI_TYPE_FLOAT32:
      return reg_tools_getMeanRMS1<float>(imageA, imageB);
   case NIFTI_TYPE_FLOAT64:
      return reg_tools_getMeanRMS1<double>(imageA, imageB);
   default:
      reg_print_fct_error("reg_tools_getMeanRMS");
      reg_print_msg_error("The image data type is not supported");
      reg_exit();
   }
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
int reg_createImagePyramid(nifti_image *inputImage, nifti_image **pyramid, int unsigned levelNumber, int unsigned levelToPerform)
{
   // FINEST LEVEL OF REGISTRATION
   pyramid[levelToPerform-1]=nifti_copy_nim_info(inputImage);
   pyramid[levelToPerform-1]->data = (void *)calloc(pyramid[levelToPerform-1]->nvox,
         pyramid[levelToPerform-1]->nbyper);
   memcpy(pyramid[levelToPerform-1]->data, inputImage->data,
         pyramid[levelToPerform-1]->nvox* pyramid[levelToPerform-1]->nbyper);
   reg_tools_changeDatatype<DTYPE>(pyramid[levelToPerform-1]);
   reg_tools_removeSCLInfo(pyramid[levelToPerform-1]);

   // Images are downsampled if appropriate
   for(unsigned int l=levelToPerform; l<levelNumber; l++)
   {
      bool downsampleAxis[8]= {false,true,true,true,false,false,false,false};
      if((pyramid[levelToPerform-1]->nx/2) < 32) downsampleAxis[1]=false;
      if((pyramid[levelToPerform-1]->ny/2) < 32) downsampleAxis[2]=false;
      if((pyramid[levelToPerform-1]->nz/2) < 32) downsampleAxis[3]=false;
      reg_downsampleImage<DTYPE>(pyramid[levelToPerform-1], 1, downsampleAxis);
   }

   // Images for each subsequent levels are allocated and downsampled if appropriate
   for(int l=levelToPerform-2; l>=0; l--)
   {
      // Allocation of the image
      pyramid[l]=nifti_copy_nim_info(pyramid[l+1]);
      pyramid[l]->data = (void *)calloc(pyramid[l]->nvox,
                                        pyramid[l]->nbyper);

      memcpy(pyramid[l]->data, pyramid[l+1]->data,
            pyramid[l]->nvox* pyramid[l]->nbyper);

      // Downsample the image if appropriate
      bool downsampleAxis[8]= {false,true,true,true,false,false,false,false};
      if((pyramid[l]->nx/2) < 32) downsampleAxis[1]=false;
      if((pyramid[l]->ny/2) < 32) downsampleAxis[2]=false;
      if((pyramid[l]->nz/2) < 32) downsampleAxis[3]=false;
      reg_downsampleImage<DTYPE>(pyramid[l], 1, downsampleAxis);
   }
   return EXIT_SUCCESS;
}
template int reg_createImagePyramid<float>(nifti_image *, nifti_image **, unsigned int , unsigned int);
template int reg_createImagePyramid<double>(nifti_image *, nifti_image **, unsigned int , unsigned int);
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
int reg_createMaskPyramid(nifti_image *inputMaskImage, int **maskPyramid, int unsigned levelNumber, int unsigned levelToPerform, int *activeVoxelNumber)
{
   // FINEST LEVEL OF REGISTRATION
   nifti_image **tempMaskImagePyramid=(nifti_image **)malloc(levelToPerform*sizeof(nifti_image *));
   tempMaskImagePyramid[levelToPerform-1]=nifti_copy_nim_info(inputMaskImage);
   tempMaskImagePyramid[levelToPerform-1]->data = (void *)calloc(tempMaskImagePyramid[levelToPerform-1]->nvox,
         tempMaskImagePyramid[levelToPerform-1]->nbyper);
   memcpy(tempMaskImagePyramid[levelToPerform-1]->data, inputMaskImage->data,
         tempMaskImagePyramid[levelToPerform-1]->nvox* tempMaskImagePyramid[levelToPerform-1]->nbyper);
   reg_tools_binarise_image(tempMaskImagePyramid[levelToPerform-1]);
   reg_tools_changeDatatype<unsigned char>(tempMaskImagePyramid[levelToPerform-1]);

   // Image is downsampled if appropriate
   for(unsigned int l=levelToPerform; l<levelNumber; l++)
   {
      bool downsampleAxis[8]= {false,true,true,true,false,false,false,false};
      if((tempMaskImagePyramid[levelToPerform-1]->nx/2) < 32) downsampleAxis[1]=false;
      if((tempMaskImagePyramid[levelToPerform-1]->ny/2) < 32) downsampleAxis[2]=false;
      if((tempMaskImagePyramid[levelToPerform-1]->nz/2) < 32) downsampleAxis[3]=false;
      reg_downsampleImage<DTYPE>(tempMaskImagePyramid[levelToPerform-1], 0, downsampleAxis);
   }
   activeVoxelNumber[levelToPerform-1]=tempMaskImagePyramid[levelToPerform-1]->nx *
         tempMaskImagePyramid[levelToPerform-1]->ny *
         tempMaskImagePyramid[levelToPerform-1]->nz;
   maskPyramid[levelToPerform-1]=(int *)malloc(activeVoxelNumber[levelToPerform-1] * sizeof(int));
   reg_tools_binaryImage2int(tempMaskImagePyramid[levelToPerform-1],
         maskPyramid[levelToPerform-1],
         activeVoxelNumber[levelToPerform-1]);

   // Images for each subsequent levels are allocated and downsampled if appropriate
   for(int l=levelToPerform-2; l>=0; l--)
   {
      // Allocation of the reference image
      tempMaskImagePyramid[l]=nifti_copy_nim_info(tempMaskImagePyramid[l+1]);
      tempMaskImagePyramid[l]->data = (void *)calloc(tempMaskImagePyramid[l]->nvox,
                                                     tempMaskImagePyramid[l]->nbyper);
      memcpy(tempMaskImagePyramid[l]->data, tempMaskImagePyramid[l+1]->data,
            tempMaskImagePyramid[l]->nvox* tempMaskImagePyramid[l]->nbyper);

      // Downsample the image if appropriate
      bool downsampleAxis[8]= {false,true,true,true,false,false,false,false};
      if((tempMaskImagePyramid[l]->nx/2) < 32) downsampleAxis[1]=false;
      if((tempMaskImagePyramid[l]->ny/2) < 32) downsampleAxis[2]=false;
      if((tempMaskImagePyramid[l]->nz/2) < 32) downsampleAxis[3]=false;
      reg_downsampleImage<DTYPE>(tempMaskImagePyramid[l], 0, downsampleAxis);

      activeVoxelNumber[l]=tempMaskImagePyramid[l]->nx *
            tempMaskImagePyramid[l]->ny *
            tempMaskImagePyramid[l]->nz;
      maskPyramid[l]=(int *)malloc(activeVoxelNumber[l] * sizeof(int));
      reg_tools_binaryImage2int(tempMaskImagePyramid[l],
                                maskPyramid[l],
                                activeVoxelNumber[l]);
   }
   for(unsigned int l=0; l<levelToPerform; ++l)
      nifti_image_free(tempMaskImagePyramid[l]);
   free(tempMaskImagePyramid);
   return EXIT_SUCCESS;
}
template int reg_createMaskPyramid<float>(nifti_image *, int **, unsigned int , unsigned int , int *);
template int reg_createMaskPyramid<double>(nifti_image *, int **, unsigned int , unsigned int , int *);
/* *************************************************************** */
/* *************************************************************** */
template <class TYPE1, class TYPE2>
int reg_tools_nanMask_image2(nifti_image *image, nifti_image *maskImage, nifti_image *outputImage)
{
   TYPE1 *imagePtr = static_cast<TYPE1 *>(image->data);
   TYPE2 *maskPtr = static_cast<TYPE2 *>(maskImage->data);
   TYPE1 *resPtr = static_cast<TYPE1 *>(outputImage->data);
   for(size_t i=0; i<image->nvox; ++i)
   {
      if(*maskPtr == 0)
         *resPtr=std::numeric_limits<TYPE1>::quiet_NaN();
      else *resPtr=*imagePtr;
      maskPtr++;
      imagePtr++;
      resPtr++;
   }
   return EXIT_SUCCESS;
}
/* *************************************************************** */
template <class TYPE1>
int reg_tools_nanMask_image1(nifti_image *image, nifti_image *maskImage, nifti_image *outputImage)
{
   switch(maskImage->datatype)
   {
   case NIFTI_TYPE_UINT8:
      return reg_tools_nanMask_image2<TYPE1,unsigned char>
            (image, maskImage, outputImage);
   case NIFTI_TYPE_INT8:
      return reg_tools_nanMask_image2<TYPE1,char>
            (image, maskImage, outputImage);
   case NIFTI_TYPE_UINT16:
      return reg_tools_nanMask_image2<TYPE1,unsigned short>
            (image, maskImage, outputImage);
   case NIFTI_TYPE_INT16:
      return reg_tools_nanMask_image2<TYPE1,short>
            (image, maskImage, outputImage);
   case NIFTI_TYPE_UINT32:
      return reg_tools_nanMask_image2<TYPE1,unsigned int>
            (image, maskImage, outputImage);
   case NIFTI_TYPE_INT32:
      return reg_tools_nanMask_image2<TYPE1,int>
            (image, maskImage, outputImage);
   case NIFTI_TYPE_FLOAT32:
      return reg_tools_nanMask_image2<TYPE1,float>
            (image, maskImage, outputImage);
   case NIFTI_TYPE_FLOAT64:
      return reg_tools_nanMask_image2<TYPE1,double>
            (image, maskImage, outputImage);
   default:
      reg_print_fct_error("reg_tools_nanMask_image1");
      reg_print_msg_error("The image data type is not supported");
      reg_exit();
   }
}
/* *************************************************************** */
int reg_tools_nanMask_image(nifti_image *image, nifti_image *maskImage, nifti_image *outputImage)
{
   // Check dimension
   if(image->nvox != maskImage->nvox || image->nvox != outputImage->nvox)
   {
      reg_print_fct_error("reg_tools_nanMask_image");
      reg_print_msg_error("Input images have different size");
      reg_exit();
   }
   // Check output data type
   if(image->datatype != outputImage->datatype)
   {
      reg_print_fct_error("reg_tools_nanMask_image");
      reg_print_msg_error("Input and output images have different data type");
      reg_exit();
   }
   switch(image->datatype)
   {
   case NIFTI_TYPE_UINT8:
      return reg_tools_nanMask_image1<unsigned char>
            (image, maskImage, outputImage);
   case NIFTI_TYPE_INT8:
      return reg_tools_nanMask_image1<char>
            (image, maskImage, outputImage);
   case NIFTI_TYPE_UINT16:
      return reg_tools_nanMask_image1<unsigned short>
            (image, maskImage, outputImage);
   case NIFTI_TYPE_INT16:
      return reg_tools_nanMask_image1<short>
            (image, maskImage, outputImage);
   case NIFTI_TYPE_UINT32:
      return reg_tools_nanMask_image1<unsigned int>
            (image, maskImage, outputImage);
   case NIFTI_TYPE_INT32:
      return reg_tools_nanMask_image1<int>
            (image, maskImage, outputImage);
   case NIFTI_TYPE_FLOAT32:
      return reg_tools_nanMask_image1<float>
            (image, maskImage, outputImage);
   case NIFTI_TYPE_FLOAT64:
      return reg_tools_nanMask_image1<double>
            (image, maskImage, outputImage);
   default:
      reg_print_fct_error("reg_tools_nanMask_image");
      reg_print_msg_error("The image data type is not supported");
      reg_exit();
   }
}
/* *************************************************************** */
/* *************************************************************** */
template <class TYPE>
int reg_tools_removeNanFromMask_core(nifti_image *image, int *mask, int time_point_index)
{
   size_t voxelNumber = (size_t)image->nx*image->ny*image->nz;
   TYPE *imagePtr = static_cast<TYPE *>(image->data);
   for(int t=0; t<time_point_index; ++t){
	imagePtr+=voxelNumber;
   }

   for(size_t i=0; i<voxelNumber; ++i){
      TYPE value=*imagePtr++;
      if(value!=value)
         mask[i]=-1;
   }

   return EXIT_SUCCESS;
}
/* *************************************************************** */
int reg_tools_removeNanFromMask(nifti_image *image,
                                int *mask, int time_point_index)
{
   switch(image->datatype)
   {
   case NIFTI_TYPE_FLOAT32:
      return reg_tools_removeNanFromMask_core<float>
            (image, mask, time_point_index);
   case NIFTI_TYPE_FLOAT64:
      return reg_tools_removeNanFromMask_core<double>
            (image, mask, time_point_index);
   default:
      reg_print_fct_error("reg_tools_removeNanFromMask");
      reg_print_msg_error("The image data type is not supported");
      reg_exit();
   }
}

/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
DTYPE reg_tools_getMinValue_core(nifti_image *image, int timepoint)
{
   if(timepoint<-1 || timepoint>=image->nt)
      reg_print_msg_error("reg_tools_getMinValue_core. The required time point does not exists");
   // Create a pointer to the image data
   DTYPE *imgPtr = static_cast<DTYPE *>(image->data);
   // Set a variable to store the minimal value
   DTYPE minValue=std::numeric_limits<DTYPE>::max();
   if(image->scl_slope==0) image->scl_slope=1.f;

   size_t voxelNumber = (size_t)image->nx*
         image->ny*image->nz;
   // Loop over all voxel to find the lowest value
   for(int time=0; time<image->nt; ++time){
      if(time==timepoint || timepoint==-1){
         for(int u=0; u<image->nu; ++u){
            DTYPE *currentVolumePtr = &imgPtr[(u*image->nt+time)*voxelNumber];
            for(size_t i=0; i<voxelNumber; ++i){
               DTYPE currentVal = (DTYPE)((float)currentVolumePtr[i] * image->scl_slope + image->scl_inter);
               minValue=currentVal<minValue?currentVal:minValue;
            }
         }
      }
   }
   // The lowest value is returned
   return minValue;
}
/* *************************************************************** */
float reg_tools_getMinValue(nifti_image *image, int timepoint)
{
   // Check the image data type
   switch(image->datatype)
   {
   case NIFTI_TYPE_UINT8:
      return reg_tools_getMinValue_core<unsigned char>(image, timepoint);
   case NIFTI_TYPE_INT8:
      return reg_tools_getMinValue_core<char>(image, timepoint);
   case NIFTI_TYPE_UINT16:
      return reg_tools_getMinValue_core<unsigned short>(image, timepoint);
   case NIFTI_TYPE_INT16:
      return reg_tools_getMinValue_core<short>(image, timepoint);
   case NIFTI_TYPE_UINT32:
      return reg_tools_getMinValue_core<unsigned int>(image, timepoint);
   case NIFTI_TYPE_INT32:
      return reg_tools_getMinValue_core<int>(image, timepoint);
   case NIFTI_TYPE_FLOAT32:
      return reg_tools_getMinValue_core<float>(image, timepoint);
   case NIFTI_TYPE_FLOAT64:
      return reg_tools_getMinValue_core<double>(image, timepoint);
   default:
      reg_print_fct_error("reg_tools_getMinValue");
      reg_print_msg_error("The image data type is not supported");
      reg_exit();
   }
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
DTYPE reg_tools_getMaxValue_core(nifti_image *image, int timepoint)
{
   if(timepoint<-1 || timepoint>=image->nt)
      reg_print_msg_error("reg_tools_getMinValue_core. The required time point does not exists");
   // Create a pointer to the image data
   DTYPE *imgPtr = static_cast<DTYPE *>(image->data);
   // Set a variable to store the minimal value
   DTYPE maxValue=std::numeric_limits<DTYPE>::min();
   if(image->scl_slope==0) image->scl_slope=1.f;

   size_t voxelNumber = (size_t)image->nx *
         image->ny * image->nz;
   // Loop over all voxel to find the lowest value
   for(int time=0; time<image->nt; ++time){
      if(time==timepoint || timepoint==-1){
         for(int u=0; u<image->nu; ++u){
            DTYPE *currentVolumePtr = &imgPtr[(u*image->nt+time)*voxelNumber];
            for(size_t i=0; i<voxelNumber; ++i){
               DTYPE currentVal = (DTYPE)((float)currentVolumePtr[i] * image->scl_slope + image->scl_inter);
               maxValue=currentVal>maxValue?currentVal:maxValue;
            } // u
         } // t
      } // if time
   } // time
   // The lowest value is returned
   return maxValue;
}
/* *************************************************************** */
float reg_tools_getMaxValue(nifti_image *image, int timepoint)
{
   // Check the image data type
   switch(image->datatype)
   {
   case NIFTI_TYPE_UINT8:
      return reg_tools_getMaxValue_core<unsigned char>(image, timepoint);
   case NIFTI_TYPE_INT8:
      return reg_tools_getMaxValue_core<char>(image, timepoint);
   case NIFTI_TYPE_UINT16:
      return reg_tools_getMaxValue_core<unsigned short>(image, timepoint);
   case NIFTI_TYPE_INT16:
      return reg_tools_getMaxValue_core<short>(image, timepoint);
   case NIFTI_TYPE_UINT32:
      return reg_tools_getMaxValue_core<unsigned int>(image, timepoint);
   case NIFTI_TYPE_INT32:
      return reg_tools_getMaxValue_core<int>(image, timepoint);
   case NIFTI_TYPE_FLOAT32:
      return reg_tools_getMaxValue_core<float>(image, timepoint);
   case NIFTI_TYPE_FLOAT64:
      return reg_tools_getMaxValue_core<double>(image, timepoint);
   default:
      reg_print_fct_error("reg_tools_getMaxValue");
      reg_print_msg_error("The image data type is not supported");
      reg_exit();
   }
}
/* *************************************************************** */
template <class DTYPE>
float reg_tools_getMeanValue_core(nifti_image *image)
{
   // Create a pointer to the image data
   DTYPE *imgPtr = static_cast<DTYPE *>(image->data);
   // Set a variable to store the minimal value
   float meanValue=0;
   if(image->scl_slope==0) image->scl_slope=1.f;
   // Loop over all voxel to find the lowest value
   for(size_t i=0; i<image->nvox; ++i)
   {
      DTYPE currentVal = (DTYPE)((float)imgPtr[i] * image->scl_slope + image->scl_inter);
      meanValue+=currentVal;
   }
   meanValue=(float)(meanValue/(double) image->nvox);
   // The lowest value is returned
   return meanValue;
}
/* *************************************************************** */
float reg_tools_getMeanValue(nifti_image *image)
{
   // Check the image data type
   switch(image->datatype)
   {
   case NIFTI_TYPE_UINT8:
      return reg_tools_getMeanValue_core<unsigned char>(image);
   case NIFTI_TYPE_INT8:
      return reg_tools_getMeanValue_core<char>(image);
   case NIFTI_TYPE_UINT16:
      return reg_tools_getMeanValue_core<unsigned short>(image);
   case NIFTI_TYPE_INT16:
      return reg_tools_getMeanValue_core<short>(image);
   case NIFTI_TYPE_UINT32:
      return reg_tools_getMeanValue_core<unsigned int>(image);
   case NIFTI_TYPE_INT32:
      return reg_tools_getMeanValue_core<int>(image);
   case NIFTI_TYPE_FLOAT32:
      return reg_tools_getMeanValue_core<float>(image);
   case NIFTI_TYPE_FLOAT64:
      return reg_tools_getMeanValue_core<double>(image);
   default:
      reg_print_fct_error("reg_tools_getMeanValue");
      reg_print_msg_error("The image data type is not supported");
      reg_exit();
   }
}
/* *************************************************************** */
template <class DTYPE>
float reg_tools_getSTDValue_core(nifti_image *image)
{
   // Create a pointer to the image data
   DTYPE *imgPtr = static_cast<DTYPE *>(image->data);
   // Set a variable to store the minimal value
   float meanValue = reg_tools_getMeanValue(image);
   float stdValue=0;
   if(image->scl_slope==0) image->scl_slope=1.f;
   // Loop over all voxel to find the lowest value
   for(size_t i=0; i<image->nvox; ++i)
   {
      DTYPE currentVal = (DTYPE)((float)imgPtr[i] * image->scl_slope + image->scl_inter);
      stdValue+=(currentVal-meanValue)*(currentVal-meanValue);
   }
   stdValue = (float) std::sqrt(stdValue/(double) image->nvox);
   // The lowest value is returned
   return stdValue;
}
/* *************************************************************** */
float reg_tools_getSTDValue(nifti_image *image)
{
   // Check the image data type
   switch(image->datatype)
   {
   case NIFTI_TYPE_UINT8:
      return reg_tools_getSTDValue_core<unsigned char>(image);
   case NIFTI_TYPE_INT8:
      return reg_tools_getSTDValue_core<char>(image);
   case NIFTI_TYPE_UINT16:
      return reg_tools_getSTDValue_core<unsigned short>(image);
   case NIFTI_TYPE_INT16:
      return reg_tools_getSTDValue_core<short>(image);
   case NIFTI_TYPE_UINT32:
      return reg_tools_getSTDValue_core<unsigned int>(image);
   case NIFTI_TYPE_INT32:
      return reg_tools_getSTDValue_core<int>(image);
   case NIFTI_TYPE_FLOAT32:
      return reg_tools_getSTDValue_core<float>(image);
   case NIFTI_TYPE_FLOAT64:
      return reg_tools_getSTDValue_core<double>(image);
   default:
      reg_print_fct_error("reg_tools_getSTDValue");
      reg_print_msg_error("The image data type is not supported");
      reg_exit();
   }
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_flippAxis_type(int nx,
                        int ny,
                        int nz,
                        int nt,
                        int nu,
                        int nv,
                        int nw,
                        void *inputArray,
                        void *outputArray,
                        std::string cmd
                        )
{
   // Allocate the outputArray if it is not allocated yet
   if(outputArray==NULL)
      outputArray=(void *)malloc(nx*ny*nz*nt*nu*nv*nw*sizeof(DTYPE));

   // Parse the cmd to check which axis have to be flipped
   char *axisName=(char *)"x\0y\0z\0t\0u\0v\0w\0";
   int increment[7]= {1,1,1,1,1,1,1};
   int start[7]= {0,0,0,0,0,0,0};
   int end[7]= {nx,ny,nz,nt,nu,nv,nw};
   for(int i=0; i<7; ++i)
   {
      if(cmd.find(axisName[i*2])!=std::string::npos)
      {
         increment[i]=-1;
         start[i]=end[i]-1;
      }
   }

   // Define the reading and writting pointers
   DTYPE *inputPtr=static_cast<DTYPE *>(inputArray);
   DTYPE *outputPtr=static_cast<DTYPE *>(outputArray);

   // Copy the data and flipp axis if required
   for(int w=0, w2=start[6]; w<nw; ++w, w2+=increment[6])
   {
      size_t index_w=w2*nx*ny*nz*nt*nu*nv;
      for(int v=0, v2=start[5]; v<nv; ++v, v2+=increment[5])
      {
         size_t index_v=index_w + v2*nx*ny*nz*nt*nu;
         for(int u=0, u2=start[4]; u<nu; ++u, u2+=increment[4])
         {
            size_t index_u=index_v + u2*nx*ny*nz*nt;
            for(int t=0, t2=start[3]; t<nt; ++t, t2+=increment[3])
            {
               size_t index_t=index_u + t2*nx*ny*nz;
               for(int z=0, z2=start[2]; z<nz; ++z, z2+=increment[2])
               {
                  size_t index_z=index_t + z2*nx*ny;
                  for(int y=0, y2=start[1]; y<ny; ++y, y2+=increment[1])
                  {
                     size_t index_y=index_z + y2*nx;
                     for(int x=0, x2=start[0]; x<nx; ++x, x2+=increment[0])
                     {
                        size_t index=index_y + x2;
                        *outputPtr++ = inputPtr[index];
                     }
                  }
               }
            }
         }
      }
   }
   return;
}
/* *************************************************************** */
void reg_flippAxis(nifti_image *image,
                   void *outputArray,
                   std::string cmd
                   )
{
   // Check the image data type
   switch(image->datatype)
   {
   case NIFTI_TYPE_UINT8:
      reg_flippAxis_type<unsigned char>
            (image->nx, image->ny, image->nz, image->nt, image->nu, image->nv, image->nw,
             image->data, outputArray, cmd);
      break;
   case NIFTI_TYPE_INT8:
      reg_flippAxis_type<char>
            (image->nx, image->ny, image->nz, image->nt, image->nu, image->nv, image->nw,
             image->data, outputArray, cmd);
      break;
   case NIFTI_TYPE_UINT16:
      reg_flippAxis_type<unsigned short>
            (image->nx, image->ny, image->nz, image->nt, image->nu, image->nv, image->nw,
             image->data, outputArray, cmd);
      break;
   case NIFTI_TYPE_INT16:
      reg_flippAxis_type<short>
            (image->nx, image->ny, image->nz, image->nt, image->nu, image->nv, image->nw,
             image->data, outputArray, cmd);
      break;
   case NIFTI_TYPE_UINT32:
      reg_flippAxis_type<unsigned int>
            (image->nx, image->ny, image->nz, image->nt, image->nu, image->nv, image->nw,
             image->data, outputArray, cmd);
      break;
   case NIFTI_TYPE_INT32:
      reg_flippAxis_type<int>
            (image->nx, image->ny, image->nz, image->nt, image->nu, image->nv, image->nw,
             image->data, outputArray, cmd);
      break;
   case NIFTI_TYPE_FLOAT32:
      reg_flippAxis_type<float>
            (image->nx, image->ny, image->nz, image->nt, image->nu, image->nv, image->nw,
             image->data, outputArray, cmd);
      break;
   case NIFTI_TYPE_FLOAT64:
      reg_flippAxis_type<double>
            (image->nx, image->ny, image->nz, image->nt, image->nu, image->nv, image->nw,
             image->data, outputArray, cmd);
      break;
   default:
      reg_print_fct_error("reg_flippAxis");
      reg_print_msg_error("The image data type is not supported");
      reg_exit();
   }
   return;
}
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
void reg_getDisplacementFromDeformation_2D(nifti_image *field)
{
   DTYPE *ptrX = static_cast<DTYPE *>(field->data);
   DTYPE *ptrY = &ptrX[field->nx*field->ny];

   mat44 matrix;
   if(field->sform_code>0)
      matrix=field->sto_xyz;
   else matrix=field->qto_xyz;

   int x, y,  index;
   DTYPE xInit, yInit;
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(field, matrix, ptrX, ptrY) \
   private(x, y, index, xInit, yInit)
#endif
   for(y=0; y<field->ny; y++)
   {
      index=y*field->nx;
      for(x=0; x<field->nx; x++)
      {

         // Get the initial control point position
         xInit = matrix.m[0][0]*(DTYPE)x
               + matrix.m[0][1]*(DTYPE)y
               + matrix.m[0][3];
         yInit = matrix.m[1][0]*(DTYPE)x
               + matrix.m[1][1]*(DTYPE)y
               + matrix.m[1][3];

         // The initial position is subtracted from every values
         ptrX[index] -= xInit;
         ptrY[index] -= yInit;
         index++;
      }
   }
}
/* *************************************************************** */
template<class DTYPE>
void reg_getDisplacementFromDeformation_3D(nifti_image *field)
{
   DTYPE *ptrX = static_cast<DTYPE *>(field->data);
   DTYPE *ptrY = &ptrX[field->nx*field->ny*field->nz];
   DTYPE *ptrZ = &ptrY[field->nx*field->ny*field->nz];

   mat44 matrix;
   if(field->sform_code>0)
      matrix=field->sto_xyz;
   else matrix=field->qto_xyz;

   int x, y, z, index;
   float xInit, yInit, zInit;
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(field, matrix, \
   ptrX, ptrY, ptrZ) \
   private(x, y, z, index, xInit, yInit, zInit)
#endif
   for(z=0; z<field->nz; z++)
   {
      index=z*field->nx*field->ny;
      for(y=0; y<field->ny; y++)
      {
         for(x=0; x<field->nx; x++)
         {
            // Get the initial control point position
            xInit = matrix.m[0][0]*static_cast<float>(x)
                  + matrix.m[0][1]*static_cast<float>(y)
                  + matrix.m[0][2]*static_cast<float>(z)
                  + matrix.m[0][3];
            yInit = matrix.m[1][0]*static_cast<float>(x)
                  + matrix.m[1][1]*static_cast<float>(y)
                  + matrix.m[1][2]*static_cast<float>(z)
                  + matrix.m[1][3];
            zInit = matrix.m[2][0]*static_cast<float>(x)
                  + matrix.m[2][1]*static_cast<float>(y)
                  + matrix.m[2][2]*static_cast<float>(z)
                  + matrix.m[2][3];

            // The initial position is subtracted from every values
            ptrX[index] -= static_cast<DTYPE>(xInit);
            ptrY[index] -= static_cast<DTYPE>(yInit);
            ptrZ[index] -= static_cast<DTYPE>(zInit);
            index++;
         }
      }
   }
}
/* *************************************************************** */
int reg_getDisplacementFromDeformation(nifti_image *field)
{
   if(field->datatype==NIFTI_TYPE_FLOAT32)
   {
      switch(field->nu)
      {
      case 2:
         reg_getDisplacementFromDeformation_2D<float>(field);
         break;
      case 3:
         reg_getDisplacementFromDeformation_3D<float>(field);
         break;
      default:
         reg_print_fct_error("reg_getDisplacementFromDeformation");
         reg_print_msg_error("Only implemented for 5D image with 2 or 3 components in the fifth dimension");
         reg_exit();
      }
   }
   else if(field->datatype==NIFTI_TYPE_FLOAT64)
   {
      switch(field->nu)
      {
      case 2:
         reg_getDisplacementFromDeformation_2D<double>(field);
         break;
      case 3:
         reg_getDisplacementFromDeformation_3D<double>(field);
         break;
      default:
         reg_print_fct_error("reg_getDisplacementFromDeformation");
         reg_print_msg_error("Only implemented for 5D image with 2 or 3 components in the fifth dimension");
         reg_exit();
      }
   }
   else
   {
      reg_print_fct_error("reg_getDisplacementFromDeformation");
      reg_print_msg_error("Only single or double floating precision have been implemented");
      reg_exit();
   }
   field->intent_code=NIFTI_INTENT_VECTOR;
   memset(field->intent_name, 0, 16);
   strcpy(field->intent_name,"NREG_TRANS");
   if(field->intent_p1==DEF_FIELD)
      field->intent_p1=DISP_FIELD;
   if(field->intent_p1==DEF_VEL_FIELD)
      field->intent_p1=DISP_VEL_FIELD;
   return EXIT_SUCCESS;
}
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
void reg_getDeformationFromDisplacement_2D(nifti_image *field)
{
   DTYPE *ptrX = static_cast<DTYPE *>(field->data);
   DTYPE *ptrY = &ptrX[field->nx*field->ny];

   mat44 matrix;
   if(field->sform_code>0)
      matrix=field->sto_xyz;
   else matrix=field->qto_xyz;

   int x, y, index;
   DTYPE xInit, yInit;
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(field, matrix, \
   ptrX, ptrY) \
   private(x, y, index, xInit, yInit)
#endif
   for(y=0; y<field->ny; y++)
   {
      index=y*field->nx;
      for(x=0; x<field->nx; x++)
      {

         // Get the initial control point position
         xInit = matrix.m[0][0]*(DTYPE)x
               + matrix.m[0][1]*(DTYPE)y
               + matrix.m[0][3];
         yInit = matrix.m[1][0]*(DTYPE)x
               + matrix.m[1][1]*(DTYPE)y
               + matrix.m[1][3];

         // The initial position is added from every values
         ptrX[index] += xInit;
         ptrY[index] += yInit;
         index++;
      }
   }
}
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
void reg_getDeformationFromDisplacement_3D(nifti_image *field)
{
   DTYPE *ptrX = static_cast<DTYPE *>(field->data);
   DTYPE *ptrY = &ptrX[field->nx*field->ny*field->nz];
   DTYPE *ptrZ = &ptrY[field->nx*field->ny*field->nz];

   mat44 matrix;
   if(field->sform_code>0)
      matrix=field->sto_xyz;
   else matrix=field->qto_xyz;

   int x, y, z, index;
   float xInit, yInit, zInit;
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(field, matrix, ptrX, ptrY, ptrZ) \
   private(x, y, z, index, xInit, yInit, zInit)
#endif
   for(z=0; z<field->nz; z++)
   {
      index=z*field->nx*field->ny;
      for(y=0; y<field->ny; y++)
      {
         for(x=0; x<field->nx; x++)
         {

            // Get the initial control point position
            xInit = matrix.m[0][0]*static_cast<float>(x)
                  + matrix.m[0][1]*static_cast<float>(y)
                  + matrix.m[0][2]*static_cast<float>(z)
                  + matrix.m[0][3];
            yInit = matrix.m[1][0]*static_cast<float>(x)
                  + matrix.m[1][1]*static_cast<float>(y)
                  + matrix.m[1][2]*static_cast<float>(z)
                  + matrix.m[1][3];
            zInit = matrix.m[2][0]*static_cast<float>(x)
                  + matrix.m[2][1]*static_cast<float>(y)
                  + matrix.m[2][2]*static_cast<float>(z)
                  + matrix.m[2][3];

            // The initial position is subtracted from every values
            ptrX[index] += static_cast<DTYPE>(xInit);
            ptrY[index] += static_cast<DTYPE>(yInit);
            ptrZ[index] += static_cast<DTYPE>(zInit);
            index++;
         }
      }
   }
}
/* *************************************************************** */
/* *************************************************************** */
int reg_getDeformationFromDisplacement(nifti_image *field)
{
   if(field->datatype==NIFTI_TYPE_FLOAT32)
   {
      switch(field->nu)
      {
      case 2:
         reg_getDeformationFromDisplacement_2D<float>(field);
         break;
      case 3:
         reg_getDeformationFromDisplacement_3D<float>(field);
         break;
      default:
         reg_print_fct_error("reg_getDeformationFromDisplacement");
         reg_print_msg_error("Only implemented for 2 or 3D deformation fields");
         reg_exit();
      }
   }
   else if(field->datatype==NIFTI_TYPE_FLOAT64)
   {
      switch(field->nu)
      {
      case 2:
         reg_getDeformationFromDisplacement_2D<double>(field);
         break;
      case 3:
         reg_getDeformationFromDisplacement_3D<double>(field);
         break;
      default:
         reg_print_fct_error("reg_getDeformationFromDisplacement");
         reg_print_msg_error("Only implemented for 2 or 3D deformation fields");
         reg_exit();
      }
   }
   else
   {
      reg_print_fct_error("reg_getDeformationFromDisplacement");
      reg_print_msg_error("Only single or double floating precision have been implemented");
      reg_exit();
   }

   field->intent_code=NIFTI_INTENT_VECTOR;
   memset(field->intent_name, 0, 16);
   strcpy(field->intent_name,"NREG_TRANS");
   if(field->intent_p1==DISP_FIELD)
      field->intent_p1=DEF_FIELD;
   if(field->intent_p1==DISP_VEL_FIELD)
      field->intent_p1=DEF_VEL_FIELD;
   return EXIT_SUCCESS;
}
/* *************************************************************** */
template <class DTYPE>
void reg_setGradientToZero_core(nifti_image *image,
                               bool x_axis,
                               bool y_axis,
                               bool z_axis)
{
   size_t voxel_number = (size_t)image->nx*image->ny*image->nz;
   DTYPE *ptr = static_cast<DTYPE *>(image->data);
   if(x_axis==true){
      for(size_t i=0; i<voxel_number; ++i)
         *ptr++=0;
   }
   else ptr += voxel_number;
   if(y_axis==true){
      for(size_t i=0; i<voxel_number; ++i)
         *ptr++=0;
   }
   else ptr += voxel_number;
   if(z_axis==true && image->nu>2){
      for(size_t i=0; i<voxel_number; ++i)
         *ptr++=0;
   }
   return;
}
/* *************************************************************** */
void reg_setGradientToZero(nifti_image *image,
                           bool x_axis,
                           bool y_axis,
                           bool z_axis=false)
{
   // Ensure that the specified image is a 5D image
   if(image->ndim != 5)
   {
      reg_print_fct_error("reg_setGradientToZero");
      reg_print_msg_error("Input image is expected to be a 5D image");
      reg_exit();
   }
   switch(image->datatype){
   case NIFTI_TYPE_FLOAT32:
      reg_setGradientToZero_core<float>(image, x_axis, y_axis, z_axis);
      break;
   case NIFTI_TYPE_FLOAT64:
      reg_setGradientToZero_core<double>(image, x_axis, y_axis, z_axis);
      break;
   default:
      reg_print_fct_error("reg_setGradientToZero");
      reg_print_msg_error("Input image is expected to be float or double");
      reg_exit();
   }
   return;
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
double reg_test_compare_arrays(DTYPE *ptrA,
                              DTYPE *ptrB,
                              size_t nvox)
{
   double maxDifference=0.0;

   for(size_t i=0; i<nvox; ++i)
   {
      double valA=(double)ptrA[i];
      double valB=(double)ptrB[i];
      if(valA!=valA || valB!=valB)
      {
         if(valA==valA || valB==valB)
         {
            reg_print_fct_warn("reg_test_compare_arrays");
            reg_print_msg_warn("Unexpected NaN in only one of the array");
            return std::numeric_limits<float>::max();
         }
      }
      else
      {
         if(valA!=0 && valB!=0)
         {
            double diffRatio=valA/valB;
            if(diffRatio<0)
            {
               diffRatio=std::abs(valA-valB);
               maxDifference=maxDifference>diffRatio?maxDifference:diffRatio;
            }
            diffRatio-=1.0;
            maxDifference=maxDifference>diffRatio?maxDifference:diffRatio;
         }
         else
         {
            double diffRatio=std::abs(valA-valB);
            maxDifference=maxDifference>diffRatio?maxDifference:diffRatio;
         }
      }
   }
   return maxDifference;
}
template double reg_test_compare_arrays<float>(float *ptrA, float *ptrB, size_t nvox);
template double reg_test_compare_arrays<double>(double *ptrA, double *ptrB, size_t nvox);
/* *************************************************************** */
template <class DTYPE>
double reg_test_compare_images1(nifti_image *imgA,
                               nifti_image *imgB)
{
   DTYPE *imgAPtr = static_cast<DTYPE *>(imgA->data);
   DTYPE *imgBPtr = static_cast<DTYPE *>(imgB->data);
   return reg_test_compare_arrays<DTYPE>(imgAPtr,imgBPtr,imgA->nvox);
}
/* *************************************************************** */
double reg_test_compare_images(nifti_image *imgA,
                              nifti_image *imgB)
{
   if(imgA->datatype!=imgB->datatype)
   {
      reg_print_fct_error("reg_test_compare_images");
      reg_print_msg_error("Input images have different datatype");
      reg_exit();
   }
   if(imgA->nvox!=imgB->nvox)
   {
      reg_print_fct_error("reg_test_compare_images");
      reg_print_msg_error("Input images have different size");
      reg_exit();
   }
   switch(imgA->datatype)
   {
   case NIFTI_TYPE_UINT8:
      return reg_test_compare_images1<unsigned char>(imgA,imgB);
   case NIFTI_TYPE_UINT16:
      return reg_test_compare_images1<unsigned short>(imgA,imgB);
   case NIFTI_TYPE_UINT32:
      return reg_test_compare_images1<unsigned int>(imgA,imgB);
   case NIFTI_TYPE_INT8:
      return reg_test_compare_images1<char>(imgA,imgB);
   case NIFTI_TYPE_INT16:
      return reg_test_compare_images1<short>(imgA,imgB);
   case NIFTI_TYPE_INT32:
      return reg_test_compare_images1<int>(imgA,imgB);
   case NIFTI_TYPE_FLOAT32:
      return reg_test_compare_images1<float>(imgA,imgB);
   case NIFTI_TYPE_FLOAT64:
      return reg_test_compare_images1<double>(imgA,imgB);
   default:
      reg_print_fct_error("reg_test_compare_images");
      reg_print_msg_error("Unsupported data type");
      reg_exit();
   }
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_tools_abs_image1(nifti_image *img)
{
   DTYPE *ptr = static_cast<DTYPE *>(img->data);
   for(size_t i=0; i<img->nvox; ++i)
      ptr[i]=static_cast<DTYPE>(fabs(static_cast<double>(ptr[i])));
}
/* *************************************************************** */
void reg_tools_abs_image(nifti_image *img)
{
   switch(img->datatype)
   {
   case NIFTI_TYPE_UINT8:
      reg_tools_abs_image1<unsigned char>(img);
      break;
   case NIFTI_TYPE_UINT16:
      reg_tools_abs_image1<unsigned short>(img);
      break;
   case NIFTI_TYPE_UINT32:
      reg_tools_abs_image1<unsigned int>(img);
      break;
   case NIFTI_TYPE_INT8:
      reg_tools_abs_image1<char>(img);
      break;
   case NIFTI_TYPE_INT16:
      reg_tools_abs_image1<short>(img);
      break;
   case NIFTI_TYPE_INT32:
      reg_tools_abs_image1<int>(img);
      break;
   case NIFTI_TYPE_FLOAT32:
      reg_tools_abs_image1<float>(img);
      break;
   case NIFTI_TYPE_FLOAT64:
      reg_tools_abs_image1<double>(img);
      break;
   default:
      reg_print_fct_error("reg_tools_abs_image");
      reg_print_msg_error("Unsupported data type");
      reg_exit();
   }
}
/* *************************************************************** */
void mat44ToCptr(mat44 mat, float* cMat)
{
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			cMat[i * 4 + j] = mat.m[i][j];
		}
	}
}
/* *************************************************************** */
void cPtrToMat44(mat44 *mat, float* cMat)
{
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			 mat->m[i][j]=cMat[i * 4 + j];
		}
	}
}
/* *************************************************************** */
void mat33ToCptr(mat33 *mat, float* cMat, const unsigned int numMats)
{
	for (size_t k = 0; k < numMats; k++)
	{
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				cMat[9*k +i * 3 + j] = mat[k].m[i][j];

			}
		}
	}
}
/* *************************************************************** */
void cPtrToMat33(mat33 *mat, float* cMat)
{
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
             mat->m[i][j]=cMat[i * 3 + j];
        }
    }
}
/* *************************************************************** */
template<typename T>
void matmnToCptr(T** mat, T* cMat, unsigned int m, unsigned int n) {
    for (unsigned int i = 0; i < m; i++)
    {
        for (unsigned int j = 0; j < n; j++)
        {
            cMat[i * n + j] = mat[i][j];
        }
    }
}
template void matmnToCptr<float>(float** mat, float* cMat, unsigned int m, unsigned int n);
template void matmnToCptr<double>(double** mat, double* cMat, unsigned int m, unsigned int n);
/* *************************************************************** */
template<typename T>
void cPtrToMatmn(T** mat, T* cMat, unsigned int m, unsigned int n) {
    for (unsigned int i = 0; i < m; i++)
    {
        for (unsigned int j = 0; j < n; j++)
        {
             mat[i][j]=cMat[i * n + j];
        }
    }
}
template void cPtrToMatmn<float>(float** mat, float* cMat, unsigned int m, unsigned int n);
template void cPtrToMatmn<double>(double** mat, double* cMat, unsigned int m, unsigned int n);
/* *************************************************************** */
void coordinateFromLinearIndex(int index, int maxValue_x, int maxValue_y, int &x, int &y, int &z)
{
    x =  index % (maxValue_x+1);
    index /= (maxValue_x+1);
    y = index % (maxValue_y+1);
    index /= (maxValue_y+1);
    z = index;
}
/* *************************************************************** */
#endif
