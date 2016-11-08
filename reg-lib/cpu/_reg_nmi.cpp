/*
 *  _reg_mutualinformation.cpp
 *
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_NMI_CPP
#define _REG_NMI_CPP

#include "_reg_nmi.h"

/* *************************************************************** */
/* *************************************************************** */
reg_nmi::reg_nmi()
   : reg_measure()
{
   this->forwardJointHistogramPro=NULL;
   this->forwardJointHistogramLog=NULL;
   this->forwardEntropyValues=NULL;
   this->backwardJointHistogramPro=NULL;
   this->backwardJointHistogramLog=NULL;
   this->backwardEntropyValues=NULL;

   for(int i=0; i<255; ++i)
   {
      this->referenceBinNumber[i]=68;
      this->floatingBinNumber[i]=68;
   }
#ifndef NDEBUG
   reg_print_msg_debug("reg_nmi constructor called");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
reg_nmi::~reg_nmi()
{
   this->ClearHistogram();
#ifndef NDEBUG
   reg_print_msg_debug("reg_nmi destructor called");
#endif
}
/* *************************************************************** */
void reg_nmi::ClearHistogram()
{
   int timepoint=this->referenceTimePoint;
   // Free the joint histograms and the entropy arrays
   if(this->forwardJointHistogramPro!=NULL)
   {
      for(int i=0; i<timepoint; ++i)
      {
         if(this->forwardJointHistogramPro[i]!=NULL)
            free(this->forwardJointHistogramPro[i]);
         this->forwardJointHistogramPro[i]=NULL;
      }
      free(this->forwardJointHistogramPro);
   }
   this->forwardJointHistogramPro=NULL;
   if(this->backwardJointHistogramPro!=NULL)
   {
      for(int i=0; i<timepoint; ++i)
      {
         if(this->backwardJointHistogramPro[i]!=NULL)
            free(this->backwardJointHistogramPro[i]);
         this->backwardJointHistogramPro[i]=NULL;
      }
      free(this->backwardJointHistogramPro);
   }
   this->backwardJointHistogramPro=NULL;

   if(this->forwardJointHistogramLog!=NULL)
   {
      for(int i=0; i<timepoint; ++i)
      {
         if(this->forwardJointHistogramLog[i]!=NULL)
            free(this->forwardJointHistogramLog[i]);
         this->forwardJointHistogramLog[i]=NULL;
      }
      free(this->forwardJointHistogramLog);
   }
   this->forwardJointHistogramLog=NULL;
   if(this->backwardJointHistogramLog!=NULL)
   {
      for(int i=0; i<timepoint; ++i)
      {
         if(this->backwardJointHistogramLog[i]!=NULL)
            free(this->backwardJointHistogramLog[i]);
         this->backwardJointHistogramLog[i]=NULL;
      }
      free(this->backwardJointHistogramLog);
   }
   this->backwardJointHistogramLog=NULL;

   if(this->forwardEntropyValues!=NULL)
   {
      for(int i=0; i<timepoint; ++i)
      {
         if(this->forwardEntropyValues[i]!=NULL)
            free(this->forwardEntropyValues[i]);
         this->forwardEntropyValues[i]=NULL;
      }
      free(this->forwardEntropyValues);
   }
   this->forwardEntropyValues=NULL;
   if(this->backwardEntropyValues!=NULL)
   {
      for(int i=0; i<timepoint; ++i)
      {
         if(this->backwardEntropyValues[i]!=NULL)
            free(this->backwardEntropyValues[i]);
         this->backwardEntropyValues[i]=NULL;
      }
      free(this->backwardEntropyValues);
   }
   this->backwardEntropyValues=NULL;
#ifndef NDEBUG
   reg_print_msg_debug("reg_nmi::ClearHistogram called");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
void reg_nmi::InitialiseMeasure(nifti_image *refImgPtr,
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
   reg_measure::InitialiseMeasure(refImgPtr,
                                  floImgPtr,
                                  maskRefPtr,
                                  warFloImgPtr,
                                  warFloGraPtr,
                                  forVoxBasedGraPtr,
                                  maskFloPtr,
                                  warRefImgPtr,
                                  warRefGraPtr,
                                  bckVoxBasedGraPtr);

   // Clear all allocated arrays
   this->ClearHistogram();
   // Extract the number of time point
   int timepoint=this->referenceTimePoint;
   // Reference and floating are resampled between 2 and bin-3
   for(int i=0; i<timepoint; ++i)
   {
      if(this->activeTimePoint[i])
      {
         reg_intensityRescale(this->referenceImagePointer,
                              i,
                              2.f,
                              this->referenceBinNumber[i]-3);
         reg_intensityRescale(this->floatingImagePointer,
                              i,
                              2.f,
                              this->floatingBinNumber[i]-3);
      }
   }
   // Create the joint histograms
   this->forwardJointHistogramPro=(double**)malloc(255*sizeof(double *));
   this->forwardJointHistogramLog=(double**)malloc(255*sizeof(double *));
   this->forwardEntropyValues=(double**)malloc(255*sizeof(double *));
   if(this->isSymmetric)
   {
      this->backwardJointHistogramPro=(double**)malloc(255*sizeof(double *));
      this->backwardJointHistogramLog=(double**)malloc(255*sizeof(double *));
      this->backwardEntropyValues=(double**)malloc(255*sizeof(double *));
   }
   for(int i=0; i<timepoint; ++i)
   {
      if(this->activeTimePoint[i])
      {
         // Compute the total number of bin
         this->totalBinNumber[i]=this->referenceBinNumber[i]*this->floatingBinNumber[i] +
               this->referenceBinNumber[i] + this->floatingBinNumber[i];
         this->forwardJointHistogramLog[i]=(double *)
               calloc(this->totalBinNumber[i],sizeof(double));
         this->forwardJointHistogramPro[i]=(double *)
               calloc(this->totalBinNumber[i],sizeof(double));
         this->forwardEntropyValues[i]=(double *)
               calloc(4,sizeof(double));
         if(this->isSymmetric)
         {
            this->backwardJointHistogramLog[i]=(double *)
                  calloc(this->totalBinNumber[i],sizeof(double));
            this->backwardJointHistogramPro[i]=(double *)
                  calloc(this->totalBinNumber[i],sizeof(double));
            this->backwardEntropyValues[i]=(double *)
                  calloc(4,sizeof(double));
         }
      }
      else
      {
         this->forwardJointHistogramLog[i]=NULL;
         this->forwardJointHistogramPro[i]=NULL;
         this->forwardEntropyValues[i]=NULL;
         if(this->isSymmetric)
         {
            this->backwardJointHistogramLog[i]=NULL;
            this->backwardJointHistogramPro[i]=NULL;
            this->backwardEntropyValues[i]=NULL;
         }
      }
   }
#ifndef NDEBUG
   char text[255];
   reg_print_msg_debug("reg_nmi::InitialiseMeasure().");
   sprintf(text, "Active time point:");
   for(int i=0; i<this->referenceImagePointer->nt; ++i)
      if(this->activeTimePoint[i])
         sprintf(text, "%s %i", text, i);
   reg_print_msg_debug(text);
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template<class PrecisionTYPE>
PrecisionTYPE GetBasisSplineValue(PrecisionTYPE x)
{
   x=fabs(x);
   PrecisionTYPE value=0.0;
   if(x<2.0)
   {
      if(x<1.0)
         value = (PrecisionTYPE)(2.0f/3.0f + (0.5f*x-1.0)*x*x);
      else
      {
         x-=2.0f;
         value = -x*x*x/6.0f;
      }
   }
   return value;
}
/* *************************************************************** */
template<class PrecisionTYPE>
PrecisionTYPE GetBasisSplineDerivativeValue(PrecisionTYPE ori)
{
   PrecisionTYPE x=fabs(ori);
   PrecisionTYPE value=0.0;
   if(x<2.0)
   {
      if(x<1.0)
         value = (PrecisionTYPE)((1.5f*x-2.0)*ori);
      else
      {
         x-=2.0f;
         value = -0.5f * x * x;
         if(ori<0.0f) value =-value;
      }
   }
   return value;
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_getNMIValue(nifti_image *referenceImage,
                     nifti_image *warpedImage,
                     bool *activeTimePoint,
                     unsigned short *referenceBinNumber,
                     unsigned short *floatingBinNumber,
                     unsigned short *totalBinNumber,
                     double **jointHistogramLog,
                     double **jointhistogramPro,
                     double **entropyValues,
                     int *referenceMask
                     )
{
   // Create pointers to the image data arrays
   DTYPE *refImagePtr = static_cast<DTYPE *>(referenceImage->data);
   DTYPE *warImagePtr = static_cast<DTYPE *>(warpedImage->data);
   // Useful variable
   size_t voxelNumber = (size_t)referenceImage->nx *
         referenceImage->ny *
         referenceImage->nz;
   // Iterate over all active time points
   for(int t=0; t<referenceImage->nt; ++t)
   {
      if(activeTimePoint[t])
      {
#ifndef NDEBUG
         char text[255];
         sprintf(text, "Computing NMI for time point %i",t);
         reg_print_msg_debug(text);
#endif
         // Define some pointers to the current histograms
         double *jointHistoProPtr = jointhistogramPro[t];
         double *jointHistoLogPtr = jointHistogramLog[t];
         // Empty the joint histogram
         memset(jointHistoProPtr,0,totalBinNumber[t]*sizeof(double));
         // Fill the joint histograms using an approximation
         DTYPE *refPtr = &refImagePtr[t*voxelNumber];
         DTYPE *warPtr = &warImagePtr[t*voxelNumber];
         for(size_t voxel=0; voxel<voxelNumber; ++voxel)
         {
            if(referenceMask[voxel]>-1)
            {
               DTYPE refValue=refPtr[voxel];
               DTYPE warValue=warPtr[voxel];
               if(refValue==refValue && warValue==warValue &&
                     refValue>=0 && warValue>=0 &&
                     refValue<referenceBinNumber[t] &&
                     warValue<floatingBinNumber[t])
               {
                  ++jointHistoProPtr[static_cast<int>(refValue) +
                        static_cast<int>(warValue) * referenceBinNumber[t]];
               }
            }
         }
         // Convolve the histogram with a cubic B-spline kernel
         double kernel[3];
         kernel[0]=kernel[2]=GetBasisSplineValue(-1.);
         kernel[1]=GetBasisSplineValue(0.);
         // Histogram is first smooth along the reference axis
         memset(jointHistoLogPtr,0,totalBinNumber[t]*sizeof(double));
         for(int f=0; f<floatingBinNumber[t]; ++f)
         {
            for(int r=0; r<referenceBinNumber[t]; ++r)
            {
               double value=0.0;
               int index = r-1;
               double *ptrHisto = &jointHistoProPtr[index+referenceBinNumber[t]*f];

               for(int it=0; it<3; it++)
               {
                  if(-1<index && index<referenceBinNumber[t])
                  {
                     value += *ptrHisto * kernel[it];
                  }
                  ++ptrHisto;
                  ++index;
               }
               jointHistoLogPtr[r+referenceBinNumber[t]*f] = value;
            }
         }
         // Histogram is then smooth along the warped floating axis
         for(int r=0; r<referenceBinNumber[t]; ++r)
         {
            for(int f=0; f<floatingBinNumber[t]; ++f)
            {
               double value=0.;
               int index = f-1;
               double *ptrHisto = &jointHistoLogPtr[r+referenceBinNumber[t]*index];

               for(int it=0; it<3; it++)
               {
                  if(-1<index && index<floatingBinNumber[t])
                  {
                     value += *ptrHisto * kernel[it];
                  }
                  ptrHisto+=referenceBinNumber[t];
                  ++index;
               }
               jointHistoProPtr[r+referenceBinNumber[t]*f] = value;
            }
         }
         // Normalise the histogram
         double activeVoxel=0.f;
         for(int i=0; i<totalBinNumber[t]; ++i)
            activeVoxel+=jointHistoProPtr[i];
         entropyValues[t][3]=activeVoxel;
         for(int i=0; i<totalBinNumber[t]; ++i)
            jointHistoProPtr[i]/=activeVoxel;
         // Marginalise over the reference axis
         for(int r=0; r<referenceBinNumber[t]; ++r)
         {
            double sum=0.;
            int index=r;
            for(int f=0; f<floatingBinNumber[t]; ++f)
            {
               sum+=jointHistoProPtr[index];
               index+=referenceBinNumber[t];
            }
            jointHistoProPtr[referenceBinNumber[t]*
                  floatingBinNumber[t]+r]=sum;
         }
         // Marginalise over the warped floating axis
         for(int f=0; f<floatingBinNumber[t]; ++f)
         {
            double sum=0.;
            int index=referenceBinNumber[t]*f;
            for(int r=0; r<referenceBinNumber[t]; ++r)
            {
               sum+=jointHistoProPtr[index];
               ++index;
            }
            jointHistoProPtr[referenceBinNumber[t]*
                  floatingBinNumber[t]+referenceBinNumber[t]+f]=sum;
         }
         // Set the log values to zero
         memset(jointHistoLogPtr,0,totalBinNumber[t]*sizeof(double));
         // Compute the entropy of the reference image
         double referenceEntropy=0.;
         for(int r=0; r<referenceBinNumber[t]; ++r)
         {
            double valPro=jointHistoProPtr[referenceBinNumber[t]*floatingBinNumber[t]+r];
            if(valPro>0)
            {
               double valLog=log(valPro);
               referenceEntropy -= valPro * valLog;
               jointHistoLogPtr[referenceBinNumber[t]*floatingBinNumber[t]+r]=valLog;
            }
         }
         entropyValues[t][0]=referenceEntropy;
         // Compute the entropy of the warped floating image
         double warpedEntropy=0.;
         for(int f=0; f<floatingBinNumber[t]; ++f)
         {
            double valPro=jointHistoProPtr[referenceBinNumber[t]*floatingBinNumber[t]+
                  referenceBinNumber[t]+f];
            if(valPro>0)
            {
               double valLog=log(valPro);
               warpedEntropy -= valPro * valLog;
               jointHistoLogPtr[referenceBinNumber[t]*floatingBinNumber[t]+
                     referenceBinNumber[t]+f]=valLog;
            }
         }
         entropyValues[t][1]=warpedEntropy;
         // Compute the joint entropy
         double jointEntropy=0.;
         for(int i=0; i<referenceBinNumber[t]*floatingBinNumber[t]; ++i)
         {
            double valPro=jointHistoProPtr[i];
            if(valPro>0)
            {
               double valLog=log(valPro);
               jointEntropy -= valPro * valLog;
               jointHistoLogPtr[i]=valLog;
            }
         }
         entropyValues[t][2]=jointEntropy;
      } // if active time point
   } // iterate over all time point in the reference image
}
/* *************************************************************** */
template void reg_getNMIValue<float>(nifti_image *,nifti_image *,bool *,unsigned short *,unsigned short *,unsigned short *,double **,double **,double **,int *);
template void reg_getNMIValue<double>(nifti_image *,nifti_image *,bool *,unsigned short *,unsigned short *,unsigned short *,double **,double **,double **,int *);
/* *************************************************************** */
/* *************************************************************** */
double reg_nmi::GetSimilarityMeasureValue()
{
   // Check that all the specified image are of the same datatype
   if(this->warpedFloatingImagePointer->datatype !=this->referenceImagePointer->datatype)
   {
      reg_print_fct_error("reg_nmi::GetSimilarityMeasureValue()");
      reg_print_msg_error("Both input images are exepected to have the same type");
      reg_exit();
   }
   switch(this->referenceImagePointer->datatype)
   {
   case NIFTI_TYPE_FLOAT32:
      reg_getNMIValue<float>
            (this->referenceImagePointer,
             this->warpedFloatingImagePointer,
             this->activeTimePoint,
             this->referenceBinNumber,
             this->floatingBinNumber,
             this->totalBinNumber,
             this->forwardJointHistogramLog,
             this->forwardJointHistogramPro,
             this->forwardEntropyValues,
             this->referenceMaskPointer
             );
      break;
   case NIFTI_TYPE_FLOAT64:
      reg_getNMIValue<double>
            (this->referenceImagePointer,
             this->warpedFloatingImagePointer,
             this->activeTimePoint,
             this->referenceBinNumber,
             this->floatingBinNumber,
             this->totalBinNumber,
             this->forwardJointHistogramLog,
             this->forwardJointHistogramPro,
             this->forwardEntropyValues,
             this->referenceMaskPointer
             );
      break;
   default:
      reg_print_fct_error("reg_nmi::GetSimilarityMeasureValue()");
      reg_print_msg_error("Unsupported datatype");
      reg_exit();
   }

   if(this->isSymmetric)
   {
      // Check that all the specified image are of the same datatype
      if(this->floatingImagePointer->datatype !=this->warpedReferenceImagePointer->datatype)
      {
         reg_print_fct_error("reg_nmi::GetSimilarityMeasureValue()");
         reg_print_msg_error("Both input images are exepected to have the same type");
         reg_exit();
      }
      switch(this->floatingImagePointer->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         reg_getNMIValue<float>
               (this->floatingImagePointer,
                this->warpedReferenceImagePointer,
                this->activeTimePoint,
                this->floatingBinNumber,
                this->referenceBinNumber,
                this->totalBinNumber,
                this->backwardJointHistogramLog,
                this->backwardJointHistogramPro,
                this->backwardEntropyValues,
                this->floatingMaskPointer
                );
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_getNMIValue<double>
               (this->floatingImagePointer,
                this->warpedReferenceImagePointer,
                this->activeTimePoint,
                this->floatingBinNumber,
                this->referenceBinNumber,
                this->totalBinNumber,
                this->backwardJointHistogramLog,
                this->backwardJointHistogramPro,
                this->backwardEntropyValues,
                this->floatingMaskPointer
                );
         break;
      default:
         reg_print_fct_error("reg_nmi::GetSimilarityMeasureValue()");
         reg_print_msg_error("Unsupported datatype");
         reg_exit();
      }
   }

   double nmi_value_forward=0.;
   double nmi_value_backward=0.;
   for(int t=0; t<this->referenceTimePoint; ++t)
   {
      if(this->activeTimePoint[t])
      {
         nmi_value_forward += (this->forwardEntropyValues[t][0] +
               this->forwardEntropyValues[t][1] ) /
               this->forwardEntropyValues[t][2];
         if(this->isSymmetric)
            nmi_value_backward += (this->backwardEntropyValues[t][0] +
                  this->backwardEntropyValues[t][1] ) /
                  this->backwardEntropyValues[t][2];
      }
   }
#ifndef NDEBUG
   reg_print_msg_debug("reg_nmi::GetSimilarityMeasureValue called");
#endif
   return nmi_value_forward+nmi_value_backward;
}
/* *************************************************************** */
template <class DTYPE>
void reg_getVoxelBasedNMIGradient2D(nifti_image *referenceImage,
                                    nifti_image *warpedImage,
                                    unsigned short *referenceBinNumber,
                                    unsigned short *floatingBinNumber,
                                    double **jointHistogramLog,
                                    double **entropyValues,
                                    nifti_image *warImgGradient,
                                    nifti_image *measureGradientImage,
                                    int *referenceMask,
                                    int current_timepoint
                                    )
{
   if(current_timepoint<0 || current_timepoint>=referenceImage->nt){
      reg_print_fct_error("reg_getVoxelBasedNMIGradient2D");
      reg_print_msg_error("The specified active timepoint is not defined in the ref/war images");
      reg_exit();
   }
   size_t voxelNumber = (size_t)referenceImage->nx*referenceImage->ny*referenceImage->nz;

   // Pointers to the image data
   DTYPE *refImagePtr = static_cast<DTYPE *>(referenceImage->data);
   DTYPE *refPtr = &refImagePtr[current_timepoint*voxelNumber];
   DTYPE *warImagePtr = static_cast<DTYPE *>(warpedImage->data);
   DTYPE *warPtr = &warImagePtr[current_timepoint*voxelNumber];

   // Pointers to the spatial gradient of the warped image
   DTYPE *warGradPtrX = static_cast<DTYPE *>(warImgGradient->data);
   DTYPE *warGradPtrY = &warGradPtrX[voxelNumber];

   // Pointers to the measure of similarity gradient
   DTYPE *measureGradPtrX = static_cast<DTYPE *>(measureGradientImage->data);
   DTYPE *measureGradPtrY = &measureGradPtrX[voxelNumber];

   // Create pointers to the current joint histogram
   double *logHistoPtr = jointHistogramLog[current_timepoint];
   double *entropyPtr = entropyValues[current_timepoint];
   double nmi = (entropyPtr[0]+entropyPtr[1])/entropyPtr[2];
   size_t referenceOffset=referenceBinNumber[current_timepoint]*floatingBinNumber[current_timepoint];
   size_t floatingOffset=referenceOffset+referenceBinNumber[current_timepoint];
   // Iterate over all voxel
   for(size_t i=0; i<voxelNumber; ++i)
   {
      // Check if the voxel belongs to the image mask
      if(referenceMask[i]>-1)
      {
         DTYPE refValue = refPtr[i];
         DTYPE warValue = warPtr[i];
         if(refValue==refValue && warValue==warValue)
         {
            DTYPE gradX = warGradPtrX[i];
            DTYPE gradY = warGradPtrY[i];

            double jointDeriv[2]= {0.};
            double refDeriv[2]= {0.};
            double warDeriv[2]= {0.};

            for(int r=(int)(refValue-1.0); r<(int)(refValue+3.0); ++r)
            {
               if(-1<r && r<referenceBinNumber[current_timepoint])
               {
                  for(int w=(int)(warValue-1.0); w<(int)(warValue+3.0); ++w)
                  {
                     if(-1<w && w<floatingBinNumber[current_timepoint])
                     {
                        double commun =
                              GetBasisSplineValue((double)refValue - (double)r) *
                              GetBasisSplineDerivativeValue((double)warValue - (double)w);
                        double jointLog = logHistoPtr[r+w*referenceBinNumber[current_timepoint]];
                        double refLog = logHistoPtr[r+referenceOffset];
                        double warLog = logHistoPtr[w+floatingOffset];
                        if(gradX==gradX){
                           jointDeriv[0] += commun * gradX * jointLog;
                           refDeriv[0] += commun * gradX * refLog;
                           warDeriv[0] += commun * gradX * warLog;
                        }
                        if(gradY==gradY){
                           jointDeriv[1] += commun * gradY * jointLog;
                           refDeriv[1] += commun * gradY * refLog;
                           warDeriv[1] += commun * gradY * warLog;
                        }
                     }
                  }
               }
            }
            measureGradPtrX[i] += (DTYPE)((refDeriv[0] + warDeriv[0] -
                  nmi * jointDeriv[0]) / (entropyPtr[2]*entropyPtr[3]));
            measureGradPtrY[i] += (DTYPE)((refDeriv[1] + warDeriv[1] -
                  nmi * jointDeriv[1]) / (entropyPtr[2]*entropyPtr[3]));
         }// Check that the values are defined
      } // mask
   } // loop over all voxel
}
/* *************************************************************** */
template void reg_getVoxelBasedNMIGradient2D<float>
(nifti_image *,nifti_image *,unsigned short *,unsigned short *,double **,double **,nifti_image *,nifti_image *,int *, int);
template void reg_getVoxelBasedNMIGradient2D<double>
(nifti_image *,nifti_image *,unsigned short *,unsigned short *,double **,double **,nifti_image *,nifti_image *,int *, int);
/* *************************************************************** */
template <class DTYPE>
void reg_getVoxelBasedNMIGradient3D(nifti_image *referenceImage,
                                    nifti_image *warpedImage,
                                    unsigned short *referenceBinNumber,
                                    unsigned short *floatingBinNumber,
                                    double **jointHistogramLog,
                                    double **entropyValues,
                                    nifti_image *warImgGradient,
                                    nifti_image *measureGradientImage,
                                    int *referenceMask,
                                    int current_timepoint
                                    )
{
   if(current_timepoint<0 || current_timepoint>=referenceImage->nt){
      reg_print_fct_error("reg_getVoxelBasedNMIGradient3D");
      reg_print_msg_error("The specified active timepoint is not defined in the ref/war images");
      reg_exit();
   }
   //
#ifdef WIN32
   long i;
   long voxelNumber = (long)referenceImage->nx*referenceImage->ny*referenceImage->nz;
#else
   size_t i;
   size_t voxelNumber = (size_t)referenceImage->nx*referenceImage->ny*referenceImage->nz;
#endif
   // Pointers to the image data
   DTYPE *refImagePtr = static_cast<DTYPE *>(referenceImage->data);
   DTYPE *refPtr = &refImagePtr[current_timepoint*voxelNumber];
   DTYPE *warImagePtr = static_cast<DTYPE *>(warpedImage->data);
   DTYPE *warPtr = &warImagePtr[current_timepoint*voxelNumber];

   // Pointers to the spatial gradient of the warped image
   DTYPE *warGradPtrX = static_cast<DTYPE *>(warImgGradient->data);
   DTYPE *warGradPtrY = &warGradPtrX[voxelNumber];
   DTYPE *warGradPtrZ = &warGradPtrY[voxelNumber];

   // Pointers to the measure of similarity gradient
   DTYPE *measureGradPtrX = static_cast<DTYPE *>(measureGradientImage->data);
   DTYPE *measureGradPtrY = &measureGradPtrX[voxelNumber];
   DTYPE *measureGradPtrZ = &measureGradPtrY[voxelNumber];

   // Create pointers to the current joint histogram
   double *logHistoPtr = jointHistogramLog[current_timepoint];
   double *entropyPtr = entropyValues[current_timepoint];
   double nmi = (entropyPtr[0]+entropyPtr[1])/entropyPtr[2];
   size_t referenceOffset=referenceBinNumber[current_timepoint]*floatingBinNumber[current_timepoint];
   size_t floatingOffset=referenceOffset+referenceBinNumber[current_timepoint];
   int r,w;
   DTYPE refValue,warValue,gradX,gradY,gradZ;
   double jointDeriv[3],refDeriv[3],warDeriv[3],commun,jointLog,refLog,warLog;
   // Iterate over all voxel
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   private(i,r,w,refValue,warValue,gradX,gradY,gradZ, \
   jointDeriv,refDeriv,warDeriv,commun,jointLog,refLog,warLog) \
   shared(voxelNumber,referenceMask,refPtr,warPtr,referenceBinNumber,floatingBinNumber, \
   logHistoPtr,referenceOffset,floatingOffset,measureGradPtrX,measureGradPtrY,measureGradPtrZ, \
   warGradPtrX,warGradPtrY,warGradPtrZ,entropyPtr,nmi,current_timepoint)
#endif // _OPENMP
   for(i=0; i<voxelNumber; ++i)
   {
      // Check if the voxel belongs to the image mask
      if(referenceMask[i]>-1)
      {
         refValue = refPtr[i];
         warValue = warPtr[i];
         if(refValue==refValue && warValue==warValue)
         {
            gradX = warGradPtrX[i];
            gradY = warGradPtrY[i];
            gradZ = warGradPtrZ[i];

            jointDeriv[0]=jointDeriv[1]=jointDeriv[2]=0.f;
            refDeriv[0]=refDeriv[1]=refDeriv[2]=0.f;
            warDeriv[0]=warDeriv[1]=warDeriv[2]=0.f;

            for(r=(int)(refValue-1.0); r<(int)(refValue+3.0); ++r)
            {
               if(-1<r && r<referenceBinNumber[current_timepoint])
               {
                  for(w=(int)(warValue-1.0); w<(int)(warValue+3.0); ++w)
                  {
                     if(-1<w && w<floatingBinNumber[current_timepoint])
                     {
                        commun= GetBasisSplineValue((double)refValue - (double)r) *
                              GetBasisSplineDerivativeValue((double)warValue - (double)w);
                        jointLog = logHistoPtr[r+w*referenceBinNumber[current_timepoint]];
                        refLog = logHistoPtr[r+referenceOffset];
                        warLog = logHistoPtr[w+floatingOffset];
                        if(gradX==gradX){
                           refDeriv[0] += commun * gradX * refLog;
                           warDeriv[0] += commun * gradX * warLog;
                           jointDeriv[0] += commun * gradX * jointLog;
                        }
                        if(gradY==gradY){
                           refDeriv[1] += commun * gradY * refLog;
                           warDeriv[1] += commun * gradY * warLog;
                           jointDeriv[1] += commun * gradY * jointLog;
                        }
                        if(gradZ==gradZ){
                           refDeriv[2] += commun * gradZ * refLog;
                           warDeriv[2] += commun * gradZ * warLog;
                           jointDeriv[2] += commun * gradZ * jointLog;
                        }
                     }
                  }
               }
            }
            measureGradPtrX[i] += (DTYPE)((refDeriv[0] + warDeriv[0] -
                  nmi * jointDeriv[0]) / (entropyPtr[2]*entropyPtr[3]));
            measureGradPtrY[i] += (DTYPE)((refDeriv[1] + warDeriv[1] -
                  nmi * jointDeriv[1]) / (entropyPtr[2]*entropyPtr[3]));
            measureGradPtrZ[i] += (DTYPE)((refDeriv[2] + warDeriv[2] -
                  nmi * jointDeriv[2]) / (entropyPtr[2]*entropyPtr[3]));
         }// Check that the values are defined
      } // mask
   } // loop over all voxel
}
/* *************************************************************** */
template void reg_getVoxelBasedNMIGradient3D<float>
(nifti_image *,nifti_image *,unsigned short *,unsigned short *,double **,double **,nifti_image *,nifti_image *,int *, int);
template void reg_getVoxelBasedNMIGradient3D<double>
(nifti_image *,nifti_image *,unsigned short *,unsigned short *,double **,double **,nifti_image *,nifti_image *,int *, int);
/* *************************************************************** */
void reg_nmi::GetVoxelBasedSimilarityMeasureGradient(int current_timepoint)
{
   // Check if the specified time point exists and is active
   reg_measure::GetVoxelBasedSimilarityMeasureGradient(current_timepoint);
   if(this->activeTimePoint[current_timepoint]==false)
      return;

   // Check if all required input images are of the same data type
   int dtype = this->referenceImagePointer->datatype;
   if(this->warpedFloatingImagePointer->datatype != dtype ||
         this->warpedFloatingGradientImagePointer->datatype != dtype ||
         this->forwardVoxelBasedGradientImagePointer->datatype != dtype
         )
   {
      reg_print_fct_error("reg_nmi::GetVoxelBasedSimilarityMeasureGradient()");
      reg_print_msg_error("Input images are exepected to be of the same type");
      reg_exit();
   }

   // Call compute similarity measure to calculate joint histogram
   this->GetSimilarityMeasureValue();

   // Compute the gradient of the nmi for the forward transformation
   if(this->referenceImagePointer->nz>1)  // 3D input images
   {
      switch(dtype)
      {
      case NIFTI_TYPE_FLOAT32:
         reg_getVoxelBasedNMIGradient3D<float>(this->referenceImagePointer,
                                               this->warpedFloatingImagePointer,
                                               this->referenceBinNumber,
                                               this->floatingBinNumber,
                                               this->forwardJointHistogramLog,
                                               this->forwardEntropyValues,
                                               this->warpedFloatingGradientImagePointer,
                                               this->forwardVoxelBasedGradientImagePointer,
                                               this->referenceMaskPointer,
                                               current_timepoint);
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_getVoxelBasedNMIGradient3D<double>(this->referenceImagePointer,
                                                this->warpedFloatingImagePointer,
                                                this->referenceBinNumber,
                                                this->floatingBinNumber,
                                                this->forwardJointHistogramLog,
                                                this->forwardEntropyValues,
                                                this->warpedFloatingGradientImagePointer,
                                                this->forwardVoxelBasedGradientImagePointer,
                                                this->referenceMaskPointer,
                                                current_timepoint);
         break;
      default:
         reg_print_fct_error("reg_nmi::GetVoxelBasedSimilarityMeasureGradient()");
         reg_print_msg_error("Unsupported datatype");
         reg_exit();
      }
   }
   else  // 2D input images
   {
      switch(dtype)
      {
      case NIFTI_TYPE_FLOAT32:
         reg_getVoxelBasedNMIGradient2D<float>(this->referenceImagePointer,
                                               this->warpedFloatingImagePointer,
                                               this->referenceBinNumber,
                                               this->floatingBinNumber,
                                               this->forwardJointHistogramLog,
                                               this->forwardEntropyValues,
                                               this->warpedFloatingGradientImagePointer,
                                               this->forwardVoxelBasedGradientImagePointer,
                                               this->referenceMaskPointer,
                                               current_timepoint);
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_getVoxelBasedNMIGradient2D<double>(this->referenceImagePointer,
                                                this->warpedFloatingImagePointer,
                                                this->referenceBinNumber,
                                                this->floatingBinNumber,
                                                this->forwardJointHistogramLog,
                                                this->forwardEntropyValues,
                                                this->warpedFloatingGradientImagePointer,
                                                this->forwardVoxelBasedGradientImagePointer,
                                                this->referenceMaskPointer,
                                                current_timepoint);
         break;
      default:
         reg_print_fct_error("reg_nmi::GetVoxelBasedSimilarityMeasureGradient()");
         reg_print_msg_error("Unsupported datatype");
         reg_exit();
      }
   }

   if(this->isSymmetric)
   {
      dtype = this->floatingImagePointer->datatype;
      if(this->warpedReferenceImagePointer->datatype != dtype ||
            this->warpedReferenceGradientImagePointer->datatype != dtype ||
            this->backwardVoxelBasedGradientImagePointer->datatype != dtype
            )
      {
         reg_print_fct_error("reg_nmi::GetVoxelBasedSimilarityMeasureGradient()");
         reg_print_msg_error("Input images are exepected to be of the same type");
         reg_exit();
      }
      // Compute the gradient of the nmi for the backward transformation
      if(this->floatingImagePointer->nz>1)  // 3D input images
      {
         switch(dtype)
         {
         case NIFTI_TYPE_FLOAT32:
            reg_getVoxelBasedNMIGradient3D<float>(this->floatingImagePointer,
                                                  this->warpedReferenceImagePointer,
                                                  this->floatingBinNumber,
                                                  this->referenceBinNumber,
                                                  this->backwardJointHistogramLog,
                                                  this->backwardEntropyValues,
                                                  this->warpedReferenceGradientImagePointer,
                                                  this->backwardVoxelBasedGradientImagePointer,
                                                  this->floatingMaskPointer,
                                                  current_timepoint);
            break;
         case NIFTI_TYPE_FLOAT64:
            reg_getVoxelBasedNMIGradient3D<double>(this->floatingImagePointer,
                                                   this->warpedReferenceImagePointer,
                                                   this->floatingBinNumber,
                                                   this->referenceBinNumber,
                                                   this->backwardJointHistogramLog,
                                                   this->backwardEntropyValues,
                                                   this->warpedReferenceGradientImagePointer,
                                                   this->backwardVoxelBasedGradientImagePointer,
                                                   this->floatingMaskPointer,
                                                   current_timepoint);
            break;
         default:
            reg_print_fct_error("reg_nmi::GetVoxelBasedSimilarityMeasureGradient()");
            reg_print_msg_error("Unsupported datatype");
            reg_exit();
         }
      }
      else  // 2D input images
      {
         switch(dtype)
         {
         case NIFTI_TYPE_FLOAT32:
            reg_getVoxelBasedNMIGradient2D<float>(this->floatingImagePointer,
                                                  this->warpedReferenceImagePointer,
                                                  this->floatingBinNumber,
                                                  this->referenceBinNumber,
                                                  this->backwardJointHistogramLog,
                                                  this->backwardEntropyValues,
                                                  this->warpedReferenceGradientImagePointer,
                                                  this->backwardVoxelBasedGradientImagePointer,
                                                  this->floatingMaskPointer,
                                                  current_timepoint);
            break;
         case NIFTI_TYPE_FLOAT64:
            reg_getVoxelBasedNMIGradient2D<double>(this->floatingImagePointer,
                                                   this->warpedReferenceImagePointer,
                                                   this->floatingBinNumber,
                                                   this->referenceBinNumber,
                                                   this->backwardJointHistogramLog,
                                                   this->backwardEntropyValues,
                                                   this->warpedReferenceGradientImagePointer,
                                                   this->backwardVoxelBasedGradientImagePointer,
                                                   this->floatingMaskPointer,
                                                   current_timepoint);
            break;
         default:
            reg_print_fct_error("reg_nmi::GetVoxelBasedSimilarityMeasureGradient()");
            reg_print_msg_error("Unsupported datatype");
            reg_exit();
         }
      }
   }
#ifndef NDEBUG
   reg_print_msg_debug("reg_nmi::GetVoxelBasedSimilarityMeasureGradient called");
#endif
}
/* *************************************************************** */
/* *************************************************************** */

#endif // _REG_NMI
