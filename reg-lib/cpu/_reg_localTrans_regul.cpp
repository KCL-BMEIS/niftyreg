/*
 *  _reg_localTrans_regul.cpp
 *
 *
 *  Created by Marc Modat on 10/05/2011.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_localTrans_regul.h"

/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
double reg_spline_approxBendingEnergyValue2D(nifti_image *splineControlPoint)
{
   size_t nodeNumber = (size_t)splineControlPoint->nx * splineControlPoint->ny;
   int a, b, x, y, index, i;

   // Create pointers to the spline coefficients
   DTYPE *splinePtrX = static_cast<DTYPE *>(splineControlPoint->data);
   DTYPE *splinePtrY = &splinePtrX[nodeNumber];

   // get the constant basis values
   DTYPE basisXX[9], basisYY[9], basisXY[9];
   set_second_order_bspline_basis_values(basisXX, basisYY, basisXY);

   double constraintValue=0.0;

   DTYPE splineCoeffX, splineCoeffY;
   DTYPE XX_x, YY_x, XY_x;
   DTYPE XX_y, YY_y, XY_y;

#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(splineControlPoint, splinePtrX, splinePtrY, \
   basisXX, basisYY, basisXY) \
   private(XX_x, YY_x, XY_x, XX_y, YY_y, XY_y, \
   x, y, a, b, index, i, \
   splineCoeffX, splineCoeffY) \
   reduction(+:constraintValue)
#endif
   for(y=1; y<splineControlPoint->ny-1; ++y)
   {
      for(x=1; x<splineControlPoint->nx-1; ++x)
      {
         XX_x=0.0, YY_x=0.0, XY_x=0.0;
         XX_y=0.0, YY_y=0.0, XY_y=0.0;

         i=0;
         for(b=-1; b<2; b++){
            for(a=-1; a<2; a++){
               index = (y+b)*splineControlPoint->nx+x+a;
               splineCoeffX = splinePtrX[index];
               splineCoeffY = splinePtrY[index];
               XX_x += basisXX[i]*splineCoeffX;
               YY_x += basisYY[i]*splineCoeffX;
               XY_x += basisXY[i]*splineCoeffX;

               XX_y += basisXX[i]*splineCoeffY;
               YY_y += basisYY[i]*splineCoeffY;
               XY_y += basisXY[i]*splineCoeffY;
               ++i;
            }
         }

         constraintValue += double(
                  XX_x*XX_x + YY_x*YY_x + 2.0*XY_x*XY_x +
                  XX_y*XX_y + YY_y*YY_y + 2.0*XY_y*XY_y );
      }
   }
   return constraintValue / (double)splineControlPoint->nvox;
}
/* *************************************************************** */
template<class DTYPE>
double reg_spline_approxBendingEnergyValue3D(nifti_image *splineControlPoint)
{
   size_t nodeNumber = (size_t)splineControlPoint->nx *
         splineControlPoint->ny * splineControlPoint->nz;
   int a, b, c, x, y, z, index, i;

   // Create pointers to the spline coefficients
   DTYPE *splinePtrX = static_cast<DTYPE *>(splineControlPoint->data);
   DTYPE *splinePtrY = &splinePtrX[nodeNumber];
   DTYPE *splinePtrZ = &splinePtrY[nodeNumber];

   // get the constant basis values
   DTYPE basisXX[27], basisYY[27], basisZZ[27], basisXY[27], basisYZ[27], basisXZ[27];
   set_second_order_bspline_basis_values(basisXX, basisYY, basisZZ, basisXY, basisYZ, basisXZ);

   double constraintValue=0.0;

   DTYPE splineCoeffX, splineCoeffY, splineCoeffZ;
   DTYPE XX_x, YY_x, ZZ_x, XY_x, YZ_x, XZ_x;
   DTYPE XX_y, YY_y, ZZ_y, XY_y, YZ_y, XZ_y;
   DTYPE XX_z, YY_z, ZZ_z, XY_z, YZ_z, XZ_z;

#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(splineControlPoint, splinePtrX, splinePtrY, splinePtrZ, \
   basisXX, basisYY, basisZZ, basisXY, basisYZ, basisXZ) \
   private(XX_x, YY_x, ZZ_x, XY_x, YZ_x, XZ_x, XX_y, YY_y, ZZ_y, XY_y, YZ_y, XZ_y, \
   XX_z, YY_z, ZZ_z, XY_z, YZ_z, XZ_z, x, y, z, a, b, c, index, i, \
   splineCoeffX, splineCoeffY, splineCoeffZ) \
   reduction(+:constraintValue)
#endif
   for(z=1; z<splineControlPoint->nz-1; ++z)
   {
      for(y=1; y<splineControlPoint->ny-1; ++y)
      {
         for(x=1; x<splineControlPoint->nx-1; ++x)
         {
            XX_x=0.0, YY_x=0.0, ZZ_x=0.0;
            XY_x=0.0, YZ_x=0.0, XZ_x=0.0;
            XX_y=0.0, YY_y=0.0, ZZ_y=0.0;
            XY_y=0.0, YZ_y=0.0, XZ_y=0.0;
            XX_z=0.0, YY_z=0.0, ZZ_z=0.0;
            XY_z=0.0, YZ_z=0.0, XZ_z=0.0;

            i=0;
            for(c=-1; c<2; c++){
               for(b=-1; b<2; b++){
                  for(a=-1; a<2; a++){
                     index = ((z+c)*splineControlPoint->ny+y+b)*splineControlPoint->nx+x+a;
                     splineCoeffX = splinePtrX[index];
                     splineCoeffY = splinePtrY[index];
                     splineCoeffZ = splinePtrZ[index];
                     XX_x += basisXX[i]*splineCoeffX;
                     YY_x += basisYY[i]*splineCoeffX;
                     ZZ_x += basisZZ[i]*splineCoeffX;
                     XY_x += basisXY[i]*splineCoeffX;
                     YZ_x += basisYZ[i]*splineCoeffX;
                     XZ_x += basisXZ[i]*splineCoeffX;

                     XX_y += basisXX[i]*splineCoeffY;
                     YY_y += basisYY[i]*splineCoeffY;
                     ZZ_y += basisZZ[i]*splineCoeffY;
                     XY_y += basisXY[i]*splineCoeffY;
                     YZ_y += basisYZ[i]*splineCoeffY;
                     XZ_y += basisXZ[i]*splineCoeffY;

                     XX_z += basisXX[i]*splineCoeffZ;
                     YY_z += basisYY[i]*splineCoeffZ;
                     ZZ_z += basisZZ[i]*splineCoeffZ;
                     XY_z += basisXY[i]*splineCoeffZ;
                     YZ_z += basisYZ[i]*splineCoeffZ;
                     XZ_z += basisXZ[i]*splineCoeffZ;
                     ++i;
                  }
               }
            }

            constraintValue += double(
                     XX_x*XX_x + YY_x*YY_x + ZZ_x*ZZ_x + 2.0*(XY_x*XY_x + YZ_x*YZ_x + XZ_x*XZ_x) +
                     XX_y*XX_y + YY_y*YY_y + ZZ_y*ZZ_y + 2.0*(XY_y*XY_y + YZ_y*YZ_y + XZ_y*XZ_y) +
                     XX_z*XX_z + YY_z*YY_z + ZZ_z*ZZ_z + 2.0*(XY_z*XY_z + YZ_z*YZ_z + XZ_z*XZ_z) );
         }
      }
   }
   return constraintValue / (double)splineControlPoint->nvox;
}
/* *************************************************************** */
extern "C++"
double reg_spline_approxBendingEnergy(nifti_image *splineControlPoint)
{
   if(splineControlPoint->nz==1)
   {
      switch(splineControlPoint->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         return reg_spline_approxBendingEnergyValue2D<float>(splineControlPoint);
      case NIFTI_TYPE_FLOAT64:
         return reg_spline_approxBendingEnergyValue2D<double>(splineControlPoint);
      default:
         reg_print_fct_error("reg_spline_approxBendingEnergy");
         reg_print_msg_error("Only implemented for single or double precision images");
         reg_exit();
      }
   }
   else
   {
      switch(splineControlPoint->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         return reg_spline_approxBendingEnergyValue3D<float>(splineControlPoint);
      case NIFTI_TYPE_FLOAT64:
         return reg_spline_approxBendingEnergyValue3D<double>(splineControlPoint);
      default:
         reg_print_fct_error("reg_spline_approxBendingEnergy");
         reg_print_msg_error("Only implemented for single or double precision images");
         reg_exit();
      }
   }
}
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
void reg_spline_approxBendingEnergyGradient2D(nifti_image *splineControlPoint,
                                              nifti_image *gradientImage,
                                              float weight)
{
   size_t nodeNumber = (size_t)splineControlPoint->nx*splineControlPoint->ny;
   int a, b, x, y, X, Y, index, i;

   // Create pointers to the spline coefficients
   DTYPE *splinePtrX = static_cast<DTYPE *>(splineControlPoint->data);
   DTYPE *splinePtrY = &splinePtrX[nodeNumber];

   // get the constant basis values
   DTYPE basisXX[9], basisYY[9], basisXY[9];
   set_second_order_bspline_basis_values(basisXX, basisYY, basisXY);

   DTYPE splineCoeffX;
   DTYPE splineCoeffY;
   DTYPE XX_x, YY_x, XY_x;
   DTYPE XX_y, YY_y, XY_y;

   DTYPE *derivativeValues = (DTYPE *)calloc(6*nodeNumber, sizeof(DTYPE));
   DTYPE *derivativeValuesPtr;

   reg_getDisplacementFromDeformation(splineControlPoint);

   // Compute the bending energy values everywhere but at the boundary
#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(splineControlPoint,splinePtrX,splinePtrY, derivativeValues, \
   basisXX, basisYY, basisXY) \
   private(a, b, i, index, x, y, derivativeValuesPtr, splineCoeffX, splineCoeffY, \
   XX_x, YY_x, XY_x, XX_y, YY_y, XY_y)
#endif
   for(y=0; y<splineControlPoint->ny; y++)
   {
      derivativeValuesPtr = &derivativeValues[6*y*splineControlPoint->nx];
      for(x=0; x<splineControlPoint->nx; x++)
      {
         XX_x=0.0, YY_x=0.0, XY_x=0.0;
         XX_y=0.0, YY_y=0.0, XY_y=0.0;

         i=0;
         for(b=-1; b<2; b++){
            for(a=-1; a<2; a++){
               if(-1<(x+a) && -1<(y+b) && (x+a)<splineControlPoint->nx && (y+b)<splineControlPoint->ny)
               {
                  index = (y+b)*splineControlPoint->nx+x+a;
                  splineCoeffX = splinePtrX[index];
                  splineCoeffY = splinePtrY[index];
                  XX_x += basisXX[i]*splineCoeffX;
                  YY_x += basisYY[i]*splineCoeffX;
                  XY_x += basisXY[i]*splineCoeffX;

                  XX_y += basisXX[i]*splineCoeffY;
                  YY_y += basisYY[i]*splineCoeffY;
                  XY_y += basisXY[i]*splineCoeffY;
               }
               ++i;
            }
         }
         *derivativeValuesPtr++ = XX_x;
         *derivativeValuesPtr++ = XX_y;
         *derivativeValuesPtr++ = YY_x;
         *derivativeValuesPtr++ = YY_y;
         *derivativeValuesPtr++ = (DTYPE)(2.0*XY_x);
         *derivativeValuesPtr++ = (DTYPE)(2.0*XY_y);
      }
   }

   DTYPE *gradientXPtr = static_cast<DTYPE *>(gradientImage->data);
   DTYPE *gradientYPtr = &gradientXPtr[nodeNumber];

   DTYPE approxRatio = (DTYPE)weight / (DTYPE)nodeNumber;
   DTYPE gradientValue[2];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(splineControlPoint, derivativeValues, gradientXPtr, gradientYPtr, \
   basisXX, basisYY, basisXY, approxRatio) \
   private(index, a, X, Y, x, y, derivativeValuesPtr, gradientValue)
#endif
   for(y=0; y<splineControlPoint->ny; y++)
   {
      index=y*splineControlPoint->nx;
      for(x=0; x<splineControlPoint->nx; x++)
      {
         gradientValue[0]=gradientValue[1]=0.0;
         a=0;
         for(Y=y-1; Y<y+2; Y++)
         {
            for(X=x-1; X<x+2; X++)
            {
               if(-1<X && -1<Y && X<splineControlPoint->nx && Y<splineControlPoint->ny)
               {
                  derivativeValuesPtr = &derivativeValues[6 * (Y*splineControlPoint->nx + X)];
                  gradientValue[0] += (*derivativeValuesPtr++) * basisXX[a];
                  gradientValue[1] += (*derivativeValuesPtr++) * basisXX[a];

                  gradientValue[0] += (*derivativeValuesPtr++) * basisYY[a];
                  gradientValue[1] += (*derivativeValuesPtr++) * basisYY[a];

                  gradientValue[0] += (*derivativeValuesPtr++) * basisXY[a];
                  gradientValue[1] += (*derivativeValuesPtr++) * basisXY[a];
               }
               a++;
            }
         }
         gradientXPtr[index] += approxRatio*gradientValue[0];
         gradientYPtr[index] += approxRatio*gradientValue[1];
         index++;
      }
   }
   reg_getDeformationFromDisplacement(splineControlPoint);
   free(derivativeValues);
}
/* *************************************************************** */
template<class DTYPE>
void reg_spline_approxBendingEnergyGradient3D(nifti_image *splineControlPoint,
                                              nifti_image *gradientImage,
                                              float weight)
{
   size_t nodeNumber = (size_t)splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz;
   int a, b, c, x, y, z, X, Y, Z, index, i;

   // Create pointers to the spline coefficients
   DTYPE *splinePtrX = static_cast<DTYPE *>(splineControlPoint->data);
   DTYPE *splinePtrY = &splinePtrX[nodeNumber];
   DTYPE *splinePtrZ = &splinePtrY[nodeNumber];

   // get the constant basis values
   DTYPE basisXX[27], basisYY[27], basisZZ[27], basisXY[27], basisYZ[27], basisXZ[27];
   set_second_order_bspline_basis_values(basisXX, basisYY, basisZZ, basisXY, basisYZ, basisXZ);

   DTYPE splineCoeffX;
   DTYPE splineCoeffY;
   DTYPE splineCoeffZ;
   DTYPE XX_x, YY_x, ZZ_x, XY_x, YZ_x, XZ_x;
   DTYPE XX_y, YY_y, ZZ_y, XY_y, YZ_y, XZ_y;
   DTYPE XX_z, YY_z, ZZ_z, XY_z, YZ_z, XZ_z;

   DTYPE *derivativeValues = (DTYPE *)calloc(18*nodeNumber, sizeof(DTYPE));
   DTYPE *derivativeValuesPtr;

   reg_getDisplacementFromDeformation(splineControlPoint);

   // Compute the bending energy values everywhere but at the boundary
#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(splineControlPoint,splinePtrX,splinePtrY,splinePtrZ, derivativeValues, \
   basisXX, basisYY, basisZZ, basisXY, basisYZ, basisXZ) \
   private(a, b, c, i, index, x, y, z, derivativeValuesPtr, splineCoeffX, splineCoeffY, \
   splineCoeffZ, XX_x, YY_x, ZZ_x, XY_x, YZ_x, XZ_x, XX_y, YY_y, \
   ZZ_y, XY_y, YZ_y, XZ_y, XX_z, YY_z, ZZ_z, XY_z, YZ_z, XZ_z)
#endif
   for(z=0; z<splineControlPoint->nz; z++)
   {
      derivativeValuesPtr = &derivativeValues[18*z*splineControlPoint->ny*splineControlPoint->nx];
      for(y=0; y<splineControlPoint->ny; y++)
      {
         for(x=0; x<splineControlPoint->nx; x++)
         {
            XX_x=0.0, YY_x=0.0, ZZ_x=0.0;
            XY_x=0.0, YZ_x=0.0, XZ_x=0.0;
            XX_y=0.0, YY_y=0.0, ZZ_y=0.0;
            XY_y=0.0, YZ_y=0.0, XZ_y=0.0;
            XX_z=0.0, YY_z=0.0, ZZ_z=0.0;
            XY_z=0.0, YZ_z=0.0, XZ_z=0.0;

            i=0;
            for(c=-1; c<2; c++){
               for(b=-1; b<2; b++){
                  for(a=-1; a<2; a++){
                     if(-1<(x+a) && -1<(y+b) && -1<(z+c) && (x+a)<splineControlPoint->nx && (y+b)<splineControlPoint->ny && (z+c)<splineControlPoint->nz)
                     {
                        index = ((z+c)*splineControlPoint->ny+y+b)*splineControlPoint->nx+x+a;
                        splineCoeffX = splinePtrX[index];
                        splineCoeffY = splinePtrY[index];
                        splineCoeffZ = splinePtrZ[index];
                        XX_x += basisXX[i]*splineCoeffX;
                        YY_x += basisYY[i]*splineCoeffX;
                        ZZ_x += basisZZ[i]*splineCoeffX;
                        XY_x += basisXY[i]*splineCoeffX;
                        YZ_x += basisYZ[i]*splineCoeffX;
                        XZ_x += basisXZ[i]*splineCoeffX;

                        XX_y += basisXX[i]*splineCoeffY;
                        YY_y += basisYY[i]*splineCoeffY;
                        ZZ_y += basisZZ[i]*splineCoeffY;
                        XY_y += basisXY[i]*splineCoeffY;
                        YZ_y += basisYZ[i]*splineCoeffY;
                        XZ_y += basisXZ[i]*splineCoeffY;

                        XX_z += basisXX[i]*splineCoeffZ;
                        YY_z += basisYY[i]*splineCoeffZ;
                        ZZ_z += basisZZ[i]*splineCoeffZ;
                        XY_z += basisXY[i]*splineCoeffZ;
                        YZ_z += basisYZ[i]*splineCoeffZ;
                        XZ_z += basisXZ[i]*splineCoeffZ;
                     }
                     ++i;
                  }
               }
            }
            *derivativeValuesPtr++ = XX_x;
            *derivativeValuesPtr++ = XX_y;
            *derivativeValuesPtr++ = XX_z;
            *derivativeValuesPtr++ = YY_x;
            *derivativeValuesPtr++ = YY_y;
            *derivativeValuesPtr++ = YY_z;
            *derivativeValuesPtr++ = ZZ_x;
            *derivativeValuesPtr++ = ZZ_y;
            *derivativeValuesPtr++ = ZZ_z;
            *derivativeValuesPtr++ = (DTYPE)(2.0*XY_x);
            *derivativeValuesPtr++ = (DTYPE)(2.0*XY_y);
            *derivativeValuesPtr++ = (DTYPE)(2.0*XY_z);
            *derivativeValuesPtr++ = (DTYPE)(2.0*YZ_x);
            *derivativeValuesPtr++ = (DTYPE)(2.0*YZ_y);
            *derivativeValuesPtr++ = (DTYPE)(2.0*YZ_z);
            *derivativeValuesPtr++ = (DTYPE)(2.0*XZ_x);
            *derivativeValuesPtr++ = (DTYPE)(2.0*XZ_y);
            *derivativeValuesPtr++ = (DTYPE)(2.0*XZ_z);
         }
      }
   }

   DTYPE *gradientXPtr = static_cast<DTYPE *>(gradientImage->data);
   DTYPE *gradientYPtr = &gradientXPtr[nodeNumber];
   DTYPE *gradientZPtr = &gradientYPtr[nodeNumber];

   DTYPE approxRatio = (DTYPE)weight / (DTYPE)nodeNumber;
   DTYPE gradientValue[3];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(splineControlPoint, derivativeValues, gradientXPtr, gradientYPtr, gradientZPtr, \
   basisXX, basisYY, basisZZ, basisXY, basisYZ, basisXZ, approxRatio) \
   private(index, a, X, Y, Z, x, y, z, derivativeValuesPtr, gradientValue)
#endif
   for(z=0; z<splineControlPoint->nz; z++)
   {
      index=z*splineControlPoint->nx*splineControlPoint->ny;
      for(y=0; y<splineControlPoint->ny; y++)
      {
         for(x=0; x<splineControlPoint->nx; x++)
         {
            gradientValue[0]=gradientValue[1]=gradientValue[2]=0.0;
            a=0;
            for(Z=z-1; Z<z+2; Z++)
            {
               for(Y=y-1; Y<y+2; Y++)
               {
                  for(X=x-1; X<x+2; X++)
                  {
                     if(-1<X && -1<Y && -1<Z && X<splineControlPoint->nx && Y<splineControlPoint->ny && Z<splineControlPoint->nz)
                     {
                        derivativeValuesPtr = &derivativeValues[18 * ((Z*splineControlPoint->ny + Y)*splineControlPoint->nx + X)];
                        gradientValue[0] += (*derivativeValuesPtr++) * basisXX[a];
                        gradientValue[1] += (*derivativeValuesPtr++) * basisXX[a];
                        gradientValue[2] += (*derivativeValuesPtr++) * basisXX[a];

                        gradientValue[0] += (*derivativeValuesPtr++) * basisYY[a];
                        gradientValue[1] += (*derivativeValuesPtr++) * basisYY[a];
                        gradientValue[2] += (*derivativeValuesPtr++) * basisYY[a];

                        gradientValue[0] += (*derivativeValuesPtr++) * basisZZ[a];
                        gradientValue[1] += (*derivativeValuesPtr++) * basisZZ[a];
                        gradientValue[2] += (*derivativeValuesPtr++) * basisZZ[a];

                        gradientValue[0] += (*derivativeValuesPtr++) * basisXY[a];
                        gradientValue[1] += (*derivativeValuesPtr++) * basisXY[a];
                        gradientValue[2] += (*derivativeValuesPtr++) * basisXY[a];

                        gradientValue[0] += (*derivativeValuesPtr++) * basisYZ[a];
                        gradientValue[1] += (*derivativeValuesPtr++) * basisYZ[a];
                        gradientValue[2] += (*derivativeValuesPtr++) * basisYZ[a];

                        gradientValue[0] += (*derivativeValuesPtr++) * basisXZ[a];
                        gradientValue[1] += (*derivativeValuesPtr++) * basisXZ[a];
                        gradientValue[2] += (*derivativeValuesPtr++) * basisXZ[a];
                     }
                     a++;
                  }
               }
            }
            gradientXPtr[index] += approxRatio*gradientValue[0];
            gradientYPtr[index] += approxRatio*gradientValue[1];
            gradientZPtr[index] += approxRatio*gradientValue[2];
            index++;
         }
      }
   }
   free(derivativeValues);
   reg_getDeformationFromDisplacement(splineControlPoint);
}
/* *************************************************************** */
extern "C++"
void reg_spline_approxBendingEnergyGradient(nifti_image *splineControlPoint,
                                            nifti_image *gradientImage,
                                            float weight)
{
   if(splineControlPoint->datatype != gradientImage->datatype)
   {
      reg_print_fct_error("reg_spline_approxBendingEnergyGradient");
      reg_print_msg_error("The input images are expected to have the same type");
      reg_exit();
   }
   if(splineControlPoint->nz==1)
   {
      switch(splineControlPoint->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         reg_spline_approxBendingEnergyGradient2D<float>
               (splineControlPoint, gradientImage, weight);
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_spline_approxBendingEnergyGradient2D<double>
               (splineControlPoint, gradientImage, weight);
         break;
      default:
         reg_print_fct_error("reg_spline_approxBendingEnergyGradient");
         reg_print_msg_error("Only implemented for single or double precision images");
         reg_exit();
      }
   }
   else
   {
      switch(splineControlPoint->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         reg_spline_approxBendingEnergyGradient3D<float>
               (splineControlPoint, gradientImage, weight);
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_spline_approxBendingEnergyGradient3D<double>
               (splineControlPoint, gradientImage, weight);
         break;
      default:
         reg_print_fct_error("reg_spline_approxBendingEnergyGradient");
         reg_print_msg_error("Only implemented for single or double precision images");
         reg_exit();
      }
   }
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
double reg_spline_approxLinearEnergyValue2D(nifti_image *splineControlPoint)
{
   size_t nodeNumber = (size_t)splineControlPoint->nx*
         splineControlPoint->ny;
   int a, b, x, y, i, index;

   double constraintValue = 0.;
   double currentValue;

   // Create pointers to the spline coefficients
   DTYPE *splinePtrX = static_cast<DTYPE *>(splineControlPoint->data);
   DTYPE *splinePtrY = &splinePtrX[nodeNumber];

   // Store the basis values since they are constant as the value is approximated
   // at the control point positions only
   DTYPE basisX[9], basisY[9];
   set_first_order_basis_values(basisX, basisY);

   DTYPE splineCoeffX;
   DTYPE splineCoeffY;

   mat33 matrix, R;

   // Matrix to use to convert the gradient from mm to voxel
   mat33 reorientation;
   if(splineControlPoint->sform_code>0)
      reorientation = reg_mat44_to_mat33(&splineControlPoint->sto_ijk);
   else reorientation = reg_mat44_to_mat33(&splineControlPoint->qto_ijk);

#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(splinePtrX, splinePtrY, splineControlPoint, \
   basisX, basisY, reorientation) \
   private(x, y, a, b, i, index, matrix, R, \
   splineCoeffX, splineCoeffY, currentValue) \
   reduction(+:constraintValue)
#endif
   for(y=1; y<splineControlPoint->ny-1; ++y){
      for(x=1; x<splineControlPoint->nx-1; ++x){

         memset(&matrix, 0, sizeof(mat33));
         matrix.m[2][2] = 1.f;

         i=0;
         for(b=-1; b<2; b++){
            for(a=-1; a<2; a++){
               index = (y+b)*splineControlPoint->nx+x+a;
               splineCoeffX = splinePtrX[index];
               splineCoeffY = splinePtrY[index];
               matrix.m[0][0] += basisX[i]*splineCoeffX;
               matrix.m[1][0] += basisY[i]*splineCoeffX;
               matrix.m[0][1] += basisX[i]*splineCoeffY;
               matrix.m[1][1] += basisY[i]*splineCoeffY;
               ++i;
            }
         }
         // Convert from mm to voxel
         matrix = nifti_mat33_mul(reorientation, matrix);
         // Removing the rotation component
         R = nifti_mat33_inverse(nifti_mat33_polar(matrix));
         matrix = nifti_mat33_mul(R, matrix);
         // Convert to displacement
         --matrix.m[0][0];
         --matrix.m[1][1];

         currentValue = 0.;
         for(b=0; b<2; b++){
            for(a=0; a<2; a++){
               currentValue += reg_pow2(0.5*(matrix.m[a][b]+matrix.m[b][a])); // symmetric part
            }
         }
         constraintValue += currentValue;
      }
   }
   return constraintValue / static_cast<double>(splineControlPoint->nvox);
}
/* *************************************************************** */
template <class DTYPE>
double reg_spline_approxLinearEnergyValue3D(nifti_image *splineControlPoint)
{
   size_t nodeNumber = (size_t)splineControlPoint->nx *
         splineControlPoint->ny * splineControlPoint->nz;
   int a, b, c, x, y, z, i, index;

   double constraintValue = 0.;
   double currentValue;

   // Create pointers to the spline coefficients
   DTYPE *splinePtrX = static_cast<DTYPE *>(splineControlPoint->data);
   DTYPE *splinePtrY = &splinePtrX[nodeNumber];
   DTYPE *splinePtrZ = &splinePtrY[nodeNumber];

   // Store the basis values since they are constant as the value is approximated
   // at the control point positions only
   DTYPE basisX[27], basisY[27], basisZ[27];
   set_first_order_basis_values(basisX, basisY, basisZ);

   DTYPE splineCoeffX;
   DTYPE splineCoeffY;
   DTYPE splineCoeffZ;

   mat33 matrix, R;

   // Matrix to use to convert the gradient from mm to voxel
   mat33 reorientation;
   if(splineControlPoint->sform_code>0)
      reorientation = reg_mat44_to_mat33(&splineControlPoint->sto_ijk);
   else reorientation = reg_mat44_to_mat33(&splineControlPoint->qto_ijk);

#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(splinePtrX, splinePtrY, splinePtrZ, splineControlPoint, \
   basisX, basisY, basisZ, reorientation) \
   private(x, y, z, a, b, c, i, index, matrix, R, \
   splineCoeffX, splineCoeffY, splineCoeffZ, currentValue) \
   reduction(+:constraintValue)
#endif
   for(z=1; z<splineControlPoint->nz-1; ++z){
      for(y=1; y<splineControlPoint->ny-1; ++y){
         for(x=1; x<splineControlPoint->nx-1; ++x){

            memset(&matrix, 0, sizeof(mat33));

            i=0;
            for(c=-1; c<2; c++){
               for(b=-1; b<2; b++){
                  for(a=-1; a<2; a++){
                     index = ((z+c)*splineControlPoint->ny+y+b)*splineControlPoint->nx+x+a;
                     splineCoeffX = splinePtrX[index];
                     splineCoeffY = splinePtrY[index];
                     splineCoeffZ = splinePtrZ[index];

                     matrix.m[0][0] += basisX[i]*splineCoeffX;
                     matrix.m[1][0] += basisY[i]*splineCoeffX;
                     matrix.m[2][0] += basisZ[i]*splineCoeffX;

                     matrix.m[0][1] += basisX[i]*splineCoeffY;
                     matrix.m[1][1] += basisY[i]*splineCoeffY;
                     matrix.m[2][1] += basisZ[i]*splineCoeffY;

                     matrix.m[0][2] += basisX[i]*splineCoeffZ;
                     matrix.m[1][2] += basisY[i]*splineCoeffZ;
                     matrix.m[2][2] += basisZ[i]*splineCoeffZ;
                     ++i;
                  }
               }
            }
            // Convert from mm to voxel
            matrix = nifti_mat33_mul(reorientation, matrix);
            // Removing the rotation component
            R = nifti_mat33_inverse(nifti_mat33_polar(matrix));
            matrix = nifti_mat33_mul(R, matrix);
            // Convert to displacement
            --matrix.m[0][0];
            --matrix.m[1][1];
            --matrix.m[2][2];

            currentValue = 0.;
            for(b=0; b<3; b++){
               for(a=0; a<3; a++){
                  currentValue += reg_pow2(0.5*(matrix.m[a][b]+matrix.m[b][a])); // symmetric part
               }
            }
            constraintValue += currentValue;
         }
      }
   }
   return constraintValue / static_cast<double>(splineControlPoint->nvox);
}
/* *************************************************************** */
double reg_spline_approxLinearEnergy(nifti_image *splineControlPoint)
{
   if(splineControlPoint->nz>1){
      switch(splineControlPoint->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         return reg_spline_approxLinearEnergyValue3D<float>(splineControlPoint);
      case NIFTI_TYPE_FLOAT64:
         return reg_spline_approxLinearEnergyValue3D<double>(splineControlPoint);
      default:
         reg_print_fct_error("reg_spline_approxLinearEnergyValue3D");
         reg_print_msg_error("Only implemented for single or double precision images");
         reg_exit();
      }
   }
   else{
      switch(splineControlPoint->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         return reg_spline_approxLinearEnergyValue2D<float>(splineControlPoint);
      case NIFTI_TYPE_FLOAT64:
         return reg_spline_approxLinearEnergyValue2D<double>(splineControlPoint);
      default:
         reg_print_fct_error("reg_spline_approxLinearEnergyValue2D");
         reg_print_msg_error("Only implemented for single or double precision images");
         reg_exit();
      }
   }
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_spline_approxLinearEnergyGradient2D(nifti_image *splineControlPoint,
                                             nifti_image *gradientImage,
                                             float weight
                                             )
{
   size_t nodeNumber = (size_t)splineControlPoint->nx*
         splineControlPoint->ny;
   int x, y, X, Y, a, b, i, index;

   // Create pointers to the spline coefficients
   DTYPE * splinePtrX = static_cast<DTYPE *>(splineControlPoint->data);
   DTYPE * splinePtrY = &splinePtrX[nodeNumber];

   // Store the basis values since they are constant as the value is approximated
   // at the control point positions only
   DTYPE basisX[9];
   DTYPE basisY[9];
   set_first_order_basis_values(basisX, basisY);

   DTYPE *derivativeValues = (DTYPE *)calloc(4*nodeNumber, sizeof(DTYPE));
   DTYPE *derivativeValuesPtr;

   // Matrix to use to convert the gradient from mm to voxel
   mat33 reorientation;
   if(splineControlPoint->sform_code>0)
      reorientation = reg_mat44_to_mat33(&splineControlPoint->sto_ijk);
   else reorientation = reg_mat44_to_mat33(&splineControlPoint->qto_ijk);

   DTYPE splineCoeffX;
   DTYPE splineCoeffY;

   mat33 matrix, R;

#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(splineControlPoint, splinePtrX, splinePtrY, \
   derivativeValues, basisX, basisY, reorientation) \
   private(x, y, a, b, i, index, derivativeValuesPtr, \
   splineCoeffX, splineCoeffY, matrix, R)
#endif
   for(y=1; y<splineControlPoint->ny-1; y++)
   {
      derivativeValuesPtr = &derivativeValues[
            4*(y*splineControlPoint->nx+1)
            ];
      for(x=1; x<splineControlPoint->nx-1; x++)
      {
         memset(&matrix, 0, sizeof(mat33));
         matrix.m[2][2]=1.f;

         i=0;
         for(b=-1; b<2; b++){
            for(a=-1; a<2; a++){
               index = (y+b)*splineControlPoint->nx+x+a;
               splineCoeffX = splinePtrX[index];
               splineCoeffY = splinePtrY[index];

               matrix.m[0][0] += basisX[i]*splineCoeffX;
               matrix.m[1][0] += basisY[i]*splineCoeffX;

               matrix.m[0][1] += basisX[i]*splineCoeffY;
               matrix.m[1][1] += basisY[i]*splineCoeffY;
               ++i;
            }
         }
         // Convert from mm to voxel
         matrix = nifti_mat33_mul(reorientation, matrix);
         // Removing the rotation component
         R = nifti_mat33_inverse(nifti_mat33_polar(matrix));
         matrix = nifti_mat33_mul(R, matrix);
         // Convert to displacement
         --matrix.m[0][0];
         --matrix.m[1][1];
         *derivativeValuesPtr++ = matrix.m[0][0];
         *derivativeValuesPtr++ = matrix.m[0][1];
         *derivativeValuesPtr++ = matrix.m[1][0];
         *derivativeValuesPtr++ = matrix.m[1][1];
      } // x
   } // y

   DTYPE *gradientXPtr = static_cast<DTYPE *>(gradientImage->data);
   DTYPE *gradientYPtr = &gradientXPtr[nodeNumber];

   DTYPE approxRatio = (DTYPE)weight / (DTYPE)(nodeNumber);

   // Matrices to be used to convert the gradient from voxel to mm
   reorientation = nifti_mat33_inverse(reorientation);

   double gradValues[2];

#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(splineControlPoint, derivativeValues, \
   gradientXPtr, gradientYPtr, \
   basisX, basisY, approxRatio, reorientation) \
   private(index, i, X, Y, x, y, a, \
   derivativeValuesPtr, gradValues, matrix)
#endif
   for(y=0; y<splineControlPoint->ny; y++)
   {
      index=y*splineControlPoint->nx;
      for(x=0; x<splineControlPoint->nx; x++)
      {
         gradValues[0]=gradValues[1]=0.0;
         i=0;
         for(Y=y-1; Y<y+2; Y++)
         {
            for(X=x-1; X<x+2; X++)
            {
               if(-1<X && -1<Y &&
                     X<splineControlPoint->nx &&
                     Y<splineControlPoint->ny)
               {
                  derivativeValuesPtr = &derivativeValues[
                        4 * (Y*splineControlPoint->nx + X)
                        ];

                  matrix.m[0][0] = (*derivativeValuesPtr++);
                  matrix.m[0][1] = (*derivativeValuesPtr++);

                  matrix.m[1][0] = (*derivativeValuesPtr++);
                  matrix.m[1][1] = (*derivativeValuesPtr++);

                  gradValues[0] -= 2.0*matrix.m[0][0]*basisX[i];
                  gradValues[1] -= 2.0*matrix.m[1][1]*basisY[i];
               }
               ++i;
            } // X
         } // Y
         matrix = reorientation;
         gradientXPtr[index] += approxRatio *
               ( matrix.m[0][0]*gradValues[0]
               + matrix.m[0][1]*gradValues[1]);
         gradientYPtr[index] += approxRatio *
               ( matrix.m[1][0]*gradValues[0]
               + matrix.m[1][1]*gradValues[1]);
         index++;
      }
   }
   free(derivativeValues);
}
/* *************************************************************** */
template <class DTYPE>
void reg_spline_approxLinearEnergyGradient3D(nifti_image *splineControlPoint,
                                             nifti_image *gradientImage,
                                             float weight
                                             )
{
   size_t nodeNumber = (size_t)splineControlPoint->nx*
         splineControlPoint->ny*splineControlPoint->nz;
   int x, y, z, X, Y, Z, a, b, c, i, index;

   // Create pointers to the spline coefficients
   DTYPE * splinePtrX = static_cast<DTYPE *>(splineControlPoint->data);
   DTYPE * splinePtrY = &splinePtrX[nodeNumber];
   DTYPE * splinePtrZ = &splinePtrY[nodeNumber];

   // Store the basis values since they are constant as the value is approximated
   // at the control point positions only
   DTYPE basisX[27];
   DTYPE basisY[27];
   DTYPE basisZ[27];
   set_first_order_basis_values(basisX, basisY, basisZ);

   DTYPE *derivativeValues = (DTYPE *)calloc(9*nodeNumber, sizeof(DTYPE));
   DTYPE *derivativeValuesPtr;

   // Matrix to use to convert the gradient from mm to voxel
   mat33 reorientation;
   if(splineControlPoint->sform_code>0)
      reorientation = reg_mat44_to_mat33(&splineControlPoint->sto_ijk);
   else reorientation = reg_mat44_to_mat33(&splineControlPoint->qto_ijk);

   DTYPE splineCoeffX;
   DTYPE splineCoeffY;
   DTYPE splineCoeffZ;

   mat33 matrix, R;

#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(splineControlPoint, splinePtrX, splinePtrY, splinePtrZ, \
   derivativeValues, basisX, basisY, basisZ, reorientation) \
   private(x, y, z, a, b, c, i, index, derivativeValuesPtr, \
   splineCoeffX, splineCoeffY, splineCoeffZ, matrix, R)
#endif
   for(z=1; z<splineControlPoint->nz-1; z++)
   {
      for(y=1; y<splineControlPoint->ny-1; y++)
      {
         derivativeValuesPtr = &derivativeValues[
               9*((z*splineControlPoint->ny+y)*splineControlPoint->nx+1)
               ];
         for(x=1; x<splineControlPoint->nx-1; x++)
         {
            memset(&matrix, 0, sizeof(mat33));

            i=0;
            for(c=-1; c<2; c++){
               for(b=-1; b<2; b++){
                  for(a=-1; a<2; a++){
                     index = ((z+c)*splineControlPoint->ny+y+b)*splineControlPoint->nx+x+a;
                     splineCoeffX = splinePtrX[index];
                     splineCoeffY = splinePtrY[index];
                     splineCoeffZ = splinePtrZ[index];

                     matrix.m[0][0] += basisX[i]*splineCoeffX;
                     matrix.m[1][0] += basisY[i]*splineCoeffX;
                     matrix.m[2][0] += basisZ[i]*splineCoeffX;

                     matrix.m[0][1] += basisX[i]*splineCoeffY;
                     matrix.m[1][1] += basisY[i]*splineCoeffY;
                     matrix.m[2][1] += basisZ[i]*splineCoeffY;

                     matrix.m[0][2] += basisX[i]*splineCoeffZ;
                     matrix.m[1][2] += basisY[i]*splineCoeffZ;
                     matrix.m[2][2] += basisZ[i]*splineCoeffZ;
                     ++i;
                  }
               }
            }
            // Convert from mm to voxel
            matrix = nifti_mat33_mul(reorientation, matrix);
            // Removing the rotation component
            R = nifti_mat33_inverse(nifti_mat33_polar(matrix));
            matrix = nifti_mat33_mul(R, matrix);
            // Convert to displacement
            --matrix.m[0][0];
            --matrix.m[1][1];
            --matrix.m[2][2];
            *derivativeValuesPtr++ = matrix.m[0][0];
            *derivativeValuesPtr++ = matrix.m[0][1];
            *derivativeValuesPtr++ = matrix.m[0][2];
            *derivativeValuesPtr++ = matrix.m[1][0];
            *derivativeValuesPtr++ = matrix.m[1][1];
            *derivativeValuesPtr++ = matrix.m[1][2];
            *derivativeValuesPtr++ = matrix.m[2][0];
            *derivativeValuesPtr++ = matrix.m[2][1];
            *derivativeValuesPtr++ = matrix.m[2][2];
         } // x
      } // y
   } // z

   DTYPE *gradientXPtr = static_cast<DTYPE *>(gradientImage->data);
   DTYPE *gradientYPtr = &gradientXPtr[nodeNumber];
   DTYPE *gradientZPtr = &gradientYPtr[nodeNumber];

   DTYPE approxRatio = (DTYPE)weight / (DTYPE)(nodeNumber);

   // Matrices to be used to convert the gradient from voxel to mm
   reorientation = nifti_mat33_inverse(reorientation);

   double gradValues[3];

#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(splineControlPoint, derivativeValues, \
   gradientXPtr, gradientYPtr, gradientZPtr, \
   basisX, basisY, basisZ, approxRatio, reorientation) \
   private(index, i, X, Y, Z, x, y, z, a, \
   derivativeValuesPtr, gradValues, matrix)
#endif
   for(z=0; z<splineControlPoint->nz; z++)
   {
      index=z*splineControlPoint->nx*splineControlPoint->ny;
      for(y=0; y<splineControlPoint->ny; y++)
      {
         for(x=0; x<splineControlPoint->nx; x++)
         {
            gradValues[0]=gradValues[1]=gradValues[2]=0.0;
            i=0;
            for(Z=z-1; Z<z+2; Z++)
            {
               for(Y=y-1; Y<y+2; Y++)
               {
                  for(X=x-1; X<x+2; X++)
                  {
                     if(-1<X && -1<Y && -1<Z &&
                           X<splineControlPoint->nx &&
                           Y<splineControlPoint->ny &&
                           Z<splineControlPoint->nz)
                     {
                        derivativeValuesPtr = &derivativeValues[
                              9 * ((Z*splineControlPoint->ny + Y)*splineControlPoint->nx + X)
                              ];

                        matrix.m[0][0] = (*derivativeValuesPtr++);
                        matrix.m[0][1] = (*derivativeValuesPtr++);
                        matrix.m[0][2] = (*derivativeValuesPtr++);

                        matrix.m[1][0] = (*derivativeValuesPtr++);
                        matrix.m[1][1] = (*derivativeValuesPtr++);
                        matrix.m[1][2] = (*derivativeValuesPtr++);

                        matrix.m[2][0] = (*derivativeValuesPtr++);
                        matrix.m[2][1] = (*derivativeValuesPtr++);
                        matrix.m[2][2] = (*derivativeValuesPtr++);

                        gradValues[0] -= 2.0*matrix.m[0][0]*basisX[i];
//                        gradValues[0] -= (matrix.m[0][1]+matrix.m[1][0])*basisY[i];
//                        gradValues[0] -= (matrix.m[0][2]+matrix.m[2][0])*basisZ[i];

                        gradValues[1] -= 2.0*matrix.m[1][1]*basisY[i];
//                        gradValues[1] -= (matrix.m[1][0]+matrix.m[0][1])*basisX[i];
//                        gradValues[1] -= (matrix.m[1][2]+matrix.m[2][1])*basisZ[i];

                        gradValues[2] -= 2.0*matrix.m[2][2]*basisZ[i];
//                        gradValues[2] -= (matrix.m[2][0]+matrix.m[0][2])*basisX[i];
//                        gradValues[2] -= (matrix.m[2][1]+matrix.m[1][2])*basisY[i];
                     }
                     ++i;
                  } // X
               } // Y
            } // Z
            matrix = reorientation;
            gradientXPtr[index] += approxRatio *
                  ( matrix.m[0][0]*gradValues[0]
                  + matrix.m[0][1]*gradValues[1]
                  + matrix.m[0][2]*gradValues[2]);
            gradientYPtr[index] += approxRatio *
                  ( matrix.m[1][0]*gradValues[0]
                  + matrix.m[1][1]*gradValues[1]
                  + matrix.m[1][2]*gradValues[2]);
            gradientZPtr[index] += approxRatio *
                  ( matrix.m[2][0]*gradValues[0]
                  + matrix.m[2][1]*gradValues[1]
                  + matrix.m[2][2]*gradValues[2]);
            index++;
         }
      }
   }
   free(derivativeValues);
}
/* *************************************************************** */
void reg_spline_approxLinearEnergyGradient(nifti_image *splineControlPoint,
                                           nifti_image *gradientImage,
                                           float weight
                                           )
{
   if(splineControlPoint->datatype != gradientImage->datatype)
   {
      reg_print_fct_error("reg_spline_linearEnergyGradient");
      reg_print_msg_error("Input images are expected to have the same datatype");
      reg_exit();
   }
   if(splineControlPoint->nz>1){
      switch(splineControlPoint->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         reg_spline_approxLinearEnergyGradient3D<float>
               (splineControlPoint, gradientImage, weight);
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_spline_approxLinearEnergyGradient3D<double>
               (splineControlPoint, gradientImage, weight);
         break;
      default:
         reg_print_fct_error("reg_spline_linearEnergyGradient");
         reg_print_msg_error("Only implemented for single or double precision images");
         reg_exit();
      }
   }
   else{
      switch(splineControlPoint->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         reg_spline_approxLinearEnergyGradient2D<float>
               (splineControlPoint, gradientImage, weight);
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_spline_approxLinearEnergyGradient2D<double>
               (splineControlPoint, gradientImage, weight);
         break;
      default:
         reg_print_fct_error("reg_spline_linearEnergyGradient");
         reg_print_msg_error("Only implemented for single or double precision images");
         reg_exit();
      }
   }
}
/* *************************************************************** */
/* *************************************************************** */
#ifdef BUILD_DEV
template <class DTYPE>
double reg_spline_approxLinearPairwise3D(nifti_image *splineControlPoint)
{
   size_t nodeNumber = (size_t)splineControlPoint->nx*
         splineControlPoint->ny*splineControlPoint->nz;
   int x, y, z, index;

   // Create pointers to the spline coefficients
   reg_getDisplacementFromDeformation(splineControlPoint);
   DTYPE * splinePtrX = static_cast<DTYPE *>(splineControlPoint->data);
   DTYPE * splinePtrY = &splinePtrX[nodeNumber];
   DTYPE * splinePtrZ = &splinePtrY[nodeNumber];

   DTYPE centralCP[3], neigbCP[3];

   double constraintValue=0;
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   private(index, x, y, z, centralCP, neigbCP) \
   shared(splineControlPoint, splinePtrX, splinePtrY, splinePtrZ) \
   reduction(+:constraintValue)
#endif // _OPENMP
   for(z=0; z<splineControlPoint->nz;++z){
      index=z*splineControlPoint->nx*splineControlPoint->ny;
      for(y=0; y<splineControlPoint->ny;++y){
         for(x=0; x<splineControlPoint->nx;++x){
            centralCP[0]=splinePtrX[index];
            centralCP[1]=splinePtrY[index];
            centralCP[2]=splinePtrZ[index];

            if(x>0){
               neigbCP[0]=splinePtrX[index-1];
               neigbCP[1]=splinePtrY[index-1];
               neigbCP[2]=splinePtrZ[index-1];
               constraintValue += (reg_pow2(centralCP[0]-neigbCP[0])+reg_pow2(centralCP[1]-neigbCP[1])+
                     reg_pow2(centralCP[2]-neigbCP[2]))/splineControlPoint->dx;
            }
            if(x<splineControlPoint->nx-1){
               neigbCP[0]=splinePtrX[index+1];
               neigbCP[1]=splinePtrY[index+1];
               neigbCP[2]=splinePtrZ[index+1];
               constraintValue += (reg_pow2(centralCP[0]-neigbCP[0])+reg_pow2(centralCP[1]-neigbCP[1])+
                     reg_pow2(centralCP[2]-neigbCP[2]))/splineControlPoint->dx;
            }

            if(y>0){
               neigbCP[0]=splinePtrX[index-splineControlPoint->nx];
               neigbCP[1]=splinePtrY[index-splineControlPoint->nx];
               neigbCP[2]=splinePtrZ[index-splineControlPoint->nx];
               constraintValue += (reg_pow2(centralCP[0]-neigbCP[0])+reg_pow2(centralCP[1]-neigbCP[1])+
                     reg_pow2(centralCP[2]-neigbCP[2]))/splineControlPoint->dy;
            }
            if(y<splineControlPoint->ny-1){
               neigbCP[0]=splinePtrX[index+splineControlPoint->nx];
               neigbCP[1]=splinePtrY[index+splineControlPoint->nx];
               neigbCP[2]=splinePtrZ[index+splineControlPoint->nx];
               constraintValue += (reg_pow2(centralCP[0]-neigbCP[0])+reg_pow2(centralCP[1]-neigbCP[1])+
                     reg_pow2(centralCP[2]-neigbCP[2]))/splineControlPoint->dy;
            }

            if(z>0){
               neigbCP[0]=splinePtrX[index-splineControlPoint->nx*splineControlPoint->ny];
               neigbCP[1]=splinePtrY[index-splineControlPoint->nx*splineControlPoint->ny];
               neigbCP[2]=splinePtrZ[index-splineControlPoint->nx*splineControlPoint->ny];
               constraintValue += (reg_pow2(centralCP[0]-neigbCP[0])+reg_pow2(centralCP[1]-neigbCP[1])+
                     reg_pow2(centralCP[2]-neigbCP[2]))/splineControlPoint->dz;
            }
            if(z<splineControlPoint->nz-1){
               neigbCP[0]=splinePtrX[index+splineControlPoint->nx*splineControlPoint->ny];
               neigbCP[1]=splinePtrY[index+splineControlPoint->nx*splineControlPoint->ny];
               neigbCP[2]=splinePtrZ[index+splineControlPoint->nx*splineControlPoint->ny];
               constraintValue += (reg_pow2(centralCP[0]-neigbCP[0])+reg_pow2(centralCP[1]-neigbCP[1])+
                     reg_pow2(centralCP[2]-neigbCP[2]))/splineControlPoint->dz;
            }
            index++;
         } // x
      } // y
   } // z
   reg_getDeformationFromDisplacement(splineControlPoint);
   return constraintValue/static_cast<double>(nodeNumber);
}
/* *************************************************************** */
double reg_spline_approxLinearPairwise(nifti_image *splineControlPoint)
{
   if(splineControlPoint->nz>1){
      switch(splineControlPoint->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         return reg_spline_approxLinearPairwise3D<float>(splineControlPoint);
      case NIFTI_TYPE_FLOAT64:
         return reg_spline_approxLinearPairwise3D<double>(splineControlPoint);
      default:
         reg_print_fct_error("reg_spline_approxLinearPairwise");
         reg_print_msg_error("Only implemented for single or double precision images");
         reg_exit();
      }
   }
   else{
      reg_print_fct_error("reg_spline_approxLinearPairwise");
      reg_print_msg_error("Not implemented in 2D yet");
      reg_exit();
   }
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_spline_approxLinearPairwiseGradient3D(nifti_image *splineControlPoint,
                                               nifti_image *gradientImage,
                                               float weight
                                               )
{
   size_t nodeNumber = (size_t)splineControlPoint->nx*
         splineControlPoint->ny*splineControlPoint->nz;
   int x, y, z, index;

   // Create pointers to the spline coefficients
   reg_getDisplacementFromDeformation(splineControlPoint);
   DTYPE *splinePtrX = static_cast<DTYPE *>(splineControlPoint->data);
   DTYPE *splinePtrY = &splinePtrX[nodeNumber];
   DTYPE *splinePtrZ = &splinePtrY[nodeNumber];

   // Pointers to the gradient image
   DTYPE *gradPtrX = static_cast<DTYPE *>(gradientImage->data);
   DTYPE *gradPtrY = &gradPtrX[nodeNumber];
   DTYPE *gradPtrZ = &gradPtrY[nodeNumber];

   DTYPE centralCP[3], neigbCP[3];

   double grad_values[3];

   DTYPE approxRatio = (DTYPE)weight / (DTYPE)(nodeNumber);
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   private(index, x, y, z, centralCP, neigbCP, grad_values) \
   shared(splineControlPoint, splinePtrX, splinePtrY, splinePtrZ, approxRatio, \
   gradPtrX, gradPtrY, gradPtrZ)
#endif // _OPENMP
   for(z=0; z<splineControlPoint->nz;++z){
      index=z*splineControlPoint->nx*splineControlPoint->ny;
      for(y=0; y<splineControlPoint->ny;++y){
         for(x=0; x<splineControlPoint->nx;++x){
            centralCP[0]=splinePtrX[index];
            centralCP[1]=splinePtrY[index];
            centralCP[2]=splinePtrZ[index];
            grad_values[0]=0;
            grad_values[1]=0;
            grad_values[2]=0;

            if(x>0){
               neigbCP[0]=splinePtrX[index-1];
               neigbCP[1]=splinePtrY[index-1];
               neigbCP[2]=splinePtrZ[index-1];
               grad_values[0] += 2. * (centralCP[0]-neigbCP[0])/splineControlPoint->dx;
               grad_values[1] += 2. * (centralCP[1]-neigbCP[1])/splineControlPoint->dx;
               grad_values[2] += 2. * (centralCP[2]-neigbCP[2])/splineControlPoint->dx;
            }
            if(x<splineControlPoint->nx-1){
               neigbCP[0]=splinePtrX[index+1];
               neigbCP[1]=splinePtrY[index+1];
               neigbCP[2]=splinePtrZ[index+1];
               grad_values[0] += 2. * (centralCP[0]-neigbCP[0])/splineControlPoint->dx;
               grad_values[1] += 2. * (centralCP[1]-neigbCP[1])/splineControlPoint->dx;
               grad_values[2] += 2. * (centralCP[2]-neigbCP[2])/splineControlPoint->dx;
            }

            if(y>0){
               neigbCP[0]=splinePtrX[index-splineControlPoint->nx];
               neigbCP[1]=splinePtrY[index-splineControlPoint->nx];
               neigbCP[2]=splinePtrZ[index-splineControlPoint->nx];
               grad_values[0] += 2. * (centralCP[0]-neigbCP[0])/splineControlPoint->dy;
               grad_values[1] += 2. * (centralCP[1]-neigbCP[1])/splineControlPoint->dy;
               grad_values[2] += 2. * (centralCP[2]-neigbCP[2])/splineControlPoint->dy;
            }
            if(y<splineControlPoint->ny-1){
               neigbCP[0]=splinePtrX[index+splineControlPoint->nx];
               neigbCP[1]=splinePtrY[index+splineControlPoint->nx];
               neigbCP[2]=splinePtrZ[index+splineControlPoint->nx];
               grad_values[0] += 2. * (centralCP[0]-neigbCP[0])/splineControlPoint->dy;
               grad_values[1] += 2. * (centralCP[1]-neigbCP[1])/splineControlPoint->dy;
               grad_values[2] += 2. * (centralCP[2]-neigbCP[2])/splineControlPoint->dy;
            }

            if(z>0){
               neigbCP[0]=splinePtrX[index-splineControlPoint->nx*splineControlPoint->ny];
               neigbCP[1]=splinePtrY[index-splineControlPoint->nx*splineControlPoint->ny];
               neigbCP[2]=splinePtrZ[index-splineControlPoint->nx*splineControlPoint->ny];
               grad_values[0] += 2. * (centralCP[0]-neigbCP[0])/splineControlPoint->dz;
               grad_values[1] += 2. * (centralCP[1]-neigbCP[1])/splineControlPoint->dz;
               grad_values[2] += 2. * (centralCP[2]-neigbCP[2])/splineControlPoint->dz;
            }
            if(z<splineControlPoint->nz-1){
               neigbCP[0]=splinePtrX[index+splineControlPoint->nx*splineControlPoint->ny];
               neigbCP[1]=splinePtrY[index+splineControlPoint->nx*splineControlPoint->ny];
               neigbCP[2]=splinePtrZ[index+splineControlPoint->nx*splineControlPoint->ny];
               grad_values[0] += 2. * (centralCP[0]-neigbCP[0])/splineControlPoint->dz;
               grad_values[1] += 2. * (centralCP[1]-neigbCP[1])/splineControlPoint->dz;
               grad_values[2] += 2. * (centralCP[2]-neigbCP[2])/splineControlPoint->dz;
            }
            gradPtrX[index] += approxRatio * static_cast<DTYPE>(grad_values[0]);
            gradPtrY[index] += approxRatio * static_cast<DTYPE>(grad_values[1]);
            gradPtrZ[index] += approxRatio * static_cast<DTYPE>(grad_values[2]);

            index++;
         } // x
      } // y
   } // z
   reg_getDeformationFromDisplacement(splineControlPoint);
}
/* *************************************************************** */
void reg_spline_approxLinearPairwiseGradient(nifti_image *splineControlPoint,
                                             nifti_image *gradientImage,
                                             float weight
                                             )
{
   if(splineControlPoint->datatype != gradientImage->datatype)
   {
      reg_print_fct_error("reg_spline_approxLinearPairwiseGradient");
      reg_print_msg_error("Input images are expected to have the same datatype");
      reg_exit();
   }
   if(splineControlPoint->nz>1){
      switch(splineControlPoint->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         reg_spline_approxLinearPairwiseGradient3D<float>
               (splineControlPoint, gradientImage, weight);
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_spline_approxLinearPairwiseGradient3D<double>
               (splineControlPoint, gradientImage, weight);
         break;
      default:
         reg_print_fct_error("reg_spline_linearEnergyGradient");
         reg_print_msg_error("Only implemented for single or double precision images");
         reg_exit();
      }
   }
   else{
      reg_print_fct_error("reg_spline_approxLinearPairwiseGradient");
      reg_print_msg_error("Not implemented for 2D images yet");
      reg_exit();
   }
}
#endif // BUILD_DEV
/* *************************************************************** */
