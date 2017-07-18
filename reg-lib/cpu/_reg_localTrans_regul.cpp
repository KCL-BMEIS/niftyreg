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
double reg_spline_linearEnergyValue2D(nifti_image *referenceImage,
                                      nifti_image *splineControlPoint)
{
   size_t voxelNumber = (size_t)referenceImage->nx *
         referenceImage->ny;
   int a, b, x, y, index, xPre, yPre;
   DTYPE basis;


   DTYPE gridVoxelSpacing[2] ={
      gridVoxelSpacing[0] = splineControlPoint->dx / referenceImage->dx,
      gridVoxelSpacing[1] = splineControlPoint->dy / referenceImage->dy
   };

   double constraintValue = 0.;
   double currentValue;

   // Create pointers to the spline coefficients
   size_t nodeNumber = (size_t)splineControlPoint->nx *
         splineControlPoint->ny * splineControlPoint->nz;
   DTYPE *splinePtrX = static_cast<DTYPE *>(splineControlPoint->data);
   DTYPE *splinePtrY = &splinePtrX[nodeNumber];
   DTYPE splineCoeffX, splineCoeffY;

   // Store the basis values since they are constant as the value is approximated
   // at the control point positions only
   DTYPE basisX[4], basisY[4];
   DTYPE firstX[4], firstY[4];

   mat33 matrix, R;

   // Matrix to use to convert the gradient from mm to voxel
   mat33 reorientation;
   if(splineControlPoint->sform_code>0)
      reorientation = reg_mat44_to_mat33(&splineControlPoint->sto_ijk);
   else reorientation = reg_mat44_to_mat33(&splineControlPoint->qto_ijk);


   for(y=0; y<referenceImage->ny; ++y){

      yPre=static_cast<int>(static_cast<DTYPE>(y)/gridVoxelSpacing[1]);
      basis=static_cast<DTYPE>(y)/gridVoxelSpacing[1]-static_cast<DTYPE>(yPre);
      if(basis<0.0) basis=0.0; //rounding error
      get_BSplineBasisValues<DTYPE>(basis, basisY, firstY);

      for(x=0; x<referenceImage->nx; ++x){

         xPre=static_cast<int>(static_cast<DTYPE>(x)/gridVoxelSpacing[0]);
         basis=static_cast<DTYPE>(x)/gridVoxelSpacing[0]-static_cast<DTYPE>(xPre);
         if(basis<0.0) basis=0.0; //rounding error
         get_BSplineBasisValues<DTYPE>(basis, basisX, firstX);

         memset(&matrix, 0, sizeof(mat33));

         for(b=0; b<4; b++){
            for(a=0; a<4; a++){
               index = (yPre+b)*splineControlPoint->nx+xPre+a;
               splineCoeffX = splinePtrX[index];
               splineCoeffY = splinePtrY[index];

               matrix.m[0][0] += firstX[a]*basisY[b]*splineCoeffX;
               matrix.m[1][0] += basisX[a]*firstY[b]*splineCoeffX;

               matrix.m[0][1] += firstX[a]*basisY[b]*splineCoeffY;
               matrix.m[1][1] += basisX[a]*firstY[b]*splineCoeffY;
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
   return constraintValue / static_cast<double>(voxelNumber*2);
}
/* *************************************************************** */
template <class DTYPE>
double reg_spline_linearEnergyValue3D(nifti_image *referenceImage,
                                      nifti_image *splineControlPoint)
{
   size_t voxelNumber = (size_t)referenceImage->nx *
         referenceImage->ny * referenceImage->nz;
   int a, b, c, x, y, z, index, xPre, yPre, zPre;
   DTYPE basis;


   DTYPE gridVoxelSpacing[3] ={
      gridVoxelSpacing[0] = splineControlPoint->dx / referenceImage->dx,
      gridVoxelSpacing[1] = splineControlPoint->dy / referenceImage->dy,
      gridVoxelSpacing[2] = splineControlPoint->dz / referenceImage->dz
   };

   double constraintValue = 0.;
   double currentValue;

   // Create pointers to the spline coefficients
   size_t nodeNumber = (size_t)splineControlPoint->nx *
         splineControlPoint->ny * splineControlPoint->nz;
   DTYPE *splinePtrX = static_cast<DTYPE *>(splineControlPoint->data);
   DTYPE *splinePtrY = &splinePtrX[nodeNumber];
   DTYPE *splinePtrZ = &splinePtrY[nodeNumber];
   DTYPE splineCoeffX, splineCoeffY, splineCoeffZ;

   // Store the basis values since they are constant as the value is approximated
   // at the control point positions only
   DTYPE basisX[4], basisY[4], basisZ[4];
   DTYPE firstX[4], firstY[4], firstZ[4];

   mat33 matrix, R;

   // Matrix to use to convert the gradient from mm to voxel
   mat33 reorientation;
   if(splineControlPoint->sform_code>0)
      reorientation = reg_mat44_to_mat33(&splineControlPoint->sto_ijk);
   else reorientation = reg_mat44_to_mat33(&splineControlPoint->qto_ijk);

   for(z=0; z<referenceImage->nz; ++z){

      zPre=static_cast<int>(static_cast<DTYPE>(z)/gridVoxelSpacing[2]);
      basis=static_cast<DTYPE>(z)/gridVoxelSpacing[2]-static_cast<DTYPE>(zPre);
      if(basis<0.0) basis=0.0; //rounding error
      get_BSplineBasisValues<DTYPE>(basis, basisZ, firstZ);

      for(y=0; y<referenceImage->ny; ++y){

         yPre=static_cast<int>(static_cast<DTYPE>(y)/gridVoxelSpacing[1]);
         basis=static_cast<DTYPE>(y)/gridVoxelSpacing[1]-static_cast<DTYPE>(yPre);
         if(basis<0.0) basis=0.0; //rounding error
         get_BSplineBasisValues<DTYPE>(basis, basisY, firstY);

         for(x=0; x<referenceImage->nx; ++x){

            xPre=static_cast<int>(static_cast<DTYPE>(x)/gridVoxelSpacing[0]);
            basis=static_cast<DTYPE>(x)/gridVoxelSpacing[0]-static_cast<DTYPE>(xPre);
            if(basis<0.0) basis=0.0; //rounding error
            get_BSplineBasisValues<DTYPE>(basis, basisX, firstX);

            memset(&matrix, 0, sizeof(mat33));

            for(c=0; c<4; c++){
               for(b=0; b<4; b++){
                  for(a=0; a<4; a++){
                     index = ((zPre+c)*splineControlPoint->ny+yPre+b)*splineControlPoint->nx+xPre+a;
                     splineCoeffX = splinePtrX[index];
                     splineCoeffY = splinePtrY[index];
                     splineCoeffZ = splinePtrZ[index];

                     matrix.m[0][0] += firstX[a]*basisY[b]*basisZ[c]*splineCoeffX;
                     matrix.m[1][0] += basisX[a]*firstY[b]*basisZ[c]*splineCoeffX;
                     matrix.m[2][0] += basisX[a]*basisY[b]*firstZ[c]*splineCoeffX;

                     matrix.m[0][1] += firstX[a]*basisY[b]*basisZ[c]*splineCoeffY;
                     matrix.m[1][1] += basisX[a]*firstY[b]*basisZ[c]*splineCoeffY;
                     matrix.m[2][1] += basisX[a]*basisY[b]*firstZ[c]*splineCoeffY;

                     matrix.m[0][2] += firstX[a]*basisY[b]*basisZ[c]*splineCoeffZ;
                     matrix.m[1][2] += basisX[a]*firstY[b]*basisZ[c]*splineCoeffZ;
                     matrix.m[2][2] += basisX[a]*basisY[b]*firstZ[c]*splineCoeffZ;
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
   return constraintValue / static_cast<double>(voxelNumber*3);
}
/* *************************************************************** */
double reg_spline_linearEnergy(nifti_image *referenceImage,
                               nifti_image *splineControlPoint)
{
   if(splineControlPoint->nz>1){
      switch(splineControlPoint->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         return reg_spline_linearEnergyValue3D<float>(referenceImage, splineControlPoint);
      case NIFTI_TYPE_FLOAT64:
         return reg_spline_linearEnergyValue3D<double>(referenceImage, splineControlPoint);
      default:
         reg_print_fct_error("reg_spline_linearEnergyValue3D");
         reg_print_msg_error("Only implemented for single or double precision images");
         reg_exit();
      }
   }
   else{
      switch(splineControlPoint->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         return reg_spline_linearEnergyValue2D<float>(referenceImage, splineControlPoint);
      case NIFTI_TYPE_FLOAT64:
         return reg_spline_linearEnergyValue2D<double>(referenceImage, splineControlPoint);
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
void reg_spline_linearEnergyGradient2D(nifti_image *referenceImage,
                                       nifti_image *splineControlPoint,
                                       nifti_image *gradientImage,
                                       float weight
                                       )
{
   size_t voxelNumber = (size_t)referenceImage->nx *
         referenceImage->ny;
   int a, b, x, y, index, xPre, yPre;
   DTYPE basis;

   DTYPE gridVoxelSpacing[2] ={
      gridVoxelSpacing[0] = splineControlPoint->dx / referenceImage->dx,
      gridVoxelSpacing[1] = splineControlPoint->dy / referenceImage->dy
   };

   // Create pointers to the spline coefficients
   size_t nodeNumber = (size_t)splineControlPoint->nx *
         splineControlPoint->ny * splineControlPoint->nz;
   DTYPE *splinePtrX = static_cast<DTYPE *>(splineControlPoint->data);
   DTYPE *splinePtrY = &splinePtrX[nodeNumber];
   DTYPE splineCoeffX, splineCoeffY;

   // Store the basis values since they are constant as the value is approximated
   // at the control point positions only
   DTYPE basisX[4], basisY[4];
   DTYPE firstX[4], firstY[4];

   mat33 matrix, R;

   DTYPE *gradientXPtr = static_cast<DTYPE *>(gradientImage->data);
   DTYPE *gradientYPtr = &gradientXPtr[nodeNumber];

   DTYPE approxRatio = (DTYPE)weight / (DTYPE)(voxelNumber);
   DTYPE gradValues[2];

   // Matrix to use to convert the gradient from mm to voxel
   mat33 reorientation;
   if(splineControlPoint->sform_code>0)
      reorientation = reg_mat44_to_mat33(&splineControlPoint->sto_ijk);
   else reorientation = reg_mat44_to_mat33(&splineControlPoint->qto_ijk);
   mat33 inv_reorientation = nifti_mat33_inverse(reorientation);

   // Loop over all voxels
   for(y=0; y<referenceImage->ny; ++y){

      yPre=static_cast<int>(static_cast<DTYPE>(y)/gridVoxelSpacing[1]);
      basis=static_cast<DTYPE>(y)/gridVoxelSpacing[1]-static_cast<DTYPE>(yPre);
      if(basis<0.0) basis=0.0; //rounding error
      get_BSplineBasisValues<DTYPE>(basis, basisY, firstY);

      for(x=0; x<referenceImage->nx; ++x){

         xPre=static_cast<int>(static_cast<DTYPE>(x)/gridVoxelSpacing[0]);
         basis=static_cast<DTYPE>(x)/gridVoxelSpacing[0]-static_cast<DTYPE>(xPre);
         if(basis<0.0) basis=0.0; //rounding error
         get_BSplineBasisValues<DTYPE>(basis, basisX, firstX);

         memset(&matrix, 0, sizeof(mat33));

         for(b=0; b<4; b++){
            for(a=0; a<4; a++){
               index = (yPre+b)*splineControlPoint->nx+xPre+a;
               splineCoeffX = splinePtrX[index];
               splineCoeffY = splinePtrY[index];

               matrix.m[0][0] += firstX[a]*basisY[b]*splineCoeffX;
               matrix.m[1][0] += basisX[a]*firstY[b]*splineCoeffX;

               matrix.m[0][1] += firstX[a]*basisY[b]*splineCoeffY;
               matrix.m[1][1] += basisX[a]*firstY[b]*splineCoeffY;
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
         for(b=0; b<4; b++){
            for(a=0; a<4; a++){
               index = (yPre+b)*splineControlPoint->nx+xPre+a;
               gradValues[0] = -2.0*matrix.m[0][0] *
                     firstX[3-a]*basisY[3-b];
               gradValues[1] = -2.0*matrix.m[1][1] *
                     basisX[3-a]*firstY[3-b];
               gradientXPtr[index] += approxRatio *
                     ( inv_reorientation.m[0][0]*gradValues[0]
                     + inv_reorientation.m[0][1]*gradValues[1]);
               gradientYPtr[index] += approxRatio *
                     ( inv_reorientation.m[1][0]*gradValues[0]
                     + inv_reorientation.m[1][1]*gradValues[1]);
            } // a
         } // b
      }
   }
   return;
}
/* *************************************************************** */
template <class DTYPE>
void reg_spline_linearEnergyGradient3D(nifti_image *referenceImage,
                                       nifti_image *splineControlPoint,
                                       nifti_image *gradientImage,
                                       float weight
                                       )
{
   size_t voxelNumber = (size_t)referenceImage->nx *
         referenceImage->ny * referenceImage->nz;
   int a, b, c, x, y, z, index, xPre, yPre, zPre;
   DTYPE basis;


   DTYPE gridVoxelSpacing[3] ={
      gridVoxelSpacing[0] = splineControlPoint->dx / referenceImage->dx,
      gridVoxelSpacing[1] = splineControlPoint->dy / referenceImage->dy,
      gridVoxelSpacing[2] = splineControlPoint->dz / referenceImage->dz
   };

   // Create pointers to the spline coefficients
   size_t nodeNumber = (size_t)splineControlPoint->nx *
         splineControlPoint->ny * splineControlPoint->nz;
   DTYPE *splinePtrX = static_cast<DTYPE *>(splineControlPoint->data);
   DTYPE *splinePtrY = &splinePtrX[nodeNumber];
   DTYPE *splinePtrZ = &splinePtrY[nodeNumber];
   DTYPE splineCoeffX, splineCoeffY, splineCoeffZ;

   // Store the basis values since they are constant as the value is approximated
   // at the control point positions only
   DTYPE basisX[4], basisY[4], basisZ[4];
   DTYPE firstX[4], firstY[4], firstZ[4];

   mat33 matrix, R;

   DTYPE *gradientXPtr = static_cast<DTYPE *>(gradientImage->data);
   DTYPE *gradientYPtr = &gradientXPtr[nodeNumber];
   DTYPE *gradientZPtr = &gradientYPtr[nodeNumber];

   DTYPE approxRatio = (DTYPE)weight / (DTYPE)(voxelNumber);
   DTYPE gradValues[3];

   // Matrix to use to convert the gradient from mm to voxel
   mat33 reorientation;
   if(splineControlPoint->sform_code>0)
      reorientation = reg_mat44_to_mat33(&splineControlPoint->sto_ijk);
   else reorientation = reg_mat44_to_mat33(&splineControlPoint->qto_ijk);
   mat33 inv_reorientation = nifti_mat33_inverse(reorientation);

   // Loop over all voxels
   for(z=0; z<referenceImage->nz; ++z){

      zPre=static_cast<int>(static_cast<DTYPE>(z)/gridVoxelSpacing[2]);
      basis=static_cast<DTYPE>(z)/gridVoxelSpacing[2]-static_cast<DTYPE>(zPre);
      if(basis<0.0) basis=0.0; //rounding error
      get_BSplineBasisValues<DTYPE>(basis, basisZ, firstZ);

      for(y=0; y<referenceImage->ny; ++y){

         yPre=static_cast<int>(static_cast<DTYPE>(y)/gridVoxelSpacing[1]);
         basis=static_cast<DTYPE>(y)/gridVoxelSpacing[1]-static_cast<DTYPE>(yPre);
         if(basis<0.0) basis=0.0; //rounding error
         get_BSplineBasisValues<DTYPE>(basis, basisY, firstY);

         for(x=0; x<referenceImage->nx; ++x){

            xPre=static_cast<int>(static_cast<DTYPE>(x)/gridVoxelSpacing[0]);
            basis=static_cast<DTYPE>(x)/gridVoxelSpacing[0]-static_cast<DTYPE>(xPre);
            if(basis<0.0) basis=0.0; //rounding error
            get_BSplineBasisValues<DTYPE>(basis, basisX, firstX);

            memset(&matrix, 0, sizeof(mat33));

            for(c=0; c<4; c++){
               for(b=0; b<4; b++){
                  for(a=0; a<4; a++){
                     index = ((zPre+c)*splineControlPoint->ny+yPre+b) *
                           splineControlPoint->nx+xPre+a;
                     splineCoeffX = splinePtrX[index];
                     splineCoeffY = splinePtrY[index];
                     splineCoeffZ = splinePtrZ[index];

                     matrix.m[0][0] += firstX[a]*basisY[b]*basisZ[c]*splineCoeffX;
                     matrix.m[1][0] += basisX[a]*firstY[b]*basisZ[c]*splineCoeffX;
                     matrix.m[2][0] += basisX[a]*basisY[b]*firstZ[c]*splineCoeffX;

                     matrix.m[0][1] += firstX[a]*basisY[b]*basisZ[c]*splineCoeffY;
                     matrix.m[1][1] += basisX[a]*firstY[b]*basisZ[c]*splineCoeffY;
                     matrix.m[2][1] += basisX[a]*basisY[b]*firstZ[c]*splineCoeffY;

                     matrix.m[0][2] += firstX[a]*basisY[b]*basisZ[c]*splineCoeffZ;
                     matrix.m[1][2] += basisX[a]*firstY[b]*basisZ[c]*splineCoeffZ;
                     matrix.m[2][2] += basisX[a]*basisY[b]*firstZ[c]*splineCoeffZ;
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
            for(c=0; c<4; c++){
               for(b=0; b<4; b++){
                  for(a=0; a<4; a++){
                     index = ((zPre+c)*splineControlPoint->ny+yPre+b) *
                           splineControlPoint->nx+xPre+a;
                     gradValues[0] = -2.0*matrix.m[0][0] *
                           firstX[3-a]*basisY[3-b]*basisZ[3-c];
                     gradValues[1] = -2.0*matrix.m[1][1] *
                           basisX[3-a]*firstY[3-b]*basisZ[3-c];
                     gradValues[2] = -2.0*matrix.m[2][2] *
                           basisX[3-a]*basisY[3-b]*firstZ[3-c];
                     gradientXPtr[index] += approxRatio *
                           ( inv_reorientation.m[0][0]*gradValues[0]
                           + inv_reorientation.m[0][1]*gradValues[1]
                           + inv_reorientation.m[0][2]*gradValues[2]);
                     gradientYPtr[index] += approxRatio *
                           ( inv_reorientation.m[1][0]*gradValues[0]
                           + inv_reorientation.m[1][1]*gradValues[1]
                           + inv_reorientation.m[1][2]*gradValues[2]);
                     gradientZPtr[index] += approxRatio *
                           ( inv_reorientation.m[2][0]*gradValues[0]
                           + inv_reorientation.m[2][1]*gradValues[1]
                           + inv_reorientation.m[2][2]*gradValues[2]);
                  } // a
               } // b
            } // c
         } // x
      } // y
   } // z
   return;
}
/* *************************************************************** */
void reg_spline_linearEnergyGradient(nifti_image *referenceImage,
                                     nifti_image *splineControlPoint,
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
         reg_spline_linearEnergyGradient3D<float>
               (referenceImage, splineControlPoint, gradientImage, weight);
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_spline_linearEnergyGradient3D<double>
               (referenceImage, splineControlPoint, gradientImage, weight);
         break;
      default:
         reg_print_fct_error("reg_spline_linearEnergyGradient3D");
         reg_print_msg_error("Only implemented for single or double precision images");
         reg_exit();
      }
   }
   else{
      switch(splineControlPoint->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         reg_spline_linearEnergyGradient2D<float>
               (referenceImage, splineControlPoint, gradientImage, weight);
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_spline_linearEnergyGradient2D<double>
               (referenceImage, splineControlPoint, gradientImage, weight);
         break;
      default:
         reg_print_fct_error("reg_spline_linearEnergyGradient2D");
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
   int x, y, a, b, i, index;

   // Create pointers to the spline coefficients
   DTYPE * splinePtrX = static_cast<DTYPE *>(splineControlPoint->data);
   DTYPE * splinePtrY = &splinePtrX[nodeNumber];

   // Store the basis values since they are constant as the value is approximated
   // at the control point positions only
   DTYPE basisX[9];
   DTYPE basisY[9];
   set_first_order_basis_values(basisX, basisY);

   // Matrix to use to convert the gradient from mm to voxel
   mat33 reorientation;
   if(splineControlPoint->sform_code>0)
      reorientation = reg_mat44_to_mat33(&splineControlPoint->sto_ijk);
   else reorientation = reg_mat44_to_mat33(&splineControlPoint->qto_ijk);
   mat33 inv_reorientation = nifti_mat33_inverse(reorientation);

   DTYPE splineCoeffX;
   DTYPE splineCoeffY;

   mat33 matrix, R;

   DTYPE *gradientXPtr = static_cast<DTYPE *>(gradientImage->data);
   DTYPE *gradientYPtr = &gradientXPtr[nodeNumber];

   DTYPE approxRatio = (DTYPE)weight / (DTYPE)(nodeNumber);
   DTYPE gradValues[2];

#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(splineControlPoint, splinePtrX, splinePtrY, \
   basisX, basisY, reorientation, inv_reorientation, \
   gradientXPtr, gradientYPtr, approxRatio) \
   private(x, y, a, b, i, index, gradValues, \
   splineCoeffX, splineCoeffY, matrix, R)
#endif
   for(y=1; y<splineControlPoint->ny-1; y++)
   {
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
            } // a
         } // b
         // Convert from mm to voxel
         matrix = nifti_mat33_mul(reorientation, matrix);
         // Removing the rotation component
         R = nifti_mat33_inverse(nifti_mat33_polar(matrix));
         matrix = nifti_mat33_mul(R, matrix);
         // Convert to displacement
         --matrix.m[0][0];
         --matrix.m[1][1];
         i=8;
         for(b=-1; b<2; b++){
            for(a=-1; a<2; a++){
               index=(y+b)*splineControlPoint->nx+x+a;
               gradValues[0] = -2.0*matrix.m[0][0]*basisX[i];
               gradValues[1] = -2.0*matrix.m[1][1]*basisY[i];

#ifdef _OPENMP
               #pragma omp atomic
#endif
               gradientXPtr[index] += approxRatio *
                     ( inv_reorientation.m[0][0]*gradValues[0]
                     + inv_reorientation.m[0][1]*gradValues[1]);
#ifdef _OPENMP
               #pragma omp atomic
#endif
               gradientYPtr[index] += approxRatio *
                     ( inv_reorientation.m[1][0]*gradValues[0]
                     + inv_reorientation.m[1][1]*gradValues[1]);
               --i;
            } // a
         } // b
      } // x
   } // y

   return;
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
   int x, y, z, a, b, c, i, index;

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

   // Matrix to use to convert the gradient from mm to voxel
   mat33 reorientation;
   if(splineControlPoint->sform_code>0)
      reorientation = reg_mat44_to_mat33(&splineControlPoint->sto_ijk);
   else reorientation = reg_mat44_to_mat33(&splineControlPoint->qto_ijk);
   mat33 inv_reorientation = nifti_mat33_inverse(reorientation);

   DTYPE splineCoeffX;
   DTYPE splineCoeffY;
   DTYPE splineCoeffZ;

   mat33 matrix, R;

   DTYPE *gradientXPtr = static_cast<DTYPE *>(gradientImage->data);
   DTYPE *gradientYPtr = &gradientXPtr[nodeNumber];
   DTYPE *gradientZPtr = &gradientYPtr[nodeNumber];

   DTYPE approxRatio = (DTYPE)weight / (DTYPE)(nodeNumber);
   DTYPE gradValues[3];

   for(z=1; z<splineControlPoint->nz-1; z++)
   {
      for(y=1; y<splineControlPoint->ny-1; y++)
      {
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
            i=26;
            for(c=-1; c<2; c++){
               for(b=-1; b<2; b++){
                  for(a=-1; a<2; a++){
                     index=((z+c)*splineControlPoint->ny+y+b)*splineControlPoint->nx+x+a;
                     gradValues[0] = -2.0*matrix.m[0][0]*basisX[i];
                     gradValues[1] = -2.0*matrix.m[1][1]*basisY[i];
                     gradValues[2] = -2.0*matrix.m[2][2]*basisZ[i];

                     gradientXPtr[index] += approxRatio *
                           ( inv_reorientation.m[0][0]*gradValues[0]
                           + inv_reorientation.m[0][1]*gradValues[1]
                           + inv_reorientation.m[0][2]*gradValues[2]);

                     gradientYPtr[index] += approxRatio *
                           ( inv_reorientation.m[1][0]*gradValues[0]
                           + inv_reorientation.m[1][1]*gradValues[1]
                           + inv_reorientation.m[1][2]*gradValues[2]);

                     gradientZPtr[index] += approxRatio *
                           ( inv_reorientation.m[2][0]*gradValues[0]
                           + inv_reorientation.m[2][1]*gradValues[1]
                           + inv_reorientation.m[2][2]*gradValues[2]);
                     --i;
                  } // a
               } // b
            } // c
         } // x
      } // y
   } // z
   return;
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
template <class DTYPE>
double reg_defField_linearEnergyValue2D(nifti_image *deformationField)
{
   size_t voxelNumber = (size_t)deformationField->nx *
         deformationField->ny;
   int a, b, x, y, X, Y, index;
   DTYPE basis[2]={1,0};
   DTYPE first[2]={-1,1};

   double constraintValue = 0.;
   double currentValue;

   // Create pointers to the deformation field
   DTYPE *defPtrX = static_cast<DTYPE *>(deformationField->data);
   DTYPE *defPtrY = &defPtrX[voxelNumber];
   DTYPE defX, defY;

   mat33 matrix, R;

   // Matrix to use to convert the gradient from mm to voxel
   mat33 reorientation;
   if(deformationField->sform_code>0)
      reorientation = reg_mat44_to_mat33(&deformationField->sto_ijk);
   else reorientation = reg_mat44_to_mat33(&deformationField->qto_ijk);

   for(y=0; y<deformationField->ny; ++y){
      Y=(y!=deformationField->ny-1)?y:y-1;
      for(x=0; x<deformationField->nx; ++x){
         X=(x!=deformationField->nx-1)?x:x-1;

         memset(&matrix, 0, sizeof(mat33));

         for(b=0; b<2; b++){
            for(a=0; a<2; a++){
               index = (Y+b)*deformationField->nx+X+a;
               defX = defPtrX[index];
               defY = defPtrY[index];

               matrix.m[0][0] += first[a]*basis[b]*defX;
               matrix.m[1][0] += basis[a]*first[b]*defX;
               matrix.m[0][1] += first[a]*basis[b]*defY;
               matrix.m[1][1] += basis[a]*first[b]*defY;
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
   return constraintValue / static_cast<double>(deformationField->nvox);
}
/* *************************************************************** */
template <class DTYPE>
double reg_defField_linearEnergyValue3D(nifti_image *deformationField)
{
   size_t voxelNumber = (size_t)deformationField->nx *
         deformationField->ny * deformationField->nz;
   int a, b, c, x, y, z, X, Y, Z, index;
   DTYPE basis[2]={1,0};
   DTYPE first[2]={-1,1};

   double constraintValue = 0.;
   double currentValue;

   // Create pointers to the deformation field
   DTYPE *defPtrX = static_cast<DTYPE *>(deformationField->data);
   DTYPE *defPtrY = &defPtrX[voxelNumber];
   DTYPE *defPtrZ = &defPtrY[voxelNumber];
   DTYPE defX, defY, defZ;

   mat33 matrix, R;

   // Matrix to use to convert the gradient from mm to voxel
   mat33 reorientation;
   if(deformationField->sform_code>0)
      reorientation = reg_mat44_to_mat33(&deformationField->sto_ijk);
   else reorientation = reg_mat44_to_mat33(&deformationField->qto_ijk);

   for(z=0; z<deformationField->nz; ++z){
      Z=(z!=deformationField->nz-1)?z:z-1;
      for(y=0; y<deformationField->ny; ++y){
         Y=(y!=deformationField->ny-1)?y:y-1;
         for(x=0; x<deformationField->nx; ++x){
            X=(x!=deformationField->nx-1)?x:x-1;

            memset(&matrix, 0, sizeof(mat33));

            for(c=0; c<2; c++){
               for(b=0; b<2; b++){
                  for(a=0; a<2; a++){
                     index = ((Z+c)*deformationField->ny+Y+b)*deformationField->nx+X+a;
                     defX = defPtrX[index];
                     defY = defPtrY[index];
                     defZ = defPtrZ[index];

                     matrix.m[0][0] += first[a]*basis[b]*basis[c]*defX;
                     matrix.m[1][0] += basis[a]*first[b]*basis[c]*defX;
                     matrix.m[2][0] += basis[a]*basis[b]*first[c]*defX;

                     matrix.m[0][1] += first[a]*basis[b]*basis[c]*defY;
                     matrix.m[1][1] += basis[a]*first[b]*basis[c]*defY;
                     matrix.m[2][1] += basis[a]*basis[b]*first[c]*defY;

                     matrix.m[0][2] += first[a]*basis[b]*basis[c]*defZ;
                     matrix.m[1][2] += basis[a]*first[b]*basis[c]*defZ;
                     matrix.m[2][2] += basis[a]*basis[b]*first[c]*defZ;
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
   return constraintValue / static_cast<double>(deformationField->nvox);
}
/* *************************************************************** */
double reg_defField_linearEnergy(nifti_image *deformationField)
{
   if(deformationField->nz>1){
      switch(deformationField->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         return reg_defField_linearEnergyValue3D<float>(deformationField);
      case NIFTI_TYPE_FLOAT64:
         return reg_defField_linearEnergyValue3D<double>(deformationField);
      default:
         reg_print_fct_error("reg_defField_linearEnergyValue3D");
         reg_print_msg_error("Only implemented for single or double precision images");
         reg_exit();
      }
   }
   else{
      switch(deformationField->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         return reg_defField_linearEnergyValue2D<float>(deformationField);
      case NIFTI_TYPE_FLOAT64:
         return reg_defField_linearEnergyValue2D<double>(deformationField);
      default:
         reg_print_fct_error("reg_defField_linearEnergyValue2D");
         reg_print_msg_error("Only implemented for single or double precision images");
         reg_exit();
      }
   }
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_defField_linearEnergyGradient2D(nifti_image *deformationField,
                                         nifti_image *gradientImage,
                                         float weight)
{
   size_t voxelNumber = (size_t)deformationField->nx *
         deformationField->ny;
   int a, b, x, y, X, Y, index;
   DTYPE basis[2]={1,0};
   DTYPE first[2]={-1,1};

   // Create pointers to the deformation field
   DTYPE *defPtrX = static_cast<DTYPE *>(deformationField->data);
   DTYPE *defPtrY = &defPtrX[voxelNumber];
   DTYPE defX, defY;

   mat33 matrix, R;

   DTYPE *gradientXPtr = static_cast<DTYPE *>(gradientImage->data);
   DTYPE *gradientYPtr = &gradientXPtr[voxelNumber];

   DTYPE approxRatio = (DTYPE)weight / (DTYPE)(voxelNumber);
   DTYPE gradValues[2];

   // Matrix to use to convert the gradient from mm to voxel
   mat33 reorientation;
   if(deformationField->sform_code>0)
      reorientation = reg_mat44_to_mat33(&deformationField->sto_ijk);
   else reorientation = reg_mat44_to_mat33(&deformationField->qto_ijk);
   mat33 inv_reorientation = nifti_mat33_inverse(reorientation);

   for(y=0; y<deformationField->ny; ++y){
      Y=(y!=deformationField->ny-1)?y:y-1;
      for(x=0; x<deformationField->nx; ++x){
         X=(x!=deformationField->nx-1)?x:x-1;

         memset(&matrix, 0, sizeof(mat33));

         for(b=0; b<2; b++){
            for(a=0; a<2; a++){
               index = (Y+b)*deformationField->nx+X+a;
               defX = defPtrX[index];
               defY = defPtrY[index];

               matrix.m[0][0] += first[a]*basis[b]*defX;
               matrix.m[1][0] += basis[a]*first[b]*defX;
               matrix.m[0][1] += first[a]*basis[b]*defY;
               matrix.m[1][1] += basis[a]*first[b]*defY;
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

         for(b=0; b<2; b++){
            for(a=0; a<2; a++){
               index = (Y+b)*deformationField->nx+X+a;
               gradValues[0] = -2.0*matrix.m[0][0] *
                     first[1-a]*basis[1-b];
               gradValues[1] = -2.0*matrix.m[1][1] *
                     basis[1-a]*first[1-b];
               gradientXPtr[index] += approxRatio *
                     ( inv_reorientation.m[0][0]*gradValues[0]
                     + inv_reorientation.m[0][1]*gradValues[1]);
               gradientYPtr[index] += approxRatio *
                     ( inv_reorientation.m[1][0]*gradValues[0]
                     + inv_reorientation.m[1][1]*gradValues[1]);
            } // a
         } // b
      }
   }
}
/* *************************************************************** */
template <class DTYPE>
void reg_defField_linearEnergyGradient3D(nifti_image *deformationField,
                                         nifti_image *gradientImage,
                                         float weight)
{
   size_t voxelNumber = (size_t)deformationField->nx *
         deformationField->ny * deformationField->nz;
   int a, b, c, x, y, z, X, Y, Z, index;
   DTYPE basis[2]={1,0};
   DTYPE first[2]={-1,1};

   // Create pointers to the deformation field
   DTYPE *defPtrX = static_cast<DTYPE *>(deformationField->data);
   DTYPE *defPtrY = &defPtrX[voxelNumber];
   DTYPE *defPtrZ = &defPtrY[voxelNumber];
   DTYPE defX, defY, defZ;

   mat33 matrix, R;

   DTYPE *gradientXPtr = static_cast<DTYPE *>(gradientImage->data);
   DTYPE *gradientYPtr = &gradientXPtr[voxelNumber];
   DTYPE *gradientZPtr = &gradientYPtr[voxelNumber];

   DTYPE approxRatio = (DTYPE)weight / (DTYPE)(voxelNumber);
   DTYPE gradValues[3];

   // Matrix to use to convert the gradient from mm to voxel
   mat33 reorientation;
   if(deformationField->sform_code>0)
      reorientation = reg_mat44_to_mat33(&deformationField->sto_ijk);
   else reorientation = reg_mat44_to_mat33(&deformationField->qto_ijk);
   mat33 inv_reorientation = nifti_mat33_inverse(reorientation);

   for(z=0; z<deformationField->nz; ++z){
      Z=(z!=deformationField->nz-1)?z:z-1;
      for(y=0; y<deformationField->ny; ++y){
         Y=(y!=deformationField->ny-1)?y:y-1;
         for(x=0; x<deformationField->nx; ++x){
            X=(x!=deformationField->nx-1)?x:x-1;

            memset(&matrix, 0, sizeof(mat33));

            for(c=0; c<2; c++){
               for(b=0; b<2; b++){
                  for(a=0; a<2; a++){
                     index = ((Z+c)*deformationField->ny+Y+b)*deformationField->nx+X+a;
                     defX = defPtrX[index];
                     defY = defPtrY[index];
                     defZ = defPtrZ[index];

                     matrix.m[0][0] += first[a]*basis[b]*basis[c]*defX;
                     matrix.m[1][0] += basis[a]*first[b]*basis[c]*defX;
                     matrix.m[2][0] += basis[a]*basis[b]*first[c]*defX;

                     matrix.m[0][1] += first[a]*basis[b]*basis[c]*defY;
                     matrix.m[1][1] += basis[a]*first[b]*basis[c]*defY;
                     matrix.m[2][1] += basis[a]*basis[b]*first[c]*defY;

                     matrix.m[0][2] += first[a]*basis[b]*basis[c]*defZ;
                     matrix.m[1][2] += basis[a]*first[b]*basis[c]*defZ;
                     matrix.m[2][2] += basis[a]*basis[b]*first[c]*defZ;
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
            for(c=0; c<2; c++){
               for(b=0; b<2; b++){
                  for(a=0; a<2; a++){
                     index = ((Z+c)*deformationField->ny+Y+b) *
                           deformationField->nx+X+a;
                     gradValues[0] = -2.0*matrix.m[0][0] *
                           first[1-a]*basis[1-b]*basis[1-c];
                     gradValues[1] = -2.0*matrix.m[1][1] *
                           basis[1-a]*first[1-b]*basis[1-c];
                     gradValues[2] = -2.0*matrix.m[2][2] *
                           basis[1-a]*basis[1-b]*first[1-c];
                     gradientXPtr[index] += approxRatio *
                           ( inv_reorientation.m[0][0]*gradValues[0]
                           + inv_reorientation.m[0][1]*gradValues[1]
                           + inv_reorientation.m[0][2]*gradValues[2]);
                     gradientYPtr[index] += approxRatio *
                           ( inv_reorientation.m[1][0]*gradValues[0]
                           + inv_reorientation.m[1][1]*gradValues[1]
                           + inv_reorientation.m[1][2]*gradValues[2]);
                     gradientZPtr[index] += approxRatio *
                           ( inv_reorientation.m[2][0]*gradValues[0]
                           + inv_reorientation.m[2][1]*gradValues[1]
                           + inv_reorientation.m[2][2]*gradValues[2]);
                  } // a
               } // b
            } // c
         }
      }
   }
}
/* *************************************************************** */
void reg_defField_linearEnergyGradient(nifti_image *deformationField,
                                       nifti_image *gradientImage,
                                       float weight)
{
   if(deformationField->nz>1){
      switch(deformationField->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         reg_defField_linearEnergyGradient3D<float>
               (deformationField, gradientImage, weight);
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_defField_linearEnergyGradient3D<double>
               (deformationField, gradientImage, weight);
         break;
      default:
         reg_print_fct_error("reg_defField_linearEnergyGradient3D");
         reg_print_msg_error("Only implemented for single or double precision images");
         reg_exit();
      }
   }
   else{
      switch(deformationField->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         reg_defField_linearEnergyGradient2D<float>
               (deformationField, gradientImage, weight);
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_defField_linearEnergyGradient2D<double>
               (deformationField, gradientImage, weight);
         break;
      default:
         reg_print_fct_error("reg_defField_linearEnergyGradient2D");
         reg_print_msg_error("Only implemented for single or double precision images");
         reg_exit();
      }
   }
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
double reg_spline_getLandmarkDistance_core(nifti_image *controlPointImage,
                                           size_t landmarkNumber,
                                           float *landmarkReference,
                                           float *landmarkFloating)
{
   int imageDim=controlPointImage->nz>1?3:2;
   size_t controlPointNumber = (size_t)controlPointImage->nx *
         controlPointImage->ny * controlPointImage->nz;
   double constraintValue=0.;
   size_t l, index;
   float ref_position[4];
   float def_position[4];
   float flo_position[4];
   int previous[3], a, b, c;
   DTYPE basisX[4], basisY[4], basisZ[4], basis;
   mat44 *gridRealToVox = &(controlPointImage->qto_ijk);
   if(controlPointImage->sform_code>0)
      gridRealToVox = &(controlPointImage->sto_ijk);
   DTYPE *gridPtrX = static_cast<DTYPE *>(controlPointImage->data);
   DTYPE *gridPtrY = &gridPtrX[controlPointNumber];
   DTYPE *gridPtrZ=NULL;
   if(imageDim>2)
      gridPtrZ = &gridPtrY[controlPointNumber];

   // Loop over all landmarks
   for(l=0;l<landmarkNumber;++l){
      // fetch the initial positions
      ref_position[0]=landmarkReference[l*imageDim];
      flo_position[0]=landmarkFloating[l*imageDim];
      ref_position[1]=landmarkReference[l*imageDim+1];
      flo_position[1]=landmarkFloating[l*imageDim+1];
      if(imageDim>2){
         ref_position[2]=landmarkReference[l*imageDim+2];
         flo_position[2]=landmarkFloating[l*imageDim+2];
      }
      else ref_position[2]=flo_position[2]=0.f;
      ref_position[3]=flo_position[3]=1.f;
      // Convert the reference position to voxel in the control point grid space
      reg_mat44_mul(gridRealToVox, ref_position, def_position);



      // Extract the corresponding nodes
      previous[0]=static_cast<int>(reg_floor(def_position[0]))-1;
      previous[1]=static_cast<int>(reg_floor(def_position[1]))-1;
      previous[2]=static_cast<int>(reg_floor(def_position[2]))-1;
      // Check that the specified landmark belongs to the input image
      if(previous[0]>-1 && previous[0]+3<controlPointImage->nx &&
         previous[1]>-1 && previous[1]+3<controlPointImage->ny &&
         ((previous[2]>-1 && previous[2]+3<controlPointImage->nz) || imageDim==2)){
         // Extract the corresponding basis values
         get_BSplineBasisValues<DTYPE>(def_position[0] - 1.f -(DTYPE)previous[0], basisX);
         get_BSplineBasisValues<DTYPE>(def_position[1] - 1.f -(DTYPE)previous[1], basisY);
         get_BSplineBasisValues<DTYPE>(def_position[2] - 1.f -(DTYPE)previous[2], basisZ);
         def_position[0]=0.f;
         def_position[1]=0.f;
         def_position[2]=0.f;
         if(imageDim>2){
            for(c=0;c<4;++c){
               for(b=0;b<4;++b){
                  for(a=0;a<4;++a){
                     index = ((previous[2]+c)*controlPointImage->ny+previous[1]+b) *
                           controlPointImage->nx+previous[0]+a;
                     basis = basisX[a] * basisY[b] * basisZ[c];
                     def_position[0] += gridPtrX[index] * basis;
                     def_position[1] += gridPtrY[index] * basis;
                     def_position[2] += gridPtrZ[index] * basis;
                  }
               }
            }
         }
         else{
            for(b=0;b<4;++b){
               for(a=0;a<4;++a){
                  index = (previous[1]+b)*controlPointImage->nx+previous[0]+a;
                  basis = basisX[a] * basisY[b];
                  def_position[0] += gridPtrX[index] * basis;
                  def_position[1] += gridPtrY[index] * basis;
               }
            }
         }
         constraintValue += reg_pow2(flo_position[0]-def_position[0]);
         constraintValue += reg_pow2(flo_position[1]-def_position[1]);
         if(imageDim>2)
            constraintValue += reg_pow2(flo_position[2]-def_position[2]);
      }
      else{
         char warning_text[255];
         if(imageDim>2)
            sprintf(warning_text, "The current landmark at position %g %g %g is ignored",
                    ref_position[0], ref_position[1], ref_position[2]);
         else
            sprintf(warning_text, "The current landmark at position %g %g is ignored",
                    ref_position[0], ref_position[1]);
         reg_print_msg_warn(warning_text);
         reg_print_msg_warn("as it is not in the space of the reference image");
      }
   }
   return constraintValue;
}
/* *************************************************************** */
double reg_spline_getLandmarkDistance(nifti_image *controlPointImage,
                                      size_t landmarkNumber,
                                      float *landmarkReference,
                                      float *landmarkFloating)
{
   if(controlPointImage->intent_p1!=CUB_SPLINE_GRID){
      reg_print_fct_error("reg_spline_getLandmarkDistance");
      reg_print_msg_error("This function is only implemented for control point grid within an Euclidean setting for now");
      reg_exit();
   }
   switch(controlPointImage->datatype)
   {
   case NIFTI_TYPE_FLOAT32:
      return reg_spline_getLandmarkDistance_core<float>
            (controlPointImage, landmarkNumber, landmarkReference, landmarkFloating);
      break;
   case NIFTI_TYPE_FLOAT64:
      return reg_spline_getLandmarkDistance_core<double>
            (controlPointImage, landmarkNumber, landmarkReference, landmarkFloating);
      break;
   default:
      reg_print_fct_error("reg_spline_getLandmarkDistance_core");
      reg_print_msg_error("Only implemented for single or double precision images");
      reg_exit();
   }
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_spline_getLandmarkDistanceGradient_core(nifti_image *controlPointImage,
                                                 nifti_image *gradientImage,
                                                 size_t landmarkNumber,
                                                 float *landmarkReference,
                                                 float *landmarkFloating,
                                                 float weight)
{
   int imageDim=controlPointImage->nz>1?3:2;
   size_t controlPointNumber = (size_t)controlPointImage->nx *
         controlPointImage->ny * controlPointImage->nz;
   size_t l, index;
   float ref_position[3];
   float def_position[3];
   float flo_position[3];
   int previous[3], a, b, c;
   DTYPE basisX[4], basisY[4], basisZ[4], basis;
   mat44 *gridRealToVox = &(controlPointImage->qto_ijk);
   if(controlPointImage->sform_code>0)
      gridRealToVox = &(controlPointImage->sto_ijk);
   DTYPE *gridPtrX = static_cast<DTYPE *>(controlPointImage->data);
   DTYPE *gradPtrX = static_cast<DTYPE *>(gradientImage->data);
   DTYPE *gridPtrY = &gridPtrX[controlPointNumber];
   DTYPE *gradPtrY = &gradPtrX[controlPointNumber];
   DTYPE *gridPtrZ=NULL;
   DTYPE *gradPtrZ=NULL;
   if(imageDim>2){
      gridPtrZ = &gridPtrY[controlPointNumber];
      gradPtrZ = &gradPtrY[controlPointNumber];
   }

   // Loop over all landmarks
   for(l=0;l<landmarkNumber;++l){
      // fetch the initial positions
      ref_position[0]=landmarkReference[l*imageDim];
      flo_position[0]=landmarkFloating[l*imageDim];
      ref_position[1]=landmarkReference[l*imageDim+1];
      flo_position[1]=landmarkFloating[l*imageDim+1];
      if(imageDim>2){
         ref_position[2]=landmarkReference[l*imageDim+2];
         flo_position[2]=landmarkFloating[l*imageDim+2];
      }
      else ref_position[2]=flo_position[2]=0.f;
      // Convert the reference position to voxel in the control point grid space
      reg_mat44_mul(gridRealToVox, ref_position, def_position);
      if(imageDim==2) def_position[2]=0.f;
      // Extract the corresponding nodes
      previous[0]=static_cast<int>(reg_floor(def_position[0]))-1;
      previous[1]=static_cast<int>(reg_floor(def_position[1]))-1;
      previous[2]=static_cast<int>(reg_floor(def_position[2]))-1;
      // Check that the specified landmark belongs to the input image
      if(previous[0]>-1 && previous[0]+3<controlPointImage->nx &&
         previous[1]>-1 && previous[1]+3<controlPointImage->ny &&
         ((previous[2]>-1 && previous[2]+3<controlPointImage->nz) || imageDim==2)){
         // Extract the corresponding basis values
         get_BSplineBasisValues<DTYPE>(def_position[0] - 1.f -(DTYPE)previous[0], basisX);
         get_BSplineBasisValues<DTYPE>(def_position[1] - 1.f -(DTYPE)previous[1], basisY);
         get_BSplineBasisValues<DTYPE>(def_position[2] - 1.f -(DTYPE)previous[2], basisZ);
         def_position[0]=0.f;
         def_position[1]=0.f;
         def_position[2]=0.f;
         if(imageDim>2){
            for(c=0;c<4;++c){
               for(b=0;b<4;++b){
                  for(a=0;a<4;++a){
                     index = ((previous[2]+c)*controlPointImage->ny+previous[1]+b) *
                           controlPointImage->nx+previous[0]+a;
                     basis = basisX[a] * basisY[b] * basisZ[c];
                     def_position[0] += gridPtrX[index] * basis;
                     def_position[1] += gridPtrY[index] * basis;
                     def_position[2] += gridPtrZ[index] * basis;
                  }
               }
            }
         }
         else{
            for(b=0;b<4;++b){
               for(a=0;a<4;++a){
                  index = (previous[1]+b)*controlPointImage->nx+previous[0]+a;
                  basis = basisX[a] * basisY[b];
                  def_position[0] += gridPtrX[index] * basis;
                  def_position[1] += gridPtrY[index] * basis;
               }
            }
         }
         def_position[0]=flo_position[0]-def_position[0];
         def_position[1]=flo_position[1]-def_position[1];
         if(imageDim>2)
            def_position[2]=flo_position[2]-def_position[2];
         if(imageDim>2){
            for(c=0;c<4;++c){
               for(b=0;b<4;++b){
                  for(a=0;a<4;++a){
                     index = ((previous[2]+c)*controlPointImage->ny+previous[1]+b) *
                           controlPointImage->nx+previous[0]+a;
                     basis = basisX[a] * basisY[b] * basisZ[c] * weight;
                     gradPtrX[index] -= def_position[0] * basis;
                     gradPtrY[index] -= def_position[1] * basis;
                     gradPtrZ[index] -= def_position[2] * basis;
                  }
               }
            }
         }
         else{
            for(b=0;b<4;++b){
               for(a=0;a<4;++a){
                  index = (previous[1]+b)*controlPointImage->nx+previous[0]+a;
                  basis = basisX[a] * basisY[b] * weight;
                  gradPtrX[index] -= def_position[0] * basis;
                  gradPtrY[index] -= def_position[1] * basis;
               }
            }
         }
      }
      else{
         char warning_text[255];
         if(imageDim>2)
            sprintf(warning_text, "The current landmark at position %g %g %g is ignored",
                    ref_position[0], ref_position[1], ref_position[2]);
         else
            sprintf(warning_text, "The current landmark at position %g %g is ignored",
                    ref_position[0], ref_position[1]);
         reg_print_msg_warn(warning_text);
         reg_print_msg_warn("as it is not in the space of the reference image");
      }
   }
}
/* *************************************************************** */
void reg_spline_getLandmarkDistanceGradient(nifti_image *controlPointImage,
                                            nifti_image *gradientImage,
                                            size_t landmarkNumber,
                                            float *landmarkReference,
                                            float *landmarkFloating,
                                            float weight)
{
   if(controlPointImage->intent_p1!=CUB_SPLINE_GRID){
      reg_print_fct_error("reg_spline_getLandmarkDistance");
      reg_print_msg_error("This function is only implemented for control point grid within an Euclidean setting for now");
      reg_exit();
   }
   switch(controlPointImage->datatype)
   {
   case NIFTI_TYPE_FLOAT32:
      reg_spline_getLandmarkDistanceGradient_core<float>
            (controlPointImage, gradientImage, landmarkNumber, landmarkReference, landmarkFloating, weight);
      break;
   case NIFTI_TYPE_FLOAT64:
      reg_spline_getLandmarkDistanceGradient_core<double>
            (controlPointImage, gradientImage, landmarkNumber, landmarkReference, landmarkFloating, weight);
      break;
   default:
      reg_print_fct_error("reg_spline_getLandmarkDistanceGradient_core");
      reg_print_msg_error("Only implemented for single or double precision images");
      reg_exit();
   }
}
/* *************************************************************** */
/* *************************************************************** */
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
/* *************************************************************** */
