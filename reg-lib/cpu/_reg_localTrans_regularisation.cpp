/*
 *  _reg_localTransformation_be.cpp
 *
 *
 *  Created by Marc Modat on 10/05/2011.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void set_first_order_basis_values(DTYPE *basisX, DTYPE *basisY)
{
   double BASIS[4], FIRST[4];get_BSplineBasisValues<double>(0, BASIS, FIRST);
   int index=0;
   for(int y=0;y<3;++y){
      for(int x=0;x<3;++x){
         basisX[index] = FIRST[x] * BASIS[y];
         basisY[index] = BASIS[x] * FIRST[y];
         printf("basisX[%i]=static_cast<DTYPE>(%g);\n", index, basisX[index]);
         printf("basisY[%i]=static_cast<DTYPE>(%g);\n", index, basisY[index]);
         index++;
      }
   }
}
/* *************************************************************** */
template <class DTYPE>
void set_first_order_basis_values(DTYPE *basisX, DTYPE *basisY, DTYPE *basisZ)
{
      basisX[0]=static_cast<DTYPE>(-0.0138889);
      basisY[0]=static_cast<DTYPE>(-0.0138889);
      basisZ[0]=static_cast<DTYPE>(-0.0138889);
      basisX[1]=static_cast<DTYPE>(0);
      basisY[1]=static_cast<DTYPE>(-0.0555556);
      basisZ[1]=static_cast<DTYPE>(-0.0555556);
      basisX[2]=static_cast<DTYPE>(0.0138889);
      basisY[2]=static_cast<DTYPE>(-0.0138889);
      basisZ[2]=static_cast<DTYPE>(-0.0138889);
      basisX[3]=static_cast<DTYPE>(-0.0555556);
      basisY[3]=static_cast<DTYPE>(0);
      basisZ[3]=static_cast<DTYPE>(-0.0555556);
      basisX[4]=static_cast<DTYPE>(0);
      basisY[4]=static_cast<DTYPE>(0);
      basisZ[4]=static_cast<DTYPE>(-0.222222);
      basisX[5]=static_cast<DTYPE>(0.0555556);
      basisY[5]=static_cast<DTYPE>(0);
      basisZ[5]=static_cast<DTYPE>(-0.0555556);
      basisX[6]=static_cast<DTYPE>(-0.0138889);
      basisY[6]=static_cast<DTYPE>(0.0138889);
      basisZ[6]=static_cast<DTYPE>(-0.0138889);
      basisX[7]=static_cast<DTYPE>(0);
      basisY[7]=static_cast<DTYPE>(0.0555556);
      basisZ[7]=static_cast<DTYPE>(-0.0555556);
      basisX[8]=static_cast<DTYPE>(0.0138889);
      basisY[8]=static_cast<DTYPE>(0.0138889);
      basisZ[8]=static_cast<DTYPE>(-0.0138889);
      basisX[9]=static_cast<DTYPE>(-0.0555556);
      basisY[9]=static_cast<DTYPE>(-0.0555556);
      basisZ[9]=static_cast<DTYPE>(0);
      basisX[10]=static_cast<DTYPE>(0);
      basisY[10]=static_cast<DTYPE>(-0.222222);
      basisZ[10]=static_cast<DTYPE>(0);
      basisX[11]=static_cast<DTYPE>(0.0555556);
      basisY[11]=static_cast<DTYPE>(-0.0555556);
      basisZ[11]=static_cast<DTYPE>(0);
      basisX[12]=static_cast<DTYPE>(-0.222222);
      basisY[12]=static_cast<DTYPE>(0);
      basisZ[12]=static_cast<DTYPE>(0);
      basisX[13]=static_cast<DTYPE>(0);
      basisY[13]=static_cast<DTYPE>(0);
      basisZ[13]=static_cast<DTYPE>(0);
      basisX[14]=static_cast<DTYPE>(0.222222);
      basisY[14]=static_cast<DTYPE>(0);
      basisZ[14]=static_cast<DTYPE>(0);
      basisX[15]=static_cast<DTYPE>(-0.0555556);
      basisY[15]=static_cast<DTYPE>(0.0555556);
      basisZ[15]=static_cast<DTYPE>(0);
      basisX[16]=static_cast<DTYPE>(0);
      basisY[16]=static_cast<DTYPE>(0.222222);
      basisZ[16]=static_cast<DTYPE>(0);
      basisX[17]=static_cast<DTYPE>(0.0555556);
      basisY[17]=static_cast<DTYPE>(0.0555556);
      basisZ[17]=static_cast<DTYPE>(0);
      basisX[18]=static_cast<DTYPE>(-0.0138889);
      basisY[18]=static_cast<DTYPE>(-0.0138889);
      basisZ[18]=static_cast<DTYPE>(0.0138889);
      basisX[19]=static_cast<DTYPE>(0);
      basisY[19]=static_cast<DTYPE>(-0.0555556);
      basisZ[19]=static_cast<DTYPE>(0.0555556);
      basisX[20]=static_cast<DTYPE>(0.0138889);
      basisY[20]=static_cast<DTYPE>(-0.0138889);
      basisZ[20]=static_cast<DTYPE>(0.0138889);
      basisX[21]=static_cast<DTYPE>(-0.0555556);
      basisY[21]=static_cast<DTYPE>(0);
      basisZ[21]=static_cast<DTYPE>(0.0555556);
      basisX[22]=static_cast<DTYPE>(0);
      basisY[22]=static_cast<DTYPE>(0);
      basisZ[22]=static_cast<DTYPE>(0.222222);
      basisX[23]=static_cast<DTYPE>(0.0555556);
      basisY[23]=static_cast<DTYPE>(0);
      basisZ[23]=static_cast<DTYPE>(0.0555556);
      basisX[24]=static_cast<DTYPE>(-0.0138889);
      basisY[24]=static_cast<DTYPE>(0.0138889);
      basisZ[24]=static_cast<DTYPE>(0.0138889);
      basisX[25]=static_cast<DTYPE>(0);
      basisY[25]=static_cast<DTYPE>(0.0555556);
      basisZ[25]=static_cast<DTYPE>(0.0555556);
      basisX[26]=static_cast<DTYPE>(0.0138889);
      basisY[26]=static_cast<DTYPE>(0.0138889);
      basisZ[26]=static_cast<DTYPE>(0.0138889);
}
/* *************************************************************** */
template <class DTYPE>
void set_second_order_basis_values(DTYPE *basisXX, DTYPE *basisYY, DTYPE *basisXY)
{
   basisXX[0]=0.166667f;
   basisYY[0]=0.166667f;
   basisXY[0]=0.25f;
   basisXX[1]=-0.333333f;
   basisYY[1]=0.666667f;
   basisXY[1]=-0.f;
   basisXX[2]=0.166667f;
   basisYY[2]=0.166667f;
   basisXY[2]=-0.25f;
   basisXX[3]=0.666667f;
   basisYY[3]=-0.333333f;
   basisXY[3]=-0.f;
   basisXX[4]=-1.33333f;
   basisYY[4]=-1.33333f;
   basisXY[4]=0.f;
   basisXX[5]=0.666667f;
   basisYY[5]=-0.333333f;
   basisXY[5]=0.f;
   basisXX[6]=0.166667f;
   basisYY[6]=0.166667f;
   basisXY[6]=-0.25f;
   basisXX[7]=-0.333333f;
   basisYY[7]=0.666667f;
   basisXY[7]=0.f;
   basisXX[8]=0.166667f;
   basisYY[8]=0.166667f;
   basisXY[8]=0.25f;
}
/* *************************************************************** */
template <class DTYPE>
void set_second_order_basis_values(DTYPE *basisXX, DTYPE *basisYY, DTYPE *basisZZ, DTYPE *basisXY, DTYPE *basisYZ, DTYPE *basisXZ)
{
   basisXX[0]=0.027778f;
   basisYY[0]=0.027778f;
   basisZZ[0]=0.027778f;
   basisXY[0]=0.041667f;
   basisYZ[0]=0.041667f;
   basisXZ[0]=0.041667f;
   basisXX[1]=-0.055556f;
   basisYY[1]=0.111111f;
   basisZZ[1]=0.111111f;
   basisXY[1]=-0.000000f;
   basisYZ[1]=0.166667f;
   basisXZ[1]=-0.000000f;
   basisXX[2]=0.027778f;
   basisYY[2]=0.027778f;
   basisZZ[2]=0.027778f;
   basisXY[2]=-0.041667f;
   basisYZ[2]=0.041667f;
   basisXZ[2]=-0.041667f;
   basisXX[3]=0.111111f;
   basisYY[3]=-0.055556f;
   basisZZ[3]=0.111111f;
   basisXY[3]=-0.000000f;
   basisYZ[3]=-0.000000f;
   basisXZ[3]=0.166667f;
   basisXX[4]=-0.222222f;
   basisYY[4]=-0.222222f;
   basisZZ[4]=0.444444f;
   basisXY[4]=0.000000f;
   basisYZ[4]=-0.000000f;
   basisXZ[4]=-0.000000f;
   basisXX[5]=0.111111f;
   basisYY[5]=-0.055556f;
   basisZZ[5]=0.111111f;
   basisXY[5]=0.000000f;
   basisYZ[5]=-0.000000f;
   basisXZ[5]=-0.166667f;
   basisXX[6]=0.027778f;
   basisYY[6]=0.027778f;
   basisZZ[6]=0.027778f;
   basisXY[6]=-0.041667f;
   basisYZ[6]=-0.041667f;
   basisXZ[6]=0.041667f;
   basisXX[7]=-0.055556f;
   basisYY[7]=0.111111f;
   basisZZ[7]=0.111111f;
   basisXY[7]=0.000000f;
   basisYZ[7]=-0.166667f;
   basisXZ[7]=-0.000000f;
   basisXX[8]=0.027778f;
   basisYY[8]=0.027778f;
   basisZZ[8]=0.027778f;
   basisXY[8]=0.041667f;
   basisYZ[8]=-0.041667f;
   basisXZ[8]=-0.041667f;
   basisXX[9]=0.111111f;
   basisYY[9]=0.111111f;
   basisZZ[9]=-0.055556f;
   basisXY[9]=0.166667f;
   basisYZ[9]=-0.000000f;
   basisXZ[9]=-0.000000f;
   basisXX[10]=-0.222222f;
   basisYY[10]=0.444444f;
   basisZZ[10]=-0.222222f;
   basisXY[10]=-0.000000f;
   basisYZ[10]=-0.000000f;
   basisXZ[10]=0.000000f;
   basisXX[11]=0.111111f;
   basisYY[11]=0.111111f;
   basisZZ[11]=-0.055556f;
   basisXY[11]=-0.166667f;
   basisYZ[11]=-0.000000f;
   basisXZ[11]=0.000000f;
   basisXX[12]=0.444444f;
   basisYY[12]=-0.222222f;
   basisZZ[12]=-0.222222f;
   basisXY[12]=-0.000000f;
   basisYZ[12]=0.000000f;
   basisXZ[12]=-0.000000f;
   basisXX[13]=-0.888889f;
   basisYY[13]=-0.888889f;
   basisZZ[13]=-0.888889f;
   basisXY[13]=0.000000f;
   basisYZ[13]=0.000000f;
   basisXZ[13]=0.000000f;
   basisXX[14]=0.444444f;
   basisYY[14]=-0.222222f;
   basisZZ[14]=-0.222222f;
   basisXY[14]=0.000000f;
   basisYZ[14]=0.000000f;
   basisXZ[14]=0.000000f;
   basisXX[15]=0.111111f;
   basisYY[15]=0.111111f;
   basisZZ[15]=-0.055556f;
   basisXY[15]=-0.166667f;
   basisYZ[15]=0.000000f;
   basisXZ[15]=-0.000000f;
   basisXX[16]=-0.222222f;
   basisYY[16]=0.444444f;
   basisZZ[16]=-0.222222f;
   basisXY[16]=0.000000f;
   basisYZ[16]=0.000000f;
   basisXZ[16]=0.000000f;
   basisXX[17]=0.111111f;
   basisYY[17]=0.111111f;
   basisZZ[17]=-0.055556f;
   basisXY[17]=0.166667f;
   basisYZ[17]=0.000000f;
   basisXZ[17]=0.000000f;
   basisXX[18]=0.027778f;
   basisYY[18]=0.027778f;
   basisZZ[18]=0.027778f;
   basisXY[18]=0.041667f;
   basisYZ[18]=-0.041667f;
   basisXZ[18]=-0.041667f;
   basisXX[19]=-0.055556f;
   basisYY[19]=0.111111f;
   basisZZ[19]=0.111111f;
   basisXY[19]=-0.000000f;
   basisYZ[19]=-0.166667f;
   basisXZ[19]=0.000000f;
   basisXX[20]=0.027778f;
   basisYY[20]=0.027778f;
   basisZZ[20]=0.027778f;
   basisXY[20]=-0.041667f;
   basisYZ[20]=-0.041667f;
   basisXZ[20]=0.041667f;
   basisXX[21]=0.111111f;
   basisYY[21]=-0.055556f;
   basisZZ[21]=0.111111f;
   basisXY[21]=-0.000000f;
   basisYZ[21]=0.000000f;
   basisXZ[21]=-0.166667f;
   basisXX[22]=-0.222222f;
   basisYY[22]=-0.222222f;
   basisZZ[22]=0.444444f;
   basisXY[22]=0.000000f;
   basisYZ[22]=0.000000f;
   basisXZ[22]=0.000000f;
   basisXX[23]=0.111111f;
   basisYY[23]=-0.055556f;
   basisZZ[23]=0.111111f;
   basisXY[23]=0.000000f;
   basisYZ[23]=0.000000f;
   basisXZ[23]=0.166667f;
   basisXX[24]=0.027778f;
   basisYY[24]=0.027778f;
   basisZZ[24]=0.027778f;
   basisXY[24]=-0.041667f;
   basisYZ[24]=0.041667f;
   basisXZ[24]=-0.041667f;
   basisXX[25]=-0.055556f;
   basisYY[25]=0.111111f;
   basisZZ[25]=0.111111f;
   basisXY[25]=0.000000f;
   basisYZ[25]=0.166667f;
   basisXZ[25]=0.000000f;
   basisXX[26]=0.027778f;
   basisYY[26]=0.027778f;
   basisZZ[26]=0.027778f;
   basisXY[26]=0.041667f;
   basisYZ[26]=0.041667f;
   basisXZ[26]=0.041667f;
}
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
   set_second_order_basis_values(basisXX, basisYY, basisXY);

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
   set_second_order_basis_values(basisXX, basisYY, basisZZ, basisXY, basisYZ, basisXZ);

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
         reg_exit(1);
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
         reg_exit(1);
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
   set_second_order_basis_values(basisXX, basisYY, basisXY);

   DTYPE splineCoeffX;
   DTYPE splineCoeffY;
   DTYPE XX_x, YY_x, XY_x;
   DTYPE XX_y, YY_y, XY_y;

   DTYPE *derivativeValues = (DTYPE *)calloc(6*nodeNumber, sizeof(DTYPE));
   DTYPE *derivativeValuesPtr;

   // Compute the bending energy values everywhere but at the boundary
#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(splineControlPoint,splinePtrX,splinePtrY, derivativeValues, \
   basisXX, basisYY, basisXY) \
   private(a, b, i, index, x, y, derivativeValuesPtr, splineCoeffX, splineCoeffY, \
   XX_x, YY_x, XY_x, XX_y, YY_y, XY_y)
#endif
   for(y=1; y<splineControlPoint->ny-1; y++)
   {
      derivativeValuesPtr = &derivativeValues[6*(y*splineControlPoint->nx+1)];
      for(x=1; x<splineControlPoint->nx-1; x++)
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
   set_second_order_basis_values(basisXX, basisYY, basisZZ, basisXY, basisYZ, basisXZ);

   DTYPE splineCoeffX;
   DTYPE splineCoeffY;
   DTYPE splineCoeffZ;
   DTYPE XX_x, YY_x, ZZ_x, XY_x, YZ_x, XZ_x;
   DTYPE XX_y, YY_y, ZZ_y, XY_y, YZ_y, XZ_y;
   DTYPE XX_z, YY_z, ZZ_z, XY_z, YZ_z, XZ_z;

   DTYPE *derivativeValues = (DTYPE *)calloc(18*nodeNumber, sizeof(DTYPE));
   DTYPE *derivativeValuesPtr;

   // Compute the bending energy values everywhere but at the boundary
#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(splineControlPoint,splinePtrX,splinePtrY,splinePtrZ, derivativeValues, \
   basisXX, basisYY, basisZZ, basisXY, basisYZ, basisXZ) \
   private(a, b, c, i, index, x, y, z, derivativeValuesPtr, splineCoeffX, splineCoeffY, \
   splineCoeffZ, XX_x, YY_x, ZZ_x, XY_x, YZ_x, XZ_x, XX_y, YY_y, \
   ZZ_y, XY_y, YZ_y, XZ_y, XX_z, YY_z, ZZ_z, XY_z, YZ_z, XZ_z)
#endif
   for(z=1; z<splineControlPoint->nz-1; z++)
   {
      for(y=1; y<splineControlPoint->ny-1; y++)
      {
         derivativeValuesPtr = &derivativeValues[18*((z*splineControlPoint->ny+y)*splineControlPoint->nx+1)];
         for(x=1; x<splineControlPoint->nx-1; x++)
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
      reg_exit(1);
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
         reg_exit(1);
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
         reg_exit(1);
      }
   }
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
double reg_spline_approxLinearEnergyValue2D(nifti_image *splineControlPoint)
{
   size_t nodeNumber = (size_t)splineControlPoint->nx *
         splineControlPoint->ny;
   int a, b, x, y, i, index;

   double constraintValue = 0.;

   // Create a new image to store the spline coefficients as displacements
   nifti_image *splineDispImg = nifti_copy_nim_info(splineControlPoint);
   splineDispImg->data = (void *)malloc(splineDispImg->nvox*splineDispImg->nbyper);
   memcpy(splineDispImg->data, splineControlPoint->data, splineDispImg->nvox*splineDispImg->nbyper);
   reg_getDisplacementFromDeformation(splineDispImg);

   // Create pointers to the spline coefficients
   DTYPE *splinePtrX = static_cast<DTYPE *>(splineDispImg->data);
   DTYPE *splinePtrY = &splinePtrX[nodeNumber];

   // Store the basis values since they are constant as the value is approximated
   // at the control point positions only
   DTYPE basisX[9], basisY[9];
   set_first_order_basis_values(basisX, basisY);

   DTYPE splineCoeffX;
   DTYPE splineCoeffY;

   mat33 matrix;

#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(splinePtrX, splinePtrY, splineDispImg, \
   basisX, basisY) \
   private(x, y, a, b, i, index, matrix, \
   splineCoeffX, splineCoeffY) \
   reduction(+:constraintValue)
#endif
   for(y=1; y<splineDispImg->ny-1; ++y){
      for(x=1; x<splineDispImg->nx-1; ++x){

         memset(&matrix, 0, sizeof(mat33));

         i=0;
         for(b=-1; b<2; b++){
            for(a=-1; a<2; a++){
               index = (y+b)*splineDispImg->nx+x+a;
               splineCoeffX = splinePtrX[index];
               splineCoeffY = splinePtrY[index];

               matrix.m[0][0] += basisX[i]*splineCoeffX;
               matrix.m[1][0] += basisY[i]*splineCoeffX;

               matrix.m[0][1] += basisX[i]*splineCoeffY;
               matrix.m[1][1] += basisY[i]*splineCoeffY;
               ++i;
            }
         }
         for(b=0; b<2; b++){
            for(a=0; a<2; a++){
               constraintValue += 0.5 *
                     (reg_pow2(matrix.m[a][b]+matrix.m[b][a]) +
                      reg_pow2(matrix.m[a][b]-matrix.m[b][a]));
            }
         }
      }
   }
   nifti_image_free(splineDispImg);
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

   // Create a new image to store the spline coefficients as displacements
   nifti_image *splineDispImg = nifti_copy_nim_info(splineControlPoint);
   splineDispImg->data = (void *)malloc(splineDispImg->nvox*splineDispImg->nbyper);
   memcpy(splineDispImg->data, splineControlPoint->data, splineDispImg->nvox*splineDispImg->nbyper);
   reg_getDisplacementFromDeformation(splineDispImg);

   // Create pointers to the spline coefficients
   DTYPE *splinePtrX = static_cast<DTYPE *>(splineDispImg->data);
   DTYPE *splinePtrY = &splinePtrX[nodeNumber];
   DTYPE *splinePtrZ = &splinePtrY[nodeNumber];

   // Store the basis values since they are constant as the value is approximated
   // at the control point positions only
   DTYPE basisX[27], basisY[27], basisZ[27];
   set_first_order_basis_values(basisX, basisY, basisZ);

   DTYPE splineCoeffX;
   DTYPE splineCoeffY;
   DTYPE splineCoeffZ;

   mat33 matrix;

#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(splinePtrX, splinePtrY, splinePtrZ, splineDispImg, \
   basisX, basisY, basisZ) \
   private(x, y, z, a, b, c, i, index, matrix, \
   splineCoeffX, splineCoeffY, splineCoeffZ) \
   reduction(+:constraintValue)
#endif
   for(z=1; z<splineDispImg->nz-1; ++z){
      for(y=1; y<splineDispImg->ny-1; ++y){
         for(x=1; x<splineDispImg->nx-1; ++x){

            memset(&matrix, 0, sizeof(mat33));

            i=0;
            for(c=-1; c<2; c++){
               for(b=-1; b<2; b++){
                  for(a=-1; a<2; a++){
                     index = ((z+c)*splineDispImg->ny+y+b)*splineDispImg->nx+x+a;
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
            for(b=0; b<3; b++){
               for(a=0; a<3; a++){
                  constraintValue += 0.5 *
                        (reg_pow2(matrix.m[a][b]+matrix.m[b][a]) +
                         reg_pow2(matrix.m[a][b]-matrix.m[b][a]));
               }
            }
         }
      }
   }
   nifti_image_free(splineDispImg);
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
         reg_exit(1);
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
         reg_exit(1);
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
   size_t nodeNumber = (size_t)splineControlPoint->nx*splineControlPoint->ny;
   int x, y, X, Y, a, b, i, index;

   // Create a new image to store the spline coefficients as displacements
   nifti_image *splineDispImg = nifti_copy_nim_info(splineControlPoint);
   splineDispImg->data = (void *)malloc(splineDispImg->nvox*splineDispImg->nbyper);
   memcpy(splineDispImg->data, splineControlPoint->data, splineDispImg->nvox*splineDispImg->nbyper);
   reg_getDisplacementFromDeformation(splineDispImg);

   // Create pointers to the spline coefficients
   DTYPE * splinePtrX = static_cast<DTYPE *>(splineDispImg->data);
   DTYPE * splinePtrY = &splinePtrX[nodeNumber];

   // Store the basis values since they are constant as the value is approximated
   // at the control point positions only
   DTYPE basisX[9];
   DTYPE basisY[9];
   set_first_order_basis_values(basisX, basisY);

   DTYPE *derivativeValues = (DTYPE *)calloc(4*nodeNumber, sizeof(DTYPE));
   DTYPE *derivativeValuesPtr;

   DTYPE splineCoeffX;
   DTYPE splineCoeffY;

   mat33 matrix;

#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(splineDispImg, splinePtrX, splinePtrY, derivativeValues, \
   basisX, basisY) \
   private(x, y, a, b,i, index, derivativeValuesPtr, \
   splineCoeffX, splineCoeffY, matrix)
#endif
   for(y=1; y<splineDispImg->ny-1; y++)
   {
      derivativeValuesPtr = &derivativeValues[4*(y*splineDispImg->nx+1)];
      for(x=1; x<splineDispImg->nx-1; x++)
      {
         memset(&matrix, 0, sizeof(mat33));

         i=0;
         for(b=-1; b<2; b++){
            for(a=-1; a<2; a++){
               index = (y+b)*splineDispImg->nx+x+a;
               splineCoeffX = splinePtrX[index];
               splineCoeffY = splinePtrY[index];

               matrix.m[0][0] += basisX[i]*splineCoeffX;
               matrix.m[1][0] += basisY[i]*splineCoeffX;

               matrix.m[0][1] += basisX[i]*splineCoeffY;
               matrix.m[1][1] += basisY[i]*splineCoeffY;
               ++i;
            }
         }
         *derivativeValuesPtr++ = matrix.m[0][0];
         *derivativeValuesPtr++ = matrix.m[0][1];
         *derivativeValuesPtr++ = matrix.m[1][0];
         *derivativeValuesPtr++ = matrix.m[1][1];
      } // x
   } // y
   nifti_image_free(splineDispImg);

   DTYPE *gradientXPtr = static_cast<DTYPE *>(gradientImage->data);
   DTYPE *gradientYPtr = &gradientXPtr[nodeNumber];

   DTYPE approxRatio = (DTYPE)weight / (DTYPE)(nodeNumber);

   double gradientValue[2];

#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(splineControlPoint, derivativeValues, gradientXPtr, gradientYPtr, \
   basisX, basisY, approxRatio) \
   private(index, i, X, Y, x, y, a, derivativeValuesPtr, gradientValue, matrix)
#endif
   for(y=0; y<splineControlPoint->ny; y++)
   {
      index=y*splineControlPoint->nx;
      for(x=0; x<splineControlPoint->nx; x++)
      {
         gradientValue[0]=gradientValue[1]=0.0;
         i=0;
         for(Y=y-1; Y<y+2; Y++)
         {
            for(X=x-1; X<x+2; X++)
            {
               if(-1<X && -1<Y && X<splineControlPoint->nx && Y<splineControlPoint->ny)
               {
                  derivativeValuesPtr = &derivativeValues[4 * (Y*splineControlPoint->nx + X)];

                  matrix.m[0][0] = (*derivativeValuesPtr++);
                  matrix.m[0][1] = (*derivativeValuesPtr++);

                  matrix.m[1][0] = (*derivativeValuesPtr++);
                  matrix.m[1][1] = (*derivativeValuesPtr++);

                  gradientValue[0] -= 2.0*matrix.m[0][0]*basisX[i];
                  gradientValue[0] -= (matrix.m[0][1]+matrix.m[1][0])*basisY[i];
                  gradientValue[0] += (matrix.m[0][1]-matrix.m[1][0])*basisY[i];

                  gradientValue[1] -= 2.0*matrix.m[1][1]*basisY[i];
                  gradientValue[1] -= (matrix.m[1][0]+matrix.m[0][1])*basisX[i];
                  gradientValue[1] += (matrix.m[1][0]-matrix.m[0][1])*basisX[i];

               }
               ++i;
            }
         }
         gradientXPtr[index] += approxRatio*gradientValue[0];
         gradientYPtr[index] += approxRatio*gradientValue[1];
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
   size_t nodeNumber = (size_t)splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz;
   int x, y, z, X, Y, Z, a, b, c, i, index;

   // Create a new image to store the spline coefficients as displacements
   nifti_image *splineDispImg = nifti_copy_nim_info(splineControlPoint);
   splineDispImg->data = (void *)malloc(splineDispImg->nvox*splineDispImg->nbyper);
   memcpy(splineDispImg->data, splineControlPoint->data, splineDispImg->nvox*splineDispImg->nbyper);
   reg_getDisplacementFromDeformation(splineDispImg);

   // Create pointers to the spline coefficients
   DTYPE * splinePtrX = static_cast<DTYPE *>(splineDispImg->data);
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

   DTYPE splineCoeffX;
   DTYPE splineCoeffY;
   DTYPE splineCoeffZ;

   mat33 matrix;

#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(splineDispImg, splinePtrX, splinePtrY, splinePtrZ, derivativeValues, \
   basisX, basisY, basisZ) \
   private(x, y, z, a, b, c, i, index, derivativeValuesPtr, \
   splineCoeffX, splineCoeffY, splineCoeffZ, matrix)
#endif
   for(z=1; z<splineDispImg->nz-1; z++)
   {
      for(y=1; y<splineDispImg->ny-1; y++)
      {
         derivativeValuesPtr = &derivativeValues[9*((z*splineDispImg->ny+y)*splineDispImg->nx+1)];
         for(x=1; x<splineDispImg->nx-1; x++)
         {
            memset(&matrix, 0, sizeof(mat33));

            i=0;
            for(c=-1; c<2; c++){
               for(b=-1; b<2; b++){
                  for(a=-1; a<2; a++){
                     index = ((z+c)*splineDispImg->ny+y+b)*splineDispImg->nx+x+a;
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
   nifti_image_free(splineDispImg);

   DTYPE *gradientXPtr = static_cast<DTYPE *>(gradientImage->data);
   DTYPE *gradientYPtr = &gradientXPtr[nodeNumber];
   DTYPE *gradientZPtr = &gradientYPtr[nodeNumber];

   DTYPE approxRatio = (DTYPE)weight / (DTYPE)(nodeNumber);

   double gradientValue[3];

#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(splineControlPoint, derivativeValues, gradientXPtr, gradientYPtr, gradientZPtr, \
   basisX, basisY, basisZ, approxRatio) \
   private(index, i, X, Y, Z, x, y, z, a, derivativeValuesPtr, gradientValue, matrix)
#endif
   for(z=0; z<splineControlPoint->nz; z++)
   {
      index=z*splineControlPoint->nx*splineControlPoint->ny;
      for(y=0; y<splineControlPoint->ny; y++)
      {
         for(x=0; x<splineControlPoint->nx; x++)
         {
            gradientValue[0]=gradientValue[1]=gradientValue[2]=0.0;
            i=0;
            for(Z=z-1; Z<z+2; Z++)
            {
               for(Y=y-1; Y<y+2; Y++)
               {
                  for(X=x-1; X<x+2; X++)
                  {
                     if(-1<X && -1<Y && -1<Z && X<splineControlPoint->nx && Y<splineControlPoint->ny && Z<splineControlPoint->nz)
                     {
                        derivativeValuesPtr = &derivativeValues[9 * ((Z*splineControlPoint->ny + Y)*splineControlPoint->nx + X)];

                        matrix.m[0][0] = (*derivativeValuesPtr++);
                        matrix.m[0][1] = (*derivativeValuesPtr++);
                        matrix.m[0][2] = (*derivativeValuesPtr++);

                        matrix.m[1][0] = (*derivativeValuesPtr++);
                        matrix.m[1][1] = (*derivativeValuesPtr++);
                        matrix.m[1][2] = (*derivativeValuesPtr++);

                        matrix.m[2][0] = (*derivativeValuesPtr++);
                        matrix.m[2][1] = (*derivativeValuesPtr++);
                        matrix.m[2][2] = (*derivativeValuesPtr++);

                        gradientValue[0] -= 2.0*matrix.m[0][0]*basisX[i];
                        gradientValue[0] -= (matrix.m[0][1]+matrix.m[1][0])*basisY[i];
                        gradientValue[0] -= (matrix.m[0][2]+matrix.m[2][0])*basisZ[i];
                        gradientValue[0] += (matrix.m[0][1]-matrix.m[1][0])*basisY[i];
                        gradientValue[0] += (matrix.m[0][2]-matrix.m[2][0])*basisZ[i];

                        gradientValue[1] -= 2.0*matrix.m[1][1]*basisY[i];
                        gradientValue[1] -= (matrix.m[1][0]+matrix.m[0][1])*basisX[i];
                        gradientValue[1] -= (matrix.m[1][2]+matrix.m[2][1])*basisZ[i];
                        gradientValue[1] += (matrix.m[1][0]-matrix.m[0][1])*basisX[i];
                        gradientValue[1] += (matrix.m[1][2]-matrix.m[2][1])*basisZ[i];

                        gradientValue[2] -= 2.0*matrix.m[2][2]*basisZ[i];
                        gradientValue[2] -= (matrix.m[2][0]+matrix.m[0][2])*basisX[i];
                        gradientValue[2] -= (matrix.m[2][1]+matrix.m[1][2])*basisY[i];
                        gradientValue[2] += (matrix.m[2][0]-matrix.m[0][2])*basisX[i];
                        gradientValue[2] += (matrix.m[2][1]-matrix.m[1][2])*basisY[i];

                     }
                     ++i;
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
      reg_exit(1);
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
         reg_exit(1);
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
         reg_exit(1);
      }
   }
}
/* *************************************************************** */
