/*
 *  _reg_localTrans_jac.cpp
 *
 *
 *  Created by Marc Modat on 10/05/2011.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_localTrans_jac.h"

#define USE_SQUARE_LOG_JAC

/* *************************************************************** */
/* *************************************************************** */
template <class DataType>
void addJacobianGradientValues(mat33 jacobianMatrix,
                               double detJac,
                               DataType basisX,
                               DataType basisY,
                               DataType *jacobianConstraint)
{
   jacobianConstraint[0] += detJac * (jacobianMatrix.m[1][1]*basisX - jacobianMatrix.m[1][0]*basisY);
   jacobianConstraint[1] += detJac * (jacobianMatrix.m[0][0]*basisY - jacobianMatrix.m[0][1]*basisX);
}
/* *************************************************************** */
template <class DataType>
void addJacobianGradientValues(mat33 jacobianMatrix,
                               double detJac,
                               DataType basisX,
                               DataType basisY,
                               DataType basisZ,
                               DataType *jacobianConstraint)
{
   jacobianConstraint[0] += detJac * (
            basisX * (jacobianMatrix.m[1][1]*jacobianMatrix.m[2][2] - jacobianMatrix.m[1][2]*jacobianMatrix.m[2][1]) +
         basisY * (jacobianMatrix.m[1][2]*jacobianMatrix.m[2][0] - jacobianMatrix.m[1][0]*jacobianMatrix.m[2][2]) +
         basisZ * (jacobianMatrix.m[1][0]*jacobianMatrix.m[2][1] - jacobianMatrix.m[1][1]*jacobianMatrix.m[2][0]) );

   jacobianConstraint[1] += detJac * (
            basisX * (jacobianMatrix.m[0][2]*jacobianMatrix.m[2][1] - jacobianMatrix.m[0][1]*jacobianMatrix.m[2][2]) +
         basisY * (jacobianMatrix.m[0][0]*jacobianMatrix.m[2][2] - jacobianMatrix.m[0][2]*jacobianMatrix.m[2][0]) +
         basisZ * (jacobianMatrix.m[0][1]*jacobianMatrix.m[2][0] - jacobianMatrix.m[0][0]*jacobianMatrix.m[2][1]) );

   jacobianConstraint[2] += detJac * (
            basisX * (jacobianMatrix.m[0][1]*jacobianMatrix.m[1][2] - jacobianMatrix.m[0][2]*jacobianMatrix.m[1][1]) +
         basisY * (jacobianMatrix.m[0][2]*jacobianMatrix.m[1][0] - jacobianMatrix.m[0][0]*jacobianMatrix.m[1][2]) +
         basisZ * (jacobianMatrix.m[0][0]*jacobianMatrix.m[1][1] - jacobianMatrix.m[0][1]*jacobianMatrix.m[1][0]) );
}
/* *************************************************************** */
/* *************************************************************** */
template<class DataType>
void reg_linear_spline_jacobian3D(nifti_image *splineControlPoint,
                                  nifti_image *referenceImage,
                                  mat33 *JacobianMatrices,
                                  DataType *JacobianDeterminants,
                                  bool approximation,
                                  bool useHeaderInformation)
{
   if(JacobianMatrices==nullptr && JacobianDeterminants==nullptr)
      NR_FATAL_ERROR("Both output pointers are nullptr");
   if(referenceImage==nullptr && approximation==false)
      NR_FATAL_ERROR("The reference image is required to compute the Jacobian at voxel position");

   // Create some pointers towards to control point grid image data
   const size_t nodeNumber = NiftiImage::calcVoxelNumber(splineControlPoint, 3);
   DataType *coeffPtrX = static_cast<DataType *>(splineControlPoint->data);
   DataType *coeffPtrY = &coeffPtrX[nodeNumber];
   DataType *coeffPtrZ = &coeffPtrY[nodeNumber];

   // Define a matrix to reorient the Jacobian matrices and normalise them by the grid spacing
   mat33 reorientation,jacobianMatrix;
   if(splineControlPoint->sform_code>0)
      reorientation = reg_mat44_to_mat33(&splineControlPoint->sto_ijk);
   else reorientation = reg_mat44_to_mat33(&splineControlPoint->qto_ijk);

   // Useful variables
   int x, y, z;
   size_t index;

   if(approximation)
   {
      // The Jacobian information is only computed at the control point positions
      for(z=1; z<splineControlPoint->nz-1; z++)
      {
         index=(z-1)*(splineControlPoint->nx-2)*(splineControlPoint->ny-2);
         for(y=1; y<splineControlPoint->ny-1; y++)
         {
            for(x=1; x<splineControlPoint->nx-1; x++)
            {
               jacobianMatrix.m[0][0] = (coeffPtrX[index+1] - coeffPtrX[-1])/2.;
               jacobianMatrix.m[0][1] = (coeffPtrX[index+splineControlPoint->nx] - coeffPtrX[index-splineControlPoint->nx])/2.;
               jacobianMatrix.m[0][2] = (coeffPtrX[index+splineControlPoint->nx*splineControlPoint->ny] - coeffPtrX[index-splineControlPoint->nx*splineControlPoint->ny])/2.;

               jacobianMatrix.m[1][0] = (coeffPtrY[index+1] - coeffPtrX[-1])/2.;
               jacobianMatrix.m[1][1] = (coeffPtrY[index+splineControlPoint->nx] - coeffPtrY[index-splineControlPoint->nx])/2.;
               jacobianMatrix.m[1][2] = (coeffPtrY[index+splineControlPoint->nx*splineControlPoint->ny] - coeffPtrY[index-splineControlPoint->nx*splineControlPoint->ny])/2.;

               jacobianMatrix.m[2][0] = (coeffPtrZ[index+1] - coeffPtrX[-1])/2.;
               jacobianMatrix.m[2][1] = (coeffPtrZ[index+splineControlPoint->nx] - coeffPtrZ[index-splineControlPoint->nx])/2.;
               jacobianMatrix.m[2][2] = (coeffPtrZ[index+splineControlPoint->nx*splineControlPoint->ny] - coeffPtrZ[index-splineControlPoint->nx*splineControlPoint->ny])/2.;

               jacobianMatrix=nifti_mat33_mul(reorientation,jacobianMatrix);
               if(JacobianMatrices!=nullptr)
                  JacobianMatrices[index]=jacobianMatrix;
               if(JacobianDeterminants!=nullptr)
                  JacobianDeterminants[index] =
                        static_cast<DataType>(nifti_mat33_determ(jacobianMatrix));
               ++index;
            } // loop over x
         } // loop over y
      } // loop over z
   } // end if approximation at the control point index only
   else
   {
      // The Jacobian matrices and determinants are computed at all voxel positions
      // The voxel are discretised using the reference image

      // If the control point grid contains an affine transformation,
      // the header information is used by default
      if(splineControlPoint->num_ext>0)
         useHeaderInformation=true;

      // Allocate variables that are used in both scenario
      DataType gridVoxelSpacing[3]=
      {
         splineControlPoint->dx / referenceImage->dx,
         splineControlPoint->dy / referenceImage->dy,
         splineControlPoint->dz / referenceImage->dz
      };
      DataType pre[3];

      if(useHeaderInformation)
      {
         // The reference image is not necessarily aligned with the grid
         mat44 transformation;
         // reference: voxel to mm
         if(referenceImage->sform_code>0)
            transformation=referenceImage->sto_xyz;
         else transformation=referenceImage->qto_xyz;
         // affine: mm to mm
         if(splineControlPoint->num_ext>0)
            transformation=reg_mat44_mul(
                     reinterpret_cast<mat44 *>(splineControlPoint->ext_list[0].edata),
                  &transformation);
         // grid: mm to voxel
         if(splineControlPoint->sform_code>0)
            transformation=reg_mat44_mul(&(splineControlPoint->sto_ijk), &transformation);
         else transformation=reg_mat44_mul(&(splineControlPoint->qto_ijk), &transformation);

         float imageCoord[3], gridCoord[3];
         for(z=0; z<referenceImage->nz; z++)
         {
            index=z*referenceImage->nx*referenceImage->ny;
            imageCoord[2]=z;
            for(y=0; y<referenceImage->ny; y++)
            {
               imageCoord[1]=y;
               for(x=0; x<referenceImage->nx; x++)
               {
                  imageCoord[0]=x;
                  // Compute the position in the grid
                  reg_mat44_mul(&transformation,imageCoord,gridCoord);
                  // Compute the anterior node coord
                  pre[0]=Floor(gridCoord[0]);
                  pre[1]=Floor(gridCoord[1]);
                  pre[2]=Floor(gridCoord[2]);
                  int controlPoint_index=(pre[2]*splineControlPoint->ny+pre[1])*splineControlPoint->nx+pre[0];

                  jacobianMatrix.m[0][0] = (coeffPtrX[controlPoint_index+1] - coeffPtrX[controlPoint_index]);
                  jacobianMatrix.m[0][1] = (coeffPtrX[controlPoint_index+splineControlPoint->nx] - coeffPtrX[controlPoint_index]);
                  jacobianMatrix.m[0][2] = (coeffPtrX[controlPoint_index+splineControlPoint->nx*splineControlPoint->ny] - coeffPtrX[controlPoint_index]);

                  jacobianMatrix.m[1][0] = (coeffPtrY[controlPoint_index+1] - coeffPtrY[controlPoint_index]);
                  jacobianMatrix.m[1][1] = (coeffPtrY[controlPoint_index+splineControlPoint->nx] - coeffPtrY[controlPoint_index]);
                  jacobianMatrix.m[1][2] = (coeffPtrY[controlPoint_index+splineControlPoint->nx*splineControlPoint->ny] - coeffPtrY[controlPoint_index]);

                  jacobianMatrix.m[2][0] = (coeffPtrZ[controlPoint_index+1] - coeffPtrZ[controlPoint_index]);
                  jacobianMatrix.m[2][1] = (coeffPtrZ[controlPoint_index+splineControlPoint->nx] - coeffPtrZ[controlPoint_index]);
                  jacobianMatrix.m[2][2] = (coeffPtrZ[controlPoint_index+splineControlPoint->nx*splineControlPoint->ny] - coeffPtrZ[controlPoint_index]);

                  // reorient the matrix
                  jacobianMatrix=nifti_mat33_mul(reorientation,
                                                 jacobianMatrix);
                  if(JacobianMatrices!=nullptr)
                     JacobianMatrices[index]=jacobianMatrix;
                  if(JacobianDeterminants!=nullptr)
                     JacobianDeterminants[index] =
                           static_cast<DataType>(nifti_mat33_determ(jacobianMatrix));
                  ++index;
               } // x
            } // y
         } // z
      }
      else
      {
         // The grid is assumed to be aligned with the reference image
         for(z=0; z<referenceImage->nz; z++)
         {
            index=z*referenceImage->nx*referenceImage->ny;
            pre[2]=(int)((DataType)z/gridVoxelSpacing[2])+1;

            for(y=0; y<referenceImage->ny; y++)
            {
               pre[1]=(int)((DataType)y/gridVoxelSpacing[1])+1;

               for(x=0; x<referenceImage->nx; x++)
               {

                  pre[0]=(int)((DataType)x/gridVoxelSpacing[0])+1;
                  int controlPoint_index=(pre[2]*splineControlPoint->ny+pre[1])*splineControlPoint->nx+pre[0];

                  jacobianMatrix.m[0][0] = (coeffPtrX[controlPoint_index+1] - coeffPtrX[controlPoint_index]);
                  jacobianMatrix.m[0][1] = (coeffPtrX[controlPoint_index+splineControlPoint->nx] - coeffPtrX[controlPoint_index]);
                  jacobianMatrix.m[0][2] = (coeffPtrX[controlPoint_index+splineControlPoint->nx*splineControlPoint->ny] - coeffPtrX[controlPoint_index]);

                  jacobianMatrix.m[1][0] = (coeffPtrY[controlPoint_index+1] - coeffPtrY[controlPoint_index]);
                  jacobianMatrix.m[1][1] = (coeffPtrY[controlPoint_index+splineControlPoint->nx] - coeffPtrY[controlPoint_index]);
                  jacobianMatrix.m[1][2] = (coeffPtrY[controlPoint_index+splineControlPoint->nx*splineControlPoint->ny] - coeffPtrY[controlPoint_index]);

                  jacobianMatrix.m[2][0] = (coeffPtrZ[controlPoint_index+1] - coeffPtrZ[controlPoint_index]);
                  jacobianMatrix.m[2][1] = (coeffPtrZ[controlPoint_index+splineControlPoint->nx] - coeffPtrZ[controlPoint_index]);
                  jacobianMatrix.m[2][2] = (coeffPtrZ[controlPoint_index+splineControlPoint->nx*splineControlPoint->ny] - coeffPtrZ[controlPoint_index]);

                  // reorient the matrix
                  jacobianMatrix=nifti_mat33_mul(reorientation,
                                                 jacobianMatrix);

                  if(JacobianMatrices!=nullptr)
                     JacobianMatrices[index]=jacobianMatrix;
                  if(JacobianDeterminants!=nullptr)
                     JacobianDeterminants[index] =
                           static_cast<DataType>(nifti_mat33_determ(jacobianMatrix));
                  ++index;
               } // loop over x
            } // loop over y
         } // loop over z
      } // end if the grid is aligned with the reference image
   } // end if no approximation
   return;
}
/* *************************************************************** */
/* *************************************************************** */
template<class DataType>
void reg_cubic_spline_jacobian2D(nifti_image *splineControlPoint,
                           nifti_image *referenceImage,
                           mat33 *JacobianMatrices,
                           DataType *JacobianDeterminants,
                           bool approximation,
                           bool useHeaderInformation)
{
   if(JacobianMatrices==nullptr && JacobianDeterminants==nullptr)
      NR_FATAL_ERROR("Both output pointers are nullptr");
   if(referenceImage==nullptr && approximation==false)
      NR_FATAL_ERROR("The reference image is required to compute the Jacobian at voxel position");

   // Create some pointers towards to control point grid image data
   const size_t nodeNumber = NiftiImage::calcVoxelNumber(splineControlPoint, 2);
   DataType *coeffPtrX = static_cast<DataType *>(splineControlPoint->data);
   DataType *coeffPtrY = &coeffPtrX[nodeNumber];

   // Define a matrice to reorient the Jacobian matrices and normalise them by the grid spacing
   mat33 reorientation,jacobianMatrix;
   if(splineControlPoint->sform_code>0)
      reorientation = reg_mat44_to_mat33(&splineControlPoint->sto_ijk);
   else reorientation = reg_mat44_to_mat33(&splineControlPoint->qto_ijk);

   // Useful variables
   int x, y, incr0;
   size_t voxelIndex;

   if(approximation)
   {
      // The Jacobian information is only computed at the control point positions
      // Note that the header information is not used here
      float basisX[9], basisY[9];
      DataType coeffX[9], coeffY[9];
	  DataType normal[3] = { 1.f / 6.f, 2.f / 3.f, 1.f / 6.f };
	  DataType first[3] = { -0.5f, 0.f, 0.5f };
      // There are six different values taken into account
      int coord=0;
      for(int b=0; b<3; ++b)
      {
         for(int a=0; a<3; ++a)
         {
            basisX[coord]=normal[b]*first[a];  // y * x'
            basisY[coord]=first[b]*normal[a]; //  y'* x
            coord++;
         }
      }
#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(splineControlPoint, coeffPtrX, coeffPtrY, \
   basisX, basisY, reorientation, JacobianMatrices, JacobianDeterminants) \
   private(x, incr0, coeffX, coeffY, jacobianMatrix, voxelIndex)
#endif
      for(y=1; y<splineControlPoint->ny-1; y++)
      {
         voxelIndex=(y-1)*(splineControlPoint->nx-2);
         for(x=1; x<splineControlPoint->nx-1; x++)
         {

            get_GridValues<DataType>(x-1,
                                  y-1,
                                  splineControlPoint,
                                  coeffPtrX,
                                  coeffPtrY,
                                  coeffX,
                                  coeffY,
                                  true, // approx
                                  false // not disp
                                  );

            memset(&jacobianMatrix,0,sizeof(mat33));
            jacobianMatrix.m[2][2]=1.f;
            for(incr0=0; incr0<9; ++incr0)
            {
               jacobianMatrix.m[0][0] += basisX[incr0]*coeffX[incr0];
               jacobianMatrix.m[0][1] += basisY[incr0]*coeffX[incr0];
               jacobianMatrix.m[1][0] += basisX[incr0]*coeffY[incr0];
               jacobianMatrix.m[1][1] += basisY[incr0]*coeffY[incr0];
            }
            jacobianMatrix=nifti_mat33_mul(reorientation,jacobianMatrix);
            if(JacobianMatrices!=nullptr)
               JacobianMatrices[voxelIndex]=jacobianMatrix;
            if(JacobianDeterminants!=nullptr)
               JacobianDeterminants[voxelIndex] =
                     static_cast<DataType>(nifti_mat33_determ(jacobianMatrix));
            ++voxelIndex;
         } // loop over x
      } // loop over y
   } // end if approximation at the control point index only
   else
   {
      // The Jacobian matrices and determinants are computed at all voxel positions
      // The voxel are discretised using the reference image

      // If the control point grid contains an affine transformation,
      // the header information is used by default
      if(splineControlPoint->num_ext>0)
         useHeaderInformation=true;

      // Allocate variables that are used in both scenarii
      int pre[2], oldPre[2];
      int coord, incr0, incr1;
      DataType xBasis[4], xFirst[4], yBasis[4], yFirst[4];
      DataType basisX[16], basisY[16];
      DataType coeffX[16], coeffY[16];
      size_t voxelIndex;

      if(useHeaderInformation)
      {
         // The reference image is not necessarly aligned with the grid
         mat44 transformation;
         // reference: voxel to mm
         if(referenceImage->sform_code>0)
            transformation=referenceImage->sto_xyz;
         else transformation=referenceImage->qto_xyz;
         // affine: mm to mm
         if(splineControlPoint->num_ext>0)
            transformation=reg_mat44_mul(
                     reinterpret_cast<mat44 *>(splineControlPoint->ext_list[0].edata),
                  &transformation);
         // grid: mm to voxel
         if(splineControlPoint->sform_code>0)
            transformation=reg_mat44_mul(&(splineControlPoint->sto_ijk), &transformation);
         else transformation=reg_mat44_mul(&(splineControlPoint->qto_ijk), &transformation);

         float imageCoord[3], gridCoord[3], basis;
         imageCoord[2]=0;
         for(y=0; y<referenceImage->ny; y++)
         {
            imageCoord[1]=y;
            oldPre[0]=oldPre[1]=999999;
            voxelIndex=y*referenceImage->nx;
            for(x=0; x<referenceImage->nx; x++)
            {
               imageCoord[0]=x;
               // Compute the position in the grid
               reg_mat44_mul(&transformation,imageCoord,gridCoord);
               // Compute the anterior node coord
               pre[0]=Floor(gridCoord[0]);
               pre[1]=Floor(gridCoord[1]);
               // Compute the basis values and their first derivatives
               basis = gridCoord[0] - pre[0];
               get_BSplineBasisValues<DataType>(basis, xBasis, xFirst);
               basis = gridCoord[1] - pre[1];
               get_BSplineBasisValues<DataType>(basis, yBasis, yFirst);
               // Compute the 16 basis values and the corresponding derivatives

               coord=0;
               for(incr0=0; incr0<4; incr0++)
               {
                  for(incr1=0; incr1<4; incr1++)
                  {
                     basisX[coord]=yBasis[incr0]*xFirst[incr1]; // y * x'
                     basisY[coord]=yFirst[incr0]*xBasis[incr1]; // y' * x
                     ++coord;
                  }
               }
               // Fetch the required coefficients
               if(oldPre[0]!=pre[0] || oldPre[1]!=pre[1])
               {

                  get_GridValues<DataType>(pre[0]-1,
                        pre[1]-1,
                        splineControlPoint,
                        coeffPtrX,
                        coeffPtrY,
                        coeffX,
                        coeffY,
                        false, // no approx
                        false // not disp
                        );
                  oldPre[0]=pre[0];
                  oldPre[1]=pre[1];
               }
               // Compute the Jacobian matrix
               memset(&jacobianMatrix, 0, sizeof(mat33));
               jacobianMatrix.m[2][2]=1.f;
               for(incr0=0; incr0<16; ++incr0)
               {
                  jacobianMatrix.m[0][0] += basisX[incr0]*coeffX[incr0];
                  jacobianMatrix.m[0][1] += basisY[incr0]*coeffX[incr0];
                  jacobianMatrix.m[1][0] += basisX[incr0]*coeffY[incr0];
                  jacobianMatrix.m[1][1] += basisY[incr0]*coeffY[incr0];
               }
               // reorient the matrix
               jacobianMatrix=nifti_mat33_mul(reorientation,
                                              jacobianMatrix);
               if(JacobianMatrices!=nullptr)
                  JacobianMatrices[voxelIndex]=jacobianMatrix;
               if(JacobianDeterminants!=nullptr)
                  JacobianDeterminants[voxelIndex] =
                        static_cast<DataType>(nifti_mat33_determ(jacobianMatrix));
               ++voxelIndex;
            } // x
         } // y
      }
      else
      {
         DataType basis;
         DataType gridVoxelSpacing[2]=
         {
            splineControlPoint->dx / referenceImage->dx,
            splineControlPoint->dy / referenceImage->dy
         };
         // The grid is assumed to be aligned with the reference image
         for(y=0; y<referenceImage->ny; y++)
         {
            voxelIndex=y*referenceImage->nx;
            oldPre[0]=oldPre[1]=999999;

            pre[1]=(int)((DataType)y/gridVoxelSpacing[1]);
            basis=(DataType)y/gridVoxelSpacing[1]-(DataType)pre[1];
            if(basis<0) basis=0; //rounding error
            get_BSplineBasisValues<DataType>(basis, yBasis, yFirst);

            for(x=0; x<referenceImage->nx; x++)
            {

               pre[0]=(int)((DataType)x/gridVoxelSpacing[0]);
               basis=(DataType)x/gridVoxelSpacing[0]-(DataType)pre[0];
               if(basis<0) basis=0; //rounding error
               get_BSplineBasisValues<DataType>(basis, xBasis, xFirst);

               coord=0;
               for(incr0=0; incr0<4; ++incr0)
               {
                  for(incr1=0; incr1<4; ++incr1)
                  {
                     basisX[coord]=yBasis[incr0]*xFirst[incr1];   // y * x'
                     basisY[coord]=yFirst[incr0]*xBasis[incr1];   // y'* x
                     coord++;
                  }
               }

               if(oldPre[0]!=pre[0] || oldPre[1]!=pre[1])
               {
                  get_GridValues<DataType>(pre[0],
                        pre[1],
                        splineControlPoint,
                        coeffPtrX,
                        coeffPtrY,
                        coeffX,
                        coeffY,
                        false, // no approx
                        false // not disp
                        );
                  oldPre[0]=pre[0];
                  oldPre[1]=pre[1];
               }
               memset(&jacobianMatrix, 0, sizeof(mat33));
               jacobianMatrix.m[2][2] = 1.f;
               for(incr0=0; incr0<16; ++incr0)
               {
                  jacobianMatrix.m[0][0] += basisX[incr0]*coeffX[incr0];
                  jacobianMatrix.m[0][1] += basisY[incr0]*coeffX[incr0];
                  jacobianMatrix.m[1][0] += basisX[incr0]*coeffY[incr0];
                  jacobianMatrix.m[1][1] += basisY[incr0]*coeffY[incr0];
               }
               jacobianMatrix=nifti_mat33_mul(reorientation,
                                              jacobianMatrix);
               if(JacobianMatrices!=nullptr)
                  JacobianMatrices[voxelIndex]=jacobianMatrix;
               if(JacobianDeterminants!=nullptr)
                  JacobianDeterminants[voxelIndex] =
                        static_cast<DataType>(nifti_mat33_determ(jacobianMatrix));
               ++voxelIndex;
            } // loop over x
         } // loop over y
      } // end if the grid is aligned with the reference image
   } // end if no approximation
   return;
}
/* *************************************************************** */
template<class DataType>
void reg_cubic_spline_jacobian3D(nifti_image *splineControlPoint,
                           nifti_image *referenceImage,
                           mat33 *JacobianMatrices,
                           DataType *JacobianDeterminants,
                           bool approximation,
                           bool useHeaderInformation)
{
   if(JacobianMatrices==nullptr && JacobianDeterminants==nullptr)
      NR_FATAL_ERROR("Both output pointers are nullptr");
   if(referenceImage==nullptr && approximation==false)
      NR_FATAL_ERROR("The reference image is required to compute the Jacobian at voxel position");

   // Create some pointers towards to control point grid image data
   const size_t nodeNumber = NiftiImage::calcVoxelNumber(splineControlPoint, 3);
   DataType *coeffPtrX = static_cast<DataType *>(splineControlPoint->data);
   DataType *coeffPtrY = &coeffPtrX[nodeNumber];
   DataType *coeffPtrZ = &coeffPtrY[nodeNumber];

   // Define a matrice to reorient the Jacobian matrices and normalise them by the grid spacing
   mat33 reorientation,jacobianMatrix;
   if(splineControlPoint->sform_code>0)
      reorientation = reg_mat44_to_mat33(&splineControlPoint->sto_ijk);
   else reorientation = reg_mat44_to_mat33(&splineControlPoint->qto_ijk);

   // Useful variables
   int x, y, z, incr0;
   size_t voxelIndex;

   if(approximation)
   {
      // The Jacobian information is only computed at the control point positions
      // Note that the header information is not used here
      float basisX[27], basisY[27], basisZ[27];
      DataType coeffX[27], coeffY[27], coeffZ[27];
	  DataType normal[3] = { 1.f / 6.f, 2.f / 3.f, 1.f / 6.f };
	  DataType first[3] = { -0.5f, 0.f, 0.5f };
      // There are six different values taken into account
      DataType tempX[9], tempY[9], tempZ[9];
      int coord=0;
      for(int c=0; c<3; c++)
      {
         for(int b=0; b<3; b++)
         {
            tempX[coord]=normal[c]*normal[b]; // z * y
            tempY[coord]=normal[c]*first[b];  // z * y"
            tempZ[coord]=first[c]*normal[b];  // z"* y
            coord++;
         }
      }
      coord=0;
      for(int bc=0; bc<9; bc++)
      {
         for(int a=0; a<3; a++)
         {
            basisX[coord]=tempX[bc]*first[a];  // z * y * x"
            basisY[coord]=tempY[bc]*normal[a]; // z * y"* x
            basisZ[coord]=tempZ[bc]*normal[a]; // z"* y * x
            coord++;
         }
      }
#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(splineControlPoint, coeffPtrX, coeffPtrY, coeffPtrZ, \
   basisX, basisY, basisZ, reorientation, JacobianMatrices, JacobianDeterminants) \
   private(x, y, incr0, coeffX, coeffY, coeffZ, jacobianMatrix, voxelIndex)
#endif
      for(z=1; z<splineControlPoint->nz-1; z++)
      {
         voxelIndex=(z-1)*(splineControlPoint->nx-2)*(splineControlPoint->ny-2);
         for(y=1; y<splineControlPoint->ny-1; y++)
         {
            for(x=1; x<splineControlPoint->nx-1; x++)
            {

               get_GridValues<DataType>(x-1,
                                     y-1,
                                     z-1,
                                     splineControlPoint,
                                     coeffPtrX,
                                     coeffPtrY,
                                     coeffPtrZ,
                                     coeffX,
                                     coeffY,
                                     coeffZ,
                                     true, // approx
                                     false // not disp
                                     );

               memset(&jacobianMatrix,0,sizeof(mat33));
               for(incr0=0; incr0<27; ++incr0)
               {
                  jacobianMatrix.m[0][0] += basisX[incr0]*coeffX[incr0];
                  jacobianMatrix.m[0][1] += basisY[incr0]*coeffX[incr0];
                  jacobianMatrix.m[0][2] += basisZ[incr0]*coeffX[incr0];
                  jacobianMatrix.m[1][0] += basisX[incr0]*coeffY[incr0];
                  jacobianMatrix.m[1][1] += basisY[incr0]*coeffY[incr0];
                  jacobianMatrix.m[1][2] += basisZ[incr0]*coeffY[incr0];
                  jacobianMatrix.m[2][0] += basisX[incr0]*coeffZ[incr0];
                  jacobianMatrix.m[2][1] += basisY[incr0]*coeffZ[incr0];
                  jacobianMatrix.m[2][2] += basisZ[incr0]*coeffZ[incr0];
               }
               jacobianMatrix=nifti_mat33_mul(reorientation,jacobianMatrix);
               if(JacobianMatrices!=nullptr)
                  JacobianMatrices[voxelIndex]=jacobianMatrix;
               if(JacobianDeterminants!=nullptr)
                  JacobianDeterminants[voxelIndex] =
                        static_cast<DataType>(nifti_mat33_determ(jacobianMatrix));
               ++voxelIndex;
            } // loop over x
         } // loop over y
      } // loop over z
   } // end if approximation at the control point index only
   else
   {
      // The Jacobian matrices and determinants are computed at all voxel positions
      // The voxel are discretised using the reference image

      // If the control point grid contains an affine transformation,
      // the header information is used by default
      if(splineControlPoint->num_ext>0)
         useHeaderInformation=true;

      // Allocate variables that are used in both scenarii
      int pre[3], oldPre[3], incr0;
      DataType basis, xBasis[4], xFirst[4], yBasis[4], yFirst[4], zBasis[4], zFirst[4];
#if USE_SSE
      union
      {
         __m128 m;
         float f[4];
      } val;
      __m128 _xBasis, _xFirst, _yBasis, _yFirst;
      __m128 tempX_x, tempX_y, tempX_z, tempY_x, tempY_y, tempY_z, tempZ_x, tempZ_y, tempZ_z;
#ifdef _WINDOWS
      union
      {
         __m128 m[4];
         __declspec(align(16)) DataType f[16];
      } tempX;
      union
      {
         __m128 m[4];
         __declspec(align(16)) DataType f[16];
      } tempY;
      union
      {
         __m128 m[4];
         __declspec(align(16)) DataType f[16];
      } tempZ;
      union
      {
         __m128 m[16];
         __declspec(align(16)) DataType f[64];
      } basisX;
      union
      {
         __m128 m[16];
         __declspec(align(16)) DataType f[64];
      } basisY;
      union
      {
         __m128 m[16];
         __declspec(align(16)) DataType f[64];
      } basisZ;
      union
      {
         __m128 m[16];
         __declspec(align(16)) DataType f[64];
      } coeffX;
      union
      {
         __m128 m[16];
         __declspec(align(16)) DataType f[64];
      } coeffY;
      union
      {
         __m128 m[16];
         __declspec(align(16)) DataType f[64];
      } coeffZ;
#else // _WINDOWS
      union
      {
         __m128 m[4];
         DataType f[16] __attribute__((aligned(16)));
      } tempX;
      union
      {
         __m128 m[4];
         DataType f[16] __attribute__((aligned(16)));
      } tempY;
      union
      {
         __m128 m[4];
         DataType f[16] __attribute__((aligned(16)));
      } tempZ;
      memset(&(tempX.f[0]),0,16*sizeof(float));
      memset(&(tempY.f[0]),0,16*sizeof(float));
      memset(&(tempZ.f[0]),0,16*sizeof(float));
      union
      {
         __m128 m[16];
         DataType f[64] __attribute__((aligned(16)));
      } basisX;
      union
      {
         __m128 m[16];
         DataType f[64] __attribute__((aligned(16)));
      } basisY;
      union
      {
         __m128 m[16];
         DataType f[64] __attribute__((aligned(16)));
      } basisZ;
      union
      {
         __m128 m[16];
         DataType f[64] __attribute__((aligned(16)));
      } coeffX;
      union
      {
         __m128 m[16];
         DataType f[64] __attribute__((aligned(16)));
      } coeffY;
      union
      {
         __m128 m[16];
         DataType f[64] __attribute__((aligned(16)));
      } coeffZ;
#endif // _WINDOWS
#else
      int coord, incr1, incr2;
      DataType tempX[16], tempY[16], tempZ[16];
      DataType basisX[64], basisY[64], basisZ[64];
      DataType coeffX[64], coeffY[64], coeffZ[64];
#endif
      DataType gridVoxelSpacing[3]=
      {
         splineControlPoint->dx / referenceImage->dx,
         splineControlPoint->dy / referenceImage->dy,
         splineControlPoint->dz / referenceImage->dz
      };
      size_t voxelIndex;

      if(useHeaderInformation)
      {
         // The reference image is not necessarly aligned with the grid
         mat44 transformation;
         // reference: voxel to mm
         if(referenceImage->sform_code>0)
            transformation=referenceImage->sto_xyz;
         else transformation=referenceImage->qto_xyz;
         // affine: mm to mm
         if(splineControlPoint->num_ext>0)
            transformation=reg_mat44_mul(
                     reinterpret_cast<mat44 *>(splineControlPoint->ext_list[0].edata),
                  &transformation);
         // grid: mm to voxel
         if(splineControlPoint->sform_code>0)
            transformation=reg_mat44_mul(&(splineControlPoint->sto_ijk), &transformation);
         else transformation=reg_mat44_mul(&(splineControlPoint->qto_ijk), &transformation);

         float imageCoord[3], gridCoord[3], basis;
         for(z=0; z<referenceImage->nz; z++)
         {
            oldPre[0]=oldPre[1]=oldPre[2]=999999;
            voxelIndex=z*referenceImage->nx*referenceImage->ny;
            imageCoord[2]=z;
            for(y=0; y<referenceImage->ny; y++)
            {
               imageCoord[1]=y;
               for(x=0; x<referenceImage->nx; x++)
               {
                  imageCoord[0]=x;
                  // Compute the position in the grid
                  reg_mat44_mul(&transformation,imageCoord,gridCoord);
                  // Compute the anterior node coord
                  pre[0]=Floor(gridCoord[0]);
                  pre[1]=Floor(gridCoord[1]);
                  pre[2]=Floor(gridCoord[2]);
                  // Compute the basis values and their first derivatives
                  basis = gridCoord[0] - pre[0];
                  get_BSplineBasisValues<DataType>(basis, xBasis, xFirst);
                  basis = gridCoord[1] - pre[1];
                  get_BSplineBasisValues<DataType>(basis, yBasis, yFirst);
                  basis = gridCoord[2] - pre[2];
                  get_BSplineBasisValues<DataType>(basis, zBasis, zFirst);
                  // Compute the 64 basis values and the corresponding derivatives
#if USE_SSE
                  val.f[0]=yBasis[0];
                  val.f[1]=yBasis[1];
                  val.f[2]=yBasis[2];
                  val.f[3]=yBasis[3];
                  _yBasis=val.m;
                  val.f[0]=yFirst[0];
                  val.f[1]=yFirst[1];
                  val.f[2]=yFirst[2];
                  val.f[3]=yFirst[3];
                  _yFirst=val.m;
                  for(incr0=0; incr0<4; ++incr0)
                  {
                     val.m=_mm_set_ps1(zBasis[incr0]);
                     tempX.m[incr0]=_mm_mul_ps(_yBasis,val.m);
                     tempY.m[incr0]=_mm_mul_ps(_yFirst,val.m);
                     val.m=_mm_set_ps1(zFirst[incr0]);
                     tempZ.m[incr0]=_mm_mul_ps(_yBasis,val.m);
                  }
                  val.f[0]=xBasis[0];
                  val.f[1]=xBasis[1];
                  val.f[2]=xBasis[2];
                  val.f[3]=xBasis[3];
                  _xBasis=val.m;
                  val.f[0]=xFirst[0];
                  val.f[1]=xFirst[1];
                  val.f[2]=xFirst[2];
                  val.f[3]=xFirst[3];
                  _xFirst=val.m;
                  for(incr0=0; incr0<16; ++incr0)
                  {
                     val.m=_mm_set_ps1(tempX.f[incr0]);
                     basisX.m[incr0]=_mm_mul_ps(_xFirst,val.m);
                     val.m=_mm_set_ps1(tempY.f[incr0]);
                     basisY.m[incr0]=_mm_mul_ps(_xBasis,val.m);
                     val.m=_mm_set_ps1(tempZ.f[incr0]);
                     basisZ.m[incr0]=_mm_mul_ps(_xBasis,val.m);
                  }
#else
                  coord=0;
                  for(incr0=0; incr0<4; incr0++)
                  {
                     for(incr1=0; incr1<4; incr1++)
                     {
                        for(incr2=0; incr2<4; incr2++)
                        {
                           basisX[coord]=zBasis[incr0]*yBasis[incr1]*xFirst[incr2]; // z * y * x'
                           basisY[coord]=zBasis[incr0]*yFirst[incr1]*xBasis[incr2]; // z * y' * x
                           basisZ[coord]=zFirst[incr0]*yBasis[incr1]*xBasis[incr2]; // z' * y * x
                           ++coord;
                        }
                     }
                  }
#endif
                  // Fetch the required coefficients
                  if(oldPre[0]!=pre[0] || oldPre[1]!=pre[1] || oldPre[2]!=pre[2])
                  {
#ifdef USE_SSE
                     get_GridValues<DataType>(pre[0]-1,
                           pre[1]-1,
                           pre[2]-1,
                           splineControlPoint,
                           coeffPtrX,
                           coeffPtrY,
                           coeffPtrZ,
                           coeffX.f,
                           coeffY.f,
                           coeffZ.f,
                           false, // no approx
                           false // not disp
                           );
#else // USE_SSE
                     get_GridValues<DataType>(pre[0]-1,
                           pre[1]-1,
                           pre[2]-1,
                           splineControlPoint,
                           coeffPtrX,
                           coeffPtrY,
                           coeffPtrZ,
                           coeffX,
                           coeffY,
                           coeffZ,
                           false, // no approx
                           false // not disp
                           );
#endif // USE_SSE
                     oldPre[0]=pre[0];
                     oldPre[1]=pre[1];
                     oldPre[2]=pre[2];
                  }
                  // Compute the Jacobian matrix
#if USE_SSE
                  tempX_x =  _mm_set_ps1(0);
                  tempX_y =  _mm_set_ps1(0);
                  tempX_z =  _mm_set_ps1(0);
                  tempY_x =  _mm_set_ps1(0);
                  tempY_y =  _mm_set_ps1(0);
                  tempY_z =  _mm_set_ps1(0);
                  tempZ_x =  _mm_set_ps1(0);
                  tempZ_y =  _mm_set_ps1(0);
                  tempZ_z =  _mm_set_ps1(0);
                  //addition and multiplication of the 16 basis value and CP position for each axis
                  for(incr0=0; incr0<16; ++incr0)
                  {
                     tempX_x = _mm_add_ps(_mm_mul_ps(basisX.m[incr0], coeffX.m[incr0]), tempX_x );
                     tempX_y = _mm_add_ps(_mm_mul_ps(basisY.m[incr0], coeffX.m[incr0]), tempX_y );
                     tempX_z = _mm_add_ps(_mm_mul_ps(basisZ.m[incr0], coeffX.m[incr0]), tempX_z );

                     tempY_x = _mm_add_ps(_mm_mul_ps(basisX.m[incr0], coeffY.m[incr0]), tempY_x );
                     tempY_y = _mm_add_ps(_mm_mul_ps(basisY.m[incr0], coeffY.m[incr0]), tempY_y );
                     tempY_z = _mm_add_ps(_mm_mul_ps(basisZ.m[incr0], coeffY.m[incr0]), tempY_z );

                     tempZ_x = _mm_add_ps(_mm_mul_ps(basisX.m[incr0], coeffZ.m[incr0]), tempZ_x );
                     tempZ_y = _mm_add_ps(_mm_mul_ps(basisY.m[incr0], coeffZ.m[incr0]), tempZ_y );
                     tempZ_z = _mm_add_ps(_mm_mul_ps(basisZ.m[incr0], coeffZ.m[incr0]), tempZ_z );
                  }

                  //the values stored in SSE variables are transferred to normal float
                  val.m = tempX_x;
                  jacobianMatrix.m[0][0] = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                  val.m = tempX_y;
                  jacobianMatrix.m[0][1] = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                  val.m = tempX_z;
                  jacobianMatrix.m[0][2] = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                  val.m = tempY_x;
                  jacobianMatrix.m[1][0] = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                  val.m = tempY_y;
                  jacobianMatrix.m[1][1] = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                  val.m = tempY_z;
                  jacobianMatrix.m[1][2] = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                  val.m = tempZ_x;
                  jacobianMatrix.m[2][0] = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                  val.m = tempZ_y;
                  jacobianMatrix.m[2][1] = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                  val.m = tempZ_z;
                  jacobianMatrix.m[2][2] = val.f[0]+val.f[1]+val.f[2]+val.f[3];
#else
                  memset(&jacobianMatrix, 0, sizeof(mat33));
                  for(incr0=0; incr0<64; ++incr0)
                  {
                     jacobianMatrix.m[0][0] += basisX[incr0]*coeffX[incr0];
                     jacobianMatrix.m[0][1] += basisY[incr0]*coeffX[incr0];
                     jacobianMatrix.m[0][2] += basisZ[incr0]*coeffX[incr0];
                     jacobianMatrix.m[1][0] += basisX[incr0]*coeffY[incr0];
                     jacobianMatrix.m[1][1] += basisY[incr0]*coeffY[incr0];
                     jacobianMatrix.m[1][2] += basisZ[incr0]*coeffY[incr0];
                     jacobianMatrix.m[2][0] += basisX[incr0]*coeffZ[incr0];
                     jacobianMatrix.m[2][1] += basisY[incr0]*coeffZ[incr0];
                     jacobianMatrix.m[2][2] += basisZ[incr0]*coeffZ[incr0];
                  }
#endif
                  // reorient the matrix
                  jacobianMatrix=nifti_mat33_mul(reorientation,
                                                 jacobianMatrix);
                  if(JacobianMatrices!=nullptr)
                     JacobianMatrices[voxelIndex]=jacobianMatrix;
                  if(JacobianDeterminants!=nullptr)
                     JacobianDeterminants[voxelIndex] =
                           static_cast<DataType>(nifti_mat33_determ(jacobianMatrix));
                  ++voxelIndex;
               } // x
            } // y
         } // z
      }
      else
      {
         // The grid is assumed to be aligned with the reference image
#ifdef _OPENMP
#ifdef USE_SSE
#pragma omp parallel for default(none) \
   shared(referenceImage, gridVoxelSpacing, splineControlPoint, \
   coeffPtrX, coeffPtrY, coeffPtrZ,reorientation, JacobianMatrices, \
   JacobianDeterminants) \
   private(x, y, pre, oldPre, basis, val, \
   _xBasis, _xFirst, _yBasis, _yFirst, \
   tempX, tempY, tempZ, basisX, basisY, basisZ, \
   xBasis, xFirst, yBasis, yFirst, zBasis, zFirst, \
   coeffX, coeffY, coeffZ, incr0, \
   jacobianMatrix, voxelIndex, \
   tempX_x, tempX_y, tempX_z, tempY_x, tempY_y, tempY_z, tempZ_x, tempZ_y, tempZ_z)
#else // _USE_SEE
#pragma omp parallel for default(none) \
   shared(referenceImage, gridVoxelSpacing, splineControlPoint, \
   coeffPtrX, coeffPtrY, coeffPtrZ, reorientation, JacobianMatrices, \
   JacobianDeterminants) \
   private(x, y, pre, oldPre, basis, \
   basisX, basisY, basisZ, coord, tempX, tempY, tempZ, \
   xBasis, xFirst, yBasis, yFirst, zBasis, zFirst, \
   coeffX, coeffY, coeffZ, incr0, incr1, \
   jacobianMatrix, voxelIndex)
#endif // _USE_SEE
#endif // _USE_OPENMP
         for(z=0; z<referenceImage->nz; z++)
         {
            voxelIndex=z*referenceImage->nx*referenceImage->ny;
            oldPre[0]=oldPre[1]=oldPre[2]=999999;

            pre[2]=(int)((DataType)z/gridVoxelSpacing[2]);
            basis=(DataType)z/gridVoxelSpacing[2]-(DataType)pre[2];
            if(basis<0) basis=0; //rounding error
            get_BSplineBasisValues<DataType>(basis, zBasis, zFirst);

            for(y=0; y<referenceImage->ny; y++)
            {

               pre[1]=(int)((DataType)y/gridVoxelSpacing[1]);
               basis=(DataType)y/gridVoxelSpacing[1]-(DataType)pre[1];
               if(basis<0) basis=0; //rounding error
               get_BSplineBasisValues<DataType>(basis, yBasis, yFirst);

#if USE_SSE
               val.f[0]=yBasis[0];
               val.f[1]=yBasis[1];
               val.f[2]=yBasis[2];
               val.f[3]=yBasis[3];
               _yBasis=val.m;
               val.f[0]=yFirst[0];
               val.f[1]=yFirst[1];
               val.f[2]=yFirst[2];
               val.f[3]=yFirst[3];
               _yFirst=val.m;
               for(incr0=0; incr0<4; ++incr0)
               {
                  val.m=_mm_set_ps1(zBasis[incr0]);
                  tempX.m[incr0]=_mm_mul_ps(_yBasis,val.m);
                  tempY.m[incr0]=_mm_mul_ps(_yFirst,val.m);
                  val.m=_mm_set_ps1(zFirst[incr0]);
                  tempZ.m[incr0]=_mm_mul_ps(_yBasis,val.m);
               }
#else
               coord=0;
               for(incr0=0; incr0<4; incr0++)
               {
                  for(incr1=0; incr1<4; incr1++)
                  {
                     tempX[coord]=zBasis[incr0]*yBasis[incr1]; // z * y
                     tempY[coord]=zBasis[incr0]*yFirst[incr1];// z * y'
                     tempZ[coord]=zFirst[incr0]*yBasis[incr1]; // z'* y
                     coord++;
                  }
               }
#endif
               for(x=0; x<referenceImage->nx; x++)
               {

                  pre[0]=(int)((DataType)x/gridVoxelSpacing[0]);
                  basis=(DataType)x/gridVoxelSpacing[0]-(DataType)pre[0];
                  if(basis<0) basis=0; //rounding error
                  get_BSplineBasisValues<DataType>(basis, xBasis, xFirst);

#if USE_SSE
                  val.f[0]=xBasis[0];
                  val.f[1]=xBasis[1];
                  val.f[2]=xBasis[2];
                  val.f[3]=xBasis[3];
                  _xBasis=val.m;
                  val.f[0]=xFirst[0];
                  val.f[1]=xFirst[1];
                  val.f[2]=xFirst[2];
                  val.f[3]=xFirst[3];
                  _xFirst=val.m;
                  for(incr0=0; incr0<16; ++incr0)
                  {
                     val.m=_mm_set_ps1(tempX.f[incr0]);
                     basisX.m[incr0]=_mm_mul_ps(_xFirst,val.m);
                     val.m=_mm_set_ps1(tempY.f[incr0]);
                     basisY.m[incr0]=_mm_mul_ps(_xBasis,val.m);
                     val.m=_mm_set_ps1(tempZ.f[incr0]);
                     basisZ.m[incr0]=_mm_mul_ps(_xBasis,val.m);
                  }
#else
                  coord=0;
                  for(incr0=0; incr0<16; ++incr0)
                  {
                     for(incr1=0; incr1<4; ++incr1)
                     {
                        basisX[coord]=tempX[incr0]*xFirst[incr1];   // z * y * x'
                        basisY[coord]=tempY[incr0]*xBasis[incr1];    // z * y'* x
                        basisZ[coord]=tempZ[incr0]*xBasis[incr1];    // z'* y * x
                        coord++;
                     }
                  }
#endif

                  if(oldPre[0]!=pre[0] || oldPre[1]!=pre[1] || oldPre[2]!=pre[2])
                  {
#ifdef USE_SSE
                     get_GridValues<DataType>(pre[0],
                           pre[1],
                           pre[2],
                           splineControlPoint,
                           coeffPtrX,
                           coeffPtrY,
                           coeffPtrZ,
                           coeffX.f,
                           coeffY.f,
                           coeffZ.f,
                           false, // no approx
                           false // not disp
                           );
#else // USE_SSE
                     get_GridValues<DataType>(pre[0],
                           pre[1],
                           pre[2],
                           splineControlPoint,
                           coeffPtrX,
                           coeffPtrY,
                           coeffPtrZ,
                           coeffX,
                           coeffY,
                           coeffZ,
                           false, // no approx
                           false // not disp
                           );
#endif // USE_SSE
                     oldPre[0]=pre[0];
                     oldPre[1]=pre[1];
                     oldPre[2]=pre[2];
                  }
#if USE_SSE
                  tempX_x =  _mm_set_ps1(0);
                  tempX_y =  _mm_set_ps1(0);
                  tempX_z =  _mm_set_ps1(0);
                  tempY_x =  _mm_set_ps1(0);
                  tempY_y =  _mm_set_ps1(0);
                  tempY_z =  _mm_set_ps1(0);
                  tempZ_x =  _mm_set_ps1(0);
                  tempZ_y =  _mm_set_ps1(0);
                  tempZ_z =  _mm_set_ps1(0);
                  //addition and multiplication of the 16 basis value and CP position for each axis
                  for(incr0=0; incr0<16; ++incr0)
                  {
                     tempX_x = _mm_add_ps(_mm_mul_ps(basisX.m[incr0], coeffX.m[incr0]), tempX_x );
                     tempX_y = _mm_add_ps(_mm_mul_ps(basisY.m[incr0], coeffX.m[incr0]), tempX_y );
                     tempX_z = _mm_add_ps(_mm_mul_ps(basisZ.m[incr0], coeffX.m[incr0]), tempX_z );

                     tempY_x = _mm_add_ps(_mm_mul_ps(basisX.m[incr0], coeffY.m[incr0]), tempY_x );
                     tempY_y = _mm_add_ps(_mm_mul_ps(basisY.m[incr0], coeffY.m[incr0]), tempY_y );
                     tempY_z = _mm_add_ps(_mm_mul_ps(basisZ.m[incr0], coeffY.m[incr0]), tempY_z );

                     tempZ_x = _mm_add_ps(_mm_mul_ps(basisX.m[incr0], coeffZ.m[incr0]), tempZ_x );
                     tempZ_y = _mm_add_ps(_mm_mul_ps(basisY.m[incr0], coeffZ.m[incr0]), tempZ_y );
                     tempZ_z = _mm_add_ps(_mm_mul_ps(basisZ.m[incr0], coeffZ.m[incr0]), tempZ_z );
                  }

                  //the values stored in SSE variables are transferred to normal float
                  val.m = tempX_x;
                  jacobianMatrix.m[0][0] = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                  val.m = tempX_y;
                  jacobianMatrix.m[0][1] = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                  val.m = tempX_z;
                  jacobianMatrix.m[0][2] = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                  val.m = tempY_x;
                  jacobianMatrix.m[1][0] = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                  val.m = tempY_y;
                  jacobianMatrix.m[1][1] = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                  val.m = tempY_z;
                  jacobianMatrix.m[1][2] = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                  val.m = tempZ_x;
                  jacobianMatrix.m[2][0] = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                  val.m = tempZ_y;
                  jacobianMatrix.m[2][1] = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                  val.m = tempZ_z;
                  jacobianMatrix.m[2][2] = val.f[0]+val.f[1]+val.f[2]+val.f[3];
#else
                  memset(&jacobianMatrix, 0, sizeof(mat33));
                  for(incr0=0; incr0<64; ++incr0)
                  {
                     jacobianMatrix.m[0][0] += basisX[incr0]*coeffX[incr0];
                     jacobianMatrix.m[0][1] += basisY[incr0]*coeffX[incr0];
                     jacobianMatrix.m[0][2] += basisZ[incr0]*coeffX[incr0];
                     jacobianMatrix.m[1][0] += basisX[incr0]*coeffY[incr0];
                     jacobianMatrix.m[1][1] += basisY[incr0]*coeffY[incr0];
                     jacobianMatrix.m[1][2] += basisZ[incr0]*coeffY[incr0];
                     jacobianMatrix.m[2][0] += basisX[incr0]*coeffZ[incr0];
                     jacobianMatrix.m[2][1] += basisY[incr0]*coeffZ[incr0];
                     jacobianMatrix.m[2][2] += basisZ[incr0]*coeffZ[incr0];
                  }
#endif
                  jacobianMatrix=nifti_mat33_mul(reorientation,
                                                 jacobianMatrix);
                  if(JacobianMatrices!=nullptr)
                     JacobianMatrices[voxelIndex]=jacobianMatrix;
                  if(JacobianDeterminants!=nullptr)
                     JacobianDeterminants[voxelIndex] =
                           static_cast<DataType>(nifti_mat33_determ(jacobianMatrix));
                  ++voxelIndex;
               } // loop over x
            } // loop over y
         } // loop over z
      } // end if the grid is aligned with the reference image
   } // end if no approximation
   return;
}
/* *************************************************************** */
double reg_spline_getJacobianPenaltyTerm(nifti_image *splineControlPoint,
                                         nifti_image *referenceImage,
                                         bool approximation,
                                         bool useHeaderInformation
                                         )
{
   // An array to store the Jacobian determinant is created
   size_t detNumber=0;
   if(approximation)
   {
      detNumber = (size_t)(splineControlPoint->nx-2) *
            (splineControlPoint->ny-2);
      if(splineControlPoint->nz>1)
         detNumber *= (size_t)(splineControlPoint->nz-2);
   }
   else detNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);

   void *JacobianDetermiantArray=malloc(detNumber*splineControlPoint->nbyper);

   // The jacobian determinants are computed
   if(splineControlPoint->nz==1)
   {
      switch(splineControlPoint->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         reg_cubic_spline_jacobian2D<float>(splineControlPoint,
                                      referenceImage,
                                      nullptr,
                                      static_cast<float *>(JacobianDetermiantArray),
                                      approximation,
                                      useHeaderInformation);
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_cubic_spline_jacobian2D<double>(splineControlPoint,
                                       referenceImage,
                                       nullptr,
                                       static_cast<double *>(JacobianDetermiantArray),
                                       approximation,
                                       useHeaderInformation);
         break;
      default:
         NR_FATAL_ERROR("Only single or double precision has been implemented");
      }
   }
   else
   {
      switch(splineControlPoint->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         reg_cubic_spline_jacobian3D<float>(splineControlPoint,
                                      referenceImage,
                                      nullptr,
                                      static_cast<float *>(JacobianDetermiantArray),
                                      approximation,
                                      useHeaderInformation);
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_cubic_spline_jacobian3D<double>(splineControlPoint,
                                       referenceImage,
                                       nullptr,
                                       static_cast<double *>(JacobianDetermiantArray),
                                       approximation,
                                       useHeaderInformation);
         break;
      default:
         NR_FATAL_ERROR("Only single or double precision has been implemented");
      }
   }
   // The jacobian determinant are averaged
   double penaltySum=0.;
   switch(splineControlPoint->datatype)
   {
   case NIFTI_TYPE_FLOAT32:
      {
         float *jacDetPtr = static_cast<float *>(JacobianDetermiantArray);
         for(size_t i=0; i<detNumber; ++i)
         {
            double logDet = log(jacDetPtr[i]);
#ifdef USE_SQUARE_LOG_JAC
            penaltySum += logDet * logDet;
#else
            penaltySum += fasb(logDet);
#endif
         }
      }
      break;
   case NIFTI_TYPE_FLOAT64:
      {
         double *jacDetPtr = static_cast<double *>(JacobianDetermiantArray);
         for(size_t i=0; i<detNumber; ++i)
         {
            double logDet = log(jacDetPtr[i]);
#ifdef USE_SQUARE_LOG_JAC
            penaltySum += logDet * logDet;
#else
            penaltySum += fasb(logDet);
#endif
         }
      }
      break;
   }
   // The allocated array is free'ed
   if(JacobianDetermiantArray)
      free(JacobianDetermiantArray);
   JacobianDetermiantArray=nullptr;
   // The penalty term value is normalised and returned
   return penaltySum/(double)detNumber;
}
/* *************************************************************** */
/* *************************************************************** */
template<class DataType>
void reg_spline_jacobianDetGradient2D(nifti_image *splineControlPoint,
                                      nifti_image *referenceImage,
                                      nifti_image *gradientImage,
                                      float weight,
                                      bool approximation,
                                      bool useHeaderInformation)
{
   size_t arraySize = 0;
   if(approximation)
      arraySize = (size_t)(splineControlPoint->nx-2) *
            (splineControlPoint->ny-2);
   else arraySize = NiftiImage::calcVoxelNumber(referenceImage, 2);
   // Allocate arrays to store determinants and matrices
   mat33 *jacobianMatrices=(mat33 *)malloc(arraySize * sizeof(mat33));
   DataType *jacobianDeterminant=(DataType *)malloc(arraySize * sizeof(DataType));

   // Compute all the required Jacobian determinants and matrices
   reg_cubic_spline_jacobian2D<DataType>(splineControlPoint,
                                referenceImage,
                                jacobianMatrices,
                                jacobianDeterminant,
                                approximation,
                                useHeaderInformation);

   // The gradient are now computed for every control point
   DataType *gradientImagePtrX = static_cast<DataType *>(gradientImage->data);
   DataType *gradientImagePtrY = &gradientImagePtrX[NiftiImage::calcVoxelNumber(gradientImage, 2)];

   // Matrices to be used to convert the gradient from voxel to mm
   mat33 jacobianMatrix, reorientation;
   if(splineControlPoint->sform_code>0)
      reorientation = reg_mat44_to_mat33(&splineControlPoint->sto_xyz);
   else reorientation = reg_mat44_to_mat33(&splineControlPoint->qto_xyz);

   // Ratio to be used for normalisation
   size_t jacobianNumber;
   if(approximation)
      jacobianNumber = NiftiImage::calcVoxelNumber(splineControlPoint, 2);
   else jacobianNumber = arraySize;
   DataType ratio[2] =
   {
      referenceImage->dx*weight / ((DataType)jacobianNumber*splineControlPoint->dx),
      referenceImage->dy*weight / ((DataType)jacobianNumber*splineControlPoint->dy)
   };

   // Only information at the control point position is considered
   if(approximation)
   {
      DataType basisX[9], basisY[9];
	  DataType normal[3] = { 1.f / 6.f, 2.f / 3.f, 1.f / 6.f };
	  DataType first[3] = { -0.5f, 0.f, 0.5f };
      DataType jacobianConstraint[2], detJac;
      size_t coord=0, jacIndex, index;
      int x, y, pixelX, pixelY;
      // INVERTED ON PURPOSE
      for(int b=2; b>-1; --b)
      {
         for(int a=2; a>-1; --a)
         {
            basisX[coord]=normal[b]*first[a];
            basisY[coord]=first[b]*normal[a];
            coord++;
         }
      }


#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(splineControlPoint, jacobianMatrices, jacobianDeterminant, basisX, basisY, \
   ratio, gradientImagePtrX, gradientImagePtrY, reorientation) \
   private(x, index, jacobianConstraint, pixelX, pixelY, jacIndex, coord, detJac, jacobianMatrix)
#endif
      for(y=0; y<splineControlPoint->ny; y++)
      {
         index=y*splineControlPoint->nx;
         for(x=0; x<splineControlPoint->nx; x++)
         {

            jacobianConstraint[0]=jacobianConstraint[1]=0;

            // Loop over all the control points in the surrounding area
            coord=0;
            for(pixelY=(int)(y-1); pixelY<(int)(y+2); ++pixelY)
            {
               if(pixelY>0 && pixelY<splineControlPoint->ny-1)
               {

                  for(pixelX=(int)(x-1); pixelX<(int)(x+2); ++pixelX)
                  {
                     if(pixelX>0 && pixelX<splineControlPoint->nx-1)
                     {

                        jacIndex = (pixelY-1) *
                              (splineControlPoint->nx-2)+pixelX-1;
                        detJac = (double)jacobianDeterminant[jacIndex];

                        if(detJac>0)
                        {
                           jacobianMatrix = jacobianMatrices[jacIndex];
#ifdef USE_SQUARE_LOG_JAC
                           detJac = 2.0*log(detJac) / detJac;
#else
                           detJac = (log(detJac)>0?1.0:-1.0) / detJac;
#endif
                           addJacobianGradientValues<DataType>(jacobianMatrix,
                                                            detJac,
                                                            basisX[coord],
                                                            basisY[coord],
                                                            jacobianConstraint);
                        }
                     } // if x
                     coord++;
                  }// x
               }// if y
               else coord+=3;
            }// y
            // (Marc) I removed the normalisation by the voxel number as each gradient has to be normalised in the same way (NMI, BE, JAC)
            gradientImagePtrX[index] += ratio[0] *
                  ( reorientation.m[0][0]*jacobianConstraint[0]
                  + reorientation.m[0][1]*jacobianConstraint[1]);
            gradientImagePtrY[index] += ratio[1] *
                  ( reorientation.m[1][0]*jacobianConstraint[0]
                  + reorientation.m[1][1]*jacobianConstraint[1]);
            ++index;
         }
      }
   } // end if approximation
   else
   {
      // All voxels are considered
      // Force to use the header information if the grid contains an affine ext
      if(splineControlPoint->num_ext>0)
         useHeaderInformation=true;
      if(useHeaderInformation)
      {
         // The header information is considered
         NR_FATAL_ERROR("Not implemented yet");
      } // end if use header information
      else
      {
         // assumes that the reference and grid image are aligned
         DataType gridVoxelSpacing[2];
         gridVoxelSpacing[0] = splineControlPoint->dx / referenceImage->dx;
         gridVoxelSpacing[1] = splineControlPoint->dy / referenceImage->dy;

         DataType xBasis, yBasis, basis;
         DataType xFirst, yFirst;
         DataType basisValues[2];
         unsigned jacIndex;

         int x, y, xPre, yPre, pixelX, pixelY, index;
         DataType jacobianConstraint[2];
         double detJac;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(splineControlPoint, gridVoxelSpacing, referenceImage, jacobianDeterminant, ratio, \
   jacobianMatrices, gradientImagePtrX, gradientImagePtrY, reorientation) \
   private(x, xPre, yPre, pixelX, pixelY, jacobianConstraint, \
   basis, xBasis, yBasis, xFirst, yFirst, jacIndex, index, detJac, \
   jacobianMatrix, basisValues)
#endif
         for(y=0; y<splineControlPoint->ny; y++)
         {
            index=y*splineControlPoint->nx;
            for(x=0; x<splineControlPoint->nx; x++)
            {

               jacobianConstraint[0]=jacobianConstraint[1]=0.;

               // Loop over all the control points in the surrounding area

               for(pixelY=Ceil((y-3)*gridVoxelSpacing[1]); pixelY<=Ceil((y+1)*gridVoxelSpacing[1]); pixelY++)
               {
                  if(pixelY>-1 && pixelY<referenceImage->ny)
                  {

                     yPre=(int)((DataType)pixelY/gridVoxelSpacing[1]);
                     basis=(DataType)pixelY/gridVoxelSpacing[1]-(DataType)yPre;
                     get_BSplineBasisValue<DataType>(basis,y-yPre,yBasis,yFirst);

                     jacIndex = pixelY*referenceImage->nx+Ceil((x-3)*gridVoxelSpacing[0]);

                     for(pixelX=Ceil((x-3)*gridVoxelSpacing[0]); pixelX<=Ceil((x+1)*gridVoxelSpacing[0]); pixelX++)
                     {
                        if(pixelX>-1 && pixelX<referenceImage->nx && (yFirst!=0 || yBasis!=0))
                        {

                           detJac = jacobianDeterminant[jacIndex];

                           xPre=(int)((DataType)pixelX/gridVoxelSpacing[0]);
                           basis=(DataType)pixelX/gridVoxelSpacing[0]-(DataType)xPre;
                           get_BSplineBasisValue<DataType>(basis,x-xPre,xBasis,xFirst);

                           if(detJac>0 && (xBasis!=0 ||xFirst!=0))
                           {

                              jacobianMatrix = jacobianMatrices[jacIndex];

                              basisValues[0] = xFirst * yBasis ;
                              basisValues[1] = xBasis * yFirst ;

                              jacobianMatrix = jacobianMatrices[jacIndex];
#ifdef USE_SQUARE_LOG_JAC
                              detJac= 2.0*log(detJac) / detJac;
#else
                              detJac = (log(detJac)>0?1.0:-1.0) / detJac;
#endif
                              addJacobianGradientValues<DataType>(jacobianMatrix,
                                                               detJac,
                                                               basisValues[0],
                                    basisValues[1],
                                    jacobianConstraint);
                           }
                        } // if x
                        jacIndex++;
                     }// x
                  }// if y
               }// y
               // (Marc) I removed the normalisation by the voxel number as each gradient has to be normalised in the same way (NMI, BE, JAC)
               gradientImagePtrX[index] += ratio[0] *
                     ( reorientation.m[0][0]*jacobianConstraint[0]
                     + reorientation.m[0][1]*jacobianConstraint[1]);
               gradientImagePtrY[index] += ratio[1] *
                     ( reorientation.m[1][0]*jacobianConstraint[0]
                     + reorientation.m[1][1]*jacobianConstraint[1]);
               index++;
            }
         }
      }
   }
   // Allocated arrays are free'ed
   free(jacobianMatrices);
   free(jacobianDeterminant);
}
/* *************************************************************** */
template<class DataType>
void reg_spline_jacobianDetGradient3D(nifti_image *splineControlPoint,
                                      nifti_image *referenceImage,
                                      nifti_image *gradientImage,
                                      float weight,
                                      bool approximation,
                                      bool useHeaderInformation)
{
   size_t arraySize = 0;
   if(approximation)
      arraySize = (size_t)(splineControlPoint->nx-2) *
            (splineControlPoint->ny-2) * (splineControlPoint->nz-2);
   else arraySize = NiftiImage::calcVoxelNumber(referenceImage, 3);
   // Allocate arrays to store determinants and matrices
   mat33 *jacobianMatrices=(mat33 *)malloc(arraySize * sizeof(mat33));
   DataType *jacobianDeterminant=(DataType *)malloc(arraySize * sizeof(DataType));

   // Compute all the required Jacobian determinants and matrices
   reg_cubic_spline_jacobian3D<DataType>(splineControlPoint,
                                referenceImage,
                                jacobianMatrices,
                                jacobianDeterminant,
                                approximation,
                                useHeaderInformation);

   // The gradient are now computed for every control point
   const size_t voxelNumber = NiftiImage::calcVoxelNumber(gradientImage, 3);
   DataType *gradientImagePtrX = static_cast<DataType *>(gradientImage->data);
   DataType *gradientImagePtrY = &gradientImagePtrX[voxelNumber];
   DataType *gradientImagePtrZ = &gradientImagePtrY[voxelNumber];

   // Matrices to be used to convert the gradient from voxel to mm
   mat33 jacobianMatrix, reorientation;
   if(splineControlPoint->sform_code>0)
      reorientation = reg_mat44_to_mat33(&splineControlPoint->sto_xyz);
   else reorientation = reg_mat44_to_mat33(&splineControlPoint->qto_xyz);

   // Ratio to be used for normalisation
   size_t jacobianNumber;
   if(approximation)
      jacobianNumber = NiftiImage::calcVoxelNumber(splineControlPoint, 3);
   else jacobianNumber = arraySize;
   DataType ratio[3] =
   {
      referenceImage->dx*weight / ((DataType)jacobianNumber*splineControlPoint->dx),
      referenceImage->dy*weight / ((DataType)jacobianNumber*splineControlPoint->dy),
      referenceImage->dz*weight / ((DataType)jacobianNumber*splineControlPoint->dz)
   };

   // Only information at the control point position is considered
   if(approximation)
   {
      DataType basisX[27], basisY[27], basisZ[27];
      DataType normal[3]= {1.f/6.f, 2.f/3.f, 1.f/6.f};
      DataType first[3]= {-0.5f, 0.f, 0.5f};
      DataType jacobianConstraint[3], detJac;
      size_t coord=0, jacIndex, index;
      int x, y, z, pixelX, pixelY, pixelZ;
      // INVERTED ON PURPOSE
      for(int c=2; c>-1; --c)
      {
         for(int b=2; b>-1; --b)
         {
            for(int a=2; a>-1; --a)
            {
               basisX[coord]=normal[c]*normal[b]*first[a];
               basisY[coord]=normal[c]*first[b]*normal[a];
               basisZ[coord]=first[c]*normal[b]*normal[a];
               coord++;
            }
         }
      }


#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(splineControlPoint, jacobianMatrices, jacobianDeterminant, basisX, basisY, basisZ, \
   ratio, gradientImagePtrX, gradientImagePtrY, gradientImagePtrZ, reorientation) \
   private(x, y, index, jacobianConstraint, pixelX, pixelY, pixelZ, jacIndex, coord, \
   detJac, jacobianMatrix)
#endif
      for(z=0; z<splineControlPoint->nz; z++)
      {
         index=z*splineControlPoint->nx*splineControlPoint->ny;
         for(y=0; y<splineControlPoint->ny; y++)
         {
            for(x=0; x<splineControlPoint->nx; x++)
            {

               jacobianConstraint[0]=jacobianConstraint[1]=jacobianConstraint[2]=0;

               // Loop over all the control points in the surrounding area
               coord=0;
               for(pixelZ=(int)(z-1); pixelZ<(int)(z+2); ++pixelZ)
               {
                  if(pixelZ>0 && pixelZ<splineControlPoint->nz-1)
                  {

                     for(pixelY=(int)(y-1); pixelY<(int)(y+2); ++pixelY)
                     {
                        if(pixelY>0 && pixelY<splineControlPoint->ny-1)
                        {

                           for(pixelX=(int)(x-1); pixelX<(int)(x+2); ++pixelX)
                           {
                              if(pixelX>0 && pixelX<splineControlPoint->nx-1)
                              {

                                 jacIndex = ((pixelZ-1)*(splineControlPoint->ny-2)+pixelY-1) *
                                       (splineControlPoint->nx-2)+pixelX-1;
                                 detJac = (double)jacobianDeterminant[jacIndex];

                                 if(detJac>0)
                                 {
                                    jacobianMatrix = jacobianMatrices[jacIndex];
#ifdef USE_SQUARE_LOG_JAC
                                    detJac = 2.0*log(detJac) / detJac;
#else
                                    detJac = (log(detJac)>0?1.0:-1.0) / detJac;
#endif
                                    addJacobianGradientValues<DataType>(jacobianMatrix,
                                                                     detJac,
                                                                     basisX[coord],
                                                                     basisY[coord],
                                                                     basisZ[coord],
                                                                     jacobianConstraint);
                                 }
                              } // if x
                              coord++;
                           }// x
                        }// if y
                        else coord+=3;
                     }// y
                  }// if z
                  else coord+=9;
               } // z
               // (Marc) I removed the normalisation by the voxel number as each gradient has to be normalised in the same way (NMI, BE, JAC)
               gradientImagePtrX[index] += ratio[0] *
                     ( reorientation.m[0][0]*jacobianConstraint[0]
                     + reorientation.m[0][1]*jacobianConstraint[1]
                     + reorientation.m[0][2]*jacobianConstraint[2]);
               gradientImagePtrY[index] += ratio[1] *
                     ( reorientation.m[1][0]*jacobianConstraint[0]
                     + reorientation.m[1][1]*jacobianConstraint[1]
                     + reorientation.m[1][2]*jacobianConstraint[2]);
               gradientImagePtrZ[index] += ratio[2] *
                     ( reorientation.m[2][0]*jacobianConstraint[0]
                     + reorientation.m[2][1]*jacobianConstraint[1]
                     + reorientation.m[2][2]*jacobianConstraint[2]);
               ++index;
            }
         }
      }
   } // end if approximation
   else
   {
      // All voxels are considered
      // Force to use the header information if the grid contains an affine ext
      if(splineControlPoint->num_ext>0)
         useHeaderInformation=true;
      if(useHeaderInformation)
      {
         // The header information is considered
         NR_FATAL_ERROR("Not implemented yet");
      } // end if use header information
      else
      {
         // assumes that the reference and grid image are aligned
         DataType gridVoxelSpacing[3];
         gridVoxelSpacing[0] = splineControlPoint->dx / referenceImage->dx;
         gridVoxelSpacing[1] = splineControlPoint->dy / referenceImage->dy;
         gridVoxelSpacing[2] = splineControlPoint->dz / referenceImage->dz;

         DataType xBasis, yBasis, zBasis, basis;
         DataType xFirst, yFirst, zFirst;
         DataType basisValues[3];
         unsigned jacIndex;

         int x, y, z, xPre, yPre, zPre, pixelX, pixelY, pixelZ, index;
         DataType jacobianConstraint[3];
         double detJac;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(splineControlPoint, gridVoxelSpacing, referenceImage, jacobianDeterminant, ratio, \
   jacobianMatrices, gradientImagePtrX, gradientImagePtrY, gradientImagePtrZ, reorientation) \
   private(x, y, xPre, yPre, zPre, pixelX, pixelY, pixelZ, jacobianConstraint, \
   basis, xBasis, yBasis, zBasis, xFirst, yFirst, zFirst, jacIndex, index, detJac, \
   jacobianMatrix, basisValues)
#endif
         for(z=0; z<splineControlPoint->nz; z++)
         {
            index=z*splineControlPoint->nx*splineControlPoint->ny;
            for(y=0; y<splineControlPoint->ny; y++)
            {
               for(x=0; x<splineControlPoint->nx; x++)
               {

                  jacobianConstraint[0]=jacobianConstraint[1]=jacobianConstraint[2]=0.;

                  // Loop over all the control points in the surrounding area
                  for(pixelZ=Ceil((z-3)*gridVoxelSpacing[2]); pixelZ<=Ceil((z+1)*gridVoxelSpacing[2]); pixelZ++)
                  {
                     if(pixelZ>-1 && pixelZ<referenceImage->nz)
                     {

                        zPre=(int)((DataType)pixelZ/gridVoxelSpacing[2]);
                        basis=(DataType)pixelZ/gridVoxelSpacing[2]-(DataType)zPre;
                        get_BSplineBasisValue<DataType>(basis,z-zPre,zBasis,zFirst);

                        for(pixelY=Ceil((y-3)*gridVoxelSpacing[1]); pixelY<=Ceil((y+1)*gridVoxelSpacing[1]); pixelY++)
                        {
                           if(pixelY>-1 && pixelY<referenceImage->ny && (zFirst!=0 || zBasis!=0))
                           {

                              yPre=(int)((DataType)pixelY/gridVoxelSpacing[1]);
                              basis=(DataType)pixelY/gridVoxelSpacing[1]-(DataType)yPre;
                              get_BSplineBasisValue<DataType>(basis,y-yPre,yBasis,yFirst);

                              jacIndex = (pixelZ*referenceImage->ny+pixelY)*referenceImage->nx+Ceil((x-3)*gridVoxelSpacing[0]);

                              for(pixelX=Ceil((x-3)*gridVoxelSpacing[0]); pixelX<=Ceil((x+1)*gridVoxelSpacing[0]); pixelX++)
                              {
                                 if(pixelX>-1 && pixelX<referenceImage->nx && (yFirst!=0 || yBasis!=0))
                                 {

                                    detJac = jacobianDeterminant[jacIndex];

                                    xPre=(int)((DataType)pixelX/gridVoxelSpacing[0]);
                                    basis=(DataType)pixelX/gridVoxelSpacing[0]-(DataType)xPre;
                                    get_BSplineBasisValue<DataType>(basis,x-xPre,xBasis,xFirst);

                                    if(detJac>0 && (xBasis!=0 ||xFirst!=0))
                                    {

                                       jacobianMatrix = jacobianMatrices[jacIndex];

                                       basisValues[0] = xFirst * yBasis * zBasis ;
                                       basisValues[1] = xBasis * yFirst * zBasis ;
                                       basisValues[2] = xBasis * yBasis * zFirst ;

                                       jacobianMatrix = jacobianMatrices[jacIndex];
#ifdef USE_SQUARE_LOG_JAC
                                       detJac= 2.0*log(detJac) / detJac;
#else
                                       detJac = (log(detJac)>0?1.0:-1.0) / detJac;
#endif
                                       addJacobianGradientValues<DataType>(jacobianMatrix,
                                                                        detJac,
                                                                        basisValues[0],
                                             basisValues[1],
                                             basisValues[2],
                                             jacobianConstraint);
                                    }
                                 } // if x
                                 jacIndex++;
                              }// x
                           }// if y
                        }// y
                     }// if z
                  } // z
                  // (Marc) I removed the normalisation by the voxel number as each gradient has to be normalised in the same way (NMI, BE, JAC)
                  gradientImagePtrX[index] += ratio[0] *
                        ( reorientation.m[0][0]*jacobianConstraint[0]
                        + reorientation.m[0][1]*jacobianConstraint[1]
                        + reorientation.m[0][2]*jacobianConstraint[2]);
                  gradientImagePtrY[index] += ratio[1] *
                        ( reorientation.m[1][0]*jacobianConstraint[0]
                        + reorientation.m[1][1]*jacobianConstraint[1]
                        + reorientation.m[1][2]*jacobianConstraint[2]);
                  gradientImagePtrZ[index] += ratio[2] *
                        ( reorientation.m[2][0]*jacobianConstraint[0]
                        + reorientation.m[2][1]*jacobianConstraint[1]
                        + reorientation.m[2][2]*jacobianConstraint[2]);
                  index++;
               }
            }
         }
      }
   }
   // Allocated arrays are free'ed
   free(jacobianMatrices);
   free(jacobianDeterminant);
}
/* *************************************************************** */
void reg_spline_getJacobianPenaltyTermGradient(nifti_image *splineControlPoint,
                                               nifti_image *referenceImage,
                                               nifti_image *gradientImage,
                                               float weight,
                                               bool approximation,
                                               bool useHeaderInformation)
{
   if(splineControlPoint->datatype != gradientImage->datatype)
      NR_FATAL_ERROR("The input images are expected to be of the same type");

   if(splineControlPoint->nz==1)
   {
      switch(splineControlPoint->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         reg_spline_jacobianDetGradient2D<float>(splineControlPoint,
                                                 referenceImage,
                                                 gradientImage,
                                                 weight,
                                                 approximation,
                                                 useHeaderInformation);
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_spline_jacobianDetGradient2D<double>(splineControlPoint,
                                                  referenceImage,
                                                  gradientImage,
                                                  weight,
                                                  approximation,
                                                  useHeaderInformation);
         break;
      default:
         NR_FATAL_ERROR("Function only usable with single or double floating precision");
      }
   }
   else
   {
      switch(splineControlPoint->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         reg_spline_jacobianDetGradient3D<float>(splineControlPoint,
                                                 referenceImage,
                                                 gradientImage,
                                                 weight,
                                                 approximation,
                                                 useHeaderInformation);
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_spline_jacobianDetGradient3D<double>(splineControlPoint,
                                                  referenceImage,
                                                  gradientImage,
                                                  weight,
                                                  approximation,
                                                  useHeaderInformation);
         break;
      default:
         NR_FATAL_ERROR("Function only usable with single or double floating precision");
      }
   }
}
/* *************************************************************** */
/* *************************************************************** */
template<class DataType>
double reg_spline_correctFolding2D(nifti_image *splineControlPoint,
                                   nifti_image *referenceImage,
                                   bool approximation,
                                   bool useHeaderInformation)
{
#ifdef WIN32
   long i;
   long jacobianNumber;
   if(approximation)
      jacobianNumber = (long)(splineControlPoint->nx-2)*(splineControlPoint->ny-2);
   else jacobianNumber = (long)NiftiImage::calcVoxelNumber(referenceImage, 2);
#else
   size_t i;
   size_t jacobianNumber;
   if(approximation)
      jacobianNumber = (size_t)(splineControlPoint->nx-2)*(splineControlPoint->ny-2);
   else jacobianNumber = NiftiImage::calcVoxelNumber(referenceImage, 2);
#endif
   mat33 *jacobianMatrices=(mat33 *)malloc(jacobianNumber*sizeof(mat33));
   DataType *jacobianDeterminant=(DataType *)malloc(jacobianNumber*sizeof(DataType));

   reg_cubic_spline_jacobian2D(splineControlPoint,
                         referenceImage,
                         jacobianMatrices,
                         jacobianDeterminant,
                         approximation,
                         useHeaderInformation);

   /* The current Penalty term value is computed */
   double penaltyTerm =0., logDet;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(jacobianNumber, jacobianDeterminant) \
   private(logDet) \
   reduction(+:penaltyTerm)
#endif
   for(i=0; i< jacobianNumber; i++)
   {
      logDet = log(jacobianDeterminant[i]);
#ifdef USE_SQUARE_LOG_JAC
      penaltyTerm += logDet*logDet;
#else
      penaltyTerm +=  fabs(log(logDet));
#endif
   }
   if(penaltyTerm==penaltyTerm)
   {
      free(jacobianDeterminant);
      free(jacobianMatrices);
      return penaltyTerm/(double)(jacobianNumber);
   }

   mat33 jacobianMatrix, reorientation;
   if(splineControlPoint->sform_code>0)
      reorientation = reg_mat44_to_mat33(&splineControlPoint->sto_xyz);
   else reorientation = reg_mat44_to_mat33(&splineControlPoint->qto_xyz);

   const size_t nodeNumber = NiftiImage::calcVoxelNumber(splineControlPoint, 3);
   DataType *controlPointPtrX = static_cast<DataType *>(splineControlPoint->data);
   DataType *controlPointPtrY = &controlPointPtrX[nodeNumber];

   DataType basisValues[2], foldingCorrection[2], gradient[2], norm;
   DataType xBasis=0, yBasis=0, xFirst=0, yFirst=0;
   int x, y, id, pixelX, pixelY, jacIndex;
   bool correctFolding;
   double detJac;

   if(approximation)
   {
      // The function discretise the Jacobian only at the control point positions

#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(splineControlPoint, jacobianDeterminant, jacobianMatrices, \
   controlPointPtrX, controlPointPtrY, reorientation) \
   private(x, pixelX, pixelY, foldingCorrection, \
   xBasis, yBasis, xFirst, yFirst, jacIndex, detJac, \
   jacobianMatrix, basisValues, norm, correctFolding, id, gradient)
#endif
      for(y=0; y<splineControlPoint->ny; y++)
      {
         for(x=0; x<splineControlPoint->nx; x++)
         {

            foldingCorrection[0]=foldingCorrection[1]=0;
            correctFolding=false;

            // Loop over all the control points in the surrounding area
            for(pixelY=(int)((y-1)); pixelY<(int)((y+2)); pixelY++)
            {
               if(pixelY>0 && pixelY<splineControlPoint->ny-1)
               {

                  for(pixelX=(int)((x-1)); pixelX<(int)((x+2)); pixelX++)
                  {
                     if(pixelX>0 && pixelX<splineControlPoint->nx-1)
                     {

                        jacIndex = (pixelY-1)*
                              (splineControlPoint->nx-2)+pixelX-1;
                        detJac = jacobianDeterminant[jacIndex];

                        if(detJac<=0)
                        {
                           get_BSplineBasisValue<DataType>(0, y-pixelY+1, yBasis, yFirst);
                           get_BSplineBasisValue<DataType>(0, x-pixelX+1, xBasis, xFirst);

                           basisValues[0] = xFirst * yBasis ;
                           basisValues[1] = xBasis * yFirst ;

                           jacobianMatrix = jacobianMatrices[jacIndex];

                           correctFolding=true;
                           addJacobianGradientValues<DataType>(jacobianMatrix,
                                                            1.0,
                                                            basisValues[0],
                                 basisValues[1],
                                 foldingCorrection);
                        } // detJac<0
                     } // if x
                  }// x
               }// if y
            }// y
            if(correctFolding)
            {
               gradient[0] = reorientation.m[0][0]*foldingCorrection[0]
                     + reorientation.m[0][1]*foldingCorrection[1];
               gradient[1] = reorientation.m[1][0]*foldingCorrection[0]
                     + reorientation.m[1][1]*foldingCorrection[1];
               norm = (DataType)(5.0 * sqrt(gradient[0]*gradient[0]
                     + gradient[1]*gradient[1]));

               if(norm>(DataType)0)
               {
                  id = y*splineControlPoint->nx+x;
                  controlPointPtrX[id] += (DataType)(gradient[0]/norm);
                  controlPointPtrY[id] += (DataType)(gradient[1]/norm);
               }
            }
         }
      }
   }
   else
   {
      // The function aims to correct the folding at every voxel positions

      if(splineControlPoint->num_ext>0)
         useHeaderInformation=true;

      int xPre, yPre;
      DataType basis;

      if(useHeaderInformation)
      {
         // The grid and reference image are not aligned
         NR_FATAL_ERROR("Not implemented yet");
      }
      else
      {
         // The grid and reference image are expected to be aligned
         DataType gridVoxelSpacing[2];
         gridVoxelSpacing[0] = splineControlPoint->dx / referenceImage->dx;
         gridVoxelSpacing[1] = splineControlPoint->dy / referenceImage->dy;

#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(splineControlPoint, gridVoxelSpacing, referenceImage, jacobianDeterminant, \
   jacobianMatrices, controlPointPtrX, controlPointPtrY, reorientation) \
   private(x, xPre, yPre, pixelX, pixelY, foldingCorrection, \
   basis, xBasis, yBasis, xFirst, yFirst, jacIndex, detJac, \
   jacobianMatrix, basisValues, norm, correctFolding, id, gradient)
#endif
         for(y=0; y<splineControlPoint->ny; y++)
         {
            for(x=0; x<splineControlPoint->nx; x++)
            {

               foldingCorrection[0]=foldingCorrection[1]=0;
               correctFolding=false;

               // Loop over all the control points in the surrounding area

               for(pixelY=Ceil((y-3)*gridVoxelSpacing[1]); pixelY<Floor((y+1)*gridVoxelSpacing[1]); pixelY++)
               {
                  if(pixelY>-1 && pixelY<referenceImage->ny)
                  {

                     for(pixelX=Ceil((x-3)*gridVoxelSpacing[0]); pixelX<Floor((x+1)*gridVoxelSpacing[0]); pixelX++)
                     {
                        if(pixelX>-1 && pixelX<referenceImage->nx)
                        {

                           jacIndex = pixelY*referenceImage->nx+pixelX;
                           detJac = jacobianDeterminant[jacIndex];

                           if(detJac<=0)
                           {

                              jacobianMatrix = jacobianMatrices[jacIndex];

                              yPre=(int)((DataType)pixelY/gridVoxelSpacing[1]);
                              basis=(DataType)pixelY/gridVoxelSpacing[1]-(DataType)yPre;
                              get_BSplineBasisValue<DataType>(basis, y-yPre,yBasis,yFirst);

                              xPre=(int)((DataType)pixelX/gridVoxelSpacing[0]);
                              basis=(DataType)pixelX/gridVoxelSpacing[0]-(DataType)xPre;
                              get_BSplineBasisValue<DataType>(basis, x-xPre,xBasis,xFirst);

                              basisValues[0]= xFirst * yBasis ;
                              basisValues[1]= xBasis * yFirst ;

                              correctFolding=true;
                              addJacobianGradientValues<DataType>(jacobianMatrix,
                                                               1.0,
                                                               basisValues[0],
                                    basisValues[1],
                                    foldingCorrection);
                           } // detJac<0
                        } // if x
                     }// x
                  }// if y
               }// y
               // (Marc) I removed the normalisation by the voxel number as each gradient has to be normalised in the same way (NMI, BE, JAC)
               if(correctFolding)
               {
                  gradient[0] = reorientation.m[0][0]*foldingCorrection[0]
                        + reorientation.m[0][1]*foldingCorrection[1];
                  gradient[1] = reorientation.m[1][0]*foldingCorrection[0]
                        + reorientation.m[1][1]*foldingCorrection[1];
                  norm = (DataType)(5.0 * sqrt(gradient[0]*gradient[0] +
                        gradient[1]*gradient[1]));

                  if(norm>0)
                  {
                     id = y*splineControlPoint->nx+x;
                     controlPointPtrX[id] += (DataType)(gradient[0]/norm);
                     controlPointPtrY[id] += (DataType)(gradient[1]/norm);
                  }
               }
            }
         }
      }
   }
   free(jacobianDeterminant);
   free(jacobianMatrices);
   return std::numeric_limits<double>::quiet_NaN();
}
/* *************************************************************** */
template<class DataType>
double reg_spline_correctFolding3D(nifti_image *splineControlPoint,
                                   nifti_image *referenceImage,
                                   bool approximation,
                                   bool useHeaderInformation)
{
#ifdef WIN32
   long i;
   long jacobianNumber;
   if(approximation)
      jacobianNumber = (long)(splineControlPoint->nx-2)*(splineControlPoint->ny-2)*(splineControlPoint->nz-2);
   else jacobianNumber = (long)NiftiImage::calcVoxelNumber(referenceImage, 3);
#else
   size_t i;
   size_t jacobianNumber;
   if(approximation)
      jacobianNumber = (size_t)(splineControlPoint->nx-2)*(splineControlPoint->ny-2)*(splineControlPoint->nz-2);
   else jacobianNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);
#endif
   mat33 *jacobianMatrices=(mat33 *)malloc(jacobianNumber*sizeof(mat33));
   DataType *jacobianDeterminant=(DataType *)malloc(jacobianNumber*sizeof(DataType));

   reg_cubic_spline_jacobian3D(splineControlPoint,
                         referenceImage,
                         jacobianMatrices,
                         jacobianDeterminant,
                         approximation,
                         useHeaderInformation);

   /* The current Penalty term value is computed */
   double penaltyTerm =0., logDet;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(jacobianNumber, jacobianDeterminant) \
   private(logDet) \
   reduction(+:penaltyTerm)
#endif
   for(i=0; i< jacobianNumber; i++)
   {
      logDet = log(jacobianDeterminant[i]);
#ifdef USE_SQUARE_LOG_JAC
      penaltyTerm += logDet*logDet;
#else
      penaltyTerm +=  fabs(log(logDet));
#endif
   }
   if(penaltyTerm==penaltyTerm)
   {
      free(jacobianDeterminant);
      free(jacobianMatrices);
      return penaltyTerm/(double)(jacobianNumber);
   }

   mat33 jacobianMatrix, reorientation;
   if(splineControlPoint->sform_code>0)
      reorientation = reg_mat44_to_mat33(&splineControlPoint->sto_xyz);
   else reorientation = reg_mat44_to_mat33(&splineControlPoint->qto_xyz);

   const size_t nodeNumber = NiftiImage::calcVoxelNumber(splineControlPoint, 3);
   DataType *controlPointPtrX = static_cast<DataType *>(splineControlPoint->data);
   DataType *controlPointPtrY = &controlPointPtrX[nodeNumber];
   DataType *controlPointPtrZ = &controlPointPtrY[nodeNumber];

   DataType basisValues[3], foldingCorrection[3], gradient[3], norm;
   DataType xBasis=0, yBasis=0, zBasis=0, xFirst=0, yFirst=0, zFirst=0;
   int x, y, z, id, pixelX, pixelY, pixelZ, jacIndex;
   bool correctFolding;
   double detJac;

   if(approximation)
   {
      // The function discretise the Jacobian only at the control point positions

#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(splineControlPoint, jacobianDeterminant, jacobianMatrices, \
   controlPointPtrX, controlPointPtrY, controlPointPtrZ, reorientation) \
   private(x, y, pixelX, pixelY, pixelZ, foldingCorrection, \
   xBasis, yBasis, zBasis, xFirst, yFirst, zFirst, jacIndex, detJac, \
   jacobianMatrix, basisValues, norm, correctFolding, id, gradient)
#endif
      for(z=0; z<splineControlPoint->nz; z++)
      {
         for(y=0; y<splineControlPoint->ny; y++)
         {
            for(x=0; x<splineControlPoint->nx; x++)
            {

               foldingCorrection[0]=foldingCorrection[1]=foldingCorrection[2]=0;
               correctFolding=false;

               // Loop over all the control points in the surrounding area
               for(pixelZ=(int)((z-1)); pixelZ<(int)((z+2)); pixelZ++)
               {
                  if(pixelZ>0 && pixelZ<splineControlPoint->nz-1)
                  {

                     for(pixelY=(int)((y-1)); pixelY<(int)((y+2)); pixelY++)
                     {
                        if(pixelY>0 && pixelY<splineControlPoint->ny-1)
                        {

                           for(pixelX=(int)((x-1)); pixelX<(int)((x+2)); pixelX++)
                           {
                              if(pixelX>0 && pixelX<splineControlPoint->nx-1)
                              {

                                 jacIndex = ((pixelZ-1)*(splineControlPoint->ny-2)+pixelY-1)*
                                       (splineControlPoint->nx-2)+pixelX-1;
                                 detJac = jacobianDeterminant[jacIndex];

                                 if(detJac<=0)
                                 {
                                    get_BSplineBasisValue<DataType>(0, z-pixelZ+1, zBasis, zFirst);
                                    get_BSplineBasisValue<DataType>(0, y-pixelY+1, yBasis, yFirst);
                                    get_BSplineBasisValue<DataType>(0, x-pixelX+1, xBasis, xFirst);

                                    basisValues[0] = xFirst * yBasis * zBasis ;
                                    basisValues[1] = xBasis * yFirst * zBasis ;
                                    basisValues[2] = xBasis * yBasis * zFirst ;

                                    jacobianMatrix = jacobianMatrices[jacIndex];

                                    correctFolding=true;
                                    addJacobianGradientValues<DataType>(jacobianMatrix,
                                                                     1.0,
                                                                     basisValues[0],
                                          basisValues[1],
                                          basisValues[2],
                                          foldingCorrection);
                                 } // detJac<0
                              } // if x
                           }// x
                        }// if y
                     }// y
                  }// if z
               } // z
               if(correctFolding)
               {
                  gradient[0] = reorientation.m[0][0]*foldingCorrection[0]
                        + reorientation.m[0][1]*foldingCorrection[1]
                        + reorientation.m[0][2]*foldingCorrection[2];
                  gradient[1] = reorientation.m[1][0]*foldingCorrection[0]
                        + reorientation.m[1][1]*foldingCorrection[1]
                        + reorientation.m[1][2]*foldingCorrection[2];
                  gradient[2] = reorientation.m[2][0]*foldingCorrection[0]
                        + reorientation.m[2][1]*foldingCorrection[1]
                        + reorientation.m[2][2]*foldingCorrection[2];
                  norm = (DataType)(5.0 * sqrt(gradient[0]*gradient[0]
                        + gradient[1]*gradient[1]
                        + gradient[2]*gradient[2]));

                  if(norm>(DataType)0)
                  {
                     id = (z*splineControlPoint->ny+y)*splineControlPoint->nx+x;
                     controlPointPtrX[id] += (DataType)(gradient[0]/norm);
                     controlPointPtrY[id] += (DataType)(gradient[1]/norm);
                     controlPointPtrZ[id] += (DataType)(gradient[2]/norm);
                  }
               }
            }
         }
      }
   }
   else
   {
      // The function aims to correct the folding at every voxel positions

      if(splineControlPoint->num_ext>0)
         useHeaderInformation=true;

      int xPre, yPre, zPre;
      DataType basis;

      if(useHeaderInformation)
      {
         // The grid and reference image are not aligned
         NR_FATAL_ERROR("Not implemented yet");
      }
      else
      {
         // The grid and reference image are expected to be aligned
         DataType gridVoxelSpacing[3];
         gridVoxelSpacing[0] = splineControlPoint->dx / referenceImage->dx;
         gridVoxelSpacing[1] = splineControlPoint->dy / referenceImage->dy;
         gridVoxelSpacing[2] = splineControlPoint->dz / referenceImage->dz;

#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(splineControlPoint, gridVoxelSpacing, referenceImage, jacobianDeterminant, \
   jacobianMatrices, controlPointPtrX, controlPointPtrY, controlPointPtrZ, reorientation) \
   private(x, y, xPre, yPre, zPre, pixelX, pixelY, pixelZ, foldingCorrection, \
   basis, xBasis, yBasis, zBasis, xFirst, yFirst, zFirst, jacIndex, detJac, \
   jacobianMatrix, basisValues, norm, correctFolding, id, gradient)
#endif
         for(z=0; z<splineControlPoint->nz; z++)
         {
            for(y=0; y<splineControlPoint->ny; y++)
            {
               for(x=0; x<splineControlPoint->nx; x++)
               {

                  foldingCorrection[0]=foldingCorrection[1]=foldingCorrection[2]=0;
                  correctFolding=false;

                  // Loop over all the control points in the surrounding area
                  for(pixelZ=Ceil((z-3)*gridVoxelSpacing[2]); pixelZ<Floor((z+1)*gridVoxelSpacing[2]); pixelZ++)
                  {
                     if(pixelZ>-1 && pixelZ<referenceImage->nz)
                     {

                        for(pixelY=Ceil((y-3)*gridVoxelSpacing[1]); pixelY<Floor((y+1)*gridVoxelSpacing[1]); pixelY++)
                        {
                           if(pixelY>-1 && pixelY<referenceImage->ny)
                           {

                              for(pixelX=Ceil((x-3)*gridVoxelSpacing[0]); pixelX<Floor((x+1)*gridVoxelSpacing[0]); pixelX++)
                              {
                                 if(pixelX>-1 && pixelX<referenceImage->nx)
                                 {

                                    jacIndex = (pixelZ*referenceImage->ny+pixelY)*referenceImage->nx+pixelX;
                                    detJac = jacobianDeterminant[jacIndex];

                                    if(detJac<=0)
                                    {

                                       jacobianMatrix = jacobianMatrices[jacIndex];

                                       zPre=(int)((DataType)pixelZ/gridVoxelSpacing[2]);
                                       basis=(DataType)pixelZ/gridVoxelSpacing[2]-(DataType)zPre;
                                       get_BSplineBasisValue<DataType>(basis, z-zPre,zBasis,zFirst);

                                       yPre=(int)((DataType)pixelY/gridVoxelSpacing[1]);
                                       basis=(DataType)pixelY/gridVoxelSpacing[1]-(DataType)yPre;
                                       get_BSplineBasisValue<DataType>(basis, y-yPre,yBasis,yFirst);

                                       xPre=(int)((DataType)pixelX/gridVoxelSpacing[0]);
                                       basis=(DataType)pixelX/gridVoxelSpacing[0]-(DataType)xPre;
                                       get_BSplineBasisValue<DataType>(basis, x-xPre,xBasis,xFirst);

                                       basisValues[0]= xFirst * yBasis * zBasis ;
                                       basisValues[1]= xBasis * yFirst * zBasis ;
                                       basisValues[2]= xBasis * yBasis * zFirst ;

                                       correctFolding=true;
                                       addJacobianGradientValues<DataType>(jacobianMatrix,
                                                                        1.0,
                                                                        basisValues[0],
                                             basisValues[1],
                                             basisValues[2],
                                             foldingCorrection);
                                    } // detJac<0
                                 } // if x
                              }// x
                           }// if y
                        }// y
                     }// if z
                  } // z
                  // (Marc) I removed the normalisation by the voxel number as each gradient has to be normalised in the same way (NMI, BE, JAC)
                  if(correctFolding)
                  {
                     gradient[0] = reorientation.m[0][0]*foldingCorrection[0]
                           + reorientation.m[0][1]*foldingCorrection[1]
                           + reorientation.m[0][2]*foldingCorrection[2];
                     gradient[1] = reorientation.m[1][0]*foldingCorrection[0]
                           + reorientation.m[1][1]*foldingCorrection[1]
                           + reorientation.m[1][2]*foldingCorrection[2];
                     gradient[2] = reorientation.m[2][0]*foldingCorrection[0]
                           + reorientation.m[2][1]*foldingCorrection[1]
                           + reorientation.m[2][2]*foldingCorrection[2];
                     norm = (DataType)(5.0 * sqrt(gradient[0]*gradient[0] +
                           gradient[1]*gradient[1] +
                           gradient[2]*gradient[2]));

                     if(norm>0)
                     {
                        id = (z*splineControlPoint->ny+y)*splineControlPoint->nx+x;
                        controlPointPtrX[id] += (DataType)(gradient[0]/norm);
                        controlPointPtrY[id] += (DataType)(gradient[1]/norm);
                        controlPointPtrZ[id] += (DataType)(gradient[2]/norm);
                     }
                  }
               }
            }
         }
      }
   }
   free(jacobianDeterminant);
   free(jacobianMatrices);
   return std::numeric_limits<double>::quiet_NaN();
}
/* *************************************************************** */
double reg_spline_correctFolding(nifti_image *splineControlPoint,
                                 nifti_image *referenceImage,
                                 bool approx)
{
   if(splineControlPoint->nz==1)
   {
      switch(splineControlPoint->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         return reg_spline_correctFolding2D<float>(splineControlPoint, referenceImage, approx, false);
         break;
      case NIFTI_TYPE_FLOAT64:
         return reg_spline_correctFolding2D<double>(splineControlPoint, referenceImage, approx, false);
         break;
      default:
         NR_FATAL_ERROR("Only implemented for single or double precision images");
         return 0;
      }
   }
   else
   {
      switch(splineControlPoint->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         return reg_spline_correctFolding3D<float>(splineControlPoint, referenceImage, approx, false);
         break;
      case NIFTI_TYPE_FLOAT64:
         return reg_spline_correctFolding3D<double>(splineControlPoint, referenceImage, approx, false);
         break;
      default:
         NR_FATAL_ERROR("Only implemented for single or double precision images");
         return 0;
      }
   }
}
/* *************************************************************** */
/* *************************************************************** */
void reg_spline_GetJacobianMap(nifti_image *splineControlPoint,
                               nifti_image *jacobianImage)
{
   if(splineControlPoint->intent_p1==LIN_SPLINE_GRID){
      if(splineControlPoint->nz==1)
      {
         NR_FATAL_ERROR("No 2D implementation for the linear spline yet");
      }
      else
      {
         switch(jacobianImage->datatype)
         {
         case NIFTI_TYPE_FLOAT32:
            reg_linear_spline_jacobian3D<float>(splineControlPoint,
                                               jacobianImage,
                                               nullptr,
                                               static_cast<float *>(jacobianImage->data),
                                               false,
                                               true);
            break;
         case NIFTI_TYPE_FLOAT64:
            reg_linear_spline_jacobian3D<double>(splineControlPoint,
                                                jacobianImage,
                                                nullptr,
                                                static_cast<double *>(jacobianImage->data),
                                                false,
                                                true);
            break;
         default:
            NR_FATAL_ERROR("Only implemented for single or double precision images");
         }
      }

   }
   else{
      if(splineControlPoint->nz==1)
      {
         switch(jacobianImage->datatype)
         {
         case NIFTI_TYPE_FLOAT32:
            reg_cubic_spline_jacobian2D<float>(splineControlPoint,
                                               jacobianImage,
                                               nullptr,
                                               static_cast<float *>(jacobianImage->data),
                                               false,
                                               true);
            break;
         case NIFTI_TYPE_FLOAT64:
            reg_cubic_spline_jacobian2D<double>(splineControlPoint,
                                                jacobianImage,
                                                nullptr,
                                                static_cast<double *>(jacobianImage->data),
                                                false,
                                                true);
            break;
         default:
            NR_FATAL_ERROR("Only implemented for single or double precision images");
         }
      }
      else
      {
         switch(jacobianImage->datatype)
         {
         case NIFTI_TYPE_FLOAT32:
            reg_cubic_spline_jacobian3D<float>(splineControlPoint,
                                               jacobianImage,
                                               nullptr,
                                               static_cast<float *>(jacobianImage->data),
                                               false,
                                               true);
            break;
         case NIFTI_TYPE_FLOAT64:
            reg_cubic_spline_jacobian3D<double>(splineControlPoint,
                                                jacobianImage,
                                                nullptr,
                                                static_cast<double *>(jacobianImage->data),
                                                false,
                                                true);
            break;
         default:
            NR_FATAL_ERROR("Only implemented for single or double precision images");
         }
      }
   }
}
/* *************************************************************** */
/* *************************************************************** */
void reg_spline_GetJacobianMatrix(nifti_image *referenceImage,
                                  nifti_image *splineControlPoint,
                                  mat33 *jacobianMatrices)
{
   if(splineControlPoint->nz==1)
   {
      switch(splineControlPoint->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         reg_cubic_spline_jacobian2D<float>(splineControlPoint,
                                      referenceImage,
                                      jacobianMatrices,
                                      nullptr,
                                      false,
                                      true);
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_cubic_spline_jacobian2D<double>(splineControlPoint,
                                       referenceImage,
                                       jacobianMatrices,
                                       nullptr,
                                       false,
                                       true);
         break;
      default:
         NR_FATAL_ERROR("Only implemented for single or double precision images");
      }
   }
   else
   {
      switch(splineControlPoint->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         reg_cubic_spline_jacobian3D<float>(splineControlPoint,
                                      referenceImage,
                                      jacobianMatrices,
                                      nullptr,
                                      false,
                                      true);
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_cubic_spline_jacobian3D<double>(splineControlPoint,
                                       referenceImage,
                                       jacobianMatrices,
                                       nullptr,
                                       false,
                                       true);
         break;
      default:
         NR_FATAL_ERROR("Only implemented for single or double precision images");
      }
   }
}
/* *************************************************************** */
/* *************************************************************** */
template <class DataType>
void reg_defField_getJacobianMap2D(nifti_image *deformationField,
                                   nifti_image *jacobianDeterminant,
                                   mat33 *jacobianMatrices)
{
   const size_t voxelNumber = NiftiImage::calcVoxelNumber(deformationField, 2);

   DataType *jacDetPtr=nullptr;
   if(jacobianDeterminant!=nullptr)
      jacDetPtr=static_cast<DataType *>(jacobianDeterminant->data);

   float spacing[3];
   mat33 reorientation, jacobianMatrix;

   if(deformationField->sform_code>0)
   {
      reg_getRealImageSpacing(deformationField,spacing);
      reorientation=nifti_mat33_inverse(nifti_mat33_polar(reg_mat44_to_mat33(&deformationField->sto_xyz)));
   }
   else
   {
      spacing[0]=deformationField->dx;
      spacing[1]=deformationField->dy;
      reorientation=nifti_mat33_inverse(nifti_mat33_polar(reg_mat44_to_mat33(&deformationField->qto_xyz)));
   }

   DataType *deformationPtrX = static_cast<DataType *>(deformationField->data);
   DataType *deformationPtrY = &deformationPtrX[voxelNumber];

   DataType basis[2]= {1.0,0};
   DataType first[2]= {-1.0,1.0};
   DataType firstX, firstY, defX, defY;

   int currentIndex, x, y, a, b, index;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(deformationField, jacobianDeterminant, jacobianMatrices, reorientation, \
   basis, first, jacDetPtr, deformationPtrX, deformationPtrY, spacing) \
   private(currentIndex, x, a, b, index, jacobianMatrix, defX, defY, firstX, firstY)
#endif
   for(y=0; y<deformationField->ny-1; ++y)
   {
      currentIndex=y*deformationField->nx;
      for(x=0; x<deformationField->nx-1; ++x)
      {
         memset(&jacobianMatrix,0,sizeof(mat33));

         for(b=0; b<2; ++b)
         {
            index=(y+b)*deformationField->nx+x;
            for(a=0; a<2; ++a)
            {

               // Compute the basis function values
               firstX=first[a]*basis[b];
               firstY=basis[a]*first[b];

               // Get the deformation field values
               defX = deformationPtrX[index];
               defY = deformationPtrY[index];

               // Symmetric difference to compute the derivatives
               jacobianMatrix.m[0][0] += firstX*defX;
               jacobianMatrix.m[0][1] += firstY*defX;
               jacobianMatrix.m[1][0] += firstX*defY;
               jacobianMatrix.m[1][1] += firstY*defY;

               ++index;
            }//a
         }//b

         // reorient and scale the Jacobian matrix
         jacobianMatrix=nifti_mat33_mul(reorientation,jacobianMatrix);
         jacobianMatrix.m[0][0] /= spacing[0];
         jacobianMatrix.m[0][1] /= spacing[1];
         jacobianMatrix.m[1][0] /= spacing[0];
         jacobianMatrix.m[1][1] /= spacing[1];

         // Update the output arrays if required
         if(jacobianDeterminant!=nullptr)
            jacDetPtr[currentIndex] = nifti_mat33_determ(jacobianMatrix);
         if(jacobianMatrices!=nullptr)
            jacobianMatrices[currentIndex]=jacobianMatrix;
         // Increment the pointer
         currentIndex++;
      }// x jacImage
   }//y jacImage
   // Sliding is assumed. The Jacobian at the boundary are then replicated

   for(y=0; y<deformationField->ny; ++y)
   {
      currentIndex=y*deformationField->nx;
      for(x=0; x<deformationField->nx; ++x)
      {
         index=currentIndex;
         if(x==deformationField->nx-1) index -= 1;
         if(y==deformationField->ny-1) index -= deformationField->nx;
         if(currentIndex!=index)
         {
            if(jacobianDeterminant!=nullptr)
               jacDetPtr[currentIndex] = jacDetPtr[index];
            if(jacobianMatrices!=nullptr)
               jacobianMatrices[currentIndex] = jacobianMatrices[index];
         }
         ++currentIndex;
      } // x
   } // y
}
/* *************************************************************** */
template <class DataType>
void reg_defField_getJacobianMap3D(nifti_image *deformationField,
                                   nifti_image *jacobianDeterminant,
                                   mat33 *jacobianMatrices)
{
   const size_t voxelNumber = NiftiImage::calcVoxelNumber(deformationField, 3);

   DataType *jacDetPtr=nullptr;
   if(jacobianDeterminant!=nullptr)
      jacDetPtr=static_cast<DataType *>(jacobianDeterminant->data);

   float spacing[3];
   mat33 reorientation, jacobianMatrix;

   if(deformationField->sform_code>0)
   {
      reg_getRealImageSpacing(deformationField,spacing);
      reorientation=nifti_mat33_inverse(nifti_mat33_polar(reg_mat44_to_mat33(&deformationField->sto_xyz)));
   }
   else
   {
      spacing[0]=deformationField->dx;
      spacing[1]=deformationField->dy;
      spacing[2]=deformationField->dz;
      reorientation=nifti_mat33_inverse(nifti_mat33_polar(reg_mat44_to_mat33(&deformationField->qto_xyz)));
   }

   DataType *deformationPtrX = static_cast<DataType *>(deformationField->data);
   DataType *deformationPtrY = &deformationPtrX[voxelNumber];
   DataType *deformationPtrZ = &deformationPtrY[voxelNumber];

   DataType basis[2]= {1.0,0};
   DataType first[2]= {-1.0,1.0};
   DataType firstX, firstY, firstZ, defX, defY, defZ;

   int currentIndex, x, y, z, a, b, c, currentZ, index;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(deformationField, jacobianDeterminant, jacobianMatrices, reorientation, \
   basis, first, jacDetPtr, deformationPtrX, deformationPtrY, deformationPtrZ, spacing) \
   private(currentIndex, x, y, a, b, c, currentZ, index, \
   jacobianMatrix, defX, defY, defZ, firstX, firstY, firstZ)
#endif
   for(z=0; z<deformationField->nz-1; ++z)
   {
      for(y=0; y<deformationField->ny-1; ++y)
      {
         currentIndex=(z*deformationField->ny+y)*deformationField->nx;
         for(x=0; x<deformationField->nx-1; ++x)
         {

            memset(&jacobianMatrix,0,sizeof(mat33));

            for(c=0; c<2; ++c)
            {
               currentZ=z+c;
               for(b=0; b<2; ++b)
               {
                  index=(currentZ*deformationField->ny+y+b)*deformationField->nx+x;
                  for(a=0; a<2; ++a)
                  {

                     // Compute the basis function values
                     firstX=first[a]*basis[b]*basis[c];
                     firstY=basis[a]*first[b]*basis[c];
                     firstZ=basis[a]*basis[b]*first[c];

                     // Get the deformation field values
                     defX = deformationPtrX[index];
                     defY = deformationPtrY[index];
                     defZ = deformationPtrZ[index];

                     // Symmetric difference to compute the derivatives
                     jacobianMatrix.m[0][0] += firstX*defX;
                     jacobianMatrix.m[0][1] += firstY*defX;
                     jacobianMatrix.m[0][2] += firstZ*defX;
                     jacobianMatrix.m[1][0] += firstX*defY;
                     jacobianMatrix.m[1][1] += firstY*defY;
                     jacobianMatrix.m[1][2] += firstZ*defY;
                     jacobianMatrix.m[2][0] += firstX*defZ;
                     jacobianMatrix.m[2][1] += firstY*defZ;
                     jacobianMatrix.m[2][2] += firstZ*defZ;

                     ++index;
                  }//a
               }//b
            }//c

            // reorient and scale the Jacobian matrix
            jacobianMatrix=nifti_mat33_mul(reorientation,jacobianMatrix);
            jacobianMatrix.m[0][0] /= spacing[0];
            jacobianMatrix.m[0][1] /= spacing[1];
            jacobianMatrix.m[0][2] /= spacing[2];
            jacobianMatrix.m[1][0] /= spacing[0];
            jacobianMatrix.m[1][1] /= spacing[1];
            jacobianMatrix.m[1][2] /= spacing[2];
            jacobianMatrix.m[2][0] /= spacing[0];
            jacobianMatrix.m[2][1] /= spacing[1];
            jacobianMatrix.m[2][2] /= spacing[2];

            // Update the output arrays if required
            if(jacobianDeterminant!=nullptr)
               jacDetPtr[currentIndex] = nifti_mat33_determ(jacobianMatrix);
            if(jacobianMatrices!=nullptr)
               jacobianMatrices[currentIndex]=jacobianMatrix;
            // Increment the pointer
            currentIndex++;
         }// x jacImage
      }//y jacImage
   }//z jacImage
   // Sliding is assumed. The Jacobian at the boundary are then replicated
   for(z=0; z<deformationField->nz; ++z)
   {
      currentIndex=z*deformationField->nx*deformationField->ny;
      for(y=0; y<deformationField->ny; ++y)
      {
         for(x=0; x<deformationField->nx; ++x)
         {
            index=currentIndex;
            if(x==deformationField->nx-1) index -= 1;
            if(y==deformationField->ny-1) index -= deformationField->nx;
            if(z==deformationField->nz-1) index -= deformationField->nx*deformationField->ny;
            if(currentIndex!=index)
            {
               if(jacobianDeterminant!=nullptr)
                  jacDetPtr[currentIndex] = jacDetPtr[index];
               if(jacobianMatrices!=nullptr)
                  jacobianMatrices[currentIndex] = jacobianMatrices[index];
            }
            ++currentIndex;
         } // x
      } // y
   } // z
}
/* *************************************************************** */
void reg_defField_getJacobianMap(nifti_image *deformationField,
                                 nifti_image *jacobianImage)
{
   if(deformationField->datatype!=jacobianImage->datatype)
      NR_FATAL_ERROR("Both input images are expected to have the same datatype");

   switch(deformationField->datatype)
   {
   case NIFTI_TYPE_FLOAT32:
      if(deformationField->nz>1)
         reg_defField_getJacobianMap3D<float>(deformationField,jacobianImage,nullptr);
      else reg_defField_getJacobianMap2D<float>(deformationField,jacobianImage,nullptr);
      break;
   case NIFTI_TYPE_FLOAT64:
      if(deformationField->nz>1)
         reg_defField_getJacobianMap3D<double>(deformationField,jacobianImage,nullptr);
      else reg_defField_getJacobianMap2D<double>(deformationField,jacobianImage,nullptr);
      break;
   default:
      NR_FATAL_ERROR("Only implemented for single or double precision images");
   }
}
/* *************************************************************** */
/* *************************************************************** */
void reg_defField_getJacobianMatrix(nifti_image *deformationField,
                                    mat33 *jacobianMatrices)
{
   switch(deformationField->datatype)
   {
   case NIFTI_TYPE_FLOAT32:
      if(deformationField->nz>1)
         reg_defField_getJacobianMap3D<float>(deformationField,nullptr,jacobianMatrices);
      else reg_defField_getJacobianMap2D<float>(deformationField,nullptr,jacobianMatrices);
      break;
   case NIFTI_TYPE_FLOAT64:
      if(deformationField->nz>1)
         reg_defField_getJacobianMap3D<double>(deformationField,nullptr,jacobianMatrices);
      else reg_defField_getJacobianMap2D<double>(deformationField,nullptr,jacobianMatrices);
      break;
   default:
      NR_FATAL_ERROR("Only implemented for single or double precision images");
   }
}
/* *************************************************************** */
template <class DataType>
void reg_defField_GetJacobianMatFromFlowField_core(mat33* jacobianMatrices,
                                                   nifti_image* flowFieldImage)
{
   // A second field is allocated to store the deformation
   nifti_image *defFieldImage = nifti_dup(*flowFieldImage, false);

   // Remove the affine component from the flow field
   if(flowFieldImage->num_ext>0)
   {
      if(flowFieldImage->ext_list[0].edata!=nullptr)
      {
         // Create a field that contains the affine component only
         reg_affine_getDeformationField(reinterpret_cast<mat44 *>(flowFieldImage->ext_list[0].edata),
               defFieldImage,
               false);
         reg_tools_subtractImageFromImage(flowFieldImage,defFieldImage,flowFieldImage);
      }
   }
   else reg_getDisplacementFromDeformation(flowFieldImage);

   // The displacement field is scaled
   float scalingValue = pow(2.0f,fabs(flowFieldImage->intent_p2));
   if(flowFieldImage->intent_p2<0)
      // backward deformation field is scaled down
      reg_tools_divideValueToImage(flowFieldImage,
                                   flowFieldImage,
                                   -scalingValue); // (/-scalingValue)
   else
      // forward deformation field is scaled down
      reg_tools_divideValueToImage(flowFieldImage,
                                   flowFieldImage,
                                   scalingValue); // (/scalingValue)

   // Conversion from displacement to deformation
   reg_getDeformationFromDisplacement(flowFieldImage);

   // The computed scaled flow field is copied over
   memcpy(defFieldImage->data, flowFieldImage->data,
          defFieldImage->nvox*defFieldImage->nbyper);

   // The Jacobian matrices are initialised with identity or the initial affine
   mat33 affineMatrix;
   reg_mat33_eye(&affineMatrix);
   if(flowFieldImage->num_ext>0)
   {
      if(flowFieldImage->ext_list[0].edata!=nullptr)
         affineMatrix = reg_mat44_to_mat33(reinterpret_cast<mat44 *>(flowFieldImage->ext_list[0].edata));
      else NR_FATAL_ERROR("The affine matrix is expected to be stored in the flow field");
   }
   const size_t voxelNumber = NiftiImage::calcVoxelNumber(flowFieldImage, 3);
   for(size_t i=0; i<voxelNumber; ++i)
      jacobianMatrices[i]=affineMatrix;

   // Create a temporary Jacobian matrix array
   mat33 *tempJacMatrix=(mat33 *)malloc(voxelNumber*sizeof(mat33));

   // The deformation field is squared and the Jacobian computed
   for(int step=0; step<fabs(flowFieldImage->intent_p2); ++step)
   {
      // The matrices are computed at every voxel for the current field
      reg_defField_getJacobianMatrix(defFieldImage,
                                     tempJacMatrix);
      // The computed matrices are composed with the previous one
      for(size_t i=0; i<voxelNumber; ++i)
         jacobianMatrices[i]=nifti_mat33_mul(tempJacMatrix[i],jacobianMatrices[i]);
      // The deformation field is applied to itself
      reg_defField_compose(defFieldImage,
                           flowFieldImage,
                           nullptr);
      // The computed scaled deformation field is copied over
      memcpy(defFieldImage->data, flowFieldImage->data,
             defFieldImage->nvox*defFieldImage->nbyper);
      NR_DEBUG("Squaring (composition) step " << int(step + 1) << "/" << int(fabs(flowFieldImage->intent_p2)));
   }
   // Allocated arrays and images are free'ed
   nifti_image_free(defFieldImage);
   free(tempJacMatrix);
   // The second half of the affine is added if required
   if(flowFieldImage->num_ext>1)
   {
      if(flowFieldImage->ext_list[1].edata!=nullptr)
         affineMatrix = reg_mat44_to_mat33(reinterpret_cast<mat44 *>(flowFieldImage->ext_list[1].edata));
      else NR_FATAL_ERROR("The affine matrix is expected to be stored in the flow field");
      for(size_t i=0; i<voxelNumber; ++i)
         jacobianMatrices[i]=nifti_mat33_mul(affineMatrix,jacobianMatrices[i]);
   }
}
/* *************************************************************** */
/* *************************************************************** */
template<class DataType>
void reg_getDetArrayFromMatArray(nifti_image *jacobianDetImage,
                                 mat33 *jacobianMatrices
                                 )
{
   const size_t voxelNumber = NiftiImage::calcVoxelNumber(jacobianDetImage, 3);
   DataType *jacDetPtr=static_cast<DataType *>(jacobianDetImage->data);
   if(jacobianDetImage->nz>1){
       for(size_t voxel=0; voxel<voxelNumber; ++voxel)
          jacDetPtr[voxel]=nifti_mat33_determ(jacobianMatrices[voxel]);
   }
   else{
       for(size_t voxel=0; voxel<voxelNumber; ++voxel)
          jacDetPtr[voxel]=
                  jacobianMatrices[voxel].m[0][0] * jacobianMatrices[voxel].m[1][1] -
                  jacobianMatrices[voxel].m[1][0] * jacobianMatrices[voxel].m[0][1];
   }
}
/* *************************************************************** */
/* *************************************************************** */
int reg_defField_GetJacobianMatFromFlowField(mat33* jacobianMatrices,
                                             nifti_image* flowFieldImage
                                             )
{
   switch(flowFieldImage->datatype)
   {
   case NIFTI_TYPE_FLOAT32:
      reg_defField_GetJacobianMatFromFlowField_core<float>(jacobianMatrices,flowFieldImage);
      break;
   case NIFTI_TYPE_FLOAT64:
      reg_defField_GetJacobianMatFromFlowField_core<double>(jacobianMatrices,flowFieldImage);
      break;
   default:
      NR_FATAL_ERROR("Unsupported data type");
   }
   return 0;
}
/* *************************************************************** */
int reg_spline_GetJacobianMatFromVelocityGrid(mat33* jacobianMatrices,
                                              nifti_image* velocityGridImage,
                                              nifti_image* referenceImage)
{
   // A new image is created to store the flow field
   nifti_image *flowFieldImage=nifti_copy_nim_info(referenceImage);
   flowFieldImage->datatype=velocityGridImage->datatype;
   flowFieldImage->nbyper=velocityGridImage->nbyper;
   flowFieldImage->ndim=flowFieldImage->dim[0]=5;
   flowFieldImage->nt=flowFieldImage->dim[4]=1;
   flowFieldImage->nu=flowFieldImage->dim[5]=referenceImage->nz>1?3:2;
   flowFieldImage->nvox = NiftiImage::calcVoxelNumber(flowFieldImage, flowFieldImage->ndim);
   flowFieldImage->data=malloc(flowFieldImage->nvox*flowFieldImage->nbyper);

   // The velocity grid image is first converted into a flow field
   reg_spline_getFlowFieldFromVelocityGrid(velocityGridImage,
                                           flowFieldImage);

   reg_defField_GetJacobianMatFromFlowField(jacobianMatrices,
                                            flowFieldImage);

   nifti_image_free(flowFieldImage);
   return 0;
}
/* *************************************************************** */
int reg_defField_GetJacobianDetFromFlowField(nifti_image* jacobianDetImage,
                                             nifti_image* flowFieldImage)
{
   // create an array of mat33
   const size_t voxelNumber = NiftiImage::calcVoxelNumber(jacobianDetImage, 3);
   mat33 *jacobianMatrices=(mat33 *)malloc(voxelNumber*sizeof(mat33));

   // Compute the Jacobian matrice array
   reg_defField_GetJacobianMatFromFlowField(jacobianMatrices, flowFieldImage);

   // Compute and store all determinant
   switch(jacobianDetImage->datatype)
   {
   case NIFTI_TYPE_FLOAT32:
      reg_getDetArrayFromMatArray<float>(jacobianDetImage,jacobianMatrices);
      break;
   case NIFTI_TYPE_FLOAT64:
      reg_getDetArrayFromMatArray<double>(jacobianDetImage,jacobianMatrices);
      break;
   default:
      NR_FATAL_ERROR("Unsupported data type");
   }
   free(jacobianMatrices);
   return 0;
}

/* *************************************************************** */
int reg_spline_GetJacobianDetFromVelocityGrid(nifti_image* jacobianDetImage,
                                              nifti_image* velocityGridImage)
{
   // A new image is created to store the flow field
   nifti_image *flowFieldImage=nifti_copy_nim_info(jacobianDetImage);
   flowFieldImage->datatype=velocityGridImage->datatype;
   flowFieldImage->nbyper=velocityGridImage->nbyper;
   flowFieldImage->ndim=flowFieldImage->dim[0]=5;
   flowFieldImage->nt=flowFieldImage->dim[4]=1;
   flowFieldImage->nu=flowFieldImage->dim[5]=jacobianDetImage->nz>1?3:2;
   flowFieldImage->nvox = NiftiImage::calcVoxelNumber(flowFieldImage, flowFieldImage->ndim);
   flowFieldImage->data=malloc(flowFieldImage->nvox*flowFieldImage->nbyper);

   // The velocity grid image is first converted into a flow field
   reg_spline_getFlowFieldFromVelocityGrid(velocityGridImage,
                                           flowFieldImage);

   reg_defField_GetJacobianDetFromFlowField(jacobianDetImage,
                                            flowFieldImage);

   nifti_image_free(flowFieldImage);

   return 0;
}
/* *************************************************************** */
/* *************************************************************** */
