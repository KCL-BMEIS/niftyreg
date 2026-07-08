/*
 *  _reg_affineTransformation.cpp
 *
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_globalTrans.h"
#include "Maths.hpp"

/* *************************************************************** */
/* *************************************************************** */
template <class FieldTYPE>
void reg_affine_deformationField2D(mat44 *affineTransformation,
                                   nifti_image *deformationFieldImage,
                                   bool composition,
                                   int *mask)
{
   const size_t voxelNumber = NiftiImage::calcVoxelNumber(deformationFieldImage, 2);
   FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationFieldImage->data);
   FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[voxelNumber];

   mat44 *referenceMatrix;
   if(deformationFieldImage->sform_code>0)
      referenceMatrix=&deformationFieldImage->sto_xyz;
   else referenceMatrix=&deformationFieldImage->qto_xyz;

   mat44 transformationMatrix;
   if(composition)
      transformationMatrix = *affineTransformation;
   else transformationMatrix = *affineTransformation * *referenceMatrix;

   double voxel[3]={0,0,0}, position[3]={0,0,0};
   int x=0, y=0;
   size_t index=0;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(deformationFieldImage, transformationMatrix, affineTransformation, \
   deformationFieldPtrX, deformationFieldPtrY, mask, composition) \
   private(voxel, position, x, index)
#endif
   for(y=0; y<deformationFieldImage->ny; y++)
   {
      index=y*deformationFieldImage->nx;
      voxel[1]=(double)y;
      voxel[2] = 0;
      for(x=0; x<deformationFieldImage->nx; x++)
      {
         voxel[0]=(double)x;
         if(mask[index]>-1)
         {
            if(composition)
            {
               voxel[0] = (double) deformationFieldPtrX[index];
               voxel[1] = (double) deformationFieldPtrY[index];
               Mat44Mul(transformationMatrix, voxel, position);
            }
            else Mat44Mul(transformationMatrix, voxel, position);

            /* the deformation field (real coordinates) is stored */
            deformationFieldPtrX[index] = (FieldTYPE) position[0];
            deformationFieldPtrY[index] = (FieldTYPE) position[1];
         }
         index++;
      }
   }
}
/* *************************************************************** */
template <class FieldTYPE>
void reg_affine_deformationField3D(mat44 *affineTransformation,
                                   nifti_image *deformationFieldImage,
                                   bool composition,
                                   int *mask)
{
   const size_t voxelNumber=NiftiImage::calcVoxelNumber(deformationFieldImage, 3);
   FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationFieldImage->data);
   FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[voxelNumber];
   FieldTYPE *deformationFieldPtrZ = &deformationFieldPtrY[voxelNumber];

   mat44 *referenceMatrix;
   if(deformationFieldImage->sform_code>0)
      referenceMatrix=&deformationFieldImage->sto_xyz;
   else referenceMatrix=&deformationFieldImage->qto_xyz;

   mat44 transformationMatrix;
   if(composition)
      transformationMatrix = *affineTransformation;
   else transformationMatrix = *affineTransformation * *referenceMatrix;

   double voxel[3]={0,0,0}, position[3]={0,0,0};
   int x=0, y=0, z=0;
   size_t index=0;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(deformationFieldImage, transformationMatrix, affineTransformation, \
   deformationFieldPtrX, deformationFieldPtrY, deformationFieldPtrZ, mask, composition) \
   private(voxel, position, x, y, index)
#endif
   for(z=0; z<deformationFieldImage->nz; z++)
   {
      index=z*deformationFieldImage->nx*deformationFieldImage->ny;
      voxel[2]=(double) z;
      for(y=0; y<deformationFieldImage->ny; y++)
      {
         voxel[1]=(double) y;
         for(x=0; x<deformationFieldImage->nx; x++)
         {
            voxel[0]=(double) x;
            if(mask[index]>-1)
            {
               if(composition)
               {
                  voxel[0]= (double) deformationFieldPtrX[index];
                  voxel[1]= (double) deformationFieldPtrY[index];
                  voxel[2]= (double) deformationFieldPtrZ[index];
               }
               Mat44Mul(transformationMatrix, voxel, position);

               /* the deformation field (real coordinates) is stored */
               deformationFieldPtrX[index] = (FieldTYPE) position[0];
               deformationFieldPtrY[index] = (FieldTYPE) position[1];
               deformationFieldPtrZ[index] = (FieldTYPE) position[2];
            }
            index++;
         }
      }
   }
}
/* *************************************************************** */
void reg_affine_getDeformationField(mat44 *affineTransformation,
                                    nifti_image *deformationField,
                                    bool compose,
                                    int *mask)
{
   int *tempMask=mask;
   if(mask==nullptr)
   {
      tempMask = (int *)calloc(NiftiImage::calcVoxelNumber(deformationField, 3), sizeof(int));
   }
   if(deformationField->nz==1)
   {
      switch(deformationField->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         reg_affine_deformationField2D<float>(affineTransformation, deformationField, compose, tempMask);
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_affine_deformationField2D<double>(affineTransformation, deformationField, compose, tempMask);
         break;
      default:
         NR_FATAL_ERROR("The deformation field data type is not supported");
      }
   }
   else
   {
      switch(deformationField->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         reg_affine_deformationField3D<float>(affineTransformation, deformationField, compose, tempMask);
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_affine_deformationField3D<double>(affineTransformation, deformationField, compose, tempMask);
         break;
      default:
         NR_FATAL_ERROR("The deformation field data type is not supported");
      }
   }
   if(mask==nullptr)
      free(tempMask);
}
/* *************************************************************** */
void estimate_rigid_transformation2D(std::vector<_reg_sorted_point2D> &points, mat44 * transformation)
{

   unsigned num_points = points.size();
   float** points1 = Matrix2dAlloc<float>(num_points, 2);
   float** points2 = Matrix2dAlloc<float>(num_points, 2);
   for (unsigned i = 0; i < num_points; i++) {
      points1[i][0] = points[i].reference[0];
      points1[i][1] = points[i].reference[1];
      points2[i][0] = points[i].warped[0];
      points2[i][1] = points[i].warped[1];
   }
   EstimateRigidLeastSquares(points1, points2, num_points, 2, transformation);
   //FREE MEMORY
   Matrix2dDealloc(num_points, points1);
   Matrix2dDealloc(num_points, points2);
}
/* *************************************************************** */
void estimate_rigid_transformation3D(std::vector<_reg_sorted_point3D> &points, mat44 * transformation)
{
   unsigned num_points = points.size();
   float** points1 = Matrix2dAlloc<float>(num_points, 3);
   float** points2 = Matrix2dAlloc<float>(num_points, 3);
   for (unsigned i = 0; i < num_points; i++) {
      points1[i][0] = points[i].reference[0];
      points1[i][1] = points[i].reference[1];
      points1[i][2] = points[i].reference[2];
      points2[i][0] = points[i].warped[0];
      points2[i][1] = points[i].warped[1];
      points2[i][2] = points[i].warped[2];
   }
   EstimateRigidLeastSquares(points1, points2, num_points, 3, transformation);
   //FREE MEMORY
   Matrix2dDealloc(num_points, points1);
   Matrix2dDealloc(num_points, points2);
}
/* *************************************************************** */
void estimate_affine_transformation2D(std::vector<_reg_sorted_point2D> &points, mat44 * transformation)
{
   unsigned num_points = points.size();
   float** points1 = Matrix2dAlloc<float>(num_points, 2);
   float** points2 = Matrix2dAlloc<float>(num_points, 2);
   for (unsigned i = 0; i < num_points; i++) {
      points1[i][0] = points[i].reference[0];
      points1[i][1] = points[i].reference[1];
      points2[i][0] = points[i].warped[0];
      points2[i][1] = points[i].warped[1];
   }

   EstimateAffineLeastSquares(points1, points2, static_cast<size_t>(num_points), 2, transformation);
   //FREE MEMORY
   Matrix2dDealloc(num_points, points1);
   Matrix2dDealloc(num_points, points2);
}
/* *************************************************************** */
// estimate an affine transformation using least square
void estimate_affine_transformation3D(std::vector<_reg_sorted_point3D> &points, mat44 * transformation)
{
   unsigned num_points = points.size();
   float** points1 = Matrix2dAlloc<float>(num_points, 3);
   float** points2 = Matrix2dAlloc<float>(num_points, 3);
   for (unsigned i = 0; i < num_points; i++) {
      points1[i][0] = points[i].reference[0];
      points1[i][1] = points[i].reference[1];
      points1[i][2] = points[i].reference[2];
      points2[i][0] = points[i].warped[0];
      points2[i][1] = points[i].warped[1];
      points2[i][2] = points[i].warped[2];
   }

   EstimateAffineLeastSquares(points1, points2, static_cast<size_t>(num_points), 3, transformation);
   //FREE MEMORY
   Matrix2dDealloc(num_points, points1);
   Matrix2dDealloc(num_points, points2);
}
/* *************************************************************** */
///LTS 2D
void optimize_2D(float* referencePosition, float* warpedPosition,
                 unsigned activeBlockNumber, int percent_to_keep, int max_iter, double tol,
                 mat44 * final, bool affine) {

   // Set the current transformation to identity
   Mat44Eye(final);

   const unsigned num_points = activeBlockNumber;
   unsigned long num_equations = num_points * 2;
   // Residuals scratch, reused across iterations.
   std::vector<std::pair<double, unsigned>> residuals(num_points);
   std::vector<_reg_sorted_point2D> top_points;

   double distance = 0;
   double lastDistance = std::numeric_limits<double>::max();

   // The initial vector with all the input points
   for (unsigned j = 0; j < num_equations; j += 2)
   {
      top_points.push_back(_reg_sorted_point2D(&referencePosition[j], &warpedPosition[j], 0));
   }
   if (affine) {
      estimate_affine_transformation2D(top_points, final);
   }
   else {
      estimate_rigid_transformation2D(top_points, final);
   }

   const unsigned long num_to_keep = (unsigned long)(num_points * (percent_to_keep / 100.0f));
   float * newWarpedPosition = new float[num_points * 2];

   mat44 lastTransformation;
   memset(&lastTransformation, 0, sizeof(mat44));

   for (int count = 0; count < max_iter; ++count)
   {
      // Transform the points in the reference
      for (unsigned j = 0; j < num_points * 2; j += 2)
         Mat33Mul(*final, reinterpret_cast<float(&)[2]>(referencePosition[j]), reinterpret_cast<float(&)[2]>(newWarpedPosition[j]));
      for (unsigned p = 0, j = 0; p < num_points; ++p, j += 2)
         residuals[p] = { SquareDistance2d(&newWarpedPosition[j], &warpedPosition[j]), j };

      // Partition so the num_to_keep smallest residuals come first (the LTS trim), then gather them.
      std::nth_element(residuals.begin(), residuals.begin() + num_to_keep, residuals.end(),
                       [](const std::pair<double, unsigned>& a, const std::pair<double, unsigned>& b) { return a.first < b.first; });
      distance = 0;
      top_points.clear();
      for (unsigned k = 0; k < num_to_keep; ++k) {
         const unsigned j = residuals[k].second;
         top_points.push_back(_reg_sorted_point2D(&referencePosition[j], &warpedPosition[j], residuals[k].first));
         distance += residuals[k].first;
      }

      // If the change is not substantial, we return
      if ((distance > lastDistance) || (lastDistance - distance) < tol)
      {
         // restore the last transformation
         memcpy(final, &lastTransformation, sizeof(mat44));
         break;
      }
      lastDistance = distance;
      memcpy(&lastTransformation, final, sizeof(mat44));
      if (affine) {
         estimate_affine_transformation2D(top_points, final);
      }
      else {
         estimate_rigid_transformation2D(top_points, final);
      }
   }
   delete[] newWarpedPosition;

}
/* *************************************************************** */
///LTS 3D
void optimize_3D(float *referencePosition, float *warpedPosition,
                 unsigned activeBlockNumber, int percent_to_keep, int max_iter, double tol,
                 mat44 *final, bool affine) {

   // Set the current transformation to identity
   Mat44Eye(final);

   const unsigned num_points = activeBlockNumber;
   unsigned long num_equations = num_points * 3;
   // Residuals scratch, reused across iterations.
   std::vector<std::pair<double, unsigned>> residuals(num_points);
   std::vector<_reg_sorted_point3D> top_points;
   double distance = 0;
   double lastDistance = std::numeric_limits<double>::max();

   // The initial vector with all the input points
   for (unsigned j = 0; j < num_equations; j+=3) {
      top_points.push_back(_reg_sorted_point3D(&referencePosition[j],
                                               &warpedPosition[j],
                                               0));
   }
   if (affine) {
      estimate_affine_transformation3D(top_points, final);
   } else {
      estimate_rigid_transformation3D(top_points, final);
   }
   unsigned long num_to_keep = (unsigned long)(num_points * (percent_to_keep/100.0f));
   float* newWarpedPosition = new float[num_points*3];

   mat44 lastTransformation;
   memset(&lastTransformation,0,sizeof(mat44));

   for (int count = 0; count < max_iter; ++count)
   {
      // Transform the points in the reference
      for (unsigned j = 0; j < num_points * 3; j+=3)
         Mat44Mul(*final, reinterpret_cast<float(&)[3]>(referencePosition[j]), reinterpret_cast<float(&)[3]>(newWarpedPosition[j]));
      for (unsigned p = 0, j = 0; p < num_points; ++p, j += 3)
         residuals[p] = { SquareDistance3d(&newWarpedPosition[j], &warpedPosition[j]), j };

      // Partition so the num_to_keep smallest residuals come first (the LTS trim), then gather them.
      std::nth_element(residuals.begin(), residuals.begin() + num_to_keep, residuals.end(),
                       [](const std::pair<double, unsigned>& a, const std::pair<double, unsigned>& b) { return a.first < b.first; });
      distance = 0;
      top_points.clear();
      for (unsigned k = 0; k < num_to_keep; ++k) {
         const unsigned j = residuals[k].second;
         top_points.push_back(_reg_sorted_point3D(&referencePosition[j], &warpedPosition[j], residuals[k].first));
         distance += residuals[k].first;
      }

      // If the change is not substantial, we return
      if ((distance > lastDistance) || (lastDistance - distance) < tol)
      {
         memcpy(final, &lastTransformation, sizeof(mat44));
         break;
      }
      lastDistance = distance;
      memcpy(&lastTransformation, final, sizeof(mat44));
      if(affine) {
         estimate_affine_transformation3D(top_points, final);
      } else {
         estimate_rigid_transformation3D(top_points, final);
      }
   }
   delete[] newWarpedPosition;
}
/* *************************************************************** */
