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
#include "_reg_maths.h"
#include "_reg_maths_eigen.h"

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
   {
      referenceMatrix=&(deformationFieldImage->sto_xyz);
   }
   else referenceMatrix=&(deformationFieldImage->qto_xyz);

   mat44 transformationMatrix;
   if(composition)
      transformationMatrix = *affineTransformation;
   else transformationMatrix = reg_mat44_mul(affineTransformation, referenceMatrix);

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
               reg_mat44_mul(&transformationMatrix, voxel, position);
            }
            else reg_mat44_mul(&transformationMatrix, voxel, position);

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
   {
      referenceMatrix=&(deformationFieldImage->sto_xyz);
   }
   else referenceMatrix=&(deformationFieldImage->qto_xyz);

   mat44 transformationMatrix;
   if(composition)
      transformationMatrix = *affineTransformation;
   else transformationMatrix = reg_mat44_mul(affineTransformation, referenceMatrix);

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
               reg_mat44_mul(&transformationMatrix, voxel, position);

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
void estimate_rigid_transformation2D(float** points1, float** points2, int num_points, mat44 * transformation)
{

   double centroid_reference[2] = { 0 };
   double centroid_warped[2] = { 0 };

   float centroid_referenceFloat[2] = { 0 };
   float centroid_warpedFloat[2] = { 0 };

   for (int j = 0; j < num_points; ++j) {
      centroid_reference[0] += (double) points1[j][0];
      centroid_reference[1] += (double) points1[j][1];
      centroid_warped[0] += (double) points2[j][0];
      centroid_warped[1] += (double) points2[j][1];
   }

   centroid_reference[0] /= static_cast<double>(num_points);
   centroid_reference[1] /= static_cast<double>(num_points);

   centroid_referenceFloat[0] = static_cast<float>(centroid_reference[0]);
   centroid_referenceFloat[1] = static_cast<float>(centroid_reference[1]);

   centroid_warped[0] /= static_cast<double>(num_points);
   centroid_warped[1] /= static_cast<double>(num_points);

   centroid_warpedFloat[0] = static_cast<float>(centroid_warped[0]);
   centroid_warpedFloat[1] = static_cast<float>(centroid_warped[1]);

   float * w = reg_matrix1DAllocate<float>(2);
   float **v = reg_matrix2DAllocate<float>(2, 2);
   float **r = reg_matrix2DAllocate<float>(2, 2);

   // Demean the input points
   for (int j = 0; j < num_points; ++j) {
      points1[j][0] = static_cast<float>(static_cast<double>(points1[j][0]) - static_cast<double>(centroid_referenceFloat[0]));
      points1[j][1] = static_cast<float>(static_cast<double>(points1[j][1]) - static_cast<double>(centroid_referenceFloat[1]));

      points2[j][0] = static_cast<float>(static_cast<double>(points2[j][0]) - static_cast<double>(centroid_warpedFloat[0]));
      points2[j][1] = static_cast<float>(static_cast<double>(points2[j][1]) - static_cast<double>(centroid_warpedFloat[1]));
   }

   float **p1t = reg_matrix2DTranspose<float>(points1, num_points, 2);
   float **u = reg_matrix2DMultiply<float>(p1t,2, num_points, points2, num_points, 2, false);

   svd(u, 2, 2, w, v);

   // Calculate transpose
   float **ut = reg_matrix2DTranspose<float>(u, 2, 2);

   // Calculate the rotation matrix
   reg_matrix2DMultiply<float>(v, 2, 2, ut, 2, 2, r, false);

   float det = reg_matrix2DDet<float>(r, 2, 2);

   // Take care of possible reflection
   if (det < 0) {
      v[0][1] = -v[0][1];
      v[1][1] = -v[1][1];
      reg_matrix2DMultiply<float>(v, 2, 2, ut, 2, 2, r, false);
   }

   // Calculate the translation
   float t[2];
   t[0] = static_cast<float>(static_cast<double>(centroid_warpedFloat[0]) - (static_cast<double>(r[0][0]) * static_cast<double>(centroid_referenceFloat[0]) +
         static_cast<double>(r[0][1]) * static_cast<double>(centroid_referenceFloat[1])));

   t[1] = static_cast<float>(static_cast<double>(centroid_warpedFloat[1]) - (static_cast<double>(r[1][0]) * static_cast<double>(centroid_referenceFloat[0]) +
         static_cast<double>(r[1][1]) * static_cast<double>(centroid_referenceFloat[1])));

   transformation->m[0][0] = r[0][0];
   transformation->m[0][1] = r[0][1];
   transformation->m[0][3] = t[0];

   transformation->m[1][0] = r[1][0];
   transformation->m[1][1] = r[1][1];
   transformation->m[1][3] = t[1];

   transformation->m[2][0] = 0.0f;
   transformation->m[2][1] = 0.0f;
   transformation->m[2][2] = 1.0f;
   transformation->m[2][3] = 0.0f;

   transformation->m[0][2] = 0.0f;
   transformation->m[1][2] = 0.0f;
   transformation->m[3][2] = 0.0f;

   transformation->m[3][0] = 0.0f;
   transformation->m[3][1] = 0.0f;
   transformation->m[3][2] = 0.0f;
   transformation->m[3][3] = 1.0f;

   // Do the deletion here
   reg_matrix2DDeallocate(2, u);
   reg_matrix1DDeallocate(w);
   reg_matrix2DDeallocate(2, v);
   reg_matrix2DDeallocate(2, ut);
   reg_matrix2DDeallocate(2, r);
   //    reg_matrix2DDeallocate(2, p1t);
   for(size_t dance=0;dance<2;++dance) free(p1t[dance]); free(p1t);
}
/* *************************************************************** */
void estimate_rigid_transformation2D(std::vector<_reg_sorted_point2D> &points, mat44 * transformation)
{

   unsigned num_points = points.size();
   float** points1 = reg_matrix2DAllocate<float>(num_points, 2);
   float** points2 = reg_matrix2DAllocate<float>(num_points, 2);
   for (unsigned i = 0; i < num_points; i++) {
      points1[i][0] = points[i].reference[0];
      points1[i][1] = points[i].reference[1];
      points2[i][0] = points[i].warped[0];
      points2[i][1] = points[i].warped[1];
   }
   estimate_rigid_transformation2D(points1, points2, num_points, transformation);
   //FREE MEMORY
   reg_matrix2DDeallocate(num_points, points1);
   reg_matrix2DDeallocate(num_points, points2);
}
/* *************************************************************** */
void estimate_rigid_transformation3D(float** points1, float** points2, int num_points, mat44 * transformation)
{

   double centroid_reference[3] = { 0 };
   double centroid_warped[3] = { 0 };

   float centroid_referenceFloat[3] = { 0 };
   float centroid_warpedFloat[3] = { 0 };


   for (int j = 0; j < num_points; ++j)
   {
      centroid_reference[0] += (double) points1[j][0];
      centroid_reference[1] += (double) points1[j][1];
      centroid_reference[2] += (double) points1[j][2];

      centroid_warped[0] += (double) points2[j][0];
      centroid_warped[1] += (double) points2[j][1];
      centroid_warped[2] += (double) points2[j][2];
   }

   centroid_reference[0] /= static_cast<double>(num_points);
   centroid_reference[1] /= static_cast<double>(num_points);
   centroid_reference[2] /= static_cast<double>(num_points);

   centroid_referenceFloat[0] = static_cast<float>(centroid_reference[0]);
   centroid_referenceFloat[1] = static_cast<float>(centroid_reference[1]);
   centroid_referenceFloat[2] = static_cast<float>(centroid_reference[2]);

   centroid_warped[0] /= static_cast<double>(num_points);
   centroid_warped[1] /= static_cast<double>(num_points);
   centroid_warped[2] /= static_cast<double>(num_points);

   centroid_warpedFloat[0] = static_cast<float>(centroid_warped[0]);
   centroid_warpedFloat[1] = static_cast<float>(centroid_warped[1]);
   centroid_warpedFloat[2] = static_cast<float>(centroid_warped[2]);

   float * w = reg_matrix1DAllocate<float>(3);
   float **v  = reg_matrix2DAllocate<float>(3, 3);
   float **r  = reg_matrix2DAllocate<float>(3, 3);

   // Demean the input points
   for (int j = 0; j < num_points; ++j) {
      points1[j][0] = static_cast<float>(static_cast<double>(points1[j][0]) - static_cast<double>(centroid_referenceFloat[0]));
      points1[j][1] = static_cast<float>(static_cast<double>(points1[j][1]) - static_cast<double>(centroid_referenceFloat[1]));
      points1[j][2] = static_cast<float>(static_cast<double>(points1[j][2]) - static_cast<double>(centroid_referenceFloat[2]));

      points2[j][0] = static_cast<float>(static_cast<double>(points2[j][0]) - static_cast<double>(centroid_warpedFloat[0]));
      points2[j][1] = static_cast<float>(static_cast<double>(points2[j][1]) - static_cast<double>(centroid_warpedFloat[1]));
      points2[j][2] = static_cast<float>(static_cast<double>(points2[j][2]) - static_cast<double>(centroid_warpedFloat[2]));
   }
   //T** reg_matrix2DTranspose(T** mat, size_t arraySizeX, size_t arraySizeY);
   //T** reg_matrix2DMultiply(T** mat1, size_t mat1X, size_t mat1Y, T** mat2, size_t mat2X, size_t mat2Y, bool transposeMat2);
   float **p1t = reg_matrix2DTranspose<float>(points1, num_points, 3);
   float **u = reg_matrix2DMultiply<float>(p1t,3, num_points, points2, num_points, 3, false);

   svd(u, 3, 3, w, v);

   // Calculate transpose
   float **ut = reg_matrix2DTranspose<float>(u, 3, 3);

   // Calculate the rotation matrix
   reg_matrix2DMultiply<float>(v, 3, 3, ut, 3, 3, r, false);

   float det = reg_matrix2DDet<float>(r, 3, 3);

   // Take care of possible reflection
   if (det < 0) {
      v[0][2] = -v[0][2];
      v[1][2] = -v[1][2];
      v[2][2] = -v[2][2];
      reg_matrix2DMultiply<float>(v, 3, 3, ut, 3, 3, r, false);
   }

   // Calculate the translation
   float t[3];
   t[0] = static_cast<float>(static_cast<double>(centroid_warpedFloat[0]) - (static_cast<double>(r[0][0]) * static_cast<double>(centroid_referenceFloat[0]) +
         static_cast<double>(r[0][1]) * static_cast<double>(centroid_referenceFloat[1]) +
         static_cast<double>(r[0][2]) * static_cast<double>(centroid_referenceFloat[2])));

   t[1] = static_cast<float>(static_cast<double>(centroid_warpedFloat[1]) - (static_cast<double>(r[1][0]) * static_cast<double>(centroid_referenceFloat[0]) +
         static_cast<double>(r[1][1]) * static_cast<double>(centroid_referenceFloat[1]) +
         static_cast<double>(r[1][2]) * static_cast<double>(centroid_referenceFloat[2])));

   t[2] = static_cast<float>(static_cast<double>(centroid_warpedFloat[2]) - (static_cast<double>(r[2][0]) * static_cast<double>(centroid_referenceFloat[0]) +
         static_cast<double>(r[2][1]) * static_cast<double>(centroid_referenceFloat[1]) +
         static_cast<double>(r[2][2]) * static_cast<double>(centroid_referenceFloat[2])));

   transformation->m[0][0] = r[0][0];
   transformation->m[0][1] = r[0][1];
   transformation->m[0][2] = r[0][2];
   transformation->m[0][3] = t[0];

   transformation->m[1][0] = r[1][0];
   transformation->m[1][1] = r[1][1];
   transformation->m[1][2] = r[1][2];
   transformation->m[1][3] = t[1];

   transformation->m[2][0] = r[2][0];
   transformation->m[2][1] = r[2][1];
   transformation->m[2][2] = r[2][2];
   transformation->m[2][3] = t[2];

   transformation->m[3][0] = 0.0f;
   transformation->m[3][1] = 0.0f;
   transformation->m[3][2] = 0.0f;
   transformation->m[3][3] = 1.0f;

   // Do the deletion here
   reg_matrix2DDeallocate(3, u);
   reg_matrix1DDeallocate(w);
   reg_matrix2DDeallocate(3, v);
   reg_matrix2DDeallocate(3, ut);
   reg_matrix2DDeallocate(3, r);
   reg_matrix2DDeallocate(3, p1t);
}
/* *************************************************************** */
void estimate_rigid_transformation3D(std::vector<_reg_sorted_point3D> &points, mat44 * transformation)
{
   unsigned num_points = points.size();
   float** points1 = reg_matrix2DAllocate<float>(num_points, 3);
   float** points2 = reg_matrix2DAllocate<float>(num_points, 3);
   for (unsigned i = 0; i < num_points; i++) {
      points1[i][0] = points[i].reference[0];
      points1[i][1] = points[i].reference[1];
      points1[i][2] = points[i].reference[2];
      points2[i][0] = points[i].warped[0];
      points2[i][1] = points[i].warped[1];
      points2[i][2] = points[i].warped[2];
   }
   estimate_rigid_transformation3D(points1, points2, num_points, transformation);
   //FREE MEMORY
   reg_matrix2DDeallocate(num_points, points1);
   reg_matrix2DDeallocate(num_points, points2);
}
/* *************************************************************** */
void estimate_affine_transformation2D(float** points1, float** points2, int num_points, mat44 * transformation)
{
   //We assume same number of points in both arrays
   int num_equations = num_points * 2;
   unsigned c = 0;
   float** A = reg_matrix2DAllocate<float>(num_equations, 6);

   for (int k = 0; k < num_points; ++k) {
      c = k * 2;

      A[c][0] = points1[k][0];
      A[c][1] = points1[k][1];
      A[c][2] = A[c][3] = A[c][5] = 0.0f;
      A[c][4] = 1.0f;

      A[c + 1][2] = points1[k][0];
      A[c + 1][3] = points1[k][1];
      A[c + 1][0] = A[c + 1][1] = A[c + 1][4] = 0.0f;
      A[c + 1][5] = 1.0f;
   }

   float* w  = reg_matrix1DAllocate<float>(6);
   float** v = reg_matrix2DAllocate<float>(6, 6);

   svd(A, num_equations, 6, w, v);

   for (unsigned k = 0; k < 6; ++k) {
      if (w[k] < 0.0001) {
         w[k] = 0.0f;
      }
      else {
         w[k] = static_cast<float>(1.0 / static_cast<double>(w[k]));
      }
   }

   // Now we can compute the pseudoinverse which is given by
   // V*inv(W)*U'
   // First compute the V * inv(w) in place.
   // Simply scale each column by the corresponding singular value
   for (unsigned k = 0; k < 6; ++k) {
      for (unsigned j = 0; j < 6; ++j) {
         v[j][k] = static_cast<float>(static_cast<double>(v[j][k]) * static_cast<double>(w[k]));
      }
   }

   float** r = reg_matrix2DAllocate<float>(6, num_equations);
   reg_matrix2DMultiply<float>(v, 6, 6, A, num_equations, 6, r, true);
   // Now r contains the pseudoinverse
   // Create vector b and then multiple r*b to get the affine paramsA
   float* b = reg_matrix1DAllocate<float>(num_equations);
   for (int k = 0; k < num_points; ++k) {
      c = k * 2;
      b[c] = points2[k][0];
      b[c + 1] = points2[k][1];
   }

   float* transform = reg_matrix1DAllocate<float>(6);
   reg_matrix2DVectorMultiply<float>(r, 6, num_equations, b, transform);

   transformation->m[0][0] = transform[0];
   transformation->m[0][1] = transform[1];
   transformation->m[0][2] = 0.0f;
   transformation->m[0][3] = transform[4];

   transformation->m[1][0] = transform[2];
   transformation->m[1][1] = transform[3];
   transformation->m[1][2] = 0.0f;
   transformation->m[1][3] = transform[5];

   transformation->m[2][0] = 0.0f;
   transformation->m[2][1] = 0.0f;
   transformation->m[2][2] = 1.0f;
   transformation->m[2][3] = 0.0f;

   transformation->m[3][0] = 0.0f;
   transformation->m[3][1] = 0.0f;
   transformation->m[3][2] = 0.0f;
   transformation->m[3][3] = 1.0f;

   // Do the deletion here
   reg_matrix1DDeallocate(transform);
   reg_matrix1DDeallocate(b);
   reg_matrix2DDeallocate(6, r);
   reg_matrix2DDeallocate(6, v);
   reg_matrix1DDeallocate(w);
   reg_matrix2DDeallocate(num_equations, A);
}
/* *************************************************************** */
void estimate_affine_transformation2D(std::vector<_reg_sorted_point2D> &points, mat44 * transformation)
{
   unsigned num_points = points.size();
   float** points1 = reg_matrix2DAllocate<float>(num_points, 2);
   float** points2 = reg_matrix2DAllocate<float>(num_points, 2);
   for (unsigned i = 0; i < num_points; i++) {
      points1[i][0] = points[i].reference[0];
      points1[i][1] = points[i].reference[1];
      points2[i][0] = points[i].warped[0];
      points2[i][1] = points[i].warped[1];
   }
   estimate_affine_transformation2D(points1, points2, num_points, transformation);
   //FREE MEMORY
   reg_matrix2DDeallocate(num_points, points1);
   reg_matrix2DDeallocate(num_points, points2);
}
/* *************************************************************** */
// estimate an affine transformation using least square
void estimate_affine_transformation3D(float** points1, float** points2, int num_points, mat44 * transformation)
{
   //We assume same number of points in both arrays

   // Create our A matrix
   // we need at least 4 points. Assuming we have that here.
   int num_equations = num_points * 3;
   unsigned c = 0;
   float** A = reg_matrix2DAllocate<float>(num_equations, 12);

   for (int k = 0; k < num_points; ++k) {
      c = k * 3;
      A[c][0] = points1[k][0];
      A[c][1] = points1[k][1];
      A[c][2] = points1[k][2];
      A[c][3] = A[c][4] = A[c][5] = A[c][6] = A[c][7] = A[c][8] = A[c][10] = A[c][11] = 0.0f;
      A[c][9] = 1.0f;

      A[c + 1][3] = points1[k][0];
      A[c + 1][4] = points1[k][1];
      A[c + 1][5] = points1[k][2];
      A[c + 1][0] = A[c + 1][1] = A[c + 1][2] = A[c + 1][6] = A[c + 1][7] = A[c + 1][8] = A[c + 1][9] = A[c + 1][11] = 0.0f;
      A[c + 1][10] = 1.0f;

      A[c + 2][6] = points1[k][0];
      A[c + 2][7] = points1[k][1];
      A[c + 2][8] = points1[k][2];
      A[c + 2][0] = A[c + 2][1] = A[c + 2][2] = A[c + 2][3] = A[c + 2][4] = A[c + 2][5] = A[c + 2][9] = A[c + 2][10] = 0.0f;
      A[c + 2][11] = 1.0f;
   }

   float* w = reg_matrix1DAllocate<float>(12);
   float** v = reg_matrix2DAllocate<float>(12, 12);
   // Now we can compute our svd
   svd(A, num_equations, 12, w, v);

   // First we make sure that the really small singular values
   // are set to 0. and compute the inverse by taking the reciprocal
   // of the entries
   for (unsigned k = 0; k < 12; ++k) {
      if (w[k] < 0.0001) {
         w[k] = 0.0f;
      }
      else {
         w[k] = static_cast<float>(1.0 / static_cast<double>(w[k]));
      }
   }

   // Now we can compute the pseudoinverse which is given by
   // V*inv(W)*U'
   // First compute the V * inv(w) in place.
   // Simply scale each column by the corresponding singular value
   for (unsigned k = 0; k < 12; ++k) {
      for (unsigned j = 0; j < 12; ++j) {
         v[j][k] = static_cast<float>(static_cast<double>(v[j][k]) * static_cast<double>(w[k]));
      }
   }

   // Now multiply the matrices together
   // Pseudoinverse = v * w * A(transpose)
   float** r = reg_matrix2DAllocate<float>(12, num_equations);
   reg_matrix2DMultiply<float>(v, 12, 12, A, num_equations, 12, r, true);
   // Now r contains the pseudoinverse
   // Create vector b and then multiple rb to get the affine paramsA
   float* b = reg_matrix1DAllocate<float>(num_equations);
   for (int k = 0; k < num_points; ++k) {
      c = k * 3;
      b[c] = points2[k][0];
      b[c + 1] = points2[k][1];
      b[c + 2] = points2[k][2];
   }

   float * transform = reg_matrix1DAllocate<float>(12);
   //mul_matvec(r, 12, num_equations, b, transform);
   reg_matrix2DVectorMultiply<float>(r, 12, num_equations, b, transform);

   transformation->m[0][0] = transform[0];
   transformation->m[0][1] = transform[1];
   transformation->m[0][2] = transform[2];
   transformation->m[0][3] = transform[9];

   transformation->m[1][0] = transform[3];
   transformation->m[1][1] = transform[4];
   transformation->m[1][2] = transform[5];
   transformation->m[1][3] = transform[10];

   transformation->m[2][0] = transform[6];
   transformation->m[2][1] = transform[7];
   transformation->m[2][2] = transform[8];
   transformation->m[2][3] = transform[11];

   transformation->m[3][0] = 0.0f;
   transformation->m[3][1] = 0.0f;
   transformation->m[3][2] = 0.0f;
   transformation->m[3][3] = 1.0f;

   // Do the deletion here
   reg_matrix1DDeallocate(transform);
   reg_matrix1DDeallocate(b);
   reg_matrix2DDeallocate(12, r);
   reg_matrix2DDeallocate(12, v);
   reg_matrix1DDeallocate(w);
   reg_matrix2DDeallocate(num_equations, A);
}
/* *************************************************************** */
// estimate an affine transformation using least square
void estimate_affine_transformation3D(std::vector<_reg_sorted_point3D> &points, mat44 * transformation)
{
   unsigned num_points = points.size();
   float** points1 = reg_matrix2DAllocate<float>(num_points, 3);
   float** points2 = reg_matrix2DAllocate<float>(num_points, 3);
   for (unsigned i = 0; i < num_points; i++) {
      points1[i][0] = points[i].reference[0];
      points1[i][1] = points[i].reference[1];
      points1[i][2] = points[i].reference[2];
      points2[i][0] = points[i].warped[0];
      points2[i][1] = points[i].warped[1];
      points2[i][2] = points[i].warped[2];
   }
   estimate_affine_transformation3D(points1, points2, num_points, transformation);
   //FREE MEMORY
   reg_matrix2DDeallocate(num_points, points1);
   reg_matrix2DDeallocate(num_points, points2);
}
/* *************************************************************** */
///LTS 2D
void optimize_2D(float* referencePosition, float* warpedPosition,
                 unsigned activeBlockNumber, int percent_to_keep, int max_iter, double tol,
                 mat44 * final, bool affine) {

   // Set the current transformation to identity
   reg_mat44_eye(final);

   const unsigned num_points = activeBlockNumber;
   unsigned long num_equations = num_points * 2;
   // Keep a sorted list of the distance measure
   std::multimap<double, _reg_sorted_point2D> queue;
   std::vector<_reg_sorted_point2D> top_points;

   double distance = 0;
   double lastDistance = std::numeric_limits<double>::max();
   unsigned long i;

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
      {
         reg_mat33_mul(final, &referencePosition[j], &newWarpedPosition[j]);
      }
      queue = std::multimap<double, _reg_sorted_point2D>();
      for (unsigned j = 0; j < num_points * 2; j += 2)
      {
         distance = get_square_distance2D(&newWarpedPosition[j], &warpedPosition[j]);
         queue.insert(std::pair<double, _reg_sorted_point2D>(distance,
                                                             _reg_sorted_point2D(&referencePosition[j], &warpedPosition[j], distance)));
      }

      distance = 0;
      i = 0;
      top_points.clear();

      for (std::multimap<double, _reg_sorted_point2D>::iterator it = queue.begin();
           it != queue.end(); ++it, ++i)
      {
         if (i >= num_to_keep) break;
         top_points.push_back((*it).second);
         distance += (*it).first;
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
   reg_mat44_eye(final);

   const unsigned num_points = activeBlockNumber;
   unsigned long num_equations = num_points * 3;
   // Keep a sorted list of the distance measure
   std::multimap<double, _reg_sorted_point3D> queue;
   std::vector<_reg_sorted_point3D> top_points;
   double distance = 0;
   double lastDistance = std::numeric_limits<double>::max();
   unsigned long i;

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
      for (unsigned j = 0; j < num_points * 3; j+=3) {
         reg_mat44_mul(final, &referencePosition[j], &newWarpedPosition[j]);
      }
      queue = std::multimap<double, _reg_sorted_point3D>();
      for (unsigned j = 0; j < num_points * 3; j+= 3)
      {
         distance = get_square_distance3D(&newWarpedPosition[j], &warpedPosition[j]);
         queue.insert(std::pair<double,
                      _reg_sorted_point3D>(distance,
                                           _reg_sorted_point3D(&referencePosition[j],
                                                               &warpedPosition[j],
                                                               distance)));
      }

      distance = 0;
      i = 0;
      top_points.clear();
      for (std::multimap<double, _reg_sorted_point3D>::iterator it = queue.begin();it != queue.end(); ++it, ++i)
      {
         if (i >= num_to_keep) break;
         top_points.push_back((*it).second);
         distance += (*it).first;
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
   delete [] newWarpedPosition;
}
/* *************************************************************** */
