/*
 * @file _reg_f3d_sym.h
 * @author Marc Modat
 * @date 10/11/2011
 *
 *  Copyright (c) 2011-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_F3D_SYM_H
#define _REG_F3D_SYM_H

#include "_reg_f3d.h"

/// @brief Symmetric Fast Free Form Deformation registration class
template <class T>
class reg_f3d_sym : public reg_f3d<T>
{
protected:
   // Optimiser related function
   virtual void SetOptimiser();

   nifti_image *floatingMaskImage;
   int **floatingMaskPyramid;
   int *currentFloatingMask;
   int *backwardActiveVoxelNumber;

   nifti_image *backwardControlPointGrid;
   nifti_image *backwardDeformationFieldImage;
   nifti_image *backwardWarped;
   nifti_image *backwardWarpedGradientImage;
   nifti_image *backwardVoxelBasedMeasureGradientImage;
   nifti_image *backwardTransformationGradient;

   double *backwardProbaJointHistogram;
   double *backwardLogJointHistogram;
   double backwardEntropies[4];

   mat33 *backwardJacobianMatrix;

   T inverseConsistencyWeight;
   double currentIC;
   double bestIC;

   virtual void AllocateWarped();
   virtual void ClearWarped();
   virtual void AllocateDeformationField();
   virtual void ClearDeformationField();
   virtual void AllocateWarpedGradient();
   virtual void ClearWarpedGradient();
   virtual void AllocateVoxelBasedMeasureGradient();
   virtual void ClearVoxelBasedMeasureGradient();
   virtual void AllocateTransformationGradient();
   virtual void ClearTransformationGradient();
   virtual T InitialiseCurrentLevel();
   virtual void ClearCurrentInputImage();

   virtual double ComputeBendingEnergyPenaltyTerm();
   virtual double ComputeLinearEnergyPenaltyTerm();
   virtual double ComputeJacobianBasedPenaltyTerm(int);
   virtual double ComputeLandmarkDistancePenaltyTerm();
   virtual void GetDeformationField();
   virtual void WarpFloatingImage(int);
   virtual void GetVoxelBasedGradient();
   virtual void GetSimilarityMeasureGradient();
   virtual void GetObjectiveFunctionGradient();
   virtual void GetBendingEnergyGradient();
   virtual void GetLinearEnergyGradient();
   virtual void GetJacobianBasedGradient();
   virtual void GetLandmarkDistanceGradient();
   virtual void SetGradientImageToZero();
   virtual T NormaliseGradient();
   virtual void SmoothGradient();
   virtual void GetApproximatedGradient();
   virtual void DisplayCurrentLevelParameters();
   virtual void PrintInitialObjFunctionValue();
   virtual void PrintCurrentObjFunctionValue(T);
   virtual void UpdateBestObjFunctionValue();
   virtual double GetObjectiveFunctionValue();

   virtual void GetInverseConsistencyErrorField(bool forceAll);
   virtual double GetInverseConsistencyPenaltyTerm();
   virtual void GetInverseConsistencyGradient();

   virtual void UpdateParameters(float);
   virtual void InitialiseSimilarity();

public:
   virtual void SetFloatingMask(nifti_image *);
   virtual void SetInverseConsistencyWeight(T);

   reg_f3d_sym(int refTimePoint,int floTimePoint);
   ~reg_f3d_sym();
   void CheckParameters();
   void Initialise();
   nifti_image *GetBackwardControlPointPositionImage();
   nifti_image **GetWarpedImage();
   bool GetSymmetricStatus()
   {
      return true;
   }
};

#endif
