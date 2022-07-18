/**
 * @file _reg_f3d.h
 * @author Marc Modat
 * @date 19/11/2010
 *
 *  Copyright (c) 2010-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_F3D_H
#define _REG_F3D_H

#include "_reg_base.h"

/// @brief Fast Free Form Deformation registration class
template <class T>
class reg_f3d : public reg_base<T>
{
protected:
   nifti_image *inputControlPointGrid; // pointer to external
   nifti_image *controlPointGrid;
   T bendingEnergyWeight;
   T linearEnergyWeight;
   T jacobianLogWeight;
   bool jacobianLogApproximation;
   T spacing[3];

   nifti_image *transformationGradient;
   bool gridRefinement;

   double currentWJac;
   double currentWBE;
   double currentWLE;
   double bestWJac;
   double bestWBE;
   double bestWLE;

   virtual void AllocateTransformationGradient();
   virtual void ClearTransformationGradient();
   virtual T InitialiseCurrentLevel();

   virtual double ComputeBendingEnergyPenaltyTerm();
   virtual double ComputeLinearEnergyPenaltyTerm();
   virtual double ComputeJacobianBasedPenaltyTerm(int);
   virtual double ComputeLandmarkDistancePenaltyTerm();

   virtual void GetBendingEnergyGradient();
   virtual void GetLinearEnergyGradient();
   virtual void GetJacobianBasedGradient();
   virtual void GetLandmarkDistanceGradient();
   virtual void SetGradientImageToZero();
   virtual T NormaliseGradient();
   virtual void SmoothGradient();
   virtual void GetObjectiveFunctionGradient();
   virtual void GetApproximatedGradient();
   void GetSimilarityMeasureGradient();

   virtual void GetDeformationField();
   virtual void DisplayCurrentLevelParameters();

   virtual double GetObjectiveFunctionValue();
   virtual void UpdateBestObjFunctionValue();
   virtual void UpdateParameters(float);
   virtual void SetOptimiser();

   virtual void PrintInitialObjFunctionValue();
   virtual void PrintCurrentObjFunctionValue(T);

   virtual void CorrectTransformation();

   void (*funcProgressCallback)(float pcntProgress, void *params);
   void *paramsProgressCallback;

public:
   reg_f3d(int refTimePoint,int floTimePoint);
   virtual ~reg_f3d();

   void SetControlPointGridImage(nifti_image *);
   void SetBendingEnergyWeight(T);
   void SetLinearEnergyWeight(T);
   void SetJacobianLogWeight(T);
   void ApproximateJacobianLog();
   void DoNotApproximateJacobianLog();
   void SetSpacing(unsigned int ,T);

   void NoGridRefinement()
   {
      this->gridRefinement=false;
   }
   // F3D2 specific options
   virtual void SetCompositionStepNumber(int)
   {
      return;
   }
   virtual void ApproximateComposition()
   {
      return;
   }
   virtual void UseSimilaritySymmetry()
   {
      return;
   }
   virtual void UseBCHUpdate(int)
   {
      return;
   }
   virtual void UseGradientCumulativeExp()
   {
      return;
   }
   virtual void DoNotUseGradientCumulativeExp()
   {
      return;
   }

   // F3D_SYM specific options
   virtual void SetFloatingMask(nifti_image *)
   {
      return;
   }
   virtual void SetInverseConsistencyWeight(T)
   {
      return;
   }
   virtual nifti_image *GetBackwardControlPointPositionImage()
   {
      return NULL;
   }

   // F3D_gpu specific option
   virtual int CheckMemoryMB()
   {
      return EXIT_SUCCESS;
   }

   virtual void CheckParameters();
   virtual void Initialise();
   virtual nifti_image *GetControlPointPositionImage();
   virtual nifti_image **GetWarpedImage();

   // Function used for testing
   virtual void reg_test_setControlPointGrid(nifti_image *cpp)
   {
      this->controlPointGrid=cpp;
   }
};

#endif
