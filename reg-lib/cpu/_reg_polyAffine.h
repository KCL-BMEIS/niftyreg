/**
 * @file _reg_polyAffine.h
 * @author Marc Modat
 * @date 16/11/2012
 *
 * Copyright (c) 2012-2018, University College London
 * Copyright (c) 2018, NiftyReg Developers.
 * All rights reserved.
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#pragma once

#include "_reg_base.h"

template <class T>
class reg_polyAffine : public reg_base<T>
{
protected:
   void GetDeformationField();
   void SetGradientImageToZero();
   void GetApproximatedGradient();
   double GetObjectiveFunctionValue();
   void UpdateParameters(float);
   T NormaliseGradient();
   void GetSimilarityMeasureGradient();
   void GetObjectiveFunctionGradient();
   void DisplayCurrentLevelParameters();
   void UpdateBestObjFunctionValue();
   void PrintCurrentObjFunctionValue(T);
   void PrintInitialObjFunctionValue();
   void AllocateTransformationGradient();
   void DeallocateTransformationGradient();

public:
   reg_polyAffine(int refTimePoints,int floTimePoints);
   ~reg_polyAffine();
};

#include "_reg_polyAffine.cpp"
