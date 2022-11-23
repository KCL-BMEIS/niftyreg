/*
 *  _reg_aladin_sym.h
 *
 *
 *  Created by David Cash on 28/02/2012.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#pragma once

#include "_reg_aladin.h"

/// @brief Symmetric Block matching registration class
template <class T>
class reg_aladin_sym : public reg_aladin<T>
{
private:
  AladinContent *backCon;
  Kernel *bAffineTransformation3DKernel, *bConvolutionKernel, *bBlockMatchingKernel, *bOptimiseKernel, *bResamplingKernel;

  virtual void InitAladinContent(nifti_image *ref,
                                 nifti_image *flo,
                                 int *mask,
                                 mat44 *transMat,
                                 size_t bytes);
  virtual void InitAladinContent(nifti_image *ref,
                                 nifti_image *flo,
                                 int *mask,
                                 mat44 *transMat,
                                 size_t bytes,
                                 unsigned int blockPercentage,
                                 unsigned int inlierLts,
                                 unsigned int blockStepSize);
  virtual void ClearAladinContent();
  virtual void CreateKernels();
  virtual void ClearKernels();

protected:
  nifti_image *InputFloatingMask;
  int **FloatingMaskPyramid;
  int *BackwardActiveVoxelNumber;

  _reg_blockMatchingParam *BackwardBlockMatchingParams;

  mat44 *BackwardTransformationMatrix;

  virtual void ClearCurrentInputImage();
  virtual void GetBackwardDeformationField();
  virtual void UpdateTransformationMatrix(int);

  virtual void DebugPrintLevelInfoStart();
  virtual void DebugPrintLevelInfoEnd();
  virtual void InitialiseRegistration();
  virtual void GetWarpedImage(int, float);

public:
  reg_aladin_sym();
  virtual ~reg_aladin_sym();
  virtual void SetInputFloatingMask(nifti_image *);
};

#include "_reg_aladin_sym.cpp"
