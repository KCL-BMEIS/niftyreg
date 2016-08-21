/*
 *  _reg_aladin_sym.h
 *
 *
 *  Created by David Cash on 28/02/2012.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_ALADIN_SYM_H
#define _REG_ALADIN_SYM_H

#include "_reg_aladin.h"

/// @brief Symmetric Block matching registration class
template <class T>
class reg_aladin_sym : public reg_aladin<T>
{
private:
  AladinContent *backCon;
  Kernel *bAffineTransformation3DKernel, *bConvolutionKernel, *bBlockMatchingKernel, *bOptimiseKernel, *bResamplingKernel;

  virtual void InitCurrentLevel(unsigned int cl);

  //virtual void ClearAladinContent();
  virtual void AllocateImages();
  virtual void ClearAllocatedImages();

  virtual void CreateKernels();
  virtual void ClearKernels();

protected:
  //virtual void ClearCurrentImagePyramid();
  virtual void ClearBlockMatchingParams();
  virtual void GetBackwardDeformationField();
  virtual void UpdateTransformationMatrix(int);

  virtual void DebugPrintLevelInfoStart();
  virtual void DebugPrintLevelInfoEnd();
  virtual void InitialiseRegistration();
  virtual void GetWarpedImage(int interp = 1);

public:
  reg_aladin_sym(int platformCodeIn);
  virtual ~reg_aladin_sym();
  virtual void SetInputFloatingMask(nifti_image *);
};

#include "_reg_aladin_sym.cpp"

#endif // _REG_ALADIN_SYM_H
