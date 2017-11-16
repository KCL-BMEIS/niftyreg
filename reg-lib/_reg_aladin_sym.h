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

  virtual void initAladinContent(nifti_image *ref,
                                 nifti_image *flo,
                                 int *mask,
                                 mat44 *transMat,
                                 size_t bytes);
  virtual void initAladinContent(nifti_image *ref,
                                 nifti_image *flo,
                                 int *mask,
                                 mat44 *transMat,
                                 size_t bytes,
                                 unsigned int blockPercentage,
                                 unsigned int inlierLts,
                                 unsigned int blockStepSize);
  virtual void clearAladinContent();
  virtual void createKernels();
  virtual void clearKernels();

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

#endif // _REG_ALADIN_SYM_H
