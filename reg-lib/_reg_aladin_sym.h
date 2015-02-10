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


template <class T>
class reg_aladin_sym : public reg_aladin<T>
{
private:
	Content *backCon;
	Kernel *bAffineTransformation3DKernel, *bConvolutionKernel, *bBlockMatchingKernel, *bOptimiseKernel, *bResamplingKernel;

	virtual void initContent(nifti_image *ref,
									 nifti_image *flo,
									 int *mask,
									 mat44 *transMat,
									 size_t bytes);
	virtual void initContent(nifti_image *ref,
									 nifti_image *flo,
									 int *mask,
									 mat44 *transMat,
									 size_t bytes,
									 unsigned int blockPercentage,
									 unsigned int inlierLts,
									 unsigned int blockStepSize);
	virtual void clearContent();
	virtual void createKernels();
	virtual void clearKernels();

protected:
   nifti_image *InputFloatingMask;
   nifti_image *CurrentBackwardWarped;
   int **FloatingMaskPyramid;
   nifti_image *BackwardDeformationFieldImage;
   int *CurrentFloatingMask;
   int *BackwardActiveVoxelNumber;

   _reg_blockMatchingParam *BackwardBlockMatchingParams;

   mat44 *BackwardTransformationMatrix;

   virtual void ClearCurrentInputImage();
   virtual void ClearBackwardWarpedImage();
   virtual void ClearBackwardDeformationField();
   virtual void GetBackwardDeformationField();
   virtual void UpdateTransformationMatrix(int);

   virtual void DebugPrintLevelInfoStart();
   virtual void DebugPrintLevelInfoEnd();
   virtual void InitialiseRegistration();
   virtual void SetCurrentImages();
   virtual void GetWarpedImage(int);

public:
   reg_aladin_sym();
   virtual ~reg_aladin_sym();


   virtual void SetInputFloatingMask(nifti_image *);
};

#include "_reg_aladin_sym.cpp"

#endif // _REG_ALADIN_SYM_H
