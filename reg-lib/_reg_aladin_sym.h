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
class reg_aladin_sym: public reg_aladin<T> {
private:
    unique_ptr<AladinContent> backCon;
    unique_ptr<Kernel> bAffineTransformation3DKernel, bConvolutionKernel, bBlockMatchingKernel, bLtsKernel, bResamplingKernel;

    virtual void InitAladinContent(nifti_image *ref,
                                   nifti_image *flo,
                                   int *mask,
                                   mat44 *transMat,
                                   size_t bytes,
                                   unsigned blockPercentage = 0,
                                   unsigned inlierLts = 0,
                                   unsigned blockStepSize = 0);
    virtual void DeinitAladinContent();
    virtual void CreateKernels();
    virtual void DeallocateKernels();

protected:
    NiftiImage inputFloatingMask;
    vector<unique_ptr<int[]>> floatingMaskPyramid;

    _reg_blockMatchingParam *backwardBlockMatchingParams;

    unique_ptr<mat44> affineTransformationBw;

    virtual void DeallocateCurrentInputImage();
    virtual void GetBackwardDeformationField();
    virtual void UpdateTransformationMatrix(int);

    virtual void DebugPrintLevelInfoStart();
    virtual void DebugPrintLevelInfoEnd();
    virtual void InitialiseRegistration();
    virtual void GetWarpedImage(int, float);

public:
    reg_aladin_sym();
    virtual void SetInputFloatingMask(NiftiImage);
};
