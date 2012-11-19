/**
 * @file _reg_f3d2.h
 * @author Marc Modat
 * @date 19/11/2011
 *
 *  Copyright (c) 2011, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_f3d_sym.h"

#ifndef _REG_F3D2_H
#define _REG_F3D2_H

template <class T>
class reg_f3d2 : public reg_f3d_sym<T>
{
  protected:
    int stepNumber;
    bool BCHUpdate;
	bool ISS;
    int BCHUpdateValue;
    mat33 *forward2backward_reorient;
    mat33 *backward2forward_reorient;

    virtual void DefineReorientationMatrices();
    virtual void GetDeformationField();
    virtual void GetInverseConsistencyErrorField();
    virtual void GetInverseConsistencyGradient();
    virtual void GetSimilarityMeasureGradient();
    virtual void UpdateParameters(float);
    virtual void ExponentiateGradient();
    virtual void UseBCHUpdate(int);
	virtual void UseInverseSclalingSquaring();

public:
    virtual void SetCompositionStepNumber(int);
    reg_f3d2(int refTimePoint,int floTimePoint);
    ~reg_f3d2();
    virtual void Initisalise();
    virtual nifti_image **GetWarpedImage();
};

#include "_reg_f3d2.cpp"

#endif
