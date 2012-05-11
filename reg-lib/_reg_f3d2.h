/*
 *  _reg_f3d2.h
 *
 *
 *  Created by Marc Modat on 19/11/2010.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_f3d_sym.h"

#ifdef BUILD_NR_DEV

#ifndef _REG_F3D2_H
#define _REG_F3D2_H


#define NR_F3D2_BCH_TYPE 1
// 0 - w=u+v
// 1 - w=u+v+0.5*[u,v]
// 2 - w=u+v+0.5*[u,v]+[u,[u,v]]/12
// 3 - w=u+v+0.5*[u,v]+[u,[u,v]]/12-[v,[u,v]]/12
// 4 - w=u+v+0.5*[u,v]+[u,[u,v]]/12-[v,[u,v]]/12-[v,[u,[u,g]]]/24

template <class T>
class reg_f3d2 : public reg_f3d_sym<T>
{
  protected:
    int stepNumber;

    virtual void GetDeformationField();
    virtual void GetInverseConsistencyErrorField();
    virtual void GetInverseConsistencyGradient();
    virtual void UpdateControlPointPosition(T);

public:
    virtual void SetCompositionStepNumber(int);
    reg_f3d2(int refTimePoint,int floTimePoint);
    ~reg_f3d2();
    virtual void Initisalise_f3d();
    virtual nifti_image **GetWarpedImage();
};

#include "_reg_f3d2.cpp"

#endif

#endif
