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

#ifndef _REG_F3D2_H
#define _REG_F3D2_H

#include "_reg_f3d.h"

template <class T>
class reg_f3d2 : public reg_f3d<T>
{
  protected:
    nifti_image *inverseDeformationFieldImage;
    nifti_image *negatedControlPointGrid;
//    bool useSymmetry;

    int AllocateDeformationField();
    int ClearDeformationField();
    int AllocateCurrentInputImage(int);
    int ClearCurrentInputImage();

    int GetDeformationField();
    int CheckStoppingCriteria(bool);

public:
    reg_f3d2(int refTimePoint,int floTimePoint);
    ~reg_f3d2();
    int Run_f3d();
    nifti_image *GetWarpedImage();
};

#include "_reg_f3d2.cpp"

#endif
