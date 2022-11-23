/*
 *  _reg_thinPlateSpline.h
 *
 *
 *  Created by Marc Modat on 22/02/2011.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#pragma once

#include "_reg_maths.h"

/* *************************************************************** */
template <class T>
class reg_tps
{
protected:
   T *positionX;
   T *positionY;
   T *positionZ;
   T *coefficientX;
   T *coefficientY;
   T *coefficientZ;
   size_t dim;
   size_t number;
   bool initialised;
   T approxInter;

   T GetTPSEuclideanDistance(size_t i, size_t j);
   T GetTPSEuclideanDistance(size_t i, T *p);
   T GetTPSweight(T dist);

public:
   reg_tps(size_t d,size_t n);
   ~reg_tps();
   void SetPosition(T*,T*,T*,T*,T*,T*);
   void SetPosition(T*,T*,T*,T*);
   void SetAproxInter(T);

   void InitialiseTPS();
   void FillDeformationField(nifti_image *deformationField);
};


#include "_reg_thinPlateSpline.cpp"
