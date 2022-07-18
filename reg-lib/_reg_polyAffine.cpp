/**
 * @file _reg_polyAffine.cpp
 * @author Marc Modat
 * @date 16/11/2012
 *
 * Copyright (c) 2012-2018, University College London
 * Copyright (c) 2018, NiftyReg Developers.
 * All rights reserved.
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_POLYAFFINE_CPP
#define _REG_POLYAFFINE_CPP

#include "_reg_polyAffine.h"

/* *************************************************************** */
/* *************************************************************** */
template <class T>
reg_polyAffine<T>::reg_polyAffine(int refTimePoint,int floTimePoint)
   : reg_base<T>::reg_base(refTimePoint,floTimePoint)
{
   this->executableName=(char *)"NiftyReg PolyAffine";

#ifndef NDEBUG
   reg_print_msg_debug("reg_polyAffine constructor called");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
reg_polyAffine<T>::~reg_polyAffine()
{
#ifndef NDEBUG
   reg_print_msg_debug("reg_polyAffine destructor called");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_polyAffine<T>::GetDeformationField()
{

}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_polyAffine<T>::SetGradientImageToZero()
{

}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_polyAffine<T>::GetApproximatedGradient()
{

}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
double reg_polyAffine<T>::GetObjectiveFunctionValue()
{

   return EXIT_SUCCESS;
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_polyAffine<T>::UpdateParameters(float stepSize)
{

}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
T reg_polyAffine<T>::NormaliseGradient()
{
   return EXIT_SUCCESS;
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_polyAffine<T>::GetSimilarityMeasureGradient()
{

}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_polyAffine<T>::GetObjectiveFunctionGradient()
{

}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_polyAffine<T>::DisplayCurrentLevelParameters()
{

}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_polyAffine<T>::UpdateBestObjFunctionValue()
{

}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_polyAffine<T>::PrintCurrentObjFunctionValue(T stepSize)
{

}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_polyAffine<T>::PrintInitialObjFunctionValue()
{

}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_polyAffine<T>::AllocateTransformationGradient()
{

}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_polyAffine<T>::ClearTransformationGradient()
{

}
/* *************************************************************** */
/* *************************************************************** */

#endif // _REG_POLYAFFINE_CPP
