/*
 *  _reg_ssd.cpp
 *  
 *
 *  Created by Marc Modat on 19/05/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_dti.h"

/* *************************************************************** */
/* *************************************************************** */
reg_dti::reg_dti()
    : reg_ssd()
{
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] reg_dti constructor called\n");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
reg_dti::~reg_dti()
{
}
/* *************************************************************** */
/* *************************************************************** */
//void reg_dti::InitialiseMeasure(nifti_image *refImgPtr,
//                                nifti_image *floImgPtr,
//                                int *maskRefPtr,
//                                nifti_image *warFloImgPtr,
//                                nifti_image *warFloGraPtr,
//                                nifti_image *forVoxBasedGraPtr,
//                                int *maskFloPtr,
//                                nifti_image *warRefImgPtr,
//                                nifti_image *warRefGraPtr,
//                                nifti_image *bckVoxBasedGraPtr)
//{
//    // Set the pointers using the parent class function
//    reg_measure::InitialiseMeasure(refImgPtr,
//                                   floImgPtr,
//                                   maskRefPtr,
//                                   warFloImgPtr,
//                                   warFloGraPtr,
//                                   forVoxBasedGraPtr,
//                                   maskFloPtr,
//                                   warRefImgPtr,
//                                   warRefGraPtr,
//                                   bckVoxBasedGraPtr);
//}
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
double reg_getDTIMeasure()
{
    return 0.;
}
/* *************************************************************** */
//double reg_dti::GetSimilarityMeasureValue()
//{
//    return 0.;
//}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_getVoxelBasedDTIMeasureGradient()
{
    return;
}
/* *************************************************************** */
//void reg_dti::GetVoxelBasedSimilarityMeasureGradient()
//{

//}
/* *************************************************************** */
/* *************************************************************** */
