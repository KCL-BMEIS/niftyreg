/**
 * @file _reg_ssd.h
 * @brief File that contains sum squared difference related function
 * @author Marc Modat
 * @date 19/05/2009
 *
 *  Created by Marc Modat on 19/05/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_DTI_H
#define _REG_DTI_H

//#include "_reg_measure.h"
#include "_reg_ssd.h" // HERE

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/// @brief DTI related measure of similarity class
class reg_dti : public reg_ssd
{
public:
    /// @brief reg_dti class constructor
    reg_dti();
//    /// @brief Initialise the reg_dti object
//	void InitialiseMeasure(nifti_image *refImgPtr,
//						   nifti_image *floImgPtr,
//						   int *maskRefPtr,
//						   nifti_image *warFloImgPtr,
//						   nifti_image *warFloGraPtr,
//						   nifti_image *forVoxBasedGraPtr,
//						   int *maskFloPtr = NULL,
//						   nifti_image *warRefImgPtr = NULL,
//						   nifti_image *warRefGraPtr = NULL,
//						   nifti_image *bckVoxBasedGraPtr = NULL);
//    /// @brief Returns the value
//    double GetSimilarityMeasureValue();
//    /// @brief Compute the voxel based gradient for DTI images
//    void GetVoxelBasedSimilarityMeasureGradient();
    /// @brief reg_dti class destructor
    ~reg_dti();
};
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

extern "C++" template <class DTYPE>
double reg_getDTIMeasure();

extern "C++" template <class DTYPE>
void reg_getVoxelBasedDTIMeasureGradient();
#endif
