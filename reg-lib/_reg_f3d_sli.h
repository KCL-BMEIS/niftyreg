/*
* @file _reg_f3d_sli.h
* @author Jamie McClelland
* @date 20/08/2017
*
*  Copyright (c) 2017, University College London. All rights reserved.
*  Centre for Medical Image Computing (CMIC)
*  See the LICENSE.txt file in the nifty_reg root folder
*
*/

#ifndef _REG_F3D_SLI_H
#define _REG_F3D_SLI_H

#include "_reg_f3d.h"

/// @brief a class for Fast Free Form Deformation registration wiht sliding regions

/*
FFD registration that allows for two sliding regions, defined in the source image
a distance map (in the source image) must be provided, giving the distance to the 
sliding interface for all voxels, with voxels in region1 having negative values,
and voxels in region2 having positive values
The deformation in each region is represented by a separate B-spline transformation
and a penalty term is used to penalise gaps/overlaps between the two regions.
*/
template <class T>
class reg_f3d_sli : public reg_f3d<T>
{
protected:
	//variables for region2 transform
	nifti_image *region2ControlPointGrid;
	nifti_image *region2DeformationFieldImage;
	nifti_image *region2VoxelBasedMeasureGradientImage;
	nifti_image *region2TransformationGradient;

	//variables for region1 transform
	nifti_image *region1DeformationFieldImage;
	nifti_image *region1VoxelBasedMeasureGradientImage;

	//variables for distance map image
	nifti_image *distanceMapImage;
	nifti_image **distanceMapPyramid;
	nifti_image *currentDistanceMap;
	nifti_image *warpedDistanceMapRegion1;
	nifti_image *warpedDistanceMapRegion2;

	//variables for penalty term
	T gapOverlapWeight;
	double currentWGO;
	double bestWGO;


	//reimplement function to get deformation field
	//combines deformation fields from each region based on warped distance maps
	virtual void GetDeformationField();

	//new functions for Gap-Overlap penalty term
	//virtual double GetGapOverlapPenaltyTerm();
	//virtual void GetGapOverlapGradient();

public:
	reg_f3d_sli(int refTimePoint, int floTimePoint);
	~reg_f3d_sli();
};

#endif
