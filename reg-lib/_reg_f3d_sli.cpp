/*
*  _reg_f3d_sli.cpp
*
*
*  Created by Jamie McClelland on 20/08/2017.
*  Copyright (c) 2017, University College London. All rights reserved.
*  Centre for Medical Image Computing (CMIC)
*  See the LICENSE.txt file in the nifty_reg root folder
*
*/


#ifndef _REG_F3D_SLI_CPP
#define _REG_F3D_SLI_CPP

#include "_reg_f3d_sli.h"

/* *************************************************************** */
/* *************************************************************** */
template <class T>
reg_f3d_sli<T>::reg_f3d_sli(int refTimePoint, int floTimePoint)
	:reg_f3d<T>::reg_f3d(refTimePoint, floTimePoint)
{
	this->executableName = (char *)"NiftyReg F3D SLI";

	this->region2ControlPointGrid = NULL;
	this->region2DeformationFieldImage = NULL;
	this->region2VoxelBasedMeasureGradientImage = NULL;
	this->region2TransformationGradient = NULL;

	this->region1DeformationFieldImage = NULL;
	this->region1VoxelBasedMeasureGradientImage = NULL;

	this->distanceMapImage = NULL;
	this->distanceMapPyramid = NULL;
	this->currentDistanceMap = NULL;
	this->warpedDistanceMapRegion1 = NULL;
	this->warpedDistanceMapRegion2 = NULL;

	this->gapOverlapWeight = 0.1;

#ifndef NDEBUG
	reg_print_fct_debug("reg_f3d_sli<T>::reg_f3d_sli");
#endif
}
/* *************************************************************** */
template <class T>
reg_f3d_sli<T>::~reg_f3d_sli()
{
	if (this->region2ControlPointGrid != NULL)
	{
		nifti_image_free(this->region2ControlPointGrid);
		this->region2ControlPointGrid = NULL;
	}

	if (this->distanceMapPyramid != NULL)
	{
		if (this->usePyramid)
		{
			for (unsigned int n = 0; n < this->levelToPerform; n++)
			{
				if (this->distanceMapPyramid[n] != NULL)
				{
					nifti_image_free(this->distanceMapPyramid[n]);
					this->distanceMapPyramid[n] = NULL;
				}
			}
		}
		else
		{
			if (this->distanceMapPyramid[0] != NULL)
			{
				nifti_image_free(this->distanceMapPyramid[0]);
				this->distanceMapPyramid[0] = NULL;
			}
		}
	}

#ifndef NDEBUG
	reg_print_fct_debug("reg_f3d_sli<T>::~reg_f3d_sli");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d_sli<T>::GetDeformationField()
{
	//get deformation field for region1
	reg_spline_getDeformationField(this->controlPointGrid,
		this->region1DeformationFieldImage,
		this->currentMask,
		false, //composition
		true); //bspline
	//get deformation field for region2
	reg_spline_getDeformationField(this->region2ControlPointGrid,
		this->region2DeformationFieldImage,
		this->currentMask,
		false, //composition
		true); //bspline

	//warp distance map using region1 def field
	reg_resampleImage(this->currentDistanceMap,
		this->warpedDistanceMapRegion1,
		this->region1DeformationFieldImage,
		this->currentMask,
		this->interpolation,
		std::numeric_limits<T>::quiet_NaN()); //set padding value to NaN
	//warp distance map using region2 def field
	reg_resampleImage(this->currentDistanceMap,
		this->warpedDistanceMapRegion2,
		this->region2DeformationFieldImage,
		this->currentMask,
		this->interpolation,
		std::numeric_limits<T>::quiet_NaN()); //set padding value to NaN

	//loop over voxels and set combined deformation field (deformationFieldImage)
	//using appropriate region, based on warped distance maps
	size_t numVox = this->region1DeformationFieldImage->nx *
		this->region1DeformationFieldImage->ny *
		this->region1DeformationFieldImage->nz;
	//need pointers 

}
/* *************************************************************** */
/* *************************************************************** */
template class reg_f3d_sli<float>;
#endif
