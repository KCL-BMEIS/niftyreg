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
	//do not use a mask as gap-overlap penalty term calculted for voxels outside mask
	reg_spline_getDeformationField(this->controlPointGrid,
		this->region1DeformationFieldImage,
		NULL, //no mask
		false, //composition
		true); //bspline
	//get deformation field for region2
	reg_spline_getDeformationField(this->region2ControlPointGrid,
		this->region2DeformationFieldImage,
		NULL, //no mask
		false, //composition
		true); //bspline

	//warp distance map using region1 def field
	reg_resampleImage(this->currentDistanceMap,
		this->warpedDistanceMapRegion1,
		this->region1DeformationFieldImage,
		NULL, //no mask
		this->interpolation,
		std::numeric_limits<T>::quiet_NaN()); //set padding value to NaN
	//warp distance map using region2 def field
	reg_resampleImage(this->currentDistanceMap,
		this->warpedDistanceMapRegion2,
		this->region2DeformationFieldImage,
		NULL, //no mask
		this->interpolation,
		std::numeric_limits<T>::quiet_NaN()); //set padding value to NaN

	//loop over voxels and set combined deformation field (deformationFieldImage)
	//using appropriate region, based on warped distance maps
	//combined def field only needs to be set within the mask
	size_t numVox = this->region1DeformationFieldImage->nx *
		this->region1DeformationFieldImage->ny *
		this->region1DeformationFieldImage->nz;
	//pointers to deformation fields
	T *region1DFPtrX = static_cast<T *>(this->region1DeformationFieldImage->data);
	T *region1DFPtrY = &region1DFPtrX[numVox];
	T *region1DFPtrZ = NULL;
	T *region2DFPtrX = static_cast<T *>(this->region2DeformationFieldImage->data);
	T *region2DFPtrY = &region2DFPtrX[numVox];
	T *region2DFPtrZ = NULL;
	T *combinedDFPtrX = static_cast<T *>(this->deformationFieldImage->data);
	T *combinedDFPtrY = &combinedDFPtrX[numVox];
	T *combinedDFPtrZ = NULL;
	//pointers to warped distance maps
	T *warpedDMR1Ptr = static_cast<T *>(this->warpedDistanceMapRegion1->data);
	T *warpedDMR2Ptr = static_cast<T *>(this->warpedDistanceMapRegion2->data);
	//are images 3D?
	if (this->region1DeformationFieldImage->nz > 1)
	{
		//extra pointers required for 3D
		region1DFPtrZ = &region1DFPtrY[numVox];
		region2DFPtrZ = &region2DFPtrY[numVox];
		combinedDFPtrZ = &combinedDFPtrY[numVox];
	}

	//loop over voxels
	for (size_t n = 0; n < numVox; n++)
	{
		//check in mask
		if (this->currentMask[n] > -1)
		{
			//warped distance maps (WDMs) will contain NaN values if the transform
			//maps the voxel outside the extent of the distance map so need to check
			//for NaN values
			if (warpedDMR1Ptr[n] != warpedDMR1Ptr[n])
			{
				if (warpedDMR2Ptr[n] != warpedDMR2Ptr[n])
				{
					//both WDMs are NaN so set combined def field to NaN
					combinedDFPtrX[n] = std::numeric_limits<T>::quiet_NaN();
					combinedDFPtrY[n] = std::numeric_limits<T>::quiet_NaN();
					if (combinedDFPtrZ != NULL)
						combinedDFPtrZ[n] = std::numeric_limits<T>::quiet_NaN();
				}
				else
				{
					//check if region2 transform maps into region1, i.e. if region2 WDM < 0
					if (warpedDMR2Ptr[n] < 0)
					{
						//set combined def field to NaN
						combinedDFPtrX[n] = std::numeric_limits<T>::quiet_NaN();
						combinedDFPtrY[n] = std::numeric_limits<T>::quiet_NaN();
						if (combinedDFPtrZ != NULL)
							combinedDFPtrZ[n] = std::numeric_limits<T>::quiet_NaN();
					}
					else
					{
						//set combined def field to region2 def field
						combinedDFPtrX[n] = region2DFPtrX[n];
						combinedDFPtrY[n] = region2DFPtrY[n];
						if (combinedDFPtrZ != NULL)
							combinedDFPtrZ[n] = region2DFPtrZ[n];
					}
				}
			}//if (warpedDMR1Ptr[n] != warpedDMR1Ptr[n])
			else
			{
				//region1 WDM is not NaN, but still need to check region2 WDM
				if (warpedDMR2Ptr[n] != warpedDMR2Ptr[n])
				{
					//region2 WDM is NaN so check if region1 transform maps into region2, i.e. if region1 WDM >= 0
					if (warpedDMR1Ptr[n] >= 0)
					{
						//set combined def field to NaN
						combinedDFPtrX[n] = std::numeric_limits<T>::quiet_NaN();
						combinedDFPtrY[n] = std::numeric_limits<T>::quiet_NaN();
						if (combinedDFPtrZ != NULL)
							combinedDFPtrZ[n] = std::numeric_limits<T>::quiet_NaN();
					}
					else
					{
						//set combined def field to region1 def field
						combinedDFPtrX[n] = region1DFPtrX[n];
						combinedDFPtrY[n] = region1DFPtrY[n];
						if (combinedDFPtrZ != NULL)
							combinedDFPtrZ[n] = region1DFPtrZ[n];
					}
				}
				else
				{
					//region1 WDM and region2 WDM are both not NaN
					//so if sum of WDMs < 0 set combined def field to region1 def field
					//if >= 0 set combined def field to region2 def field
					if ((warpedDMR1Ptr[n] + warpedDMR2Ptr[n]) < 0)
					{
						combinedDFPtrX[n] = region1DFPtrX[n];
						combinedDFPtrY[n] = region1DFPtrY[n];
						if (combinedDFPtrZ != NULL)
							combinedDFPtrZ[n] = region1DFPtrZ[n];
					}
					else
					{
						combinedDFPtrX[n] = region2DFPtrX[n];
						combinedDFPtrY[n] = region2DFPtrY[n];
						if (combinedDFPtrZ != NULL)
							combinedDFPtrZ[n] = region2DFPtrZ[n];
					}
				}//else (warpedDMR2Ptr[n] != warpedDMR2Ptr[n])
			}//else (warpedDMR1Ptr[n] != warpedDMR1Ptr[n])
		}//if (this->currentMask[n] > -1)
		//not in mask so set combined def field to NaN
	}//for (size_t n = 0; n < numVox; n++)


#ifndef NDEBUG
	reg_print_fct_debug("reg_f3d_sli<T>::GetDeformationField()");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d_sli<T>::GetSimilarityMeasureGradient()
{
	//get voxel-based similairty gradient
	this->GetVoxelBasedGradient();

	//The voxel based gradient images for each region are filled with zeros
	reg_tools_multiplyValueToImage(this->region1VoxelBasedMeasureGradientImage,
		this->region1VoxelBasedMeasureGradientImage,
		0.f);
	reg_tools_multiplyValueToImage(this->region2VoxelBasedMeasureGradientImage,
		this->region2VoxelBasedMeasureGradientImage,
		0.f);

	//pointers to warped distance maps
	T *warpedDMR1Ptr = static_cast<T *>(this->warpedDistanceMapRegion1->data);
	T *warpedDMR2Ptr = static_cast<T *>(this->warpedDistanceMapRegion2->data);
	//pointers to voxel-based similarity gradients
	T *combinedVBMGPtr = static_cast<T *>(this->voxelBasedMeasureGradient->data);
	T *region1VBMGPtr = static_cast<T *>(this->region1VoxelBasedMeasureGradientImage->data);
	T *region2VBMGPtr = static_cast<T *>(this->region2VoxelBasedMeasureGradientImage->data);

	//loop over voxels and split voxel-based gradient between two regions
	//based on warped distance maps (WDMs).
	//Note - GetDeformationField() will be called before this method, so
	//WDMs will have already been calculated
	size_t numVox = this->voxelBasedMeasureGradient->nx *
		this->voxelBasedMeasureGradient->ny *
		this->voxelBasedMeasureGradient->nz;
	for (size_t n = 0; n < numVox; n++)
	{
		//is in mask?
		if (this->currentMask[n] > -1)
		{
			//need to check for NaNs in WDMs
			//if WDM1 is NaN and WDM2 >= 0 (indicating region2 transform maps into region 2)
			//then copy voxel-based gradient value in to region2VoxelBasedMeasureGradientImage
			if (warpedDMR1Ptr[n] != warpedDMR1Ptr[n] && warpedDMR2Ptr[n] >= 0)
			{
				region2VBMGPtr[n] = combinedVBMGPtr[n];
			}
			//if WDM2 is NaN and WDM1 < 0 (indicating region1 transform maps into region 1)
			//then copy voxel-based gradient value in to region1VoxelBasedMeasureGradientImage
			if (warpedDMR2Ptr[n] != warpedDMR2Ptr[n] && warpedDMR1Ptr[n] < 0)
			{
				region1VBMGPtr[n] = combinedVBMGPtr[n];
			}
			//if both WDMs are not NaN then assign voxel-based gradient value to correct region
			//based on WDMs
			if (warpedDMR1Ptr[n] == warpedDMR1Ptr[n] && warpedDMR2Ptr[n] == warpedDMR2Ptr[n])
			{
				//if sum of WDMs < 0 assign value to region 1, else assign to region 2
				if (warpedDMR1Ptr[n] + warpedDMR2Ptr[n] < 0)
					region1VBMGPtr[n] = combinedVBMGPtr[n];
				else
					region2VBMGPtr[n] = combinedVBMGPtr[n];
			}
		}
	}


	//convert voxel-based gradienta to CPG gradients for both regions
	
	//first convolve voxel-based gardient with a spline kernel
	int kernel_type = CUBIC_SPLINE_KERNEL;
	// Convolution along the x axis
	float currentNodeSpacing[3];
	currentNodeSpacing[0] = currentNodeSpacing[1] = currentNodeSpacing[2] = this->controlPointGrid->dx;
	bool activeAxis[3] = { 1, 0, 0 };
	reg_tools_kernelConvolution(this->region1VoxelBasedMeasureGradientImage,
		currentNodeSpacing,
		kernel_type,
		NULL, // mask
		NULL, // all volumes are considered as active
		activeAxis);
	reg_tools_kernelConvolution(this->region2VoxelBasedMeasureGradientImage,
		currentNodeSpacing,
		kernel_type,
		NULL, // mask
		NULL, // all volumes are considered as active
		activeAxis);
	// Convolution along the y axis
	currentNodeSpacing[0] = currentNodeSpacing[1] = currentNodeSpacing[2] = this->controlPointGrid->dy;
	activeAxis[0] = 0;
	activeAxis[1] = 1;
	reg_tools_kernelConvolution(this->region1VoxelBasedMeasureGradientImage,
		currentNodeSpacing,
		kernel_type,
		NULL, // mask
		NULL, // all volumes are considered as active
		activeAxis);
	reg_tools_kernelConvolution(this->region2VoxelBasedMeasureGradientImage,
		currentNodeSpacing,
		kernel_type,
		NULL, // mask
		NULL, // all volumes are considered as active
		activeAxis);
	// Convolution along the z axis if required
	if (this->region1VoxelBasedMeasureGradientImage->nz > 1)
	{
		currentNodeSpacing[0] = currentNodeSpacing[1] = currentNodeSpacing[2] = this->controlPointGrid->dz;
		activeAxis[1] = 0;
		activeAxis[2] = 1;
		reg_tools_kernelConvolution(this->region1VoxelBasedMeasureGradientImage,
			currentNodeSpacing,
			kernel_type,
			NULL, // mask
			NULL, // all volumes are considered as active
			activeAxis);
		reg_tools_kernelConvolution(this->region2VoxelBasedMeasureGradientImage,
			currentNodeSpacing,
			kernel_type,
			NULL, // mask
			NULL, // all volumes are considered as active
			activeAxis);
	}
	//now resample voxel-based gradients at control points to get transformationGradients
	//the gradients need to be reorientated to account for the transformation from distance
	//map image coordinates to world coordinates
	mat44 reorientation;
	if (this->currentFloating->sform_code>0)
		reorientation = this->currentFloating->sto_ijk;
	else reorientation = this->currentFloating->qto_ijk;
	reg_voxelCentric2NodeCentric(this->transformationGradient,
		this->region1VoxelBasedMeasureGradientImage,
		this->similarityWeight,
		false, // no update
		&reorientation);
	reg_voxelCentric2NodeCentric(this->region2TransformationGradient,
		this->region2VoxelBasedMeasureGradientImage,
		this->similarityWeight,
		false, // no update
		&reorientation);

	
#ifndef NDEBUG
	reg_print_fct_debug("reg_f3d_sli<T>::GetSimilarityMeasureGradient()");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
double reg_f3d_sli<T>::ComputeGapOverlapPenaltyTerm()
{
	//NOTE - this method assumes the current warped distance maps (WDMs) have already
	//been calculated by calling the GetDeformationField() method prior to calling this
	//method. The GetDeformtionField method will usually be called when warping the image
	//to calculate the image similarities, so this prevents re-calculating the WDMs
	//unnecessarily, but if the image similarities all have a weight of 0 and therefore
	//the warped image is not calculated, the GetDeformationField() method must still be
	//called.

	//NOTE2 - the gap-overlap penalty term is calculate at all voxels within the reference
	//image, even if they are outside the mask or have a NaN value in the reference or
	//warped image - this is to ensure the transformations for the 2 regions are free of
	//gaps and overlaps, even in areas where the images are not being used to drive the
	//registration

	if (this->gapOverlapWeight <= 0)
		return 0.;

	//loop over all voxels and sum up gap-overlap penalty term values from each voxel.
	//the gap-overlap penalty term is defined as -WDM1*WDM2 if WDM1*WDM2<0 (i.e. the
	//WDMs point to different regions, indicating a gap or overlap), and 0 otherwise
	double gapOverlapTotal = 0.;
	double gapOverlapValue = 0.;
	
	//pointers to warped distance maps
	T *warpedDMR1Ptr = static_cast<T *>(this->warpedDistanceMapRegion1->data);
	T *warpedDMR2Ptr = static_cast<T *>(this->warpedDistanceMapRegion2->data);

	size_t numVox = this->warpedDistanceMapRegion1->nx *
		this->warpedDistanceMapRegion1->ny *
		this->warpedDistanceMapRegion1->nz;
	for (size_t n = 0; n < numVox; n++)
	{
		gapOverlapValue = warpedDMR1Ptr[n] * warpedDMR2Ptr[n];
		//if NaN value in either WDM then gapOverlapValue = NaN, so will fail
		//test for less than 0
		if (gapOverlapValue < 0)
			gapOverlapTotal -= gapOverlapValue;
	}

	//normalise by the number of voxels and return weighted value
	gapOverlapTotal /= double(numVox);
	return double(this->gapOverlapWeight) * gapOverlapTotal;

#ifndef NDEBUG
	reg_print_fct_debug("reg_f3d_sli<T>::ComputeGapOverlapPenaltyTerm()");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d_sli<T>::GetGapOverlapGradient()
{
	//NOTE - this method assumes the deformation fields and the WDMs for each region
	//have already been calculated by calling the GetDeformationField() method prior
	//to calling this method.

	//first calculate gap-overlap gradient with respect to def field for each region
	//then convolve these with a B-spline kernal to get the gap-overlap gradient with
	//respect to the transform (i.e. the CPG) for each region
	//
	//the gap-overlap gradient with respect to the def field for region 1 is:
	//dGO/dDF1 = -WDM2*(dWDM1/dDF1) if WDM1*WDM2<0 else 0
	//where dWMD1/dDF1 is the spatial gradient of the distance map warped by the def
	//field for region 1
	//
	//the gap-overlap gradient with respect to the def field for region 2 is:
	//dGO/dDF2 = -WDM1*(dWDM2/dDF2) if WDM1*WDM2<0 else 0
	//where dWMD2/dDF2 is the spatial gradient of the distance map warped by the def
	//field for region 2

	//The gap-overlap gradients WRT the def fields for each region are filled with zeros
	reg_tools_multiplyValueToImage(this->gapOverlapGradientWRTDefFieldRegion1,
		this->gapOverlapGradientWRTDefFieldRegion1,
		0.f);
	reg_tools_multiplyValueToImage(this->gapOverlapGradientWRTDefFieldRegion2,
		this->gapOverlapGradientWRTDefFieldRegion2,
		0.f);

	//calculate the spatial gradient of the distance map warped by the def field from
	//each region
	reg_getImageGradient(this->currentDistanceMap,
		this->warpedDistanceMapGradientRegion1,
		this->region1DeformationFieldImage,
		this->currentMask,
		this->interpolation,
		this->warpedPaddingValue,
		0);//timepoint 0
	reg_getImageGradient(this->currentDistanceMap,
		this->warpedDistanceMapGradientRegion2,
		this->region2DeformationFieldImage,
		this->currentMask,
		this->interpolation,
		this->warpedPaddingValue,
		0);//timepoint 0

	//pointers to warped distance maps
	T *warpedDMR1Ptr = static_cast<T *>(this->warpedDistanceMapRegion1->data);
	T *warpedDMR2Ptr = static_cast<T *>(this->warpedDistanceMapRegion2->data);
	//pointers to warped spatial gradients
	size_t numVox = this->warpedDistanceMapRegion1->nx *
		this->warpedDistanceMapRegion1->ny *
		this->warpedDistanceMapRegion1->nz;
	T *warpedDMGradR1PtrX = static_cast<T *>(this->warpedDistanceMapGradientRegion1->data);
	T *warpedDMGradR1PtrY = &warpedDMGradR1PtrX[numVox];
	T *warpedDMGradR1PtrZ = NULL;
	T *warpedDMGradR2PtrX = static_cast<T *>(this->warpedDistanceMapGradientRegion2->data);
	T *warpedDMGradR2PtrY = &warpedDMGradR2PtrX[numVox];
	T *warpedDMGradR2PtrZ = NULL;
	//pointers to the gap-overlap gradients WRT def field for each region
	T *gapOverlapGradR1PtrX = static_cast<T *>(this->gapOverlapGradientWRTDefFieldRegion1->data);
	T *gapOverlapGradR1PtrY = &gapOverlapGradR1PtrX[numVox];
	T *gapOverlapGradR1PtrZ = NULL;
	T *gapOverlapGradR2PtrX = static_cast<T *>(this->gapOverlapGradientWRTDefFieldRegion2->data);
	T *gapOverlapGradR2PtrY = &gapOverlapGradR2PtrX[numVox];
	T *gapOverlapGradR2PtrZ = NULL;
	//check for 3D
	if (this->warpedDistanceMapGradientRegion1->nz > 1)
	{
		warpedDMGradR1PtrZ = &warpedDMGradR1PtrY[numVox];
		warpedDMGradR2PtrZ = &warpedDMGradR2PtrY[numVox];
		gapOverlapGradR1PtrZ = &gapOverlapGradR1PtrY[numVox];
		gapOverlapGradR2PtrZ = &gapOverlapGradR2PtrY[numVox];
	}

	//loop over all voxels and calculate gap-overlap gradient with respect to def field
	//for each region
	for (size_t n = 0; n < numVox; n++)
	{
		if (warpedDMR1Ptr[n] * warpedDMR2Ptr[n] < 0)
		{
			//dGO / dDF1 = -WDM2*(dWDM1 / dDF1)
			gapOverlapGradR1PtrX[n] = warpedDMR2Ptr[n] * warpedDMGradR1PtrX[n];
			gapOverlapGradR1PtrY[n] = warpedDMR2Ptr[n] * warpedDMGradR1PtrY[n];
			//dGO / dDF2 = -WDM1*(dWDM2 / dDF2)
			gapOverlapGradR2PtrX[n] = warpedDMR1Ptr[n] * warpedDMGradR2PtrX[n];
			gapOverlapGradR2PtrY[n] = warpedDMR1Ptr[n] * warpedDMGradR2PtrY[n];
			//check for 3D
			if (gapOverlapGradR1PtrZ != NULL)
			{
				gapOverlapGradR1PtrZ[n] = warpedDMR2Ptr[n] * warpedDMGradR1PtrZ[n];
				gapOverlapGradR2PtrZ[n] = warpedDMR1Ptr[n] * warpedDMGradR2PtrZ[n];
			}
		}//if (warpedDMR1Ptr[n] * warpedDMR2Ptr[n])
	}//for (size_t n = 0; n < numVox; n++)

	//the gap-overlap gradient WRT the def field is convolved with a B-spline kernel
	//to calculate the gradient WRT the CPG for each region
	int kernel_type = CUBIC_SPLINE_KERNEL;
	// Convolution along the x axis
	float currentNodeSpacing[3];
	currentNodeSpacing[0] = currentNodeSpacing[1] = currentNodeSpacing[2] = this->controlPointGrid->dx;
	bool activeAxis[3] = { 1, 0, 0 };
	reg_tools_kernelConvolution(this->gapOverlapGradientWRTDefFieldRegion1,
		currentNodeSpacing,
		kernel_type,
		NULL, // mask
		NULL, // all volumes are considered as active
		activeAxis);
	reg_tools_kernelConvolution(this->gapOverlapGradientWRTDefFieldRegion2,
		currentNodeSpacing,
		kernel_type,
		NULL, // mask
		NULL, // all volumes are considered as active
		activeAxis);
	// Convolution along the y axis
	currentNodeSpacing[0] = currentNodeSpacing[1] = currentNodeSpacing[2] = this->controlPointGrid->dy;
	activeAxis[0] = 0;
	activeAxis[1] = 1;
	reg_tools_kernelConvolution(this->gapOverlapGradientWRTDefFieldRegion1,
		currentNodeSpacing,
		kernel_type,
		NULL, // mask
		NULL, // all volumes are considered as active
		activeAxis);
	reg_tools_kernelConvolution(this->gapOverlapGradientWRTDefFieldRegion2,
		currentNodeSpacing,
		kernel_type,
		NULL, // mask
		NULL, // all volumes are considered as active
		activeAxis);
	// Convolution along the z axis if required
	if (this->gapOverlapGradientWRTDefFieldRegion1->nz > 1)
	{
		currentNodeSpacing[0] = currentNodeSpacing[1] = currentNodeSpacing[2] = this->controlPointGrid->dz;
		activeAxis[1] = 0;
		activeAxis[2] = 1;
		reg_tools_kernelConvolution(this->gapOverlapGradientWRTDefFieldRegion1,
			currentNodeSpacing,
			kernel_type,
			NULL, // mask
			NULL, // all volumes are considered as active
			activeAxis);
		reg_tools_kernelConvolution(this->gapOverlapGradientWRTDefFieldRegion2,
			currentNodeSpacing,
			kernel_type,
			NULL, // mask
			NULL, // all volumes are considered as active
			activeAxis);
	}

	//the voxel-wise gradients are now resampled at the CPG locations and added to the 
	//transformation gradients for each region
	//the gradients need to be reorientated to account for the transformation from distance
	//map image coordinates to world coordinates
	mat44 reorientation;
	if (this->currentDistanceMap->sform_code>0)
		reorientation = this->currentDistanceMap->sto_ijk;
	else reorientation = this->currentDistanceMap->qto_ijk;
	reg_voxelCentric2NodeCentric(this->transformationGradient,
		this->gapOverlapGradientWRTDefFieldRegion1,
		this->gapOverlapWeight,
		true, // update the transformation gradient
		&reorientation);
	reg_voxelCentric2NodeCentric(this->region2TransformationGradient,
		this->gapOverlapGradientWRTDefFieldRegion2,
		this->gapOverlapWeight,
		true, // update the transformation gradient
		&reorientation);

}
/* *************************************************************** */
/* *************************************************************** */
template class reg_f3d_sli<float>;
#endif
