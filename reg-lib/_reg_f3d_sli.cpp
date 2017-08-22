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
	//pointers to deformation fields
	T *region1DFPtrX = static_cast<T *>(this->region1DeformationFieldImage->data);
	T *region1DFPtrY = &region1DFPtrX[numVox];
	T *region2DFPtrX = static_cast<T *>(this->region2DeformationFieldImage->data);
	T *region2DFPtrY = &region2DFPtrX[numVox];
	T *combinedDFPtrX = static_cast<T *>(this->deformationFieldImage->data);
	T *combinedDFPtrY = &combinedDFPtrX[numVox];
	//pointers to warped distance maps
	T *warpedDMR1Ptr = static_cast<T *>(this->warpedDistanceMapRegion1->data);
	T *warpedDMR2Ptr = static_cast<T *>(this->warpedDistanceMapRegion2->data);
	//are images 3D?
	if (this->region1DeformationFieldImage->nz > 1)
	{
		//extra pointers required for 3D
		T *region1DFPtrZ = &region1DFPtrY[numVox];
		T *region2DFPtrZ = &region2DFPtrY[numVox];
		T *combinedDFPtrZ = &combinedDFPtrY[numVox];

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
							combinedDFPtrZ[n] = std::numeric_limits<T>::quiet_NaN();
						}
						else
						{
							//set combined def field to region2 def field
							combinedDFPtrX[n] = region2DFPtrX[n];
							combinedDFPtrY[n] = region2DFPtrY[n];
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
							combinedDFPtrZ[n] = std::numeric_limits<T>::quiet_NaN();
						}
						else
						{
							//set combined def field to region1 def field
							combinedDFPtrX[n] = region1DFPtrX[n];
							combinedDFPtrY[n] = region1DFPtrY[n];
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
							combinedDFPtrZ[n] = region1DFPtrZ[n];
						}
						else
						{
							combinedDFPtrX[n] = region2DFPtrX[n];
							combinedDFPtrY[n] = region2DFPtrY[n];
							combinedDFPtrZ[n] = region2DFPtrZ[n];
						}
					}//else (warpedDMR2Ptr[n] != warpedDMR2Ptr[n])
				}//else (warpedDMR1Ptr[n] != warpedDMR1Ptr[n])
			}//if (this->currentMask[n] > -1)
			//not in mask so set combined def field to NaN
			else
			{
				combinedDFPtrX[n] = std::numeric_limits<T>::quiet_NaN();
				combinedDFPtrY[n] = std::numeric_limits<T>::quiet_NaN();
				combinedDFPtrZ[n] = std::numeric_limits<T>::quiet_NaN();
			}
		}//for (size_t n = 0; n < numVox; n++)
	}//if (this->region1DeformationFieldImage->nz > 1)
	else
	{
		//images are 2D
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
					}
					else
					{
						//check if region2 transform maps into region1, i.e. if region2 WDM < 0
						if (warpedDMR2Ptr[n] < 0)
						{
							//set combined def field to NaN
							combinedDFPtrX[n] = std::numeric_limits<T>::quiet_NaN();
							combinedDFPtrY[n] = std::numeric_limits<T>::quiet_NaN();
						}
						else
						{
							//set combined def field to region2 def field
							combinedDFPtrX[n] = region2DFPtrX[n];
							combinedDFPtrY[n] = region2DFPtrY[n];
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
						}
						else
						{
							//set combined def field to region1 def field
							combinedDFPtrX[n] = region1DFPtrX[n];
							combinedDFPtrY[n] = region1DFPtrY[n];
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
						}
						else
						{
							combinedDFPtrX[n] = region2DFPtrX[n];
							combinedDFPtrY[n] = region2DFPtrY[n];
						}
					}//else (warpedDMR2Ptr[n] != warpedDMR2Ptr[n])
				}//else (warpedDMR1Ptr[n] != warpedDMR1Ptr[n])
			}//if (this->currentMask[n] > -1)
			//not in mask so set combined def field to NaN
			else
			{
				combinedDFPtrX[n] = std::numeric_limits<T>::quiet_NaN();
				combinedDFPtrY[n] = std::numeric_limits<T>::quiet_NaN();
			}
		}//for (size_t n = 0; n < numVox; n++)
	}//else (this->region1DeformationFieldImage->nz > 1)

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


	//convert region 1 voxel-based gradient to CPG gradient
	
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
		activeAxis
		);
	// Convolution along the y axis
	currentNodeSpacing[0] = currentNodeSpacing[1] = currentNodeSpacing[2] = this->controlPointGrid->dy;
	activeAxis[0] = 0;
	activeAxis[1] = 1;
	reg_tools_kernelConvolution(this->region1VoxelBasedMeasureGradientImage,
		currentNodeSpacing,
		kernel_type,
		NULL, // mask
		NULL, // all volumes are considered as active
		activeAxis
		);
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
			activeAxis
			);
	}
	//now resample voxel-based gradient at control points to get transformationGradient
	mat44 reorientation;
	if (this->currentFloating->sform_code>0)
		reorientation = this->currentFloating->sto_ijk;
	else reorientation = this->currentFloating->qto_ijk;
	reg_voxelCentric2NodeCentric(this->transformationGradient,
		this->region1VoxelBasedMeasureGradientImage,
		this->similarityWeight,
		false, // no update
		&reorientation
		);


	//convert region 2 voxel=based gradient to CPG gradient

	//first convolve voxel-based gardient with a spline kernel
	// Convolution along the x axis
	currentNodeSpacing[0] = currentNodeSpacing[1] = currentNodeSpacing[2] = this->region2ControlPointGrid->dx;
	activeAxis[0] = 1;
	activeAxis[2] = 0;
	reg_tools_kernelConvolution(this->region2VoxelBasedMeasureGradientImage,
		currentNodeSpacing,
		kernel_type,
		NULL, // mask
		NULL, // all volumes are considered as active
		activeAxis
		);
	// Convolution along the y axis
	currentNodeSpacing[0] = currentNodeSpacing[1] = currentNodeSpacing[2] = this->region2ControlPointGrid->dy;
	activeAxis[0] = 0;
	activeAxis[1] = 1;
	reg_tools_kernelConvolution(this->region2VoxelBasedMeasureGradientImage,
		currentNodeSpacing,
		kernel_type,
		NULL, // mask
		NULL, // all volumes are considered as active
		activeAxis
		);
	// Convolution along the z axis if required
	if (this->region2VoxelBasedMeasureGradientImage->nz > 1)
	{
		currentNodeSpacing[0] = currentNodeSpacing[1] = currentNodeSpacing[2] = this->region2ControlPointGrid->dz;
		activeAxis[1] = 0;
		activeAxis[2] = 1;
		reg_tools_kernelConvolution(this->region2VoxelBasedMeasureGradientImage,
			currentNodeSpacing,
			kernel_type,
			NULL, // mask
			NULL, // all volumes are considered as active
			activeAxis
			);
	}
	//now resample voxel-based gradient at control points to get transformationGradient
	if (this->currentFloating->sform_code>0)
		reorientation = this->currentFloating->sto_ijk;
	else reorientation = this->currentFloating->qto_ijk;
	reg_voxelCentric2NodeCentric(this->region2TransformationGradient,
		this->region2VoxelBasedMeasureGradientImage,
		this->similarityWeight,
		false, // no update
		&reorientation
		);

	
#ifndef NDEBUG
	reg_print_fct_debug("reg_f3d_sli<T>::GetSimilarityMeasureGradient");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template class reg_f3d_sli<float>;
#endif
