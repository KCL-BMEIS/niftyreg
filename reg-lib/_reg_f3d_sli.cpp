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
	this->executableName = (char *)"NiftyReg F3D Sliding regions";

	this->inputRegion2ControlPointGrid = NULL;
	this->region2ControlPointGrid = NULL;
	this->region2DeformationFieldImage = NULL;
	this->region2VoxelBasedMeasureGradientImage = NULL;
	this->region2TransformationGradient = NULL;

	this->region1DeformationFieldImage = NULL;
	this->region1VoxelBasedMeasureGradientImage = NULL;

	this->inputDistanceMap = NULL;
	this->distanceMapPyramid = NULL;
	this->currentDistanceMap = NULL;

	this->warpedDistanceMapRegion1 = NULL;
	this->warpedDistanceMapRegion2 = NULL;
	this->warpedDistanceMapGradientRegion1 = NULL;
	this->warpedDistanceMapGradientRegion2 = NULL;

	this->gapOverlapGradientWRTDefFieldRegion1 = NULL;
	this->gapOverlapGradientWRTDefFieldRegion2 = NULL;

	this->gapOverlapWeight = 0.1;
	this->currentWGO = 0;
	this->bestWGO = 0;

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
void reg_f3d_sli<T>::AllocateDeformationField()
{
	//clear any previously allocated deformation fields
	this->ClearDeformationField();

	//call method from reg_base to allocate combined deformation field
	reg_base<T>::AllocateDeformationField();

	//now allocate def fields for regions 1 and 2 using header info from combined def field
	this->region1DeformationFieldImage = nifti_copy_nim_info(this->deformationFieldImage);
	this->region1DeformationFieldImage->data = (void *)calloc(this->region1DeformationFieldImage->nvox,
		this->region1DeformationFieldImage->nbyper);
	this->region1DeformationFieldImage = nifti_copy_nim_info(this->deformationFieldImage);
	this->region2DeformationFieldImage->data = (void *)calloc(this->region2DeformationFieldImage->nvox,
		this->region2DeformationFieldImage->nbyper);

#ifndef NDEBUG
	reg_print_fct_debug("reg_f3d_sli<T>::AllocateDeformationField");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d_sli<T>::ClearDeformationField()
{
	//call method from reg_base to clear combined def field
	reg_base<T>::ClearDeformationField();

	//now clear def fields for regions 1 and 2
	if (this->region1DeformationFieldImage != NULL)
	{
		nifti_image_free(this->region1DeformationFieldImage);
		this->region1DeformationFieldImage == NULL;
	}
	if (this->region2DeformationFieldImage != NULL)
	{
		nifti_image_free(this->region2DeformationFieldImage);
		this->region2DeformationFieldImage == NULL;
	}

#ifndef NDEBUG
	reg_print_fct_debug("reg_f3d_sli<T>::ClearDeformationField");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d_sli<T>::AllocateWarped()
{
	//clear any previously allocated warped images
	this->ClearWarped();

	//call method from reg_base to allocate warped floating image
	reg_base<T>::AllocateWarped();

	//Allocate warped distance maps for region 1 and region 2
	//use header info from warped image, but update some info using header from current distance map
	this->warpedDistanceMapRegion1 = nifti_copy_nim_info(this->warped);
	this->warpedDistanceMapRegion2 = nifti_copy_nim_info(this->warped);
	this->warpedDistanceMapRegion1->dim[0] = this->warpedDistanceMapRegion1->ndim =
		this->warpedDistanceMapRegion2->dim[0] = this->warpedDistanceMapRegion2->ndim =
		this->currentDistanceMap->ndim;
	this->warpedDistanceMapRegion1->dim[4] = this->warpedDistanceMapRegion1->nt =
		this->warpedDistanceMapRegion2->dim[4] = this->warpedDistanceMapRegion2->nt =
		this->currentDistanceMap->nt;
	this->warpedDistanceMapRegion1->nvox = this->warpedDistanceMapRegion2->nvox =
		this->warpedDistanceMapRegion1->nx *
		this->warpedDistanceMapRegion1->ny *
		this->warpedDistanceMapRegion1->nz *
		this->warpedDistanceMapRegion1->nt;
	this->warpedDistanceMapRegion1->datatype = this->warpedDistanceMapRegion2->datatype = this->currentDistanceMap->datatype;
	this->warpedDistanceMapRegion1->nbyper = this->warpedDistanceMapRegion2->nbyper = this->currentDistanceMap->nbyper;
	//now allocate memory for warped distance maps data
	this->warpedDistanceMapRegion1->data = (void *)calloc(this->warpedDistanceMapRegion1->nvox, this->warpedDistanceMapRegion1->nbyper);
	this->warpedDistanceMapRegion2->data = (void *)calloc(this->warpedDistanceMapRegion2->nvox, this->warpedDistanceMapRegion2->nbyper);

#ifndef NDEBUG
	reg_print_fct_debug("reg_f3d_sli<T>::AllocateWarped");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d_sli<T>::ClearWarped()
{
	//call method from reg_base to clear warped floating image
	reg_base<T>::ClearWarped();

	//now clear warped distance maps
	if (this->warpedDistanceMapRegion1 != NULL)
	{
		nifti_image_free(this->warpedDistanceMapRegion1);
		this->warpedDistanceMapRegion1 = NULL;
	}
	if (this->warpedDistanceMapRegion2 != NULL)
	{
		nifti_image_free(this->warpedDistanceMapRegion2);
		this->warpedDistanceMapRegion2 = NULL;
	}

#ifndef NDEBUG
	reg_print_fct_debug("reg_f3d_sli<T>::ClearWarped");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
double reg_f3d_sli<T>::GetObjectiveFunctionValue()
{
	//call method from reg_f3d to calculate objective function value for similarity
	//measure and other penalty terms
	double objFuncValue = reg_f3d<T>::GetObjectiveFunctionValue();

	//calculate weighted gap-overlap penalty term
	this->currentWGO = this->ComputeGapOverlapPenaltyTerm();

#ifndef NDEBUG
	char text[255];
	sprintf(text, " | (wGO) %g", this->currentWGO);
	reg_print_msg_debug(text);
	reg_print_fct_debug("reg_f3d<T>::GetObjectiveFunctionValue");
#endif

	//return objective function value with weighted gap-overlap value subtracted
	return objFuncValue - this->currentWGO;
}
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

	//NOTE2 - the gap-overlap penalty term is calculated at all voxels within the reference
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
double reg_f3d_sli <T>::ComputeBendingEnergyPenaltyTerm()
{
	//check if penalty term used, i.e. weight > 0
	if (this->bendingEnergyWeight <= 0) return 0.;

	//calculate the bending energy penalty term for region 1
	double region1PenaltyTerm = reg_f3d<T>::ComputeBendingEnergyPenaltyTerm();

	//calculate the bending energy penalty term for region 2
	double region2PenaltyTerm = this->bendingEnergyWeight * reg_spline_approxBendingEnergy(this->region2ControlPointGrid);
#ifndef NDEBUG
	reg_print_fct_debug("reg_f3d_sli<T>::ComputeBendingEnergyPenaltyTerm");
#endif
	return region1PenaltyTerm + region2PenaltyTerm;
}
/* *************************************************************** */
template <class T>
double reg_f3d_sli<T>::ComputeLinearEnergyPenaltyTerm()
{
	//check if penalty term used, i.e. weight > 0
	if (this->linearEnergyWeight <= 0) return 0.;

	//calculate the linear energy penalty term for region 1
	double region1PenaltyTerm = reg_f3d<T>::ComputeLinearEnergyPenaltyTerm();

	//calculate the bending energy penalty term for region 2
	double region2PenaltyTerm = this->linearEnergyWeight * reg_spline_approxLinearEnergy(this->region2ControlPointGrid);

#ifndef NDEBUG
	reg_print_fct_debug("reg_f3d_sli<T>::ComputeLinearEnergyPenaltyTerm");
#endif
	return region1PenaltyTerm + region2PenaltyTerm;
}
/* *************************************************************** */
template <class T>
double reg_f3d_sli<T>::ComputeJacobianBasedPenaltyTerm(int type)
{
	//check if penalty term used, i.e. weight > 0
	if (this->jacobianLogWeight <= 0) return 0.;

	//Jacobian penalty term not currently implemented for sliding region registrations
	//so throw error
	reg_print_fct_error("reg_f3d_sli<T>::ComputeJacobianBasedPenaltyTerm");
	reg_print_msg_error("Jacobian penalty term not currently implemented for sliding region registrations");
	reg_exit();
}
/* *************************************************************** */
template <class T>
double reg_f3d_sli<T>::ComputeLandmarkDistancePenaltyTerm()
{
	//check if penalty term used, i.e. weight > 0
	if (this->landmarkRegWeight <= 0) return 0.;

	//Landmark penalty term not currently implemented for sliding region registrations
	//so throw error
	reg_print_fct_error("reg_f3d_sli<T>::ComputeLandmarkDistancePenaltyTerm");
	reg_print_msg_error("Landmark distance penalty term not currently implemented for sliding region registrations");
	reg_exit();
}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
void reg_f3d_sli<T>::GetObjectiveFunctionGradient()
{
	//note - cannot call method from reg_f3d as objective function gradient will
	//be smoothed before the gap-overlap gradient is added to it, so need to
	//reproduce code here

	//check if gradient is approximated
	if (!this->useApproxGradient)
	{
		// Compute the gradient of the similarity measure
		if (this->similarityWeight>0)
		{
			this->WarpFloatingImage(this->interpolation);
			this->GetSimilarityMeasureGradient();
		}
		else
		{
			this->SetGradientImageToZero();
		}
		// Compute the penalty term gradients if required
		this->GetBendingEnergyGradient();
		this->GetJacobianBasedGradient();
		this->GetLinearEnergyGradient();
		this->GetLandmarkDistanceGradient();
		//include the gap-penalty term gradient
		this->GetGapOverlapGradient();
	}
	else
	{
		this->GetApproximatedGradient();
	}

	//increment the optimiser iteration number 
	this->optimiser->IncrementCurrentIterationNumber();

	// Smooth the gradient if required
	this->SmoothGradient();
#ifndef NDEBUG
	reg_print_fct_debug("reg_f3d_sli<T>::GetObjectiveFunctionGradient");
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
template <class T>
void reg_f3d_sli<T>::GetBendingEnergyGradient()
{
	//check if bending energy used
	if (this->bendingEnergyWeight <= 0) return;

	//calculate bending energy gradient for region 1 transform
	reg_f3d<T>::GetBendingEnergyGradient();

	//calculate bending energy gradient for region 2 transform
	reg_spline_approxBendingEnergyGradient(this->region2ControlPointGrid,
		this->region2TransformationGradient,
		this->bendingEnergyWeight);

#ifndef NDEBUG
	reg_print_fct_debug("reg_f3d_sli<T>::GetBendingEnergyGradient");
#endif
	return;
}
/* *************************************************************** */
template <class T>
void reg_f3d_sli<T>::GetLinearEnergyGradient()
{
	//check if linear energy used
	if (this->linearEnergyWeight <= 0) return;

	//calculate linear energy gradient for region 1 transform
	reg_f3d<T>::GetLinearEnergyGradient();

	//calculate linear energy gradient for region 2 transform
	reg_spline_approxLinearEnergyGradient(this->region2ControlPointGrid,
		this->region2TransformationGradient,
		this->linearEnergyWeight);

#ifndef NDEBUG
	reg_print_fct_debug("reg_f3d_sli<T>::GetLinearEnergyGradient");
#endif
	return;
}
/* *************************************************************** */
template <class T>
void reg_f3d_sli<T>::GetJacobianBasedGradient()
{
	//check if penalty term used, i.e. weight > 0
	if (this->jacobianLogWeight <= 0) return;

	//Jacobian penalty term not currently implemented for sliding region registrations
	//so throw error
	reg_print_fct_error("reg_f3d_sli<T>::GetJacobianBasedGradient");
	reg_print_msg_error("Jacobian penalty term not currently implemented for sliding region registrations");
	reg_exit();
}
/* *************************************************************** */
template <class T>
void reg_f3d_sli<T>::GetLandmarkDistanceGradient()
{
	//check if penalty term used, i.e. weight > 0
	if (this->landmarkRegWeight <= 0) return;

	//Landmark penalty term not currently implemented for sliding region registrations
	//so throw error
	reg_print_fct_error("reg_f3d_sli<T>::GetLandmarkDistanceGradient");
	reg_print_msg_error("Landmark distance penalty term not currently implemented for sliding region registrations");
	reg_exit();
}
/* *************************************************************** */
template <class T>
void reg_f3d_sli<T>::SetGradientImageToZero()
{
	//call method from reg_f3d to set region 1 gradient image to 0
	reg_f3d<T>::SetGradientImageToZero();

	//set region 2 gradient image to 0
	T* nodeGradPtr = static_cast<T *>(this->region2TransformationGradient->data);
	for (size_t i = 0; i<this->region2TransformationGradient->nvox; ++i)
		*nodeGradPtr++ = 0;

#ifndef NDEBUG
	reg_print_fct_debug("reg_f3d_sli<T>::SetGradientImageToZero");
#endif
}
/* *************************************************************** */
template <class T>
T reg_f3d_sli<T>::NormaliseGradient()
{
	// call method from reg_f3d to calculate max length of region 1 gradient image
	// note - this method does not normalise the gradient (as the executable name
	// is not "NiftyReg F3D"), it will just return the max length
	T region1MaxValue = reg_f3d<T>::NormaliseGradient();

	// The max length of the region 2 gradient image is calculated
	T maxGradValue = 0;
	size_t voxNumber = this->region2TransformationGradient->nx *
		this->region2TransformationGradient->ny *
		this->region2TransformationGradient->nz;
	//pointers to gradient data
	T *r2PtrX = static_cast<T *>(this->region2TransformationGradient->data);
	T *r2PtrY = &r2PtrX[voxNumber];
	T *r2PtrZ = NULL;
	//check for 3D
	if (this->region2TransformationGradient->nz > 1)
		r2PtrZ = &r2PtrY[voxNumber];
	//loop over voxels, calculate length of gradient vector (ignoring dimension(s) not
	//being optimised), and store value if greater than current max value
	for (size_t i = 0; i<voxNumber; i++)
	{
		T valX = 0, valY = 0, valZ = 0;
		if (this->optimiseX == true)
			valX = *r2PtrX++;
		if (this->optimiseY == true)
			valY = *r2PtrY++;
		if (r2PtrZ != NULL && this->optimiseZ == true)
			valZ = *r2PtrZ++;
		T length = (T)(sqrt(valX*valX + valY*valY + valZ*valZ));
		maxGradValue = (length > maxGradValue) ? length : maxGradValue;
	}
	
	// The largest value between the region 1 and region 2 gradients is kept
	maxGradValue = maxGradValue>region1MaxValue ? maxGradValue : region1MaxValue;
#ifndef NDEBUG
	char text[255];
	sprintf(text, "Objective function gradient maximal length: %g", maxGradValue);
	reg_print_msg_debug(text);
#endif

	// The region 1 gradient is normalised
	T *r1Ptr = static_cast<T *>(this->transformationGradient->data);
	for (size_t i = 0; i < this->transformationGradient->nvox; ++i)
	{
		*r1Ptr++ /= maxGradValue;
	}
	// The region 2 gradient is normalised
	T *r2Ptr = static_cast<T *>(this->region2TransformationGradient->data);
	for (size_t i = 0; i<this->region2TransformationGradient->nvox; ++i)
	{
		*r2Ptr++ /= maxGradValue;
	}

#ifndef NDEBUG
	reg_print_fct_debug("reg_f3d_sli<T>::NormaliseGradient");
#endif
	// Returns the largest gradient distance
	return maxGradValue;
}
/* *************************************************************** */
template <class T>
void reg_f3d_sli<T>::SmoothGradient()
{
	//check if gradients require smoothing
	if (this->gradientSmoothingSigma != 0)
	{
		//call method from reg_f3d to smooth gradient for region 1 transform
		reg_f3d<T>::SmoothGradient();
		
		//smooth the gradient for region 2 transform
		float kernel = fabs(this->gradientSmoothingSigma);
		reg_tools_kernelConvolution(this->region2TransformationGradient,
			&kernel,
			GAUSSIAN_KERNEL);
	}

#ifndef NDEBUG
	reg_print_fct_debug("reg_f3d_sli<T>::SmoothGradient");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d_sli<T>::GetApproximatedGradient()
{
	//call method from reg_f3d to approximate gradient for region 1
	reg_f3d<T>::GetApproximatedGradient();

	// approximate gradient for region 2 using finite differences
	//
	//pointers to region 2 CPG and gradient
	T *r2CPGPtr = static_cast<T *>(this->region2ControlPointGrid->data);
	T *r2GradPtr = static_cast<T *>(this->region2TransformationGradient->data);
	//amount to increase/decrease CPG values by
	//equal to floating voxel size in x dimension / 1000
	T eps = this->currentFloating->dx / 1000.f;
	//loop over CPG values
	for (size_t i = 0; i<this->region2ControlPointGrid->nvox; i++)
	{
		//get best CPG value from optimiser
		T currentValue = this->optimiser->GetBestDOF_b()[i];
		//increase the value by eps and calculate new objective function value
		r2CPGPtr[i] = currentValue + eps;
		double valPlus = this->GetObjectiveFunctionValue();
		//decrease the value by eps and calculate new objective function value
		r2CPGPtr[i] = currentValue - eps;
		double valMinus = this->GetObjectiveFunctionValue();
		//reset CPG to best value
		r2CPGPtr[i] = currentValue;
		//set the value of gradient by approximating using finite differences
		r2GradPtr[i] = -(T)((valPlus - valMinus) / (2.0*eps));
	}
#ifndef NDEBUG
	reg_print_fct_debug("reg_f3d_sli<T>::GetApproximatedGradient");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d_sli<T>::AllocateWarpedGradient()
{
	//clear any previously allocated warped gradient images
	this->ClearWarpedGradient();

	//call method from reg_base to allocate warped (floating) gradient image
	reg_base<T>::AllocateWarpedGradient();

	//allocate warped distance map gradient images using header info from
	//warped (floating) gradient image
	this->warpedDistanceMapGradientRegion1 = nifti_copy_nim_info(this->warImgGradient);
	this->warpedDistanceMapGradientRegion2 = nifti_copy_nim_info(this->warImgGradient);
	this->warpedDistanceMapGradientRegion1->data = (void *)calloc(this->warpedDistanceMapGradientRegion1->nvox,
		this->warpedDistanceMapGradientRegion1->nbyper);
	this->warpedDistanceMapGradientRegion2->data = (void *)calloc(this->warpedDistanceMapGradientRegion2->nvox,
		this->warpedDistanceMapGradientRegion2->nbyper);

#ifndef NDEBUG
	reg_print_fct_debug("reg_f3d_sli<T>::AllocateWarpedGradient");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d_sli<T>::ClearWarpedGradient()
{
	//call method from reg_base to clear warped (floating) gradient image
	reg_base<T>::ClearWarpedGradient();

	//now clear warped distance map gradient images
	if (this->warpedDistanceMapGradientRegion1 != NULL)
	{
		nifti_image_free(this->warpedDistanceMapGradientRegion1);
		this->warpedDistanceMapGradientRegion1 = NULL;
	}
	if (this->warpedDistanceMapGradientRegion2 != NULL)
	{
		nifti_image_free(this->warpedDistanceMapGradientRegion2);
		this->warpedDistanceMapGradientRegion2 = NULL;
	}

#ifndef NDEBUG
		reg_print_fct_debug("reg_f3d_sli<T>::ClearWarpedGradient");
#endif
	
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d_sli<T>::AllocateVoxelBasedMeasureGradient()
{
	//clear any previously allocated images
	this->ClearVoxelBasedMeasureGradient();

	//call method from reg_base to allocate voxel-based similarity measure gradient image
	//for combined transform
	reg_base<T>::AllocateVoxelBasedMeasureGradient();

	//allocate voxel-based similarity measure gradient images for each region
	this->region1VoxelBasedMeasureGradientImage = nifti_copy_nim_info(this->voxelBasedMeasureGradient);
	this->region2VoxelBasedMeasureGradientImage = nifti_copy_nim_info(this->voxelBasedMeasureGradient);
	this->region1VoxelBasedMeasureGradientImage->data = (void *)calloc(this->region1VoxelBasedMeasureGradientImage->nvox,
		this->region1VoxelBasedMeasureGradientImage->nbyper);
	this->region2VoxelBasedMeasureGradientImage->data = (void *)calloc(this->region2VoxelBasedMeasureGradientImage->nvox,
		this->region2VoxelBasedMeasureGradientImage->nbyper);

	//allocate voxel-based gap-overlap peanlty term gradient images
	this->gapOverlapGradientWRTDefFieldRegion1 = nifti_copy_nim_info(this->voxelBasedMeasureGradient);
	this->gapOverlapGradientWRTDefFieldRegion2 = nifti_copy_nim_info(this->voxelBasedMeasureGradient);
	this->gapOverlapGradientWRTDefFieldRegion1->data = (void *)calloc(this->gapOverlapGradientWRTDefFieldRegion1->nvox,
		this->gapOverlapGradientWRTDefFieldRegion1->nbyper);
	this->gapOverlapGradientWRTDefFieldRegion2->data = (void *)calloc(this->gapOverlapGradientWRTDefFieldRegion2->nvox,
		this->gapOverlapGradientWRTDefFieldRegion2->nbyper);

#ifndef NDEBUG
	reg_print_fct_debug("reg_f3d_sli<T>::AllocateVoxelBasedMeasureGradient");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d_sli<T>::ClearVoxelBasedMeasureGradient()
{
	//call method from reg_base to clear voxel-based similarity gradient image for combined transform
	reg_base<T>::ClearVoxelBasedMeasureGradient();

	//clear voxel-based similarity gradient images for each region
	if (this->region1VoxelBasedMeasureGradientImage != NULL)
	{
		nifti_image_free(this->region1VoxelBasedMeasureGradientImage);
		this->region1VoxelBasedMeasureGradientImage = NULL;
	}
	if (this->region2VoxelBasedMeasureGradientImage != NULL)
	{
		nifti_image_free(this->region2VoxelBasedMeasureGradientImage);
		this->region2VoxelBasedMeasureGradientImage = NULL;
	}

	//clear voxel-based gap-overlap penalty term gradient images
	if (this->gapOverlapGradientWRTDefFieldRegion1 != NULL)
	{
		nifti_image_free(this->gapOverlapGradientWRTDefFieldRegion1);
		this->gapOverlapGradientWRTDefFieldRegion1 = NULL;
	}
	if (this->gapOverlapGradientWRTDefFieldRegion2 != NULL)
	{
		nifti_image_free(this->gapOverlapGradientWRTDefFieldRegion2);
		this->gapOverlapGradientWRTDefFieldRegion2 = NULL;
	}

#ifndef NDEBUG
	reg_print_fct_debug("reg_f3d_sli<T>::ClearVoxelBasedMeasureGradient");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d_sli<T>::AllocateTransformationGradient()
{
	//clear any previously allocated transformation gradients
	this->ClearTransformationGradient();

	//call method from reg_f3d to allocate transformation gradient for region 1
	reg_f3d<T>::AllocateTransformationGradient();

	//allocate transformation gradient image for region 2
	this->region2TransformationGradient = nifti_copy_nim_info(this->region2ControlPointGrid);
	this->region2TransformationGradient->data = (void *)calloc(this->region2TransformationGradient->nvox,
		this->region2TransformationGradient->nbyper);

#ifndef NDEBUG
	reg_print_fct_debug("reg_f3d_sli<T>::AllocateTransformationGradient");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d_sli<T>::ClearTransformationGradient()
{
	//call method from reg_f3d to clear transformation gradient for region 1
	reg_f3d<T>::ClearTransformationGradient();

	//clear transformation gradient image for region 2
	if (this->region2TransformationGradient != NULL)
	{
		nifti_image_free(this->region2TransformationGradient);
		this->region2TransformationGradient = NULL;
	}

#ifndef NDEBUG
	reg_print_fct_debug("reg_f3d_sli<T>::ClearTransformationGradient");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
T reg_f3d_sli<T>::InitialiseCurrentLevel()
{
	//call method from reg_f3d to calculate max step size for this level and to
	//refine gpg for region 1 and modify bending energy weight (and linear energy
	//weight?) if required
	T maxStepSize = reg_f3d<T>::InitialiseCurrentLevel();

	// Refine the control point grid for region 2 if required
	if (this->gridRefinement && this->currentLevel > 0)
		reg_spline_refineControlPointGrid(this->region2ControlPointGrid, this->currentReference);

	//set current distance map
	if(this->usePyramid)
		this->currentDistanceMap = this->distanceMapPyramid[this->currentLevel];
	else
		this->currentDistanceMap = this->distanceMapPyramid[0];

#ifndef NDEBUG
	reg_print_fct_debug("reg_f3d_sli<T>::InitialiseCurrentLevel");
#endif

	//return max step size
	return maxStepSize;
}
/* *************************************************************** */
template <class T>
void reg_f3d_sli<T>::ClearCurrentInputImage()
{
	//call method from reg_base to clear current reference, floating, and mask image
	reg_base<T>::ClearCurrentInputImage();

	//clear current distance map image
	this->currentDistanceMap = NULL;

#ifndef NDEBUG
	reg_print_fct_debug("reg_f3d_sli<T>::ClearCurrentInputImage");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_f3d_sli<T>::SetOptimiser()
{
	//create new optimiser object
	//if useConjGradient set then create new conjugate gradient optimiser
	if (this->useConjGradient)
		this->optimiser = new reg_conjugateGradient<T>();
	//else create standard (gradient ascent) optimiser
	else this->optimiser = new reg_optimiser<T>();

	//initialise optimiser passing pointers to data from transforms and gradients
	//from both regions
	this->optimiser->Initialise(this->controlPointGrid->nvox,//number of voxels in region 1 CPG
		this->controlPointGrid->nz>1 ? 3 : 2,
		this->optimiseX,
		this->optimiseY,
		this->optimiseZ,
		this->maxiterationNumber,
		0, // currentIterationNumber
		this, //this object, which implements interface for interacting with optimiser
		static_cast<T *>(this->controlPointGrid->data),//pointer to data from region 1 CPG
		static_cast<T *>(this->transformationGradient->data),//pointer to data from region 1 gradient
		this->region2ControlPointGrid->nvox,//number of voxels in region 2 CPG
		static_cast<T *>(this->region2ControlPointGrid->data),//pointer to data from region 2 CPG
		static_cast<T *>(this->region2TransformationGradient->data));//pointer to data from region 2 gradient

#ifndef NDEBUG
	reg_print_fct_debug("reg_f3d_sli<T>::SetOptimiser");
#endif
}
/* *************************************************************** */
template <class T>
void reg_f3d_sli<T>::UpdateParameters(float scale)
{
	// First update the transformation for region 1
	reg_f3d<T>::UpdateParameters(scale);

	// Create some pointers to the relevant arrays
	//noet - call '_b' methods from optimiser to access the region 2 data
	T *currentDOFRegion2 = this->optimiser->GetCurrentDOF_b();
	T *bestDOFRegion2 = this->optimiser->GetBestDOF_b();
	T *gradientRegion2 = this->optimiser->GetGradient_b();

	// update the CPG values for region 2
	size_t voxNumberRegion2 = this->optimiser->GetVoxNumber_b();
	// Update the values for the x-axis displacement
	if (this->optimiser->GetOptimiseX() == true)
	{
		for (size_t i = 0; i < voxNumberRegion2; ++i)
		{
			currentDOFRegion2[i] = bestDOFRegion2[i] + scale * gradientRegion2[i];
		}
	}
	// Update the values for the y-axis displacement
	if (this->optimiser->GetOptimiseY() == true)
	{
		//pointers to y-axis displacements
		T *currentDOFYRegion2 = &currentDOFRegion2[voxNumberRegion2];
		T *bestDOFYRegion2 = &bestDOFRegion2[voxNumberRegion2];
		T *gradientYRegion2 = &gradientRegion2[voxNumberRegion2];
		for (size_t i = 0; i < voxNumberRegion2; ++i)
		{
			currentDOFYRegion2[i] = bestDOFYRegion2[i] + scale * gradientYRegion2[i];
		}
	}
	// Update the values for the z-axis displacement
	if (this->optimiser->GetOptimiseZ() == true && this->optimiser->GetNDim()>2)
	{
		//pointers to z-axis displacements
		T *currentDOFZRegion2 = &currentDOFRegion2[2 * voxNumberRegion2];
		T *bestDOFZRegion2 = &bestDOFRegion2[2 * voxNumberRegion2];
		T *gradientZRegion2 = &gradientRegion2[2 * voxNumberRegion2];
		for (size_t i = 0; i < voxNumberRegion2; ++i)
		{
			currentDOFZRegion2[i] = bestDOFZRegion2[i] + scale * gradientZRegion2[i];
		}
	}

#ifndef NDEBUG
	reg_print_fct_debug("reg_f3d_sli<T>::UpdateParameters");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d_sli<T>::UpdateBestObjFunctionValue()
{
	// call method from reg_f3d to update all values except gap-overlap term
	reg_f3d<T>::UpdateBestObjFunctionValue();
	//now update best gap-overlap value
	this->bestWGO = this->currentWGO;

#ifndef NDEBUG
	reg_print_fct_debug("reg_f3d_sli<T>::UpdateBestObjFunctionValue");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d_sli<T>::PrintInitialObjFunctionValue()
{
	// if verbose not set don't display anything
	if (!this->verbose) return;

	// format text with initial total objective function value
	char text[255];
	sprintf(text, "Initial objective function: %g",	this->optimiser->GetBestObjFunctionValue());
	// and similarity measure(s) value
	sprintf(text + strlen(text), " = (wSIM)%g", this->bestWMeasure);
	// and values of penalty terms
	if (this->bendingEnergyWeight > 0)
		sprintf(text + strlen(text), " - (wBE)%.2e", this->bestWBE);
	if (this->linearEnergyWeight > 0)
		sprintf(text + strlen(text), " - (wLE)%.2e", this->bestWLE);
	//jacobian and landmark penalty terms not currently implemented for sliding region registrations
	//if (this->jacobianLogWeight>0)
	//	sprintf(text + strlen(text), " - (wJAC)%.2e", this->bestWJac);
	//if (this->landmarkRegWeight>0)
	//	sprintf(text + strlen(text), " - (wLAN)%.2e", this->bestWLand);
	if (this->gapOverlapWeight > 0)
		sprintf(text + strlen(text), " - (wGO)%.2e", this->bestWGO);

	//display text
	reg_print_info(this->executableName, text);

#ifndef NDEBUG
	reg_print_fct_debug("reg_f3d_sli<T>::PrintInitialObjFunctionValue");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d_sli<T>::PrintCurrentObjFunctionValue(T currentSize)
{
	// if verbose not set don't display anything
	if (!this->verbose) return;

	// format text with iteration number and current total objective function value
	char text[255];
	sprintf(text, "[%i] Current objective function: %g",
		(int)this->optimiser->GetCurrentIterationNumber(),
		this->optimiser->GetBestObjFunctionValue());
	// and similarity measure(s) value
	sprintf(text + strlen(text), " = (wSIM)%g", this->bestWMeasure);
	// and values of penalty terms
	if (this->bendingEnergyWeight > 0)
		sprintf(text + strlen(text), " - (wBE)%.2e", this->bestWBE);
	if (this->linearEnergyWeight > 0)
		sprintf(text + strlen(text), " - (wLE)%.2e", this->bestWLE);
	//jacobian and landmark penalty terms not currently implemented for sliding region registrations
	//if (this->jacobianLogWeight>0)
	//	sprintf(text + strlen(text), " - (wJAC)%.2e", this->bestWJac);
	//if (this->landmarkRegWeight>0)
	//	sprintf(text + strlen(text), " - (wLAN)%.2e", this->bestWLand);
	if (this->gapOverlapWeight > 0)
		sprintf(text + strlen(text), " - (wGO)%.2e", this->bestWGO);
	//add current step size to text
	sprintf(text + strlen(text), " [+ %g mm]", currentSize);

	//display text
	reg_print_info(this->executableName, text);

#ifndef NDEBUG
	reg_print_fct_debug("reg_f3d_sli<T>::PrintCurrentObjFunctionValue");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
void reg_f3d_sli<T>::SetDistanceMapImage(nifti_image *distanceMapImage)
{
	this->inputDistanceMap = distanceMapImage;

#ifndef NDEBUG
	reg_print_fct_debug("reg_f3d_sli<T>::SetDistanceMapImage");
#endif
}
/* *************************************************************** */
template<class T>
void reg_f3d_sli<T>::SetGapOverlapWeight(T weight)
{
	this->gapOverlapWeight = weight;

#ifndef NDEBUG
	reg_print_fct_debug("reg_f3d_sli<T>::SetGapOverlapWeight");
#endif
}
/* *************************************************************** */
template<class T>
nifti_image * reg_f3d_sli<T>::GetRegion2ControlPointPositionImage()
{
	// Create a control point grid nifti image
	nifti_image *returnedControlPointGrid = nifti_copy_nim_info(this->region2ControlPointGrid);

	// Allocate the new image data array
	returnedControlPointGrid->data = (void *)malloc(returnedControlPointGrid->nvox*returnedControlPointGrid->nbyper);
	
	// Copy the final region2 control point grid image
	memcpy(returnedControlPointGrid->data, this->region2ControlPointGrid->data,
		returnedControlPointGrid->nvox * returnedControlPointGrid->nbyper);
	
	// Return the new control point grid
#ifndef NDEBUG
	reg_print_fct_debug("reg_f3d_sli<T>::GetRegion2ControlPointPositionImage");
#endif
	return returnedControlPointGrid;
}
/* *************************************************************** */
template<class T>
void reg_f3d_sli<T>::SetRegion2ControlPointGridImage(nifti_image *controlPointGrid)
{
	this->inputRegion2ControlPointGrid = controlPointGrid;

#ifndef NDEBUG
	reg_print_fct_debug("reg_f3d_sli<T>::SetRegion2ControlPointGridImage");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
void reg_f3d_sli<T>::CheckParameters()
{
	//call method from reg_f3d to check standard parameters
	reg_f3d<T>::CheckParameters();

	//check the distance map has been defined
	if (this->inputDistanceMap == NULL)
	{
		reg_print_fct_error("reg_f3d_sli::CheckParameters()");
		reg_print_msg_error("The distance map image is not defined");
		reg_exit();
	}
	else
	{
		//and has the same dimensions as the floating (source) image
		if (this->inputDistanceMap->nx != this->inputFloating->nx ||
			this->inputDistanceMap->ny != this->inputFloating->ny ||
			this->inputDistanceMap->nz != this->inputFloating->nz)
		{
			reg_print_fct_error("reg_f3d_sli<T>::CheckParameters()");
			reg_print_msg_error("The distance map has different dimensions to the floating image");
			reg_exit();
		}
	}

	//check if an input CPG has only been provided for one region
	if ((this->inputControlPointGrid != NULL && this->inputRegion2ControlPointGrid == NULL) ||
		(this->inputControlPointGrid == NULL && this->inputRegion2ControlPointGrid != NULL))
	{
		reg_print_fct_error("reg_f3d_sli<T>::CheckParameters()");
		reg_print_msg_error("An input Control Point Grid has only been provided for one region");
		reg_print_msg_error("You must provide a Control Point Grid for both regions (or none)");
		reg_exit();
	}
	
	//if input CPGs provided for both regions check they have the same dimensions
	if (this->inputControlPointGrid != NULL && this->inputRegion2ControlPointGrid != NULL)
	{
		if (this->inputControlPointGrid->nx != this->inputRegion2ControlPointGrid->nx ||
			this->inputControlPointGrid->ny != this->inputRegion2ControlPointGrid->ny ||
			this->inputControlPointGrid->nz != this->inputRegion2ControlPointGrid->nz)
		{
			reg_print_fct_error("reg_f3d_sli<T>::CheckParameters()");
			reg_print_msg_error("The input Control Point Grids for the two regions have different dimensions");
			reg_exit();
		}
	}


	//check if jacobian or landmark penalty term weights have been set - if so throw error as
	//these terms are not yet implemented for sliding region registrations
	if (this->jacobianLogWeight > 0)
	{
		reg_print_fct_error("reg_f3d_sli<T>::CheckParameters()");
		reg_print_msg_error("Jacobian penalty term weight > 0");
		reg_print_msg_error("Jacobian penalty term has not yet been implemented to work with sliding region registrations");
		reg_exit();
	}
	if (this->landmarkRegWeight > 0)
	{
		reg_print_fct_error("reg_f3d_sli<T>::CheckParameters()");
		reg_print_msg_error("Landmark penalty term weight > 0");
		reg_print_msg_error("Landmark penalty term has not yet been implemented to work with sliding region registrations");
		reg_exit();
	}

	// check if penalty term weights >= 1
	T penaltySum = this->bendingEnergyWeight + this->linearEnergyWeight + this->gapOverlapWeight;
	if (penaltySum >= 1)
	{
		//display a warning saying images will be ignored for the registration
		reg_print_fct_warn("reg_f3d_sli<T>::CheckParameters()");
		reg_print_msg_warn("Penalty term weights greater than or equal to 1");
		reg_print_msg_warn("The images will be ignored during the registration ");
		this->similarityWeight = 0;
		this->bendingEnergyWeight /= penaltySum;
		this->linearEnergyWeight /= penaltySum;
		this->gapOverlapWeight /= penaltySum;
	}
	else this->similarityWeight = 1.0 - penaltySum;

#ifndef NDEBUG
	reg_print_fct_debug("reg_f3d_sli<T>::CheckParameters");
#endif
	return;
}
/* *************************************************************** */
template<class T>
void reg_f3d_sli<T>::Initialise()
{
	//call method from reg_f3d to initialise image pyramids and region 1 control point grid
	reg_f3d<T>::Initialise();

	//initialise control point grid for region 2
	//
	//check if an input grid has been provided
	if (this->inputRegion2ControlPointGrid != NULL)
	{
		//if so use input grid to initialise region 2 control point grid
		this->region2ControlPointGrid = nifti_copy_nim_info(this->inputRegion2ControlPointGrid);
		this->region2ControlPointGrid->data = (void *)malloc(this->region2ControlPointGrid->nvox * this->region2ControlPointGrid->nbyper);
		memcpy(this->region2ControlPointGrid->data, this->inputRegion2ControlPointGrid->data, 
			this->region2ControlPointGrid->nvox * this->region2ControlPointGrid->nbyper);
	}
	else
	{
		//if not copy grid from region 1
		this->region2ControlPointGrid = nifti_copy_nim_info(this->controlPointGrid);
		this->region2ControlPointGrid->data = (void *)malloc(this->region2ControlPointGrid->nvox * this->region2ControlPointGrid->nbyper);
		memcpy(this->region2ControlPointGrid->data, this->controlPointGrid->data,
			this->region2ControlPointGrid->nvox * this->region2ControlPointGrid->nbyper);
	}

	//check if image pyramids are being used for multi-resolution
	if (this->usePyramid)
	{
		//create image pyramid for distance map, with one image for each resolution level
		this->distanceMapPyramid = (nifti_image **)malloc(this->levelToPerform * sizeof(nifti_image *));
		reg_createImagePyramid<T>(this->inputDistanceMap, this->distanceMapPyramid, this->levelNumber, this->levelToPerform);
	}
	else
	{
		//image pyramids are not used, so create pyramid with just one level (i.e. copy of input image)
		this->distanceMapPyramid = (nifti_image **)malloc(sizeof(nifti_image *));
		reg_createImagePyramid<T>(this->inputDistanceMap, this->distanceMapPyramid, 1, 1);
	}

#ifdef NDEBUG
	if (this->verbose)
	{
#endif
		//print out some info:
		std::string text;
		//name of distance map image
		text = stringFormat("Distance map image used for sliding regions: %s", this->inputDistanceMap->fname);
		reg_print_info(this->executableName, text.c_str());
		//weight of gap-overlap penalty term
		text = stringFormat("Gap-overlap penalty term weight: %g", this->gapOverlapWeight);
		reg_print_info(this->executableName, text.c_str());
		reg_print_info(this->executableName, "");

#ifdef NDEBUG
	}
#endif
#ifndef NDEBUG
	reg_print_fct_debug("reg_f3d<T>::Initialise");
#endif

}
/* *************************************************************** */
/* *************************************************************** */
template class reg_f3d_sli<float>;
#endif
