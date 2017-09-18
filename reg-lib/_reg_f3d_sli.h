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
	nifti_image *inputRegion2ControlPointGrid; //pointer to external
	nifti_image *region2ControlPointGrid;
	nifti_image *region2DeformationFieldImage;
	nifti_image *region2VoxelBasedMeasureGradientImage;
	nifti_image *region2TransformationGradient;

	//variables for region1 transform
	nifti_image *region1DeformationFieldImage;
	nifti_image *region1VoxelBasedMeasureGradientImage;

	//variables for distance map image
	nifti_image *inputDistanceMap; //pointer to external
	nifti_image **distanceMapPyramid;
	nifti_image *currentDistanceMap;

	//variables for the distance map warped by the transform for each region
	nifti_image *warpedDistanceMapRegion1;
	nifti_image *warpedDistanceMapRegion2;

	//variables for the spatial gradient of distance map warped by the transform
	//for each region
	nifti_image *warpedDistanceMapGradientRegion1;
	nifti_image *warpedDistanceMapGradientRegion2;

	//variables for gap-overlap penalty term
	T gapOverlapWeight;
	double currentWGO;
	double bestWGO;

	//variables for the gradient of the penalty term with respect to the def field
	//for each region
	nifti_image *gapOverlapGradientWRTDefFieldRegion1;
	nifti_image *gapOverlapGradientWRTDefFieldRegion2;


	//reimplement method to get deformation field
	//combines deformation fields from each region based on warped distance maps
	//at each voxel, if sum of warped distance maps < 0 use region 1 def field else
	//use region 2 def field. if one warped distance map has a value of NaN (due to
	//transform mapping outside distance map image) and other warped distance map
	//maps to the same region as used to warp it (i.e. region 1 warped distance map
	// < 0 or region 2 warped distance map >= 0) then use def field from non-NaN
	//region, else combined def field set to NaN. If both warped distance maps are
	//NaN then combined def field set to NaN.
	virtual void GetDeformationField();
	//reimplement methods to allocate/clear deformation field
	//these methods will allocate/clear the deformation fields for each region as well
	//as the combined deformation field.
	virtual void AllocateDeformationField();
	virtual void ClearDeformationField();
	//reimplement methods to allocate/clear the warped images so that the warped
	//distance maps are also allocated/cleared
	virtual void AllocateWarped();
	virtual void ClearWarped();


	//methods to calculate objective function
	//note - no need to reimplement method to get similarity measure value as ths will
	//be calculated using the combined deformation field using the existing methods
	//
	//reimplement method to calculate objective function value to also include
	//value of gap-overlap penalty term
	virtual double GetObjectiveFunctionValue();
	//new method to calculate gap-overlap penalty term
	//at each voxel, the warped distance maps are multiplied together - if the result
	//is less than 0 (indicating the transforms for the two regions map the voxel into
	//different regiions, i.e. a gap or overlap is present) then the negative of the
	//result is added to the gap-overlap penalty term.
	virtual double ComputeGapOverlapPenaltyTerm();
	//reimplement methods to calculate prenalty terms using transforms for both regions
	virtual double ComputeBendingEnergyPenaltyTerm();
	virtual double ComputeLinearEnergyPenaltyTerm();
	//Jacobian penalty term not currently implemented to work with sliding region registrations
	//this method will throw an error if called
	virtual double ComputeJacobianBasedPenaltyTerm(int);
	//Landmark distance penalty term not currently implemented to work with sliding region registrations
	//this method will throw an error if called
	virtual double ComputeLandmarkDistancePenaltyTerm();


	//methods to calculate objective function gradient
	//
	//reimplement method to calculate objective function gradient to include gradient of
	//gap-ovlerlap penalty term
	virtual void GetObjectiveFunctionGradient();
	//reimplement method to convert voxel-based similarity gradient to CPG based
	//gradient(s). splits voxel-based gradient between two regions, based on warped
	//distance maps, and then converts voxel-based gradient for each region to CPG
	//gradients
	virtual void GetSimilarityMeasureGradient();
	//new method to calculate the gap-overlap penalty term gradient
	virtual void GetGapOverlapGradient();
	//reimplement methods to calculate penalty term gradients for transforms for both regions
	virtual void GetBendingEnergyGradient();
	virtual void GetLinearEnergyGradient();
	//Jacobian penalty term not currently implemented to work with sliding region registrations
	//this method will throw an error if called
	virtual void GetJacobianBasedGradient();
	//Landmark distance penalty term not currently implemented to work with sliding region registrations
	//this method will throw an error if called
	virtual void GetLandmarkDistanceGradient();
	//reimplement method to set gradient image to zero to set gradient images for both regions to 0
	virtual void SetGradientImageToZero();
	//reimplement method to normalise gradient so that gradients for both regions are normalised
	//using the max value over both gradient images
	virtual T NormaliseGradient();
	//reimplement method to smooth gradient so that gradients for both regions are smoothed
	virtual void SmoothGradient();
	//remiplement method to approximate gradient so that gradients for both regions are approximated
	virtual void GetApproximatedGradient();
	//reimplement methods to allocate/clear warped gradient images so that the warped
	//distance map gradients are also allocated/cleared
	virtual void AllocateWarpedGradient();
	virtual void ClearWarpedGradient();
	//reimplement methods to allocate/clear 'voxel-based' similarity measure gradient
	//image (i.e. the similarity measure gradient WRT the def field) - these methods
	//will now also allocate/clear the 'voxel-based' similarity measure gradients for
	//each region and the 'voxel-based' gap-overlap penalty term gradient images
	virtual void AllocateVoxelBasedMeasureGradient();
	virtual void ClearVoxelBasedMeasureGradient();
	//reimplement methods to allocate/clear transformation gradient images - these methods
	//will now also allocate/clear the transformation gradient image for region 2
	virtual void AllocateTransformationGradient();
	virtual void ClearTransformationGradient();

	//reimplement method to initialise current level to refine CPGs for transforms for both
	//regions and to set the current distance map image
	virtual T InitialiseCurrentLevel();
	//reimplement method to clear current input images to also clear current distance map
	//image
	virtual void ClearCurrentInputImage();

	//reimplement method for setting optimiser so that region 2 transform data and gradient
	//data also passed to optimiser.
	//note - no modifications to optimiser required as it can already jointly optimise 2
	//transforms for use with symmetric registrations
	virtual void SetOptimiser();
	//reimplement method for updating parameters so that region 2 transform is updated as well
	virtual void UpdateParameters(float);
	//reimplement method for updating best objective function value so that gap-overlap value
	//is updated as well
	virtual void UpdateBestObjFunctionValue();
	//reimplement methods for printing objective function value so that gap-overlap value is
	//also printed
	virtual void PrintInitialObjFunctionValue();
	virtual void PrintCurrentObjFunctionValue(T);
	
public:
	//constructor and destructor methods
	reg_f3d_sli(int refTimePoint, int floTimePoint);
	~reg_f3d_sli();


	//new method to set distance map image
	virtual void SetDistanceMapImage(nifti_image *);

	//new method to set gap-overlap penalty term weight
	virtual void SetGapOverlapWeight(T);

	//new methods to get and set transform for region 2
	//note - used similar method names as for methods for region 1 (i.e. standard methods from reg_f3d)
	//hence get method called ...Position... and set method called ...Grid...
	virtual nifti_image *GetRegion2ControlPointPositionImage();
	virtual void SetRegion2ControlPointGridImage(nifti_image *);


	//reimplement method to check parameters so that also checks if distance map has been set
	//and has same dimensions as floating image.
	//Also checks if an input control point grid has been set for one region but not the other,
	//and throws an error if so. If input control point grids have been set for both regions
	//then checks they have the same dimensions.
	//And checks that jacobian and landmark penalty terms have not been set (as not yet
	//implemented for sliding region registrations) and normalises penalty term weights
	virtual void CheckParameters();

	//reimplement method to initialise registration so that also initialises CPG for region 2
	//and image pyramid for distance map image
	virtual void Initialise();
};

#endif
