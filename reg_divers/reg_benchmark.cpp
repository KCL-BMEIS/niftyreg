/*
 *  benchmark.cpp
 *
 *
 *  Created by Marc Modat on 15/11/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */
 
 #include "_reg_resampling.h"
#include "_reg_affineTransformation.h"
#include "_reg_bspline.h"
#include "_reg_mutualinformation.h"
#include "_reg_ssd.h"
#include "_reg_tools.h"
#include "_reg_blockMatching.h"

#include "_reg_cudaCommon.h"
#include "_reg_resampling_gpu.h"
#include "_reg_affineTransformation_gpu.h"
#include "_reg_bspline_gpu.h"
#include "_reg_mutualinformation_gpu.h"
#include "_reg_tools_gpu.h"
#include "_reg_blockMatching_gpu.h"
 
 #define GRID_SPACING 10
 #define BINNING 68
 
 int main(int argc, char **argv)
 {
 	//The image dimension is user-defined
	if(argc<2){
		fprintf(stderr, "The image dimension is expected.\n");
		fprintf(stderr, "Exit ...\n");
		return 1;	
	}
	int dimension = atoi(argv[1]);
	
	// The target, source and result images are created
	 int dim_img[8];
     dim_img[0]=3;
     dim_img[1]=dimension;
     dim_img[2]=dimension;
     dim_img[3]=dimension;
     dim_img[4]=dim_img[5]=dim_img[6]=dim_img[7]=1;
	 nifti_image *targetImage = nifti_make_new_nim(dim_img, NIFTI_TYPE_FLOAT32, true);
	 nifti_image *sourceImage = nifti_make_new_nim(dim_img, NIFTI_TYPE_FLOAT32, true);
	 nifti_image *resultImage = nifti_make_new_nim(dim_img, NIFTI_TYPE_FLOAT32, true);
     targetImage->sform_code=0;
     sourceImage->sform_code=0;
     resultImage->sform_code=0;

	 // The target and source images are filled with random number
	 float *targetPtr=static_cast<float *>(targetImage->data);
	 float *sourcePtr=static_cast<float *>(sourceImage->data);
	 for(int i=0;i<targetImage->nvox;++i){
	 	*targetPtr+=0.0f;	
	 	*sourcePtr+=0.0f;	
	 }
     
     // Deformation field image is created
     dim_img[0]=5;
     dim_img[1]=dimension;
     dim_img[2]=dimension;
     dim_img[3]=dimension;
     dim_img[5]=3;
     dim_img[4]=dim_img[6]=dim_img[7]=1;
	 nifti_image *deformationFieldImage = nifti_make_new_nim(dim_img, NIFTI_TYPE_FLOAT32, true);
     targetImage->sform_code=0;
     sourceImage->sform_code=0;
     resultImage->sform_code=0;
	 
	 // Joint histogram creation
	double *probaJointHistogram=(double *)malloc(BINNING*(BINNING+2)*sizeof(double));
	double *logJointHistogram=(double *)malloc(BINNING*(BINNING+2)*sizeof(double));
	
	// Affine transformation
	mat44 *affineTransformation = (mat44 *)calloc(1,sizeof(mat44));
	affineTransformation->m[0][0]=1.0;
	affineTransformation->m[1][1]=1.0;
	affineTransformation->m[2][2]=1.0;
	affineTransformation->m[3][3]=1.0;
	
	// A control point image is created
    dim_img[0]=5;
    dim_img[1]=(int)floor(targetImage->nx*targetImage->dx/GRID_SPACING)+4;
    dim_img[2]=(int)floor(targetImage->ny*targetImage->dy/GRID_SPACING)+4;
    dim_img[3]=(int)floor(targetImage->nz*targetImage->dz/GRID_SPACING)+4;
    dim_img[5]=3;
    dim_img[4]=dim_img[6]=dim_img[7]=1;
    nifti_image *controlPointImage = nifti_make_new_nim(dim_img, NIFTI_TYPE_FLOAT32, true);
    controlPointImage->cal_min=0;
    controlPointImage->cal_max=0;
    controlPointImage->pixdim[0]=1.0f;
    controlPointImage->pixdim[1]=controlPointImage->dx=GRID_SPACING;
    controlPointImage->pixdim[2]=controlPointImage->dy=GRID_SPACING;
    controlPointImage->pixdim[3]=controlPointImage->dz=GRID_SPACING;
    controlPointImage->pixdim[4]=controlPointImage->dt=1.0f;
    controlPointImage->pixdim[5]=controlPointImage->du=1.0f;
    controlPointImage->pixdim[6]=controlPointImage->dv=1.0f;
    controlPointImage->pixdim[7]=controlPointImage->dw=1.0f;
    controlPointImage->qform_code=targetImage->qform_code;
    controlPointImage->sform_code=targetImage->sform_code;
    float qb, qc, qd, qx, qy, qz, dx, dy, dz, qfac;
    nifti_mat44_to_quatern( targetImage->qto_xyz, &qb, &qc, &qd, &qx, &qy, &qz, &dx, &dy, &dz, &qfac);
    controlPointImage->quatern_b=qb;
    controlPointImage->quatern_c=qc;
    controlPointImage->quatern_d=qd;
    controlPointImage->qfac=qfac;
    controlPointImage->qto_xyz = nifti_quatern_to_mat44(qb, qc, qd, qx, qy, qz,
        controlPointImage->dx, controlPointImage->dy, controlPointImage->dz, qfac);
    float originIndex[3];
    float originReal[3];
    originIndex[0] = -1.0f;
    originIndex[1] = -1.0f;
    originIndex[2] = -1.0f;
    reg_mat44_mul(&(controlPointImage->qto_xyz), originIndex, originReal);
    controlPointImage->qto_xyz.m[0][3] = controlPointImage->qoffset_x = originReal[0];
    controlPointImage->qto_xyz.m[1][3] = controlPointImage->qoffset_y = originReal[1];
    controlPointImage->qto_xyz.m[2][3] = controlPointImage->qoffset_z = originReal[2];
    controlPointImage->qto_ijk = nifti_mat44_inverse(controlPointImage->qto_xyz);
    if(reg_bspline_initialiseControlPointGridWithAffine(affineTransformation, controlPointImage))
    	return 1;
    
    // Different gradient images
	nifti_image *resultGradientImage = nifti_copy_nim_info(deformationFieldImage);
	resultGradientImage->datatype = NIFTI_TYPE_FLOAT32;
	resultGradientImage->nbyper = sizeof(float);
	resultGradientImage->data = (void *)calloc(resultGradientImage->nvox, resultGradientImage->nbyper);
	nifti_image *voxelNMIGradientImage = nifti_copy_nim_info(deformationFieldImage);
	voxelNMIGradientImage->datatype = NIFTI_TYPE_FLOAT32;
	voxelNMIGradientImage->nbyper = sizeof(float);
	voxelNMIGradientImage->data = (void *)calloc(voxelNMIGradientImage->nvox, voxelNMIGradientImage->nbyper);
	nifti_image *nodeNMIGradientImage = nifti_copy_nim_info(controlPointImage);
	nodeNMIGradientImage->datatype = NIFTI_TYPE_FLOAT32;
	nodeNMIGradientImage->nbyper = sizeof(float);
	nodeNMIGradientImage->data = (void *)calloc(nodeNMIGradientImage->nvox, nodeNMIGradientImage->nbyper);
	
	// Conjugate gradient arrays
	float *conjugateG = (float *)calloc(nodeNMIGradientImage->nvox, sizeof(float));
	float *conjugateH = (float *)calloc(nodeNMIGradientImage->nvox, sizeof(float));

	time_t start,end;

	/* Functions to be tested
		- affine deformation field
		- spline deformation field
		- linear interpolation
		- block matching computation
		- spatial gradient computation
		- voxel-based NMI gradient computation
		- node-based NMI gradient computation
		- conjugate gradient computation
		- bending-energy computation
		- bending-energy gradient computation
		- gradient form voxel to real space
	*/

	// AFFINE DEFORMATION FIELD CREATION
	time(&start);
	for(int i=0; i<100; ++i){
		reg_affine_positionField(	affineTransformation,
									targetImage,
									deformationFieldImage);
	}
	time(&end);
	int minutes = (int)floorf(float(end-start)/60.0f);
	int seconds = (int)(end-start - 60*minutes);
	printf("CPU - 100 affine deformation field computations - %i min %i sec\n", minutes, seconds);

	// SPLINE DEFORMATION FIELD CREATION
	time(&start);
	for(int i=0; i<100; ++i){
		reg_bspline<float>(	controlPointImage,
			                targetImage,
			                deformationFieldImage,
			                NULL,
			                0);
	}
	time(&end);
	minutes = (int)floorf(float(end-start)/60.0f);
	seconds = (int)(end-start - 60*minutes);
	printf("CPU - 100 spline deformation field computations - %i min %i sec\n", minutes, seconds);

	/* Monsieur Propre */
	nifti_image_free(targetImage);
	nifti_image_free(sourceImage);
	nifti_image_free(resultImage);
	nifti_image_free(controlPointImage);
	nifti_image_free(deformationFieldImage);
	nifti_image_free(resultGradientImage);
	nifti_image_free(voxelNMIGradientImage);
	nifti_image_free(nodeNMIGradientImage);
	free(probaJointHistogram);
	free(logJointHistogram);
	free(conjugateG);
	free(conjugateH);
 	return 0;
 }