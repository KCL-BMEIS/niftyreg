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


void Usage(char *);

int main(int argc, char **argv)
{
    int dimension = 100;
    float gridSpacing = 10.0f;
    int binning = 68;
    char *outputFileName = "benchmark_result.txt";

    for(int i=1;i<argc;i++){
        if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 ||
            strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 ||
            strcmp(argv[i], "--h")==0 || strcmp(argv[i], "--help")==0){
            Usage(argv[0]);
            return 0;
        }
        else if(strcmp(argv[i], "-dim") == 0){
            dimension=atoi(argv[++i]);
        }
        else if(strcmp(argv[i], "-sp") == 0){
            gridSpacing=atof(argv[++i]);
        }
        else if(strcmp(argv[i], "-bin") == 0){
            binning=atoi(argv[++i]);
        }
        else if(strcmp(argv[i], "-o") == 0){
            outputFileName=argv[++i];
        }
        else{
            fprintf(stderr,"Err:\tParameter %s unknown.\n",argv[i]);
            Usage(argv[0]);
            return 1;
        }
    }
	
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
    int *maskImage = (int *)malloc(targetImage->nvox*sizeof(int));

	// The target and source images are filled with random number
	float *targetPtr=static_cast<float *>(targetImage->data);
	float *sourcePtr=static_cast<float *>(sourceImage->data);
    srand((unsigned)time(0));
	for(int i=0;i<targetImage->nvox;++i){
	    *targetPtr++ = (float)(binning-4)*(float)rand()/(float)RAND_MAX + 2.0f;
	    *sourcePtr++ = (float)(binning-4)*(float)rand()/(float)RAND_MAX + 2.0f;
        maskImage[i]=i;
	}

nifti_set_filenames(targetImage, "temp.nii", 0, 0);
nifti_image_write(targetImage);

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
	double *probaJointHistogram=(double *)malloc(binning*(binning+2)*sizeof(double));
	double *logJointHistogram=(double *)malloc(binning*(binning+2)*sizeof(double));
	
	// Affine transformation
	mat44 *affineTransformation = (mat44 *)calloc(1,sizeof(mat44));
	affineTransformation->m[0][0]=1.0;
	affineTransformation->m[1][1]=1.0;
	affineTransformation->m[2][2]=1.0;
	affineTransformation->m[3][3]=1.0;
	
	// A control point image is created
    dim_img[0]=5;
    dim_img[1]=(int)floor(targetImage->nx*targetImage->dx/gridSpacing)+4;
    dim_img[2]=(int)floor(targetImage->ny*targetImage->dy/gridSpacing)+4;
    dim_img[3]=(int)floor(targetImage->nz*targetImage->dz/gridSpacing)+4;
    dim_img[5]=3;
    dim_img[4]=dim_img[6]=dim_img[7]=1;
    nifti_image *controlPointImage = nifti_make_new_nim(dim_img, NIFTI_TYPE_FLOAT32, true);
    controlPointImage->cal_min=0;
    controlPointImage->cal_max=0;
    controlPointImage->pixdim[0]=1.0f;
    controlPointImage->pixdim[1]=controlPointImage->dx=gridSpacing;
    controlPointImage->pixdim[2]=controlPointImage->dy=gridSpacing;
    controlPointImage->pixdim[3]=controlPointImage->dz=gridSpacing;
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
    int minutes, seconds, maxIt;

    //
    FILE *outputFile;
    outputFile=fopen(outputFileName, "w");

	/* Functions to be tested
		- affine deformation field
		- spline deformation field
		- linear interpolation
		- block matching computation
		- spatial gradient computation
		- voxel-based NMI gradient computation
		- node-based NMI gradient computation
		- bending-energy computation
		- bending-energy gradient computation
	*/

	// AFFINE DEFORMATION FIELD CREATION
    maxIt=1000000 / dimension;
	time(&start);
	for(int i=0; i<maxIt; ++i){
		reg_affine_positionField(	affineTransformation,
									targetImage,
									deformationFieldImage);
	}
	time(&end);
	minutes = (int)floorf(float(end-start)/60.0f);
	seconds = (int)(end-start - 60*minutes);
	fprintf(outputFile, "CPU - %i affine deformation field computations - %i min %i sec\n", maxIt, minutes, seconds);
    printf("Affine deformation done\n");

	// SPLINE DEFORMATION FIELD CREATION
    maxIt=50000 / dimension;
	time(&start);
	for(int i=0; i<maxIt; ++i){
		reg_bspline<float>(	controlPointImage,
			                targetImage,
			                deformationFieldImage,
			                maskImage,
			                0);
	}
	time(&end);
	minutes = (int)floorf(float(end-start)/60.0f);
	seconds = (int)(end-start - 60*minutes);
    fprintf(outputFile, "CPU - %i spline deformation field computations - %i min %i sec\n", maxIt, minutes, seconds);
    printf("Spline deformation done\n");

    // LINEAR INTERPOLATION
    maxIt=100000 / dimension;
    time(&start);
    for(int i=0; i<maxIt; ++i){
        reg_resampleSourceImage<float>( targetImage,
                                        sourceImage,
                                        resultImage,
                                        deformationFieldImage,
                                        maskImage,
                                        1,
                                        0);
    }
    time(&end);
    minutes = (int)floorf(float(end-start)/60.0f);
    seconds = (int)(end-start - 60*minutes);
    fprintf(outputFile, "CPU - %i linear interpolation computations - %i min %i sec\n", maxIt, minutes, seconds);
    printf("Linear interpolation done\n");

    // BLOCK MATCHING
    _reg_blockMatchingParam blockMatchingParams;
    initialise_block_matching_method(   targetImage,
                                        &blockMatchingParams,
                                        100,    // percentage of block kept
                                        50,     // percentage of inlier in the optimisation process
                                        maskImage);
    maxIt=2000 / dimension;
    time(&start);
    for(int i=0; i<maxIt; ++i){
        block_matching_method<float>(   targetImage,
                                        resultImage,
                                        &blockMatchingParams,
                                        maskImage);
    }
    time(&end);
    minutes = (int)floorf(float(end-start)/60.0f);
    seconds = (int)(end-start - 60*minutes);
    fprintf(outputFile, "CPU - %i block matching computations - %i min %i sec\n", maxIt, minutes, seconds);
    printf("Block-matching done\n");

    // SPATIAL GRADIENT COMPUTATION
    maxIt=100000 / dimension;
    time(&start);
    for(int i=0; i<maxIt; ++i){
        reg_getSourceImageGradient<float>(  targetImage,
                                            sourceImage,
                                            resultGradientImage,
                                            deformationFieldImage,
                                            maskImage,
                                            1);
    }
    time(&end);
    minutes = (int)floorf(float(end-start)/60.0f);
    seconds = (int)(end-start - 60*minutes);
    fprintf(outputFile, "CPU - %i spatial gradient computations - %i min %i sec\n", maxIt, minutes, seconds);
    printf("Spatial gradient done\n");

    // JOINT HISTOGRAM COMPUTATION
    double entropies[4];
    reg_getEntropies<double>(   targetImage,
                                resultImage,
                                2,
                                binning,
                                probaJointHistogram,
                                logJointHistogram,
                                entropies,
                                maskImage);


    // VOXEL-BASED NMI GRADIENT COMPUTATION
    maxIt=1000 / dimension;
    time(&start);
    for(int i=0; i<maxIt; ++i){
        reg_getVoxelBasedNMIGradientUsingPW<double>(targetImage,
                                                    resultImage,
                                                    2,
                                                    resultGradientImage,
                                                    binning,
                                                    logJointHistogram,
                                                    entropies,
                                                    voxelNMIGradientImage,
                                                    maskImage);
    }
    time(&end);
    minutes = (int)floorf(float(end-start)/60.0f);
    seconds = (int)(end-start - 60*minutes);
    fprintf(outputFile, "CPU - %i voxel-based NMI gradient computations - %i min %i sec\n", maxIt, minutes, seconds);
    printf("Voxel-based NMI gradient done\n");

    // NODE-BASED NMI GRADIENT COMPUTATION
    maxIt=10000 / dimension;
    int smoothingRadius[3];
    smoothingRadius[0] = (int)floor( 2.0*controlPointImage->dx/targetImage->dx );
    smoothingRadius[1] = (int)floor( 2.0*controlPointImage->dy/targetImage->dy );
    smoothingRadius[2] = (int)floor( 2.0*controlPointImage->dz/targetImage->dz );
    time(&start);
    for(int i=0; i<maxIt; ++i){
        reg_smoothImageForCubicSpline<float>(voxelNMIGradientImage,smoothingRadius);
        reg_voxelCentric2NodeCentric(nodeNMIGradientImage,voxelNMIGradientImage);
    }
    time(&end);
    minutes = (int)floorf(float(end-start)/60.0f);
    seconds = (int)(end-start - 60*minutes);
    fprintf(outputFile, "CPU - %i node-based NMI gradient computations - %i min %i sec\n", maxIt, minutes, seconds);
    printf("Node-based NMI gradient done\n");

    // BENDING ENERGY COMPUTATION
    maxIt=10000000 / dimension;
    time(&start);
    for(int i=0; i<maxIt; ++i){
        reg_bspline_bendingEnergy<float>(controlPointImage, targetImage,1);
    }
    time(&end);
    minutes = (int)floorf(float(end-start)/60.0f);
    seconds = (int)(end-start - 60*minutes);
    fprintf(outputFile, "CPU - %i BE computations - %i min %i sec\n", maxIt, minutes, seconds);
    printf("BE gradient done\n");

    // BENDING ENERGY GRADIENT COMPUTATION
    maxIt=5000000 / dimension;
    time(&start);
    for(int i=0; i<maxIt; ++i){
        reg_bspline_bendingEnergyGradient<float>(   controlPointImage,
                                                    targetImage,
                                                    nodeNMIGradientImage,
                                                    0.01f);
    }
    time(&end);
    minutes = (int)floorf(float(end-start)/60.0f);
    seconds = (int)(end-start - 60*minutes);
    fprintf(outputFile, "CPU - %i BE gradient computations - %i min %i sec\n", maxIt, minutes, seconds);
    printf("BE gradient done\n");

    fclose(outputFile);

	/* Monsieur Propre */
	nifti_image_free(targetImage);
	nifti_image_free(sourceImage);
	nifti_image_free(resultImage);
	nifti_image_free(controlPointImage);
	nifti_image_free(deformationFieldImage);
	nifti_image_free(resultGradientImage);
	nifti_image_free(voxelNMIGradientImage);
	nifti_image_free(nodeNMIGradientImage);
    free(maskImage);
	free(probaJointHistogram);
	free(logJointHistogram);
	free(conjugateG);
	free(conjugateH);
    return 0;
}

void Usage(char *exec)
{
    printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
    printf("Usage:\t%s [OPTIONS].\n",exec);
    printf("\t-dim <int>\tImage dimension [100]\n");
    printf("\t-bin <int>\tBin number [68]\n");
    printf("\t-sp <float>\tControl point grid spacing [10]\n");
    printf("\t-o <char*>\t Output file name [benchmark_result.txt]\n");
    printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
    return;
}
