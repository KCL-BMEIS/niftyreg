/*
 *  reg_resample.cpp
 *
 *
 *  Created by Marc Modat on 18/05/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _MM_RESAMPLE_CPP
#define _MM_RESAMPLE_CPP

#include "_reg_resampling.h"
#include "_reg_affineTransformation.h"
#include "_reg_bspline.h"
#include "_reg_bspline_comp.h"
#include "_reg_tools.h"

#define PrecisionTYPE float

typedef struct{
	char *targetImageName;
	char *sourceImageName;
    char *affineMatrixName;
    char *inputCPPName;
    char *inputDEFName;
	char *outputResultName;
    char *outputBlankName;
	PrecisionTYPE sourceBGValue;
}PARAM;
typedef struct{
	bool targetImageFlag;
	bool sourceImageFlag;
	bool affineMatrixFlag;
    bool affineFlirtFlag;
    bool inputCPPFlag;
    bool inputDEFFlag;
	bool outputResultFlag;
    bool outputBlankFlag;
    bool NNInterpolationFlag;
    bool TRIInterpolationFlag;
}FLAG;


void PetitUsage(char *exec)
{
	fprintf(stderr,"Usage:\t%s -target <targetImageName> -source <sourceImageName> [OPTIONS].\n",exec);
	fprintf(stderr,"\tSee the help for more details (-h).\n");
	return;
}
void Usage(char *exec)
{
	printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
	printf("Usage:\t%s -target <filename> -source <filename> [OPTIONS].\n",exec);
	printf("\t-target <filename>\tFilename of the target image (mandatory)\n");
	printf("\t-source <filename>\tFilename of the source image (mandatory)\n\n");

	printf("* * OPTIONS * *\n");
    printf("\t*\tOnly one of the following tranformation is taken into account\n");
    printf("\t-aff <filename>\t\tFilename which contains an affine transformation (Affine*Target=Source)\n");
    printf("\t-affFlirt <filename>\t\tFilename which contains a radiological flirt affine transformation\n");
    printf("\t-cpp <filename>\t\tFilename of the control point grid image\n");
    printf("\t-def <filename>\t\tFilename of the deformation field image\n");

    printf("\t*\tThere are no limit for the required output number from the following\n");
    printf("\t-result <filename> \tFilename of the resampled image [none]\n");
    printf("\t-blank <filename> \tFilename of the resampled blank grid [none]\n");

    printf("\t*\tOthers\n");
	printf("\t-NN \t\t\tUse a Nearest Neighbor interpolation for the source resampling (cubic spline by default)\n");
	printf("\t-TRI \t\t\tUse a Trilinear interpolation for the source resampling (cubic spline by default)\n");
	printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
	return;
}

int main(int argc, char **argv)
{
	PARAM *param = (PARAM *)calloc(1,sizeof(PARAM));
	FLAG *flag = (FLAG *)calloc(1,sizeof(FLAG));
	
	/* read the input parameter */
	for(int i=1;i<argc;i++){
		if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 ||
		   strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 ||
		   strcmp(argv[i], "--h")==0 || strcmp(argv[i], "--help")==0){
			Usage(argv[0]);
			return 0;
		}
		else if(strcmp(argv[i], "-target") == 0){
			param->targetImageName=argv[++i];
			flag->targetImageFlag=1;
        }
        else if(strcmp(argv[i], "-source") == 0){
            param->sourceImageName=argv[++i];
            flag->sourceImageFlag=1;
        }
		else if(strcmp(argv[i], "-aff") == 0){
			param->affineMatrixName=argv[++i];
			flag->affineMatrixFlag=1;
		}
		else if(strcmp(argv[i], "-affFlirt") == 0){
			param->affineMatrixName=argv[++i];
			flag->affineMatrixFlag=1;
			flag->affineFlirtFlag=1;
		}
		else if(strcmp(argv[i], "-result") == 0){
			param->outputResultName=argv[++i];
			flag->outputResultFlag=1;
        }
        else if(strcmp(argv[i], "-cpp") == 0){
            param->inputCPPName=argv[++i];
            flag->inputCPPFlag=1;
        }
        else if(strcmp(argv[i], "-def") == 0){
            param->inputDEFName=argv[++i];
            flag->inputDEFFlag=1;
        }
		else if(strcmp(argv[i], "-NN") == 0){
			flag->NNInterpolationFlag=1;
		}
		else if(strcmp(argv[i], "-TRI") == 0){
			flag->TRIInterpolationFlag=1;
		}
		else if(strcmp(argv[i], "-blank") == 0){
			param->outputBlankName=argv[++i];
			flag->outputBlankFlag=1;
		}
		else{
			fprintf(stderr,"Err:\tParameter %s unknown.\n",argv[i]);
			PetitUsage(argv[0]);
			return 1;
		}
	}
	
	if(!flag->targetImageFlag || !flag->sourceImageFlag){
		fprintf(stderr,"Err:\tThe target and the source image have both to be defined.\n");
		PetitUsage(argv[0]);
		return 1;
	}
	
    /* Check the number of input images */
    if( ((unsigned int)flag->affineMatrixFlag
        + (unsigned int)flag->affineFlirtFlag
        + (unsigned int)flag->inputCPPFlag
        + (unsigned int)flag->inputDEFFlag) > 1){
        fprintf(stderr,"Err:\tOnly one input transformation has to be assigned.\n");
        PetitUsage(argv[0]);
        return 1;
    }

	/* Read the target image */
    nifti_image *targetImage = nifti_image_read(param->targetImageName,false);
	if(targetImage == NULL){
		fprintf(stderr,"** ERROR Error when reading the target image: %s\n",param->targetImageName);
		return 1;
	}
    reg_checkAndCorrectDimension(targetImage);
	
	/* Read the source image */
    nifti_image *sourceImage = nifti_image_read(param->sourceImageName,true);
	if(sourceImage == NULL){
		fprintf(stderr,"** ERROR Error when reading the source image: %s\n",param->sourceImageName);
		return 1;
	}
    reg_checkAndCorrectDimension(sourceImage);

	/* *********************************** */
	/* DISPLAY THE REGISTRATION PARAMETERS */
	/* *********************************** */
	printf("\n* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
	printf("Command line:\n");
	for(int i=0;i<argc;i++) printf(" %s", argv[i]);
	printf("\n\n");
	printf("Parameters\n");
	printf("Target image name: %s\n",targetImage->fname);
    printf("\t%ix%ix%i voxels, %i volumes\n",targetImage->nx,targetImage->ny,targetImage->nz,targetImage->nt);
	printf("\t%gx%gx%g mm\n",targetImage->dx,targetImage->dy,targetImage->dz);
	printf("Source image name: %s\n",sourceImage->fname);
    printf("\t%ix%ix%i voxels, %i volumes\n",sourceImage->nx,sourceImage->ny,sourceImage->nz,sourceImage->nt);
	printf("\t%gx%gx%g mm\n",sourceImage->dx,sourceImage->dy,sourceImage->dz);
	printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n\n");

    /* *********************** */
    /* READ THE TRANSFORMATION */
    /* *********************** */
    nifti_image *controlPointImage = NULL;
    nifti_image *deformationFieldImage = NULL;
    mat44 *affineTransformationMatrix = (mat44 *)calloc(1,sizeof(mat44));
    if(flag->inputCPPFlag){
#ifndef NDEBUG
        printf("Name of the control point image: %s\n", param->inputCPPName);
#endif
        controlPointImage = nifti_image_read(param->inputCPPName,true);
        if(controlPointImage == NULL){
            fprintf(stderr,"** ERROR Error when reading the control point image: %s\n",param->inputCPPName);
            return 1;
        }
        reg_checkAndCorrectDimension(controlPointImage);
    }
    else if(flag->inputDEFFlag){
#ifndef NDEBUG
        printf("Name of the deformation field image: %s\n", param->inputDEFName);
#endif
        deformationFieldImage = nifti_image_read(param->inputDEFName,true);
        if(deformationFieldImage == NULL){
            fprintf(stderr,"** ERROR Error when reading the deformation field image: %s\n",param->inputDEFName);
            return 1;
        }
        reg_checkAndCorrectDimension(deformationFieldImage);
    }
    else if(flag->affineMatrixFlag){
#ifndef NDEBUG
        printf("Name of affine transformation: %s\n", param->affineMatrixName);
#endif
        // Check first if the specified affine file exist
        if(FILE *aff=fopen(param->affineMatrixName, "r")){
            fclose(aff);
        }
        else{
            fprintf(stderr,"The specified input affine file (%s) can not be read\n",param->affineMatrixName);
            return 1;
        }
        reg_tool_ReadAffineFile(	affineTransformationMatrix,
                                    targetImage,
                                    sourceImage,
                                    param->affineMatrixName,
                                    flag->affineFlirtFlag);
    }
    else{
        // identity transformation is considered
        affineTransformationMatrix->m[0][0]=1.0;
        affineTransformationMatrix->m[1][1]=1.0;
        affineTransformationMatrix->m[2][2]=1.0;
        affineTransformationMatrix->m[3][3]=1.0;
    }

    // Allocate and copmute the deformation field if necessary
    if(!flag->inputDEFFlag){
#ifndef NDEBUG
        printf("Allocation of the deformation field\n");
#endif
        // Allocate
        deformationFieldImage = nifti_copy_nim_info(targetImage);
        deformationFieldImage->dim[0]=deformationFieldImage->ndim=5;
        deformationFieldImage->dim[1]=deformationFieldImage->nx=targetImage->nx;
        deformationFieldImage->dim[2]=deformationFieldImage->ny=targetImage->ny;
        deformationFieldImage->dim[3]=deformationFieldImage->nz=targetImage->nz;
        deformationFieldImage->dim[4]=deformationFieldImage->nt=1;deformationFieldImage->pixdim[4]=deformationFieldImage->dt=1.0;
        if(targetImage->nz>1) deformationFieldImage->dim[5]=deformationFieldImage->nu=3;
        else deformationFieldImage->dim[5]=deformationFieldImage->nu=2;
        deformationFieldImage->pixdim[5]=deformationFieldImage->du=1.0;
        deformationFieldImage->dim[6]=deformationFieldImage->nv=1;deformationFieldImage->pixdim[6]=deformationFieldImage->dv=1.0;
        deformationFieldImage->dim[7]=deformationFieldImage->nw=1;deformationFieldImage->pixdim[7]=deformationFieldImage->dw=1.0;
        deformationFieldImage->nvox=deformationFieldImage->nx*deformationFieldImage->ny*deformationFieldImage->nz*deformationFieldImage->nt*deformationFieldImage->nu;
        deformationFieldImage->datatype = NIFTI_TYPE_FLOAT32;
        deformationFieldImage->nbyper = sizeof(float);
        deformationFieldImage->data = (void *)calloc(deformationFieldImage->nvox, deformationFieldImage->nbyper);
        //Computation
        if(flag->inputCPPFlag){
#ifndef NDEBUG
            printf("Computation of the deformation field from the CPP image\n");
#endif
            reg_bspline<float>(	controlPointImage,
                                targetImage,
                                deformationFieldImage,
                                NULL,
                                0);
        }
        else{
#ifndef NDEBUG
            printf("Computation of the deformation field from the affine transformation\n");
#endif
            reg_affine_positionField(   affineTransformationMatrix,
                                        targetImage,
                                        deformationFieldImage);
        }
    }

    /* ************************* */
    /* RESAMPLE THE SOURCE IMAGE */
    /* ************************* */
    if(flag->outputResultFlag){
        int inter=3;
        if(flag->TRIInterpolationFlag) inter=1;
        else if(flag->NNInterpolationFlag) inter=0;

        nifti_image *resultImage = nifti_copy_nim_info(targetImage);
        resultImage->dim[0]=resultImage->ndim=sourceImage->dim[0];
        resultImage->dim[4]=resultImage->nt=sourceImage->dim[4];
        resultImage->cal_min=sourceImage->cal_min;
        resultImage->cal_max=sourceImage->cal_max;
        resultImage->scl_slope=sourceImage->scl_slope;
        resultImage->scl_inter=sourceImage->scl_inter;
        resultImage->datatype = sourceImage->datatype;
        resultImage->nbyper = sourceImage->nbyper;
        resultImage->nvox = resultImage->dim[1] * resultImage->dim[2] * resultImage->dim[3] * resultImage->dim[4];
        resultImage->data = (void *)calloc(resultImage->nvox, resultImage->nbyper);
        reg_resampleSourceImage<double>(targetImage,
                                        sourceImage,
                                        resultImage,
                                        deformationFieldImage,
                                        NULL,
                                        inter,
                                        0);
        nifti_set_filenames(resultImage, param->outputResultName, 0, 0);
        nifti_image_write(resultImage);
        printf("Resampled image has been saved: %s\n", param->outputResultName);
        nifti_image_free(resultImage);
    }

    /* *********************** */
    /* RESAMPLE A REGULAR GRID */
    /* *********************** */
    if(flag->outputBlankFlag){
        nifti_image *gridImage = nifti_copy_nim_info(sourceImage);
        gridImage->cal_min=0;
        gridImage->cal_max=255;
        gridImage->datatype = NIFTI_TYPE_UINT8;
        gridImage->nbyper = sizeof(unsigned char);
        gridImage->data = (void *)calloc(gridImage->nvox, gridImage->nbyper);
        unsigned char *gridImageValuePtr = static_cast<unsigned char *>(gridImage->data);
        for(int z=0; z<gridImage->nz;z++){
            for(int y=0; y<gridImage->ny;y++){
                for(int x=0; x<gridImage->nx;x++){
                    if(targetImage->nz>1){
                        if( x/10==(float)x/10.0 || y/10==(float)y/10.0 || z/10==(float)z/10.0)
                            *gridImageValuePtr = 255;
                    }
                    else{
                        if( x/10==(float)x/10.0 || x==targetImage->nx-1 || y/10==(float)y/10.0 || y==targetImage->ny-1)
                            *gridImageValuePtr = 255;
                    }
                    gridImageValuePtr++;
                }
            }
        }

        nifti_image *resultImage = nifti_copy_nim_info(targetImage);
        resultImage->dim[0]=resultImage->ndim=3;
        resultImage->dim[4]=resultImage->nt=1;
        resultImage->cal_min=sourceImage->cal_min;
        resultImage->cal_max=sourceImage->cal_max;
        resultImage->scl_slope=sourceImage->scl_slope;
        resultImage->scl_inter=sourceImage->scl_inter;
        resultImage->datatype =NIFTI_TYPE_UINT8;
        resultImage->nbyper = sizeof(unsigned char);
        resultImage->data = (void *)calloc(resultImage->nvox, resultImage->nbyper);
        reg_resampleSourceImage<double>(targetImage,
                                        gridImage,
                                        resultImage,
                                        deformationFieldImage,
                                        NULL,
                                        1,
                                       0);
        nifti_set_filenames(resultImage, param->outputBlankName, 0, 0);
        nifti_image_write(resultImage);
        nifti_image_free(resultImage);
        nifti_image_free(gridImage);
        printf("Resampled grid has been saved: %s\n", param->outputBlankName);
    }

    nifti_image_free(targetImage);
    nifti_image_free(sourceImage);
    nifti_image_free(controlPointImage);
    nifti_image_free(deformationFieldImage);
    free(affineTransformationMatrix);

	
	free(flag);
	free(param);
	
	return 0;
}

#endif
