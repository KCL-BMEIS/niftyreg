/*
 *  reg_transform.cpp
 *
 *
 *  Created by Marc Modat on 08/11/2010.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _MM_TRANSFORM_CPP
#define _MM_TRANSFORM_CPP

#include "_reg_resampling.h"
#include "_reg_affineTransformation.h"
#include "_reg_bspline.h"
#include "_reg_bspline_comp.h"
#include "_reg_tools.h"

#define PrecisionTYPE float

typedef struct{
	char *targetImageName;
	
	char *inputAffineName;
	char *inputSourceImageName;
	char *inputFirstCPPName;
	char *inputSecondCPPName;
	char *inputDeformationName;
	char *inputDisplacementName;
	char *outputSourceImageName;
	char *outputDeformationName;
	char *outputDisplacementName;
	char *outputAffineName;
}PARAM;
typedef struct{
	bool targetImageFlag;
	bool composeTransformation1Flag;
	bool composeTransformation2Flag;
	bool def2dispFlag;
	bool disp2defFlag;
	bool updateSformFlag;
	bool invertAffineFlag;
}FLAG;


void PetitUsage(char *exec)
{
	fprintf(stderr,"Usage:\t%s -target <targetImageName> [OPTIONS].\n",exec);
	fprintf(stderr,"\tSee the help for more details (-h).\n");
	return;
}
void Usage(char *exec)
{
	printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
	printf("Usage:\t%s -target <filename> -source <filename> [OPTIONS].\n",exec);
	printf("\t-target <filename>\tFilename of the target image (mandatory)\n");
	
	printf("\n* * OPTIONS * *\n");
	printf("\t-comp1 <filename1>  <filename2> <filename3>\n");
		printf("\t\tComposition of two lattices of control points. CPP2(CPP1(x)).\n");
		printf("\t\tFilename1 of lattice of control point that contains the second deformation (CPP2).\n");
		printf("\t\tFilename2 of lattice of control point that contains the initial deformation (CPP1).\n");
		printf("\t\tFilename3 of the output deformation field.\n");
	printf("\t-comp2 <filename1>  <filename2> <filename3>\n");
		printf("\t\tComposition of a deformation field with a lattice of control points. CPP(DEF(x)).\n");
		printf("\t\tFilename1 of lattice of control point that contains the second deformation (CPP).\n");
		printf("\t\tFilename2 of the deformation field to be used as initial deformation (DEF).\n");
		printf("\t\tFilename3 of the output deformation field.\n");
	printf("\t-def2disp <filename1>  <filename2>\n");
		printf("\t\tConvert a deformation field into a displacement field.\n");
		printf("\t\tFilename1: deformation field x'=T(x)\n");
		printf("\t\tFilename2: displacement field x'=x+T(x)\n");
	printf("\t-disp2def <filename1>  <filename2>\n");
		printf("\t\tConvert a displacement field into a deformation field.\n");
		printf("\t\tFilename1: displacement field x'=x+T(x)\n");
	printf("\t\tFilename2: deformation field x'=T(x)\n");
		printf("\t-updSform <filename1> <filename2> <filename3>\n");
		printf("\t\tUpdate the sform of a Floating (Source) image using an affine transformation.\n");
		printf("\t\tFilename1: Image to be updated\n");
		printf("\t\tFilename2: Affine transformation defined as Affine x Reference = Floating\n");
		printf("\t\tFilename3: Updated image.\n");
	printf("\t-invAffine <filename1> <filename2>\n");
		printf("\t\tInvert an affine transformation matrix\n");
		printf("\t\tFilename1: Input affine matrix\n");
		printf("\t\tFilename2: Inverted affine matrix\n");
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
		else if(strcmp(argv[i], "-comp1") == 0){
			param->inputSecondCPPName=argv[++i];
			param->inputFirstCPPName=argv[++i];
			param->outputDeformationName=argv[++i];
			flag->composeTransformation1Flag=1;
		}
		else if(strcmp(argv[i], "-comp2") == 0){
			param->inputSecondCPPName=argv[++i];
			param->inputDeformationName=argv[++i];
			param->outputDeformationName=argv[++i];
			flag->composeTransformation2Flag=1;
		}
		else if(strcmp(argv[i], "-def2disp") == 0){
			param->inputDeformationName=argv[++i];
			param->outputDisplacementName=argv[++i];
			flag->def2dispFlag=1;
		}
		else if(strcmp(argv[i], "-disp2def") == 0){
			param->inputDisplacementName=argv[++i];
			param->outputDeformationName=argv[++i];
			flag->disp2defFlag=1;
		}
		else if(strcmp(argv[i], "-updSform") == 0){
			param->inputSourceImageName=argv[++i];
			param->inputAffineName=argv[++i];
			param->outputSourceImageName=argv[++i];
			flag->updateSformFlag=1;
		}
		else if(strcmp(argv[i], "-invAffine") == 0){
			param->inputAffineName=argv[++i];
			param->outputAffineName=argv[++i];
			flag->invertAffineFlag=1;
		}
		else{
			fprintf(stderr,"Err:\tParameter %s unknown.\n",argv[i]);
			PetitUsage(argv[0]);
			return 1;
		}
	}
	
	if(!flag->targetImageFlag){
		fprintf(stderr,"Err:\tThe target image has to be defined.\n");
		PetitUsage(argv[0]);
		return 1;
	}

	/* Read the target image */
	nifti_image *targetImage = nifti_image_read(param->targetImageName,false);
	if(targetImage == NULL){
		fprintf(stderr,"** ERROR Error when reading the target image: %s\n",param->targetImageName);
		PetitUsage(argv[0]);
		return 1;
	}
	
	/* ********************* */
	/* START THE COMPOSITION */
	/* ********************* */
	if(flag->composeTransformation1Flag || flag->composeTransformation2Flag){

		nifti_image *secondControlPointImage = nifti_image_read(param->inputSecondCPPName,true);
		if(secondControlPointImage == NULL){
			fprintf(stderr,"** ERROR Error when reading the control point image: %s\n",param->inputSecondCPPName);
			PetitUsage(argv[0]);
			return 1;
		}

		// Here should be a check for the control point image. Does it suit the target image space.
		//TODO

		// Check if the input deformation can be read
		nifti_image *deformationFieldImage = NULL;

		if(flag->composeTransformation1Flag){
			// Read the initial deformation control point grid
			nifti_image *firstControlPointImage = nifti_image_read(param->inputFirstCPPName,true);
			if(firstControlPointImage == NULL){
				fprintf(stderr,"** ERROR Error when reading the control point image: %s\n",param->inputFirstCPPName);
				PetitUsage(argv[0]);
				return 1;
			}

			// Create the deformation field image
			deformationFieldImage = nifti_copy_nim_info(targetImage);
			deformationFieldImage->cal_min=0;
			deformationFieldImage->cal_max=0;
			deformationFieldImage->scl_slope = 1.0f;
			deformationFieldImage->scl_inter = 0.0f;
			deformationFieldImage->dim[0]=deformationFieldImage->ndim=5;
			deformationFieldImage->dim[1]=deformationFieldImage->nx=targetImage->nx;
			deformationFieldImage->dim[2]=deformationFieldImage->ny=targetImage->ny;
			deformationFieldImage->dim[3]=deformationFieldImage->nz=targetImage->nz;
			deformationFieldImage->dim[4]=deformationFieldImage->nt=1;deformationFieldImage->pixdim[4]=deformationFieldImage->dt=1.0;
			if(targetImage->nz>1)
				deformationFieldImage->dim[5]=deformationFieldImage->nu=3;
			else deformationFieldImage->dim[5]=deformationFieldImage->nu=2;
			deformationFieldImage->pixdim[5]=deformationFieldImage->du=1.0;
			deformationFieldImage->dim[6]=deformationFieldImage->nv=1;deformationFieldImage->pixdim[6]=deformationFieldImage->dv=1.0;
			deformationFieldImage->dim[7]=deformationFieldImage->nw=1;deformationFieldImage->pixdim[7]=deformationFieldImage->dw=1.0;
			deformationFieldImage->nvox=deformationFieldImage->nx*deformationFieldImage->ny*deformationFieldImage->nz*deformationFieldImage->nt*deformationFieldImage->nu;
			if(sizeof(PrecisionTYPE)==4) deformationFieldImage->datatype = NIFTI_TYPE_FLOAT32;
			else deformationFieldImage->datatype = NIFTI_TYPE_FLOAT64;
			deformationFieldImage->nbyper = sizeof(PrecisionTYPE);
			deformationFieldImage->data = (void *)calloc(deformationFieldImage->nvox, deformationFieldImage->nbyper);
			
			//Compute the initial deformation
			reg_bspline<PrecisionTYPE>(	firstControlPointImage,
										targetImage,
										deformationFieldImage,
										NULL,
									   0);
			nifti_image_free(firstControlPointImage);
		}
		else{
			// Read the deformation field
			deformationFieldImage = nifti_image_read(param->inputDeformationName,true);
			if(deformationFieldImage == NULL){
				fprintf(stderr,"** ERROR Error when reading the input deformation field image: %s\n",param->inputDeformationName);
				PetitUsage(argv[0]);
				return 1;
			}
		}
		
		// Are the deformation field and the target image defined in the same space
		//TODO
		
		// The deformation field is updated through composition
		reg_bspline<PrecisionTYPE>(	secondControlPointImage,
									targetImage,
									deformationFieldImage,
									NULL,
									2);
		nifti_image_free(secondControlPointImage);
		
		// Ouput the composed deformation field
		nifti_set_filenames(deformationFieldImage, param->outputDeformationName, 0, 0);
        nifti_image_write(deformationFieldImage);
        nifti_image_free(deformationFieldImage);
        printf("Composed deformation field has been saved: %s\n", param->outputDeformationName);
	}
	
	/* ******************** */
	/* START THE CONVERSION */
	/* ******************** */
	if(flag->disp2defFlag){
		// Read the input displacement field
		nifti_image *image = nifti_image_read(param->inputDisplacementName,true);
		if(image == NULL){
			fprintf(stderr,"** ERROR Error when reading the input displacement field image: %s\n",param->inputDisplacementName);
			PetitUsage(argv[0]);
			return 1;
		}
		// Conversion from displacement to deformation
		reg_getPositionFromDisplacement<PrecisionTYPE>(image);
		
		// Ouput the deformation field
		nifti_set_filenames(image, param->outputDeformationName, 0, 0);
        nifti_image_write(image);
        nifti_image_free(image);
        printf("The deformation field has been saved: %s\n", param->outputDeformationName);
	}
	if(flag->def2dispFlag){
		// Read the input deformation field
		nifti_image *image = nifti_image_read(param->inputDeformationName,true);
		if(image == NULL){
			fprintf(stderr,"** ERROR Error when reading the input deformation field image: %s\n",param->inputDeformationName);
			PetitUsage(argv[0]);
			return 1;
		}
		// Conversion from displacement to deformation
		reg_getDisplacementFromPosition<PrecisionTYPE>(image);
		
		// Ouput the deformation field
		nifti_set_filenames(image, param->outputDisplacementName, 0, 0);
        nifti_image_write(image);
        nifti_image_free(image);
        printf("The displacement field has been saved: %s\n", param->outputDisplacementName);
	}
	
	/* ******************* */
	/* UPDATE IMAGE S_FORM */
	/* ******************* */
	if(flag->updateSformFlag){
		// Read the input image
		nifti_image *image = nifti_image_read(param->inputSourceImageName,true);
		if(image == NULL){
			fprintf(stderr,"** ERROR Error when reading the input image: %s\n",param->inputSourceImageName);
			PetitUsage(argv[0]);
			return 1;
		}
		// Read the affine transformation
		mat44 *affineTransformation = (mat44 *)calloc(1,sizeof(mat44));
		reg_tool_ReadAffineFile(	affineTransformation,
									targetImage,
									image,
									param->inputAffineName,
									0);
		//Invert the affine transformation since the flaoting is updated
		*affineTransformation = nifti_mat44_inverse(*affineTransformation);

		// Update the sform
		if(image->sform_code>0){
			image->sto_xyz = reg_mat44_mul(affineTransformation, &(image->sto_xyz));
		}
		else{
			image->sform_code = 1;
			image->sto_xyz = reg_mat44_mul(affineTransformation, &(image->qto_xyz));
		}
		image->sto_ijk = nifti_mat44_inverse(image->sto_xyz);
		free(affineTransformation);
		
		// Write the output image
		nifti_set_filenames(image, param->outputSourceImageName, 0, 0);
        nifti_image_write(image);
        nifti_image_free(image);
        printf("The sform has been updated and the image saved: %s\n", param->outputSourceImageName);
	}
	
	/* **************************** */
	/* INVERT AFFINE TRANSFORMATION */
	/* **************************** */
	if(flag->invertAffineFlag){
		// First read the affine transformation from a file
		mat44 *affineTransformation = (mat44 *)calloc(1,sizeof(mat44));
		reg_tool_ReadAffineFile(	affineTransformation,
									targetImage,
									targetImage,
									param->inputAffineName,
									0);
		
		// Invert the matrix
		*affineTransformation = nifti_mat44_inverse(*affineTransformation);
		
		// Output the new affine transformation
		reg_tool_WriteAffineFile(	affineTransformation,
									param->outputAffineName);
		free(affineTransformation);
	}
	
	nifti_image_free(targetImage);
	
	free(flag);
	free(param);
	
	return 0;
}

#endif