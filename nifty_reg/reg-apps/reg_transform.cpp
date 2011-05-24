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
#include "_reg_globalTransformation.h"
#include "_reg_localTransformation.h"
#include "_reg_tools.h"
// #include "_reg_thinPlateSpline.h"

#ifdef _USE_NR_DOUBLE
    #define PrecisionTYPE double
#else
    #define PrecisionTYPE float
#endif

typedef struct{
    char *targetImageName;
    char *sourceImageName;

    char *inputSourceImageName;
    char *inputAffineName;
    char *inputFirstCPPName;
    char *inputSecondCPPName;
    char *inputDeformationName;
    char *inputDisplacementName;
    char *outputSourceImageName;
    char *outputDeformationName;
    char *outputDisplacementName;
    char *outputAffineName;
    char *cpp2defInputName;
    char *cpp2defOutputName;
    char *outputVelName;
}PARAM;
typedef struct{
    bool targetImageFlag;
    bool cpp2defFlag;
    bool composeTransformation1Flag;
    bool composeTransformation2Flag;
    bool def2dispFlag;
    bool disp2defFlag;
    bool updateSformFlag;
    bool aff2defFlag;
    bool invertAffineFlag;
    bool invertVelFlag;
    bool invertDefFlag;
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
    printf("Usage:\t%s -target <filename> [OPTIONS].\n",exec);
        printf("\t-target <filename>\tFilename of the target image (mandatory)\n");
	
    printf("\n* * OPTIONS * *\n");
    printf("\t-cpp2def <filename1>  <filename2>\n");
        printf("\t\tConversion from control point position to deformation field.\n");
        printf("\t\tFilename1 of input lattice of control point positions (CPP).\n");
        printf("\t\tFilename2 of the output deformation field image (DEF).\n");
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
        printf("\t\tFilename2: deformation field x'=T(x)\n");
    printf("\t-aff2def <filename1> <filename2> <filename3> <filename3>\n");
        printf("\t\tApply a non-rigid deformation (stored as CPP) to an initial affine transformation.\n\t\tReturns a deformation field.\n");
        printf("\t\tFilename1: Affine transformation defined as Affine x Reference = Floating\n");
        printf("\t\tFilename2: Image used as a target for the non-rigid step.\n");
        printf("\t\tFilename3: Control point position filename\n");
        printf("\t\tFilename4: Updated deformation field.\n");
    printf("\t-invAffine <filename1> <filename2>\n");
        printf("\t\tInvert an affine transformation matrix\n");
        printf("\t\tFilename1: Input affine matrix\n");
        printf("\t\tFilename2: Inverted affine matrix\n");
//    printf("\t-invDef <filename1> <filename2> <filename3>\n");
//        printf("\t\tInvert a deformation field\n");
//        printf("\t\tFilename1: Input deformation field filename\n");
//        printf("\t\tFilename2: Source image filename\n");
//        printf("\t\tFilename3: output deformation field filename\n");
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
        else if(strcmp(argv[i], "-aff2def") == 0){
            param->inputAffineName=argv[++i];
            param->sourceImageName=argv[++i];
            param->inputFirstCPPName=argv[++i];
            param->outputDeformationName=argv[++i];
            flag->aff2defFlag=1;
        }
        else if(strcmp(argv[i], "-invAffine") == 0){
            param->inputAffineName=argv[++i];
            param->outputAffineName=argv[++i];
            flag->invertAffineFlag=1;
        }
        else if(strcmp(argv[i], "-invDef") == 0){
            param->inputDeformationName=argv[++i];
            param->sourceImageName=argv[++i];
            param->outputDeformationName=argv[++i];
            flag->invertDefFlag=1;
        }
        else if(strcmp(argv[i], "-cpp2def") == 0){
            param->cpp2defInputName=argv[++i];
            param->cpp2defOutputName=argv[++i];
            flag->cpp2defFlag=1;
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
        fprintf(stderr,"[NiftyReg ERROR] Error when reading the target image: %s\n",param->targetImageName);
        PetitUsage(argv[0]);
        return 1;
    }
    reg_checkAndCorrectDimension(targetImage);

    /* *************************************** */
    /* GENERATE THE DEFORMATION FIELD FROM CPP */
    /* *************************************** */
    if(flag->cpp2defFlag){
        // Read the control point image
        nifti_image *controlPointImage = nifti_image_read(param->cpp2defInputName,true);
        if(controlPointImage == NULL){
            fprintf(stderr,"[NiftyReg ERROR] Error when reading the control point position image: %s\n",param->cpp2defInputName);
            PetitUsage(argv[0]);
            return 1;
        }
        reg_checkAndCorrectDimension(controlPointImage);
        // Allocate the deformation field
        nifti_image *deformationFieldImage = nifti_copy_nim_info(targetImage);
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
        deformationFieldImage->datatype = controlPointImage->datatype;
        deformationFieldImage->nbyper = controlPointImage->nbyper;
        deformationFieldImage->data = (void *)calloc(deformationFieldImage->nvox, deformationFieldImage->nbyper);
        //Computation of the deformation field
        if(controlPointImage->pixdim[5]>1)
            reg_getDeformationFieldFromVelocityGrid(controlPointImage,
                                                    deformationFieldImage,
                                                    NULL, //mask
                                                    false // approximation
                                                    );
         else
            reg_spline(controlPointImage,
                       targetImage,
                       deformationFieldImage,
                       NULL,
                       false, //composition
                       true // bspline
                       );
        nifti_image_free(controlPointImage);
        // Ouput the deformation field image
        nifti_set_filenames(deformationFieldImage, param->cpp2defOutputName, 0, 0);
        nifti_image_write(deformationFieldImage);
        printf("The deformation field has been saved: %s\n", param->cpp2defOutputName);
        nifti_image_free(deformationFieldImage);
    }

    /* ********************* */
    /* START THE COMPOSITION */
    /* ********************* */
	if(flag->composeTransformation1Flag || flag->composeTransformation2Flag){

            nifti_image *secondControlPointImage = nifti_image_read(param->inputSecondCPPName,true);
            if(secondControlPointImage == NULL){
            fprintf(stderr,"[NiftyReg ERROR] Error when reading the control point image: %s\n",param->inputSecondCPPName);
			PetitUsage(argv[0]);
			return 1;
        }
        reg_checkAndCorrectDimension(secondControlPointImage);

        // Here should be a check for the control point image. Does it suit the target image space.
        //TODO

        // Check if the input deformation can be read
        nifti_image *deformationFieldImage = NULL;

        if(flag->composeTransformation1Flag){
            // Read the initial deformation control point grid
            nifti_image *firstControlPointImage = nifti_image_read(param->inputFirstCPPName,true);
            if(firstControlPointImage == NULL){
                fprintf(stderr,"[NiftyReg ERROR] Error when reading the control point image: %s\n",param->inputFirstCPPName);
                PetitUsage(argv[0]);
                return 1;
            }
            reg_checkAndCorrectDimension(firstControlPointImage);

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
            reg_spline(firstControlPointImage,
                       targetImage,
                       deformationFieldImage,
                       NULL,
                       false, //composition
                       true // bspline
                       );
            nifti_image_free(firstControlPointImage);
    }
    else{
                    // Read the deformation field
                    deformationFieldImage = nifti_image_read(param->inputDeformationName,true);
                    if(deformationFieldImage == NULL){
                        fprintf(stderr,"[NiftyReg ERROR] Error when reading the input deformation field image: %s\n",param->inputDeformationName);
                        PetitUsage(argv[0]);
                        return 1;
                    }
                    reg_checkAndCorrectDimension(deformationFieldImage);
		}
		
		// Are the deformation field and the target image defined in the same space
		//TODO
		
		// The deformation field is updated through composition
                reg_spline(secondControlPointImage,
                           targetImage,
                           deformationFieldImage,
                           NULL,
                           true, //composition
                           true // bspline
                           );
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
                fprintf(stderr,"[NiftyReg ERROR] Error when reading the input displacement field image: %s\n",param->inputDisplacementName);
                PetitUsage(argv[0]);
                return 1;
            }
            reg_checkAndCorrectDimension(image);
            // Conversion from displacement to deformation
            reg_getDeformationFromDisplacement(image);
		
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
                fprintf(stderr,"[NiftyReg ERROR] Error when reading the input deformation field image: %s\n",param->inputDeformationName);
                PetitUsage(argv[0]);
                return 1;
            }
            reg_checkAndCorrectDimension(image);
            // Conversion from displacement to deformation
            reg_getDisplacementFromDeformation(image);

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
                fprintf(stderr,"[NiftyReg ERROR] Error when reading the input image: %s\n",
                        param->inputSourceImageName);
                PetitUsage(argv[0]);
                return 1;
            }
            reg_checkAndCorrectDimension(image);
            // Read the affine transformation
            mat44 *affineTransformation = (mat44 *)calloc(1,sizeof(mat44));
            reg_tool_ReadAffineFile(affineTransformation,
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
            reg_tool_ReadAffineFile(affineTransformation,
                                    targetImage,
                                    targetImage,
                                    param->inputAffineName,
                                    0);

            // Invert the matrix
            *affineTransformation = nifti_mat44_inverse(*affineTransformation);

            // Output the new affine transformation
            reg_tool_WriteAffineFile(affineTransformation,
                                     param->outputAffineName);
            free(affineTransformation);
	}

//    /* ************************ */
//    /* INVERT DEFORMATION FIELD */
//    /* ************************ */
//    if(flag->invertDefFlag){
//        // Read the input deformation field
//        nifti_image *deformationField = nifti_image_read(param->inputDeformationName,true);
//        if(deformationField == NULL){
//            fprintf(stderr,"[NiftyReg ERROR] Error when reading the input deformation field image: %s\n",param->inputDeformationName);
//            PetitUsage(argv[0]);
//            return 1;
//        }
//        reg_checkAndCorrectDimension(deformationField);
//
//        nifti_image *sourceImage = nifti_image_read(param->sourceImageName,false);
//        if(sourceImage == NULL){
//            fprintf(stderr,"[NiftyReg ERROR] Error when reading the source image: %s\n",param->sourceImageName);
//            PetitUsage(argv[0]);
//            return 1;
//        }
//        reg_checkAndCorrectDimension(sourceImage);
//
//        thinPlateSpline *tps=NULL;
//        reg_initThinPlateSplineFromDeformationField(deformationField,
//                                                    tps,
//                                                    false);
//
//        nifti_image *outputDeformationFieldImage = nifti_copy_nim_info(sourceImage);
//        outputDeformationFieldImage->cal_min=0;
//        outputDeformationFieldImage->cal_max=0;
//        outputDeformationFieldImage->scl_slope = 1.0f;
//        outputDeformationFieldImage->scl_inter = 0.0f;
//        outputDeformationFieldImage->dim[0]=outputDeformationFieldImage->ndim=5;
//        outputDeformationFieldImage->dim[1]=outputDeformationFieldImage->nx=sourceImage->nx;
//        outputDeformationFieldImage->dim[2]=outputDeformationFieldImage->ny=sourceImage->ny;
//        outputDeformationFieldImage->dim[3]=outputDeformationFieldImage->nz=sourceImage->nz;
//        outputDeformationFieldImage->dim[4]=outputDeformationFieldImage->nt=1;outputDeformationFieldImage->pixdim[4]=outputDeformationFieldImage->dt=1.0;
//        if(targetImage->nz>1)
//            outputDeformationFieldImage->dim[5]=outputDeformationFieldImage->nu=3;
//        else outputDeformationFieldImage->dim[5]=outputDeformationFieldImage->nu=2;
//        outputDeformationFieldImage->pixdim[5]=outputDeformationFieldImage->du=1.0;
//        outputDeformationFieldImage->dim[6]=outputDeformationFieldImage->nv=1;
//        outputDeformationFieldImage->pixdim[6]=outputDeformationFieldImage->dv=1.0;
//        outputDeformationFieldImage->dim[7]=outputDeformationFieldImage->nw=1;
//        outputDeformationFieldImage->pixdim[7]=outputDeformationFieldImage->dw=1.0;
//        outputDeformationFieldImage->nvox=outputDeformationFieldImage->nx*outputDeformationFieldImage->ny*
//                                          outputDeformationFieldImage->nz*outputDeformationFieldImage->nt*
//                                          outputDeformationFieldImage->nu;
//        if(sizeof(PrecisionTYPE)==4) outputDeformationFieldImage->datatype = NIFTI_TYPE_FLOAT32;
//        else outputDeformationFieldImage->datatype = NIFTI_TYPE_FLOAT64;
//        outputDeformationFieldImage->nbyper = sizeof(PrecisionTYPE);
//        outputDeformationFieldImage->data = (void *)calloc(outputDeformationFieldImage->nvox, outputDeformationFieldImage->nbyper);
//
//        reg_TPSfillDeformationField(tps, outputDeformationFieldImage);
//
//        nifti_set_filenames(outputDeformationFieldImage,param->outputDeformationName,0,0);
//        nifti_image_write(outputDeformationFieldImage);
//        nifti_image_free(sourceImage);
//        nifti_image_free(deformationField);
//        nifti_image_free(outputDeformationFieldImage);
//    }

    /* *************************** */
    /* APPLY AFFINE TO DEFORMATION */
    /* *************************** */
    if(flag->aff2defFlag){
        // Read the input control point position
        nifti_image *controlPointPosition = nifti_image_read(param->inputFirstCPPName,true);
        if(controlPointPosition == NULL){
            fprintf(stderr,"[NiftyReg ERROR] Error when reading the input control point position image: %s\n",param->inputFirstCPPName);
            PetitUsage(argv[0]);
            return 1;
        }
        reg_checkAndCorrectDimension(controlPointPosition);

        // Read the input control point position
        nifti_image *targetImage2 = nifti_image_read(param->sourceImageName,true);
        if(targetImage2 == NULL){
            fprintf(stderr,"[NiftyReg ERROR] Error when reading the nrr-step target image: %s\n",param->sourceImageName);
            PetitUsage(argv[0]);
            return 1;
        }
        reg_checkAndCorrectDimension(targetImage2);

        // Read the affine transformation
        mat44 *affineTransformation = (mat44 *)calloc(1,sizeof(mat44));
        reg_tool_ReadAffineFile(affineTransformation,
                                targetImage,
                                targetImage2,
                                param->inputAffineName,
                                0);

        reg_mat44_disp(affineTransformation, (char *)"affineTransformation");

        // An initial deformation field is created
        nifti_image *deformationFieldImage = nifti_copy_nim_info(targetImage);
        deformationFieldImage->cal_min=0;
        deformationFieldImage->cal_max=0;
        deformationFieldImage->scl_slope = 1.0f;
        deformationFieldImage->scl_inter = 0.0f;
        deformationFieldImage->dim[0]=deformationFieldImage->ndim=5;
        deformationFieldImage->dim[1]=deformationFieldImage->nx=targetImage->nx;
        deformationFieldImage->dim[2]=deformationFieldImage->ny=targetImage->ny;
        deformationFieldImage->dim[3]=deformationFieldImage->nz=targetImage->nz;
        deformationFieldImage->dim[4]=deformationFieldImage->nt=1;
        deformationFieldImage->pixdim[4]=deformationFieldImage->dt=1.0;
        if(targetImage->nz>1)
        deformationFieldImage->dim[5]=deformationFieldImage->nu=3;
        else deformationFieldImage->dim[5]=deformationFieldImage->nu=2;
        deformationFieldImage->pixdim[5]=deformationFieldImage->du=1.0;
        deformationFieldImage->dim[6]=deformationFieldImage->nv=1;
        deformationFieldImage->pixdim[6]=deformationFieldImage->dv=1.0;
        deformationFieldImage->dim[7]=deformationFieldImage->nw=1;
        deformationFieldImage->pixdim[7]=deformationFieldImage->dw=1.0;
        deformationFieldImage->nvox=deformationFieldImage->nx*deformationFieldImage->ny
                                    *deformationFieldImage->nz*deformationFieldImage->nt
                                    *deformationFieldImage->nu;
        if(sizeof(PrecisionTYPE)==4) deformationFieldImage->datatype = NIFTI_TYPE_FLOAT32;
        else deformationFieldImage->datatype = NIFTI_TYPE_FLOAT64;
        deformationFieldImage->nbyper = sizeof(PrecisionTYPE);
        deformationFieldImage->data = (void *)calloc(deformationFieldImage->nvox,
                                                     deformationFieldImage->nbyper);

        // The deformation field is field with the affine transformation
        reg_affine_positionField(affineTransformation, targetImage, deformationFieldImage);

        // The deformation field is composed with the CPP file
        reg_spline(controlPointPosition,
                   targetImage2,
                   deformationFieldImage,
                   NULL,
                   true, //composition
                   true // bspline
                   );

        nifti_set_filenames(deformationFieldImage, param->outputDeformationName, 0, 0);
        nifti_image_write(deformationFieldImage);
        nifti_image_free(deformationFieldImage);
        nifti_image_free(controlPointPosition);
        nifti_image_free(targetImage2);
    }

	nifti_image_free(targetImage);
	
	free(flag);
	free(param);
	
	return 0;
}

#endif
