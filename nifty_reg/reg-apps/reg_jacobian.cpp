/*
 *  reg_jacobian.cpp
 *
 *
 *  Created by Marc Modat on 15/11/2010.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _MM_JACOBIAN_CPP
#define _MM_JACOBIAN_CPP

#include "_reg_globalTransformation.h"
#include "_reg_localTransformation.h"
#include "_reg_tools.h"
#include "_reg_resampling.h"

#ifdef _USE_NR_DOUBLE
    #define PrecisionTYPE double
#else
    #define PrecisionTYPE float
#endif

typedef struct{
    char *referenceImageName;
    char *inputDEFName;
    char *inputCPPName;
    char *jacobianMapName;
    char *jacobianMatrixName;
    char *logJacobianMapName;
}PARAM;
typedef struct{
    bool referenceImageFlag;
    bool inputDEFFlag;
    bool inputCPPFlag;
    bool jacobianMapFlag;
    bool jacobianMatrixFlag;
    bool logJacobianMapFlag;
}FLAG;


void PetitUsage(char *exec)
{
    fprintf(stderr,"Usage:\t%s -target <referenceImage> [OPTIONS].\n",exec);
    fprintf(stderr,"\tSee the help for more details (-h).\n");
    return;
}
void Usage(char *exec)
{
    printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
    printf("Usage:\t%s -target <filename> [OPTIONS].\n",exec);
    printf("\t-target <filename>\tFilename of the target image (mandatory)\n");

    printf("\n* * INPUT (Only one will be used) * *\n");
    printf("\t-def <filename>\n");
        printf("\t\tFilename of the deformation field (from reg_transform).\n");
    printf("\t-cpp <filename>\n");
        printf("\t\tFilename of the control point position lattice (from reg_f3d).\n");
    printf("\n* * OUTPUT * *\n");
    printf("\t-jac <filename>\n");
        printf("\t\tFilename of the Jacobian determinant map.\n");
    printf("\t-jacM <filename>\n");
        printf("\t\tFilename of the Jacobian matrix map. (9 values are stored as a 5D nifti).\n");
    printf("\t-jacL <filename>\n");
        printf("\t\tFilename of the Log of the Jacobian determinant map.\n");
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
            param->referenceImageName=argv[++i];
            flag->referenceImageFlag=1;
        }
        else if(strcmp(argv[i], "-def") == 0){
            param->inputDEFName=argv[++i];
            flag->inputDEFFlag=1;
        }
        else if(strcmp(argv[i], "-cpp") == 0){
            param->inputCPPName=argv[++i];
            flag->inputCPPFlag=1;
        }
        else if(strcmp(argv[i], "-jac") == 0){
            param->jacobianMapName=argv[++i];
            flag->jacobianMapFlag=1;
        }
        else if(strcmp(argv[i], "-jacM") == 0){
            param->jacobianMatrixName=argv[++i];
            flag->jacobianMatrixFlag=1;
        }
        else if(strcmp(argv[i], "-jacL") == 0){
            param->logJacobianMapName=argv[++i];
            flag->logJacobianMapFlag=1;
        }
         else{
             fprintf(stderr,"Err:\tParameter %s unknown.\n",argv[i]);
             PetitUsage(argv[0]);
             return 1;
         }
    }

    /* ************** */
    /* READ REFERENCE */
    /* ************** */
    nifti_image *image = nifti_image_read(param->referenceImageName,false);
    if(image == NULL){
        fprintf(stderr,"** ERROR Error when reading the target image: %s\n",param->referenceImageName);
        return 1;
    }
    reg_checkAndCorrectDimension(image);

    /* ******************* */
    /* READ TRANSFORMATION */
    /* ******************* */
    nifti_image *controlPointImage=NULL;
    nifti_image *deformationFieldImage=NULL;
    if(flag->inputCPPFlag){
        controlPointImage = nifti_image_read(param->inputCPPName,true);
        if(controlPointImage == NULL){
            fprintf(stderr,"** ERROR Error when reading the control point image: %s\n",param->inputCPPName);
            nifti_image_free(image);
            return 1;
        }
        reg_checkAndCorrectDimension(controlPointImage);
    }
    else if(flag->inputDEFFlag){
        deformationFieldImage = nifti_image_read(param->inputDEFName,true);
        if(deformationFieldImage == NULL){
            fprintf(stderr,"** ERROR Error when reading the deformation field image: %s\n",param->inputDEFName);
            nifti_image_free(image);
            return 1;
        }
        reg_checkAndCorrectDimension(deformationFieldImage);
    }
    else{
        fprintf(stderr, "No transformation has been provided.\n");
        nifti_image_free(image);
        return 1;
    }

    /* ******************** */
    /* COMPUTE JACOBIAN MAP */
    /* ******************** */
    if(flag->jacobianMapFlag || flag->logJacobianMapFlag){
        // Create first the Jacobian map image
        nifti_image *jacobianImage = nifti_copy_nim_info(image);
        jacobianImage->cal_min=0;
        jacobianImage->cal_max=0;
        jacobianImage->scl_slope = 1.0f;
        jacobianImage->scl_inter = 0.0f;
        if(sizeof(PrecisionTYPE)==8)
            jacobianImage->datatype = NIFTI_TYPE_FLOAT64;
        else jacobianImage->datatype = NIFTI_TYPE_FLOAT32;
        jacobianImage->nbyper = sizeof(PrecisionTYPE);
        jacobianImage->data = (void *)calloc(jacobianImage->nvox, jacobianImage->nbyper);

        // Compute the determinant
        if(flag->inputCPPFlag){
            if(controlPointImage->pixdim[5]>1){
                reg_bspline_GetJacobianMapFromVelocityField(controlPointImage,
                                                            jacobianImage);
            }
            else{
                reg_bspline_GetJacobianMap(controlPointImage,
                                           jacobianImage
                                           );
            }
        }
        else if(flag->inputDEFFlag){
            reg_defField_getJacobianMap(deformationFieldImage,
                                        jacobianImage);
        }
        else{
            fprintf(stderr, "No transformation has been provided.\n");
            nifti_image_free(image);
            return 1;
        }

        // Export the Jacobian determinant map
        if(flag->jacobianMapFlag){
            nifti_set_filenames(jacobianImage, param->jacobianMapName, 0, 0);
            memset(jacobianImage->descrip, 0, 80);
            strcpy (jacobianImage->descrip,"Jacobian determinant map created using NiftyReg");
            nifti_image_write(jacobianImage);
            printf("Jacobian map image has been saved: %s\n", param->jacobianMapName);
        }
        else if(flag->logJacobianMapFlag){
            PrecisionTYPE *jacPtr=static_cast<PrecisionTYPE *>(jacobianImage->data);
            for(unsigned int i=0;i<jacobianImage->nvox;i++){
                *jacPtr = log(*jacPtr);
                jacPtr++;
            }
            nifti_set_filenames(jacobianImage, param->logJacobianMapName, 0, 0);
            memset(jacobianImage->descrip, 0, 80);
            strcpy (jacobianImage->descrip,"Log Jacobian determinant map created using NiftyReg");
            nifti_image_write(jacobianImage);
            printf("Log Jacobian map image has been saved: %s\n", param->logJacobianMapName);
        }
        nifti_image_free(jacobianImage);
    }

    /* *********************** */
    /* COMPUTE JACOBIAN MATRIX */
    /* *********************** */
    if(flag->jacobianMatrixFlag){
        // Create first the Jacobian matrix image
        nifti_image *jacobianImage = nifti_copy_nim_info(image);
        jacobianImage->cal_min=0;
        jacobianImage->cal_max=0;
        jacobianImage->scl_slope = 1.0f;
        jacobianImage->scl_inter = 0.0f;
        jacobianImage->dim[0] = 5;
        if(image->nz>1) jacobianImage->dim[5] = jacobianImage->nu = 3;
        else jacobianImage->dim[5] = jacobianImage->nu = 2;
        if(sizeof(PrecisionTYPE)==8)
            jacobianImage->datatype = NIFTI_TYPE_FLOAT64;
        else jacobianImage->datatype = NIFTI_TYPE_FLOAT32;
        jacobianImage->nbyper = sizeof(PrecisionTYPE);
        jacobianImage->nvox *= jacobianImage->nu;
        jacobianImage->data = (void *)calloc(jacobianImage->nvox, jacobianImage->nbyper);

        // Compute the determinant
        if(flag->inputCPPFlag){
            reg_bspline_GetJacobianMatrix(controlPointImage,
                                          jacobianImage
                                          );
        }
        else{//TODO
        }

        // Export the Jacobian matrix image
        nifti_set_filenames(jacobianImage, param->jacobianMatrixName, 0, 0);
        strcpy (jacobianImage->descrip,"Jacobian determinant matrix created using NiftyReg");
        nifti_image_write(jacobianImage);
        nifti_image_write(jacobianImage);
        printf("Jacobian map image has been saved: %s\n", param->jacobianMatrixName);
        nifti_image_free(jacobianImage);
    }

    nifti_image_free(controlPointImage);
    nifti_image_free(deformationFieldImage);
    nifti_image_free(image);

    return EXIT_SUCCESS;
}

#endif
