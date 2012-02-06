/*
 *  reg_aladin.cpp
 *
 *
 *  Created by Marc Modat and Pankaj Daga on 12/08/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

/* TODO
 - All
 */

/* TOFIX
 - None (so far)
 */

#ifndef _MM_ALADIN_CPP
#define _MM_ALADIN_CPP


#include "_reg_aladin.h"
#include "_reg_tools.h"

#ifdef _WINDOWS
#include <time.h>
#endif

#ifdef _USE_NR_DOUBLE
#define PrecisionTYPE double
#else
#define PrecisionTYPE float
#endif

void PetitUsage(char *exec)
{
    fprintf(stderr,"Aladin - Seb.\n");
    fprintf(stderr,"Usage:\t%s -target <targetImageName> -source <sourceImageName> [OPTIONS].\n",exec);
    fprintf(stderr,"\tSee the help for more details (-h).\n");
    return;
}
void Usage(char *exec)
{
    printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
    printf("Block Matching algorithm for global registration.\n");
    printf("Based on Ourselin et al., \"Reconstructing a 3D structure from serial histological sections\",\n");
    printf("Image and Vision Computing, 2001\n");
    printf("This code has been written by Marc Modat (m.modat@ucl.ac.uk) and Pankaj Daga,\n");
    printf("for any comment, please contact them.\n");
    printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
    printf("Usage:\t%s -target <filename> -source <filename> [OPTIONS].\n",exec);
    printf("\t-target <filename>\tFilename of the target image (mandatory)\n");
    printf("\t-source <filename>\tFilename of the source image (mandatory)\n");
    printf("* * OPTIONS * *\n");
    printf("\t-aff <filename>\t\tFilename which contains the output affine transformation [outputAffine.txt]\n");
    printf("\t-rigOnly\t\tTo perform a rigid registration only (rigid+affine by default)\n");
    printf("\t-affDirect\t\tDirectly optimize 12 DoF affine [default is rigid initially then affine]\n");
    printf("\t-inaff <filename>\tFilename which contains an input affine transformation (Affine*Target=Source) [none]\n");
    printf("\t-affFlirt <filename>\tFilename which contains an input affine transformation from Flirt [none]\n");
    printf("\t-tmask <filename>\tFilename of a mask image in the target space\n");
    printf("\t-result <filename>\tFilename of the resampled image [outputResult.nii]\n");
    printf("\t-maxit <int>\t\tNumber of iteration per level [5]\n");
    printf("\t-smooT <float>\t\tSmooth the target image using the specified sigma (mm) [0]\n");
    printf("\t-smooS <float>\t\tSmooth the source image using the specified sigma (mm) [0]\n");
    printf("\t-ln <int>\t\tNumber of level to perform [3]\n");
    printf("\t-lp <int>\t\tOnly perform the first levels [ln]\n");

    printf("\t-nac\t\t\tUse the nifti header origins to initialise the translation\n");

    printf("\t-%%v <int>\t\tPercentage of block to use [50]\n");
    printf("\t-%%i <int>\t\tPercentage of inlier for the LTS [50]\n");
#ifdef _USE_CUDA	
    printf("\t-gpu \t\t\tTo use the GPU implementation [no]\n");
#endif
    printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
    return;
}

int main(int argc, char **argv)
{
    time_t start; time(&start);
    reg_aladin<PrecisionTYPE> *REG = new reg_aladin<PrecisionTYPE>;
    char *targetImageName=NULL;
    int targetImageFlag=0;

    char *sourceImageName=NULL;
    int sourceImageFlag=0;

    char *outputAffineName=NULL;
    int outputAffineFlag=0;

    char *inputAffineName=NULL;
    int inputAffineFlag=0;
    int flirtAffineFlag=0;

    char *targetMaskName=NULL;
    int targetMaskFlag=0;

    char *outputResultName=NULL;
    int outputResultFlag=0;

    /* read the input parameter */
    for(int i=1;i<argc;i++){
        if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 ||
           strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 ||
           strcmp(argv[i], "--h")==0 || strcmp(argv[i], "--help")==0){
            Usage(argv[0]);
            return 0;
        }
        else if(strcmp(argv[i], "-target") == 0){
            targetImageName=argv[++i];
            targetImageFlag=1;
        }
        else if(strcmp(argv[i], "-source") == 0){
            sourceImageName=argv[++i];
            sourceImageFlag=1;
        }
        else if(strcmp(argv[i], "-aff") == 0){
            outputAffineName=argv[++i];
            outputAffineFlag=1;
        }
        else if(strcmp(argv[i], "-inaff") == 0){
            inputAffineName=argv[++i];
            inputAffineFlag=1;
        }
        else if(strcmp(argv[i], "-affFlirt") == 0){
            inputAffineName=argv[++i];
            inputAffineFlag=1;
            flirtAffineFlag=1;
        }
        else if(strcmp(argv[i], "-tmask") == 0){
            targetMaskName=argv[++i];
            targetMaskFlag=1;
        }
        else if(strcmp(argv[i], "-result") == 0){
            outputResultName=argv[++i];
            outputResultFlag=1;
        }
        else if(strcmp(argv[i], "-maxit") == 0){
            REG->SetMaxIterations(atoi(argv[++i]));
        }
        else if(strcmp(argv[i], "-ln") == 0){
            REG->SetNumberOfLevels(atoi(argv[++i]));
        }
        else if(strcmp(argv[i], "-lp") == 0){
            REG->SetLevelsToPerform(atoi(argv[++i]));
        }
        else if(strcmp(argv[i], "-smooT") == 0){
            REG->SetReferenceSigma((float)(atof(argv[++i])));
        }
        else if(strcmp(argv[i], "-smooS") == 0){
            REG->SetFloatingSigma((float)(atof(argv[++i])));
        }
        else if(strcmp(argv[i], "-rigOnly") == 0){
            REG->PerformAffineOff();
            REG->PerformRigidOn();
            }
        else if(strcmp(argv[i], "-affDirect") == 0){
            REG->PerformAffineOn();
            REG->PerformRigidOff();
        }
        else if(strcmp(argv[i], "-nac") == 0){
            REG->AlignCentreOff();
        }
        else if(strcmp(argv[i], "-%v") == 0){
            REG->SetBlockPercentage(atof(argv[++i]));
        }
        else if(strcmp(argv[i], "-%i") == 0){
            REG->SetInlierLts(atof(argv[++i]));
        }
        else if(strcmp(argv[i], "-NN") == 0){
            REG->SetInterpolationToNearestNeighbor();
        }
        else if(strcmp(argv[i], "-LIN") == 0){
            REG->SetInterpolationToTrilinear();
        }
        else if(strcmp(argv[i], "-CUB") == 0){
            REG->SetInterpolationToCubic();
        }
#ifdef _USE_CUDA
        else if(strcmp(argv[i], "-gpu") == 0){
            REG->UseGpuOn();
        }
#endif
        else{
            fprintf(stderr,"Err:\tParameter %s unknown.\n",argv[i]);
            PetitUsage(argv[0]);
            return 1;
        }
    }

    if(!targetImageFlag || !sourceImageFlag){
        fprintf(stderr,"Err:\tThe target and the source image have to be defined.\n");
        PetitUsage(argv[0]);
        return 1;
    }


    if(REG->GetLevelsToPerform() > REG->GetNumberOfLevels())
        REG->SetLevelsToPerform(REG->GetNumberOfLevels());

    /* Read the target and source images */
    nifti_image *targetHeader = nifti_image_read(targetImageName,true);
    nifti_image *sourceHeader = nifti_image_read(sourceImageName,true);
    REG->SetInputReference(targetHeader);
    REG->SetInputFloating(sourceHeader);

    REG->SetInputTransform(inputAffineName,flirtAffineFlag);

    /* read the target mask image */
    nifti_image *targetMaskImage=NULL;
    if(targetMaskFlag){
        targetMaskImage = nifti_image_read(targetMaskName,true);
        if(targetMaskImage == NULL){
            fprintf(stderr,"* ERROR Error when reading the target naask image: %s\n",targetImageName);
            return 1;
        }
        reg_checkAndCorrectDimension(targetMaskImage);
        /* check the dimension */
        for(int i=1; i<=targetHeader->dim[0]; i++){
            if(targetHeader->dim[i]!=targetMaskImage->dim[i]){
                fprintf(stderr,"* ERROR The target image and its mask do not have the same dimension\n");
                return 1;
            }
        }
        REG->SetInputMask(targetMaskImage);
    }
    REG->Run();

    // The warped image is saved
    nifti_image *outputResultImage=REG->GetFinalWarpedImage();
    if(!outputResultFlag) outputResultName=(char *)"outputResult.nii";
    nifti_set_filenames(outputResultImage,outputResultName,0,0);
    nifti_image_write(outputResultImage);
    nifti_image_free(outputResultImage);

    /* The affine transformation is saved */
    if(outputAffineFlag)
        reg_tool_WriteAffineFile(REG->GetTransformationMatrix(), outputAffineName);
    else reg_tool_WriteAffineFile(REG->GetTransformationMatrix(), (char *)"outputAffine.txt");

    nifti_image_free(targetHeader);
    nifti_image_free(sourceHeader);

    delete REG;
    time_t end; time(&end);
    int minutes=(int)floorf((end-start)/60.0f);
    int seconds=(int)(end-start - 60*minutes);
    printf("Registration Performed in %i min %i sec\n", minutes, seconds);
    printf("Have a good day !\n");

    return 0;
}

#endif
