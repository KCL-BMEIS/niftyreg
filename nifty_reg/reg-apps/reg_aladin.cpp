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

#define RIGID 0
#define AFFINE 1

#include "_reg_resampling.h"
#include "_reg_globalTransformation.h"
#include "_reg_blockMatching.h"
#include "_reg_tools.h"

#ifdef _USE_CUDA
#include "_reg_cudaCommon.h"
#include "_reg_resampling_gpu.h"
#include "_reg_globalTransformation_gpu.h"
#include "_reg_blockMatching_gpu.h"
#endif

#ifdef _WINDOWS
#include <time.h>
#endif

#ifdef _USE_NR_DOUBLE
#define PrecisionTYPE double
#else
#define PrecisionTYPE float
#endif

typedef struct{
    char *targetImageName;
    char *sourceImageName;
    char *inputAffineName;
    char *outputResultName;
    char *outputAffineName;
    char *targetMaskName;

    int maxIteration;

    int backgroundIndex[3];
    PrecisionTYPE sourceBGValue;

    float targetSigmaValue;
    float sourceSigmaValue;
    int levelNumber;
    int level2Perform;

    int block_percent_to_use;
    int inlier_lts;
}PARAM;

typedef struct{
    bool targetImageFlag;
    bool sourceImageFlag;
    bool inputAffineFlag;
    bool flirtAffineFlag;
    bool levelNumberFlag;
    bool level2PerformFlag;
    bool outputResultFlag;
    bool outputAffineFlag;
    bool targetMaskFlag;

    bool maxIterationFlag;

    bool backgroundIndexFlag;

    bool alignCenterFlag;

    bool rigidFlag;
    bool affineFlag;

    bool targetSigmaFlag;
    bool sourceSigmaFlag;
    bool pyramidFlag;
    bool useGPUFlag;
    bool twoDimRegistration;
}FLAG;
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

    printf("\t-bgi <int> <int> <int>\tForce the background value during\n\t\t\t\tresampling to have the same value as this voxel in the source image [none]\n");

    printf("\t-%%v <int>\t\tPercentage of block to use [50]\n");
    printf("\t-%%i <int>\t\tPercentage of inlier for the LTS [50]\n");
#ifdef _USE_CUDA	
    printf("\t-gpu \t\t\tTo use the GPU implementation [no]\n");
#endif
    printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
    return;
}
#define CONVERGENCE_EPS 0.00001
bool reg_test_convergence(mat44 *mat)
{
    bool convergence=true;
    if((fabsf(mat->m[0][0])-1.0f)>CONVERGENCE_EPS) convergence=false;
    if((fabsf(mat->m[1][1])-1.0f)>CONVERGENCE_EPS) convergence=false;
    if((fabsf(mat->m[2][2])-1.0f)>CONVERGENCE_EPS) convergence=false;

    if((fabsf(mat->m[0][1])-0.0f)>CONVERGENCE_EPS) convergence=false;
    if((fabsf(mat->m[0][2])-0.0f)>CONVERGENCE_EPS) convergence=false;
    if((fabsf(mat->m[0][3])-0.0f)>CONVERGENCE_EPS) convergence=false;

    if((fabsf(mat->m[1][0])-0.0f)>CONVERGENCE_EPS) convergence=false;
    if((fabsf(mat->m[1][2])-0.0f)>CONVERGENCE_EPS) convergence=false;
    if((fabsf(mat->m[1][3])-0.0f)>CONVERGENCE_EPS) convergence=false;

    if((fabsf(mat->m[2][0])-0.0f)>CONVERGENCE_EPS) convergence=false;
    if((fabsf(mat->m[2][1])-0.0f)>CONVERGENCE_EPS) convergence=false;
    if((fabsf(mat->m[2][3])-0.0f)>CONVERGENCE_EPS) convergence=false;

    return convergence;
}

int main(int argc, char **argv)
{
    time_t start; time(&start);

    PARAM *param = (PARAM *)calloc(1,sizeof(PARAM));
    FLAG *flag = (FLAG *)calloc(1,sizeof(FLAG));

    flag->affineFlag=1;
    flag->rigidFlag=1;
    param->block_percent_to_use=50;
    param->inlier_lts=50;
    flag->alignCenterFlag=1;

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
            param->outputAffineName=argv[++i];
            flag->outputAffineFlag=1;
        }
        else if(strcmp(argv[i], "-inaff") == 0){
            param->inputAffineName=argv[++i];
            flag->inputAffineFlag=1;
        }
        else if(strcmp(argv[i], "-affFlirt") == 0){
            param->inputAffineName=argv[++i];
            flag->inputAffineFlag=1;
            flag->flirtAffineFlag=1;
        }
        else if(strcmp(argv[i], "-tmask") == 0){
            param->targetMaskName=argv[++i];
            flag->targetMaskFlag=1;
        }
        else if(strcmp(argv[i], "-result") == 0){
            param->outputResultName=argv[++i];
            flag->outputResultFlag=1;
        }
        else if(strcmp(argv[i], "-maxit") == 0){
            param->maxIteration=atoi(argv[++i]);
            flag->maxIterationFlag=1;
        }
        else if(strcmp(argv[i], "-ln") == 0){
            param->levelNumber=atoi(argv[++i]);
            flag->levelNumberFlag=1;
        }
        else if(strcmp(argv[i], "-lp") == 0){
            param->level2Perform=atoi(argv[++i]);
            flag->level2PerformFlag=1;
        }
        else if(strcmp(argv[i], "-smooT") == 0){
            param->targetSigmaValue=(float)(atof(argv[++i]));
            flag->targetSigmaFlag=1;
        }
        else if(strcmp(argv[i], "-smooS") == 0){
            param->sourceSigmaValue=(float)(atof(argv[++i]));
            flag->sourceSigmaFlag=1;
        }
        else if(strcmp(argv[i], "-rigOnly") == 0){
            flag->affineFlag=0;
        }
        else if(strcmp(argv[i], "-affDirect") == 0){
            flag->rigidFlag=0;
        }
        else if(strcmp(argv[i], "-nac") == 0){
            flag->alignCenterFlag=0;
        }
        else if(strcmp(argv[i], "-bgi") == 0){
            param->backgroundIndex[0]=atoi(argv[++i]);
            param->backgroundIndex[1]=atoi(argv[++i]);
            param->backgroundIndex[2]=atoi(argv[++i]);
            flag->backgroundIndexFlag=1;
        }
        else if(strcmp(argv[i], "-%v") == 0){
            param->block_percent_to_use=atoi(argv[++i]);
        }
        else if(strcmp(argv[i], "-%i") == 0){
            param->inlier_lts=atoi(argv[++i]);
        }
#ifdef _USE_CUDA
        else if(strcmp(argv[i], "-gpu") == 0){
            flag->useGPUFlag=1;
        }
#endif
        else{
            fprintf(stderr,"Err:\tParameter %s unknown.\n",argv[i]);
            PetitUsage(argv[0]);
            return 1;
        }
    }

    if(!flag->targetImageFlag || !flag->sourceImageFlag){
        fprintf(stderr,"Err:\tThe target and the source image have to be defined.\n");
        PetitUsage(argv[0]);
        return 1;
    }

    if(!flag->levelNumberFlag) param->levelNumber=3;

    /* Read the maximum number of iteration */
    if(!flag->maxIterationFlag) param->maxIteration=5;

    if(!flag->level2PerformFlag) param->level2Perform=param->levelNumber;

    param->level2Perform=param->level2Perform<param->levelNumber?param->level2Perform:param->levelNumber;

    /* Read the target and source images */
    nifti_image *targetHeader = nifti_image_read(param->targetImageName,false);
    if(targetHeader == NULL){
        fprintf(stderr,"** ERROR Error when reading the target image: %s\n",param->targetImageName);
        return 1;
    }
    reg_checkAndCorrectDimension(targetHeader);
    nifti_image *sourceHeader = nifti_image_read(param->sourceImageName,false);
    if(sourceHeader == NULL){
        fprintf(stderr,"** ERROR Error when reading the source image: %s\n",param->sourceImageName);
        return 1;
    }
    reg_checkAndCorrectDimension(sourceHeader);

    /* Flag for 2D registration */
    if(sourceHeader->nz==1 || targetHeader->nz==1){
        flag->twoDimRegistration=1;
#ifdef _USE_CUDA 
        if(flag->useGPUFlag){
            printf("\n[WARNING] The GPU 2D version has not been implemented yet [/WARNING]\n");
            printf("[WARNING] >>> Exit <<< [/WARNING]\n\n");
            return 1;
        }
#endif
    }

    /* Check the source background index */
    if(!flag->backgroundIndexFlag) param->sourceBGValue = 0.0;
    else{
        if(param->backgroundIndex[0] < 0 || param->backgroundIndex[1] < 0 || param->backgroundIndex[2] < 0
                || param->backgroundIndex[0] >= sourceHeader->dim[1] || param->backgroundIndex[1] >= sourceHeader->dim[2] || param->backgroundIndex[2] >= sourceHeader->dim[3]){
            fprintf(stderr,"The specified index (%i %i %i) for background does not belong to the source image (out of bondary)\n",
                    param->backgroundIndex[0], param->backgroundIndex[1], param->backgroundIndex[2]);
            return 1;
        }
    }

    /* Read the input affine tranformation is defined otherwise assign it to identity */
    mat44 *affineTransformation = (mat44 *)calloc(1,sizeof(mat44));
    affineTransformation->m[0][0]=1.0;
    affineTransformation->m[1][1]=1.0;
    affineTransformation->m[2][2]=1.0;
    affineTransformation->m[3][3]=1.0;
    if(flag->alignCenterFlag){
        mat44 *sourceMatrix;
        if(sourceHeader->sform_code>0)
            sourceMatrix = &(sourceHeader->sto_xyz);
        else sourceMatrix = &(sourceHeader->qto_xyz);
        mat44 *targetMatrix;
        if(targetHeader->sform_code>0)
            targetMatrix = &(targetHeader->sto_xyz);
        else targetMatrix = &(targetHeader->qto_xyz);
        float sourceCenter[3];
        sourceCenter[0]=(float)(sourceHeader->nx)/2.0f;
        sourceCenter[1]=(float)(sourceHeader->ny)/2.0f;
        sourceCenter[2]=(float)(sourceHeader->nz)/2.0f;
        float targetCenter[3];
        targetCenter[0]=(float)(targetHeader->nx)/2.0f;
        targetCenter[1]=(float)(targetHeader->ny)/2.0f;
        targetCenter[2]=(float)(targetHeader->nz)/2.0f;
        float sourceRealPosition[3]; reg_mat44_mul(sourceMatrix, sourceCenter, sourceRealPosition);
        float targetRealPosition[3]; reg_mat44_mul(targetMatrix, targetCenter, targetRealPosition);
        affineTransformation->m[0][3]=sourceRealPosition[0]-targetRealPosition[0];
        affineTransformation->m[1][3]=sourceRealPosition[1]-targetRealPosition[1];
        affineTransformation->m[2][3]=sourceRealPosition[2]-targetRealPosition[2];
    }

    if(flag->inputAffineFlag){
        // Check first if the specified affine file exist
        if(FILE *aff=fopen(param->inputAffineName, "r")){
            fclose(aff);
        }
        else{
            fprintf(stderr,"The specified input affine file (%s) can not be read\n",param->inputAffineName);
            return 1;
        }
        reg_tool_ReadAffineFile(	affineTransformation,
                                targetHeader,
                                sourceHeader,
                                param->inputAffineName,
                                flag->flirtAffineFlag);
    }

    /* read and binarise the target mask image */
    nifti_image *targetMaskImage=NULL;
    if(flag->targetMaskFlag){
        targetMaskImage = nifti_image_read(param->targetMaskName,true);
        if(targetMaskImage == NULL){
            fprintf(stderr,"* ERROR Error when reading the target naask image: %s\n",param->targetMaskName);
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
        reg_tool_binarise_image(targetMaskImage);
    }

    /* *********************************** */
    /* DISPLAY THE REGISTRATION PARAMETERS */
    /* *********************************** */
    printf("\n* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
    printf("Command line:\n %s",argv[0]);
    for(int i=1;i<argc;i++)
        printf(" %s",argv[i]);
    printf("\n\n");
    printf("Parameters\n");
    printf("Target image name: %s\n",targetHeader->fname);
    printf("\t%ix%ix%i voxels\n",targetHeader->nx,targetHeader->ny,targetHeader->nz);
    printf("\t%gx%gx%g mm\n",targetHeader->dx,targetHeader->dy,targetHeader->dz);
    printf("Source image name: %s\n",sourceHeader->fname);
    printf("\t%ix%ix%i voxels\n",sourceHeader->nx,sourceHeader->ny,sourceHeader->nz);
    printf("\t%gx%gx%g mm\n",sourceHeader->dx,sourceHeader->dy,sourceHeader->dz);
    printf("Maximum iteration number: %i ",param->maxIteration);
    if(flag->inputAffineFlag) printf("\n");
    else printf("(%i during the first level)\n",2*param->maxIteration);
    printf("Percentage of blocks: %i %%",param->block_percent_to_use);
    if(flag->inputAffineFlag) printf("\n");
    else printf(" (100%% during the first level)\n");
#ifdef _USE_CUDA
    if(flag->useGPUFlag) printf("The GPU implementation is used\n");
    else printf("The CPU implementation is used\n");
#endif
    printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n\n");

    /* ********************** */
    /* START THE REGISTRATION */
    /* ********************** */

    for(int level=0; level<param->level2Perform; level++){
        /* Read the target and source image */
        nifti_image *targetImage = nifti_image_read(param->targetImageName,true);

        if(targetImage->data == NULL){
            fprintf(stderr, "** ERROR Error when reading the target image: %s\n", param->targetImageName);
            return 1;
        }
        reg_checkAndCorrectDimension(targetImage);
        reg_changeDatatype<PrecisionTYPE>(targetImage);
        nifti_image *sourceImage = nifti_image_read(param->sourceImageName,true);
        if(sourceImage->data == NULL){
            fprintf(stderr, "** ERROR Error when reading the source image: %s\n", param->sourceImageName);
            return 1;
        }
        reg_checkAndCorrectDimension(sourceImage);
        reg_changeDatatype<PrecisionTYPE>(sourceImage);

        // Twice more iterations are performed during the first level
        // All the blocks are used during the first level
        int maxNumberOfIterationToPerform=param->maxIteration;
        int percentageOfBlockToUse=param->block_percent_to_use;
        if(level==0 && !flag->inputAffineFlag){
            maxNumberOfIterationToPerform*=2;
            percentageOfBlockToUse=100;
        }

        /* declare the target mask array */
        int *targetMask;
        int activeVoxelNumber=0;

        /* downsample the input images if appropriate */
        nifti_image *tempMaskImage=NULL;
        if(flag->targetMaskFlag){
            tempMaskImage = nifti_copy_nim_info(targetMaskImage);
            tempMaskImage->data = (void *)malloc(tempMaskImage->nvox * tempMaskImage->nbyper);
            memcpy(tempMaskImage->data, targetMaskImage->data, tempMaskImage->nvox*tempMaskImage->nbyper);
        }

        for(int l=level; l<param->levelNumber-1; l++){
            int ratio = (int)powf(2.0f,param->levelNumber-param->levelNumber+l+1.0f);

            bool sourceDownsampleAxis[8]={true,true,true,true,true,true,true,true};
            if((sourceHeader->nx/ratio) < 32) sourceDownsampleAxis[1]=false;
            if((sourceHeader->ny/ratio) < 32) sourceDownsampleAxis[2]=false;
            if((sourceHeader->nz/ratio) < 32) sourceDownsampleAxis[3]=false;
            reg_downsampleImage<PrecisionTYPE>(sourceImage, 1, sourceDownsampleAxis);

            bool targetDownsampleAxis[8]={true,true,true,true,true,true,true,true};
            if((targetHeader->nx/ratio) < 32) targetDownsampleAxis[1]=false;
            if((targetHeader->ny/ratio) < 32) targetDownsampleAxis[2]=false;
            if((targetHeader->nz/ratio) < 32) targetDownsampleAxis[3]=false;
            reg_downsampleImage<PrecisionTYPE>(targetImage, 1, targetDownsampleAxis);

            if(flag->targetMaskFlag){
                reg_downsampleImage<PrecisionTYPE>(tempMaskImage, 0, targetDownsampleAxis);
            }
        }
        targetMask = (int *)malloc(targetImage->nvox*sizeof(int));
        if(flag->targetMaskFlag){
            reg_tool_binaryImage2int(tempMaskImage, targetMask, activeVoxelNumber);
            nifti_image_free(tempMaskImage);
        }
        else{
            for(unsigned int i=0; i<targetImage->nvox; i++)
                targetMask[i]=i;
            activeVoxelNumber=targetImage->nvox;
        }


        /* smooth the input image if appropriate */
        if(flag->targetSigmaFlag){
            bool smoothAxis[8]={true,true,true,true,true,true,true,true};
            reg_gaussianSmoothing<PrecisionTYPE>(targetImage, param->targetSigmaValue, smoothAxis);
        }
        if(flag->sourceSigmaFlag){
            bool smoothAxis[8]={true,true,true,true,true,true,true,true};
            reg_gaussianSmoothing<PrecisionTYPE>(sourceImage, param->sourceSigmaValue, smoothAxis);
        }

        /* allocate the deformation Field image */
        nifti_image *positionFieldImage = nifti_copy_nim_info(targetImage);
        positionFieldImage->dim[0]=positionFieldImage->ndim=5;
        positionFieldImage->dim[1]=positionFieldImage->nx=targetImage->nx;
        positionFieldImage->dim[2]=positionFieldImage->ny=targetImage->ny;
        positionFieldImage->dim[3]=positionFieldImage->nz=targetImage->nz;
        positionFieldImage->dim[4]=positionFieldImage->nt=1;positionFieldImage->pixdim[4]=positionFieldImage->dt=1.0;
        if(flag->twoDimRegistration) positionFieldImage->dim[5]=positionFieldImage->nu=2;
        else positionFieldImage->dim[5]=positionFieldImage->nu=3;
        positionFieldImage->pixdim[5]=positionFieldImage->du=1.0;
        positionFieldImage->dim[6]=positionFieldImage->nv=1;positionFieldImage->pixdim[6]=positionFieldImage->dv=1.0;
        positionFieldImage->dim[7]=positionFieldImage->nw=1;positionFieldImage->pixdim[7]=positionFieldImage->dw=1.0;
        positionFieldImage->nvox=positionFieldImage->nx*positionFieldImage->ny*positionFieldImage->nz*positionFieldImage->nt*positionFieldImage->nu;
        if(sizeof(PrecisionTYPE)==4) positionFieldImage->datatype = NIFTI_TYPE_FLOAT32;
        else positionFieldImage->datatype = NIFTI_TYPE_FLOAT64;
        positionFieldImage->nbyper = sizeof(PrecisionTYPE);
#ifdef _USE_CUDA
        if(flag->useGPUFlag){
            positionFieldImage->data=NULL;
        }
        else
#endif
            positionFieldImage->data = (void *)calloc(positionFieldImage->nvox, positionFieldImage->nbyper);

        /* allocate the result image */
        nifti_image *resultImage = nifti_copy_nim_info(targetImage);
        resultImage->datatype = sourceImage->datatype;
        resultImage->nbyper = sourceImage->nbyper;
        resultImage->dim[0] = sourceImage->dim[0];
        resultImage->dim[4] = resultImage->nt = sourceImage->nt;
        resultImage->nvox=resultImage->nx*resultImage->ny*resultImage->nz*resultImage->nt;
#ifdef _USE_CUDA
        if(flag->useGPUFlag){
            cudaMallocHost(&resultImage->data,resultImage->nvox*resultImage->nbyper);
        }
        else
#endif
            resultImage->data = (void *)calloc(resultImage->nvox, resultImage->nbyper);

        /* Set the padding value */
        if(flag->backgroundIndexFlag){
            int index[3];
            index[0]=param->backgroundIndex[0];
            index[1]=param->backgroundIndex[1];
            index[2]=param->backgroundIndex[2];
            if(flag->pyramidFlag){
                for(int l=level; l<param->levelNumber-1; l++){
                    index[0] /= 2;
                    index[1] /= 2;
                    index[2] /= 2;
                }
            }
            param->sourceBGValue = (float)(reg_tool_GetIntensityValue(sourceImage, index));
        }
        else param->sourceBGValue = 0;

        /* initialise the block matching */
        _reg_blockMatchingParam blockMatchingParams;
        initialise_block_matching_method(targetImage,
                                         &blockMatchingParams,
                                         percentageOfBlockToUse,    // percentage of block kept
                                         param->inlier_lts,         // percentage of inlier in the optimisation process
                                         targetMask,
                                         flag->useGPUFlag);

        mat44 updateAffineMatrix;

#ifdef _USE_CUDA
        /* initialise the cuda array if necessary */
        float *targetImageArray_d=NULL;
        cudaArray *sourceImageArray_d=NULL;
        float *resultImageArray_d=NULL;
        float4 *positionFieldImageArray_d=NULL;
        int *targetMask_d=NULL;

        float *targetPosition_d=NULL;
        float *resultPosition_d=NULL;
        int *activeBlock_d=NULL;

        if(flag->useGPUFlag){
            /* The data are transfered from the host to the device */
            if(cudaCommon_allocateArrayToDevice<float>(&targetImageArray_d, targetImage->dim)) return 1;
            if(cudaCommon_transferNiftiToArrayOnDevice<float>(&targetImageArray_d, targetImage)) return 1;
            if(cudaCommon_allocateArrayToDevice<float>(&sourceImageArray_d, sourceImage->dim)) return 1;
            if(cudaCommon_transferNiftiToArrayOnDevice<float>(&sourceImageArray_d,sourceImage)) return 1;
            if(cudaCommon_allocateArrayToDevice<float>(&resultImageArray_d, targetImage->dim)) return 1;
            if(cudaCommon_allocateArrayToDevice<float4>(&positionFieldImageArray_d, targetImage->dim)) return 1;

            // Index of the active voxel is stored
            int *targetMask_h;cudaMallocHost(&targetMask_h, activeVoxelNumber*sizeof(int));
            int *targetMask_h_ptr = &targetMask_h[0];
            for(unsigned int i=0;i<targetImage->nvox;i++){
                if(targetMask[i]!=-1) *targetMask_h_ptr++=i;
            }
            cudaMalloc(&targetMask_d, activeVoxelNumber*sizeof(int));
            cudaMemcpy(targetMask_d, targetMask_h, activeVoxelNumber*sizeof(int), cudaMemcpyHostToDevice);
            cudaFreeHost(targetMask_h);

            cudaMalloc(&targetPosition_d, blockMatchingParams.activeBlockNumber*3*sizeof(float));
            cudaMalloc(&resultPosition_d, blockMatchingParams.activeBlockNumber*3*sizeof(float));

            cudaMalloc(&activeBlock_d,
                       blockMatchingParams.blockNumber[0]*blockMatchingParams.blockNumber[1]*blockMatchingParams.blockNumber[2]*sizeof(int));
            cudaMemcpy(activeBlock_d, blockMatchingParams.activeBlock,
                       blockMatchingParams.blockNumber[0]*blockMatchingParams.blockNumber[1]*blockMatchingParams.blockNumber[2]*sizeof(int),
                       cudaMemcpyHostToDevice);
        }
#endif

        /* Display some parameters specific to the current level */
        printf("Current level %i / %i\n", level+1, param->levelNumber);
        printf("Target image size: \t%ix%ix%i voxels\t%gx%gx%g mm\n",
               targetImage->nx, targetImage->ny, targetImage->nz, targetImage->dx, targetImage->dy, targetImage->dz);
        printf("Source image size: \t%ix%ix%i voxels\t%gx%gx%g mm\n",
               sourceImage->nx, sourceImage->ny, sourceImage->nz, sourceImage->dx, sourceImage->dy, sourceImage->dz);
        if(flag->twoDimRegistration)
            printf("Block size = [4 4 1]\n");
        else printf("Block size = [4 4 4]\n");
        printf("Block number = [%i %i %i]\n", blockMatchingParams.blockNumber[0],
               blockMatchingParams.blockNumber[1], blockMatchingParams.blockNumber[2]);
#ifndef NDEBUG
        if(targetImage->sform_code>0)
            reg_mat44_disp(&targetImage->sto_xyz, (char *)"[DEBUG] Target image matrix (sform sto_xyz)");
        else reg_mat44_disp(&targetImage->qto_xyz, (char *)"[DEBUG] Target image matrix (qform qto_xyz)");
        if(sourceImage->sform_code>0)
            reg_mat44_disp(&sourceImage->sto_xyz, (char *)"[DEBUG] Source image matrix (sform sto_xyz)");
        else reg_mat44_disp(&sourceImage->qto_xyz, (char *)"[DEBUG] Source image matrix (qform qto_xyz)");
#endif
        printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
        reg_mat44_disp(affineTransformation, (char *)"Initial affine transformation:");

        /* ****************** */
        /* Rigid registration */
        /* ****************** */
        int iteration=0;

        if((flag->rigidFlag && !flag->affineFlag) || (flag->affineFlag && flag->rigidFlag && level==0)){
            while(iteration<maxNumberOfIterationToPerform){
                /* Compute the affine transformation deformation field */
#ifdef _USE_CUDA
                if(flag->useGPUFlag){
                    reg_affine_positionField_gpu(	affineTransformation,
                                                 targetImage,
                                                 &positionFieldImageArray_d);
                    /* Resample the source image */
                    reg_resampleSourceImage_gpu(resultImage,
                                                sourceImage,
                                                &resultImageArray_d,
                                                &sourceImageArray_d,
                                                &positionFieldImageArray_d,
                                                &targetMask_d,
                                                activeVoxelNumber,
                                                param->sourceBGValue);
                    /* Compute the correspondances between blocks */
                    block_matching_method_gpu(	targetImage,
                                              resultImage,
                                              &blockMatchingParams,
                                              &targetImageArray_d,
                                              &resultImageArray_d,
                                              &targetPosition_d,
                                              &resultPosition_d,
                                              &activeBlock_d);
                    /* update  the affine transformation matrix */
                    optimize_gpu(	&blockMatchingParams,
                                 &updateAffineMatrix,
                                 &targetPosition_d,
                                 &resultPosition_d,
                                 RIGID);
                }
                else{
#endif
                    reg_affine_positionField(	affineTransformation,
                                             targetImage,
                                             positionFieldImage);
                    /* Resample the source image */
                    reg_resampleSourceImage(targetImage,
                                            sourceImage,
                                            resultImage,
                                            positionFieldImage,
                                            targetMask,
                                            1,
                                            param->sourceBGValue);
                    /* Compute the correspondances between blocks */
                    block_matching_method<PrecisionTYPE>(   targetImage,
                                                         resultImage,
                                                         &blockMatchingParams,
                                                         targetMask);
                    /* update  the affine transformation matrix */
                    optimize(&blockMatchingParams,
                             &updateAffineMatrix,
                             RIGID);
#ifdef _USE_CUDA
                }
#endif
                // the affine transformation is updated
                *affineTransformation = reg_mat44_mul( affineTransformation, &(updateAffineMatrix));
#ifndef NDEBUG
                printf("[DEBUG] -Rigid- iteration %i - ",iteration);
                reg_mat44_disp(&updateAffineMatrix, (char *)"[DEBUG] updateMatrix");
                reg_mat44_disp(affineTransformation, (char *)"[DEBUG] updated affine");
#endif

                if(reg_test_convergence(&updateAffineMatrix)) break;
                iteration++;
            }
        }

        /* ******************* */
        /* Affine registration */
        /* ******************* */
        iteration=0;
        if(flag->affineFlag){
            while(iteration<maxNumberOfIterationToPerform){
                /* Compute the affine transformation deformation field */
#ifdef _USE_CUDA
                if(flag->useGPUFlag){
                    reg_affine_positionField_gpu(	affineTransformation,
                                                 targetImage,
                                                 &positionFieldImageArray_d);
                    /* Resample the source image */
                    reg_resampleSourceImage_gpu(	resultImage,
                                                sourceImage,
                                                &resultImageArray_d,
                                                &sourceImageArray_d,
                                                &positionFieldImageArray_d,
                                                &targetMask_d,
                                                activeVoxelNumber,
                                                param->sourceBGValue);
                    /* Compute the correspondances between blocks */
                    block_matching_method_gpu(	targetImage,
                                              resultImage,
                                              &blockMatchingParams,
                                              &targetImageArray_d,
                                              &resultImageArray_d,
                                              &targetPosition_d,
                                              &resultPosition_d,
                                              &activeBlock_d);

                    /* update  the affine transformation matrix */
                    optimize_gpu(	&blockMatchingParams,
                                 &updateAffineMatrix,
                                 &targetPosition_d,
                                 &resultPosition_d,
                                 AFFINE);
                }
                else{
#endif
                    reg_affine_positionField(	affineTransformation,
                                             targetImage,
                                             positionFieldImage);
                    /* Resample the source image */
                    reg_resampleSourceImage(	targetImage,
                                            sourceImage,
                                            resultImage,
                                            positionFieldImage,
                                            targetMask,
                                            1,
                                            param->sourceBGValue);
                    /* Compute the correspondances between blocks */
                    block_matching_method<PrecisionTYPE>(	targetImage,
                                                         resultImage,
                                                         &blockMatchingParams,
                                                         targetMask);
                    /* update  the affine transformation matrix */
                    optimize(	&blockMatchingParams,
                             &updateAffineMatrix,
                             AFFINE);
#ifdef _USE_CUDA
                }
#endif

                // the affine transformation is updated
                *affineTransformation = reg_mat44_mul( affineTransformation, &(updateAffineMatrix));
#ifndef NDEBUG
                printf("[DEBUG] -Affine- iteration %i - ",iteration);
                reg_mat44_disp(&updateAffineMatrix, (char *)"[DEBUG] updateMatrix");
                reg_mat44_disp(affineTransformation, (char *)"[DEBUG] updated affine");
#endif
                if(reg_test_convergence(&updateAffineMatrix)) break;
                iteration++;
            }
        }

        free(targetMask);

#ifdef _USE_CUDA
        if(flag->useGPUFlag){
            /* The data are transfered from the host to the device */
            cudaCommon_free<float>(&targetImageArray_d);
            cudaCommon_free(&sourceImageArray_d);
            cudaCommon_free<float>(&resultImageArray_d);
            cudaCommon_free<float4>(&positionFieldImageArray_d);
            cudaCommon_free<int>(&activeBlock_d);
            cudaFreeHost(resultImage->data);
            resultImage->data=NULL;
        }
#endif

        if(level==(param->level2Perform-1)){
            /* ****************** */
            /* OUTPUT THE RESULTS */
            /* ****************** */

#ifdef _USE_CUDA
            if(flag->useGPUFlag && param->level2Perform==param->levelNumber)
                positionFieldImage->data = (void *)calloc(positionFieldImage->nvox, positionFieldImage->nbyper);
            else
#endif
                if(param->level2Perform != param->levelNumber){
                    if(positionFieldImage->data)free(positionFieldImage->data);
                    positionFieldImage->dim[1]=positionFieldImage->nx=targetHeader->nx;
                    positionFieldImage->dim[2]=positionFieldImage->ny=targetHeader->ny;
                    positionFieldImage->dim[3]=positionFieldImage->nz=targetHeader->nz;
                    positionFieldImage->dim[4]=positionFieldImage->nt=1;positionFieldImage->pixdim[4]=positionFieldImage->dt=1.0;
                    if(flag->twoDimRegistration)
                        positionFieldImage->dim[5]=positionFieldImage->nu=2;
                    else positionFieldImage->dim[5]=positionFieldImage->nu=3;
                    positionFieldImage->pixdim[5]=positionFieldImage->du=1.0;
                    positionFieldImage->dim[6]=positionFieldImage->nv=1;positionFieldImage->pixdim[6]=positionFieldImage->dv=1.0;
                    positionFieldImage->dim[7]=positionFieldImage->nw=1;positionFieldImage->pixdim[7]=positionFieldImage->dw=1.0;
                    positionFieldImage->nvox=positionFieldImage->nx*positionFieldImage->ny*positionFieldImage->nz*positionFieldImage->nt*positionFieldImage->nu;
                    positionFieldImage->data = (void *)calloc(positionFieldImage->nvox, positionFieldImage->nbyper);
                }

            /* The corresponding deformation field is evaluated and saved */
            reg_affine_positionField(	affineTransformation,
                                     targetHeader,
                                     positionFieldImage);

            /* The result image is resampled using a cubic spline interpolation */
            nifti_image_free(sourceImage);
            sourceImage = nifti_image_read(param->sourceImageName,true); // reload the source image with the correct intensity values
            nifti_image_free(resultImage);
            resultImage = nifti_copy_nim_info(targetHeader);
            resultImage->cal_min=sourceImage->cal_min;
            resultImage->cal_max=sourceImage->cal_max;
            resultImage->scl_slope=sourceImage->scl_slope;
            resultImage->scl_inter=sourceImage->scl_inter;
            resultImage->datatype = sourceImage->datatype;
            resultImage->nbyper = sourceImage->nbyper;
            resultImage->nt = resultImage->dim[4] = sourceImage->nt;
            resultImage->nvox=resultImage->nx*resultImage->ny*resultImage->nz*resultImage->nt;
            resultImage->data = (void *)calloc(resultImage->nvox, resultImage->nbyper);
            reg_resampleSourceImage(targetHeader,
                                    sourceImage,
                                    resultImage,
                                    positionFieldImage,
                                    NULL,
                                    3,
                                    param->sourceBGValue);
            if(!flag->outputResultFlag) param->outputResultName=(char *)"outputResult.nii";
            nifti_set_filenames(resultImage, param->outputResultName, 0, 0);
            nifti_image_write(resultImage);

        }
        nifti_image_free(positionFieldImage);
        nifti_image_free(resultImage);
        nifti_image_free(targetImage);
        nifti_image_free(sourceImage);
        reg_mat44_disp(affineTransformation, (char *)"Final affine transformation:");
#ifndef NDEBUG
        mat33 tempMat;
        for(int i=0; i<3; i++){
            for(int j=0; j<3; j++){
                tempMat.m[i][j] = affineTransformation->m[i][j];
            }
        }
        printf("[DEBUG] Matrix determinant %g\n", nifti_mat33_determ(tempMat));
#endif
        printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\n");
    }


    /* The affine transformation is saved */
    if(flag->outputAffineFlag)
        reg_tool_WriteAffineFile(affineTransformation, param->outputAffineName);
    else reg_tool_WriteAffineFile(affineTransformation, (char *)"outputAffine.txt");

    free(affineTransformation);
    nifti_image_free(targetHeader);
    nifti_image_free(sourceHeader);

    free(flag);
    free(param);

    time_t end; time(&end);
    int minutes=(int)floorf((end-start)/60.0f);
    int seconds=(int)(end-start - 60*minutes);
    printf("Registration Performed in %i min %i sec\n", minutes, seconds);
    printf("Have a good day !\n");

    return 0;
}

#endif
