#include "_reg_aladin.h"
#ifndef _REG_ALADIN_CPP
#define _REG_ALADIN_CPP
template <class T> reg_aladin<T>::reg_aladin ()
{
    ExecutableName=(char*) "reg_aladin";
    InputReference = 0;
    InputFloating = 0;
    InputMask = 0;
    OutputImage = 0;
    UseInputTransform=0;
    InputTransform = 0;
    OutputTransform=0;
    Verbose = 0;
    ImageDimension=3;
    MaxIterations = 5;
    BackgroundIndex[0]=BackgroundIndex[1]=BackgroundIndex[2] = 0;
    NumberOfLevels = 3;
    LevelsToPerform = 3;
    PerformRigid=1;
    PerformAffine=1;
    BlockPercentage=50;
    InlierLts=50;
    AlignCentre=1;
    Interpolation=1;
    UseGpu=0;
    UseTargetMask=0;
    SmoothSource=0;
    SourceSigma=0.0;
    SmoothTarget=0;
    TargetSigma=0.0;
}

template <class T> reg_aladin<T>::~reg_aladin()
{
    if(this->InputTransform)
    {
        delete this->InputTransform;
        this->InputTransform=0;
    }
    if(this->OutputImage)
    {
        nifti_image_free(this->OutputImage);
    }
    //Others to consider:OutputTransform
}
template <class T>
bool reg_aladin<T>::TestMatrixConvergence(mat44 *mat)
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

template <class T>
int reg_aladin<T>::Check()
{
    //This does all the initial checking
    if(this->InputReference == 0)
    {
        fprintf(stderr,"** ERROR Error when reading the target image. No image specified or not able to read \n");
        return 1;
    }
    reg_checkAndCorrectDimension(this->InputReference);

    if(this->InputFloating == 0)
    {
        fprintf(stderr,"** ERROR Error when reading the source image: No image specified or not able to read \n");
        return 1;
    }
    /* Flag for 2D registration */
    if(this->InputReference->nz==1 || this->InputFloating->nz==1)
    {
        this->ImageDimension=2;
#ifdef _USE_CUDA
        if(this->UseGpu){
            printf("\n[WARNING] The GPU 2D version has not been implemented yet [/WARNING]\n");
            printf("[WARNING] >>> Exit <<< [/WARNING]\n\n");
            return 1;
        }
#endif
    }

    /* Check the source background index */
    if(!this->UseBackgroundIndex) this->SourceBackgroundValue = 0.0;
    else{
        if(this->BackgroundIndex[0] < 0 || this->BackgroundIndex[1] < 0 || this->BackgroundIndex[2] < 0
           || this->BackgroundIndex[0] >= this->InputFloating->dim[1] || this->BackgroundIndex[1] >= this->InputFloating->dim[2] || this->BackgroundIndex[2] >= this->InputFloating->dim[3]){
            fprintf(stderr,"The specified index (%i %i %i) for background does not belong to the source image (out of bondary)\n",
                    this->BackgroundIndex[0], this->BackgroundIndex[1], this->BackgroundIndex[2]);
            return 1;
        }
    }
return 0;
}

template <class T>
int reg_aladin<T>::Print()
{
    if(this->InputReference == 0)
    {
        fprintf(stderr,"** ERROR Error when reading the target image. No image is loaded\n");
        return 1;
    }
    if(this->InputFloating == 0)
    {
        fprintf(stderr,"** ERROR Error when reading the source image. No image is loaded\n");
        return 1;
    }

    /* *********************************** */
    /* DISPLAY THE REGISTRATION PARAMETERS */
    /* *********************************** */
    printf("Parameters\n");
    printf("Target image name: %s\n",this->InputReference->fname);
    printf("\t%ix%ix%i voxels\n",this->InputReference->nx,this->InputReference->ny,this->InputReference->nz);
    printf("\t%gx%gx%g mm\n",this->InputReference->dx,this->InputReference->dy,this->InputReference->dz);
    printf("Source image name: %s\n",this->InputFloating->fname);
    printf("\t%ix%ix%i voxels\n",this->InputFloating->nx,this->InputFloating->ny,this->InputFloating->nz);
    printf("\t%gx%gx%g mm\n",this->InputFloating->dx,this->InputFloating->dy,this->InputFloating->dz);
    printf("Maximum iteration number: %i ",this->MaxIterations);
    if(this->UseInputTransform) printf("\n");
    else printf("(%i during the first level)\n",2*this->MaxIterations);
    printf("Percentage of blocks: %i %%",this->BlockPercentage);
    if(this->UseInputTransform) printf("\n");
    else printf(" (100%% during the first level)\n");
#ifdef _USE_CUDA
    if(this->UseGpu) printf("The GPU implementation is used\n");
    else printf("The CPU implementation is used\n");
#endif

}

template <class T>
int reg_aladin<T>::SetInputTransform(char *filename, int flirtFlag)
{
    //Assume all flags, images, and such have been set up.
    if(this->InputReference == NULL){
        fprintf(stderr,"** ERROR Error when reading the target image. No target image loaded\n");
        return 1;
    }
    if(this->InputFloating == NULL){
        fprintf(stderr,"** ERROR Error when reading the source image. No source image loaded\n");
        return 1;
    }
    //Do any aligning of the centre before we start with the transformation
    //Need to check if it is NULL or not
    if(this->InputTransform == NULL)
    {
        this->InputTransform = new mat44;
    }
    for(int i=0; i< 4; i++)
    {
        for(int j=0; j < 4; j++)
        {
            this->InputTransform->m[i][j]=0.0;
        }
        this->InputTransform->m[i][i]=1.0;
    }
    if(this->AlignCentre)
    {
        mat44 *sourceMatrix;
        if(this->InputFloating->sform_code>0)
            sourceMatrix = &(this->InputFloating->sto_xyz);
        else sourceMatrix = &(this->InputFloating->qto_xyz);
        mat44 *targetMatrix;
        if(this->InputReference->sform_code>0)
            targetMatrix = &(this->InputReference->sto_xyz);
        else targetMatrix = &(this->InputReference->qto_xyz);
        float sourceCenter[3];
        sourceCenter[0]=(float)(this->InputFloating->nx)/2.0f;
        sourceCenter[1]=(float)(this->InputFloating->ny)/2.0f;
        sourceCenter[2]=(float)(this->InputFloating->nz)/2.0f;
        float targetCenter[3];
        targetCenter[0]=(float)(this->InputReference->nx)/2.0f;
        targetCenter[1]=(float)(this->InputReference->ny)/2.0f;
        targetCenter[2]=(float)(this->InputReference->nz)/2.0f;
        float sourceRealPosition[3]; reg_mat44_mul(sourceMatrix, sourceCenter, sourceRealPosition);
        float targetRealPosition[3]; reg_mat44_mul(targetMatrix, targetCenter, targetRealPosition);
        this->InputTransform->m[0][3]=sourceRealPosition[0]-targetRealPosition[0];
        this->InputTransform->m[1][3]=sourceRealPosition[1]-targetRealPosition[1];
        this->InputTransform->m[2][3]=sourceRealPosition[2]-targetRealPosition[2];

    }
    if(filename != NULL)
    {
        if(FILE *aff=fopen(filename, "r")){
            fclose(aff);
        }
        else{
            fprintf(stderr,"The specified input affine file (%s) can not be read\n",filename);
            return 1;
        }
        reg_tool_ReadAffineFile(	this->InputTransform,
                                        this->InputReference,
                                        this->InputFloating,
                                        filename,
                                        flirtFlag);
        //Good time to create OutputTransform and set it to InputTransform
        this->UseInputTransform=1;
    }
    if(this->OutputTransform == 0)
        this->OutputTransform = new mat44;
}

template <class T>
void reg_aladin<T>::Run()
{
    for(int i=0; i < 4; i++)
    {
        for(int j=0; j < 4; j++)
        {
            this->OutputTransform->m[i][j]=this->InputTransform->m[i][j];
        }
    }

    //Main loop:
    for(int CurrentLevel=0; CurrentLevel < this->LevelsToPerform; CurrentLevel++)
    {
        //Create a temporary copy of the target image.
        //It will be downsampled and smoothed
//        nifti_image *targetImage = nifti_copy_nim_info(this->InputReference);
        nifti_image *targetImage = nifti_image_read(this->InputReference->fname,true);
//        targetImage->data = (void*)malloc(targetImage->nvox * targetImage->nbyper);
//        memcpy(targetImage->data,this->InputReference->data,targetImage->nvox * targetImage->nbyper);
        reg_checkAndCorrectDimension(targetImage);
        reg_changeDatatype<T>(targetImage);

        //Create a temporary copy of the source image
        //It will be downsampled and smoothed
//        nifti_image *sourceImage = nifti_copy_nim_info(this->InputFloating);
        nifti_image *sourceImage = nifti_image_read(this->InputFloating->fname,true);
//        sourceImage->data = (void*)malloc(sourceImage->nvox * sourceImage->nbyper);
//        memcpy(sourceImage->data,this->InputFloating->data,sourceImage->nvox * sourceImage->nbyper);
        reg_checkAndCorrectDimension(sourceImage);
        reg_changeDatatype<T>(sourceImage);

        // Twice more iterations are performed during the first level
        // All the blocks are used during the first level
        int maxNumberOfIterationToPerform=this->MaxIterations;
        int percentageOfBlockToUse=this->BlockPercentage;
        if(CurrentLevel==0 && !this->UseInputTransform){
            maxNumberOfIterationToPerform*=2;
            percentageOfBlockToUse=100;
        }
        /* declare the target mask array */
        int *targetMask;
        int activeVoxelNumber=0;

        /* downsample the input images if appropriate */
        nifti_image *tempMaskImage=NULL;
        if(this->InputMask && this->UseTargetMask){
            tempMaskImage = nifti_copy_nim_info(this->InputMask);
            tempMaskImage->data = (void *)malloc(tempMaskImage->nvox * tempMaskImage->nbyper);
            memcpy(tempMaskImage->data, this->InputMask->data, tempMaskImage->nvox*tempMaskImage->nbyper);
        }

        for(int l=CurrentLevel; l< (this->NumberOfLevels-1); l++){
            int ratio = (int)powf(2.0f,l+1.0f);

            bool sourceDownsampleAxis[8]={true,true,true,true,true,true,true,true};
            if((this->InputFloating->nx/ratio) < 32) sourceDownsampleAxis[1]=false;
            if((this->InputFloating->ny/ratio) < 32) sourceDownsampleAxis[2]=false;
            if((this->InputFloating->nz/ratio) < 32) sourceDownsampleAxis[3]=false;
            reg_downsampleImage<T>(sourceImage, 1, sourceDownsampleAxis);

            bool targetDownsampleAxis[8]={true,true,true,true,true,true,true,true};
            if((this->InputReference->nx/ratio) < 32) targetDownsampleAxis[1]=false;
            if((this->InputReference->ny/ratio) < 32) targetDownsampleAxis[2]=false;
            if((this->InputReference->nz/ratio) < 32) targetDownsampleAxis[3]=false;
            reg_downsampleImage<T>(targetImage, 1, targetDownsampleAxis);

            if(this->UseTargetMask){
                reg_downsampleImage<T>(tempMaskImage, 0, targetDownsampleAxis);
            }
        }

        //This determines the active voxels based on the mask on each level
        //and makes a binary model, which converts it to images.
        targetMask = (int *)malloc(targetImage->nvox*sizeof(int));
        if(this->UseTargetMask){
            reg_tool_binaryImage2int(tempMaskImage, targetMask, activeVoxelNumber);
            nifti_image_free(tempMaskImage);
        }
        else{
            for(unsigned int i=0; i<targetImage->nvox; i++)
                targetMask[i]=i;
            activeVoxelNumber=targetImage->nvox;
        }


        /* smooth the input image if appropriate */
        if(this->SmoothTarget){
            bool smoothAxis[8]={true,true,true,true,true,true,true,true};
            reg_gaussianSmoothing<T>(targetImage, this->TargetSigma, smoothAxis);
        }
        if(this->SmoothSource){
            bool smoothAxis[8]={true,true,true,true,true,true,true,true};
            reg_gaussianSmoothing<T>(sourceImage, this->SourceSigma, smoothAxis);
        }

        /* allocate the deformation Field image */
        nifti_image *positionFieldImage = nifti_copy_nim_info(targetImage);
        positionFieldImage->dim[0]=positionFieldImage->ndim=5;
        positionFieldImage->dim[1]=positionFieldImage->nx=targetImage->nx;
        positionFieldImage->dim[2]=positionFieldImage->ny=targetImage->ny;
        positionFieldImage->dim[3]=positionFieldImage->nz=targetImage->nz;
        positionFieldImage->dim[4]=positionFieldImage->nt=1;positionFieldImage->pixdim[4]=positionFieldImage->dt=1.0;
        if(this->ImageDimension==2) positionFieldImage->dim[5]=positionFieldImage->nu=2;
        else positionFieldImage->dim[5]=positionFieldImage->nu=3;
        positionFieldImage->pixdim[5]=positionFieldImage->du=1.0;
        positionFieldImage->dim[6]=positionFieldImage->nv=1;positionFieldImage->pixdim[6]=positionFieldImage->dv=1.0;
        positionFieldImage->dim[7]=positionFieldImage->nw=1;positionFieldImage->pixdim[7]=positionFieldImage->dw=1.0;
        positionFieldImage->nvox=positionFieldImage->nx*positionFieldImage->ny*positionFieldImage->nz*positionFieldImage->nt*positionFieldImage->nu;
        if(sizeof(T)==4) positionFieldImage->datatype = NIFTI_TYPE_FLOAT32;
        else positionFieldImage->datatype = NIFTI_TYPE_FLOAT64;
        positionFieldImage->nbyper = sizeof(T);
#ifdef _USE_CUDA
        if(this->UseGpu){
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
            //Need to figure out what we are doing with result data
            resultImage->data = (void *)calloc(resultImage->nvox, resultImage->nbyper);

        /* Set the padding value */
        if(this->UseBackgroundIndex){
            int index[3];
            index[0]=this->BackgroundIndex[0];
            index[1]=this->BackgroundIndex[1];
            index[2]=this->BackgroundIndex[2];
            //Where is this defined???
            //if(flag->pyramidFlag){
             //   for(int l=CurrentLevel; l<(this->NumberOfLevels-1); l++){
             //       index[0] /= 2;
             //       index[1] /= 2;
             //       index[2] /= 2;
            //   }
            //}
            this->SourceBackgroundValue = (T)(reg_tool_GetIntensityValue(sourceImage, index));
        }
        else this->SourceBackgroundValue = 0;

        /* initialise the block matching */
        _reg_blockMatchingParam blockMatchingParams;
        initialise_block_matching_method(targetImage,
                                         &blockMatchingParams,
                                         percentageOfBlockToUse,    // percentage of block kept
                                         this->InlierLts,         // percentage of inlier in the optimisation process
                                         targetMask,
                                         this->UseGpu);

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

        if(this->UseGpu){
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
        printf("Current level %i / %i\n", CurrentLevel+1, this->NumberOfLevels);
        printf("Target image size: \t%ix%ix%i voxels\t%gx%gx%g mm\n",
               targetImage->nx, targetImage->ny, targetImage->nz, targetImage->dx, targetImage->dy, targetImage->dz);
        printf("Source image size: \t%ix%ix%i voxels\t%gx%gx%g mm\n",
               sourceImage->nx, sourceImage->ny, sourceImage->nz, sourceImage->dx, sourceImage->dy, sourceImage->dz);
        if(this->ImageDimension==2)
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
        reg_mat44_disp(this->OutputTransform, (char *)"Initial affine transformation:");

        /* ****************** */
        /* Rigid registration */
        /* ****************** */
        int iteration=0;

        if((this->PerformRigid && !this->PerformAffine) || (this->PerformAffine && this->PerformRigid && CurrentLevel==0))
        {
            int ratio=1;
            if(this->PerformAffine && this->PerformRigid && CurrentLevel==0) ratio=4;
            while(iteration<maxNumberOfIterationToPerform*ratio)
            {
                /* Compute the affine transformation deformation field */
#ifdef _USE_CUDA
                if(this->UseGpu)
                {
                    reg_affine_positionField_gpu(	this->InputTransform,
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
                                                this->SourceBackgroundValue);
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
                else
                {
#endif
                    reg_affine_positionField(	this->OutputTransform,
                                                targetImage,
                                                positionFieldImage);
                    /* Resample the source image */
                    reg_resampleSourceImage(targetImage,
                                            sourceImage,
                                            resultImage,
                                            positionFieldImage,
                                            targetMask,
                                            this->Interpolation,
                                            this->SourceBackgroundValue);
                    /* Compute the correspondances between blocks */
                    block_matching_method<T>(   targetImage,
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
                // the affine transformation is updated. We don't have this
                //variable, so this should be the output transformation
                *(this->OutputTransform) = reg_mat44_mul( this->OutputTransform, &(updateAffineMatrix));
#ifndef NDEBUG
                printf("[DEBUG] -Rigid- iteration %i - ",iteration);
                reg_mat44_disp(&updateAffineMatrix, (char *)"[DEBUG] updateMatrix");
                reg_mat44_disp(this->OutputTransform, (char *)"[DEBUG] updated affine");
#endif

                if(this->TestMatrixConvergence(&updateAffineMatrix)) break;
                iteration++;
            }
        }

        /* ******************* */
        /* Affine registration */
        /* ******************* */
        iteration=0;
        if(this->PerformAffine)
        {
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
                    reg_affine_positionField(	this->OutputTransform,
                                                targetImage,
                                                positionFieldImage);
                    /* Resample the source image */
                    reg_resampleSourceImage(	targetImage,
                                                sourceImage,
                                                resultImage,
                                                positionFieldImage,
                                                targetMask,
                                                this->Interpolation,
                                                this->SourceBackgroundValue);
                    /* Compute the correspondances between blocks */
                    block_matching_method<T>(	targetImage,
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
                *(this->OutputTransform) = reg_mat44_mul( this->OutputTransform, &(updateAffineMatrix));
#ifndef NDEBUG
                printf("[DEBUG] -Affine- iteration %i - ",iteration);
                reg_mat44_disp(&updateAffineMatrix, (char *)"[DEBUG] updateMatrix");
                reg_mat44_disp(this->OutputTransform, (char *)"[DEBUG] updated affine");
#endif
                if(this->TestMatrixConvergence(&updateAffineMatrix)) break;
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

        if(CurrentLevel==(this->LevelsToPerform-1))
        {
            /* ****************** */
            /* OUTPUT THE RESULTS */
            /* ****************** */
#ifdef _USE_CUDA
            if(this->UseGpu && this->LevelsToPerform==this->NumberOfLevels)
                positionFieldImage->data = (void *)calloc(positionFieldImage->nvox, positionFieldImage->nbyper);
#endif
            if(this->LevelsToPerform != this->NumberOfLevels)
            {
                if(positionFieldImage->data)free(positionFieldImage->data);
                if(positionFieldImage) free(positionFieldImage);
                fprintf(stderr,"Position Field image freed.");
                positionFieldImage = nifti_copy_nim_info(this->InputReference);
                positionFieldImage->dim[0]=positionFieldImage->ndim=5;
                positionFieldImage->dim[1]=positionFieldImage->nx=this->InputReference->nx;
                positionFieldImage->dim[2]=positionFieldImage->ny=this->InputReference->ny;
                positionFieldImage->dim[3]=positionFieldImage->nz=this->InputReference->nz;
                positionFieldImage->dim[4]=positionFieldImage->nt=1;
                if(this->ImageDimension==2) positionFieldImage->dim[5]=positionFieldImage->nu=2;
                else positionFieldImage->dim[5]=positionFieldImage->nu=3;
                positionFieldImage->dim[6]=positionFieldImage->nv=1;
                positionFieldImage->dim[7]=positionFieldImage->nw=1;
                positionFieldImage->nvox=positionFieldImage->nx*positionFieldImage->ny*
                                         positionFieldImage->nz*positionFieldImage->nt*
                                         positionFieldImage->nu;
                positionFieldImage->pixdim[5]=positionFieldImage->du=1.0;
                positionFieldImage->pixdim[4]=positionFieldImage->dt=1.0;
                positionFieldImage->pixdim[6]=positionFieldImage->dv=1.0;
                positionFieldImage->pixdim[7]=positionFieldImage->dw=1.0;
                if(sizeof(T)==4) positionFieldImage->datatype = NIFTI_TYPE_FLOAT32;
                else positionFieldImage->datatype = NIFTI_TYPE_FLOAT64;
                positionFieldImage->nbyper = sizeof(T);
                positionFieldImage->data = (void *)calloc(positionFieldImage->nvox, positionFieldImage->nbyper);
            }

            /* The corresponding deformation field is evaluated and saved */
            reg_affine_positionField(	this->OutputTransform,
                                        this->InputReference,
                                        positionFieldImage);

            /* The result image is resampled using a cubic spline interpolation */
           //Might want to make this a member of the class rather than a temp value
            this->OutputImage = nifti_copy_nim_info(this->InputReference);
            this->OutputImage->cal_min=this->InputFloating->cal_min;
            this->OutputImage->cal_max=this->InputFloating->cal_max;
            this->OutputImage->scl_slope=this->InputFloating->scl_slope;
            this->OutputImage->scl_inter=this->InputFloating->scl_inter;
            this->OutputImage->datatype = this->InputFloating->datatype;
            this->OutputImage->nbyper = this->InputFloating->nbyper;
            this->OutputImage->nt = this->OutputImage->dim[4] = this->InputFloating->nt;
            this->OutputImage->nvox=this->OutputImage->nx*this->OutputImage->ny*this->OutputImage->nz*this->OutputImage->nt;
            this->OutputImage->data = (void *)calloc(this->OutputImage->nvox, this->OutputImage->nbyper);
            reg_resampleSourceImage(this->InputReference,
                                    this->InputFloating,
                                    this->OutputImage,
                                    positionFieldImage,
                                    NULL,
                                    3,
                                    this->SourceBackgroundValue);

        }
        nifti_image_free(positionFieldImage);
        nifti_image_free(resultImage);
        nifti_image_free(targetImage);
        nifti_image_free(sourceImage);
        reg_mat44_disp(this->OutputTransform, (char *)"Final affine transformation:");
#ifndef NDEBUG
        mat33 tempMat;
        for(int i=0; i<3; i++){
            for(int j=0; j<3; j++){
                tempMat.m[i][j] = this->OutputTransform->m[i][j];
            }
        }
        printf("[DEBUG] Matrix determinant %g\n", nifti_mat33_determ(tempMat));
#endif
        printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\n");
    }

}

#endif //#ifndef _REG_ALADIN_CPP
