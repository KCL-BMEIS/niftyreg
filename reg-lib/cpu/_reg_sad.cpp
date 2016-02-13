#include "_reg_sad.h"

//#define USE_LOG_SAD

/* *************************************************************** */
/* *************************************************************** */
reg_sad::reg_sad()
    : reg_measure()
{
#ifndef NDEBUG
    reg_print_msg_debug("reg_sad constructor called");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
void reg_sad::InitialiseMeasure(nifti_image *refImgPtr,
                                nifti_image *floImgPtr,
                                int *maskRefPtr,
                                nifti_image *warFloImgPtr,
                                nifti_image *warFloGraPtr,
                                nifti_image *forVoxBasedGraPtr,
                                int *maskFloPtr,
                                nifti_image *warRefImgPtr,
                                nifti_image *warRefGraPtr,
                                nifti_image *bckVoxBasedGraPtr)
{
    // Set the pointers using the parent class function
    reg_measure::InitialiseMeasure(refImgPtr,
                                   floImgPtr,
                                   maskRefPtr,
                                   warFloImgPtr,
                                   warFloGraPtr,
                                   forVoxBasedGraPtr,
                                   maskFloPtr,
                                   warRefImgPtr,
                                   warRefGraPtr,
                                   bckVoxBasedGraPtr);

    // Check that the input images have the same number of time point
    if(this->referenceImagePointer->nt != this->floatingImagePointer->nt)
    {
        reg_print_fct_error("reg_sad::InitialiseMeasure");
        reg_print_msg_error("This number of time point should be the same for both input images");
        reg_exit();
    }
    // Input images are normalised between 0 and 1
    for(int i=0; i<this->referenceImagePointer->nt; ++i)
    {
        if(this->activeTimePoint[i])
        {
            reg_intensityRescale(this->referenceImagePointer,
                                 i,
                                 0.f,
                                 1.f);
            reg_intensityRescale(this->floatingImagePointer,
                                 i,
                                 0.f,
                                 1.f);
        }
    }
#ifndef NDEBUG
    char text[255];
    reg_print_msg_debug("reg_sad::InitialiseMeasure().");
    sprintf(text, "Active time point:");
    for(int i=0; i<this->referenceImagePointer->nt; ++i)
        if(this->activeTimePoint[i])
            sprintf(text, "%s %i", text, i);
    reg_print_msg_debug(text);
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
double reg_getSADValue(nifti_image *referenceImage,
                       nifti_image *warpedImage,
                       bool *activeTimePoint,
                       nifti_image *jacobianDetImage,
                       int *mask,
                       float *currentValue
                       )
{

#ifdef _WIN32
    long voxel;
    long voxelNumber = (long)referenceImage->nx*referenceImage->ny*referenceImage->nz;
#else
    size_t voxel;
    size_t voxelNumber = (size_t)referenceImage->nx*referenceImage->ny*referenceImage->nz;
#endif
    // Create pointers to the reference and warped image data
    DTYPE *referencePtr=static_cast<DTYPE *>(referenceImage->data);
    DTYPE *warpedPtr=static_cast<DTYPE *>(warpedImage->data);
    // Create a pointer to the Jacobian determinant image if defined
    DTYPE *jacDetPtr=NULL;
    if(jacobianDetImage!=NULL)
        jacDetPtr=static_cast<DTYPE *>(jacobianDetImage->data);


    double SAD_global=0.0, n=0.0;
    double refValue, warValue, diff;

    // Loop over the different time points
    for(int time=0; time<referenceImage->nt; ++time)
    {
        if(activeTimePoint[time])
        {
            // Create pointers to the current time point of the reference and warped images
            DTYPE *currentRefPtr=&referencePtr[time*voxelNumber];
            DTYPE *currentWarPtr=&warpedPtr[time*voxelNumber];

            double SAD_local=0.;
            n=0.;
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
    shared(referenceImage, currentRefPtr, currentWarPtr, mask, \
    jacobianDetImage, jacDetPtr, voxelNumber) \
    private(voxel, refValue, warValue, diff) \
    reduction(+:SAD_local) \
    reduction(+:n)
#endif
            for(voxel=0; voxel<voxelNumber; ++voxel)
            {
                // Check if the current voxel belongs to the mask
                if(mask[voxel]>-1)
                {
                    // Ensure that both ref and warped values are defined
                    refValue = (double)(currentRefPtr[voxel] * referenceImage->scl_slope +
                                        referenceImage->scl_inter);
                    warValue = (double)(currentWarPtr[voxel] * referenceImage->scl_slope +
                                        referenceImage->scl_inter);
                    if(refValue==refValue && warValue==warValue)
                    {
                        diff = std::abs(refValue-warValue);
                        //						if(diff>0) diff=log(diff);
                        // Jacobian determinant modulation of the sad if required
                        if(jacDetPtr!=NULL)
                        {
                            SAD_local += diff * jacDetPtr[voxel];
                            n += jacDetPtr[voxel];
                        }
                        else
                        {
                            SAD_local += diff;
                            n += 1.0;
                        }
                    }
                }
            }
            currentValue[time]=-SAD_local;
            SAD_global -= SAD_local/n;
        }
    }
    return SAD_global;
}
template double reg_getSADValue<float>(nifti_image *,nifti_image *,bool *,nifti_image *,int *, float *);
template double reg_getSADValue<double>(nifti_image *,nifti_image *,bool *,nifti_image *,int *, float *);
/* *************************************************************** */
double reg_sad::GetSimilarityMeasureValue()
{
    // Check that all the specified image are of the same datatype
    if(this->warpedFloatingImagePointer->datatype != this->referenceImagePointer->datatype)
    {
        reg_print_fct_error("reg_sad::GetSimilarityMeasureValue");
        reg_print_msg_error("Both input images are exepected to have the same type");
        reg_exit();
    }
    double SADValue;
    switch(this->referenceImagePointer->datatype)
    {
    case NIFTI_TYPE_FLOAT32:
        SADValue = reg_getSADValue<float>
                   (this->referenceImagePointer,
                    this->warpedFloatingImagePointer,
                    this->activeTimePoint,
                    NULL, // HERE TODO this->forwardJacDetImagePointer,
                    this->referenceMaskPointer,
                    this->currentValue
                    );
        break;
    case NIFTI_TYPE_FLOAT64:
        SADValue = reg_getSADValue<double>
                   (this->referenceImagePointer,
                    this->warpedFloatingImagePointer,
                    this->activeTimePoint,
                    NULL, // HERE TODO this->forwardJacDetImagePointer,
                    this->referenceMaskPointer,
                    this->currentValue
                    );
        break;
    default:
        reg_print_fct_error("reg_sad::GetSimilarityMeasureValue");
        reg_print_msg_error("Warped pixel type unsupported");
        reg_exit();
    }

    // Backward computation
    if(this->isSymmetric)
    {
        // Check that all the specified image are of the same datatype
        if(this->warpedReferenceImagePointer->datatype != this->floatingImagePointer->datatype)
        {
            reg_print_fct_error("reg_sad::GetSimilarityMeasureValue");
            reg_print_msg_error("Both input images are exepected to have the same type");
            reg_exit();
        }
        switch(this->floatingImagePointer->datatype)
        {
        case NIFTI_TYPE_FLOAT32:
            SADValue += reg_getSADValue<float>
                        (this->floatingImagePointer,
                         this->warpedReferenceImagePointer,
                         this->activeTimePoint,
                         NULL, // HERE TODO this->backwardJacDetImagePointer,
                         this->floatingMaskPointer,
                         this->currentValue
                         );
            break;
        case NIFTI_TYPE_FLOAT64:
            SADValue += reg_getSADValue<double>
                        (this->floatingImagePointer,
                         this->warpedReferenceImagePointer,
                         this->activeTimePoint,
                         NULL, // HERE TODO this->backwardJacDetImagePointer,
                         this->floatingMaskPointer,
                         this->currentValue
                         );
            break;
        default:
            reg_print_fct_error("reg_sad::GetSimilarityMeasureValue");
            reg_print_msg_error("Warped pixel type unsupported");
            reg_exit();
        }
    }
    return SADValue;
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_getVoxelBasedSADGradient(nifti_image *referenceImage,
                                  nifti_image *warpedImage,
                                  bool *activeTimePoint,
                                  nifti_image *warImgGradient,
                                  nifti_image *sadGradientImage,
                                  nifti_image *jacobianDetImage,
                                  int *mask)
{
    // Create pointers to the reference and warped images
#ifdef _WIN32
    long voxel;
    long voxelNumber = (long)referenceImage->nx*referenceImage->ny*referenceImage->nz;
#else
    size_t voxel;
    size_t voxelNumber = (size_t)referenceImage->nx*referenceImage->ny*referenceImage->nz;
#endif

    DTYPE *refPtr=static_cast<DTYPE *>(referenceImage->data);
    DTYPE *warPtr=static_cast<DTYPE *>(warpedImage->data);

    // Pointer to the warped image gradient
    DTYPE *spatialGradPtr=static_cast<DTYPE *>(warImgGradient->data);

    // Create a pointer to the Jacobian determinant values if defined
    DTYPE *jacDetPtr=NULL;
    if(jacobianDetImage!=NULL)
        jacDetPtr=static_cast<DTYPE *>(jacobianDetImage->data);

    // Create an array to store the computed gradient per time point
    DTYPE *sadGradPtrX=static_cast<DTYPE *>(sadGradientImage->data);
    DTYPE *sadGradPtrY = &sadGradPtrX[voxelNumber];
    DTYPE *sadGradPtrZ = NULL;
    if(referenceImage->nz>1)
        sadGradPtrZ = &sadGradPtrY[voxelNumber];

    double refValue, warValue, common;
    // Loop over the different time points
    for(int time=0; time<referenceImage->nt; ++time)
    {
        if(activeTimePoint[time])
        {
            // Create some pointers to the current time point image to be accessed
            DTYPE *currentRefPtr=&refPtr[time*voxelNumber];
            DTYPE *currentWarPtr=&warPtr[time*voxelNumber];

            // Create some pointers to the spatial gradient of the current warped volume
            DTYPE *spatialGradPtrX = &spatialGradPtr[time*voxelNumber];
            DTYPE *spatialGradPtrY = &spatialGradPtr[(referenceImage->nt+time)*voxelNumber];
            DTYPE *spatialGradPtrZ = NULL;
            if(referenceImage->nz>1)
                spatialGradPtrZ=&spatialGradPtr[(2*referenceImage->nt+time)*voxelNumber];


#if defined (_OPENMP)
#pragma omp parallel for default(none) \
    shared(referenceImage, warpedImage, currentRefPtr, currentWarPtr, time, \
    mask, jacDetPtr, spatialGradPtrX, spatialGradPtrY, spatialGradPtrZ, \
    sadGradPtrX, sadGradPtrY, sadGradPtrZ, voxelNumber) \
    private(voxel, refValue, warValue, common)
#endif
            for(voxel=0; voxel<voxelNumber; voxel++)
            {
                if(mask[voxel]>-1)
                {
                    refValue = (double)(currentRefPtr[voxel] * referenceImage->scl_slope +
                                        referenceImage->scl_inter);
                    warValue = (double)(currentWarPtr[voxel] * warpedImage->scl_slope +
                                        warpedImage->scl_inter);
                    if(refValue==refValue && warValue==warValue)
                    {
                        common = -1.0 * std::abs(refValue - warValue);
                        if(jacDetPtr!=NULL)
                            common *= jacDetPtr[voxel];

                        if(spatialGradPtrX[voxel]==spatialGradPtrX[voxel])
                            sadGradPtrX[voxel] += (DTYPE)(common * spatialGradPtrX[voxel]);
                        if(spatialGradPtrY[voxel]==spatialGradPtrY[voxel])
                            sadGradPtrY[voxel] += (DTYPE)(common * spatialGradPtrY[voxel]);

                        if(sadGradPtrZ!=NULL)
                        {
                            if(spatialGradPtrZ[voxel]==spatialGradPtrZ[voxel])
                                sadGradPtrZ[voxel] += (DTYPE)(common * spatialGradPtrZ[voxel]);
                        }
                    }
                }
            }
        }
    }// loop over time points
}
/* *************************************************************** */
template void reg_getVoxelBasedSADGradient<float>
(nifti_image *,nifti_image *,bool *,nifti_image *,nifti_image *,nifti_image *, int *);
template void reg_getVoxelBasedSADGradient<double>
(nifti_image *,nifti_image *,bool *,nifti_image *,nifti_image *,nifti_image *, int *);
/* *************************************************************** */
void reg_sad::GetVoxelBasedSimilarityMeasureGradient()
{
    // Check if all required input images are of the same data type
    int dtype = this->referenceImagePointer->datatype;
    if(this->warpedFloatingImagePointer->datatype != dtype ||
            this->warpedFloatingGradientImagePointer->datatype != dtype ||
            this->forwardVoxelBasedGradientImagePointer->datatype != dtype
            )
    {
        reg_print_fct_error("reg_sad::GetVoxelBasedSimilarityMeasureGradient");
        reg_print_msg_error("Input images are exepected to be of the same type");
        reg_exit();
    }
    // Compute the gradient of the sad for the forward transformation
    switch(dtype)
    {
    case NIFTI_TYPE_FLOAT32:
        reg_getVoxelBasedSADGradient<float>
                (this->referenceImagePointer,
                 this->warpedFloatingImagePointer,
                 this->activeTimePoint,
                 this->warpedFloatingGradientImagePointer,
                 this->forwardVoxelBasedGradientImagePointer,
                 NULL, // HERE TODO this->forwardJacDetImagePointer,
                 this->referenceMaskPointer
                 );
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_getVoxelBasedSADGradient<double>
                (this->referenceImagePointer,
                 this->warpedFloatingImagePointer,
                 this->activeTimePoint,
                 this->warpedFloatingGradientImagePointer,
                 this->forwardVoxelBasedGradientImagePointer,
                 NULL, // HERE TODO this->forwardJacDetImagePointer,
                 this->referenceMaskPointer
                 );
        break;
    default:
        reg_print_fct_error("reg_sad::GetVoxelBasedSimilarityMeasureGradient");
        reg_print_msg_error("Unsupported datatype");
        reg_exit();
    }
    // Compute the gradient of the sad for the backward transformation
    if(this->isSymmetric)
    {
        dtype = this->floatingImagePointer->datatype;
        if(this->warpedReferenceImagePointer->datatype != dtype ||
                this->warpedReferenceGradientImagePointer->datatype != dtype ||
                this->backwardVoxelBasedGradientImagePointer->datatype != dtype
                )
        {
            reg_print_fct_error("reg_sad::GetVoxelBasedSimilarityMeasureGradient");
            reg_print_msg_error("Input images are exepected to be of the same type");
            reg_exit();
        }
        // Compute the gradient of the nmi for the backward transformation
        switch(dtype)
        {
        case NIFTI_TYPE_FLOAT32:
            reg_getVoxelBasedSADGradient<float>
                    (this->floatingImagePointer,
                     this->warpedReferenceImagePointer,
                     this->activeTimePoint,
                     this->warpedReferenceGradientImagePointer,
                     this->backwardVoxelBasedGradientImagePointer,
                     NULL, // HERE TODO this->backwardJacDetImagePointer,
                     this->floatingMaskPointer
                     );
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_getVoxelBasedSADGradient<double>
                    (this->floatingImagePointer,
                     this->warpedReferenceImagePointer,
                     this->activeTimePoint,
                     this->warpedReferenceGradientImagePointer,
                     this->backwardVoxelBasedGradientImagePointer,
                     NULL, // HERE TODO this->backwardJacDetImagePointer,
                     this->floatingMaskPointer
                     );
            break;
        default:
            reg_print_fct_error("reg_sad::GetVoxelBasedSimilarityMeasureGradient");
            reg_print_msg_error("Unsupported datatype");
            reg_exit();
        }
    }
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void GetDiscretisedValue_core3D(nifti_image *controlPointGridImage,
                                float *discretisedValue,
                                int discretise_radius,
                                int discretise_step,
                                nifti_image *refImage,
                                nifti_image *warImage,
                                int *mask,
                                float costWeight)
{
    int cpx, cpy, cpz, t, x, y, z, a, b, c, blockIndex, voxIndex, voxIndex_t, discretisedIndex;
    int nD_discrete_valueNumber = pow((discretise_radius / discretise_step) * 2 + 1, 3);
    //output matrix = discretisedValue (first dimension displacement label, second dim. control point)
    float gridVox[3], imageVox[3];
    float currentValue;
    // Define the transformation matrices
    mat44 *grid_vox2mm = &controlPointGridImage->qto_xyz;
    if(controlPointGridImage->sform_code>0)
        grid_vox2mm = &controlPointGridImage->sto_xyz;
    mat44 *image_mm2vox = &refImage->qto_ijk;
    if(refImage->sform_code>0)
        image_mm2vox = &refImage->sto_ijk;
    mat44 grid2img_vox = reg_mat44_mul(image_mm2vox, grid_vox2mm);

    // Compute the block size
    int blockSize[3]={
        (int)reg_ceil(controlPointGridImage->dx / refImage->dx),
        (int)reg_ceil(controlPointGridImage->dy / refImage->dy),
        (int)reg_ceil(controlPointGridImage->dz / refImage->dz),
    };
    int voxelBlockNumber = blockSize[0] * blockSize[1] * blockSize[2] * refImage->nt;
    // Allocate some static memory
    float refBlockValue[voxelBlockNumber];
    float warBlockValue[voxelBlockNumber];

    //DEBUG
    //int gridDim_x = refImage->nx / blockSize[0];
    //int gridDim_y = refImage->ny / blockSize[1];
    //int gridDim_z = refImage->nz / blockSize[2];
    //DEBUG

    // Pointers to the input image
    size_t voxelNumber = (size_t)refImage->nx *
                         refImage->ny * refImage->nz;
    DTYPE *refImgPtr = static_cast<DTYPE *>(refImage->data);
    DTYPE *warImgPtr = static_cast<DTYPE *>(warImage->data);
    DTYPE *currentRefPtr = NULL;
    DTYPE *currentWarPtr = NULL;

    // Loop over all control points
    for(cpz=0; cpz<controlPointGridImage->nz; ++cpz){
        gridVox[2] = cpz;
        for(cpy=0; cpy<controlPointGridImage->ny; ++cpy){
            gridVox[1] = cpy;
            for(cpx=0; cpx<controlPointGridImage->nx; ++cpx){
                gridVox[0] = cpx;
                //DEBUG
                //for(cpz=0;cpz<gridDim_z;cpz++) {
                //    for(cpy=0;cpy<gridDim_y;cpy++) {
                //        for(cpx=0;cpx<gridDim_x;cpx++) {
                //DEBUG
                // Compute the corresponding image voxel position
                reg_mat44_mul(&grid2img_vox, gridVox, imageVox);
                imageVox[0]=reg_round(imageVox[0]);
                imageVox[1]=reg_round(imageVox[1]);
                imageVox[2]=reg_round(imageVox[2]);
                //DEBUG
                //imageVox[0]=cpx*blockSize[0];
                //imageVox[1]=cpy*blockSize[1];
                //imageVox[2]=cpz*blockSize[2];
                //DEBUG
                // Extract the block in the reference image
                blockIndex = 0;
                for(z=imageVox[2]-blockSize[2]/2; z<imageVox[2]+blockSize[2]/2; ++z){
                    for(y=imageVox[1]-blockSize[1]/2; y<imageVox[1]+blockSize[1]/2; ++y){
                        for(x=imageVox[0]-blockSize[0]/2; x<imageVox[0]+blockSize[0]/2; ++x){
                //DEBUG
                //for(z=imageVox[2]; z<imageVox[2]+blockSize[2]; ++z){
                //    for(y=imageVox[1]; y<imageVox[1]+blockSize[1]; ++y){
                //        for(x=imageVox[0]; x<imageVox[0]+blockSize[0]; ++x){
                //DEBUG
                            if(x>-1 && x<refImage->nx && y>-1 && y<refImage->ny && z>-1 && z<refImage->nz) {
                                voxIndex = x+y*refImage->nx+z*refImage->nx*refImage->ny;
                            if(mask[voxIndex]>-1){
                                for(t=0; t<refImage->nt; ++t){
                                    voxIndex_t = voxIndex+t*refImage->nx*refImage->ny*refImage->nz;
                                    refBlockValue[blockIndex] = refImgPtr[voxIndex_t];
                                    blockIndex++;
                                } //t
                            }
                            } else {
                                for(t=0; t<refImage->nt; ++t){
                                    refBlockValue[blockIndex] = 0.f;
                                    blockIndex++;
                                }
                            }
                        } // x
                    } // y
                } // z

                // Loop over the discretised value
                int start_c=imageVox[2]-discretise_radius;
                int end_c=imageVox[2]+discretise_radius;

                discretisedIndex=0;
                for(c=start_c; c<=end_c; c+=discretise_step){
                    for(b=imageVox[1]-discretise_radius; b<=imageVox[1]+discretise_radius; b+=discretise_step){
                        for(a=imageVox[0]-discretise_radius; a<=imageVox[0]+discretise_radius; a+=discretise_step){
                            blockIndex = 0;
                            for(z=c-blockSize[2]/2; z<c+blockSize[2]/2; ++z){
                                for(y=b-blockSize[1]/2; y<b+blockSize[1]/2; ++y){
                                    for(x=a-blockSize[0]/2; x<a+blockSize[0]/2; ++x){
                            //DEBUG
                            //for(z=c; z<c+blockSize[2]; ++z){
                            //    for(y=b; y<b+blockSize[1]; ++y){
                            //        for(x=a; x<a+blockSize[0]; ++x){
                            //DEBUG
                                        if(x>-1 && x<warImage->nx && y>-1 && y<warImage->ny && z>-1 && z<warImage->nz){
                                            voxIndex = x+y*warImage->nx+z*warImage->nx*warImage->ny;
                                            for(t=0; t<warImage->nt; ++t){
                                                voxIndex_t = voxIndex+t*warImage->nx*warImage->ny*warImage->nz;
                                                warBlockValue[blockIndex]=warImgPtr[voxIndex_t];
                                                blockIndex++;
                                            }
                                        } else {
                                            for(t=0; t<warImage->nt; ++t){
                                                warBlockValue[blockIndex]=0.f;
                                                blockIndex++;
                                            }
                                        } // if defined
                                    } // x
                                } // y
                            } // z
                            currentValue = 0;
                            blockIndex = 0;
                            for(blockIndex = 0;blockIndex<voxelBlockNumber;blockIndex++) {
                                if((warBlockValue[blockIndex] == warBlockValue[blockIndex]) &&
                                   (refBlockValue[blockIndex] == refBlockValue[blockIndex])) {
                                        currentValue += std::abs(warBlockValue[blockIndex]-refBlockValue[blockIndex]);
                                }
                            }
                            currentValue=currentValue/(float)voxelBlockNumber;
                            //DEBUG
                            std::cout<<"currentValue="<<currentValue<<std::endl;
                            //DEBUG
                            discretisedValue[discretisedIndex+
                                    cpx*nD_discrete_valueNumber+
                                    cpy*nD_discrete_valueNumber*controlPointGridImage->nx+
                                    cpz*nD_discrete_valueNumber*controlPointGridImage->nx*controlPointGridImage->ny]=currentValue*costWeight;
                            //DEBUG
                            //discretisedValue[discretisedIndex+
                            //          cpx*nD_discrete_valueNumber+
                            //          cpy*nD_discrete_valueNumber*gridDim_x+
                            //          cpz*nD_discrete_valueNumber*gridDim_x*gridDim_y]=currentValue;
                            //DEBUG
                            ++discretisedIndex;
                        } // a
                    } // b
                } // c
            } // cpx
        } // cpy
    } // cpz
}
/* *************************************************************** */
template <class DTYPE>
void GetDiscretisedValue_core2D(nifti_image *controlPointGridImage,
                                float *discretisedValue,
                                int discretise_radius,
                                int discretise_step,
                                nifti_image *refImage,
                                nifti_image *warImage,
                                int *mask,
                                float costWeight)
{
    reg_print_fct_warn("GetDiscretisedValue_core2D");
    reg_print_msg_warn("No yet implemented");
    reg_exit();
}
/* *************************************************************** */
void reg_sad::GetDiscretisedValue(nifti_image *controlPointGridImage,
                                  float *discretisedValue,
                                  int discretise_radius,
                                  int discretise_step,
                                  float costWeight)
{
    if(referenceImagePointer->nz > 1) {
        switch(this->referenceImagePointer->datatype)
        {
        case NIFTI_TYPE_FLOAT32:
            GetDiscretisedValue_core3D<float>
                    (controlPointGridImage,
                     discretisedValue,
                     discretise_radius,
                     discretise_step,
                     this->referenceImagePointer,
                     this->warpedFloatingImagePointer,
                     this->referenceMaskPointer,
                     costWeight
                     );
            break;
        case NIFTI_TYPE_FLOAT64:
            GetDiscretisedValue_core3D<double>
                    (controlPointGridImage,
                     discretisedValue,
                     discretise_radius,
                     discretise_step,
                     this->referenceImagePointer,
                     this->warpedFloatingImagePointer,
                     this->referenceMaskPointer,
                     costWeight
                     );
            break;
        default:
            reg_print_fct_error("reg_sad::GetDiscretisedValue");
            reg_print_msg_error("Unsupported datatype");
            reg_exit();
        }
    } else {
        switch(this->referenceImagePointer->datatype)
        {
        case NIFTI_TYPE_FLOAT32:
            GetDiscretisedValue_core2D<float>
                    (controlPointGridImage,
                     discretisedValue,
                     discretise_radius,
                     discretise_step,
                     this->referenceImagePointer,
                     this->warpedFloatingImagePointer,
                     this->referenceMaskPointer,
                     costWeight
                     );
            break;
        case NIFTI_TYPE_FLOAT64:
            GetDiscretisedValue_core2D<double>
                    (controlPointGridImage,
                     discretisedValue,
                     discretise_radius,
                     discretise_step,
                     this->referenceImagePointer,
                     this->warpedFloatingImagePointer,
                     this->referenceMaskPointer,
                     costWeight
                     );
            break;
        default:
            reg_print_fct_error("reg_sad::GetDiscretisedValue");
            reg_print_msg_error("Unsupported datatype");
            reg_exit();
        }
    }
}
/* *************************************************************** */
