/*
 *  _reg_ssd.cpp
 *
 *
 *  Created by Marc Modat on 19/05/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_ssd.h"

//#define USE_LOG_SSD
//#define MRF_USE_SAD

/* *************************************************************** */
/* *************************************************************** */
reg_ssd::reg_ssd()
    : reg_measure()
{
    memset(this->normalizeTimePoint,0,255*sizeof(bool) );
#ifndef NDEBUG
    reg_print_msg_debug("reg_ssd constructor called");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
void reg_ssd::InitialiseMeasure(nifti_image *refImgPtr,
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
        reg_print_fct_error("reg_ssd::InitialiseMeasure");
        reg_print_msg_error("This number of time point should be the same for both input images");
        reg_exit();
    }
    // Input images are normalised between 0 and 1
    for(int i=0; i<this->referenceImagePointer->nt; ++i)
    {
        if(this->activeTimePoint[i] && normalizeTimePoint[i])
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
#ifdef MRF_USE_SAD
    reg_print_msg_warn("SAD is used instead of SSD");
#endif
#ifndef NDEBUG
    char text[255];
    reg_print_msg_debug("reg_ssd::InitialiseMeasure().");
    sprintf(text, "Active time point:");
    for(int i=0; i<this->referenceImagePointer->nt; ++i)
        if(this->activeTimePoint[i])
            sprintf(text, "%s %i", text, i);
    reg_print_msg_debug(text);
    sprintf(text, "Normalize time point:");
    for(int i=0; i<this->referenceImagePointer->nt; ++i)
        if(this->normalizeTimePoint[i])
            sprintf(text, "%s %i", text, i);
    reg_print_msg_debug(text);
#endif
}
/* *************************************************************** */
/* *************************************************************** */
void reg_ssd::SetNormalizeTimepoint(int timepoint, bool normalize)
{
   this->normalizeTimePoint[timepoint]=normalize;
}
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
double reg_getSSDValue(nifti_image *referenceImage,
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


    double SSD_global=0.0, n=0.0;
    double refValue, warValue, diff;

    // Loop over the different time points
    for(int time=0; time<referenceImage->nt; ++time)
    {
        if(activeTimePoint[time])
        {
            // Create pointers to the current time point of the reference and warped images
            DTYPE *currentRefPtr=&referencePtr[time*voxelNumber];
            DTYPE *currentWarPtr=&warpedPtr[time*voxelNumber];

            double SSD_local=0.;
            n=0.;
            int nRef = 0;
            int nWar = 0;
            int nMask = 0;
/*#if defined (_OPENMP)
#pragma omp parallel for default(none) \
    shared(referenceImage, currentRefPtr, currentWarPtr, mask, \
    jacobianDetImage, jacDetPtr, voxelNumber) \
    private(voxel, refValue, warValue, diff) \
    reduction(+:SSD_local) \
    reduction(+:n)
#endif
*/
            for(voxel=0; voxel<voxelNumber; ++voxel)
            {
                // Ensure that both ref and warped values are defined
                refValue = (double)(currentRefPtr[voxel] * referenceImage->scl_slope +
                                    referenceImage->scl_inter);
                warValue = (double)(currentWarPtr[voxel] * referenceImage->scl_slope +
                                    referenceImage->scl_inter);
                //
                //DEBUG
                if(refValue == refValue) {
                    nRef = nRef + 1;
                }
                if(warValue == warValue) {
                    nWar = nWar + 1;
                }
                //DEBUG
                //
                // Check if the current voxel belongs to the mask
                if(mask[voxel]>-1)
                {
                    nMask = nMask +1;
                    // Ensure that both ref and warped values are defined
                    refValue = (double)(currentRefPtr[voxel] * referenceImage->scl_slope +
                                        referenceImage->scl_inter);
                    warValue = (double)(currentWarPtr[voxel] * referenceImage->scl_slope +
                                        referenceImage->scl_inter);
                    //DEBUG
//                    if(refValue == refValue) {
//                        nRef = nRef + 1;
//                    }
//                    if(warValue == warValue) {
//                        nWar = nWar + 1;
//                    }
                    //DEBUG
                    if(refValue==refValue && warValue==warValue)
                    {
#ifdef MRF_USE_SAD
                        diff = fabs(refValue-warValue);
#else
                        diff = reg_pow2(refValue-warValue);
#endif
                        // Jacobian determinant modulation of the ssd if required
                        if(jacDetPtr!=NULL)
                        {
                            SSD_local += diff * jacDetPtr[voxel];
                            n += jacDetPtr[voxel];
                        }
                        else
                        {
                            SSD_local += diff;
                            n += 1.0;
                        }
                    }
                }
            }
            currentValue[time]=-SSD_local;
            SSD_global -= SSD_local/n;
        }
    }
    return SSD_global;
}
template double reg_getSSDValue<float>(nifti_image *,nifti_image *,bool *,nifti_image *,int *, float *);
template double reg_getSSDValue<double>(nifti_image *,nifti_image *,bool *,nifti_image *,int *, float *);
/* *************************************************************** */
double reg_ssd::GetSimilarityMeasureValue()
{
    // Check that all the specified image are of the same datatype
    if(this->warpedFloatingImagePointer->datatype != this->referenceImagePointer->datatype)
    {
        reg_print_fct_error("reg_ssd::GetSimilarityMeasureValue");
        reg_print_msg_error("Both input images are exepected to have the same type");
        reg_exit();
    }
    double SSDValue=0;
    switch(this->referenceImagePointer->datatype)
    {
    case NIFTI_TYPE_FLOAT32:
        SSDValue = reg_getSSDValue<float>
                   (this->referenceImagePointer,
                    this->warpedFloatingImagePointer,
                    this->activeTimePoint,
                    NULL, // HERE TODO this->forwardJacDetImagePointer,
                    this->referenceMaskPointer,
                    this->currentValue
                    );
        break;
    case NIFTI_TYPE_FLOAT64:
        SSDValue = reg_getSSDValue<double>
                   (this->referenceImagePointer,
                    this->warpedFloatingImagePointer,
                    this->activeTimePoint,
                    NULL, // HERE TODO this->forwardJacDetImagePointer,
                    this->referenceMaskPointer,
                    this->currentValue
                    );
        break;
    default:
        reg_print_fct_error("reg_ssd::GetSimilarityMeasureValue");
        reg_print_msg_error("Warped pixel type unsupported");
        reg_exit();
    }

    // Backward computation
    if(this->isSymmetric)
    {
        // Check that all the specified image are of the same datatype
        if(this->warpedReferenceImagePointer->datatype != this->floatingImagePointer->datatype)
        {
            reg_print_fct_error("reg_ssd::GetSimilarityMeasureValue");
            reg_print_msg_error("Both input images are exepected to have the same type");
            reg_exit();
        }
        switch(this->floatingImagePointer->datatype)
        {
        case NIFTI_TYPE_FLOAT32:
            SSDValue += reg_getSSDValue<float>
                        (this->floatingImagePointer,
                         this->warpedReferenceImagePointer,
                         this->activeTimePoint,
                         NULL, // HERE TODO this->backwardJacDetImagePointer,
                         this->floatingMaskPointer,
                         this->currentValue
                         );
            break;
        case NIFTI_TYPE_FLOAT64:
            SSDValue += reg_getSSDValue<double>
                        (this->floatingImagePointer,
                         this->warpedReferenceImagePointer,
                         this->activeTimePoint,
                         NULL, // HERE TODO this->backwardJacDetImagePointer,
                         this->floatingMaskPointer,
                         this->currentValue
                         );
            break;
        default:
            reg_print_fct_error("reg_ssd::GetSimilarityMeasureValue");
            reg_print_msg_error("Warped pixel type unsupported");
            reg_exit();
        }
    }
    return SSDValue;
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_getVoxelBasedSSDGradient(nifti_image *referenceImage,
                                  nifti_image *warpedImage,
                                  nifti_image *warImgGradient,
                                  nifti_image *measureGradientImage,
                                  nifti_image *jacobianDetImage,
                                  int *mask,
                                  int current_timepoint)
{
    if(current_timepoint<0 || current_timepoint>=referenceImage->nt){
        reg_print_fct_error("reg_getVoxelBasedNMIGradient2D");
        reg_print_msg_error("The specified active timepoint is not defined in the ref/war images");
        reg_exit();
    }
    // Create pointers to the reference and warped images
#ifdef _WIN32
    long voxel;
    long voxelNumber = (long)referenceImage->nx*referenceImage->ny*referenceImage->nz;
#else
    size_t voxel;
    size_t voxelNumber = (size_t)referenceImage->nx*referenceImage->ny*referenceImage->nz;
#endif
    // Pointers to the image data
    DTYPE *refImagePtr = static_cast<DTYPE *>(referenceImage->data);
    DTYPE *currentRefPtr=&refImagePtr[current_timepoint*voxelNumber];
    DTYPE *warImagePtr = static_cast<DTYPE *>(warpedImage->data);
    DTYPE *currentWarPtr=&warImagePtr[current_timepoint*voxelNumber];

    // Pointers to the spatial gradient of the warped image
    DTYPE *spatialGradPtrX = static_cast<DTYPE *>(warImgGradient->data);
    DTYPE *spatialGradPtrY = &spatialGradPtrX[voxelNumber];
    DTYPE *spatialGradPtrZ = NULL;
    if(referenceImage->nz>1)
        spatialGradPtrZ=&spatialGradPtrY[voxelNumber];

    // Pointers to the measure of similarity gradient
    DTYPE *measureGradPtrX = static_cast<DTYPE *>(measureGradientImage->data);
    DTYPE *measureGradPtrY = &measureGradPtrX[voxelNumber];
    DTYPE *measureGradPtrZ = NULL;
    if(referenceImage->nz>1)
        measureGradPtrZ=&measureGradPtrY[voxelNumber];

    // Create a pointer to the Jacobian determinant values if defined
    DTYPE *jacDetPtr=NULL;
    if(jacobianDetImage!=NULL)
        jacDetPtr=static_cast<DTYPE *>(jacobianDetImage->data);

    double refValue, warValue, common;

#if defined (_OPENMP)
#pragma omp parallel for default(none) \
    shared(referenceImage, warpedImage, currentRefPtr, currentWarPtr, \
    mask, jacDetPtr, spatialGradPtrX, spatialGradPtrY, spatialGradPtrZ, \
    measureGradPtrX, measureGradPtrY, measureGradPtrZ, voxelNumber) \
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
#ifdef MRF_USE_SAD
                common = refValue>warValue?-1.f:1.f;
                common *= (refValue - warValue);
#else
                common = -2.0 * (refValue - warValue) / (float)referenceImage->nt;
#endif
                if(jacDetPtr!=NULL)
                    common *= jacDetPtr[voxel];

                if(spatialGradPtrX[voxel]==spatialGradPtrX[voxel])
                    measureGradPtrX[voxel] += (DTYPE)(common * spatialGradPtrX[voxel]);
                if(spatialGradPtrY[voxel]==spatialGradPtrY[voxel])
                    measureGradPtrY[voxel] += (DTYPE)(common * spatialGradPtrY[voxel]);

                if(measureGradPtrZ!=NULL)
                {
                    if(spatialGradPtrZ[voxel]==spatialGradPtrZ[voxel])
                        measureGradPtrZ[voxel] += (DTYPE)(common * spatialGradPtrZ[voxel]);
                }
            }
        }
    }
}
/* *************************************************************** */
template void reg_getVoxelBasedSSDGradient<float>
(nifti_image *,nifti_image *,nifti_image *,nifti_image *,nifti_image *, int *, int);
template void reg_getVoxelBasedSSDGradient<double>
(nifti_image *,nifti_image *,nifti_image *,nifti_image *,nifti_image *, int *, int);
/* *************************************************************** */
void reg_ssd::GetVoxelBasedSimilarityMeasureGradient(int current_timepoint)
{
    // Check if the specified time point exists and is active
    reg_measure::GetVoxelBasedSimilarityMeasureGradient(current_timepoint);
    if(this->activeTimePoint[current_timepoint]==false)
        return;

    // Check if all required input images are of the same data type
    int dtype = this->referenceImagePointer->datatype;
    if(this->warpedFloatingImagePointer->datatype != dtype ||
            this->warpedFloatingGradientImagePointer->datatype != dtype ||
            this->forwardVoxelBasedGradientImagePointer->datatype != dtype
            )
    {
        reg_print_fct_error("reg_ssd::GetVoxelBasedSimilarityMeasureGradient");
        reg_print_msg_error("Input images are exepected to be of the same type");
        reg_exit();
    }
    // Compute the gradient of the ssd for the forward transformation
    switch(dtype)
    {
    case NIFTI_TYPE_FLOAT32:
        reg_getVoxelBasedSSDGradient<float>
                (this->referenceImagePointer,
                 this->warpedFloatingImagePointer,
                 this->warpedFloatingGradientImagePointer,
                 this->forwardVoxelBasedGradientImagePointer,
                 NULL, // HERE TODO this->forwardJacDetImagePointer,
                 this->referenceMaskPointer,
                 current_timepoint
                 );
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_getVoxelBasedSSDGradient<double>
                (this->referenceImagePointer,
                 this->warpedFloatingImagePointer,
                 this->warpedFloatingGradientImagePointer,
                 this->forwardVoxelBasedGradientImagePointer,
                 NULL, // HERE TODO this->forwardJacDetImagePointer,
                 this->referenceMaskPointer,
                 current_timepoint
                 );
        break;
    default:
        reg_print_fct_error("reg_ssd::GetVoxelBasedSimilarityMeasureGradient");
        reg_print_msg_error("Unsupported datatype");
        reg_exit();
    }
    // Compute the gradient of the ssd for the backward transformation
    if(this->isSymmetric)
    {
        dtype = this->floatingImagePointer->datatype;
        if(this->warpedReferenceImagePointer->datatype != dtype ||
                this->warpedReferenceGradientImagePointer->datatype != dtype ||
                this->backwardVoxelBasedGradientImagePointer->datatype != dtype
                )
        {
            reg_print_fct_error("reg_ssd::GetVoxelBasedSimilarityMeasureGradient");
            reg_print_msg_error("Input images are exepected to be of the same type");
            reg_exit();
        }
        // Compute the gradient of the nmi for the backward transformation
        switch(dtype)
        {
        case NIFTI_TYPE_FLOAT32:
            reg_getVoxelBasedSSDGradient<float>
                    (this->floatingImagePointer,
                     this->warpedReferenceImagePointer,
                     this->warpedReferenceGradientImagePointer,
                     this->backwardVoxelBasedGradientImagePointer,
                     NULL, // HERE TODO this->backwardJacDetImagePointer,
                     this->floatingMaskPointer,
                     current_timepoint
                     );
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_getVoxelBasedSSDGradient<double>
                    (this->floatingImagePointer,
                     this->warpedReferenceImagePointer,
                     this->warpedReferenceGradientImagePointer,
                     this->backwardVoxelBasedGradientImagePointer,
                     NULL, // HERE TODO this->backwardJacDetImagePointer,
                     this->floatingMaskPointer,
                     current_timepoint
                     );
            break;
        default:
            reg_print_fct_error("reg_ssd::GetVoxelBasedSimilarityMeasureGradient");
            reg_print_msg_error("Unsupported datatype");
            reg_exit();
        }
    }
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void GetDiscretisedValueSSD_core3D(nifti_image *controlPointGridImage,
                                   float *discretisedValue,
                                   int discretise_radius,
                                   int discretise_step,
                                   nifti_image *refImage,
                                   nifti_image *warImage,
                                   int *mask)
{
    int cpx, cpy, cpz, t, x, y, z, a, b, c, blockIndex, discretisedIndex;
    size_t voxIndex, voxIndex_t;
    int label_1D_number = (discretise_radius / discretise_step) * 2 + 1;
    int label_2D_number = label_1D_number*label_1D_number;
    int label_nD_number = label_2D_number*label_1D_number;
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
    int currentControlPoint = 0;

    // Allocate some static memory
    float* refBlockValue = (float *) malloc(voxelBlockNumber*sizeof(float));

    // Pointers to the input image
    size_t voxelNumber = (size_t)refImage->nx*
                         refImage->ny*refImage->nz;
    DTYPE *refImgPtr = static_cast<DTYPE *>(refImage->data);
    DTYPE *warImgPtr = static_cast<DTYPE *>(warImage->data);

    // Create a padded version of the warped image to avoid doundary condition check
    int warPaddedOffset [3] = {
        discretise_radius + blockSize[0],
        discretise_radius + blockSize[1],
        discretise_radius + blockSize[2],
    };
    int warPaddedDim[4] = {
        warImage->nx + 2 * warPaddedOffset[0] + blockSize[0],
        warImage->ny + 2 * warPaddedOffset[1] + blockSize[1],
        warImage->nz + 2 * warPaddedOffset[2] + blockSize[2],
        warImage->nt
    };

    //DTYPE padding_value = std::numeric_limits<DTYPE>::quiet_NaN();
    DTYPE padding_value = 0.0;

    size_t warPaddedVoxelNumber = (size_t)warPaddedDim[0] *
                                  warPaddedDim[1] * warPaddedDim[2];
    DTYPE *paddedWarImgPtr = (DTYPE *)calloc(warPaddedVoxelNumber*warPaddedDim[3], sizeof(DTYPE));
    for(voxIndex=0; voxIndex<warPaddedVoxelNumber*warPaddedDim[3]; ++voxIndex)
        paddedWarImgPtr[voxIndex]=padding_value;
    voxIndex=0;
    voxIndex_t=0;
    for(t=0; t<warImage->nt; ++t){
        for(z=warPaddedOffset[2]; z<warPaddedDim[2]-warPaddedOffset[2]-blockSize[2]; ++z){
            for(y=warPaddedOffset[1]; y<warPaddedDim[1]-warPaddedOffset[1]-blockSize[1]; ++y){
                voxIndex= t * warPaddedVoxelNumber + (z*warPaddedDim[1]+y)*warPaddedDim[0]+warPaddedOffset[0];
                for(x=warPaddedOffset[0]; x<warPaddedDim[0]-warPaddedOffset[0]-blockSize[0]; ++x){
                    paddedWarImgPtr[voxIndex]=warImgPtr[voxIndex_t];
                    ++voxIndex;
                    ++voxIndex_t;
                }
            }
        }
    }

    int definedValueNumber;

    // Loop over all control points
    for(cpz=1; cpz<controlPointGridImage->nz-1; ++cpz){
        gridVox[2] = cpz;
        for(cpy=1; cpy<controlPointGridImage->ny-1; ++cpy){
            gridVox[1] = cpy;
            currentControlPoint=(cpz*controlPointGridImage->ny+cpy)*controlPointGridImage->nx+1;
            for(cpx=1; cpx<controlPointGridImage->nx-1; ++cpx){
                gridVox[0] = cpx;
                // Compute the corresponding image voxel position
                reg_mat44_mul(&grid2img_vox, gridVox, imageVox);
                imageVox[0]=reg_round(imageVox[0]);
                imageVox[1]=reg_round(imageVox[1]);
                imageVox[2]=reg_round(imageVox[2]);

                // Extract the block in the reference image
                blockIndex = 0;
                definedValueNumber = 0;
                for(z=imageVox[2]-blockSize[2]/2; z<imageVox[2]+blockSize[2]/2; ++z){
                    for(y=imageVox[1]-blockSize[1]/2; y<imageVox[1]+blockSize[1]/2; ++y){
                        for(x=imageVox[0]-blockSize[0]/2; x<imageVox[0]+blockSize[0]/2; ++x){
                            if(x>-1 && x<refImage->nx && y>-1 && y<refImage->ny && z>-1 && z<refImage->nz) {
                                voxIndex = (z*refImage->ny+y)*refImage->nx+x;
                                if(mask[voxIndex]>-1){
                                    for(t=0; t<refImage->nt; ++t){
                                        voxIndex_t = t*voxelNumber + voxIndex;
                                        refBlockValue[blockIndex] = refImgPtr[voxIndex_t];
                                        if(refBlockValue[blockIndex]==refBlockValue[blockIndex])
                                            ++definedValueNumber;
                                        blockIndex++;
                                    } //t
                                }
                                else{
                                    for(t=0; t<refImage->nt; ++t){
                                        refBlockValue[blockIndex] = padding_value;
                                        blockIndex++;
                                    } // t
                                }
                            }
                            else{
                                for(t=0; t<refImage->nt; ++t){
                                    refBlockValue[blockIndex] = padding_value;
                                    blockIndex++;
                                } // t
                            } // mask
                        } // x
                    } // y
                } // z
                // Loop over the discretised value
                if(definedValueNumber>0){

                    DTYPE warpedValue;
                    int paddedImageVox[3] = {
                        imageVox[0]+warPaddedOffset[0],
                        imageVox[1]+warPaddedOffset[1],
                        imageVox[2]+warPaddedOffset[2]
                    };
                    int cc;
                    double currentSum;
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
    shared(label_1D_number, label_2D_number, label_nD_number, discretise_step, discretise_radius, \
    paddedImageVox, blockSize, warPaddedDim, paddedWarImgPtr, refBlockValue, warPaddedVoxelNumber, \
    discretisedValue, currentControlPoint, voxelBlockNumber) \
    private(a, b, c, cc, x, y, z, t, discretisedIndex, blockIndex, \
    currentValue, warpedValue, voxIndex, voxIndex_t, definedValueNumber, currentSum)
#endif
                    for(cc=0; cc<label_1D_number; ++cc){
                        discretisedIndex = cc * label_2D_number;
                        c = paddedImageVox[2]-discretise_radius + cc*discretise_step;
                        for(b=paddedImageVox[1]-discretise_radius; b<=paddedImageVox[1]+discretise_radius; b+=discretise_step){
                            for(a=paddedImageVox[0]-discretise_radius; a<=paddedImageVox[0]+discretise_radius; a+=discretise_step){

                                blockIndex = 0;
                                currentSum = 0.;
                                definedValueNumber = 0;

                                for(z=c-blockSize[2]/2; z<c+blockSize[2]/2; ++z){
                                    for(y=b-blockSize[1]/2; y<b+blockSize[1]/2; ++y){
                                        for(x=a-blockSize[0]/2; x<a+blockSize[0]/2; ++x){
                                            voxIndex = (z*warPaddedDim[1]+y)*warPaddedDim[0]+x;
                                            for(t=0; t<warPaddedDim[3]; ++t){
                                                voxIndex_t = t*warPaddedVoxelNumber + voxIndex;
                                                warpedValue = paddedWarImgPtr[voxIndex_t];
#ifdef MRF_USE_SAD
                                                currentValue = fabs(warpedValue-refBlockValue[blockIndex]);
#else
                                                currentValue = reg_pow2(warpedValue-refBlockValue[blockIndex]);
#endif
                                                if(currentValue==currentValue){
                                                    currentSum -= currentValue;
                                                    ++definedValueNumber;
                                                }
                                                blockIndex++;
                                            }
                                        } // x
                                    } // y
                                } // z
                                discretisedValue[currentControlPoint * label_nD_number + discretisedIndex] =
                                        currentSum / static_cast<float>(definedValueNumber);
                                ++discretisedIndex;
                            } // a
                        } // b
                    } // cc
                } // defined value in the reference block
                ++currentControlPoint;
            } // cpx
        } // cpy
    } // cpz
    free(paddedWarImgPtr);
    free(refBlockValue);
    // Deal with the labels that contains NaN values
    for(int node=0; node<controlPointGridImage->nx*controlPointGridImage->ny*controlPointGridImage->nz; ++node){
        int definedValueNumber=0;
        float *discretisedValuePtr = &discretisedValue[node * label_nD_number];
        float meanValue=0;
        for(int label=0; label<label_nD_number;++label){
            if(discretisedValuePtr[label]==discretisedValuePtr[label]){
                ++definedValueNumber;
                meanValue+=discretisedValuePtr[label];
            }
        }
        if(definedValueNumber==0){
            for(int label=0; label<label_nD_number;++label){
                discretisedValuePtr[label]=0;
            }
        }
        else if(definedValueNumber<label_nD_number){
            // Needs to be altered for efficiency
            int label=0;
            // Loop over all labels
            int label_x, label2_x, label_y, label2_y, label_z, label2_z, label2;
            float min_distance, current_distance;
            for(label_z=0; label_z<label_1D_number;++label_z){
                for(label_y=0; label_y<label_1D_number;++label_y){
                    for(label_x=0; label_x<label_1D_number;++label_x){
                        // check if the current label is defined
                        if(discretisedValuePtr[label]!=discretisedValuePtr[label]){
                            label2=0;
                            min_distance=std::numeric_limits<float>::max();
                            // Loop again over all label to detect the defined values
                            for(label2_z=0; label2_z<label_1D_number;++label2_z){
                                for(label2_y=0; label2_y<label_1D_number;++label2_y){
                                    for(label2_x=0; label2_x<label_1D_number;++label2_x){
                                        // Check if the value is defined
                                        if(discretisedValuePtr[label2]==discretisedValuePtr[label2]){
                                            // compute the distance between label and label2
                                            current_distance = reg_pow2(label_x-label2_x)+reg_pow2(label_y-label2_y)+reg_pow2(label_z-label2_z);
                                            if(current_distance<min_distance){
                                                min_distance=current_distance;
                                                discretisedValuePtr[label] = discretisedValuePtr[label2];
                                            }
                                        } // Check if label2 is defined
                                        ++label2;
                                    } // x
                                } // y
                            } // z
                        } // check if undefined label
                        ++label;
                    } //x
                } // y
            } // z

        } // node with undefined label
    } // node
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void GetDiscretisedValueSSD_core3D_2(nifti_image *controlPointGridImage,
                                     float *discretisedValue,
                                     int discretise_radius,
                                     int discretise_step,
                                     nifti_image *refImage,
                                     nifti_image *warImage,
                                     int *mask)
{
    //
    int cpx, cpy, cpz, t, x, y, z, a, b, c, blockIndex, blockIndex_t, discretisedIndex;
    size_t voxIndex, voxIndex_t;
    const int label_1D_number = (discretise_radius / discretise_step) * 2 + 1;
    const int label_2D_number = label_1D_number*label_1D_number;
    const int label_nD_number = label_2D_number*label_1D_number;
    //output matrix = discretisedValue (first dimension displacement label, second dim. control point)
    float gridVox[3], imageVox[3];
    float currentValue;
    double currentSum;
    // Define the transformation matrices
    mat44 *grid_vox2mm = &controlPointGridImage->qto_xyz;
    if(controlPointGridImage->sform_code>0)
        grid_vox2mm = &controlPointGridImage->sto_xyz;
    mat44 *image_mm2vox = &refImage->qto_ijk;
    if(refImage->sform_code>0)
        image_mm2vox = &refImage->sto_ijk;
    mat44 grid2img_vox = reg_mat44_mul(image_mm2vox, grid_vox2mm);

    // Compute the block size
    const int blockSize[3]={
        (int)reg_ceil(controlPointGridImage->dx / refImage->dx),
        (int)reg_ceil(controlPointGridImage->dy / refImage->dy),
        (int)reg_ceil(controlPointGridImage->dz / refImage->dz),
    };
    const int voxelBlockNumber = blockSize[0] * blockSize[1] * blockSize[2];
    const int voxelBlockNumber_t = blockSize[0] * blockSize[1] * blockSize[2] * refImage->nt;
    int currentControlPoint = 0;

    // Pointers to the input image
    const size_t voxelNumber = (size_t)refImage->nx*
                         refImage->ny*refImage->nz;
    DTYPE *refImgPtr = static_cast<DTYPE *>(refImage->data);
    DTYPE *warImgPtr = static_cast<DTYPE *>(warImage->data);

    DTYPE padding_value = 0.0;

    int definedValueNumber, idBlock, timeV;

    int threadNumber = 1;
    int tid = 0;
#if defined (_OPENMP)
    threadNumber=omp_get_max_threads();
#endif

    // Allocate some static memory
    float** refBlockValue = (float **) malloc(threadNumber*sizeof(float *));
    for(a=0;a<threadNumber;++a)
       refBlockValue[a] = (float *) malloc(voxelBlockNumber_t*sizeof(float));

    // Loop over all control points
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(controlPointGridImage, refImage, warImage, grid2img_vox, blockSize, \
   padding_value, refBlockValue, mask, refImgPtr, warImgPtr, discretise_radius, \
   discretise_step, discretisedValue) \
   private(cpx, cpy, cpz, x, y, z, a, b, c, t, currentControlPoint, gridVox, imageVox, \
   voxIndex, idBlock, blockIndex, definedValueNumber, tid, \
   timeV, voxIndex_t, blockIndex_t, discretisedIndex, currentSum, currentValue)
#endif
    for(cpz=0; cpz<controlPointGridImage->nz; ++cpz){
#if defined (_OPENMP)
       tid=omp_get_thread_num();
#endif
        gridVox[2] = cpz;
        for(cpy=0; cpy<controlPointGridImage->ny; ++cpy){
            gridVox[1] = cpy;
            for(cpx=0; cpx<controlPointGridImage->nx; ++cpx){
                gridVox[0] = cpx;
                currentControlPoint=controlPointGridImage->ny*controlPointGridImage->nx*cpz +
                      controlPointGridImage->nx*cpy+cpx;

                // Compute the corresponding image voxel position
                reg_mat44_mul(&grid2img_vox, gridVox, imageVox);
                imageVox[0]=reg_round(imageVox[0]);
                imageVox[1]=reg_round(imageVox[1]);
                imageVox[2]=reg_round(imageVox[2]);

                //INIT
                for(idBlock=0;idBlock<voxelBlockNumber_t;idBlock++) {
                    refBlockValue[tid][idBlock]=padding_value;
                }

                // Extract the block in the reference image
                blockIndex = 0;
                definedValueNumber = 0;
                for(z=imageVox[2]-blockSize[2]/2; z<imageVox[2]+blockSize[2]/2; ++z) {
                    for(y=imageVox[1]-blockSize[1]/2; y<imageVox[1]+blockSize[1]/2; ++y) {
                        for(x=imageVox[0]-blockSize[0]/2; x<imageVox[0]+blockSize[0]/2; ++x) {
                            if(x>-1 && x<refImage->nx && y>-1 && y<refImage->ny && z>-1 && z<refImage->nz) {
                                voxIndex = refImage->ny*refImage->nx*z+refImage->nx*y+x;
                                if(mask[voxIndex]>-1){
                                    for(timeV=0; timeV<refImage->nt; ++timeV){
                                        voxIndex_t = timeV*voxelNumber + voxIndex;
                                        blockIndex_t = timeV*voxelBlockNumber + blockIndex;
                                        refBlockValue[tid][blockIndex_t] = refImgPtr[voxIndex_t];
                                        if(refBlockValue[tid][blockIndex_t]==refBlockValue[tid][blockIndex_t]) {
                                            ++definedValueNumber;
                                        }
                                        else refBlockValue[tid][blockIndex_t] = 0;
                                    } // timeV
                                } //inside mask
                            } //inside image
                            blockIndex++;
                        } // x
                    } // y
                } // z
                // Loop over the discretised value
                if(definedValueNumber>0){

                    discretisedIndex=0;
                    for(c=imageVox[2]-discretise_radius; c<=imageVox[2]+discretise_radius; c+=discretise_step){
                        for(b=imageVox[1]-discretise_radius; b<=imageVox[1]+discretise_radius; b+=discretise_step){
                            for(a=imageVox[0]-discretise_radius; a<=imageVox[0]+discretise_radius; a+=discretise_step){

                                blockIndex = 0;
                                currentSum = 0.;
                                definedValueNumber = 0;

                                for(z=c-blockSize[2]/2; z<c+blockSize[2]/2; ++z){
                                    for(y=b-blockSize[1]/2; y<b+blockSize[1]/2; ++y){
                                        for(x=a-blockSize[0]/2; x<a+blockSize[0]/2; ++x){

                                            if(x>-1 && x<warImage->nx && y>-1 && y<warImage->ny && z>-1 && z<warImage->nz) {
                                                voxIndex = warImage->ny*warImage->nx*z+warImage->nx*y+x;
                                                for(t=0; t<warImage->nt; ++t){
                                                    voxIndex_t = t*voxelNumber + voxIndex;
                                                    blockIndex_t = t*voxelBlockNumber + blockIndex;
                                                    if(warImgPtr[voxIndex_t]==warImgPtr[voxIndex_t]) {
#ifdef MRF_USE_SAD
                                                    currentValue = fabs(warImgPtr[voxIndex_t]-refBlockValue[tid][blockIndex_t]);
#else
                                                    currentValue = reg_pow2(warImgPtr[voxIndex_t]-refBlockValue[tid][blockIndex_t]);
#endif
                                                    } else {
#ifdef MRF_USE_SAD
                                                    currentValue = fabs(0-refBlockValue[tid][blockIndex_t]);
#else
                                                    currentValue = reg_pow2(0-refBlockValue[tid][blockIndex_t]);
#endif
                                                    }

                                                    if(currentValue==currentValue){
                                                        currentSum -= currentValue;
                                                        ++definedValueNumber;
                                                    }
                                                }
                                            } //inside image
                                            else {
                                                for(t=0; t<warImage->nt; ++t){
                                                    blockIndex_t = t*voxelBlockNumber + blockIndex;
#ifdef MRF_USE_SAD
                                                    currentValue = fabs(0-refBlockValue[tid][blockIndex_t]);
#else
                                                    currentValue = reg_pow2(0-refBlockValue[tid][blockIndex_t]);
#endif
                                                    if(currentValue==currentValue){
                                                        currentSum -= currentValue;
                                                        ++definedValueNumber;
                                                    }
                                                }
                                            }
                                            blockIndex++;
                                        } // x
                                    } // y
                                } // z
                                discretisedValue[currentControlPoint * label_nD_number + discretisedIndex] = currentSum;
                                ++discretisedIndex;
                            } // a
                        } // b
                    } // cc
                } // defined value in the reference block
                ++currentControlPoint;
            } // cpx
        } // cpy
    } // cpz
    for(a=0;a<threadNumber;++a)
       free(refBlockValue[a]);
    free(refBlockValue);

    // Deal with the labels that contains NaN values
    for(int node=0; node<controlPointGridImage->nx*controlPointGridImage->ny*controlPointGridImage->nz; ++node){
        int definedValueNumber=0;
        float *discretisedValuePtr = &discretisedValue[node * label_nD_number];
        float meanValue=0;
        for(int label=0; label<label_nD_number;++label){
            if(discretisedValuePtr[label]==discretisedValuePtr[label]){
                ++definedValueNumber;
                meanValue+=discretisedValuePtr[label];
            }
        }
        if(definedValueNumber==0){
            for(int label=0; label<label_nD_number;++label){
                discretisedValuePtr[label]=0;
            }
        }
        else if(definedValueNumber<label_nD_number){
            // Needs to be altered for efficiency
            int label=0;
            // Loop over all labels
            int label_x, label2_x, label_y, label2_y, label_z, label2_z, label2;
            float min_distance, current_distance;
            for(label_z=0; label_z<label_1D_number;++label_z){
                for(label_y=0; label_y<label_1D_number;++label_y){
                    for(label_x=0; label_x<label_1D_number;++label_x){
                        // check if the current label is defined
                        if(discretisedValuePtr[label]!=discretisedValuePtr[label]){
                            label2=0;
                            min_distance=std::numeric_limits<float>::max();
                            // Loop again over all label to detect the defined values
                            for(label2_z=0; label2_z<label_1D_number;++label2_z){
                                for(label2_y=0; label2_y<label_1D_number;++label2_y){
                                    for(label2_x=0; label2_x<label_1D_number;++label2_x){
                                        // Check if the value is defined
                                        if(discretisedValuePtr[label2]==discretisedValuePtr[label2]){
                                            // compute the distance between label and label2
                                            current_distance = reg_pow2(label_x-label2_x)+reg_pow2(label_y-label2_y)+reg_pow2(label_z-label2_z);
                                            if(current_distance<min_distance){
                                                min_distance=current_distance;
                                                discretisedValuePtr[label] = discretisedValuePtr[label2];
                                            }
                                        } // Check if label2 is defined
                                        ++label2;
                                    } // x
                                } // y
                            } // z
                        } // check if undefined label
                        ++label;
                    } //x
                } // y
            } // z

        } // node with undefined label
    } // node
}
/* *************************************************************** */
template <class DTYPE>
void GetDiscretisedValueSSD_core2D(nifti_image *controlPointGridImage,
                                   float *discretisedValue,
                                   int discretise_radius,
                                   int discretise_step,
                                   nifti_image *refImage,
                                   nifti_image *warImage,
                                   int *mask)
{
    reg_print_fct_warn("GetDiscretisedValue_core2D");
    reg_print_msg_warn("No yet implemented");
    reg_exit();
}
/* *************************************************************** */
void reg_ssd::GetDiscretisedValue(nifti_image *controlPointGridImage,
                                  float *discretisedValue,
                                  int discretise_radius,
                                  int discretise_step)
{
    if(referenceImagePointer->nz > 1) {
        switch(this->referenceImagePointer->datatype)
        {
        case NIFTI_TYPE_FLOAT32:
            GetDiscretisedValueSSD_core3D_2<float>
                    (controlPointGridImage,
                     discretisedValue,
                     discretise_radius,
                     discretise_step,
                     this->referenceImagePointer,
                     this->warpedFloatingImagePointer,
                     this->referenceMaskPointer
                     );
            break;
        case NIFTI_TYPE_FLOAT64:
            GetDiscretisedValueSSD_core3D_2<double>
                    (controlPointGridImage,
                     discretisedValue,
                     discretise_radius,
                     discretise_step,
                     this->referenceImagePointer,
                     this->warpedFloatingImagePointer,
                     this->referenceMaskPointer
                     );
            break;
        default:
            reg_print_fct_error("reg_ssd::GetDiscretisedValue");
            reg_print_msg_error("Unsupported datatype");
            reg_exit();
        }
    } else {
        switch(this->referenceImagePointer->datatype)
        {
        case NIFTI_TYPE_FLOAT32:
            GetDiscretisedValueSSD_core2D<float>
                    (controlPointGridImage,
                     discretisedValue,
                     discretise_radius,
                     discretise_step,
                     this->referenceImagePointer,
                     this->warpedFloatingImagePointer,
                     this->referenceMaskPointer
                     );
            break;
        case NIFTI_TYPE_FLOAT64:
            GetDiscretisedValueSSD_core2D<double>
                    (controlPointGridImage,
                     discretisedValue,
                     discretise_radius,
                     discretise_step,
                     this->referenceImagePointer,
                     this->warpedFloatingImagePointer,
                     this->referenceMaskPointer
                     );
            break;
        default:
            reg_print_fct_error("reg_ssd::GetDiscretisedValue");
            reg_print_msg_error("Unsupported datatype");
            reg_exit();
        }
    }
}
/* *************************************************************** */
