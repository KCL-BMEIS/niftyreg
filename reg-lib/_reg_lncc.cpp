/**
 * @file  _reg_lncc.cpp
 *
 *
 *  Created by Aileen Cordes on 10/11/2012.
 *  Copyright (c) 2012, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_LNCC_CPP
#define _REG_LNCC_CPP

#include "_reg_lncc.h"

template<class DTYPE>
double reg_getLNCC1(nifti_image *referenceImage,
                    nifti_image *warpedImage,
                    float gaussianStandardDeviation,
                    int *mask
                    )
{
    nifti_image *localMeanReferenceImage = nifti_copy_nim_info(referenceImage);
    localMeanReferenceImage->data = (void *) malloc(referenceImage->nvox*sizeof(DTYPE));
    DTYPE *localMeanReferenceImage_ptr = static_cast<DTYPE *>(localMeanReferenceImage->data);
    memcpy(localMeanReferenceImage_ptr, referenceImage->data, referenceImage->nvox*sizeof(DTYPE));

    nifti_image *localCorrelationImage = nifti_copy_nim_info(warpedImage);
    localCorrelationImage->data = (void *) malloc(warpedImage->nvox*sizeof(DTYPE));
    DTYPE *localCorrelationImage_ptr = static_cast<DTYPE *>(localCorrelationImage->data);
    memcpy(localCorrelationImage_ptr, warpedImage->data, warpedImage->nvox*sizeof(DTYPE));

    nifti_image *localMeanWarpedImage = nifti_copy_nim_info(warpedImage);
    localMeanWarpedImage->data = (void *) malloc(warpedImage->nvox*sizeof(DTYPE));
    DTYPE *localMeanWarpedImage_ptr = static_cast<DTYPE *>(localMeanWarpedImage->data);
    memcpy(localMeanWarpedImage_ptr, warpedImage->data, warpedImage->nvox*sizeof(DTYPE));

    nifti_image *localStdReferenceImage = nifti_copy_nim_info(warpedImage);
    localStdReferenceImage->data = (void *) malloc(warpedImage->nvox*sizeof(DTYPE));
    DTYPE *localStdReferenceImage_ptr = static_cast<DTYPE *>(localStdReferenceImage->data);
    memcpy(localStdReferenceImage_ptr, warpedImage->data, warpedImage->nvox*sizeof(DTYPE));

    nifti_image *localStdWarpedImage = nifti_copy_nim_info(warpedImage);
    localStdWarpedImage->data = (void *) malloc(referenceImage->nvox*sizeof(DTYPE));
    DTYPE *localStdWarpedImage_ptr = static_cast<DTYPE *>(localStdWarpedImage->data);
    memcpy(localStdWarpedImage_ptr, warpedImage->data, referenceImage->nvox*sizeof(DTYPE));

    for(size_t voxel=0; voxel<referenceImage->nvox; ++voxel){
        if(localMeanWarpedImage_ptr[voxel]!=localMeanWarpedImage_ptr[voxel])
            localMeanReferenceImage_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
        if(localMeanReferenceImage_ptr[voxel]!=localMeanReferenceImage_ptr[voxel])
            localMeanWarpedImage_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
    }

    bool axis[8];
    axis[1]=true;
    axis[2]=true;
    if(referenceImage->nz>1)
        axis[3]=true;

    for(size_t voxel=0; voxel<referenceImage->nvox; ++voxel){
        if(mask[voxel]>-1){
            localCorrelationImage_ptr[voxel] = localMeanReferenceImage_ptr[voxel]*localMeanWarpedImage_ptr[voxel];
            localStdReferenceImage_ptr[voxel] = localMeanReferenceImage_ptr[voxel]*localMeanReferenceImage_ptr[voxel];
            localStdWarpedImage_ptr[voxel] = localMeanWarpedImage_ptr[voxel]*localMeanWarpedImage_ptr[voxel];
        }
    }

    reg_gaussianSmoothing<DTYPE>(localMeanReferenceImage, gaussianStandardDeviation, axis);
    reg_gaussianSmoothing<DTYPE>(localStdReferenceImage, gaussianStandardDeviation, axis);
    reg_gaussianSmoothing<DTYPE>(localMeanWarpedImage, gaussianStandardDeviation, axis);
    reg_gaussianSmoothing<DTYPE>(localStdWarpedImage, gaussianStandardDeviation, axis);
    reg_gaussianSmoothing<DTYPE>(localCorrelationImage, gaussianStandardDeviation, axis);

    int n=0;
    // We then iterate over every voxel to compute the LNCC
    double lncc_value = 0;
    double referenceMeanValue, warpedMeanValue, referenceVarValue, warpedVarValue, correlationValue;
    for(size_t voxel=0; voxel<referenceImage->nvox; ++voxel){
        // Check if the current voxel belongs to the mask
        if(mask[voxel]>-1){

            referenceMeanValue = localMeanReferenceImage_ptr[voxel];
            warpedMeanValue = localMeanWarpedImage_ptr[voxel];
            referenceVarValue = localStdReferenceImage_ptr[voxel];
            warpedVarValue = localStdWarpedImage_ptr[voxel];
            correlationValue = localCorrelationImage_ptr[voxel];
            if(referenceMeanValue==referenceMeanValue &&
               warpedMeanValue==warpedMeanValue &&
               referenceVarValue==referenceVarValue &&
               warpedVarValue==warpedVarValue &&
               correlationValue==correlationValue){

                referenceVarValue -= referenceMeanValue*referenceMeanValue;
                warpedVarValue -= warpedMeanValue*warpedMeanValue;
                // Sanity check
                if (referenceVarValue < 1e-6) referenceVarValue = 0.;
                if (warpedVarValue < 1e-6) warpedVarValue = 0.;

                referenceVarValue=sqrt(referenceVarValue);
                warpedVarValue=sqrt(warpedVarValue);
                correlationValue -= referenceMeanValue * warpedMeanValue;

                if (referenceVarValue !=0 && warpedVarValue !=0){
                    lncc_value += fabs(correlationValue /
                                       (referenceVarValue*warpedVarValue));
                    n++;

                    // to be removed
                    localCorrelationImage_ptr[voxel]=fabs(correlationValue /
                        (referenceVarValue*warpedVarValue));
                }
                // to be removed
                else localCorrelationImage_ptr[voxel]=0;
            }
        }
    }

    nifti_image_free(localMeanReferenceImage);
    nifti_image_free(localCorrelationImage);
    nifti_image_free(localMeanWarpedImage);
    nifti_image_free(localStdReferenceImage);
    nifti_image_free(localStdWarpedImage);

    return lncc_value/(double)n;
}

/* *************************************************************** */

double reg_getLNCC(nifti_image *referenceImage,
                   nifti_image *warpedImage,
                   float gaussianStandardDeviation,
                   int *mask
                   )
{
    // Check that all input images are of the same type
    if(referenceImage->datatype != warpedImage->datatype){
        fprintf(stderr,"[NiftyReg ERROR] reg_getLNCC\n");
        fprintf(stderr,"[NiftyReg ERROR] Input images are expected to have the same type\n");
        reg_exit(1);
    }

    // Check that both input images have the same size
    for(int i=0;i<5;++i){
        if(referenceImage->dim[i] != warpedImage->dim[i]){
            fprintf(stderr,"[NiftyReg ERROR] reg_getLNCC\n");
            fprintf(stderr,"[NiftyReg ERROR] Input images are expected to have the same dimension");
            reg_exit(1);
        }
    }

    switch ( referenceImage->datatype ){
    case NIFTI_TYPE_FLOAT32:
        return reg_getLNCC1<float>(referenceImage,warpedImage,gaussianStandardDeviation,mask);
        break;
    case NIFTI_TYPE_FLOAT64:
        return reg_getLNCC1<double>(referenceImage,warpedImage,gaussianStandardDeviation,mask);
        break;
    default:
        fprintf(stderr,"[NiftyReg ERROR] warped pixel type unsupported in the LNCC computation function.\n");
        reg_exit(1);
    }
    return 0.0;
}

/* *************************************************************** */

template <class DTYPE>
void reg_getVoxelBasedLNCCGradient1(nifti_image *referenceImage,
                                    nifti_image *warpedImage,
                                    nifti_image *warpedImageGradient,
                                    nifti_image *lnccGradientImage,
                                    float gaussianStandardDeviation,
                                    int *mask
                                    )
{
    DTYPE *referenceImage_ptr = static_cast<DTYPE *>(referenceImage->data);
    DTYPE *warpedImage_ptr = static_cast<DTYPE *>(warpedImage->data);

    nifti_image *localMeanReferenceImage = nifti_copy_nim_info(referenceImage);
    localMeanReferenceImage->data = (void *) malloc(referenceImage->nvox*sizeof(DTYPE));
    DTYPE *localMeanReferenceImage_ptr = static_cast<DTYPE *>(localMeanReferenceImage->data);
    memcpy(localMeanReferenceImage_ptr, referenceImage->data, referenceImage->nvox*sizeof(DTYPE));

    nifti_image *localCorrelationImage = nifti_copy_nim_info(warpedImage);
    localCorrelationImage->data = (void *) malloc(warpedImage->nvox*sizeof(DTYPE));
    DTYPE *localCorrelationImage_ptr = static_cast<DTYPE *>(localCorrelationImage->data);
    memcpy(localCorrelationImage_ptr, warpedImage->data, warpedImage->nvox*sizeof(DTYPE));

    nifti_image *localMeanWarpedImage = nifti_copy_nim_info(warpedImage);
    localMeanWarpedImage->data = (void *) malloc(warpedImage->nvox*sizeof(DTYPE));
    DTYPE *localMeanWarpedImage_ptr = static_cast<DTYPE *>(localMeanWarpedImage->data);
    memcpy(localMeanWarpedImage_ptr, warpedImage->data, warpedImage->nvox*sizeof(DTYPE));

    nifti_image *localStdReferenceImage = nifti_copy_nim_info(warpedImage);
    localStdReferenceImage->data = (void *) malloc(warpedImage->nvox*sizeof(DTYPE));
    DTYPE *localStdReferenceImage_ptr = static_cast<DTYPE *>(localStdReferenceImage->data);
    memcpy(localStdReferenceImage_ptr, warpedImage->data, warpedImage->nvox*sizeof(DTYPE));

    nifti_image *localStdWarpedImage = nifti_copy_nim_info(warpedImage);
    localStdWarpedImage->data = (void *) malloc(referenceImage->nvox*sizeof(DTYPE));
    DTYPE *localStdWarpedImage_ptr = static_cast<DTYPE *>(localStdWarpedImage->data);
    memcpy(localStdWarpedImage_ptr, warpedImage->data, referenceImage->nvox*sizeof(DTYPE));

    for(size_t voxel=0; voxel<referenceImage->nvox; ++voxel){
        if(localMeanReferenceImage_ptr[voxel]!=localMeanReferenceImage_ptr[voxel])
            localMeanWarpedImage_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
        if(localMeanWarpedImage_ptr[voxel]!=localMeanWarpedImage_ptr[voxel])
            localMeanReferenceImage_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
    }

    bool axis[8];
    axis[1]=true;
    axis[2]=true;
    if(referenceImage->nz>1)
        axis[3]=true;

    for(size_t voxel=0; voxel<referenceImage->nvox; ++voxel){
        if(mask[voxel]>-1 && localMeanReferenceImage_ptr[voxel]==localMeanReferenceImage_ptr[voxel]){
            localCorrelationImage_ptr[voxel] = localMeanReferenceImage_ptr[voxel]*localMeanWarpedImage_ptr[voxel];
//            localCorrelationImage_ptr[voxel] = localMeanWarpedImage_ptr[voxel]*localMeanWarpedImage_ptr[voxel];
            localStdReferenceImage_ptr[voxel] = localMeanReferenceImage_ptr[voxel]*localMeanReferenceImage_ptr[voxel];
            localStdWarpedImage_ptr[voxel] = localMeanWarpedImage_ptr[voxel]*localMeanWarpedImage_ptr[voxel];
        }     
    }

    reg_gaussianSmoothing<DTYPE>(localMeanReferenceImage, gaussianStandardDeviation, axis);
    reg_gaussianSmoothing<DTYPE>(localMeanWarpedImage, gaussianStandardDeviation, axis);
    reg_gaussianSmoothing<DTYPE>(localStdReferenceImage, gaussianStandardDeviation, axis);
    reg_gaussianSmoothing<DTYPE>(localStdWarpedImage, gaussianStandardDeviation, axis);
    reg_gaussianSmoothing<DTYPE>(localCorrelationImage, gaussianStandardDeviation, axis);

    double referenceMeanValue, warpedMeanValue, referenceVarValue, warpedVarValue;
    for(size_t voxel=0; voxel<referenceImage->nvox; ++voxel){
        // Check if the current voxel belongs to the mask
        if(mask[voxel]>-1){
            referenceMeanValue = localMeanReferenceImage_ptr[voxel];
            warpedMeanValue = localMeanWarpedImage_ptr[voxel];
            referenceVarValue = localStdReferenceImage_ptr[voxel];
            warpedVarValue = localStdWarpedImage_ptr[voxel];
            if(referenceMeanValue==referenceMeanValue &&
               warpedMeanValue==warpedMeanValue &&
               referenceVarValue==referenceVarValue &&
               warpedVarValue==warpedVarValue){

                localStdReferenceImage_ptr[voxel] -= referenceMeanValue*referenceMeanValue;
                localStdWarpedImage_ptr[voxel] -= warpedMeanValue*warpedMeanValue;
                // Sanity check
                if (localStdReferenceImage_ptr[voxel] < 1e-6) localStdReferenceImage_ptr[voxel] = 0;
                if (localStdWarpedImage_ptr[voxel]< 1e-6) localStdWarpedImage_ptr[voxel] = 0;

                localStdReferenceImage_ptr[voxel]=sqrt(localStdReferenceImage_ptr[voxel]);
                localStdWarpedImage_ptr[voxel]=sqrt(localStdWarpedImage_ptr[voxel]);
                localCorrelationImage_ptr[voxel] -= referenceMeanValue * warpedMeanValue;

            }
            else{
                localStdReferenceImage_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
                localStdWarpedImage_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
                localCorrelationImage_ptr[voxel] = std::numeric_limits<DTYPE>::quiet_NaN();

            }
        }
    } 


    size_t voxelNumber = referenceImage->nx*referenceImage->ny*referenceImage->nz;

    nifti_image *img1 = nifti_copy_nim_info(warpedImage);
    img1->data=(void *)malloc(img1->nvox*img1->nbyper);
    DTYPE *img1_ptr = static_cast<DTYPE *>(img1->data);

    nifti_image *img2 = nifti_copy_nim_info(warpedImage);
    img2->data=(void *)malloc(img2->nvox*img2->nbyper);
    DTYPE *img2_ptr = static_cast<DTYPE *>(img2->data);

    nifti_image *img3 = nifti_copy_nim_info(warpedImage);
    img3->data=(void *)malloc(img3->nvox*img3->nbyper);
    DTYPE *img3_ptr = static_cast<DTYPE *>(img3->data);


    double ref_stdValue, war_stdValue;
    for(size_t voxel=0; voxel<referenceImage->nvox; ++voxel){
        // Check if the current voxel belongs to the mask
        ref_stdValue = localStdReferenceImage_ptr[voxel];
        war_stdValue = localStdWarpedImage_ptr[voxel];
        if(mask[voxel]>-1){
            img1_ptr[voxel] = 1.0/(ref_stdValue*war_stdValue);
            img2_ptr[voxel] = localCorrelationImage_ptr[voxel]/(ref_stdValue*war_stdValue*war_stdValue*war_stdValue);
            img3_ptr[voxel] = (localCorrelationImage_ptr[voxel]/(war_stdValue*war_stdValue)*localMeanWarpedImage_ptr[voxel] - localMeanReferenceImage_ptr[voxel])/(ref_stdValue*war_stdValue);
            if(img1_ptr[voxel]!=img1_ptr[voxel] || isinf(img1_ptr[voxel])!=0 ||
               img2_ptr[voxel]!=img2_ptr[voxel] || isinf(img2_ptr[voxel])!=0 ||
               img3_ptr[voxel]!=img3_ptr[voxel] || isinf(img3_ptr[voxel])!=0)
                img1_ptr[voxel]=img2_ptr[voxel]=img3_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
        }
        else{
             img1_ptr[voxel]=img2_ptr[voxel]=img3_ptr[voxel]=0;
        }
    }

    reg_gaussianSmoothing<DTYPE>(img1, gaussianStandardDeviation, axis);
    reg_gaussianSmoothing<DTYPE>(img2, gaussianStandardDeviation, axis);
    reg_gaussianSmoothing<DTYPE>(img3, gaussianStandardDeviation, axis);

    // Create pointers to the voxel based gradient image - warpeds
    DTYPE *lnccGradPtrX=static_cast<DTYPE *>(lnccGradientImage->data);
    DTYPE *lnccGradPtrY = &lnccGradPtrX[voxelNumber];
    DTYPE *lnccGradPtrZ = NULL;
    // Create the z-axis pointers if the images are volume
    if(referenceImage->nz>1)
        lnccGradPtrZ = &lnccGradPtrY[voxelNumber];

    // Set all the gradient values to zero
    for(size_t voxel=0;voxel<lnccGradientImage->nvox;++voxel)
        lnccGradPtrX[voxel]=0;

    // Create some pointers to the spatial gradient of the warped volume
    DTYPE *spatialGradPtrX=static_cast<DTYPE *>(warpedImageGradient->data);
    DTYPE *spatialGradPtrY=&spatialGradPtrX[voxelNumber];
    DTYPE *spatialGradPtrZ=NULL;
    if(referenceImage->nz>1)
        spatialGradPtrZ=&spatialGradPtrY[voxelNumber];

    DTYPE gradX, gradY, gradZ;
    double common,refImageValue,warImageValue,img1Value,img2Value,img3Value;
    for(size_t voxel=0; voxel<voxelNumber; ++voxel){
        if(mask[voxel]>-1){
            refImageValue=referenceImage_ptr[voxel];
            warImageValue=warpedImage_ptr[voxel];
            img1Value=img1_ptr[voxel];
            img2Value=img2_ptr[voxel];
            img3Value=img3_ptr[voxel];
            if(refImageValue==refImageValue &&
               warImageValue==warImageValue && img1Value==img1Value && img2Value==img2Value && img2Value==img2Value){
                common = refImageValue*img1Value
                        - warImageValue*img2Value
                        + img3Value;

                //img3_ptr[voxel]=common;

                gradX = (DTYPE)(common * spatialGradPtrX[voxel]);
                gradY = (DTYPE)(common * spatialGradPtrY[voxel]);
                if(referenceImage->nz>1)
                    gradZ = (DTYPE)(common * spatialGradPtrZ[voxel]);                 
                lnccGradPtrX[voxel] += -gradX;
                lnccGradPtrY[voxel] += -gradY;
                if(referenceImage->nz>1)
                    lnccGradPtrZ[voxel] += -gradZ;

            }
        }
    }

    nifti_image_free(localMeanReferenceImage);
    nifti_image_free(localCorrelationImage);
    nifti_image_free(localMeanWarpedImage);
    nifti_image_free(localStdReferenceImage);
    nifti_image_free(localStdWarpedImage);

    nifti_image_free(img1);
    nifti_image_free(img2);
    nifti_image_free(img3);
}

/* *************************************************************** */
void reg_getVoxelBasedLNCCGradient(nifti_image *referenceImage,
                                   nifti_image *warpedImage,
                                   nifti_image *warpedImageGradient,
                                   nifti_image *lnccGradientImage,
                                   float gaussianStandardDeviation,
                                   int *mask
                                   )
{
    if(referenceImage->datatype != warpedImage->datatype ||
            warpedImageGradient->datatype != lnccGradientImage->datatype ||
            referenceImage->datatype != warpedImageGradient->datatype){
        fprintf(stderr,"[NiftyReg ERROR] reg_getVoxelBasedLNCCGradient\n");
        fprintf(stderr,"[NiftyReg ERROR] Input images are expected to have the same type\n");
        reg_exit(1);
    }
    
    switch ( referenceImage->datatype ){
    case NIFTI_TYPE_FLOAT32:
        reg_getVoxelBasedLNCCGradient1<float>
                (referenceImage, warpedImage,warpedImageGradient,lnccGradientImage, gaussianStandardDeviation, mask);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_getVoxelBasedLNCCGradient1<double>
                (referenceImage, warpedImage,warpedImageGradient,lnccGradientImage, gaussianStandardDeviation, mask);
        break;
    default:
        fprintf(stderr,"[NiftyReg ERROR] reference pixel type unsupported in the LNCC gradient computation function.\n");
        reg_exit(1);
    }
}
/* *************************************************************** */
template <class DTYPE>
void reg_getLocalStd1(nifti_image *image,
                      nifti_image *localMeanImage,
                      nifti_image *localStdImage,
                      float gaussianStandardDeviation,
                      int *mask
                      )
{
    DTYPE *image_Ptr=static_cast<DTYPE *>(image->data);
    DTYPE *localMeanImage_Ptr=static_cast<DTYPE *>(localMeanImage->data);
    DTYPE *localStdImage_Ptr=static_cast<DTYPE *>(localStdImage->data);

    for(size_t voxel=0; voxel<image->nvox; ++voxel){
        if(mask[voxel]>-1 && image_Ptr[voxel]==image_Ptr[voxel] ){
            localStdImage_Ptr[voxel] = image_Ptr[voxel]*image_Ptr[voxel];
        }
    }

    bool axis[8];
    axis[1]=true;
    axis[2]=true;
    if(image->nz>1)
        axis[3]=true;
    reg_gaussianSmoothing<DTYPE>(localStdImage, gaussianStandardDeviation, axis);

    for(size_t voxel=0; voxel<image->nvox; ++voxel){
        if(mask[voxel]>-1 && localMeanImage_Ptr[voxel]==localMeanImage_Ptr[voxel] && localStdImage_Ptr[voxel]==localStdImage_Ptr[voxel] && image_Ptr[voxel]==image_Ptr[voxel]){
            localStdImage_Ptr[voxel] -= localMeanImage_Ptr[voxel]*localMeanImage_Ptr[voxel];
            if (localStdImage_Ptr[voxel] < 0) localStdImage_Ptr[voxel] = 0;
            localStdImage_Ptr[voxel]=sqrt(localStdImage_Ptr[voxel]);
        }

    }
}

/* *************************************************************** */
void reg_getLocalStd(nifti_image *image,
                     nifti_image *localMeanImage,
                     nifti_image *localStdImage,
                     float gaussianStandardDeviation,
                     int *mask
                     )
{
    for(int i=0;i<5;++i){
        if(image->dim[i] != localStdImage->dim[i] || image->dim[i] != localMeanImage->dim[i]){
            fprintf(stderr,"[NiftyReg ERROR] reg_getLocalStd\n");
            fprintf(stderr,"[NiftyReg ERROR] Input images are expected to have the same dimension");
            reg_exit(1);
        }
    }

    switch ( image->datatype ){
    case NIFTI_TYPE_FLOAT32:
        reg_getLocalStd1<float>
                (image, localMeanImage, localStdImage, gaussianStandardDeviation, mask);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_getLocalStd1<double>
                (image, localMeanImage, localStdImage, gaussianStandardDeviation, mask);
        break;
    default:
        fprintf(stderr,"[NiftyReg ERROR] reference pixel type unsupported in the local standard deviation computation function.\n");
        reg_exit(1);
    }
}

/* *************************************************************** */
template <class DTYPE>
void reg_getLocalMean1(nifti_image *image,
                       nifti_image *localMeanImage,
                       float gaussianStandardDeviation
                       )
{
    bool axis[8];
    axis[1]=true;
    axis[2]=true;
    if(image->nz>1)
        axis[3]=true;

    memcpy(localMeanImage->data, image->data, localMeanImage->nvox*localMeanImage->nbyper);
    reg_gaussianSmoothing<DTYPE>(localMeanImage, gaussianStandardDeviation, axis);

}

/* *************************************************************** */
void reg_getLocalMean(nifti_image *image,
                      nifti_image *localMeanImage,
                      float gaussianStandardDeviation
                      )
{ 
    for(int i=0;i<5;++i){
        if(image->dim[i] != localMeanImage->dim[i]){
            fprintf(stderr,"[NiftyReg ERROR] reg_getLocalMean\n");
            fprintf(stderr,"[NiftyReg ERROR] Input images are expected to have the same dimension");
            reg_exit(1);
        }
    }

    switch ( image->datatype ){
    case NIFTI_TYPE_FLOAT32:
        reg_getLocalMean1<float>
                (image, localMeanImage, gaussianStandardDeviation);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_getLocalMean1<double>
                (image, localMeanImage, gaussianStandardDeviation);
        break;
    default:
        fprintf(stderr,"[NiftyReg ERROR] reference pixel type unsupported in the local mean computation function.\n");
        reg_exit(1);
    }
}

/* *************************************************************** */
template <class DTYPE>
void reg_getLocalCorrelation1(nifti_image *referenceImage,
                              nifti_image *warpedImage,
                              nifti_image *localMeanReferenceImage,
                              nifti_image *localMeanWarpedImage,
                              nifti_image *localCorrelationImage,
                              float gaussianStandardDeviation,
                              int *mask
                              )
{
    DTYPE *refPtr=static_cast<DTYPE *>(referenceImage->data);
    DTYPE *warPtr=static_cast<DTYPE *>(warpedImage->data);
    DTYPE *reflocalMeanPtr=static_cast<DTYPE *>(localMeanReferenceImage->data);
    DTYPE *warlocalMeanPtr=static_cast<DTYPE *>(localMeanWarpedImage->data);
    DTYPE *localCorrelationPtr=static_cast<DTYPE *>(localCorrelationImage->data);

    for(size_t voxel=0; voxel<referenceImage->nvox; ++voxel){
        if(mask[voxel]>-1){
            localCorrelationPtr[voxel] = refPtr[voxel]*warPtr[voxel];
        }
    }

    bool axis[8];
    axis[1]=true;
    axis[2]=true;
    if(referenceImage->nz>1)
        axis[3]=true;
    reg_gaussianSmoothing<DTYPE>(localCorrelationImage, gaussianStandardDeviation, axis);

    for(size_t voxel=0; voxel<referenceImage->nvox; ++voxel){
        if(mask[voxel]>-1){
            localCorrelationPtr[voxel] -= reflocalMeanPtr[voxel]*warlocalMeanPtr[voxel];
        }
    }
}

/* *************************************************************** */
void reg_getLocalCorrelation(nifti_image *referenceImage,
                             nifti_image *warpedImage,
                             nifti_image *localMeanReferenceImage,
                             nifti_image *localMeanWarpedImage,
                             nifti_image *localCorrelationImage,
                             float gaussianStandardDeviation,
                             int *mask
                             )
{
    for(int i=0;i<5;++i){
        if(referenceImage->dim[i] != warpedImage->dim[i]){
            fprintf(stderr,"[NiftyReg ERROR] reg_getLocalCorrelation\n");
            fprintf(stderr,"[NiftyReg ERROR] Input images are expected to have the same dimension");
            reg_exit(1);
        }
    }

    switch ( referenceImage->datatype ){
    case NIFTI_TYPE_FLOAT32:
        reg_getLocalCorrelation1<float>
                (referenceImage, warpedImage, localMeanReferenceImage, localMeanWarpedImage, localCorrelationImage, gaussianStandardDeviation, mask);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_getLocalCorrelation1<double>
                (referenceImage, warpedImage, localMeanReferenceImage, localMeanWarpedImage, localCorrelationImage, gaussianStandardDeviation, mask);
        break;
    default:
        fprintf(stderr,"[NiftyReg ERROR] reference pixel type unsupported in the local correlation computation function.\n");
        reg_exit(1);
    }
}

/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_meanFilter1(nifti_image *image, int radius, int *mask){

    DTYPE *imagePtr = static_cast<DTYPE *>(image->data);
    int voxelNumber = image->nx*image->ny*image->nz;

    int block_size=radius*2+1;
    
    DTYPE value;
    int current_index;

    DTYPE *warpedValue=(DTYPE *)malloc(voxelNumber * sizeof(DTYPE));

    //Smoothing along the X axis
    int x,y,z,index;
    for(z=0; z<image->nz; z++){
        for(y=0; y<image->ny; y++){
            index=z*image->nx*image->ny+y*image->nx;
            value=0;
            for (int i=0; i<=radius;i++){
                if(imagePtr[index+i]==imagePtr[index+i])
                    value += imagePtr[index+i];
            }
            value/=(block_size);
            warpedValue[index]=value;
            for(x=1; x<image->nx; x++){
                current_index=index+x;
                if (mask[current_index]>-1){
                    if (imagePtr[current_index]==imagePtr[current_index]){
                        if ((x-radius-1)>-1){
                            if(imagePtr[current_index-radius-1]==imagePtr[current_index-radius-1])
                                value-=imagePtr[current_index-radius-1]/block_size;
                        }
                        if ((x+radius)<image->nx){
                            if(imagePtr[current_index+radius]==imagePtr[current_index+radius])
                                value+=imagePtr[current_index+radius]/block_size;
                        }
                        warpedValue[current_index]=value;
                    }
                    else{
                        if ((x-radius-1)>-1){
                            if(imagePtr[current_index-radius-1]==imagePtr[current_index-radius-1])
                                value-=imagePtr[current_index-radius-1]/block_size;
                        }
                        if ((x+radius)<image->nx){
                            if(imagePtr[current_index+radius]==imagePtr[current_index+radius])
                                value+=imagePtr[current_index+radius]/block_size;
                        }
                    }
                    warpedValue[current_index]=imagePtr[current_index];

                }
            }
        }
    }
    
    for(int i=0; i<voxelNumber; i++)
        imagePtr[i]=(DTYPE)warpedValue[i];

    //Smoothing along the Y axis
    for(z=0; z<image->nz; z++){
        for(x=0; x<image->nx; x++){
            index=z*image->nx*image->ny+x;
            value=0;
            for (int i=0; i<=radius;i++){
                if(imagePtr[index+i*image->nx]==imagePtr[index+i*image->nx])
                    value += imagePtr[index+i*image->nx];
            }
            value/=block_size;
            warpedValue[index]=value;
            for(y=1; y<image->ny; y++){
                current_index=index+y*image->nx;
                if (mask[current_index]>-1){
                    if ((y-(radius+1))>-1){
                        if(imagePtr[current_index-(radius+1)*image->nx]==imagePtr[current_index-(radius+1)*image->nx])  value-= imagePtr[current_index-(radius+1)*image->nx]/block_size;
                    }
                    if ((y+radius)<image->ny){
                        if(imagePtr[current_index+radius*image->nx]==imagePtr[current_index+radius*image->nx]) value+= imagePtr[current_index+radius*image->nx]/block_size;
                    }
                    if ( imagePtr[current_index]== imagePtr[current_index]){
                        warpedValue[current_index]=value;
                    }
                }
            }
        }
    }

    for(int i=0; i<voxelNumber; i++)
        imagePtr[i]=(DTYPE)warpedValue[i];

    //Smoothing along the Z axis
    if(image->nz>1){
        
        for(y=0; y<image->ny; y++){
            for(x=0; x<image->nx; x++){
                index=y*image->nx+x;
                value=0;
                for (int i=0; i<=radius;i++){
                    if(imagePtr[index+i*image->nx*image->ny]==imagePtr[index+i*image->nx*image->ny])
                        value += imagePtr[index+i*image->nx*image->ny];
                }
                value/=block_size;
                warpedValue[index]=value;
                for(z=1; z<image->nz; z++){
                    current_index=index+z*image->nx*image->ny;
                    if (mask[current_index]>-1){
                        if ((z-(radius+1))>-1) {
                            if(imagePtr[current_index-(radius+1)*image->nx*image->ny]==imagePtr[current_index-(radius+1)*image->nx*image->ny]) value-=imagePtr[current_index-(radius+1)*image->nx*image->ny]/block_size;
                        }
                        if ((z+radius)<image->nz) {
                            if( imagePtr[current_index+radius*image->nx*image->ny]==imagePtr[current_index+radius*image->nx*image->ny]) value+=imagePtr[current_index+radius*image->nx*image->ny]/block_size;
                        }
                        if (imagePtr[current_index]==imagePtr[current_index]){
                            warpedValue[current_index]=value;
                        }
                    }
                }
            }
        }
    }

    for(int i=0; i<voxelNumber; i++)
        imagePtr[i]=(DTYPE)warpedValue[i];
    free(warpedValue);
}
/* *************************************************************** */
void reg_meanFilter(nifti_image *image, int radius, int *mask)
{   
    switch ( image->datatype ){
    case NIFTI_TYPE_FLOAT32:
        reg_meanFilter1<float>(image, radius, mask);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_meanFilter1<double>(image, radius, mask);
        break;
    default:
        fprintf(stderr,"[NiftyReg ERROR] reference pixel type unsupported in the mean filter computation function.\n");
        reg_exit(1);
    }
}
/* *************************************************************** */
/* *************************************************************** */
#endif

