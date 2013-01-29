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
    reg_gaussianSmoothing<DTYPE>(localMeanWarpedImage, gaussianStandardDeviation, axis);
    reg_gaussianSmoothing<DTYPE>(localStdReferenceImage, gaussianStandardDeviation, axis);
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
        exit(1);
    }

    // Check that both input images have the same size
    for(int i=0;i<5;++i){
        if(referenceImage->dim[i] != warpedImage->dim[i]){
            fprintf(stderr,"[NiftyReg ERROR] reg_getLNCC\n");
            fprintf(stderr,"[NiftyReg ERROR] Input images are expected to have the same dimension");
            exit(1);
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
        exit(1);
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
//                localStdReferenceImage_ptr[voxel]=0;
//                localStdWarpedImage_ptr[voxel]=0;
//                localCorrelationImage_ptr[voxel] = 0;
                localStdReferenceImage_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
                localStdWarpedImage_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
                localCorrelationImage_ptr[voxel] = std::numeric_limits<DTYPE>::quiet_NaN();

            }
        }
    } 

//    nifti_set_filenames(localCorrelationImage, "test2.nii",0,0);
//    nifti_image_write(localCorrelationImage);

//    nifti_set_filenames(localStdWarpedImage, "test.nii",0,0);
//    nifti_image_write(localStdWarpedImage);

    size_t voxelNumber = referenceImage->nx*referenceImage->ny*referenceImage->nz;

    /*    DTYPE *refPtr=static_cast<DTYPE *>(referenceImage->data);
        DTYPE *warPtr=static_cast<DTYPE *>(warpedImage->data);

    DTYPE *referenceImage_ptr=static_cast<DTYPE *>(referenceImage->data);
        DTYPE *warpedImage_ptr=static_cast<DTYPE *>(warpedImage->data);
    DTYPE *localMeanReferenceImage_ptr = static_cast<DTYPE *>(localMeanReferenceImage->data);
    DTYPE *localStdReferenceImage_ptr = static_cast<DTYPE *>(localStdReferenceImage->data);
    DTYPE *localMeanWarpedImage_ptr = static_cast<DTYPE *>(localMeanWarpedImage->data);
    DTYPE *localStdWarpedImage_ptr = static_cast<DTYPE *>(localStdWarpedImage->data);
    DTYPE *localCorrelationImage_ptr = static_cast<DTYPE *>(localCorrelationImage->data);

    bool axis[8];
    axis[1]=true;
    axis[2]=true;
    if(referenceImage->nz>1)
    axis[3]=true;
*/
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
//            img3_ptr[voxel] = img2_ptr[voxel]*localMeanWarpedImage_ptr[voxel] - localMeanReferenceImage_ptr[voxel]/(ref_stdValue*war_stdValue);
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
        exit(1);
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
        exit(1);
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
            exit(1);
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
        exit(1);
    }
}

/* *************************************************************** */
template <class DTYPE>
void reg_getLocalMean1(nifti_image *image,
                       nifti_image *localMeanImage,
                       float gaussianStandardDeviation,
                       int *mask
                       )
{
    bool axis[8];
    axis[1]=true;
    axis[2]=true;
    if(image->nz>1)
        axis[3]=true;

    memcpy(localMeanImage->data, image->data, localMeanImage->nvox*localMeanImage->nbyper);
    reg_gaussianSmoothing<DTYPE>(localMeanImage, gaussianStandardDeviation, axis);
    // reg_meanFilter(localMeanImage, 5, mask);

}

/* *************************************************************** */
void reg_getLocalMean(nifti_image *image,
                      nifti_image *localMeanImage,
                      float gaussianStandardDeviation,
                      int *mask
                      )
{ 
    for(int i=0;i<5;++i){
        if(image->dim[i] != localMeanImage->dim[i]){
            fprintf(stderr,"[NiftyReg ERROR] reg_getLocalMean\n");
            fprintf(stderr,"[NiftyReg ERROR] Input images are expected to have the same dimension");
            exit(1);
        }
    }

    switch ( image->datatype ){
    case NIFTI_TYPE_FLOAT32:
        reg_getLocalMean1<float>
                (image, localMeanImage, gaussianStandardDeviation, mask);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_getLocalMean1<double>
                (image, localMeanImage, gaussianStandardDeviation, mask);
        break;
    default:
        fprintf(stderr,"[NiftyReg ERROR] reference pixel type unsupported in the local mean computation function.\n");
        exit(1);
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
            exit(1);
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
        exit(1);
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
        exit(1);
    }
}
#endif

/* *************************************************************** */
/* *************************************************************** */

template<class DTYPE>
double reg_getLNCC_wml1(nifti_image *referenceImage,
                        nifti_image *warpedImage,
                        nifti_image *localMeanreferenceImage, //
                        nifti_image *localStdReferenceImage, //
                        nifti_image *localMeanWarpedImage, //
                        nifti_image *localStdWarpedImage, //
                        nifti_image *localCorrelationImage, //
                        float gaussianStandardDeviation,
                        int *mask,
                        double alpha
                        )
{
    /*   double lncc = reg_getLNCC(referenceImage,warpedImage,localMeanreferenceImage,localStdReferenceImage,localMeanWarpedImage,localStdWarpedImage,localCorrelationImage,
                        gaussianStandardDeviation,mask);

    nifti_image *referenceImageGradientX = nifti_copy_nim_info(referenceImage);
    referenceImageGradientX->data=(void *)malloc(referenceImageGradientX->nvox*referenceImageGradientX->nbyper);
    memcpy(referenceImageGradientX->data, referenceImage->data, referenceImageGradientX->nvox*referenceImageGradientX->nbyper);
    DTYPE *referenceImageGradientX_ptr = static_cast<DTYPE *>(referenceImageGradientX->data);

    nifti_image *referenceImageGradientY = nifti_copy_nim_info(referenceImage);
    referenceImageGradientY->data=(void *)malloc(referenceImageGradientY->nvox*referenceImageGradientY->nbyper);
    memcpy(referenceImageGradientY->data, referenceImage->data, referenceImageGradientY->nvox*referenceImageGradientY->nbyper);
    DTYPE *referenceImageGradientY_ptr = static_cast<DTYPE *>(referenceImageGradientY->data);

    nifti_image *warpedImageGradientX = nifti_copy_nim_info(warpedImage);
    warpedImageGradientX->data=(void *)malloc(warpedImageGradientX->nvox*warpedImageGradientX->nbyper);
    memcpy(warpedImageGradientX->data, warpedImage->data, warpedImageGradientX->nvox*warpedImageGradientX->nbyper);
    DTYPE *warpedImageGradientX_ptr = static_cast<DTYPE *>(warpedImageGradientX->data);

    nifti_image *warpedImageGradientY = nifti_copy_nim_info(warpedImage);
    warpedImageGradientY->data=(void *)malloc(warpedImageGradientY->nvox*warpedImageGradientY->nbyper);
    memcpy(warpedImageGradientY->data, warpedImage->data, warpedImageGradientY->nvox*warpedImageGradientY->nbyper);
    DTYPE *warpedImageGradientY_ptr = static_cast<DTYPE *>(warpedImageGradientY->data);

    reg_sobelFilter(referenceImageGradientX,1);
    reg_sobelFilter(referenceImageGradientY,2);
    reg_sobelFilter(warpedImageGradientX,1);
    reg_sobelFilter(warpedImageGradientY,2);
    double g=0.001;
    for(size_t voxel=0; voxel<referenceImage->nvox; ++voxel){
        if (fabs(fabs(referenceImageGradientX_ptr[voxel])-fabs(warpedImageGradientX_ptr[voxel]))>g){
            referenceImageGradientX_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
            warpedImageGradientX_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
        }
        if (fabs(fabs(referenceImageGradientY_ptr[voxel])-fabs(warpedImageGradientY_ptr[voxel]))>g){
            referenceImageGradientY_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
            warpedImageGradientY_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
        }
    }

    double lnccX, lnccY;

    lnccX = reg_getLNCC(referenceImageGradientX,warpedImageGradientX,localMeanreferenceImage,localStdReferenceImage,localMeanWarpedImage,localStdWarpedImage,localCorrelationImage,
                        gaussianStandardDeviation,mask);
    lnccY = reg_getLNCC(referenceImageGradientY,warpedImageGradientY,localMeanreferenceImage,localStdReferenceImage,localMeanWarpedImage,localStdWarpedImage,localCorrelationImage,
                        gaussianStandardDeviation,mask);

    memcpy(referenceImageGradientX->data, referenceImage->data, referenceImageGradientX->nvox*referenceImageGradientX->nbyper);
    memcpy(referenceImageGradientY->data, referenceImage->data, referenceImageGradientY->nvox*referenceImageGradientY->nbyper);
    memcpy(warpedImageGradientX->data, warpedImage->data, warpedImageGradientX->nvox*warpedImageGradientX->nbyper);
    memcpy(warpedImageGradientY->data, warpedImage->data, warpedImageGradientY->nvox*warpedImageGradientY->nbyper);


    reg_sobelFilter(referenceImageGradientX,1);
    reg_sobelFilter(referenceImageGradientY,2);
    reg_sobelFilter(warpedImageGradientX,1);
    reg_sobelFilter(warpedImageGradientY,2);

    double lnccX2, lnccY2;

    lnccX2 = reg_getLNCC(referenceImageGradientX,warpedImageGradientX,localMeanreferenceImage,localStdReferenceImage,localMeanWarpedImage,localStdWarpedImage,localCorrelationImage,
                        gaussianStandardDeviation,mask);
    lnccY2 = reg_getLNCC(referenceImageGradientY,warpedImageGradientY,localMeanreferenceImage,localStdReferenceImage,localMeanWarpedImage,localStdWarpedImage,localCorrelationImage,
                        gaussianStandardDeviation,mask);

    nifti_image_free(referenceImageGradientX);
    nifti_image_free(referenceImageGradientY);
    nifti_image_free(warpedImageGradientX);
    nifti_image_free(warpedImageGradientY);

    return alpha*lncc + (1-alpha)*(lnccX+lnccY)/2;
*/
    ////////////
    // We create some pointer to read the image data
    DTYPE *referenceImage_ptr = static_cast<DTYPE *>(referenceImage->data);
    DTYPE *warpedImage_ptr = static_cast<DTYPE *>(warpedImage->data);

    memcpy(localCorrelationImage->data, warpedImage->data, localCorrelationImage->nvox*localCorrelationImage->nbyper);
    memcpy(localMeanreferenceImage->data, referenceImage->data, referenceImage->nvox*referenceImage->nbyper);
    memcpy(localMeanWarpedImage->data, warpedImage->data, localMeanWarpedImage->nvox*localMeanWarpedImage->nbyper);
    memcpy(localStdReferenceImage->data, warpedImage->data, localStdReferenceImage->nvox*localStdReferenceImage->nbyper);
    memcpy(localStdWarpedImage->data, warpedImage->data, localStdWarpedImage->nvox*localStdWarpedImage->nbyper);

    DTYPE *localMeanReferenceImage_ptr = static_cast<DTYPE *>(localMeanreferenceImage->data);
    DTYPE *localMeanWarpedImage_ptr = static_cast<DTYPE *>(localMeanWarpedImage->data);
    DTYPE *localStdReferenceImage_ptr = static_cast<DTYPE *>(localStdReferenceImage->data);
    DTYPE *localStdWarpedImage_ptr = static_cast<DTYPE *>(localStdWarpedImage->data);
    DTYPE *localCorrelationImage_ptr = static_cast<DTYPE *>(localCorrelationImage->data);

    double s=0;

    for(size_t voxel=0; voxel<referenceImage->nvox; ++voxel){
        if(warpedImage_ptr[voxel]!=warpedImage_ptr[voxel]) localMeanReferenceImage_ptr[voxel]=warpedImage_ptr[voxel];
        /*       if (((referenceImage_ptr[voxel]>14 && referenceImage_ptr[voxel]<15) || (warpedImage_ptr[voxel]>14 && warpedImage_ptr[voxel]<15))){
            localMeanReferenceImage_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
            localMeanWarpedImage_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
            localStdReferenceImage_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
            localStdWarpedImage_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
            localCorrelationImage_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
        }*/
    }

    bool axis[8];
    axis[1]=true;
    axis[2]=true;
    if(referenceImage->nz>1)
        axis[3]=true;

    reg_gaussianSmoothing<DTYPE>(localMeanreferenceImage, gaussianStandardDeviation, axis);
    reg_gaussianSmoothing<DTYPE>(localMeanWarpedImage, gaussianStandardDeviation, axis);

    for(size_t voxel=0; voxel<referenceImage->nvox; ++voxel){
        if(mask[voxel]>-1 && localMeanReferenceImage_ptr[voxel]==localMeanReferenceImage_ptr[voxel]){
            localCorrelationImage_ptr[voxel] = referenceImage_ptr[voxel]*warpedImage_ptr[voxel];
            localStdReferenceImage_ptr[voxel] = referenceImage_ptr[voxel]*referenceImage_ptr[voxel];
            localStdWarpedImage_ptr[voxel] = warpedImage_ptr[voxel]*warpedImage_ptr[voxel];
        }
    }
    reg_gaussianSmoothing<DTYPE>(localStdReferenceImage, gaussianStandardDeviation, axis);
    reg_gaussianSmoothing<DTYPE>(localStdWarpedImage, gaussianStandardDeviation, axis);
    reg_gaussianSmoothing<DTYPE>(localCorrelationImage, gaussianStandardDeviation, axis);

    int n=0;
    // We then iterate over every voxel to compute the LNCC
    double lncc_value_wml = 0;
    double referenceMeanValue, warpedMeanValue, referenceVarValue, warpedVarValue;
    for(size_t voxel=0; voxel<referenceImage->nvox; ++voxel){
        // Check if the current voxel belongs to the mask
        if(mask[voxel]>-1){

            referenceMeanValue = localMeanReferenceImage_ptr[voxel];
            warpedMeanValue = localMeanWarpedImage_ptr[voxel];
            referenceVarValue = localStdReferenceImage_ptr[voxel];
            warpedVarValue = localStdWarpedImage_ptr[voxel];
            if(referenceMeanValue==referenceMeanValue && warpedMeanValue==warpedMeanValue && referenceVarValue==referenceVarValue && warpedVarValue==warpedVarValue){

                localStdReferenceImage_ptr[voxel] -= localMeanReferenceImage_ptr[voxel]*localMeanReferenceImage_ptr[voxel];
                localStdWarpedImage_ptr[voxel] -= localMeanWarpedImage_ptr[voxel]*localMeanWarpedImage_ptr[voxel];
                // Sanity check
                if (localStdReferenceImage_ptr[voxel] < 0) localStdReferenceImage_ptr[voxel] = 0;
                if (localStdWarpedImage_ptr[voxel] < 0) localStdWarpedImage_ptr[voxel] = 0;

                localStdReferenceImage_ptr[voxel]=sqrt(localStdReferenceImage_ptr[voxel]);
                localStdWarpedImage_ptr[voxel]=sqrt(localStdWarpedImage_ptr[voxel]);
                localCorrelationImage_ptr[voxel] -= (localMeanReferenceImage_ptr[voxel] * localMeanWarpedImage_ptr[voxel]);

                if ((localStdReferenceImage_ptr[voxel]*localStdWarpedImage_ptr[voxel])  !=0 && localCorrelationImage_ptr[voxel]==localCorrelationImage_ptr[voxel]
                        && localStdReferenceImage_ptr[voxel]==localStdReferenceImage_ptr[voxel] && localStdWarpedImage_ptr[voxel]==localStdWarpedImage_ptr[voxel]){
                    n++;
                    lncc_value_wml += fabs(localCorrelationImage_ptr[voxel] /
                                           (localStdReferenceImage_ptr[voxel]*localStdWarpedImage_ptr[voxel]));

                    //  localCorrelationImage_ptr[voxel]=fabs(localCorrelationImage_ptr[voxel] /
                    //                                        (localStdReferenceImage_ptr[voxel]*localStdWarpedImage_ptr[voxel]));
                }
            }
        }
    }
    if (n!=0)
        lncc_value_wml/=(double)(n);

    return lncc_value_wml;

    //////

    /*
    memcpy(localCorrelationImage->data, warpedImage->data, localCorrelationImage->nvox*localCorrelationImage->nbyper);
    memcpy(localMeanreferenceImage->data, referenceImage->data, referenceImage->nvox*referenceImage->nbyper);
    memcpy(localMeanWarpedImage->data, warpedImage->data, localMeanWarpedImage->nvox*localMeanWarpedImage->nbyper);
    memcpy(localStdReferenceImage->data, warpedImage->data, localStdReferenceImage->nvox*localStdReferenceImage->nbyper);
    memcpy(localStdWarpedImage->data, warpedImage->data, localStdWarpedImage->nvox*localStdWarpedImage->nbyper);

    for(size_t voxel=0; voxel<referenceImage->nvox; ++voxel){
        if(warpedImage_ptr[voxel]!=warpedImage_ptr[voxel]) localMeanReferenceImage_ptr[voxel]=warpedImage_ptr[voxel];
    }


    reg_gaussianSmoothing<DTYPE>(localMeanreferenceImage, gaussianStandardDeviation, axis);
    reg_gaussianSmoothing<DTYPE>(localMeanWarpedImage, gaussianStandardDeviation, axis);

    for(size_t voxel=0; voxel<referenceImage->nvox; ++voxel){
        if(mask[voxel]>-1 && localMeanReferenceImage_ptr[voxel]==localMeanReferenceImage_ptr[voxel]){
        localCorrelationImage_ptr[voxel] = referenceImage_ptr[voxel]*warpedImage_ptr[voxel];
        localStdReferenceImage_ptr[voxel] = referenceImage_ptr[voxel]*referenceImage_ptr[voxel];
        localStdWarpedImage_ptr[voxel] = warpedImage_ptr[voxel]*warpedImage_ptr[voxel];
        }
    }

    reg_gaussianSmoothing<DTYPE>(localStdReferenceImage, gaussianStandardDeviation, axis);
    reg_gaussianSmoothing<DTYPE>(localStdWarpedImage, gaussianStandardDeviation, axis);
    reg_gaussianSmoothing<DTYPE>(localCorrelationImage, gaussianStandardDeviation, axis);

    n=0;
    // We then iterate over every voxel to compute the LNCC
    double lncc_value = 0;
   // double referenceMeanValue, warpedMeanValue, referenceVarValue, warpedVarValue;
    for(size_t voxel=0; voxel<referenceImage->nvox; ++voxel){
        // Check if the current voxel belongs to the mask
                if(mask[voxel]>-1){

            referenceMeanValue = localMeanReferenceImage_ptr[voxel];
            warpedMeanValue = localMeanWarpedImage_ptr[voxel];
            referenceVarValue = localStdReferenceImage_ptr[voxel];
            warpedVarValue = localStdWarpedImage_ptr[voxel];
            if(referenceMeanValue==referenceMeanValue && warpedMeanValue==warpedMeanValue && referenceVarValue==referenceVarValue && warpedVarValue==warpedVarValue){

                localStdReferenceImage_ptr[voxel] -= localMeanReferenceImage_ptr[voxel]*localMeanReferenceImage_ptr[voxel];
                localStdWarpedImage_ptr[voxel] -= localMeanWarpedImage_ptr[voxel]*localMeanWarpedImage_ptr[voxel];
                // Sanity check
                if (localStdReferenceImage_ptr[voxel] < 0) localStdReferenceImage_ptr[voxel] = 0;
                if (localStdWarpedImage_ptr[voxel] < 0) localStdWarpedImage_ptr[voxel] = 0;

                localStdReferenceImage_ptr[voxel]=sqrt(localStdReferenceImage_ptr[voxel]);
                localStdWarpedImage_ptr[voxel]=sqrt(localStdWarpedImage_ptr[voxel]);
                localCorrelationImage_ptr[voxel] -= (localMeanReferenceImage_ptr[voxel] * localMeanWarpedImage_ptr[voxel]);

            if ((localStdReferenceImage_ptr[voxel]*localStdWarpedImage_ptr[voxel])!=0 && localCorrelationImage_ptr[voxel]==localCorrelationImage_ptr[voxel]
                    && localStdReferenceImage_ptr[voxel]==localStdReferenceImage_ptr[voxel] && localStdWarpedImage_ptr[voxel]==localStdWarpedImage_ptr[voxel]){
                n++;
                lncc_value += fabs(localCorrelationImage_ptr[voxel] /
                (localStdReferenceImage_ptr[voxel]*localStdWarpedImage_ptr[voxel]));

                }
            }
        }
    }

if (n!=0)
    lncc_value/=(double)(n);
else
    lncc_value=0;

////////

return alpha*lncc_value+(1-alpha)*lncc_value_wml;
*/
}

/* *************************************************************** */

double reg_getLNCC_wml(nifti_image *referenceImage,
                       nifti_image *warpedImage,
                       nifti_image *localMeanreferenceImage,
                       nifti_image *localStdReferenceImage,
                       nifti_image *localMeanWarpedImage,
                       nifti_image *localStdWarpedImage,
                       nifti_image *localCorrelationImage,
                       float gaussianStandardDeviation,
                       int *mask,
                       double alpha
                       )
{
    // Check that all input images are of the same type
    if(referenceImage->datatype != warpedImage->datatype){
        fprintf(stderr,"[NiftyReg ERROR] reg_getLNCC\n");
        fprintf(stderr,"[NiftyReg ERROR] Input images are expected to have the same type\n");
        exit(1);
    }

    // Check that both input images have the same size
    for(int i=0;i<5;++i){
        if(referenceImage->dim[i] != warpedImage->dim[i]){
            fprintf(stderr,"[NiftyReg ERROR] reg_getLNCC\n");
            fprintf(stderr,"[NiftyReg ERROR] Input images are expected to have the same dimension");
            exit(1);
        }
    }

    switch ( referenceImage->datatype ){
    case NIFTI_TYPE_FLOAT32:
        return reg_getLNCC_wml1<float>(referenceImage,warpedImage,localMeanreferenceImage,localStdReferenceImage,localMeanWarpedImage,localStdWarpedImage,localCorrelationImage,gaussianStandardDeviation,mask,alpha);
        break;
    case NIFTI_TYPE_FLOAT64:
        return reg_getLNCC_wml1<double>(referenceImage,warpedImage,localMeanreferenceImage,localStdReferenceImage,localMeanWarpedImage,localStdWarpedImage,localCorrelationImage,gaussianStandardDeviation,mask,alpha);
        break;
    default:
        fprintf(stderr,"[NiftyReg ERROR] warped pixel type unsupported in the LNCC computation function.\n");
        exit(1);
    }
    return 0.0;
}

/* ******************************************************************** */

template <class DTYPE>
void reg_getVoxelBasedLNCCGradient_wml1(nifti_image *referenceImage,
                                        nifti_image *warpedImage,
                                        nifti_image *localMeanReferenceImage, //
                                        nifti_image *localStdReferenceImage, //
                                        nifti_image *localMeanWarpedImage, //
                                        nifti_image *localStdWarpedImage, //
                                        nifti_image *localCorrelationImage, //
                                        nifti_image *warpedImageGradient,
                                        nifti_image *lnccGradientImage,
                                        float gaussianStandardDeviation,
                                        int *mask,
                                        double alpha
                                        )
{
    /*  reg_getVoxelBasedLNCCGradient(referenceImage,warpedImage,localMeanReferenceImage,localStdReferenceImage,localMeanWarpedImage,
                               localStdWarpedImage,localCorrelationImage,warpedImageGradient,lnccGradientImage,
                                gaussianStandardDeviation,mask);
    DTYPE *lnccGradientImage_ptr = static_cast<DTYPE *>(lnccGradientImage->data);

    nifti_image *referenceImageGradientX = nifti_copy_nim_info(referenceImage);
    referenceImageGradientX->data=(void *)malloc(referenceImageGradientX->nvox*referenceImageGradientX->nbyper);
    memcpy(referenceImageGradientX->data, referenceImage->data, referenceImageGradientX->nvox*referenceImageGradientX->nbyper);
    DTYPE *referenceImageGradientX_ptr = static_cast<DTYPE *>(referenceImageGradientX->data);

    nifti_image *referenceImageGradientY = nifti_copy_nim_info(referenceImage);
    referenceImageGradientY->data=(void *)malloc(referenceImageGradientY->nvox*referenceImageGradientY->nbyper);
    memcpy(referenceImageGradientY->data, referenceImage->data, referenceImageGradientY->nvox*referenceImageGradientY->nbyper);
    DTYPE *referenceImageGradientY_ptr = static_cast<DTYPE *>(referenceImageGradientY->data);

    nifti_image *warpedImageGradientX = nifti_copy_nim_info(warpedImage);
    warpedImageGradientX->data=(void *)malloc(warpedImageGradientX->nvox*warpedImageGradientX->nbyper);
    memcpy(warpedImageGradientX->data, warpedImage->data, warpedImageGradientX->nvox*warpedImageGradientX->nbyper);
    DTYPE *warpedImageGradientX_ptr = static_cast<DTYPE *>(warpedImageGradientX->data);

    nifti_image *warpedImageGradientY = nifti_copy_nim_info(warpedImage);
    warpedImageGradientY->data=(void *)malloc(warpedImageGradientY->nvox*warpedImageGradientY->nbyper);
    memcpy(warpedImageGradientY->data, warpedImage->data, warpedImageGradientY->nvox*warpedImageGradientY->nbyper);
    DTYPE *warpedImageGradientY_ptr = static_cast<DTYPE *>(warpedImageGradientY->data);

    reg_sobelFilter(referenceImageGradientX,1);
    reg_sobelFilter(referenceImageGradientY,2);
    reg_sobelFilter(warpedImageGradientX,1);
    reg_sobelFilter(warpedImageGradientY,2);
    double g=0.001;
    for(size_t voxel=0; voxel<referenceImage->nvox; ++voxel){
        if (fabs(fabs(referenceImageGradientX_ptr[voxel])-fabs(warpedImageGradientX_ptr[voxel]))>g){
            referenceImageGradientX_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
            warpedImageGradientX_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
        }
        if (fabs(fabs(referenceImageGradientY_ptr[voxel])-fabs(warpedImageGradientY_ptr[voxel]))>g){
            referenceImageGradientY_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
            warpedImageGradientY_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
        }
    }

    nifti_image *lnccGradientImageX = nifti_copy_nim_info(lnccGradientImage);
    lnccGradientImageX->data=(void *)malloc(lnccGradientImage->nvox*lnccGradientImage->nbyper);

    nifti_image *lnccGradientImageY = nifti_copy_nim_info(lnccGradientImage);
    lnccGradientImageY->data=(void *)malloc(lnccGradientImage->nvox*lnccGradientImage->nbyper);

    reg_getVoxelBasedLNCCGradient(referenceImageGradientX,warpedImageGradientX,localMeanReferenceImage,localStdReferenceImage,localMeanWarpedImage,
                               localStdWarpedImage,localCorrelationImage,warpedImageGradient,lnccGradientImageX,
                                gaussianStandardDeviation,mask);
    reg_getVoxelBasedLNCCGradient(referenceImageGradientY,warpedImageGradientY,localMeanReferenceImage,localStdReferenceImage,localMeanWarpedImage,
                               localStdWarpedImage,localCorrelationImage,warpedImageGradient,lnccGradientImageY,
                                gaussianStandardDeviation,mask);
    DTYPE *lnccGradientImageX_ptr = static_cast<DTYPE *>(lnccGradientImageX->data);
    DTYPE *lnccGradientImageY_ptr = static_cast<DTYPE *>(lnccGradientImageY->data);

    for(size_t voxel=0; voxel<lnccGradientImage->nvox; ++voxel){
        lnccGradientImage_ptr[voxel]=lnccGradientImage_ptr[voxel]*alpha+(1-alpha)*(lnccGradientImageX_ptr[voxel]+lnccGradientImageY_ptr[voxel])/2;
    }

    nifti_image_free(referenceImageGradientX);
    nifti_image_free(referenceImageGradientY);
    nifti_image_free(warpedImageGradientX);
    nifti_image_free(warpedImageGradientY);
    nifti_image_free(lnccGradientImageX);
    nifti_image_free(lnccGradientImageY);

    return;
*/

    /////
    size_t voxelNumber = referenceImage->nx*referenceImage->ny*referenceImage->nz;

    memcpy(localCorrelationImage->data, warpedImage->data, localCorrelationImage->nvox*localCorrelationImage->nbyper);
    memcpy(localMeanReferenceImage->data, referenceImage->data, referenceImage->nvox*referenceImage->nbyper);
    memcpy(localMeanWarpedImage->data, warpedImage->data, localMeanWarpedImage->nvox*localMeanWarpedImage->nbyper);
    memcpy(localStdReferenceImage->data, warpedImage->data, localStdReferenceImage->nvox*localStdReferenceImage->nbyper);
    memcpy(localStdWarpedImage->data, warpedImage->data, localStdWarpedImage->nvox*localStdWarpedImage->nbyper);

    DTYPE *refPtr=static_cast<DTYPE *>(referenceImage->data);
    DTYPE *warPtr=static_cast<DTYPE *>(warpedImage->data);

    DTYPE *referenceImage_ptr=static_cast<DTYPE *>(referenceImage->data);
    DTYPE *warpedImage_ptr=static_cast<DTYPE *>(warpedImage->data);
    DTYPE *localMeanReferenceImage_ptr = static_cast<DTYPE *>(localMeanReferenceImage->data);
    DTYPE *localStdReferenceImage_ptr = static_cast<DTYPE *>(localStdReferenceImage->data);
    DTYPE *localMeanWarpedImage_ptr = static_cast<DTYPE *>(localMeanWarpedImage->data);
    DTYPE *localStdWarpedImage_ptr = static_cast<DTYPE *>(localStdWarpedImage->data);
    DTYPE *localCorrelationImage_ptr = static_cast<DTYPE *>(localCorrelationImage->data);
    double s=0;

    for(size_t voxel=0; voxel<referenceImage->nvox; ++voxel){
        if(warpedImage_ptr[voxel]!=warpedImage_ptr[voxel]) localMeanReferenceImage_ptr[voxel]=warpedImage_ptr[voxel];
        /*      if (((referenceImage_ptr[voxel]>14 && referenceImage_ptr[voxel]<15) || (warpedImage_ptr[voxel]>14 && warpedImage_ptr[voxel]<15))){
            localMeanReferenceImage_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
            localMeanWarpedImage_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
            localStdReferenceImage_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
            localStdWarpedImage_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
            localCorrelationImage_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
        }*/
    }

    bool axis[8];
    axis[1]=true;
    axis[2]=true;
    if(referenceImage->nz>1)
        axis[3]=true;

    reg_gaussianSmoothing<DTYPE>(localMeanReferenceImage, gaussianStandardDeviation, axis);
    reg_gaussianSmoothing<DTYPE>(localMeanWarpedImage, gaussianStandardDeviation, axis);

    for(size_t voxel=0; voxel<referenceImage->nvox; ++voxel){
        if(mask[voxel]>-1 && localMeanReferenceImage_ptr[voxel]==localMeanReferenceImage_ptr[voxel]){
            localCorrelationImage_ptr[voxel] = referenceImage_ptr[voxel]*warpedImage_ptr[voxel];
            localStdReferenceImage_ptr[voxel] = referenceImage_ptr[voxel]*referenceImage_ptr[voxel];
            localStdWarpedImage_ptr[voxel] = warpedImage_ptr[voxel]*warpedImage_ptr[voxel];
        }
    }

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
            if(referenceMeanValue==referenceMeanValue && warpedMeanValue==warpedMeanValue && referenceVarValue==referenceVarValue && warpedVarValue==warpedVarValue){

                localStdReferenceImage_ptr[voxel] -= localMeanReferenceImage_ptr[voxel]*localMeanReferenceImage_ptr[voxel];
                localStdWarpedImage_ptr[voxel] -= localMeanWarpedImage_ptr[voxel]*localMeanWarpedImage_ptr[voxel];
                // Sanity check
                if (localStdReferenceImage_ptr[voxel] < 0) localStdReferenceImage_ptr[voxel] = 0;
                if (localStdWarpedImage_ptr[voxel] < 0) localStdWarpedImage_ptr[voxel] = 0;

                localStdReferenceImage_ptr[voxel]=sqrt(localStdReferenceImage_ptr[voxel]);
                localStdWarpedImage_ptr[voxel]=sqrt(localStdWarpedImage_ptr[voxel]);
                localCorrelationImage_ptr[voxel] -= (localMeanReferenceImage_ptr[voxel] * localMeanWarpedImage_ptr[voxel]);
            }
        }
    }

    nifti_image *img1 = nifti_copy_nim_info(warpedImage);
    img1->data=(void *)malloc(img1->nvox*img1->nbyper);
    DTYPE *img1_ptr = static_cast<DTYPE *>(img1->data);

    nifti_image *img2 = nifti_copy_nim_info(warpedImage);
    img2->data=(void *)malloc(img2->nvox*img2->nbyper);
    DTYPE *img2_ptr = static_cast<DTYPE *>(img2->data);

    nifti_image *img3 = nifti_copy_nim_info(warpedImage);
    img3->data=(void *)malloc(img3->nvox*img3->nbyper);
    DTYPE *img3_ptr = static_cast<DTYPE *>(img3->data);

    for(size_t voxel=0;voxel<img1->nvox;++voxel){
        img1_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
        img2_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
        img3_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
    }

    double ref_stdValue, war_stdValue;
    for(size_t voxel=0; voxel<referenceImage->nvox; ++voxel){
        // Check if the current voxel belongs to the mask
        if(mask[voxel]>-1){
            ref_stdValue = localStdReferenceImage_ptr[voxel];
            war_stdValue = localStdWarpedImage_ptr[voxel];
            if (ref_stdValue != 0 && war_stdValue !=0){
                img1_ptr[voxel] = 1/(ref_stdValue*war_stdValue);
                img2_ptr[voxel] = localCorrelationImage_ptr[voxel]/(ref_stdValue*war_stdValue*war_stdValue*war_stdValue);
                img3_ptr[voxel] = img2_ptr[voxel]*localMeanWarpedImage_ptr[voxel] -localMeanReferenceImage_ptr[voxel]/(ref_stdValue*war_stdValue);
            }
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
        lnccGradPtrX[voxel]= 0;

    // Create some pointers to the spatial gradient of the warped volume
    DTYPE *spatialGradPtrX=static_cast<DTYPE *>(warpedImageGradient->data);
    DTYPE *spatialGradPtrY=&spatialGradPtrX[voxelNumber];
    DTYPE *spatialGradPtrZ=NULL;
    if(referenceImage->nz>1)
        spatialGradPtrZ=&spatialGradPtrY[voxelNumber];

    double common,refImageValue,warImageValue,img1Value,img2Value,img3Value;
    for(size_t voxel=0; voxel<voxelNumber; ++voxel){
        if(mask[voxel]>-1){
            refImageValue=referenceImage_ptr[voxel];
            warImageValue=warpedImage_ptr[voxel];
            img1Value=img1_ptr[voxel];
            img2Value=img2_ptr[voxel];
            img3Value=img3_ptr[voxel];
            DTYPE gradX=0, gradY=0, gradZ=0;
            if (refImageValue==refImageValue && warImageValue==warImageValue && img1Value==img1Value && img2Value==img2Value && img3Value==img3Value){
                common = refImageValue*img1Value
                        -warImageValue*img2Value
                        +img3Value;
                gradX = (DTYPE)(common * spatialGradPtrX[voxel]);
                if(gradX==gradX && isinf(gradX)==0){
                    lnccGradPtrX[voxel] += - alpha*gradX;
                }
                gradY = (DTYPE)(common * spatialGradPtrY[voxel]);
                if(gradY==gradY && isinf(gradY)==0){
                    lnccGradPtrY[voxel] += - alpha*gradY;
                }
                if(referenceImage->nz>1){
                    if(spatialGradPtrZ[voxel]==spatialGradPtrZ[voxel]){
                        gradZ = (DTYPE)(common * spatialGradPtrZ[voxel]);
                        lnccGradPtrZ[voxel] += -alpha*gradZ;
                    }
                }
            }
        }
    }

    //// with wml
    /*    memcpy(localCorrelationImage->data, warpedImage->data, localCorrelationImage->nvox*localCorrelationImage->nbyper);
    memcpy(localMeanReferenceImage->data, referenceImage->data, referenceImage->nvox*referenceImage->nbyper);
    memcpy(localMeanWarpedImage->data, warpedImage->data, localMeanWarpedImage->nvox*localMeanWarpedImage->nbyper);
    memcpy(localStdReferenceImage->data, warpedImage->data, localStdReferenceImage->nvox*localStdReferenceImage->nbyper);
    memcpy(localStdWarpedImage->data, warpedImage->data, localStdWarpedImage->nvox*localStdWarpedImage->nbyper);

    for(size_t voxel=0; voxel<referenceImage->nvox; ++voxel){
        if(warpedImage_ptr[voxel]!=warpedImage_ptr[voxel]) localMeanReferenceImage_ptr[voxel]=warpedImage_ptr[voxel];
        if (fabs(warpedImage_ptr[voxel]-referenceImage_ptr[voxel])>s && ((referenceImage_ptr[voxel]>14 && referenceImage_ptr[voxel]<15) || (warpedImage_ptr[voxel]>14 && warpedImage_ptr[voxel]<15))){
        //   if (fabs(warpedImage_ptr[voxel]-referenceImage_ptr[voxel])>s){
            localMeanReferenceImage_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
            localMeanWarpedImage_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
            localStdReferenceImage_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
            localStdWarpedImage_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
            localCorrelationImage_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
        }
    }

    reg_gaussianSmoothing<DTYPE>(localMeanReferenceImage, gaussianStandardDeviation, axis);
    reg_gaussianSmoothing<DTYPE>(localMeanWarpedImage, gaussianStandardDeviation, axis);

    for(size_t voxel=0; voxel<referenceImage->nvox; ++voxel){
        if(mask[voxel]>-1 && localMeanReferenceImage_ptr[voxel]==localMeanReferenceImage_ptr[voxel]){
        localCorrelationImage_ptr[voxel] = referenceImage_ptr[voxel]*warpedImage_ptr[voxel];
        localStdReferenceImage_ptr[voxel] = referenceImage_ptr[voxel]*referenceImage_ptr[voxel];
        localStdWarpedImage_ptr[voxel] = warpedImage_ptr[voxel]*warpedImage_ptr[voxel];
        }
    }

    reg_gaussianSmoothing<DTYPE>(localStdReferenceImage, gaussianStandardDeviation, axis);
    reg_gaussianSmoothing<DTYPE>(localStdWarpedImage, gaussianStandardDeviation, axis);
    reg_gaussianSmoothing<DTYPE>(localCorrelationImage, gaussianStandardDeviation, axis);

    for(size_t voxel=0; voxel<referenceImage->nvox; ++voxel){
        // Check if the current voxel belongs to the mask
                if(mask[voxel]>-1){
            referenceMeanValue = localMeanReferenceImage_ptr[voxel];
            warpedMeanValue = localMeanWarpedImage_ptr[voxel];
            referenceVarValue = localStdReferenceImage_ptr[voxel];
            warpedVarValue = localStdWarpedImage_ptr[voxel];
            if(referenceMeanValue==referenceMeanValue && warpedMeanValue==warpedMeanValue && referenceVarValue==referenceVarValue && warpedVarValue==warpedVarValue){

                localStdReferenceImage_ptr[voxel] -= localMeanReferenceImage_ptr[voxel]*localMeanReferenceImage_ptr[voxel];
                localStdWarpedImage_ptr[voxel] -= localMeanWarpedImage_ptr[voxel]*localMeanWarpedImage_ptr[voxel];
                // Sanity check
                if (localStdReferenceImage_ptr[voxel] < 0) localStdReferenceImage_ptr[voxel] = 0;
                if (localStdWarpedImage_ptr[voxel] < 0) localStdWarpedImage_ptr[voxel] = 0;

                localStdReferenceImage_ptr[voxel]=sqrt(localStdReferenceImage_ptr[voxel]);
                localStdWarpedImage_ptr[voxel]=sqrt(localStdWarpedImage_ptr[voxel]);
                localCorrelationImage_ptr[voxel] -= (localMeanReferenceImage_ptr[voxel] * localMeanWarpedImage_ptr[voxel]);
            }
        }
    }

    for(size_t voxel=0;voxel<img1->nvox;++voxel){
    img1_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
    img2_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
    img3_ptr[voxel]=std::numeric_limits<DTYPE>::quiet_NaN();
    }

    for(size_t voxel=0; voxel<referenceImage->nvox; ++voxel){
        // Check if the current voxel belongs to the mask
            if(mask[voxel]>-1){
            ref_stdValue = localStdReferenceImage_ptr[voxel];
            war_stdValue = localStdWarpedImage_ptr[voxel];
            if (ref_stdValue != 0 && war_stdValue !=0){
            img1_ptr[voxel] = 1/(ref_stdValue*war_stdValue);
            img2_ptr[voxel] = localCorrelationImage_ptr[voxel]/(ref_stdValue*war_stdValue*war_stdValue*war_stdValue);
            img3_ptr[voxel] = img2_ptr[voxel]*localMeanWarpedImage_ptr[voxel] -localMeanReferenceImage_ptr[voxel]/(ref_stdValue*war_stdValue);
            }
        }
    }

    reg_gaussianSmoothing<DTYPE>(img1, gaussianStandardDeviation, axis);
    reg_gaussianSmoothing<DTYPE>(img2, gaussianStandardDeviation, axis);
    reg_gaussianSmoothing<DTYPE>(img3, gaussianStandardDeviation, axis);

    for(size_t voxel=0; voxel<voxelNumber; ++voxel){
        if(mask[voxel]>-1){
        refImageValue=referenceImage_ptr[voxel];
        warImageValue=warpedImage_ptr[voxel];
        img1Value=img1_ptr[voxel];
        img2Value=img2_ptr[voxel];
        img3Value=img3_ptr[voxel];
        if (refImageValue==refImageValue && warImageValue==warImageValue && img1Value==img1Value && img2Value==img2Value && img3Value==img3Value){
        common = refImageValue*img1Value
            -warImageValue*img2Value
            +img3Value;
                gradX = (DTYPE)(common * spatialGradPtrX[voxel]);
                gradY = (DTYPE)(common * spatialGradPtrY[voxel]);
                if(referenceImage->nz>1)
                    gradZ = (DTYPE)(common * spatialGradPtrZ[voxel]);
                lnccGradPtrX[voxel] -= (1-alpha)*gradX;
                lnccGradPtrY[voxel] -= (1-alpha)*gradY;
                if(referenceImage->nz>1)
                        lnccGradPtrZ[voxel] -= (1-alpha)*gradZ;
        }
        }
    }
*/
    ////
    nifti_image_free(img1);
    nifti_image_free(img2);
    nifti_image_free(img3);
}

/* *************************************************************** */
void reg_getVoxelBasedLNCCGradient_wml(nifti_image *referenceImage,
                                       nifti_image *warpedImage,
                                       nifti_image *localMeanReferenceImage,
                                       nifti_image *localStdReferenceImage,
                                       nifti_image *localMeanWarpedImage,
                                       nifti_image *localStdWarpedImage,
                                       nifti_image *localCorrelationImage,
                                       nifti_image *warpedImageGradient,
                                       nifti_image *lnccGradientImage,
                                       float gaussianStandardDeviation,
                                       int *mask,
                                       double alpha
                                       )
{
    if(referenceImage->datatype != warpedImage->datatype ||
            warpedImageGradient->datatype != lnccGradientImage->datatype ||
            referenceImage->datatype != warpedImageGradient->datatype){
        fprintf(stderr,"[NiftyReg ERROR] reg_getVoxelBasedLNCCGradient\n");
        fprintf(stderr,"[NiftyReg ERROR] Input images are expected to have the same type\n");
        exit(1);
    }

    switch ( referenceImage->datatype ){
    case NIFTI_TYPE_FLOAT32:
        reg_getVoxelBasedLNCCGradient_wml1<float>
                (referenceImage, warpedImage,localMeanReferenceImage,localStdReferenceImage,localMeanWarpedImage,localStdWarpedImage, localCorrelationImage,warpedImageGradient,lnccGradientImage, gaussianStandardDeviation, mask, alpha);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_getVoxelBasedLNCCGradient_wml1<double>
                (referenceImage, warpedImage,localMeanReferenceImage,localStdReferenceImage,localMeanWarpedImage, localStdWarpedImage,localCorrelationImage,warpedImageGradient,lnccGradientImage, gaussianStandardDeviation, mask, alpha);
        break;
    default:
        fprintf(stderr,"[NiftyReg ERROR] reference pixel type unsupported in the LNCC gradient computation function.\n");
        exit(1);
    }
}

/* *************************************************************** */
template <class PrecisionTYPE, class ImageTYPE>
void reg_sobelFilter1(nifti_image *image, int axis){
    PrecisionTYPE *imagePtr = static_cast<ImageTYPE *>(image->data);

    int timePoint = image->nt;
    if(timePoint==0) timePoint=1;
    int field = image->nu;
    if(field==0) field=1;

    int voxelNumber = image->nx*image->ny*image->nz;

    int index, startingIndex, x, i, j, t, current, n, radius, increment;
    PrecisionTYPE value;
    PrecisionTYPE *kernel1 = new PrecisionTYPE[3];
    PrecisionTYPE *kernel2 = new PrecisionTYPE[3];
    // Loop over the dimension higher than 3
    for(t=0; t<timePoint*field; t++){
        ImageTYPE *timeImagePtr = &imagePtr[t * voxelNumber];
        PrecisionTYPE *warpedValue=(PrecisionTYPE *)malloc(voxelNumber * sizeof(PrecisionTYPE));
        // Loop over the 3 dimensions

        if (axis==1){

            kernel1[0]=1;
            kernel1[1]=0;
            kernel1[2]=-1;

            kernel2[0]=1;
            kernel2[1]=2;
            kernel2[2]=1;
        }
        if(axis==2){

            kernel1[0]=1;
            kernel1[1]=2;
            kernel1[2]=1;

            kernel2[0]=1;
            kernel2[1]=0;
            kernel2[2]=-1;
        }

        for(n=1;n<3;n++){
            if(n==1){
                // Define the variable to increment in the 1D array
                increment=1;
                switch(n){
                case 1: increment=1;break;
                case 2: increment=image->nx;break;
                case 3: increment=image->nx*image->ny;break;
                }
                // Loop over the different voxel

                for(index=0;index<voxelNumber;index+=image->dim[n]){
                    for(x=0; x<image->dim[n]; x++){
                        startingIndex=index+x;

                        current = startingIndex - increment*radius;
                        value=0;

                        // Check if the central voxel is a NaN
                        if(timeImagePtr[startingIndex]==timeImagePtr[startingIndex]){

                            for(j=-1; j<=1; j++){
                                if(-1<current && current<(int)voxelNumber){
                                    if(timeImagePtr[current]==timeImagePtr[current]){
                                        value += (PrecisionTYPE)(timeImagePtr[current]*kernel1[j+1]);

                                    }
                                }
                                current += increment;
                            }
                            warpedValue[startingIndex]=value;
                        }
                        else{
                            warpedValue[startingIndex]=timeImagePtr[startingIndex];
                        }
                    }
                }

                for(i=0; i<voxelNumber; i++)
                    timeImagePtr[i]=(ImageTYPE)warpedValue[i];
                delete[] kernel1;
            }
            if(n==2){
                // Define the variable to increment in the 1D array
                increment=1;
                switch(n){
                case 1: increment=1;break;
                case 2: increment=image->nx;break;
                case 3: increment=image->nx*image->ny;break;
                }
                // Loop over the different voxel

                for(index=0;index<voxelNumber;index+=image->dim[n]){
                    for(x=0; x<image->dim[n]; x++){
                        startingIndex=index+x;

                        current = startingIndex - increment*radius;
                        value=0;

                        // Check if the central voxel is a NaN
                        if(timeImagePtr[startingIndex]==timeImagePtr[startingIndex]){

                            for(j=-1; j<=1; j++){
                                if(-1<current && current<(int)voxelNumber){
                                    if(timeImagePtr[current]==timeImagePtr[current]){
                                        value += (PrecisionTYPE)(timeImagePtr[current]*kernel2[j+1]);

                                    }
                                }
                                current += increment;
                            }
                            warpedValue[startingIndex]=value ;
                        }
                        else{
                            warpedValue[startingIndex]=timeImagePtr[startingIndex];
                        }
                    }
                }

                for(i=0; i<voxelNumber; i++)
                    timeImagePtr[i]=(ImageTYPE)warpedValue[i];
                delete[] kernel2;
            }
        }
        free(warpedValue);
    }

}

/* *************************************************************** */
void reg_sobelFilter(nifti_image *image, int axis)
{
    switch ( image->datatype ){
    case NIFTI_TYPE_FLOAT32:
        reg_sobelFilter1<float,float>
                (image, axis);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_sobelFilter1<double,double>
                (image, axis);
        break;
    default:
        fprintf(stderr,"[NiftyReg ERROR] reference pixel type unsupported in the LNCC gradient computation function.\n");
        exit(1);
    }
}
