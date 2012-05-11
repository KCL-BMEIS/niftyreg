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

/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
double reg_getSSD1(nifti_image *targetImage,
                   nifti_image *resultImage,
                   nifti_image *jacobianDetImage,
                   int *mask
                   )
{
    DTYPE *targetPtr=static_cast<DTYPE *>(targetImage->data);
    DTYPE *resultPtr=static_cast<DTYPE *>(resultImage->data);
    DTYPE *jacDetPtr=NULL;
    if(jacobianDetImage!=NULL)
        jacDetPtr=static_cast<DTYPE *>(jacobianDetImage->data);

    int i;
    double SSD=0.0, n=0.0;
    double targetValue, resultValue, diff;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(targetImage, targetPtr, resultPtr, mask, \
    jacobianDetImage, jacDetPtr) \
    private(i, targetValue, resultValue, diff) \
    reduction(+:SSD) \
    reduction(+:n)
#endif
    for(i=0; i<(int)targetImage->nvox;i++){
        if(mask[i]>-1){
            targetValue = (double)targetPtr[i];
            resultValue = (double)resultPtr[i];
            if(targetValue==targetValue && resultValue==resultValue){
                diff = (targetValue-resultValue);
                if(jacobianDetImage!=NULL){
                    SSD += diff * diff * jacDetPtr[i];
                    n += jacDetPtr[i];
                }
                else{
                    SSD += diff * diff;
                    n += 1.0;
                }
            }
        }
    }

    return SSD/n;
}
/* *************************************************************** */
double reg_getSSD(nifti_image *targetImage,
                  nifti_image *resultImage,
                  nifti_image *jacobianDetImage,
                  int *mask
                  )
{
    if(jacobianDetImage!=NULL){
        if(targetImage->datatype != resultImage->datatype || targetImage->datatype != jacobianDetImage->datatype){
            fprintf(stderr,"[NiftyReg ERROR] reg_getSSD\n");
            fprintf(stderr,"[NiftyReg ERROR] Input images are expected to have the same type\n");
            exit(1);
        }
    }
    else{
        if(targetImage->datatype != resultImage->datatype){
            fprintf(stderr,"[NiftyReg ERROR] reg_getSSD\n");
            fprintf(stderr,"[NiftyReg ERROR] Input images are expected to have the same type\n");
            exit(1);
        }
    }

    switch ( targetImage->datatype ){
        case NIFTI_TYPE_FLOAT32:
            return reg_getSSD1<float>(targetImage,resultImage, jacobianDetImage, mask);
            break;
        case NIFTI_TYPE_FLOAT64:
            return reg_getSSD1<double>(targetImage,resultImage, jacobianDetImage, mask);
            break;
        default:
            fprintf(stderr,"[NiftyReg ERROR] Result pixel type unsupported in the SSD computation function.\n");
            exit(1);
	}
	return 0.0;
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_getVoxelBasedSSDGradient1(nifti_image *targetImage,
                                   nifti_image *resultImage,
                                   nifti_image *resultImageGradient,
                                   nifti_image *ssdGradientImage,
                                   nifti_image *jacobianDetImage,
                                   float maxSD,
                                   int *mask
                                   )
{
    DTYPE *targetPtr=static_cast<DTYPE *>(targetImage->data);
    DTYPE *resultPtr=static_cast<DTYPE *>(resultImage->data);

    DTYPE *spatialGradPtrX=static_cast<DTYPE *>(resultImageGradient->data);
    DTYPE *spatialGradPtrY = &spatialGradPtrX[resultImageGradient->nx*resultImageGradient->ny*resultImageGradient->nz];
    DTYPE *spatialGradPtrZ = NULL;
    if(targetImage->nz>1) spatialGradPtrZ = &spatialGradPtrY[resultImageGradient->nx*resultImageGradient->ny*resultImageGradient->nz];

    DTYPE *ssdGradPtrX=static_cast<DTYPE *>(ssdGradientImage->data);
    DTYPE *ssdGradPtrY = &ssdGradPtrX[ssdGradientImage->nx*ssdGradientImage->ny*ssdGradientImage->nz];
    DTYPE *ssdGradPtrZ = NULL;
    if(targetImage->nz>1) ssdGradPtrZ = &ssdGradPtrY[ssdGradientImage->nx*ssdGradientImage->ny*ssdGradientImage->nz];

    DTYPE *jacDetPtr=NULL;
    if(jacobianDetImage!=NULL)
        jacDetPtr=static_cast<DTYPE *>(jacobianDetImage->data);

    DTYPE gradX, gradY, gradZ;
    double targetValue, resultValue, common;
    int i;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(targetImage, targetPtr, resultPtr, maxSD, mask, jacDetPtr, jacobianDetImage, \
    spatialGradPtrX, spatialGradPtrY, spatialGradPtrZ, ssdGradPtrX, ssdGradPtrY, ssdGradPtrZ) \
    private(i, targetValue, resultValue, common, gradX, gradY, gradZ)
#endif
    for(i=0; i<(int)targetImage->nvox;i++){
        if(mask[i]>-1){
            targetValue = targetPtr[i];
            resultValue = resultPtr[i];
            gradX=0;
            gradY=0;
            gradZ=0;
            if(targetValue==targetValue && resultValue==resultValue){
                common = - 2.0 * (targetValue - resultValue);
                gradX = (DTYPE)(common * spatialGradPtrX[i]/maxSD);
                gradY = (DTYPE)(common * spatialGradPtrY[i]/maxSD);
                if(targetImage->nz>1) gradZ = (DTYPE)(common * spatialGradPtrZ[i]/maxSD);
            }
            if(jacobianDetImage!=NULL){
                ssdGradPtrX[i] = gradX * jacDetPtr[i];
                ssdGradPtrY[i] = gradY * jacDetPtr[i];
                if(targetImage->nz>1) ssdGradPtrZ[i] = gradZ * jacDetPtr[i];
            }
            else{
                ssdGradPtrX[i] = gradX;
                ssdGradPtrY[i] = gradY;
                if(targetImage->nz>1) ssdGradPtrZ[i] = gradZ;
            }
        }
    }
}
/* *************************************************************** */
void reg_getVoxelBasedSSDGradient(nifti_image *targetImage,
                                  nifti_image *resultImage,
                                  nifti_image *resultImageGradient,
                                  nifti_image *ssdGradientImage,
                                  nifti_image *jacobianDeterminantImage,
                                  float maxSD,
                                  int *mask
                                  )
{
    if(targetImage->datatype != resultImage->datatype ||
       resultImageGradient->datatype != ssdGradientImage->datatype ||
       targetImage->datatype != resultImageGradient->datatype){
        fprintf(stderr,"[NiftyReg ERROR] reg_getVoxelBasedSSDGradient\n");
        fprintf(stderr,"[NiftyReg ERROR] Input images are expected to have the same type\n");
        exit(1);
    }
    if(jacobianDeterminantImage!=NULL){
        if(targetImage->datatype != jacobianDeterminantImage->datatype){
            fprintf(stderr,"[NiftyReg ERROR] reg_getVoxelBasedSSDGradient\n");
            fprintf(stderr,"[NiftyReg ERROR] Input images are expected to have the same type\n");
            exit(1);
        }
    }
    switch ( targetImage->datatype ){
        case NIFTI_TYPE_FLOAT32:
            reg_getVoxelBasedSSDGradient1<float>
                (targetImage, resultImage, resultImageGradient, ssdGradientImage, jacobianDeterminantImage,maxSD, mask);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_getVoxelBasedSSDGradient1<double>
                (targetImage, resultImage, resultImageGradient, ssdGradientImage, jacobianDeterminantImage,maxSD, mask);
            break;
        default:
            fprintf(stderr,"[NiftyReg ERROR] Target pixel type unsupported in the SSD gradient computation function.\n");
            exit(1);
	}
}
/* *************************************************************** */
/* *************************************************************** */
