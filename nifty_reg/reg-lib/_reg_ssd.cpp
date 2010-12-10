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
template<class PrecisionTYPE, class ResultTYPE, class TargetTYPE>
PrecisionTYPE reg_getSSD2(	nifti_image *targetImage,
                            nifti_image *resultImage,
                            int *mask
 							)
{
	TargetTYPE *targetPtr=static_cast<TargetTYPE *>(targetImage->data);
	ResultTYPE *resultPtr=static_cast<ResultTYPE *>(resultImage->data);

    int *maskPtr = &mask[0];
	
	PrecisionTYPE SSD=0.0;
	PrecisionTYPE n=0.0;
	
	for(unsigned int i=0; i<targetImage->nvox;i++){
        if(*maskPtr++>-1){
            TargetTYPE targetValue = *targetPtr;
            ResultTYPE resultValue = *resultPtr;
            if(targetValue==targetValue && resultValue==resultValue){
                SSD += (PrecisionTYPE)((targetValue-resultValue)*(targetValue-resultValue));
                n += 1.0;
            }
        }
        targetPtr++;
        resultPtr++;
	}
    return SSD/n;
}
/* *************************************************************** */
template<class PrecisionTYPE, class ResultTYPE>
PrecisionTYPE reg_getSSD1(	nifti_image *targetImage,
                            nifti_image *resultImage,
                            int *mask
 							)
{
	switch ( targetImage->datatype ){
#ifdef _NR_DEV
		case NIFTI_TYPE_UINT8:
            return reg_getSSD2<PrecisionTYPE,ResultTYPE,unsigned char>(targetImage,resultImage, mask);
			break;
		case NIFTI_TYPE_INT8:
            return reg_getSSD2<PrecisionTYPE,ResultTYPE,char>(targetImage,resultImage, mask);
			break;
		case NIFTI_TYPE_UINT16:
            return reg_getSSD2<PrecisionTYPE,ResultTYPE,unsigned short>(targetImage,resultImage, mask);
			break;
		case NIFTI_TYPE_INT16:
            return reg_getSSD2<PrecisionTYPE,ResultTYPE,short>(targetImage,resultImage, mask);
			break;
		case NIFTI_TYPE_UINT32:
            return reg_getSSD2<PrecisionTYPE,ResultTYPE,unsigned int>(targetImage,resultImage, mask);
			break;
		case NIFTI_TYPE_INT32:
            return reg_getSSD2<PrecisionTYPE,ResultTYPE,int>(targetImage,resultImage, mask);
			break;
        case NIFTI_TYPE_FLOAT64:
            return reg_getSSD2<PrecisionTYPE,ResultTYPE,double>(targetImage,resultImage, mask);
            break;
#endif
		case NIFTI_TYPE_FLOAT32:
            return reg_getSSD2<PrecisionTYPE,ResultTYPE,float>(targetImage,resultImage, mask);
            break;
		default:
            fprintf(stderr,"[NiftyReg ERROR] Target pixel type unsupported in the SSD computation function.");
            exit(1);
	}
	return 0.0;
}
/* *************************************************************** */
template<class PrecisionTYPE>
PrecisionTYPE reg_getSSD(	nifti_image *targetImage,
                            nifti_image *resultImage,
                            int *mask
 							)
{
	switch ( resultImage->datatype ){
#ifdef _NR_DEV
		case NIFTI_TYPE_UINT8:
            return reg_getSSD1<PrecisionTYPE,unsigned char>(targetImage,resultImage, mask);
			break;
		case NIFTI_TYPE_INT8:
            return reg_getSSD1<PrecisionTYPE,char>(targetImage,resultImage, mask);
			break;
		case NIFTI_TYPE_UINT16:
            return reg_getSSD1<PrecisionTYPE,unsigned short>(targetImage,resultImage, mask);
			break;
		case NIFTI_TYPE_INT16:
            return reg_getSSD1<PrecisionTYPE,short>(targetImage,resultImage, mask);
			break;
		case NIFTI_TYPE_UINT32:
            return reg_getSSD1<PrecisionTYPE,unsigned int>(targetImage,resultImage, mask);
			break;
		case NIFTI_TYPE_INT32:
            return reg_getSSD1<PrecisionTYPE,int>(targetImage,resultImage, mask);
			break;
        case NIFTI_TYPE_FLOAT64:
            return reg_getSSD1<PrecisionTYPE,double>(targetImage,resultImage, mask);
            break;
#endif
		case NIFTI_TYPE_FLOAT32:
            return reg_getSSD1<PrecisionTYPE,float>(targetImage,resultImage, mask);
            break;
		default:
            fprintf(stderr,"[NiftyReg ERROR] Result pixel type unsupported in the SSD computation function.");
            exit(1);
	}
	return 0.0;
}
template float reg_getSSD<float>(nifti_image *, nifti_image *, int *);
template double reg_getSSD<double>(nifti_image *, nifti_image *, int *);
/* *************************************************************** */
/* *************************************************************** */
template <class PrecisionTYPE, class TargetTYPE, class ResultTYPE, class GradientImgTYPE, class SSDGradTYPE>
void reg_getVoxelBasedSSDGradient4(	nifti_image *targetImage,
									nifti_image *resultImage,
									nifti_image *resultImageGradient,
									float maxSD,
                                    nifti_image *ssdGradientImage,
                                    int *mask
									)
{
	TargetTYPE *targetPtr=static_cast<TargetTYPE *>(targetImage->data);
	ResultTYPE *resultPtr=static_cast<ResultTYPE *>(resultImage->data);

	GradientImgTYPE *spatialGradPtrX=static_cast<GradientImgTYPE *>(resultImageGradient->data);
	GradientImgTYPE *spatialGradPtrY = &spatialGradPtrX[resultImageGradient->nx*resultImageGradient->ny*resultImageGradient->nz];
	GradientImgTYPE *spatialGradPtrZ = NULL;
	if(targetImage->nz>1) spatialGradPtrZ = &spatialGradPtrY[resultImageGradient->nx*resultImageGradient->ny*resultImageGradient->nz];

	SSDGradTYPE *ssdGradPtrX=static_cast<SSDGradTYPE *>(ssdGradientImage->data);
	SSDGradTYPE *ssdGradPtrY = &ssdGradPtrX[ssdGradientImage->nx*ssdGradientImage->ny*ssdGradientImage->nz];
	SSDGradTYPE *ssdGradPtrZ = NULL;
	if(targetImage->nz>1) ssdGradPtrZ = &ssdGradPtrY[ssdGradientImage->nx*ssdGradientImage->ny*ssdGradientImage->nz];

    int *maskPtr = &mask[0];

	for(unsigned int i=0; i<targetImage->nvox;i++){
        if(*maskPtr++>-1){
            TargetTYPE targetValue = *targetPtr;
            ResultTYPE resultValue = *resultPtr;
            PrecisionTYPE gradX=0;
            PrecisionTYPE gradY=0;
            PrecisionTYPE gradZ=0;
            if(targetValue==targetValue && resultValue==resultValue){
                PrecisionTYPE common = (PrecisionTYPE)(- 2.0 * (targetValue - resultValue));
                gradX = (PrecisionTYPE)(common * (*spatialGradPtrX)/maxSD);
                gradY = (PrecisionTYPE)(common * (*spatialGradPtrY)/maxSD);
                if(targetImage->nz>1) gradZ = (PrecisionTYPE)(common * (*spatialGradPtrZ)/maxSD);
            }
            *ssdGradPtrX = (SSDGradTYPE)gradX;
            *ssdGradPtrY = (SSDGradTYPE)gradY;
            if(targetImage->nz>1) *ssdGradPtrZ = (SSDGradTYPE)gradZ;
        }
        targetPtr++;
        resultPtr++;
        ssdGradPtrX++;
        ssdGradPtrY++;
        spatialGradPtrX++;
        spatialGradPtrY++;
        if(targetImage->nz>1){
            ssdGradPtrZ++;
            spatialGradPtrZ++;
        }
	}
}
/* *************************************************************** */
template <class PrecisionTYPE, class TargetTYPE, class ResultTYPE, class GradientImgTYPE>
void reg_getVoxelBasedSSDGradient3(	nifti_image *targetImage,
									nifti_image *resultImage,
									nifti_image *resultImageGradient,
									float maxSD,
                                    nifti_image *ssdGradientImage,
                                    int *mask
									)
{
	switch ( ssdGradientImage->datatype ){
		case NIFTI_TYPE_FLOAT32:
			reg_getVoxelBasedSSDGradient4<PrecisionTYPE,TargetTYPE,ResultTYPE,GradientImgTYPE,float>
                (targetImage, resultImage, resultImageGradient, maxSD, ssdGradientImage, mask);
			break;
#ifdef _NR_DEV
		case NIFTI_TYPE_FLOAT64:
			reg_getVoxelBasedSSDGradient4<PrecisionTYPE,TargetTYPE,ResultTYPE,GradientImgTYPE,double>
                (targetImage, resultImage, resultImageGradient, maxSD, ssdGradientImage, mask);
			break;
#endif
		default:
            fprintf(stderr,"[NiftyReg ERROR] SSD gradient pixel type unsupported in the SSD gradient computation function.");
            exit(1);
	}
}
/* *************************************************************** */
template <class PrecisionTYPE, class TargetTYPE, class ResultTYPE>
void reg_getVoxelBasedSSDGradient2(	nifti_image *targetImage,
									nifti_image *resultImage,
									nifti_image *resultImageGradient,
									float maxSD,
                                    nifti_image *ssdGradientImage,
                                    int *mask
									)
{
	switch ( resultImageGradient->datatype ){
		case NIFTI_TYPE_FLOAT32:
			reg_getVoxelBasedSSDGradient3<PrecisionTYPE,TargetTYPE,ResultTYPE,float>
                (targetImage, resultImage, resultImageGradient, maxSD, ssdGradientImage, mask);
			break;
#ifdef _NR_DEV
		case NIFTI_TYPE_FLOAT64:
			reg_getVoxelBasedSSDGradient3<PrecisionTYPE,TargetTYPE,ResultTYPE,double>
                (targetImage, resultImage, resultImageGradient, maxSD, ssdGradientImage, mask);
			break;
#endif
		default:
            fprintf(stderr,"[NiftyReg ERROR] Spatial gradient pixel type unsupported in the SSD gradient computation function.");
            exit(1);
	}
}
/* *************************************************************** */
template <class PrecisionTYPE, class TargetTYPE>
void reg_getVoxelBasedSSDGradient1(	nifti_image *targetImage,
									nifti_image *resultImage,
									nifti_image *resultImageGradient,
									float maxSD,
                                    nifti_image *ssdGradientImage,
                                    int *mask
									)
{
	switch ( resultImage->datatype ){
#ifdef _NR_DEV
		case NIFTI_TYPE_UINT8:
			reg_getVoxelBasedSSDGradient2<PrecisionTYPE,TargetTYPE,unsigned char>
                (targetImage, resultImage, resultImageGradient, maxSD, ssdGradientImage, mask);
			break;
		case NIFTI_TYPE_INT8:
			reg_getVoxelBasedSSDGradient2<PrecisionTYPE,TargetTYPE,char>
                (targetImage, resultImage, resultImageGradient, maxSD, ssdGradientImage, mask);
			break;
		case NIFTI_TYPE_UINT16:
			reg_getVoxelBasedSSDGradient2<PrecisionTYPE,TargetTYPE,unsigned short>
                (targetImage, resultImage, resultImageGradient, maxSD, ssdGradientImage, mask);
			break;
		case NIFTI_TYPE_INT16:
			reg_getVoxelBasedSSDGradient2<PrecisionTYPE,TargetTYPE,short>
                (targetImage, resultImage, resultImageGradient, maxSD, ssdGradientImage, mask);
			break;
		case NIFTI_TYPE_UINT32:
			reg_getVoxelBasedSSDGradient2<PrecisionTYPE,TargetTYPE,unsigned int>
                (targetImage, resultImage, resultImageGradient, maxSD, ssdGradientImage, mask);
			break;
		case NIFTI_TYPE_INT32:
			reg_getVoxelBasedSSDGradient2<PrecisionTYPE,TargetTYPE,int>
                (targetImage, resultImage, resultImageGradient, maxSD, ssdGradientImage, mask);
			break;
        case NIFTI_TYPE_FLOAT64:
            reg_getVoxelBasedSSDGradient2<PrecisionTYPE,TargetTYPE,double>
                (targetImage, resultImage, resultImageGradient, maxSD, ssdGradientImage, mask);
            break;
#endif
		case NIFTI_TYPE_FLOAT32:
			reg_getVoxelBasedSSDGradient2<PrecisionTYPE,TargetTYPE,float>
                (targetImage, resultImage, resultImageGradient, maxSD, ssdGradientImage, mask);
            break;
		default:
            fprintf(stderr,"[NiftyReg ERROR] Result pixel type unsupported in the SSD gradient computation function.");
            exit(1);
	}
}
/* *************************************************************** */
template <class PrecisionTYPE>
void reg_getVoxelBasedSSDGradient(	nifti_image *targetImage,
									nifti_image *resultImage,
									nifti_image *resultImageGradient,
									float maxSD,
                                    nifti_image *ssdGradientImage,
                                    int *mask
									)
{
	switch ( targetImage->datatype ){
#ifdef _NR_DEV
		case NIFTI_TYPE_UINT8:
			reg_getVoxelBasedSSDGradient1<PrecisionTYPE,unsigned char>
                (targetImage, resultImage, resultImageGradient, maxSD, ssdGradientImage, mask);
			break;
		case NIFTI_TYPE_INT8:
			reg_getVoxelBasedSSDGradient1<PrecisionTYPE,char>
                (targetImage, resultImage, resultImageGradient, maxSD, ssdGradientImage, mask);
			break;
		case NIFTI_TYPE_UINT16:
			reg_getVoxelBasedSSDGradient1<PrecisionTYPE,unsigned short>
                (targetImage, resultImage, resultImageGradient, maxSD, ssdGradientImage, mask);
			break;
		case NIFTI_TYPE_INT16:
			reg_getVoxelBasedSSDGradient1<PrecisionTYPE,short>
                (targetImage, resultImage, resultImageGradient, maxSD, ssdGradientImage, mask);
			break;
		case NIFTI_TYPE_UINT32:
			reg_getVoxelBasedSSDGradient1<PrecisionTYPE,unsigned int>
                (targetImage, resultImage, resultImageGradient, maxSD, ssdGradientImage, mask);
			break;
		case NIFTI_TYPE_INT32:
			reg_getVoxelBasedSSDGradient1<PrecisionTYPE,int>
                (targetImage, resultImage, resultImageGradient, maxSD, ssdGradientImage, mask);
			break;
        case NIFTI_TYPE_FLOAT64:
            reg_getVoxelBasedSSDGradient1<PrecisionTYPE,double>
                (targetImage, resultImage, resultImageGradient, maxSD, ssdGradientImage, mask);
            break;
#endif
		case NIFTI_TYPE_FLOAT32:
			reg_getVoxelBasedSSDGradient1<PrecisionTYPE,float>
                (targetImage, resultImage, resultImageGradient, maxSD, ssdGradientImage, mask);
            break;
		default:
            fprintf(stderr,"[NiftyReg ERROR] Target pixel type unsupported in the SSD gradient computation function.");
            exit(1);
	}
}
template void reg_getVoxelBasedSSDGradient<float>(nifti_image *, nifti_image *, nifti_image *, float, nifti_image *, int *);
template void reg_getVoxelBasedSSDGradient<double>(nifti_image *, nifti_image *, nifti_image *, float, nifti_image *, int *);
/* *************************************************************** */
/* *************************************************************** */
