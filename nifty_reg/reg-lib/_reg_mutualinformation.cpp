/*
 *  _reg_mutualinformation.cpp
 *  
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_MUTUALINFORMATION_CPP
#define _REG_MUTUALINFORMATION_CPP

#include "_reg_mutualinformation.h"

/* *************************************************************** */
template<class PrecisionTYPE>
PrecisionTYPE GetBasisSplineValue(PrecisionTYPE x)
{
	x=fabs(x);
	PrecisionTYPE value=0.0;
	if(x<2.0)
		if(x<1.0)
			value = (PrecisionTYPE)(2.0f/3.0f + (0.5f*x-1.0)*x*x);
		else{
			x-=2.0f;
			value = -x*x*x/6.0f;
	}
	return value;
}
/* *************************************************************** */
template<class PrecisionTYPE>
PrecisionTYPE GetBasisSplineDerivativeValue(PrecisionTYPE ori)
{
	PrecisionTYPE x=fabs(ori);
	PrecisionTYPE value=0.0;
	if(x<2.0)
		if(x<1.0)
			value = (PrecisionTYPE)((1.5f*x-2.0)*ori);
		else{
			x-=2.0f;
			value = -0.5f * x * x;
			if(ori<0.0f)value =-value;
	}
	return value;
}
/* *************************************************************** */
/* *************************************************************** */
extern "C++" template<class PrecisionTYPE, class TargetTYPE, class ResultTYPE>
void reg_getEntropies3(	nifti_image *targetImage,
                        nifti_image *resultImage,
                        int type,
                        int binning,
                        PrecisionTYPE *probaJointHistogram,
                        PrecisionTYPE *logJointHistogram,
                        PrecisionTYPE *entropies,
                        int *mask)
{
	TargetTYPE *targetPtr = static_cast<TargetTYPE *>(targetImage->data);
	ResultTYPE *resultPtr = static_cast<ResultTYPE *>(resultImage->data);

    int *maskPtr = &mask[0];

	memset(probaJointHistogram, 0, binning*(binning+2) * sizeof(PrecisionTYPE));
	memset(logJointHistogram, 0, binning*(binning+2) * sizeof(PrecisionTYPE));
	PrecisionTYPE voxelNumber=0.0;
	
	int targetIndex;
	int resultIndex;
	if(type==1){ // parzen windows approach to fill the joint histogram
		for(int z=0; z<targetImage->nz; z++){ // loop over the target space
			for(int y=0; y<targetImage->ny; y++){
				for(int x=0; x<targetImage->nx; x++){
					TargetTYPE targetValue=*targetPtr++;
					ResultTYPE resultValue=*resultPtr++;
                    if( targetValue>0.0f &&
                        resultValue>0.0f &&
                        targetValue<(TargetTYPE)binning &&
                        resultValue<(ResultTYPE)binning &&
						*maskPtr++>-1 &&
						targetValue==targetValue &&
						resultValue==resultValue){
                        // The two is added because the image is resample between 2 and bin +2
                        // if 64 bins are used the histogram will have 68 bins et the image will be between 2 and 65
						for(int t=(int)(targetValue-1.0); t<(int)(targetValue+2.0); t++){
							if(-1<t && t<binning){
								for(int r=(int)(resultValue-1.0); r<(int)(resultValue+2.0); r++){
									if(-1<r && r<binning){
										PrecisionTYPE coeff = GetBasisSplineValue<PrecisionTYPE>
                                            ((PrecisionTYPE)t-(PrecisionTYPE)targetValue) *
											GetBasisSplineValue<PrecisionTYPE>
                                            ((PrecisionTYPE)r-(PrecisionTYPE)resultValue);
										probaJointHistogram[t*binning+r] += coeff;
										voxelNumber += coeff;
									} // O<j<bin
								} // j
							} // 0<i<bin
						} // i
					} // targetValue>0 && resultValue>0
				} // x
			} // y
		} // z
	}
	else{ // classical trilinear interpolation only
		for(unsigned int index=0; index<targetImage->nvox; index++){
            if(*maskPtr++>-1){
                TargetTYPE targetInt = *targetPtr;
                ResultTYPE resultInt = *resultPtr;
                if( targetInt>(TargetTYPE)(0) &&
                    targetInt<(TargetTYPE)(binning) &&
                    resultInt>(ResultTYPE)(0) &&
                    resultInt<(ResultTYPE)(binning) &&
                    targetInt==targetInt &&
                    resultInt==resultInt){
	                probaJointHistogram[(unsigned int)((floorf((float)targetInt) * binning + floorf((float)resultInt)))]++;
	                voxelNumber++;
                }
            }
            targetPtr++;
            resultPtr++;
		}
	}
	if(type==2){ // parzen windows approximation by smoothing the classical approach
		// the logJointHistogram array is used as a temporary array
		PrecisionTYPE window[3];
		window[0]=window[2]=GetBasisSplineValue((PrecisionTYPE)(-1.0));
		window[1]=GetBasisSplineValue((PrecisionTYPE)(0.0));
		
		//The joint histogram is smoothed along the target axis
		for(int s=0; s<binning; s++){
			for(int t=0; t<binning; t++){
	
				PrecisionTYPE value=(PrecisionTYPE)(0.0);
				targetIndex = t-1;
				PrecisionTYPE *ptrHisto = &probaJointHistogram[targetIndex*binning+s];
				
				for(int it=0; it<3; it++){
					if(-1<targetIndex && targetIndex<binning){
						value += *ptrHisto * window[it];
					}
					ptrHisto += binning;
					targetIndex++;
				}
				logJointHistogram[t*binning+s] = value;
			}
		}
		//The joint histogram is smoothed along the source axis
		for(int t=0; t<binning; t++){
			for(int s=0; s<binning; s++){
	
				PrecisionTYPE value=0.0;
				resultIndex = s-1;
				PrecisionTYPE *ptrHisto = &logJointHistogram[t*binning+resultIndex];
				
				for(int it=0; it<3; it++){
					if(-1<resultIndex && resultIndex<binning){
						value += *ptrHisto * window[it];
					}
					ptrHisto++;
					resultIndex++;
				}
				probaJointHistogram[t*binning+s] = value;
			}
		}
		memset(logJointHistogram, 0, binning*(binning+2) * sizeof(PrecisionTYPE));
	}
	
	for(int index=0; index<binning*binning; index++)
		probaJointHistogram[index] /= voxelNumber;

	// The marginal probability are stored first
	targetIndex = binning*binning;
	resultIndex = targetIndex+binning;
	for(int t=0; t<binning; t++){
		PrecisionTYPE sum=0.0;
		unsigned int coord=t*binning;
		for(int r=0; r<binning; r++){
			sum += probaJointHistogram[coord++];
		}
		probaJointHistogram[targetIndex++] = sum;
	}
	for(int r=0; r<binning; r++){
		PrecisionTYPE sum=0.0;
		unsigned int coord=r;
		for(int t=0; t<binning; t++){
			sum += probaJointHistogram[coord];
			coord += binning;
		}
		probaJointHistogram[resultIndex++] = sum;
	}

	PrecisionTYPE tEntropy = 0.0;
	PrecisionTYPE rEntropy = 0.0;
	PrecisionTYPE jEntropy = 0.0;

	targetIndex = binning*binning;
	resultIndex = targetIndex+binning;
	for(int tr=0; tr<binning; tr++){
		PrecisionTYPE targetValue = probaJointHistogram[targetIndex];
		PrecisionTYPE resultValue = probaJointHistogram[resultIndex];
		PrecisionTYPE targetLog=0.0;
		PrecisionTYPE resultLog=0.0;
		if(targetValue)	targetLog = log(targetValue);
		if(resultValue)	resultLog = log(resultValue);
		tEntropy -= targetValue*targetLog;
		rEntropy -= resultValue*resultLog;
		logJointHistogram[targetIndex++] = targetLog;
		logJointHistogram[resultIndex++] = resultLog;
	}

	for(int tr=0; tr<binning*binning; tr++){
		PrecisionTYPE jointValue = probaJointHistogram[tr];
		PrecisionTYPE jointLog = 0.0;
		if(jointValue)	jointLog = log(jointValue);
		jEntropy -= jointValue*jointLog;
		logJointHistogram[tr] = jointLog;
	}

	entropies[0]=tEntropy;	// target image entropy
	entropies[1]=rEntropy;	// result image entropy
	entropies[2]=jEntropy;	// joint entropy
	entropies[3]=voxelNumber;	// Number of voxel

	return;
}
/* *************************************************************** */
extern "C++" template<class PrecisionTYPE, class TargetTYPE>
void reg_getEntropies2(	nifti_image *targetImage,
 						nifti_image *resultImage,
	 					int type,
	 					int binning,
	 					PrecisionTYPE *probaJointHistogram,
	 					PrecisionTYPE *logJointHistogram,
						PrecisionTYPE *entropies,
						int *mask
 					)
{
	switch(resultImage->datatype){
		case NIFTI_TYPE_UINT8:
			reg_getEntropies3<PrecisionTYPE,TargetTYPE,unsigned char>
                (targetImage, resultImage, type, binning, probaJointHistogram, logJointHistogram, entropies, mask);
			break;
		case NIFTI_TYPE_INT8:
			reg_getEntropies3<PrecisionTYPE,TargetTYPE,char>
                (targetImage, resultImage, type, binning, probaJointHistogram, logJointHistogram, entropies, mask);
			break;
		case NIFTI_TYPE_UINT16:
			reg_getEntropies3<PrecisionTYPE,TargetTYPE,unsigned short>
                (targetImage, resultImage, type, binning, probaJointHistogram, logJointHistogram, entropies, mask);
			break;
		case NIFTI_TYPE_INT16:
			reg_getEntropies3<PrecisionTYPE,TargetTYPE,short>
                (targetImage, resultImage, type, binning, probaJointHistogram, logJointHistogram, entropies, mask);
			break;
		case NIFTI_TYPE_UINT32:
			reg_getEntropies3<PrecisionTYPE,TargetTYPE,unsigned int>
                (targetImage, resultImage, type, binning, probaJointHistogram, logJointHistogram, entropies, mask);
			break;
		case NIFTI_TYPE_INT32:
			reg_getEntropies3<PrecisionTYPE,TargetTYPE,int>
                (targetImage, resultImage, type, binning, probaJointHistogram, logJointHistogram, entropies, mask);
			break;
		case NIFTI_TYPE_FLOAT32:
			reg_getEntropies3<PrecisionTYPE,TargetTYPE,float>
                (targetImage, resultImage, type, binning, probaJointHistogram, logJointHistogram, entropies, mask);
			break;
		case NIFTI_TYPE_FLOAT64:
			reg_getEntropies3<PrecisionTYPE,TargetTYPE,double>
                (targetImage, resultImage, type, binning, probaJointHistogram, logJointHistogram, entropies, mask);
			break;
		default:
			printf("err\treg_getEntropies\tThe result image data type is not supported\n");
			return;
	}
	return;
}
/* *************************************************************** */
extern "C++" template<class PrecisionTYPE>
void reg_getEntropies(	nifti_image *targetImage,
 						nifti_image *resultImage,
	 					int type,
	 					int binning,
	 					PrecisionTYPE *probaJointHistogram,
	 					PrecisionTYPE *logJointHistogram,
					    PrecisionTYPE *entropies,
                        int *mask
 					)
{
	switch(targetImage->datatype){
		case NIFTI_TYPE_UINT8:
			reg_getEntropies2<PrecisionTYPE,unsigned char>
                (targetImage, resultImage, type, binning, probaJointHistogram, logJointHistogram, entropies, mask);
			break;
		case NIFTI_TYPE_INT8:
			reg_getEntropies2<PrecisionTYPE,char>
                (targetImage, resultImage, type, binning, probaJointHistogram, logJointHistogram, entropies, mask);
			break;
		case NIFTI_TYPE_UINT16:
			reg_getEntropies2<PrecisionTYPE,unsigned short>
                (targetImage, resultImage, type, binning, probaJointHistogram, logJointHistogram, entropies, mask);
			break;
		case NIFTI_TYPE_INT16:
			reg_getEntropies2<PrecisionTYPE,short>
                (targetImage, resultImage, type, binning, probaJointHistogram, logJointHistogram, entropies, mask);
			break;
		case NIFTI_TYPE_UINT32:
			reg_getEntropies2<PrecisionTYPE,unsigned int>
                (targetImage, resultImage, type, binning, probaJointHistogram, logJointHistogram, entropies, mask);
			break;
		case NIFTI_TYPE_INT32:
			reg_getEntropies2<PrecisionTYPE,int>
                (targetImage, resultImage, type, binning, probaJointHistogram, logJointHistogram, entropies, mask);
			break;
		case NIFTI_TYPE_FLOAT32:
			reg_getEntropies2<PrecisionTYPE,float>
                (targetImage, resultImage, type, binning, probaJointHistogram, logJointHistogram, entropies, mask);
			break;
		case NIFTI_TYPE_FLOAT64:
			reg_getEntropies2<PrecisionTYPE,double>
                (targetImage, resultImage, type, binning, probaJointHistogram, logJointHistogram, entropies, mask);
			break;
		default:
			printf("err\treg_getEntropies\tThe target image data type is not supported\n");
			return;
	}
	return;
}
/* *************************************************************** */
template void reg_getEntropies<float>(nifti_image *, nifti_image *, int, int, float *, float *, float *, int *);
template void reg_getEntropies<double>(nifti_image *, nifti_image *, int, int, double *, double *, double *, int *);
/* *************************************************************** */
/* *************************************************************** */
/* *************************************************************** */
template<class PrecisionTYPE,class TargetTYPE,class ResultTYPE,class ResultGradientTYPE,class NMIGradientTYPE>
void reg_getVoxelBasedNMIGradientUsingPW2D(	nifti_image *targetImage,
						nifti_image *resultImage,
                        int type,
						nifti_image *resultImageGradient,
						int binning,
						PrecisionTYPE *logJointHistogram,
						PrecisionTYPE *entropies,
						nifti_image *nmiGradientImage,
                        int *mask)
{
	TargetTYPE *targetPtr = static_cast<TargetTYPE *>(targetImage->data);
	ResultTYPE *resultPtr = static_cast<ResultTYPE *>(resultImage->data);
	ResultGradientTYPE *resultGradientPtrX = static_cast<ResultGradientTYPE *>(resultImageGradient->data);
	ResultGradientTYPE *resultGradientPtrY = &resultGradientPtrX[resultImage->nvox];
	NMIGradientTYPE *nmiGradientPtrX = static_cast<NMIGradientTYPE *>(nmiGradientImage->data);
	NMIGradientTYPE *nmiGradientPtrY = &nmiGradientPtrX[resultImage->nvox];
	
    int *maskPtr = &mask[0];

	// In a first time the NMI gradient is computed for every voxel
	memset(nmiGradientPtrX,0,nmiGradientImage->nvox*nmiGradientImage->nbyper);

	PrecisionTYPE NMI = (entropies[0]+entropies[1])/entropies[2];

	unsigned int binningSquare = binning*binning;

	for(int y=0; y<targetImage->ny; y++){
		for(int x=0; x<targetImage->nx; x++){

               if(*maskPtr++>-1){
                   TargetTYPE targetValue = *targetPtr;
                   ResultTYPE resultValue = *resultPtr;
                   if(targetValue>0.0f &&
                      resultValue>0.0f &&
                      targetValue<(TargetTYPE)binning &&
                      resultValue<(ResultTYPE)binning &&
					  targetValue==targetValue &&
					  resultValue==resultValue){
                   // The two is added because the image is resample between 2 and bin +2
                   // if 64 bins are used the histogram will have 68 bins et the image will be between 2 and 65

                       if(type!=1){
                           targetValue = (TargetTYPE)floor((double)targetValue);
                           resultValue = (ResultTYPE)floor((double)resultValue);
                       }

					PrecisionTYPE resDeriv[2];
					resDeriv[0] = (PrecisionTYPE)(*resultGradientPtrX);
					resDeriv[1] = (PrecisionTYPE)(*resultGradientPtrY);
					   
					if(resDeriv[0]==resDeriv[0] && resDeriv[1]==resDeriv[1]){

						PrecisionTYPE jointEntropyDerivative_X = 0.0;
						PrecisionTYPE movingEntropyDerivative_X = 0.0;
						PrecisionTYPE fixedEntropyDerivative_X = 0.0;

						PrecisionTYPE jointEntropyDerivative_Y = 0.0;
						PrecisionTYPE movingEntropyDerivative_Y = 0.0;
						PrecisionTYPE fixedEntropyDerivative_Y = 0.0;
						
						for(int t=(int)(targetValue-1.0); t<(int)(targetValue+2.0); t++){
							if(-1<t && t<binning){
								for(int r=(int)(resultValue-1.0); r<(int)(resultValue+2.0); r++){
									if(-1<r && r<binning){
										PrecisionTYPE commonValue =  GetBasisSplineValue<PrecisionTYPE>((PrecisionTYPE)t-(PrecisionTYPE)targetValue) * 
											GetBasisSplineDerivativeValue<PrecisionTYPE>((PrecisionTYPE)r-(PrecisionTYPE)resultValue);
										
										PrecisionTYPE jointLog = logJointHistogram[t*binning+r];
										PrecisionTYPE targetLog = logJointHistogram[binningSquare+t];
										PrecisionTYPE resultLog = logJointHistogram[binningSquare+binning+r];
										
										PrecisionTYPE temp = commonValue * resDeriv[0];
										jointEntropyDerivative_X -= temp * jointLog;
										fixedEntropyDerivative_X -= temp * targetLog;
										movingEntropyDerivative_X -= temp * resultLog;
										
										temp = commonValue * resDeriv[1];
										jointEntropyDerivative_Y -= temp * jointLog;
										fixedEntropyDerivative_Y -= temp * targetLog;
										movingEntropyDerivative_Y -= temp * resultLog;										
									} // O<t<bin
								} // t
							} // 0<r<bin
						} // r

						PrecisionTYPE temp = (PrecisionTYPE)(entropies[2]);
						// (Marc) I removed the normalisation by the voxel number as each gradient has to be normalised in the same way (NMI, BE, JAC)
						*nmiGradientPtrX = (NMIGradientTYPE)((fixedEntropyDerivative_X + movingEntropyDerivative_X - NMI * jointEntropyDerivative_X) / temp);
						*nmiGradientPtrY = (NMIGradientTYPE)((fixedEntropyDerivative_Y + movingEntropyDerivative_Y - NMI * jointEntropyDerivative_Y) / temp);
							
					} // gradient nan
				} // value > 0
               }// mask > -1
               targetPtr++;
               resultPtr++;
               nmiGradientPtrX++;
               nmiGradientPtrY++;
               resultGradientPtrX++;
               resultGradientPtrY++;
		}
	}
}
/* *************************************************************** */
template<class PrecisionTYPE,class TargetTYPE,class ResultTYPE,class ResultGradientTYPE,class NMIGradientTYPE>
void reg_getVoxelBasedNMIGradientUsingPW3D(	nifti_image *targetImage,
						nifti_image *resultImage,
                        int type,
						nifti_image *resultImageGradient,
						int binning,
						PrecisionTYPE *logJointHistogram,
						PrecisionTYPE *entropies,
						nifti_image *nmiGradientImage,
                        int *mask)
{
	TargetTYPE *targetPtr = static_cast<TargetTYPE *>(targetImage->data);
	ResultTYPE *resultPtr = static_cast<ResultTYPE *>(resultImage->data);
	ResultGradientTYPE *resultGradientPtrX = static_cast<ResultGradientTYPE *>(resultImageGradient->data);
	ResultGradientTYPE *resultGradientPtrY = &resultGradientPtrX[resultImage->nvox];
	ResultGradientTYPE *resultGradientPtrZ = &resultGradientPtrY[resultImage->nvox];
	NMIGradientTYPE *nmiGradientPtrX = static_cast<NMIGradientTYPE *>(nmiGradientImage->data);
	NMIGradientTYPE *nmiGradientPtrY = &nmiGradientPtrX[resultImage->nvox];
	NMIGradientTYPE *nmiGradientPtrZ = &nmiGradientPtrY[resultImage->nvox];
	
    int *maskPtr = &mask[0];

	// In a first time the NMI gradient is computed for every voxel
	memset(nmiGradientPtrX,0,nmiGradientImage->nvox*nmiGradientImage->nbyper);

	PrecisionTYPE NMI = (entropies[0]+entropies[1])/entropies[2];

	unsigned int binningSquare = binning*binning;

	for(int z=0; z<targetImage->nz; z++){
		for(int y=0; y<targetImage->ny; y++){
			for(int x=0; x<targetImage->nx; x++){

                if(*maskPtr++>-1){
                    TargetTYPE targetValue = *targetPtr;
                    ResultTYPE resultValue = *resultPtr;
                    if(targetValue>0.0f &&
                       resultValue>0.0f &&
                       targetValue<(TargetTYPE)binning &&
                       resultValue<(ResultTYPE)binning &&
					   targetValue==targetValue &&
					   resultValue==resultValue){
						// The two is added because the image is resample between 2 and bin +2
						// if 64 bins are used the histogram will have 68 bins et the image will be between 2 and 65

                        if(type!=1){
                            targetValue = (TargetTYPE)floor((double)targetValue);
                            resultValue = (ResultTYPE)floor((double)resultValue);
                        }

						PrecisionTYPE resDeriv[3];
						resDeriv[0] = (PrecisionTYPE)(*resultGradientPtrX);
						resDeriv[1] = (PrecisionTYPE)(*resultGradientPtrY);
						resDeriv[2] = (PrecisionTYPE)(*resultGradientPtrZ);
						
						if(resDeriv[0]==resDeriv[0] && resDeriv[1]==resDeriv[1] && resDeriv[2]==resDeriv[2]){

							PrecisionTYPE jointEntropyDerivative_X = 0.0;
							PrecisionTYPE movingEntropyDerivative_X = 0.0;
							PrecisionTYPE fixedEntropyDerivative_X = 0.0;

							PrecisionTYPE jointEntropyDerivative_Y = 0.0;
							PrecisionTYPE movingEntropyDerivative_Y = 0.0;
							PrecisionTYPE fixedEntropyDerivative_Y = 0.0;

							PrecisionTYPE jointEntropyDerivative_Z = 0.0;
							PrecisionTYPE movingEntropyDerivative_Z = 0.0;
							PrecisionTYPE fixedEntropyDerivative_Z = 0.0;
							
							for(int t=(int)(targetValue-1.0); t<(int)(targetValue+2.0); t++){
								if(-1<t && t<binning){
									for(int r=(int)(resultValue-1.0); r<(int)(resultValue+2.0); r++){
										if(-1<r && r<binning){
											PrecisionTYPE commonValue =  GetBasisSplineValue<PrecisionTYPE>((PrecisionTYPE)t-(PrecisionTYPE)targetValue) * 
												GetBasisSplineDerivativeValue<PrecisionTYPE>((PrecisionTYPE)r-(PrecisionTYPE)resultValue);
											
											PrecisionTYPE jointLog = logJointHistogram[t*binning+r];
											PrecisionTYPE targetLog = logJointHistogram[binningSquare+t];
											PrecisionTYPE resultLog = logJointHistogram[binningSquare+binning+r];
											
											PrecisionTYPE temp = commonValue * resDeriv[0];
											jointEntropyDerivative_X -= temp * jointLog;
											fixedEntropyDerivative_X -= temp * targetLog;
											movingEntropyDerivative_X -= temp * resultLog;
											
											temp = commonValue * resDeriv[1];
											jointEntropyDerivative_Y -= temp * jointLog;
											fixedEntropyDerivative_Y -= temp * targetLog;
											movingEntropyDerivative_Y -= temp * resultLog;
											
											temp = commonValue * resDeriv[2];
											jointEntropyDerivative_Z -= temp * jointLog;
											fixedEntropyDerivative_Z -= temp * targetLog;
											movingEntropyDerivative_Z -= temp * resultLog;
											
										} // O<t<bin
									} // t
								} // 0<r<bin
							} // r
						
							// (Marc) I removed the normalisation by the voxel number as each gradient has to be normalised in the same way (NMI, BE, JAC)
							PrecisionTYPE temp = (PrecisionTYPE)(entropies[2]);
							*nmiGradientPtrX = (NMIGradientTYPE)((fixedEntropyDerivative_X + movingEntropyDerivative_X - NMI * jointEntropyDerivative_X) / temp);
							*nmiGradientPtrY = (NMIGradientTYPE)((fixedEntropyDerivative_Y + movingEntropyDerivative_Y - NMI * jointEntropyDerivative_Y) / temp);
							*nmiGradientPtrZ = (NMIGradientTYPE)((fixedEntropyDerivative_Z + movingEntropyDerivative_Z - NMI * jointEntropyDerivative_Z) / temp);
								
						}
					} // value > 0
                }// mask > -1
                targetPtr++;
                resultPtr++;
                nmiGradientPtrX++;
                nmiGradientPtrY++;
                nmiGradientPtrZ++;
                resultGradientPtrX++;
                resultGradientPtrY++;
                resultGradientPtrZ++;
			}
		}
	}
}
/* *************************************************************** */
template<class PrecisionTYPE,class TargetTYPE,class ResultTYPE,class ResultGradientTYPE>
void reg_getVoxelBasedNMIGradientUsingPW3(	nifti_image *targetImage,
                                            nifti_image *resultImage,
                                            int type,
                                            nifti_image *resultImageGradient,
                                            int binning,
                                            PrecisionTYPE *logJointHistogram,
                                            PrecisionTYPE *entropies,
                                            nifti_image *nmiGradientImage,
                                            int *mask)
{
	if(nmiGradientImage->nz>1){
		switch(nmiGradientImage->datatype){
			case NIFTI_TYPE_FLOAT32:
				reg_getVoxelBasedNMIGradientUsingPW3D<PrecisionTYPE,TargetTYPE,ResultTYPE,ResultGradientTYPE,float>
	                (targetImage, resultImage, type, resultImageGradient, binning, logJointHistogram, entropies, nmiGradientImage, mask);
				break;
			case NIFTI_TYPE_FLOAT64:
				reg_getVoxelBasedNMIGradientUsingPW3D<PrecisionTYPE,TargetTYPE,ResultTYPE,ResultGradientTYPE,double>
	                (targetImage, resultImage, type, resultImageGradient, binning, logJointHistogram, entropies, nmiGradientImage, mask);
				break;
			default:
				printf("err\treg_getVoxelBasedNMIGradientUsingPW\tThe result image gradient data type is not supported\n");
				return;
		}
	}else{
		switch(nmiGradientImage->datatype){
			case NIFTI_TYPE_FLOAT32:
				reg_getVoxelBasedNMIGradientUsingPW2D<PrecisionTYPE,TargetTYPE,ResultTYPE,ResultGradientTYPE,float>
	                (targetImage, resultImage, type, resultImageGradient, binning, logJointHistogram, entropies, nmiGradientImage, mask);
				break;
			case NIFTI_TYPE_FLOAT64:
				reg_getVoxelBasedNMIGradientUsingPW2D<PrecisionTYPE,TargetTYPE,ResultTYPE,ResultGradientTYPE,double>
	                (targetImage, resultImage, type, resultImageGradient, binning, logJointHistogram, entropies, nmiGradientImage, mask);
				break;
			default:
				printf("err\treg_getVoxelBasedNMIGradientUsingPW\tThe result image gradient data type is not supported\n");
				return;
		}
	}
}
/* *************************************************************** */
template<class PrecisionTYPE,class TargetTYPE,class ResultTYPE>
void reg_getVoxelBasedNMIGradientUsingPW2(	nifti_image *targetImage,
                                            nifti_image *resultImage,
                                            int type,
                                            nifti_image *resultImageGradient,
                                            int binning,
                                            PrecisionTYPE *logJointHistogram,
                                            PrecisionTYPE *entropies,
                                            nifti_image *nmiGradientImage,
                                            int *mask)
{
	switch(resultImageGradient->datatype){
		case NIFTI_TYPE_FLOAT32:
			reg_getVoxelBasedNMIGradientUsingPW3<PrecisionTYPE,TargetTYPE,ResultTYPE,float>
                (targetImage, resultImage, type, resultImageGradient, binning, logJointHistogram, entropies, nmiGradientImage, mask);
			break;
		case NIFTI_TYPE_FLOAT64:
			reg_getVoxelBasedNMIGradientUsingPW3<PrecisionTYPE,TargetTYPE,ResultTYPE,double>
                (targetImage, resultImage, type, resultImageGradient, binning, logJointHistogram, entropies, nmiGradientImage, mask);
			break;
		default:
			printf("err\treg_getVoxelBasedNMIGradientUsingPW\tThe result image gradient data type is not supported\n");
			return;
	}
}
/* *************************************************************** */
template<class PrecisionTYPE,class TargetTYPE>
void reg_getVoxelBasedNMIGradientUsingPW1(  nifti_image *targetImage,
                                            nifti_image *resultImage,
                                            int type,
                                            nifti_image *resultImageGradient,
                                            int binning,
                                            PrecisionTYPE *logJointHistogram,
                                            PrecisionTYPE *entropies,
                                            nifti_image *nmiGradientImage,
                                            int *mask)
{
	switch(resultImage->datatype){
				case NIFTI_TYPE_UINT8:
			reg_getVoxelBasedNMIGradientUsingPW2<PrecisionTYPE,TargetTYPE,unsigned char>
                (targetImage, resultImage, type, resultImageGradient, binning, logJointHistogram, entropies, nmiGradientImage, mask);
			break;
		case NIFTI_TYPE_INT8:
			reg_getVoxelBasedNMIGradientUsingPW2<PrecisionTYPE,TargetTYPE,char>
                (targetImage, resultImage, type, resultImageGradient, binning, logJointHistogram, entropies, nmiGradientImage, mask);
			break;
		case NIFTI_TYPE_UINT16:
			reg_getVoxelBasedNMIGradientUsingPW2<PrecisionTYPE,TargetTYPE,unsigned short>
                (targetImage, resultImage, type, resultImageGradient, binning, logJointHistogram, entropies, nmiGradientImage, mask);
			break;
		case NIFTI_TYPE_INT16:
			reg_getVoxelBasedNMIGradientUsingPW2<PrecisionTYPE,TargetTYPE,short>
                (targetImage, resultImage, type, resultImageGradient, binning, logJointHistogram, entropies, nmiGradientImage, mask);
			break;
		case NIFTI_TYPE_UINT32:
			reg_getVoxelBasedNMIGradientUsingPW2<PrecisionTYPE,TargetTYPE,unsigned int>
                (targetImage, resultImage, type, resultImageGradient, binning, logJointHistogram, entropies, nmiGradientImage, mask);
			break;
		case NIFTI_TYPE_INT32:
			reg_getVoxelBasedNMIGradientUsingPW2<PrecisionTYPE,TargetTYPE,int>
                (targetImage, resultImage, type, resultImageGradient, binning, logJointHistogram, entropies, nmiGradientImage, mask);
			break;
		case NIFTI_TYPE_FLOAT32:
			reg_getVoxelBasedNMIGradientUsingPW2<PrecisionTYPE,TargetTYPE,float>
                (targetImage, resultImage, type, resultImageGradient, binning, logJointHistogram, entropies, nmiGradientImage, mask);
			break;
		case NIFTI_TYPE_FLOAT64:
			reg_getVoxelBasedNMIGradientUsingPW2<PrecisionTYPE,TargetTYPE,double>
                (targetImage, resultImage, type, resultImageGradient, binning, logJointHistogram, entropies, nmiGradientImage, mask);
			break;
		default:
			printf("err\treg_getVoxelBasedNMIGradientUsingPW\tThe result image data type is not supported\n");
			return;
	}
}
/* *************************************************************** */
template<class PrecisionTYPE>
void reg_getVoxelBasedNMIGradientUsingPW(   nifti_image *targetImage,
                                            nifti_image *resultImage,
                                            int type,
                                            nifti_image *resultImageGradient,
                                            int binning,
                                            PrecisionTYPE *logJointHistogram,
                                            PrecisionTYPE *entropies,
                                            nifti_image *nmiGradientImage,
                                            int *mask)
{
	switch(targetImage->datatype){
				case NIFTI_TYPE_UINT8:
			reg_getVoxelBasedNMIGradientUsingPW1<PrecisionTYPE,unsigned char>
                (targetImage, resultImage, type, resultImageGradient, binning, logJointHistogram, entropies, nmiGradientImage, mask);
			break;
		case NIFTI_TYPE_INT8:
			reg_getVoxelBasedNMIGradientUsingPW1<PrecisionTYPE,char>
                (targetImage, resultImage, type, resultImageGradient, binning, logJointHistogram, entropies, nmiGradientImage, mask);
			break;
		case NIFTI_TYPE_UINT16:
			reg_getVoxelBasedNMIGradientUsingPW1<PrecisionTYPE,unsigned short>
                (targetImage, resultImage, type, resultImageGradient, binning, logJointHistogram, entropies, nmiGradientImage, mask);
			break;
		case NIFTI_TYPE_INT16:
			reg_getVoxelBasedNMIGradientUsingPW1<PrecisionTYPE,short>
                (targetImage, resultImage, type, resultImageGradient, binning, logJointHistogram, entropies, nmiGradientImage, mask);
			break;
		case NIFTI_TYPE_UINT32:
			reg_getVoxelBasedNMIGradientUsingPW1<PrecisionTYPE,unsigned int>
                (targetImage, resultImage, type, resultImageGradient, binning, logJointHistogram, entropies, nmiGradientImage, mask);
			break;
		case NIFTI_TYPE_INT32:
			reg_getVoxelBasedNMIGradientUsingPW1<PrecisionTYPE,int>
                (targetImage, resultImage, type, resultImageGradient, binning, logJointHistogram, entropies, nmiGradientImage, mask);
			break;
		case NIFTI_TYPE_FLOAT32:
			reg_getVoxelBasedNMIGradientUsingPW1<PrecisionTYPE,float>
                (targetImage, resultImage, type, resultImageGradient, binning, logJointHistogram, entropies, nmiGradientImage, mask);
			break;
		case NIFTI_TYPE_FLOAT64:
			reg_getVoxelBasedNMIGradientUsingPW1<PrecisionTYPE,double>
                (targetImage, resultImage, type, resultImageGradient, binning, logJointHistogram, entropies, nmiGradientImage, mask);
			break;
		default:
			printf("err\treg_getVoxelBasedNMIGradientUsingPW\tThe target image data type is not supported\n");
			return;
	}
}
/* *************************************************************** */
template void reg_getVoxelBasedNMIGradientUsingPW<float>(nifti_image *, nifti_image *, int, nifti_image *, int, float *, float *, nifti_image *, int *);
template void reg_getVoxelBasedNMIGradientUsingPW<double>(nifti_image *, nifti_image *,int, nifti_image *, int, double *, double *, nifti_image *, int *);
/* *************************************************************** */
/* *************************************************************** */

#endif
