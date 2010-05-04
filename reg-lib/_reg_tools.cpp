/*
 *  _reg_tools.h
 *  
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_TOOLS_CPP
#define _REG_TOOLS_CPP

#include "_reg_tools.h"

/* *************************************************************** */

// No round() function available in windows.
#ifdef _WINDOWS
template<class PrecisionType>
int round(PrecisionType x)
{
   return int(x > 0.0 ? x + 0.5 : x - 0.5);
}
#endif

template<class DTYPE>
void reg_intensityRescale2(	nifti_image *image,
			                float newMin,
			                float newMax,
                            float lowThr,
                            float upThr
			)
{
	DTYPE *imagePtr = static_cast<DTYPE *>(image->data);
	DTYPE currentMin=0;
	DTYPE currentMax=0;
	switch(image->datatype){
		case NIFTI_TYPE_UINT8:
			currentMin=(DTYPE)std::numeric_limits<unsigned char>::max();
			currentMax=0;
			break;
		case NIFTI_TYPE_INT8:
			currentMin=(DTYPE)std::numeric_limits<char>::max();
			currentMax=(DTYPE)-std::numeric_limits<char>::max();
			break;
		case NIFTI_TYPE_UINT16:
			currentMin=(DTYPE)std::numeric_limits<unsigned short>::max();
			currentMax=0;
			break;
		case NIFTI_TYPE_INT16:
			currentMin=(DTYPE)std::numeric_limits<char>::max();
			currentMax=-(DTYPE)std::numeric_limits<char>::max();
			break;
		case NIFTI_TYPE_UINT32:
			currentMin=(DTYPE)std::numeric_limits<unsigned int>::max();
			currentMax=0;
			break;
		case NIFTI_TYPE_INT32:
			currentMin=(DTYPE)std::numeric_limits<int>::max();
			currentMax=-(DTYPE)std::numeric_limits<int>::max();
			break;
		case NIFTI_TYPE_FLOAT32:
			currentMin=(DTYPE)std::numeric_limits<float>::max();
			currentMax=-(DTYPE)std::numeric_limits<float>::max();
			break;
		case NIFTI_TYPE_FLOAT64:
			currentMin=(DTYPE)std::numeric_limits<double>::max();
			currentMax=-(DTYPE)std::numeric_limits<double>::max();
			break;
	}

    if(image->scl_slope==0) image->scl_slope=1.0f;
	for(unsigned int index=0; index<image->nvox; index++){
		DTYPE value = (DTYPE)(*imagePtr++ * image->scl_slope + image->scl_inter);
        if(value==value){
		    currentMin=(currentMin<value)?currentMin:value;
		    currentMax=(currentMax>value)?currentMax:value;
        }
    }

    if(currentMin<lowThr) currentMin=(DTYPE)lowThr;
    if(currentMax>upThr) currentMax=(DTYPE)upThr;
	
	double currentDiff = (double)(currentMax-currentMin);
	double newDiff = (double)(newMax-newMin);

    image->cal_min=newMin * image->scl_slope + image->scl_inter;
    image->cal_max=newMax * image->scl_slope + image->scl_inter;

	imagePtr = static_cast<DTYPE *>(image->data);

	for(unsigned int index=0; index<image->nvox; index++){
		double value = (double)*imagePtr * image->scl_slope + image->scl_inter;
        if(value==value){
            if(value<currentMin){
                value = newMin;
            }
            else if(value>currentMax){
                value = newMax;
            }
            else{
		        value = (value-(double)currentMin)/currentDiff;
		        value = value * newDiff + newMin;
            }
        }
		*imagePtr++=(DTYPE)value;
	}
}
void reg_intensityRescale(	nifti_image *image,
				            float newMin,
				            float newMax,
                            float lowThr,
                            float upThr
			)
{
	switch(image->datatype){
		case NIFTI_TYPE_UINT8:
			reg_intensityRescale2<unsigned char>(image, newMin, newMax, lowThr, upThr);
			break;
		case NIFTI_TYPE_INT8:
			reg_intensityRescale2<char>(image, newMin, newMax, lowThr, upThr);
			break;
		case NIFTI_TYPE_UINT16:
			reg_intensityRescale2<unsigned short>(image, newMin, newMax, lowThr, upThr);
			break;
		case NIFTI_TYPE_INT16:
			reg_intensityRescale2<short>(image, newMin, newMax, lowThr, upThr);
			break;
		case NIFTI_TYPE_UINT32:
			reg_intensityRescale2<unsigned int>(image, newMin, newMax, lowThr, upThr);
			break;
		case NIFTI_TYPE_INT32:
			reg_intensityRescale2<int>(image, newMin, newMax, lowThr, upThr);
			break;
		case NIFTI_TYPE_FLOAT32:
			reg_intensityRescale2<float>(image, newMin, newMax, lowThr, upThr);
			break;
		case NIFTI_TYPE_FLOAT64:
			reg_intensityRescale2<double>(image, newMin, newMax, lowThr, upThr);
			break;
		default:
			printf("err\treg_intensityRescale\tThe image data type is not supported\n");
			return;
	}
}
/* *************************************************************** */
/* *************************************************************** */
template <class PrecisionTYPE, class DTYPE>
void reg_smoothImageForCubicSpline1(nifti_image *image,
								    int radius[]
								   )
{
	DTYPE *imageArray = static_cast<DTYPE *>(image->data);
	
	/* a temp image array is first created */
	DTYPE *tempArray  = (DTYPE *)malloc(image->nvox * sizeof(DTYPE));
	
	int timePoint = image->nt;
	if(timePoint==0) timePoint=1;
	int field = image->nu;
	if(field==0) field=1;
	
	/* Smoothing along the X axis */
	int windowSize = 2*radius[0] + 1;
	PrecisionTYPE *window = (PrecisionTYPE *)calloc(windowSize,sizeof(PrecisionTYPE));
//    PrecisionTYPE coeffSum=0.0;
	for(int it=-radius[0]; it<=radius[0]; it++){
		PrecisionTYPE coeff = (PrecisionTYPE)(fabs(2.0*(PrecisionTYPE)it/(PrecisionTYPE)radius[0]));
		if(coeff<1.0)	window[it+radius[0]] = (PrecisionTYPE)(2.0/3.0 - coeff*coeff + 0.5*coeff*coeff*coeff);
		else		window[it+radius[0]] = (PrecisionTYPE)(-(coeff-2.0)*(coeff-2.0)*(coeff-2.0)/6.0);
//        coeffSum += window[it+radius[0]];
	}
//	for(int it=0;it<windowSize;it++) window[it] /= coeffSum;
	for(int t=0;t<timePoint;t++){
		for(int u=0;u<field;u++){
			
			DTYPE *readingValue=&imageArray[(t+u*timePoint)*image->nx*image->ny*image->nz];
			DTYPE *writtingValue=&tempArray[(t+u*timePoint)*image->nx*image->ny*image->nz];
			int i=0;
			for(int z=0; z<image->nz; z++){
				for(int y=0; y<image->ny; y++){
					for(int x=0; x<image->nx; x++){
						
						PrecisionTYPE finalValue=0.0;
						
						int index = i - radius[0];
						int X = x - radius[0];
						
						for(int it=0; it<windowSize; it++){
							if(-1<X && X<image->nx){
								DTYPE imageValue = readingValue[index];
								PrecisionTYPE windowValue = window[it];
								finalValue += (PrecisionTYPE)imageValue * windowValue;
							}
							index++;
							X++;
						}
						
						writtingValue[i++] = (DTYPE)finalValue;
					}
				}
			}
		}
	}
	
	/* Smoothing along the Y axis */
	windowSize = 2*radius[1] + 1;
	free(window);
	window = (PrecisionTYPE *)calloc(windowSize,sizeof(PrecisionTYPE));
//    coeffSum=0.0;
	for(int it=-radius[1]; it<=radius[1]; it++){
		PrecisionTYPE coeff = (PrecisionTYPE)(fabs(2.0*(PrecisionTYPE)it/(PrecisionTYPE)radius[1]));
		if(coeff<1.0)	window[it+radius[1]] = (PrecisionTYPE)(2.0/3.0 - coeff*coeff + 0.5*coeff*coeff*coeff);
		else		window[it+radius[1]] = (PrecisionTYPE)(-(coeff-2.0)*(coeff-2.0)*(coeff-2.0)/6.0);
//        coeffSum += window[it+radius[1]];
	}
//    for(int it=0;it<windowSize;it++)window[it] /= coeffSum;
	for(int t=0;t<timePoint;t++){
		for(int u=0;u<field;u++){
			
			DTYPE *readingValue=&tempArray[(t+u*timePoint)*image->nx*image->ny*image->nz];
			DTYPE *writtingValue=&imageArray[(t+u*timePoint)*image->nx*image->ny*image->nz];
			int i=0;
			for(int z=0; z<image->nz; z++){
				for(int y=0; y<image->ny; y++){
					for(int x=0; x<image->nx; x++){
						
						PrecisionTYPE finalValue=0.0;
						
						int index = i - image->nx*radius[1];
						int Y = y - radius[1];
						
						for(int it=0; it<windowSize; it++){
							if(-1<Y && Y<image->ny){
								DTYPE imageValue = readingValue[index];
								PrecisionTYPE windowValue = window[it];
								finalValue += (PrecisionTYPE)imageValue * windowValue;
							}
							index+=image->nx;
							Y++;
						}
						
						writtingValue[i++] = (DTYPE)finalValue;
					}
				}
			}
		}
	}
	if(image->nz>1){
		/* Smoothing along the Z axis */
		windowSize = 2*radius[2] + 1;
		free(window);
		window = (PrecisionTYPE *)calloc(windowSize,sizeof(PrecisionTYPE));
//	    coeffSum=0.0;
		for(int it=-radius[2]; it<=radius[2]; it++){
			PrecisionTYPE coeff = (PrecisionTYPE)(fabs(2.0*(PrecisionTYPE)it/(PrecisionTYPE)radius[2]));
			if(coeff<1.0)	window[it+radius[2]] = (PrecisionTYPE)(2.0/3.0 - coeff*coeff + 0.5*coeff*coeff*coeff);
			else		window[it+radius[2]] = (PrecisionTYPE)(-(coeff-2.0)*(coeff-2.0)*(coeff-2.0)/6.0);
//			coeffSum += window[it+radius[2]];
		}
//	    for(int it=0;it<windowSize;it++)window[it] /= coeffSum;
		for(int t=0;t<timePoint;t++){
			for(int u=0;u<field;u++){
				
				DTYPE *readingValue=&imageArray[(t+u*timePoint)*image->nx*image->ny*image->nz];
				DTYPE *writtingValue=&tempArray[(t+u*timePoint)*image->nx*image->ny*image->nz];
				int i=0;
				for(int z=0; z<image->nz; z++){
					for(int y=0; y<image->ny; y++){
						for(int x=0; x<image->nx; x++){
							
							PrecisionTYPE finalValue=0.0;
							
							int index = i - image->nx*image->ny*radius[2];
							int Z = z - radius[2];
							
							for(int it=0; it<windowSize; it++){
								if(-1<Z && Z<image->nz){
									DTYPE imageValue = readingValue[index];
									PrecisionTYPE windowValue = window[it];
									finalValue += (PrecisionTYPE)imageValue * windowValue;
								}
								index+=image->nx*image->ny;
								Z++;
							}
							
							writtingValue[i++] = (DTYPE)finalValue;
						}
					}
				}
			}
		}
        memcpy(imageArray,tempArray,image->nvox * sizeof(DTYPE));
	}
	free(window);
	free(tempArray);
}
/* *************************************************************** */
template <class PrecisionTYPE>
void reg_smoothImageForCubicSpline(	nifti_image *image,
								  int radius[]
								  )
{
	switch(image->datatype){
		case NIFTI_TYPE_UINT8:
			reg_smoothImageForCubicSpline1<PrecisionTYPE,unsigned char>(image, radius);
			break;
		case NIFTI_TYPE_INT8:
			reg_smoothImageForCubicSpline1<PrecisionTYPE,char>(image, radius);
			break;
		case NIFTI_TYPE_UINT16:
			reg_smoothImageForCubicSpline1<PrecisionTYPE,unsigned short>(image, radius);
			break;
		case NIFTI_TYPE_INT16:
			reg_smoothImageForCubicSpline1<PrecisionTYPE,short>(image, radius);
			break;
		case NIFTI_TYPE_UINT32:
			reg_smoothImageForCubicSpline1<PrecisionTYPE,unsigned int>(image, radius);
			break;
		case NIFTI_TYPE_INT32:
			reg_smoothImageForCubicSpline1<PrecisionTYPE,int>(image, radius);
			break;
		case NIFTI_TYPE_FLOAT32:
			reg_smoothImageForCubicSpline1<PrecisionTYPE,float>(image, radius);
			break;
		case NIFTI_TYPE_FLOAT64:
			reg_smoothImageForCubicSpline1<PrecisionTYPE,double>(image, radius);
			break;
		default:
			printf("err\treg_smoothImage\tThe image data type is not supported\n");
			return;
	}
}
/* *************************************************************** */
template void reg_smoothImageForCubicSpline<float>(nifti_image *, int[]);
template void reg_smoothImageForCubicSpline<double>(nifti_image *, int[]);
/* *************************************************************** */
/* *************************************************************** */
template <class PrecisionTYPE, class DTYPE>
void reg_smoothImageForTrilinear1(	nifti_image *image,
								   int radius[]
								   )
{
	DTYPE *imageArray = static_cast<DTYPE *>(image->data);
	
	/* a temp image array is first created */
	DTYPE *tempArray  = (DTYPE *)malloc(image->nvox * sizeof(DTYPE));
	
	int timePoint = image->nt;
	if(timePoint==0) timePoint=1;
	int field = image->nu;
	if(field==0) field=1;
	
	/* Smoothing along the X axis */
	int windowSize = 2*radius[0] + 1;
	// 	printf("window size along X: %i\n", windowSize);
	PrecisionTYPE *window = (PrecisionTYPE *)calloc(windowSize,sizeof(PrecisionTYPE));
    PrecisionTYPE coeffSum=0.0;
	for(int it=-radius[0]; it<=radius[0]; it++){
		PrecisionTYPE coeff = (PrecisionTYPE)(fabs(1.0 -fabs((PrecisionTYPE)(it)/(PrecisionTYPE)radius[0] )));
		window[it+radius[0]] = coeff;
        coeffSum += coeff;
	}
    for(int it=0;it<windowSize;it++){
//printf("coeff[%i] = %g -> ", it, window[it]);
        window[it] /= coeffSum;
//printf("%g\n", window[it]);
    }
	for(int t=0;t<timePoint;t++){
		for(int u=0;u<field;u++){
			
			DTYPE *readingValue=&imageArray[(t+u*timePoint)*image->nx*image->ny*image->nz];
			DTYPE *writtingValue=&tempArray[(t+u*timePoint)*image->nx*image->ny*image->nz];
			int i=0;
			for(int z=0; z<image->nz; z++){
				for(int y=0; y<image->ny; y++){
					for(int x=0; x<image->nx; x++){
						
						PrecisionTYPE finalValue=0.0;
						
						int index = i - radius[0];
						int X = x - radius[0];
						
						for(int it=0; it<windowSize; it++){
							if(-1<X && X<image->nx){
								DTYPE imageValue = readingValue[index];
								PrecisionTYPE windowValue = window[it];
                                if(windowValue==windowValue)
								    finalValue += (PrecisionTYPE)imageValue * windowValue;
							}
							index++;
							X++;
						}
						
						writtingValue[i++] = (DTYPE)finalValue;
					}
				}
			}
		}
	}
	
	/* Smoothing along the Y axis */
	windowSize = 2*radius[1] + 1;
	// 	printf("window size along Y: %i\n", windowSize);
	free(window);
	window = (PrecisionTYPE *)calloc(windowSize,sizeof(PrecisionTYPE));
    coeffSum=0.0;
	for(int it=-radius[1]; it<=radius[1]; it++){
		PrecisionTYPE coeff = (PrecisionTYPE)(fabs(1.0 -fabs((PrecisionTYPE)(it)/(PrecisionTYPE)radius[0] )));
		window[it+radius[1]] = coeff;
        coeffSum += coeff;
	}
    for(int it=0;it<windowSize;it++) window[it] /= coeffSum;
	for(int t=0;t<timePoint;t++){
		for(int u=0;u<field;u++){
			
			DTYPE *readingValue=&tempArray[(t+u*timePoint)*image->nx*image->ny*image->nz];
			DTYPE *writtingValue=&imageArray[(t+u*timePoint)*image->nx*image->ny*image->nz];
			int i=0;
			for(int z=0; z<image->nz; z++){
				for(int y=0; y<image->ny; y++){
					for(int x=0; x<image->nx; x++){
						
						PrecisionTYPE finalValue=0.0;
						
						int index = i - image->nx*radius[1];
						int Y = y - radius[1];
						
						for(int it=0; it<windowSize; it++){
							if(-1<Y && Y<image->ny){
								DTYPE imageValue = readingValue[index];
								PrecisionTYPE windowValue = window[it];
                                if(windowValue==windowValue)
								    finalValue += (PrecisionTYPE)imageValue * windowValue;
							}
							index+=image->nx;
							Y++;
						}
						
						writtingValue[i++] = (DTYPE)finalValue;
					}
				}
			}
		}
	}
	
	/* Smoothing along the Z axis */
	windowSize = 2*radius[2] + 1;
	// 	printf("window size along Z: %i\n", windowSize);
	free(window);
	window = (PrecisionTYPE *)calloc(windowSize,sizeof(PrecisionTYPE));
    coeffSum=0.0;
	for(int it=-radius[2]; it<=radius[2]; it++){
		PrecisionTYPE coeff = (PrecisionTYPE)(fabs(1.0 -fabs((PrecisionTYPE)(it)/(PrecisionTYPE)radius[0] )));
		window[it+radius[2]] = coeff;
        coeffSum += coeff;
	}
    for(int it=0;it<windowSize;it++) window[it] /= coeffSum;
	for(int t=0;t<timePoint;t++){
		for(int u=0;u<field;u++){
			
			DTYPE *readingValue=&imageArray[(t+u*timePoint)*image->nx*image->ny*image->nz];
			DTYPE *writtingValue=&tempArray[(t+u*timePoint)*image->nx*image->ny*image->nz];
			int i=0;
			for(int z=0; z<image->nz; z++){
				for(int y=0; y<image->ny; y++){
					for(int x=0; x<image->nx; x++){
						
						PrecisionTYPE finalValue=0.0;
						
						int index = i - image->nx*image->ny*radius[2];
						int Z = z - radius[2];
						
						for(int it=0; it<windowSize; it++){
							if(-1<Z && Z<image->nz){
								DTYPE imageValue = readingValue[index];
								PrecisionTYPE windowValue = window[it];
                                if(windowValue==windowValue)
								    finalValue += (PrecisionTYPE)imageValue * windowValue;
							}
							index+=image->nx*image->ny;
							Z++;
						}
						
						writtingValue[i++] = (DTYPE)finalValue;
					}
				}
			}
		}
	}
	free(window);
	memcpy(imageArray,tempArray,image->nvox * sizeof(DTYPE));
	free(tempArray);
}
/* *************************************************************** */
template <class PrecisionTYPE>
void reg_smoothImageForTrilinear(	nifti_image *image,
								  int radius[]
								  )
{
	switch(image->datatype){
		case NIFTI_TYPE_UINT8:
			reg_smoothImageForTrilinear1<PrecisionTYPE,unsigned char>(image, radius);
			break;
		case NIFTI_TYPE_INT8:
			reg_smoothImageForTrilinear1<PrecisionTYPE,char>(image, radius);
			break;
		case NIFTI_TYPE_UINT16:
			reg_smoothImageForTrilinear1<PrecisionTYPE,unsigned short>(image, radius);
			break;
		case NIFTI_TYPE_INT16:
			reg_smoothImageForTrilinear1<PrecisionTYPE,short>(image, radius);
			break;
		case NIFTI_TYPE_UINT32:
			reg_smoothImageForCubicSpline1<PrecisionTYPE,unsigned int>(image, radius);
			break;
		case NIFTI_TYPE_INT32:
			reg_smoothImageForTrilinear1<PrecisionTYPE,int>(image, radius);
			break;
		case NIFTI_TYPE_FLOAT32:
			reg_smoothImageForTrilinear1<PrecisionTYPE,float>(image, radius);
			break;
		case NIFTI_TYPE_FLOAT64:
			reg_smoothImageForTrilinear1<PrecisionTYPE,double>(image, radius);
			break;
		default:
			printf("err\treg_smoothImage\tThe image data type is not supported\n");
			return;
	}
}
/* *************************************************************** */
template void reg_smoothImageForTrilinear<float>(nifti_image *, int[]);
template void reg_smoothImageForTrilinear<double>(nifti_image *, int[]);
/* *************************************************************** */
/* *************************************************************** */
template <class PrecisionTYPE, class DTYPE>
PrecisionTYPE reg_getMaximalLength2D(nifti_image *image)
{
	DTYPE *dataPtrX = static_cast<DTYPE *>(image->data);
	DTYPE *dataPtrY = &dataPtrX[image->nx*image->ny*image->nz];
	
	PrecisionTYPE max=0.0;
	
	for(int i=0; i<image->nx*image->ny*image->nz; i++){
		PrecisionTYPE valX = (PrecisionTYPE)(*dataPtrX++);
		PrecisionTYPE valY = (PrecisionTYPE)(*dataPtrY++);
		PrecisionTYPE length = (PrecisionTYPE)(sqrt(valX*valX + valY*valY));
		max = (length>max)?length:max;
	}
	return max;
}
/* *************************************************************** */
template <class PrecisionTYPE, class DTYPE>
PrecisionTYPE reg_getMaximalLength3D(nifti_image *image)
{
	DTYPE *dataPtrX = static_cast<DTYPE *>(image->data);
	DTYPE *dataPtrY = &dataPtrX[image->nx*image->ny*image->nz];
	DTYPE *dataPtrZ = &dataPtrY[image->nx*image->ny*image->nz];
	
	PrecisionTYPE max=0.0;
	
	for(int i=0; i<image->nx*image->ny*image->nz; i++){
		PrecisionTYPE valX = (PrecisionTYPE)(*dataPtrX++);
		PrecisionTYPE valY = (PrecisionTYPE)(*dataPtrY++);
		PrecisionTYPE valZ = (PrecisionTYPE)(*dataPtrZ++);
		PrecisionTYPE length = (PrecisionTYPE)(sqrt(valX*valX + valY*valY + valZ*valZ));
		max = (length>max)?length:max;
	}
	return max;
}
/* *************************************************************** */
template <class PrecisionTYPE>
		PrecisionTYPE reg_getMaximalLength(nifti_image *image)
{
	if(image->nz==1){
		switch(image->datatype){
			case NIFTI_TYPE_FLOAT32:
				return reg_getMaximalLength2D<PrecisionTYPE,float>(image);
				break;
			case NIFTI_TYPE_FLOAT64:
				return reg_getMaximalLength2D<PrecisionTYPE,double>(image);
				break;
		}
	}
	else{
		switch(image->datatype){
			case NIFTI_TYPE_FLOAT32:
				return reg_getMaximalLength3D<PrecisionTYPE,float>(image);
				break;
			case NIFTI_TYPE_FLOAT64:
				return reg_getMaximalLength3D<PrecisionTYPE,double>(image);
				break;
		}
	}
	return 0;
}
/* *************************************************************** */
template float reg_getMaximalLength<float>(nifti_image *);
template double reg_getMaximalLength<double>(nifti_image *);
/* *************************************************************** */
/* *************************************************************** */
template <class NewTYPE, class DTYPE>
void reg_changeDatatype1(nifti_image *image)
{
	// the initial array is saved and freeed
	DTYPE *initialValue = (DTYPE *)malloc(image->nvox*sizeof(DTYPE));
	memcpy(initialValue, image->data, image->nvox*sizeof(DTYPE));
	
	// the new array is allocated and then filled
	if(sizeof(NewTYPE)==sizeof(unsigned char)) image->datatype = NIFTI_TYPE_UINT8;
	else if(sizeof(NewTYPE)==sizeof(float)) image->datatype = NIFTI_TYPE_FLOAT32;
	else if(sizeof(NewTYPE)==sizeof(double)) image->datatype = NIFTI_TYPE_FLOAT64;
	else{
		printf("err\treg_changeDatatype\tOnly change to unsigned char, float or double are supported\n");
		free(initialValue);
		return;
	}
	free(image->data);
	image->nbyper = sizeof(NewTYPE);
	image->data = (void *)calloc(image->nvox,sizeof(NewTYPE));
	NewTYPE *dataPtr = static_cast<NewTYPE *>(image->data);
	for(unsigned int i=0; i<image->nvox; i++) dataPtr[i] = (NewTYPE)(initialValue[i]);

	free(initialValue);
	return;
}
/* *************************************************************** */
template <class NewTYPE>
void reg_changeDatatype(nifti_image *image)
{
	switch(image->datatype){
		case NIFTI_TYPE_UINT8:
			reg_changeDatatype1<NewTYPE,unsigned char>(image);
			break;
		case NIFTI_TYPE_INT8:
			reg_changeDatatype1<NewTYPE,char>(image);
			break;
		case NIFTI_TYPE_UINT16:
			reg_changeDatatype1<NewTYPE,unsigned short>(image);
			break;
		case NIFTI_TYPE_INT16:
			reg_changeDatatype1<NewTYPE,short>(image);
			break;
		case NIFTI_TYPE_UINT32:
			reg_changeDatatype1<NewTYPE,unsigned int>(image);
			break;
		case NIFTI_TYPE_INT32:
			reg_changeDatatype1<NewTYPE,int>(image);
			break;
		case NIFTI_TYPE_FLOAT32:
			reg_changeDatatype1<NewTYPE,float>(image);
			break;
		case NIFTI_TYPE_FLOAT64:
			reg_changeDatatype1<NewTYPE,double>(image);
			break;
		default:
			printf("err\treg_changeDatatype\tThe initial image data type is not supported\n");
			return;
	}
}
/* *************************************************************** */
template void reg_changeDatatype<unsigned char>(nifti_image *);
template void reg_changeDatatype<float>(nifti_image *);
template void reg_changeDatatype<double>(nifti_image *);
/* *************************************************************** */
/* *************************************************************** */
template <class ImageTYPE>
double reg_tool_GetIntensityValue1(nifti_image *image,
								 int *index)
{
	ImageTYPE *imgPtr = static_cast<ImageTYPE *>(image->data);
	return (double) imgPtr[(index[2]*image->ny+index[1])*image->nx+index[0]];
}
/* *************************************************************** */
double reg_tool_GetIntensityValue(nifti_image *image,
								  int *index)
{
	switch(image->datatype){
		case NIFTI_TYPE_UINT8:
			return reg_tool_GetIntensityValue1<unsigned char>(image, index);
			break;
		case NIFTI_TYPE_INT8:
			return reg_tool_GetIntensityValue1<char>(image, index);
			break;
		case NIFTI_TYPE_UINT16:
			return reg_tool_GetIntensityValue1<unsigned short>(image, index);
			break;
		case NIFTI_TYPE_INT16:
			reg_tool_GetIntensityValue1<short>(image, index);
			break;
		case NIFTI_TYPE_UINT32:
			return reg_tool_GetIntensityValue1<unsigned int>(image, index);
			break;
		case NIFTI_TYPE_INT32:
			return reg_tool_GetIntensityValue1<int>(image, index);
			break;
		case NIFTI_TYPE_FLOAT32:
			return reg_tool_GetIntensityValue1<float>(image, index);
			break;
		case NIFTI_TYPE_FLOAT64:
			return reg_tool_GetIntensityValue1<double>(image, index);
			break;
		default:
			printf("err\treg_changeDatatype\tThe initial image data type is not supported\n");
			break;
	}
	
	return 0.0;
}
/* *************************************************************** */
/* *************************************************************** */
template <class TYPE1, class TYPE2>
void reg_tools_addSubMulDivImages2( nifti_image *img1,
                                    nifti_image *img2,
                                    nifti_image *res,
                                    int type)
{
    TYPE1 *img1Ptr = static_cast<TYPE1 *>(img1->data);
    TYPE2 *img2Ptr = static_cast<TYPE2 *>(img2->data);
    TYPE1 *resPtr = static_cast<TYPE1 *>(res->data);
    switch(type){
        case 0:
            for(unsigned int i=0; i<res->nvox; i++)
                *resPtr++ = (TYPE1)(*img1Ptr++ + *img2Ptr++);
            break;
        case 1:
            for(unsigned int i=0; i<res->nvox; i++)
                *resPtr++ = (TYPE1)(*img1Ptr++ - *img2Ptr++);
            break;
        case 2:
            for(unsigned int i=0; i<res->nvox; i++)
                *resPtr++ = (TYPE1)(*img1Ptr++ * *img2Ptr++);
            break;
        case 3:
            for(unsigned int i=0; i<res->nvox; i++)
                *resPtr++ = (TYPE1)(*img1Ptr++ / *img2Ptr++);
            break;
    }
}
/* *************************************************************** */
template <class TYPE1>
void reg_tools_addSubMulDivImages1( nifti_image *img1,
                                    nifti_image *img2,
                                    nifti_image *res,
                                    int type)
{
    switch(img1->datatype){
        case NIFTI_TYPE_UINT8:
            reg_tools_addSubMulDivImages2<TYPE1,unsigned char>(img1, img2, res, type);
            break;
        case NIFTI_TYPE_INT8:
            reg_tools_addSubMulDivImages2<TYPE1,char>(img1, img2, res, type);
            break;
        case NIFTI_TYPE_UINT16:
            reg_tools_addSubMulDivImages2<TYPE1,unsigned short>(img1, img2, res, type);
            break;
        case NIFTI_TYPE_INT16:
            reg_tools_addSubMulDivImages2<TYPE1,short>(img1, img2, res, type);
            break;
        case NIFTI_TYPE_UINT32:
            reg_tools_addSubMulDivImages2<TYPE1,unsigned int>(img1, img2, res, type);
            break;
        case NIFTI_TYPE_INT32:
            reg_tools_addSubMulDivImages2<TYPE1,int>(img1, img2, res, type);
            break;
        case NIFTI_TYPE_FLOAT32:
            reg_tools_addSubMulDivImages2<TYPE1,float>(img1, img2, res, type);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_tools_addSubMulDivImages2<TYPE1,double>(img1, img2, res, type);
            break;
        default:
            fprintf(stderr,"err\treg_tools_addSubMulDivImages1\tSecond image data type is not supported\n");
            return;
    }
}
/* *************************************************************** */
void reg_tools_addSubMulDivImages(  nifti_image *img1,
                                    nifti_image *img2,
                                    nifti_image *res,
                                    int type)
{
    
    if(img1->dim[0]!=img2->dim[0] ||
       img1->dim[1]!=img2->dim[1] ||
       img1->dim[2]!=img2->dim[2] ||
       img1->dim[3]!=img2->dim[3] ||
       img1->dim[4]!=img2->dim[4] ||
       img1->dim[5]!=img2->dim[5] ||
       img1->dim[6]!=img2->dim[6] ||
       img1->dim[7]!=img2->dim[7]){
        fprintf(stderr,"err\treg_tools_addSubMulDivImages\tBoth images do not have the same dimension\n");
        return;
    }

    if(img1->datatype != res->datatype){
        fprintf(stderr,"err\treg_tools_addSubMulDivImages\tFirst and result image do not have the same data type\n");
        return;
    }
    switch(img1->datatype){
        case NIFTI_TYPE_UINT8:
            reg_tools_addSubMulDivImages1<unsigned char>(img1, img2, res, type);
            break;
        case NIFTI_TYPE_INT8:
            reg_tools_addSubMulDivImages1<char>(img1, img1, res, type);
            break;
        case NIFTI_TYPE_UINT16:
            reg_tools_addSubMulDivImages1<unsigned short>(img1, img2, res, type);
            break;
        case NIFTI_TYPE_INT16:
            reg_tools_addSubMulDivImages1<short>(img1, img2, res, type);
            break;
        case NIFTI_TYPE_UINT32:
            reg_tools_addSubMulDivImages1<unsigned int>(img1, img2, res, type);
            break;
        case NIFTI_TYPE_INT32:
            reg_tools_addSubMulDivImages1<int>(img1, img2, res, type);
            break;
        case NIFTI_TYPE_FLOAT32:
            reg_tools_addSubMulDivImages1<float>(img1, img2, res, type);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_tools_addSubMulDivImages1<double>(img1, img2, res, type);
            break;
        default:
            fprintf(stderr,"err\treg_tools_addSubMulDivImages1\tFirst image data type is not supported\n");
            return;
    }
}
/* *************************************************************** */
/* *************************************************************** */
template <class TYPE1>
void reg_tools_addSubMulDivValue1(  nifti_image *img1,
                                    nifti_image *res,
                                    float val,
                                    int type)
{
    TYPE1 *img1Ptr = static_cast<TYPE1 *>(img1->data);
    TYPE1 *resPtr = static_cast<TYPE1 *>(res->data);
    switch(type){
        case 0:
            printf("+ %g\n",val);
            for(unsigned int i=0; i<res->nvox; i++)
                *resPtr++ = (TYPE1)(*img1Ptr++ + val);
            break;
        case 1:
            printf("- %g\n",val);
            for(unsigned int i=0; i<res->nvox; i++)
                *resPtr++ = (TYPE1)(*img1Ptr++ - val);
            break;
        case 2:
            printf("* %g\n",val);
            for(unsigned int i=0; i<res->nvox; i++)
                *resPtr++ = (TYPE1)(*img1Ptr++ * val);
            break;
        case 3:
            printf("/ %g\n",val);
            for(unsigned int i=0; i<res->nvox; i++)
                *resPtr++ = (TYPE1)(*img1Ptr++ / val);
            break;
    }
}
/* *************************************************************** */
void reg_tools_addSubMulDivValue(   nifti_image *img1,
                                    nifti_image *res,
                                    float val,
                                    int type)
{
    if(img1->datatype != res->datatype){
        fprintf(stderr,"err\treg_tools_addSubMulDivValue\tInput and result image do not have the same data type\n");
        return;
    }
    switch(img1->datatype){
        case NIFTI_TYPE_UINT8:
            reg_tools_addSubMulDivValue1<unsigned char>
                (img1, res, val, type);
            break;
        case NIFTI_TYPE_INT8:
            reg_tools_addSubMulDivValue1<char>
                (img1, res, val, type);
            break;
        case NIFTI_TYPE_UINT16:
            reg_tools_addSubMulDivValue1<unsigned short>
                (img1, res, val, type);
            break;
        case NIFTI_TYPE_INT16:
            reg_tools_addSubMulDivValue1<short>
                (img1, res, val, type);
            break;
        case NIFTI_TYPE_UINT32:
            reg_tools_addSubMulDivValue1<unsigned int>
                (img1, res, val, type);
            break;
        case NIFTI_TYPE_INT32:
            reg_tools_addSubMulDivValue1<int>
                (img1, res, val, type);
            break;
        case NIFTI_TYPE_FLOAT32:
            reg_tools_addSubMulDivValue1<float>
                (img1, res, val, type);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_tools_addSubMulDivValue1<double>
                (img1, res, val, type);
            break;
        default:
            fprintf(stderr,"err\treg_tools_addSubMulDivImages1\tFirst image data type is not supported\n");
            return;
    }
}
/* *************************************************************** */
/* *************************************************************** */
template <class PrecisionTYPE, class ImageTYPE>
void reg_gaussianSmoothing1(nifti_image *image,
			                float sigma,
                            bool axisToSmooth[8])
{
	ImageTYPE *imagePtr = static_cast<ImageTYPE *>(image->data);

    int timePoint = image->nt;
    if(timePoint==0) timePoint=1;
    int field = image->nu;
    if(field==0) field=1;

    unsigned int voxelNumber = image->nx*image->ny*image->nz;

    for(int t=0; t<timePoint*field; t++){
        ImageTYPE *timeImagePtr = &imagePtr[t * voxelNumber];
        PrecisionTYPE *resultValue=(PrecisionTYPE *)malloc(voxelNumber * sizeof(PrecisionTYPE));
	    for(int n=1; n<4; n++){
            if(axisToSmooth[n]==true && image->dim[n]>1){
		        float currentSigma;
		        if(sigma>0) currentSigma=sigma/image->pixdim[n];
		        else currentSigma=fabs(sigma); // voxel based if negative value
		        int radius=(int)ceil(currentSigma*3.0f);
                if(radius>0){
		            PrecisionTYPE *kernel = new PrecisionTYPE[2*radius+1];
		            PrecisionTYPE kernelSum=0;
		            for(int i=-radius; i<=radius; i++){
			            kernel[radius+i]=(PrecisionTYPE)(exp( -(i*i)/(2.0*currentSigma*currentSigma)) / (currentSigma*2.506628274631));
                        // 2.506... = sqrt(2*pi)
			            kernelSum += kernel[radius+i];
		            }
		            for(int i=-radius; i<=radius; i++) kernel[radius+i] /= kernelSum;
#ifndef NDEBUG
		            printf("[DEBUG] smoothing dim[%i] radius[%i] kernelSum[%g]\n", n, radius, kernelSum);
#endif
		            int increment=1;
		            switch(n){
			            case 1: increment=1;break;
			            case 2: increment=image->nx;break;
			            case 3: increment=image->nx*image->ny;break;
			            case 4: increment=image->nx*image->ny*image->nz;break;
			            case 5: increment=image->nx*image->ny*image->nz*image->nt;break;
			            case 6: increment=image->nx*image->ny*image->nz*image->nt*image->nu;break;
			            case 7: increment=image->nx*image->ny*image->nz*image->nt*image->nu*image->nv;break;
		            }
		            unsigned int index=0;
		            while(index<voxelNumber){
			            for(int x=0; x<image->dim[n]; x++){
				            int current = index - increment*radius;
				            PrecisionTYPE value=0;
				            for(int j=-radius; j<=radius; j++){
					            if(-1<current && current<(int)voxelNumber){
                                    if(timeImagePtr[current]==timeImagePtr[current])
    						            value += (PrecisionTYPE)(timeImagePtr[current]*kernel[j+radius]);
					            }
					            current += increment;
				            }
				            resultValue[index]=value;
				            index++;
			            }
		            }
		            for(unsigned int i=0; i<voxelNumber; i++) timeImagePtr[i]=(ImageTYPE)resultValue[i];
		            delete[] kernel;
                }
            }
	    }
        free(resultValue);
    }
}
/* *************************************************************** */
template <class PrecisionTYPE>
void reg_gaussianSmoothing(	nifti_image *image,
							float sigma,
                            bool smoothXYZ[8])
{
    bool axisToSmooth[8];
    if(smoothXYZ==NULL){
        for(int i=0; i<8; i++) axisToSmooth[i]=true;
    }
    else{
        for(int i=0; i<8; i++) axisToSmooth[i]=smoothXYZ[i];
    }

	if(sigma==0.0) return;
	switch(image->datatype){
		case NIFTI_TYPE_UINT8:
			reg_gaussianSmoothing1<PrecisionTYPE,unsigned char>(image, sigma, axisToSmooth);
			break;
		case NIFTI_TYPE_INT8:
			reg_gaussianSmoothing1<PrecisionTYPE,char>(image, sigma, axisToSmooth);
			break;
		case NIFTI_TYPE_UINT16:
			reg_gaussianSmoothing1<PrecisionTYPE,unsigned short>(image, sigma, axisToSmooth);
			break;
		case NIFTI_TYPE_INT16:
			reg_gaussianSmoothing1<PrecisionTYPE,short>(image, sigma, axisToSmooth);
			break;
		case NIFTI_TYPE_UINT32:
			reg_gaussianSmoothing1<PrecisionTYPE,unsigned int>(image, sigma, axisToSmooth);
			break;
		case NIFTI_TYPE_INT32:
			reg_gaussianSmoothing1<PrecisionTYPE,int>(image, sigma, axisToSmooth);
			break;
		case NIFTI_TYPE_FLOAT32:
			reg_gaussianSmoothing1<PrecisionTYPE,float>(image, sigma, axisToSmooth);
			break;
		case NIFTI_TYPE_FLOAT64:
			reg_gaussianSmoothing1<PrecisionTYPE,double>(image, sigma, axisToSmooth);
			break;
		default:
			printf("err\treg_smoothImage\tThe image data type is not supported\n");
			return;
	}
}
template void reg_gaussianSmoothing<float>(nifti_image *, float, bool[8]);
template void reg_gaussianSmoothing<double>(nifti_image *, float, bool[8]);
/* *************************************************************** */
/* *************************************************************** */
template <class PrecisionTYPE, class ImageTYPE>
void reg_downsampleImage1(nifti_image *image, int type, bool downsampleAxis[8])
{
    if(type==1){
	    /* the input image is first smooth */
	    reg_gaussianSmoothing<float>(image, -0.7f, downsampleAxis);
    }

	/* the values are copied */
	ImageTYPE *oldValues = (ImageTYPE *)malloc(image->nvox * image->nbyper);
	ImageTYPE *imagePtr = static_cast<ImageTYPE *>(image->data);
	memcpy(oldValues, imagePtr, image->nvox*image->nbyper);
	free(image->data);

	int oldDim[4];
	for(int i=1; i<4; i++){
		oldDim[i]=image->dim[i];
		if(image->dim[i]>1 && downsampleAxis[i]==true) image->dim[i]=(int)(image->dim[i]/2.0);
		if(image->pixdim[i]>0 && downsampleAxis[i]==true) image->pixdim[i]=image->pixdim[i]*2.0f;
	}
	image->nx=image->dim[1];
	image->ny=image->dim[2];
	image->nz=image->dim[3];
	image->dx=image->pixdim[1];
	image->dy=image->pixdim[2];
	image->dz=image->pixdim[3];
	if(image->nt<=0 || image->dim[4]<=0) image->nt=image->dim[4]=1;
	if(image->nu<=0 || image->dim[5]<=0) image->nu=image->dim[5]=1;
	if(image->nv<=0 || image->dim[6]<=0) image->nv=image->dim[6]=1;
	if(image->nw<=0 || image->dim[7]<=0) image->nw=image->dim[7]=1;
	mat44 oldMat; memset(&oldMat, 0, sizeof(mat44));
	for(int i=0; i<3; i++){
		for(int j=0; j<3; j++){
			oldMat.m[i][j]=image->qto_ijk.m[i][j];
            if(downsampleAxis[j+1]==true){
			    image->qto_xyz.m[i][j]=image->qto_xyz.m[i][j]*2.0f;
			    image->sto_xyz.m[i][j]=image->sto_xyz.m[i][j]*2.0f;
            }
		}
	}
	oldMat.m[0][3]=image->qto_ijk.m[0][3];
	oldMat.m[1][3]=image->qto_ijk.m[1][3];
	oldMat.m[2][3]=image->qto_ijk.m[2][3];
	oldMat.m[3][3]=image->qto_ijk.m[3][3];
	
	image->qto_ijk = nifti_mat44_inverse(image->qto_xyz);
	image->sto_ijk = nifti_mat44_inverse(image->sto_xyz);

	image->nvox=image->nx*image->ny*image->nz*image->nt*image->nu*image->nv*image->nw;
	
	image->data=(void *)calloc(image->nvox, image->nbyper);
	imagePtr = static_cast<ImageTYPE *>(image->data);

	PrecisionTYPE real[3], position[3], relative, xBasis[2], yBasis[2], zBasis[2], intensity;
	int previous[3];

	for(int tuvw=0; tuvw<image->nt*image->nu*image->nv*image->nw; tuvw++){
		ImageTYPE *valuesPtrTUVW = &oldValues[tuvw*oldDim[1]*oldDim[2]*oldDim[3]];
		for(int z=0; z<image->nz; z++){
			for(int y=0; y<image->ny; y++){
				for(int x=0; x<image->nx; x++){
					real[0]=x*image->qto_xyz.m[0][0] + y*image->qto_xyz.m[0][1] + z*image->qto_xyz.m[0][2] + image->qto_xyz.m[0][3];
					real[1]=x*image->qto_xyz.m[1][0] + y*image->qto_xyz.m[1][1] + z*image->qto_xyz.m[1][2] + image->qto_xyz.m[1][3];
					real[2]=x*image->qto_xyz.m[2][0] + y*image->qto_xyz.m[2][1] + z*image->qto_xyz.m[2][2] + image->qto_xyz.m[2][3];
					position[0]=real[0]*oldMat.m[0][0] + real[1]*oldMat.m[0][1] + real[2]*oldMat.m[0][2] + oldMat.m[0][3];
					position[1]=real[0]*oldMat.m[1][0] + real[1]*oldMat.m[1][1] + real[2]*oldMat.m[1][2] + oldMat.m[1][3];
					position[2]=real[0]*oldMat.m[2][0] + real[1]*oldMat.m[2][1] + real[2]*oldMat.m[2][2] + oldMat.m[2][3];
					/* trilinear interpolation */
					previous[0] = (int)floor(position[0]);
					previous[1] = (int)floor(position[1]);
					previous[2] = (int)floor(position[2]);
					
					// basis values along the x axis
					relative=position[0]-(PrecisionTYPE)previous[0];
					if(relative<0) relative=0.0; // rounding error correction
					xBasis[0]= (PrecisionTYPE)(1.0-relative);
					xBasis[1]= relative;
					// basis values along the y axis
					relative=position[1]-(PrecisionTYPE)previous[1];
					if(relative<0) relative=0.0; // rounding error correction
					yBasis[0]= (PrecisionTYPE)(1.0-relative);
					yBasis[1]= relative;
					// basis values along the z axis
					relative=position[2]-(PrecisionTYPE)previous[2];
					if(relative<0) relative=0.0; // rounding error correction
					zBasis[0]= (PrecisionTYPE)(1.0-relative);
					zBasis[1]= relative;
					intensity=0;
					for(short c=0; c<2; c++){
						short Z= previous[2]+c;
						if(-1<Z && Z<oldDim[3]){
							ImageTYPE *zPointer = &valuesPtrTUVW[Z*oldDim[1]*oldDim[2]];
							PrecisionTYPE yTempNewValue=0.0;
							for(short b=0; b<2; b++){
								short Y= previous[1]+b;
								if(-1<Y && Y<oldDim[2]){
									ImageTYPE *yzPointer = &zPointer[Y*oldDim[1]];
									ImageTYPE *xyzPointer = &yzPointer[previous[0]];
									PrecisionTYPE xTempNewValue=0.0;
									for(short a=0; a<2; a++){
										if(-1<(previous[0]+a) && (previous[0]+a)<oldDim[1]){
											const ImageTYPE coeff = *xyzPointer;
											xTempNewValue +=  (PrecisionTYPE)(coeff * xBasis[a]);
										}
										xyzPointer++;
									}
									yTempNewValue += (xTempNewValue * yBasis[b]);
								}
							}
							intensity += yTempNewValue * zBasis[c];
						}
					}
                    switch(image->datatype){
                        case NIFTI_TYPE_FLOAT32:
                            (*imagePtr)=(ImageTYPE)intensity;
                            break;
                        case NIFTI_TYPE_FLOAT64:
                            (*imagePtr)=(ImageTYPE)intensity;
                            break;
                        case NIFTI_TYPE_UINT8:
                            (*imagePtr)=(ImageTYPE)(intensity>0?round(intensity):0);
                            break;
                        case NIFTI_TYPE_UINT16:
                            (*imagePtr)=(ImageTYPE)(intensity>0?round(intensity):0);
                            break;
                        case NIFTI_TYPE_UINT32:
                            (*imagePtr)=(ImageTYPE)(intensity>0?round(intensity):0);
                            break;
                        default:
                            (*imagePtr)=(ImageTYPE)round(intensity);
                            break;
                    }
                    imagePtr++;
				}
			}
		}
	}

	free(oldValues);
}
/* *************************************************************** */
template <class PrecisionTYPE>
void reg_downsampleImage(nifti_image *image, int type, bool downsampleAxis[8])
{
	switch(image->datatype){
		case NIFTI_TYPE_UINT8:
			reg_downsampleImage1<PrecisionTYPE,unsigned char>(image, type, downsampleAxis);
			break;
		case NIFTI_TYPE_INT8:
			reg_downsampleImage1<PrecisionTYPE,char>(image, type, downsampleAxis);
			break;
		case NIFTI_TYPE_UINT16:
			reg_downsampleImage1<PrecisionTYPE,unsigned short>(image, type, downsampleAxis);
			break;
		case NIFTI_TYPE_INT16:
			reg_downsampleImage1<PrecisionTYPE,short>(image, type, downsampleAxis);
			break;
		case NIFTI_TYPE_UINT32:
			reg_downsampleImage1<PrecisionTYPE,unsigned int>(image, type, downsampleAxis);
			break;
		case NIFTI_TYPE_INT32:
			reg_downsampleImage1<PrecisionTYPE,int>(image, type, downsampleAxis);
			break;
		case NIFTI_TYPE_FLOAT32:
			reg_downsampleImage1<PrecisionTYPE,float>(image, type, downsampleAxis);
			break;
		case NIFTI_TYPE_FLOAT64:
			reg_downsampleImage1<PrecisionTYPE,double>(image, type, downsampleAxis);
			break;
		default:
			printf("err\treg_downsampleImage\tThe image data type is not supported\n");
			return;
	}
}
template void reg_downsampleImage<float>(nifti_image *, int, bool[8]);
template void reg_downsampleImage<double>(nifti_image *, int, bool[8]);
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_tool_binarise_image1(nifti_image *image)
{
    DTYPE *dataPtr=static_cast<DTYPE *>(image->data);
    for(unsigned i=0; i<image->nvox; i++){
        *dataPtr = (*dataPtr)!=0?(DTYPE)1:(DTYPE)0;
        dataPtr++;
    }
}
/* *************************************************************** */
void reg_tool_binarise_image(nifti_image *image)
{
    switch(image->datatype){
        case NIFTI_TYPE_UINT8:
            reg_tool_binarise_image1<unsigned char>(image);
            break;
        case NIFTI_TYPE_INT8:
            reg_tool_binarise_image1<char>(image);
            break;
        case NIFTI_TYPE_UINT16:
            reg_tool_binarise_image1<unsigned short>(image);
            break;
        case NIFTI_TYPE_INT16:
            reg_tool_binarise_image1<short>(image);
            break;
        case NIFTI_TYPE_UINT32:
            reg_tool_binarise_image1<unsigned int>(image);
            break;
        case NIFTI_TYPE_INT32:
            reg_tool_binarise_image1<int>(image);
            break;
        case NIFTI_TYPE_FLOAT32:
            reg_tool_binarise_image1<float>(image);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_tool_binarise_image1<double>(image);
            break;
        default:
            printf("err\treg_tool_binarise_image\tThe image data type is not supported\n");
            return;
    }
}

/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_tool_binaryImage2int1(nifti_image *image, int *array, int &activeVoxelNumber)
{
    // Active voxel are different from -1
    activeVoxelNumber=0;
    DTYPE *dataPtr=static_cast<DTYPE *>(image->data);
    for(unsigned i=0; i<image->nvox; i++){
        if(*dataPtr++ != 0){
            array[i]=1;
            activeVoxelNumber++;
        }
        else{
            array[i]=-1;
        }
    }
}
/* *************************************************************** */
void reg_tool_binaryImage2int(nifti_image *image, int *array, int &activeVoxelNumber)
{
    switch(image->datatype){
        case NIFTI_TYPE_UINT8:
            reg_tool_binaryImage2int1<unsigned char>(image, array, activeVoxelNumber);
            break;
        case NIFTI_TYPE_INT8:
            reg_tool_binaryImage2int1<char>(image, array, activeVoxelNumber);
            break;
        case NIFTI_TYPE_UINT16:
            reg_tool_binaryImage2int1<unsigned short>(image, array, activeVoxelNumber);
            break;
        case NIFTI_TYPE_INT16:
            reg_tool_binaryImage2int1<short>(image, array, activeVoxelNumber);
            break;
        case NIFTI_TYPE_UINT32:
            reg_tool_binaryImage2int1<unsigned int>(image, array, activeVoxelNumber);
            break;
        case NIFTI_TYPE_INT32:
            reg_tool_binaryImage2int1<int>(image, array, activeVoxelNumber);
            break;
        case NIFTI_TYPE_FLOAT32:
            reg_tool_binaryImage2int1<float>(image, array, activeVoxelNumber);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_tool_binaryImage2int1<double>(image, array, activeVoxelNumber);
            break;
        default:
            printf("err\treg_tool_binarise_image\tThe image data type is not supported\n");
            return;
    }
}
/* *************************************************************** */
/* *************************************************************** */
template <class ATYPE,class BTYPE>
double reg_tools_getMeanRMS2(nifti_image *imageA, nifti_image *imageB)
{
    ATYPE *imageAPtrX = static_cast<ATYPE *>(imageA->data);
    BTYPE *imageBPtrX = static_cast<BTYPE *>(imageB->data);
    ATYPE *imageAPtrY=NULL;
    BTYPE *imageBPtrY=NULL;
    ATYPE *imageAPtrZ=NULL;
    BTYPE *imageBPtrZ=NULL;
    if(imageA->dim[5]>1){
        imageAPtrY = &imageAPtrX[imageA->nx*imageA->ny*imageA->nz];
        imageBPtrY = &imageBPtrX[imageA->nx*imageA->ny*imageA->nz];
    }
    if(imageA->dim[5]>2){
        imageAPtrZ = &imageAPtrY[imageA->nx*imageA->ny*imageA->nz];
        imageBPtrZ = &imageBPtrY[imageA->nx*imageA->ny*imageA->nz];
    }
    double sum=0.0f;
    double rms;
    double diff;
    for(int i=0; i<imageA->nx*imageA->ny*imageA->nz; i++){
        diff = (double)*imageAPtrX++ - (double)*imageBPtrX++;
        rms = diff * diff;
        if(imageA->dim[5]>1){
            diff = (double)*imageAPtrY++ - (double)*imageBPtrY++;
            rms += diff * diff;
        }
        if(imageA->dim[5]>2){
            diff = (double)*imageAPtrZ++ - (double)*imageBPtrZ++;
            rms += diff * diff;
        }
        sum += sqrt(rms);
    }
    return sum/(double)(imageA->nx*imageA->ny*imageA->nz);
}
/* *************************************************************** */
template <class ATYPE>
double reg_tools_getMeanRMS1(nifti_image *imageA, nifti_image *imageB)
{
    switch(imageB->datatype){
        case NIFTI_TYPE_UINT8:
            return reg_tools_getMeanRMS2<ATYPE,unsigned char>(imageA, imageB);
        case NIFTI_TYPE_INT8:
            return reg_tools_getMeanRMS2<ATYPE,char>(imageA, imageB);
        case NIFTI_TYPE_UINT16:
            return reg_tools_getMeanRMS2<ATYPE,unsigned short>(imageA, imageB);
        case NIFTI_TYPE_INT16:
            return reg_tools_getMeanRMS2<ATYPE,short>(imageA, imageB);
        case NIFTI_TYPE_UINT32:
            return reg_tools_getMeanRMS2<ATYPE,unsigned int>(imageA, imageB);
        case NIFTI_TYPE_INT32:
            return reg_tools_getMeanRMS2<ATYPE,int>(imageA, imageB);
        case NIFTI_TYPE_FLOAT32:
            return reg_tools_getMeanRMS2<ATYPE,float>(imageA, imageB);
        case NIFTI_TYPE_FLOAT64:
            return reg_tools_getMeanRMS2<ATYPE,double>(imageA, imageB);
        default:
            printf("err\treg_tools_getMeanRMS\tThe image data type is not supported\n");
            return -1;
    }
}
/* *************************************************************** */
double reg_tools_getMeanRMS(nifti_image *imageA, nifti_image *imageB)
{
    switch(imageA->datatype){
        case NIFTI_TYPE_UINT8:
            return reg_tools_getMeanRMS1<unsigned char>(imageA, imageB);
        case NIFTI_TYPE_INT8:
            return reg_tools_getMeanRMS1<char>(imageA, imageB);
        case NIFTI_TYPE_UINT16:
            return reg_tools_getMeanRMS1<unsigned short>(imageA, imageB);
        case NIFTI_TYPE_INT16:
            return reg_tools_getMeanRMS1<short>(imageA, imageB);
        case NIFTI_TYPE_UINT32:
            return reg_tools_getMeanRMS1<unsigned int>(imageA, imageB);
        case NIFTI_TYPE_INT32:
            return reg_tools_getMeanRMS1<int>(imageA, imageB);
        case NIFTI_TYPE_FLOAT32:
            return reg_tools_getMeanRMS1<float>(imageA, imageB);
        case NIFTI_TYPE_FLOAT64:
            return reg_tools_getMeanRMS1<double>(imageA, imageB);
        default:
            printf("err\treg_tools_getMeanRMS\tThe image data type is not supported\n");
            return -1;
    }
}

/* *************************************************************** */
/* *************************************************************** */
#endif
