/*
 *  reg_tools.cpp
 *
 *
 *  Created by Marc Modat and Pankaj Daga on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */
#ifndef MM_TOOLS_CPP
#define MM_TOOLS_CPP

#include "_reg_resampling.h"
#include "_reg_affineTransformation.h"
#include "_reg_bspline.h"
#include "_reg_tools.h"

#define PrecisionTYPE float

typedef struct{
	char *inputImageName;
	char *outputImageName;
    char *addImageName;
    char *rmsImageName;
	int smoothValue;
}PARAM;
typedef struct{
	bool inputImageFlag;
	bool outputImageFlag;
    bool addImageFlag;
    bool rmsImageFlag;
	bool smoothValueFlag;
	bool gradientImageFlag;
}FLAG;


void PetitUsage(char *exec)
{
	fprintf(stderr,"Usage:\t%s -in  <targetImageName> [OPTIONS].\n",exec);
	fprintf(stderr,"\tSee the help for more details (-h).\n");
	return;
}
void Usage(char *exec)
{
	printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
	printf("Usage:\t%s -in <filename> -out <filename> [OPTIONS].\n",exec);
	printf("\t-in <filename>\tFilename of the input image image (mandatory)\n");
	printf("* * OPTIONS * *\n");
    printf("\t-out <filename>\t\tFilename out the output image [output.nii]\n");
	printf("\t-grad\t\t\t4D spatial gradient of the input image\n");
	printf("\t-add <filename>\t\tThis image is added to the input\n");
    printf("\t-smo <int>\t\tThe input image is smoothed using a b-spline curve\n");
    printf("\t-rms <filename>\tCompute the mean rms between both image\n");
	printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
	return;
}

int main(int argc, char **argv)
{
	PARAM *param = (PARAM *)calloc(1,sizeof(PARAM));
	FLAG *flag = (FLAG *)calloc(1,sizeof(FLAG));
	
	/* read the input parameter */
	for(int i=1;i<argc;i++){
		if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 ||
			strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 ||
			strcmp(argv[i], "--h")==0 || strcmp(argv[i], "--help")==0){
			Usage(argv[0]);
			return 0;
		}
		else if(strcmp(argv[i], "-in") == 0){
			param->inputImageName=argv[++i];
			flag->inputImageFlag=1;
		}
		else if(strcmp(argv[i], "-out") == 0){
			param->outputImageName=argv[++i];
			flag->outputImageFlag=1;
		}
		else if(strcmp(argv[i], "-grad") == 0){
			flag->gradientImageFlag=1;
		}
        else if(strcmp(argv[i], "-add") == 0){
            param->addImageName=argv[++i];
            flag->addImageFlag=1;
        }
        else if(strcmp(argv[i], "-rms") == 0){
            param->rmsImageName=argv[++i];
            flag->rmsImageFlag=1;
        }
		else if(strcmp(argv[i], "-smo") == 0){
			param->smoothValue=atoi(argv[++i]);
			flag->smoothValueFlag=1;
		}
		 else{
			 fprintf(stderr,"Err:\tParameter %s unknown.\n",argv[i]);
			 PetitUsage(argv[0]);
			 return 1;
		 }
	}
		
	/* Read the image */
	nifti_image *image = nifti_image_read(param->inputImageName,true);
	if(image == NULL){
		fprintf(stderr,"** ERROR Error when reading the target image: %s\n",param->inputImageName);
		return 1;
	}
	
	/* spatial gradient image */
	if(flag->gradientImageFlag){
		nifti_image *spatialGradientImage = nifti_copy_nim_info(image);
		spatialGradientImage->dim[0]=spatialGradientImage->ndim=4;
		spatialGradientImage->dim[1]=spatialGradientImage->nx=image->nx;
		spatialGradientImage->dim[2]=spatialGradientImage->ny=image->ny;
		spatialGradientImage->dim[3]=spatialGradientImage->nz=image->nz;
		spatialGradientImage->dim[4]=spatialGradientImage->nt=3;spatialGradientImage->pixdim[4]=spatialGradientImage->dt=1.0;
		spatialGradientImage->dim[5]=spatialGradientImage->nu=1;spatialGradientImage->pixdim[5]=spatialGradientImage->du=1.0;
		spatialGradientImage->dim[6]=spatialGradientImage->nv=1;spatialGradientImage->pixdim[6]=spatialGradientImage->dv=1.0;
		spatialGradientImage->dim[7]=spatialGradientImage->nw=1;spatialGradientImage->pixdim[7]=spatialGradientImage->dw=1.0;
		spatialGradientImage->nvox=spatialGradientImage->nx*spatialGradientImage->ny*spatialGradientImage->nz*spatialGradientImage->nt*spatialGradientImage->nu;
		if(sizeof(PrecisionTYPE)==4) spatialGradientImage->datatype = NIFTI_TYPE_FLOAT32;
		else spatialGradientImage->datatype = NIFTI_TYPE_FLOAT64;
		spatialGradientImage->nbyper = sizeof(PrecisionTYPE);
		spatialGradientImage->data = (void *)malloc(spatialGradientImage->nvox * spatialGradientImage->nbyper);
		if(flag->outputImageFlag)
			nifti_set_filenames(spatialGradientImage, param->outputImageName, 0, 0);
		else nifti_set_filenames(spatialGradientImage, "output.nii", 0, 0);
		
		mat44 *affineTransformation = (mat44 *)calloc(1,sizeof(mat44));
		affineTransformation->m[0][0]=1.0;
		affineTransformation->m[1][1]=1.0;
		affineTransformation->m[2][2]=1.0;
		affineTransformation->m[3][3]=1.0;
		nifti_image *fakepositionField = nifti_copy_nim_info(spatialGradientImage);
		fakepositionField->data = (void *)malloc(fakepositionField->nvox * fakepositionField->nbyper);
		reg_affine_positionField(	affineTransformation,
						            image,
						            fakepositionField);
		free(affineTransformation);
		reg_getSourceImageGradient<PrecisionTYPE>(	image,
								                    image,
								                    spatialGradientImage,
								                    fakepositionField,
                                                    NULL,
								                    3); // cubic spline interpolation
		nifti_image_free(fakepositionField);
		nifti_image_write(spatialGradientImage);
		nifti_image_free(spatialGradientImage);
	}
	
	if(flag->smoothValueFlag){
		nifti_image *smoothImg = nifti_copy_nim_info(image);
		smoothImg->data = (void *)malloc(smoothImg->nvox * smoothImg->nbyper);
		memcpy(smoothImg->data, image->data, smoothImg->nvox*smoothImg->nbyper);
		if(flag->outputImageFlag)
			nifti_set_filenames(smoothImg, param->outputImageName, 0, 0);
		else nifti_set_filenames(smoothImg, "output.nii", 0, 0);
		printf("%i\n", param->smoothValue);
		int radius[3];radius[0]=radius[1]=radius[2]=param->smoothValue;
		reg_smoothImageForCubicSpline<PrecisionTYPE>(smoothImg, radius);
		nifti_image_write(smoothImg);
		nifti_image_free(smoothImg);
	}

    if(flag->addImageFlag){
        nifti_image *imageToAdd = nifti_image_read(param->addImageName,true);
        if(imageToAdd == NULL){
            fprintf(stderr,"** ERROR Error when reading the image to add: %s\n",param->addImageName);
            return 1;
        }
        // Check image dimension
        if(image->dim[0]!=imageToAdd->dim[0] ||
           image->dim[1]!=imageToAdd->dim[1] ||
           image->dim[2]!=imageToAdd->dim[2] ||
           image->dim[3]!=imageToAdd->dim[3] ||
           image->dim[4]!=imageToAdd->dim[4] ||
           image->dim[5]!=imageToAdd->dim[5] ||
           image->dim[6]!=imageToAdd->dim[6] ||
           image->dim[7]!=imageToAdd->dim[7]){
            fprintf(stderr,"Both images do not have the same dimension\n");
            return 1;
        }
        nifti_image *sumImage = nifti_copy_nim_info(image);
        sumImage->data = (void *)malloc(sumImage->nvox * sumImage->nbyper);
        if(flag->outputImageFlag)
            nifti_set_filenames(sumImage, param->outputImageName, 0, 0);
        else nifti_set_filenames(sumImage, "output.nii", 0, 0);

        reg_tools_addImages(image, imageToAdd, sumImage);
        nifti_image_write(sumImage);
        nifti_image_free(sumImage);
        nifti_image_free(imageToAdd);
    }

    if(flag->rmsImageFlag){
        nifti_image *image2 = nifti_image_read(param->rmsImageName,true);
        if(image2 == NULL){
            fprintf(stderr,"** ERROR Error when reading the image to add: %s\n",param->rmsImageName);
            return 1;
        }
        // Check image dimension
        if(image->dim[0]!=image2->dim[0] ||
           image->dim[1]!=image2->dim[1] ||
           image->dim[2]!=image2->dim[2] ||
           image->dim[3]!=image2->dim[3] ||
           image->dim[4]!=image2->dim[4] ||
           image->dim[5]!=image2->dim[5] ||
           image->dim[6]!=image2->dim[6] ||
           image->dim[7]!=image2->dim[7]){
            fprintf(stderr,"Both images do not have the same dimension\n");
            return 1;
        }

        double meanRMSerror = reg_tools_getMeanRMS(image, image2);
        printf("%g\n", meanRMSerror);
        nifti_image_free(image2);
    }

	nifti_image_free(image);
	return 0;
}

#endif
