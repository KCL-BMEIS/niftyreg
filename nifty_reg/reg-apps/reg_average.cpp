/*
 *  reg_average.cpp
 *
 *
 *  Created by Marc Modat on 29/10/2011.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */
#ifndef MM_AVERAGE_CPP
#define MM_AVERAGE_CPP

#include "_reg_tools.h"

#ifdef _USE_NR_DOUBLE
#define PrecisionTYPE double
#else
#define PrecisionTYPE float
#endif

void usage(char *exec)
{
    printf("usage:\n\t%s <outputFileName> <inputFileName1> <inputFileName2> <inputFileName3> ...\n", exec);
}

int main(int argc, char **argv)
{
    if(argc<3){
        usage(argv[0]);
        EXIT_SUCCESS;
    }
    // Read the first image to average
    nifti_image *tempImage=nifti_image_read(argv[2],false);
    if(tempImage==NULL){
        fprintf(stderr, "The following image can not be read: %s\n", argv[2]);
        return EXIT_FAILURE;
    }
    reg_checkAndCorrectDimension(tempImage);

    // Create the average image
    nifti_image *average_image=nifti_copy_nim_info(tempImage);
    nifti_image_free(tempImage);tempImage=NULL;
    average_image->datatype=NIFTI_TYPE_FLOAT32;
    if(sizeof(PrecisionTYPE)==sizeof(double))
        average_image->datatype=NIFTI_TYPE_FLOAT64;
    average_image->nbyper=sizeof(PrecisionTYPE);
    average_image->data=(void *)malloc(average_image->nvox*average_image->nbyper);
    nifti_set_filenames(average_image,argv[1],0,0);
    reg_tools_addSubMulDivValue(average_image,average_image,0.f,2);

    int imageTotalNumber=0;
    for(unsigned int i=2;i<argc;++i){
        nifti_image *tempImage=nifti_image_read(argv[i],true);
        if(tempImage==NULL){
            fprintf(stderr, "[!] The following image can not be read: %s\n", argv[i]);
            return EXIT_FAILURE;
        }
        reg_checkAndCorrectDimension(tempImage);
        if(average_image->nvox!=tempImage->nvox){
            fprintf(stderr, "[!] All images must have the same size. Error when processing: %s\n", argv[i]);
            return EXIT_FAILURE;
        }
        reg_tools_addSubMulDivImages(average_image,tempImage,average_image,0);
        imageTotalNumber++;
        nifti_image_free(tempImage);tempImage=NULL;
    }
    reg_tools_addSubMulDivValue(average_image,average_image,(float)imageTotalNumber,3);
    nifti_image_write(average_image);
    nifti_image_free(average_image);

    return EXIT_SUCCESS;
}

#endif
