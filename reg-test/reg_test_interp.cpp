/*
 *  reg_test_interp.cpp
 *
 *
 *  Created by Marc Modat on 10/05/2012.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_resampling.h"
#include "_reg_tools.h"

#define EPS 0.001

void usage(char *exec)
{
    printf("Usage:\n");
    printf("\t%s floatingImage deformationField interpolationType expectedImage\n", exec);
}

int main(int argc, char **argv)
{
    // Check the number of arguments
    if(argc!=5){
        fprintf(stderr, "Four arguments are expected\n");
        usage(argv[0]);
		exit(1);
    }

    // Read the floating image
    nifti_image *floatingImage=nifti_image_read(argv[1],true);
    if(floatingImage==NULL){
        fprintf(stderr, "Error when reading the floating image: %s\n", argv[1]);
        usage(argv[0]);
        reg_exit(1);
    }

    // Read the deformation field image and perform some quick test about dimension
    nifti_image *deformationFieldImage=nifti_image_read(argv[2], true);
    if(deformationFieldImage==NULL){
        fprintf(stderr, "Error when reading the deformation field image: %s\n", argv[2]);
        usage(argv[0]);
        reg_exit(1);
    }
    if(deformationFieldImage->ndim!=5){
        fprintf(stderr, "The deformation field image is expected to have 5D\n");
        usage(argv[0]);
        reg_exit(1);
    }
    if(deformationFieldImage->nz==1 && deformationFieldImage->nu!=2){
        fprintf(stderr, "The deformation field image does not correspond to a 2D image\n");
        usage(argv[0]);
        reg_exit(1);
    }
    if(deformationFieldImage->nz>1 && deformationFieldImage->nu!=3){
        fprintf(stderr, "The deformation field image does not correspond to a 3D image\n");
        usage(argv[0]);
        reg_exit(1);
    }

    // Read the interpolation type
    int interpolationType = atoi(argv[3]);

    // Read the expected image and check the dimension
    nifti_image *expectedImage=nifti_image_read(argv[4], true);
    if(deformationFieldImage==NULL){
        fprintf(stderr, "Error when reading the expected image: %s\n", argv[4]);
        usage(argv[0]);
        reg_exit(1);
    }
    if(expectedImage->nx != deformationFieldImage->nx ||
       expectedImage->ny != deformationFieldImage->ny ||
       expectedImage->nz != deformationFieldImage->nz){
        fprintf(stderr, "The deformation field and expected image do not match\n");
        usage(argv[0]);
        reg_exit(1);
    }

    // Create and allocate a warped image that has the dimension and type of the expected image
    nifti_image *warpedImage = nifti_copy_nim_info(expectedImage);
    warpedImage->data=(void *)malloc(warpedImage->nvox*warpedImage->nbyper);

    // Resample the floating image in the space of the expected image
    reg_resampleImage(floatingImage,
                      warpedImage,
                      deformationFieldImage,
                      NULL, // no mask is used
                      interpolationType,
                      0.f); // padding value

    // Compute the maximal difference between the warped and expected image
    double  difference = reg_test_compare_images(expectedImage,warpedImage);
#ifndef NDEBUG
		printf("[NiftyReg DEBUG] [dim=%i] Interpolation difference: %g\n",
			   expectedImage->nz>1?3:2,
			   difference);
		nifti_set_filenames(expectedImage, "reg_test_interp_exp.nii",0,0);
		nifti_set_filenames(warpedImage,"reg_test_interp_res.nii",0,0);
		nifti_image_write(expectedImage);
		nifti_image_write(warpedImage);
		reg_tools_divideImageToImage(expectedImage,warpedImage,warpedImage);
		reg_tools_substractValueToImage(warpedImage,warpedImage,1.f);
		reg_tools_abs_image(warpedImage);
		nifti_set_filenames(warpedImage,"reg_test_interp_diff.nii",0,0);
		nifti_image_write(warpedImage);
#endif

    // Clean all the allocated memory
    nifti_image_free(floatingImage);
    nifti_image_free(deformationFieldImage);
    nifti_image_free(warpedImage);
    nifti_image_free(expectedImage);

    // Check if the test failed or passed
	if(difference>EPS){
		fprintf(stderr, "Max difference: %g - Threshold: %g\n",difference, EPS);
        return 1;
    }
	return 0;
}
