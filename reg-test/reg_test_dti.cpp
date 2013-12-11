/*
 *  reg_test_dti.cpp
 *
 *
 *  Created by Ivor Simpson on 28/06/2013
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_resampling.h"
#include "_reg_globalTransformation.h"
#include "_reg_localTransformation.h"
#include "_reg_tools.h"
#include "_reg_base.h"

#define EPS 0.001

void usage(char *exec)
{
    printf("Usage:\n");
    printf("\t%s floatingImage deformationField interpolationType expectedImage\n", exec);
}

int main(int argc, char **argv)
{
    // Check the number of arguments
    if(argc!=4){
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

    /*
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
    */
    // Read the expected image and check the dimension
    nifti_image *expectedImage=nifti_image_read(argv[2], true);
    /*if(deformationFieldImage==NULL){
        fprintf(stderr, "Error when reading the expected image: %s\n", argv[2]);
        usage(argv[0]);
        reg_exit(1);
    }
    if(expectedImage->nx != deformationFieldImage->nx ||
       expectedImage->ny != deformationFieldImage->ny ||
       expectedImage->nz != deformationFieldImage->nz){
        fprintf(stderr, "The deformation field and expected image do not match\n");
        usage(argv[0]);
        reg_exit(1);
    }*/
    // Initialise the deformation field
    nifti_image *deformationFieldImage = nifti_copy_nim_info(expectedImage);
    deformationFieldImage->nt = deformationFieldImage->dim[4] = 1;
    deformationFieldImage->nu = deformationFieldImage->dim[5] = 3;
    deformationFieldImage->pixdim[4]=deformationFieldImage->dt=1.0;
    deformationFieldImage->ndim = 5;
    deformationFieldImage->nvox = deformationFieldImage->nx*deformationFieldImage->ny*deformationFieldImage->nz*deformationFieldImage->nt*deformationFieldImage->nu;
    deformationFieldImage->dim[1] = deformationFieldImage->nx;
    deformationFieldImage->dim[2] = deformationFieldImage->ny;
    deformationFieldImage->dim[3] = deformationFieldImage->nz;
    deformationFieldImage->nbyper = sizeof(float);
    deformationFieldImage->dim[6]=deformationFieldImage->nv=1;
    deformationFieldImage->pixdim[6]=deformationFieldImage->dv=1.0;
    deformationFieldImage->dim[7]=deformationFieldImage->nw=1;
    deformationFieldImage->pixdim[7]=deformationFieldImage->dw=1.0;
    deformationFieldImage->intent_code=NIFTI_INTENT_VECTOR;
    memset(deformationFieldImage->intent_name, 0, 16);
    strcpy(deformationFieldImage->intent_name,"NREG_TRANS");
    deformationFieldImage->intent_p1=DEF_FIELD;

    deformationFieldImage->data=(void *)malloc(deformationFieldImage->nvox*deformationFieldImage->nbyper);

    // Interpolation type
    int interpolationType = 1;

    // Create and allocate a warped image that has the dimension and type of the expected image
    nifti_image *warpedImage = nifti_copy_nim_info(expectedImage);
    warpedImage->data=(void *)malloc(warpedImage->nvox*warpedImage->nbyper);

    std::cerr << "Read the images";
    mat44 affineMat;
    reg_tool_ReadAffineFile(&affineMat,
                            floatingImage,
                            expectedImage,
                            argv[3],
                            false);
    //reg_mat44_eye(&affineMat);
    /*double anglex = (45.0f/180.0f)*(3.14159f);
    affineMat.m[1][1] = cos(anglex);
    affineMat.m[1][2] = -sin(anglex);
    affineMat.m[2][1] = sin(anglex);
    affineMat.m[2][2] = cos(anglex);
    reg_mat44_disp(&affineMat,(char *)"Affine matrix");*/

    // Make a deformation field from an affine transformation matrix
    reg_affine_getDeformationField(&affineMat,
                                deformationFieldImage);
    std::cerr << "Made a deformation field from the affine matrix";

    bool dti_timepoint[255];
    for( unsigned int  i = 0; i < 255; i++ )
    {
        if( i > 0  && i < 7 )
            dti_timepoint[i] = true;
        else
            dti_timepoint[i] = false;
    }

    unsigned int noVoxs = expectedImage->nx*expectedImage->ny*expectedImage->nz;//for()

    mat33 * jacMat = new mat33[noVoxs];
    /*for(unsigned i = 0; i < noVoxs; i++ )
    {
        jacMat[i] = reg_mat44_to_mat33(&affineMat);
    }*/
    // We could generate the jacobian matrices from the B-spline cp grid
    nifti_image * cpGridImage = NULL;
    float spacing[3] = {10,10,10};
    reg_createControlPointGrid<float>(&cpGridImage, expectedImage,spacing);
    reg_affine_getDeformationField(&affineMat,cpGridImage);
    float * cpPtr = static_cast<float *>(cpGridImage->data);
    for(unsigned int c = 0; c <cpGridImage->nvox; c++)
    {
        //std::cerr << *cpPtr;
        *cpPtr = *cpPtr+5*((float)(rand())/RAND_MAX)-2.5;
        //std::cerr << '\t' << *cpPtr <<'\n';

        cpPtr++;
    }
    reg_spline_getDeformationField(cpGridImage,
                                    deformationFieldImage,
                                    NULL,
                                    false,
                                     true);
    std::cerr << "Made a CP grid";
	reg_spline_GetJacobianMatrix(floatingImage,
								 cpGridImage,
								 jacMat);

    //reg_getDeformationFromDisplacement(deformationFieldImage);

    // Resample the floating image in the space of the expected image
    reg_resampleImage(floatingImage,
                      warpedImage,
                      deformationFieldImage,
                      NULL, // no mask is used
                      interpolationType,
                      std::numeric_limits<float>::quiet_NaN(),
                      dti_timepoint,
                      jacMat); // padding value
    nifti_image *outputImage = warpedImage;
    char * outputImageName=(char *)"warpedImage.nii";
    memset(outputImage->descrip, 0, 80);
    strcpy (outputImage->descrip,"Warped image from NiftyReg (reg_f3d)");
    reg_io_WriteImageFile(outputImage,outputImageName);
    nifti_image_free(outputImage);outputImage=NULL;

    // Convert deformation field to a displacement field so it will work with dtitk
    reg_getDisplacementFromDeformation(deformationFieldImage);
    outputImage = deformationFieldImage;
    outputImageName=(char *)"displacementField.nii";
    memset(outputImage->descrip, 0, 80);
    strcpy (outputImage->descrip,"Control point position from NiftyReg (reg_f3d)");
    reg_io_WriteImageFile(outputImage,outputImageName);
    nifti_image_free(outputImage);outputImage=NULL;



    /*// Compute the maximal difference between the warped and expected image
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
#endif*/

    // Clean all the allocated memory
    nifti_image_free(floatingImage);
    //nifti_image_free(warpedImage);
    nifti_image_free(expectedImage);

    // Check if the test failed or passed
    /*if(difference>EPS){
        fprintf(stderr, "Max difference: %g - Threshold: %g\n",difference, EPS);
        return 1;
    }
    return 0;*/
}

