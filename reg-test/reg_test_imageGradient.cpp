#include "_reg_ReadWriteImage.h"
#include "_reg_globalTrans.h"
#include "_reg_tools.h"
#include "_reg_mind.h"

#define EPS 0.000001

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <image to process> <expected gradient image> <type=0|1>\n", argv[0]);
        return EXIT_FAILURE;
    }
    char *inputImageName = argv[1];
    // Read the input image
    nifti_image *inputImage = reg_io_ReadImageFile(inputImageName);
    if (inputImage == NULL) {
        reg_print_msg_error("The input image could not be read");
        return EXIT_FAILURE;
    }
    //Convert the image in float
    reg_tools_changeDatatype<float>(inputImage);
    //
    char *expectedImageName = argv[2];
    // Read the expected image
    nifti_image *expectedImage = reg_io_ReadImageFile(expectedImageName);
    if (expectedImage == NULL) {
        reg_print_msg_error("The expected image could not be read");
        return EXIT_FAILURE;
    }

    int usedMethod = atoi(argv[3]);
    // Read the expected image
    if(usedMethod != 0 && usedMethod != 1 && usedMethod != 3) {
        reg_print_msg_error("The current method is not supported - should be 0, 1 or 3");
        return EXIT_FAILURE;
    }
    int dim = (inputImage->nz > 1) ? 3 : 2;

    // Allocate a gradient image
    nifti_image *gradientImage = nifti_copy_nim_info(inputImage);
    gradientImage->dim[0]=gradientImage->ndim=5;
    gradientImage->dim[5]=gradientImage->nu=dim;
    gradientImage->nvox = (size_t)gradientImage->nx*gradientImage->ny*
                      gradientImage->nz*gradientImage->nt*gradientImage->nu;
    gradientImage->nbyper=sizeof(float);
    gradientImage->datatype=NIFTI_TYPE_FLOAT32;
    gradientImage->data=(void *)malloc(gradientImage->nvox*gradientImage->nbyper);

    // Allocate a temporary file to compute the gradient's timepoint one at the time
    nifti_image *tempGradImage = nifti_copy_nim_info(gradientImage);
    tempGradImage->dim[4]=tempGradImage->nt=1;
    tempGradImage->nvox = (size_t)tempGradImage->nx*tempGradImage->ny*
                      tempGradImage->nz*tempGradImage->nt*tempGradImage->nu;
    tempGradImage->data=(void *)malloc(tempGradImage->nvox*tempGradImage->nbyper);

    // Declare a deformation field image
    nifti_image *defFieldImage = NULL;
    // Allocate a deformation field image if required
    if(usedMethod > 0)
    {
        defFieldImage = nifti_copy_nim_info(inputImage);
        defFieldImage->dim[0]=defFieldImage->ndim=5;
        defFieldImage->dim[4]=defFieldImage->nt=1;
        defFieldImage->dim[5]=defFieldImage->nu=dim;
        defFieldImage->nvox = (size_t)defFieldImage->nx*defFieldImage->ny *
                                 defFieldImage->nz*defFieldImage->nu;
        defFieldImage->nbyper=sizeof(float);
        defFieldImage->datatype=NIFTI_TYPE_FLOAT32;
        defFieldImage->intent_code=NIFTI_INTENT_VECTOR;
        memset(defFieldImage->intent_name, 0, 16);
        strcpy(defFieldImage->intent_name,"NREG_TRANS");
        defFieldImage->intent_p1=DISP_FIELD;
        // Set the deformation field to identity
        defFieldImage->data = (void *)calloc(defFieldImage->nvox, defFieldImage->nbyper);
        reg_getDeformationFromDisplacement(defFieldImage);
    }

    // Allocate a mask array
    int *mask = (int *)calloc(inputImage->nvox,sizeof(int));

    // Setup pointers over the gradient images
    float *tempGradImgPtr = static_cast<float *>(tempGradImage->data);

    float *gradImagePtr = static_cast<float *>(gradientImage->data);
    // Loop over the input image timepoints
    for(int time=0; time<inputImage->nt; ++time){
        if(usedMethod == 0){
            // Compute the gradient using symmetric difference
            reg_getImageGradient_symDiff(inputImage,
                                         tempGradImage,
                                         mask,
                                         0,
                                         time);
        }
        else if(usedMethod == 3){
            // Compute the gradient from the deformation field using spline interpolation
            // Given an identity transformation, since gives the same as symmetric
            // difference with a kernel of [-1/2 0 1/2]
            reg_getImageGradient(inputImage,
                                 tempGradImage,
                                 defFieldImage,
                                 mask,
                                 3,
                                 0.f,
                                 time);
        }
        else{
            // Compute the gradient from the deformation field using linear interpolation
            reg_getImageGradient(inputImage,
                                 tempGradImage,
                                 defFieldImage,
                                 mask,
                                 1,
                                 std::numeric_limits<float>::quiet_NaN(),
                                 time);
        }
        // Copy the single time point gradient in the less effective way known to mankind
        for(int u=0; u<gradientImage->nu; ++u){
            for(int z=0; z<gradientImage->nz; ++z){
                for(int y=0; y<gradientImage->ny; ++y){
                    for(int x=0; x<gradientImage->nx; ++x){
                        size_t voxIndex_gradImg=
                                gradientImage->nx*gradientImage->ny*gradientImage->nz*gradientImage->nt*u +
                                gradientImage->nx*gradientImage->ny*gradientImage->nz*time +
                                gradientImage->nx*gradientImage->ny*z +
                                gradientImage->nx*y +
                                x;
                        size_t voxIndex_tempGrad=
                                tempGradImage->nx*tempGradImage->ny*tempGradImage->nz*tempGradImage->nt*u +
                                tempGradImage->nx*tempGradImage->ny*z +
                                tempGradImage->nx*y +
                                x;
                        gradImagePtr[voxIndex_gradImg]=tempGradImgPtr[voxIndex_tempGrad];
                    }
                }
            }
        }
    }

    // Free the allocated arrays and images
    if(defFieldImage!=NULL)
        nifti_image_free(defFieldImage);
    nifti_image_free(tempGradImage);
    free(mask);

    //Compute the difference between the computed and expected image
    reg_tools_substractImageToImage(gradientImage, expectedImage, expectedImage);

    // Extract the maximal absolute value
    reg_tools_abs_image(expectedImage);
    double max_difference = reg_tools_getMaxValue(expectedImage, -1);


    reg_io_WriteImageFile(gradientImage, "res.nii.gz");
    reg_io_WriteImageFile(expectedImage, "diff.nii.gz");

    nifti_image_free(inputImage);
    nifti_image_free(expectedImage);
    nifti_image_free(gradientImage);

    if (max_difference > EPS){
        fprintf(stderr, "reg_test_imageGradient error too large: %g (>%g)\n",
                max_difference, EPS);
        return EXIT_FAILURE;
    }
#ifndef NDEBUG
    fprintf(stdout, "reg_test_imageGradient ok: %g (<%g)\n", max_difference, EPS);
#endif
    return EXIT_SUCCESS;
}
