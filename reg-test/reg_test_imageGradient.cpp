
//TEST CHANGE DATATYPE
#include "_reg_ReadWriteImage.h"
#include "_reg_globalTrans.h"
#include "_reg_tools.h"
#include "_reg_mind.h"
//
#define EPS 0.000001
//
int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <image to process> <expected gradient image> <m>\n", argv[0]);
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
    if(usedMethod != 0 && usedMethod != 1) {
        reg_print_msg_error("The current method is not supported - should be 0 or 1");
        return EXIT_FAILURE;
    }

    int dim = (inputImage->nz > 1) ? 3 : 2;
    //NB VOXEL
    //int voxelNumber = inputImage->nx*inputImage->ny*inputImage->nz;
    // COMPUTE THE GRADIENT OF THE IMAGE
    nifti_image *gradImg = nifti_copy_nim_info(inputImage);
    gradImg->dim[0]=gradImg->ndim=5;
    gradImg->dim[5]=gradImg->nu=dim;
    gradImg->nvox = (size_t)gradImg->nx*
                    gradImg->ny*
                    gradImg->nz*
                    gradImg->nt*
                    gradImg->nu;
    gradImg->data=(void *)malloc(gradImg->nvox*gradImg->nbyper);
    //GRAD IMAGE "3D"
    nifti_image *gradImg3D = nifti_copy_nim_info(inputImage);
    gradImg3D->dim[0]=gradImg3D->ndim=5;
    gradImg3D->dim[4]=gradImg3D->nt=1;
    gradImg3D->dim[5]=gradImg3D->nu=dim;
    gradImg3D->nvox = (size_t)gradImg3D->nx*
                      gradImg3D->ny*
                      gradImg3D->nz*
                      gradImg3D->nt*
                      gradImg3D->nu;
    gradImg3D->data=(void *)malloc(gradImg3D->nvox*gradImg3D->nbyper);

    // Create a mask
    //int *mask = (int *)malloc(inputImage->nvox*sizeof(int));
    //for (size_t i = 0; i < inputImage->nvox; ++i) {
    //   mask[i] = i;
    //}

    // Compute the gradient of the warped floating descriptor image
    if(usedMethod == 0) {

        // Create an identity transformation
        nifti_image *identityDefField = nifti_copy_nim_info(inputImage);
        identityDefField->dim[0]=identityDefField->ndim=5;
        identityDefField->dim[4]=identityDefField->nt=1;
        identityDefField->dim[5]=identityDefField->nu=dim;
        identityDefField->nvox = (size_t)identityDefField->nx *
                                 identityDefField->ny *
                                 identityDefField->nz *
                                 identityDefField->nu;
        identityDefField->datatype=NIFTI_TYPE_FLOAT32;
        identityDefField->nbyper=sizeof(float);
        identityDefField->data = (void *)calloc(identityDefField->nvox,
                                                identityDefField->nbyper);
        identityDefField->intent_code=NIFTI_INTENT_VECTOR;
        memset(identityDefField->intent_name, 0, 16);
        strcpy(identityDefField->intent_name,"NREG_TRANS");
        identityDefField->intent_p1=DISP_FIELD;
        reg_getDeformationFromDisplacement(identityDefField);

        reg_getImageGradient(inputImage,
                             gradImg,
                             identityDefField,
                             NULL,
                             1,
                             std::numeric_limits<float>::quiet_NaN(),
                             0);
        //
        nifti_image_free(identityDefField);identityDefField=NULL;
        //
    } else {
        float *gradPtr = static_cast<float *>(gradImg->data);
        float *grad3DPtr = static_cast<float *>(gradImg3D->data);
        int *mask = (int *)calloc(inputImage->nvox,sizeof(int));
        for(int t=0; t<inputImage->nt; ++t){
            spatialGradient<float>(inputImage,gradImg3D,mask,t);
            //It is not very optimised...
            for(int u=0; u<gradImg3D->nu; ++u){
                for(int z=0; z<gradImg3D->nz; ++z){
                    for(int y=0; y<gradImg3D->ny; ++y){
                        for(int x=0; x<gradImg3D->nx; ++x){
                            size_t voxIndex_gradImg=
                                     gradImg->nx*gradImg->ny*gradImg->nz*gradImg->nt*u
                                    +gradImg->nx*gradImg->ny*gradImg->nz*t
                                    +gradImg->nx*gradImg->ny*z
                                    +gradImg->nx*y
                                    +x;
                            size_t voxIndex_gradImg3D=
                                     gradImg3D->nx*gradImg3D->ny*gradImg3D->nz*gradImg3D->nt*u
                                    +gradImg3D->nx*gradImg3D->ny*gradImg3D->nz*0
                                    +gradImg3D->nx*gradImg3D->ny*z
                                    +gradImg3D->nx*y
                                    +x;
                            gradPtr[voxIndex_gradImg]=grad3DPtr[voxIndex_gradImg3D];
                        }
                    }
                }
            }
        }
        free(mask);
    }
    //
    //Compute the difference between the computed and expected image
    //
    reg_tools_substractImageToImage(gradImg, expectedImage, expectedImage);

    reg_tools_abs_image(expectedImage);
    double max_difference = reg_tools_getMaxValue(expectedImage, -1);

    nifti_image_free(inputImage);
    nifti_image_free(expectedImage);
    nifti_image_free(gradImg);

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
