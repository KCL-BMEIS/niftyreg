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
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <image to process> <expected MIND image>\n", argv[0]);
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
    int dim = (inputImage->nz > 1) ? 3 : 2;
    // COMPUTE THE MIND DESCRIPTOR
    //MIND image
    nifti_image *MIND_img = nifti_copy_nim_info(inputImage);
    if(dim == 2) {
        MIND_img->data=(void *)calloc(inputImage->nvox * 4,inputImage->nbyper);
        MIND_img->dim[0] = 4;
        MIND_img->dim[4] = 4;
        MIND_img->nt = MIND_img->dim[4];
        MIND_img->nvox = MIND_img->nvox*MIND_img->dim[4];
    }
    else if (dim == 3) {
        MIND_img->data=(void *)calloc(inputImage->nvox * 6,inputImage->nbyper);
        MIND_img->dim[0] = 4;
        MIND_img->dim[4] = 6;
        MIND_img->nt = MIND_img->dim[4];
        MIND_img->nvox = MIND_img->nvox*MIND_img->dim[4];
    } else {
        reg_print_msg_error("dimension not supported");
        return EXIT_FAILURE;
    }
    //
    GetMINDImageDesciptor(inputImage,MIND_img);
    //
    //Compute the difference between the computed and expected image
    //
    reg_tools_substractImageToImage(MIND_img, expectedImage, expectedImage);
    reg_tools_abs_image(expectedImage);
    double max_difference = reg_tools_getMaxValue(expectedImage);

    nifti_image_free(inputImage);
    nifti_image_free(expectedImage);
    nifti_image_free(MIND_img);

    if (max_difference > EPS){
        fprintf(stderr, "reg_test_MINDDescriptor error too large: %g (>%g)\n",
            max_difference, EPS);
        return EXIT_FAILURE;
    }
#ifndef NDEBUG
    fprintf(stdout, "reg_test_MINDDescriptor ok: %g (<%g)\n", max_difference, EPS);
#endif
    return EXIT_SUCCESS;
}

