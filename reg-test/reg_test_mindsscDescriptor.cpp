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
        fprintf(stderr, "Usage: %s <image to process> <expected MIND-SSC image>\n", argv[0]);
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
    if(dim<2 || dim>3){
        reg_print_msg_error("dimension not supported");
        return EXIT_FAILURE;
    }
    // COMPUTE THE MIND DESCRIPTOR
    int lengthDescritor = 12;
    if(dim == 2) {
        lengthDescritor = 4;
    }
    //MINDSSC image
    nifti_image *MINDSSC_img = nifti_copy_nim_info(inputImage);
    MINDSSC_img->ndim = MINDSSC_img->dim[0] = 4;
    MINDSSC_img->nt = MINDSSC_img->dim[4] = lengthDescritor;
    MINDSSC_img->nvox = MINDSSC_img->nvox*lengthDescritor;
    MINDSSC_img->data=(void *)calloc(MINDSSC_img->nvox,MINDSSC_img->nbyper);

    // Compute the MIND descriptor
    int *mask = (int *)calloc(inputImage->nvox, sizeof(int));
    GetMINDSSCImageDesciptor(inputImage,MINDSSC_img, mask, 1, 0);
    free(mask);
    //
    //Compute the difference between the computed and expected image
    //
    reg_tools_substractImageToImage(MINDSSC_img, expectedImage, expectedImage);
    reg_tools_abs_image(expectedImage);
    double max_difference = reg_tools_getMaxValue(expectedImage, -1);

    nifti_image_free(inputImage);
    nifti_image_free(expectedImage);
    nifti_image_free(MINDSSC_img);

    if (max_difference > EPS){
        fprintf(stderr, "reg_test_MINDSSCDescriptor error too large: %g (>%g)\n",
            max_difference, EPS);
        return EXIT_FAILURE;
    }
#ifndef NDEBUG
    fprintf(stdout, "reg_test_MINDSSCDescriptor ok: %g (<%g)\n", max_difference, EPS);
#endif
    return EXIT_SUCCESS;
}
