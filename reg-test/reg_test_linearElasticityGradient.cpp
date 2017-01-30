#include "_reg_ReadWriteImage.h"
#include "_reg_localTrans_regul.h"

#define EPS 0.000001

int main(int argc, char **argv)
{
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <refImage> <inputTrans> <expectedGradient> <type>\n", argv[0]);
        return EXIT_FAILURE;
    }

    char *inputRefImageName = argv[1];
    char *inputTransFileName = argv[2];
    char *expectedGradFileName = argv[3];
    int computationType = atoi(argv[4]);

    // Read the input reference image
    nifti_image *referenceImage = reg_io_ReadImageFile(inputRefImageName);
    if (referenceImage == NULL) {
        reg_print_msg_error("The input reference image could not be read");
        return EXIT_FAILURE;
    }
    // Read the transformation file
    nifti_image *transImage = reg_io_ReadImageFile(inputTransFileName);
    if (transImage == NULL) {
        reg_print_msg_error("The transformation image could not be read");
        return EXIT_FAILURE;
    }
    // Read the expected gradient file
    nifti_image *expectedGradientImage = reg_io_ReadImageFile(expectedGradFileName);
    if (expectedGradientImage == NULL) {
        reg_print_msg_error("The expected gradient image could not be read");
        return EXIT_FAILURE;
    }

    // Compute the linear elasticity gradient
    nifti_image *obtainedGradient = nifti_copy_nim_info(expectedGradientImage);
    obtainedGradient->data=(void *)calloc(obtainedGradient->nvox,obtainedGradient->nbyper);
    switch(computationType){
    case 0: // Approximation based on the control point grid
       reg_spline_approxLinearEnergyGradient(transImage,
                                             obtainedGradient,
                                             1.f);
       break;
    case 1: // Dense based on the control point grid
       reg_spline_linearEnergyGradient(referenceImage,
                                       transImage,
                                       obtainedGradient,
                                       1.f);
       break;
    case 2: // Dense based on the deformation field
       reg_defField_linearEnergyGradient(transImage,
                                         obtainedGradient,
                                         1.f);
       break;
    default:
       reg_print_msg_error("Unexpected computation type");
       reg_exit();
    }
    // Compute the difference between the computed and expected gradient
    nifti_image *diff_field = nifti_copy_nim_info(obtainedGradient);
    diff_field->data = (void *)malloc(diff_field->nvox*diff_field->nbyper);
    reg_tools_substractImageToImage(obtainedGradient, expectedGradientImage, diff_field);
    reg_tools_abs_image(diff_field);
    double max_difference = reg_tools_getMaxValue(diff_field, -1);

    // Free allocated images
    nifti_image_free(diff_field);
    nifti_image_free(obtainedGradient);
    nifti_image_free(expectedGradientImage);
    nifti_image_free(referenceImage);
    nifti_image_free(transImage);

    if (max_difference > EPS){
        fprintf(stderr, "reg_test_linearElasticityGradient error too large: %g ( > %g)\n",
                max_difference, EPS);
        return EXIT_FAILURE;
    }
#ifndef NDEBUG
    fprintf(stdout, "reg_test_linearElasticityGradient ok: %g (<%g)\n",
            max_difference, EPS);
#endif

    return EXIT_SUCCESS;
}

