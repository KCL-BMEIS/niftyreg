#include "_reg_ReadWriteImage.h"
#include "_reg_ReadWriteMatrix.h"
#include "_reg_localTrans.h"
#include "_reg_tools.h"

#include "AffineDeformationFieldKernel.h"

#define EPS 0.0001

int main(int argc, char **argv)
{
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <refImage> <inputGrid> <expectedField> <useComp> <platformCode>\n", argv[0]);
        return EXIT_FAILURE;
    }

    char *inputRefImageName = argv[1];
    char *inputCPPFileName = argv[2];
    char *inputDefImageName = argv[3];
    bool useComposition = atoi(argv[4]);
//    int platformCode = atoi(argv[5]);

    // Read the input reference image
    nifti_image *referenceImage = reg_io_ReadImageFile(inputRefImageName);
    if (referenceImage == NULL) {
        reg_print_msg_error("The input reference image could not be read");
        return EXIT_FAILURE;
    }
    nifti_image *cppImage = reg_io_ReadImageFile(inputCPPFileName);
    if (cppImage == NULL) {
        reg_print_msg_error("The control point grid image could not be read");
        return EXIT_FAILURE;
    }

    // Read the input deformation field image image
    nifti_image *expectedDefField = reg_io_ReadImageFile(inputDefImageName);
    if (expectedDefField == NULL){
        reg_print_msg_error("The input deformation field image could not be read");
        return EXIT_FAILURE;
    }
    // Check the dimension of the input images
    if (referenceImage->nx != expectedDefField->nx ||
        referenceImage->ny != expectedDefField->ny ||
        referenceImage->nz != expectedDefField->nz ||
        (referenceImage->nz > 1 ? 3 : 2) != expectedDefField->nu){
        reg_print_msg_error("The input reference and deformation field images do not have corresponding sizes");
        return EXIT_FAILURE;
    }

    // Create a deformation field
    nifti_image *test_field = nifti_copy_nim_info(expectedDefField);
    test_field->data = (void *)malloc(test_field->nvox*test_field->nbyper);

    if(useComposition)
    {
       // Set the deformation to identity
       reg_tools_multiplyValueToImage(test_field, test_field, 0.f);
       test_field->intent_p1=DISP_FIELD;
       reg_getDeformationFromDisplacement(test_field);

       // Compute the deformation field throught composition
       reg_spline_getDeformationField(cppImage,
                                      test_field,
                                      NULL,
                                      true,
                                      true);
    }
    else{
       // Compute the deformation field from scratch
       reg_spline_getDeformationField(cppImage,
                                      test_field,
                                      NULL,
                                      false,
                                      true);
    }

    // Compute the difference between the computed and expected deformation fields
    nifti_image *diff_field = nifti_copy_nim_info(expectedDefField);
    diff_field->data = (void *) malloc(diff_field->nvox*diff_field->nbyper);
    reg_tools_substractImageToImage(expectedDefField, test_field, diff_field);
    reg_tools_abs_image(diff_field);
    double max_difference = reg_tools_getMaxValue(diff_field, -1);

    // Delete all allocated images
    nifti_image_free(referenceImage);
    nifti_image_free(expectedDefField);
    nifti_image_free(cppImage);
    nifti_image_free(test_field);
    nifti_image_free(diff_field);

    // Check if the obtained difference is below a specific threshold
    if (max_difference > EPS){
        fprintf(stderr, "reg_test_bspline_deformation_field from blank error too large: %g (>%g)\n",
                max_difference, EPS);
        // return on a failed test
        return EXIT_FAILURE;
    }

#ifndef NDEBUG
    fprintf(stdout, "reg_test_bspline_deformation_field ok 1: %g (<%g)\n",
            max_difference, EPS);
#endif

    // return on a successful test
    return EXIT_SUCCESS;
}

