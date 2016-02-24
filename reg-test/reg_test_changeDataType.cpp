//TEST CHANGE DATATYPE
#include "_reg_ReadWriteImage.h"
#include "_reg_globalTrans.h"
#include "_reg_tools.h"
//
#define EPS 0.000001
//
int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <path to the image to cast> <cast value: float - double> <expected casted image>\n", argv[0]);
        return EXIT_FAILURE;
    }
    //
    char str_float[] = "float";
    char str_double[] = "double";
    char str_uchar[] = "uchar";
    //
    char *inputImageName = argv[1];
    // Read the input image
    nifti_image *referenceImage = reg_io_ReadImageFile(inputImageName);
    if (referenceImage == NULL) {
        reg_print_msg_error("The input reference image could not be read");
        return EXIT_FAILURE;
    }
    //
    char* castValue = argv[2];
    if (strcmp(castValue, str_float) != 0 && strcmp(castValue, str_double) != 0 && strcmp(castValue, str_uchar) != 0) {
        reg_print_msg_error("The cast value is wrong - it should be uchar, float or double");
        return EXIT_FAILURE;
    }
    //
    char *expectedImageName = argv[3];
    // Read the input image
    nifti_image *expectedImage = reg_io_ReadImageFile(expectedImageName);
    if (expectedImage == NULL) {
        reg_print_msg_error("The expected image could not be read");
        return EXIT_FAILURE;
    }
    //
    ///////////////////////////////////////////////////////////////////////////////////////
#ifndef NDEBUG
    //TEST CHANGE DATATYPE --> WE CAN ONLY UPGRADE THE DATATYPE !
    //FIRST DETECT THE DATATYPE OF THE INPUT IMAGE
    char* inputDataType = nifti_datatype_string(referenceImage->datatype);
    char text[255];
    sprintf(text, "The input image datatype is: %s", inputDataType);
    reg_print_msg_debug(text);
    //
    char text3[255];
    sprintf(text3, "The cast value is: %s", castValue);
    reg_print_msg_debug(text3);
    //DETECT THE DATATYPE OF THE EXPECTED IMAGE
    char* expectedDataType = nifti_datatype_string(expectedImage->datatype);
    char text2[255];
    sprintf(text2, "The expected image datatype is: %s", expectedDataType);
    reg_print_msg_debug(text2);
#endif
    ///////////////////////////////////////////////////////////////////////////////////////
    if (strcmp(castValue, str_float) == 0) {
#ifndef NDEBUG
        reg_print_msg_debug("cast image to float")
#endif
            reg_tools_changeDatatype<float>(referenceImage);
    }
    else if (strcmp(castValue, str_double) == 0) {
#ifndef NDEBUG
        reg_print_msg_debug("cast image to double")
#endif
            reg_tools_changeDatatype<double>(referenceImage);
    }
    else if (strcmp(castValue, str_uchar) == 0) {
#ifndef NDEBUG
        reg_print_msg_debug("cast image to unsigned char")
#endif
            reg_tools_changeDatatype<unsigned char>(referenceImage);
    }
    else {
        reg_print_msg_error("The reference image could not be casted");
        return EXIT_FAILURE;
    }
    //
    // Compute the difference between the computed and inputed deformation field
    reg_tools_substractImageToImage(referenceImage, expectedImage, expectedImage);
    reg_tools_abs_image(expectedImage);
    double max_difference = reg_tools_getMaxValue(expectedImage, -1);

    nifti_image_free(referenceImage);
    nifti_image_free(expectedImage);

    if (max_difference > EPS){
        fprintf(stderr, "reg_test_changeDataType error too large: %g (>%g)\n",
            max_difference, EPS);
        return EXIT_FAILURE;
    }
#ifndef NDEBUG
    fprintf(stdout, "reg_test_changeDataType ok: %g (<%g)\n", max_difference, EPS);
#endif
    return EXIT_SUCCESS;
}
