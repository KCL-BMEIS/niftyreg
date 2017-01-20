#include "_reg_ReadWriteImage.h"
#include "_reg_tools.h"

#define EPS 0.0001

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <refImage> <expectedImage> <convolutionType>\n", argv[0]);
        return EXIT_FAILURE;
    }

    char *inputImageName = argv[1];
    char *expectedFileName = argv[2];
    int convolutionType = atoi(argv[3]);

    // Read the input reference image
    nifti_image *referenceImage = reg_io_ReadImageFile(inputImageName);
    if (referenceImage == NULL) {
        reg_print_msg_error("The input reference image could not be read");
        return EXIT_FAILURE;
    }
    reg_tools_changeDatatype<double>(referenceImage);

    // Apply the convolution
    float spacing[3]={-5.f,-5.f,-5.f};
    reg_tools_kernelConvolution(referenceImage,
                                spacing,
                                convolutionType);


    // Read the input reference image
    nifti_image *expectedFile = reg_io_ReadImageFile(expectedFileName);
    if (expectedFile == NULL) {
        reg_print_msg_error("The expected result image could not be read");
        return EXIT_FAILURE;
    }
    reg_tools_changeDatatype<double>(expectedFile);

    // Compute the difference between the computed and expected deformation fields
    nifti_image *diff_file = nifti_copy_nim_info(expectedFile);
    diff_file->data = (void *) malloc(diff_file->nvox*diff_file->nbyper);
    reg_tools_substractImageToImage(expectedFile, referenceImage, diff_file);
    reg_tools_abs_image(diff_file);
    double max_difference = reg_tools_getMaxValue(diff_file, -1);

    nifti_image_free(referenceImage);
    nifti_image_free(expectedFile);

    if (max_difference > EPS){
        fprintf(stderr, "reg_test_convolution error too large: %g (>%g)\n",
                max_difference, EPS);
        reg_io_WriteImageFile(diff_file, "diff_file.nii.gz");
        return EXIT_FAILURE;
    }
#ifndef NDEBUG
    fprintf(stdout, "reg_test_bspline_deformation_field ok: %g (<%g)\n",
            max_difference, EPS);
#endif
    nifti_image_free(diff_file);

    return EXIT_SUCCESS;
}

