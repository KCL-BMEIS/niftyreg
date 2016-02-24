#include "_reg_ReadWriteImage.h"
#include "_reg_ReadWriteMatrix.h"
#include "_reg_globalTrans.h"
#include "_reg_tools.h"

#include "Kernel.h"
#include "AffineDeformationFieldKernel.h"
#include "Platform.h"

#include "AladinContent.h"
#ifdef _USE_CUDA
#include "CUDAAladinContent.h"
#endif

#ifdef _USE_OPENCL
#include "CLAladinContent.h"
#endif

#define EPS 0.000001
#define EPS_SINGLE 0.0001

void test(AladinContent *con, int platformCode) {

    Platform *platform = new Platform(platformCode);

    Kernel *affineDeformKernel = platform->createKernel(AffineDeformationFieldKernel::getName(), con);
    affineDeformKernel->castTo<AffineDeformationFieldKernel>()->calculate();

    delete affineDeformKernel;
    delete platform;
}

int main(int argc, char **argv)
{
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <refImage> <inputMatrix> <expectedField> <platformCode>\n", argv[0]);
        return EXIT_FAILURE;
    }

    char *inputRefImageName = argv[1];
    char *inputMatFileName = argv[2];
    char *inputDefImageName = argv[3];
    int platformCode = atoi(argv[4]);

    // Read the input reference image
    nifti_image *referenceImage = reg_io_ReadImageFile(inputRefImageName);
    if (referenceImage == NULL) {
        reg_print_msg_error("The input reference image could not be read");
        return EXIT_FAILURE;
    }
    // Read the input affine matrix
    mat44 *inputMatrix = (mat44 *)malloc(sizeof(mat44));
    reg_tool_ReadAffineFile(inputMatrix, inputMatFileName);

    // Read the input deformation field image image
    nifti_image *inputDeformationField = reg_io_ReadImageFile(inputDefImageName);
    if (inputDeformationField == NULL){
        reg_print_msg_error("The input deformation field image could not be read");
        return EXIT_FAILURE;
    }
    // Check the dimension of the input images
    if (referenceImage->nx != inputDeformationField->nx ||
            referenceImage->ny != inputDeformationField->ny ||
            referenceImage->nz != inputDeformationField->nz ||
            (referenceImage->nz > 1 ? 3 : 2) != inputDeformationField->nu){
        reg_print_msg_error("The input reference and deformation field images do not have corresponding sizes");
        return EXIT_FAILURE;
    }

    // Create a deformation field
    nifti_image *test_field = nifti_copy_nim_info(inputDeformationField);
    test_field->data = (void *) malloc(test_field->nvox*test_field->nbyper);

    // Compute the affine deformation field
    AladinContent *con = NULL;
    if (platformCode == NR_PLATFORM_CPU) {
        con = new AladinContent(referenceImage, NULL, NULL, inputMatrix, sizeof(float));
    }
#ifdef _USE_CUDA
    else if (platformCode == NR_PLATFORM_CUDA) {
        con = new CudaAladinContent(referenceImage, NULL, NULL, inputMatrix, sizeof(float));
    }
#endif
#ifdef _USE_OPENCL
    else if (platformCode == NR_PLATFORM_CL) {
        con = new ClAladinContent(referenceImage, NULL, NULL, inputMatrix, sizeof(float));
    }
#endif
    else {
        reg_print_msg_error("The platform code is not suppoted");
        return EXIT_FAILURE;
    }
    //Check if the platform used is double capable
    bool isDouble = con->isCurrentComputationDoubleCapable();
    double proper_eps = EPS;
    if(isDouble == 0) {
        proper_eps = EPS_SINGLE;
    }

    //CPU or GPU code
    reg_tools_changeDatatype<float>(referenceImage);
    test(con, platformCode);
    test_field = con->getCurrentDeformationField();

    // Compute the difference between the computed and inputed deformation field
    nifti_image *diff_field = nifti_copy_nim_info(inputDeformationField);
    diff_field->data = (void *) malloc(diff_field->nvox*diff_field->nbyper);
    reg_tools_substractImageToImage(inputDeformationField, test_field, diff_field);
    reg_tools_abs_image(diff_field);
    double max_difference = reg_tools_getMaxValue(diff_field, -1);

    nifti_image_free(referenceImage);
    nifti_image_free(inputDeformationField);

    delete con;
    free(inputMatrix);

    if (max_difference > proper_eps){
        fprintf(stderr, "reg_test_affine_deformation_field error too large: %g (>%g)\n",
                max_difference, proper_eps);
        return EXIT_FAILURE;
    }
#ifndef NDEBUG
    fprintf(stdout, "reg_test_affine_deformation_field ok: %g (<%g)\n",
            max_difference, proper_eps);
#endif

    return EXIT_SUCCESS;
}

