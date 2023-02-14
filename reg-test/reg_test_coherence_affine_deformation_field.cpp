#include "_reg_ReadWriteImage.h"
#include "_reg_ReadWriteMatrix.h"
#include "_reg_globalTrans.h"
#include "_reg_tools.h"

#include "Kernel.h"
#include "AffineDeformationFieldKernel.h"
#include "Platform.h"
#include "AladinContent.h"

#define EPS 0.000001
#define EPS_SINGLE 0.0001

void test(AladinContent *con, Platform *platform) {
    unique_ptr<Kernel> affineDeformKernel{ platform->CreateKernel(AffineDeformationFieldKernel::GetName(), con) };
    affineDeformKernel->castTo<AffineDeformationFieldKernel>()->Calculate();
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <refImage> <inputMatrix> <expectedField> <platformType>\n", argv[0]);
        return EXIT_FAILURE;
    }

    char *inputRefImageName = argv[1];
    char *inputMatFileName = argv[2];
    char *inputDefImageName = argv[3];
    PlatformType platformType{ atoi(argv[4]) };

    // Read the input reference image
    nifti_image *referenceImage = reg_io_ReadImageFile(inputRefImageName);
    if (referenceImage == nullptr) {
        reg_print_msg_error("The input reference image could not be read");
        return EXIT_FAILURE;
    }
    // Read the input affine matrix
    mat44 *inputMatrix = (mat44 *)malloc(sizeof(mat44));
    reg_tool_ReadAffineFile(inputMatrix, inputMatFileName);

    // Read the input deformation field image image
    nifti_image *inputDeformationField = reg_io_ReadImageFile(inputDefImageName);
    if (inputDeformationField == nullptr) {
        reg_print_msg_error("The input deformation field image could not be read");
        return EXIT_FAILURE;
    }
    // Check the dimension of the input images
    if (referenceImage->nx != inputDeformationField->nx ||
        referenceImage->ny != inputDeformationField->ny ||
        referenceImage->nz != inputDeformationField->nz ||
        (referenceImage->nz > 1 ? 3 : 2) != inputDeformationField->nu) {
        reg_print_msg_error("The input reference and deformation field images do not have corresponding sizes");
        return EXIT_FAILURE;
    }

    // Create a deformation field
    nifti_image *test_field_cpu = nifti_copy_nim_info(inputDeformationField);
    test_field_cpu->data = malloc(test_field_cpu->nvox * test_field_cpu->nbyper);

    nifti_image *test_field_gpu = nifti_copy_nim_info(inputDeformationField);
    test_field_gpu->data = malloc(test_field_gpu->nvox * test_field_gpu->nbyper);

    // Compute the affine deformation field
    unique_ptr<Platform> platformCpu{ new Platform(PlatformType::Cpu) };
    unique_ptr<AladinContent> conCpu{ new AladinContent(referenceImage, nullptr, nullptr, inputMatrix, sizeof(float)) };
    unique_ptr<Platform> platformGpu{ new Platform(platformType) };
    unique_ptr<AladinContentCreator> contentCreator{ dynamic_cast<AladinContentCreator*>(platformGpu->CreateContentCreator(ContentType::Aladin)) };
    unique_ptr<AladinContent> conGpu{ contentCreator->Create(referenceImage, nullptr, nullptr, inputMatrix, sizeof(float)) };

    //Check if the platform used is double capable
    bool isDouble = conGpu->IsCurrentComputationDoubleCapable();
    double proper_eps = EPS;
    if (isDouble == 0) {
        proper_eps = EPS_SINGLE;
    }

    //CPU or GPU code
    reg_tools_changeDatatype<float>(referenceImage);
    test(conCpu.get(), platformCpu.get());
    test_field_cpu = conCpu->GetDeformationField();

    test(conGpu.get(), platformGpu.get());
    test_field_gpu = conGpu->GetDeformationField();

    // Compute the difference between the computed and inputted deformation field
    nifti_image *diff_field = nifti_copy_nim_info(inputDeformationField);
    diff_field->data = malloc(diff_field->nvox * diff_field->nbyper);
    reg_tools_subtractImageFromImage(inputDeformationField, test_field_cpu, diff_field);
    reg_tools_abs_image(diff_field);
    double max_difference = reg_tools_GetMaxValue(diff_field, -1);

    nifti_image_free(referenceImage);
    nifti_image_free(inputDeformationField);
    free(inputMatrix);

    if (max_difference > proper_eps) {
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
