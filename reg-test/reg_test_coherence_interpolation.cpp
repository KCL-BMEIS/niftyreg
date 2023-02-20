#include "_reg_ReadWriteImage.h"
#include "_reg_resampling.h"
#include "_reg_tools.h"

#include "ResampleImageKernel.h"
#include "Platform.h"
#include "AladinContent.h"

#define EPS 0.000001
#define EPS_SINGLE 0.0001

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <refImage> <inputDefField> <order> <platformType>\n", argv[0]);
        return EXIT_FAILURE;
    }

    char *inputRefImageName = argv[1];
    char *inputDefImageName = argv[2];
    int interpolation = atoi(argv[3]);
    PlatformType platformType{ atoi(argv[4]) };

    if (platformType != PlatformType::Cuda && platformType != PlatformType::OpenCl) {
        reg_print_msg_error("Unexpected platform code");
        return EXIT_FAILURE;
    }

    // Read the input reference image
    nifti_image *referenceImage = reg_io_ReadImageFile(inputRefImageName);
    if (referenceImage == nullptr) {
        reg_print_msg_error("The input reference image could not be read");
        return EXIT_FAILURE;
    }
    reg_tools_changeDatatype<float>(referenceImage);
    // Read the input deformation field image image
    nifti_image *inputDeformationField = reg_io_ReadImageFile(inputDefImageName);
    if (inputDeformationField == nullptr) {
        reg_print_msg_error("The input deformation field image could not be read");
        return EXIT_FAILURE;
    }
    reg_tools_changeDatatype<float>(inputDeformationField);

    // Check the dimension of the input images
    if (referenceImage->nx != inputDeformationField->nx ||
        referenceImage->ny != inputDeformationField->ny ||
        referenceImage->nz != inputDeformationField->nz ||
        (referenceImage->nz > 1 ? 3 : 2) != inputDeformationField->nu) {
        reg_print_msg_error("The input reference and deformation field images do not have corresponding sizes");
        return EXIT_FAILURE;
    }

    // Initialise warped images
    nifti_image *cpuWarped = nifti_dup(*referenceImage, false);
    nifti_image *gpuWarped = nifti_dup(*referenceImage, false);

    int *tempMask = (int *)calloc(referenceImage->nvox, sizeof(int));

    // CPU platform
    unique_ptr<Platform> platformCpu{ new Platform(PlatformType::Cpu) };
    unique_ptr<AladinContent> conCpu{ new AladinContent(nullptr, referenceImage, nullptr, sizeof(float)) };
    conCpu->SetWarped(cpuWarped);
    conCpu->SetDeformationField(inputDeformationField);
    conCpu->SetReferenceMask(tempMask);
    unique_ptr<Kernel> resampleImageKernel_cpu{ platformCpu->CreateKernel(ResampleImageKernel::GetName(), conCpu) };
    resampleImageKernel_cpu->castTo<ResampleImageKernel>()->Calculate(interpolation,
                                                                      std::numeric_limits<float>::quiet_NaN());
    cpuWarped = conCpu->GetWarped();

    // GPU platform
    unique_ptr<Platform> platformGpu{ new Platform(platformType) };
    unique_ptr<AladinContentCreator> contentCreator{ dynamic_cast<AladinContentCreator*>(platformGpu->CreateContentCreator(ContentType::Aladin)) };
    unique_ptr<AladinContent> conGpu{ contentCreator->Create(nullptr, referenceImage, nullptr, sizeof(float)) };
    conGpu->SetWarped(gpuWarped);
    conGpu->SetDeformationField(inputDeformationField);
    conGpu->SetReferenceMask(tempMask);

    unique_ptr<Kernel> resampleImageKernel_gpu{ platformGpu->CreateKernel(ResampleImageKernel::GetName(), conGpu) };
    resampleImageKernel_gpu->castTo<ResampleImageKernel>()->Calculate(interpolation,
                                                                      std::numeric_limits<float>::quiet_NaN());
    gpuWarped = conGpu->GetWarped();

    //Check if the platform used is double capable
    double proper_eps = EPS;
    if (conGpu->IsCurrentComputationDoubleCapable() == 0) {
        proper_eps = EPS_SINGLE;
    }

    // Compute the difference between the warped images
    nifti_image *diff_field = nifti_dup(*referenceImage, false);

    // Compute the difference between the computed and inputted warped image
    reg_tools_subtractImageFromImage(cpuWarped, gpuWarped, diff_field);
    reg_tools_abs_image(diff_field);
    double max_difference = reg_tools_GetMaxValue(diff_field, -1);

    // free the allocated images
    nifti_image_free(referenceImage);
    nifti_image_free(cpuWarped);
    nifti_image_free(gpuWarped);
    nifti_image_free(inputDeformationField);

    if (max_difference > proper_eps) {
        fprintf(stderr, "reg_test_interpolation error too large: %g (>%g)\n",
                max_difference, proper_eps);
        return EXIT_FAILURE;
    }
#ifndef NDEBUG
    fprintf(stdout, "reg_test_interpolation ok: %g ( < %g )\n", max_difference, proper_eps);
#endif
    return EXIT_SUCCESS;
}
