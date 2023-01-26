#include "_reg_ReadWriteImage.h"
#include "_reg_resampling.h"
#include "_reg_tools.h"

#include "ResampleImageKernel.h"
#include "Platform.h"
#include "AladinContent.h"
#ifdef _USE_CUDA
#include "CudaAladinContent.h"
#endif
#ifdef _USE_OPENCL
#include "ClAladinContent.h"
#endif
#define EPS 0.000001
#define EPS_SINGLE 0.0001

int main(int argc, char **argv)
{
    if(argc!=5)
    {
        fprintf(stderr, "Usage: %s <refImage> <inputDefField> <order> <platformType>\n", argv[0]);
        return EXIT_FAILURE;
    }

    char *inputRefImageName=argv[1];
    char *inputDefImageName=argv[2];
    int interpolation=atoi(argv[3]);
    PlatformType platformType{atoi(argv[4])};
#ifndef _USE_CUDA
   if(platformType == PlatformType::Cuda){
      reg_print_msg_error("NiftyReg has not been compiled with CUDA");
      return EXIT_FAILURE;
   }
#endif
#ifndef _USE_OPENCL
   if(platformType == PlatformType::OpenCl){
      reg_print_msg_error("NiftyReg has not been compiled with OpenCL");
      return EXIT_FAILURE;
   }
#endif
   if(platformType!=PlatformType::Cuda && platformType!=PlatformType::OpenCl){
      reg_print_msg_error("Unexpected platform code");
      return EXIT_FAILURE;
   }

    // Read the input reference image
    nifti_image *referenceImage = reg_io_ReadImageFile(inputRefImageName);
    if(referenceImage==nullptr){
        reg_print_msg_error("The input reference image could not be read");
        return EXIT_FAILURE;
    }
    reg_tools_changeDatatype<float>(referenceImage);
    // Read the input deformation field image image
    nifti_image *inputDeformationField = reg_io_ReadImageFile(inputDefImageName);
    if(inputDeformationField==nullptr){
        reg_print_msg_error("The input deformation field image could not be read");
        return EXIT_FAILURE;
    }
    reg_tools_changeDatatype<float>(inputDeformationField);

    // Check the dimension of the input images
    if(referenceImage->nx != inputDeformationField->nx ||
            referenceImage->ny != inputDeformationField->ny ||
            referenceImage->nz != inputDeformationField->nz ||
            (referenceImage->nz>1?3:2) != inputDeformationField->nu){
        reg_print_msg_error("The input reference and deformation field images do not have corresponding sizes");
        return EXIT_FAILURE;
    }

    // Initialise warped images
    nifti_image *cpu_warped=nifti_copy_nim_info(referenceImage);
    cpu_warped->data=(void *)malloc(cpu_warped->nvox*cpu_warped->nbyper);
    nifti_image *gpu_warped=nifti_copy_nim_info(referenceImage);
    gpu_warped->data=(void *)malloc(gpu_warped->nvox*gpu_warped->nbyper);

    int *tempMask = (int *)calloc(referenceImage->nvox, sizeof(int));

    // CPU platform
    AladinContent *con_cpu = new AladinContent(nullptr, referenceImage, nullptr, sizeof(float));
    con_cpu->SetWarped(cpu_warped);
    con_cpu->SetDeformationField(inputDeformationField);
    con_cpu->SetReferenceMask(tempMask);
    Platform *platform_cpu = new Platform(PlatformType::Cpu);
    Kernel *resampleImageKernel_cpu = platform_cpu->CreateKernel(ResampleImageKernel::GetName(), con_cpu);
    resampleImageKernel_cpu->castTo<ResampleImageKernel>()->Calculate(interpolation,
                                                                      std::numeric_limits<float>::quiet_NaN());
    delete resampleImageKernel_cpu;
    delete platform_cpu;
    cpu_warped = con_cpu->GetWarped();

    // GPU platform
    AladinContent *con_gpu = nullptr;
#ifdef _USE_CUDA
    if (platformType == PlatformType::Cuda) {
        con_gpu = new CudaAladinContent(nullptr, referenceImage, nullptr, sizeof(float));
    }
#endif
#ifdef _USE_OPENCL
    if (platformType == PlatformType::OpenCl) {
        con_gpu = new ClAladinContent(nullptr, referenceImage, nullptr, sizeof(float));
    }
#endif
    con_gpu->SetWarped(gpu_warped);
    con_gpu->SetDeformationField(inputDeformationField);
    con_gpu->SetReferenceMask(tempMask);
    Platform *platform_gpu = nullptr;
#ifdef _USE_CUDA
    if (platformType == PlatformType::Cuda)
       platform_gpu = new Platform(PlatformType::Cuda);
#endif
#ifdef _USE_OPENCL
    if (platformType == PlatformType::OpenCl) {
       platform_gpu = new Platform(PlatformType::OpenCl);
    }
#endif
    Kernel *resampleImageKernel_gpu = platform_gpu->CreateKernel(ResampleImageKernel::GetName(), con_gpu);
    resampleImageKernel_gpu->castTo<ResampleImageKernel>()->Calculate(interpolation,
                                                                      std::numeric_limits<float>::quiet_NaN());
    delete resampleImageKernel_gpu;
    delete platform_gpu;
    gpu_warped = con_gpu->GetWarped();

    //Check if the platform used is double capable
    double proper_eps = EPS;
    if(con_gpu->IsCurrentComputationDoubleCapable() == 0) {
        proper_eps = EPS_SINGLE;
    }

    // Compute the difference between the warped images
    nifti_image *diff_field = nifti_copy_nim_info(referenceImage);
    diff_field->data = (void *)malloc(diff_field->nvox*diff_field->nbyper);

    // Compute the difference between the computed and inputed warped image
    reg_tools_subtractImageFromImage(cpu_warped, gpu_warped, diff_field);
    reg_tools_abs_image(diff_field);
    double max_difference = reg_tools_GetMaxValue(diff_field, -1);

    // free the allocated images
    nifti_image_free(referenceImage);
    nifti_image_free(cpu_warped);
    nifti_image_free(gpu_warped);
    nifti_image_free(inputDeformationField);

    if(max_difference>proper_eps){
        fprintf(stderr, "reg_test_interpolation error too large: %g (>%g)\n",
                max_difference, proper_eps);
        return EXIT_FAILURE;
    }
#ifndef NDEBUG
    fprintf(stdout, "reg_test_interpolation ok: %g ( < %g )\n", max_difference, proper_eps);
#endif
    return EXIT_SUCCESS;
}
