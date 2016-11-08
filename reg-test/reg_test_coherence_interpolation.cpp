#include "_reg_ReadWriteImage.h"
#include "_reg_resampling.h"
#include "_reg_tools.h"

#include "ResampleImageKernel.h"
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

int main(int argc, char **argv)
{
    if(argc!=5)
    {
        fprintf(stderr, "Usage: %s <refImage> <inputDefField> <order> <platformCode>\n", argv[0]);
        return EXIT_FAILURE;
    }

    char *inputRefImageName=argv[1];
    char *inputDefImageName=argv[2];
    int interpolation=atoi(argv[3]);
    int platformCode = atoi(argv[4]);
#ifndef _USE_CUDA
   if(platformCode == NR_PLATFORM_CUDA){
      reg_print_msg_error("NiftyReg has not been compiled with CUDA");
      return EXIT_FAILURE;
   }
#endif
#ifndef _USE_OPENCL
   if(platformCode == NR_PLATFORM_CL){
      reg_print_msg_error("NiftyReg has not been compiled with OpenCL");
      return EXIT_FAILURE;
   }
#endif
   if(platformCode!=NR_PLATFORM_CUDA && platformCode!=NR_PLATFORM_CL){
      reg_print_msg_error("Unexpected platform code");
      return EXIT_FAILURE;
   }

    // Read the input reference image
    nifti_image *referenceImage = reg_io_ReadImageFile(inputRefImageName);
    if(referenceImage==NULL){
        reg_print_msg_error("The input reference image could not be read");
        return EXIT_FAILURE;
    }
    reg_tools_changeDatatype<float>(referenceImage);
    // Read the input deformation field image image
    nifti_image *inputDeformationField = reg_io_ReadImageFile(inputDefImageName);
    if(inputDeformationField==NULL){
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
    AladinContent *con_cpu = new AladinContent(NULL, referenceImage, NULL, sizeof(float));
    con_cpu->setCurrentWarped(cpu_warped);
    con_cpu->setCurrentDeformationField(inputDeformationField);
    con_cpu->setCurrentReferenceMask(tempMask, cpu_warped->nvox);
    Platform *platform_cpu = new Platform(NR_PLATFORM_CPU);
    Kernel *resampleImageKernel_cpu = platform_cpu->createKernel(ResampleImageKernel::getName(), con_cpu);
    resampleImageKernel_cpu->castTo<ResampleImageKernel>()->calculate(interpolation,
                                                                      std::numeric_limits<float>::quiet_NaN());
    delete resampleImageKernel_cpu;
    delete platform_cpu;
    cpu_warped = con_cpu->getCurrentWarped(referenceImage->datatype);

    // GPU platform
    AladinContent *con_gpu = NULL;
#ifdef _USE_CUDA
    if (platformCode == NR_PLATFORM_CUDA) {
        con_gpu = new CudaAladinContent(NULL, referenceImage, NULL, sizeof(float));
    }
#endif
#ifdef _USE_OPENCL
    if (platformCode == NR_PLATFORM_CL) {
        con_gpu = new ClAladinContent(NULL, referenceImage, NULL, sizeof(float));
    }
#endif
    con_gpu->setCurrentWarped(gpu_warped);
    con_gpu->setCurrentDeformationField(inputDeformationField);
    con_gpu->setCurrentReferenceMask(tempMask, gpu_warped->nvox);
    Platform *platform_gpu = NULL;
#ifdef _USE_CUDA
    if (platformCode == NR_PLATFORM_CUDA)
       platform_gpu = new Platform(NR_PLATFORM_CUDA);
#endif
#ifdef _USE_OPENCL
    if (platformCode == NR_PLATFORM_CL) {
       platform_gpu = new Platform(NR_PLATFORM_CL);
    }
#endif
    Kernel *resampleImageKernel_gpu = platform_gpu->createKernel(ResampleImageKernel::getName(), con_gpu);
    resampleImageKernel_gpu->castTo<ResampleImageKernel>()->calculate(interpolation,
                                                                      std::numeric_limits<float>::quiet_NaN());
    delete resampleImageKernel_gpu;
    delete platform_gpu;
    gpu_warped = con_gpu->getCurrentWarped(referenceImage->datatype);

    //Check if the platform used is double capable
    double proper_eps = EPS;
    if(con_gpu->isCurrentComputationDoubleCapable() == 0) {
        proper_eps = EPS_SINGLE;
    }

    // Compute the difference between the warped images
    nifti_image *diff_field = nifti_copy_nim_info(referenceImage);
    diff_field->data = (void *)malloc(diff_field->nvox*diff_field->nbyper);

    // Compute the difference between the computed and inputed warped image
    reg_tools_substractImageToImage(cpu_warped, gpu_warped, diff_field);
    reg_tools_abs_image(diff_field);
    double max_difference = reg_tools_getMaxValue(diff_field, -1);

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
