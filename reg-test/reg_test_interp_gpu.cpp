#include "_reg_resampling.h"
#include "_reg_resampling_gpu.h"
#include "_reg_tools.h"

#include "_reg_common_gpu.h"

#include <limits.h>

#define EPS 0.001

int main(int argc, char **argv)
{
    if(argc!=5){
        fprintf(stderr, "Usage: %s <floatingImage> <deformationField> <dim> <type>\n", argv[0]);
        fprintf(stderr, "<dim>\tImages dimension (2,3)\n");
        fprintf(stderr, "<type>\tTest type:\n");
        fprintf(stderr, "\t\t- linear interpolation (\"lin\")\n");
        fprintf(stderr, "\t\t- linear interpolation gradient (\"lgrad\")\n");
        return EXIT_FAILURE;
    }
    int dimension=atoi(argv[3]);
    char *type=argv[4];

    // Check and setup the GPU card
    CUcontext ctx;
    if(cudaCommon_setCUDACard(&ctx, true))
        return EXIT_FAILURE;

    // Read the input floating image and converts it to a float
    nifti_image *floatingImage = nifti_image_read(argv[1],true);
    if(floatingImage==NULL){
        fprintf(stderr, "Error when reading the floating image: %s\n", argv[1]);
        return EXIT_FAILURE;
    }
    reg_tools_changeDatatype<float>(floatingImage);

    // Read the input deformation field image and converts it to a float
    nifti_image *deformationField = nifti_image_read(argv[2],true);
    if(deformationField==NULL){
        fprintf(stderr, "Error when reading the deformation field image: %s\n", argv[2]);
        return EXIT_FAILURE;
    }
    reg_tools_changeDatatype<float>(deformationField);

    // Create the floating image array onto the device
    cudaArray *floatingImage_device=NULL;
    NR_CUDA_SAFE_CALL(cudaCommon_allocateArrayToDevice<float>(&floatingImage_device,floatingImage->dim))
    NR_CUDA_SAFE_CALL(cudaCommon_transferNiftiToArrayOnDevice<float>(&floatingImage_device,floatingImage))

    // Create the deformation field array onto the device
    float4 *deformationField_device=NULL;
    NR_CUDA_SAFE_CALL(cudaCommon_allocateArrayToDevice<float4>(&deformationField_device,deformationField->dim))
    NR_CUDA_SAFE_CALL(cudaCommon_transferNiftiToArrayOnDevice<float4>(&deformationField_device,deformationField))

    // Create a mask on both the host and the device
    int *mask_host=(int *)malloc(floatingImage->nx*floatingImage->ny*floatingImage->nz*sizeof(int));
    for(size_t i=0; i<floatingImage->nx*floatingImage->ny*floatingImage->nz; ++i)
        mask_host[i]=i;
    int *mask_device=NULL;
    NR_CUDA_SAFE_CALL(cudaMalloc(&mask_device,floatingImage->nx*floatingImage->ny*floatingImage->nz*sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpy(mask_device,mask_host,floatingImage->nx*floatingImage->ny*floatingImage->nz*sizeof(int),cudaMemcpyHostToDevice))

    float maxDifferenceLIN=0;
    float maxDifferenceLGRAD=0;

    if(strcmp(type,"lin")==0){
        // Create a warped image on the host
        nifti_image *warpedImage=nifti_copy_nim_info(deformationField);
        warpedImage->dim[0]=warpedImage->ndim=dimension;
        warpedImage->nu=warpedImage->dim[5]=1;
        warpedImage->nvox=warpedImage->nx*warpedImage->ny*warpedImage->nz*warpedImage->nt;
        warpedImage->data=(void *)malloc(warpedImage->nvox*warpedImage->nbyper);
        // Create a warped image on the device
        float *warpedImage_device=NULL;
        NR_CUDA_SAFE_CALL(cudaCommon_allocateArrayToDevice<float>(&warpedImage_device,warpedImage->dim))
        // Warp the floating image using the CPU
        reg_resampleImage(floatingImage,
                          warpedImage,
                          deformationField,
                          mask_host,
                          1,
                          std::numeric_limits<float>::quiet_NaN());
        // Warp the floating image using the GPU
        reg_resampleImage_gpu(floatingImage,
                              &warpedImage_device,
                              &floatingImage_device,
                              &deformationField_device,
                              &mask_device,
                              warpedImage->nx*warpedImage->ny*warpedImage->nz,
                              std::numeric_limits<float>::quiet_NaN());
        // Transfer the warped floating image from the device to the host
        nifti_image *warpedImage_temp=nifti_copy_nim_info(warpedImage);
        warpedImage_temp->data=(void *)malloc(warpedImage_temp->nvox*warpedImage_temp->nbyper);
        NR_CUDA_SAFE_CALL(cudaCommon_transferFromDeviceToNifti<float>(warpedImage_temp,&warpedImage_device))
        // Compute the difference between both warped image
        maxDifferenceLIN=reg_test_compare_images(warpedImage,warpedImage_temp);
//#ifndef NDEBUG
        printf("[NiftyReg DEBUG] [dim=%i] Linear interpolation difference: %g\n",
               dimension,
               maxDifferenceLIN);
        nifti_set_filenames(warpedImage,"reg_test_interp_cpu.nii",0,0);
        nifti_set_filenames(warpedImage_temp,"reg_test_interp_gpu.nii",0,0);
        nifti_image_write(warpedImage);
        nifti_image_write(warpedImage_temp);
        reg_tools_substractImageToImage(warpedImage,warpedImage_temp,warpedImage);
        reg_tools_divideValueToImage(warpedImage,warpedImage,255.f);
        nifti_set_filenames(warpedImage,"reg_test_interp_diff.nii",0,0);
        nifti_image_write(warpedImage);
//#endif
        // Free the allocated images
        nifti_image_free(warpedImage);
        nifti_image_free(warpedImage_temp);
        cudaCommon_free(&warpedImage_device);
    }
    else if(strcmp(type,"lgrad")==0){
        // Create a warped image gradient on the host
        nifti_image *warpedImageGradient=nifti_copy_nim_info(deformationField);
        warpedImageGradient->data=(void *)malloc(warpedImageGradient->nvox*warpedImageGradient->nbyper);
        // Create a warped image on the device
        float4 *warpedImageGradient_device=NULL;
        NR_CUDA_SAFE_CALL(cudaCommon_allocateArrayToDevice<float4>(&warpedImageGradient_device,warpedImageGradient->dim))
        // Warp the floating image using the CPU
        reg_getImageGradient(floatingImage,
                             warpedImageGradient,
                             deformationField,
                             mask_host,
                             1,
                             std::numeric_limits<float>::quiet_NaN());
        // Warp the floating image using the GPU
        reg_getImageGradient_gpu(floatingImage,
                                 &floatingImage_device,
                                 &deformationField_device,
                                 &warpedImageGradient_device,
                                 warpedImageGradient->nx*warpedImageGradient->ny*warpedImageGradient->nz,
                                 std::numeric_limits<float>::quiet_NaN());
        // Transfer the warped floating image from the device to the host
        nifti_image *warpedImageGradient_temp=nifti_copy_nim_info(warpedImageGradient);
        warpedImageGradient_temp->data=(void *)malloc(warpedImageGradient_temp->nvox*warpedImageGradient_temp->nbyper);
        NR_CUDA_SAFE_CALL(cudaCommon_transferFromDeviceToNifti<float4>(warpedImageGradient_temp,&warpedImageGradient_device))
        // Compute the difference between both warped image
        maxDifferenceLIN=reg_test_compare_images(warpedImageGradient,warpedImageGradient_temp);
//#ifndef NDEBUG
        printf("[NiftyReg DEBUG] [dim=%i] Gradient computation through linear interpolation difference: %g\n",
               dimension,
               maxDifferenceLIN);
        nifti_set_filenames(warpedImageGradient,"reg_test_interp_cpu.nii",0,0);
        nifti_set_filenames(warpedImageGradient_temp,"reg_test_interp_gpu.nii",0,0);
        nifti_image_write(warpedImageGradient);
        nifti_image_write(warpedImageGradient_temp);
        reg_tools_divideImageToImage(warpedImageGradient,warpedImageGradient_temp,warpedImageGradient);
        reg_tools_substractValueToImage(warpedImageGradient,warpedImageGradient,1.f);
        nifti_set_filenames(warpedImageGradient,"reg_test_interp_diff.nii",0,0);
        nifti_image_write(warpedImageGradient);
//#endif
        // Free the allocated images
        nifti_image_free(warpedImageGradient);
        nifti_image_free(warpedImageGradient_temp);
        cudaCommon_free(&warpedImageGradient_device);
    }

    // Clean the allocated arrays
    nifti_image_free(floatingImage);
    nifti_image_free(deformationField);
    free(mask_host);
    cudaCommon_free(&floatingImage_device);
    cudaCommon_free(&deformationField_device);
    cudaCommon_free(&mask_device);

    cudaCommon_unsetCUDACard(&ctx);

    if(maxDifferenceLIN>EPS){
        fprintf(stderr,
                "[dim=%i] Linear interpolation difference too high: %g\n",
                dimension,
                maxDifferenceLIN);
        return EXIT_FAILURE;
    }
    else if(maxDifferenceLGRAD>EPS){
        fprintf(stderr,
                "[dim=%i]  Gradient computation through linear interpolation difference too high: %g\n",
                dimension,
                maxDifferenceLIN);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
