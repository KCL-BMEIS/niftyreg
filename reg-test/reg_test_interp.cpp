/*
 *  reg_test_interp.cpp
 *
 *
 *  Created by Marc Modat on 10/05/2012.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_resampling.h"
#include "_reg_ReadWriteImage.h"

#ifdef _USE_CUDA
#define GPU_EPS 0.001
#include "_reg_resampling_gpu.h"
#endif

#define size 64

int main(int argc, char **argv)
{
    char msg[255];
    sprintf(msg,"Usage: %s dim type",argv[0]);
    if(argc!=3){
        reg_print_msg_error(msg);
        return EXIT_FAILURE;
    }
    const int dim=atoi(argv[1]);
    const int type=atoi(argv[2]);
    if(dim!=2 && dim!=3){
        reg_print_msg_error(msg);
        reg_print_msg_error("Expected value for dim are 2 and 3");
        return EXIT_FAILURE;
    }
#ifdef _USE_CUDA
    if(type!=0 && type!=1 && type!=3 && type!=4){
        reg_print_msg_error(msg);
        reg_print_msg_error("Expected value for type are 0, 1 and 3 (cpu) or 4 (gpu)");
        return EXIT_FAILURE;
    }
#else
    if(type!=0 && type!=1 && type!=3){
        reg_print_msg_error(msg);
        reg_print_msg_error("Expected value for type are 0, 1 and 3");
        return EXIT_FAILURE;
    }
#endif

    const int upRatio = 4;

    // Create a floating image
    int image_dim[8]={dim,size,size,dim==2?1:size,1,1,1,1};
    nifti_image *floating=nifti_make_new_nim(image_dim,NIFTI_TYPE_FLOAT32,true);
    reg_checkAndCorrectDimension(floating);

    // The floating image is filled with a cosine function
    float *floPtr = static_cast<float *>(floating->data);
    for(int z=0; z<floating->nz; ++z){
        for(int y=0; y<floating->ny; ++y){
            for(int x=0; x<floating->nx; ++x){
                *floPtr++=cos((float)x)*cos((float)y)*cos((float)z);
            }
        }
    }

    // Create a warped image
    image_dim[1] = (image_dim[1]-1)*upRatio;
    image_dim[2] = (image_dim[2]-1)*upRatio;
    if(dim>2)
        image_dim[3] = (image_dim[3]-1)*upRatio;
    nifti_image *warped=nifti_make_new_nim(image_dim,NIFTI_TYPE_FLOAT32,true);
    warped->pixdim[1]=warped->dx=1.f/(float)upRatio;
    warped->pixdim[2]=warped->dy=1.f/(float)upRatio;
    if(dim>2)
        warped->pixdim[3]=warped->dz=1.f/(float)upRatio;
    reg_checkAndCorrectDimension(warped);

    // Create an identity deformation field
    nifti_image *deformationField=nifti_copy_nim_info(warped);
    deformationField->dim[0]=deformationField->ndim=5;
    deformationField->dim[5]=deformationField->nu=dim;
    deformationField->nvox = (size_t)deformationField->nx *
            deformationField->ny * deformationField->nz *
            deformationField->nt * deformationField->nu ;
    deformationField->data = (void *)calloc(deformationField->nvox,
                                            deformationField->nbyper);
    reg_tools_multiplyValueToImage(deformationField,deformationField,0.f);
    reg_getDeformationFromDisplacement(deformationField);
    deformationField->intent_p1=DEF_FIELD;

    // Resample the floating image
    reg_resampleImage(floating,
                      warped,
                      deformationField,
                      NULL,
                      type>3?1:type,
                      std::numeric_limits<float>::quiet_NaN());

#ifdef _USE_CUDA
    if(type==4){
        // Check and setup the GPU card
        CUcontext ctx;
        if(cudaCommon_setCUDACard(&ctx, true))
            return EXIT_FAILURE;

		// Allocate the floating image on the device
        cudaArray *floating_device=NULL;
        NR_CUDA_SAFE_CALL(cudaCommon_allocateArrayToDevice<float>
                          (&floating_device,floating->dim))
		// Transfer the floating image on the device
        NR_CUDA_SAFE_CALL(cudaCommon_transferNiftiToArrayOnDevice<float>
						  (&floating_device,floating))

		// Allocate the warped image on the device
        float *warped_device=NULL;
        NR_CUDA_SAFE_CALL(cudaCommon_allocateArrayToDevice<float>
                          (&warped_device,warped->dim))

		// Allocate the deformation field array on the device
        float4 *deformationField_device=NULL;
        NR_CUDA_SAFE_CALL(cudaCommon_allocateArrayToDevice<float4>
                          (&deformationField_device,deformationField->dim))

		// Transfer the deformation field onto the device
        NR_CUDA_SAFE_CALL(cudaCommon_transferNiftiToArrayOnDevice<float4>
                          (&deformationField_device,deformationField))

		// Create a mask image and transfer it onto the device
		const size_t voxelNumber = warped->nvox;
        int *mask_host=(int *)malloc(voxelNumber*sizeof(int));
        for(size_t i=0; i<voxelNumber; ++i)
            mask_host[i]=i;
        int *mask_device=NULL;
        NR_CUDA_SAFE_CALL(cudaMalloc(&mask_device,voxelNumber*sizeof(int)))
        NR_CUDA_SAFE_CALL(cudaMemcpy(mask_device,mask_host,voxelNumber*sizeof(int),cudaMemcpyHostToDevice))
        free(mask_host);

		// Resample the floating image
        reg_resampleImage_gpu(floating,
                              &warped_device,
                              &floating_device,
                              &deformationField_device,
                              &mask_device,
                              warped->nx*warped->ny*warped->nz,
                              std::numeric_limits<float>::quiet_NaN());
		// Free the unncessary arrays on the device
        cudaCommon_free(&floating_device);
        cudaCommon_free(&deformationField_device);
        cudaCommon_free(&mask_device);

		// Allocate another warped image to transfer the data
		nifti_image *warped_host=nifti_copy_nim_info(warped);
		warped_host->data = (void *)malloc(warped_host->nvox *
										   warped_host->nbyper);
		// Transfer the warped image from the device to the host
        NR_CUDA_SAFE_CALL(cudaCommon_transferFromDeviceToNifti<float>
                          (warped_host,&warped_device))
        cudaCommon_free(&warped_device);
		cudaCommon_unsetCUDACard(&ctx);

        float max_diff_gpu=0.f;
        float* cpuPtr = static_cast<float *>(warped->data);
        float* gpuPtr = static_cast<float *>(warped_host->data);
        for(size_t i=0; i<voxelNumber; ++i){
            float diff = fabs(*cpuPtr++ - *gpuPtr++);
            if(diff==diff)
                max_diff_gpu = max_diff_gpu>diff?max_diff_gpu:diff;
        }
		nifti_image_free(warped_host);
        if(max_diff_gpu>GPU_EPS){
            fprintf(stderr, "Difference between CPU and GPU too high: %g > %g\n",max_diff_gpu,GPU_EPS);
            return EXIT_FAILURE;
        }
    }
#endif

    // Assess the difference
    float max_diff=0.f;
    float *warPtr = static_cast<float *>(warped->data);
    for(int z=0; z<warped->nz; ++z){
        for(int y=0; y<warped->ny; ++y){
            for(int x=0; x<warped->nx; ++x){
                float obtained = *warPtr++;
                if(obtained==obtained){
                    float expected;
                    if(type==0)
                        expected =
                                cos(float(reg_round((float)x/(float)upRatio)))*
                                cos(float(reg_round((float)y/(float)upRatio)))*
                                cos(float(reg_round((float)z/(float)upRatio)));
                    else expected =
                            cos((float)x/(float)upRatio)*
                            cos((float)y/(float)upRatio)*
                            cos((float)z/(float)upRatio);
                    float diff = fabsf(expected-obtained);
                    max_diff = max_diff>diff?max_diff:diff;
                }
            }
        }
    }

    nifti_image_free(floating);
    nifti_image_free(warped);
    nifti_image_free(deformationField);

    // Set the maximal error
    float maxError=0.f;
	if((type==1 || type==4) && dim==2)
        maxError = 0.2294f;
	if(type==3 && dim==2)
        maxError = 0.0426f;
	if((type==1 || type==4) && dim==3)
        maxError = 0.3230f;
	if(type==3 && dim==3)
        maxError = 0.0631f;

    // Check if the test failed or passed
    if(max_diff>maxError){
        fprintf(stderr, "Error: %g > %g\n",max_diff,maxError);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
