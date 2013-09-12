#include "_reg_tools.h"
#include "_reg_tools_gpu.h"

#include "_reg_common_gpu.h"

#define EPS 0.001
#define SIZE 128

int main(int argc, char **argv)
{
	if(argc!=3){
		fprintf(stderr, "Usage: %s <dim> <type>\n", argv[0]);
		fprintf(stderr, "<dim>\tImages dimension (2,3)\n");
		fprintf(stderr, "<type>\tTest type:\n");
		fprintf(stderr, "\t\t- Gaussian kernel convolution (\"gaussian\")\n");
		fprintf(stderr, "\t\t- Cubic spline kernel convolution (\"spline\")\n");
		return EXIT_FAILURE;
	}
	int dimension=atoi(argv[1]);
	char *type=argv[2];

    // Check and setup the GPU card
    CUcontext ctx;
    if(cudaCommon_setCUDACard(&ctx, true))
		return EXIT_FAILURE;

	// Create the input images
	int nii_dim[8]={5,SIZE,SIZE,SIZE,1,dimension,1,1};
	if(dimension==2)
		nii_dim[3]=1;
	nifti_image *input = nifti_make_new_nim(nii_dim, NIFTI_TYPE_FLOAT32, true);
	reg_checkAndCorrectDimension(input);
	// Fill the input images with random value
	float *inputPtr=static_cast<float *>(input->data);
	for(size_t i=0; i<input->nvox; ++i){
		inputPtr[i] = (float)rand()/(float)RAND_MAX;
	}

	// Allocate the image image on the GPU
	float4 *input_gpu=NULL;
	cudaCommon_allocateArrayToDevice<float4>(&input_gpu, input->dim);
	cudaCommon_transferNiftiToArrayOnDevice<float4>(&input_gpu, input);

	float maxDifferenceGAUSSIAN=0;
	float maxDifferenceSPLINE=0;

	if(strcmp(type,"gaussian")==0){
		// First define the size of the spline kernel to use
        // Gaussian convolution on the CPU
        float test[3]={5,5,5};
        reg_tools_kernelConvolution(input,
                                    test,
                                    0 // Gaussian kernel
                                    );
		// Gaussian convolution on the GPU
		bool axisToSmooth[8]={1,1,1,1,1,1,1,1};
		reg_gaussianSmoothing_gpu(input,
								  &input_gpu,
								  5.f,
								  axisToSmooth);

		nifti_image *tempImage = nifti_make_new_nim(nii_dim, NIFTI_TYPE_FLOAT32, true);
		reg_checkAndCorrectDimension(tempImage);
		cudaCommon_transferFromDeviceToNifti<float4>(tempImage, &input_gpu);
        maxDifferenceGAUSSIAN=reg_test_compare_images(input,tempImage);

		char name[255];
		sprintf(name, "gaussian_%i_cpu.nii",dimension);
		nifti_set_filenames(input,name,0,0);
		nifti_image_write(input);
		sprintf(name, "gaussian_%i_gpu.nii",dimension);
		nifti_set_filenames(tempImage,name,0,0);
		nifti_image_write(tempImage);
		printf("ERROR GAUSSIAN %i: %g\n",dimension,maxDifferenceGAUSSIAN);

		nifti_image_free(tempImage);
	}
	else if(strcmp(type,"spline")==0){
		// First define the size of the spline kernel to use
        float test[3]={5,5,5};
        // Spline convolution on the CPU
        reg_tools_kernelConvolution(input,
                                    test,
                                    1 // Cubic spline kernel
                                    );
		// Spline convolution on the GPU
		reg_smoothImageForCubicSpline_gpu(input,
										  &input_gpu,
                                          test);
		// The GPU result is transfered on the host
		nifti_image *tempImage = nifti_make_new_nim(nii_dim, NIFTI_TYPE_FLOAT32, true);
		reg_checkAndCorrectDimension(tempImage);
		cudaCommon_transferFromDeviceToNifti<float4>(tempImage, &input_gpu);
        maxDifferenceSPLINE=reg_test_compare_images(input,tempImage);

		char name[255];
		sprintf(name, "spline_%i_cpu.nii",dimension);
		nifti_set_filenames(input,name,0,0);
		nifti_image_write(input);
		sprintf(name, "spline_%i_gpu.nii",dimension);
		nifti_set_filenames(tempImage,name,0,0);
		nifti_image_write(tempImage);
		printf("ERROR SPLINE %i: %g\n",dimension,maxDifferenceSPLINE);

		nifti_image_free(tempImage);
	}

    // Clean the allocate arrays and images
    nifti_image_free(input);
    cudaCommon_free(&input_gpu);
    cudaCommon_unsetCUDACard(&ctx);

	if(maxDifferenceGAUSSIAN>EPS){
		fprintf(stderr,
				"[dim=%i] Gaussian difference too high: %g\n",
				dimension,
				maxDifferenceGAUSSIAN);
		return EXIT_FAILURE;
	}
	else if(maxDifferenceSPLINE>EPS){
		fprintf(stderr,
				"[dim=%i] Spline difference too high: %g\n",
				dimension,
				maxDifferenceSPLINE);
		return EXIT_FAILURE;
    }

	return EXIT_SUCCESS;
}

