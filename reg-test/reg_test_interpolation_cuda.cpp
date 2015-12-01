#include "_reg_ReadWriteImage.h"
#include "_reg_resampling.h"
#include "_reg_tools.h"

#include "Kernel.h"
//#include "Kernels.h"
#include "ResampleImageKernel.h"
#include "Platform.h"
#include "cuda/CudaContent.h"

#define EPS 0.000001

void test(Content *con, const unsigned int interp) {

	Platform *cudaPlatform = new Platform(NR_PLATFORM_CUDA);
	Kernel* resamplingKernel = cudaPlatform->createKernel(ResampleImageKernel::getName(), con);

	//run kernel
	resamplingKernel->castTo<ResampleImageKernel>()->calculate(interp, 0);

	delete resamplingKernel;
	delete cudaPlatform;
}

void flag(std::string msg, std::string flag) {
	std::cout << msg << ": " << flag << std::endl;
}
int main(int argc, char **argv) {
	if (argc != 5) {
		fprintf(stderr, "Usage: %s <floImage> <inputDefField> <expectedWarpedImage> <order>\n", argv[0]);
		return EXIT_FAILURE;
	}

	char *inputfloatingImageName = argv[1];
	char *inputDefImageName = argv[2];
	char *inputWarpedImageName = argv[3];
	int interpolation = atoi(argv[4]);

	// Read the input floating image
	nifti_image *floatingImage = reg_io_ReadImageFile(inputfloatingImageName);
	if (floatingImage == NULL) {
		reg_print_msg_error("The input floating image could not be read");
		return EXIT_FAILURE;
	}
	// Read the input deformation field image image
	nifti_image *inputDeformationField = reg_io_ReadImageFile(inputDefImageName);
	if (inputDeformationField == NULL) {
		reg_print_msg_error("The input deformation field image could not be read");
		return EXIT_FAILURE;
	}
	// Read the input reference image
	nifti_image *warpedImage = reg_io_ReadImageFile(inputWarpedImageName);
	if (warpedImage == NULL) {
		reg_print_msg_error("The input warped image could not be read");
		return EXIT_FAILURE;
	}
	// Check the dimension of the input images
	if (warpedImage->nx != inputDeformationField->nx || warpedImage->ny != inputDeformationField->ny || warpedImage->nz != inputDeformationField->nz || (warpedImage->nz > 1 ? 3 : 2) != inputDeformationField->nu) {
		reg_print_msg_error("The input warped and deformation field images do not have corresponding sizes");
		return EXIT_FAILURE;
	}
	if ((floatingImage->nz > 1) != (warpedImage->nz > 1) || floatingImage->nt != warpedImage->nt) {
		reg_print_msg_error("The input floating and warped images do not have corresponding sizes");
		return EXIT_FAILURE;
	}

	// Create a deformation field
	nifti_image *test_warped = nifti_copy_nim_info(warpedImage);
	test_warped->data = (void *) malloc(test_warped->nvox * test_warped->nbyper);

	// Compute the non-linear deformation field
	int* tempMask = (int *) calloc(test_warped->nvox, sizeof(int));
	reg_tools_changeDatatype<float>(test_warped);

	Content *con = new CudaContent(NULL, floatingImage, NULL, sizeof(float));
	con->setCurrentWarped(test_warped);
	con->setCurrentDeformationField(inputDeformationField);
	con->setCurrentReferenceMask(tempMask, test_warped->nvox);

	test(con, interpolation);
	test_warped = con->getCurrentWarped(warpedImage->datatype);

	// Compute the difference between the computed and inputed warped image
	reg_tools_substractImageToImage(warpedImage, test_warped, test_warped);
	reg_tools_abs_image(test_warped);
	double max_difference = reg_tools_getMaxValue(test_warped);

#ifndef NDEBUG
	if (max_difference > EPS) {
		const char* tmpdir = getenv("TMPDIR");
		char filename[255];
		if(tmpdir!=NULL)
			sprintf(filename,"%s/difference_wapr_cuda_%i.nii", tmpdir, interpolation);
		else sprintf(filename,"./difference_warp_cuda_%i.nii", interpolation);
		reg_io_WriteImageFile(test_warped,filename);
		reg_print_msg_error("Saving temp warped image:");
		reg_print_msg_error(filename);
	}
#endif

	nifti_image_free(floatingImage);
	nifti_image_free(warpedImage);
	//they are freed in context
	/*nifti_image_free(inputDeformationField);
	nifti_image_free(test_warped);*/

	free(tempMask);
	delete con;

//	std::cout<<"max diff: "<<max_difference<<std::endl;
	if (max_difference > EPS) {
		fprintf(stderr, "reg_test_interpolation_cuda error too large: %g (>%g)\n", max_difference, EPS);
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
