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

void test(AladinContent *con, const unsigned int interp, int platformCode) {

    Platform *platform = new Platform(platformCode);

    Kernel *resampleImageKernel = platform->createKernel(ResampleImageKernel::getName(), con);
    resampleImageKernel->castTo<ResampleImageKernel>()->calculate(interp, std::numeric_limits<float>::quiet_NaN());
    //resampleImageKernel->castTo<ResampleImageKernel>()->calculate(interp, 0);

    delete resampleImageKernel;
    delete platform;
}


int main(int argc, char **argv)
{
    if(argc!=6)
    {
        fprintf(stderr, "Usage: %s <floImage> <inputDefField> <expectedWarpedImage> <order> <platformCode>\n", argv[0]);
        return EXIT_FAILURE;
    }

    char *inputfloatingImageName=argv[1];
    char *inputDefImageName=argv[2];
    char *inputWarpedImageName=argv[3];
    int interpolation=atoi(argv[4]);
    int platformCode = atoi(argv[5]);

    // Read the input floating image
    nifti_image *floatingImage = reg_io_ReadImageFile(inputfloatingImageName);
    if(floatingImage==NULL){
        reg_print_msg_error("The input floating image could not be read");
        return EXIT_FAILURE;
    }
    reg_tools_changeDatatype<float>(floatingImage);
    // Read the input deformation field image image
    nifti_image *inputDeformationField = reg_io_ReadImageFile(inputDefImageName);
    if(inputDeformationField==NULL){
        reg_print_msg_error("The input deformation field image could not be read");
        return EXIT_FAILURE;
    }
    reg_tools_changeDatatype<float>(inputDeformationField);
    // Read the input reference image
    nifti_image *warpedImage = reg_io_ReadImageFile(inputWarpedImageName);
    if(warpedImage==NULL){
        reg_print_msg_error("The input warped image could not be read");
        return EXIT_FAILURE;
    }
    //The expected warped image contains NaN, let's change them to quietNaN to be consitent
    reg_tools_changeDatatype<float>(warpedImage);
    float* wid = (float*) warpedImage->data;
    for (size_t i = 0; i < warpedImage->nvox; i++) {
        if (wid[i] != wid[i]) {
            wid[i] = std::numeric_limits<float>::quiet_NaN();
        }
    }

    // Check the dimension of the input images
    if(warpedImage->nx != inputDeformationField->nx ||
            warpedImage->ny != inputDeformationField->ny ||
            warpedImage->nz != inputDeformationField->nz ||
            (warpedImage->nz>1?3:2) != inputDeformationField->nu){
        reg_print_msg_error("The input warped and deformation field images do not have corresponding sizes");
        return EXIT_FAILURE;
    }
    if((floatingImage->nz>1) != (warpedImage->nz>1) ||
            floatingImage->nt != warpedImage->nt){
        reg_print_msg_error("The input floating and warped images do not have corresponding sizes");
        return EXIT_FAILURE;
    }

    // Initialize a deformation field image
    nifti_image *test_warped=nifti_copy_nim_info(warpedImage);
    test_warped->data=(void *)malloc(test_warped->nvox*test_warped->nbyper);
    //test_warped->data = (void *)calloc(test_warped->nvox, sizeof(float));

    //CPU - GPU code
    int *tempMask = (int *)calloc(test_warped->nvox, sizeof(int));

    AladinContent *con = NULL;
    if (platformCode == NR_PLATFORM_CPU) {
        con = new AladinContent(NULL, floatingImage, NULL, sizeof(float));
    }
#ifdef _USE_CUDA
    else if (platformCode == NR_PLATFORM_CUDA) {
        con = new CudaAladinContent(NULL, floatingImage, NULL, sizeof(float));
    }
#endif
#ifdef _USE_OPENCL
    else if (platformCode == NR_PLATFORM_CL) {
        con = new ClAladinContent(NULL, floatingImage, NULL, sizeof(float));
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

    con->setCurrentWarped(test_warped);
    con->setCurrentDeformationField(inputDeformationField);
    con->setCurrentReferenceMask(tempMask, test_warped->nvox);

    test(con, interpolation, platformCode);
    test_warped = con->getCurrentWarped(warpedImage->datatype);//check

    // Compute the difference between the computed and inputed deformation field
    nifti_image *diff_field = nifti_copy_nim_info(test_warped);
    diff_field->data = (void *)malloc(diff_field->nvox*diff_field->nbyper);

    // Compute the difference between the computed and inputed warped image
    reg_tools_substractImageToImage(warpedImage, test_warped, diff_field);
    reg_tools_abs_image(diff_field);
    double max_difference = reg_tools_getMaxValue(diff_field, -1);

#ifndef NDEBUG
    if (max_difference > proper_eps) {
        const char* tmpdir = getenv("TMPDIR");
        char filename[255];
        if(tmpdir!=NULL)
            sprintf(filename, "%s/difference_warp_%iD_%i.nii", tmpdir, (diff_field->nz>1 ? 3 : 2), interpolation);
        else sprintf(filename, "./difference_warp_%iD_%i.nii", (diff_field->nz>1 ? 3 : 2), interpolation);
        reg_io_WriteImageFile(diff_field, filename);
        reg_print_msg_error("Saving temp warped image:");
        reg_print_msg_error(filename);
    }
#endif

    nifti_image_free(floatingImage);
    nifti_image_free(warpedImage);
    nifti_image_free(inputDeformationField);
    nifti_image_free(test_warped);

    if(max_difference>proper_eps){
        fprintf(stderr, "reg_test_interpolation error too large: %g (>%g)\n",
                max_difference, proper_eps);
        return EXIT_FAILURE;
    }
#ifndef NDEBUG
    fprintf(stdout, "reg_test_interpolation ok: %g ( <%g )\n", max_difference, proper_eps);
#endif
    return EXIT_SUCCESS;
}
