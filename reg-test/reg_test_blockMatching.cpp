#include "_reg_ReadWriteImage.h"
#include "_reg_ReadWriteMatrix.h"
#include "_reg_blockMatching.h"
#include "_reg_tools.h"
#include "_reg_globalTrans.h"

#include "BlockMatchingKernel.h"
#include "Platform.h"

#include "Content.h"
#ifdef _USE_CUDA
#include "CudaContent.h"
#endif
#ifdef _USE_OPENCL
#include "CLContent.h"
#endif

#define EPS 0.000001

void test(Content *con, int platformCode) {

    Platform *platform = new Platform(platformCode);

    Kernel *blockMatchingKernel = platform->createKernel(BlockMatchingKernel::getName(), con);
    blockMatchingKernel->castTo<BlockMatchingKernel>()->calculate();

    delete blockMatchingKernel;
    delete platform;
}

int main(int argc, char **argv)
{

    if (argc != 5) {
        fprintf(stderr, "Usage: %s <refImage> <warpedImage> <expectedBlockMatchingMatrix> <platformCode>\n", argv[0]);
        return EXIT_FAILURE;
    }

    char *inputRefImageName = argv[1];
    char *inputWarpedImageName = argv[2];
    char* expectedBlockMatchingMatrixName = argv[3];
    int   platformCode = atoi(argv[4]);

    // Read the input reference image
    nifti_image *referenceImage = reg_io_ReadImageFile(inputRefImageName);
    if (referenceImage == NULL){
        reg_print_msg_error("The input reference image could not be read");
        return EXIT_FAILURE;
    }
    reg_tools_changeDatatype<float>(referenceImage);

    // Read the input floating image
    nifti_image *warpedImage = reg_io_ReadImageFile(inputWarpedImageName);
    if (warpedImage == NULL){
        reg_print_msg_error("The input warped image could not be read");
        return EXIT_FAILURE;
    }
    reg_tools_changeDatatype<float>(warpedImage);

    // Read the expected block matching matrix
    std::pair<size_t, size_t> inputMatrixSize = reg_tool_sizeInputMatrixFile(expectedBlockMatchingMatrixName);
    size_t m = inputMatrixSize.first;
    size_t n = inputMatrixSize.second;
    float **inputSVDMatrix = reg_tool_ReadMatrixFile<float>(expectedBlockMatchingMatrixName, m, n);

    // Create a mask - Why ?
    int *mask = (int *)malloc(referenceImage->nvox*sizeof(int));
    for (size_t i = 0; i < referenceImage->nvox; ++i) {
        mask[i] = i;
    }

    _reg_blockMatchingParam* blockMatchingParams;

    // Platforms
    Content *con = NULL;
    if (platformCode == NR_PLATFORM_CPU) {
        con = new Content(referenceImage, NULL, mask, sizeof(float), 100, 100, 1);
    }
#ifdef _USE_CUDA
    else if (platformCode == NR_PLATFORM_CUDA) {
        con = new CudaContent(referenceImage, NULL, mask, sizeof(float), 100, 100, 1);
    }
#endif
#ifdef _USE_OPENCL
    else if (platformCode == NR_PLATFORM_CL) {
        con = new ClContent(referenceImage, NULL, mask, sizeof(float), 100, 100, 1);
    }
#endif
    else {
        reg_print_msg_error("The platform code is not suppoted");
        return EXIT_FAILURE;
    }
    con->setCurrentWarped(warpedImage);
    //con->setCurrentWarped(referenceImage);
    test(con, platformCode);
    blockMatchingParams = con->getBlockMatchingParams();

#ifndef NDEBUG

    std::cout << "blockMatchingParams->definedActiveBlock = " << blockMatchingParams->definedActiveBlock << std::endl;

    int positionIndex = 0;
    for (int z = 0; z < 1; z += 3){
        for (int y = 1; y < blockMatchingParams->blockNumber[1]; y=y+3){
            for (int x = 1; x < blockMatchingParams->blockNumber[0]; x=x+3){
                positionIndex = 3*(y * blockMatchingParams->blockNumber[0] + x);
                std::cout << "ref position - warped position: ";
                std::cout << blockMatchingParams->referencePosition[positionIndex] << " ";
                std::cout << blockMatchingParams->referencePosition[positionIndex + 1] << " ";
                std::cout << blockMatchingParams->warpedPosition[positionIndex] << " ";
                std::cout << blockMatchingParams->warpedPosition[positionIndex + 1] << std::endl;
            }
        }
    }

#endif
    /*
    nifti_image_free(referenceImage);
    nifti_image_free(floatingImage);
    free(mask);
    delete con;
    */
    return EXIT_SUCCESS;
}

