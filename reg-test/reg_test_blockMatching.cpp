#include "_reg_ReadWriteImage.h"
#include "_reg_ReadWriteMatrix.h"
#include "_reg_blockMatching.h"
#include "_reg_tools.h"
#include "_reg_globalTrans.h"

#include "BlockMatchingKernel.h"
#include "Platform.h"
#include "AladinContent.h"

#define EPS 0.000001

void check_matching_difference(int dim,
                               float* referencePosition,
                               float* warpedPosition,
                               float* expectedReferencePositions,
                               float* expectedWarpedPosition,
                               float &max_difference) {
    float difference;
    for (int i = 0; i < dim; ++i) {
        difference = fabsf(referencePosition[i] - expectedReferencePositions[i]);
        max_difference = std::max(difference, max_difference);
        if (difference > EPS) {
#ifndef NDEBUG
            fprintf(stderr, "reg_test_blockMatching reference position failed %g>%g\n", difference, EPS);
            if (dim == 2) {
                fprintf(stderr, "Reference. NR [%g %g] Expected [%g %g]\n",
                        referencePosition[0], referencePosition[1],
                        expectedReferencePositions[0], expectedReferencePositions[1]);
                fprintf(stderr, "Warped. NR [%g %g] Expected [%g %g]\n",
                        warpedPosition[0], warpedPosition[1],
                        expectedWarpedPosition[0], expectedWarpedPosition[1]);
            } else {
                fprintf(stderr, "Reference. NR [%g %g %g] Expected [%g %g %g]\n",
                        referencePosition[0], referencePosition[1], referencePosition[2],
                        expectedReferencePositions[0], expectedReferencePositions[1], expectedReferencePositions[2]);
                fprintf(stderr, "Warped. NR [%g %g %g] Expected [%g %g %g]\n",
                        warpedPosition[0], warpedPosition[1], warpedPosition[2],
                        expectedWarpedPosition[0], expectedWarpedPosition[1], expectedWarpedPosition[2]);
            }
            reg_exit();
#endif
        }
        difference = fabsf(warpedPosition[i] - expectedWarpedPosition[i]);
        max_difference = std::max(difference, max_difference);
        if (difference > EPS) {
#ifndef NDEBUG
            fprintf(stderr, "reg_test_blockMatching warped position failed %g>%g\n", difference, EPS);
            if (dim == 2) {
                fprintf(stderr, "Reference. NR [%g %g] Expected [%g %g]\n",
                        referencePosition[0], referencePosition[1],
                        expectedReferencePositions[0], expectedReferencePositions[1]);
                fprintf(stderr, "Warped. NR [%g %g] Expected [%g %g]\n",
                        warpedPosition[0], warpedPosition[1],
                        expectedWarpedPosition[0], expectedWarpedPosition[1]);
            } else {
                fprintf(stderr, "Reference. NR [%g %g %g] Expected [%g %g %g]\n",
                        referencePosition[0], referencePosition[1], referencePosition[2],
                        expectedReferencePositions[0], expectedReferencePositions[1], expectedReferencePositions[2]);
                fprintf(stderr, "Warped. NR [%g %g %g] Expected [%g %g %g]\n",
                        warpedPosition[0], warpedPosition[1], warpedPosition[2],
                        expectedWarpedPosition[0], expectedWarpedPosition[1], expectedWarpedPosition[2]);
            }
            reg_exit();
#endif
        }
    }
}

void test(AladinContent *con, Platform *platform) {
    std::unique_ptr<Kernel> blockMatchingKernel{ platform->CreateKernel(BlockMatchingKernel::GetName(), con) };
    blockMatchingKernel->castTo<BlockMatchingKernel>()->Calculate();
}

int main(int argc, char **argv) {

    if (argc != 5) {
        fprintf(stderr, "Usage: %s <refImage> <warpedImage> <expectedBlockMatchingMatrix> <platformType>\n", argv[0]);
        return EXIT_FAILURE;
    }

    char *inputRefImageName = argv[1];
    char *inputWarpedImageName = argv[2];
    char *expectedBlockMatchingMatrixName = argv[3];
    PlatformType platformType{ atoi(argv[4]) };

    // Read the input reference image
    nifti_image *referenceImage = reg_io_ReadImageFile(inputRefImageName);
    if (referenceImage == nullptr) {
        reg_print_msg_error("The input reference image could not be read");
        return EXIT_FAILURE;
    }
    reg_tools_changeDatatype<float>(referenceImage);
    //dim
    int imgDim = referenceImage->dim[0];

    // Read the input floating image
    nifti_image *warpedImage = reg_io_ReadImageFile(inputWarpedImageName);
    if (warpedImage == nullptr) {
        reg_print_msg_error("The input warped image could not be read");
        return EXIT_FAILURE;
    }
    reg_tools_changeDatatype<float>(warpedImage);

    // Read the expected block matching matrix
    std::pair<size_t, size_t> inputMatrixSize = reg_tool_sizeInputMatrixFile(expectedBlockMatchingMatrixName);
    size_t m = inputMatrixSize.first;
    size_t n = inputMatrixSize.second;
    float **expectedBlockMatchingMatrix = reg_tool_ReadMatrixFile<float>(expectedBlockMatchingMatrixName, m, n);

    // Create a mask
    int *mask = (int *)malloc(referenceImage->nvox * sizeof(int));
    for (size_t i = 0; i < referenceImage->nvox; ++i) {
        mask[i] = i;
    }

    _reg_blockMatchingParam* blockMatchingParams;

    // Platforms
    std::unique_ptr<Platform> platform{ new Platform(platformType) };
    std::unique_ptr<AladinContentCreator> contentCreator{ dynamic_cast<AladinContentCreator*>(platform->CreateContentCreator(ContentType::Aladin)) };
    std::unique_ptr<AladinContent> con{ contentCreator->Create(referenceImage, nullptr, mask, sizeof(float), 100, 100, 1) };
    con->SetWarped(warpedImage);
    //con->SetWarped(referenceImage);
    test(con.get(), platform.get());
    blockMatchingParams = con->GetBlockMatchingParams();

#ifndef NDEBUG
    std::cout << "blockMatchingParams->definedActiveBlock = " << blockMatchingParams->definedActiveBlockNumber << std::endl;
#endif

    float max_difference = 0;

    int blockIndex = 0;
    int positionIndex = 0;
    int matrixIndex = 0;

    unsigned int zMax = 2;
    if (imgDim == 3)
        zMax = blockMatchingParams->blockNumber[2] - 1;


    for (unsigned int z = 1; z < zMax; z += 3) {
        for (unsigned int y = 1; y < blockMatchingParams->blockNumber[1] - 1; y += 3) {
            for (unsigned int x = 1; x < blockMatchingParams->blockNumber[0] - 1; x += 3) {

                if (imgDim == 3) {
                    blockIndex = (z * blockMatchingParams->blockNumber[1] + y) * blockMatchingParams->blockNumber[0] + x;
                } else {
                    blockIndex = y * blockMatchingParams->blockNumber[0] + x;
                }

                positionIndex = imgDim * blockMatchingParams->totalBlock[blockIndex];

                if (positionIndex > -1) {
                    check_matching_difference(imgDim,
                                              &blockMatchingParams->referencePosition[positionIndex],
                                              &blockMatchingParams->warpedPosition[positionIndex],
                                              &expectedBlockMatchingMatrix[matrixIndex][0],
                                              &expectedBlockMatchingMatrix[matrixIndex][3],
                                              max_difference);
                    matrixIndex++;
                }
            }
        }
    }

    free(mask);
    reg_matrix2DDeallocate(m, expectedBlockMatchingMatrix);
    nifti_image_free(referenceImage);

    if (max_difference > EPS) {
#ifndef NDEBUG
        fprintf(stdout, "reg_test_blockMatching failed: %g (>%g)\n", max_difference, EPS);
#endif
        return EXIT_FAILURE;
    }
#ifndef NDEBUG
    printf("All good (%g<%g)\n", max_difference, EPS);
#endif
    return EXIT_SUCCESS;
}
