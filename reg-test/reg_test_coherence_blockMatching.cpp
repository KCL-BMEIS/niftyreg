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
                               float* cpuRefPos,
                               float* cpuWarPos,
                               float* gpuRefPos,
                               float* gpuWarPos,
                               float &max_difference) {
    bool cpu_finite = cpuWarPos[0] == cpuWarPos[0] ? true : false;
    bool gpu_finite = gpuWarPos[0] == gpuWarPos[0] ? true : false;

    if (!cpu_finite && !gpu_finite) return;

    if (cpu_finite != gpu_finite) {
        max_difference = std::numeric_limits<float>::max();
        return;
    }

    float difference;
    for (int i = 0; i < dim; ++i) {
        difference = fabsf(cpuRefPos[i] - gpuRefPos[i]);
        max_difference = std::max(difference, max_difference);
        if (difference > EPS) {
#ifndef NDEBUG
            fprintf(stderr, "reg_test_blockMatching reference position failed %g>%g\n", difference, EPS);
            if (dim == 2) {
                fprintf(stderr, "Reference. CPU [%g %g] GPU [%g %g]\n",
                        cpuRefPos[0], cpuRefPos[1],
                        gpuRefPos[0], gpuRefPos[1]);
                fprintf(stderr, "Warped. CPU [%g %g] GPU [%g %g]\n",
                        cpuWarPos[0], cpuWarPos[1],
                        gpuWarPos[0], gpuWarPos[1]);
            } else {
                fprintf(stderr, "Reference. CPU [%g %g %g] GPU [%g %g %g]\n",
                        cpuRefPos[0], cpuRefPos[1], cpuRefPos[2],
                        gpuRefPos[0], gpuRefPos[1], gpuRefPos[2]);
                fprintf(stderr, "Warped. CPU [%g %g %g] GPU [%g %g %g]\n",
                        cpuWarPos[0], cpuWarPos[1], cpuWarPos[2],
                        gpuWarPos[0], gpuWarPos[1], gpuWarPos[2]);
            }
            reg_exit();
#endif
        }
        difference = fabsf(cpuWarPos[i] - gpuWarPos[i]);
        max_difference = std::max(difference, max_difference);
        if (difference > EPS) {
#ifndef NDEBUG
            fprintf(stderr, "reg_test_blockMatching warped position failed %g>%g\n", difference, EPS);
            if (dim == 2) {
                fprintf(stderr, "Reference. CPU [%g %g] GPU [%g %g]\n",
                        cpuRefPos[0], cpuRefPos[1],
                        gpuRefPos[0], gpuRefPos[1]);
                fprintf(stderr, "Warped. CPU [%g %g] GPU [%g %g]\n",
                        cpuWarPos[0], cpuWarPos[1],
                        gpuWarPos[0], gpuWarPos[1]);
            } else {
                fprintf(stderr, "Reference. CPU [%g %g %g] GPU [%g %g %g]\n",
                        cpuRefPos[0], cpuRefPos[1], cpuRefPos[2],
                        gpuRefPos[0], gpuRefPos[1], gpuRefPos[2]);
                fprintf(stderr, "Warped. CPU [%g %g %g] GPU [%g %g %g]\n",
                        cpuWarPos[0], cpuWarPos[1], cpuWarPos[2],
                        gpuWarPos[0], gpuWarPos[1], gpuWarPos[2]);
            }
            reg_exit();
#endif
        }
    }
}

void test(AladinContent *con, Platform *platform) {
    unique_ptr<Kernel> blockMatchingKernel{ platform->CreateKernel(BlockMatchingKernel::GetName(), con) };
    blockMatchingKernel->castTo<BlockMatchingKernel>()->Calculate();
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <refImage> <warpedImage> <platformType>\n", argv[0]);
        return EXIT_FAILURE;
    }

    char *inputRefImageName = argv[1];
    char *inputWarpedImageName = argv[2];
    PlatformType platformType{ atoi(argv[3]) };

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
    //dim
    int imgDim = referenceImage->dim[0];

    // Read the input floating image
    nifti_image *warpedImage = reg_io_ReadImageFile(inputWarpedImageName);
    if (warpedImage == nullptr) {
        reg_print_msg_error("The input warped image could not be read");
        return EXIT_FAILURE;
    }
    reg_tools_changeDatatype<float>(warpedImage);

    // Create a mask
    int *mask = (int *)malloc(referenceImage->nvox * sizeof(int));
    for (size_t i = 0; i < referenceImage->nvox; ++i) mask[i] = i;

    // CPU Platform
    unique_ptr<Platform> platformCpu{ new Platform(PlatformType::Cpu) };
    unique_ptr<AladinContent> conCpu{ new AladinContent(referenceImage, nullptr, mask, sizeof(float), 100, 100, 1) };
    conCpu->SetWarped(warpedImage);
    test(conCpu.get(), platformCpu.get());
    _reg_blockMatchingParam *blockMatchingParams_cpu = conCpu->GetBlockMatchingParams();

#ifndef NDEBUG
    std::cout << "blockMatchingParams_cpu->activeBlockNumber = " << blockMatchingParams_cpu->activeBlockNumber << std::endl;
    std::cout << "blockMatchingParams_cpu->definedActiveBlockNumber = " << blockMatchingParams_cpu->definedActiveBlockNumber << std::endl;
#endif

    // GPU Platform
    unique_ptr<Platform> platformGpu{ new Platform(platformType) };
    unique_ptr<AladinContentCreator> contentCreator{ dynamic_cast<AladinContentCreator*>(platformGpu->CreateContentCreator(ContentType::Aladin)) };
    unique_ptr<AladinContent> conGpu{ contentCreator->Create(referenceImage, nullptr, mask, sizeof(float), 100, 100, 1) };
    conGpu->SetWarped(warpedImage);
    test(conGpu.get(), platformGpu.get());
    _reg_blockMatchingParam *blockMatchingParams_gpu = conGpu->GetBlockMatchingParams();

#ifndef NDEBUG
    std::cout << "blockMatchingParams_gpu->activeBlockNumber = " << blockMatchingParams_gpu->activeBlockNumber << std::endl;
    std::cout << "blockMatchingParams_gpu->definedActiveBlockNumber = " << blockMatchingParams_gpu->definedActiveBlockNumber << std::endl;
#endif

    float max_difference = 0;

    if (blockMatchingParams_cpu->definedActiveBlockNumber != blockMatchingParams_gpu->definedActiveBlockNumber) {
        reg_print_msg_error("The number of defined active blockNumber blocks vary accros platforms");
        char out_text[255];
        sprintf(out_text, "activeBlockNumber CPU: %i", blockMatchingParams_cpu->activeBlockNumber);
        reg_print_msg_error(out_text);
        sprintf(out_text, "activeBlockNumber GPU: %i", blockMatchingParams_gpu->activeBlockNumber);
        reg_print_msg_error(out_text);
        sprintf(out_text, "definedActiveBlockNumber CPU: %i", blockMatchingParams_cpu->definedActiveBlockNumber);
        reg_print_msg_error(out_text);
        sprintf(out_text, "definedActiveBlockNumber CPU: %i", blockMatchingParams_gpu->definedActiveBlockNumber);
        reg_print_msg_error(out_text);
        return EXIT_FAILURE;
    }

    for (int i = 0; i < blockMatchingParams_cpu->activeBlockNumber * imgDim; i += imgDim) {
        check_matching_difference(imgDim,
                                  &blockMatchingParams_cpu->referencePosition[i],
                                  &blockMatchingParams_cpu->warpedPosition[i],
                                  &blockMatchingParams_gpu->referencePosition[i],
                                  &blockMatchingParams_gpu->warpedPosition[i],
                                  max_difference);
    }
    size_t test_cpu = 0, test_gpu = 0;
    for (int i = 0; i < blockMatchingParams_cpu->activeBlockNumber * imgDim; i += imgDim) {
        test_cpu = (blockMatchingParams_cpu->warpedPosition[i] == blockMatchingParams_cpu->warpedPosition[i]) ? test_cpu + 1 : test_cpu;
        test_gpu = (blockMatchingParams_gpu->warpedPosition[i] == blockMatchingParams_gpu->warpedPosition[i]) ? test_gpu + 1 : test_gpu;
    }
    printf("CPU: %zu - GPU: %zu\n", test_cpu, test_gpu);

    free(mask);
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
