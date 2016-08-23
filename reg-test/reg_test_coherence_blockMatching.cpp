#include "_reg_ReadWriteImage.h"
#include "_reg_ReadWriteMatrix.h"
#include "_reg_blockMatching.h"
#include "_reg_tools.h"
#include "_reg_globalTrans.h"

#include "BlockMatchingKernel.h"
#include "Platform.h"

#include "AladinContent.h"
#ifdef _USE_CUDA
#include "CUDAAladinContent.h"
#endif
#ifdef _USE_OPENCL
#include "CLAladinContent.h"
#endif

#include <algorithm>

#define EPS 0.000001

void check_matching_difference(int dim,
                               float* cpuRefPos,
                               float* cpuWarPos,
                               float* gpuRefPos,
                               float* gpuWarPos,
                               float &max_difference)
{
   bool cpu_finite = cpuWarPos[0]==cpuWarPos[0] ? true : false;
   bool gpu_finite = gpuWarPos[0]==gpuWarPos[0] ? true : false;

   if(!cpu_finite && !gpu_finite) return;

   if(cpu_finite!=gpu_finite){
      max_difference = std::numeric_limits<float>::max();
      return;
   }

   float difference;
   for (int i = 0; i < dim; ++i) {
      difference = fabsf(cpuRefPos[i] - gpuRefPos[i]);
      max_difference = std::max(difference, max_difference);
      if (difference > EPS){
#ifndef NDEBUG
         fprintf(stderr, "reg_test_blockMatching reference position failed %g>%g\n", difference, EPS);
         if(dim==2){
            fprintf(stderr, "Reference. CPU [%g %g] GPU [%g %g]\n",
                    cpuRefPos[0], cpuRefPos[1],
                  gpuRefPos[0], gpuRefPos[1]);
            fprintf(stderr, "Warped. CPU [%g %g] GPU [%g %g]\n",
                    cpuWarPos[0], cpuWarPos[1],
                  gpuWarPos[0], gpuWarPos[1]);
         }
         else{
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
      if (difference > EPS){
#ifndef NDEBUG
         fprintf(stderr, "reg_test_blockMatching warped position failed %g>%g\n", difference, EPS);
         if(dim==2){
            fprintf(stderr, "Reference. CPU [%g %g] GPU [%g %g]\n",
                    cpuRefPos[0], cpuRefPos[1],
                  gpuRefPos[0], gpuRefPos[1]);
            fprintf(stderr, "Warped. CPU [%g %g] GPU [%g %g]\n",
                    cpuWarPos[0], cpuWarPos[1],
                  gpuWarPos[0], gpuWarPos[1]);
         }
         else{
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

void test(GlobalContent *con, int platformCode) {

   Platform *platform = new Platform(platformCode);

   Kernel *blockMatchingKernel = platform->createKernel(BlockMatchingKernel::getName(), con);
   blockMatchingKernel->castTo<BlockMatchingKernel>()->calculate();

   delete blockMatchingKernel;
   delete platform;
}

int main(int argc, char **argv)
{

   if (argc != 4) {
      fprintf(stderr, "Usage: %s <refImage> <warpedImage> <platformCode>\n", argv[0]);
      return EXIT_FAILURE;
   }

   char *inputRefImageName = argv[1];
   char *inputWarpedImageName = argv[2];
   int   platformCode = atoi(argv[3]);
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
   if (referenceImage == NULL){
      reg_print_msg_error("The input reference image could not be read");
      return EXIT_FAILURE;
   }
   reg_tools_changeDatatype<float>(referenceImage);
   //dim
   int imgDim = referenceImage->dim[0];

   // Read the input floating image
   nifti_image *warpedImage = reg_io_ReadImageFile(inputWarpedImageName);
   if (warpedImage == NULL){
      reg_print_msg_error("The input warped image could not be read");
      return EXIT_FAILURE;
   }
   reg_tools_changeDatatype<float>(warpedImage);

   // Create a mask
   int *mask = (int *)malloc(referenceImage->nvox*sizeof(int));
   for (size_t i = 0; i < referenceImage->nvox; ++i) mask[i] = i;

   // CPU Platform
   _reg_blockMatchingParam* blockMatchingParams_cpu = NULL;
   AladinContent *con_cpu = NULL;
   con_cpu = new AladinContent(NR_PLATFORM_CPU);
   con_cpu->setCurrentReference(referenceImage);
   con_cpu->setCurrentReferenceMask(mask, referenceImage->nvox);
   con_cpu->setCurrentWarped(warpedImage);
   con_cpu->setInlierLts(100);
   con_cpu->setPercentageOfBlock(100);
   con_cpu->setBlockStepSize(1);
   con_cpu->InitBlockMatchingParams();
   test(con_cpu, NR_PLATFORM_CPU);
   blockMatchingParams_cpu = con_cpu->getBlockMatchingParams();

#ifndef NDEBUG
   std::cout << "blockMatchingParams_cpu->definedActiveBlock = " << blockMatchingParams_cpu->definedActiveBlockNumber << std::endl;
#endif

   // GPU Platform
   AladinContent *con_gpu = NULL;
   _reg_blockMatchingParam* blockMatchingParams_gpu = NULL;
#ifdef _USE_CUDA
   if (platformCode == NR_PLATFORM_CUDA) {
       con_gpu = new CudaAladinContent();
   }
#endif
#ifdef _USE_OPENCL
   if (platformCode == NR_PLATFORM_CL) {
       con_gpu = new ClAladinContent();
   }
#endif
   con_gpu->setCurrentReference(referenceImage);
   con_gpu->setCurrentReferenceMask(mask, referenceImage->nvox);
   con_gpu->setCurrentWarped(warpedImage);
   con_gpu->setInlierLts(100);
   con_gpu->setPercentageOfBlock(100);
   con_gpu->setBlockStepSize(1);
   con_gpu->InitBlockMatchingParams();
   test(con_gpu, platformCode);
   blockMatchingParams_gpu = con_gpu->getBlockMatchingParams();

#ifndef NDEBUG
   std::cout << "blockMatchingParams_gpu->definedActiveBlock = " << blockMatchingParams_gpu->definedActiveBlockNumber << std::endl;
#endif

   float max_difference = 0;

   if(blockMatchingParams_cpu->definedActiveBlockNumber != blockMatchingParams_gpu->definedActiveBlockNumber){
      reg_print_msg_error("The number of defined active blockNumber blocks vary accros platforms");
      return EXIT_FAILURE;
   }

   for(int i=0; i<blockMatchingParams_cpu->activeBlockNumber*imgDim; i+=imgDim){
      check_matching_difference(imgDim,
                                &blockMatchingParams_cpu->referencePosition[i],
                                &blockMatchingParams_cpu->warpedPosition[i],
                                &blockMatchingParams_gpu->referencePosition[i],
                                &blockMatchingParams_gpu->warpedPosition[i],
                                max_difference);
   }
   size_t test_cpu=0, test_gpu=0;
   for(int i=0; i<blockMatchingParams_cpu->activeBlockNumber*imgDim; i+=imgDim){
       test_cpu = blockMatchingParams_cpu->warpedPosition[i]==blockMatchingParams_cpu->warpedPosition[i]?++test_cpu:test_cpu;
       test_gpu = blockMatchingParams_gpu->warpedPosition[i]==blockMatchingParams_gpu->warpedPosition[i]?++test_gpu:test_gpu;
   }
   printf("CPU: %lu - GPU: %lu\n", test_cpu, test_gpu);

   delete con_gpu;
   //delete con_cpu;
   free(mask);
   nifti_image_free(referenceImage);

   if(max_difference>EPS){
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

