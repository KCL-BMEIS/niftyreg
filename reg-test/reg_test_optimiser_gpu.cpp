#include <time.h>
#include "_reg_f3d_gpu.h"

#define EPS 0.001
#define SIZE 64

int main(int argc, char **argv)
{
   if(argc!=3)
   {
      fprintf(stderr, "Usage: %s <dim> <type>\n", argv[0]);
      fprintf(stderr, "<dim>\tImages dimension (2,3)\n");
      fprintf(stderr, "<type>\tTest type:\n");
      fprintf(stderr, "\t\t- Gradient descent (\"descent\")\n");
      fprintf(stderr, "\t\t- Conjugate gradient descent (\"conjugate\")\n");
      return EXIT_FAILURE;
   }
   int dimension=atoi(argv[1]);
   char *type=argv[2];
   if(strcmp(type,"conjugate")!=0)
      type=(char *)"descent";

   // Check and setup the GPU card
   CUcontext ctx;
   if(cudaCommon_setCUDACard(&ctx, true))
      return EXIT_FAILURE;

   // Create fake registration objects
   reg_f3d<float> *reg_cpu=new reg_f3d<float>(1,1);
   reg_f3d_gpu *reg_gpu=new reg_f3d_gpu(1,1);

   // Set all objective function weight to 0 to avoid any computation
   reg_cpu->SetBendingEnergyWeight(0);
   reg_cpu->SetJacobianLogWeight(0);
   reg_cpu->SetInverseConsistencyWeight(0);
   reg_cpu->SetL2NormDisplacementWeight(0);
   reg_cpu->SetLinearEnergyWeights(0,0);
   reg_gpu->SetBendingEnergyWeight(0);
   reg_gpu->SetJacobianLogWeight(0);
   reg_gpu->SetInverseConsistencyWeight(0);
   reg_gpu->SetL2NormDisplacementWeight(0);
   reg_gpu->SetLinearEnergyWeights(0,0);

   // Useful variable
   const size_t nodeNumber=(size_t)pow((double)SIZE,dimension)+.5;

   // Create the optimiser
   reg_optimiser<float> *optimiser_cpu=NULL;
   reg_optimiser_gpu *optimiser_gpu=NULL;

   // Check which opitmiser type should be used
   if(strcmp(type,"conjugate")==0)
   {
      optimiser_cpu=new reg_conjugateGradient<float>();
      optimiser_gpu=new reg_conjugateGradient_gpu();
   }
   else
   {
      optimiser_cpu=new reg_optimiser<float>();
      optimiser_gpu=new reg_optimiser_gpu();
   }

   // Create some random arrays on the cpu
   float *values_forward_cpu=(float *)malloc(nodeNumber*dimension*sizeof(float));
   float *grad_forward_cpu=(float *)malloc(nodeNumber*dimension*sizeof(float));
   srand(time(0));
   for(size_t i=0; i<nodeNumber*dimension; ++i)
   {
      values_forward_cpu[i]= 100.f * ((float)rand()/(float)RAND_MAX) - 50.f; //[-50;50]
      grad_forward_cpu[i]= 10.f * ((float)rand()/(float)RAND_MAX) - 5.f; //[-5;5];
   }

   // Allocate some arrays on the GPU
   float4 *values_forward_gpu=NULL;
   float4 *grad_forward_gpu=NULL;
   NR_CUDA_SAFE_CALL(cudaMalloc(&values_forward_gpu,nodeNumber*sizeof(float4)))
   NR_CUDA_SAFE_CALL(cudaMalloc(&grad_forward_gpu,nodeNumber*sizeof(float4)))

   // Create a fake nifti image to ease the transfer from GPU to CPU
   int dim[8]= {5,SIZE,SIZE,SIZE,1,3,1,1};
   if(dimension==2)
   {
      dim[3]=1;
      dim[5]=2;
   }
   nifti_image *image=nifti_make_new_nim(dim,NIFTI_TYPE_FLOAT32,false);

   // Transfer the random values from the CPU to the GPU
   image->data=static_cast<void *>(values_forward_cpu);
   cudaCommon_transferNiftiToArrayOnDevice<float4>(&values_forward_gpu,image);
   image->data=static_cast<void *>(grad_forward_cpu);
   cudaCommon_transferNiftiToArrayOnDevice<float4>(&grad_forward_gpu,image);
   image->data=NULL;

   // Initialise the CPU optimiser
   reg_cpu->reg_test_setControlPointGrid(image);
   reg_cpu->reg_test_setOptimiser(optimiser_cpu);
   optimiser_cpu->Initialise(nodeNumber*dimension,
                             dimension,
                             true,
                             true,
                             true,
                             10,
                             0,
                             reg_cpu,
                             values_forward_cpu,
                             grad_forward_cpu
                            );

   // Initialise the GPU optimiser
   reg_gpu->reg_test_setControlPointGrid(image);
   reg_gpu->reg_test_setOptimiser(optimiser_gpu);
   optimiser_gpu->Initialise(nodeNumber*dimension,
                             dimension,
                             true,
                             true,
                             true,
                             10,
                             0,
                             reg_gpu,
                             reinterpret_cast<float *>(values_forward_gpu),
                             reinterpret_cast<float *>(grad_forward_gpu)
                            );

   // Run the optimiser once
   optimiser_cpu->reg_test_optimiser();
   optimiser_gpu->reg_test_optimiser();

   // If conjugate gradient, the code is run twice to ensure both conjugate function are used
   if(strcmp(type,"conjugate")==0)
   {
      // modify the gradient values
      for(size_t i=0; i<nodeNumber*dimension; ++i)
         grad_forward_cpu[i]= 10.f * ((float)rand()/(float)RAND_MAX) - 5.f; //[-5;5];
      // transfer the new values onto the device
      image->data=static_cast<void *>(grad_forward_cpu);
      cudaCommon_transferNiftiToArrayOnDevice<float4>(&grad_forward_gpu,image);
      // run the update again
      optimiser_cpu->reg_test_optimiser();
      optimiser_gpu->reg_test_optimiser();
   }

#ifndef NDEBUG
   // Save the CPU result image
   image->data=static_cast<void *>(values_forward_cpu);
   nifti_set_filenames(image,"optimiser_test_cpu.nii",0,0);
   nifti_image_write(image);
#endif
   // Compare the CPU and the GPU results
   float *values_forward_gpu_transfered=(float *)calloc(image->nvox,sizeof(float));
   image->data=static_cast<void *>(values_forward_gpu_transfered);
   cudaCommon_transferFromDeviceToNifti<float4>(image,&values_forward_gpu);
   float maxDifference=reg_test_compare_arrays(values_forward_gpu_transfered,values_forward_cpu,nodeNumber*dimension);
#ifndef NDEBUG
   // Save the GPU result
   nifti_set_filenames(image,"optimiser_test_gpu.nii",0,0);
   nifti_image_write(image);
   // Print out the maximal error value
   printf("[NiftyReg DEBUG] [dim=%i] Gradient difference (%s): %g\n",
          dimension,
          type,
          maxDifference);
#endif
   // Free all allocated arrays
   image->data=NULL;
   nifti_image_free(image);
   reg_cpu->reg_test_setControlPointGrid(NULL);
   reg_gpu->reg_test_setControlPointGrid(NULL);
   delete optimiser_cpu;
   reg_cpu->reg_test_setOptimiser(NULL);
   delete optimiser_gpu;
   reg_gpu->reg_test_setOptimiser(NULL);
   delete reg_cpu;
   delete reg_gpu;
   free(values_forward_cpu);
   free(grad_forward_cpu);
   cudaCommon_free<float4>(&values_forward_gpu);
   cudaCommon_free<float4>(&grad_forward_gpu);
   free(values_forward_gpu_transfered);

   cudaCommon_unsetCUDACard(&ctx);

   // Check if the maximal difference is too large
   if(maxDifference>EPS)
   {
      fprintf(stderr,
              "[dim=%i] Gradient difference too high: %g\n",
              dimension,
              maxDifference);
      return EXIT_FAILURE;
   }
   return EXIT_SUCCESS;
}

