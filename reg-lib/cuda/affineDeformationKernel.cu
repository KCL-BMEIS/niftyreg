#include "CudaTools.hpp"

/* *************************************************************** */
__device__ __inline__ double getPosition(float* matrix, double* voxel, const unsigned idx)
{
   unsigned long index = idx * 4;
   return (double)matrix[index++] * voxel[0] +
          (double)matrix[index++] * voxel[1] +
          (double)matrix[index++] * voxel[2] +
          (double)matrix[index];
}
/* *************************************************************** */
__global__ void affineKernel(float *transformationMatrix,
                             float4 *defField,
                             const int *mask,
                             const uint3 dims,
                             const bool composition)
{
   // Get the current coordinate
   const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
   const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
   const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;
   const unsigned index = x + dims.x * (y + z * dims.y);

   if (z<dims.z && y<dims.y && x<dims.x && mask[index] >= 0)
   {
      // The deformation field is an interleaved float4 (world coords in .x/.y/.z), shared with the
      // Compute-path resampler.
      const float4 current = defField[index];
      double voxel[3];
      voxel[0] = composition ? current.x : x;
      voxel[1] = composition ? current.y : y;
      voxel[2] = composition ? current.z : z;

      /* the deformation field (real coordinates) is stored */
      float4 result;
      result.x = (float)getPosition(transformationMatrix, voxel, 0);
      result.y = (float)getPosition(transformationMatrix, voxel, 1);
      result.z = (float)getPosition(transformationMatrix, voxel, 2);
      result.w = 0;
      defField[index] = result;
   }
}
/* *************************************************************** */
void launchAffine(mat44 *affineTransformation,
                  nifti_image *deformationField,
                  float4 *def_d,
                  const int *mask_d,
                  float *trans_d,
                  bool compose) {

   const unsigned xThreads = 8;
   const unsigned yThreads = 8;
   const unsigned zThreads = 8;

   const unsigned xBlocks = ((deformationField->nx % xThreads) == 0) ? (deformationField->nx / xThreads) : (deformationField->nx / xThreads) + 1;
   const unsigned yBlocks = ((deformationField->ny % yThreads) == 0) ? (deformationField->ny / yThreads) : (deformationField->ny / yThreads) + 1;
   const unsigned zBlocks = ((deformationField->nz % zThreads) == 0) ? (deformationField->nz / zThreads) : (deformationField->nz / zThreads) + 1;

   dim3 G1_b(xBlocks, yBlocks, zBlocks);
   dim3 B1_b(xThreads, yThreads, zThreads);

   float* trans = (float *)malloc(16 * sizeof(float));
   const mat44 *targetMatrix = (deformationField->sform_code > 0) ? &(deformationField->sto_xyz) : &(deformationField->qto_xyz);
   mat44 transformationMatrix = compose ? *affineTransformation : *affineTransformation * *targetMatrix;
   mat44ToCptr(transformationMatrix, trans);
   NR_CUDA_SAFE_CALL(cudaMemcpy(trans_d, trans, 16 * sizeof(float), cudaMemcpyHostToDevice));
   free(trans);

   uint3 dims_d = make_uint3(deformationField->nx, deformationField->ny, deformationField->nz);
   affineKernel<<<G1_b, B1_b>>>(trans_d, def_d, mask_d, dims_d, compose);
   NR_CUDA_CHECK_KERNEL(G1_b, B1_b);
}
