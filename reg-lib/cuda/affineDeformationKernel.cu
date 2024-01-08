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
                             float *defField,
                             const int *mask,
                             const uint3 dims,
                             const unsigned voxelNumber,
                             const bool composition)
{
   // Get the current coordinate
   const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
   const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
   const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;
   const unsigned index = x + dims.x * (y + z * dims.y);

   if (z<dims.z && y<dims.y && x<dims.x && mask[index] >= 0)
   {
      double voxel[3];
      float *deformationFieldPtrX = &defField[index];
      float *deformationFieldPtrY = &deformationFieldPtrX[voxelNumber];
      float *deformationFieldPtrZ = &deformationFieldPtrY[voxelNumber];

      voxel[0] = composition ? *deformationFieldPtrX : x;
      voxel[1] = composition ? *deformationFieldPtrY : y;
      voxel[2] = composition ? *deformationFieldPtrZ : z;

      /* the deformation field (real coordinates) is stored */
      *deformationFieldPtrX = (float)getPosition(transformationMatrix, voxel, 0);
      *deformationFieldPtrY = (float)getPosition(transformationMatrix, voxel, 1);
      *deformationFieldPtrZ = (float)getPosition(transformationMatrix, voxel, 2);
   }
}
/* *************************************************************** */
void launchAffine(mat44 *affineTransformation,
                  nifti_image *deformationField,
                  float *def_d,
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
   mat44 transformationMatrix = compose ? *affineTransformation : reg_mat44_mul(affineTransformation, targetMatrix);
   mat44ToCptr(transformationMatrix, trans);
   NR_CUDA_SAFE_CALL(cudaMemcpy(trans_d, trans, 16 * sizeof(float), cudaMemcpyHostToDevice));
   free(trans);

   uint3 dims_d = make_uint3(deformationField->nx, deformationField->ny, deformationField->nz);
   affineKernel<<<G1_b, B1_b>>>(trans_d, def_d, mask_d, dims_d, (unsigned)NiftiImage::calcVoxelNumber(deformationField, 3), compose);
   NR_CUDA_CHECK_KERNEL(G1_b, B1_b);
}
