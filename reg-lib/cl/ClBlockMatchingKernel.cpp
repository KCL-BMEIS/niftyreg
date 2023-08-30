#include "ClBlockMatchingKernel.h"
#include "config.h"
#include <fstream>

/* *************************************************************** */
ClBlockMatchingKernel::ClBlockMatchingKernel(Content *conIn) : BlockMatchingKernel() {
   //populate the ClAladinContent object ptr
   ClAladinContent *con = static_cast<ClAladinContent*>(conIn);

   //path to kernel file
   const char *niftyreg_install_dir = getenv("NIFTYREG_INSTALL_DIR");
   const char *niftyreg_src_dir = getenv("NIFTYREG_SRC_DIR");

   std::string clInstallPath;
   std::string clSrcPath;
   //src dir
   if (niftyreg_src_dir != nullptr) {
      clSrcPath = niftyreg_src_dir + "/reg-lib/cl/"s;
   } else clSrcPath = CL_KERNELS_SRC_PATH;
   //install dir
   if (niftyreg_install_dir != nullptr) {
      clInstallPath = niftyreg_install_dir + "/include/cl/"s;
   } else clInstallPath = CL_KERNELS_PATH;
   std::string clKernel("blockMatchingKernel.cl");
   //Let's check if we did an install
   std::string clKernelPath = (clInstallPath + clKernel);
   std::ifstream kernelFile(clKernelPath.c_str(), std::ios::in);
   if (kernelFile.is_open() == 0) {
      //"clKernel.cl probably not installed - let's use the src location"
      clKernelPath = (clSrcPath + clKernel);
   }

   //get opencl context params
   sContext = &ClContextSingleton::GetInstance();
   clContext = sContext->GetContext();
   commandQueue = sContext->GetCommandQueue();
   program = sContext->CreateProgram(clKernelPath.c_str());

   // Create OpenCL kernel
   cl_int errNum;
   if (con->GetBlockMatchingParams()->dim == 3) {
      kernel = clCreateKernel(program, "blockMatchingKernel3D", &errNum);
   } else {
      kernel = clCreateKernel(program, "blockMatchingKernel2D", &errNum);
   }
   sContext->CheckErrNum(errNum, "Error setting bm kernel.");

   //get cl ptrs
   clTotalBlock = con->GetTotalBlockClmem();
   clReferenceImageArray = con->GetReferenceImageArrayClmem();
   clWarpedImageArray = con->GetWarpedImageClmem();
   clWarpedPosition = con->GetWarpedPositionClmem();
   clReferencePosition = con->GetReferencePositionClmem();
   clMask = con->GetMaskClmem();
   clReferenceMat = con->GetRefMatClmem();

   //get cpu ptrs
   reference = con->AladinContent::GetReference();
   params = con->AladinContent::GetBlockMatchingParams();

}
/* *************************************************************** */
void ClBlockMatchingKernel::Calculate() {
   if (params->stepSize != 1 || params->voxelCaptureRange != 3)
      NR_FATAL_ERROR("The block matching OpenCL kernel supports only a single step size");
   cl_int errNum;
   params->definedActiveBlockNumber = 0;
   cl_mem cldefinedBlock = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                          sizeof(int), &(params->definedActiveBlockNumber), &errNum);
   sContext->CheckErrNum(errNum, "ClBlockMatchingKernel::calculate failed to allocate memory (cldefinedBlock) ");

   const cl_uint4 imageSize = {{(cl_uint)reference->nx,
      (cl_uint)reference->ny,
      (cl_uint)reference->nz,
      (cl_uint)0}};

   size_t globalWorkSize[3] = {(size_t)params->blockNumber[0] * 4,
      (size_t)params->blockNumber[1] * 4,
      (size_t)params->blockNumber[2] * 4};
   size_t localWorkSize[3] = {4, 4, 4};
   unsigned sMemSize = 1728; // (3*4)^3
   if (reference->nz == 1) {
      globalWorkSize[2] = 1;
      localWorkSize[2] = 1;
      sMemSize = 144; // (3*4)^2
   }

   errNum = clSetKernelArg(kernel, 0, sMemSize * sizeof(cl_float), nullptr);
   sContext->CheckErrNum(errNum, "Error setting shared memory.");
   errNum = clSetKernelArg(kernel, 1, sizeof(cl_mem), &clWarpedImageArray);
   sContext->CheckErrNum(errNum, "Error setting resultImageArray.");
   errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &clReferenceImageArray);
   sContext->CheckErrNum(errNum, "Error setting targetImageArray.");
   errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &clWarpedPosition);
   sContext->CheckErrNum(errNum, "Error setting resultPosition.");
   errNum |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &clReferencePosition);
   sContext->CheckErrNum(errNum, "Error setting targetPosition.");
   errNum |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &clTotalBlock);
   sContext->CheckErrNum(errNum, "Error setting mask.");
   errNum |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &clMask);
   sContext->CheckErrNum(errNum, "Error setting mask.");
   errNum |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &clReferenceMat);
   sContext->CheckErrNum(errNum, "Error setting targetMatrix_xyz.");
   errNum |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &cldefinedBlock);
   sContext->CheckErrNum(errNum, "Error setting cldefinedBlock.");
   errNum |= clSetKernelArg(kernel, 9, sizeof(cl_uint4), &imageSize);
   sContext->CheckErrNum(errNum, "Error setting image size.");

   errNum = clEnqueueNDRangeKernel(commandQueue, kernel, params->dim, nullptr,
                                   globalWorkSize, localWorkSize, 0, nullptr, nullptr);
   sContext->CheckErrNum(errNum, "Error queuing blockmatching kernel for execution ");

   errNum = clFinish(commandQueue);
   sContext->CheckErrNum(errNum, "Error after clFinish ClBlockMatchingKernel");

   errNum = clEnqueueReadBuffer(commandQueue, cldefinedBlock, CL_TRUE, 0, sizeof(int),
                                &(params->definedActiveBlockNumber), 0, nullptr, nullptr);
   sContext->CheckErrNum(errNum, "Error reading  var after ClBlockMatchingKernel execution ");

   if (params->definedActiveBlockNumber == 0)
      NR_FATAL_ERROR("Unexpected error in the ClBlockMatchingKernel execution");

   clReleaseMemObject(cldefinedBlock);
}
/* *************************************************************** */
ClBlockMatchingKernel::~ClBlockMatchingKernel() {
   if (kernel != 0)
      clReleaseKernel(kernel);
   if (program != 0)
      clReleaseProgram(program);
}
/* *************************************************************** */
