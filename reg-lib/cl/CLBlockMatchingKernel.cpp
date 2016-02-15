#include "CLBlockMatchingKernel.h"
#include "config.h"
#include <fstream>

CLBlockMatchingKernel::CLBlockMatchingKernel(AladinContent *conIn, std::string name) :
   BlockMatchingKernel(name) {
   //populate the CLAladinContent object ptr
   this->con = static_cast<ClAladinContent*>(conIn);

   //path to kernel file
   const char* niftyreg_install_dir = getenv("NIFTYREG_INSTALL_DIR");
   const char* niftyreg_src_dir = getenv("NIFTYREG_SRC_DIR");

   std::string clInstallPath;
   std::string clSrcPath;
   //src dir
   if (niftyreg_src_dir != NULL){
      char opencl_kernel_path[255];
      sprintf(opencl_kernel_path, "%s/reg-lib/cl/", niftyreg_src_dir);
      clSrcPath = opencl_kernel_path;
   }
   else clSrcPath = CL_KERNELS_SRC_PATH;
   //install dir
   if(niftyreg_install_dir!=NULL){
      char opencl_kernel_path[255];
      sprintf(opencl_kernel_path, "%s/include/cl/", niftyreg_install_dir);
      clInstallPath = opencl_kernel_path;
   }
   else clInstallPath = CL_KERNELS_PATH;
   std::string clKernel("blockMatchingKernel.cl");
   //Let's check if we did an install
   std::string clKernelPath = (clInstallPath + clKernel);
   std::ifstream kernelFile(clKernelPath.c_str(), std::ios::in);
   if (kernelFile.is_open() == 0) {
      //"clKernel.cl propbably not installed - let's use the src location"
      clKernelPath = (clSrcPath + clKernel);
   }

   //get opencl context params
   this->sContext = &CLContextSingletton::Instance();
   this->clContext = this->sContext->getContext();
   this->commandQueue = this->sContext->getCommandQueue();
   this->program = this->sContext->CreateProgram(clKernelPath.c_str());

   // Create OpenCL kernel
   cl_int errNum;
   if (this->con->getBlockMatchingParams()->dim == 3) {
      this->kernel = clCreateKernel(program, "blockMatchingKernel3D", &errNum);
   }
   else {
      this->kernel = clCreateKernel(program, "blockMatchingKernel2D", &errNum);
   }
   this->sContext->checkErrNum(errNum, "Error setting bm kernel.");

   //get cl ptrs
   this->clTotalBlock = this->con->getTotalBlockClmem();
   this->clReferenceImageArray = this->con->getReferenceImageArrayClmem();
   this->clWarpedImageArray = this->con->getWarpedImageClmem();
   this->clWarpedPosition = this->con->getWarpedPositionClmem();
   this->clReferencePosition = this->con->getReferencePositionClmem();
   this->clMask = this->con->getMaskClmem();
   this->clReferenceMat = this->con->getRefMatClmem();

   //get cpu ptrs
   this->reference = this->con->AladinContent::getCurrentReference();
   this->params = this->con->AladinContent::getBlockMatchingParams();

}
/* *************************************************************** */
void CLBlockMatchingKernel::calculate()
{
   if (this->params->stepSize!=1 || this->params->voxelCaptureRange!=3){
      reg_print_msg_error("The block Mathching OpenCL kernel supports only a stepsize of 1");
      reg_exit();
   }
   cl_int errNum;
   this->params->definedActiveBlockNumber = 0;
   cl_mem cldefinedBlock = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                          sizeof(int), &(this->params->definedActiveBlockNumber), &errNum);
   this->sContext->checkErrNum(errNum, "CLBlockMatchingKernel::calculate failed to allocate memory (cldefinedBlock) ");

   const cl_uint4 imageSize ={{(cl_uint)this->reference->nx,
                               (cl_uint)this->reference->ny,
                               (cl_uint)this->reference->nz,
                               (cl_uint)0}};

   size_t globalWorkSize[3] = { (size_t)params->blockNumber[0] * 4,
                                (size_t)params->blockNumber[1] * 4,
                                (size_t)params->blockNumber[2] * 4};
   size_t localWorkSize[3] = {4, 4, 4};
   unsigned int sMemSize = 1728; // (3*4)^3
   if(this->reference->nz==1){
      globalWorkSize[2] = 1;
      localWorkSize[2] = 1;
      sMemSize = 144; // (3*4)^2
   }

   errNum = clSetKernelArg(kernel, 0, sMemSize * sizeof(cl_float), NULL);
   this->sContext->checkErrNum(errNum, "Error setting shared memory.");
   errNum = clSetKernelArg(kernel, 1, sizeof(cl_mem), &this->clWarpedImageArray);
   this->sContext->checkErrNum(errNum, "Error setting resultImageArray.");
   errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &this->clReferenceImageArray);
   this->sContext->checkErrNum(errNum, "Error setting targetImageArray.");
   errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &this->clWarpedPosition);
   this->sContext->checkErrNum(errNum, "Error setting resultPosition.");
   errNum |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &this->clReferencePosition);
   this->sContext->checkErrNum(errNum, "Error setting targetPosition.");
   errNum |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &this->clTotalBlock);
   this->sContext->checkErrNum(errNum, "Error setting mask.");
   errNum |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &this->clMask);
   this->sContext->checkErrNum(errNum, "Error setting mask.");
   errNum |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &this->clReferenceMat);
   this->sContext->checkErrNum(errNum, "Error setting targetMatrix_xyz.");
   errNum |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &cldefinedBlock);
   this->sContext->checkErrNum(errNum, "Error setting cldefinedBlock.");
   errNum |= clSetKernelArg(kernel, 9, sizeof(cl_uint4), &imageSize);
   this->sContext->checkErrNum(errNum, "Error setting image size.");

   errNum = clEnqueueNDRangeKernel(this->commandQueue, kernel, params->dim, NULL,
                                   globalWorkSize, localWorkSize, 0, NULL, NULL);
   this->sContext->checkErrNum(errNum, "Error queuing blockmatching kernel for execution ");

   errNum = clFinish(this->commandQueue);
   this->sContext->checkErrNum(errNum, "Error after clFinish CLBlockMatchingKernel");

   errNum = clEnqueueReadBuffer(this->commandQueue, cldefinedBlock, CL_TRUE, 0, sizeof(int),
                                &(this->params->definedActiveBlockNumber), 0, NULL, NULL);
   sContext->checkErrNum(errNum, "Error reading  var after CLBlockMatchingKernel execution ");

   if(this->params->definedActiveBlockNumber == 0) {
      reg_print_msg_error("Unexpected error in the CLBlockMatchingKernel execution");
      reg_exit();
   }
   clReleaseMemObject(cldefinedBlock);
}
/* *************************************************************** */
CLBlockMatchingKernel::~CLBlockMatchingKernel() {
   if (kernel != 0)
      clReleaseKernel(kernel);
   if (program != 0)
      clReleaseProgram(program);
}
