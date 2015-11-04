#include "CLBlockMatchingKernel.h"
#include "config.h"
#include <fstream>

CLBlockMatchingKernel::CLBlockMatchingKernel(AladinContent *conIn, std::string name) :
        BlockMatchingKernel(name) {
    //populate the CLAladinContent object ptr
    con = static_cast<ClAladinContent*>(conIn);

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
    sContext = &CLContextSingletton::Instance();
    clContext = sContext->getContext();
    commandQueue = sContext->getCommandQueue();
    program = sContext->CreateProgram(clKernelPath.c_str());

    // Create OpenCL kernel
    cl_int errNum;
    if (con->getBlockMatchingParams()->dim == 3) {
        kernel = clCreateKernel(program, "blockMatchingKernel3D", &errNum);
    }
    else {
        kernel = clCreateKernel(program, "blockMatchingKernel2D", &errNum);
    }
    sContext->checkErrNum(errNum, "Error setting bm kernel.");

    //get cl ptrs
    clTotalBlock = con->getTotalBlockClmem();
    clReferenceImageArray = con->getReferenceImageArrayClmem();
    clWarpedImageArray = con->getWarpedImageClmem();
    clWarpedPosition = con->getWarpedPositionClmem();
    clReferencePosition = con->getReferencePositionClmem();
    clMask = con->getMaskClmem();
    clReferenceMat = con->getRefMatClmem();

    //get cpu ptrs
    reference = con->AladinContent::getCurrentReference();
    params = con->AladinContent::getBlockMatchingParams();

}
/* *************************************************************** */
void CLBlockMatchingKernel::calculate() {
    // Copy some required parameters over to the device
    int *definedBlock = (int*)malloc(sizeof(int));
    *definedBlock = 0;
    cl_int errNum;
    cl_mem cldefinedBlock = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), definedBlock, &errNum);
    this->sContext->checkErrNum(errNum, "CLBlockMatchingKernel::calculate failed to allocate memory (cldefinedBlock) ");

    const unsigned int blockRange = params->voxelCaptureRange%4?params->voxelCaptureRange/4+1:params->voxelCaptureRange/4;
    const unsigned int stepSize = params->stepSize;

    const unsigned int numBlocks = blockRange * 2 + 1;

    cl_uint3 imageSize = {{(cl_uint)this->reference->nx,
                                  (cl_uint)this->reference->ny,
                                  (cl_uint)this->reference->nz,
                                 (cl_uint)0 }};
    
    if (imageSize.z > 1) {
        const size_t globalWorkSize[3] = { (size_t)params->blockNumber[0] * 4,
            (size_t)params->blockNumber[1] * 4,
            (size_t)params->blockNumber[2] * 4 };
        const size_t localWorkSize[3] = { 4, 4, 4 };
        const unsigned int sMemSize = numBlocks*numBlocks*numBlocks * 64;
        
        errNum = clSetKernelArg(kernel, 0, sMemSize * sizeof(cl_float), NULL);
        sContext->checkErrNum(errNum, "Error setting shared memory.");
        errNum = clSetKernelArg(kernel, 1, sizeof(cl_mem), &this->clWarpedImageArray);
        sContext->checkErrNum(errNum, "Error setting resultImageArray.");
        errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &this->clReferenceImageArray);
        sContext->checkErrNum(errNum, "Error setting targetImageArray.");
        errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &this->clWarpedPosition);
        sContext->checkErrNum(errNum, "Error setting resultPosition.");
        errNum |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &this->clReferencePosition);
        sContext->checkErrNum(errNum, "Error setting targetPosition.");
        errNum |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &this->clTotalBlock);
        sContext->checkErrNum(errNum, "Error setting mask.");
        errNum |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &this->clMask);
        sContext->checkErrNum(errNum, "Error setting mask.");
        errNum |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &this->clReferenceMat);
        sContext->checkErrNum(errNum, "Error setting targetMatrix_xyz.");
        errNum |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &cldefinedBlock);
        sContext->checkErrNum(errNum, "Error setting cldefinedBlock.");
        errNum |= clSetKernelArg(kernel, 9, sizeof(cl_uint3), &imageSize);
        sContext->checkErrNum(errNum, "Error setting image size.");
        errNum |= clSetKernelArg(kernel, 10, sizeof(cl_uint), &blockRange);
        sContext->checkErrNum(errNum, "Error setting blockRange.");
        errNum |= clSetKernelArg(kernel, 11, sizeof(cl_uint), &stepSize);
        sContext->checkErrNum(errNum, "Error setting step size.");

        errNum = clEnqueueNDRangeKernel(commandQueue, kernel, params->dim, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
        sContext->checkErrNum(errNum, "Error queuing blockmatching kernel for execution ");
    }
    else {
        const size_t globalWorkSize[2] = { (size_t)params->blockNumber[0] * 4,
            (size_t)params->blockNumber[1] * 4};
        const size_t localWorkSize[2] = { 4, 4};
        const unsigned int sMemSize = numBlocks*numBlocks * 16;

        errNum = clSetKernelArg(kernel, 0, sMemSize * sizeof(cl_float), NULL);
        sContext->checkErrNum(errNum, "Error setting shared memory.");
        errNum = clSetKernelArg(kernel, 1, sizeof(cl_mem), &this->clWarpedImageArray);
        sContext->checkErrNum(errNum, "Error setting resultImageArray.");
        errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &this->clReferenceImageArray);
        sContext->checkErrNum(errNum, "Error setting targetImageArray.");
        errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &this->clWarpedPosition);
        sContext->checkErrNum(errNum, "Error setting resultPosition.");
        errNum |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &this->clReferencePosition);
        sContext->checkErrNum(errNum, "Error setting targetPosition.");
        errNum |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &this->clTotalBlock);
        sContext->checkErrNum(errNum, "Error setting mask.");
        errNum |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &this->clMask);
        sContext->checkErrNum(errNum, "Error setting mask.");
        errNum |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &this->clReferenceMat);
        sContext->checkErrNum(errNum, "Error setting targetMatrix_xyz.");
        errNum |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &cldefinedBlock);
        sContext->checkErrNum(errNum, "Error setting cldefinedBlock.");
        errNum |= clSetKernelArg(kernel, 9, sizeof(cl_uint3), &imageSize);
        sContext->checkErrNum(errNum, "Error setting image size.");
        errNum |= clSetKernelArg(kernel, 10, sizeof(cl_uint), &blockRange);
        sContext->checkErrNum(errNum, "Error setting blockRange.");
        errNum |= clSetKernelArg(kernel, 11, sizeof(cl_uint), &stepSize);
        sContext->checkErrNum(errNum, "Error setting step size.");

        errNum = clEnqueueNDRangeKernel(commandQueue, kernel, params->dim, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
        sContext->checkErrNum(errNum, "Error queuing blockmatching kernel for execution ");
    }

    errNum = clEnqueueReadBuffer(commandQueue, cldefinedBlock, CL_TRUE, 0, sizeof(int), definedBlock, 0, NULL, NULL);
    sContext->checkErrNum(errNum, "Error reading  var after CLBlockMatchingKernel execution ");
    params->definedActiveBlockNumber = *definedBlock;

    //PATCH TO CHECK IF EVERYTHING GOES WELL - DID NOT FIND ANOTHER WAY BECAUSE errNum = 0
    //I do not know why
    if(params->definedActiveBlockNumber == 0) {
        reg_print_msg_error("error in the CLBlockMatchingKernel execution (should be a memory problem)");
        reg_exit(1);
    }

    errNum = clFinish(commandQueue);
    sContext->checkErrNum(errNum, "Error after clFinish CLBlockMatchingKernel");

    free(definedBlock);
    clReleaseMemObject(cldefinedBlock);
}
/* *************************************************************** */
CLBlockMatchingKernel::~CLBlockMatchingKernel() {
    if (kernel != 0)
        clReleaseKernel(kernel);
    if (program != 0)
        clReleaseProgram(program);
}
