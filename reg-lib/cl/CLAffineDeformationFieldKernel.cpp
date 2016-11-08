#include "CLAffineDeformationFieldKernel.h"
#include "config.h"

#include "_reg_tools.h"

CLAffineDeformationFieldKernel::CLAffineDeformationFieldKernel(AladinContent *conIn, std::string nameIn) :
    AffineDeformationFieldKernel(nameIn) {
    //populate the CLAladinContent object ptr
    con = static_cast<ClAladinContent*>(conIn);

    //path to kernel files
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

    std::string clKernel("affineDeformationKernel.cl");

    //Let's check if we did an install
    std::string clKernelPath = (clInstallPath + clKernel);
    std::ifstream kernelFile(clKernelPath.c_str(), std::ios::in);
    if (kernelFile.is_open() == 0) {
        //"affineDeformationKernel.cl propbably not installed - let's use the src location"
        clKernelPath = (clSrcPath + clKernel);
    }

    //get opencl context params
    sContext = &CLContextSingletton::Instance();
    clContext = sContext->getContext();
    commandQueue = sContext->getCommandQueue();
    program = sContext->CreateProgram(clKernelPath.c_str());

    //get cpu ptrs
    this->deformationFieldImage = con->AladinContent::getCurrentDeformationField();
    this->affineTransformation = con->AladinContent::getTransformationMatrix();
    this->ReferenceMatrix = (this->deformationFieldImage->sform_code > 0) ? &(this->deformationFieldImage->sto_xyz) : &(this->deformationFieldImage->qto_xyz);

    cl_int errNum;
    // Create OpenCL kernel
    if(this->deformationFieldImage->nz>1)
        kernel = clCreateKernel(program, "affineKernel3D", &errNum);
    else kernel = clCreateKernel(program, "affineKernel2D", &errNum);
    sContext->checkErrNum(errNum, "Error setting kernel CLAffineDeformationFieldKernel.");

    //get cl ptrs
    clDeformationField = con->getDeformationFieldArrayClmem();
    clMask = con->getMaskClmem();

    //set some final kernel args
    errNum = clSetKernelArg(this->kernel, 2, sizeof(cl_mem), &this->clMask);
    sContext->checkErrNum(errNum, "Error setting clMask.");

}
/* *************************************************************** */
void CLAffineDeformationFieldKernel::calculate(bool compose) {
    //localWorkSize[0]*localWorkSize[1]*localWorkSize[2]... should be lower than the value specified by CL_DEVICE_MAX_WORK_GROUP_SIZE
    cl_uint maxWG = 0;
    cl_int errNum;
    std::size_t paramValueSize;
    errNum = clGetDeviceInfo(sContext->getDeviceId(), CL_DEVICE_MAX_WORK_GROUP_SIZE, 0, NULL, &paramValueSize);
    sContext->checkErrNum(errNum, "Failed to getDeviceId() OpenCL device info ");
    cl_uint * info = (cl_uint *) alloca(sizeof(cl_uint) * paramValueSize);
    errNum = clGetDeviceInfo(sContext->getDeviceId(), CL_DEVICE_MAX_WORK_GROUP_SIZE, paramValueSize, info, NULL);
    sContext->checkErrNum(errNum, "Failed to getDeviceId() OpenCL device info ");
    maxWG = *info;

    //8=default value
    unsigned int xThreads = 8;
    unsigned int yThreads = 8;
    unsigned int zThreads = 8;

    while(xThreads*yThreads*zThreads > maxWG) {
        xThreads = xThreads/2;
        yThreads = yThreads/2;
        zThreads = zThreads/2;
    }

    const unsigned int xBlocks = ((this->deformationFieldImage->nx % xThreads) == 0) ?
                (this->deformationFieldImage->nx / xThreads) : (this->deformationFieldImage->nx / xThreads) + 1;
    const unsigned int yBlocks = ((this->deformationFieldImage->ny % yThreads) == 0) ?
                (this->deformationFieldImage->ny / yThreads) : (this->deformationFieldImage->ny / yThreads) + 1;
    const unsigned int zBlocks = ((this->deformationFieldImage->nz % zThreads) == 0) ?
                (this->deformationFieldImage->nz / zThreads) : (this->deformationFieldImage->nz / zThreads) + 1;
    //const cl_uint dims = this->deformationFieldImage->nz>1?3:2;
    //Back to the old version... at least I could compile
    const cl_uint dims = 3;
    const size_t globalWorkSize[dims] = { xBlocks * xThreads, yBlocks * yThreads, zBlocks * zThreads };
    const size_t localWorkSize[dims] = { xThreads, yThreads, zThreads };

    mat44 transformationMatrix = (compose == true) ?
                *this->affineTransformation : reg_mat44_mul(this->affineTransformation, ReferenceMatrix);

    float* trans = (float *) malloc(16 * sizeof(float));
    mat44ToCptr(transformationMatrix, trans);

    cl_uint3 pms_d = {{ (cl_uint)this->deformationFieldImage->nx,
                        (cl_uint)this->deformationFieldImage->ny,
                        (cl_uint)this->deformationFieldImage->nz,
                        (cl_uint)0 }};

    cl_mem cltransMat = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(float) * 16, trans, &errNum);
    this->sContext->checkErrNum(errNum,
                                "CLAffineDeformationFieldKernel::calculate failed to allocate memory (cltransMat): ");

    cl_uint composition = compose;
    errNum = clSetKernelArg(this->kernel, 0, sizeof(cl_mem), &cltransMat);
    sContext->checkErrNum(errNum, "Error setting cltransMat.");
    errNum |= clSetKernelArg(this->kernel, 1, sizeof(cl_mem), &this->clDeformationField);
    sContext->checkErrNum(errNum, "Error setting clDeformationField.");
    errNum |= clSetKernelArg(this->kernel, 3, sizeof(cl_uint3), &pms_d);
    sContext->checkErrNum(errNum, "Error setting kernel arguments.");
    errNum |= clSetKernelArg(this->kernel, 4, sizeof(cl_uint), &composition);
    sContext->checkErrNum(errNum, "Error setting kernel arguments.");

    errNum = clEnqueueNDRangeKernel(this->commandQueue, kernel, dims, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    sContext->checkErrNum(errNum, "Error queuing CLAffineDeformationFieldKernel for execution");
    clFinish(commandQueue);

    free(trans);
    clReleaseMemObject(cltransMat);
    return;
}
/* *************************************************************** */
CLAffineDeformationFieldKernel::~CLAffineDeformationFieldKernel() {
    if (kernel != 0)
        clReleaseKernel(kernel);
    if (program != 0)
        clReleaseProgram(program);
}
