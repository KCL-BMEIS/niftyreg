#include "CLResampleImageKernel.h"
#include "config.h"

#include "_reg_tools.h"

/* *************************************************************** */
CLResampleImageKernel::CLResampleImageKernel(Content *conIn, std::string name) : ResampleImageKernel(name) {
    //populate the CLContext object ptr
    con = static_cast<ClContent*>(conIn);

    //path to kernel file
    const char* niftyreg_install_dir = getenv("NIFTYREG_INSTALL_DIR");
    std::string clInstallPath;
    if(niftyreg_install_dir!=NULL){
        char opencl_kernel_path[255];
        sprintf(opencl_kernel_path, "%s/include/cl/", niftyreg_install_dir);
        clInstallPath = opencl_kernel_path;
    }
    else clInstallPath = CL_KERNELS_PATH;
    std::string clKernel("resampleKernel.cl");

    //get opencl context params
    sContext = &CLContextSingletton::Instance();
    clContext = sContext->getContext();
    commandQueue = sContext->getCommandQueue();
    program = sContext->CreateProgram((clInstallPath + clKernel).c_str());

    //get cpu ptrs
    floatingImage = con->Content::getCurrentFloating();
    warpedImage = con->Content::getCurrentWarped();
    mask = con->Content::getCurrentReferenceMask();

    //get cl ptrs
    clCurrentFloating = con->getFloatingImageArrayClmem();
    clCurrentDeformationField = con->getDeformationFieldArrayClmem();
    clCurrentWarped = con->getWarpedImageClmem();
    clMask = con->getMaskClmem();
    floMat = con->getFloMatClmem();

    //init kernel
    kernel = 0;
}
/* *************************************************************** */
void CLResampleImageKernel::calculate(int interp,
                                                  float paddingValue,
                                                  bool *dti_timepoint,
                                                  mat33 *jacMat) {
    cl_int errNum;
    // Define the DTI indices if required
    if(dti_timepoint!=NULL || jacMat!=NULL){
        reg_print_fct_error("CLResampleImageKernel::calculate");
        reg_print_msg_error("The DTI resampling has not yet been implemented with the OpenCL platform. Exit.");
        reg_exit(1);
    }

    kernel = clCreateKernel(program, "ResampleImage3D", &errNum);
    sContext->checkErrNum(errNum, "Error setting kernel ResampleImage3D.");

    long targetVoxelNumber = (long) warpedImage->nx * warpedImage->ny * warpedImage->nz;
    const unsigned int maxThreads = sContext->getMaxThreads();
    const unsigned int maxBlocks = sContext->getMaxBlocks();

    unsigned int blocks = (targetVoxelNumber % maxThreads) ? (targetVoxelNumber / maxThreads) + 1 : targetVoxelNumber / maxThreads;
    //blocks = min_cl(blocks, maxBlocks);
    blocks = std::min(blocks, maxBlocks);

    const cl_uint dims = 1;
    const size_t globalWorkSize[dims] = { blocks * maxThreads };
    const size_t localWorkSize[dims] = { maxThreads };

    int numMats = 0; //needs to be a parameter
    float* jacMat_h = (float*) malloc(9 * numMats * sizeof(float));

    cl_long2 voxelNumber = {{ (cl_long)warpedImage->nx * warpedImage->ny * warpedImage->nz,
                                      (cl_long)floatingImage->nx * floatingImage->ny * floatingImage->nz }};
    cl_uint3 fi_xyz = {{ (cl_uint)floatingImage->nx,
                                (cl_uint)floatingImage->ny,
                                (cl_uint)floatingImage->nz }};
    cl_uint2 wi_tu = {{ (cl_uint)warpedImage->nt,
                              (cl_uint)warpedImage->nu }};

    if (numMats)
        mat33ToCptr(jacMat, jacMat_h, numMats);
    int datatype = con->getFloatingDatatype();

    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &this->clCurrentFloating);
    sContext->checkErrNum(errNum, "Error setting interp kernel arguments 0.");
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &this->clCurrentDeformationField);
    sContext->checkErrNum(errNum, "Error setting interp kernel arguments 1.");
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &this->clCurrentWarped);
    sContext->checkErrNum(errNum, "Error setting interp kernel arguments 2.");
    errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &this->clMask);
    sContext->checkErrNum(errNum, "Error setting interp kernel arguments 3.");
    errNum |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &this->floMat);
    sContext->checkErrNum(errNum, "Error setting interp kernel arguments 4.");
    errNum |= clSetKernelArg(kernel, 5, sizeof(cl_long2), &voxelNumber);
    sContext->checkErrNum(errNum, "Error setting interp kernel arguments 5.");
    errNum |= clSetKernelArg(kernel, 6, sizeof(cl_uint3), &fi_xyz);
    sContext->checkErrNum(errNum, "Error setting interp kernel arguments 6.");
    errNum |= clSetKernelArg(kernel, 7, sizeof(cl_uint2), &wi_tu);
    sContext->checkErrNum(errNum, "Error setting interp kernel arguments 7.");
    errNum |= clSetKernelArg(kernel, 8, sizeof(float), &paddingValue);
    sContext->checkErrNum(errNum, "Error setting interp kernel arguments 8.");
    errNum |= clSetKernelArg(kernel, 9, sizeof(cl_int), &interp);
    sContext->checkErrNum(errNum, "Error setting interp kernel arguments 9.");
    errNum |= clSetKernelArg(kernel, 10, sizeof(cl_int), &datatype);
    sContext->checkErrNum(errNum, "Error setting interp kernel arguments 10.");

    errNum = clEnqueueNDRangeKernel(commandQueue, kernel, dims, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    sContext->checkErrNum(errNum, "Error queuing interp kernel for execution: ");

    clFinish(commandQueue);
}
/* *************************************************************** */
CLResampleImageKernel::~CLResampleImageKernel() {
    if (kernel != 0)
        clReleaseKernel(kernel);
    if (program != 0)
        clReleaseProgram(program);
}
/* *************************************************************** */
