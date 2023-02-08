#include "ClResampleImageKernel.h"
#include "config.h"
#include "_reg_tools.h"
#include <algorithm>

/* *************************************************************** */
ClResampleImageKernel::ClResampleImageKernel(Content *conIn) : ResampleImageKernel() {
    //populate the CLContext object ptr
    ClAladinContent *con = static_cast<ClAladinContent*>(conIn);

    //path to kernel file
    const char *niftyreg_install_dir = getenv("NIFTYREG_INSTALL_DIR");
    const char *niftyreg_src_dir = getenv("NIFTYREG_SRC_DIR");

    std::string clInstallPath;
    std::string clSrcPath;
    //src dir
    if (niftyreg_src_dir != nullptr) {
        char opencl_kernel_path[255];
        sprintf(opencl_kernel_path, "%s/reg-lib/cl/", niftyreg_src_dir);
        clSrcPath = opencl_kernel_path;
    } else clSrcPath = CL_KERNELS_SRC_PATH;
    //install dir
    if (niftyreg_install_dir != nullptr) {
        char opencl_kernel_path[255];
        sprintf(opencl_kernel_path, "%s/include/cl/", niftyreg_install_dir);
        clInstallPath = opencl_kernel_path;
    } else clInstallPath = CL_KERNELS_PATH;
    std::string clKernel("resampleKernel.cl");
    //Let's check if we did an install
    std::string clKernelPath = (clInstallPath + clKernel);
    std::ifstream kernelFile(clKernelPath.c_str(), std::ios::in);
    if (kernelFile.is_open() == 0) {
        //"clKernel.cl propbably not installed - let's use the src location"
        clKernelPath = (clSrcPath + clKernel);
    }

    //get opencl context params
    sContext = &ClContextSingleton::Instance();
    clContext = sContext->GetContext();
    commandQueue = sContext->GetCommandQueue();
    program = sContext->CreateProgram(clKernelPath.c_str());

    //get cpu ptrs
    floatingImage = con->AladinContent::GetFloating();
    warpedImage = con->AladinContent::GetWarped();
    mask = con->AladinContent::GetReferenceMask();

    //get cl ptrs
    clFloating = con->GetFloatingImageArrayClmem();
    clDeformationField = con->GetDeformationFieldArrayClmem();
    clWarped = con->GetWarpedImageClmem();
    clMask = con->GetMaskClmem();
    floMat = con->GetFloMatClmem();

    //init kernel
    kernel = 0;
}
/* *************************************************************** */
void ClResampleImageKernel::Calculate(int interp,
                                      float paddingValue,
                                      bool *dti_timepoint,
                                      mat33 *jacMat) {
    cl_int errNum;
    // Define the DTI indices if required
    if (dti_timepoint != nullptr || jacMat != nullptr) {
        reg_print_fct_error("ClResampleImageKernel::calculate");
        reg_print_msg_error("The DTI resampling has not yet been implemented with the OpenCL platform. Exit.");
        reg_exit();
    }

    if (this->floatingImage->nz > 1) {
        this->kernel = clCreateKernel(program, "ResampleImage3D", &errNum);
    } else if (this->floatingImage->nz == 1) {
        //2D case
        this->kernel = clCreateKernel(program, "ResampleImage2D", &errNum);
    } else {
        reg_print_fct_error("ClResampleImageKernel::calculate");
        reg_print_msg_error("The image dimension is not supported. Exit.");
        reg_exit();
    }
    sContext->checkErrNum(errNum, "Error setting kernel ResampleImage.");

    const size_t targetVoxelNumber = CalcVoxelNumber(*this->warpedImage);
    const unsigned int maxThreads = sContext->GetMaxThreads();
    const unsigned int maxBlocks = sContext->GetMaxBlocks();

    unsigned int blocks = (targetVoxelNumber % maxThreads) ? (targetVoxelNumber / maxThreads) + 1 : targetVoxelNumber / maxThreads;
    blocks = std::min(blocks, maxBlocks);

    const cl_uint dims = 1;
    const size_t globalWorkSize[dims] = {blocks * maxThreads};
    const size_t localWorkSize[dims] = {maxThreads};

    //    int numMats = 0; //needs to be a parameter
    //    float* jacMat_h = (float*) malloc(9 * numMats * sizeof(float));

    cl_long2 voxelNumber = {{(cl_long)CalcVoxelNumber(*warpedImage), (cl_long)CalcVoxelNumber(*this->floatingImage)}};
    cl_uint3 fi_xyz = {{(cl_uint)floatingImage->nx, (cl_uint)floatingImage->ny, (cl_uint)floatingImage->nz}};
    cl_uint2 wi_tu = {{(cl_uint)warpedImage->nt, (cl_uint)warpedImage->nu}};

    //    if (numMats)
    //        mat33ToCptr(jacMat, jacMat_h, numMats);

    int datatype = this->floatingImage->datatype;

    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &this->clFloating);
    sContext->checkErrNum(errNum, "Error setting interp kernel arguments 0.");
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &this->clDeformationField);
    sContext->checkErrNum(errNum, "Error setting interp kernel arguments 1.");
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &this->clWarped);
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

    errNum = clEnqueueNDRangeKernel(commandQueue, kernel, dims, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
    sContext->checkErrNum(errNum, "Error queuing interp kernel for execution: ");

    clFinish(commandQueue);
}
/* *************************************************************** */
ClResampleImageKernel::~ClResampleImageKernel() {
    if (kernel != 0)
        clReleaseKernel(kernel);
    if (program != 0)
        clReleaseProgram(program);
}
/* *************************************************************** */
