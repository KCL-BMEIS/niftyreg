#include "ClAffineDeformationFieldKernel.h"
#include "config.h"
#include "_reg_tools.h"

/* *************************************************************** */
ClAffineDeformationFieldKernel::ClAffineDeformationFieldKernel(Content *conIn) : AffineDeformationFieldKernel() {
    //populate the ClAladinContent object ptr
    ClAladinContent *con = static_cast<ClAladinContent*>(conIn);

    //path to kernel files
    const char* niftyreg_install_dir = getenv("NIFTYREG_INSTALL_DIR");
    const char* niftyreg_src_dir = getenv("NIFTYREG_SRC_DIR");

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

    std::string clKernel("affineDeformationKernel.cl");

    //Let's check if we did an install
    std::string clKernelPath = (clInstallPath + clKernel);
    std::ifstream kernelFile(clKernelPath.c_str(), std::ios::in);
    if (kernelFile.is_open() == 0) {
        //"affineDeformationKernel.cl probably not installed - let's use the src location"
        clKernelPath = (clSrcPath + clKernel);
    }

    //get opencl context params
    sContext = &ClContextSingleton::GetInstance();
    clContext = sContext->GetContext();
    commandQueue = sContext->GetCommandQueue();
    program = sContext->CreateProgram(clKernelPath.c_str());

    //get cpu ptrs
    deformationFieldImage = con->AladinContent::GetDeformationField();
    affineTransformation = con->AladinContent::GetTransformationMatrix();
    referenceMatrix = AladinContent::GetXYZMatrix(*deformationFieldImage);

    cl_int errNum;
    // Create OpenCL kernel
    if (deformationFieldImage->nz > 1)
        kernel = clCreateKernel(program, "affineKernel3D", &errNum);
    else kernel = clCreateKernel(program, "affineKernel2D", &errNum);
    sContext->CheckErrNum(errNum, "Error setting kernel ClAffineDeformationFieldKernel.");

    //get cl ptrs
    clDeformationField = con->GetDeformationFieldArrayClmem();
    clMask = con->GetMaskClmem();

    //set some final kernel args
    errNum = clSetKernelArg(kernel, 2, sizeof(cl_mem), &clMask);
    sContext->CheckErrNum(errNum, "Error setting clMask.");

}
/* *************************************************************** */
void ClAffineDeformationFieldKernel::Calculate(bool compose) {
    //localWorkSize[0]*localWorkSize[1]*localWorkSize[2]... should be lower than the value specified by CL_DEVICE_MAX_WORK_GROUP_SIZE
    cl_uint maxWG = 0;
    cl_int errNum;
    std::size_t paramValueSize;
    errNum = clGetDeviceInfo(sContext->GetDeviceId(), CL_DEVICE_MAX_WORK_GROUP_SIZE, 0, nullptr, &paramValueSize);
    sContext->CheckErrNum(errNum, "Failed to GetDeviceId() OpenCL device info ");
    cl_uint * info = (cl_uint *)alloca(sizeof(cl_uint) * paramValueSize);
    errNum = clGetDeviceInfo(sContext->GetDeviceId(), CL_DEVICE_MAX_WORK_GROUP_SIZE, paramValueSize, info, nullptr);
    sContext->CheckErrNum(errNum, "Failed to GetDeviceId() OpenCL device info ");
    maxWG = *info;

    //8=default value
    unsigned xThreads = 8;
    unsigned yThreads = 8;
    unsigned zThreads = 8;

    while (xThreads * yThreads * zThreads > maxWG) {
        xThreads = xThreads / 2;
        yThreads = yThreads / 2;
        zThreads = zThreads / 2;
    }

    const unsigned xBlocks = ((deformationFieldImage->nx % xThreads) == 0) ?
        (deformationFieldImage->nx / xThreads) : (deformationFieldImage->nx / xThreads) + 1;
    const unsigned yBlocks = ((deformationFieldImage->ny % yThreads) == 0) ?
        (deformationFieldImage->ny / yThreads) : (deformationFieldImage->ny / yThreads) + 1;
    const unsigned zBlocks = ((deformationFieldImage->nz % zThreads) == 0) ?
        (deformationFieldImage->nz / zThreads) : (deformationFieldImage->nz / zThreads) + 1;
    //const cl_uint dims = deformationFieldImage->nz>1?3:2;
    //Back to the old version... at least I could compile
    const cl_uint dims = 3;
    const size_t globalWorkSize[dims] = {xBlocks * xThreads, yBlocks * yThreads, zBlocks * zThreads};
    const size_t localWorkSize[dims] = {xThreads, yThreads, zThreads};

    mat44 transformationMatrix = compose ? *affineTransformation : reg_mat44_mul(affineTransformation, referenceMatrix);

    float* trans = (float *)malloc(16 * sizeof(float));
    mat44ToCptr(transformationMatrix, trans);

    cl_uint3 pms_d = {{(cl_uint)deformationFieldImage->nx,
        (cl_uint)deformationFieldImage->ny,
        (cl_uint)deformationFieldImage->nz,
        (cl_uint)0}};

    cl_mem cltransMat = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(float) * 16, trans, &errNum);
    sContext->CheckErrNum(errNum,
                          "ClAffineDeformationFieldKernel::calculate failed to allocate memory (cltransMat): ");

    cl_uint composition = compose;
    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cltransMat);
    sContext->CheckErrNum(errNum, "Error setting cltransMat.");
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &clDeformationField);
    sContext->CheckErrNum(errNum, "Error setting clDeformationField.");
    errNum |= clSetKernelArg(kernel, 3, sizeof(cl_uint3), &pms_d);
    sContext->CheckErrNum(errNum, "Error setting kernel arguments.");
    errNum |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &composition);
    sContext->CheckErrNum(errNum, "Error setting kernel arguments.");

    errNum = clEnqueueNDRangeKernel(commandQueue, kernel, dims, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
    sContext->CheckErrNum(errNum, "Error queuing ClAffineDeformationFieldKernel for execution");
    clFinish(commandQueue);

    free(trans);
    clReleaseMemObject(cltransMat);
    return;
}
/* *************************************************************** */
ClAffineDeformationFieldKernel::~ClAffineDeformationFieldKernel() {
    if (kernel != 0)
        clReleaseKernel(kernel);
    if (program != 0)
        clReleaseProgram(program);
}
/* *************************************************************** */
