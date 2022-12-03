#include "Platform.h"

/* *************************************************************** */
Platform::Platform(int platformCodeIn) {
    platformCode = platformCodeIn;
    if (platformCode == NR_PLATFORM_CPU) {
        kernelFactory = new CpuKernelFactory();
        computeFactory = new ComputeFactory();
        platformName = "cpu_platform";
    }
#ifdef _USE_CUDA
    else if (platformCode == NR_PLATFORM_CUDA) {
        kernelFactory = new CudaKernelFactory();
        computeFactory = new CudaComputeFactory();
        platformName = "cuda_platform";
    }
#endif
#ifdef _USE_OPENCL
    else if (platformCode == NR_PLATFORM_CL) {
        kernelFactory = new ClKernelFactory();
        computeFactory = new ClComputeFactory();
        platformName = "cl_platform";
    }
#endif
}
/* *************************************************************** */
Compute* Platform::CreateCompute(Content *con) const {
    return computeFactory->Produce(con);
}
/* *************************************************************** */
Kernel* Platform::CreateKernel(const std::string& name, Content *con) const {
    return kernelFactory->Produce(name, con);
}
/* *************************************************************** */
template<typename Type>
reg_optimiser<Type>* Platform::CreateOptimiser(F3dContent *con,
                                               InterfaceOptimiser *opt,
                                               size_t maxIterationNumber,
                                               bool useConjGradient,
                                               bool optimiseX,
                                               bool optimiseY,
                                               bool optimiseZ) {
    reg_optimiser<Type> *optimiser;
    nifti_image *controlPointGrid = con->F3dContent::GetControlPointGrid();
    Type *controlPointGridData, *transformationGradientData;

    if (platformCode == NR_PLATFORM_CPU) {
        optimiser = useConjGradient ? new reg_conjugateGradient<Type>() : new reg_optimiser<Type>();
        controlPointGridData = (Type*)controlPointGrid->data;
        transformationGradientData = (Type*)con->F3dContent::GetTransformationGradient()->data;
    }
#ifdef _USE_CUDA
    else if (platformCode == NR_PLATFORM_CUDA) {
        optimiser = dynamic_cast<reg_optimiser<Type>*>(useConjGradient ? new reg_conjugateGradient_gpu() : new reg_optimiser_gpu());
        controlPointGridData = (Type*)dynamic_cast<CudaF3dContent*>(con)->GetControlPointGridCuda();
        transformationGradientData = (Type*)dynamic_cast<CudaF3dContent*>(con)->GetTransformationGradientCuda();
    }
#endif

    optimiser->Initialise(controlPointGrid->nvox,
                          controlPointGrid->nz > 1 ? 3 : 2,
                          optimiseX,
                          optimiseY,
                          optimiseZ,
                          maxIterationNumber,
                          0, // currentIterationNumber,
                          opt,
                          controlPointGridData,
                          transformationGradientData);

    return optimiser;
}
template reg_optimiser<float>* Platform::CreateOptimiser(F3dContent*, InterfaceOptimiser*, size_t, bool, bool, bool, bool);
template reg_optimiser<double>* Platform::CreateOptimiser(F3dContent*, InterfaceOptimiser*, size_t, bool, bool, bool, bool);
/* *************************************************************** */
std::string Platform::GetName() {
    return platformName;
}
/* *************************************************************** */
unsigned Platform::GetGpuIdx() {
    return gpuIdx;
}
/* *************************************************************** */
void Platform::SetGpuIdx(unsigned gpuIdxIn) {
    if (platformCode == NR_PLATFORM_CPU) {
        gpuIdx = 999;
    }
#ifdef _USE_CUDA
    else if (platformCode == NR_PLATFORM_CUDA) {
        CudaContextSingleton *cudaContext = &CudaContextSingleton::Instance();
        if (gpuIdxIn != 999) {
            gpuIdx = gpuIdxIn;
            cudaContext->SetCudaIdx(gpuIdxIn);
        }
    }
#endif
#ifdef _USE_OPENCL
    else if (platformCode == NR_PLATFORM_CL) {
        ClContextSingleton *sContext = &ClContextSingleton::Instance();
        if (gpuIdxIn != 999) {
            gpuIdx = gpuIdxIn;
            sContext->SetClIdx(gpuIdxIn);
        }

        std::size_t paramValueSize;
        sContext->checkErrNum(clGetDeviceInfo(sContext->GetDeviceId(), CL_DEVICE_TYPE, 0, nullptr, &paramValueSize), "Failed to find OpenCL device info ");
        cl_device_type *field = (cl_device_type *)alloca(sizeof(cl_device_type) * paramValueSize);
        sContext->checkErrNum(clGetDeviceInfo(sContext->GetDeviceId(), CL_DEVICE_TYPE, paramValueSize, field, nullptr), "Failed to find OpenCL device info ");
        if (CL_DEVICE_TYPE_CPU == *field) {
            reg_print_fct_error("Platform::setClIdx");
            reg_print_msg_error("The OpenCL kernels only support GPU devices for now. Exit");
            reg_exit();
        }
    }
#endif
}
/* *************************************************************** */
int Platform::GetPlatformCode() {
    return platformCode;
}
/* *************************************************************** */
//void Platform::SetPlatformCode(const int platformCodeIn) {
//    platformCode = platformCodeIn;
//}
/* *************************************************************** */
Platform::~Platform() {
    delete kernelFactory;
    delete computeFactory;
}
/* *************************************************************** */
