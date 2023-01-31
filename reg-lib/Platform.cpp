#include "Platform.h"
#include "CpuKernelFactory.h"
#ifdef _USE_CUDA
#include "CudaKernelFactory.h"
#include "CudaF3dContent.h"
#include "CudaComputeFactory.h"
#include "CudaContextSingleton.h"
#include "CudaMeasureFactory.h"
#include "_reg_optimiser_gpu.h"
#endif
#ifdef _USE_OPENCL
#include "ClKernelFactory.h"
#include "ClComputeFactory.h"
#include "ClContextSingleton.h"
#endif

/* *************************************************************** */
Platform::Platform(const PlatformType& platformTypeIn) {
    platformType = platformTypeIn;
    if (platformType == PlatformType::Cpu) {
        kernelFactory = new CpuKernelFactory();
        computeFactory = new ComputeFactory();
        measureFactory = new MeasureFactory();
        platformName = "cpu_platform";
    }
#ifdef _USE_CUDA
    else if (platformType == PlatformType::Cuda) {
        kernelFactory = new CudaKernelFactory();
        computeFactory = new CudaComputeFactory();
        measureFactory = new CudaMeasureFactory();
        platformName = "cuda_platform";
    }
#endif
#ifdef _USE_OPENCL
    else if (platformType == PlatformType::OpenCl) {
        kernelFactory = new ClKernelFactory();
        computeFactory = new ClComputeFactory();
        platformName = "cl_platform";
    }
#endif
}
/* *************************************************************** */
Compute* Platform::CreateCompute(Content& con) const {
    return computeFactory->Produce(con);
}
/* *************************************************************** */
Kernel* Platform::CreateKernel(const std::string& name, Content *con) const {
    return kernelFactory->Produce(name, con);
}
/* *************************************************************** */
template<typename Type>
reg_optimiser<Type>* Platform::CreateOptimiser(F3dContent& con,
                                               InterfaceOptimiser& opt,
                                               size_t maxIterationNumber,
                                               bool useConjGradient,
                                               bool optimiseX,
                                               bool optimiseY,
                                               bool optimiseZ,
                                               F3dContent *conBw) const {
    reg_optimiser<Type> *optimiser;
    nifti_image *controlPointGrid = con.F3dContent::GetControlPointGrid();
    nifti_image *controlPointGridBw = conBw ? conBw->F3dContent::GetControlPointGrid() : nullptr;
    Type *controlPointGridData, *transformationGradientData;
    Type *controlPointGridDataBw = nullptr, *transformationGradientDataBw = nullptr;

    if (platformType == PlatformType::Cpu) {
        optimiser = useConjGradient ? new reg_conjugateGradient<Type>() : new reg_optimiser<Type>();
        controlPointGridData = (Type*)controlPointGrid->data;
        transformationGradientData = (Type*)con.GetTransformationGradient()->data;
        if (conBw) {
            controlPointGridDataBw = (Type*)controlPointGridBw->data;
            transformationGradientDataBw = (Type*)conBw->GetTransformationGradient()->data;
        }
    }
#ifdef _USE_CUDA
    else if (platformType == PlatformType::Cuda) {
        optimiser = dynamic_cast<reg_optimiser<Type>*>(useConjGradient ? new reg_conjugateGradient_gpu() : new reg_optimiser_gpu());
        controlPointGridData = (Type*)dynamic_cast<CudaF3dContent&>(con).GetControlPointGridCuda();
        transformationGradientData = (Type*)dynamic_cast<CudaF3dContent&>(con).GetTransformationGradientCuda();
        if (conBw) {
            controlPointGridDataBw = (Type*)dynamic_cast<CudaF3dContent*>(conBw)->GetControlPointGridCuda();
            transformationGradientDataBw = (Type*)dynamic_cast<CudaF3dContent*>(conBw)->GetTransformationGradientCuda();
        }
    }
#endif

    optimiser->Initialise(controlPointGrid->nvox,
                          controlPointGrid->nz > 1 ? 3 : 2,
                          optimiseX,
                          optimiseY,
                          optimiseZ,
                          maxIterationNumber,
                          0, // currentIterationNumber,
                          &opt,
                          controlPointGridData,
                          transformationGradientData,
                          controlPointGridBw ? controlPointGridBw->nvox : 0,
                          controlPointGridDataBw,
                          transformationGradientDataBw);

    return optimiser;
}
template reg_optimiser<float>* Platform::CreateOptimiser(F3dContent&, InterfaceOptimiser&, size_t, bool, bool, bool, bool, F3dContent*) const;
template reg_optimiser<double>* Platform::CreateOptimiser(F3dContent&, InterfaceOptimiser&, size_t, bool, bool, bool, bool, F3dContent*) const;
/* *************************************************************** */
Measure* Platform::CreateMeasure() const {
    return measureFactory->Produce();
}
/* *************************************************************** */
std::string Platform::GetName() const {
    return platformName;
}
/* *************************************************************** */
unsigned int Platform::GetGpuIdx() const {
    return gpuIdx;
}
/* *************************************************************** */
void Platform::SetGpuIdx(unsigned gpuIdxIn) {
    if (platformType == PlatformType::Cpu) {
        gpuIdx = 999;
    }
#ifdef _USE_CUDA
    else if (platformType == PlatformType::Cuda) {
        CudaContextSingleton *cudaContext = &CudaContextSingleton::Instance();
        if (gpuIdxIn != 999) {
            gpuIdx = gpuIdxIn;
            cudaContext->SetCudaIdx(gpuIdxIn);
        }
    }
#endif
#ifdef _USE_OPENCL
    else if (platformType == PlatformType::OpenCl) {
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
PlatformType Platform::GetPlatformType() const {
    return platformType;
}
/* *************************************************************** */
Platform::~Platform() {
    delete kernelFactory;
    delete computeFactory;
}
/* *************************************************************** */
