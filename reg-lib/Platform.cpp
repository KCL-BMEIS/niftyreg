#include "Platform.h"
#include "CpuKernelFactory.h"
// USE_CUDA marks the CUDA platform as available in both build modes. When
// USE_CUDA_PLUGIN is also set, CUDA is reached through the runtime plugin (no CUDA
// headers or symbols here); otherwise it is compiled in statically.
#if defined(USE_CUDA) && !defined(USE_CUDA_PLUGIN)
#include "CudaContext.hpp"
#include "CudaF3dContent.h"
#include "CudaComputeFactory.h"
#include "CudaContentCreatorFactory.h"
#include "CudaKernelFactory.h"
#include "CudaMeasureCreatorFactory.hpp"
#include "CudaOptimiser.hpp"
#elif defined(USE_CUDA_PLUGIN)
#include "CudaPluginInterface.h"
#include <type_traits>
#endif
#ifdef USE_OPENCL
#include "ClContextSingleton.h"
#include "ClComputeFactory.h"
#include "ClContentCreatorFactory.h"
#include "ClKernelFactory.h"
#endif

/* *************************************************************** */
Platform::Platform(const PlatformType platformTypeIn) {
    platformType = platformTypeIn;
    if (platformType == PlatformType::Cpu) {
        platformName = "CPU";
        computeFactory = new ComputeFactory();
        contentCreatorFactory = new ContentCreatorFactory();
        kernelFactory = new CpuKernelFactory();
        measureCreatorFactory = new MeasureCreatorFactory();
    }
#if defined(USE_CUDA) && !defined(USE_CUDA_PLUGIN)
    else if (platformType == PlatformType::Cuda) {
        platformName = "CUDA";
        SetGpuIdx(999);
        computeFactory = new CudaComputeFactory();
        contentCreatorFactory = new CudaContentCreatorFactory();
        kernelFactory = new CudaKernelFactory();
        measureCreatorFactory = new CudaMeasureCreatorFactory();
    }
#elif defined(USE_CUDA_PLUGIN)
    else if (platformType == PlatformType::Cuda) {
        // The CUDA implementation lives in the runtime plugin. An explicit CUDA
        // request that cannot be honoured is a hard error - never a silent CPU
        // fallback, which would let pipelines record CPU results as GPU runs
        CudaPluginInterface *cudaPlugin = loadCudaPlugin();
        if (!cudaPlugin)
            NR_FATAL_ERROR("The CUDA platform was requested but " + getCudaPluginLoadError() +
                           "\nUsing the CUDA platform requires the plugin alongside this installation "
                           "and the CUDA runtime libraries (CUDA toolkit) on this machine.");
        platformName = "CUDA";
        SetGpuIdx(999);
        computeFactory = cudaPlugin->CreateComputeFactory();
        contentCreatorFactory = cudaPlugin->CreateContentCreatorFactory();
        kernelFactory = cudaPlugin->CreateKernelFactory();
        measureCreatorFactory = cudaPlugin->CreateMeasureCreatorFactory();
    }
#endif
#ifdef USE_OPENCL
    else if (platformType == PlatformType::OpenCl) {
        platformName = "OpenCL";
        SetGpuIdx(999);
        computeFactory = new ClComputeFactory();
        contentCreatorFactory = new ClContentCreatorFactory();
        kernelFactory = new ClKernelFactory();
    }
#endif
    else NR_FATAL_ERROR("Unsupported platform type");
}
/* *************************************************************** */
Platform::~Platform() {
    delete computeFactory;
    delete contentCreatorFactory;
    delete kernelFactory;
    delete measureCreatorFactory;
}
/* *************************************************************** */
std::string Platform::GetName() const {
    return platformName;
}
/* *************************************************************** */
PlatformType Platform::GetPlatformType() const {
    return platformType;
}
/* *************************************************************** */
void Platform::SetGpuIdx(unsigned gpuIdxIn) {
    if (platformType == PlatformType::Cpu) {
        gpuIdx = 999;
    }
#if defined(USE_CUDA) && !defined(USE_CUDA_PLUGIN)
    else if (platformType == PlatformType::Cuda) {
        CudaContext& cudaContext = CudaContext::GetInstance();
        if (gpuIdxIn != 999) {
            gpuIdx = gpuIdxIn;
            cudaContext.SetCudaIdx(gpuIdxIn);
        }
    }
#elif defined(USE_CUDA_PLUGIN)
    else if (platformType == PlatformType::Cuda) {
        // Forwarded unconditionally: with the default index (999) the plugin still
        // creates the CUDA context eagerly (max-Gflops card pick) BEFORE any device
        // allocation, matching the static build's initialisation order
        if (gpuIdxIn != 999)
            gpuIdx = gpuIdxIn;
        CudaPluginInterface *cudaPlugin = loadCudaPlugin();
        if (cudaPlugin)
            cudaPlugin->SetGpuIdx(gpuIdxIn);
    }
#endif
#ifdef USE_OPENCL
    else if (platformType == PlatformType::OpenCl) {
        ClContextSingleton& clContext = ClContextSingleton::GetInstance();
        if (gpuIdxIn != 999) {
            gpuIdx = gpuIdxIn;
            clContext.SetClIdx(gpuIdxIn);
        }

        cl_device_type field;
        clContext.CheckErrNum(clGetDeviceInfo(clContext.GetDeviceId(), CL_DEVICE_TYPE, sizeof(field), &field, nullptr), "Failed to find OpenCL device info");
        if (CL_DEVICE_TYPE_CPU == field)
            NR_FATAL_ERROR("The OpenCL kernels only support GPU devices for now");
    }
#endif
}
/* *************************************************************** */
Compute* Platform::CreateCompute(Content& con) const {
    return computeFactory->Produce(con);
}
/* *************************************************************** */
ContentCreator* Platform::CreateContentCreator(const ContentType conType) const {
    return contentCreatorFactory->Produce(conType);
}
/* *************************************************************** */
Kernel* Platform::CreateKernel(const std::string& name, Content *con) const {
    return kernelFactory->Produce(name, con);
}
/* *************************************************************** */
MeasureCreator* Platform::CreateMeasureCreator() const {
    return measureCreatorFactory->Produce();
}
/* *************************************************************** */
template<typename Type>
Optimiser<Type>* Platform::CreateOptimiser(F3dContent& con,
                                           InterfaceOptimiser& opt,
                                           size_t maxIterationNumber,
                                           bool useConjGradient,
                                           bool optimiseX,
                                           bool optimiseY,
                                           bool optimiseZ,
                                           F3dContent *conBw) const {
#ifdef USE_CUDA_PLUGIN
    if (platformType == PlatformType::Cuda) {
        // The concrete CUDA optimiser needs the CudaF3dContent device pointers, so
        // it is built inside the plugin. It is single precision only (CudaOptimiser
        // derives from Optimiser<float>); the double path stays on the CPU
        if constexpr (std::is_same_v<Type, float>) {
            CudaPluginInterface *cudaPlugin = loadCudaPlugin();
            if (!cudaPlugin)
                NR_FATAL_ERROR("The CUDA plugin is unavailable: " + getCudaPluginLoadError());
            return cudaPlugin->CreateOptimiser(con, opt, maxIterationNumber, useConjGradient,
                                               optimiseX, optimiseY, optimiseZ, conBw);
        } else {
            NR_FATAL_ERROR("The CUDA platform only supports single-precision optimisation");
            return nullptr;
        }
    }
#endif
    Optimiser<Type> *optimiser;
    nifti_image *controlPointGrid = con.F3dContent::GetControlPointGrid();
    nifti_image *controlPointGridBw = conBw ? static_cast<nifti_image*>(conBw->F3dContent::GetControlPointGrid()) : nullptr;
    Type *controlPointGridData, *transformationGradientData;
    Type *controlPointGridDataBw = nullptr, *transformationGradientDataBw = nullptr;

    if (platformType == PlatformType::Cpu) {
        optimiser = useConjGradient ? new ConjugateGradient<Type>() : new Optimiser<Type>();
        controlPointGridData = static_cast<Type*>(controlPointGrid->data);
        transformationGradientData = static_cast<Type*>(con.GetTransformationGradient()->data);
        if (conBw) {
            controlPointGridDataBw = static_cast<Type*>(controlPointGridBw->data);
            transformationGradientDataBw = static_cast<Type*>(conBw->GetTransformationGradient()->data);
        }
    }
#if defined(USE_CUDA) && !defined(USE_CUDA_PLUGIN)
    else if (platformType == PlatformType::Cuda) {
        optimiser = dynamic_cast<Optimiser<Type>*>(useConjGradient ? new CudaConjugateGradient() : new CudaOptimiser());
        controlPointGridData = reinterpret_cast<Type*>(dynamic_cast<CudaF3dContent&>(con).GetControlPointGridCuda());
        transformationGradientData = reinterpret_cast<Type*>(dynamic_cast<CudaF3dContent&>(con).GetTransformationGradientCuda());
        if (conBw) {
            controlPointGridDataBw = reinterpret_cast<Type*>(dynamic_cast<CudaF3dContent*>(conBw)->GetControlPointGridCuda());
            transformationGradientDataBw = reinterpret_cast<Type*>(dynamic_cast<CudaF3dContent*>(conBw)->GetTransformationGradientCuda());
        }
    }
#endif

    optimiser->Initialise(controlPointGrid->nvox,
                          controlPointGrid->nz > 1 ? 3 : 2,
                          optimiseX,
                          optimiseY,
                          optimiseZ,
                          maxIterationNumber,
                          0,
                          &opt,
                          controlPointGridData,
                          transformationGradientData,
                          controlPointGridBw ? controlPointGridBw->nvox : 0,
                          controlPointGridDataBw,
                          transformationGradientDataBw);

    return optimiser;
}
template Optimiser<float>* Platform::CreateOptimiser(F3dContent&, InterfaceOptimiser&, size_t, bool, bool, bool, bool, F3dContent*) const;
template Optimiser<double>* Platform::CreateOptimiser(F3dContent&, InterfaceOptimiser&, size_t, bool, bool, bool, bool, F3dContent*) const;
/* *************************************************************** */
