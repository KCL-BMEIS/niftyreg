// CUDA runtime plugin implementation.
//
// Compiled into the plugin shared library (libniftyreg_cuda.so / niftyreg_cuda.dll)
// with USE_CUDA, so it may freely use the concrete Cuda* classes and CUDA headers.
// It exposes them to the host through the CUDA-free CudaPluginInterface, mirroring
// exactly what Platform's PlatformType::Cuda branch does in a static USE_CUDA build.
//
// In a static host build the CPU libraries are linked into this shared object so it
// always loads with no unresolved symbols; the shared C++ classes are then
// duplicated between the host and the plugin, which is safe with the toolchains used
// here because RTTI is matched by name across module boundaries (libstdc++ always;
// MSVC likewise). In a shared host build the plugin links the same shared libraries
// as the executables, so nothing is duplicated.

#include "CudaPluginInterface.h"

// Base factory types. The Cuda*Factory headers assume their base is already
// included; CudaMeasureCreatorFactory.hpp in particular does not pull in
// MeasureCreatorFactory.hpp (in the static build Platform.h has already included
// it). Include them explicitly here.
#include "ComputeFactory.h"
#include "ContentCreatorFactory.h"
#include "KernelFactory.h"
#include "MeasureCreatorFactory.hpp"

#include "CudaComputeFactory.h"
#include "CudaContentCreatorFactory.h"
#include "CudaKernelFactory.h"
#include "CudaMeasureCreatorFactory.hpp"
#include "CudaContext.hpp"
#include "CudaF3dContent.h"
#include "CudaOptimiser.hpp"
#include "Optimiser.hpp"
#include "_reg_cudainfo.h"

#include <cuda.h>

// Optimiser, InterfaceOptimiser, CudaContext, CudaOptimiser and
// CudaConjugateGradient all live in the NiftyReg namespace.
using namespace NiftyReg;

namespace {
/* *************************************************************** */
// True when a device of the given compute capability (major*10+minor) can run this
// build's kernels. NR_CUDA_COMPILED_ARCHS is the comma-joined
// CMAKE_CUDA_ARCHITECTURES the plugin was compiled with (e.g.
// "60-real,61-real,70-real,75-real,80-real,86-real,89"): a plain (or -virtual)
// entry embeds PTX, which JIT-compiles on that architecture and anything newer; a
// -real entry embeds SASS only, which runs on the same major architecture at an
// equal or higher minor revision.
bool DeviceIsSupported(const int computeCapability) {
    const std::string archs(NR_CUDA_COMPILED_ARCHS);
    size_t start = 0;
    while (start < archs.size()) {
        size_t end = archs.find(',', start);
        if (end == std::string::npos) end = archs.size();
        const std::string entry = archs.substr(start, end - start);
        start = end + 1;
        int arch = 0;
        size_t digit = 0;
        while (digit < entry.size() && entry[digit] >= '0' && entry[digit] <= '9')
            arch = arch * 10 + (entry[digit++] - '0');
        if (arch == 0) continue;
        if (entry.find("-real") != std::string::npos) {
            if (computeCapability / 10 == arch / 10 && computeCapability >= arch)
                return true;
        } else if (computeCapability >= arch) {
            return true;
        }
    }
    return false;
}
/* *************************************************************** */
class CudaPluginImpl: public CudaPluginInterface {
public:
    bool IsAvailable() override {
        // Driver API only: unlike the runtime API, none of these calls creates a
        // CUDA context, so probing leaves no state behind for the later best-card
        // selection to trip over
        if (cuInit(0) != CUDA_SUCCESS) return false;
        int deviceCount = 0;
        if (cuDeviceGetCount(&deviceCount) != CUDA_SUCCESS) return false;
        for (int deviceIdx = 0; deviceIdx < deviceCount; deviceIdx++) {
            CUdevice device;
            int major = 0, minor = 0;
            if (cuDeviceGet(&device, deviceIdx) != CUDA_SUCCESS) continue;
            if (cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device) != CUDA_SUCCESS ||
                cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device) != CUDA_SUCCESS)
                continue;
            if (DeviceIsSupported(major * 10 + minor))
                return true;
        }
        return false;
    }

    void SetGpuIdx(unsigned gpuIdx) override {
        // GetInstance() runs unconditionally so the default index (999) still
        // creates the context eagerly (max-Gflops card pick) before any allocation,
        // exactly as the static build's Platform::SetGpuIdx does
        CudaContext& cudaContext = CudaContext::GetInstance();
        if (gpuIdx != 999)
            cudaContext.SetCudaIdx(gpuIdx);
    }

    ComputeFactory* CreateComputeFactory() override { return new CudaComputeFactory(); }
    ContentCreatorFactory* CreateContentCreatorFactory() override { return new CudaContentCreatorFactory(); }
    KernelFactory* CreateKernelFactory() override { return new CudaKernelFactory(); }
    MeasureCreatorFactory* CreateMeasureCreatorFactory() override { return new CudaMeasureCreatorFactory(); }

    Optimiser<float>* CreateOptimiser(F3dContent& con,
                                      InterfaceOptimiser& opt,
                                      size_t maxIterationNumber,
                                      bool useConjGradient,
                                      bool optimiseX,
                                      bool optimiseY,
                                      bool optimiseZ,
                                      F3dContent *conBw) override {
        // Mirrors Platform::CreateOptimiser<float>'s PlatformType::Cuda branch
        nifti_image *controlPointGrid = con.F3dContent::GetControlPointGrid();
        nifti_image *controlPointGridBw = conBw ? conBw->F3dContent::GetControlPointGrid() : nullptr;

        Optimiser<float> *optimiser = useConjGradient ? new CudaConjugateGradient() : new CudaOptimiser();
        float *controlPointGridData = reinterpret_cast<float*>(dynamic_cast<CudaF3dContent&>(con).GetControlPointGridCuda());
        float *transformationGradientData = reinterpret_cast<float*>(dynamic_cast<CudaF3dContent&>(con).GetTransformationGradientCuda());
        float *controlPointGridDataBw = nullptr, *transformationGradientDataBw = nullptr;
        if (conBw) {
            controlPointGridDataBw = reinterpret_cast<float*>(dynamic_cast<CudaF3dContent*>(conBw)->GetControlPointGridCuda());
            transformationGradientDataBw = reinterpret_cast<float*>(dynamic_cast<CudaF3dContent*>(conBw)->GetTransformationGradientCuda());
        }

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

    void ShowDeviceInfo() override {
        showCUDAInfo();
    }
};
/* *************************************************************** */
} // namespace
/* *************************************************************** */
extern "C" NR_CUDA_PLUGIN_EXPORT int nrCudaPluginAbi() {
    return NR_CUDA_PLUGIN_ABI;
}
/* *************************************************************** */
extern "C" NR_CUDA_PLUGIN_EXPORT const char* nrCudaPluginVersion() {
    return NR_VERSION;
}
/* *************************************************************** */
extern "C" NR_CUDA_PLUGIN_EXPORT CudaPluginInterface* nrCreateCudaPlugin() {
    static CudaPluginImpl instance;
    return &instance;
}
/* *************************************************************** */
