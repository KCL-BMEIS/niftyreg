#pragma once

// Host-side interface to the CUDA runtime plugin (libniftyreg_cuda.so / niftyreg_cuda.dll).
//
// When NiftyReg is built with USE_CUDA_PLUGIN, the CUDA platform is packaged as a
// shared library that the host binaries load at run time (see CudaPluginLoader.cpp),
// so a single build offers GPU acceleration where CUDA is usable and runs CPU-only
// everywhere else, without the host binaries linking any CUDA library.
//
// This header is compiled into the CPU/host targets and therefore MUST NOT pull in
// any CUDA headers or types. The plugin vends only the abstract factory/optimiser
// base types declared in reg-lib (never the concrete Cuda* subclasses), so no CUDA
// symbol crosses the boundary; objects created by the plugin are used purely through
// virtual calls on these base pointers.
//
// ABI rule: the loader and the plugin may come from different builds, so the
// boundary is versioned. The plugin exports nrCudaPluginAbi(), which the loader
// checks BEFORE any virtual call (a plugin with a different vtable layout makes even
// IsAvailable() unsafe), refusing a plugin whose value differs from
// NR_CUDA_PLUGIN_ABI. This interface is append-only: adding, removing, reordering or
// changing the signature of ANY virtual below requires incrementing NR_CUDA_PLUGIN_ABI.

#include <cstddef>
#include <string>

constexpr int NR_CUDA_PLUGIN_ABI = 1;

class ComputeFactory;
class ContentCreatorFactory;
class KernelFactory;
class MeasureCreatorFactory;
class F3dContent;
// Optimiser and InterfaceOptimiser live in the NiftyReg namespace; the factories and
// F3dContent are in the global namespace. They are referenced fully qualified below
// so this header stays consistent with the real declarations Platform.cpp pulls in.
namespace NiftyReg {
class InterfaceOptimiser;
template<typename Type> class Optimiser;
}

class CudaPluginInterface {
public:
    virtual ~CudaPluginInterface() = default;

    // True when at least one CUDA device is present AND its compute capability is
    // covered by the architectures the plugin was compiled for. Uses the CUDA driver
    // API only, so no CUDA context is created (a context left on the wrong device
    // would poison the later best-card selection, and exclusive-compute devices can
    // refuse a second context).
    virtual bool IsAvailable() = 0;

    // Select the CUDA device. Must be called (with 999 for the automatic best-card
    // pick) BEFORE any content allocation, so the CudaContext singleton creates the
    // device context first - mirroring the static build's initialisation order.
    virtual void SetGpuIdx(unsigned gpuIdx) = 0;

    // Factories mirroring Platform's CUDA branch. Ownership passes to the caller.
    virtual ComputeFactory* CreateComputeFactory() = 0;
    virtual ContentCreatorFactory* CreateContentCreatorFactory() = 0;
    virtual KernelFactory* CreateKernelFactory() = 0;
    virtual MeasureCreatorFactory* CreateMeasureCreatorFactory() = 0;

    // Build and initialise the CUDA optimiser. The CUDA optimiser is single
    // precision only (CudaOptimiser derives from Optimiser<float>), so only the
    // float instantiation is offered; the double path stays on the CPU.
    virtual NiftyReg::Optimiser<float>* CreateOptimiser(F3dContent& con,
                                                        NiftyReg::InterfaceOptimiser& opt,
                                                        size_t maxIterationNumber,
                                                        bool useConjGradient,
                                                        bool optimiseX,
                                                        bool optimiseY,
                                                        bool optimiseZ,
                                                        F3dContent *conBw) = 0;

    // Print the per-device information block (used by reg_gpuinfo)
    virtual void ShowDeviceInfo() = 0;
};

// Exported (C linkage) symbols the plugin shared library provides. nrCudaPluginAbi
// and nrCudaPluginVersion are plain C functions so they are safe to call before the
// ABI has been validated; nrCreateCudaPlugin returns a singleton (never freed).
extern "C" {
int nrCudaPluginAbi();
const char* nrCudaPluginVersion();
CudaPluginInterface* nrCreateCudaPlugin();
}

// Decoration for the definitions of the exported symbols in the plugin. MSVC DLLs
// export nothing by default, so the dllexport is load-bearing on Windows.
#if defined(_WIN32)
#define NR_CUDA_PLUGIN_EXPORT __declspec(dllexport)
#else
#define NR_CUDA_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

// Host-side loader (CudaPluginLoader.cpp, compiled into _reg_platform): locate and
// load the plugin once, validate its ABI and check a usable device is present.
// Returns the cached plugin, or nullptr when CUDA is unavailable for any reason -
// the reason is then available from getCudaPluginLoadError(). Thread-safe.
CudaPluginInterface* loadCudaPlugin();
const std::string& getCudaPluginLoadError();
