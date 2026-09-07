// Host-side loader for the CUDA runtime plugin.
//
// Compiled into _reg_platform only when USE_CUDA_PLUGIN is set. The plugin is
// searched relative to the module that contains this code (the executable when the
// host libraries are static, lib_reg_platform.so when they are shared), then by bare
// name through the caller's RPATH and the system search paths, and finally in the
// configured install tree. Resolving relative to the containing module rather than
// relying on the executable's RPATH keeps the lookup correct for shared-library
// builds and for external programs that link NiftyReg.
//
// The load is attempted at most once per process (magic static, thread-safe) and the
// result is cached. Any failure - plugin missing, exported symbols missing, ABI
// mismatch, or no usable CUDA device - makes loadCudaPlugin() return nullptr with
// the reason available from getCudaPluginLoadError(); callers decide whether that is
// fatal (Platform, on an explicit CUDA request) or informational (reg_gpuinfo). The
// plugin is never unloaded: once any of its code has run it may hold static state,
// and keeping the handle for the lifetime of the process is harmless.

#include "CudaPluginInterface.h"
#include "Debug.hpp"

#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <climits>
#include <cstdlib>
#endif

namespace {
/* *************************************************************** */
#ifdef _WIN32
constexpr const char *pluginFileName = "niftyreg_cuda.dll";
#else
constexpr const char *pluginFileName = "libniftyreg_cuda.so";
#endif
/* *************************************************************** */
std::string& LoadError() {
    static std::string loadError;
    return loadError;
}
/* *************************************************************** */
// Directory of the module (executable or shared library) containing this function
std::string GetThisModuleDir() {
    std::string path;
#ifdef _WIN32
    HMODULE module = nullptr;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCSTR>(&loadCudaPlugin), &module))
        return "";
    char buffer[MAX_PATH];
    const DWORD length = GetModuleFileNameA(module, buffer, MAX_PATH);
    if (length == 0 || length >= MAX_PATH) return "";
    path.assign(buffer, length);
#else
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&loadCudaPlugin), &info) || !info.dli_fname)
        return "";
    // dli_fname can be relative (e.g. ./reg_f3d); make it absolute while the
    // working directory is still the one the process started in
    char resolved[PATH_MAX];
    path = realpath(info.dli_fname, resolved) ? resolved : info.dli_fname;
#endif
    const size_t separator = path.find_last_of("/\\");
    return separator == std::string::npos ? "" : path.substr(0, separator);
}
/* *************************************************************** */
// Try to open one candidate; appends the failure reason to error when unsuccessful
void* OpenLibrary(const std::string& path, std::string& error) {
#ifdef _WIN32
    // Fail quietly when the plugin or one of its dependent DLLs (e.g.
    // cudart64_*.dll) is missing - without this, Windows pops a blocking system
    // error dialog, which would hang headless runs
    DWORD previousErrorMode = 0;
    SetThreadErrorMode(SEM_FAILCRITICALERRORS, &previousErrorMode);
    HMODULE handle = LoadLibraryA(path.c_str());
    SetThreadErrorMode(previousErrorMode, nullptr);
    if (!handle)
        error += "\n  " + path + ": error " + std::to_string(GetLastError());
    return handle;
#else
    // RTLD_NOW so a missing or mismatched dependency of the plugin (its CUDA
    // runtime libraries) is detected here, cleanly, rather than surfacing mid-run
    void *handle = dlopen(path.c_str(), RTLD_NOW);
    if (!handle)
        error += std::string("\n  ") + dlerror();
    return handle;
#endif
}
/* *************************************************************** */
void* GetSymbol(void *handle, const char *name) {
#ifdef _WIN32
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name));
#else
    return dlsym(handle, name);
#endif
}
/* *************************************************************** */
CudaPluginInterface* LoadPlugin() {
    std::string& error = LoadError();

    // Candidates in order: next to the containing module; the lib/ sibling of a
    // module in bin/ (installed executables, plugin installed to lib/); the bare
    // name (build-tree RPATH, LD_LIBRARY_PATH/PATH, system paths); and the
    // configured install tree as a last resort (external programs linking static
    // NiftyReg, whose own executable lives outside the install prefix)
    std::vector<std::string> candidates;
    const std::string moduleDir = GetThisModuleDir();
    if (!moduleDir.empty()) {
        candidates.push_back(moduleDir + "/" + pluginFileName);
        candidates.push_back(moduleDir + "/../lib/" + pluginFileName);
    }
    candidates.push_back(pluginFileName);
#ifdef NR_INSTALL_LIBDIR
    candidates.push_back(std::string(NR_INSTALL_LIBDIR) + "/" + pluginFileName);
#endif

    error = std::string("the CUDA plugin (") + pluginFileName + ") could not be loaded:";
    void *handle = nullptr;
    for (const std::string& candidate : candidates)
        if ((handle = OpenLibrary(candidate, error)))
            break;
    if (!handle) return nullptr;

    // Validate the ABI before making any virtual call: a plugin from a different
    // build may lay out the vtable differently, so nothing on CudaPluginInterface is
    // safe to invoke until the version has been checked through plain C symbols
    const auto getAbi = reinterpret_cast<int(*)()>(GetSymbol(handle, "nrCudaPluginAbi"));
    const auto getVersion = reinterpret_cast<const char*(*)()>(GetSymbol(handle, "nrCudaPluginVersion"));
    const auto createPlugin = reinterpret_cast<CudaPluginInterface*(*)()>(GetSymbol(handle, "nrCreateCudaPlugin"));
    if (!getAbi || !getVersion || !createPlugin) {
        error = "the loaded library is not a NiftyReg CUDA plugin (exported symbols missing)";
        return nullptr;
    }
    if (getAbi() != NR_CUDA_PLUGIN_ABI) {
        error = "the CUDA plugin has an incompatible ABI (plugin " + std::to_string(getAbi()) +
                ", expected " + std::to_string(NR_CUDA_PLUGIN_ABI) + "; plugin version " + getVersion() +
                ", host version " NR_VERSION ") - it likely belongs to a different NiftyReg build";
        return nullptr;
    }
    if (std::string(getVersion()) != NR_VERSION)
        NR_WARN("The CUDA plugin was built from NiftyReg " << getVersion() <<
                " but the host is " << NR_VERSION << "; the plugin ABI matches, continuing");

    CudaPluginInterface *plugin = createPlugin();
    if (!plugin) {
        error = "the CUDA plugin failed to initialise";
        return nullptr;
    }
    if (!plugin->IsAvailable()) {
        error = "the CUDA plugin loaded but no usable CUDA device is available (no device present, "
                "or no device's compute capability is covered by this build)";
        return nullptr;
    }

    error.clear();
    return plugin;
}
/* *************************************************************** */
} // namespace
/* *************************************************************** */
CudaPluginInterface* loadCudaPlugin() {
    // Magic static: the load is attempted once, thread-safely; the result (possibly
    // nullptr) and the error message are immutable afterwards
    static CudaPluginInterface *const plugin = LoadPlugin();
    return plugin;
}
/* *************************************************************** */
const std::string& getCudaPluginLoadError() {
    loadCudaPlugin();
    return LoadError();
}
/* *************************************************************** */
