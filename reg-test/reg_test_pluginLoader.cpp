// Tests the CUDA runtime-plugin loader (USE_CUDA_PLUGIN builds only).
//
// The loader's outcome depends on the machine (plugin present or not, CUDA device
// present or not), so every check below asserts properties that must hold in BOTH
// outcomes: the result is cached and consistent, the error message exists exactly
// when loading failed, and Platform's behaviour matches the loader's verdict - a
// CUDA Platform on success, a hard error (never a silent CPU fallback) on failure.

#include "CudaPluginInterface.h"
#include "Platform.h"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Plugin loader", "[unit][plugin]") {
    CudaPluginInterface *plugin = loadCudaPlugin();

    SECTION("The result is cached: repeated calls return the same pointer") {
        REQUIRE(loadCudaPlugin() == plugin);
        REQUIRE(loadCudaPlugin() == plugin);
    }

    SECTION("The error message exists exactly when loading failed") {
        if (plugin)
            REQUIRE(getCudaPluginLoadError().empty());
        else
            REQUIRE_FALSE(getCudaPluginLoadError().empty());
    }

    SECTION("A loaded plugin reports an available device") {
        // loadCudaPlugin() returns nullptr unless IsAvailable() held, and the
        // answer must be stable across calls
        if (plugin) {
            REQUIRE(plugin->IsAvailable());
            REQUIRE(plugin->IsAvailable());
        }
    }

    SECTION("Platform honours the loader's verdict") {
        if (plugin) {
            Platform platform(PlatformType::Cuda);
            REQUIRE(platform.GetName() == "CUDA");
            REQUIRE(platform.GetPlatformType() == PlatformType::Cuda);
        } else {
            // An explicit CUDA request that cannot be honoured is a hard error
            REQUIRE_THROWS_AS(Platform(PlatformType::Cuda), std::runtime_error);
        }
    }

    SECTION("The CPU platform is unaffected by the plugin's availability") {
        Platform platform(PlatformType::Cpu);
        REQUIRE(platform.GetName() == "CPU");
    }
}
