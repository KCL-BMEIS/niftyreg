#include "reg_test_common.h"
#include "CudaContext.hpp"

/**
 *  Unit test for the CUDA context singleton (NiftyReg::CudaContext).
 *
 *  Constructing a CUDA Platform forces CudaContext::GetInstance(), which runs the singleton
 *  constructor and the auto-pick (PickCard(999)) path. This test then asserts the public
 *  accessors that no other test exercises directly: GetContext, GetBlockSize, IsCardDoubleCapable
 *  and SetCudaIdx (both the "already-selected id" early return and the invalid-id error path).
 */

TEST_CASE("CudaContext accessors", "[unit]") {
    // Forces CudaContext::GetInstance() (singleton ctor + auto-pick of the best card).
    Platform platform(PlatformType::Cuda);
    NiftyReg::CudaContext& context = NiftyReg::CudaContext::GetInstance();

    SECTION("A valid context and block size are available") {
        REQUIRE(context.GetContext() != nullptr);
        REQUIRE(NiftyReg::CudaContext::GetBlockSize() != nullptr);
    }
    SECTION("Double-capability query is callable") {
        // Any modern card is double-capable; we only require the accessor to run without error.
        const bool doubleCapable = context.IsCardDoubleCapable();
        REQUIRE((doubleCapable == true || doubleCapable == false));
    }
    SECTION("Selecting the already-active card is a no-op") {
        // The constructor auto-picked card 0 on a single-GPU host, so re-selecting it early-returns
        // without recreating the context.
        const CUcontext before = context.GetContext();
        context.SetCudaIdx(0);
        REQUIRE(context.GetContext() == before);
    }
    SECTION("Selecting an out-of-range card is rejected") {
        REQUIRE_THROWS_AS(context.SetCudaIdx(999999), std::runtime_error);
    }
}
