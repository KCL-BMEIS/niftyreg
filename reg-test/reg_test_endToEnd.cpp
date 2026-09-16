#include "reg_test_common.h"
#include "_reg_aladin_sym.h"
#include "_reg_f3d2.h"
#include <catch2/matchers/catch_matchers_string.hpp>

/**
 *  End-to-end smoke tests for the registration front-ends
 *
 *  reg_aladin, reg_aladin_sym, reg_f3d and reg_f3d2 are run with (close to) their default
 *  settings on small, well-behaved 2D and 3D images, on every available platform. The tests
 *  only require the pipelines to complete and to hand back finite, correctly sized outputs: they
 *  are not accuracy tests (the per-component oracle tests are), they make sure that a crash, a
 *  hang or an exception anywhere along the pyramid / kernel / optimiser / final-warp chain is
 *  caught by the suite.
 */

namespace {

// Smooth, textured, strictly positive intensities: a few cosine products at different frequencies
// plus a fine ripple, so that no region is flat (block matching needs variance in every block and
// NMI needs a spread of intensities) and nothing is periodic at the block or grid scale
NiftiImage MakeSmoothImage(const vector<NiftiImage::dim_t>& dims) {
    NiftiImage image(dims, NIFTI_TYPE_FLOAT32);
    mat44 identity;
    Mat44Eye(&identity);
    image->sform_code = 1;
    image->sto_xyz = identity;
    image->sto_ijk = identity;
    image->qform_code = 0;
    const int nx = image->nx, ny = image->ny, nz = image->nz;
    auto data = image.data();
    size_t index = 0;
    for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x, ++index) {
                const float depth = nz > 1 ? cosf(0.27f * z + 0.7f) : 1.f;
                data[index] = 100.f
                    + 40.f * cosf(0.31f * x + 0.4f) * cosf(0.23f * y + 1.1f) * depth
                    + 25.f * cosf(0.11f * x - 0.07f * y + 0.19f * z + 2.f)
                    + 10.f * sinf(0.53f * x + 0.61f * y + 0.47f * z);
            }
    return image;
}

// The same anatomy seen one voxel away with a different intensity mapping: a small, smooth
// misalignment every algorithm should cope with, and non-identical intensities so that NMI
// has something to do
NiftiImage MakeFloating(const NiftiImage& reference) {
    NiftiImage floating(reference, NiftiImage::Copy::Image);
    for (int axis = 0; axis < (reference->nz > 1 ? 3 : 2); ++axis)
        floating->sto_xyz.m[axis][3] += 1.f;
    floating->sto_ijk = nifti_mat44_inverse(floating->sto_xyz);
    auto data = floating.data();
    for (size_t i = 0; i < floating.nVoxels(); ++i)
        data[i] = static_cast<float>(data[i]) * 1.1f + 5.f;
    return floating;
}

size_t CountFinite(const NiftiImage& image) {
    size_t count = 0;
    const auto data = image.data();
    for (size_t i = 0; i < image.nVoxels(); ++i)
        if (std::isfinite(static_cast<float>(data[i]))) ++count;
    return count;
}

void RequireSameGrid(const NiftiImage& image, const NiftiImage& reference) {
    REQUIRE(image);
    REQUIRE(image->nx == reference->nx);
    REQUIRE(image->ny == reference->ny);
    REQUIRE(image->nz == reference->nz);
}

// The warped image is padded with NaN where the floating image does not reach, so it is only
// required to be mostly defined; a grid or a matrix has to be finite everywhere
void RequireMostlyFinite(const NiftiImage& image) {
    REQUIRE(CountFinite(image) > image.nVoxels() / 2);
}
void RequireAllFinite(const NiftiImage& image) {
    REQUIRE(image);
    REQUIRE(CountFinite(image) == image.nVoxels());
}

template<class Registration>
void RunAladin(const NiftiImage& reference, const NiftiImage& floating, const PlatformType platformType) {
    // Defaults otherwise: rigid then affine, 3 levels, 5 iterations per level, 50% of the blocks,
    // centre alignment, linear interpolation
    Registration reg;
    reg.SetInputReference(reference);
    reg.SetInputFloating(floating);
    reg.SetPlatformType(platformType);
    reg.SetVerbose(false);
    REQUIRE_NOTHROW(reg.Run());

    const mat44 *matrix = reg.GetTransformationMatrix();
    REQUIRE(matrix != nullptr);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            REQUIRE(std::isfinite(matrix->m[i][j]));

    NiftiImage warped;
    REQUIRE_NOTHROW(warped = reg.GetFinalWarpedImage());
    RequireSameGrid(warped, reference);
    RequireMostlyFinite(warped);
}

template<class Registration>
void RunF3d(const NiftiImage& reference, const NiftiImage& floating, const PlatformType platformType, const bool symmetric) {
    // Defaults otherwise: 5-voxel control-point spacing, NMI, bending-energy penalty, pyramid
    Registration reg(1, 1);
    reg.SetReferenceImage(reference);
    reg.SetFloatingImage(floating);
    reg.SetPlatformType(platformType);
    reg.DoNotPrintOutInformation();
    reg.SetLevelNumber(2);
    reg.SetLevelToPerform(2);
    reg.SetMaximalIterationNumber(5);
    REQUIRE_NOTHROW(reg.Run());

    NiftiImage grid;
    REQUIRE_NOTHROW(grid = reg.GetControlPointPositionImage());
    RequireAllFinite(grid);

    NiftiImage backwardGrid;
    REQUIRE_NOTHROW(backwardGrid = reg.GetBackwardControlPointPositionImage());
    if (symmetric)
        RequireAllFinite(backwardGrid);
    else
        REQUIRE_FALSE(backwardGrid);

    vector<NiftiImage> warped;
    REQUIRE_NOTHROW(warped = reg.GetWarpedImage());
    REQUIRE(warped.size() == (symmetric ? 2u : 1u));
    RequireSameGrid(warped[0], reference);
    RequireMostlyFinite(warped[0]);
    if (symmetric) {
        RequireSameGrid(warped[1], floating);
        RequireMostlyFinite(warped[1]);
    }
}

// A 2D image large enough for the pyramid to downsample it (an axis is halved only from 64
// voxels), and a 3D image small enough to keep the Debug-build runtime low
const vector<vector<NiftiImage::dim_t>> kAllDims{ { 64, 64 }, { 32, 32, 32 } };

std::string Label(const vector<NiftiImage::dim_t>& dims, const PlatformType platformType) {
    return (dims.size() == 3 ? "3D"s : "2D"s) + " - platform " + std::to_string(static_cast<int>(platformType));
}

// The OpenCL backend implements the reg_aladin kernels only; the non-rigid registrations have
// to be run on the other platforms and to refuse it explicitly (see the last test case)
bool SupportsNonRigid(const PlatformType platformType) {
    return platformType != PlatformType::OpenCl;
}

} // namespace

TEST_CASE("reg_aladin runs end to end on well-behaved images", "[reg_aladin][smoke]") {
    for (const auto& dims : kAllDims) {
        const NiftiImage reference = MakeSmoothImage(dims);
        const NiftiImage floating = MakeFloating(reference);
        for (const auto& platformType : PlatformTypes) {
            SECTION(Label(dims, platformType)) {
                RunAladin<reg_aladin<float>>(reference, floating, platformType);
            }
        }
    }
}

TEST_CASE("reg_aladin_sym runs end to end on well-behaved images", "[reg_aladin][smoke]") {
    for (const auto& dims : kAllDims) {
        const NiftiImage reference = MakeSmoothImage(dims);
        const NiftiImage floating = MakeFloating(reference);
        for (const auto& platformType : PlatformTypes) {
            SECTION(Label(dims, platformType)) {
                RunAladin<reg_aladin_sym<float>>(reference, floating, platformType);
            }
        }
    }
}

TEST_CASE("reg_f3d runs end to end on well-behaved images", "[reg_f3d][smoke]") {
    for (const auto& dims : kAllDims) {
        const NiftiImage reference = MakeSmoothImage(dims);
        const NiftiImage floating = MakeFloating(reference);
        for (const auto& platformType : PlatformTypes) {
            if (!SupportsNonRigid(platformType)) continue;
            SECTION(Label(dims, platformType)) {
                RunF3d<reg_f3d<float>>(reference, floating, platformType, false);
            }
        }
    }
}

TEST_CASE("reg_f3d2 runs end to end on well-behaved images", "[reg_f3d][smoke]") {
    for (const auto& dims : kAllDims) {
        const NiftiImage reference = MakeSmoothImage(dims);
        const NiftiImage floating = MakeFloating(reference);
        for (const auto& platformType : PlatformTypes) {
            if (!SupportsNonRigid(platformType)) continue;
            SECTION(Label(dims, platformType)) {
                RunF3d<reg_f3d2<float>>(reference, floating, platformType, true);
            }
        }
    }
}

#ifdef USE_OPENCL
TEST_CASE("reg_f3d refuses the OpenCL platform with a diagnosis", "[reg_f3d][smoke]") {
    // OpenCL provides no similarity measure: asking for it must raise a readable error, not
    // reach into an unset factory
    const NiftiImage reference = MakeSmoothImage({ 32, 32, 32 });
    const NiftiImage floating = MakeFloating(reference);
    reg_f3d<float> reg(1, 1);
    reg.SetReferenceImage(reference);
    reg.SetFloatingImage(floating);
    reg.DoNotPrintOutInformation();
    REQUIRE_THROWS_WITH(reg.SetPlatformType(PlatformType::OpenCl),
                        Catch::Matchers::ContainsSubstring("does not provide similarity measures"));
}
#endif
