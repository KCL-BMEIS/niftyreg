#include "reg_test_common.h"
#include "_reg_aladin_sym.h"
#include "_reg_f3d2.h"
#include <catch2/matchers/catch_matchers_string.hpp>
#include <functional>

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

namespace {

// Raw buffers on purpose below: NiftiImage::data() applies scl_slope/scl_inter on read and inverts it
// on write, which would hide (or double) the scaling these tests are about

// The same intensities stored as int16 with a slope and an intercept (real = stored * slope + inter)
NiftiImage ToScaledInt16(const NiftiImage& image, const float slope, const float inter) {
    NiftiImage scaled(image, NiftiImage::Copy::ImageInfo);
    scaled->datatype = NIFTI_TYPE_INT16;
    scaled->nbyper = sizeof(short);
    scaled.realloc();
    scaled->scl_slope = slope;
    scaled->scl_inter = inter;
    const float *in = static_cast<const float*>(image->data);
    short *out = static_cast<short*>(scaled->data);
    for (size_t i = 0; i < image.nVoxels(); ++i)
        out[i] = static_cast<short>(std::lround((in[i] - inter) / slope));
    return scaled;
}

// The real values of a scaled int16 image, as a float image without scaling
NiftiImage ToRealFloat(const NiftiImage& image) {
    NiftiImage real(image, NiftiImage::Copy::ImageInfo);
    real->datatype = NIFTI_TYPE_FLOAT32;
    real->nbyper = sizeof(float);
    real.realloc();
    real->scl_slope = 1;
    real->scl_inter = 0;
    const short *in = static_cast<const short*>(image->data);
    float *out = static_cast<float*>(real->data);
    for (size_t i = 0; i < image.nVoxels(); ++i)
        out[i] = in[i] * image->scl_slope + image->scl_inter;
    return real;
}

// Median rather than mean: the padding of an integer warped image is stored as 0, i.e. the intercept
double MedianRealValue(const NiftiImage& image) {
    REQUIRE(image->datatype == NIFTI_TYPE_INT16);
    const float slope = image->scl_slope == 0 ? 1.f : image->scl_slope;
    const short *data = static_cast<const short*>(image->data);
    vector<double> values(data, data + image.nVoxels());
    for (auto& value : values) value = value * slope + image->scl_inter;
    std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
    return values[values.size() / 2];
}

NiftiImage RunF3dOnCpu(const NiftiImage& reference, const NiftiImage& floating,
                       const std::function<void(reg_f3d<float>&)>& configure = nullptr) {
    reg_f3d<float> reg(reference->nt, floating->nt);
    reg.SetReferenceImage(reference);
    reg.SetFloatingImage(floating);
    reg.SetPlatformType(PlatformType::Cpu);
    reg.DoNotPrintOutInformation();
    reg.SetLevelNumber(2);
    reg.SetLevelToPerform(2);
    reg.SetMaximalIterationNumber(5);
    if (configure) configure(reg);
    reg.Run();
    return reg.GetControlPointPositionImage();
}

bool SameValues(const NiftiImage& a, const NiftiImage& b) {
    for (size_t i = 0; i < a.nVoxels(); ++i)
        if (static_cast<float>(a.data()[i]) != static_cast<float>(b.data()[i])) return false;
    return true;
}

} // namespace

TEST_CASE("reg_f3d and reg_f3d2 write each warped image with the scaling of the image it resamples", "[reg_f3d][intensity]") {
    // The warped image holds the stored values of the image it resamples, but is allocated from the
    // other image's header: a CT floating (int16, intercept -1024) used to come out 1024 HU off
    const NiftiImage reference = ToScaledInt16(MakeSmoothImage({ 32, 32, 32 }), 2.f, 10.f);
    const NiftiImage floating = ToScaledInt16(MakeFloating(MakeSmoothImage({ 32, 32, 32 })), 0.5f, -1024.f);
    for (const bool symmetric : { false, true }) {
        INFO((symmetric ? "reg_f3d2" : "reg_f3d"));
        std::unique_ptr<reg_f3d<float>> reg(symmetric ? new reg_f3d2<float>(1, 1) : new reg_f3d<float>(1, 1));
        reg->SetReferenceImage(reference);
        reg->SetFloatingImage(floating);
        reg->SetPlatformType(PlatformType::Cpu);
        reg->DoNotPrintOutInformation();
        reg->SetLevelNumber(2);
        reg->SetLevelToPerform(2);
        reg->SetMaximalIterationNumber(5);
        REQUIRE_NOTHROW(reg->Run());
        const vector<NiftiImage> warped = reg->GetWarpedImage();
        REQUIRE(warped[0]->scl_slope == floating->scl_slope);
        REQUIRE(warped[0]->scl_inter == floating->scl_inter);
        // Misalignment and padding move the median by a few units; the wrong scaling moved it by ~1000
        REQUIRE(std::abs(MedianRealValue(warped[0]) - MedianRealValue(floating)) < 50);
        if (symmetric) {
            REQUIRE(warped[1]->scl_slope == reference->scl_slope);
            REQUIRE(warped[1]->scl_inter == reference->scl_inter);
            REQUIRE(std::abs(MedianRealValue(warped[1]) - MedianRealValue(reference)) < 50);
        }
    }
}

TEST_CASE("reg_f3d thresholds each image with its own thresholds", "[reg_f3d][intensity]") {
    const NiftiImage reference = MakeSmoothImage({ 32, 32, 32 });
    const NiftiImage floating = MakeFloating(reference);
    const NiftiImage unthresholded = RunF3dOnCpu(reference, floating);

    SECTION("A floating threshold changes the registration") {
        // The floating pyramid used to be clamped with the reference thresholds, ignoring --fUpTh/--fLwTh
        const NiftiImage thresholded = RunF3dOnCpu(reference, floating, [](reg_f3d<float>& reg) { reg.SetFloatingThresholdUp(0, 110.f); });
        REQUIRE_FALSE(SameValues(thresholded, unthresholded));
    }
    SECTION("A reference threshold does not reach the floating image") {
        // Just above the reference maximum the threshold leaves the reference untouched; the floating
        // image (1.1 * reference + 5) goes higher and used to be clamped by it
        float referenceMax = std::numeric_limits<float>::lowest();
        for (size_t i = 0; i < reference.nVoxels(); ++i)
            referenceMax = std::max(referenceMax, static_cast<float>(reference.data()[i]));
        const NiftiImage thresholded = RunF3dOnCpu(reference, floating, [&](reg_f3d<float>& reg) { reg.SetReferenceThresholdUp(0, referenceMax + 1); });
        REQUIRE(SameValues(thresholded, unthresholded));
    }
    SECTION("The robust range is taken in real units") {
        // The 2-98% range used to be read from the stored values and applied to the real ones
        const NiftiImage scaledReference = ToScaledInt16(reference, 0.5f, -1024.f);
        const NiftiImage scaledFloating = ToScaledInt16(floating, 0.5f, -1024.f);
        auto robust = [](reg_f3d<float>& reg) { reg.UseRobustRange(); };
        REQUIRE(SameValues(RunF3dOnCpu(scaledReference, scaledFloating, robust),
                           RunF3dOnCpu(ToRealFloat(scaledReference), ToRealFloat(scaledFloating), robust)));
    }
    SECTION("The robust range ignores NaN voxels") {
        NiftiImage maskedFloating(floating, NiftiImage::Copy::Image);
        auto data = maskedFloating.data();
        for (size_t i = 0; i < maskedFloating.nVoxels() / 4; ++i)
            data[i] = std::numeric_limits<float>::quiet_NaN();
        const NiftiImage grid = RunF3dOnCpu(reference, maskedFloating, [](reg_f3d<float>& reg) { reg.UseRobustRange(); });
        RequireAllFinite(grid);
    }
}

TEST_CASE("reg_aladin's intensity centre of mass skips NaN voxels", "[reg_aladin][intensity]") {
    // One NaN voxel used to turn the centre of mass, hence the initial transformation, into NaN, and the
    // registration then aborted for lack of block correspondences
    const NiftiImage reference = MakeSmoothImage({ 32, 32, 32 });
    NiftiImage floating = MakeFloating(reference);
    static_cast<float*>(floating->data)[0] = std::numeric_limits<float>::quiet_NaN();
    reg_aladin<float> reg;
    reg.SetInputReference(reference);
    reg.SetInputFloating(floating);
    reg.SetPlatformType(PlatformType::Cpu);
    reg.SetAlignCentreMass(2);
    reg.SetVerbose(false);
    REQUIRE_NOTHROW(reg.Run());
    const mat44 *matrix = reg.GetTransformationMatrix();
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            REQUIRE(std::isfinite(matrix->m[i][j]));
}
