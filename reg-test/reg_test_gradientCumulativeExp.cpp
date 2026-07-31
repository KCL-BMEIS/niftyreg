// OpenCL is not supported for this test
#undef USE_OPENCL

#include "reg_test_common.h"
#include "_reg_tools.h"

/*
    Gradient accumulation through exponentiation on the CPU, checked against closed-form expectations
    (reg_f3d2::ExponentiateGradient, implemented by Compute::ExponentiateGradient). reg_f3d -vel runs
    this on every objective gradient.

    The operation integrates the voxel-based measure gradient along the path of the *backward*
    transformation: it builds the squaringNumber+1 intermediate deformation fields of the backward
    velocity grid, resamples the gradient through each of them accumulating as it goes, and divides the
    total by 2^squaringNumber. The number of squaring steps comes from the backward control point
    grid's intent_p2, which reg_f3d2 sets to 6.

    Cross-backend comparison is covered separately; what these cases add is an expectation the
    operation has to meet regardless of how it is implemented, so that a property both backends got
    wrong would still be caught.

      1. A zero backward velocity field makes every intermediate field the identity, so each of the
         squaringNumber accumulation steps doubles the gradient and the final division by
         2^squaringNumber returns it. This pins the accumulation count against the normalisation:
         getting either wrong shifts the result by a factor of two.
         It is checked to a tolerance rather than exactly. The doubling and the halving are both powers
         of two, so they are exact, but the intermediate fields are not exactly the identity: they come
         from a cubic B-spline evaluation of the control point grid, which reproduces the identity only
         to rounding, and the resampling carries that through. The residual is largest with few
         squaring steps, since the flow field is divided by 2^squaringNumber and a larger count starts
         closer to the identity.
      2. Linearity in the gradient. The operation is a sum of resamplings of its input, so scaling the
         input scales the output. A power of two keeps the scaling exact, so this stays an equality.
      3. Zero squaring steps is the identity operation. Pinned because it is the default, and because
         it silently disables the integration the -vel gradient depends on.
*/

namespace {

// A float32 image with identity sform, so the deformation fields and the gradient share a trivial
// geometry and an identity field resamples exactly.
NiftiImage MakeReference(bool is3D) {
    const NiftiImage::dim_t size = 8;
    std::vector<NiftiImage::dim_t> dims(is3D ? 3 : 2, size);
    NiftiImage img(dims, NIFTI_TYPE_FLOAT32);
    mat44 eye;
    Mat44Eye(&eye);
    img->sform_code = 1;
    img->sto_xyz = eye;
    img->sto_ijk = eye;
    img->qform_code = 0;
    return img;
}

// A deterministic, non-trivial gradient.
void FillGradient(NiftiImage& gradient, float scale) {
    auto ptr = gradient.data();
    for (size_t i = 0; i < gradient.nVoxels(); ++i)
        ptr[i] = scale * static_cast<float>(std::sin(0.37 * static_cast<double>(i)) + 0.5);
}

// Run Compute::ExponentiateGradient with a zero backward velocity field of the given squaring count,
// and return the resulting voxel-based measure gradient.
NiftiImage ExponentiateWithZeroVelocity(const NiftiImage& reference, int squaringSteps, float gradientScale) {
    NiftiImage referenceFw(reference), referenceBw(reference);
    NiftiImage controlPointGrid = CreateControlPointGrid(reference);

    // A zero *displacement* velocity field, i.e. the control point positions themselves
    NiftiImage controlPointGridBw = CreateControlPointGrid(reference);
    controlPointGridBw->intent_p1 = SPLINE_VEL_GRID;
    controlPointGridBw->intent_p2 = static_cast<float>(squaringSteps);

    unique_ptr<F3dContent> content{ new F3dContent(referenceFw, referenceFw, controlPointGrid) };
    unique_ptr<F3dContent> contentBw{ new F3dContent(referenceBw, referenceBw, controlPointGridBw) };
    content->SetDeformationField(CreateDeformationField(reference));

    FillGradient(content->GetVoxelBasedMeasureGradient(), gradientScale);
    content->UpdateVoxelBasedMeasureGradient();

    Platform platform(PlatformType::Cpu);
    unique_ptr<Compute> compute{ platform.CreateCompute(*content) };
    compute->ExponentiateGradient(*contentBw);

    return NiftiImage(content->GetVoxelBasedMeasureGradient(), NiftiImage::Copy::Image);
}

NiftiImage ReferenceGradient(const NiftiImage& reference, float gradientScale) {
    NiftiImage gradient = CreateDeformationField(reference);
    FillGradient(gradient, gradientScale);
    return gradient;
}

void RequireSameData(const NiftiImage& actual, const NiftiImage& expected) {
    const auto actualPtr = actual.data();
    const auto expectedPtr = expected.data();
    REQUIRE(actual.nVoxels() == expected.nVoxels());
    for (size_t i = 0; i < actual.nVoxels(); ++i) {
        INFO("voxel " << i);
        REQUIRE(static_cast<float>(actualPtr[i]) == static_cast<float>(expectedPtr[i]));
    }
}

void RequireCloseData(const NiftiImage& actual, const NiftiImage& expected, double tolerance) {
    const auto actualPtr = actual.data();
    const auto expectedPtr = expected.data();
    REQUIRE(actual.nVoxels() == expected.nVoxels());
    for (size_t i = 0; i < actual.nVoxels(); ++i) {
        const double difference = std::abs(static_cast<double>(actualPtr[i]) - static_cast<double>(expectedPtr[i]));
        INFO("voxel " << i << ": " << static_cast<float>(actualPtr[i]) << " vs "
             << static_cast<float>(expectedPtr[i]) << ", diff " << std::scientific << difference);
        REQUIRE(difference < tolerance);
    }
}

} // namespace

TEST_CASE("Gradient cumulative exponentiation leaves the gradient unchanged for a zero velocity field", "[unit]") {
    for (const bool is3D : { false, true }) {
        for (const int squaringSteps : { 1, 3, 6 }) {
            SECTION(std::string(is3D ? "3D" : "2D") + ", " + std::to_string(squaringSteps) + " squaring steps") {
                const NiftiImage reference = MakeReference(is3D);
                const NiftiImage expected = ReferenceGradient(reference, 1.f);
                const NiftiImage actual = ExponentiateWithZeroVelocity(reference, squaringSteps, 1.f);
                // See the note on property 1: bounded by the B-spline reproduction of the identity
                // grid, not exact. Far below the factor of two a miscount would produce.
                RequireCloseData(actual, expected, 1e-5);
            }
        }
    }
}

TEST_CASE("Gradient cumulative exponentiation is linear in the gradient", "[unit]") {
    for (const bool is3D : { false, true }) {
        SECTION(is3D ? "3D" : "2D") {
            const NiftiImage reference = MakeReference(is3D);
            // A power of two keeps the scaling exact, so this stays an equality check
            constexpr float scale = 4.f;
            const NiftiImage single = ExponentiateWithZeroVelocity(reference, 6, 1.f);
            const NiftiImage scaled = ExponentiateWithZeroVelocity(reference, 6, scale);

            NiftiImage expected(single, NiftiImage::Copy::Image);
            reg_tools_multiplyValueToImage(single, expected, scale);
            RequireSameData(scaled, expected);
        }
    }
}

TEST_CASE("Gradient cumulative exponentiation is a no-op without squaring steps", "[unit]") {
    // intent_p2 == 0 means zero accumulation steps and a division by 2^0. Pinned down because it is
    // the default, and because it silently disables the integration the -vel gradient relies on.
    for (const bool is3D : { false, true }) {
        SECTION(is3D ? "3D" : "2D") {
            const NiftiImage reference = MakeReference(is3D);
            const NiftiImage expected = ReferenceGradient(reference, 1.f);
            const NiftiImage actual = ExponentiateWithZeroVelocity(reference, 0, 1.f);
            RequireSameData(actual, expected);
        }
    }
}
