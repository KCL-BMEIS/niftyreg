#include "reg_test_common.h"
#include "_reg_aladin_sym.h"
#include "_reg_f3d.h"
#include "_reg_localTrans_jac.h"
#include "_reg_resampling.h"
#include <catch2/matchers/catch_matchers_string.hpp>

/**
 *  Input validation of the registration front-ends (reg_aladin, reg_aladin_sym, reg_f3d) and of
 *  the library entry points that pair a 2D or 3D kernel with buffers they did not allocate
 *
 *  The pipelines are dimensioned from the reference image, so the front-ends have to reject what
 *  they cannot handle before any buffer is allocated:
 *    - a 2D image paired with a 3D one: the warped image would be allocated with the wrong voxel
 *      count and a 2D deformation field cannot sample a 3D floating image,
 *    - an axis thinner than a block (BLOCK_WIDTH voxels), which cannot hold a single block.
 *  When an axis is at least a block wide but only holds a single row of blocks, every block corner
 *  shares the same coordinate along it: the correspondences are coplanar, the affine least-squares
 *  fit is singular, and the estimate has to be rejected rather than inverted or fed to the matrix
 *  logarithm of the symmetric update. A rigid transformation is still well defined from coplanar
 *  points and has to go through.
 *  A positive control on the smallest admissible images checks that the validation does not
 *  reject valid inputs and that a pure translation is recovered.
 *
 *  Below the front-ends, each resampling, composition, spline evaluation, inversion and Jacobian
 *  entry point selects its 2D or 3D kernel from one argument and indexes the others as if they
 *  agreed, so a caller assembling its own buffers can make a kernel run past the smaller one.
 *  The second half of this file gives each entry point a mismatched pair and a matched control.
 */

using Catch::Matchers::ContainsSubstring;

namespace {

// Random float32 image with an identity sform
NiftiImage MakeImage(std::mt19937& gen, const vector<NiftiImage::dim_t>& dims) {
    NiftiImage image(dims, NIFTI_TYPE_FLOAT32);
    mat44 identity;
    Mat44Eye(&identity);
    image->sform_code = 1;
    image->sto_xyz = identity;
    image->sto_ijk = identity;
    image->qform_code = 0;
    std::uniform_real_distribution<float> distr(0, 1);
    auto data = image.data();
    for (auto itr = data.begin(); itr != data.end(); ++itr)
        *itr = distr(gen);
    return image;
}

// The same voxel data with the origin moved: registering it onto the original has to recover the
// translation from the block matching alone (the centre alignment is disabled below)
NiftiImage Translate(const NiftiImage& image, const float translation[3]) {
    NiftiImage translated(image, NiftiImage::Copy::Image);
    for (int i = 0; i < 3; ++i)
        translated->sto_xyz.m[i][3] += translation[i];
    translated->sto_ijk = nifti_mat44_inverse(translated->sto_xyz);
    return translated;
}

mat44 TranslationMatrix(const float translation[3]) {
    mat44 matrix;
    Mat44Eye(&matrix);
    for (int i = 0; i < 3; ++i)
        matrix.m[i][3] = translation[i];
    return matrix;
}

template<class Registration>
mat44 Register(const NiftiImage& reference, const NiftiImage& floating, const PlatformType platformType, const bool affine) {
    Registration reg;
    reg.SetInputReference(reference);
    reg.SetInputFloating(floating);
    reg.SetPlatformType(platformType);
    reg.SetVerbose(false);
    reg.SetNumberOfLevels(1);
    reg.SetLevelsToPerform(1);
    reg.SetMaxIterations(5);
    reg.SetAlignCentre(false);
    reg.SetPerformAffine(affine);
    reg.Run();
    return *reg.GetTransformationMatrix();
}

void RequireMatrixNear(const mat44& actual, const mat44& expected, const float tolerance) {
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            INFO("element [" << i << "][" << j << "]");
            REQUIRE(std::abs(actual.m[i][j] - expected.m[i][j]) < tolerance);
        }
}

constexpr NiftiImage::dim_t kSize = 16;   // enough blocks for the affine correspondence minimum

} // namespace

TEST_CASE("reg_aladin rejects a 2D image paired with a 3D one", "[reg_aladin][validation]") {
    std::mt19937 gen(0);
    const NiftiImage image2d = MakeImage(gen, { kSize, kSize });
    const NiftiImage image3d = MakeImage(gen, { kSize, kSize, kSize });
    const auto matcher = ContainsSubstring("must both be 2D or both be 3D");

    SECTION("2D reference, 3D floating") {
        REQUIRE_THROWS_WITH(Register<reg_aladin<float>>(image2d, image3d, PlatformType::Cpu, true), matcher);
        REQUIRE_THROWS_WITH(Register<reg_aladin_sym<float>>(image2d, image3d, PlatformType::Cpu, true), matcher);
    }
    SECTION("3D reference, 2D floating") {
        REQUIRE_THROWS_WITH(Register<reg_aladin<float>>(image3d, image2d, PlatformType::Cpu, true), matcher);
        REQUIRE_THROWS_WITH(Register<reg_aladin_sym<float>>(image3d, image2d, PlatformType::Cpu, true), matcher);
    }
}

TEST_CASE("reg_aladin rejects an axis thinner than a block", "[reg_aladin][validation]") {
    std::mt19937 gen(0);
    constexpr NiftiImage::dim_t thin = BLOCK_WIDTH - 1;
    const auto matcher = ContainsSubstring("block matching requires at least " + std::to_string(BLOCK_WIDTH));

    // Every axis of a 2D and of a 3D image, thin on either side of the registration
    const vector<vector<NiftiImage::dim_t>> thinDims{
        { thin, kSize, kSize }, { kSize, thin, kSize }, { kSize, kSize, thin },
        { thin, kSize }, { kSize, thin }
    };
    for (const auto& dims : thinDims) {
        const vector<NiftiImage::dim_t> validDims(dims.size(), kSize);
        const NiftiImage thinImage = MakeImage(gen, dims);
        const NiftiImage validImage = MakeImage(gen, validDims);
        std::string label;
        for (const auto& dim : dims) label += std::to_string(dim) + " ";
        SECTION("Thin image " + label) {
            REQUIRE_THROWS_WITH(Register<reg_aladin<float>>(thinImage, validImage, PlatformType::Cpu, true), matcher);
            REQUIRE_THROWS_WITH(Register<reg_aladin<float>>(validImage, thinImage, PlatformType::Cpu, true), matcher);
            REQUIRE_THROWS_WITH(Register<reg_aladin_sym<float>>(thinImage, validImage, PlatformType::Cpu, true), matcher);
        }
    }

    SECTION("An axis exactly one block wide is accepted") {
        // In-plane translation only: the single row of blocks along z cannot observe a z shift
        const float translation[3] = { 1, 1, 0 };
        const NiftiImage reference = MakeImage(gen, { kSize, kSize, BLOCK_WIDTH });
        const NiftiImage floating = Translate(reference, translation);
        REQUIRE_NOTHROW(Register<reg_aladin<float>>(reference, floating, PlatformType::Cpu, false));
    }
}

TEST_CASE("reg_aladin rejects a singular affine estimate from a single row of blocks", "[reg_aladin][validation]") {
    // Fewer than BLOCK_WIDTH + BLOCK_WIDTH / 2 + 1 slices: the second row of blocks along z is less
    // than half filled and unused, so every reference block corner lies in the plane z = 0
    constexpr NiftiImage::dim_t singleRow = BLOCK_WIDTH + BLOCK_WIDTH / 2;
    constexpr NiftiImage::dim_t size = 32;   // enough active blocks for the affine correspondence minimum
    const float translation[3] = { 1, 1, 0 };
    std::mt19937 gen(0);
    const NiftiImage reference = MakeImage(gen, { size, size, singleRow });
    const NiftiImage floating = Translate(reference, translation);
    const mat44 expected = TranslationMatrix(translation);

    for (const auto& platformType : PlatformTypes) {
        SECTION("Platform " + std::to_string(static_cast<int>(platformType))) {
            const auto matcher = ContainsSubstring("transformation is singular");
            REQUIRE_THROWS_WITH(Register<reg_aladin<float>>(reference, floating, platformType, true), matcher);
            REQUIRE_THROWS_WITH(Register<reg_aladin_sym<float>>(reference, floating, platformType, true), matcher);

            // Coplanar correspondences still determine a rigid transformation
            mat44 rigid;
            REQUIRE_NOTHROW(rigid = Register<reg_aladin<float>>(reference, floating, platformType, false));
            RequireMatrixNear(rigid, expected, 0.05f);
            REQUIRE_NOTHROW(rigid = Register<reg_aladin_sym<float>>(reference, floating, platformType, false));
            RequireMatrixNear(rigid, expected, 0.05f);
        }
    }
}

TEST_CASE("reg_aladin recovers a translation on the smallest admissible images", "[reg_aladin][validation]") {
    std::mt19937 gen(0);
    const vector<vector<NiftiImage::dim_t>> allDims{ { kSize, kSize }, { kSize, kSize, kSize } };

    for (const auto& dims : allDims) {
        const bool is3d = dims.size() == 3;
        const float translation[3] = { 1, 1, is3d ? 1.f : 0.f };
        const NiftiImage reference = MakeImage(gen, dims);
        const NiftiImage floating = Translate(reference, translation);
        const mat44 expected = TranslationMatrix(translation);

        for (const auto& platformType : PlatformTypes) {
            SECTION((is3d ? "3D" : "2D") + " - platform "s + std::to_string(static_cast<int>(platformType))) {
                mat44 affine;
                REQUIRE_NOTHROW(affine = Register<reg_aladin<float>>(reference, floating, platformType, true));
                RequireMatrixNear(affine, expected, 0.05f);
                REQUIRE_NOTHROW(affine = Register<reg_aladin_sym<float>>(reference, floating, platformType, true));
                RequireMatrixNear(affine, expected, 0.05f);
            }
        }
    }
}

TEST_CASE("reg_f3d rejects a 2D image paired with a 3D one", "[reg_f3d][validation]") {
    std::mt19937 gen(0);
    const NiftiImage image2d = MakeImage(gen, { kSize, kSize });
    const NiftiImage image3d = MakeImage(gen, { kSize, kSize, kSize });
    const auto matcher = ContainsSubstring("must both be 2D or both be 3D");

    for (const bool reference3d : { false, true }) {
        SECTION(reference3d ? "3D reference, 2D floating" : "2D reference, 3D floating") {
            reg_f3d<float> reg(1, 1);
            reg.SetReferenceImage(reference3d ? image3d : image2d);
            reg.SetFloatingImage(reference3d ? image2d : image3d);
            reg.DoNotPrintOutInformation();
            REQUIRE_THROWS_WITH(reg.Run(), matcher);
        }
    }
}

/* *************************************************************** */
// Library entry points: each selects a 2D or 3D kernel from one argument and indexes the others
/* *************************************************************** */

namespace {

constexpr NiftiImage::dim_t kSmall = 8;   // large enough for a cubic spline grid, small enough for the Debug build

// A deformation field on the image's grid labelled as a flow field
NiftiImage MakeFlowField(const NiftiImage& reference) {
    NiftiImage flow = CreateDeformationField(reference);
    flow->intent_p1 = DEF_VEL_FIELD;
    flow->intent_p2 = 6;
    return flow;
}

// A control point grid re-labelled as a stationary velocity grid
NiftiImage MakeVelocityGrid(const NiftiImage& reference) {
    NiftiImage grid = CreateControlPointGrid(reference, kProductionGridSpacing);
    grid->intent_p1 = SPLINE_VEL_GRID;
    grid->intent_p2 = 6;
    return grid;
}

const auto kDimensionalityMatcher = ContainsSubstring("must both be 2D or both be 3D");
const auto kGridMatcher = ContainsSubstring("same grid");

} // namespace

TEST_CASE("The dimensionality helpers tell 2D inputs from 3D ones", "[validation]") {
    std::mt19937 gen(0);
    const NiftiImage image2d = MakeImage(gen, { kSmall, kSmall });
    const NiftiImage image3d = MakeImage(gen, { kSmall, kSmall, kSmall });
    const NiftiImage grid2d = CreateControlPointGrid(image2d, kProductionGridSpacing);
    const NiftiImage field3d = CreateDeformationField(image3d);

    REQUIRE(reg_haveSameDimensionality(image2d, image2d));
    REQUIRE(reg_haveSameDimensionality(image3d, field3d));
    REQUIRE(reg_haveSameDimensionality(image2d, grid2d));
    REQUIRE_FALSE(reg_haveSameDimensionality(image2d, image3d));
    REQUIRE_FALSE(reg_haveSameDimensionality(grid2d, field3d));

    // The diagnosis names both inputs and says which is which
    const std::string message = reg_dimensionalityMismatchMessage(image3d, "reference image", image2d, "floating image");
    REQUIRE_THAT(message, ContainsSubstring("The reference image and the floating image must both be 2D or both be 3D"));
    REQUIRE_THAT(message, ContainsSubstring("the reference image is 3D and the floating image is 2D"));

    REQUIRE_THROWS_WITH(reg_checkSameDimensionality(image2d, "reference image", image3d, "floating image"),
                        ContainsSubstring("the reference image is 2D and the floating image is 3D"));
    REQUIRE_NOTHROW(reg_checkSameDimensionality(image3d, "reference image", image3d, "floating image"));
}

TEST_CASE("Content rejects a 2D image paired with a 3D one", "[validation]") {
    std::mt19937 gen(0);
    NiftiImage image2d = MakeImage(gen, { kSmall, kSmall });
    NiftiImage image3d = MakeImage(gen, { kSmall, kSmall, kSmall });

    // The warped image takes the floating image's dimensions on the reference grid: a 2D floating
    // image would leave a single slice of a 3D grid
    REQUIRE_THROWS_WITH(Content(image3d, image2d), kDimensionalityMatcher);
    REQUIRE_THROWS_WITH(Content(image2d, image3d), kDimensionalityMatcher);

    // Matched images: the warped image spans the reference grid
    NiftiImage floating3d(image3d, NiftiImage::Copy::Image);
    Content content(image3d, floating3d);
    REQUIRE(content.GetWarped().nVoxelsPerVolume() == image3d.nVoxelsPerVolume());
    REQUIRE(content.GetDeformationField()->nu == 3);
}

TEST_CASE("reg_resampleImage rejects a floating image or a warped image that disagrees with the field", "[validation]") {
    std::mt19937 gen(0);
    NiftiImage image2d = MakeImage(gen, { kSmall, kSmall });
    NiftiImage image3d = MakeImage(gen, { kSmall, kSmall, kSmall });
    NiftiImage smaller3d = MakeImage(gen, { kSmall - 2, kSmall - 2, kSmall - 2 });
    NiftiImage field2d = CreateDeformationField(image2d);
    NiftiImage field3d = CreateDeformationField(image3d);
    NiftiImage warped2d(image2d, NiftiImage::Copy::Image);
    NiftiImage warped3d(image3d, NiftiImage::Copy::Image);
    NiftiImage warpedSmaller3d(smaller3d, NiftiImage::Copy::Image);

    SECTION("2D floating image through a 3D field") {
        REQUIRE_THROWS_WITH(reg_resampleImage(image2d, warped3d, field3d, nullptr, 1, 0.f), kDimensionalityMatcher);
    }
    SECTION("3D floating image through a 2D field") {
        REQUIRE_THROWS_WITH(reg_resampleImage(image3d, warped2d, field2d, nullptr, 1, 0.f), kDimensionalityMatcher);
    }
    SECTION("Warped image on another grid than the field") {
        REQUIRE_THROWS_WITH(reg_resampleImage(image3d, warpedSmaller3d, field3d, nullptr, 1, 0.f), kGridMatcher);
    }
    SECTION("Matched inputs resample") {
        REQUIRE_NOTHROW(reg_resampleImage(image3d, warped3d, field3d, nullptr, 1, 0.f));
        REQUIRE_NOTHROW(reg_resampleImage(image2d, warped2d, field2d, nullptr, 1, 0.f));
    }
}

TEST_CASE("reg_resampleGradient rejects fields with different numbers of components", "[validation]") {
    std::mt19937 gen(0);
    const NiftiImage image2d = MakeImage(gen, { kSmall, kSmall });
    const NiftiImage image3d = MakeImage(gen, { kSmall, kSmall, kSmall });
    const NiftiImage smaller3d = MakeImage(gen, { kSmall - 2, kSmall - 2, kSmall - 2 });
    NiftiImage gradient3d = CreateDeformationField(image3d);
    NiftiImage field3d = CreateDeformationField(image3d);
    NiftiImage warped2d = CreateDeformationField(image2d);
    NiftiImage warped3d = CreateDeformationField(image3d);
    NiftiImage warpedSmaller3d = CreateDeformationField(smaller3d);

    SECTION("Different numbers of components") {
        REQUIRE_THROWS_WITH(reg_resampleGradient(gradient3d, warped2d, field3d, 1, 0.f), ContainsSubstring("same number of components"));
    }
    SECTION("Two-component fields on a 3D grid") {
        // The 3D kernel is chosen from the warped grid and reads a third component no field carries
        for (NiftiImage *field : { &gradient3d, &warped3d, &field3d }) {
            field->setDim(NiftiDim::U, 2);
            field->realloc();
        }
        REQUIRE_THROWS_WITH(reg_resampleGradient(gradient3d, warped3d, field3d, 1, 0.f), ContainsSubstring("two components when 2D"));
    }
    SECTION("Warped gradient on another grid than the field") {
        REQUIRE_THROWS_WITH(reg_resampleGradient(gradient3d, warpedSmaller3d, field3d, 1, 0.f), kGridMatcher);
    }
    SECTION("Matched fields resample") {
        REQUIRE_NOTHROW(reg_resampleGradient(gradient3d, warped3d, field3d, 1, 0.f));
    }
}

TEST_CASE("reg_defField_compose rejects fields of different dimensionality", "[validation]") {
    std::mt19937 gen(0);
    const NiftiImage image2d = MakeImage(gen, { kSmall, kSmall });
    const NiftiImage image3d = MakeImage(gen, { kSmall, kSmall, kSmall });
    NiftiImage field2d = CreateDeformationField(image2d);
    NiftiImage field3d = CreateDeformationField(image3d);
    NiftiImage other3d = CreateDeformationField(image3d);

    // The kernel follows the field to update and reads as many components from the look-up field
    REQUIRE_THROWS_WITH(reg_defField_compose(field3d, field2d, nullptr), kDimensionalityMatcher);
    REQUIRE_THROWS_WITH(reg_defField_compose(field2d, field3d, nullptr), kDimensionalityMatcher);
    REQUIRE_NOTHROW(reg_defField_compose(field3d, other3d, nullptr));
}

TEST_CASE("Spline evaluation rejects a grid whose dimensionality differs from the field", "[validation]") {
    std::mt19937 gen(0);
    const NiftiImage image2d = MakeImage(gen, { kSmall, kSmall });
    const NiftiImage image3d = MakeImage(gen, { kSmall, kSmall, kSmall });
    NiftiImage grid2d = CreateControlPointGrid(image2d, kProductionGridSpacing);
    NiftiImage grid3d = CreateControlPointGrid(image3d, kProductionGridSpacing);
    NiftiImage field2d = CreateDeformationField(image2d);
    NiftiImage field3d = CreateDeformationField(image3d);

    SECTION("Cubic B-spline grid") {
        REQUIRE_THROWS_WITH(reg_spline_getDeformationField(grid3d, field2d, nullptr, true, true), kDimensionalityMatcher);
        REQUIRE_THROWS_WITH(reg_spline_getDeformationField(grid2d, field3d, nullptr, true, true), kDimensionalityMatcher);
        REQUIRE_NOTHROW(reg_spline_getDeformationField(grid3d, field3d, nullptr, true, true));
        REQUIRE_NOTHROW(reg_spline_getDeformationField(grid2d, field2d, nullptr, true, true));
    }
    SECTION("Velocity grid") {
        // The flow field takes the deformation field's geometry before the grid is evaluated into it
        NiftiImage velocity3d = MakeVelocityGrid(image3d);
        NiftiImage velocity2d = MakeVelocityGrid(image2d);
        REQUIRE_THROWS_WITH(reg_spline_getDefFieldFromVelocityGrid(velocity3d, field2d, false), kDimensionalityMatcher);
        REQUIRE_THROWS_WITH(reg_spline_getFlowFieldFromVelocityGrid(velocity2d, field3d), kDimensionalityMatcher);
        REQUIRE_NOTHROW(reg_spline_getDefFieldFromVelocityGrid(velocity3d, field3d, false));
    }
}

TEST_CASE("Flow field exponentiation rejects a deformation field on another grid", "[validation]") {
    std::mt19937 gen(0);
    const NiftiImage image2d = MakeImage(gen, { kSmall, kSmall });
    const NiftiImage image3d = MakeImage(gen, { kSmall, kSmall, kSmall });
    const NiftiImage smaller3d = MakeImage(gen, { kSmall - 2, kSmall - 2, kSmall - 2 });
    NiftiImage field2d = CreateDeformationField(image2d);
    NiftiImage field3d = CreateDeformationField(image3d);
    NiftiImage fieldSmaller3d = CreateDeformationField(smaller3d);

    // Both fields serve as ping-pong buffers of the squaring and are copied into one another
    REQUIRE_THROWS_WITH(reg_defField_getDeformationFieldFromFlowField(MakeFlowField(image3d), field2d, false), kGridMatcher);
    REQUIRE_THROWS_WITH(reg_defField_getDeformationFieldFromFlowField(MakeFlowField(image3d), fieldSmaller3d, false), kGridMatcher);
    REQUIRE_NOTHROW(reg_defField_getDeformationFieldFromFlowField(MakeFlowField(image3d), field3d, false));
}

TEST_CASE("reg_defFieldInvert rejects a 2D output field", "[validation]") {
    std::mt19937 gen(0);
    const NiftiImage image2d = MakeImage(gen, { kSmall, kSmall });
    const NiftiImage image3d = MakeImage(gen, { kSmall, kSmall, kSmall });
    NiftiImage field3d = CreateDeformationField(image3d);
    NiftiImage output2d = CreateDeformationField(image2d);
    NiftiImage output3d = CreateDeformationField(image3d);

    // The inverse is written with three components on the output grid
    REQUIRE_THROWS_WITH(reg_defFieldInvert(field3d, output2d, 1.e-6f), ContainsSubstring("3D as well"));
    REQUIRE_NOTHROW(reg_defFieldInvert(field3d, output3d, 1.e-6f));
}

TEST_CASE("Jacobian maps reject a transformation that disagrees with the image they fill", "[validation]") {
    std::mt19937 gen(0);
    NiftiImage image2d = MakeImage(gen, { kSmall, kSmall });
    NiftiImage image3d = MakeImage(gen, { kSmall, kSmall, kSmall });
    NiftiImage smaller3d = MakeImage(gen, { kSmall - 2, kSmall - 2, kSmall - 2 });
    NiftiImage grid2d = CreateControlPointGrid(image2d, kProductionGridSpacing);
    NiftiImage grid3d = CreateControlPointGrid(image3d, kProductionGridSpacing);
    NiftiImage jacobian2d(image2d, NiftiImage::Copy::Image);
    NiftiImage jacobian3d(image3d, NiftiImage::Copy::Image);
    NiftiImage jacobianSmaller3d(smaller3d, NiftiImage::Copy::Image);

    SECTION("Determinant map from a spline grid") {
        REQUIRE_THROWS_WITH(reg_spline_GetJacobianMap(grid3d, jacobian2d), kDimensionalityMatcher);
        REQUIRE_THROWS_WITH(reg_spline_GetJacobianMap(grid2d, jacobian3d), kDimensionalityMatcher);
        REQUIRE_NOTHROW(reg_spline_GetJacobianMap(grid3d, jacobian3d));
    }
    SECTION("Matrix map from a spline grid") {
        vector<mat33> matrices(image3d.nVoxelsPerVolume());
        REQUIRE_THROWS_WITH(reg_spline_GetJacobianMatrix(image2d, grid3d, matrices.data()), kDimensionalityMatcher);
        REQUIRE_NOTHROW(reg_spline_GetJacobianMatrix(image3d, grid3d, matrices.data()));
    }
    SECTION("Determinant map from a deformation field on another grid") {
        NiftiImage field3d = CreateDeformationField(image3d);
        REQUIRE_THROWS_WITH(reg_defField_getJacobianMap(field3d, jacobianSmaller3d), kGridMatcher);
        REQUIRE_NOTHROW(reg_defField_getJacobianMap(field3d, jacobian3d));
    }
    SECTION("Determinant map from a flow field on another grid") {
        REQUIRE_THROWS_WITH(reg_defField_GetJacobianDetFromFlowField(jacobianSmaller3d, MakeFlowField(image3d)), kGridMatcher);
        REQUIRE_NOTHROW(reg_defField_GetJacobianDetFromFlowField(jacobian3d, MakeFlowField(image3d)));
    }
}
