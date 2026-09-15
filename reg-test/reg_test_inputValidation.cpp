#include "reg_test_common.h"
#include "_reg_aladin_sym.h"
#include "_reg_f3d.h"
#include <catch2/matchers/catch_matchers_string.hpp>

/**
 *  Input validation of the registration front-ends (reg_aladin, reg_aladin_sym, reg_f3d)
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
