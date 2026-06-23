// OpenCL is not supported for this test
#undef USE_OPENCL

#include "reg_test_common.h"
#include "_reg_f3d.h"
#include <algorithm>
#include <cmath>
#include <cstring>

/*
    This test file checks the behaviour of the reg_f3d -freshgrid option.

    When a control point grid is provided as an input (-incpp), the default
    behaviour reuses that grid as the coarsest pyramid level and refines it
    down - so the final spacing is derived from the input grid and compounds
    across chained stages. The -freshgrid option (UseFreshGrid) instead builds
    a brand new grid at the spacing requested through SetSpacing (-sx/-sy/-sz),
    matching Elastix's behaviour, while still carrying the input grid's
    deformation forward as a warm start.

    The following cases are covered:
    - No input grid: a fresh grid is built at the requested spacing
    - Input grid, default: the grid keeps the input grid's geometry
    - Input grid + freshgrid: the grid uses the requested spacing, not the
      input grid's, and is initialised from the input grid's deformation
*/

// Small subclass to drive the protected Initialise() and inspect the
// protected control point grid and spacing
template<class T>
class TestF3d: public reg_f3d<T> {
public:
    TestF3d(int refTimePoints, int floTimePoints): reg_f3d<T>(refTimePoints, floTimePoints) {}
    using reg_f3d<T>::Initialise;
    NiftiImage& GetGrid() { return this->controlPointGrid; }
    T GetSpacing(int i) { return this->spacing[i]; }
};

class FreshGridTest {
protected:
    // Constant translation (in mm) stored in the input control point grid
    static constexpr float tx = 1.5f, ty = -2.f, tz = 0.5f;
    // Registration parameters
    static constexpr unsigned levelNumber = 3;
    static constexpr float finalSpacing = 4.f;     // requested spacing (-sx)
    static constexpr float inputSpacing = 8.f;     // input grid spacing
    NiftiImage reference;
    NiftiImage floating;

public:
    FreshGridTest() {
        constexpr NiftiImage::dim_t dimSize = 16;
        reference = NiftiImage({ dimSize, dimSize, dimSize }, NIFTI_TYPE_FLOAT32);
        floating = NiftiImage({ dimSize, dimSize, dimSize }, NIFTI_TYPE_FLOAT32);
        // reg_f3d::Initialise() logs the image filenames; give the in-memory test
        // images a name (real images are always read from a file) so the logging
        // does not stream a null pointer. strdup is used so nifti_image_free can
        // safely release it.
        reference->fname = strdup("reference");
        floating->fname = strdup("floating");
    }

    // Build a coarse input grid holding a constant translation
    NiftiImage CreateInputGrid() {
        NiftiImage inputGrid;
        const float spacing[3] = { inputSpacing, inputSpacing, inputSpacing };
        reg_createControlPointGrid<float>(inputGrid, reference, spacing);
        mat44 translation;
        Mat44Eye(&translation);
        translation.m[0][3] = tx;
        translation.m[1][3] = ty;
        translation.m[2][3] = tz;
        reg_affine_getDeformationField(&translation, inputGrid);
        return inputGrid;
    }

    unique_ptr<TestF3d<float>> CreateReg() {
        unique_ptr<TestF3d<float>> reg(new TestF3d<float>(1, 1));
        reg->SetReferenceImage(reference);
        reg->SetFloatingImage(floating);
        reg->SetLevelNumber(levelNumber);
        reg->SetLevelToPerform(levelNumber);
        reg->SetSpacing(0, finalSpacing);
        reg->SetSpacing(1, finalSpacing);
        reg->SetSpacing(2, finalSpacing);
        reg->DoNotPrintOutInformation();
        return reg;
    }
};

TEST_CASE_METHOD(FreshGridTest, "Fresh grid - no input grid", "[unit]") {
    // Without an input grid, the coarsest level spacing is finalSpacing * 2^(levelNumber-1)
    const float expectedSpacing = finalSpacing * powf(2, levelNumber - 1);

    auto reg = CreateReg();
    reg->Initialise();
    NiftiImage& grid = reg->GetGrid();

    REQUIRE(std::abs(grid->dx - expectedSpacing) < EPS);
    REQUIRE(std::abs(grid->dy - expectedSpacing) < EPS);
    REQUIRE(std::abs(grid->dz - expectedSpacing) < EPS);
}

TEST_CASE_METHOD(FreshGridTest, "Fresh grid - input grid, default behaviour", "[unit]") {
    // The default behaviour keeps the input grid: its spacing and dimensions are preserved
    NiftiImage inputGrid = CreateInputGrid();
    const auto inputNx = inputGrid->nx;

    auto reg = CreateReg();
    reg->SetControlPointGridImage(inputGrid);
    reg->Initialise();
    NiftiImage& grid = reg->GetGrid();

    REQUIRE(std::abs(grid->dx - inputSpacing) < EPS);
    REQUIRE(grid->nx == inputNx);
    // The final spacing is back-computed from the input grid (compounding behaviour)
    REQUIRE(std::abs(reg->GetSpacing(0) - inputSpacing / powf(2, levelNumber - 1)) < EPS);
}

TEST_CASE_METHOD(FreshGridTest, "Fresh grid - input grid with freshgrid", "[unit]") {
    NiftiImage inputGrid = CreateInputGrid();
    const auto inputNx = inputGrid->nx;
    const float expectedSpacing = finalSpacing * powf(2, levelNumber - 1);

    auto reg = CreateReg();
    reg->SetControlPointGridImage(inputGrid);
    reg->UseFreshGrid();
    reg->Initialise();
    NiftiImage& grid = reg->GetGrid();

    SECTION("Grid uses the requested spacing, not the input grid spacing") {
        REQUIRE(std::abs(grid->dx - expectedSpacing) < EPS);
        REQUIRE(std::abs(grid->dy - expectedSpacing) < EPS);
        REQUIRE(std::abs(grid->dz - expectedSpacing) < EPS);
        // The fresh grid geometry differs from the input grid
        REQUIRE(grid->nx != inputNx);
        // The requested final spacing is preserved (no compounding)
        REQUIRE(std::abs(reg->GetSpacing(0) - finalSpacing) < EPS);
    }

    SECTION("Input grid deformation is carried forward as a warm start") {
        // Evaluate the fresh grid's deformation over the reference image
        NiftiImage deformation = CreateDeformationField(reference);
        // Keep a copy of the identity field (each voxel holds its world position)
        NiftiImage world(deformation, NiftiImage::Copy::Image);
        reg_spline_getDeformationField(grid, deformation, nullptr, false, true);

        const auto defPtr = deformation.data();
        const auto worldPtr = world.data();
        const size_t voxelsPerVolume = deformation.nVoxelsPerVolume();
        const int nx = deformation->nx, ny = deformation->ny, nz = deformation->nz;

        // Only check the central region, which has full grid support and so
        // reproduces the constant translation exactly
        double maxErr = 0;
        bool checkedAnyVoxel = false;
        for (int z = nz / 4; z < 3 * nz / 4; ++z) {
            for (int y = ny / 4; y < 3 * ny / 4; ++y) {
                for (int x = nx / 4; x < 3 * nx / 4; ++x) {
                    const size_t v = (static_cast<size_t>(z) * ny + y) * nx + x;
                    const double dispX = static_cast<float>(defPtr[v]) - static_cast<float>(worldPtr[v]);
                    const double dispY = static_cast<float>(defPtr[v + voxelsPerVolume]) - static_cast<float>(worldPtr[v + voxelsPerVolume]);
                    const double dispZ = static_cast<float>(defPtr[v + 2 * voxelsPerVolume]) - static_cast<float>(worldPtr[v + 2 * voxelsPerVolume]);
                    maxErr = std::max(maxErr, std::abs(dispX - tx));
                    maxErr = std::max(maxErr, std::abs(dispY - ty));
                    maxErr = std::max(maxErr, std::abs(dispZ - tz));
                    checkedAnyVoxel = true;
                }
            }
        }
        REQUIRE(checkedAnyVoxel);
        // The warm-started grid reproduces the input translation (not an identity transform)
        REQUIRE(maxErr < 0.001);
    }
}
