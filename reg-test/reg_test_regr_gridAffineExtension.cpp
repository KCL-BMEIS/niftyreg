#include "reg_test_common.h"
#include "CudaF3dContent.h"

/**
 *  Deformation field from a control point grid that carries affine extensions, CPU against CUDA.
 *
 *  reg_createSymmetricControlPointGrids stores the square root of the initial affine twice as
 *  extensions on each grid (what reg_f3d -vel -aff sets up): the first is applied before the spline
 *  and the second composed after it. A backend that ignores them returns half of the affine, or none
 *  of it, and nothing else in the suite puts an extension on a grid given to CUDA.
 *
 *  With a zero velocity (or an identity spline) the field has to be the affine itself, which gives an
 *  oracle that does not depend on either implementation. A CPU/CUDA comparison alone would pass if
 *  both ignored the extensions.
**/

enum class GridKind { Velocity, Spline };

// Field of the forward (or backward) grid created from the affine, on the given platform
static NiftiImage gridAffineDeformationField(const PlatformType platformType,
                                             const NiftiImage& referenceIn,
                                             const mat44& affine,
                                             const GridKind gridKind,
                                             const bool backward,
                                             const float amplitude) {
    NiftiImage reference(referenceIn);
    const float spacing[3]{ reference->dx * kProductionGridSpacing,
                            reference->dy * kProductionGridSpacing,
                            reference->dz * kProductionGridSpacing };
    NiftiImage forwardGrid, backwardGrid;
    reg_createSymmetricControlPointGrids<float>(forwardGrid, backwardGrid, reference, reference, &affine, spacing);
    NiftiImage& grid = backward ? backwardGrid : forwardGrid;
    REQUIRE(grid->num_ext == 2);
    if (gridKind == GridKind::Velocity)
        grid->intent_p1 = SPLINE_VEL_GRID;

    // Same seed on both platforms, so they are given the same grid
    std::mt19937 gen(0);
    std::uniform_real_distribution<float> distr(-amplitude, amplitude);
    auto gridPtr = grid.data();
    if (amplitude > 0)
        for (size_t i = 0; i < grid.nVoxels(); ++i)
            gridPtr[i] = static_cast<float>(gridPtr[i]) + distr(gen);

    unique_ptr<F3dContent> content;
    if (platformType == PlatformType::Cuda)
        content.reset(new CudaF3dContent(reference, reference, grid));
    else content.reset(new F3dContent(reference, reference, grid));
    Platform platform(platformType);
    unique_ptr<Compute> compute{ platform.CreateCompute(*content) };

    if (gridKind == GridKind::Velocity)
        compute->GetDefFieldFromVelocityGrid(true);
    else compute->GetDeformationField(false, true);

    return NiftiImage(DeformationFieldForComparison(*content), NiftiImage::Copy::Image);
}

class GridAffineExtensionTest {
protected:
    using TestCase = std::tuple<std::string, NiftiImage, NiftiImage, NiftiImage, NiftiImage, bool>;
    // name, expected affine field, identity field, cpu, cuda, whether the grid is an identity

    inline static vector<TestCase> testCases;

public:
    GridAffineExtensionTest() {
        if (!testCases.empty())
            return;

        for (const bool is3d : { false, true }) {
            NiftiImage reference = is3d ? makeImage({ 16, 17, 15 }) : makeImage({ 17, 18 });
            setAnisotropicSform(reference);

            // The translation spans several voxels
            const mat44 affine = makeRotationScalingAffine(is3d, 0.3f, { 6.5f, -4.25f, 5.75f });
            const mat44 inverseAffine = nifti_mat44_inverse(affine);
            mat44 identity;
            Mat44Eye(&identity);

            for (const GridKind gridKind : { GridKind::Velocity, GridKind::Spline }) {
                for (const bool backward : { false, true }) {
                    for (const float amplitude : { 0.f, 0.5f }) {
                        const std::string name = (is3d ? "3D "s : "2D "s) +
                            (gridKind == GridKind::Velocity ? "velocity grid " : "spline grid ") +
                            (backward ? "backward " : "forward ") +
                            (amplitude > 0 ? "random" : "identity");
                        testCases.push_back({
                            name,
                            ExpectedAffineField(reference, backward ? inverseAffine : affine),
                            ExpectedAffineField(reference, identity),
                            gridAffineDeformationField(PlatformType::Cpu, reference, affine, gridKind, backward, amplitude),
                            gridAffineDeformationField(PlatformType::Cuda, reference, affine, gridKind, backward, amplitude),
                            amplitude == 0
                        });
                    }
                }
            }
        }
    }
};

TEST_CASE_METHOD(GridAffineExtensionTest, "Regression Grid Affine Extension", "[regression]") {
    for (auto&& testCase : testCases) {
        auto&& [sectionName, expected, identityField, defFieldCpu, defFieldCuda, isIdentityGrid] = testCase;

        SECTION(sectionName) {
            NR_COUT << "\n**************** Section " << sectionName << " ****************" << std::endl;

            // The affine moves every voxel, so a field left at identity means it was not applied
            RequireChanged(identityField, defFieldCpu, "CPU deformation field");
            RequireChanged(identityField, defFieldCuda, "CUDA deformation field");

            const Deviation cpuVsCuda = CompareImages(defFieldCpu, defFieldCuda);
            ReportDeviation("CPU vs CUDA", cpuVsCuda);
            const Deviation cpuVsExpected = CompareImages(defFieldCpu, expected);
            const Deviation cudaVsExpected = CompareImages(defFieldCuda, expected);
            ReportDeviation("CPU vs affine", cpuVsExpected);
            ReportDeviation("CUDA vs affine", cudaVsExpected);

            REQUIRE(cpuVsCuda.max == 0);
            if (isIdentityGrid) {
                // Positions reach tens of millimetres in single precision, and the velocity path
                // goes through a matrix square root and the squaring steps
                REQUIRE(cpuVsExpected.max < 1e-3);
                REQUIRE(cudaVsExpected.max < 1e-3);
            } else {
                // The perturbation has to show, or this case repeats the identity one
                REQUIRE(cpuVsExpected.max > 1e-2);
                REQUIRE(cudaVsExpected.max > 1e-2);
            }
        }
    }
}
