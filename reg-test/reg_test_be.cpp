// OpenCL is not supported for this test
#undef USE_OPENCL

#include "reg_test_common.h"

/*
    The approximated bending energy, checked against closed forms.

    The value sums, over the interior control points, the squared second derivatives the 3x3(x3)
    B-spline knot stencils produce, and divides by nvox. Three families of grid make that a known
    number without restating the stencils:

      - identity and AFFINE grids: every second derivative of an affine function vanishes, so the
        energy is zero. The affine includes shear as well as scaling - a stencil that accidentally
        picked up a first-derivative term would survive a pure scaling test;
      - a QUADRATIC displacement a*(index)^2 added along one axis of one component: the (1,-2,1)
        second-derivative stencil gives exactly 2a, every other term has a vanishing factor
        (sum(second) = 0, sum(first) = 0 across the perpendicular stencils), so the energy is
        4 a^2 N_interior / nvox. The derivation is three lines of algebra on the stencil, not a copy
        of the loop; the tolerance absorbs the float literals the production basis tables hold
        (their partition of unity is 1 to ~1e-6, not exactly 1).

    The second TEST_CASE compares the analytical bending-energy GRADIENT against central finite
    differences of the value, the check that ties the two definitions together (and which found the
    linear-energy gradient inconsistent). Scale and shape are gated separately, as there.
*/

class BendingEnergyTest {
protected:
    using TestData = std::tuple<std::string, NiftiImage, NiftiImage, float>;
    using TestCase = std::tuple<std::string, float, float>;

    inline static vector<TestCase> testCases;

public:
    BendingEnergyTest() {
        if (!testCases.empty())
            return;

        // Create 2D and 3D reference images
        constexpr NiftiImage::dim_t dimSize = 8;
        NiftiImage reference2d({ dimSize, dimSize }, NIFTI_TYPE_FLOAT32);
        NiftiImage reference3d({ dimSize, dimSize, dimSize }, NIFTI_TYPE_FLOAT32);
        setIdentitySform(reference2d);
        setIdentitySform(reference3d);

        NiftiImage controlPointGrid2d = CreateControlPointGrid(reference2d);
        NiftiImage controlPointGrid3d = CreateControlPointGrid(reference3d);

        vector<TestData> testData;
        testData.emplace_back(TestData("BE identity 2D", reference2d, controlPointGrid2d, 0.f));
        testData.emplace_back(TestData("BE identity 3D", reference3d, controlPointGrid3d, 0.f));

        // A general affine, with shear and translation, applied to the node positions: all second
        // derivatives vanish, so the bending energy must still be zero
        {
            mat44 affine;
            Mat44Eye(&affine);
            affine.m[0][0] = 0.8f; affine.m[0][1] = 0.15f; affine.m[0][3] = -2.5f;
            affine.m[1][0] = -0.1f; affine.m[1][1] = 1.2f; affine.m[1][3] = 1.75f;
            NiftiImage affineGrid2d(controlPointGrid2d, NiftiImage::Copy::Image);
            ApplyAffineToGrid(affineGrid2d, affine);
            testData.emplace_back(TestData("BE affine 2D", reference2d, std::move(affineGrid2d), 0.f));

            affine.m[0][2] = 0.05f; affine.m[1][2] = -0.08f;
            affine.m[2][0] = 0.12f; affine.m[2][1] = -0.06f; affine.m[2][2] = 1.1f; affine.m[2][3] = 0.5f;
            NiftiImage affineGrid3d(controlPointGrid3d, NiftiImage::Copy::Image);
            ApplyAffineToGrid(affineGrid3d, affine);
            testData.emplace_back(TestData("BE affine 3D", reference3d, std::move(affineGrid3d), 0.f));
        }

        // Quadratic displacement along each axis: u_c(index) = a * (index along that axis)^2.
        // Every interior node's second derivative along that axis is exactly 2a; every other term
        // vanishes through sum(second) = 0 or sum(first) = 0. Energy = 4 a^2 N_interior / nvox.
        constexpr float amplitude = 0.25f;
        const auto addQuadratic = [](NiftiImage& grid, int axis, int component, float a) {
            const size_t volume = grid.nVoxelsPerVolume();
            const int nx = grid->nx, ny = grid->ny, nz = grid->nz;
            auto ptr = grid.data();
            for (int k = 0; k < nz; ++k)
                for (int j = 0; j < ny; ++j)
                    for (int i = 0; i < nx; ++i) {
                        const int idx = axis == 0 ? i : (axis == 1 ? j : k);
                        const size_t index = (static_cast<size_t>(k) * ny + j) * nx + i;
                        ptr[component * volume + index] =
                            static_cast<float>(ptr[component * volume + index]) + a * idx * idx;
                    }
        };
        const auto quadraticExpected = [](const NiftiImage& grid, float a) {
            const size_t interior = static_cast<size_t>(grid->nx - 2) * (grid->ny - 2) *
                                    (grid->nz > 1 ? grid->nz - 2 : 1);
            return static_cast<float>(4.0 * a * a * interior / static_cast<double>(grid->nvox));
        };
        for (const auto& [label, axis, component] : { std::tuple{ "x-axis, x-component", 0, 0 },
                                                      std::tuple{ "y-axis, y-component", 1, 1 },
                                                      std::tuple{ "y-axis, x-component", 1, 0 } }) {
            NiftiImage grid2d(controlPointGrid2d, NiftiImage::Copy::Image);
            addQuadratic(grid2d, axis, component, amplitude);
            testData.emplace_back(TestData(std::string("BE quadratic 2D ") + label, reference2d,
                                           std::move(grid2d), quadraticExpected(controlPointGrid2d, amplitude)));
        }
        for (const auto& [label, axis, component] : { std::tuple{ "x-axis, x-component", 0, 0 },
                                                      std::tuple{ "z-axis, z-component", 2, 2 },
                                                      std::tuple{ "z-axis, x-component", 2, 0 } }) {
            NiftiImage grid3d(controlPointGrid3d, NiftiImage::Copy::Image);
            addQuadratic(grid3d, axis, component, amplitude);
            testData.emplace_back(TestData(std::string("BE quadratic 3D ") + label, reference3d,
                                           std::move(grid3d), quadraticExpected(controlPointGrid3d, amplitude)));
        }

        // Compute the bending energy for each case on every platform
        for (auto&& data : testData) {
            for (auto&& platformType : PlatformTypes) {
                auto [testName, reference, controlPointGrid, expected] = data;
                shared_ptr<Platform> platform{ new Platform(platformType) };
                unique_ptr<F3dContentCreator> contentCreator{ dynamic_cast<F3dContentCreator*>(platform->CreateContentCreator(ContentType::F3d)) };
                unique_ptr<F3dContent> content{ contentCreator->Create(reference, reference, controlPointGrid) };
                unique_ptr<Compute> compute{ platform->CreateCompute(*content) };
                const float be = static_cast<float>(compute->ApproxBendingEnergy());
                testCases.push_back({ testName + " " + platform->GetName(), be, expected });
            }
        }
    }
};

TEST_CASE_METHOD(BendingEnergyTest, "Bending Energy", "[unit]") {
    for (auto&& testCase : testCases) {
        auto&& [testName, result, expected] = testCase;

        SECTION(testName) {
            NR_COUT << "  " << std::setw(44) << std::left << testName
                    << " result = " << std::scientific << std::setprecision(6) << result
                    << " expected = " << expected << std::endl;
            // Relative bound: the production basis tables hold 6-digit float literals whose
            // partition of unity is off by ~1e-6, which enters the quadratic case squared
            INFO(testName << ": result " << result << ", expected " << expected);
            REQUIRE(std::abs(result - expected) < 1e-5 * (1.0 + std::abs(expected)));
        }
    }
}

class BendingEnergyGradientFiniteDiffTest {
protected:
    using TestCase = std::tuple<std::string, NiftiImage, NiftiImage>;
    inline static vector<TestCase> testCases;

public:
    BendingEnergyGradientFiniteDiffTest() {
        if (!testCases.empty())
            return;

        constexpr float weight = 1.f;
        constexpr double h = 1e-3;   // finite-difference step (coefficients are O(1..10) mm)
        std::mt19937 gen(0);
        std::uniform_real_distribution<double> distr(-0.5, 0.5);

        Platform platformCpu(PlatformType::Cpu);

        for (const int dim : { 2, 3 }) {
            std::vector<NiftiImage::dim_t> dims(dim, 8);
            NiftiImage reference(dims, NIFTI_TYPE_FLOAT32);
            setIdentitySform(reference);

            // Double-precision content so the finite differences resolve well below the gradient
            NiftiImage controlPointGrid;
            const float spacing[3]{ reference->dx * 2, reference->dy * 2, reference->dz * 2 };
            reg_createControlPointGrid<double>(controlPointGrid, reference, spacing);
            NiftiImage floating(reference);
            {
                auto cpgPtr = controlPointGrid.data();
                for (size_t j = 0; j < controlPointGrid.nVoxels(); j++)
                    cpgPtr[j] = static_cast<double>(cpgPtr[j]) + distr(gen);
            }

            const std::string testName = std::to_string(dim) + "D near-identity";

            // Analytical gradient
            NiftiImage refA(reference), floA(floating), cpgA(controlPointGrid);
            unique_ptr<F3dContent> contentA{ new F3dContent(refA, floA, cpgA, nullptr, nullptr, nullptr, sizeof(double)) };
            unique_ptr<Compute> computeA{ platformCpu.CreateCompute(*contentA) };
            computeA->ApproxBendingEnergyGradient(weight);
            NiftiImage analytical = contentA->GetTransformationGradient();

            // Numerical gradient via central finite differences on the value
            NiftiImage refN(reference), floN(floating), cpgN(controlPointGrid);
            unique_ptr<F3dContent> contentN{ new F3dContent(refN, floN, cpgN, nullptr, nullptr, nullptr, sizeof(double)) };
            unique_ptr<Compute> computeN{ platformCpu.CreateCompute(*contentN) };
            NiftiImage& liveCpg = contentN->GetControlPointGrid();
            auto cpgPtr = liveCpg.data();

            NiftiImage numerical(analytical, NiftiImage::Copy::ImageInfoAndAllocData);
            auto numPtr = numerical.data();
            for (size_t i = 0; i < liveCpg.nVoxels(); ++i) {
                const double c = cpgPtr[i];
                cpgPtr[i] = c + h;
                const double ePlus = computeN->ApproxBendingEnergy();
                cpgPtr[i] = c - h;
                const double eMinus = computeN->ApproxBendingEnergy();
                cpgPtr[i] = c;   // restore
                numPtr[i] = weight * (ePlus - eMinus) / (2 * h);
            }

            testCases.push_back({ testName, std::move(analytical), std::move(numerical) });
        }
    }
};

/*
    The analytical bending-energy gradient against central finite differences of the value: the check
    that the gradient is the derivative of the number the objective reports, which comparing two
    backends can never establish.
*/
TEST_CASE_METHOD(BendingEnergyGradientFiniteDiffTest, "Bending Energy Gradient Finite Difference", "[unit]") {
    for (auto&& testCase : testCases) {
        auto&& [testName, analytical, numerical] = testCase;

        SECTION(testName) {
            const auto anaPtr = analytical.data();
            const auto numPtr = numerical.data();

            double maxAbs = 0;
            for (size_t i = 0; i < numerical.nVoxels(); ++i)
                maxAbs = std::max(maxAbs, std::abs(static_cast<double>(numPtr[i])));
            const double significant = 0.1 * maxAbs;

            double maxAbsDiff = 0, sumRatio = 0;
            size_t ratioCount = 0;
            for (size_t i = 0; i < analytical.nVoxels(); ++i) {
                const double a = anaPtr[i];
                const double n = numPtr[i];
                maxAbsDiff = std::max(maxAbsDiff, std::abs(a - n));
                if (std::abs(n) > significant) { sumRatio += a / n; ++ratioCount; }
            }
            const double meanRatio = ratioCount ? sumRatio / static_cast<double>(ratioCount) : 0;

            NR_COUT << "  " << testName << ": mean(analytical/numerical) = " << std::fixed
                    << std::setprecision(10) << meanRatio << " over " << ratioCount
                    << " entries, max |analytical-numerical| = " << std::scientific << maxAbsDiff
                    << " (max |numerical| = " << maxAbs << ")" << std::endl;

            REQUIRE(ratioCount > 0);
            INFO("mean analytical/numerical = " << meanRatio);
            REQUIRE(std::abs(meanRatio - 1.0) < 1e-2);
            INFO("max residual " << maxAbsDiff << " vs max |numerical| " << maxAbs);
            REQUIRE(maxAbsDiff <= 5e-3 * maxAbs);
        }
    }
}
