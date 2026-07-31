#include "reg_test_common.h"
#include "CudaF3dContent.h"

/**
 *  Update velocity field regression test to ensure the CPU and CUDA versions yield the same output
**/

class UpdateVelocityFieldTest {
protected:
    using TestData = std::tuple<std::string, NiftiImage, NiftiImage, NiftiImage, float>;
    using TestCase = std::tuple<std::string, NiftiImage, NiftiImage>;

    inline static vector<TestCase> testCases;

public:
    UpdateVelocityFieldTest() {
        if (!testCases.empty())
            return;

        // Create a random number generator
        std::mt19937 gen(0);
        std::uniform_real_distribution<float> distr(-1, 1);

        // Create 2D and 3D reference images
        constexpr NiftiImage::dim_t dimSize = 4;
        NiftiImage reference2d({ dimSize, dimSize }, NIFTI_TYPE_FLOAT32);
        NiftiImage reference3d({ dimSize, dimSize, dimSize }, NIFTI_TYPE_FLOAT32);

        // Create 2D and 3D control point grids
        NiftiImage controlPointGrid2d = CreateControlPointGrid(reference2d);
        NiftiImage controlPointGrid3d = CreateControlPointGrid(reference3d);

        // Create transformation gradient images and fill them with random values
        NiftiImage transGrad2d(controlPointGrid2d, NiftiImage::Copy::ImageInfoAndAllocData);
        NiftiImage transGrad3d(controlPointGrid3d, NiftiImage::Copy::ImageInfoAndAllocData);
        auto transGrad2dPtr = transGrad2d.data();
        auto transGrad3dPtr = transGrad3d.data();
        for (size_t i = 0; i < transGrad2d.nVoxels(); i++)
            transGrad2dPtr[i] = distr(gen);
        for (size_t i = 0; i < transGrad3d.nVoxels(); i++)
            transGrad3dPtr[i] = distr(gen);

        // Add the test data
        vector<TestData> testData;
        testData.emplace_back(TestData(
            "2D",
            std::move(reference2d),
            std::move(controlPointGrid2d),
            std::move(transGrad2d),
            distr(gen)  // scale
        ));
        testData.emplace_back(TestData(
            "3D",
            std::move(reference3d),
            std::move(controlPointGrid3d),
            std::move(transGrad3d),
            distr(gen)  // scale
        ));

        // Create the platforms
        Platform platformCpu(PlatformType::Cpu);
        Platform platformCuda(PlatformType::Cuda);

        for (auto&& testData : testData) {
            for (int optimiseX = 0; optimiseX < 2; optimiseX++) {
                for (int optimiseY = 0; optimiseY < 2; optimiseY++) {
                    for (int optimiseZ = 0; optimiseZ < 2; optimiseZ++) {
                        // Get the test data
                        auto&& [testName, reference, controlPointGrid, transGrad, scale] = testData;
                        testName += " scale=" + std::to_string(scale) + " " + (optimiseX ? "X" : "noX") + " " + (optimiseY ? "Y" : "noY") + " " + (optimiseZ ? "Z" : "noZ");

                        // Create images
                        NiftiImage referenceCpu(reference), referenceCuda(reference);
                        NiftiImage cppCpu(controlPointGrid), cppCuda(controlPointGrid);

                        // Create the content
                        unique_ptr<F3dContent> contentCpu{ new F3dContent(referenceCpu, referenceCpu, cppCpu) };
                        unique_ptr<F3dContent> contentCuda{ new CudaF3dContent(referenceCuda, referenceCuda, cppCuda) };

                        // Set the transformation gradient image to host the computation
                        contentCpu->GetTransformationGradient().copyData(transGrad);
                        contentCpu->UpdateTransformationGradient();
                        contentCuda->F3dContent::GetTransformationGradient().copyData(transGrad);
                        contentCuda->UpdateTransformationGradient();

                        // Create the computes
                        unique_ptr<Compute> computeCpu{ platformCpu.CreateCompute(*contentCpu) };
                        unique_ptr<Compute> computeCuda{ platformCuda.CreateCompute(*contentCuda) };

                        // Update the velocity field
                        computeCpu->UpdateVelocityField(scale, optimiseX, optimiseY, optimiseZ);
                        computeCuda->UpdateVelocityField(scale, optimiseX, optimiseY, optimiseZ);

                        // Save the results for testing
                        testCases.push_back({ testName, std::move(contentCpu->GetTransformationGradient()),
                                            std::move(contentCuda->GetTransformationGradient()) });
                    }
                }
            }
        }
    }
};

TEST_CASE_METHOD(UpdateVelocityFieldTest, "Regression Update Velocity Field", "[regression]") {
    // Loop over all generated test cases
    for (auto&& testCase : testCases) {
        // Retrieve test information
        auto&& [sectionName, transGradCpu, transGradCuda] = testCase;

        SECTION(sectionName) {
            // The comparison below is only meaningful if the operation ran at all
            RequireNonZero(transGradCpu, "the updated velocity field");
            NR_COUT << "\n**************** Section " << sectionName << " ****************" << std::endl;

            // Increase the precision for the output
            NR_COUT << std::fixed << std::setprecision(10);

            // Check the results
            const auto transGradCpuPtr = transGradCpu.data();
            const auto transGradCudaPtr = transGradCuda.data();
            for (size_t i = 0; i < transGradCpu.nVoxels(); i++) {
                const float transGradCpuVal = transGradCpuPtr[i];
                const float transGradCudaVal = transGradCudaPtr[i];
                const float diff = abs(transGradCpuVal - transGradCudaVal);
                if (diff > 0) {
                    NR_COUT << "[i]=" << i;
                    NR_COUT << " | diff=" << diff;
                    NR_COUT << " | CPU=" << transGradCpuVal;
                    NR_COUT << " | CUDA=" << transGradCudaVal << std::endl;
                }
                REQUIRE(diff == 0);
            }
        }
    }
}

namespace {

constexpr int kSquaringSteps = 6;

// An image with smooth, non-degenerate intensity content, optionally on anisotropic voxels with a
// shifted origin as a downsampled pyramid level has
NiftiImage MakeVelImage(bool is3D, float phase, bool anisotropic) {
    const NiftiImage::dim_t size = 12;
    std::vector<NiftiImage::dim_t> dims(is3D ? 3 : 2, size);
    NiftiImage img(dims, NIFTI_TYPE_FLOAT32);
    mat44 mat;
    Mat44Eye(&mat);
    if (anisotropic) {
        mat.m[0][0] = 1.7f; mat.m[1][1] = 2.3f; mat.m[2][2] = is3D ? 1.1f : 1.f;
        mat.m[0][3] = -3.5f; mat.m[1][3] = 2.25f; mat.m[2][3] = is3D ? -1.75f : 0.f;
    }
    setSform(img, mat);
    img->dx = img->pixdim[1] = mat.m[0][0];
    img->dy = img->pixdim[2] = mat.m[1][1];
    img->dz = img->pixdim[3] = is3D ? mat.m[2][2] : 1.f;
    const int nx = img->nx, ny = img->ny, nz = img->nz;
    auto ptr = img.data();
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const double x = 2.0 * M_PI * i / nx, y = 2.0 * M_PI * j / ny;
                const double z = nz > 1 ? 2.0 * M_PI * k / nz : 0.0;
                ptr[(static_cast<size_t>(k) * ny + j) * nx + i] =
                    static_cast<float>(100.0 * (1.5 + std::sin(x + phase) * std::cos(y) * std::cos(z)));
            }
    return img;
}

// A grid carrying a smooth, small stationary velocity, as -vel optimises
NiftiImage MakeVelocityGrid(const NiftiImage& reference, float spacingInVoxels) {
    NiftiImage grid;
    const float gridSpacing[3]{ reference->dx * spacingInVoxels, reference->dy * spacingInVoxels,
                                reference->dz * spacingInVoxels };
    reg_createControlPointGrid<float>(grid, reference, gridSpacing);
    grid->intent_p1 = SPLINE_VEL_GRID;
    grid->intent_p2 = kSquaringSteps;
    const size_t volume = grid.nVoxelsPerVolume();
    const int nx = grid->nx, ny = grid->ny, nz = grid->nz;
    const int components = nz > 1 ? 3 : 2;
    auto ptr = grid.data();
    for (int c = 0; c < components; ++c)
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    const size_t index = c * volume + (static_cast<size_t>(k) * ny + j) * nx + i;
                    const double x = 2.0 * M_PI * i / nx, y = 2.0 * M_PI * j / ny;
                    const double z = nz > 1 ? 2.0 * M_PI * k / nz : 0.0;
                    ptr[index] = static_cast<float>(ptr[index]) +
                        0.3f * static_cast<float>(std::sin(x + c) * std::cos(y) * (nz > 1 ? std::cos(z) : 1.0));
                }
    return grid;
}

struct Backends {
    Platform platformCpu{ PlatformType::Cpu };
    Platform platformCuda{ PlatformType::Cuda };
    NiftiImage refCpu, floCpu, gridCpu, refCuda, floCuda, gridCuda;
    unique_ptr<F3dContent> contentCpu, contentCuda;
    unique_ptr<Compute> computeCpu, computeCuda;

    Backends(const NiftiImage& reference, const NiftiImage& floating, const NiftiImage& grid):
        refCpu(reference), floCpu(floating), gridCpu(grid),
        refCuda(reference), floCuda(floating), gridCuda(grid) {
        contentCpu.reset(new F3dContent(refCpu, floCpu, gridCpu));
        contentCuda.reset(new CudaF3dContent(refCuda, floCuda, gridCuda));
        computeCpu.reset(platformCpu.CreateCompute(*contentCpu));
        computeCuda.reset(platformCuda.CreateCompute(*contentCuda));
    }
};

} // namespace

TEST_CASE("Regression Velocity Field Exponentiation", "[regression]") {
    /*
        GetDefFieldFromVelocityGrid: the scaling-and-squaring integration that turns the stationary
        velocity grid into a deformation field. It runs on every objective evaluation of a -vel
        registration, forwards and backwards, so the two backends have to agree on it exactly before
        any comparison further down the iteration means anything.

        Grid spacings are swept because the number of nodes contributing to each voxel, and hence the
        arrangement of the sum, changes with them; the anisotropic geometry is included because an
        identity sform puts every sub-voxel offset on an exact binary fraction, where two orderings of
        the same products cannot disagree.
    */
    for (const bool is3D : { false, true })
        for (const auto& [label, spacing, anisotropic] : { std::tuple{ "grid 2 voxels", 2.f, false },
                                                           std::tuple{ "grid 2 voxels, anisotropic", 2.f, true },
                                                           std::tuple{ "grid 1 voxel (as refined)", 1.f, false } }) {
            const std::string name = std::string(is3D ? "3D" : "2D") + ", " + label;
            SECTION(name) {
                const NiftiImage reference = MakeVelImage(is3D, 0.f, anisotropic);
                const NiftiImage floating = MakeVelImage(is3D, 0.6f, anisotropic);
                Backends b(reference, floating, MakeVelocityGrid(reference, spacing));

                b.computeCpu->GetDefFieldFromVelocityGrid(false);
                b.computeCuda->GetDefFieldFromVelocityGrid(false);

                // A deformation field holds positions, so an all-zero one means nothing ran
                RequireNonZero(b.contentCpu->GetDeformationField(), "the CPU deformation field");
                RequireNonZero(b.contentCuda->GetDeformationField(), "the CUDA deformation field");

                const Deviation deviation = CompareImages(b.contentCpu->GetDeformationField(),
                                                          b.contentCuda->GetDeformationField());
                ReportDeviation(name, deviation);
                INFO(name << ": " << deviation.differing << " values differ, max " << deviation.max);
                REQUIRE(deviation.max == 0);
            }
        }
}

TEST_CASE("Regression Velocity Field Step Number Update", "[regression]") {
    /*
        The same integration with updateStepNumber set, which reg_f3d2 passes on the forward pass
        (in reg_f3d2::GetDeformationField). That derives the number of squaring steps from the flow field's
        extrema and writes it back to the grid's intent_p2, so the step count is data-dependent rather
        than fixed. Both backends have to derive the same one: a difference of a single step halves or
        doubles the integration interval, and every deformation field afterwards differs by far more
        than rounding.
    */
    for (const bool is3D : { false, true }) {
        SECTION(is3D ? "3D" : "2D") {
            const NiftiImage reference = MakeVelImage(is3D, 0.f, false);
            const NiftiImage floating = MakeVelImage(is3D, 0.6f, false);
            Backends b(reference, floating, MakeVelocityGrid(reference, 2.f));

            b.computeCpu->GetDefFieldFromVelocityGrid(true);
            b.computeCuda->GetDefFieldFromVelocityGrid(true);

            const float stepsCpu = b.contentCpu->GetControlPointGrid()->intent_p2;
            const float stepsCuda = b.contentCuda->GetControlPointGrid()->intent_p2;
            NR_COUT << "  derived squaring steps: CPU = " << stepsCpu << ", CUDA = " << stepsCuda << std::endl;
            INFO("CPU derived " << stepsCpu << " squaring steps, CUDA derived " << stepsCuda);
            REQUIRE(stepsCpu == stepsCuda);
            // And it has to have been derived, not left at the value the grid was built with
            REQUIRE(stepsCpu != 0);

            const Deviation deviation = CompareImages(b.contentCpu->GetDeformationField(),
                                                      b.contentCuda->GetDeformationField());
            ReportDeviation(std::string(is3D ? "3D" : "2D") + ", step number update", deviation);
            REQUIRE(deviation.max == 0);
        }
    }
}
