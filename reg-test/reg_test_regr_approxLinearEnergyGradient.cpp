#include "reg_test_common.h"
#include "_reg_nmi.h"
#include "CudaF3dContent.h"

/**
 *  Approximate linear energy and approximate linear energy gradient regression tests
 *  to ensure the CPU and CUDA versions yield the same output
**/

class ApproxLinearEnergyGradientTest {
protected:
    using TestData = std::tuple<std::string, NiftiImage&, NiftiImage&, NiftiImage&, float>;
    using TestCase = std::tuple<std::string, double, double, NiftiImage, NiftiImage>;

    inline static vector<TestCase> testCases;

public:
    ApproxLinearEnergyGradientTest() {
        if (!testCases.empty())
            return;

        // Create a random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> distr(0, 10);

        // Create 2D reference, floating and control point grid images
        constexpr NiftiImage::dim_t size = 4;
        vector<NiftiImage::dim_t> dim{ size, size };
        NiftiImage reference2d(dim, NIFTI_TYPE_FLOAT32);
        NiftiImage floating2d(dim, NIFTI_TYPE_FLOAT32);
        NiftiImage controlPointGrid = CreateControlPointGrid(reference2d);
        NiftiImage controlPointGrid2d[3]{ controlPointGrid, controlPointGrid, controlPointGrid };

        // Create 3D reference, floating and control point grid images
        dim.push_back(size);
        NiftiImage reference3d(dim, NIFTI_TYPE_FLOAT32);
        NiftiImage floating3d(dim, NIFTI_TYPE_FLOAT32);
        controlPointGrid = CreateControlPointGrid(reference3d);
        NiftiImage controlPointGrid3d[3]{ controlPointGrid, controlPointGrid, controlPointGrid };

        // Fill control point grids with random values
        for (int i = 0; i < 3; i++) {
            auto controlPointGridPtr = controlPointGrid2d[i].data();
            for (size_t j = 0; j < controlPointGrid2d[i].nVoxels(); j++)
                controlPointGridPtr[j] = distr(gen);
            controlPointGridPtr = controlPointGrid3d[i].data();
            for (size_t j = 0; j < controlPointGrid3d[i].nVoxels(); j++)
                controlPointGridPtr[j] = distr(gen);
        }

        // Create the data container for the regression test
        vector<TestData> testData;
        for (int i = 0; i < 3; i++) {
            const float weight = distr(gen);
            testData.emplace_back(TestData(
                "2D weight: "s + std::to_string(weight),
                reference2d,
                floating2d,
                controlPointGrid2d[i],
                weight
            ));
            testData.emplace_back(TestData(
                "3D weight: "s + std::to_string(weight),
                reference3d,
                floating3d,
                controlPointGrid3d[i],
                weight
            ));
        }

        // Create the platforms
        Platform platformCpu(PlatformType::Cpu);
        Platform platformCuda(PlatformType::Cuda);

        for (auto&& testData : testData) {
            // Get the test data
            auto&& [testName, reference, floating, controlPointGrid, weight] = testData;

            // Create images
            NiftiImage referenceCpu(reference), referenceCuda(reference);
            NiftiImage floatingCpu(floating), floatingCuda(floating);
            NiftiImage controlPointGridCpu(controlPointGrid), controlPointGridCuda(controlPointGrid);

            // Create the contents
            unique_ptr<F3dContent> contentCpu{ new F3dContent(
                referenceCpu,
                floatingCpu,
                controlPointGridCpu,
                nullptr,
                nullptr,
                nullptr,
                sizeof(float)
            ) };
            unique_ptr<F3dContent> contentCuda{ new CudaF3dContent(
                referenceCuda,
                floatingCuda,
                controlPointGridCuda,
                nullptr,
                nullptr,
                nullptr,
                sizeof(float)
            ) };

            // Create the computes
            unique_ptr<Compute> computeCpu{ platformCpu.CreateCompute(*contentCpu) };
            unique_ptr<Compute> computeCuda{ platformCuda.CreateCompute(*contentCuda) };

            // Compute the approximate linear energy for CPU and CUDA
            const double approxLinearEnergyCpu = computeCpu->ApproxLinearEnergy();
            const double approxLinearEnergyCuda = computeCuda->ApproxLinearEnergy();

            // Compute the approximate linear energy gradient for CPU and CUDA
            computeCpu->ApproxLinearEnergyGradient(weight);
            computeCuda->ApproxLinearEnergyGradient(weight);

            // Get the transformation gradients
            NiftiImage transGradCpu(contentCpu->GetTransformationGradient(), NiftiImage::Copy::Image);
            NiftiImage transGradCuda(contentCuda->GetTransformationGradient(), NiftiImage::Copy::Image);

            // Save for testing
            testCases.push_back({ testName, approxLinearEnergyCpu, approxLinearEnergyCuda, std::move(transGradCpu), std::move(transGradCuda) });
        }
    }
};

TEST_CASE_METHOD(ApproxLinearEnergyGradientTest, "Regression Approximate Linear Energy Gradient", "[regression]") {
    // Loop over all generated test cases
    for (auto&& testCase : testCases) {
        // Retrieve test information
        auto&& [testName, approxLinearEnergyCpu, approxLinearEnergyCuda, transGradCpu, transGradCuda] = testCase;

        SECTION(testName) {
            NR_COUT << "\n**************** Section " << testName << " ****************" << std::endl;

            // Increase the precision for the output
            NR_COUT << std::fixed << std::setprecision(10);

            // Check the approximate linear energy
            NR_COUT << "Approx Linear Energy: " << approxLinearEnergyCpu << " " << approxLinearEnergyCuda << std::endl;
            REQUIRE(abs(approxLinearEnergyCpu - approxLinearEnergyCuda) < EPS);

            // Check the transformation gradients
            const auto transGradCpuPtr = transGradCpu.data();
            const auto transGradCudaPtr = transGradCuda.data();
            for (size_t i = 0; i < transGradCpu.nVoxels(); ++i) {
                const float cpuVal = transGradCpuPtr[i];
                const float cudaVal = transGradCudaPtr[i];
                const auto diff = abs(cpuVal - cudaVal);
                if (diff > EPS)
                    NR_COUT << i << " " << cpuVal << " " << cudaVal << std::endl;
                REQUIRE(diff < EPS);
            }
        }
    }
}
