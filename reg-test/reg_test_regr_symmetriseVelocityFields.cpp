#include "reg_test_common.h"
#include "CudaF3dContent.h"

/**
 *  Symmetrise velocity fields regression test to ensure the CPU and CUDA versions yield the same output
**/

class SymmetriseVelocityFieldsTest {
protected:
    using TestData = std::tuple<std::string, NiftiImage, NiftiImage, NiftiImage>;
    using TestCase = std::tuple<std::string, NiftiImage, NiftiImage, NiftiImage, NiftiImage>;

    inline static vector<TestCase> testCases;

public:
    SymmetriseVelocityFieldsTest() {
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
        NiftiImage controlPointGridBw2d = CreateControlPointGrid(reference2d);
        NiftiImage controlPointGrid3d = CreateControlPointGrid(reference3d);
        NiftiImage controlPointGridBw3d = CreateControlPointGrid(reference3d);

        // Add random values to the control point grid coefficients
        // No += or + operator for RNifti::NiftiImageData:Element
        // so reverting to old school for now
        float *cpp2dPtr = static_cast<float*>(controlPointGrid2d->data);
        float *cpp2dBwPtr = static_cast<float*>(controlPointGridBw2d->data);
        float *cpp3dPtr = static_cast<float*>(controlPointGrid3d->data);
        float *cpp3dBwPtr = static_cast<float*>(controlPointGridBw3d->data);
        for (size_t i = 0; i < controlPointGrid2d.nVoxels(); ++i) {
            cpp2dPtr[i] += distr(gen);
            cpp2dBwPtr[i] += distr(gen);
        }
        for (size_t i = 0; i < controlPointGrid3d.nVoxels(); ++i) {
            cpp3dPtr[i] += distr(gen);
            cpp3dBwPtr[i] += distr(gen);
        }

        // Create the affine matrices and fill them with random values
        std::array<mat44, 2> matrices{};
        for (int i = 0; i < matrices.size(); ++i)
            for (int j = 0; j < 4; ++j)
                for (int k = 0; k < 4; ++k)
                    matrices[i].m[j][k] = j == k ? distr(gen) : 0;

        // Add the test data
        vector<TestData> testData;
        testData.emplace_back(TestData(
            "2D",
            std::move(reference2d),
            std::move(controlPointGrid2d),
            std::move(controlPointGridBw2d)
        ));
        testData.emplace_back(TestData(
            "3D",
            std::move(reference3d),
            std::move(controlPointGrid3d),
            std::move(controlPointGridBw3d)
        ));

        // Create the platforms
        Platform platformCpu(PlatformType::Cpu);
        Platform platformCuda(PlatformType::Cuda);

        for (auto&& testData : testData) {
            // Make a copy of the test data
            auto [testName, reference, controlPointGrid, controlPointGridBw] = testData;

            // Set the affine matrices
            controlPointGrid->sform_code = 0;
            controlPointGrid->qto_xyz = matrices[0];
            controlPointGridBw->sform_code = 1;
            controlPointGridBw->sto_xyz = matrices[1];

            // Create images
            NiftiImage referenceCpu(reference), referenceCuda(reference);
            NiftiImage cppCpu(controlPointGrid), cppCuda(controlPointGrid);
            NiftiImage cppBwCpu(controlPointGrid), cppBwCuda(controlPointGrid);

            // Create the content
            unique_ptr<F3dContent> contentCpu{ new F3dContent(referenceCpu, referenceCpu, cppCpu) };
            unique_ptr<F3dContent> contentBwCpu{ new F3dContent(referenceCpu, referenceCpu, cppBwCpu) };
            unique_ptr<F3dContent> contentCuda{ new CudaF3dContent(referenceCuda, referenceCuda, cppCuda) };
            unique_ptr<F3dContent> contentBwCuda{ new CudaF3dContent(referenceCuda, referenceCuda, cppBwCuda) };

            // Create the computes
            unique_ptr<Compute> computeCpu{ platformCpu.CreateCompute(*contentCpu) };
            unique_ptr<Compute> computeCuda{ platformCuda.CreateCompute(*contentCuda) };

            // Symmetrise the velocity fields
            computeCpu->SymmetriseVelocityFields(*contentBwCpu);
            computeCuda->SymmetriseVelocityFields(*contentBwCuda);

            // Get the results of CUDA since CPU results are already inplace
            contentCuda->GetControlPointGrid();
            contentBwCuda->GetControlPointGrid();

            // Save for testing
            testCases.push_back({ testName, std::move(cppCpu), std::move(cppBwCpu), std::move(cppCuda), std::move(cppBwCuda) });
        }
    }
};

TEST_CASE_METHOD(SymmetriseVelocityFieldsTest, "Regression Symmetrise Velocity Fields", "[regression]") {
    // Loop over all generated test cases
    for (auto&& testCase : testCases) {
        // Retrieve test information
        auto&& [sectionName, cppCpu, cppBwCpu, cppCuda, cppBwCuda] = testCase;

        SECTION(sectionName) {
            NR_COUT << "\n**************** Section " << sectionName << " ****************" << std::endl;

            // Increase the precision for the output
            NR_COUT << std::fixed << std::setprecision(10);

            // Check the results
            const auto cppCpuPtr = cppCpu.data();
            const auto cppBwCpuPtr = cppBwCpu.data();
            const auto cppCudaPtr = cppCuda.data();
            const auto cppBwCudaPtr = cppBwCuda.data();
            for (size_t i = 0; i < cppCpu.nVoxels(); i++) {
                const float cppCpuVal = cppCpuPtr[i];
                const float cppCudaVal = cppCudaPtr[i];
                const float diff = abs(cppCpuVal - cppCudaVal);
                if (diff > 0) {
                    NR_COUT << "[i]=" << i;
                    NR_COUT << " | diff=" << diff;
                    NR_COUT << " | CPU=" << cppCpuVal;
                    NR_COUT << " | CUDA=" << cppCudaVal << std::endl;
                }
                REQUIRE(diff == 0);
                // Check the results of the backwards
                const float cppBwCpuVal = cppBwCpuPtr[i];
                const float cppBwCudaVal = cppBwCudaPtr[i];
                const float diffBw = abs(cppBwCpuVal - cppBwCudaVal);
                if (diffBw > 0) {
                    NR_COUT << "[i]=" << i;
                    NR_COUT << " | diffBw=" << diffBw;
                    NR_COUT << " | CPU=" << cppBwCpuVal;
                    NR_COUT << " | CUDA=" << cppBwCudaVal << std::endl;
                }
                REQUIRE(diffBw == 0);
            }
        }
    }
}
