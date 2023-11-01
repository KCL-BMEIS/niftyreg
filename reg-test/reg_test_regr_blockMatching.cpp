#include "reg_test_common.h"
#include "_reg_blockMatching.h"
#include "CpuBlockMatchingKernel.h"
#include "CudaBlockMatchingKernel.h"

/**
 *  Block matching regression test to ensure the CPU and CUDA versions yield the same output
 */

class BMTest {
protected:
    using TestData = std::tuple<std::string, NiftiImage, NiftiImage, int>;
    using TestCase = std::tuple<std::string, unique_ptr<_reg_blockMatchingParam>, unique_ptr<_reg_blockMatchingParam>>;

    inline static vector<TestCase> testCases;

public:
    BMTest() {
        if (!testCases.empty())
            return;

        std::mt19937 gen(0);
        std::uniform_real_distribution<float> distr(0, 1);

        // Create a reference and floating 2D images
        constexpr NiftiImage::dim_t size = 128;
        vector<NiftiImage::dim_t> dim{ size, size };
        NiftiImage reference2d(dim, NIFTI_TYPE_FLOAT32);
        NiftiImage floating2d(dim, NIFTI_TYPE_FLOAT32);

        // Create a reference 3D image
        dim.push_back(size);
        NiftiImage reference3d(dim, NIFTI_TYPE_FLOAT32);
        NiftiImage floating3d(dim, NIFTI_TYPE_FLOAT32);

        // Fill images with random values
        auto ref2dPtr = reference2d.data();
        auto flo2dPtr = floating2d.data();
        for (size_t i = 0; i < reference2d.nVoxels(); ++i) {
            ref2dPtr[i] = distr(gen);
            flo2dPtr[i] = distr(gen);
        }

        // Fill images with random values
        auto ref3dPtr = reference3d.data();
        auto flo3dPtr = floating3d.data();
        for (size_t i = 0; i < reference3d.nVoxels(); ++i) {
            ref3dPtr[i] = distr(gen);
            flo3dPtr[i] = distr(gen);
        }

        // Create the data container for the regression test
        vector<TestData> testData;
        for (int b = 50; b <= 100; b += 50) {
            testData.emplace_back(TestData(
                "BlockMatching 2D block " + std::to_string(b),
                reference2d,
                floating2d,
                b
            ));
            testData.emplace_back(TestData(
                "BlockMatching 3D block " + std::to_string(b),
                reference3d,
                floating3d,
                b
            ));
        }

        for (auto&& data : testData) {
            // Get the test data
            auto&& [testName, reference, floating, block] = data;

            // Create images
            NiftiImage referenceCpu(reference), referenceCuda(reference);
            NiftiImage floatingCpu(floating), floatingCuda(floating);
            NiftiImage warpedCpu(floating), warpedCuda(floating);

            // Create the contents
            unique_ptr<AladinContent> contentCpu{ new AladinContent(
                referenceCpu,
                floatingCpu,
                nullptr,
                nullptr,
                sizeof(float),
                100,
                block,
                1
            ) };
            unique_ptr<AladinContent> contentCuda{ new CudaAladinContent(
                referenceCuda,
                floatingCuda,
                nullptr,
                nullptr,
                sizeof(float),
                100,
                block,
                1
            ) };

            // Initialise the warped images
            contentCpu->SetWarped(warpedCpu.disown());
            contentCuda->SetWarped(warpedCuda.disown());

            // Initialise the block matching
            unique_ptr<BlockMatchingKernel> kernelCpu{ new CpuBlockMatchingKernel(contentCpu.get()) };
            unique_ptr<BlockMatchingKernel> kernelCuda{ new CudaBlockMatchingKernel(contentCuda.get()) };

            // Do the computation
            kernelCpu->Calculate();
            kernelCuda->Calculate();

            // Retrieve the information
            unique_ptr<_reg_blockMatchingParam> blockMatchingParamsCpu{ new _reg_blockMatchingParam(contentCpu->GetBlockMatchingParams()) };
            unique_ptr<_reg_blockMatchingParam> blockMatchingParamsCuda{ new _reg_blockMatchingParam(contentCuda->GetBlockMatchingParams()) };

            testCases.push_back({ testName, std::move(blockMatchingParamsCpu), std::move(blockMatchingParamsCuda) });
        }
    }
};

TEST_CASE_METHOD(BMTest, "Regression Block Matching", "[regression]") {
    // Loop over all generated test cases
    for (auto&& testCase : this->testCases) {
        // Retrieve test information
        auto&& [testName, blockMatchingParamsCpu, blockMatchingParamsCuda] = testCase;

        SECTION(testName) {
            NR_COUT << "\n**************** Section " << testName << " ****************" << std::endl;

            // Increase the precision for the output
            NR_COUT << std::fixed << std::setprecision(10);

            // Ensure both approaches retrieve the same number of voxels
            REQUIRE(blockMatchingParamsCpu->activeBlockNumber == blockMatchingParamsCuda->activeBlockNumber);

            // Loop over the block and ensure all values are identical
            for (int b = 0; b < blockMatchingParamsCpu->activeBlockNumber; ++b) {
                for (int d = 0; d < (int)blockMatchingParamsCpu->dim; ++d) {
                    const int i = b * (int)blockMatchingParamsCpu->dim + d;
                    const auto refPosCpu = blockMatchingParamsCpu->referencePosition[i];
                    const auto refPosCuda = blockMatchingParamsCuda->referencePosition[i];
                    auto diff = abs(refPosCpu - refPosCuda);
                    if (diff > 0) {
                        NR_COUT << "Ref[" << b << "/" << blockMatchingParamsCpu->activeBlockNumber << ":" << d << "] CPU:";
                        NR_COUT << refPosCpu << " | CUDA:" << refPosCuda << std::endl;
                    }
                    REQUIRE(diff == 0);
                    const auto warPosCpu = blockMatchingParamsCpu->warpedPosition[i];
                    const auto warPosCuda = blockMatchingParamsCuda->warpedPosition[i];
                    diff = abs(warPosCpu - warPosCuda);
                    if (diff > 0) {
                        NR_COUT << "War[" << b << "/" << blockMatchingParamsCpu->activeBlockNumber << ":" << d << "] CPU:";
                        NR_COUT << warPosCpu << " | CUDA:" << warPosCuda << std::endl;
                    }
                    REQUIRE(diff == 0);
                }
            }
        }
    }
}
