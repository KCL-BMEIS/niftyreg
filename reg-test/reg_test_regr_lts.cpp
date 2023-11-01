#include "reg_test_common.h"
#include "_reg_blockMatching.h"
#include "CpuBlockMatchingKernel.h"

#include "LtsKernel.h"
#include "CpuLtsKernel.h"
#include "CudaLtsKernel.h"

/**
 *  LTS regression test to ensure the CPU and CUDA versions yield the same output
 */

class LtsTest {
protected:
    using TestData = std::tuple<std::string, NiftiImage, NiftiImage, int, int>;
    using TestCase = std::tuple<std::string, unique_ptr<mat44>, unique_ptr<mat44>>;

    inline static vector<TestCase> testCases;

public:
    LtsTest() {
        if (!testCases.empty())
            return;

        // Create a random number generator
        std::mt19937 gen(0);
        std::uniform_real_distribution<float> distr(0, 1);

        // Create a reference and floating 2D images
        constexpr NiftiImage::dim_t size = 64;
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
        for (int t = 0; t <= 1; ++t) {
            for (int i = 20; i <= 100; i += 20) {
                testData.emplace_back(TestData(
                    "BlockMatching 2D type " + std::to_string(t) + " inlier " + std::to_string(i),
                    reference2d,
                    floating2d,
                    t,
                    i
                ));
                testData.emplace_back(TestData(
                    "BlockMatching 3D type " + std::to_string(t) + " inlier " + std::to_string(i),
                    reference3d,
                    floating3d,
                    t,
                    i
                ));
            }
        }

        for (auto&& data : testData) {
            // Get the test data
            auto&& [testName, reference, floating, ttype, inlier] = data;

            // Create identity transformations
            unique_ptr<mat44> matCpu{ new mat44 }; reg_mat44_eye(matCpu.get());
            unique_ptr<mat44> matCuda{ new mat44 }; reg_mat44_eye(matCuda.get());

            // Create images
            NiftiImage referenceCpu(reference), referenceCuda(reference);
            NiftiImage floatingCpu(floating), floatingCuda(floating);
            NiftiImage warpedCpu(floating), warpedCuda(floating);

            // Create the contents
            unique_ptr<AladinContent> contentCpu{ new AladinContent(
                referenceCpu,
                floatingCpu,
                nullptr,
                matCpu.get(),
                sizeof(float),
                inlier,
                100,
                1
            ) };
            unique_ptr<AladinContent> contentCuda{ new CudaAladinContent(
                referenceCuda,
                floatingCuda,
                nullptr,
                matCuda.get(),
                sizeof(float),
                inlier,
                100,
                1
            ) };

            // Initialise the warped images
            contentCpu->SetWarped(warpedCpu.disown());
            contentCuda->SetWarped(warpedCuda.disown());

            // Initialise the block matching and run it on the CPU
            unique_ptr<BlockMatchingKernel> bmKernelCpu{ new CpuBlockMatchingKernel(contentCpu.get()) };
            bmKernelCpu->Calculate();

            // Set the CUDA block matching parameters
            _reg_blockMatchingParam *blockMatchingParamsCuda = new _reg_blockMatchingParam(contentCpu->GetBlockMatchingParams());
            contentCuda->SetBlockMatchingParams(blockMatchingParamsCuda);

            // Initialise the optimise kernels
            unique_ptr<LtsKernel> kernelCpu{ new CpuLtsKernel(contentCpu.get()) };
            unique_ptr<LtsKernel> kernelCuda{ new CudaLtsKernel(contentCuda.get()) };

            // Compute the transformations
            kernelCpu->Calculate(ttype);
            kernelCuda->Calculate(ttype);

            // Save the matrices for testing
            testCases.push_back({ testName, std::move(matCpu), std::move(matCuda) });
        }
    }
};

TEST_CASE_METHOD(LtsTest, "Regression LTS", "[regression]") {
    // Loop over all generated test cases
    for (auto&& testCase : this->testCases) {
        // Retrieve test information
        auto&& [testName, matCpu, matCuda] = testCase;

        SECTION(testName) {
            NR_COUT << "\n**************** Section " << testName << " ****************" << std::endl;

            // Increase the precision for the output
            NR_COUT << std::fixed << std::setprecision(10);

            // Loop over the matrix values and ensure they are identical
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    const auto mCpu = matCpu->m[i][j];
                    const auto mCuda = matCuda->m[i][j];
                    const auto diff = abs(mCpu - mCuda);
                    if (diff > 0)
                        NR_COUT << i << " " << j << " " << mCpu << " " << mCuda << std::endl;
                    REQUIRE(diff == 0);
                }
            }
        }
    }
}
