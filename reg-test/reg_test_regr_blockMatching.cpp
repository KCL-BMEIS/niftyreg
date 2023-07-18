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

        // Create a random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
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
        const auto ref2dPtr = reference2d.data();
        auto ref2dItr = ref2dPtr.begin();
        const auto flo2dPtr = floating2d.data();
        auto flo2dItr = flo2dPtr.begin();
        for (int y = 0; y < reference2d->ny; ++y)
            for (int x = 0; x < reference2d->nx; ++x) {
                *ref2dItr++ = distr(gen);
                *flo2dItr++ = distr(gen);
            }

        // Fill images with random values
        const auto ref3dPtr = reference3d.data();
        auto ref3dItr = ref3dPtr.begin();
        const auto flo3dPtr = floating3d.data();
        auto flo3dItr = flo3dPtr.begin();
        for (int z = 0; z < reference3d->nz; ++z)
            for (int y = 0; y < reference3d->ny; ++y)
                for (int x = 0; x < reference3d->nx; ++x) {
                    *ref3dItr++ = distr(gen);
                    *flo3dItr++ = distr(gen);
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
            std::unique_ptr<BlockMatchingKernel> kernelCpu{ new CpuBlockMatchingKernel(contentCpu.get()) };
            std::unique_ptr<BlockMatchingKernel> kernelCuda{ new CudaBlockMatchingKernel(contentCuda.get()) };

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

TEST_CASE_METHOD(BMTest, "Regression BlockMatching", "[regression]") {
    // Loop over all generated test cases
    for (auto&& testCase : this->testCases) {
        // Retrieve test information
        auto&& [testName, blockMatchingParamsCpu, blockMatchingParamsCuda] = testCase;

        SECTION(testName) {
            std::cout << "\n**************** Section " << testName << " ****************" << std::endl;

            // Ensure both approaches retrieve the same number of voxels
            REQUIRE(blockMatchingParamsCpu->activeBlockNumber == blockMatchingParamsCuda->activeBlockNumber);

            // Loop over the block and ensure all values are identical
            for (int b = 0; b < blockMatchingParamsCpu->activeBlockNumber; ++b) {
                for(int d = 0; d<(int)blockMatchingParamsCpu->dim; ++d){

                    const int i = b*(int)blockMatchingParamsCpu->dim+d;
                    const auto refPosCpu = blockMatchingParamsCpu->referencePosition[i];
                    const auto refPosCuda = blockMatchingParamsCuda->referencePosition[i];
                    if(fabs(refPosCpu - refPosCuda) > EPS){
                        std::cout << "Ref[" << b << "/" << blockMatchingParamsCpu->activeBlockNumber << ":" << d << "] CPU:";
                        std::cout << refPosCpu << " | CUDA:" << refPosCuda << std::endl;
                        std::cout.flush();
                    }
                    REQUIRE(fabs(refPosCpu - refPosCuda) < EPS);
                    const auto warPosCpu = blockMatchingParamsCpu->warpedPosition[i];
                    const auto warPosCuda = blockMatchingParamsCuda->warpedPosition[i];
                    if(fabs(warPosCpu - warPosCuda) > EPS){
                        std::cout << "War[" << b << "/" << blockMatchingParamsCpu->activeBlockNumber << ":" << d << "] CPU:";
                        std::cout << warPosCpu << " | CUDA:" << warPosCuda << std::endl;
                        std::cout.flush();
                    }
                    REQUIRE(fabs(warPosCpu - warPosCuda) < EPS);
                }
            }
        }
    }
}
