#include "reg_test_common.h"
#include "_reg_blockMatching.h"
#include "CpuBlockMatchingKernel.h"

#include "OptimiseKernel.h"
#include "CpuOptimiseKernel.h"
#include "CudaOptimiseKernel.h"

/*
    This test file contains a regression test to ensure the CPU and GPU version yield the same output
*/

class LTSTest {
    /*
    Class to call the LTS function
    */
protected:
    using TestData = std::tuple<std::string, NiftiImage, NiftiImage, int, int>;
    using TestCase = std::tuple<std::string, mat44 *, mat44 *>;
    inline static vector<TestCase> testCases;
    NiftiImage reference2d;
    NiftiImage floating2d;
    NiftiImage reference3d;
    NiftiImage floating3d;
public:
    ~LTSTest() {
        std::cout << "Calling destructor" << std::endl;
    }
    LTSTest() {
        std::cout << "Calling constructor" << std::endl;
        if (!testCases.empty())
            return;

        // Create a random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> distr(0, 1);

        // Create a reference and floating 2D images
        NiftiImage::dim_t size = 64;
        vector<NiftiImage::dim_t> dim{ size, size };
        this->reference2d = NiftiImage(dim, NIFTI_TYPE_FLOAT32);
        this->floating2d = NiftiImage(dim, NIFTI_TYPE_FLOAT32);

        // Create a reference 3D image
        dim.push_back(size);
        this->reference3d = NiftiImage(dim, NIFTI_TYPE_FLOAT32);
        this->floating3d = NiftiImage(dim, NIFTI_TYPE_FLOAT32);

        // Fill images with random values
        float *ref2dPtr = static_cast<float *>(reference2d->data);
        float *flo2dPtr = static_cast<float *>(floating2d->data);
        for (int y = 0; y < reference2d->ny; ++y)
            for (int x = 0; x < reference2d->nx; ++x) {
                *ref2dPtr++ = distr(gen);
                *flo2dPtr++ = distr(gen);
            }

        // Fill images with random values
        float *ref3dPtr = static_cast<float *>(reference3d->data);
        float *flo3dPtr = static_cast<float *>(floating3d->data);
        for (int z = 0; z < reference3d->nz; ++z)
            for (int y = 0; y < reference3d->ny; ++y)
                for (int x = 0; x < reference3d->nx; ++x) {
                    *ref3dPtr++ = distr(gen);
                    *flo3dPtr++ = distr(gen);
                }


        // Create the data container for the regression test
        vector<TestData> testData;
        for(int t=0; t<=1; ++t){
            for(int i=20; i<=100; i+=20){
                testData.emplace_back(TestData(
                    "BlockMatching 2D type " + std::to_string(t) + " inlier " + std::to_string(i),
                    std::move(NiftiImage(this->reference2d)),
                    std::move(NiftiImage(this->floating2d)),
                    t,
                    i
                ));
                testData.emplace_back(TestData(
                    "BlockMatching 3D type " + std::to_string(t) + " inlier " + std::to_string(i),
                    std::move(NiftiImage(this->reference3d)),
                    std::move(NiftiImage(this->floating3d)),
                    t,
                    i
                ));
            }
        }

        for (auto&& data : testData) {
            unique_ptr<Platform> platformCPU{ new Platform(PlatformType::Cpu) };
            unique_ptr<Platform> platformCUDA{ new Platform(PlatformType::Cuda) };
            // Make a copy of the test data
            auto&& [testName, reference, floating, ttype, inlier] = data;
            // Create content creator
            unique_ptr<AladinContentCreator> contentCreatorCPU{
                dynamic_cast<AladinContentCreator*>(platformCPU->CreateContentCreator(ContentType::Aladin))
            };
            unique_ptr<AladinContentCreator> contentCreatorCUDA{
                dynamic_cast<AladinContentCreator*>(platformCUDA->CreateContentCreator(ContentType::Aladin))
            };
            // Create identity transformations
            mat44 *matCPU = new mat44; reg_mat44_eye(matCPU);
            mat44 *matCUDA = new mat44; reg_mat44_eye(matCUDA);
            // Create the contents
            unique_ptr<AladinContent> contentCPU{ contentCreatorCPU->Create(
                NiftiImage(reference).disown(),
                NiftiImage(floating).disown(),
                nullptr,
                matCPU,
                sizeof(float),
                inlier,
                100,
                1
            )};
            unique_ptr<AladinContent> contentCUDA{ contentCreatorCUDA->Create(
                NiftiImage(reference).disown(),
                NiftiImage(floating).disown(),
                nullptr,
                matCUDA,
                sizeof(float),
                inlier,
                100,
                1
            )};
            // Initialise the warped image
            contentCPU->SetWarped(NiftiImage(floating).disown());
            contentCUDA->SetWarped(NiftiImage(floating).disown());

            // Initialise the block matching and run it on the CPU
            std::unique_ptr<Kernel> BMKernelCPU = nullptr;
            BMKernelCPU.reset(platformCPU->CreateKernel(BlockMatchingKernel::GetName(), contentCPU.get()));
            BMKernelCPU->template castTo<CpuBlockMatchingKernel>()->Calculate();

            // Set the CUDA block matching parameteters
            _reg_blockMatchingParam *blockMatchingParamsCPU = new _reg_blockMatchingParam(contentCPU->GetBlockMatchingParams());
            contentCUDA->SetBlockMatchingParams(blockMatchingParamsCPU);

            // Compute a transformations
            std::unique_ptr<Kernel> kernelCPU = nullptr;
            kernelCPU.reset(platformCPU->CreateKernel(OptimiseKernel::GetName(), contentCPU.get()));
            kernelCPU->template castTo<CpuOptimiseKernel>()->Calculate(ttype);
            std::unique_ptr<Kernel> kernelCUDA = nullptr;
            kernelCUDA.reset(platformCUDA->CreateKernel(OptimiseKernel::GetName(), contentCUDA.get()));
            kernelCUDA->template castTo<CudaOptimiseKernel>()->Calculate(ttype);

            // Save the matrices for testing
            testCases.push_back({
                testName,
                matCPU,
                matCUDA
            });
            contentCPU.reset();
            contentCUDA.reset();
        }
    }
};

TEST_CASE_METHOD(LTSTest, "Regression LTS", "[regression]") {
    // Loop over all generated test cases
    for (auto&& testCase : this->testCases) {
        // Retrieve test information
        auto&& [testName, mat_cpu, mat_cuda] = testCase;

        SECTION(testName) {

            // Loop over the matrix values and ensure they are identical
            for(int j=0; j<4; ++j){
                for(int i=0; i<4; ++i){
                    REQUIRE(fabs(mat_cpu->m[i][j] - mat_cuda->m[i][j]) < EPS);
                }
            }
            delete mat_cpu;
            delete mat_cuda;
        }
    }
}