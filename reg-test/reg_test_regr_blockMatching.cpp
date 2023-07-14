#include "reg_test_common.h"
#include "_reg_blockMatching.h"
#include "CpuBlockMatchingKernel.h"
#include "CudaBlockMatchingKernel.h"

/*
    This test file contains a regression test to ensure the CPU and GPU version yield the same output
*/

class BMTest {
    /*
    Class to call the block matching function
    */
protected:
    using TestData = std::tuple<std::string, NiftiImage, NiftiImage, int>;
    using TestCase = std::tuple<std::string, _reg_blockMatchingParam *, _reg_blockMatchingParam *>;
    inline static vector<TestCase> testCases;
    NiftiImage reference2d;
    NiftiImage floating2d;
    NiftiImage reference3d;
    NiftiImage floating3d;
public:
    ~BMTest() {
        std::cout << "Calling destructor" << std::endl;
    }
    BMTest() {
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
        for(int b=50; b<=100; b+=50){
            testData.emplace_back(TestData(
                "BlockMatching 2D block " + std::to_string(b),
                std::move(NiftiImage(this->reference2d)),
                std::move(NiftiImage(this->floating2d)),
                b
            ));
            testData.emplace_back(TestData(
                "BlockMatching 3D block " + std::to_string(b),
                std::move(NiftiImage(this->reference3d)),
                std::move(NiftiImage(this->floating3d)),
                b
            ));
        }

        for (auto&& data : testData) {
            unique_ptr<Platform> platformCPU{ new Platform(PlatformType::Cpu) };
            unique_ptr<Platform> platformCUDA{ new Platform(PlatformType::Cuda) };
            // Make a copy of the test data
            auto&& [testName, reference, floating, block] = data;
            // Create content creator
            unique_ptr<AladinContentCreator> contentCreatorCPU{
                dynamic_cast<AladinContentCreator*>(platformCPU->CreateContentCreator(ContentType::Aladin))
            };
            unique_ptr<AladinContentCreator> contentCreatorCUDA{
                dynamic_cast<AladinContentCreator*>(platformCUDA->CreateContentCreator(ContentType::Aladin))
            };
            // Create the contents
            unique_ptr<AladinContent> contentCPU{ contentCreatorCPU->Create(
                NiftiImage(reference).disown(),
                NiftiImage(floating).disown(),
                nullptr,
                nullptr,
                sizeof(float),
                100,
                block,
                1
            )};
            unique_ptr<AladinContent> contentCUDA{ contentCreatorCUDA->Create(
                NiftiImage(reference).disown(),
                NiftiImage(floating).disown(),
                nullptr,
                nullptr,
                sizeof(float),
                100,
                block,
                1
            )};
            // Initialise the warped image
            contentCPU->SetWarped(NiftiImage(floating).disown());
            contentCUDA->SetWarped(NiftiImage(floating).disown());
            // Initialise the block matching
            std::unique_ptr<Kernel> kernelCPU = nullptr;
            kernelCPU.reset(platformCPU->CreateKernel(BlockMatchingKernel::GetName(), contentCPU.get()));
            std::unique_ptr<Kernel> kernelCUDA = nullptr;
            kernelCUDA.reset(platformCUDA->CreateKernel(BlockMatchingKernel::GetName(), contentCUDA.get()));

            // run the computation
            kernelCPU->template castTo<CpuBlockMatchingKernel>()->Calculate();
            kernelCUDA->template castTo<CudaBlockMatchingKernel>()->Calculate();

            // Retrieve the information
            _reg_blockMatchingParam *blockMatchingParamsCPU = new _reg_blockMatchingParam(contentCPU->GetBlockMatchingParams());
            _reg_blockMatchingParam *blockMatchingParamsCUDA = new _reg_blockMatchingParam(contentCUDA->GetBlockMatchingParams());

            testCases.push_back({
                testName,
                blockMatchingParamsCPU,
                blockMatchingParamsCUDA
            });
            contentCPU.reset();
            contentCUDA.reset();
        }
    }
};

TEST_CASE_METHOD(BMTest, "Regression BlockMatching", "[regression]") {
    // Loop over all generated test cases
    for (auto&& testCase : this->testCases) {
        // Retrieve test information
        auto&& [testName, blockMatchingParamsCPU, blockMatchingParamsCUDA] = testCase;

        SECTION(testName) {

            // Ensure both approaches retreive the same number of voxel
            REQUIRE(blockMatchingParamsCPU->activeBlockNumber==blockMatchingParamsCUDA->activeBlockNumber);

            // Loop over the block and ensure all values are identical
            for(int b=0; b<blockMatchingParamsCPU->activeBlockNumber*blockMatchingParamsCPU->dim; ++b){
                float delta = blockMatchingParamsCPU->referencePosition[b] - blockMatchingParamsCUDA->referencePosition[b];
                if(fabs(delta) > EPS){
                    std::cout << "HERE " << delta << std::endl;
                    std::cout.flush();
                }
                REQUIRE(fabs(delta) < EPS);
                delta = blockMatchingParamsCPU->warpedPosition[b] - blockMatchingParamsCUDA->warpedPosition[b];
                if(fabs(delta) > EPS){
                    std::cout << "HERE " << delta << std::endl;
                    std::cout.flush();
                }
                REQUIRE(fabs(delta) < EPS);
            }
            delete blockMatchingParamsCPU;
            delete blockMatchingParamsCUDA;
        }
    }
}