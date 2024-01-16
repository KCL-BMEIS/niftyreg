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
                        NiftiImage transGradCpu = contentCpu->GetTransformationGradient();
                        transGradCpu.copyData(transGrad);
                        transGradCpu.disown();
                        contentCpu->UpdateTransformationGradient();
                        NiftiImage transGradCuda = contentCuda->GetTransformationGradient();
                        transGradCuda.copyData(transGrad);
                        transGradCuda.disown();
                        contentCuda->UpdateTransformationGradient();

                        // Create the computes
                        unique_ptr<Compute> computeCpu{ platformCpu.CreateCompute(*contentCpu) };
                        unique_ptr<Compute> computeCuda{ platformCuda.CreateCompute(*contentCuda) };

                        // Update the velocity field
                        computeCpu->UpdateVelocityField(scale, optimiseX, optimiseY, optimiseZ);
                        computeCuda->UpdateVelocityField(scale, optimiseX, optimiseY, optimiseZ);

                        // Get the results
                        transGradCpu = NiftiImage(contentCpu->GetTransformationGradient(), NiftiImage::Copy::Image);
                        transGradCuda = NiftiImage(contentCuda->GetTransformationGradient(), NiftiImage::Copy::Image);

                        // Save for testing
                        testCases.push_back({ testName, std::move(transGradCpu), std::move(transGradCuda) });
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
