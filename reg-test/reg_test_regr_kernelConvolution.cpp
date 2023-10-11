#include "reg_test_common.h"
#include "CudaContent.h"
#include "CudaKernelConvolution.hpp"

/**
 *  Kernel convolution regression test to ensure the CPU and CUDA versions yield the same output
**/

class KernelConvolutionTest {
protected:
    using TestData = std::tuple<std::string, NiftiImage&, vector<float>, int, bool*, bool*>;
    using TestCase = std::tuple<std::string, NiftiImage, NiftiImage>;

    inline static vector<TestCase> testCases;

public:
    KernelConvolutionTest() {
        if (!testCases.empty())
            return;

        // Create a random number generator
        std::mt19937 gen(0);
        std::uniform_real_distribution<float> distr(0, 1);

        // Create images
        constexpr int imageCount = 8;
        constexpr NiftiImage::dim_t size = 16;
        vector<NiftiImage::dim_t> dims[imageCount]{ { size, size },
                                                   { size, size, 1, 1, 2 },
                                                   { size, size, 1, 1, 3 },
                                                   { size, size, 1, 2, 2 },
                                                   { size, size, size },
                                                   { size, size, size, 2, 1 },
                                                   { size, size, size, 3, 1 },
                                                   { size, size, size, 2, 2 } };
        NiftiImage images[imageCount];

        // Fill images with random values
        for (int i = 0; i < imageCount; i++) {
            images[i] = NiftiImage(dims[i], NIFTI_TYPE_FLOAT32);
            auto imagePtr = images[i].data();
            for (size_t j = 0; j < images[i].nVoxels(); j++)
                imagePtr[j] = distr(gen);
        }

        // Create a lambda to concatenate strings for std::accumulate
        auto strConcat = [](const std::string& str, const auto& val) { return str + " "s + std::to_string(val); };

        // Create the data container for the regression test
        constexpr int kernelTypeCount = 4;
        distr.param(std::uniform_real_distribution<float>::param_type(1, 10));  // Change the range of the distribution
        vector<TestData> testData;
        for (int i = 0; i < imageCount; i++) {
            for (int kernelType = 0; kernelType < kernelTypeCount; kernelType++) {
                vector<float> sigmaValues(images[i]->nt * images[i]->nu);
                std::generate(sigmaValues.begin(), sigmaValues.end(), [&]() { return distr(gen); });
                const std::string sigmaStr = std::accumulate(sigmaValues.begin(), sigmaValues.end(), ""s, strConcat);
                const std::string dimsStr = std::accumulate(dims[i].begin(), dims[i].end(), ""s, strConcat);
                testData.emplace_back(TestData(
                    "Kernel: "s + std::to_string(kernelType) + " Sigma:"s + sigmaStr + " Dims:"s + dimsStr,
                    images[i],
                    std::move(sigmaValues),
                    kernelType,
                    nullptr,
                    nullptr
                ));
            }
        }

        // Define time points and axes to smooth
        constexpr auto timePointCount = 4;
        bool timePoints[timePointCount][4]{ { true, false, false, false },
                                           { false, true, false, false },
                                           { false, false, true, false },
                                           { false, false, false, true } };
        bool axes[timePointCount][3]{ { true, false, false },
                                     { false, true, false },
                                     { false, false, true },
                                     { true, true, true } };

        // Add the time points and axes to the latest test data
        for (int i = 0, latestIndex = int(testData.size()) - timePointCount; i < timePointCount; i++, latestIndex++) {
            auto&& [testName, image, sigmaValues, kernelType, activeTimePoints, activeAxes] = testData[latestIndex];
            const std::string timePointsStr = std::accumulate(timePoints[i], timePoints[i] + 4, ""s, strConcat);
            const std::string axesStr = std::accumulate(axes[i], axes[i] + 3, ""s, strConcat);
            testData.emplace_back(TestData(
                testName + " TimePoints:"s + timePointsStr + " Axes:"s + axesStr,
                image,
                sigmaValues,
                kernelType,
                timePoints[i],
                axes[i]
            ));
        }

        // Create the platforms
        Platform platformCpu(PlatformType::Cpu);
        Platform platformCuda(PlatformType::Cuda);

        for (auto&& testData : testData) {
            // Get the test data
            auto&& [testName, image, sigmaValues, kernelType, activeTimePoints, activeAxes] = testData;

            // Create images
            NiftiImage imageCpu(image), imageCuda(image);

            // Create the contents
            unique_ptr<Content> contentCpu{ new Content(
                imageCpu,
                imageCpu,
                nullptr,
                nullptr,
                sizeof(float)
            ) };
            unique_ptr<CudaContent> contentCuda{ new CudaContent(
                imageCuda,
                imageCuda,
                nullptr,
                nullptr,
                sizeof(float)
            ) };

            // Use deformation fields to store images
            contentCpu->SetDeformationField(imageCpu.disown());
            contentCuda->SetDeformationField(imageCuda.disown());

            // Create the kernel convolution function for CUDA
            auto cudaKernelConvolution = Cuda::KernelConvolution<ConvKernelType(0)>;
            switch (kernelType) {
            case 1: cudaKernelConvolution = Cuda::KernelConvolution<ConvKernelType(1)>; break;
            case 2: cudaKernelConvolution = Cuda::KernelConvolution<ConvKernelType(2)>; break;
            case 3: cudaKernelConvolution = Cuda::KernelConvolution<ConvKernelType(3)>; break;
            }

            // Compute the kernel convolution for CPU and CUDA
            reg_tools_kernelConvolution(contentCpu->GetDeformationField(), sigmaValues.data(), ConvKernelType(kernelType), nullptr, activeTimePoints, activeAxes);
            cudaKernelConvolution(contentCuda->Content::GetDeformationField(), contentCuda->GetDeformationFieldCuda(), sigmaValues.data(), activeTimePoints, activeAxes);

            // Get the images
            imageCpu = NiftiImage(contentCpu->GetDeformationField(), NiftiImage::Copy::Image);
            imageCuda = NiftiImage(contentCuda->GetDeformationField(), NiftiImage::Copy::Image);

            // Save for testing
            testCases.push_back({ testName, std::move(imageCpu), std::move(imageCuda) });
        }
    }
};

TEST_CASE_METHOD(KernelConvolutionTest, "Regression Kernel Convolution", "[regression]") {
    // Loop over all generated test cases
    for (auto&& testCase : testCases) {
        // Retrieve test information
        auto&& [testName, imageCpu, imageCuda] = testCase;

        SECTION(testName) {
            NR_COUT << "\n**************** Section " << testName << " ****************" << std::endl;

            // Increase the precision for the output
            NR_COUT << std::fixed << std::setprecision(10);

            // Check the images
            const auto imageCpuPtr = imageCpu.data();
            const auto imageCudaPtr = imageCuda.data();
            for (size_t i = 0; i < imageCpu.nVoxels(); ++i) {
                const float cpuVal = imageCpuPtr[i];
                const float cudaVal = imageCudaPtr[i];
                if (cpuVal != cpuVal && cudaVal != cudaVal) continue;  // Skip NaN values
                const float diff = fabs(cpuVal - cudaVal);
                if (diff > EPS)
                    NR_COUT << i << " " << cpuVal << " " << cudaVal << std::endl;
                REQUIRE(diff < EPS);
            }
        }
    }
}
