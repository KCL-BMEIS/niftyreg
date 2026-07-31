#include "reg_test_common.h"
#include "CudaContent.h"
#include "CudaKernelConvolution.hpp"
#include "CudaF3dContent.h"

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
            contentCpu->SetDeformationField(std::move(imageCpu));
            contentCuda->SetDeformationField(std::move(imageCuda));

            // Create the kernel convolution function for CUDA
            auto cudaKernelConvolution = Cuda::KernelConvolution<ConvKernelType(0)>;
            switch (kernelType) {
            case 1: cudaKernelConvolution = Cuda::KernelConvolution<ConvKernelType(1)>; break;
            case 2: cudaKernelConvolution = Cuda::KernelConvolution<ConvKernelType(2)>; break;
            case 3: cudaKernelConvolution = Cuda::KernelConvolution<ConvKernelType(3)>; break;
            }

            // Compute the kernel convolution for CPU and CUDA
            reg_tools_kernelConvolution(contentCpu->GetDeformationField(), sigmaValues.data(), ConvKernelType(kernelType), nullptr, activeTimePoints, activeAxes);
            cudaKernelConvolution(contentCuda->Content::GetDeformationField(), contentCuda->GetDeformationFieldCuda(), sigmaValues.data(), activeTimePoints, activeAxes, nullptr, nullptr);

            // Save the results for testing
            testCases.push_back({ testName, std::move(contentCpu->GetDeformationField()), std::move(contentCuda->GetDeformationField()) });
        }
    }
};

/**
 * Multi-image convolution regression for every kernel type, with a mask and a shared NaN
 * pattern, for both accumulation types (CPU) and including a size whose z-axis sliding window
 * overflows L2 so the packed path takes the column-parallel variant while the single-channel
 * reference stays voxel-parallel (CUDA).
**/
class KernelConvolutionMultiTest {
protected:
    using TestCase = std::tuple<std::string, vector<NiftiImage>, vector<NiftiImage>>;

    inline static vector<TestCase> testCases;

public:
    KernelConvolutionMultiTest() {
        if (!testCases.empty())
            return;

        std::mt19937 gen(1);
        std::uniform_real_distribution<float> distr(0, 1);

        // Geometries: 2D, 3D, and a 3D size large enough that the CUDA packed z pass takes the
        // column-parallel variant (sliding window > L2/2) while the single-channel path does not
        const vector<vector<NiftiImage::dim_t>> dimsList{ { 16, 16 }, { 16, 16, 16 }, { 192, 192, 8 } };
        constexpr int kernelTypeCount = 4;
        const float sigma = -5.f;  // voxel-based, same for every image (the multi-image contract)

        // Fill an image with random values; NaN out the same voxel set in every image (the
        // multi-image functions require an identical NaN pattern across images)
        auto makeImages = [&](const vector<NiftiImage::dim_t>& dims, const int imageCount) {
            vector<NiftiImage> images(imageCount);
            for (int c = 0; c < imageCount; c++) {
                images[c] = NiftiImage(dims, NIFTI_TYPE_FLOAT32);
                auto imagePtr = images[c].data();
                for (size_t i = 0; i < images[c].nVoxels(); i++)
                    imagePtr[i] = distr(gen);
            }
            for (size_t i = 0; i < images[0].nVoxels(); i += 37)
                for (int c = 0; c < imageCount; c++)
                    images[c].data()[i] = std::numeric_limits<float>::quiet_NaN();
            return images;
        };
        auto makeMask = [](const size_t voxelNumber) {
            vector<int> mask(voxelNumber);
            for (size_t i = 0; i < voxelNumber; i++)
                mask[i] = i % 7 == 0 ? -1 : 0;
            return mask;
        };

        for (const auto& dims : dimsList) {
            for (int kernelType = 0; kernelType < kernelTypeCount; kernelType++) {
                /* ------------------------------- CPU: multi vs single ------------------------------ */
                for (const int imageCount : { 2, 3, 4 }) {
                    for (const bool useFloatAcc : { false, true }) {
                        const vector<NiftiImage> images = makeImages(dims, imageCount);
                        const vector<int> mask = makeMask(images[0].nVoxels());

                        // Reference: each image convolved ALONE (density recomputed every time)
                        vector<NiftiImage> expected(images.begin(), images.end());
                        ConvolutionWorkspace wsSingle;
                        wsSingle.useFloatAccumulation = useFloatAcc;
                        for (int c = 0; c < imageCount; c++) {
                            wsSingle.densityValid = false;
                            reg_tools_kernelConvolution(expected[c], &sigma, ConvKernelType(kernelType),
                                                        mask.data(), nullptr, nullptr, &wsSingle);
                        }

                        // Actual: one multi-image sweep
                        vector<NiftiImage> actual(images.begin(), images.end());
                        vector<nifti_image*> actualPtrs(imageCount);
                        for (int c = 0; c < imageCount; c++) actualPtrs[c] = actual[c];
                        ConvolutionWorkspace wsMulti;
                        wsMulti.useFloatAccumulation = useFloatAcc;
                        reg_tools_kernelConvolutionMulti(actualPtrs.data(), imageCount, &sigma,
                                                         ConvKernelType(kernelType), mask.data(), &wsMulti);

                        const std::string dimsStr = std::accumulate(dims.begin(), dims.end(), ""s,
                            [](const std::string& str, const auto& val) { return str + " "s + std::to_string(val); });
                        testCases.push_back({ "CPU Multi Kernel: "s + std::to_string(kernelType) +
                                              " Images: "s + std::to_string(imageCount) +
                                              (useFloatAcc ? " FloatAcc" : " DoubleAcc") +
                                              " Dims:"s + dimsStr,
                                              std::move(expected), std::move(actual) });
                    }
                }

                /* ------------- CPU: workspace-less / mask-less / single-image entry points --------- */
                if (&dims == &dimsList[0]) {
                    const vector<NiftiImage> images = makeImages(dims, 2);

                    // N=2 with neither workspace nor mask (locally allocated scratch, all-active mask)
                    vector<NiftiImage> expected(images.begin(), images.end());
                    for (int c = 0; c < 2; c++)
                        reg_tools_kernelConvolution(expected[c], &sigma, ConvKernelType(kernelType));
                    vector<NiftiImage> actual(images.begin(), images.end());
                    vector<nifti_image*> actualPtrs{ actual[0], actual[1] };
                    reg_tools_kernelConvolutionMulti(actualPtrs.data(), 2, &sigma, ConvKernelType(kernelType));

                    // N=1 delegates to the single-image convolution
                    NiftiImage oneExpected(images[0]), oneActual(images[0]);
                    reg_tools_kernelConvolution(oneExpected, &sigma, ConvKernelType(kernelType));
                    nifti_image *onePtr = oneActual;
                    reg_tools_kernelConvolutionMulti(&onePtr, 1, &sigma, ConvKernelType(kernelType));
                    expected.push_back(std::move(oneExpected));
                    actual.push_back(std::move(oneActual));

                    testCases.push_back({ "CPU Multi Kernel: "s + std::to_string(kernelType) +
                                          " NoWorkspaceNoMask", std::move(expected), std::move(actual) });
                }

                /* ---------------------------- CUDA: packed vs single-channel ----------------------- */
                {
                    // Four images packed as the four time points of one image: CudaContent stores an
                    // nt=4 deformation field as one float4 per voxel, i.e. one image per lane. The
                    // NaN pattern must be identical across the lanes (multi-image contract), so the
                    // NaNs are injected per 3D voxel into all four planes.
                    vector<NiftiImage::dim_t> dims4(dims);
                    dims4.resize(4, 1);
                    dims4[3] = 4;
                    NiftiImage packedImage(dims4, NIFTI_TYPE_FLOAT32);
                    const size_t voxelNumber = packedImage.nVoxels() / 4;  // 4 time points (lanes)
                    auto packedPtr = packedImage.data();
                    for (size_t i = 0; i < packedImage.nVoxels(); i++)
                        packedPtr[i] = distr(gen);
                    for (size_t i = 0; i < voxelNumber; i += 37)
                        for (int t = 0; t < 4; t++)
                            packedPtr[t * voxelNumber + i] = std::numeric_limits<float>::quiet_NaN();
                    const vector<int> mask = makeMask(voxelNumber);
                    int *maskCudaPtr = nullptr;
                    Cuda::Allocate<int>(&maskCudaPtr, voxelNumber);
                    Cuda::UniquePtr<int> maskCuda(maskCudaPtr);
                    Cuda::TransferFromHostToDevice<int>(maskCudaPtr, mask.data(), voxelNumber);

                    // Reference: the single-channel convolution applied to each lane in turn (the
                    // same sigma for every lane)
                    NiftiImage imageSingle(packedImage), imagePacked(packedImage);
                    unique_ptr<CudaContent> contentSingle{ new CudaContent(imageSingle, imageSingle, nullptr, nullptr, sizeof(float)) };
                    unique_ptr<CudaContent> contentPacked{ new CudaContent(imagePacked, imagePacked, nullptr, nullptr, sizeof(float)) };
                    contentSingle->SetDeformationField(std::move(imageSingle));
                    contentPacked->SetDeformationField(std::move(imagePacked));

                    const float sigma4[4]{ sigma, sigma, sigma, sigma };
                    Cuda::KernelConvolutionWorkspace wsCudaSingle, wsCudaPacked;
                    auto cudaSingle = Cuda::KernelConvolution<ConvKernelType(0), float>;
                    auto cudaPacked = Cuda::KernelConvolutionPacked<ConvKernelType(0), float>;
                    switch (kernelType) {
                    case 1: cudaSingle = Cuda::KernelConvolution<ConvKernelType(1), float>; cudaPacked = Cuda::KernelConvolutionPacked<ConvKernelType(1), float>; break;
                    case 2: cudaSingle = Cuda::KernelConvolution<ConvKernelType(2), float>; cudaPacked = Cuda::KernelConvolutionPacked<ConvKernelType(2), float>; break;
                    case 3: cudaSingle = Cuda::KernelConvolution<ConvKernelType(3), float>; cudaPacked = Cuda::KernelConvolutionPacked<ConvKernelType(3), float>; break;
                    }
                    cudaSingle(contentSingle->Content::GetDeformationField(), contentSingle->GetDeformationFieldCuda(),
                               sigma4, nullptr, nullptr, maskCudaPtr, &wsCudaSingle);
                    cudaPacked(contentPacked->Content::GetDeformationField(), contentPacked->GetDeformationFieldCuda(),
                               sigma4, maskCudaPtr, &wsCudaPacked);

                    const std::string dimsStr = std::accumulate(dims.begin(), dims.end(), ""s,
                        [](const std::string& str, const auto& val) { return str + " "s + std::to_string(val); });
                    vector<NiftiImage> expected, actual;
                    expected.push_back(std::move(contentSingle->GetDeformationField()));
                    actual.push_back(std::move(contentPacked->GetDeformationField()));
                    testCases.push_back({ "CUDA Packed Kernel: "s + std::to_string(kernelType) + " Dims:"s + dimsStr,
                                          std::move(expected), std::move(actual) });
                }
            }
        }
    }
};

TEST_CASE_METHOD(KernelConvolutionMultiTest, "Regression Kernel Convolution Multi", "[regression]") {
    for (auto&& testCase : testCases) {
        auto&& [testName, expected, actual] = testCase;

        SECTION(testName) {
            NR_COUT << "\n**************** Section " << testName << " ****************" << std::endl;
            NR_COUT << std::fixed << std::setprecision(10);

            for (size_t c = 0; c < expected.size(); ++c) {
                const auto expectedPtr = expected[c].data();
                const auto actualPtr = actual[c].data();
                for (size_t i = 0; i < expected[c].nVoxels(); ++i) {
                    const float expectedVal = expectedPtr[i];
                    const float actualVal = actualPtr[i];
                    if (expectedVal != expectedVal && actualVal != actualVal) continue;  // Skip NaN values
                    const float diff = fabs(expectedVal - actualVal);
                    if (diff > 0)
                        NR_COUT << "img " << c << " vox " << i << " " << expectedVal << " " << actualVal << std::endl;
                    REQUIRE(diff == 0);
                }
            }
        }
    }
}

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
                REQUIRE(diff == 0);
            }
        }
    }
}

namespace {

// A control point grid and a filled voxel-based gradient, which is what the two convolution wrappers
// below operate on
struct GradientFixture {
    Platform platformCpu{ PlatformType::Cpu };
    Platform platformCuda{ PlatformType::Cuda };
    NiftiImage refCpu, floCpu, gridCpu, refCuda, floCuda, gridCuda;
    unique_ptr<F3dContent> contentCpu, contentCuda;
    unique_ptr<Compute> computeCpu, computeCuda;

    GradientFixture(bool is3D, float gridSpacingInVoxels, bool anisotropic) {
        const NiftiImage reference = MakeReferenceImage(is3D, anisotropic);
        refCpu = reference; floCpu = reference; refCuda = reference; floCuda = reference;
        NiftiImage grid;
        const float spacing[3]{ reference->dx * gridSpacingInVoxels, reference->dy * gridSpacingInVoxels,
                                reference->dz * gridSpacingInVoxels };
        reg_createControlPointGrid<float>(grid, reference, spacing);
        gridCpu = grid; gridCuda = grid;
        contentCpu.reset(new F3dContent(refCpu, floCpu, gridCpu));
        contentCuda.reset(new CudaF3dContent(refCuda, floCuda, gridCuda));
        computeCpu.reset(platformCpu.CreateCompute(*contentCpu));
        computeCuda.reset(platformCuda.CreateCompute(*contentCuda));
    }

    static NiftiImage MakeReferenceImage(bool is3D, bool anisotropic) {
        const NiftiImage::dim_t size = 12;
        std::vector<NiftiImage::dim_t> dims(is3D ? 3 : 2, size);
        NiftiImage img(dims, NIFTI_TYPE_FLOAT32);
        if (anisotropic) setAnisotropicSform(img);
        else {
            setIdentitySform(img);
            img->dx = img->pixdim[1] = 1.f; img->dy = img->pixdim[2] = 1.f; img->dz = img->pixdim[3] = 1.f;
        }
        auto ptr = img.data();
        for (size_t i = 0; i < img.nVoxels(); ++i)
            ptr[i] = static_cast<float>(100.0 * (1.5 + std::sin(0.31 * double(i))));
        return img;
    }

    void FillVoxelBasedGradient() {
        const auto fill = [](NiftiImage& image) {
            auto ptr = image.data();
            for (size_t i = 0; i < image.nVoxels(); ++i)
                ptr[i] = static_cast<float>(std::sin(0.41 * double(i)) * 0.05);
        };
        fill(contentCpu->GetVoxelBasedMeasureGradient());
        contentCpu->UpdateVoxelBasedMeasureGradient();
        fill(contentCuda->DefContent::GetVoxelBasedMeasureGradient());
        contentCuda->UpdateVoxelBasedMeasureGradient();
    }

    void FillTransformationGradient() {
        const auto fill = [](NiftiImage& image) {
            auto ptr = image.data();
            for (size_t i = 0; i < image.nVoxels(); ++i)
                ptr[i] = static_cast<float>(std::cos(0.23 * double(i)) * 0.4);
        };
        fill(contentCpu->GetTransformationGradient());
        contentCpu->UpdateTransformationGradient();
        fill(contentCuda->F3dContent::GetTransformationGradient());
        contentCuda->UpdateTransformationGradient();
    }
};

} // namespace

TEST_CASE("Regression Kernel Convolution - voxel-based gradient smoothing", "[regression]") {
    /*
        ConvolveVoxelBasedMeasureGradient, as opposed to the KernelConvolution primitive above.

        The wrapper is not a single convolution: it makes three passes, one per axis, each with a
        cubic kernel whose width is that axis's control point spacing, and each renormalising by its
        own smoothed density. It also carries a scratch workspace across the three. So the axes are
        smoothed with different kernel widths and the passes are not interchangeable - none of which
        the primitive's own test can reach, since that one convolves an image with a sigma it is
        handed.

        Anisotropic spacings are included precisely because they make the three passes differ from
        each other; an isotropic grid would let a pass that used the wrong axis's width still agree.
    */
    for (const bool is3D : { false, true })
        for (const auto& [label, spacing, anisotropic] : { std::tuple{ "grid 2 voxels", 2.f, false },
                                                           std::tuple{ "grid 2 voxels, anisotropic", 2.f, true },
                                                           std::tuple{ "grid 5 voxels", 5.f, false } }) {
            const std::string name = std::string(is3D ? "3D" : "2D") + ", " + label;
            SECTION(name) {
                GradientFixture f(is3D, spacing, anisotropic);
                f.FillVoxelBasedGradient();
                const NiftiImage before(f.contentCpu->GetVoxelBasedMeasureGradient(), NiftiImage::Copy::Image);

                f.computeCpu->ConvolveVoxelBasedMeasureGradient(1.f);
                f.computeCuda->ConvolveVoxelBasedMeasureGradient(1.f);

                RequireChanged(before, f.contentCpu->GetVoxelBasedMeasureGradient(),
                               "the CPU voxel-based measure gradient");
                const Deviation deviation = CompareImages(f.contentCpu->GetVoxelBasedMeasureGradient(),
                                                          f.contentCuda->GetVoxelBasedMeasureGradient());
                ReportDeviation(name, deviation);
                INFO(name << ": " << deviation.differing << " values differ, max " << deviation.max);
                REQUIRE(deviation.max == 0);
            }
        }
}

TEST_CASE("Regression Kernel Convolution - transformation gradient smoothing", "[regression]") {
    /*
        SmoothGradient, the Gaussian smoothing applied to the control point gradient under -smoothGrad.

        Its own wrinkle is the sigma argument: the convolution reads one sigma per time point, and a
        control point gradient always has two or three components, so a caller passing a single value
        reads past the end of what it supplied.

        The cross-backend comparison cannot see that. Both backends would read the same out-of-bounds
        value and agree with each other while both being wrong - which is the general limit of
        comparing two implementations rather than checking a result. So the last section compares the
        wrapper against the primitive called with a correctly sized sigma array, which is an
        expectation the wrapper has to meet on its own.
    */
    for (const bool is3D : { false, true })
        for (const bool anisotropic : { false, true }) {
            const std::string name = std::string(is3D ? "3D" : "2D") +
                (anisotropic ? ", anisotropic" : ", identity sform");
            SECTION(name) {
                GradientFixture f(is3D, 2.f, anisotropic);
                f.FillTransformationGradient();
                const NiftiImage before(f.contentCpu->GetTransformationGradient(), NiftiImage::Copy::Image);

                f.computeCpu->SmoothGradient(1.5f);
                f.computeCuda->SmoothGradient(1.5f);

                RequireChanged(before, f.contentCpu->GetTransformationGradient(),
                               "the CPU transformation gradient");
                const Deviation deviation = CompareImages(f.contentCpu->GetTransformationGradient(),
                                                          f.contentCuda->GetTransformationGradient());
                ReportDeviation(name, deviation);
                INFO(name << ": " << deviation.differing << " values differ, max " << deviation.max);
                REQUIRE(deviation.max == 0);
            }
        }
}

TEST_CASE("Regression Kernel Convolution - gradient smoothing applies its sigma to every component",
          "[regression]") {
    /*
        SmoothGradient(s) has to equal a Gaussian convolution with s supplied for every time point.
        Anything the wrapper does with its sigma argument other than broadcasting it - passing the
        scalar straight through, so that components after the first take whatever follows it in
        memory - shows up here, and shows up nowhere in a comparison of two backends that would both
        read the same wrong value.
    */
    for (const bool is3D : { false, true }) {
        SECTION(is3D ? "3D" : "2D") {
            constexpr float sigma = 1.5f;
            GradientFixture f(is3D, 2.f, false);
            f.FillTransformationGradient();

            // The expectation: the same convolution, with the sigma array the primitive expects
            NiftiImage expected(f.contentCpu->GetTransformationGradient(), NiftiImage::Copy::Image);
            const std::vector<float> sigmaPerTimePoint(expected->nt * expected->nu, sigma);
            reg_tools_kernelConvolution(expected, sigmaPerTimePoint.data(), ConvKernelType::Gaussian);

            f.computeCpu->SmoothGradient(sigma);

            const Deviation deviation = CompareImages(f.contentCpu->GetTransformationGradient(), expected);
            ReportDeviation(std::string(is3D ? "3D" : "2D") + ", sigma per component", deviation);
            INFO("max deviation " << deviation.max << " over " << deviation.differing << " values");
            REQUIRE(deviation.max == 0);
        }
    }
}
