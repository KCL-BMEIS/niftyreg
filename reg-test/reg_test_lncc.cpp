// OpenCL and CUDA are not supported for this test yet
#undef USE_OPENCL
#undef USE_CUDA

#include "reg_test_common.h"
#include "_reg_lncc.h"

/*
    This test file contains the following unit tests:
    test function: LNCC computation and its voxel-wise gradient
    In 2D and 3D
*/

class LnccTest {
public:
    LnccTest() {
        if (!testCases.empty())
            return;

        // Create a random number generator
        std::mt19937 gen(0);
        std::uniform_real_distribution<float> distr(0, 1);

        // Create reference and floating 2D images
        vector<NiftiImage::dim_t> dim{ 16, 16 };
        NiftiImage reference2d(dim, NIFTI_TYPE_FLOAT32);
        NiftiImage floating2d(dim, NIFTI_TYPE_FLOAT32);

        // Create reference and floating 3D images
        dim.push_back(16);
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

        // Create the object to compute the expected values
        vector<TestData> testData;
        testData.emplace_back(TestData(
            "LNCC 2D -1",
            reference2d,
            floating2d,
            -1.f,
            GetLNCCNoConv(1, reference2d, floating2d)
        ));
        testData.emplace_back(TestData(
            "LNCC 2D -1 same image",
            reference2d,
            reference2d,
            -1.f,
            1.0
        ));
        testData.emplace_back(TestData(
            "LNCC 2D -5",
            reference2d,
            floating2d,
            -5.f,
            GetLNCCNoConv(5, reference2d, floating2d)
        ));
        testData.emplace_back(TestData(
            "LNCC 2D -5 same image",
            reference2d,
            reference2d,
            -5.f,
            1.0
        ));
        reg_tools_multiplyValueToImage(reference2d, floating2d, -1.f);
        testData.emplace_back(TestData(
            "LNCC 2D -1 same image negated",
            reference2d,
            floating2d,
            -1.f,
            1.0
        ));
        testData.emplace_back(TestData(
            "LNCC 2D -5 same image negated",
            reference2d,
            floating2d,
            -5.f,
            1.0
        ));
        testData.emplace_back(TestData(
            "LNCC 3D -1",
            reference3d,
            floating3d,
            -1.f,
            GetLNCCNoConv(1, reference3d, floating3d)
        ));
        testData.emplace_back(TestData(
            "LNCC 3D -1 same image",
            reference3d,
            reference3d,
            -1.f,
            1.0
        ));
        testData.emplace_back(TestData(
            "LNCC 3D -5",
            reference3d,
            floating3d,
            -5.f,
            GetLNCCNoConv(5, reference3d, floating3d)
        ));
        testData.emplace_back(TestData(
            "LNCC 3D -5 same image",
            reference3d,
            reference3d,
            -5.f,
            1.0
        ));
        reg_tools_multiplyValueToImage(reference3d, floating3d, -1.f);
        testData.emplace_back(TestData(
            "LNCC 3D -1 same image negated",
            reference3d,
            floating3d,
            -1.f,
            1.0
        ));
        testData.emplace_back(TestData(
            "LNCC 3D -5 same image negated",
            reference3d,
            floating3d,
            -5.f,
            1.0
        ));
        for (auto&& data : testData) {
            for (auto&& platformType : PlatformTypes) {
                // Create the platform
                unique_ptr<Platform> platform{ new Platform(platformType) };
                // Make a copy of the test data
                auto [testName, reference, floating, sigma, expLncc] = data;
                // Create the content creator
                unique_ptr<DefContentCreator> contentCreator{
                    dynamic_cast<DefContentCreator*>(platform->CreateContentCreator(ContentType::Def))
                };
                // Create the content
                unique_ptr<DefContent> content{ contentCreator->Create(reference, floating) };
                // Initialise the warped image using the nearest-neighbour interpolation
                unique_ptr<Compute> compute{ platform->CreateCompute(*content) };
                compute->ResampleImage(0, 0);
                content->SetWarped(NiftiImage(floating));
                // Create the measure creator
                unique_ptr<MeasureCreator> measureCreator{ platform->CreateMeasureCreator() };
                // Use LNCC as a measure
                unique_ptr<reg_lncc> measure_lncc{ dynamic_cast<reg_lncc*>(measureCreator->Create(MeasureType::Lncc)) };
                measure_lncc->SetKernelStandardDeviation(0, sigma);
                measure_lncc->SetTimePointWeight(0, 1.0); // weight initially set to default value of 1.0
                measureCreator->Initialise(*measure_lncc, *content);
                const double lncc = measure_lncc->GetSimilarityMeasureValue();
                // Save the results for testing
                testCases.push_back({ testName, lncc, expLncc });
            }
        }
    }

protected:
    struct Kernel {
        // float[] not float: the buffer below is allocated with new[], so a unique_ptr<float> would
        // destroy it with delete rather than delete[] - undefined behaviour, and what AddressSanitizer
        // reports as an alloc-dealloc-mismatch
        unique_ptr<float[]> ptr;
        int radius[3];
        int size[3];
    };

    using LocalStats = std::tuple<double, double>;
    using TestData = std::tuple<std::string, NiftiImage, NiftiImage, float, double>;
    using TestCase = std::tuple<std::string, double, double>;
    inline static vector<TestCase> testCases;

    double GetLNCCNoConv(int kernelStd, const NiftiImage& ref, const NiftiImage& flo) {
        // Compute the kernel
        Kernel kernel = InitialiseKernel(ref, (float)abs(kernelStd));
        double lncc = 0, voxelNumber = 0;
        for (int z = 0; z < ref->nz; ++z) {
            for (int y = 0; y < ref->ny; ++y) {
                for (int x = 0; x < ref->nx; ++x) {
                    lncc += abs(GetLocalCC(x, y, z, kernel, ref, flo, GetLocalMeans(x, y, z, kernel, ref, flo)));
                    voxelNumber++;
                }
            }
        }
        return lncc / voxelNumber;
    }

    Kernel InitialiseKernel(const NiftiImage& ref, const float kernelStdVoxel) {
        Kernel kernel;
        kernel.radius[0] = static_cast<int>(3.f * kernelStdVoxel);
        kernel.radius[1] = static_cast<int>(3.f * kernelStdVoxel);
        kernel.radius[2] = 0;
        if (ref->ndim > 2)
            kernel.radius[2] = static_cast<int>(3.f * kernelStdVoxel);
        kernel.size[0] = kernel.radius[0] * 2 + 1;
        kernel.size[1] = kernel.radius[1] * 2 + 1;
        kernel.size[2] = kernel.radius[2] * 2 + 1;
        kernel.ptr = unique_ptr<float[]>(new float[kernel.size[0] * kernel.size[1] * kernel.size[2]]);
        float *kernelPtr = kernel.ptr.get();

        for (int z = -kernel.radius[2]; z <= kernel.radius[2]; z++) {
            const float z_value = static_cast<float>(
                exp(-(z * z) / (2.0 * Square(kernelStdVoxel))) / (kernelStdVoxel * 2.506628274631));
            for (int y = -kernel.radius[1]; y <= kernel.radius[1]; y++) {
                const float y_value = static_cast<float>(
                    exp(-(y * y) / (2.0 * Square(kernelStdVoxel))) / (kernelStdVoxel * 2.506628274631));
                for (int x = -kernel.radius[0]; x <= kernel.radius[0]; x++) {
                    const float x_value = static_cast<float>(
                        exp(-(x * x) / (2.0 * Square(kernelStdVoxel))) / (kernelStdVoxel * 2.506628274631));
                    *kernelPtr++ = x_value * y_value * z_value;
                }
            }
        }

        return kernel;
    }

    LocalStats GetLocalMeans(const int x, const int y, const int z, const Kernel& kernel,
                             const NiftiImage& ref, const NiftiImage& flo) {
        double meanRef = 0, meanFlo = 0, kernelSum = 0;
        const float *kernelPtr = kernel.ptr.get();
        const auto refPtr = ref.data();
        const auto floPtr = flo.data();
        for (int k = -kernel.radius[2]; k <= kernel.radius[2]; k++) {
            int zz = z + k;
            if (0 <= zz && zz < ref->nz) {
                for (int j = -kernel.radius[1]; j <= kernel.radius[1]; j++) {
                    int yy = y + j;
                    if (0 <= yy && yy < ref->ny) {
                        for (int i = -kernel.radius[0]; i <= kernel.radius[0]; i++) {
                            int xx = x + i;
                            if (0 <= xx && xx < ref->nx) {
                                const double kernelValue = *kernelPtr;
                                const int index = (zz * ref->ny + yy) * ref->nx + xx;
                                meanRef += kernelValue * static_cast<double>(refPtr[index]);
                                meanFlo += kernelValue * static_cast<double>(floPtr[index]);
                                kernelSum += kernelValue;
                            }
                            kernelPtr++;
                        }
                    } else kernelPtr += kernel.size[0];
                }
            } else kernelPtr += kernel.size[0] * kernel.size[1];
        }
        return LocalStats(meanRef / kernelSum, meanFlo / kernelSum);
    }

    double GetLocalCC(const int x, const int y, const int z, const Kernel& kernel,
                      const NiftiImage& ref, const NiftiImage& flo, const LocalStats& means) {
        const float *kernelPtr = kernel.ptr.get();
        const auto refPtr = ref.data();
        const auto floPtr = flo.data();
        const auto [meanRef, meanFlo] = means;
        double varRef = 0, varFlo = 0, wdiff = 0, kernelSum = 0;
        for (int k = -kernel.radius[2]; k <= kernel.radius[2]; k++) {
            int zz = z + k;
            if (0 <= zz && zz < ref->nz) {
                for (int j = -kernel.radius[1]; j <= kernel.radius[1]; j++) {
                    int yy = y + j;
                    if (0 <= yy && yy < ref->ny) {
                        for (int i = -kernel.radius[0]; i <= kernel.radius[0]; i++) {
                            int xx = x + i;
                            if (0 <= xx && xx < ref->nx) {
                                const int index = (zz * ref->ny + yy) * ref->nx + xx;
                                const float refValue = refPtr[index];
                                const float floValue = floPtr[index];
                                const float kernelValue = *kernelPtr;
                                varRef += kernelValue * (refValue - meanRef) * (refValue - meanRef);
                                varFlo += kernelValue * (floValue - meanFlo) * (floValue - meanFlo);
                                wdiff += kernelValue * (refValue - meanRef) * (floValue - meanFlo);
                                kernelSum += kernelValue;
                            }
                            kernelPtr++;
                        }
                    } else kernelPtr += kernel.size[0];
                }

            } else kernelPtr += kernel.size[0] * kernel.size[1];
        }
        varRef /= kernelSum;
        varFlo /= kernelSum;
        wdiff /= kernelSum;
        return wdiff / (sqrt(varRef) * sqrt(varFlo));
    }
};

TEST_CASE_METHOD(LnccTest, "LNCC", "[unit][GetSimilarityMeasureValue]") {
    // Loop over all generated test cases
    for (auto&& testCase : testCases) {
        // Retrieve test information
        auto&& [testName, lncc, expLncc] = testCase;

        SECTION(testName) {
            NR_COUT << "\n**************** Section " << testName << " ****************" << std::endl;

            // Increase the precision for the output
            NR_COUT << std::fixed << std::setprecision(10);

            const double diff = abs(lncc - expLncc);
            if (diff > 0)
                NR_COUT << lncc << " " << expLncc << std::endl;
            REQUIRE(diff < EPS);
        }
    }
}

namespace {

// Compute the LNCC value for a reference/warped pair through the public measure API
double LnccValue(const NiftiImage& reference, const NiftiImage& warped, float sigma,
                 int *mask = nullptr, ConvKernelType kernelType = ConvKernelType::Gaussian) {
    NiftiImage ref(reference), flo(warped);
    Platform platform(PlatformType::Cpu);
    unique_ptr<DefContentCreator> creator{
        dynamic_cast<DefContentCreator*>(platform.CreateContentCreator(ContentType::Def)) };
    unique_ptr<DefContent> content{ creator->Create(ref, flo, nullptr, mask) };
    content->SetWarped(NiftiImage(warped));
    unique_ptr<MeasureCreator> measureCreator{ platform.CreateMeasureCreator() };
    unique_ptr<reg_lncc> measure{ dynamic_cast<reg_lncc*>(measureCreator->Create(MeasureType::Lncc)) };
    measure->SetKernelStandardDeviation(0, sigma);
    measure->SetKernelType(kernelType);
    measure->SetTimePointWeight(0, 1.0);
    measureCreator->Initialise(*measure, *content);
    return measure->GetSimilarityMeasureValue();
}

NiftiImage MakeLnccImage(bool is3D, unsigned seed) {
    std::vector<NiftiImage::dim_t> dims(is3D ? 3 : 2, 16);
    NiftiImage img(dims, NIFTI_TYPE_FLOAT32);
    setIdentitySform(img);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> distr(0.f, 1.f);
    auto ptr = img.data();
    for (size_t i = 0; i < img.nVoxels(); ++i)
        ptr[i] = distr(gen);
    return img;
}

} // namespace

TEST_CASE("LNCC is invariant to affine intensity rescaling", "[unit]") {
    /*
        The defining property of a correlation measure, and one no reference implementation is needed
        for: replacing the reference by a*ref + b rescales the local means and deviations by the same
        factors the covariance gains, so the value must not move (beyond convolution rounding on the
        rescaled intensities). A negative `a` flips the correlation's sign, which the measure's
        absolute value absorbs.
    */
    for (const bool is3D : { false, true })
        for (const float a : { 2.5f, -1.5f }) {
            SECTION(std::string(is3D ? "3D" : "2D") + " a=" + std::to_string(a)) {
                const NiftiImage reference = MakeLnccImage(is3D, 1);
                const NiftiImage warped = MakeLnccImage(is3D, 2);
                NiftiImage rescaled(reference, NiftiImage::Copy::Image);
                {
                    auto ptr = rescaled.data();
                    for (size_t i = 0; i < rescaled.nVoxels(); ++i)
                        ptr[i] = a * static_cast<float>(ptr[i]) + 10.f;
                }
                const double plain = LnccValue(reference, warped, -3.f);
                const double scaled = LnccValue(rescaled, warped, -3.f);
                NR_COUT << "  " << (is3D ? "3D" : "2D") << " a=" << a << ": " << std::fixed
                        << std::setprecision(10) << plain << " vs " << scaled << std::endl;
                REQUIRE(plain > 0.01);   // vacuity: the measure saw real structure
                REQUIRE(std::abs(plain - scaled) < 1e-5);
            }
        }
}

TEST_CASE("LNCC excludes NaN voxels exactly as masked voxels", "[unit]") {
    // The combined mask drops a voxel when any input holds NaN there; dropping the same voxels via
    // the reference mask must give the same value, or padded and masked runs would score differently
    for (const bool is3D : { false, true }) {
        SECTION(is3D ? "3D" : "2D") {
            const NiftiImage reference = MakeLnccImage(is3D, 3);
            const NiftiImage warped = MakeLnccImage(is3D, 4);
            NiftiImage warpedWithNan(warped, NiftiImage::Copy::Image);
            std::vector<int> mask(reference.nVoxelsPerVolume(), 0);
            {
                auto ptr = warpedWithNan.data();
                for (size_t i = 0; i < warpedWithNan.nVoxels(); ++i)
                    if (i % 7 == 0) {
                        ptr[i] = std::numeric_limits<float>::quiet_NaN();
                        mask[i] = -1;
                    }
            }
            const double viaNan = LnccValue(reference, warpedWithNan, -3.f);
            const double viaMask = LnccValue(reference, warped, -3.f, mask.data());
            const double unrestricted = LnccValue(reference, warped, -3.f);
            NR_COUT << "  " << (is3D ? "3D" : "2D") << ": NaN " << std::fixed << std::setprecision(10)
                    << viaNan << ", mask " << viaMask << ", neither " << unrestricted << std::endl;
            REQUIRE(viaNan != unrestricted);   // the exclusion did something
            REQUIRE(viaNan == viaMask);        // and both routes agree exactly
        }
    }
}

TEST_CASE("LNCC of a constant image is excluded by the zero-variance guard", "[unit]") {
    /*
        A constant reference has zero local variance everywhere, so every voxel's correlation is
        0/0; production skips non-finite voxels (lncc == lncc && !isinf) and divides by the count of
        those that survived. With NO surviving voxel that is 0/0 again - the returned value is NaN,
        and that IS the current contract, pinned here so a change to the guard is a decision. The
        objective function is protected upstream (a constant image inside the mask does not occur in
        a real registration's overlap), but any caller feeding a flat region a small kernel should
        know the measure can return NaN rather than 0.
    */
    for (const bool is3D : { false, true }) {
        SECTION(is3D ? "3D" : "2D") {
            NiftiImage reference = MakeLnccImage(is3D, 5);
            { auto ptr = reference.data(); for (size_t i = 0; i < reference.nVoxels(); ++i) ptr[i] = 1.f; }
            const NiftiImage warped = MakeLnccImage(is3D, 6);
            const double value = LnccValue(reference, warped, -3.f);
            NR_COUT << "  " << (is3D ? "3D" : "2D") << " constant reference: " << value << std::endl;
            REQUIRE(std::isnan(value));
        }
    }
}

TEST_CASE("LNCC with a box kernel keeps the closed-form properties", "[unit]") {
    // The mean (box) kernel is the -lnccMean variant: same measure, different smoothing. The
    // identical-image and intensity-invariance closed forms hold for any kernel.
    for (const bool is3D : { false, true }) {
        SECTION(is3D ? "3D" : "2D") {
            const NiftiImage reference = MakeLnccImage(is3D, 7);
            const double same = LnccValue(reference, reference, -3.f, nullptr, ConvKernelType::Mean);
            NR_COUT << "  " << (is3D ? "3D" : "2D") << " box kernel, same image: " << std::fixed
                    << std::setprecision(10) << same << std::endl;
            REQUIRE(std::abs(same - 1.0) < 1e-5);
        }
    }
}
