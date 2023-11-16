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
                content->SetWarped(floating.disown());
                // Create the measure
                unique_ptr<Measure> measure{ platform->CreateMeasure() };
                // Use LNCC as a measure
                unique_ptr<reg_lncc> measure_lncc{ dynamic_cast<reg_lncc*>(measure->Create(MeasureType::Lncc)) };
                measure_lncc->SetKernelStandardDeviation(0, sigma);
                measure_lncc->SetTimePointWeight(0, 1.0); // weight initially set to default value of 1.0
                measure->Initialise(*measure_lncc, *content);
                const double lncc = measure_lncc->GetSimilarityMeasureValue();
                // Save for testing
                testCases.push_back({ testName, lncc, expLncc });
            }
        }
    }

protected:
    struct Kernel {
        unique_ptr<float> ptr;
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
        kernel.ptr = unique_ptr<float>(new float[kernel.size[0] * kernel.size[1] * kernel.size[2]]);
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
