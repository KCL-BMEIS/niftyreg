// OpenCL and CUDA are not supported for this test yet
#undef _USE_OPENCL
#undef _USE_CUDA

#include "reg_test_common.h"
#include "_reg_lncc.h"

/*
    This test file contains the following unit tests:
    test function: LNCC computation and its voxel-wise gradient
    In 2D and 3D
*/

class LNCCTest {
public:
    LNCCTest() {
        if (!testCases.empty())
            return;

        // Create a random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> distr(0, 1);

        // Create reference and floating 2D images
        vector<NiftiImage::dim_t> dim{ 16, 16 };
        NiftiImage reference2d(dim, NIFTI_TYPE_FLOAT32);
        NiftiImage floating2d(dim, NIFTI_TYPE_FLOAT32);

        // Create reference and floating 3D images
        dim.push_back(16);
        NiftiImage reference3d(dim, NIFTI_TYPE_FLOAT32);
        NiftiImage floating3d(dim, NIFTI_TYPE_FLOAT32);

        // Create corresponding identify control point grids
        NiftiImage cpp2d(CreateControlPointGrid(reference2d));
        NiftiImage cpp3d(CreateControlPointGrid(reference3d));

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
            cpp2d,
            -1.f,
            GetLNCCNoConv(1, reference2d, floating2d)
        ));
        testData.emplace_back(TestData(
            "LNCC 2D -1 same image",
            reference2d,
            reference2d,
            cpp2d,
            -1.f,
            1.0
        ));
        testData.emplace_back(TestData(
            "LNCC 2D -5",
            reference2d,
            floating2d,
            cpp2d,
            -5.f,
            GetLNCCNoConv(5, reference2d, floating2d)
        ));
        testData.emplace_back(TestData(
            "LNCC 2D -5 same image",
            reference2d,
            reference2d,
            cpp2d,
            -5.f,
            1.0
        ));
        reg_tools_multiplyValueToImage(reference2d, floating2d, -1.f);
        testData.emplace_back(TestData(
            "LNCC 2D -1 same image negated",
            reference2d,
            floating2d,
            cpp2d,
            -1.f,
            1.0
        ));
        testData.emplace_back(TestData(
            "LNCC 2D -5 same image negated",
            reference2d,
            floating2d,
            cpp2d,
            -5.f,
            1.0
        ));
        testData.emplace_back(TestData(
            "LNCC 3D -1",
            reference3d,
            floating3d,
            cpp3d,
            -1.f,
            GetLNCCNoConv(1, reference3d, floating3d)
        ));
        testData.emplace_back(TestData(
            "LNCC 3D -1 same image",
            reference3d,
            reference3d,
            cpp3d,
            -1.f,
            1.0
        ));
        testData.emplace_back(TestData(
            "LNCC 3D -5",
            reference3d,
            floating3d,
            cpp3d,
            -5.f,
            GetLNCCNoConv(5, reference3d, floating3d)
        ));
        testData.emplace_back(TestData(
            "LNCC 3D -5 same image",
            reference3d,
            reference3d,
            cpp3d,
            -5.f,
            1.0
        ));
        reg_tools_multiplyValueToImage(reference3d, floating3d, -1.f);
        testData.emplace_back(TestData(
            "LNCC 3D -1 same image negated",
            reference3d,
            floating3d,
            cpp3d,
            -1.f,
            1.0
        ));
        testData.emplace_back(TestData(
            "LNCC 3D -5 same image negated",
            reference3d,
            floating3d,
            cpp3d,
            -5.f,
            1.0
        ));
        for (auto&& data : testData) {
            for (auto&& platformType : PlatformTypes) {
                // Create the platform
                shared_ptr<Platform> platform{ new Platform(platformType) };
                // Make a copy of the test data
                auto td = data;
                auto&& [testName, reference, floating, cpp, sigma, result] = td;
                // Create the content creator
                unique_ptr<F3dContentCreator> contentCreator{
                    dynamic_cast<F3dContentCreator*>(platform->CreateContentCreator(ContentType::F3d))
                };
                // Create the content
                unique_ptr<F3dContent> content{ contentCreator->Create(reference, floating, cpp) };
                // Initialise the warped image using the nearest-neighbour interpolation
                unique_ptr<Compute> compute{ platform->CreateCompute(*content) };
                compute->ResampleImage(0, 0);
                content->SetWarped(floating.disown());
                // Create the measure
                unique_ptr<Measure> measure{ platform->CreateMeasure() };
                // Use LNCC as a measure
                unique_ptr<reg_lncc> measure_lncc{ dynamic_cast<reg_lncc*>(measure->Create(MeasureType::Lncc)) };
                measure_lncc->SetKernelStandardDeviation(0, sigma);
                measure_lncc->SetTimepointWeight(0, 1.0); // weight initially set to default value of 1.0
                measure->Initialise(*measure_lncc, *content);

                testCases.push_back({ std::move(content), std::move(measure_lncc), platform, std::move(td) });
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
    using TestData = std::tuple<std::string, NiftiImage, NiftiImage, NiftiImage, float, double>;
    using TestCase = std::tuple<unique_ptr<Content>, unique_ptr<reg_lncc>, shared_ptr<Platform>, TestData>;
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

    Kernel InitialiseKernel(const NiftiImage& ref, const float& kernelStdVoxel) {
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
                exp(-(z * z) / (2.0 * reg_pow2(kernelStdVoxel))) / (kernelStdVoxel * 2.506628274631));
            for (int y = -kernel.radius[1]; y <= kernel.radius[1]; y++) {
                const float y_value = static_cast<float>(
                    exp(-(y * y) / (2.0 * reg_pow2(kernelStdVoxel))) / (kernelStdVoxel * 2.506628274631));
                for (int x = -kernel.radius[0]; x <= kernel.radius[0]; x++) {
                    const float x_value = static_cast<float>(
                        exp(-(x * x) / (2.0 * reg_pow2(kernelStdVoxel))) / (kernelStdVoxel * 2.506628274631));
                    *kernelPtr++ = x_value * y_value * z_value;
                }
            }
        }

        return kernel;
    }

    LocalStats GetLocalMeans(const int& x, const int& y, const int& z, const Kernel& kernel,
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
                                const double& kernelValue = *kernelPtr;
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

    double GetLocalCC(const int& x, const int& y, const int& z, const Kernel& kernel,
                      const NiftiImage& ref, const NiftiImage& flo, const LocalStats& means) {
        const float *kernelPtr = kernel.ptr.get();
        const auto refPtr = ref.data();
        const auto floPtr = flo.data();
        const auto& [meanRef, meanFlo] = means;
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

TEST_CASE_METHOD(LNCCTest, "LNCC", "[GetSimilarityMeasureValue]") {
    // Loop over all generated test cases
    for (auto&& testCase : testCases) {
        // Retrieve test information
        auto&& [content, measure, platform, testData] = testCase;
        auto&& [testName, reference, floating, cpp, sigma, value] = testData;

        SECTION(testName) {
            std::cout << "\n**************** Section " << testName << " ****************" << std::endl;
            const double lncc = measure->GetSimilarityMeasureValue();
            std::cout << lncc << " " << value << std::endl;
            REQUIRE(fabs(lncc - value) < EPS);
        }
    }
}
