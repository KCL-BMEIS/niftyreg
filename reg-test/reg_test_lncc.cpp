// OpenCL and CUDA are not supported for this test yet
#undef _USE_OPENCL
#undef _USE_CUDA

#include "reg_test_common.h"
#include "_reg_lncc.h"

/*
    This test file contains the following unit tests:
    test function: LNCC computation and its voxel wise gradient
    In 2D and 3D
*/

class LNCCTest {
    /*
    Class to compute the LNCC between two values without any convolution
    Will take some time, don't judge me!!
    */
public:
    LNCCTest() {
        if (!testCases.empty())
            return;

        // Create a random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> distr(0, 1);

        // Create a reference and floating 2D images
        vector<NiftiImage::dim_t> dim{ 16, 16 };
        reference2d = NiftiImage(dim, NIFTI_TYPE_FLOAT32);
        floating2d = NiftiImage(dim, NIFTI_TYPE_FLOAT32);

        // Create a reference 3D image
        dim.push_back(16);
        reference3d = NiftiImage(dim, NIFTI_TYPE_FLOAT32);
        floating3d = NiftiImage(dim, NIFTI_TYPE_FLOAT32);

        // Create corresponding identify control point grids
        cpp2d = CreateControlPointGrid(reference2d);
        cpp3d = CreateControlPointGrid(reference3d);

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

        // Create the object to compute the expected values
        vector<TestData> testData;
        this->_ref = reference2d;
        this->_flo = floating2d;
        testData.emplace_back(TestData(
            "LNCC 2D -1",
            std::move(NiftiImage(reference2d)),
            std::move(NiftiImage(floating2d)),
            std::move(NiftiImage(cpp2d)),
            -1.f,
            this->GetLNCCNoConv(1)
        ));
        testData.emplace_back(TestData(
            "LNCC 2D -1 same image",
            std::move(NiftiImage(reference2d)),
            std::move(NiftiImage(reference2d)),
            std::move(NiftiImage(cpp2d)),
            -1.f,
            1.f
        ));
        testData.emplace_back(TestData(
            "LNCC 2D -5",
            std::move(NiftiImage(reference2d)),
            std::move(NiftiImage(floating2d)),
            std::move(NiftiImage(cpp2d)),
            -5.f,
            this->GetLNCCNoConv(5)
        ));
        testData.emplace_back(TestData(
            "LNCC 2D -5 same image",
            std::move(NiftiImage(reference2d)),
            std::move(NiftiImage(reference2d)),
            std::move(NiftiImage(cpp2d)),
            -5.f,
            1.f
        ));
        reg_tools_multiplyValueToImage(reference2d, floating2d, -1.f);
        testData.emplace_back(TestData(
            "LNCC 2D -1 same image negated",
            std::move(NiftiImage(reference2d)),
            std::move(NiftiImage(floating2d)),
            std::move(NiftiImage(cpp2d)),
            -1.f,
            1.f
        ));
        testData.emplace_back(TestData(
            "LNCC 2D -5 same image negated",
            std::move(NiftiImage(reference2d)),
            std::move(NiftiImage(floating2d)),
            std::move(NiftiImage(cpp2d)),
            -5.f,
            1.f
        ));
        this->_ref = reference3d;
        this->_flo = floating3d;
        testData.emplace_back(TestData(
            "LNCC 3D -1",
            std::move(NiftiImage(reference3d)),
            std::move(NiftiImage(floating3d)),
            std::move(NiftiImage(cpp3d)),
            -1.f,
            this->GetLNCCNoConv(1)
        ));
        testData.emplace_back(TestData(
            "LNCC 3D -1 same image",
            std::move(NiftiImage(reference3d)),
            std::move(NiftiImage(reference3d)),
            std::move(NiftiImage(cpp3d)),
            -1.f,
            1.f
        ));
        testData.emplace_back(TestData(
            "LNCC 3D -5",
            std::move(NiftiImage(reference3d)),
            std::move(NiftiImage(floating3d)),
            std::move(NiftiImage(cpp3d)),
            -5.f,
            this->GetLNCCNoConv(5)
        ));
        testData.emplace_back(TestData(
            "LNCC 3D -5 same image",
            std::move(NiftiImage(reference3d)),
            std::move(NiftiImage(reference3d)),
            std::move(NiftiImage(cpp3d)),
            -5.f,
            1.f
        ));
        reg_tools_multiplyValueToImage(reference3d, floating3d, -1.f);
        testData.emplace_back(TestData(
            "LNCC 3D -1 same image negated",
            std::move(NiftiImage(reference3d)),
            std::move(NiftiImage(floating3d)),
            std::move(NiftiImage(cpp3d)),
            -1.f,
            1.f
        ));
        testData.emplace_back(TestData(
            "LNCC 3D -5 same image negated",
            std::move(NiftiImage(reference3d)),
            std::move(NiftiImage(floating3d)),
            std::move(NiftiImage(cpp3d)),
            -5.f,
            1.f
        ));
        for (auto&& data : testData) {
            for (auto&& platformType : PlatformTypes) {
                shared_ptr<Platform> platform{ new Platform(platformType) };
                // Make a copy of the test data
                auto td = data;
                auto&& [testName, reference, floating, cpp, sigma, result] = td;
                // Create content creator
                unique_ptr<F3dContentCreator> contentCreator{
                    dynamic_cast<F3dContentCreator*>(platform->CreateContentCreator(ContentType::F3d))
                };
                // Create the content
                unique_ptr<F3dContent> content{ contentCreator->Create(reference, floating, cpp) };
                // Initialise the warped image using nearest neigh interpolation
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

    ~LNCCTest() {
        if (this->_kernel != nullptr) delete[] this->_kernel;
    }

protected:
    NiftiImage reference2d;
    NiftiImage reference3d;
    NiftiImage floating2d;
    NiftiImage floating3d;
    NiftiImage cpp2d;
    NiftiImage cpp3d;
    nifti_image *_ref = nullptr;
    nifti_image *_flo = nullptr;
    float *_kernel = nullptr;
    float _kernelStdVoxel=5;
    int _kernel_radius[3];
    int _kernel_size[3];
    using LocalStats = std::tuple<float, float>;
    using TestData = std::tuple<std::string, NiftiImage, NiftiImage, NiftiImage, int, float>;
    using TestCase = std::tuple<unique_ptr<Content>, unique_ptr<reg_lncc>, shared_ptr<Platform>, TestData>;

    inline static vector<TestCase> testCases;

    float GetLNCCNoConv(int kernelStd) {
        double lncc_value = 0;
        // Compute the kernel
        this->_kernelStdVoxel = fabs(kernelStd);
        this->InitialiseKernel();
        float lncc = 0;
        float voxelNumber = 0;
        for (int z = 0; z < this->_ref->nz; ++z) {
            for (int y = 0; y < this->_ref->ny; ++y) {
                for (int x = 0; x < this->_ref->nx; ++x) {
                    lncc += fabs(this->GetLocalCC(x, y, z, this->GetLocalMeans(x, y, z)));
                    voxelNumber++;
                }
            }
        }
        return lncc / voxelNumber;
    }

    void InitialiseKernel() {
        if (this->_kernel != nullptr) {
            delete[] this->_kernel;
        }
        this->_kernel_radius[0] = 3 * this->_kernelStdVoxel;
        this->_kernel_radius[1] = 3 * this->_kernelStdVoxel;
        this->_kernel_radius[2] = 0;
        if (this->_ref->ndim > 2)
            this->_kernel_radius[2] = 3 * this->_kernelStdVoxel;
        this->_kernel_size[0] = this->_kernel_radius[0] * 2 + 1;
        this->_kernel_size[1] = this->_kernel_radius[1] * 2 + 1;
        this->_kernel_size[2] = this->_kernel_radius[2] * 2 + 1;
        this->_kernel = new float[this->_kernel_size[0] *
            this->_kernel_size[1] *
            this->_kernel_size[2]];
        float *kernelPtr = this->_kernel;

        for (int z = -this->_kernel_radius[2]; z <= this->_kernel_radius[2]; z++) {
            float z_value = static_cast<float>(
                exp(-(z * z) / (2.0 * reg_pow2(this->_kernelStdVoxel))) /
                (this->_kernelStdVoxel * 2.506628274631)
                );
            for (int y = -this->_kernel_radius[1]; y <= this->_kernel_radius[1]; y++) {
                float y_value = static_cast<float>(
                    exp(-(y * y) / (2.0 * reg_pow2(this->_kernelStdVoxel))) /
                    (this->_kernelStdVoxel * 2.506628274631)
                    );
                for (int x = -this->_kernel_radius[0]; x <= this->_kernel_radius[0]; x++) {
                    float x_value = static_cast<float>(
                        exp(-(x * x) / (2.0 * reg_pow2(this->_kernelStdVoxel))) /
                        (this->_kernelStdVoxel * 2.506628274631)
                        );
                    *kernelPtr++ = x_value * y_value * z_value;
                }
            }
        }
    }

    LocalStats GetLocalMeans(int x, int y, int z) {
        double mean_ref = 0.;
        double mean_flo = 0.;
        double sum_kernel = 0.;
        float *kernelPtr = this->_kernel;
        float *refPtr = static_cast<float *>(this->_ref->data);
        float *floPtr = static_cast<float *>(this->_flo->data);
        for (int k = -this->_kernel_radius[2]; k <= this->_kernel_radius[2]; k++) {
            int zz = z + k;
            if (0 <= zz && zz < this->_ref->nz) {
                for (int j = -this->_kernel_radius[1]; j <= this->_kernel_radius[1]; j++) {
                    int yy = y + j;
                    if (0 <= yy && yy < this->_ref->ny) {
                        for (int i = -this->_kernel_radius[0]; i <= this->_kernel_radius[0]; i++) {
                            int xx = x + i;
                            if (0 <= xx && xx < this->_ref->nx) {
                                double kernelValue = *kernelPtr;
                                int index = (zz * this->_ref->ny + yy) * this->_ref->nx + xx;
                                mean_ref += kernelValue * refPtr[index];
                                mean_flo += kernelValue * floPtr[index];
                                sum_kernel += kernelValue;
                            }
                            kernelPtr++;
                        }
                    } else kernelPtr += this->_kernel_size[0];
                }
            } else kernelPtr += this->_kernel_size[0] * this->_kernel_size[1];
        }
        return LocalStats(mean_ref / sum_kernel, mean_flo / sum_kernel);
    }

    float GetLocalCC(int x, int y, int z, LocalStats means) {
        float *kernelPtr = this->_kernel;
        float *refPtr = static_cast<float *>(this->_ref->data);
        float *floPtr = static_cast<float *>(this->_flo->data);
        auto &&[mean_ref, mean_flo] = means;
        double var_ref = 0.;
        double var_flo = 0.;
        double wdiff = 0.;
        double sum_kernel = 0.;
        for (int k = -this->_kernel_radius[2]; k <= this->_kernel_radius[2]; k++) {
            int zz = z + k;
            if (0 <= zz && zz < this->_ref->nz) {
                for (int j = -this->_kernel_radius[1]; j <= this->_kernel_radius[1]; j++) {
                    int yy = y + j;
                    if (0 <= yy && yy < this->_ref->ny) {
                        for (int i = -this->_kernel_radius[0]; i <= this->_kernel_radius[0]; i++) {
                            int xx = x + i;
                            if (0 <= xx && xx < this->_ref->nx) {
                                int index = (zz * this->_ref->ny + yy) * this->_ref->nx + xx;
                                float refValue = refPtr[index];
                                float floValue = floPtr[index];
                                float kernelValue = *kernelPtr;
                                var_ref += kernelValue * (refValue - mean_ref) * (refValue - mean_ref);
                                var_flo += kernelValue * (floValue - mean_flo) * (floValue - mean_flo);
                                wdiff += kernelValue * (refValue - mean_ref) * (floValue - mean_flo);
                                sum_kernel += kernelValue;
                            }
                            kernelPtr++;
                        }
                    } else kernelPtr += this->_kernel_size[0];
                }

            } else kernelPtr += this->_kernel_size[0] * this->_kernel_size[1];
        }
        var_ref /= sum_kernel;
        var_flo /= sum_kernel;
        wdiff /= sum_kernel;
        return wdiff / (sqrtf(var_ref) * sqrtf(var_flo));
    }
};

TEST_CASE_METHOD(LNCCTest, "LNCC", "[GetSimilarityMeasureValue]") {
    // Loop over all generated test cases
    for (auto&& testCase : this->testCases) {
        // Retrieve test information
        auto&& [content, measure, platform, testData] = testCase;
        auto&& [testName, reference, floating, cpp, sigma, value] = testData;

        SECTION(testName) {
            float lncc = measure->GetSimilarityMeasureValue();
            std::cout << lncc << " " << value << std::endl;
            REQUIRE(fabs(lncc - value) < EPS);
            content.reset();
        }
    }
}