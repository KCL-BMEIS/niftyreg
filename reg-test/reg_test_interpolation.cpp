// OpenCL is not supported for this test
#undef _USE_OPENCL

#include "_reg_ReadWriteMatrix.h"
#include "_reg_tools.h"

#include "Kernel.h"
#include "ResampleImageKernel.h"
#include "Platform.h"
#include "AladinContent.h"

#include <list>
#include <catch2/catch_test_macros.hpp>

#define EPS_SINGLE 0.0001

/*
    This test file contains the following unit tests:
    test function: image resampling
    In 2D and 3D
    linear
    cubic
*/


typedef std::tuple<std::string, nifti_image*, nifti_image*, int, float*> TestData;
typedef std::tuple<unique_ptr<AladinContent>, unique_ptr<Platform>> ContentDesc;

template <typename T>
void interpCubicSplineKernel(T relative, T (&basis)[4]) {
    if (relative < 0) relative = 0; //reg_rounding error
    const T relative2 = relative * relative;
    basis[0] = (relative * ((2.f - relative) * relative - 1.f)) / 2.f;
    basis[1] = (relative2 * (3.f * relative - 5.f) + 2.f) / 2.f;
    basis[2] = (relative * ((4.f - 3.f * relative) * relative + 1.f)) / 2.f;
    basis[3] = (relative - 1.f) * relative2 / 2.f;
}

TEST_CASE("Resampling", "[resampling]") {
    // Create a reference 2D image
    int dimFlo[8] = { 2, 4, 4, 1, 1, 1, 1, 1 };
    nifti_image *reference2d = nifti_make_new_nim(dimFlo, NIFTI_TYPE_FLOAT32, true);
    reg_checkAndCorrectDimension(reference2d);

    // Fill image with distance from identity
    auto *ref2dPtr = static_cast<float*>(reference2d->data);
    for (auto y = 0; y < reference2d->ny; ++y) {
        for (auto x = 0; x < reference2d->nx; ++x) {
            *ref2dPtr = sqrtf(float(x * x) + float(y * y));
            ref2dPtr++;
        }
    }

    // Create a corresponding 2D deformation field
    int dimDef[8] = { 5, 1, 1, 1, 1, 2, 1, 1 };
    nifti_image *deformationField2d = nifti_make_new_nim(dimDef, NIFTI_TYPE_FLOAT32, true);
    reg_checkAndCorrectDimension(deformationField2d);
    auto *def2dPtr = static_cast<float*>(deformationField2d->data);
    def2dPtr[0] = 1.2f;
    def2dPtr[1] = 1.3f;

    // Create a reference 3D image
    dimFlo[0] = 3; dimFlo[3] = 4;
    nifti_image *reference3d = nifti_make_new_nim(dimFlo, NIFTI_TYPE_FLOAT32, true);
    reg_checkAndCorrectDimension(reference3d);

    // Fill image with distance from identity
    auto *ref3dPtr = static_cast<float*>(reference3d->data);
    for (auto z = 0; z < reference3d->nz; ++z) {
        for (auto y = 0; y < reference3d->ny; ++y) {
            for (auto x = 0; x < reference3d->nx; ++x) {
                *ref3dPtr = sqrtf(float(x * x) + float(y * y) + float(z * z));
                ref3dPtr++;
            }
        }
    }

    // Create a corresponding 3D deformation field
    dimDef[5] = 3;
    nifti_image *deformationField3d = nifti_make_new_nim(dimDef, NIFTI_TYPE_FLOAT32, true);
    reg_checkAndCorrectDimension(deformationField3d);
    auto *def3dPtr = static_cast<float*>(deformationField3d->data);
    def3dPtr[0] = 1.2f;
    def3dPtr[1] = 1.3f;
    def3dPtr[2] = 1.4f;

    // Generate the different use cases
    std::vector<TestData> testCases;

    // Linear interpolation - 2D
    // coordinate in image: [1.2, 1.3]
    float resLinear2d[1] = {0};
    ref2dPtr = static_cast<float*>(reference2d->data);
    for (int y = 1; y <= 2; ++y) {
        for (int x = 1; x <= 2; ++x) {
            resLinear2d[0] += ref2dPtr[y * dimFlo[1] + x] *
                abs(2.0f - (float)x - 0.2f) *
                abs(2.0f - (float)y - 0.3f);
        }
    }
    // create the test case
    testCases.emplace_back(TestData(
        "Linear 2D",
        reference2d,
        deformationField2d,
        1,
        resLinear2d)
    );

    // Nearest neighbour interpolation - 2D
    // coordinate in image: [1.2, 1.3]
    float resNearest2d[1];
    resNearest2d[0] = ref2dPtr[1 * dimFlo[1] + 1];
    // create the test case
    testCases.emplace_back(TestData(
        "Nearest Neighbour 2D",
        reference2d,
        deformationField2d,
        0,
        resNearest2d)
    );

    // Cubic spline interpolation - 2D
    // coordinate in image: [1.2, 1.3]
    float resCubic2d[1] = {0};
    float xBasis[4], yBasis[4];
    interpCubicSplineKernel(0.2f, xBasis);
    interpCubicSplineKernel(0.3f, yBasis);
    for (int y = 0; y <= 3; ++y) {
        float resX = 0;
        for (int x = 0; x <= 3; ++x) {
            resX += ref2dPtr[y * dimFlo[1] + x] * xBasis[x];
        }
        resCubic2d[0] += resX * yBasis[y];
    }

    // create the test case
    testCases.emplace_back(TestData(
        "Cubic Spline 2D",
        reference2d,
        deformationField2d,
        3,
        resCubic2d)
    );

    // Linear interpolation - 3D
    // coordinate in image: [1.2, 1.3, 1.4]
    float resLinear3d[1] = {0};
    ref3dPtr = static_cast<float*>(reference3d->data);
    for (int z = 1; z <= 2; ++z) {
        for (int y = 1; y <= 2; ++y) {
            for (int x = 1; x <= 2; ++x) {
                resLinear3d[0] += ref3dPtr[z * dimFlo[1] * dimFlo[2] + y * dimFlo[1] + x] *
                    abs(2.0f - (float)x - 0.2f) *
                    abs(2.0f - (float)y - 0.3f) *
                    abs(2.0f - (float)z - 0.4f);
            }
        }
    }

    // create the test case
    testCases.emplace_back(TestData(
        "Linear 3D",
        reference3d,
        deformationField3d,
        1,
        resLinear3d)
    );

    // Nearest neighbour interpolation - 3D
    // coordinate in image: [1.2, 1.3, 1.4]
    float resNearest3d[1];
    resNearest3d[0] = ref3dPtr[1 * dimFlo[2] * dimFlo[1] + 1 * dimFlo[1] + 1];
    // create the test case
    testCases.emplace_back(TestData(
        "Nearest Neighbour 3D",
        reference3d,
        deformationField3d,
        0,
        resNearest3d)
    );

    // Cubic spline interpolation - 3D
    // coordinate in image: [1.2, 1.3, 1.4]
    float resCubic3d[1] = {0};
    float zBasis[4];
    interpCubicSplineKernel(0.4f, zBasis);
    for (int z = 0; z <= 3; ++z) {
        float resY = 0;
        for (int y = 0; y <= 3; ++y) {
            float resX = 0;
            for (int x = 0; x <= 3; ++x) {
                resX += ref3dPtr[z * dimFlo[1] * dimFlo[2] + y * dimFlo[1] + x] * xBasis[x];
            }
            resY += resX * yBasis[y];
        }
        resCubic3d[0] += resY * zBasis[z];
    }

    // create the test case
    testCases.emplace_back(TestData(
        "Cubic Spline 3D",
        reference3d,
        deformationField3d,
        3,
        resCubic3d)
    );

    // Loop over all generated test cases to create all content and run all tests
    for (auto&& testCase : testCases) {
        // Retrieve test information
        auto&& [testName, reference, defField, interp, testResult] = testCase;

        // Accumulate all required contents with a vector
        std::vector<ContentDesc> contentDescs;
        for (auto&& platformType : PlatformTypes) {
            unique_ptr<Platform> platform{ new Platform(platformType) };
            unique_ptr<AladinContentCreator> contentCreator{ dynamic_cast<AladinContentCreator*>(platform->CreateContentCreator(ContentType::Aladin)) };
            unique_ptr<AladinContent> content{ contentCreator->Create(reference, reference) };
            contentDescs.push_back(ContentDesc(std::move(content), std::move(platform)));
        }

        // Loop over all possibles contents for each test
        for (auto&& contentDesc : contentDescs) {
            auto&& [content, platform] = contentDesc;
            SECTION(testName + " " + platform->GetName()) {
                // Create and set a warped image to host the computation
                nifti_image *warped = nifti_copy_nim_info(defField);
                warped->ndim = warped->dim[0] = defField->nu;
                warped->dim[1] = warped->nx = 1;
                warped->dim[2] = warped->ny = 1;
                warped->dim[3] = warped->nz = 1;
                warped->dim[5] = warped->nu = 1;
                warped->nvox = CalcVoxelNumber(*warped, warped->ndim);
                warped->data = malloc(warped->nvox * warped->nbyper);
                content->SetWarped(warped);
                // Set the deformation field
                content->SetDeformationField(defField);
                // Initialise the platform to run current content and retrieve deformation field
                unique_ptr<Kernel> resampleKernel{ platform->CreateKernel(ResampleImageKernel::GetName(), content.get()) };
                // args = interpolation and padding

                resampleKernel->castTo<ResampleImageKernel>()->Calculate(interp, 0);
                warped = content->GetWarped();

                // Check all values
                auto *warpedPtr = static_cast<float*>(warped->data);
                for (size_t i = 0; i < warped->nvox; ++i) {
                    std::cout << i << " " << warpedPtr[i] << " " << testResult[i] << std::endl;
                    REQUIRE(fabs(warpedPtr[i] - testResult[i]) < EPS_SINGLE);
                }
            }
        }
    }
    // Only free-ing ref as the rest if cleared by content destructor
    nifti_image_free(reference2d);
    nifti_image_free(reference3d);
}
