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


typedef std::tuple<std::string, nifti_image*, nifti_image*, float*> TestData;
typedef std::tuple<std::unique_ptr<AladinContent>, std::unique_ptr<Platform>> ContentDesc;

TEST_CASE("Resampling", "[resampling]") {
    // Create a reference 2D image
    int dim_flo[8] = { 2, 4, 4, 1, 1, 1, 1, 1 };
    nifti_image *reference2d = nifti_make_new_nim(dim_flo, NIFTI_TYPE_FLOAT32, true);
    reg_checkAndCorrectDimension(reference2d);

    // Fill image with distance from identity
    auto* ref2dPrt = static_cast<float*>(reference2d->data);
    for (auto y = 0; y < reference2d->ny; ++y) {
        for (auto x = 0; x < reference2d->nx; ++x) {
            *ref2dPrt = sqrtf(float(x * x) + float(y * y));
            ref2dPrt++;
        }
    }

    // Create a corresponding 2D deformation field
    int dim_def[8] = {5, 1, 1, 1, 1, 2, 1, 1};
    nifti_image *deformationField2D = nifti_make_new_nim(dim_def, NIFTI_TYPE_FLOAT32, true);
    reg_checkAndCorrectDimension(deformationField2D);
    auto* def2dPrt = static_cast<float*>(deformationField2D->data);
    def2dPrt[0] = 1.2;
    def2dPrt[1] = 1.3;

    // Create a reference 3D image
    dim_flo[0] = 3; dim_flo[3] = 4;
    nifti_image *reference3d = nifti_make_new_nim(dim_flo, NIFTI_TYPE_FLOAT32, true);
    reg_checkAndCorrectDimension(reference3d);

    // Fill image with distance from identity
    auto *ref3dPrt = static_cast<float*>(reference3d->data);
    for (auto z = 0; z < reference3d->nz; ++z) {
        for (auto y = 0; y < reference3d->ny; ++y) {
            for (auto x = 0; x < reference3d->nx; ++x) {
                *ref3dPrt = sqrtf(float(x * x) + float(y * y) + float(z * z));
                ref3dPrt++;
            }
        }
    }

    // Create a corresponding 2D deformation field
    dim_def[5] = 3;
    nifti_image *deformationField3D = nifti_make_new_nim(dim_def, NIFTI_TYPE_FLOAT32, true);
    reg_checkAndCorrectDimension(deformationField3D);
    auto* def3dPrt = static_cast<float*>(deformationField3D->data);
    def3dPrt[0] = 1.2;
    def3dPrt[1] = 1.3;
    def3dPrt[2] = 1.4;

    // Generate the different use cases
    std::vector<TestData> testCases;

    // Linear interpolation - 2D
    // coordinate in image: [1.2, 1.3]
    auto *res_linear_2d = new float[1];
    res_linear_2d[0] = 0;
    for (auto y=1; y<2; ++y){
        for (auto x=1; x<2; ++x){
            res_linear_2d[0] += ref2dPrt[y*dim_flo[1]+
                                         x] *
                                abs(2.0 - (float)x - 0.2) *
                                abs(2.0 - (float)y - 0.3);
        }
    }

    // create the test case
    testCases.emplace_back(TestData(
        "Linear 2D",
        reference2d,
        deformationField2D,
        res_linear_2d)
    );

    // Linear interpolation - 23D
    // coordinate in image: [1.2, 1.3, 1.4]
    auto *res_linear_3d = new float[1];
    res_linear_3d[0] = 0;
    for (auto z=1; z<2; ++z){
        for (auto y=1; y<2; ++y){
            for (auto x=1; x<2; ++x) {
                res_linear_3d[0] += ref2dPrt[z * dim_flo[1]* dim_flo[2] +
                                             y * dim_flo[1] +
                                             x] *
                                    abs(2.0 - (float) x - 0.2) *
                                    abs(2.0 - (float) y - 0.3) *
                                    abs(2.0 - (float) z - 0.4);
            }
        }
    }

    // create the test case
    testCases.emplace_back(TestData(
            "Linear 3D",
            reference3d,
            deformationField3D,
            res_linear_3d)
    );

    // Loop over all generated test cases to create all content and run all tests
    for (auto&& testCase : testCases) {
        // Retrieve test information
        auto&& [testName, reference, defField, testResult] = testCase;

        // Accumulate all required contents with a vector
        std::vector<ContentDesc> contentDescs;
        for (auto&& platformType : PlatformTypes) {
            std::unique_ptr<Platform> platform{ new Platform(platformType) };
            std::unique_ptr<AladinContentCreator> contentCreator{ dynamic_cast<AladinContentCreator*>(platform->CreateContentCreator(ContentType::Aladin)) };
            std::unique_ptr<AladinContent> content{ contentCreator->Create(reference, reference) };
            contentDescs.push_back(ContentDesc(std::move(content), std::move(platform)));
        }
        // Loop over all possibles contents for each test
        for (auto&& contentDesc : contentDescs) {
            auto&& [content, platform] = contentDesc;
            SECTION(testName + " " + platform->GetName()) {
                // Create and set a warped image to host the computation
                nifti_image *warped = nifti_copy_nim_info(reference);
                warped->data = malloc(warped->nvox * warped->nbyper);
                content->SetWarped(warped);
                // Set the deformation field
                content->SetDeformationField(defField);
                // Initialise the platform to run current content and retrieve deformation field
                std::unique_ptr<Kernel> resampleKernel{ platform->CreateKernel(ResampleImageKernel::GetName(), content.get()) };
                // args = interpolation and padding
                std::list<int> interp = { 0, 1, 3 };
                for (auto it : interp) {
                    resampleKernel->castTo<ResampleImageKernel>()->Calculate(it, 0);
                    warped = content->GetWarped();

                    // Check all values
                    auto *warpedPtr = static_cast<float*>(warped->data);
                    for (size_t i = 0; i < CalcVoxelNumber(*warped); ++i) {
                        std::cout << i << " " << static_cast<float*>(reference->data)[i] << " " << warpedPtr[i] << " " << testResult[i] << std::endl;
                        REQUIRE(fabs(warpedPtr[i] - testResult[i]) < EPS_SINGLE);
                    }
                }
            }
        }
    }
    // Only free-ing ref as the rest if cleared by content destructor
    nifti_image_free(reference2d);
    nifti_image_free(reference3d);
}
