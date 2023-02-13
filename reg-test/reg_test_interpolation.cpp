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
    identity
    translation
    affine
*/


typedef std::tuple<std::string, nifti_image*, nifti_image*, float*> TestData;
typedef std::tuple<std::unique_ptr<AladinContent>, std::unique_ptr<Platform>> ContentDesc;

TEST_CASE("Resampling", "[resampling]") {
    // Create a reference 2D image
    int dim[8] = { 2, 2, 2, 1, 1, 1, 1, 1 };
    nifti_image *reference2d = nifti_make_new_nim(dim, NIFTI_TYPE_FLOAT32, true);
    reg_checkAndCorrectDimension(reference2d);

    // Fill image with distance from identity
    auto* ref2dPrt = static_cast<float*>(reference2d->data);
    for (float y = 0; y < reference2d->ny; ++y) {
        for (float x = 0; x < reference2d->nx; ++x) {
            *ref2dPrt = sqrtf(x * x + y * y);
            ref2dPrt++;
        }
    }

    // Create a corresponding deformation field

    // Create a reference 3D image
    dim[0] = 3; dim[3] = 2;
    nifti_image *reference3d = nifti_make_new_nim(dim, NIFTI_TYPE_FLOAT32, true);
    reg_checkAndCorrectDimension(reference3d);

    // Fill image with distance from identity
    auto *ref3dPrt = static_cast<float*>(reference3d->data);
    for (float z = 0; z < reference3d->nz; ++z) {
        for (float y = 0; y < reference3d->ny; ++y) {
            for (float x = 0; x < reference3d->nx; ++x) {
                *ref3dPrt = sqrtf(x * x + y * y + z * z);
                ref3dPrt++;
            }
        }
    }

    // Generate the different use cases
    std::vector<TestData> testCases;

    // Identity use case - 2D
    // First create an identity displacement field and then convert it into a deformation
    nifti_image *idField2d = nifti_copy_nim_info(reference2d);
    idField2d->ndim = idField2d->dim[0] = 5;
    idField2d->nu = idField2d->dim[5] = 2;
    idField2d->nvox = CalcVoxelNumber(*idField2d, idField2d->ndim);
    idField2d->data = (void *)calloc(idField2d->nvox, idField2d->nbyper);
    reg_getDeformationFromDisplacement(idField2d);
    float res2[4];
    memcpy(res2, reference2d->data, reference2d->nvox * sizeof(float));
    // create the test case
    testCases.emplace_back(TestData(
        "identity 2D",
        reference2d,
        idField2d,
        res2)
    );

    // Identity use case - 3D
    nifti_image *idField3d = nifti_copy_nim_info(reference3d);
    idField3d->ndim = idField3d->dim[0] = 5;
    idField3d->nu = idField3d->dim[5] = 3;
    idField3d->nvox = CalcVoxelNumber(*idField3d, idField3d->ndim);
    idField3d->data = calloc(idField3d->nvox, idField3d->nbyper);
    reg_getDeformationFromDisplacement(idField3d);
    float res3[8];
    memcpy(res3, reference3d->data, reference3d->nvox * sizeof(float));
    // create the test case
    testCases.emplace_back(TestData(
        "identity 3D",
        reference3d,
        idField3d,
        res3)
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
