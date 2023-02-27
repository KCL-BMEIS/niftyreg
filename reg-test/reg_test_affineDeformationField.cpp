#include "reg_test_common.h"

#define EPS 0.0001

/*
    This test file contains the following unit tests:
    test function: creation of a deformation field from an affine matrix
    In 2D and 3D
    identity
    translation
    affine
*/


typedef std::tuple<std::string, nifti_image*, mat44*, float*, float*, float*> TestData;
typedef std::tuple<unique_ptr<AladinContent>, unique_ptr<Platform>> ContentDesc;

TEST_CASE("Affine deformation field", "[AffineDefField]") {
    // Create a reference 2D image
    int dim[8] = { 2, 2, 2, 1, 1, 1, 1, 1 };
    nifti_image *reference2d = nifti_make_new_nim(dim, NIFTI_TYPE_FLOAT32, true);
    reg_checkAndCorrectDimension(reference2d);

    // Create a reference 3D image
    dim[0] = 3;
    dim[3] = 2;
    nifti_image *reference3d = nifti_make_new_nim(dim, NIFTI_TYPE_FLOAT32, true);
    reg_checkAndCorrectDimension(reference3d);

    // Generate the different test cases
    std::vector<TestData> testCases;

    // Identity use case - 2D
    mat44 identity;
    reg_mat44_eye(&identity);
    // Test order [0,0] [1,0] [0,1] [1,1]
    float identityResult2x[4] = { 0, 1, 0, 1 };
    float identityResult2y[4] = { 0, 0, 1, 1 };
    testCases.emplace_back(TestData(
        "identity 2D",
        reference2d,
        &identity,
        identityResult2x,
        identityResult2y,
        nullptr)
    );

    // Identity use case - 3D
    // Test order [0,0,0] [1,0,0] [0,1,0] [1,1,0],[0,0,1] [1,0,1] [0,1,1] [1,1,1]
    float identityResult3x[8] = { 0, 1, 0, 1, 0, 1, 0, 1 };
    float identityResult3y[8] = { 0, 0, 1, 1, 0, 0, 1, 1 };
    float identityResult3z[8] = { 0, 0, 0, 0, 1, 1, 1, 1 };
    testCases.emplace_back(TestData(
        "identity 3D",
        reference3d,
        &identity,
        identityResult3x,
        identityResult3y,
        identityResult3z)
    );

    // Translation - 2D
    mat44 translation;
    reg_mat44_eye(&translation);
    translation.m[0][3] = -0.5;
    translation.m[1][3] = 1.5;
    translation.m[2][3] = 0.75;
    // Test order [0,0] [1,0] [0,1] [1,1]
    float translationResult2x[4] = { -0.5, .5, -0.5, .5 };
    float translationResult2y[4] = { 1.5, 1.5, 2.5, 2.5 };
    testCases.emplace_back(TestData(
        "translation 2D",
        reference2d,
        &translation,
        translationResult2x,
        translationResult2y,
        nullptr)
    );

    // Translation - 3D
    // Test order [0,0,0] [1,0,0] [0,1,0] [1,1,0],[0,0,1] [1,0,1] [0,1,1] [1,1,1]
    float translationResult3x[8] = { -0.5, .5, -0.5, .5, -0.5, .5, -0.5, .5 };
    float translationResult3y[8] = { 1.5, 1.5, 2.5, 2.5, 1.5, 1.5, 2.5, 2.5 };
    float translationResult3z[8] = { .75, .75, .75, .75, 1.75, 1.75, 1.75, 1.75 };
    testCases.emplace_back(TestData(
        "translation 3D",
        reference3d,
        &translation,
        translationResult3x,
        translationResult3y,
        translationResult3z)
    );

    // Full affine - 2D
    // Test order [0,0] [1,0] [0,1] [1,1]
    mat44 affine;
    reg_mat44_eye(&affine);
    affine.m[0][3] = -0.5;
    affine.m[1][3] = 1.5;
    affine.m[2][3] = 0.75;
    for (auto i = 0; i < 4; ++i) {
        for (auto j = 0; j < 4; ++j) {
            affine.m[i][j] += (((float)rand() / (RAND_MAX)) - 0.5f) / 10.f;
        }
    }
    float affineResult2x[4];
    float affineResult2y[4];
    for (auto i = 0; i < 4; ++i) {
        auto x = identityResult2x[i];
        auto y = identityResult2y[i];
        affineResult2x[i] = affine.m[0][3] + affine.m[0][0] * x + affine.m[0][1] * y;
        affineResult2y[i] = affine.m[1][3] + affine.m[1][0] * x + affine.m[1][1] * y;

    }
    testCases.emplace_back(TestData(
        "full affine 2D",
        reference2d,
        &affine,
        affineResult2x,
        affineResult2y,
        nullptr)
    );

    // Full affine - 3D
    // Test order [0,0,0] [1,0,0] [0,1,0] [1,1,0],[0,0,1] [1,0,1] [0,1,1] [1,1,1]
    float affineResult3x[8];
    float affineResult3y[8];
    float affineResult3z[8];
    for (auto i = 0; i < 8; ++i) {
        auto x = identityResult3x[i];
        auto y = identityResult3y[i];
        auto z = identityResult3z[i];
        affineResult3x[i] = affine.m[0][3] + affine.m[0][0] * x + affine.m[0][1] * y + affine.m[0][2] * z;
        affineResult3y[i] = affine.m[1][3] + affine.m[1][0] * x + affine.m[1][1] * y + affine.m[1][2] * z;
        affineResult3z[i] = affine.m[2][3] + affine.m[2][0] * x + affine.m[2][1] * y + affine.m[2][2] * z;
    }
    testCases.emplace_back(TestData(
        "affine 3D",
        reference3d,
        &affine,
        affineResult3x,
        affineResult3y,
        affineResult3z)
    );

    // Loop over all generated test cases
    for (auto&& testCase : testCases) {
        // Retrieve test information
        auto&& [testName, reference, testMat, testResX, testResY, testResZ] = testCase;

        // Accumulate all required contents with a vector
        std::vector<ContentDesc> contentDescs;
        for (auto&& platformType : PlatformTypes) {
            unique_ptr<Platform> platform{ new Platform(platformType) };
            unique_ptr<AladinContentCreator> contentCreator{ dynamic_cast<AladinContentCreator*>(platform->CreateContentCreator(ContentType::Aladin)) };
            unique_ptr<AladinContent> content{ contentCreator->Create(reference, reference, nullptr, testMat, sizeof(float)) };
            contentDescs.push_back({ std::move(content), std::move(platform) });
        }
        // Loop over all possibles contents for each test
        for (auto&& contentDesc : contentDescs) {
            auto&& [content, platform] = contentDesc;
            SECTION(testName + " " + platform->GetName()) {
                // Do the calculation
                unique_ptr<Kernel> affineDeformKernel{ platform->CreateKernel(AffineDeformationFieldKernel::GetName(), content.get()) };
                affineDeformKernel->castTo<AffineDeformationFieldKernel>()->Calculate();

                // Check all values
                nifti_image *defField = content->GetDeformationField();
                auto defFieldPtrX = static_cast<float*>(defField->data);
                const size_t voxelNumber = CalcVoxelNumber(*defField);
                auto defFieldPtrY = &defFieldPtrX[voxelNumber];
                auto defFieldPtrZ = &defFieldPtrY[voxelNumber];
                for (size_t i = 0; i < voxelNumber; ++i) {
                    REQUIRE(fabs(defFieldPtrX[i] - testResX[i]) < EPS);
                    REQUIRE(fabs(defFieldPtrY[i] - testResY[i]) < EPS);
                    if (testResZ)
                        REQUIRE(fabs(defFieldPtrZ[i] - testResZ[i]) < EPS);
                }
            }
        }
    }
    // Clean up
    nifti_image_free(reference2d);
    nifti_image_free(reference3d);
}
