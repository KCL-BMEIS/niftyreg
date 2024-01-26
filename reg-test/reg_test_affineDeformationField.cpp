#include "reg_test_common.h"

/*
    This test file contains the following unit tests:
    test function: creation of a deformation field from an affine matrix
    In 2D and 3D
    Identity
    Translation
    Affine
*/

struct float3 {
    float x, y, z;

    std::string to_string() const {
        return "(" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ")";
    }
};

class AffineDeformationFieldTest {
protected:
    using TestData = std::tuple<std::string, NiftiImage&, mat44, vector<float3>>;
    using TestCase = std::tuple<std::string, NiftiImage, vector<float3>>;

    inline static vector<TestCase> testCases;

public:
    AffineDeformationFieldTest() {
        if (!testCases.empty())
            return;

        // Create reference images
        constexpr NiftiImage::dim_t size = 2;
        NiftiImage reference2d({ size, size }, NIFTI_TYPE_FLOAT32);
        NiftiImage reference3d({ size, size, size }, NIFTI_TYPE_FLOAT32);

        // Data container for the test data
        vector<TestData> testData;

        // Identity use case - 2D
        mat44 identity;
        reg_mat44_eye(&identity);
        // Test order [0,0] [1,0] [0,1] [1,1]
        vector<float3> identityResult2d{ { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 1, 1, 0 } };
        testData.emplace_back(TestData(
            "2D Identity",
            reference2d,
            identity,
            identityResult2d
        ));

        // Identity use case - 3D
        // Test order [0,0,0] [1,0,0] [0,1,0] [1,1,0],[0,0,1] [1,0,1] [0,1,1] [1,1,1]
        vector<float3> identityResult3d{ { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 1, 1, 0 }, { 0, 0, 1 }, { 1, 0, 1 }, { 0, 1, 1 }, { 1, 1, 1 } };
        testData.emplace_back(TestData(
            "3D Identity",
            reference3d,
            identity,
            identityResult3d
        ));

        // Translation - 2D
        mat44 translation;
        reg_mat44_eye(&translation);
        translation.m[0][3] = -0.5;
        translation.m[1][3] = 1.5;
        translation.m[2][3] = 0.75;
        // Test order [0,0] [1,0] [0,1] [1,1]
        vector<float3> translationResult2d{ { -0.5f, 1.5f, 0 }, { 0.5f, 1.5f, 0 }, { -0.5f, 2.5f, 0 }, { 0.5f, 2.5f, 0 } };
        testData.emplace_back(TestData(
            "2D Translation",
            reference2d,
            translation,
            std::move(translationResult2d)
        ));

        // Translation - 3D
        // Test order [0,0,0] [1,0,0] [0,1,0] [1,1,0],[0,0,1] [1,0,1] [0,1,1] [1,1,1]
        vector<float3> translationResult3d{ { -0.5f, 1.5f, 0.75f }, { 0.5f, 1.5f, 0.75f },
                                            { -0.5f, 2.5f, 0.75f }, { 0.5f, 2.5f, 0.75f },
                                            { -0.5f, 1.5f, 1.75f }, { 0.5f, 1.5f, 1.75f },
                                            { -0.5f, 2.5f, 1.75f }, { 0.5f, 2.5f, 1.75f } };
        testData.emplace_back(TestData(
            "3D Translation",
            reference3d,
            translation,
            std::move(translationResult3d)
        ));

        // Full affine - 2D
        // Test order [0,0] [1,0] [0,1] [1,1]
        mat44 affine;
        reg_mat44_eye(&affine);
        affine.m[0][3] = -0.5;
        affine.m[1][3] = 1.5;
        affine.m[2][3] = 0.75;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                affine.m[i][j] += ((static_cast<float>(rand()) / RAND_MAX) - 0.5f) / 10.f;
        vector<float3> affineResult2d(4);
        for (int i = 0; i < 4; ++i) {
            double x = identityResult2d[i].x;
            double y = identityResult2d[i].y;
            affineResult2d[i].x = static_cast<float>(affine.m[0][3] + affine.m[0][0] * x + affine.m[0][1] * y);
            affineResult2d[i].y = static_cast<float>(affine.m[1][3] + affine.m[1][0] * x + affine.m[1][1] * y);

        }
        testData.emplace_back(TestData(
            "2D Affine",
            reference2d,
            affine,
            std::move(affineResult2d)
        ));

        // Full affine - 3D
        // Test order [0,0,0] [1,0,0] [0,1,0] [1,1,0],[0,0,1] [1,0,1] [0,1,1] [1,1,1]
        vector<float3> affineResult3d(8);
        for (int i = 0; i < 8; ++i) {
            double x = identityResult3d[i].x;
            double y = identityResult3d[i].y;
            double z = identityResult3d[i].z;
            affineResult3d[i].x = static_cast<float>(affine.m[0][3] + affine.m[0][0] * x + affine.m[0][1] * y + affine.m[0][2] * z);
            affineResult3d[i].y = static_cast<float>(affine.m[1][3] + affine.m[1][0] * x + affine.m[1][1] * y + affine.m[1][2] * z);
            affineResult3d[i].z = static_cast<float>(affine.m[2][3] + affine.m[2][0] * x + affine.m[2][1] * y + affine.m[2][2] * z);
        }
        testData.emplace_back(TestData(
            "3D Affine",
            reference3d,
            affine,
            std::move(affineResult3d)
        ));

        for (auto&& testData : testData) {
            for (auto&& platformType : PlatformTypes) {
                // Make a copy of the test data
                auto [testName, reference, transMat, expRes] = testData;

                // Create the platform
                unique_ptr<Platform> platform{ new Platform(platformType) };
                testName += " "s + platform->GetName();

                // Create the content for Aladin
                unique_ptr<AladinContentCreator> aladinContentCreator{ dynamic_cast<AladinContentCreator*>(platform->CreateContentCreator(ContentType::Aladin)) };
                unique_ptr<AladinContent> aladinContent{ aladinContentCreator->Create(reference, reference, nullptr, &transMat, sizeof(float)) };

                // Do the calculation for Aladin
                unique_ptr<Kernel> affineDeformKernel{ platform->CreateKernel(AffineDeformationFieldKernel::GetName(), aladinContent.get()) };
                affineDeformKernel->castTo<AffineDeformationFieldKernel>()->Calculate();

                // Get the result
                NiftiImage defField(aladinContent->GetDeformationField(), NiftiImage::Copy::Image);

                // Save for testing
                testCases.push_back({ testName + " - Aladin", std::move(defField), std::move(expRes) });
            }
        }
    }
};

TEST_CASE_METHOD(AffineDeformationFieldTest, "Affine Deformation Field", "[unit]") {
    // Loop over all possibles contents for each test
    for (auto&& testCase : testCases) {
        auto&& [testName, defField, expected] = testCase;
        SECTION(testName) {
            NR_COUT << "\n**************** Section " << testName << " ****************" << std::endl;

            // Increase the precision for the output
            NR_COUT << std::fixed << std::setprecision(10);

            // Check all values
            const bool is3d = defField->nz > 1;
            const size_t voxelNumber = defField.nVoxelsPerVolume();
            const auto defFieldPtrX = defField.data(0);
            const auto defFieldPtrY = defField.data(1);
            const auto defFieldPtrZ = defField.data(2);
            for (auto i = 0; i < voxelNumber; i++) {
                float3 result{ static_cast<float>(defFieldPtrX[i]), static_cast<float>(defFieldPtrY[i]), is3d ? defFieldPtrZ[i] : 0.f };
                float3 diff{ abs(result.x - expected[i].x), abs(result.y - expected[i].y), abs(result.z - expected[i].z) };
                if (diff.x > 0 || diff.y > 0 || diff.z > 0) {
                    NR_COUT << "[i]=" << i;
                    NR_COUT << " | diff=" << diff.to_string();
                    NR_COUT << " | Result=" << result.to_string();
                    NR_COUT << " | Expected=" << expected[i].to_string() << std::endl;
                }
                REQUIRE((diff.x == 0 && diff.y == 0 && diff.z == 0));
            }
        }
    }
}
