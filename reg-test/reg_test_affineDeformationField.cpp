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
    using TestData = std::tuple<std::string, NiftiImage&, NiftiImage, mat44, vector<float3>>;
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
        Mat44Eye(&identity);
        // Test order [0,0] [1,0] [0,1] [1,1]
        vector<float3> identityResult2d{ { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 1, 1, 0 } };
        testData.emplace_back(TestData(
            "2D Identity",
            reference2d,
            NiftiImage(),
            identity,
            identityResult2d
        ));

        // Identity use case - 3D
        // Test order [0,0,0] [1,0,0] [0,1,0] [1,1,0],[0,0,1] [1,0,1] [0,1,1] [1,1,1]
        vector<float3> identityResult3d{ { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 1, 1, 0 }, { 0, 0, 1 }, { 1, 0, 1 }, { 0, 1, 1 }, { 1, 1, 1 } };
        testData.emplace_back(TestData(
            "3D Identity",
            reference3d,
            NiftiImage(),
            identity,
            identityResult3d
        ));

        // Translation - 2D
        mat44 translation;
        Mat44Eye(&translation);
        translation.m[0][3] = -0.5;
        translation.m[1][3] = 1.5;
        translation.m[2][3] = 0.75;
        // Test order [0,0] [1,0] [0,1] [1,1]
        vector<float3> translationResult2d{ { -0.5f, 1.5f, 0 }, { 0.5f, 1.5f, 0 }, { -0.5f, 2.5f, 0 }, { 0.5f, 2.5f, 0 } };
        testData.emplace_back(TestData(
            "2D Translation",
            reference2d,
            NiftiImage(),
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
            NiftiImage(),
            translation,
            std::move(translationResult3d)
        ));

        // Create deformation fields and fill them with random values
        NiftiImage defField2d = CreateDeformationField(reference2d);
        NiftiImage defField3d = CreateDeformationField(reference3d);
        auto defField2dPtr = defField2d.data();
        auto defField2dPtrX = defField2d.data(0);
        auto defField2dPtrY = defField2d.data(1);
        auto defField3dPtr = defField3d.data();
        auto defField3dPtrX = defField3d.data(0);
        auto defField3dPtrY = defField3d.data(1);
        auto defField3dPtrZ = defField3d.data(2);
        for (auto i = 0; i < defField2d.nVoxels(); i++)
            defField2dPtr[i] = static_cast<float>(rand()) / RAND_MAX;
        for (auto i = 0; i < defField3d.nVoxels(); i++)
            defField3dPtr[i] = static_cast<float>(rand()) / RAND_MAX;

        // Full affine - 2D
        // Test order [0,0] [1,0] [0,1] [1,1]
        mat44 affine;
        Mat44Eye(&affine);
        affine.m[0][3] = -0.5;
        affine.m[1][3] = 1.5;
        affine.m[2][3] = 0.75;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                affine.m[i][j] += ((static_cast<float>(rand()) / RAND_MAX) - 0.5f) / 10.f;
        vector<float3> affineResult2d(4);
        for (char compose = 0; compose < 2; compose++) {
            for (int i = 0; i < 4; i++) {
                double x = compose ? defField2dPtrX[i] : identityResult2d[i].x;
                double y = compose ? defField2dPtrY[i] : identityResult2d[i].y;
                affineResult2d[i].x = static_cast<float>(affine.m[0][3] + affine.m[0][0] * x + affine.m[0][1] * y);
                affineResult2d[i].y = static_cast<float>(affine.m[1][3] + affine.m[1][0] * x + affine.m[1][1] * y);
            }
            testData.emplace_back(TestData(
                "2D Affine"s + (compose ? " with Composition" : ""),
                reference2d,
                compose ? std::move(defField2d) : NiftiImage(),
                affine,
                affineResult2d
            ));
        }

        // Full affine - 3D
        // Test order [0,0,0] [1,0,0] [0,1,0] [1,1,0],[0,0,1] [1,0,1] [0,1,1] [1,1,1]
        vector<float3> affineResult3d(8);
        for (char compose = 0; compose < 2; compose++) {
            for (int i = 0; i < 8; i++) {
                double x = compose ? defField3dPtrX[i] : identityResult3d[i].x;
                double y = compose ? defField3dPtrY[i] : identityResult3d[i].y;
                double z = compose ? defField3dPtrZ[i] : identityResult3d[i].z;
                affineResult3d[i].x = static_cast<float>(affine.m[0][3] + affine.m[0][0] * x + affine.m[0][1] * y + affine.m[0][2] * z);
                affineResult3d[i].y = static_cast<float>(affine.m[1][3] + affine.m[1][0] * x + affine.m[1][1] * y + affine.m[1][2] * z);
                affineResult3d[i].z = static_cast<float>(affine.m[2][3] + affine.m[2][0] * x + affine.m[2][1] * y + affine.m[2][2] * z);
            }
            testData.emplace_back(TestData(
                "3D Affine"s + (compose ? " with Composition" : ""),
                reference3d,
                compose ? std::move(defField3d) : NiftiImage(),
                affine,
                affineResult3d
            ));
        }

        for (auto&& testData : testData) {
            for (auto&& platformType : PlatformTypes) {
                // Make a copy of the test data
                auto [testName, reference, defField, transMat, expRes] = testData;

                // Create the platform
                unique_ptr<Platform> platform{ new Platform(platformType) };
                testName += " "s + platform->GetName();

                // Create the content for Aladin
                unique_ptr<AladinContentCreator> aladinContentCreator{ dynamic_cast<AladinContentCreator*>(platform->CreateContentCreator(ContentType::Aladin)) };
                unique_ptr<AladinContent> aladinContent{ aladinContentCreator->Create(reference, reference, nullptr, &transMat, sizeof(float)) };

                // Set the deformation field if composition is required
                if (defField)
                    aladinContent->SetDeformationField(NiftiImage(defField));

                // Do the calculation for Aladin
                unique_ptr<Kernel> affineDeformKernel{ platform->CreateKernel(AffineDeformationFieldKernel::GetName(), aladinContent.get()) };
                affineDeformKernel->castTo<AffineDeformationFieldKernel>()->Calculate(defField);

                // Save the results for testing
                testCases.push_back({ testName + " - Aladin", std::move(aladinContent->GetDeformationField()), expRes });

                // Do the calculation also for Compute using Content
                // Skip OpenCL as it is not supported
                if (platform->GetPlatformType() == PlatformType::OpenCl)
                    continue;

                // Create the content
                unique_ptr<ContentCreator> contentCreator{ platform->CreateContentCreator() };
                unique_ptr<Content> content{ contentCreator->Create(reference, reference, nullptr, &transMat, sizeof(float)) };

                // Set the deformation field if composition is required
                if (defField)
                    content->SetDeformationField(NiftiImage(defField));

                // Do the calculation
                unique_ptr<Compute> compute{ platform->CreateCompute(*content) };
                compute->GetAffineDeformationField(defField);

                // Save the results for testing
                testCases.push_back({ testName, std::move(content->GetDeformationField()), std::move(expRes) });
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

TEST_CASE("Affine deformation field composes with a non-identity sform", "[unit]") {
    /*
        Without composition the field is (A * S) * voxel, S being the deformation field's voxel-to-mm
        matrix (production selects the sform when sform_code > 0, the qform otherwise). The cases
        above all run with an
        identity S, where a mishandled composition is invisible; this one installs an anisotropic,
        sheared S and checks the product against the same expression evaluated independently - the
        matrix product in float exactly as IEEE defines it, the per-voxel application in double.
    */
    for (const bool is3D : { false, true }) {
        std::vector<NiftiImage::dim_t> dims(is3D ? 3 : 2, 4);
        NiftiImage reference(dims, NIFTI_TYPE_FLOAT32);
        mat44 sform;
        Mat44Eye(&sform);
        sform.m[0][0] = 1.3f; sform.m[0][1] = 0.2f;  sform.m[0][3] = -3.f;
        sform.m[1][0] = -0.1f; sform.m[1][1] = 0.85f; sform.m[1][3] = 2.f;
        if (is3D) {
            sform.m[0][2] = 0.05f; sform.m[1][2] = -0.07f;
            sform.m[2][0] = 0.15f; sform.m[2][1] = -0.1f; sform.m[2][2] = 1.1f; sform.m[2][3] = 0.75f;
        }
        setSform(reference, sform);

        mat44 affine;
        Mat44Eye(&affine);
        affine.m[0][0] = 1.05f; affine.m[0][1] = -0.12f; affine.m[0][3] = 1.5f;
        affine.m[1][0] = 0.08f; affine.m[1][1] = 0.9f;   affine.m[1][3] = -0.5f;
        if (is3D) { affine.m[2][2] = 1.15f; affine.m[2][3] = 0.25f; }

        for (auto&& platformType : PlatformTypes) {
            Platform platform(platformType);
            if (platform.GetPlatformType() == PlatformType::OpenCl)
                continue;
            SECTION(std::string(is3D ? "3D" : "2D") + " " + platform.GetName()) {
                NiftiImage ref(reference);
                unique_ptr<ContentCreator> creator{ platform.CreateContentCreator() };
                unique_ptr<Content> content{ creator->Create(ref, ref, nullptr, &affine, sizeof(float)) };
                unique_ptr<Compute> compute{ platform.CreateCompute(*content) };
                compute->GetAffineDeformationField(false);
                NiftiImage& field = content->GetDeformationField();

                // The composition the operation performs, evaluated independently: the mat44 product
                // in float (as IEEE defines it), the application per voxel in double
                const mat44 composed = affine * sform;
                const size_t volume = field.nVoxelsPerVolume();
                const auto ptr = field.data();
                const int nx = field->nx, ny = field->ny, nz = field->nz;
                const int components = is3D ? 3 : 2;
                for (int k = 0; k < nz; ++k)
                    for (int j = 0; j < ny; ++j)
                        for (int i = 0; i < nx; ++i) {
                            const size_t index = (static_cast<size_t>(k) * ny + j) * nx + i;
                            for (int c = 0; c < components; ++c) {
                                const double expected = double(composed.m[c][0]) * i +
                                                        double(composed.m[c][1]) * j +
                                                        double(composed.m[c][2]) * k +
                                                        double(composed.m[c][3]);
                                const float actual = ptr[c * volume + index];
                                INFO("voxel (" << i << "," << j << "," << k << ") component " << c);
                                REQUIRE(std::abs(actual - expected) < 1e-5);
                            }
                        }
            }
        }
    }
}
