// OpenCL is not supported for this test
#undef USE_OPENCL

#include "reg_test_common.h"

/*
    This test file contains the following unit tests:
    test functions: composition of deformation field
*/


class ComposeDeformationFieldTest {
protected:
    using TestData = std::tuple<std::string, NiftiImage&, NiftiImage, NiftiImage, NiftiImage>;
    using TestCase = std::tuple<std::string, NiftiImage, NiftiImage>;

    inline static vector<TestCase> testCases;

public:
    ComposeDeformationFieldTest() {
        if (!testCases.empty())
            return;

        // Create reference images
        constexpr NiftiImage::dim_t size = 5;
        NiftiImage reference2d({ size, size }, NIFTI_TYPE_FLOAT32);
        NiftiImage reference3d({ size, size, size }, NIFTI_TYPE_FLOAT32);

        // Data container for the test data
        vector<TestData> testData;

        // Create affine deformation fields
        NiftiImage defField2d = CreateDeformationField(reference2d);
        NiftiImage defField3d = CreateDeformationField(reference3d);
        NiftiImage outDefField2d = CreateDeformationField(reference2d);
        NiftiImage outDefField3d = CreateDeformationField(reference3d);
        NiftiImage expDefField2d = CreateDeformationField(reference2d);
        NiftiImage expDefField3d = CreateDeformationField(reference3d);

        // Identity transformation tests
        testData.emplace_back(TestData(
            "2D ID",
            reference2d,
            defField2d,
            outDefField2d,
            expDefField2d
        ));
        testData.emplace_back(TestData(
            "3D ID",
            reference3d,
            defField3d,
            outDefField3d,
            expDefField3d
        ));

        // Scaling transformation tests
        float *defField2dPtr = static_cast<float*>(defField2d->data);
        float *defField3dPtr = static_cast<float*>(defField3d->data);
        float *outDefField2dPtr = static_cast<float*>(outDefField2d->data);
        float *outDefField3dPtr = static_cast<float*>(outDefField3d->data);
        for (size_t i = 0; i < defField2d.nVoxels(); i++)
            defField2dPtr[i] *= 1.11f;
        for (size_t i = 0; i < defField3d.nVoxels(); i++)
            defField3dPtr[i] *= 1.11f;
        for (size_t i = 0; i < outDefField2d.nVoxels(); i++)
            outDefField2dPtr[i] /= 1.11f;
        for (size_t i = 0; i < outDefField3d.nVoxels(); i++)
            outDefField3dPtr[i] /= 1.11f;

        testData.emplace_back(TestData(
            "2D Scaling",
            reference2d,
            defField2d,
            outDefField2d,
            expDefField2d
        ));
        testData.emplace_back(TestData(
            "3D Scaling",
            reference3d,
            defField3d,
            outDefField3d,
            expDefField3d
        ));

        // Check boundary conditions. The default behavior is to use the embedded
        // affine transformation in the deformation field and shift the boundary
        // transformation for padding.
        reg_tools_multiplyValueToImage(defField2d, defField2d, 0.f);
        reg_tools_multiplyValueToImage(defField3d, defField3d, 0.f);
        reg_tools_multiplyValueToImage(outDefField2d, outDefField2d, 0.f);
        reg_tools_multiplyValueToImage(outDefField3d, outDefField3d, 0.f);
        reg_tools_multiplyValueToImage(expDefField2d, expDefField2d, 0.f);
        reg_tools_multiplyValueToImage(expDefField3d, expDefField3d, 0.f);
        reg_getDeformationFromDisplacement(defField2d);
        reg_getDeformationFromDisplacement(defField3d);
        reg_getDeformationFromDisplacement(outDefField2d);
        reg_getDeformationFromDisplacement(outDefField3d);
        reg_getDeformationFromDisplacement(expDefField2d);
        reg_getDeformationFromDisplacement(expDefField3d);
        float *expDefField2dPtr = static_cast<float*>(expDefField2d->data);
        float *expDefField3dPtr = static_cast<float*>(expDefField3d->data);
        for (size_t i = 0; i < defField2d.nVoxels(); i++)
            defField2dPtr[i] += 1.f;
        for (size_t i = 0; i < defField3d.nVoxels(); i++)
            defField3dPtr[i] += 1.f;
        for (size_t i = 0; i < outDefField2d.nVoxels(); i++)
            outDefField2dPtr[i] += 3.f;
        for (size_t i = 0; i < outDefField3d.nVoxels(); i++)
            outDefField3dPtr[i] += 3.f;
        for (size_t i = 0; i < expDefField2d.nVoxels(); i++)
            expDefField2dPtr[i] += 4.f;
        for (size_t i = 0; i < expDefField3d.nVoxels(); i++)
            expDefField3dPtr[i] += 4.f;
        testData.emplace_back(TestData(
            "2D Padding",
            reference2d,
            defField2d,
            outDefField2d,
            expDefField2d
        ));
        testData.emplace_back(TestData(
            "3D Padding",
            reference3d,
            defField3d,
            outDefField3d,
            expDefField3d
        ));

        // Run the actual computation with the provided input data
        for (auto&& data : testData) {
            // Get the test data
            auto&& [testName, reference, defField, outDefField, expDefField] = data;
            for (auto&& platformType : PlatformTypes) {
                unique_ptr<Platform> platform{ new Platform(platformType) };
                unique_ptr<ContentCreator> contentCreator{ dynamic_cast<ContentCreator*>(platform->CreateContentCreator()) };
                // Create the content and the compute
                unique_ptr<Content> content{ contentCreator->Create(reference, reference) };
                unique_ptr<Compute> compute{ platform->CreateCompute(*content) };
                // Run the compose
                content->SetDeformationField(NiftiImage(outDefField).disown());
                compute->DefFieldCompose(defField);
                // Get the result
                NiftiImage resDefField(content->GetDeformationField(), NiftiImage::Copy::Image);
                // Save for testing
                testCases.push_back({ testName + " "s + platform->GetName(), std::move(resDefField), expDefField });
            }
        }
    }
};

TEST_CASE_METHOD(ComposeDeformationFieldTest, "Compose Deformation Field", "[unit]") {
    // Loop over all generated test cases
    for (auto&& testCase : testCases) {
        // Retrieve test information
        auto&& [testName, result, expected] = testCase;

        SECTION(testName) {
            std::cout << "\n**************** Section " << testName << " ****************" << std::endl;

            // Increase the precision for the output
            NR_COUT << std::fixed << std::setprecision(10);

            // Check the deformation fields
            const auto resPtr = result.data();
            const auto expPtr = expected.data();
            for (auto i = 0; i < expected.nVoxels(); i++) {
                const float resVal = resPtr[i];
                const float expVal = expPtr[i];
                const float diff = abs(resVal - expVal);
                if (diff > 0) {
                    std::cout << "[i]=" << i;
                    std::cout << " | diff=" << diff;
                    std::cout << " | Result=" << resVal;
                    std::cout << " | Expected=" << expVal << std::endl;
                }
                REQUIRE(diff < EPS);
            }
        }
    }
}
