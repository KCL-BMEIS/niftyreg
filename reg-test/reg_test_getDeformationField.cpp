// OpenCL is not supported for this test
#undef USE_OPENCL

#include "reg_test_common.h"

/*
    This test file contains the following unit tests:
    test functions: creation of a deformation field from a control point grid
    In 2D and 3D
    Cubic spline
*/


class GetDeformationFieldTest {
protected:
    using TestData = std::tuple<std::string, NiftiImage, NiftiImage, NiftiImage>;
    using TestDataComp = std::tuple<std::string, NiftiImage, NiftiImage, NiftiImage, NiftiImage>;
    using TestCase = std::tuple<std::string, NiftiImage, NiftiImage>;

    inline static vector<TestCase> testCases;

public:
    GetDeformationFieldTest() {
        if (!testCases.empty())
            return;

        // Create reference images
        constexpr NiftiImage::dim_t size = 5;
        NiftiImage reference2d({ size, size }, NIFTI_TYPE_FLOAT32);
        NiftiImage reference3d({ size, size, size }, NIFTI_TYPE_FLOAT32);

        // Data container for the test data
        vector<TestData> testData;

        // Identity transformation tests
        // Create an affine transformation b-spline parametrisation
        NiftiImage controlPointGrid2d = CreateControlPointGrid(reference2d);
        NiftiImage controlPointGrid3d = CreateControlPointGrid(reference3d);
        // Create the expected deformation field result with an identity
        NiftiImage expDefField2d = CreateDeformationField(reference2d);
        NiftiImage expDefField3d = CreateDeformationField(reference3d);
        testData.emplace_back(TestData(
            "2D ID",
            reference2d,
            controlPointGrid2d,
            expDefField2d
        ));
        testData.emplace_back(TestData(
            "3D ID",
            reference3d,
            controlPointGrid3d,
            expDefField3d
        ));

        // Translation transformation tests - translation of 2 along each axis
        float *cpp2dPtr = static_cast<float*>(controlPointGrid2d->data);
        float *cpp3dPtr = static_cast<float*>(controlPointGrid3d->data);
        float *expDefField2dPtr = static_cast<float*>(expDefField2d->data);
        float *expDefField3dPtr = static_cast<float*>(expDefField3d->data);
        for (size_t i = 0; i < controlPointGrid2d.nVoxels(); i++)
            cpp2dPtr[i] += 2.f;
        for (size_t i = 0; i < controlPointGrid3d.nVoxels(); i++)
            cpp3dPtr[i] += 2.f;
        for (size_t i = 0; i < expDefField2d.nVoxels(); i++)
            expDefField2dPtr[i] += 2.f;
        for (size_t i = 0; i < expDefField3d.nVoxels(); i++)
            expDefField3dPtr[i] += 2.f;

        testData.emplace_back(TestData(
            "2D Trans",
            reference2d,
            controlPointGrid2d,
            expDefField2d
        ));
        testData.emplace_back(TestData(
            "3D Trans",
            reference3d,
            controlPointGrid3d,
            expDefField3d
        ));

        // Scaling transformation tests
        for (size_t i = 0; i < controlPointGrid2d.nVoxels(); i++)
            cpp2dPtr[i] = (cpp2dPtr[i] - 2.f) * 1.1f;
        for (size_t i = 0; i < controlPointGrid3d.nVoxels(); i++)
            cpp3dPtr[i] = (cpp3dPtr[i] - 2.f) * 1.1f;
        for (size_t i = 0; i < expDefField2d.nVoxels(); i++)
            expDefField2dPtr[i] = (expDefField2dPtr[i] - 2.f) * 1.1f;
        for (size_t i = 0; i < expDefField3d.nVoxels(); i++)
            expDefField3dPtr[i] = (expDefField3dPtr[i] - 2.f) * 1.1f;

        testData.emplace_back(TestData(
            "2D Scaling",
            reference2d,
            controlPointGrid2d,
            expDefField2d
        ));
        testData.emplace_back(TestData(
            "3D Scaling",
            reference3d,
            controlPointGrid3d,
            expDefField3d
        ));

        // Run the actual computation with the provided input data
        for (auto&& data : testData) {
            for (auto&& platformType : PlatformTypes) {
                unique_ptr<Platform> platform{ new Platform(platformType) };
                unique_ptr<F3dContentCreator> contentCreator{ dynamic_cast<F3dContentCreator*>(platform->CreateContentCreator(ContentType::F3d)) };
                // Make a copy of the test data
                auto [testName, reference, controlPointGrid, expDefField] = data;
                // Create the content and the compute
                unique_ptr<F3dContent> content{ contentCreator->Create(reference, reference, controlPointGrid) };
                unique_ptr<Compute> compute{ platform->CreateCompute(*content) };
                // Compute the deformation field
                compute->GetDeformationField(false, true); // no composition - use bspline
                // Retrieve the deformation field
                NiftiImage defField(content->GetDeformationField(), NiftiImage::Copy::Image);
                // Save for testing
                testCases.push_back({ testName + " "s + platform->GetName(), std::move(defField), std::move(expDefField) });
            }
        }

        // Data container for the test data related to composition
        vector<TestDataComp> testDataComp;

        // Ensures composition of identity transformation yield identity
        NiftiImage defField2d = CreateDeformationField(reference2d);
        NiftiImage defField3d = CreateDeformationField(reference3d);
        reg_tools_multiplyValueToImage(expDefField2d, expDefField2d, 0.f);
        reg_tools_multiplyValueToImage(expDefField3d, expDefField3d, 0.f);
        reg_tools_multiplyValueToImage(controlPointGrid2d, controlPointGrid2d, 0.f);
        reg_tools_multiplyValueToImage(controlPointGrid3d, controlPointGrid3d, 0.f);
        reg_getDeformationFromDisplacement(expDefField2d);
        reg_getDeformationFromDisplacement(expDefField3d);
        reg_getDeformationFromDisplacement(controlPointGrid2d);
        reg_getDeformationFromDisplacement(controlPointGrid3d);
        testDataComp.emplace_back(TestDataComp(
            "2D Composition ID",
            reference2d,
            controlPointGrid2d,
            defField2d,
            expDefField2d
        ));
        testDataComp.emplace_back(TestDataComp(
            "3D Composition ID",
            reference3d,
            controlPointGrid3d,
            defField3d,
            expDefField3d
        ));

        // Ensures composition from zooming and and out goes back identity ID
        float *defField2dPtr = static_cast<float*>(defField2d->data);
        float *defField3dPtr = static_cast<float*>(defField3d->data);
        for (size_t i = 0; i < controlPointGrid2d.nVoxels(); i++)
            cpp2dPtr[i] *= 1.1f;
        for (size_t i = 0; i < controlPointGrid3d.nVoxels(); i++)
            cpp3dPtr[i] *= 1.1f;
        for (size_t i = 0; i < defField2d.nVoxels(); i++)
            defField2dPtr[i] /= 1.1f;
        for (size_t i = 0; i < defField3d.nVoxels(); i++)
            defField3dPtr[i] /= 1.1f;
        testDataComp.emplace_back(TestDataComp(
            "2D Composition Scaling",
            reference2d,
            controlPointGrid2d,
            defField2d,
            expDefField2d
        ));
        testDataComp.emplace_back(TestDataComp(
            "3D Composition Scaling",
            reference3d,
            controlPointGrid3d,
            defField3d,
            expDefField3d
        ));

        for (auto&& data : testDataComp) {
            for (auto&& platformType : PlatformTypes) {
                unique_ptr<Platform> platform{ new Platform(platformType) };
                unique_ptr<F3dContentCreator> contentCreator{ dynamic_cast<F3dContentCreator*>(platform->CreateContentCreator(ContentType::F3d)) };
                // Make a copy of the test data
                auto [testName, reference, controlPointGrid, defField, expDefField] = data;
                // Create the content and the compute
                unique_ptr<F3dContent> content{ contentCreator->Create(reference, reference, controlPointGrid) };
                unique_ptr<Compute> compute{ platform->CreateCompute(*content) };
                // Compute the deformation field
                content->SetDeformationField(defField.disown());
                compute->GetDeformationField(true, true); // with composition - use bspline
                // Retrieve the deformation field
                defField = NiftiImage(content->GetDeformationField(), NiftiImage::Copy::Image);
                // Save for testing
                testCases.push_back({ testName + " "s + platform->GetName(), std::move(defField), std::move(expDefField) });
            }
        }
    }
};

TEST_CASE_METHOD(GetDeformationFieldTest, "Deformation Field from B-spline Grid", "[unit]") {
    // Loop over all generated test cases
    for (auto&& testCase : testCases) {
        // Retrieve test information
        auto&& [testName, result, expected] = testCase;

        SECTION(testName) {
            NR_COUT << "\n**************** Section " << testName << " ****************" << std::endl;

            // Increase the precision for the output
            NR_COUT << std::fixed << std::setprecision(10);

            const auto resPtr = result.data();
            const auto expPtr = expected.data();
            for (auto i = 0; i < expected.nVoxels(); i++) {
                const float resVal = resPtr[i];
                const float expVal = expPtr[i];
                const float diff = abs(resVal - expVal);
                if (diff > 0) {
                    NR_COUT << "[i]=" << i;
                    NR_COUT << " | diff=" << diff;
                    NR_COUT << " | Result=" << resVal;
                    NR_COUT << " | Expected=" << expVal << std::endl;
                }
                REQUIRE(diff < EPS);
            }
        }
    }
}
