// OpenCL is not supported for this test
#undef _USE_OPENCL

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

        // Create a random number generator
        std::mt19937 gen(0);
        std::uniform_real_distribution<float> distr(0, 1);

        // Create a 2D reference image
        NiftiImage::dim_t size = 5;
        vector<NiftiImage::dim_t> dimFlo{ size, size };
        NiftiImage reference2d(dimFlo, NIFTI_TYPE_FLOAT32);

        // Create a 3D reference image
        dimFlo.push_back(size);
        NiftiImage reference3d(dimFlo, NIFTI_TYPE_FLOAT32);

        // Data container for the test data
        vector<TestData> testData;

        // Identity transformation tests
        // Create an affine transformation b-spline parametrisation
        NiftiImage controlPointGrid2d = CreateControlPointGrid(reference2d);
        NiftiImage controlPointGrid3d = CreateControlPointGrid(reference3d);
        // Create the expected deformation field result with an identity
        NiftiImage deformationField2d = CreateDeformationField(reference2d);
        NiftiImage deformationField3d = CreateDeformationField(reference3d);
        testData.emplace_back(TestData(
            "2D ID",
            reference2d,
            controlPointGrid2d,
            deformationField2d
        ));
        testData.emplace_back(TestData(
            "3D ID",
            reference3d,
            controlPointGrid3d,
            deformationField3d
        ));

        // Translation transformation tests - translation of 2 along each axis
        float *cpp2dPtr = static_cast<float*>(controlPointGrid2d->data);
        float *cpp3dPtr = static_cast<float*>(controlPointGrid3d->data);
        float *def2dPtr = static_cast<float*>(deformationField2d->data);
        float *def3dPtr = static_cast<float*>(deformationField3d->data);
        for (size_t i = 0; i < controlPointGrid2d.nVoxels(); i++)
            cpp2dPtr[i] += 2.f;
        for (size_t i = 0; i < controlPointGrid3d.nVoxels(); i++)
            cpp3dPtr[i] += 2.f;
        for (size_t i = 0; i < deformationField2d.nVoxels(); i++)
            def2dPtr[i] += 2.f;
        for (size_t i = 0; i < deformationField3d.nVoxels(); i++)
            def3dPtr[i] += 2.f;

        testData.emplace_back(TestData(
            "2D Trans",
            reference2d,
            controlPointGrid2d,
            deformationField2d
        ));
        testData.emplace_back(TestData(
            "3D Trans",
            reference3d,
            controlPointGrid3d,
            deformationField3d
        ));

        // Scaling transformation tests
        for (size_t i = 0; i < controlPointGrid2d.nVoxels(); i++)
            cpp2dPtr[i] = (cpp2dPtr[i] - 2.f) * 1.1f;
        for (size_t i = 0; i < controlPointGrid3d.nVoxels(); i++)
            cpp3dPtr[i] = (cpp3dPtr[i] - 2.f) * 1.1f;
        for (size_t i = 0; i < deformationField2d.nVoxels(); i++)
            def2dPtr[i] = (def2dPtr[i] - 2.f) * 1.1f;
        for (size_t i = 0; i < deformationField3d.nVoxels(); i++)
            def3dPtr[i] = (def3dPtr[i] - 2.f) * 1.1f;

        testData.emplace_back(TestData(
            "2D scaling",
            reference2d,
            (controlPointGrid2d),
            (deformationField2d)
        ));
        testData.emplace_back(TestData(
            "3D scaling",
            reference3d,
            controlPointGrid3d,
            deformationField3d
        ));

        // Run the actual computation with the provided input data
        for (auto&& data : testData) {
            for (auto&& platformType : PlatformTypes) {
                shared_ptr<Platform> platform{ new Platform(platformType) };
                unique_ptr<F3dContentCreator> contentCreator{ dynamic_cast<F3dContentCreator*>(platform->CreateContentCreator(ContentType::F3d)) };
                // Make a copy of the test data
                auto [testName, reference, controlPointGrid, defFieldExp] = data;
                // Add content
                unique_ptr<F3dContent> content{ contentCreator->Create(reference, reference, controlPointGrid) };
                // Add compute
                unique_ptr<Compute> compute{ platform->CreateCompute(*content) };
                // Compute the deformation field
                compute->GetDeformationField(false, true); // no composition - use bspline
                // Retrieve the deformation field
                NiftiImage defField(content->GetDeformationField(), NiftiImage::Copy::Image);
                // Save for testing
                testCases.push_back({ testName + " " + platform->GetName(), std::move(defField), std::move(defFieldExp) });
            }
        }

        // Data container for the test data related to composition
        vector<TestDataComp> testDataComp;

        // Ensures composition of identity transformation yield identity
        NiftiImage deformationFieldInput2d = CreateDeformationField(reference2d);
        NiftiImage deformationFieldInput3d = CreateDeformationField(reference3d);
        reg_tools_multiplyValueToImage(deformationField2d, deformationField2d, 0.f);
        reg_tools_multiplyValueToImage(deformationField3d, deformationField3d, 0.f);
        reg_tools_multiplyValueToImage(controlPointGrid2d, controlPointGrid2d, 0.f);
        reg_tools_multiplyValueToImage(controlPointGrid3d, controlPointGrid3d, 0.f);
        reg_getDeformationFromDisplacement(deformationField2d);
        reg_getDeformationFromDisplacement(deformationField3d);
        reg_getDeformationFromDisplacement(controlPointGrid2d);
        reg_getDeformationFromDisplacement(controlPointGrid3d);
        testDataComp.emplace_back(TestDataComp(
            "2D composition ID",
            reference3d,
            controlPointGrid2d,
            deformationFieldInput2d,
            deformationField2d
        ));
        testDataComp.emplace_back(TestDataComp(
            "3D composition ID",
            reference3d,
            controlPointGrid3d,
            deformationFieldInput3d,
            deformationField3d
        ));

        // Ensures composition from zooming and and out goes back identity ID
        float *def2dInPtr = static_cast<float*>(deformationFieldInput2d->data);
        float *def3dInPtr = static_cast<float*>(deformationFieldInput3d->data);
        for (size_t i = 0; i < controlPointGrid2d.nVoxels(); i++)
            cpp2dPtr[i] *= 1.1f;
        for (size_t i = 0; i < controlPointGrid3d.nVoxels(); i++)
            cpp3dPtr[i] *= 1.1f;
        for (size_t i = 0; i < deformationFieldInput2d.nVoxels(); i++)
            def2dInPtr[i] /= 1.1f;
        for (size_t i = 0; i < deformationFieldInput3d.nVoxels(); i++)
            def3dInPtr[i] /= 1.1f;
        testDataComp.emplace_back(TestDataComp(
            "2D composition scaling",
            reference3d,
            controlPointGrid2d,
            deformationFieldInput2d,
            deformationField2d
        ));
        testDataComp.emplace_back(TestDataComp(
            "3D composition scaling",
            reference3d,
            controlPointGrid3d,
            deformationFieldInput3d,
            deformationField3d
        ));

        for (auto&& data : testDataComp) {
            for (auto&& platformType : { PlatformType::Cpu }) {
                shared_ptr<Platform> platform{ new Platform(platformType) };
                unique_ptr<F3dContentCreator> contentCreator{ dynamic_cast<F3dContentCreator*>(platform->CreateContentCreator(ContentType::F3d)) };
                // Make a copy of the test data
                auto [testName, reference, controlPointGrid, defField, defFieldExp] = data;
                // Add content
                unique_ptr<F3dContent> content{ contentCreator->Create(reference, reference, controlPointGrid) };
                content->SetDeformationField(defField.disown());
                // Add compute
                unique_ptr<Compute> compute{ platform->CreateCompute(*content) };
                // Compute the deformation field
                compute->GetDeformationField(true, true); // with composition - use bspline
                // Retrieve the deformation field
                defField = NiftiImage(content->GetDeformationField(), NiftiImage::Copy::Image);
                // Save for testing
                testCases.push_back({ testName + " " + platform->GetName(), std::move(defField), std::move(defFieldExp) });
            }
        }

    }
};

TEST_CASE_METHOD(GetDeformationFieldTest, "Deformation field from b-spline grid", "[unit]") {
    // Loop over all generated test cases
    for (auto&& testCase : testCases) {
        // Retrieve test information
        auto&& [testName, result, expected] = testCase;

        SECTION(testName) {
            NR_COUT << "\n**************** Section " << testName << " ****************" << std::endl;
            float *resPtr = static_cast<float*>(result->data);
            float *expPtr = static_cast<float*>(expected->data);
            for (unsigned i = 0; i < expected.nVoxels(); ++i) {
                const double diff = fabs(resPtr[i] - expPtr[i]);
                if (diff > EPS) {
                    NR_COUT << "[i]=" << i;
                    NR_COUT << " | diff=" << diff;
                    NR_COUT << " | Result=" << resPtr[i];
                    NR_COUT << " | Expected=" << expPtr[i] << std::endl;
                }
                REQUIRE(diff < EPS);
            }
        }
    }
}
