// OpenCL is not supported for this test
#undef _USE_OPENCL

#include "reg_test_common.h"

/*
    This test file contains the following unit tests:
    test functions: composition of deformation field
*/


class ComposeDeformationFieldTest {
protected:
    using TestData = std::tuple<std::string, NiftiImage, NiftiImage, NiftiImage, NiftiImage>;
    using TestCase = std::tuple<std::string, NiftiImage, NiftiImage>;

    inline static vector<TestCase> testCases;

public:
    ComposeDeformationFieldTest() {
        if (!testCases.empty())
            return;

        // Create a 2D reference image
        NiftiImage::dim_t size = 5;
        vector<NiftiImage::dim_t> dimFlo{ size, size };
        NiftiImage reference2d(dimFlo, NIFTI_TYPE_FLOAT32);

        // Create a 3D reference image
        dimFlo.push_back(size);
        NiftiImage reference3d(dimFlo, NIFTI_TYPE_FLOAT32);

        // Data container for the test data
        vector<TestData> testData;

        // Create affine deformation fields
        NiftiImage inDefField2d = CreateDeformationField(reference2d);
        NiftiImage inDefField3d = CreateDeformationField(reference3d);
        NiftiImage defField2d = CreateDeformationField(reference2d);
        NiftiImage defField3d = CreateDeformationField(reference3d);
        NiftiImage outDefField2d = CreateDeformationField(reference2d);
        NiftiImage outDefField3d = CreateDeformationField(reference3d);

        // Identity transformation tests
        testData.emplace_back(TestData(
            "2D ID",
            reference2d,
            inDefField2d,
            defField2d,
            outDefField2d
        ));
        testData.emplace_back(TestData(
            "3D ID",
            reference3d,
            inDefField3d,
            defField3d,
            outDefField3d
        ));

        // Scaling transformation tests
        float * inDefField2dPtr = static_cast<float *>(inDefField2d->data);
        float * inDefField3dPtr = static_cast<float *>(inDefField3d->data);
        float * def2dPtr = static_cast<float *>(defField2d->data);
        float * def3dPtr = static_cast<float *>(defField3d->data);
        for(size_t i=0; i<inDefField2d.nVoxels(); i++)
            inDefField2dPtr[i] /= 1.11f;
        for(size_t i=0; i<inDefField3d.nVoxels(); i++)
            inDefField3dPtr[i] /= 1.11f;
        for(size_t i=0; i<defField2d.nVoxels(); i++)
            def2dPtr[i] *= 1.11f;
        for(size_t i=0; i<defField3d.nVoxels(); i++)
            def3dPtr[i] *= 1.11f;

        testData.emplace_back(TestData(
            "2D scaling",
            reference2d,
            inDefField2d,
            defField2d,
            outDefField2d
        ));
        testData.emplace_back(TestData(
            "3D scaling",
            reference3d,
            inDefField3d,
            defField3d,
            outDefField3d
        ));

        // Check boundary conditions. The default behavior is to use the embedded
        // affine transformation in the deformation field and shift the boundary
        // transformation for padding.
        reg_tools_multiplyValueToImage(defField2d, defField2d, 0.f);
        reg_tools_multiplyValueToImage(defField3d, defField3d, 0.f);
        reg_tools_multiplyValueToImage(inDefField2d, inDefField2d, 0.f);
        reg_tools_multiplyValueToImage(inDefField3d, inDefField3d, 0.f);
        reg_tools_multiplyValueToImage(outDefField2d, outDefField2d, 0.f);
        reg_tools_multiplyValueToImage(outDefField3d, outDefField3d, 0.f);
        reg_getDeformationFromDisplacement(defField2d);
        reg_getDeformationFromDisplacement(defField3d);
        reg_getDeformationFromDisplacement(inDefField2d);
        reg_getDeformationFromDisplacement(inDefField3d);
        reg_getDeformationFromDisplacement(outDefField2d);
        reg_getDeformationFromDisplacement(outDefField3d);
        float * outDefField2dPtr = static_cast<float *>(outDefField2d->data);
        float * outDefField3dPtr = static_cast<float *>(outDefField3d->data);
        for(size_t i=0; i<inDefField2d.nVoxels(); i++)
            inDefField2dPtr[i] += 3.f;
        for(size_t i=0; i<inDefField3d.nVoxels(); i++)
            inDefField3dPtr[i] += 3.f;
        for(size_t i=0; i<defField2d.nVoxels(); i++)
            def2dPtr[i] += 1.f;
        for(size_t i=0; i<defField3d.nVoxels(); i++)
            def3dPtr[i] += 1.f;
        for(size_t i=0; i<outDefField2d.nVoxels(); i++)
            outDefField2dPtr[i] += 4.f;
        for(size_t i=0; i<outDefField3d.nVoxels(); i++)
            outDefField3dPtr[i] += 4.f;
        testData.emplace_back(TestData(
            "2D padding",
            reference2d,
            inDefField2d,
            defField2d,
            outDefField2d
        ));
        testData.emplace_back(TestData(
            "3D padding",
            reference3d,
            inDefField3d,
            defField3d,
            outDefField3d
        ));

        // Run the actual computation with the provided input data
        for (auto&& data : testData) {
            auto&& [testName, reference, inDefField, defField, expectedField] = data;
            // Run the compose on CPU only for now
            reg_defField_compose(defField, inDefField, nullptr);
            // Check the results
            testCases.push_back({testName + " CPU", inDefField, expectedField});
        }

    }
};

TEST_CASE_METHOD(ComposeDeformationFieldTest, "Compose deformation field", "[unit]") {
    // Loop over all generated test cases
    for (auto&& testCase : testCases) {
        // Retrieve test information
        auto&& [testName, result, expected] = testCase;

        SECTION(testName) {
            std::cout << "\n**************** Section " << testName << " ****************" << std::endl;
            float *resPtr = static_cast<float *>(result->data);
            float *expPtr = static_cast<float *>(expected->data);
            for(unsigned i=0; i<expected.nVoxels();++i){
                const double diff = fabs(resPtr[i] - expPtr[i]);
                if (diff > EPS){
                    std::cout << "[i]=" << i;
                    std::cout << " | diff=" << diff;
                    std::cout << " | Result=" << resPtr[i];
                    std::cout << " | Expected=" << expPtr[i] << std::endl;
                }
                REQUIRE(diff < EPS);
            }
        }
    }
}
