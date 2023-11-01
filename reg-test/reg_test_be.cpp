// OpenCL is not supported for this test
#undef USE_OPENCL

#include "reg_test_common.h"

/*
    This test file contains the following unit tests:
    - BE computation for an identity transformation
    - BE computation for an affine transformation
    - BE computation for non-linear transformation
*/


class BendingEnergyTest {
protected:
    using TestData = std::tuple<std::string, NiftiImage, NiftiImage, float>;
    using TestCase = std::tuple<std::string, float, float>;

    inline static vector<TestCase> testCases;

public:
    BendingEnergyTest() {
        if (!testCases.empty())
            return;

        // Create a random number generator
        std::mt19937 gen(0);
        std::uniform_real_distribution<float> distr(-1, 1);

        // Create 2D and 3D reference images
        constexpr NiftiImage::dim_t dimSize = 4;
        NiftiImage reference2d({ dimSize, dimSize }, NIFTI_TYPE_FLOAT32);
        NiftiImage reference3d({ dimSize, dimSize, dimSize }, NIFTI_TYPE_FLOAT32);

        // Create 2D and 3D control point grids
        NiftiImage controlPointGrid2d = CreateControlPointGrid(reference2d);
        NiftiImage controlPointGrid3d = CreateControlPointGrid(reference3d);

        // Add the test data
        vector<TestData> testData;
        testData.emplace_back(TestData(
            "BE identity 2D",
            reference2d,
            controlPointGrid2d,
            0.f
        ));
        testData.emplace_back(TestData(
            "BE identity 3D",
            reference3d,
            controlPointGrid3d,
            0.f
        ));
        // Add random values to the control point grid coefficients
        // No += or + operator for RNifti::NiftiImageData:Element
        // so reverting to old school for now
        float *cpp2dPtr = static_cast<float*>(controlPointGrid2d->data);
        float *cpp3dPtr = static_cast<float*>(controlPointGrid3d->data);
        for (size_t i = 0; i < controlPointGrid2d.nVoxels(); ++i)
            cpp2dPtr[i] += distr(gen);
        for (size_t i = 0; i < controlPointGrid3d.nVoxels(); ++i)
            cpp3dPtr[i] += distr(gen);
        // Add the test data
        testData.emplace_back(TestData(
            "BE random 2D",
            reference2d,
            controlPointGrid2d,
            this->GetBe2d(controlPointGrid2d)
        ));
        testData.emplace_back(TestData(
            "BE random 3D",
            reference3d,
            controlPointGrid3d,
            this->GetBe3d(controlPointGrid3d)
        ));

        // Set some scaling transformation in the transformations
        mat44 affine2d, affine3d;
        reg_mat44_eye(&affine2d);
        reg_mat44_eye(&affine3d);
        affine3d.m[0][0] = affine2d.m[0][0] = 0.8f;
        affine3d.m[1][1] = affine2d.m[1][1] = 1.2f;
        affine3d.m[2][2] = 1.1f;
        reg_affine_getDeformationField(&affine2d, controlPointGrid2d);
        reg_affine_getDeformationField(&affine3d, controlPointGrid3d);

        // Add the test data
        testData.emplace_back(TestData(
            "BE scaling 2D",
            reference2d,
            controlPointGrid2d,
            0.f
        ));
        testData.emplace_back(TestData(
            "BE scaling 3D",
            reference3d,
            controlPointGrid3d,
            0.f
        ));

        // Compute the Bending energy for each use case
        for (auto&& data : testData) {
            for (auto&& platformType : PlatformTypes) {
                // Make a copy of the test data
                auto [testName, reference, controlPointGrid, expected] = data;
                // Add content
                shared_ptr<Platform> platform{ new Platform(platformType) };
                unique_ptr<F3dContentCreator> contentCreator{ dynamic_cast<F3dContentCreator*>(platform->CreateContentCreator(ContentType::F3d)) };
                unique_ptr<F3dContent> content{ contentCreator->Create(reference, reference, controlPointGrid) };
                unique_ptr<Compute> compute{ platform->CreateCompute(*content) };
                float be = static_cast<float>(compute->ApproxBendingEnergy());
                testCases.push_back({ testName + " " + platform->GetName(), be, expected });
            }
        }
    }

    float GetBe2d(const NiftiImage& cpp) {
        // variable to store the bending energy and the normalisation value
        double be = 0;

        // The BSpine basis values are known since the control points all have a relative position equal to 0
        float basis[3], first[3], second[3];
        basis[0] = 1.f / 6.f; basis[1] = 4.f / 6.f; basis[2] = 1.f / 6.f;
        first[0] = -0.5f; first[1] = 0.f; first[2] = 0.5f;
        second[0] = 1.f; second[1] = -2.f; second[2] = 1.f;

        // the first and last control points along each axis are
        // ignored for lack of support
        const auto cppPtr = cpp.data();
        for (int y = 1; y < cpp->dim[2] - 1; ++y) {
            for (int x = 1; x < cpp->dim[1] - 1; ++x) {
                // The BE is computed as
                // BE=dXX/dx^2 + dYY/dy^2 + dXX/dy^2 + dYY/dx^2 + 2 * [dXY/dx^2 + dXY/dy^2]
                float XX_x = 0, YY_x = 0, XY_x = 0;
                float XX_y = 0, YY_y = 0, XY_y = 0;
                for (unsigned j = 0; j < 3; ++j) {
                    for (unsigned i = 0; i < 3; ++i) {
                        unsigned cpIndex = (y + j - 1) * cpp->dim[1] + x + i - 1;
                        float x_val = cppPtr[cpIndex];
                        float y_val = cppPtr[cpIndex + cpp.nVoxelsPerVolume()];
                        XX_x += x_val * second[i] * basis[j];
                        YY_x += x_val * basis[i] * second[j];
                        XY_x += x_val * first[i] * first[j];
                        XX_y += y_val * second[i] * basis[j];
                        YY_y += y_val * basis[i] * second[j];
                        XY_y += y_val * first[i] * first[j];
                    }
                }
                be += XX_x * XX_x + YY_x * YY_x + XX_y * XX_y + YY_y * YY_y + 2.0 * XY_x * XY_x + 2.0 * XY_y * XY_y;
            }
        }
        return float(be / (double)cpp.nVoxels());
    }

    float GetBe3d(const NiftiImage& cpp) {
        // variable to store the bending energy and the normalisation value
        double be = 0;

        // The BSpine basis values are known since the control points all have a relative position equal to 0
        float basis[3], first[3], second[3];
        basis[0] = 1.f / 6.f; basis[1] = 4.f / 6.f; basis[2] = 1.f / 6.f;
        first[0] = -0.5f; first[1] = 0.f; first[2] = 0.5f;
        second[0] = 1.f; second[1] = -2.f; second[2] = 1.f;

        const auto cppPtr = cpp.data();
        // the first and last control points along each axis are
        // ignored for lack of support
        for (int z = 1; z < cpp->nz - 1; ++z) {
            for (int y = 1; y < cpp->ny - 1; ++y) {
                for (int x = 1; x < cpp->nx - 1; ++x) {
                    float XX_x = 0, YY_x = 0, ZZ_x = 0, XY_x = 0, YZ_x = 0, XZ_x = 0;
                    float XX_y = 0, YY_y = 0, ZZ_y = 0, XY_y = 0, YZ_y = 0, XZ_y = 0;
                    float XX_z = 0, YY_z = 0, ZZ_z = 0, XY_z = 0, YZ_z = 0, XZ_z = 0;
                    for (unsigned k = 0; k < 3; ++k) {
                        for (unsigned j = 0; j < 3; ++j) {
                            for (unsigned i = 0; i < 3; ++i) {
                                unsigned cpIndex = ((z + k - 1) * cpp->ny + y + j - 1) * cpp->nx + x + i - 1;
                                float x_val = cppPtr[cpIndex];
                                float y_val = cppPtr[cpIndex + cpp.nVoxelsPerVolume()];
                                float z_val = cppPtr[cpIndex + 2 * cpp.nVoxelsPerVolume()];
                                XX_x += x_val * second[i] * basis[j] * basis[k];
                                YY_x += x_val * basis[i] * second[j] * basis[k];
                                ZZ_x += x_val * basis[i] * basis[j] * second[k];
                                XY_x += x_val * first[i] * first[j] * basis[k];
                                YZ_x += x_val * basis[i] * first[j] * first[k];
                                XZ_x += x_val * first[i] * basis[j] * first[k];

                                XX_y += y_val * second[i] * basis[j] * basis[k];
                                YY_y += y_val * basis[i] * second[j] * basis[k];
                                ZZ_y += y_val * basis[i] * basis[j] * second[k];
                                XY_y += y_val * first[i] * first[j] * basis[k];
                                YZ_y += y_val * basis[i] * first[j] * first[k];
                                XZ_y += y_val * first[i] * basis[j] * first[k];

                                XX_z += z_val * second[i] * basis[j] * basis[k];
                                YY_z += z_val * basis[i] * second[j] * basis[k];
                                ZZ_z += z_val * basis[i] * basis[j] * second[k];
                                XY_z += z_val * first[i] * first[j] * basis[k];
                                YZ_z += z_val * basis[i] * first[j] * first[k];
                                XZ_z += z_val * first[i] * basis[j] * first[k];
                            }
                        }
                    }
                    be += XX_x * XX_x + YY_x * YY_x + ZZ_x * ZZ_x + \
                        XX_y * XX_y + YY_y * YY_y + ZZ_y * ZZ_y + \
                        XX_z * XX_z + YY_z * YY_z + ZZ_z * ZZ_z + \
                        2.0 * XY_x * XY_x + 2.0 * YZ_x * YZ_x + 2.0 * XZ_x * XZ_x + \
                        2.0 * XY_y * XY_y + 2.0 * YZ_y * YZ_y + 2.0 * XZ_y * XZ_y + \
                        2.0 * XY_z * XY_z + 2.0 * YZ_z * YZ_z + 2.0 * XZ_z * XZ_z;
                }
            }
        }
        return float(be / (double)cpp.nVoxels());
    }
};

TEST_CASE_METHOD(BendingEnergyTest, "Bending Energy", "[unit]") {
    // Loop over all generated test cases
    for (auto&& testCase : testCases) {
        // Retrieve test information
        auto&& [testName, result, expected] = testCase;

        SECTION(testName) {
            NR_COUT << "\n**************** Section " << testName << " ****************" << std::endl;

            // Increase the precision for the output
            NR_COUT << std::fixed << std::setprecision(10);

            const auto diff = abs(result - expected);
            if (diff > 0)
                NR_COUT << "Result=" << result << " | Expected=" << expected << std::endl;
            REQUIRE(diff < EPS);
        }
    }
}
