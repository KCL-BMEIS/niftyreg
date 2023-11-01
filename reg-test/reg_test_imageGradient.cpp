// OpenCL is not supported for this test
#undef USE_OPENCL

#include "reg_test_common.h"

/*
    This test file contains the following unit tests:
    test function: image gradient
    In 2D and 3D
    Linear
    Cubic spline
*/


typedef std::tuple<std::string, NiftiImage, NiftiImage, int, float*> TestData;
typedef std::tuple<unique_ptr<DefContent>, unique_ptr<Platform>> ContentDesc;

TEST_CASE("Image Gradient", "[unit]") {
    // Create a reference 2D image
    vector<NiftiImage::dim_t> dimFlo{ 4, 4 };
    NiftiImage reference2d(dimFlo, NIFTI_TYPE_FLOAT32);

    // Fill image with distance from identity
    const auto ref2dPtr = reference2d.data();
    auto ref2dItr = ref2dPtr.begin();
    for (int y = 0; y < reference2d->ny; ++y)
        for (int x = 0; x < reference2d->nx; ++x)
            *ref2dItr++ = sqrtf(static_cast<float>(x * x + y * y));

    // Create a corresponding 2D deformation field
    vector<NiftiImage::dim_t> dimDef{ 1, 1, 1, 1, 2 };
    NiftiImage deformationField2d(dimDef, NIFTI_TYPE_FLOAT32);
    auto def2dPtr = deformationField2d.data();
    def2dPtr[0] = 1.2f;
    def2dPtr[1] = 1.3f;

    // Create a reference 3D image
    dimFlo.push_back(4);
    NiftiImage reference3d(dimFlo, NIFTI_TYPE_FLOAT32);

    // Fill image with distance from identity
    const auto ref3dPtr = reference3d.data();
    auto ref3dItr = ref3dPtr.begin();
    for (int z = 0; z < reference3d->nz; ++z)
        for (int y = 0; y < reference3d->ny; ++y)
            for (int x = 0; x < reference3d->nx; ++x)
                *ref3dItr++ = sqrtf(static_cast<float>(x * x + y * y + z * z));

    // Create a corresponding 3D deformation field
    dimDef[4] = 3;
    NiftiImage deformationField3d(dimDef, NIFTI_TYPE_FLOAT32);
    auto def3dPtr = deformationField3d.data();
    def3dPtr[0] = 1.2f;
    def3dPtr[1] = 1.3f;
    def3dPtr[2] = 1.4f;

    // Generate the different test cases
    vector<TestData> testCases;

    // Linear image gradient - 2D
    // coordinate in image: [1.2, 1.3]
    float resLinear2d[2] = {};
    const float derivLinear[2] = { -1, 1 };
    const float xBasisLinear[2] = { 0.8f, 0.2f };
    const float yBasisLinear[2] = { 0.7f, 0.3f };
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            const float coeff = ref2dPtr[(y + 1) * dimFlo[1] + (x + 1)];
            resLinear2d[0] += coeff * derivLinear[x] * yBasisLinear[y];
            resLinear2d[1] += coeff * xBasisLinear[x] * derivLinear[y];
        }
    }

    // Create the test case
    testCases.emplace_back(TestData(
        "Linear 2D",
        reference2d,
        deformationField2d,
        1,
        resLinear2d
    ));

    // Cubic spline image gradient - 2D
    // coordinate in image: [1.2, 1.3]
    float resCubic2d[2] = {};
    float xBasisCubic[4], yBasisCubic[4];
    float xDerivCubic[4], yDerivCubic[4];
    InterpCubicSplineKernel(0.2f, xBasisCubic, xDerivCubic);
    InterpCubicSplineKernel(0.3f, yBasisCubic, yDerivCubic);
    for (int y = 0; y <= 3; ++y) {
        for (int x = 0; x <= 3; ++x) {
            const float coeff = ref2dPtr[y * dimFlo[1] + x];
            resCubic2d[0] += coeff * xDerivCubic[x] * yBasisCubic[y];
            resCubic2d[1] += coeff * xBasisCubic[x] * yDerivCubic[y];
        }
    }

    // Create the test case
    testCases.emplace_back(TestData(
        "Cubic Spline 2D",
        reference2d,
        deformationField2d,
        3,
        resCubic2d
    ));

    // Linear image gradient - 3D
    // coordinate in image: [1.2, 1.3, 1.4]
    float resLinear3d[3] = {};
    const float zBasisLinear[2] = { 0.6f, 0.4f };
    for (int z = 0; z < 2; ++z) {
        for (int y = 0; y < 2; ++y) {
            for (int x = 0; x < 2; ++x) {
                const float coeff = ref3dPtr[(z + 1) * dimFlo[1] * dimFlo[2] + (y + 1) * dimFlo[1] + (x + 1)];
                resLinear3d[0] += coeff * derivLinear[x] * yBasisLinear[y] * zBasisLinear[z];
                resLinear3d[1] += coeff * xBasisLinear[x] * derivLinear[y] * zBasisLinear[z];
                resLinear3d[2] += coeff * xBasisLinear[x] * yBasisLinear[y] * derivLinear[z];
            }
        }
    }

    // Create the test case
    testCases.emplace_back(TestData(
        "Linear 3D",
        reference3d,
        deformationField3d,
        1,
        resLinear3d
    ));

    // Cubic spline image gradient - 3D
    // coordinate in image: [1.2, 1.3, 1.4]
    float resCubic3d[3] = {};
    float zBasisCubic[4], zDerivCubic[4];
    InterpCubicSplineKernel(0.4f, zBasisCubic, zDerivCubic);
    for (int z = 0; z <= 3; ++z) {
        for (int y = 0; y <= 3; ++y) {
            for (int x = 0; x <= 3; ++x) {
                const float coeff = ref3dPtr[z * dimFlo[1] * dimFlo[2] + y * dimFlo[1] + x];
                resCubic3d[0] += coeff * xDerivCubic[x] * yBasisCubic[y] * zBasisCubic[z];
                resCubic3d[1] += coeff * xBasisCubic[x] * yDerivCubic[y] * zBasisCubic[z];
                resCubic3d[2] += coeff * xBasisCubic[x] * yBasisCubic[y] * zDerivCubic[z];
            }
        }
    }

    // Create the test case
    testCases.emplace_back(TestData(
        "Cubic Spline 3D",
        reference3d,
        deformationField3d,
        3,
        resCubic3d
    ));

    // Loop over all generated test cases
    for (auto&& testCase : testCases) {
        // Retrieve test information
        auto&& [testName, reference, defField, interp, testResult] = testCase;
        // Accumulate all required contents with a vector
        vector<ContentDesc> contentDescs;
        for (auto&& platformType : PlatformTypes) {
            if (platformType == PlatformType::Cuda && interp != 1)
                continue;   // CUDA platform only supports linear interpolation
            unique_ptr<Platform> platform{ new Platform(platformType) };
            unique_ptr<DefContentCreator> contentCreator{ dynamic_cast<DefContentCreator*>(platform->CreateContentCreator(ContentType::Def)) };
            unique_ptr<DefContent> content{ contentCreator->Create(reference, reference) };
            contentDescs.push_back({ std::move(content), std::move(platform) });
        }

        // Loop over all possibles contents for each test
        for (auto&& contentDesc : contentDescs) {
            auto&& [content, platform] = contentDesc;
            const std::string sectionName = testName + " " + platform->GetName();
            SECTION(sectionName) {
                NR_COUT << "\n**************** Section " << sectionName << " ****************" << std::endl;

                // Increase the precision for the output
                NR_COUT << std::fixed << std::setprecision(10);

                // Set the warped gradient image to host the computation
                NiftiImage warpedGradient(content->GetWarpedGradient());
                warpedGradient.setDim(NiftiDim::NDim, defField->ndim);
                warpedGradient.setDim(NiftiDim::X, 1);
                warpedGradient.setDim(NiftiDim::Y, 1);
                warpedGradient.setDim(NiftiDim::Z, 1);
                warpedGradient.setDim(NiftiDim::U, defField->nu);
                warpedGradient.recalcVoxelNumber();
                warpedGradient.disown();

                // Set the deformation field
                content->SetDeformationField(defField.disown());

                // Do the computation
                unique_ptr<Compute> compute{ platform->CreateCompute(*content) };
                compute->GetImageGradient(interp, 0, 0);

                // Check all values
                warpedGradient = content->GetWarpedGradient();
                const auto warpedGradPtr = warpedGradient.data();
                const size_t nVoxels = warpedGradient.nVoxels();
                warpedGradient.disown();
                for (size_t i = 0; i < nVoxels; ++i) {
                    const float warpedGradVal = warpedGradPtr[i];
                    const auto diff = abs(warpedGradVal - testResult[i]);
                    if (diff > 0)
                        NR_COUT << i << " " << warpedGradVal << " " << testResult[i] << std::endl;
                    REQUIRE(diff < EPS);
                }
            }
        }
    }
}
