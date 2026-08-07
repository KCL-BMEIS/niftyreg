// OpenCL is not supported for this test
#undef USE_OPENCL

#include "reg_test_common.h"

/*
    This test file checks the interpolation *kernels* (the point-wise interpolation
    weights). Each case resamples a single voxel through a constant one-point deformation
    field and compares against an analytically computed value, in 2D and 3D, for:
    Nearest neighbour
    Linear
    Cubic spline
*/


typedef std::tuple<std::string, NiftiImage, NiftiImage, int, float*> TestData;
typedef std::tuple<unique_ptr<Content>, shared_ptr<Platform>> ContentDesc;

TEST_CASE("Interpolation kernels", "[unit]") {
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

    // Linear interpolation - 2D
    // coordinate in image: [1.2, 1.3]
    float resLinear2d[1] = {};
    for (int y = 1; y <= 2; ++y) {
        for (int x = 1; x <= 2; ++x) {
            resLinear2d[0] += static_cast<float>(ref2dPtr[y * dimFlo[1] + x]) *
                abs(2.0f - static_cast<float>(x) - 0.2f) *
                abs(2.0f - static_cast<float>(y) - 0.3f);
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

    // Nearest neighbour interpolation - 2D
    // coordinate in image: [1.2, 1.3]
    float resNearest2d[1];
    resNearest2d[0] = ref2dPtr[1 * dimFlo[1] + 1];

    // Create the test case
    testCases.emplace_back(TestData(
        "Nearest Neighbour 2D",
        reference2d,
        deformationField2d,
        0,
        resNearest2d
    ));

    // Cubic spline interpolation - 2D
    // coordinate in image: [1.2, 1.3]
    float resCubic2d[1] = {};
    float xBasis[4], yBasis[4];
    InterpCubicSplineKernel(0.2f, xBasis);
    InterpCubicSplineKernel(0.3f, yBasis);
    for (int y = 0; y <= 3; ++y)
        for (int x = 0; x <= 3; ++x)
            resCubic2d[0] += static_cast<float>(ref2dPtr[y * dimFlo[1] + x]) * xBasis[x] * yBasis[y];

    // Create the test case
    testCases.emplace_back(TestData(
        "Cubic Spline 2D",
        reference2d,
        deformationField2d,
        3,
        resCubic2d
    ));

    // Linear interpolation - 3D
    // coordinate in image: [1.2, 1.3, 1.4]
    float resLinear3d[1] = {};
    for (int z = 1; z <= 2; ++z) {
        for (int y = 1; y <= 2; ++y) {
            for (int x = 1; x <= 2; ++x) {
                resLinear3d[0] += static_cast<float>(ref3dPtr[z * dimFlo[1] * dimFlo[2] + y * dimFlo[1] + x]) *
                    abs(2.0f - static_cast<float>(x) - 0.2f) *
                    abs(2.0f - static_cast<float>(y) - 0.3f) *
                    abs(2.0f - static_cast<float>(z) - 0.4f);
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

    // Nearest neighbour interpolation - 3D
    // coordinate in image: [1.2, 1.3, 1.4]
    float resNearest3d[1];
    resNearest3d[0] = ref3dPtr[1 * dimFlo[2] * dimFlo[1] + 1 * dimFlo[1] + 1];

    // Create the test case
    testCases.emplace_back(TestData(
        "Nearest Neighbour 3D",
        reference3d,
        deformationField3d,
        0,
        resNearest3d
    ));

    // Cubic spline interpolation - 3D
    // coordinate in image: [1.2, 1.3, 1.4]
    float resCubic3d[1] = {};
    float zBasis[4];
    InterpCubicSplineKernel(0.4f, zBasis);
    for (int z = 0; z <= 3; ++z)
        for (int y = 0; y <= 3; ++y)
            for (int x = 0; x <= 3; ++x)
                resCubic3d[0] += static_cast<float>(ref3dPtr[z * dimFlo[1] * dimFlo[2] + y * dimFlo[1] + x]) * xBasis[x] * yBasis[y] * zBasis[z];

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
            shared_ptr<Platform> platform{ new Platform(platformType) };
            // CUDA supports linear interpolation only (both the Aladin and Base content resample
            // through Cuda::ResampleImage); nearest/cubic are covered on the CPU.
            if (platformType == PlatformType::Cuda && interp != 1)
                continue;
            // Add Aladin content
            unique_ptr<AladinContentCreator> aladinContentCreator{ dynamic_cast<AladinContentCreator*>(platform->CreateContentCreator(ContentType::Aladin)) };
            unique_ptr<AladinContent> aladinContent{ aladinContentCreator->Create(reference, reference) };
            contentDescs.push_back(ContentDesc(std::move(aladinContent), platform));
            // Add Base content
            unique_ptr<ContentCreator> contentCreator{ dynamic_cast<ContentCreator*>(platform->CreateContentCreator()) };
            unique_ptr<Content> content{ contentCreator->Create(reference, reference) };
            contentDescs.push_back({ std::move(content), platform });
        }

        // Loop over all possibles contents for each test
        for (auto&& contentDesc : contentDescs) {
            auto&& [content, platform] = contentDesc;
            const bool isAladinContent = dynamic_cast<AladinContent*>(content.get());
            auto contentName = isAladinContent ? "Aladin" : "Base";
            const std::string sectionName = testName + " " + platform->GetName() + " - " + contentName;
            SECTION(sectionName) {
                NR_COUT << "\n**************** Section " << sectionName << " ****************" << std::endl;

                // Increase the precision for the output
                NR_COUT << std::fixed << std::setprecision(10);

                // Create and set a warped image to host the computation
                NiftiImage warped(defField, NiftiImage::Copy::ImageInfo);
                warped.setDim(NiftiDim::NDim, defField->nu);
                warped.setDim(NiftiDim::X, 1);
                warped.setDim(NiftiDim::Y, 1);
                warped.setDim(NiftiDim::Z, 1);
                warped.setDim(NiftiDim::U, 1);
                warped.realloc();
                content->SetWarped(std::move(warped));

                // Set the deformation field
                content->SetDeformationField(std::move(defField));

                // Do the computation
                if (isAladinContent) {
                    unique_ptr<Kernel> resampleKernel{ platform->CreateKernel(ResampleImageKernel::GetName(), content.get()) };
                    resampleKernel->castTo<ResampleImageKernel>()->Calculate(interp, 0);
                } else {
                    unique_ptr<Compute> compute{ platform->CreateCompute(*content) };
                    compute->ResampleImage(interp, 0);
                }

                // Check all values
                warped = std::move(content->GetWarped());
                const auto warpedPtr = warped.data();
                const size_t nVoxels = warped.nVoxels();
                for (size_t i = 0; i < nVoxels; ++i) {
                    const float warpedValue = warpedPtr[i];
                    const float diff = abs(warpedValue - testResult[i]);
                    if (diff > 0)
                        NR_COUT << i << " " << warpedValue << " " << testResult[i] << std::endl;
                    REQUIRE(diff < EPS);
                }
            }
        }
    }
}

TEST_CASE("Interpolation kernels reproduce a linear ramp", "[unit]") {
    /*
        The cases above check one hand-computed point per kernel. This one checks the property that
        makes interpolation trustworthy between voxels: both the linear and the cubic (Catmull-Rom)
        kernels have linear precision, so sampling f(v) = c0 + cx vx + cy vy [+ cz vz] at ANY position
        must return exactly the ramp's value there - independent of any weight table, copied or not.
        Positions are multiples of 0.25 so the float weights are exact; the ramp coefficients are
        dyadic; tolerance covers the cubic kernel's float literals.
    */
    const double c0 = 1.5, cx = 0.75, cy = -0.5, cz = 0.25;
    for (const bool is3D : { false, true })
        for (const int interp : { 1, 3 }) {
            SECTION(std::string(is3D ? "3D" : "2D") + (interp == 3 ? " cubic" : " linear")) {
                std::vector<NiftiImage::dim_t> dims(is3D ? 3 : 2, 8);
                NiftiImage floating(dims, NIFTI_TYPE_FLOAT32);
                setIdentitySform(floating);
                {
                    auto ptr = floating.data();
                    const int nx = floating->nx, ny = floating->ny, nz = floating->nz;
                    for (int k = 0; k < nz; ++k)
                        for (int j = 0; j < ny; ++j)
                            for (int i = 0; i < nx; ++i)
                                ptr[(static_cast<size_t>(k) * ny + j) * nx + i] = static_cast<float>(
                                    c0 + cx * i + cy * j + (is3D ? cz * k : 0.0));
                }

                // Interior sample positions on quarter-voxel offsets (cubic stencil needs a margin)
                const double positions[][3] = {
                    { 3.25, 3.5, 3.75 }, { 2.5, 4.25, 3.0 }, { 4.75, 2.75, 4.5 }, { 3.0, 3.0, 3.0 },
                };
                for (const auto& p : positions) {
                    // One-voxel warped image and one-point deformation field, as the cases above
                    NiftiImage defField({ 1, 1, 1, 1, is3D ? 3 : 2 }, NIFTI_TYPE_FLOAT32);
                    auto defPtr = defField.data();
                    defPtr[0] = static_cast<float>(p[0]);
                    defPtr[1] = static_cast<float>(p[1]);
                    if (is3D) defPtr[2] = static_cast<float>(p[2]);

                    Platform platform(PlatformType::Cpu);
                    unique_ptr<ContentCreator> creator{ platform.CreateContentCreator() };
                    NiftiImage ref(floating), flo(floating);
                    unique_ptr<Content> content{ creator->Create(ref, flo) };
                    NiftiImage warped(defField, NiftiImage::Copy::ImageInfo);
                    warped.setDim(NiftiDim::NDim, defField->nu);
                    warped.setDim(NiftiDim::X, 1);
                    warped.setDim(NiftiDim::Y, 1);
                    warped.setDim(NiftiDim::Z, 1);
                    warped.setDim(NiftiDim::U, 1);
                    warped.realloc();
                    content->SetWarped(std::move(warped));
                    content->SetDeformationField(std::move(defField));
                    unique_ptr<Compute> compute{ platform.CreateCompute(*content) };
                    compute->ResampleImage(interp, 0);
                    const float actual = static_cast<float>(content->GetWarped().data()[0]);

                    const double expected = c0 + cx * p[0] + cy * p[1] + (is3D ? cz * p[2] : 0.0);
                    INFO("position (" << p[0] << "," << p[1] << "," << p[2] << ")");
                    REQUIRE(std::abs(actual - expected) < 1e-5);
                }
            }
        }
}
