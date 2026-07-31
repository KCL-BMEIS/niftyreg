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
                NiftiImage& warpedGradient = content->GetWarpedGradient();
                warpedGradient.setDim(NiftiDim::NDim, defField->ndim);
                warpedGradient.setDim(NiftiDim::X, 1);
                warpedGradient.setDim(NiftiDim::Y, 1);
                warpedGradient.setDim(NiftiDim::Z, 1);
                warpedGradient.setDim(NiftiDim::U, defField->nu);
                warpedGradient.recalcVoxelNumber();

                // Set the deformation field
                content->SetDeformationField(std::move(defField));

                // Do the computation
                unique_ptr<Compute> compute{ platform->CreateCompute(*content) };
                compute->GetImageGradient(interp, 0, 0);

                // Check all values
                content->GetWarpedGradient();
                const auto warpedGradPtr = warpedGradient.data();
                const size_t nVoxels = warpedGradient.nVoxels();
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

namespace {

// Full-grid gradient through Compute::GetImageGradient: floating image, deformation field in, warped
// gradient out. Returns the warped gradient.
NiftiImage GradientGrid(NiftiImage& floating, NiftiImage defField, int interp, float padding) {
    Platform platform(PlatformType::Cpu);
    unique_ptr<DefContentCreator> creator{
        dynamic_cast<DefContentCreator*>(platform.CreateContentCreator(ContentType::Def)) };
    NiftiImage ref(floating), flo(floating);
    unique_ptr<DefContent> content{ creator->Create(ref, flo) };
    content->SetDeformationField(std::move(defField));
    unique_ptr<Compute> compute{ platform.CreateCompute(*content) };
    compute->GetImageGradient(interp, padding, 0);
    NiftiImage out = std::move(content->GetWarpedGradient());
    return out;
}

} // namespace

TEST_CASE("Image gradient of a linear ramp is its slope", "[unit]") {
    /*
        f(v) = c0 + cx vx + cy vy + cz vz: both the linear and the cubic gradient kernels have linear
        precision, so away from the boundary the gradient is exactly (cx, cy, cz) at ANY sub-voxel
        position - the closed form the single-point cases above cannot provide. (The production
        gradient only implements linear and cubic; there is no sinc gradient path.)
    */
    const double slope[3] = { 0.75, -0.5, 0.25 };
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
                                    1.5 + slope[0] * i + slope[1] * j + (is3D ? slope[2] * k : 0.0));
                }

                // Identity plus a fractional offset, so the sampling is genuinely interpolated
                NiftiImage defField = CreateDeformationField(floating);
                const size_t nVox = defField.nVoxelsPerVolume();
                {
                    auto dp = defField.data();
                    for (size_t i = 0; i < nVox; ++i) {
                        dp[i] = static_cast<float>(dp[i]) + 0.37f;
                        dp[nVox + i] = static_cast<float>(dp[nVox + i]) - 0.21f;
                        if (is3D) dp[2 * nVox + i] = static_cast<float>(dp[2 * nVox + i]) + 0.13f;
                    }
                }

                NiftiImage gradient = GradientGrid(floating, std::move(defField), interp, 0.f);
                const auto ptr = gradient.data();
                const int components = is3D ? 3 : 2;

                // Stay far enough inside that no kernel tap reaches the boundary after the offsets
                const int lo = 3, hi = 8 - 4;
                const int nx = 8, ny = 8, nz = is3D ? 8 : 1;
                size_t checked = 0;
                double maxDeviation = 0;
                for (int k = (is3D ? lo : 0); k <= (is3D ? hi : 0); ++k)
                    for (int j = lo; j <= hi; ++j)
                        for (int i = lo; i <= hi; ++i) {
                            const size_t index = (static_cast<size_t>(k) * ny + j) * nx + i;
                            for (int c = 0; c < components; ++c)
                                maxDeviation = std::max(maxDeviation,
                                    std::abs(static_cast<double>(ptr[c * nVox + index]) - slope[c]));
                            ++checked;
                        }
                NR_COUT << "  " << (is3D ? "3D" : "2D") << (interp == 3 ? " cubic" : " linear")
                        << ": " << checked << " voxels, max |gradient - slope| = "
                        << std::scientific << maxDeviation << std::endl;
                REQUIRE(checked > 0);
                REQUIRE(maxDeviation < 1e-5);
            }
        }
}

TEST_CASE("Image gradient of a constant image is zero", "[unit]") {
    // Every finite difference of a constant vanishes; with the padding set to the same constant the
    // boundary taps do too, so the gradient is exactly zero at every voxel
    constexpr float k = 2.5f;
    for (const bool is3D : { false, true })
        for (const int interp : { 1, 3 }) {
            SECTION(std::string(is3D ? "3D" : "2D") + (interp == 3 ? " cubic" : " linear")) {
                std::vector<NiftiImage::dim_t> dims(is3D ? 3 : 2, 8);
                NiftiImage floating(dims, NIFTI_TYPE_FLOAT32);
                setIdentitySform(floating);
                { auto ptr = floating.data(); for (size_t i = 0; i < floating.nVoxels(); ++i) ptr[i] = k; }

                NiftiImage defField = CreateDeformationField(floating);
                { auto dp = defField.data(); for (size_t i = 0; i < defField.nVoxels(); ++i) dp[i] = static_cast<float>(dp[i]) + 0.3f; }

                NiftiImage gradient = GradientGrid(floating, std::move(defField), interp, k);
                const auto ptr = gradient.data();
                for (size_t i = 0; i < gradient.nVoxels(); ++i) {
                    INFO("value " << i << " = " << static_cast<float>(ptr[i]));
                    // The linear weights cancel exactly ((1-t)+t and -1+1); the cubic kernel's float
                    // weights only sum to 0/1 within rounding, so its boundary voxels carry ~1e-7
                    if (interp == 1)
                        REQUIRE(static_cast<float>(ptr[i]) == 0.f);
                    else
                        REQUIRE(std::abs(static_cast<float>(ptr[i])) < 1e-5f);
                }
            }
        }
}

TEST_CASE("Image gradient under NaN padding is zeroed outside the field of view", "[unit]") {
    /*
        Production replaces a non-finite gradient with zero (TrilinearImageGradient: grad != grad -> 0).
        With NaN padding, any voxel whose stencil leaves the floating image therefore holds exactly
        zero, and fully-interior voxels are untouched by the padding value. Pinned as the contract the
        measures rely on: an out-of-FOV voxel contributes no gradient rather than a NaN that would
        poison every later reduction.
    */
    const double slope[3] = { 0.75, -0.5, 0.25 };
    for (const bool is3D : { false, true }) {
        SECTION(is3D ? "3D" : "2D") {
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
                                1.5 + slope[0] * i + slope[1] * j + (is3D ? slope[2] * k : 0.0));
            }

            // Push the x sampling positions up by 4.5 voxels: the right half of the grid samples
            // outside the floating image
            NiftiImage defField = CreateDeformationField(floating);
            const size_t nVox = defField.nVoxelsPerVolume();
            { auto dp = defField.data(); for (size_t i = 0; i < nVox; ++i) dp[i] = static_cast<float>(dp[i]) + 4.5f; }

            NiftiImage gradient = GradientGrid(floating, std::move(defField),
                                               1, std::numeric_limits<float>::quiet_NaN());
            const auto ptr = gradient.data();
            const int components = is3D ? 3 : 2;
            const int nx = 8, ny = 8, nz = is3D ? 8 : 1;
            size_t zeroed = 0, interior = 0;
            for (int k = 0; k < nz; ++k)
                for (int j = 0; j < ny; ++j)
                    for (int i = 0; i < nx; ++i) {
                        const size_t index = (static_cast<size_t>(k) * ny + j) * nx + i;
                        const double x = i + 4.5;   // sampled x position
                        // A voxel is poisoned when ANY tap leaves the image - including the
                        // zero-weight tap of a position exactly on the last row/slice (0 * NaN is
                        // NaN, the same contract the resampler pins at the exact upper edge)
                        const bool outOfFov = x > 6.5 || j == ny - 1 || (is3D && k == nz - 1);
                        if (outOfFov) {
                            for (int c = 0; c < components; ++c) {
                                INFO("out-of-FOV voxel (" << i << "," << j << "," << k << ") component " << c);
                                REQUIRE(static_cast<float>(ptr[c * nVox + index]) == 0.f);
                            }
                            ++zeroed;
                        } else {
                            // Fully inside: the ramp's slope, exactly as in the interior case
                            for (int c = 0; c < components; ++c) {
                                INFO("interior voxel (" << i << "," << j << "," << k << ") component " << c);
                                REQUIRE(std::abs(static_cast<double>(ptr[c * nVox + index]) - slope[c]) < 1e-5);
                            }
                            ++interior;
                        }
                    }
            NR_COUT << "  " << (is3D ? "3D" : "2D") << " NaN padding: " << zeroed << " zeroed, "
                    << interior << " interior" << std::endl;
            REQUIRE(zeroed > 0);
            REQUIRE(interior > 0);
        }
    }
}
