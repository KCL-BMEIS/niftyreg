// OpenCL is not supported for this test
#undef USE_OPENCL

#include "reg_test_common.h"

/*
    CPU unit test of the resampling *operation*

    It exercises, over a full grid and at hand-placed sample coordinates:
      - invariants: identity, integer shift, constant image, fully-out-of-FOV, full coverage;
      - per-tap boundary behaviour and padding (finite and NaN) across all interpolation orders
        (nearest 0, linear 1, cubic 3, windowed-sinc 4);
      - masks (masked-out output voxels left untouched, in both 2D and 3D);
      - multiple time points / channels (no bleed between volumes);
      - realistic geometry (reference != floating, non-square dims, non-identity sform).

    Expected values for the boundary/geometry cases come from mirrorResample(): an
    independent re-implementation of one output voxel of the kernel that reuses the
    trusted weight functions from reg_test_common.h.
*/

constexpr int kNN = 0, kLinear = 1, kCubic = 3, kSinc = 4;
const int allOrders[] = { kNN, kLinear, kCubic, kSinc };

static std::string orderName(int order) {
    switch (order) {
    case kNN: return "nearest";
    case kLinear: return "linear";
    case kSinc: return "sinc";
    default: return "cubic";
    }
}

// Kernel stencil geometry, mirroring _reg_resampling.cpp:346-367
static void kernelGeom(int order, int& size, int& offset) {
    switch (order) {
    case kNN: size = 2; offset = 0; break;
    case kLinear: size = 2; offset = 0; break;
    case kSinc: size = SINC_KERNEL_SIZE; offset = SINC_KERNEL_RADIUS; break;
    default: size = 4; offset = 1; break; // cubic spline
    }
}

// Per-axis weights, dispatching to the replicas in reg_test_common.h
static void kernelWeights(int order, float relative, float* basis) {
    if (relative < 0) relative = 0; // reg_rounding error (matches the lib kernels)
    switch (order) {
    case kNN:
        basis[0] = basis[1] = 0;
        if (relative >= 0.5f) basis[1] = 1; else basis[0] = 1;
        break;
    case kLinear:
        basis[1] = relative;
        basis[0] = 1.f - relative;
        break;
    case kSinc: {
        int j = 0;
        float sum = 0.f;
        for (int i = -SINC_KERNEL_RADIUS; i < SINC_KERNEL_RADIUS; ++i) {
            float x = relative - static_cast<float>(i);
            if (x == 0)
                basis[j] = 1.f;
            else if (fabs(x) >= static_cast<float>(SINC_KERNEL_RADIUS))
                basis[j] = 0;
            else {
                double pi_x = M_PI * static_cast<double>(x);
                basis[j] = static_cast<float>(static_cast<double>(SINC_KERNEL_RADIUS) *
                    sin(pi_x) * sin(pi_x / static_cast<double>(SINC_KERNEL_RADIUS)) / (pi_x * pi_x));
            }
            sum += basis[j];
            j++;
        }
        for (int i = 0; i < SINC_KERNEL_SIZE; ++i) basis[i] /= sum;
    } break;
    default: {
        float b[4];
        InterpCubicSplineKernel(relative, b);
        for (int i = 0; i < 4; ++i) basis[i] = b[i];
    } break;
    }
}

// Independent reference re-implementation of one output voxel of reg_resampleImage, for a float32
// floating image. `world` is the deformation-field value (world coordinates of the reference
// space). Mirrors ResampleImage3D/2D: maps world -> floating voxel coords via floating->sto_ijk,
// gathers the stencil, applying paddingValue per out-of-FOV tap, accumulating in float (FP32),
// matching ResampleImage2D/3D.
static float mirrorResample(const NiftiImage& floating, int order, const float world[3], float pad, bool is3D) {
    int size, offset;
    kernelGeom(order, size, offset);
    const mat44 mat = floating->sform_code > 0 ? floating->sto_ijk : floating->qto_ijk;
    float w[3] = { world[0], world[1], is3D ? world[2] : 0.f };
    float position[3];
    Mat44Mul(mat, w, position);

    const int nx = floating->nx, ny = floating->ny, nz = floating->nz;
    auto floPtr = floating.data();
    auto floVal = [&](int X, int Y, int Z) -> float {
        return static_cast<float>(floPtr[(static_cast<size_t>(Z) * ny + Y) * nx + X]);
    };

    // Mirror ResampleImage2D/3D's FP32 accumulation for every order: `relative` is formed in float,
    // the (double) weight functions are cast down to a float basis, and the blend is done in float.
    int prev[3];
    for (int d = 0; d < 3; ++d) prev[d] = Floor<int>(position[d]);
    float basisX[SINC_KERNEL_SIZE], basisY[SINC_KERNEL_SIZE], basisZ[SINC_KERNEL_SIZE];
    kernelWeights(order, position[0] - static_cast<float>(prev[0]), basisX);
    kernelWeights(order, position[1] - static_cast<float>(prev[1]), basisY);
    if (is3D) kernelWeights(order, position[2] - static_cast<float>(prev[2]), basisZ);
    prev[0] -= offset; prev[1] -= offset; prev[2] -= offset;

    float intensity = 0;
    if (is3D) {
        for (int c = 0; c < size; ++c) {
            const int Z = prev[2] + c;
            float yTemp = 0;
            for (int b = 0; b < size; ++b) {
                const int Y = prev[1] + b;
                float xTemp = 0;
                for (int a = 0; a < size; ++a) {
                    const int X = prev[0] + a;
                    const bool inFov = -1 < X && X < nx && -1 < Y && Y < ny && -1 < Z && Z < nz;
                    xTemp += (inFov ? floVal(X, Y, Z) : pad) * basisX[a];
                }
                yTemp += xTemp * basisY[b];
            }
            intensity += yTemp * basisZ[c];
        }
    } else {
        for (int b = 0; b < size; ++b) {
            const int Y = prev[1] + b;
            float xTemp = 0;
            for (int a = 0; a < size; ++a) {
                const int X = prev[0] + a;
                const bool inFov = -1 < X && X < nx && -1 < Y && Y < ny;
                xTemp += (inFov ? floVal(X, Y, 0) : pad) * basisX[a];
            }
            intensity += xTemp * basisY[b];
        }
    }
    return intensity;
}

// Resample a single output voxel that maps (in world coords) to `world`, sampling `floating`.
// Drives the real operation through Compute::ResampleImage with a one-voxel warped image and a
// one-point deformation field (the same shape reg_test_interpolation uses).
static float probeResample(NiftiImage& floating, int order, const float world[3], float pad, bool is3D) {
    Platform platform(PlatformType::Cpu);
    unique_ptr<ContentCreator> creator{ platform.CreateContentCreator() };
    NiftiImage ref(floating), flo(floating);
    unique_ptr<Content> content{ creator->Create(ref, flo) };

    NiftiImage defField({ 1, 1, 1, 1, is3D ? 3 : 2 }, NIFTI_TYPE_FLOAT32);
    auto defPtr = defField.data();
    defPtr[0] = world[0];
    defPtr[1] = world[1];
    if (is3D) defPtr[2] = world[2];

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
    compute->ResampleImage(order, pad);

    NiftiImage result = std::move(content->GetWarped());
    auto rPtr = result.data();
    return static_cast<float>(rPtr[0]);
}

// Full-grid resample. `defField` is moved in. If `prefill` is non-null the warped buffer is
// filled with *prefill before resampling (to detect untouched / uncovered voxels).
static NiftiImage resampleGrid(NiftiImage& reference, NiftiImage& floating, NiftiImage defField,
                               int* mask, int order, float pad, const float* prefill = nullptr) {
    Platform platform(PlatformType::Cpu);
    unique_ptr<ContentCreator> creator{ platform.CreateContentCreator() };
    unique_ptr<Content> content{ creator->Create(reference, floating, mask) };
    if (prefill) {
        auto wp = content->GetWarped().data();
        const size_t n = content->GetWarped().nVoxels();
        for (size_t i = 0; i < n; ++i) wp[i] = *prefill;
    }
    content->SetDeformationField(std::move(defField));
    unique_ptr<Compute> compute{ platform.CreateCompute(*content) };
    compute->ResampleImage(order, pad);
    NiftiImage out = std::move(content->GetWarped());
    return out;
}

static void requireMatch(float got, float expected) {
    if (std::isnan(expected))
        REQUIRE(std::isnan(got));
    else
        REQUIRE(got == expected);
}

// For closed-form invariants compared against an analytic value (floating, shifted, padding).
// Nearest/linear/cubic weights are exactly 0/1 at integer positions, so they are bit-exact;
// windowed-sinc weights are only approximately 0/1 (sin(M_PI*k) is ~1e-16, not 0), so sinc is
// asserted within a small tolerance. (Sinc is still checked bit-exactly against mirrorResample.)
static void requireResampleEq(float got, float expected, int order) {
    if (order == kSinc) {
        const float scale = std::abs(expected) > 1.f ? std::abs(expected) : 1.f;
        REQUIRE(std::abs(got - expected) <= 1e-4f * scale);
    } else {
        REQUIRE(got == expected);
    }
}

TEST_CASE("Resample image operation", "[unit]") {
    constexpr NiftiImage::dim_t D = 8; // big enough that the 6-tap sinc stencil has interior room

    SECTION("Identity deformation => warped == floating") {
        for (bool is3D : { false, true }) {
            NiftiImage floating = is3D ? makeImage({ D, D, D }) : makeImage({ D, D });
            for (int order : allOrders) {
                INFO((is3D ? "3D " : "2D ") << orderName(order));
                NiftiImage reference(floating);
                NiftiImage def = CreateDeformationField(reference); // identity (world coords)
                NiftiImage warped = resampleGrid(reference, floating, std::move(def), nullptr, order, 0.f);
                auto wp = warped.data();
                auto fp = floating.data();
                for (size_t i = 0; i < warped.nVoxels(); ++i)
                    requireResampleEq(static_cast<float>(wp[i]), static_cast<float>(fp[i]), order);
            }
        }
    }

    SECTION("Integer-translation deformation => warped == shifted floating") {
        const int shift = 2;
        for (bool is3D : { false, true }) {
            NiftiImage floating = is3D ? makeImage({ D, D, D }) : makeImage({ D, D });
            const int nx = floating->nx, ny = floating->ny, nz = floating->nz;
            for (int order : allOrders) {
                INFO((is3D ? "3D " : "2D ") << orderName(order));
                NiftiImage reference(floating);
                NiftiImage def = CreateDeformationField(reference);
                const size_t nVox = def.nVoxelsPerVolume();
                auto dp = def.data();
                for (size_t i = 0; i < nVox; ++i) // shift the x component by `shift` voxels (identity sform)
                    dp[i] = static_cast<float>(dp[i]) + shift;
                NiftiImage warped = resampleGrid(reference, floating, std::move(def), nullptr, order, 0.f);
                auto wp = warped.data();
                auto fp = floating.data();
                for (int z = 0; z < nz; ++z)
                    for (int y = 0; y < ny; ++y)
                        for (int x = 0; x < nx; ++x) {
                            const size_t idx = (static_cast<size_t>(z) * ny + y) * nx + x;
                            const int sx = x + shift;
                            const float expected = sx < nx ? static_cast<float>(fp[(static_cast<size_t>(z) * ny + y) * nx + sx]) : 0.f;
                            requireResampleEq(static_cast<float>(wp[idx]), expected, order);
                        }
            }
        }
    }

    SECTION("Constant image => warped == constant at interior voxels") {
        const float k = 1.f;
        const float subVoxel = 0.3f;
        for (bool is3D : { false, true }) {
            std::vector<NiftiImage::dim_t> dims = is3D ? std::vector<NiftiImage::dim_t>{ D, D, D }
                                                       : std::vector<NiftiImage::dim_t>{ D, D };
            NiftiImage floating(dims, NIFTI_TYPE_FLOAT32);
            setIdentitySform(floating);
            { auto fp = floating.data(); for (size_t i = 0; i < floating.nVoxels(); ++i) fp[i] = k; }
            for (int order : allOrders) {
                INFO((is3D ? "3D " : "2D ") << orderName(order));
                NiftiImage reference(floating);
                NiftiImage def = CreateDeformationField(reference);
                { auto dp = def.data(); for (size_t i = 0; i < def.nVoxels(); ++i) dp[i] = static_cast<float>(dp[i]) + subVoxel; }
                NiftiImage warped = resampleGrid(reference, floating, std::move(def), nullptr, order, 0.f);
                auto wp = warped.data();
                const int nx = floating->nx, ny = floating->ny, nz = floating->nz;
                const int lo = SINC_KERNEL_RADIUS, hiX = nx - 1 - SINC_KERNEL_RADIUS; // fully-interior band for every order
                const int hiY = ny - 1 - SINC_KERNEL_RADIUS, hiZ = nz - 1 - SINC_KERNEL_RADIUS;
                for (int z = (is3D ? lo : 0); z <= (is3D ? hiZ : 0); ++z)
                    for (int y = lo; y <= hiY; ++y)
                        for (int x = lo; x <= hiX; ++x) {
                            const size_t idx = (static_cast<size_t>(z) * ny + y) * nx + x;
                            REQUIRE(std::abs(static_cast<float>(wp[idx]) - k) < 1e-5f);
                        }
            }
        }
    }

    SECTION("Fully-out-of-FOV deformation => warped == padding") {
        for (bool is3D : { false, true }) {
            NiftiImage floating = is3D ? makeImage({ D, D, D }) : makeImage({ D, D });
            for (int order : allOrders) {
                for (float pad : { 0.f, std::numeric_limits<float>::quiet_NaN() }) {
                    INFO((is3D ? "3D " : "2D ") << orderName(order) << " pad=" << pad);
                    NiftiImage reference(floating);
                    NiftiImage def = CreateDeformationField(reference);
                    const size_t nVox = def.nVoxelsPerVolume();
                    auto dp = def.data();
                    for (size_t i = 0; i < nVox; ++i) // push the x component far outside the FOV
                        dp[i] = static_cast<float>(dp[i]) + 1000.f;
                    NiftiImage warped = resampleGrid(reference, floating, std::move(def), nullptr, order, pad);
                    auto wp = warped.data();
                    for (size_t i = 0; i < warped.nVoxels(); ++i) {
                        if (std::isnan(pad)) REQUIRE(std::isnan(static_cast<float>(wp[i])));
                        else REQUIRE(static_cast<float>(wp[i]) == 0.f);
                    }
                }
            }
        }
    }

    SECTION("Large-grid coverage => no voxel left at its initial value") {
        // Primarily a GPU launch-config guard; on the CPU it confirms every active voxel is written.
        for (bool is3D : { false, true }) {
            NiftiImage floating = is3D ? makeImage({ 12, 11, 10 }) : makeImage({ 12, 11 });
            const float sentinel = -12345.f;
            for (int order : allOrders) {
                INFO((is3D ? "3D " : "2D ") << orderName(order));
                NiftiImage reference(floating);
                NiftiImage def = CreateDeformationField(reference); // identity -> samples == floating (!= sentinel)
                NiftiImage warped = resampleGrid(reference, floating, std::move(def), nullptr, order, 0.f, &sentinel);
                auto wp = warped.data();
                for (size_t i = 0; i < warped.nVoxels(); ++i)
                    REQUIRE(static_cast<float>(wp[i]) != sentinel);
            }
        }
    }

    SECTION("Boundary, padding and NaN behaviour (probe vs reference impl)") {
        // Probe the x axis across the FOV while keeping y,z at an interior integer (4) so every
        // order's y/z stencil is fully in-FOV and only x exercises the boundary.
        const double cs[] = { 4.3, 4.0, 0.0, 7.0, -0.3, 7.4, 0.5, 6.5, 1.5, 5.5, 2.5, -1.0, 8.0, -2.5, 9.7 };
        const float pads[] = { 0.f, -1.f, 100.f, std::numeric_limits<float>::quiet_NaN() };
        for (bool is3D : { false, true }) {
            NiftiImage floating = is3D ? makeImage({ D, D, D }) : makeImage({ D, D });
            for (int order : allOrders)
                for (float pad : pads)
                    for (double c : cs) {
                        const float world[3] = { static_cast<float>(c), 4.f, 4.f };
                        INFO((is3D ? "3D " : "2D ") << orderName(order) << " c=" << c << " pad=" << pad);
                        requireMatch(probeResample(floating, order, world, pad, is3D),
                                     mirrorResample(floating, order, world, pad, is3D));
                    }
        }
    }

    SECTION("Exact upper edge under NaN padding is NaN (pinned), finite otherwise") {
        // A sample landing exactly on the last valid voxel (c == D-1) returns flo[D-1] with finite
        // padding, but NaN under NaN padding, because the zero-weight out-of-FOV tap does 0*NaN.
        // This is the resampler's current contract; the CUDA/refactor work must preserve it.
        for (bool is3D : { false, true }) {
            NiftiImage floating = is3D ? makeImage({ D, D, D }) : makeImage({ D, D });
            auto fp = floating.data();
            const int nx = floating->nx, ny = floating->ny;
            const float edge[3] = { static_cast<float>(D - 1), 4.f, 4.f };
            const float onVoxel = static_cast<float>(fp[(static_cast<size_t>(is3D ? 4 : 0) * ny + 4) * nx + (D - 1)]);
            for (int order : allOrders) {
                INFO((is3D ? "3D " : "2D ") << orderName(order));
                requireResampleEq(probeResample(floating, order, edge, 0.f, is3D), onVoxel, order);
                REQUIRE(std::isnan(probeResample(floating, order, edge, std::numeric_limits<float>::quiet_NaN(), is3D)));
            }
        }
    }

    SECTION("Nearest-neighbour tie at k+0.5 rounds up") {
        for (bool is3D : { false, true }) {
            NiftiImage floating = is3D ? makeImage({ D, D, D }) : makeImage({ D, D });
            auto fp = floating.data();
            const int nx = floating->nx, ny = floating->ny;
            const int yz = is3D ? 4 : 0;
            const float tie[3] = { 4.5f, 4.f, 4.f };  // -> rounds up to x = 5
            const float below[3] = { 4.4f, 4.f, 4.f }; // -> rounds down to x = 4
            REQUIRE(probeResample(floating, kNN, tie, 0.f, is3D) ==
                    static_cast<float>(fp[(static_cast<size_t>(yz) * ny + 4) * nx + 5]));
            REQUIRE(probeResample(floating, kNN, below, 0.f, is3D) ==
                    static_cast<float>(fp[(static_cast<size_t>(yz) * ny + 4) * nx + 4]));
        }
    }

    SECTION("Masked-out voxels are left untouched (2D and 3D)") {
        const float sentinel = -777.f;
        for (bool is3D : { false, true }) {
            NiftiImage floating = is3D ? makeImage({ D, D, D }) : makeImage({ D, D });
            NiftiImage reference(floating);
            const size_t nVox = reference.nVoxelsPerVolume();
            // Partial mask: deactivate every third voxel; also deactivate one that maps out-of-FOV.
            std::vector<int> mask(nVox, 0);
            for (size_t i = 0; i < nVox; ++i) if (i % 3 == 0) mask[i] = -1;
            for (int order : allOrders) {
                INFO((is3D ? "3D " : "2D ") << orderName(order));
                NiftiImage def = CreateDeformationField(reference); // identity
                NiftiImage warped = resampleGrid(reference, floating, std::move(def), mask.data(), order, 0.f, &sentinel);
                auto wp = warped.data();
                auto fp = floating.data();
                for (size_t i = 0; i < nVox; ++i) {
                    if (mask[i] > -1)
                        requireResampleEq(static_cast<float>(wp[i]), static_cast<float>(fp[i]), order); // active: resampled (identity)
                    else
                        REQUIRE(static_cast<float>(wp[i]) == sentinel);                                 // masked-out: untouched
                }
            }
        }
    }

    SECTION("Multiple time points do not bleed into each other") {
        NiftiImage floating = makeImage({ D, D, D, 3 }); // nt = 3, distinct values per volume
        NiftiImage reference = makeImage({ D, D, D });   // 3D deformation grid, nt = 1
        for (int order : allOrders) {
            INFO(orderName(order));
            NiftiImage def = CreateDeformationField(reference); // identity
            NiftiImage warped = resampleGrid(reference, floating, std::move(def), nullptr, order, 0.f);
            REQUIRE(warped->nt == 3);
            auto wp = warped.data();
            auto fp = floating.data();
            for (size_t i = 0; i < warped.nVoxels(); ++i)
                requireResampleEq(static_cast<float>(wp[i]), static_cast<float>(fp[i]), order);
        }
    }

    SECTION("Realistic geometry: reference != floating, non-square dims, non-identity sform") {
        NiftiImage floating = makeImage({ D, D, D });
        mat44 m;
        Mat44Eye(&m);
        m.m[0][0] = 1.2f; m.m[1][1] = 0.9f; m.m[2][2] = 1.1f; // anisotropic scale
        m.m[0][1] = 0.1f; m.m[0][2] = 0.05f; m.m[1][2] = -0.07f; // shear
        m.m[0][3] = 0.5f; m.m[1][3] = 0.3f; m.m[2][3] = 0.2f;   // translation
        setSform(floating, m);

        NiftiImage reference = makeImage({ 6, 5, 4 }); // non-square output grid, identity sform
        const size_t nVox = reference.nVoxelsPerVolume();
        for (int order : allOrders) {
            for (float pad : { 0.f, std::numeric_limits<float>::quiet_NaN() }) {
                INFO(orderName(order) << " pad=" << pad);
                NiftiImage def = CreateDeformationField(reference);
                auto dp = def.data();
                for (size_t i = 0; i < nVox; ++i) { // add a fractional offset so sampling is interpolated
                    dp[i] = static_cast<float>(dp[i]) + 0.37f;
                    dp[nVox + i] = static_cast<float>(dp[nVox + i]) - 0.21f;
                    dp[2 * nVox + i] = static_cast<float>(dp[2 * nVox + i]) + 0.13f;
                }
                // Snapshot the expected values before the deformation field is moved into the resampler.
                std::vector<float> expected(nVox);
                for (size_t i = 0; i < nVox; ++i) {
                    const float world[3] = { static_cast<float>(dp[i]),
                                             static_cast<float>(dp[nVox + i]),
                                             static_cast<float>(dp[2 * nVox + i]) };
                    expected[i] = mirrorResample(floating, order, world, pad, true);
                }
                NiftiImage warped = resampleGrid(reference, floating, std::move(def), nullptr, order, pad);
                auto wp = warped.data();
                for (size_t i = 0; i < nVox; ++i)
                    requireMatch(static_cast<float>(wp[i]), expected[i]);
            }
        }
    }
}
