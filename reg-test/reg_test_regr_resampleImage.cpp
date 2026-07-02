// OpenCL is not supported for this test
#undef USE_OPENCL

#include "reg_test_common.h"

/*
    CPU<->CUDA bit-exact regression test for the forward image-resampling *operation*
    For each parameter combination the same reference / floating /
    deformation-field / mask / padding is warped on the CPU platform and on the CUDA platform,
    through the identical Platform/Content/Compute abstraction, and the two warped images
    are required to match.

    The build sets USE_CUDA_FMA=OFF (--fmad=false + matched host -ffp-contract) and both
    resamplers accumulate the linear blend in double, so finite voxels must be equal.

    Two comparison subtleties this test handles:
      - NaN padding: an out-of-FOV tap contributes paddingValue * weight, and a zero-weight tap
        still yields 0*NaN = NaN, so voxels with any out-of-FOV tap are legitimately NaN. Since
        NaN == NaN is false, NaN is compared explicitly (a voxel must be NaN on both platforms or
        finite-and-equal on both).
      - Masked-out voxels: neither kernel writes them (CPU gates on mask; CUDA iterates a compacted
        active-voxel list), and the CUDA warped buffer is uninitialised cudaMalloc while the CPU
        buffer is untouched host memory - so masked-out voxels are undefined and platform-specific.
        Masked cases therefore compare ACTIVE voxels only; "masked-out left untouched" is a
        per-platform property already pinned by the CPU-only reg_test_resampleImage.

    The CUDA resampler is linear-only (CudaResampling.cu fatal-errors otherwise), so interpolation
    is fixed at 1 here
*/

constexpr int kLinear = 1; // the only interpolation the GPU supports

// ---- deformation-field builders ----
// CreateDeformationField() yields the identity transform (each field value is the world
// coordinate of that reference voxel). The variants below perturb it to exercise different code
// paths shared by both platforms.
enum FieldKind {
    kIdentity,  // warped == floating (all interpolation weights 0/1)
    kPerturbed, // identity + U(-1.5, 1.5) per component -> interior + boundary straddle + some out-of-FOV, incl. samples near voxel 0
    kIntShift,  // identity + integer x-shift -> pure addressing, no interpolation
    kFullyOut   // identity + huge x-shift -> every sample fully out-of-FOV (all-padding branch)
};

static NiftiImage buildField(const NiftiImage& reference, FieldKind kind, std::mt19937& gen) {
    NiftiImage def = CreateDeformationField(reference); // identity, world coordinates
    const size_t nVoxPerVol = def.nVoxelsPerVolume();
    auto dp = def.data();
    switch (kind) {
    case kIdentity:
        break;
    case kPerturbed: {
        std::uniform_real_distribution<float> d(-1.5f, 1.5f);
        const size_t n = def.nVoxels(); // every component of every voxel
        for (size_t i = 0; i < n; ++i)
            dp[i] = static_cast<float>(dp[i]) + d(gen);
        break;
    }
    case kIntShift:
        for (size_t i = 0; i < nVoxPerVol; ++i) // x component only
            dp[i] = static_cast<float>(dp[i]) + 3.f;
        break;
    case kFullyOut:
        for (size_t i = 0; i < nVoxPerVol; ++i) // push x far outside the FOV
            dp[i] = static_cast<float>(dp[i]) + 1000.f;
        break;
    }
    return def;
}

// Warp `floating` onto `reference`'s grid on the given platform, through the same
// ContentType::Base / Compute::ResampleImage path reg_resample will use. `defField` is copied in;
// `mask` (0 = active, -1 = inactive) must outlive the call. Returns the warped image - for CUDA,
// GetWarped() performs the device->host download (UpdateWarped() is the opposite direction and
// would clobber the result), and on the CPU it is a plain accessor, so the same call serves both.
static NiftiImage warpImage(PlatformType platformType, const NiftiImage& reference,
                            const NiftiImage& floating, NiftiImage defField,
                            std::vector<int>& mask, float pad) {
    Platform platform(platformType);
    unique_ptr<ContentCreator> creator{ platform.CreateContentCreator(ContentType::Base) };
    NiftiImage ref(reference), flo(floating); // fresh deep copies so each platform is independent
    unique_ptr<Content> content{ creator->Create(ref, flo, mask.data()) };
    content->SetDeformationField(std::move(defField)); // CudaContent override uploads to device
    unique_ptr<Compute> compute{ platform.CreateCompute(*content) };
    compute->ResampleImage(kLinear, pad);
    NiftiImage warped = std::move(content->GetWarped());
    return warped;
}

class ResampleImageTest {
protected:
    using TestCase = std::tuple<std::string, NiftiImage, NiftiImage, std::vector<int>>;
    inline static vector<TestCase> testCases;

public:
    ResampleImageTest() {
        if (!testCases.empty())
            return;

        std::mt19937 gen(0);

        // Run the same inputs on CPU and CUDA and store both warped images plus the mask.
        auto addCase = [&](const std::string& name, NiftiImage& reference, NiftiImage& floating,
                           NiftiImage field, std::vector<int> mask, float pad) {
            NiftiImage warpedCpu = warpImage(PlatformType::Cpu, reference, floating, field, mask, pad);
            NiftiImage warpedCuda = warpImage(PlatformType::Cuda, reference, floating, field, mask, pad);
            testCases.push_back({ name, std::move(warpedCpu), std::move(warpedCuda), std::move(mask) });
        };

        const float nan = std::numeric_limits<float>::quiet_NaN();
        auto padName = [](float pad) -> std::string {
            if (std::isnan(pad)) return "NaN";
            if (pad == 0.f) return "0";
            if (pad == -1.f) return "-1";
            if (pad == 100.f) return "100";
            return std::to_string(pad);
        };

        for (bool is3D : { false, true }) {
            const std::string dimTag = is3D ? "3D " : "2D ";
            const std::vector<NiftiImage::dim_t> dims = is3D ? std::vector<NiftiImage::dim_t>{ 6, 7, 5 }
                                                             : std::vector<NiftiImage::dim_t>{ 7, 8 };

            // --- Group A: core matrix. perturbed field x pad{0,-1,NaN} x mask{none,partial}, tp=1 ---
            {
                NiftiImage floating = makeImage(dims);
                NiftiImage reference(floating);
                const size_t nVox = reference.nVoxelsPerVolume();
                std::vector<int> noMask(nVox, 0);
                std::vector<int> partMask(nVox, 0);
                for (size_t i = 0; i < nVox; ++i)
                    if (i % 3 == 0) partMask[i] = -1;
                for (float pad : { 0.f, -1.f, nan }) {
                    addCase(dimTag + "core pad=" + padName(pad) + " nomask", reference, floating,
                            buildField(reference, kPerturbed, gen), noMask, pad);
                    addCase(dimTag + "core pad=" + padName(pad) + " masked", reference, floating,
                            buildField(reference, kPerturbed, gen), partMask, pad);
                }
            }

            // --- Group B: multiple time points / channels (no mask, tp=3)
            {
                std::vector<NiftiImage::dim_t> floDims = dims; // ref==flo spatial dims, 3 time points
                floDims.resize(4, 1);
                floDims[3] = 3;
                NiftiImage floating = makeImage(floDims);
                NiftiImage reference = makeImage(dims);
                std::vector<int> noMask(reference.nVoxelsPerVolume(), 0);
                for (float pad : { 0.f, nan })
                    addCase(dimTag + "multi-tp matched pad=" + padName(pad), reference, floating,
                            buildField(reference, kPerturbed, gen), noMask, pad);
            }
            {
                // ref != flo spatial dims + 3 time points -> warpedVoxelNumber != floVoxelNumber, so
                // this diverges / goes out of bounds unless the warped stride uses warpedVoxelNumber.
                NiftiImage floating = is3D ? makeImage({ 7, 8, 6, 3 }) : makeImage({ 7, 8, 1, 3 });
                NiftiImage reference = is3D ? makeImage({ 5, 6, 4 }) : makeImage({ 5, 6 });
                std::vector<int> noMask(reference.nVoxelsPerVolume(), 0);
                for (float pad : { 0.f, nan })
                    addCase(dimTag + "multi-tp mismatched pad=" + padName(pad), reference, floating,
                            buildField(reference, kPerturbed, gen), noMask, pad);
            }

            // --- Group C: diagnostics (no mask, tp=1) ---
            {
                NiftiImage floating = makeImage(dims);
                NiftiImage reference(floating);
                std::vector<int> noMask(reference.nVoxelsPerVolume(), 0);
                addCase(dimTag + "identity pad=0", reference, floating,
                        buildField(reference, kIdentity, gen), noMask, 0.f);
                addCase(dimTag + "int-shift pad=0", reference, floating,
                        buildField(reference, kIntShift, gen), noMask, 0.f);
                addCase(dimTag + "fully-out pad=100", reference, floating,
                        buildField(reference, kFullyOut, gen), noMask, 100.f);
                addCase(dimTag + "fully-out pad=NaN", reference, floating,
                        buildField(reference, kFullyOut, gen), noMask, nan);
            }

            // --- Group D: grid mismatch. reference != floating spatial dims (identity sform),
            //     perturbed field, no mask, tp=1. Exercises resampling between differently-sized
            //     grids and the axis indexing that the ref==floating groups (A/C) do not.
            //     NOTE: a non-identity (sheared / arbitrary-scale) floating sform is deliberately
            //     NOT used - it is not bit-exact CPU<->CUDA and so cannot be asserted with ==. The
            //     CPU stores the world->voxel coordinate in float (_reg_resampling.cpp, float
            //     position[3]) while CUDA keeps it in double (TransformInterpolate<double>,
            //     CudaResampling.cu:25)
            {
                NiftiImage floating = is3D ? makeImage({ 7, 8, 6 }) : makeImage({ 7, 8 });
                NiftiImage reference = is3D ? makeImage({ 5, 6, 4 }) : makeImage({ 5, 6 });
                std::vector<int> noMask(reference.nVoxelsPerVolume(), 0);
                for (float pad : { 0.f, nan })
                    addCase(dimTag + "grid-mismatch pad=" + padName(pad), reference, floating,
                            buildField(reference, kPerturbed, gen), noMask, pad);
            }
        }
    }
};

TEST_CASE_METHOD(ResampleImageTest, "Regression Resample Image", "[regression]") {
    for (auto&& testCase : testCases) {
        auto&& [sectionName, warpedCpu, warpedCuda, mask] = testCase;

        SECTION(sectionName) {
            NR_COUT << "\n**************** Section " << sectionName << " ****************" << std::endl;
            NR_COUT << std::fixed << std::setprecision(10);

            REQUIRE(warpedCpu.nVoxels() == warpedCuda.nVoxels());
            const auto cpuPtr = warpedCpu.data();
            const auto cudaPtr = warpedCuda.data();
            const size_t total = warpedCpu.nVoxels();
            const size_t spatial = mask.size(); // reference voxels per volume; maps every timepoint

            for (size_t i = 0; i < total; ++i) {
                if (mask[i % spatial] == -1) // masked-out: undefined per-platform, skip
                    continue;
                const float cpuVal = cpuPtr[i];
                const float cudaVal = cudaPtr[i];
                const bool cpuNan = std::isnan(cpuVal);
                const bool cudaNan = std::isnan(cudaVal);
                if (cpuNan || cudaNan) {
                    if (cpuNan != cudaNan)
                        NR_COUT << "[i]=" << i << " | NaN mismatch | CPU=" << cpuVal << " | CUDA=" << cudaVal << std::endl;
                    REQUIRE(cpuNan == cudaNan);
                } else {
                    const float diff = std::abs(cpuVal - cudaVal);
                    if (diff > 0) {
                        NR_COUT << "[i]=" << i;
                        NR_COUT << " | diff=" << diff;
                        NR_COUT << " | CPU=" << cpuVal;
                        NR_COUT << " | CUDA=" << cudaVal << std::endl;
                    }
                    REQUIRE(diff == 0);
                }
            }
        }
    }
}
