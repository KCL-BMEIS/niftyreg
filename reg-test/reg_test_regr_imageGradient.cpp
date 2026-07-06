// OpenCL is not supported for this test
#undef USE_OPENCL

#include "reg_test_common.h"

/*
    CPU<->CUDA bit-exact regression for Compute::GetImageGradient

    For timepoint t the floating volume is fetched at floOffset = t * floVoxelNumber from a single
    whole-buffer texture.

    Coverage (all bit-exact, diff == 0, NaN-aware):
      - 2D and 3D,
      - activeTimePoint 0 and 1 on a 2-timepoint floating with matched ref/flo grids,
      - activeTimePoint 2 on a 3-timepoint floating with ref != flo spatial dims (floVoxelNumber !=
        reference voxel number), which goes out of bounds / selects the wrong volume unless the
        fetch offset uses floVoxelNumber,
      - a sheared floating sform (the world->voxel coordinate path)
      - paddingValue 0, a nonzero finite value (-1), and NaN. The deformation field is perturbed
        enough to push taps out of FOV, so the padding path is exercised.

*/

// Compute the image gradient of `floating` at `activeTimePoint` on the given platform with the given
// padding value, and return it on the host (GetWarpedGradient() downloads it on CUDA). Linear interp.
static NiftiImage imageGradient(PlatformType platformType, const NiftiImage& reference,
                                const NiftiImage& floating, NiftiImage defField,
                                int activeTimePoint, float paddingValue) {
    Platform platform(platformType);
    unique_ptr<DefContentCreator> creator{ dynamic_cast<DefContentCreator*>(platform.CreateContentCreator(ContentType::Def)) };
    NiftiImage ref(reference), flo(floating);
    unique_ptr<DefContent> content{ creator->Create(ref, flo) };
    content->SetDeformationField(std::move(defField));
    unique_ptr<Compute> compute{ platform.CreateCompute(*content) };
    compute->GetImageGradient(1, paddingValue, activeTimePoint); // linear interpolation
    NiftiImage warpedGradient = std::move(content->GetWarpedGradient());
    return warpedGradient;
}

// Perturbed-identity deformation field. The +-1.5 spread pushes some taps out of FOV so the padding
// path is exercised, while interior taps still sample real floating data.
static NiftiImage perturbedField(const NiftiImage& reference, std::mt19937& gen) {
    NiftiImage field = CreateDeformationField(reference); // identity, world coordinates
    std::uniform_real_distribution<float> d(-1.5f, 1.5f);
    auto p = field.data();
    for (size_t i = 0; i < field.nVoxels(); ++i)
        p[i] = static_cast<float>(p[i]) + d(gen);
    return field;
}

class ImageGradientRegrTest {
protected:
    using TestCase = std::tuple<std::string, NiftiImage, NiftiImage>;
    inline static vector<TestCase> testCases;

public:
    ImageGradientRegrTest() {
        if (!testCases.empty())
            return;

        std::mt19937 gen(0);
        const float nan = std::numeric_limits<float>::quiet_NaN();
        auto padName = [](float pad) -> std::string {
            if (std::isnan(pad)) return "NaN";
            if (pad == 0.f) return "0";
            if (pad == -1.f) return "-1";
            return std::to_string(pad);
        };

        // Run the same inputs on CPU and CUDA and store both gradient images.
        auto addCase = [&](const std::string& name, NiftiImage& reference, NiftiImage& floating,
                           NiftiImage field, int activeTimePoint, float pad) {
            NiftiImage gradCpu = imageGradient(PlatformType::Cpu, reference, floating, field, activeTimePoint, pad);
            NiftiImage gradCuda = imageGradient(PlatformType::Cuda, reference, floating, field, activeTimePoint, pad);
            testCases.push_back({ name, std::move(gradCpu), std::move(gradCuda) });
        };

        for (bool is3D : { false, true }) {
            const std::string dimTag = is3D ? "3D " : "2D ";

            // --- Matched grids, 2-timepoint floating, activeTimePoint {0,1}.
            // makeImage values are index+0.5 so volume 1 differs from volume 0 -> activeTimePoint>=1
            // must select the right volume (the multi-timepoint texture-offset path). ---
            {
                NiftiImage floating = is3D ? makeImage({ 6, 7, 5, 2 }) : makeImage({ 7, 8, 1, 2 });
                NiftiImage reference = is3D ? makeImage({ 6, 7, 5 }) : makeImage({ 7, 8 });
                for (int activeTimePoint : { 0, 1 })
                    for (float pad : { 0.f, -1.f, nan })
                        addCase(dimTag + "activeTimePoint=" + std::to_string(activeTimePoint) + " pad=" + padName(pad),
                                reference, floating, perturbedField(reference, gen), activeTimePoint, pad);
            }

            // --- Mismatched ref/flo spatial dims + 3 time points, activeTimePoint=2.
            // floVoxelNumber != reference voxel number, so the timepoint fetch offset must use
            // floVoxelNumber, and the
            // multiplier is >1. ---
            {
                NiftiImage floating = is3D ? makeImage({ 7, 8, 6, 3 }) : makeImage({ 7, 8, 1, 3 });
                NiftiImage reference = is3D ? makeImage({ 5, 6, 4 }) : makeImage({ 5, 6 });
                for (float pad : { 0.f, nan })
                    addCase(dimTag + "mismatched-dims activeTimePoint=2 pad=" + padName(pad),
                            reference, floating, perturbedField(reference, gen), 2, pad);
            }

            // --- Sheared floating sform, activeTimePoint=1.
            // Exercises the world->voxel coordinate path shared with the resampler ---
            {
                NiftiImage floating = is3D ? makeImage({ 7, 8, 6, 2 }) : makeImage({ 7, 8, 1, 2 });
                mat44 m;
                Mat44Eye(&m);
                m.m[0][0] = 1.2f; m.m[1][1] = 0.9f;
                m.m[0][1] = 0.1f; m.m[1][0] = -0.06f;   // in-plane shear
                m.m[0][3] = 0.5f; m.m[1][3] = 0.3f;     // in-plane translation
                if (is3D) { m.m[2][2] = 1.1f; m.m[0][2] = 0.05f; m.m[1][2] = -0.07f; m.m[2][3] = 0.2f; }
                setSform(floating, m);
                NiftiImage reference = is3D ? makeImage({ 5, 6, 4 }) : makeImage({ 5, 6 });
                for (float pad : { 0.f, -1.f, nan })
                    addCase(dimTag + "sheared-sform activeTimePoint=1 pad=" + padName(pad),
                            reference, floating, perturbedField(reference, gen), 1, pad);
            }
        }
    }
};

TEST_CASE_METHOD(ImageGradientRegrTest, "Regression Image Gradient", "[regression]") {
    for (auto&& testCase : testCases) {
        auto&& [sectionName, gradCpu, gradCuda] = testCase;

        SECTION(sectionName) {
            NR_COUT << "\n**************** Section " << sectionName << " ****************" << std::endl;
            NR_COUT << std::fixed << std::setprecision(10);

            REQUIRE(gradCpu.nVoxels() == gradCuda.nVoxels());
            const auto cpuPtr = gradCpu.data();
            const auto cudaPtr = gradCuda.data();
            for (size_t i = 0; i < gradCpu.nVoxels(); ++i) {
                const float cpuVal = cpuPtr[i];
                const float cudaVal = cudaPtr[i];
                const bool cpuNan = std::isnan(cpuVal);
                const bool cudaNan = std::isnan(cudaVal);
                if (cpuNan || cudaNan) {
                    if (cpuNan != cudaNan)
                        NR_COUT << "[i]=" << i << " | NaN mismatch | CPU=" << cpuVal << " | CUDA=" << cudaVal << std::endl;
                    REQUIRE(cpuNan == cudaNan);
                } else {
                    const float diff = abs(cpuVal - cudaVal);
                    if (diff > 0)
                        NR_COUT << "[i]=" << i << " | diff=" << diff << " | CPU=" << cpuVal << " | CUDA=" << cudaVal << std::endl;
                    REQUIRE(diff == 0);
                }
            }
        }
    }
}
