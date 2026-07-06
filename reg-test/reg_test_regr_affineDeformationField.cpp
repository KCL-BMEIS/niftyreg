// OpenCL is not supported for this test
#undef USE_OPENCL

#include "reg_test_common.h"

/*
    CPU<->CUDA bit-exact regression for Compute::GetAffineDeformationField
    The affine field build is deterministic per-voxel, so the two platforms are required to be
    bit-exact (== 0).
*/

// Build the deformation field from an affine matrix on the given platform (compose=false, the
// reg_resample case) and return it on the host (GetDeformationField() downloads it on CUDA).
static NiftiImage affineDeformationField(PlatformType platformType, const NiftiImage& reference, mat44& affine) {
    Platform platform(platformType);
    unique_ptr<ContentCreator> creator{ platform.CreateContentCreator(ContentType::Base) };
    NiftiImage ref(reference), flo(reference); // floating is unused by the affine field build
    unique_ptr<Content> content{ creator->Create(ref, flo, nullptr, &affine) };
    unique_ptr<Compute> compute{ platform.CreateCompute(*content) };
    compute->GetAffineDeformationField(false);
    NiftiImage defField = std::move(content->GetDeformationField());
    return defField;
}

class AffineDeformationFieldRegrTest {
protected:
    using TestCase = std::tuple<std::string, NiftiImage, NiftiImage>;
    inline static vector<TestCase> testCases;

public:
    AffineDeformationFieldRegrTest() {
        if (!testCases.empty())
            return;

        // A general affine: anisotropic scale + shear + translation.
        mat44 affine;
        Mat44Eye(&affine);
        affine.m[0][0] = 1.1f; affine.m[1][1] = 0.95f; affine.m[2][2] = 1.05f;
        affine.m[0][1] = 0.08f; affine.m[0][2] = -0.04f; affine.m[1][2] = 0.06f;
        affine.m[0][3] = 1.5f; affine.m[1][3] = -0.8f; affine.m[2][3] = 0.3f;

        // A non-identity (anisotropic + sheared) reference sform.
        mat44 sheared;
        Mat44Eye(&sheared);
        sheared.m[0][0] = 1.2f; sheared.m[1][1] = 0.9f; sheared.m[2][2] = 1.1f;
        sheared.m[0][1] = 0.1f; sheared.m[0][2] = 0.05f; sheared.m[1][2] = -0.07f;
        sheared.m[0][3] = 0.5f; sheared.m[1][3] = 0.3f; sheared.m[2][3] = 0.2f;

        for (bool is3D : { false, true }) {
            const std::string dimTag = is3D ? "3D " : "2D ";
            for (int shearedSform = 0; shearedSform < 2; shearedSform++) {
                NiftiImage reference = is3D ? makeImage({ 6, 7, 5 }) : makeImage({ 7, 8 });
                if (shearedSform) setSform(reference, sheared); // makeImage installs an identity sform otherwise
                NiftiImage defFieldCpu = affineDeformationField(PlatformType::Cpu, reference, affine);
                NiftiImage defFieldCuda = affineDeformationField(PlatformType::Cuda, reference, affine);
                testCases.push_back({ dimTag + (shearedSform ? "sheared sform" : "identity sform"),
                                      std::move(defFieldCpu), std::move(defFieldCuda) });
            }
        }
    }
};

TEST_CASE_METHOD(AffineDeformationFieldRegrTest, "Regression Affine Deformation Field", "[regression]") {
    for (auto&& testCase : testCases) {
        auto&& [sectionName, defFieldCpu, defFieldCuda] = testCase;

        SECTION(sectionName) {
            NR_COUT << "\n**************** Section " << sectionName << " ****************" << std::endl;
            NR_COUT << std::fixed << std::setprecision(10);

            REQUIRE(defFieldCpu.nVoxels() == defFieldCuda.nVoxels());
            const auto cpuPtr = defFieldCpu.data();
            const auto cudaPtr = defFieldCuda.data();
            for (size_t i = 0; i < defFieldCpu.nVoxels(); ++i) {
                const float cpuVal = cpuPtr[i];
                const float cudaVal = cudaPtr[i];
                const float diff = abs(cpuVal - cudaVal);
                if (diff > 0)
                    NR_COUT << "[i]=" << i << " | diff=" << diff << " | CPU=" << cpuVal << " | CUDA=" << cudaVal << std::endl;
                REQUIRE(diff == 0);
            }
        }
    }
}
