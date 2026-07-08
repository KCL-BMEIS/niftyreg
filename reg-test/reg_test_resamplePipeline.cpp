// OpenCL is not supported for this test
#undef USE_OPENCL

#include "reg_test_common.h"
#include "_reg_resampling.h"
#include "_reg_globalTrans.h"
#include "_reg_tools.h"

/*
    CPU pipeline equivalence: direct reg-lib calls vs the Platform/Content/Compute abstraction.

    For every transform type reg_resample supports (affine / cubic-spline grid / displacement field
    / spline velocity grid, in 2D and 3D) this test builds the deformation field and warps the
    floating image two ways, and asserts the warped images are bit-identical:
      - Path A: the direct reg-lib functions (reg_affine_getDeformationField,
        reg_spline_getDeformationField, reg_getDeformationFromDisplacement + reg_defField_compose,
        reg_spline_getDefFieldFromVelocityGrid, then reg_resampleImage).
      - Path B: the Platform -> Content -> Compute abstraction (GetAffineDeformationField,
        GetDeformationField, GetDefFieldFromVelocityGrid on a Base/F3d content, then ResampleImage).
    Both paths ultimately call the same reg-lib functions on identical inputs, so the result is
    deterministic and compared with ==.

    These two code paths coexist permanently (Compute is a thin wrapper over the reg-lib functions),
    so this stays a valid CPU equivalence/regression check no matter which path a caller uses. That
    is also what makes wiring reg_resample onto Content/Compute safe: the two pipelines are proven to
    agree per transform type. The test is library-level and intentionally does not cover
    reg_resample's CLI parsing or file I/O.

    Note: a content's deformation field is identity-initialised and reg_resample builds a spline
    grid's field with composition=true, so Path A initialises its field to identity and uses
    composition=true to match Compute::GetDeformationField(true, true).
*/

constexpr int kInterp = 3;   // reg_resample default (cubic spline)
constexpr float kPad = 0.f;  // reg_resample default padding

// A float32 image with identity sform and distinct values (so warping is non-trivial).
static NiftiImage makeImg(const std::vector<NiftiImage::dim_t>& dims) {
    NiftiImage img(dims, NIFTI_TYPE_FLOAT32);
    mat44 eye;
    Mat44Eye(&eye);
    img->sform_code = 1;
    img->sto_xyz = eye;
    img->sto_ijk = eye;
    img->qform_code = 0;
    auto p = img.data();
    for (size_t i = 0; i < img.nVoxels(); ++i)
        p[i] = std::sin(0.3f * static_cast<float>(i)) + 1.5f;
    return img;
}

// Add a smooth, deterministic, bounded perturbation to a field/grid (so the transform is non-identity).
static void perturb(NiftiImage& img, float scale) {
    auto p = img.data();
    for (size_t i = 0; i < img.nVoxels(); ++i)
        p[i] = static_cast<float>(p[i]) + scale * std::sin(0.7f * static_cast<float>(i) + 0.3f);
}

// Warp directly with reg_resampleImage, allocating the warped image exactly as Content::AllocateWarped.
static NiftiImage warpDirect(NiftiImage& reference, NiftiImage& floating, NiftiImage& def) {
    NiftiImage warped(reference, NiftiImage::Copy::ImageInfo);
    warped.setDim(NiftiDim::NDim, floating->ndim);
    warped.setDim(NiftiDim::T, floating->nt);
    warped.setPixDim(NiftiDim::T, 1);
    warped->datatype = floating->datatype;
    warped->nbyper = floating->nbyper;
    warped.realloc();
    reg_resampleImage(floating, warped, def, nullptr, kInterp, kPad);
    return warped;
}

static void requireSameWarped(const NiftiImage& a, const NiftiImage& b) {
    REQUIRE(a.nVoxels() == b.nVoxels());
    auto ap = a.data();
    auto bp = b.data();
    for (size_t i = 0; i < a.nVoxels(); ++i) {
        const float av = static_cast<float>(ap[i]);
        const float bv = static_cast<float>(bp[i]);
        if (std::isnan(av) || std::isnan(bv))
            REQUIRE(std::isnan(av) == std::isnan(bv));
        else
            REQUIRE(av == bv);
    }
}

TEST_CASE("Resample pipeline CPU equivalence: direct vs Compute", "[unit]") {
    constexpr NiftiImage::dim_t S = 8;

    SECTION("Affine") {
        for (bool is3D : { false, true }) {
            INFO((is3D ? "3D" : "2D"));
            NiftiImage image = is3D ? makeImg({ S, S, S }) : makeImg({ S, S });
            mat44 aff;
            Mat44Eye(&aff);
            aff.m[0][0] = 0.98f; aff.m[0][1] = -0.17f; aff.m[1][0] = 0.17f; aff.m[1][1] = 0.98f;
            aff.m[0][3] = 1.5f; aff.m[1][3] = -0.8f; if (is3D) aff.m[2][3] = 0.3f;

            // Path A: direct reg-lib construction from the affine matrix, then reg_resampleImage
            NiftiImage defA = CreateDeformationField(image);
            reg_affine_getDeformationField(&aff, defA, false, nullptr);
            NiftiImage warpedA = warpDirect(image, image, defA);

            // Path B: Compute::GetAffineDeformationField + ResampleImage on a Base content
            Platform platform(PlatformType::Cpu);
            unique_ptr<ContentCreator> creator{ platform.CreateContentCreator() };
            NiftiImage ref(image), flo(image);
            unique_ptr<Content> content{ creator->Create(ref, flo, nullptr, &aff) };
            unique_ptr<Compute> compute{ platform.CreateCompute(*content) };
            compute->GetAffineDeformationField(false);
            compute->ResampleImage(kInterp, kPad);
            NiftiImage warpedB = std::move(content->GetWarped());

            requireSameWarped(warpedA, warpedB);
        }
    }

    SECTION("Cubic spline grid") {
        for (bool is3D : { false, true }) {
            INFO((is3D ? "3D" : "2D"));
            NiftiImage image = is3D ? makeImg({ S, S, S }) : makeImg({ S, S });
            NiftiImage cpg = CreateControlPointGrid(image);
            perturb(cpg, 0.8f);

            // Path A: direct construction with composition=true (as reg_resample builds a spline grid)
            NiftiImage defA = CreateDeformationField(image); // identity
            reg_spline_getDeformationField(cpg, defA, nullptr, true, true);
            NiftiImage warpedA = warpDirect(image, image, defA);

            // Path B: Compute::GetDeformationField + ResampleImage on an F3d content
            Platform platform(PlatformType::Cpu);
            unique_ptr<F3dContentCreator> creator{ dynamic_cast<F3dContentCreator*>(platform.CreateContentCreator(ContentType::F3d)) };
            NiftiImage ref(image), flo(image), cpgB(cpg);
            unique_ptr<F3dContent> content{ creator->Create(ref, flo, cpgB) };
            unique_ptr<Compute> compute{ platform.CreateCompute(*content) };
            compute->GetDeformationField(true, true);
            compute->ResampleImage(kInterp, kPad);
            NiftiImage warpedB = std::move(content->GetWarped());

            requireSameWarped(warpedA, warpedB);
        }
    }

    SECTION("Displacement field") {
        for (bool is3D : { false, true }) {
            INFO((is3D ? "3D" : "2D"));
            NiftiImage image = is3D ? makeImg({ S, S, S }) : makeImg({ S, S });

            // Build a displacement-field input transform
            NiftiImage disp = CreateDeformationField(image); // identity deformation
            perturb(disp, 0.5f);                              // -> non-identity deformation
            reg_getDisplacementFromDeformation(disp);         // -> displacement field
            disp->intent_p1 = DISP_FIELD;

            // Path A: direct construction (displacement -> deformation, then compose)
            NiftiImage defA = CreateDeformationField(image); // identity
            NiftiImage dispA(disp);
            reg_getDeformationFromDisplacement(dispA);
            reg_defField_compose(dispA, defA, nullptr);
            NiftiImage warpedA = warpDirect(image, image, defA);

            // Path B: there is no Compute entry point for displacement -> deformation, so reuse the
            // directly-built field and route only the resample through Compute.
            Platform platform(PlatformType::Cpu);
            unique_ptr<ContentCreator> creator{ platform.CreateContentCreator() };
            NiftiImage ref(image), flo(image);
            unique_ptr<Content> content{ creator->Create(ref, flo) };
            content->SetDeformationField(NiftiImage(defA));
            unique_ptr<Compute> compute{ platform.CreateCompute(*content) };
            compute->ResampleImage(kInterp, kPad);
            NiftiImage warpedB = std::move(content->GetWarped());

            requireSameWarped(warpedA, warpedB);
        }
    }

    SECTION("Spline velocity grid") {
        for (bool is3D : { false, true }) {
            INFO((is3D ? "3D" : "2D"));
            NiftiImage image = is3D ? makeImg({ S, S, S }) : makeImg({ S, S });
            NiftiImage vel = CreateControlPointGrid(image);
            vel->intent_p1 = SPLINE_VEL_GRID;
            vel->intent_p2 = 6; // number of squaring steps
            perturb(vel, 0.1f); // small stationary velocity -> stable exponentiation

            // Path A: direct construction (spline velocity grid -> deformation via scaling-and-squaring)
            NiftiImage defA = CreateDeformationField(image); // identity
            reg_spline_getDefFieldFromVelocityGrid(vel, defA, false);
            NiftiImage warpedA = warpDirect(image, image, defA);

            // Path B: Compute::GetDefFieldFromVelocityGrid + ResampleImage on an F3d content
            Platform platform(PlatformType::Cpu);
            unique_ptr<F3dContentCreator> creator{ dynamic_cast<F3dContentCreator*>(platform.CreateContentCreator(ContentType::F3d)) };
            NiftiImage ref(image), flo(image), velB(vel);
            unique_ptr<F3dContent> content{ creator->Create(ref, flo, velB) };
            unique_ptr<Compute> compute{ platform.CreateCompute(*content) };
            compute->GetDefFieldFromVelocityGrid(false);
            compute->ResampleImage(kInterp, kPad);
            NiftiImage warpedB = std::move(content->GetWarped());

            requireSameWarped(warpedA, warpedB);
        }
    }
}
