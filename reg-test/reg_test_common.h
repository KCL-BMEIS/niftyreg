#define NR_TESTING  // Enable testing
#define EPS     0.000001f

#include <array>
#include <random>
#include <iomanip>
#include <numeric>
#include <cmath>
#include <catch2/catch_test_macros.hpp>
#include "_reg_lncc.h"
#include "_reg_localTrans.h"
#include "_reg_nmi.h"
#include "AffineDeformationFieldKernel.h"
#include "Platform.h"
#include "ResampleImageKernel.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef SINC_KERNEL_RADIUS
#define SINC_KERNEL_RADIUS 3
#endif
#ifndef SINC_KERNEL_SIZE
#define SINC_KERNEL_SIZE (SINC_KERNEL_RADIUS * 2)
#endif

template<typename T>
void InterpCubicSplineKernel(T relative, T (&basis)[4]) {
    if (relative < 0) relative = 0; //reg_rounding error
    const T relative2 = relative * relative;
    basis[0] = (relative * ((2.f - relative) * relative - 1.f)) / 2.f;
    basis[1] = (relative2 * (3.f * relative - 5.f) + 2.f) / 2.f;
    basis[2] = (relative * ((4.f - 3.f * relative) * relative + 1.f)) / 2.f;
    basis[3] = (relative - 1.f) * relative2 / 2.f;
}

template<typename T>
void InterpCubicSplineKernel(T relative, T (&basis)[4], T (&derivative)[4]) {
    InterpCubicSplineKernel(relative, basis);
    if (relative < 0) relative = 0; //reg_rounding error
    const T relative2 = relative * relative;
    derivative[0] = (4.f * relative - 3.f * relative2 - 1.f) / 2.f;
    derivative[1] = (9.f * relative - 10.f) * relative / 2.f;
    derivative[2] = (8.f * relative - 9.f * relative2 + 1.f) / 2.f;
    derivative[3] = (3.f * relative - 2.f) * relative / 2.f;
}


// reg_f3d's default control point spacing (-sx -5), i.e. the geometry almost every real run uses.
// Worth preferring over the 2-voxel default below when a test has no reason to need something else:
// a 5-voxel spacing is what selects the look-up-table branch of the 3D cubic spline evaluation,
// so tests built on 2-voxel grids never reach it.
constexpr float kProductionGridSpacing = 5.f;

// The spacing is in voxels. Prefer kProductionGridSpacing, or sweep several values, when the choice
// is arbitrary; the default of 2 is only a default.
NiftiImage CreateControlPointGrid(const NiftiImage& reference, const float spacingInVoxels = 2.f) {
    const float gridSpacing[3] = { reference->dx * spacingInVoxels,
                                   reference->dy * spacingInVoxels,
                                   reference->dz * spacingInVoxels };

    // Create and allocate the control point image
    // It is initialised with an identity transformation by default
    NiftiImage controlPointGrid;
    reg_createControlPointGrid<float>(controlPointGrid, reference, gridSpacing);

    return controlPointGrid;
}

NiftiImage CreateDeformationField(const NiftiImage& reference) {
    // Create and allocate a deformation field
    // It is initialised with an identity transformation by default
    NiftiImage deformationField;
    reg_createDeformationField<float>(deformationField, reference);

    return deformationField;
}

// Install an identity sform (world coordinates == voxel coordinates) on an image.
void setIdentitySform(NiftiImage& img) {
    mat44 eye;
    Mat44Eye(&eye);
    img->sform_code = 1;
    img->sto_xyz = eye;
    img->sto_ijk = eye;
    img->qform_code = 0;
}

// Install an arbitrary sform (sto_xyz) on an image, deriving sto_ijk as its inverse.
void setSform(NiftiImage& img, const mat44& m) {
    img->sform_code = 1;
    img->sto_xyz = m;
    img->sto_ijk = nifti_mat44_inverse(m);
    img->qform_code = 0;
}

// Install an anisotropic sform with a shifted origin, i.e. what a real acquisition and every
// downsampled pyramid level look like. An identity sform puts every interpolation weight on an exact
// binary fraction, where float and double arithmetic cannot disagree and the order of a sum does not
// matter, so a test built only on one cannot see a precision or association difference at all. Worth
// pairing with the identity case rather than replacing it.
void setAnisotropicSform(NiftiImage& img) {
    mat44 m;
    Mat44Eye(&m);
    m.m[0][0] = 1.4f; m.m[1][1] = 0.8f; m.m[2][2] = img->nz > 1 ? 1.9f : 1.f;
    m.m[0][3] = -6.5f; m.m[1][3] = 3.25f; m.m[2][3] = img->nz > 1 ? -2.75f : 0.f;
    setSform(img, m);
    img->dx = img->pixdim[1] = m.m[0][0];
    img->dy = img->pixdim[2] = m.m[1][1];
    img->dz = img->pixdim[3] = img->nz > 1 ? m.m[2][2] : 1.f;
}

// Vacuity guards. A comparison only means something if the operation being compared actually ran -
// an output left at its initial value compares equal to anything else left at its initial value, and
// the test passes without exercising the code it names. Assert that the output was computed first.
//
// This bites most often where the amount of work is driven by a parameter that defaults to nothing:
// a zero iteration count, an empty mask, a weight of zero. Prefer setting such a parameter explicitly
// in the test over relying on whatever the default happens to be.
void RequireNonZero(const NiftiImage& img, const std::string& what) {
    const auto ptr = img.data();
    bool nonZero = false;
    for (size_t i = 0; i < img.nVoxels() && !nonZero; ++i)
        nonZero = static_cast<float>(ptr[i]) != 0.f;
    INFO(what << " is entirely zero, so the operation that should have filled it did not run");
    REQUIRE(nonZero);
}

void RequireChanged(const NiftiImage& before, const NiftiImage& after, const std::string& what) {
    REQUIRE(before.nVoxels() == after.nVoxels());
    const auto beforePtr = before.data();
    const auto afterPtr = after.data();
    bool changed = false;
    for (size_t i = 0; i < after.nVoxels() && !changed; ++i)
        changed = static_cast<float>(afterPtr[i]) != static_cast<float>(beforePtr[i]);
    INFO(what << " is unchanged, so the operation under test was a no-op");
    REQUIRE(changed);
}

// Read an image out of a content for comparison. Always go through the virtual getter: a device-backed
// content overrides these to transfer its data back to the host first, so a Content::-qualified call
// would yield whatever the host buffer last held and compare that instead.
template<class ContentType>
NiftiImage& DeformationFieldForComparison(ContentType& con) { return con.GetDeformationField(); }
template<class ContentType>
NiftiImage& WarpedForComparison(ContentType& con) { return con.GetWarped(); }
template<class ContentType>
NiftiImage& VoxelBasedMeasureGradientForComparison(ContentType& con) { return con.GetVoxelBasedMeasureGradient(); }
template<class ContentType>
NiftiImage& TransformationGradientForComparison(ContentType& con) { return con.GetTransformationGradient(); }
template<class ContentType>
NiftiImage& ControlPointGridForComparison(ContentType& con) { return con.GetControlPointGrid(); }

// How far apart two images are, and over how many voxels. Reported rather than only asserted, so a
// test that gates on a property can still show the magnitude behind it.
struct Deviation {
    double max = 0;
    size_t differing = 0;
};

Deviation CompareImages(const NiftiImage& a, const NiftiImage& b) {
    Deviation deviation;
    const auto aPtr = a.data();
    const auto bPtr = b.data();
    REQUIRE(a.nVoxels() == b.nVoxels());
    for (size_t i = 0; i < a.nVoxels(); ++i) {
        const double difference = std::abs(static_cast<double>(aPtr[i]) - static_cast<double>(bPtr[i]));
        if (difference > 0) ++deviation.differing;
        deviation.max = std::max(deviation.max, difference);
    }
    return deviation;
}

void ReportDeviation(const std::string& what, const Deviation& deviation) {
    NR_COUT << "  " << std::setw(52) << std::left << what
            << " max = " << std::scientific << std::setprecision(3) << deviation.max
            << " (" << deviation.differing << " differing)" << std::endl;
}

// The two helpers below express the reproduction property that a cubic B-spline satisfies: with the
// control points holding a function's values at the nodes, the spline evaluates to that function. For
// an affine it holds exactly, which makes it an oracle that does not depend on how the repo evaluates
// splines - unlike comparing against a second copy of the same algorithm.

// Replace every control point's stored position p by affine(p), so the grid parametrises the affine
void ApplyAffineToGrid(NiftiImage& grid, const mat44& affine) {
    const size_t volume = grid.nVoxelsPerVolume();
    const int components = grid->nz > 1 ? 3 : 2;
    auto ptr = grid.data();
    for (size_t i = 0; i < volume; ++i) {
        float position[3]{ 0, 0, 0 }, transformed[3];
        for (int c = 0; c < components; ++c)
            position[c] = static_cast<float>(ptr[c * volume + i]);
        Mat44Mul(affine, position, transformed);
        for (int c = 0; c < components; ++c)
            ptr[c * volume + i] = transformed[c];
    }
}

// The deformation field an affine produces, evaluated directly per voxel
NiftiImage ExpectedAffineField(const NiftiImage& reference, const mat44& affine) {
    NiftiImage field = CreateDeformationField(reference);
    const size_t volume = field.nVoxelsPerVolume();
    const int nx = field->nx, ny = field->ny, nz = field->nz;
    const int components = nz > 1 ? 3 : 2;
    const mat44 voxelToReal = reference->sform_code > 0 ? reference->sto_xyz : reference->qto_xyz;
    auto ptr = field.data();
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                float voxel[3]{ float(i), float(j), float(k) }, world[3], transformed[3];
                Mat44Mul(voxelToReal, voxel, world);
                Mat44Mul(affine, world, transformed);
                const size_t index = (static_cast<size_t>(k) * ny + j) * nx + i;
                for (int c = 0; c < components; ++c)
                    ptr[c * volume + index] = transformed[c];
            }
    return field;
}

// A float32 image with identity sform, filled with distinct fractional values (unique per voxel
// and per volume, so neighbouring voxels and multi-timepoint volumes are all distinguishable).
NiftiImage makeImage(const std::vector<NiftiImage::dim_t>& dims) {
    NiftiImage img(dims, NIFTI_TYPE_FLOAT32);
    setIdentitySform(img);
    auto ptr = img.data();
    const size_t n = img.nVoxels();
    for (size_t i = 0; i < n; ++i)
        ptr[i] = static_cast<float>(i) + 0.5f;
    return img;
}

