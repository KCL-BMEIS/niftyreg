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


NiftiImage CreateControlPointGrid(const NiftiImage& reference) {
    // Set the spacing for the control point grid to 2 voxel along each axis
    const float gridSpacing[3] = { reference->dx * 2, reference->dy * 2, reference->dz * 2 };

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

