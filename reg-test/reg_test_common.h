#define NR_TESTING  // Enable testing
#define EPS     0.000001f

#include <array>
#include <random>
#include <iomanip>
#include <numeric>
#include <catch2/catch_test_macros.hpp>
#include "_reg_lncc.h"
#include "_reg_localTrans.h"
#include "_reg_nmi.h"
#include "AffineDeformationFieldKernel.h"
#include "Platform.h"
#include "ResampleImageKernel.h"


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
