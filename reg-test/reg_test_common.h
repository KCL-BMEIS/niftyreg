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

inline void InterpNearestNeighKernel(double relative, double *basis) {
    if (relative < 0) relative = 0; // reg_rounding error
    basis[0] = basis[1] = 0;
    if (relative >= 0.5)
        basis[1] = 1;
    else basis[0] = 1;
}

inline void InterpLinearKernel(double relative, double *basis) {
    if (relative < 0) relative = 0; // reg_rounding error
    basis[1] = relative;
    basis[0] = 1.0 - relative;
}

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


inline void InterpWindowedSincKernel(double relative, double *basis) {
    if (relative < 0) relative = 0; // reg_rounding error
    int j = 0;
    double sum = 0.;
    for (int i = -SINC_KERNEL_RADIUS; i < SINC_KERNEL_RADIUS; ++i) {
        double x = relative - static_cast<double>(i);
        if (x == 0)
            basis[j] = 1.0;
        else if (fabs(x) >= static_cast<double>(SINC_KERNEL_RADIUS))
            basis[j] = 0;
        else {
            double pi_x = M_PI * x;
            basis[j] = static_cast<double>(SINC_KERNEL_RADIUS) *
                sin(pi_x) *
                sin(pi_x / static_cast<double>(SINC_KERNEL_RADIUS)) /
                (pi_x * pi_x);
        }
        sum += basis[j];
        j++;
    }
    for (int i = 0; i < SINC_KERNEL_SIZE; ++i)
        basis[i] /= sum;
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
