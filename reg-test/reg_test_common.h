// Enable testing
#define NR_TESTING

#include "Platform.h"
#include "ResampleImageKernel.h"
#include "_reg_localTrans.h"

#include <list>
#include <catch2/catch_test_macros.hpp>


template <typename T>
void interpCubicSplineKernel(T relative, T (&basis)[4]) {
    if (relative < 0) relative = 0; //reg_rounding error
    const T relative2 = relative * relative;
    basis[0] = (relative * ((2.f - relative) * relative - 1.f)) / 2.f;
    basis[1] = (relative2 * (3.f * relative - 5.f) + 2.f) / 2.f;
    basis[2] = (relative * ((4.f - 3.f * relative) * relative + 1.f)) / 2.f;
    basis[3] = (relative - 1.f) * relative2 / 2.f;
}

