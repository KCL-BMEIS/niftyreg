#define NR_TESTING  // Enable testing
#define EPS     0.000001

#include <array>
#include <random>
#include <catch2/catch_test_macros.hpp>
#include "_reg_lncc.h"
#include "_reg_localTrans.h"
#include "Platform.h"
#include "ResampleImageKernel.h"
#include "AffineDeformationFieldKernel.h"


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
    float gridSpacing[3] = { reference->dx * 2, reference->dy * 2, reference->dz * 2 };

    // Create and allocate the control point image
    NiftiImage controlPointGrid;
    reg_createControlPointGrid<float>(controlPointGrid, reference, gridSpacing);

    return controlPointGrid;
}

NiftiImage CreateDeformationField(const NiftiImage& reference) {
    // Create and allocate a deformation field
    NiftiImage deformationField(reference, NiftiImage::Copy::ImageInfo);
    deformationField.setDim(NiftiDim::NDim, 5);
    if (reference->dim[0] == 2)
        deformationField.setDim(NiftiDim::Z, 1);
    deformationField.setDim(NiftiDim::T, 1);
    deformationField.setPixDim(NiftiDim::T, 1);
    deformationField.setDim(NiftiDim::U, reference->nz > 1 ? 3 : 2);
    deformationField.setPixDim(NiftiDim::U, 1);
    deformationField.setDim(NiftiDim::V, 1);
    deformationField.setPixDim(NiftiDim::V, 1);
    deformationField.setDim(NiftiDim::W, 1);
    deformationField.setPixDim(NiftiDim::W, 1);
    deformationField->datatype = NIFTI_TYPE_FLOAT32;
    deformationField->intent_code = NIFTI_INTENT_VECTOR;
    memset(deformationField->intent_name, 0, sizeof(deformationField->intent_name));
    strcpy(deformationField->intent_name, "NREG_TRANS");
    deformationField->intent_p1 = DISP_FIELD;
    deformationField->scl_slope = 1;
    deformationField->scl_inter = 0;
    deformationField.realloc();
    reg_getDeformationFromDisplacement(deformationField);
    return deformationField;
}
