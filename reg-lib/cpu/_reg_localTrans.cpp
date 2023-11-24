/*
 *  _reg_localTrans.cpp
 *
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_localTrans.h"
#include "_reg_maths_eigen.h"

// Due to SSE usage creates incorrect test results
#if defined(BUILD_TESTS) && !defined(NDEBUG)
#undef USE_SSE
#endif

/* *************************************************************** */
template <class DataType>
void reg_createControlPointGrid(NiftiImage& controlPointGridImage,
                                const NiftiImage& referenceImage,
                                const float *spacing) {
    // Define the control point grid dimensions
    vector<NiftiImage::dim_t> dims{
        Ceil(referenceImage->nx * referenceImage->dx / spacing[0] + 3.f),
        Ceil(referenceImage->ny * referenceImage->dy / spacing[1] + 3.f),
        referenceImage->nz > 1 ? Ceil(referenceImage->nz * referenceImage->dz / spacing[2] + 3.f) : 1,
        1,
        referenceImage->nz > 1 ? 3 : 2
    };

    // Create the new control point grid image and allocate its space
    controlPointGridImage = NiftiImage(dims, sizeof(DataType) == sizeof(float) ? NIFTI_TYPE_FLOAT32 : NIFTI_TYPE_FLOAT64);

    // Fill the header information
    controlPointGridImage->cal_min = 0;
    controlPointGridImage->cal_max = 0;
    controlPointGridImage->pixdim[0] = 1.0f;
    controlPointGridImage.setPixDim(NiftiDim::X, spacing[0]);
    controlPointGridImage.setPixDim(NiftiDim::Y, spacing[1]);
    controlPointGridImage.setPixDim(NiftiDim::Z, referenceImage->nz > 1 ? spacing[2] : 1.0f);
    controlPointGridImage.setPixDim(NiftiDim::T, 1.0f);
    controlPointGridImage.setPixDim(NiftiDim::U, 1.0f);
    controlPointGridImage.setPixDim(NiftiDim::V, 1.0f);
    controlPointGridImage.setPixDim(NiftiDim::W, 1.0f);

    // Reproduce the orientation of the reference image and add a one voxel shift
    if (referenceImage->qform_code + referenceImage->sform_code > 0) {
        controlPointGridImage->qform_code = referenceImage->qform_code;
        controlPointGridImage->sform_code = referenceImage->sform_code;
    } else {
        controlPointGridImage->qform_code = 1;
        controlPointGridImage->sform_code = 0;
    }

    // The qform (and sform) are set for the control point position image
    controlPointGridImage->quatern_b = referenceImage->quatern_b;
    controlPointGridImage->quatern_c = referenceImage->quatern_c;
    controlPointGridImage->quatern_d = referenceImage->quatern_d;
    controlPointGridImage->qoffset_x = referenceImage->qoffset_x;
    controlPointGridImage->qoffset_y = referenceImage->qoffset_y;
    controlPointGridImage->qoffset_z = referenceImage->qoffset_z;
    controlPointGridImage->qfac = referenceImage->qfac;
    controlPointGridImage->qto_xyz = nifti_quatern_to_mat44(controlPointGridImage->quatern_b,
                                                            controlPointGridImage->quatern_c,
                                                            controlPointGridImage->quatern_d,
                                                            controlPointGridImage->qoffset_x,
                                                            controlPointGridImage->qoffset_y,
                                                            controlPointGridImage->qoffset_z,
                                                            controlPointGridImage->dx,
                                                            controlPointGridImage->dy,
                                                            controlPointGridImage->dz,
                                                            controlPointGridImage->qfac);

    // Origin is shifted from 1 control point in the qform
    float originIndex[3];
    float originReal[3];
    originIndex[0] = -1.0f;
    originIndex[1] = -1.0f;
    originIndex[2] = 0.0f;
    if (referenceImage->nz > 1) originIndex[2] = -1.0f;
    reg_mat44_mul(&controlPointGridImage->qto_xyz, originIndex, originReal);
    controlPointGridImage->qto_xyz.m[0][3] = controlPointGridImage->qoffset_x = originReal[0];
    controlPointGridImage->qto_xyz.m[1][3] = controlPointGridImage->qoffset_y = originReal[1];
    controlPointGridImage->qto_xyz.m[2][3] = controlPointGridImage->qoffset_z = originReal[2];

    controlPointGridImage->qto_ijk = nifti_mat44_inverse(controlPointGridImage->qto_xyz);

    // Update the sform if required
    if (controlPointGridImage->sform_code > 0) {
        float scalingRatio[3];
        scalingRatio[0] = controlPointGridImage->dx / referenceImage->dx;
        scalingRatio[1] = controlPointGridImage->dy / referenceImage->dy;
        scalingRatio[2] = controlPointGridImage->dz / referenceImage->dz;

        controlPointGridImage->sto_xyz.m[0][0] = referenceImage->sto_xyz.m[0][0] * scalingRatio[0];
        controlPointGridImage->sto_xyz.m[1][0] = referenceImage->sto_xyz.m[1][0] * scalingRatio[0];
        controlPointGridImage->sto_xyz.m[2][0] = referenceImage->sto_xyz.m[2][0] * scalingRatio[0];
        controlPointGridImage->sto_xyz.m[3][0] = referenceImage->sto_xyz.m[3][0];
        controlPointGridImage->sto_xyz.m[0][1] = referenceImage->sto_xyz.m[0][1] * scalingRatio[1];
        controlPointGridImage->sto_xyz.m[1][1] = referenceImage->sto_xyz.m[1][1] * scalingRatio[1];
        controlPointGridImage->sto_xyz.m[2][1] = referenceImage->sto_xyz.m[2][1] * scalingRatio[1];
        controlPointGridImage->sto_xyz.m[3][1] = referenceImage->sto_xyz.m[3][1];
        controlPointGridImage->sto_xyz.m[0][2] = referenceImage->sto_xyz.m[0][2] * scalingRatio[2];
        controlPointGridImage->sto_xyz.m[1][2] = referenceImage->sto_xyz.m[1][2] * scalingRatio[2];
        controlPointGridImage->sto_xyz.m[2][2] = referenceImage->sto_xyz.m[2][2] * scalingRatio[2];
        controlPointGridImage->sto_xyz.m[3][2] = referenceImage->sto_xyz.m[3][2];
        controlPointGridImage->sto_xyz.m[0][3] = referenceImage->sto_xyz.m[0][3];
        controlPointGridImage->sto_xyz.m[1][3] = referenceImage->sto_xyz.m[1][3];
        controlPointGridImage->sto_xyz.m[2][3] = referenceImage->sto_xyz.m[2][3];
        controlPointGridImage->sto_xyz.m[3][3] = referenceImage->sto_xyz.m[3][3];

        // Origin is shifted from 1 control point in the sform
        reg_mat44_mul(&controlPointGridImage->sto_xyz, originIndex, originReal);
        controlPointGridImage->sto_xyz.m[0][3] = originReal[0];
        controlPointGridImage->sto_xyz.m[1][3] = originReal[1];
        controlPointGridImage->sto_xyz.m[2][3] = originReal[2];
        controlPointGridImage->sto_ijk = nifti_mat44_inverse(controlPointGridImage->sto_xyz);
    }

    // The grid is initialised with an identity transformation
    reg_getDeformationFromDisplacement(controlPointGridImage);
    controlPointGridImage->intent_code = NIFTI_INTENT_VECTOR;
    controlPointGridImage.setIntentName("NREG_TRANS"s);
    controlPointGridImage->intent_p1 = CUB_SPLINE_GRID;
}
template void reg_createControlPointGrid<float>(NiftiImage&, const NiftiImage&, const float*);
template void reg_createControlPointGrid<double>(NiftiImage&, const NiftiImage&, const float*);
/* *************************************************************** */
template <class DataType>
void reg_createSymmetricControlPointGrids(NiftiImage& forwardGridImage,
                                          NiftiImage& backwardGridImage,
                                          const NiftiImage& referenceImage,
                                          const NiftiImage& floatingImage,
                                          const mat44 *forwardAffineTrans,
                                          const float *spacing) {
    // We specified a space which is in-between both input images
    // Get the reference image space
    mat44 referenceImageSpace = referenceImage->qto_xyz;
    if (referenceImage->sform_code > 0)
        referenceImageSpace = referenceImage->sto_xyz;
    NR_MAT44_DEBUG(referenceImageSpace, "Input reference image orientation");
    // // Get the floating image space
    mat44 floatingImageSpace = floatingImage->qto_xyz;
    if (floatingImage->sform_code > 0)
        floatingImageSpace = floatingImage->sto_xyz;
    NR_MAT44_DEBUG(floatingImageSpace, "Input floating image orientation");
    // Check if an affine transformation is specified
    mat44 halfForwardAffine, halfBackwardAffine;
    if (forwardAffineTrans != nullptr) {
        // Compute half of the affine transformation - ref to flo
        halfForwardAffine = reg_mat44_logm(forwardAffineTrans);
        halfForwardAffine = reg_mat44_mul(&halfForwardAffine, .5f);
        halfForwardAffine = reg_mat44_expm(&halfForwardAffine);
        // Compute half of the affine transformation - flo to ref
        // Note that this is done twice for symmetry consideration
        halfBackwardAffine = nifti_mat44_inverse(*forwardAffineTrans);
        halfBackwardAffine = reg_mat44_logm(&halfBackwardAffine);
        halfBackwardAffine = reg_mat44_mul(&halfBackwardAffine, .5f);
        halfBackwardAffine = reg_mat44_expm(&halfBackwardAffine);
        NR_WARN("Note that the symmetry of the registration is affected by the input affine transformation");
    } else {
        reg_mat44_eye(&halfForwardAffine);
        reg_mat44_eye(&halfBackwardAffine);
    }

    // Update the reference and floating transformation to propagate to a mid space
    referenceImageSpace = reg_mat44_mul(&halfForwardAffine, &referenceImageSpace);
    floatingImageSpace = reg_mat44_mul(&halfBackwardAffine, &floatingImageSpace);

    // Define the largest field of view in the mid space
    float minPosition[3] = { 0, 0, 0 }, maxPosition[3] = { 0, 0, 0 };
    if (referenceImage->nz > 1)  // 3D
    {
        float referenceImageCorners[8][3] = {
            { 0, 0, 0 },
            { float(referenceImage->nx), 0, 0 },
            { 0, float(referenceImage->ny), 0 },
            { float(referenceImage->nx), float(referenceImage->ny), 0 },
            { 0, 0, float(referenceImage->nz) },
            { float(referenceImage->nx), 0, float(referenceImage->nz) },
            { 0, float(referenceImage->ny), float(referenceImage->nz) },
            { float(referenceImage->nx), float(referenceImage->ny), float(referenceImage->nz) }
        };
        float floatingImageCorners[8][3] = {
            { 0, 0, 0 },
            { float(floatingImage->nx), 0, 0 },
            { 0, float(floatingImage->ny), 0 },
            { float(floatingImage->nx), float(floatingImage->ny), 0 },
            { 0, 0, float(floatingImage->nz) },
            { float(floatingImage->nx), 0, float(floatingImage->nz) },
            { 0, float(floatingImage->ny), float(floatingImage->nz) },
            { float(floatingImage->nx), float(floatingImage->ny), float(floatingImage->nz) }
        };
        float out[3];
        for (int c = 0; c < 8; ++c) {
            reg_mat44_mul(&referenceImageSpace, referenceImageCorners[c], out);
            referenceImageCorners[c][0] = out[0];
            referenceImageCorners[c][1] = out[1];
            referenceImageCorners[c][2] = out[2];
            reg_mat44_mul(&floatingImageSpace, floatingImageCorners[c], out);
            floatingImageCorners[c][0] = out[0];
            floatingImageCorners[c][1] = out[1];
            floatingImageCorners[c][2] = out[2];

        }
        minPosition[0] = std::min(referenceImageCorners[0][0], floatingImageCorners[0][0]);
        minPosition[1] = std::min(referenceImageCorners[0][1], floatingImageCorners[0][1]);
        minPosition[2] = std::min(referenceImageCorners[0][2], floatingImageCorners[0][2]);
        maxPosition[0] = std::max(referenceImageCorners[0][0], floatingImageCorners[0][0]);
        maxPosition[1] = std::max(referenceImageCorners[0][1], floatingImageCorners[0][1]);
        maxPosition[2] = std::max(referenceImageCorners[0][2], floatingImageCorners[0][2]);
        for (int c = 1; c < 8; ++c) {
            minPosition[0] = std::min(minPosition[0], referenceImageCorners[c][0]);
            minPosition[0] = std::min(minPosition[0], floatingImageCorners[c][0]);
            minPosition[1] = std::min(minPosition[1], referenceImageCorners[c][1]);
            minPosition[1] = std::min(minPosition[1], floatingImageCorners[c][1]);
            minPosition[2] = std::min(minPosition[2], referenceImageCorners[c][2]);
            minPosition[2] = std::min(minPosition[2], floatingImageCorners[c][2]);
            maxPosition[0] = std::max(maxPosition[0], referenceImageCorners[c][0]);
            maxPosition[0] = std::max(maxPosition[0], floatingImageCorners[c][0]);
            maxPosition[1] = std::max(maxPosition[1], referenceImageCorners[c][1]);
            maxPosition[1] = std::max(maxPosition[1], floatingImageCorners[c][1]);
            maxPosition[2] = std::max(maxPosition[2], referenceImageCorners[c][2]);
            maxPosition[2] = std::max(maxPosition[2], floatingImageCorners[c][2]);
        }
    } else { // 2D
        float referenceImageCorners[4][2] = {
            { 0, 0 },
            { float(referenceImage->nx), 0 },
            { 0, float(referenceImage->ny) },
            { float(referenceImage->nx), float(referenceImage->ny) }
        };
        float floatingImageCorners[4][2] = {
            { 0, 0 },
            { float(floatingImage->nx), 0 },
            { 0, float(floatingImage->ny) },
            { float(floatingImage->nx), float(floatingImage->ny) }
        };
        float out[2];
        for (int c = 0; c < 4; ++c) {
            out[0] = referenceImageCorners[c][0] * referenceImageSpace.m[0][0]
                + referenceImageCorners[c][1] * referenceImageSpace.m[0][1]
                + referenceImageSpace.m[0][3];
            out[1] = referenceImageCorners[c][0] * referenceImageSpace.m[1][0]
                + referenceImageCorners[c][1] * referenceImageSpace.m[1][1]
                + referenceImageSpace.m[1][3];
            referenceImageCorners[c][0] = out[0];
            referenceImageCorners[c][1] = out[1];
            out[0] = floatingImageCorners[c][0] * floatingImageSpace.m[0][0]
                + floatingImageCorners[c][1] * floatingImageSpace.m[0][1]
                + floatingImageSpace.m[0][3];
            out[1] = floatingImageCorners[c][0] * floatingImageSpace.m[1][0]
                + floatingImageCorners[c][1] * floatingImageSpace.m[1][1]
                + floatingImageSpace.m[1][3];
            floatingImageCorners[c][0] = out[0];
            floatingImageCorners[c][1] = out[1];

        }
        minPosition[0] = std::min(referenceImageCorners[0][0], floatingImageCorners[0][0]);
        minPosition[1] = std::min(referenceImageCorners[0][1], floatingImageCorners[0][1]);
        maxPosition[0] = std::max(referenceImageCorners[0][0], floatingImageCorners[0][0]);
        maxPosition[1] = std::max(referenceImageCorners[0][1], floatingImageCorners[0][1]);
        for (int c = 1; c < 4; ++c) {
            minPosition[0] = std::min(minPosition[0], referenceImageCorners[c][0]);
            minPosition[0] = std::min(minPosition[0], floatingImageCorners[c][0]);
            minPosition[1] = std::min(minPosition[1], referenceImageCorners[c][1]);
            minPosition[1] = std::min(minPosition[1], floatingImageCorners[c][1]);
            maxPosition[0] = std::max(maxPosition[0], referenceImageCorners[c][0]);
            maxPosition[0] = std::max(maxPosition[0], floatingImageCorners[c][0]);
            maxPosition[1] = std::max(maxPosition[1], referenceImageCorners[c][1]);
            maxPosition[1] = std::max(maxPosition[1], floatingImageCorners[c][1]);
        }
    }

    // Compute the dimension of the control point grids
    const vector<NiftiImage::dim_t> dims{
        Ceil((maxPosition[0] - minPosition[0]) / spacing[0] + 3.f),
        Ceil((maxPosition[1] - minPosition[1]) / spacing[1] + 3.f),
        referenceImage->nz > 1 ? Ceil((maxPosition[2] - minPosition[2]) / spacing[2] + 3.f) : 1,
        1,
        referenceImage->nz > 1 ? 3 : 2
    };

    // Create the control point grid image
    forwardGridImage = NiftiImage(dims, sizeof(DataType) == sizeof(float) ? NIFTI_TYPE_FLOAT32 : NIFTI_TYPE_FLOAT64);
    backwardGridImage = NiftiImage(dims, sizeof(DataType) == sizeof(float) ? NIFTI_TYPE_FLOAT32 : NIFTI_TYPE_FLOAT64);

    // Set the control point grid spacing
    forwardGridImage.setPixDim(NiftiDim::X, spacing[0]);
    backwardGridImage.setPixDim(NiftiDim::X, spacing[0]);
    forwardGridImage.setPixDim(NiftiDim::Y, spacing[1]);
    backwardGridImage.setPixDim(NiftiDim::Y, spacing[1]);
    forwardGridImage.setPixDim(NiftiDim::Z, referenceImage->nz > 1 ? spacing[2] : 1.0f);
    backwardGridImage.setPixDim(NiftiDim::Z, referenceImage->nz > 1 ? spacing[2] : 1.0f);
    // Set the control point grid image orientation
    forwardGridImage->qform_code = backwardGridImage->qform_code = 0;
    forwardGridImage->sform_code = backwardGridImage->sform_code = 1;
    reg_mat44_eye(&forwardGridImage->sto_xyz);
    reg_mat44_eye(&backwardGridImage->sto_xyz);
    reg_mat44_eye(&forwardGridImage->sto_ijk);
    reg_mat44_eye(&backwardGridImage->sto_ijk);
    for (unsigned i = 0; i < 3; ++i) {
        if (referenceImage->nz > 1 || i < 2) {
            forwardGridImage->sto_xyz.m[i][i] = backwardGridImage->sto_xyz.m[i][i] = spacing[i];
            forwardGridImage->sto_xyz.m[i][3] = backwardGridImage->sto_xyz.m[i][3] = minPosition[i] - spacing[i];
        } else {
            forwardGridImage->sto_xyz.m[i][i] = backwardGridImage->sto_xyz.m[i][i] = 1.f;
            forwardGridImage->sto_xyz.m[i][3] = backwardGridImage->sto_xyz.m[i][3] = 0.f;
        }
    }
    forwardGridImage->sto_ijk = backwardGridImage->sto_ijk = nifti_mat44_inverse(forwardGridImage->sto_xyz);
    // Set the intent type
    forwardGridImage->intent_code = backwardGridImage->intent_code = NIFTI_INTENT_VECTOR;
    forwardGridImage.setIntentName("NREG_TRANS"s);
    backwardGridImage.setIntentName("NREG_TRANS"s);
    forwardGridImage->intent_p1 = backwardGridImage->intent_p1 = CUB_SPLINE_GRID;
    // Set the affine matrices
    mat44 identity;
    reg_mat44_eye(&identity);
    if (forwardGridImage->ext_list != nullptr)
        free(forwardGridImage->ext_list);
    if (backwardGridImage->ext_list != nullptr)
        free(backwardGridImage->ext_list);
    forwardGridImage->num_ext = 0;
    backwardGridImage->num_ext = 0;
    if (identity != halfForwardAffine && identity != halfBackwardAffine) {
        // Create extensions to store the affine parametrisations for the forward transformation
        forwardGridImage->num_ext = 2;
        forwardGridImage->ext_list = (nifti1_extension*)malloc(2 * sizeof(nifti1_extension));
        forwardGridImage->ext_list[0].esize = 16 * sizeof(float) + 16;
        forwardGridImage->ext_list[1].esize = 16 * sizeof(float) + 16;
        forwardGridImage->ext_list[0].ecode = NIFTI_ECODE_IGNORE;
        forwardGridImage->ext_list[1].ecode = NIFTI_ECODE_IGNORE;
        forwardGridImage->ext_list[0].edata = (char*)calloc(forwardGridImage->ext_list[0].esize - 8, sizeof(float));
        forwardGridImage->ext_list[1].edata = (char*)calloc(forwardGridImage->ext_list[1].esize - 8, sizeof(float));
        memcpy(forwardGridImage->ext_list[0].edata, &halfForwardAffine, sizeof(mat44));
        memcpy(forwardGridImage->ext_list[1].edata, &halfForwardAffine, sizeof(mat44));
        NR_MAT44_DEBUG(halfForwardAffine, "Forward transformation half-affine");
        // Create extensions to store the affine parametrisations for the backward transformation
        backwardGridImage->num_ext = 2;
        backwardGridImage->ext_list = (nifti1_extension*)malloc(2 * sizeof(nifti1_extension));
        backwardGridImage->ext_list[0].esize = 16 * sizeof(float) + 16;
        backwardGridImage->ext_list[1].esize = 16 * sizeof(float) + 16;
        backwardGridImage->ext_list[0].ecode = NIFTI_ECODE_IGNORE;
        backwardGridImage->ext_list[1].ecode = NIFTI_ECODE_IGNORE;
        backwardGridImage->ext_list[0].edata = (char*)calloc(backwardGridImage->ext_list[0].esize - 8, sizeof(float));
        backwardGridImage->ext_list[1].edata = (char*)calloc(backwardGridImage->ext_list[1].esize - 8, sizeof(float));
        memcpy(backwardGridImage->ext_list[0].edata, &halfBackwardAffine, sizeof(mat44));
        memcpy(backwardGridImage->ext_list[1].edata, &halfBackwardAffine, sizeof(mat44));
        NR_MAT44_DEBUG(halfBackwardAffine, "Backward transformation half-affine");
    }
    // Convert the parametrisations into deformation fields
    reg_getDeformationFromDisplacement(forwardGridImage);
    reg_getDeformationFromDisplacement(backwardGridImage);
}
template void reg_createSymmetricControlPointGrids<float>(NiftiImage&, NiftiImage&, const NiftiImage&, const NiftiImage&, const mat44*, const float*);
template void reg_createSymmetricControlPointGrids<double>(NiftiImage&, NiftiImage&, const NiftiImage&, const NiftiImage&, const mat44*, const float*);
/* *************************************************************** */
template <class DataType>
void reg_createDeformationField(NiftiImage& deformationFieldImage,
                                const nifti_image *referenceImage) {
    // The header information from the reference image are copied over
    deformationFieldImage = NiftiImage(const_cast<nifti_image*>(referenceImage), NiftiImage::Copy::ImageInfo);
    // The dimension are updated to store the deformation vector along U index
    // in a 5D image
    deformationFieldImage.setDim(NiftiDim::NDim, 5);
    if (referenceImage->dim[0] == 2)
        deformationFieldImage.setDim(NiftiDim::Z, 1);
    deformationFieldImage.setDim(NiftiDim::T, 1);
    deformationFieldImage.setPixDim(NiftiDim::T, 1);
    deformationFieldImage.setDim(NiftiDim::U, referenceImage->nz > 1 ? 3 : 2);
    deformationFieldImage.setPixDim(NiftiDim::U, 1);
    deformationFieldImage.setDim(NiftiDim::V, 1);
    deformationFieldImage.setPixDim(NiftiDim::V, 1);
    deformationFieldImage.setDim(NiftiDim::W, 1);
    deformationFieldImage.setPixDim(NiftiDim::W, 1);
    // The deformation stores floating scalar
    deformationFieldImage->datatype = sizeof(DataType) == sizeof(float) ? NIFTI_TYPE_FLOAT32 : NIFTI_TYPE_FLOAT64;
    deformationFieldImage->nbyper = sizeof(DataType);
    deformationFieldImage->intent_code = NIFTI_INTENT_VECTOR;
    memset(deformationFieldImage->intent_name, 0, sizeof(deformationFieldImage->intent_name));
    strcpy(deformationFieldImage->intent_name, "NREG_TRANS");
    deformationFieldImage->scl_slope = 1;
    deformationFieldImage->scl_inter = 0;

    // The data is allocated given the new size and filled in with zero to represent an identity displacement field
    deformationFieldImage.realloc();
    deformationFieldImage->intent_p1 = DISP_FIELD;
    // The displacement field is converted into a deformation field
    reg_getDeformationFromDisplacement(deformationFieldImage);
}
template void reg_createDeformationField<float>(NiftiImage&, const nifti_image*);
template void reg_createDeformationField<double>(NiftiImage&, const nifti_image*);
/* *************************************************************** */
template<class DataType>
void reg_linear_spline_getDeformationField3D(nifti_image *splineControlPoint,
                                             nifti_image *deformationField,
                                             int *mask,
                                             bool composition) {
    int coord;

    const size_t splineControlPointVoxelNumber = NiftiImage::calcVoxelNumber(splineControlPoint, 3);
    DataType *controlPointPtrX = static_cast<DataType*>(splineControlPoint->data);
    DataType *controlPointPtrY = &controlPointPtrX[splineControlPointVoxelNumber];
    DataType *controlPointPtrZ = &controlPointPtrY[splineControlPointVoxelNumber];

    const size_t deformationFieldVoxelNumber = NiftiImage::calcVoxelNumber(deformationField, 3);
    DataType *fieldPtrX = static_cast<DataType*>(deformationField->data);
    DataType *fieldPtrY = &fieldPtrX[deformationFieldVoxelNumber];
    DataType *fieldPtrZ = &fieldPtrY[deformationFieldVoxelNumber];

    int x, y, z, a, b, c, xPre, yPre, zPre, index;
    DataType xBasis[2], yBasis[2], zBasis[2], real[3];

    if (composition) { // Composition of deformation fields
        // read the ijk sform or qform, as appropriate
        mat44 referenceMatrix_real_to_voxel;
        if (splineControlPoint->sform_code > 0)
            referenceMatrix_real_to_voxel = splineControlPoint->sto_ijk;
        else referenceMatrix_real_to_voxel = splineControlPoint->qto_ijk;

        DataType voxel[3];

        for (z = 0; z < deformationField->nz; z++) {
            index = z * deformationField->nx * deformationField->ny;
            for (y = 0; y < deformationField->ny; y++) {
                for (x = 0; x < deformationField->nx; x++) {
                    if (mask[index] > -1) {
                        // The previous position at the current pixel position is read
                        real[0] = fieldPtrX[index];
                        real[1] = fieldPtrY[index];
                        real[2] = fieldPtrZ[index];

                        // From real to pixel position in the control point space
                        voxel[0] =
                            referenceMatrix_real_to_voxel.m[0][0] * real[0] +
                            referenceMatrix_real_to_voxel.m[0][1] * real[1] +
                            referenceMatrix_real_to_voxel.m[0][2] * real[2] +
                            referenceMatrix_real_to_voxel.m[0][3];
                        voxel[1] =
                            referenceMatrix_real_to_voxel.m[1][0] * real[0] +
                            referenceMatrix_real_to_voxel.m[1][1] * real[1] +
                            referenceMatrix_real_to_voxel.m[1][2] * real[2] +
                            referenceMatrix_real_to_voxel.m[1][3];
                        voxel[2] =
                            referenceMatrix_real_to_voxel.m[2][0] * real[0] +
                            referenceMatrix_real_to_voxel.m[2][1] * real[1] +
                            referenceMatrix_real_to_voxel.m[2][2] * real[2] +
                            referenceMatrix_real_to_voxel.m[2][3];

                        // The spline coefficients are computed
                        xPre = Floor(voxel[0]);
                        xBasis[1] = voxel[0] - static_cast<DataType>(xPre);
                        if (xBasis[1] < 0) xBasis[1] = 0; //rounding error
                        xBasis[0] = 1.f - xBasis[1];

                        yPre = Floor(voxel[1]);
                        yBasis[1] = voxel[1] - static_cast<DataType>(yPre);
                        if (yBasis[1] < 0) yBasis[1] = 0; //rounding error
                        yBasis[0] = 1.f - yBasis[1];

                        zPre = Floor(voxel[2]);
                        zBasis[1] = voxel[2] - static_cast<DataType>(zPre);
                        if (zBasis[1] < 0) zBasis[1] = 0; //rounding error
                        zBasis[0] = 1.f - zBasis[1];

                        real[0] = 0;
                        real[1] = 0;
                        real[2] = 0;
                        for (c = 0; c < 2; c++) {
                            for (b = 0; b < 2; b++) {
                                for (a = 0; a < 2; a++) {
                                    DataType tempValue = xBasis[a] * yBasis[b] * zBasis[c];
                                    coord = ((zPre + c) * splineControlPoint->ny + yPre + b) * splineControlPoint->nx + xPre + a;
                                    real[0] += controlPointPtrX[coord] * tempValue;
                                    real[1] += controlPointPtrY[coord] * tempValue;
                                    real[2] += controlPointPtrZ[coord] * tempValue;
                                }
                            }
                        }
                        fieldPtrX[index] = real[0];
                        fieldPtrY[index] = real[1];
                        fieldPtrZ[index] = real[2];
                    } // mask
                    index++;
                }
            }
        }
    } else {  // !composition
        DataType gridVoxelSpacing[3];
        gridVoxelSpacing[0] = splineControlPoint->dx / deformationField->dx;
        gridVoxelSpacing[1] = splineControlPoint->dy / deformationField->dy;
        gridVoxelSpacing[2] = splineControlPoint->dz / deformationField->dz;
        DataType tempValue;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
   private(x, y, a, b, c, xPre, yPre, zPre, xBasis, yBasis, zBasis, real, index, coord, tempValue) \
   shared(deformationField, gridVoxelSpacing, mask, fieldPtrX, fieldPtrY, fieldPtrZ, \
   controlPointPtrX, controlPointPtrY, controlPointPtrZ, splineControlPoint)
#endif // _OPENMP
        for (z = 0; z < deformationField->nz; z++) {
            index = z * deformationField->nx * deformationField->ny;

            zPre = static_cast<int>(static_cast<DataType>(z) / gridVoxelSpacing[2]);
            zBasis[1] = static_cast<DataType>(z) / gridVoxelSpacing[2] - static_cast<DataType>(zPre);
            if (zBasis[1] < 0) zBasis[1] = 0; //rounding error
            zBasis[0] = 1.f - zBasis[1];
            zPre++;

            for (y = 0; y < deformationField->ny; y++) {
                yPre = static_cast<int>(static_cast<DataType>(y) / gridVoxelSpacing[1]);
                yBasis[1] = static_cast<DataType>(y) / gridVoxelSpacing[1] - static_cast<DataType>(yPre);
                if (yBasis[1] < 0) yBasis[1] = 0; //rounding error
                yBasis[0] = 1.f - yBasis[1];
                yPre++;

                for (x = 0; x < deformationField->nx; x++) {
                    real[0] = 0;
                    real[1] = 0;
                    real[2] = 0;

                    if (mask[index] > -1) {
                        xPre = static_cast<int>(static_cast<DataType>(x) / gridVoxelSpacing[0]);
                        xBasis[1] = static_cast<DataType>(x) / gridVoxelSpacing[0] - static_cast<DataType>(xPre);
                        if (xBasis[1] < 0) xBasis[1] = 0; //rounding error
                        xBasis[0] = 1.f - xBasis[1];
                        xPre++;
                        real[0] = 0;
                        real[1] = 0;
                        real[2] = 0;
                        for (c = 0; c < 2; c++) {
                            for (b = 0; b < 2; b++) {
                                for (a = 0; a < 2; a++) {
                                    tempValue = xBasis[a] * yBasis[b] * zBasis[c];
                                    coord = ((zPre + c) * splineControlPoint->ny + yPre + b) * splineControlPoint->nx + xPre + a;
                                    real[0] += controlPointPtrX[coord] * tempValue;
                                    real[1] += controlPointPtrY[coord] * tempValue;
                                    real[2] += controlPointPtrZ[coord] * tempValue;
                                }
                            }
                        }
                    }// mask
                    fieldPtrX[index] = real[0];
                    fieldPtrY[index] = real[1];
                    fieldPtrZ[index] = real[2];
                    index++;
                } // x
            } // y
        } // z
    }// from a deformation field
}
/* *************************************************************** */
template<class DataType>
void reg_cubic_spline_getDeformationField2D(nifti_image *splineControlPoint,
                                            nifti_image *deformationField,
                                            int *mask,
                                            bool composition,
                                            bool bspline) {
#if USE_SSE
    union {
        __m128 m;
        float f[4];
    } val;
    __m128 tempCurrent, tempX, tempY;
#ifdef _WIN32
    __declspec(align(16)) DataType xBasis[4];
    __declspec(align(16)) DataType yBasis[4];
    union {
        __m128 m[16];
        __declspec(align(16)) DataType f[16];
    } xControlPointCoordinates;
    union {
        __m128 m[16];
        __declspec(align(16)) DataType f[16];
    } yControlPointCoordinates;
    union u1 {
        __m128 m[4];
        __declspec(align(16)) DataType f[16];
    } xyBasis;
#else // _WIN32
    DataType xBasis[4] __attribute__((aligned(16)));
    DataType yBasis[4] __attribute__((aligned(16)));
    union {
        __m128 m[16];
        DataType f[16] __attribute__((aligned(16)));
    } xControlPointCoordinates;
    union {
        __m128 m[16];
        DataType f[16] __attribute__((aligned(16)));
    } yControlPointCoordinates;
    union u1 {
        __m128 m[4];
        DataType f[16] __attribute__((aligned(16)));
    } xyBasis;
#endif // _WIN32
#else // USE_SSE
    DataType xBasis[4];
    DataType yBasis[4];
    DataType xyBasis[16];
    DataType xControlPointCoordinates[16];
    DataType yControlPointCoordinates[16];
#endif // USE_SSE

    DataType *controlPointPtrX = static_cast<DataType*>(splineControlPoint->data);
    DataType *controlPointPtrY = &controlPointPtrX[NiftiImage::calcVoxelNumber(splineControlPoint, 2)];

    DataType *fieldPtrX = static_cast<DataType*>(deformationField->data);
    DataType *fieldPtrY = &fieldPtrX[NiftiImage::calcVoxelNumber(deformationField, 2)];

    DataType gridVoxelSpacing[2];
    gridVoxelSpacing[0] = splineControlPoint->dx / deformationField->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / deformationField->dy;

    DataType basis, xReal, yReal, xVoxel, yVoxel;
    int x, y, a, b, xPre, yPre, oldXpre, oldYpre;
    size_t index, coord;

    if (composition) { // Composition of deformation fields
        // read the ijk sform or qform, as appropriate
        const mat44 *referenceMatrix_real_to_voxel;
        if (splineControlPoint->sform_code > 0)
            referenceMatrix_real_to_voxel = &splineControlPoint->sto_ijk;
        else referenceMatrix_real_to_voxel = &splineControlPoint->qto_ijk;

        for (y = 0; y < deformationField->ny; y++) {
            index = y * deformationField->nx;
            oldXpre = oldYpre = -99;
            for (x = 0; x < deformationField->nx; x++) {
                if (mask[index] > -1) {
                    // The previous position at the current pixel position is read
                    xReal = fieldPtrX[index];
                    yReal = fieldPtrY[index];

                    // From real to pixel position in the CPP
                    xVoxel = referenceMatrix_real_to_voxel->m[0][0] * xReal
                        + referenceMatrix_real_to_voxel->m[0][1] * yReal
                        + referenceMatrix_real_to_voxel->m[0][3];
                    yVoxel = referenceMatrix_real_to_voxel->m[1][0] * xReal
                        + referenceMatrix_real_to_voxel->m[1][1] * yReal
                        + referenceMatrix_real_to_voxel->m[1][3];

                    // The spline coefficients are computed
                    xPre = Floor(xVoxel);
                    basis = xVoxel - static_cast<DataType>(xPre--);
                    if (basis < 0) basis = 0; //rounding error
                    if (bspline) get_BSplineBasisValues<DataType>(basis, xBasis);
                    else get_SplineBasisValues<DataType>(basis, xBasis);

                    yPre = Floor(yVoxel);
                    basis = yVoxel - static_cast<DataType>(yPre--);
                    if (basis < 0) basis = 0; //rounding error
                    if (bspline) get_BSplineBasisValues<DataType>(basis, yBasis);
                    else get_SplineBasisValues<DataType>(basis, yBasis);

                    if (xVoxel >= 0 && xVoxel <= deformationField->nx - 1 &&
                        yVoxel >= 0 && yVoxel <= deformationField->ny - 1) {
                        // The control point positions are extracted
                        if (oldXpre != xPre || oldYpre != yPre) {
#ifdef USE_SSE
                            get_GridValues<DataType>(xPre,
                                                     yPre,
                                                     splineControlPoint,
                                                     controlPointPtrX,
                                                     controlPointPtrY,
                                                     xControlPointCoordinates.f,
                                                     yControlPointCoordinates.f,
                                                     false,  // no approximation
                                                     false); // not a displacement field
#else // USE_SSE
                            get_GridValues<DataType>(xPre,
                                                     yPre,
                                                     splineControlPoint,
                                                     controlPointPtrX,
                                                     controlPointPtrY,
                                                     xControlPointCoordinates,
                                                     yControlPointCoordinates,
                                                     false,  // no approximation
                                                     false); // not a displacement field
#endif // USE_SSE
                            oldXpre = xPre;
                            oldYpre = yPre;
                        }
#if USE_SSE
                        coord = 0;
                        for (b = 0; b < 4; b++)
                            for (a = 0; a < 4; a++)
                                xyBasis.f[coord++] = xBasis[a] * yBasis[b];

                        tempX = _mm_set_ps1(0);
                        tempY = _mm_set_ps1(0);
                        //addition and multiplication of the 16 basis value and CP position for each axis
                        for (a = 0; a < 4; a++) {
                            tempX = _mm_add_ps(_mm_mul_ps(xyBasis.m[a], xControlPointCoordinates.m[a]), tempX);
                            tempY = _mm_add_ps(_mm_mul_ps(xyBasis.m[a], yControlPointCoordinates.m[a]), tempY);
                        }
                        //the values stored in SSE variables are transferred to normal float
                        val.m = tempX;
                        xReal = val.f[0] + val.f[1] + val.f[2] + val.f[3];
                        val.m = tempY;
                        yReal = val.f[0] + val.f[1] + val.f[2] + val.f[3];
#else
                        xReal = 0;
                        yReal = 0;
                        for (b = 0; b < 4; b++) {
                            for (a = 0; a < 4; a++) {
                                DataType tempValue = xBasis[a] * yBasis[b];
                                xReal += xControlPointCoordinates[b * 4 + a] * tempValue;
                                yReal += yControlPointCoordinates[b * 4 + a] * tempValue;
                            }
                        }
#endif
                    }

                    fieldPtrX[index] = xReal;
                    fieldPtrY[index] = yReal;
                }
                index++;
            }
        }
    } else { // starting deformation field is blank - !composition
#ifdef _OPENMP
#ifdef USE_SSE
#pragma  omp parallel for default(none) \
   shared(deformationField, gridVoxelSpacing, splineControlPoint, controlPointPtrX, \
   controlPointPtrY, mask, fieldPtrX, fieldPtrY, bspline) \
   private(x, a, xPre, yPre, oldXpre, oldYpre, index, xReal, yReal, basis, \
   val, xBasis, yBasis, tempCurrent, xyBasis, tempX, tempY, \
   xControlPointCoordinates, yControlPointCoordinates)
#else // USE_SSE
#pragma  omp parallel for default(none) \
   shared(deformationField, gridVoxelSpacing, splineControlPoint, controlPointPtrX, \
   controlPointPtrY, mask, fieldPtrX, fieldPtrY, bspline) \
   private(x, a, xPre, yPre, oldXpre, oldYpre, index, xReal, yReal, basis, coord, \
   xBasis, yBasis, xyBasis, xControlPointCoordinates, yControlPointCoordinates)
#endif // _USE_SEE
#endif // _OPENMP
        for (y = 0; y < deformationField->ny; y++) {
            index = y * deformationField->nx;
            oldXpre = oldYpre = -99;

            yPre = static_cast<int>(static_cast<DataType>(y) / gridVoxelSpacing[1]);
            basis = static_cast<DataType>(y) / gridVoxelSpacing[1] - static_cast<DataType>(yPre);
            if (basis < 0) basis = 0; // rounding error
            if (bspline) get_BSplineBasisValues<DataType>(basis, yBasis);
            else get_SplineBasisValues<DataType>(basis, yBasis);

            for (x = 0; x < deformationField->nx; x++) {
                xPre = static_cast<int>(static_cast<DataType>(x) / gridVoxelSpacing[0]);
                basis = static_cast<DataType>(x) / gridVoxelSpacing[0] - static_cast<DataType>(xPre);
                if (basis < 0) basis = 0; // rounding error
                if (bspline) get_BSplineBasisValues<DataType>(basis, xBasis);
                else get_SplineBasisValues<DataType>(basis, xBasis);
#if USE_SSE
                val.f[0] = static_cast<float>(xBasis[0]);
                val.f[1] = static_cast<float>(xBasis[1]);
                val.f[2] = static_cast<float>(xBasis[2]);
                val.f[3] = static_cast<float>(xBasis[3]);
                tempCurrent = val.m;
                for (a = 0; a < 4; a++) {
                    val.m = _mm_set_ps1(static_cast<float>(yBasis[a]));
                    xyBasis.m[a] = _mm_mul_ps(tempCurrent, val.m);
                }
#else
                coord = 0;
                for (a = 0; a < 4; a++) {
                    xyBasis[coord++] = xBasis[0] * yBasis[a];
                    xyBasis[coord++] = xBasis[1] * yBasis[a];
                    xyBasis[coord++] = xBasis[2] * yBasis[a];
                    xyBasis[coord++] = xBasis[3] * yBasis[a];
                }
#endif
                if (oldXpre != xPre || oldYpre != yPre) {
#ifdef USE_SSE
                    get_GridValues<DataType>(xPre,
                                             yPre,
                                             splineControlPoint,
                                             controlPointPtrX,
                                             controlPointPtrY,
                                             xControlPointCoordinates.f,
                                             yControlPointCoordinates.f,
                                             false,  // no approximation
                                             false); // not a deformation field
#else // USE_SSE
                    get_GridValues<DataType>(xPre,
                                             yPre,
                                             splineControlPoint,
                                             controlPointPtrX,
                                             controlPointPtrY,
                                             xControlPointCoordinates,
                                             yControlPointCoordinates,
                                             false,  // no approximation
                                             false); // not a deformation field
#endif // USE_SSE
                    oldXpre = xPre;
                    oldYpre = yPre;
                }

                xReal = 0;
                yReal = 0;

                if (mask[index] > -1) {
#if USE_SSE
                    tempX = _mm_set_ps1(0);
                    tempY = _mm_set_ps1(0);
                    //addition and multiplication of the 64 basis value and CP displacement for each axis
                    for (a = 0; a < 4; a++) {
                        tempX = _mm_add_ps(_mm_mul_ps(xyBasis.m[a], xControlPointCoordinates.m[a]), tempX);
                        tempY = _mm_add_ps(_mm_mul_ps(xyBasis.m[a], yControlPointCoordinates.m[a]), tempY);
                    }
                    //the values stored in SSE variables are transferred to normal float
                    val.m = tempX;
                    xReal = val.f[0] + val.f[1] + val.f[2] + val.f[3];
                    val.m = tempY;
                    yReal = val.f[0] + val.f[1] + val.f[2] + val.f[3];
#else
                    for (a = 0; a < 16; a++) {
                        xReal += xControlPointCoordinates[a] * xyBasis[a];
                        yReal += yControlPointCoordinates[a] * xyBasis[a];
                    }
#endif
                }// mask
                fieldPtrX[index] = (DataType)xReal;
                fieldPtrY[index] = (DataType)yReal;
                index++;
            } // x
        } // y
    } // composition
}
/* *************************************************************** */
template<class DataType>
void reg_cubic_spline_getDeformationField3D(nifti_image *splineControlPoint,
                                            nifti_image *deformationField,
                                            int *mask,
                                            bool composition,
                                            bool bspline,
                                            bool forceNoLut = false) {
#if USE_SSE
    union {
        __m128 m;
        float f[4];
    } val;
    __m128 tempX, tempY, tempZ, tempCurrent;
    __m128 xBasis_sse, yBasis_sse, zBasis_sse, temp_basis_sse, basis_sse;

#ifdef _WIN32
    __declspec(align(16)) DataType temp[4];
    __declspec(align(16)) DataType zBasis[4];
    union {
        __m128 m[16];
        __declspec(align(16)) DataType f[16];
    } xControlPointCoordinates;
    union {
        __m128 m[16];
        __declspec(align(16)) DataType f[16];
    } yControlPointCoordinates;
    union {
        __m128 m[16];
        __declspec(align(16)) DataType f[16];
    } zControlPointCoordinates;
#else // _WIN32
    DataType temp[4] __attribute__((aligned(16)));
    DataType zBasis[4] __attribute__((aligned(16)));
    union {
        __m128 m[16];
        DataType f[16] __attribute__((aligned(16)));
    } xControlPointCoordinates;
    union {
        __m128 m[16];
        DataType f[16] __attribute__((aligned(16)));
    } yControlPointCoordinates;
    union {
        __m128 m[16];
        DataType f[16] __attribute__((aligned(16)));
    } zControlPointCoordinates;
#endif // _WIN32
#else // USE_SSE
    DataType temp[4];
    DataType zBasis[4];
    DataType xControlPointCoordinates[64];
    DataType yControlPointCoordinates[64];
    DataType zControlPointCoordinates[64];
    int coord;
#endif // USE_SSE

    const size_t splineControlPointVoxelNumber = NiftiImage::calcVoxelNumber(splineControlPoint, 3);
    DataType *controlPointPtrX = static_cast<DataType*>(splineControlPoint->data);
    DataType *controlPointPtrY = &controlPointPtrX[splineControlPointVoxelNumber];
    DataType *controlPointPtrZ = &controlPointPtrY[splineControlPointVoxelNumber];

    const size_t deformationFieldVoxelNumber = NiftiImage::calcVoxelNumber(deformationField, 3);
    DataType *fieldPtrX = static_cast<DataType*>(deformationField->data);
    DataType *fieldPtrY = &fieldPtrX[deformationFieldVoxelNumber];
    DataType *fieldPtrZ = &fieldPtrY[deformationFieldVoxelNumber];

    DataType basis, oldBasis = 1.1f;

    int x, y, z, a, b, c, oldPreX, oldPreY, oldPreZ, xPre, yPre, zPre, index;
    DataType real[3];

    if (composition) {  // Composition of deformation fields
        // read the ijk sform or qform, as appropriate
        mat44 referenceMatrix_real_to_voxel;
        if (splineControlPoint->sform_code > 0)
            referenceMatrix_real_to_voxel = splineControlPoint->sto_ijk;
        else referenceMatrix_real_to_voxel = splineControlPoint->qto_ijk;
#ifdef USE_SSE
#ifdef _WIN32
        __declspec(align(16)) DataType xBasis[4];
        __declspec(align(16)) DataType yBasis[4];
#else
        DataType xBasis[4] __attribute__((aligned(16)));
        DataType yBasis[4] __attribute__((aligned(16)));
#endif
#else // USE_SSE
        DataType xBasis[4], yBasis[4];
#endif // USE_SSE

        DataType voxel[3];

#ifdef _OPENMP
#ifdef USE_SSE
#pragma omp parallel for default(none) \
   private(x, y, b, c, oldPreX, oldPreY, oldPreZ, xPre, yPre, zPre, real, \
   index, voxel, basis, xBasis, yBasis, zBasis, xControlPointCoordinates, \
   yControlPointCoordinates, zControlPointCoordinates,  \
   tempX, tempY, tempZ, xBasis_sse, yBasis_sse, zBasis_sse, \
   temp_basis_sse, basis_sse, val) \
   shared(deformationField, fieldPtrX, fieldPtrY, fieldPtrZ, referenceMatrix_real_to_voxel, \
   bspline, controlPointPtrX, controlPointPtrY, controlPointPtrZ, \
   splineControlPoint, mask)
#else
#pragma omp parallel for default(none) \
   private(x, y, a, b, c, oldPreX, oldPreY, oldPreZ, xPre, yPre, zPre, real, \
   index, voxel, basis, xBasis, yBasis, zBasis, xControlPointCoordinates, \
   yControlPointCoordinates, zControlPointCoordinates, coord) \
   shared(deformationField, fieldPtrX, fieldPtrY, fieldPtrZ, referenceMatrix_real_to_voxel, \
   bspline, controlPointPtrX, controlPointPtrY, controlPointPtrZ, \
   splineControlPoint, mask)
#endif // USE_SSE
#endif // _OPENMP
        for (z = 0; z < deformationField->nz; z++) {
            index = z * deformationField->nx * deformationField->ny;
            oldPreX = oldPreY = oldPreZ = -99;
            for (y = 0; y < deformationField->ny; y++) {
                for (x = 0; x < deformationField->nx; x++) {
                    if (mask[index] > -1) {
                        // The previous position at the current pixel position is read
                        real[0] = fieldPtrX[index];
                        real[1] = fieldPtrY[index];
                        real[2] = fieldPtrZ[index];

                        // From real to pixel position in the control point space
                        voxel[0] =
                            referenceMatrix_real_to_voxel.m[0][0] * real[0] +
                            referenceMatrix_real_to_voxel.m[0][1] * real[1] +
                            referenceMatrix_real_to_voxel.m[0][2] * real[2] +
                            referenceMatrix_real_to_voxel.m[0][3];
                        voxel[1] =
                            referenceMatrix_real_to_voxel.m[1][0] * real[0] +
                            referenceMatrix_real_to_voxel.m[1][1] * real[1] +
                            referenceMatrix_real_to_voxel.m[1][2] * real[2] +
                            referenceMatrix_real_to_voxel.m[1][3];
                        voxel[2] =
                            referenceMatrix_real_to_voxel.m[2][0] * real[0] +
                            referenceMatrix_real_to_voxel.m[2][1] * real[1] +
                            referenceMatrix_real_to_voxel.m[2][2] * real[2] +
                            referenceMatrix_real_to_voxel.m[2][3];

                        // The spline coefficients are computed
                        xPre = Floor(voxel[0]);
                        basis = voxel[0] - static_cast<DataType>(xPre--);
                        if (basis < 0) basis = 0; //rounding error
                        if (bspline) get_BSplineBasisValues<DataType>(basis, xBasis);
                        else get_SplineBasisValues<DataType>(basis, xBasis);

                        yPre = Floor(voxel[1]);
                        basis = voxel[1] - static_cast<DataType>(yPre--);
                        if (basis < 0) basis = 0; //rounding error
                        if (bspline) get_BSplineBasisValues<DataType>(basis, yBasis);
                        else get_SplineBasisValues<DataType>(basis, yBasis);

                        zPre = Floor(voxel[2]);
                        basis = voxel[2] - static_cast<DataType>(zPre--);
                        if (basis < 0) basis = 0; //rounding error
                        if (bspline) get_BSplineBasisValues<DataType>(basis, zBasis);
                        else get_SplineBasisValues<DataType>(basis, zBasis);

                        // The control point positions are extracted
                        if (xPre != oldPreX || yPre != oldPreY || zPre != oldPreZ) {
#ifdef USE_SSE
                            get_GridValues<DataType>(xPre,
                                                     yPre,
                                                     zPre,
                                                     splineControlPoint,
                                                     controlPointPtrX,
                                                     controlPointPtrY,
                                                     controlPointPtrZ,
                                                     xControlPointCoordinates.f,
                                                     yControlPointCoordinates.f,
                                                     zControlPointCoordinates.f,
                                                     false,  // no approximation
                                                     false); // not a deformation field
#else // USE_SSE
                            get_GridValues<DataType>(xPre,
                                                     yPre,
                                                     zPre,
                                                     splineControlPoint,
                                                     controlPointPtrX,
                                                     controlPointPtrY,
                                                     controlPointPtrZ,
                                                     xControlPointCoordinates,
                                                     yControlPointCoordinates,
                                                     zControlPointCoordinates,
                                                     false,  // no approximation
                                                     false); // not a deformation field
#endif // USE_SSE
                            oldPreX = xPre;
                            oldPreY = yPre;
                            oldPreZ = zPre;
                        }

#if USE_SSE
                        tempX = _mm_set_ps1(0);
                        tempY = _mm_set_ps1(0);
                        tempZ = _mm_set_ps1(0);
                        val.f[0] = static_cast<float>(xBasis[0]);
                        val.f[1] = static_cast<float>(xBasis[1]);
                        val.f[2] = static_cast<float>(xBasis[2]);
                        val.f[3] = static_cast<float>(xBasis[3]);
                        xBasis_sse = val.m;

                        //addition and multiplication of the 16 basis value and CP position for each axis
                        for (c = 0; c < 4; c++) {
                            for (b = 0; b < 4; b++) {
                                yBasis_sse = _mm_set_ps1(static_cast<float>(yBasis[b]));
                                zBasis_sse = _mm_set_ps1(static_cast<float>(zBasis[c]));
                                temp_basis_sse = _mm_mul_ps(yBasis_sse, zBasis_sse);
                                basis_sse = _mm_mul_ps(temp_basis_sse, xBasis_sse);

                                tempX = _mm_add_ps(_mm_mul_ps(basis_sse, xControlPointCoordinates.m[c * 4 + b]), tempX);
                                tempY = _mm_add_ps(_mm_mul_ps(basis_sse, yControlPointCoordinates.m[c * 4 + b]), tempY);
                                tempZ = _mm_add_ps(_mm_mul_ps(basis_sse, zControlPointCoordinates.m[c * 4 + b]), tempZ);
                            }
                        }
                        //the values stored in SSE variables are transferred to normal float
                        val.m = tempX;
                        real[0] = val.f[0] + val.f[1] + val.f[2] + val.f[3];
                        val.m = tempY;
                        real[1] = val.f[0] + val.f[1] + val.f[2] + val.f[3];
                        val.m = tempZ;
                        real[2] = val.f[0] + val.f[1] + val.f[2] + val.f[3];
#else
                        real[0] = 0;
                        real[1] = 0;
                        real[2] = 0;
                        coord = 0;
                        for (c = 0; c < 4; c++) {
                            for (b = 0; b < 4; b++) {
                                for (a = 0; a < 4; a++) {
                                    DataType tempValue = xBasis[a] * yBasis[b] * zBasis[c];
                                    real[0] += xControlPointCoordinates[coord] * tempValue;
                                    real[1] += yControlPointCoordinates[coord] * tempValue;
                                    real[2] += zControlPointCoordinates[coord] * tempValue;
                                    coord++;
                                }
                            }
                        }
#endif
                        fieldPtrX[index] = real[0];
                        fieldPtrY[index] = real[1];
                        fieldPtrZ[index] = real[2];
                    }
                    index++;
                }
            }
        }
    } else { // !composition
        DataType gridVoxelSpacing[3];
        gridVoxelSpacing[0] = splineControlPoint->dx / deformationField->dx;
        gridVoxelSpacing[1] = splineControlPoint->dy / deformationField->dy;
        gridVoxelSpacing[2] = splineControlPoint->dz / deformationField->dz;

#ifdef USE_SSE
#ifdef _WIN32
        union u1 {
            __m128 m[4];
            __declspec(align(16)) DataType f[16];
        } yzBasis;
        union u2 {
            __m128 m[16];
            __declspec(align(16)) DataType f[64];
        } xyzBasis;
#else // _WIN32
        union {
            __m128 m[4];
            DataType f[16] __attribute__((aligned(16)));
        } yzBasis;
        union {
            __m128 m[16];
            DataType f[64] __attribute__((aligned(16)));
        } xyzBasis;
#endif // _WIN32
#else // USE_SSE
        DataType yzBasis[16], xyzBasis[64];
#endif // USE_SSE

        // Assess if lookup table can be used
        if (gridVoxelSpacing[0] == 5. && gridVoxelSpacing[0] == 5. && gridVoxelSpacing[0] == 5. && forceNoLut == false) {
            // Assign a single array that will contain all coefficients
            DataType *coefficients = (DataType*)malloc(125 * 64 * sizeof(DataType));
            // Compute and store all required coefficients
            int coeff_index;
#ifdef _OPENMP
#ifdef USE_SSE
#pragma omp parallel for default(none) \
    private(x, y, a, coeff_index, basis, zBasis, temp, val, tempCurrent, yzBasis) \
    shared(coefficients, bspline)
#else //  USE_SSE
#pragma omp parallel for default(none) \
    private(x, y, a, coeff_index, basis, zBasis, temp, yzBasis, coord) \
    shared(coefficients, bspline)
#endif // USE_SSE
#endif // _OPENMP
            for (z = 0; z < 5; ++z) {
                coeff_index = z * 5 * 5 * 64;
                basis = static_cast<DataType>(z) / 5.f;
                if (bspline) get_BSplineBasisValues<DataType>(basis, zBasis);
                else get_SplineBasisValues<DataType>(basis, zBasis);
                for (y = 0; y < 5; ++y) {
                    basis = static_cast<DataType>(y) / 5.f;
                    if (bspline) get_BSplineBasisValues<DataType>(basis, temp);
                    else get_SplineBasisValues<DataType>(basis, temp);
#if USE_SSE
                    val.f[0] = static_cast<float>(temp[0]);
                    val.f[1] = static_cast<float>(temp[1]);
                    val.f[2] = static_cast<float>(temp[2]);
                    val.f[3] = static_cast<float>(temp[3]);
                    tempCurrent = val.m;
                    for (a = 0; a < 4; a++) {
                        val.m = _mm_set_ps1(static_cast<float>(zBasis[a]));
                        yzBasis.m[a] = _mm_mul_ps(tempCurrent, val.m);
                    }
#else
                    coord = 0;
                    for (a = 0; a < 4; a++) {
                        yzBasis[coord++] = temp[0] * zBasis[a];
                        yzBasis[coord++] = temp[1] * zBasis[a];
                        yzBasis[coord++] = temp[2] * zBasis[a];
                        yzBasis[coord++] = temp[3] * zBasis[a];
                    }
#endif

                    for (x = 0; x < 5; ++x) {
                        basis = static_cast<DataType>(x) / 5.f;
                        if (bspline) get_BSplineBasisValues<DataType>(basis, temp);
                        else get_SplineBasisValues<DataType>(basis, temp);
#if USE_SSE
                        val.f[0] = static_cast<float>(temp[0]);
                        val.f[1] = static_cast<float>(temp[1]);
                        val.f[2] = static_cast<float>(temp[2]);
                        val.f[3] = static_cast<float>(temp[3]);
                        tempCurrent = val.m;
                        for (a = 0; a < 16; ++a) {
                            val.m = _mm_set_ps1(static_cast<float>(yzBasis.f[a]));
                            val.m = _mm_mul_ps(tempCurrent, val.m);
                            coefficients[coeff_index++] = val.f[0];
                            coefficients[coeff_index++] = val.f[1];
                            coefficients[coeff_index++] = val.f[2];
                            coefficients[coeff_index++] = val.f[3];
                        }
#else
                        for (a = 0; a < 16; a++) {
                            coefficients[coeff_index++] = temp[0] * yzBasis[a];
                            coefficients[coeff_index++] = temp[1] * yzBasis[a];
                            coefficients[coeff_index++] = temp[2] * yzBasis[a];
                            coefficients[coeff_index++] = temp[3] * yzBasis[a];
                        }
#endif
                    } //x
                } // y
            } // z

            // Loop over block of 5x5x5 voxels
#if USE_SSE
            int coord;
#endif // USE_SSE
#ifdef _OPENMP
#ifdef USE_SSE
#pragma omp parallel for default(none) \
   private(x, y, z, a, b, c, xPre, yPre, real, \
   index, coeff_index, coord, tempX, tempY, tempZ, val,\
   xControlPointCoordinates, yControlPointCoordinates, zControlPointCoordinates) \
   shared(deformationField, fieldPtrX, fieldPtrY, fieldPtrZ, splineControlPoint, mask, \
   gridVoxelSpacing, bspline, controlPointPtrX, controlPointPtrY, controlPointPtrZ, \
   coefficients)
#else //  USE_SSE
#pragma omp parallel for default(none) \
   private(x, y, z, a, b, c, xPre, yPre, real, \
   index, coeff_index, coord, basis, \
   xControlPointCoordinates, yControlPointCoordinates, zControlPointCoordinates) \
   shared(deformationField, fieldPtrX, fieldPtrY, fieldPtrZ, splineControlPoint, mask, \
   gridVoxelSpacing, bspline, controlPointPtrX, controlPointPtrY, controlPointPtrZ, \
   coefficients)
#endif // USE_SSE
#endif // _OPENMP
            for (zPre = 0; zPre < splineControlPoint->nz - 3; zPre++) {
                for (yPre = 0; yPre < splineControlPoint->ny - 3; yPre++) {
                    for (xPre = 0; xPre < splineControlPoint->nx - 3; xPre++) {
#if USE_SSE
                        get_GridValues<DataType>(xPre,
                                                 yPre,
                                                 zPre,
                                                 splineControlPoint,
                                                 controlPointPtrX,
                                                 controlPointPtrY,
                                                 controlPointPtrZ,
                                                 xControlPointCoordinates.f,
                                                 yControlPointCoordinates.f,
                                                 zControlPointCoordinates.f,
                                                 false,  // no approximation
                                                 false); // not a deformation field
#else // USE_SSE
                        get_GridValues<DataType>(xPre,
                                                 yPre,
                                                 zPre,
                                                 splineControlPoint,
                                                 controlPointPtrX,
                                                 controlPointPtrY,
                                                 controlPointPtrZ,
                                                 xControlPointCoordinates,
                                                 yControlPointCoordinates,
                                                 zControlPointCoordinates,
                                                 false,  // no approximation
                                                 false); // not a deformation field
#endif // USE_SSE
                        coeff_index = 0;
                        for (c = 0; c < 5; ++c) {
                            z = zPre * 5 + c;
                            if (z < deformationField->nz) {
                                for (b = 0; b < 5; ++b) {
                                    y = yPre * 5 + b;
                                    if (y < deformationField->ny) {
                                        index = (z * deformationField->ny + y) * deformationField->nx + xPre * 5;
                                        for (a = 0; a < 5; ++a) {
                                            x = xPre * 5 + a;
                                            if (x<deformationField->nx && mask[index]>-1) {
#if USE_SSE
                                                tempX = _mm_set_ps1(0);
                                                tempY = _mm_set_ps1(0);
                                                tempZ = _mm_set_ps1(0);
                                                for (coord = 0; coord < 16; ++coord) {
                                                    val.m = _mm_set_ps(static_cast<float>(coefficients[coeff_index + 3]),
                                                                       static_cast<float>(coefficients[coeff_index + 2]),
                                                                       static_cast<float>(coefficients[coeff_index + 1]),
                                                                       static_cast<float>(coefficients[coeff_index]));
                                                    coeff_index += 4;
                                                    tempX = _mm_add_ps(_mm_mul_ps(val.m, xControlPointCoordinates.m[coord]), tempX);
                                                    tempY = _mm_add_ps(_mm_mul_ps(val.m, yControlPointCoordinates.m[coord]), tempY);
                                                    tempZ = _mm_add_ps(_mm_mul_ps(val.m, zControlPointCoordinates.m[coord]), tempZ);
                                                }
                                                // The values stored in SSE variables are transferred to normal float
#ifdef __SSE3__
                                                val.m = _mm_hadd_ps(tempX, tempY);
                                                val.m = _mm_hadd_ps(val.m, tempZ);
                                                real[0] = val.f[0];
                                                real[1] = val.f[1];
                                                real[2] = val.f[2] + val.f[3];
#else
                                                val.m = tempX;
                                                real[0] = val.f[0] + val.f[1] + val.f[2] + val.f[3];
                                                val.m = tempY;
                                                real[1] = val.f[0] + val.f[1] + val.f[2] + val.f[3];
                                                val.m = tempZ;
                                                real[2] = val.f[0] + val.f[1] + val.f[2] + val.f[3];
#endif
#else // USE_SSE
                                                real[0] = real[1] = real[2] = 0;
                                                for (coord = 0; coord < 64; ++coord) {
                                                    basis = coefficients[coeff_index++];
                                                    real[0] += xControlPointCoordinates[coord] * basis;
                                                    real[1] += yControlPointCoordinates[coord] * basis;
                                                    real[2] += zControlPointCoordinates[coord] * basis;
                                                }
#endif // USE_SSE
                                                fieldPtrX[index] = real[0];
                                                fieldPtrY[index] = real[1];
                                                fieldPtrZ[index] = real[2];
                                            } // x defined
                                            else coeff_index += 64;
                                            index++;
                                        } // a
                                    } // y defined
                                    else coeff_index += 5 * 64;
                                } // b
                            } // z defined
                            else coeff_index += 5 * 5 * 64;
                        } // c
                    } // xPre
                } // yPre
            } // zPre
            free(coefficients);
        } else { // if spacings!=5 voxels
#ifdef _OPENMP
#ifdef USE_SSE
#pragma omp parallel for default(none) \
    private(x, y, a, xPre, yPre, zPre, real, \
    index, basis, xyzBasis, yzBasis, zBasis, temp, xControlPointCoordinates, \
    yControlPointCoordinates, zControlPointCoordinates, oldBasis, \
    tempX, tempY, tempZ, xBasis_sse, yBasis_sse, zBasis_sse, \
    temp_basis_sse, basis_sse, val, tempCurrent) \
    shared(deformationField, fieldPtrX, fieldPtrY, fieldPtrZ, splineControlPoint, mask, \
    gridVoxelSpacing, bspline, controlPointPtrX, controlPointPtrY, controlPointPtrZ)
#else //  USE_SSE
#pragma omp parallel for default(none) \
    private(x, y, a, xPre, yPre, zPre, real, \
    index, basis, xyzBasis, yzBasis, zBasis, temp, xControlPointCoordinates, \
    yControlPointCoordinates, zControlPointCoordinates, oldBasis, coord) \
    shared(deformationField, fieldPtrX, fieldPtrY, fieldPtrZ, splineControlPoint, mask, \
    gridVoxelSpacing, bspline, controlPointPtrX, controlPointPtrY, controlPointPtrZ)
#endif // USE_SSE
#endif // _OPENMP
            for (z = 0; z < deformationField->nz; z++) {
                index = z * deformationField->nx * deformationField->ny;
                oldBasis = 1.1f;

                zPre = static_cast<int>(static_cast<DataType>(z) / gridVoxelSpacing[2]);
                basis = static_cast<DataType>(z) / gridVoxelSpacing[2] - static_cast<DataType>(zPre);
                if (basis < 0) basis = 0; //rounding error
                if (bspline) get_BSplineBasisValues<DataType>(basis, zBasis);
                else get_SplineBasisValues<DataType>(basis, zBasis);

                for (y = 0; y < deformationField->ny; y++) {
                    yPre = static_cast<int>(static_cast<DataType>(y) / gridVoxelSpacing[1]);
                    basis = static_cast<DataType>(y) / gridVoxelSpacing[1] - static_cast<DataType>(yPre);
                    if (basis < 0) basis = 0; //rounding error
                    if (bspline) get_BSplineBasisValues<DataType>(basis, temp);
                    else get_SplineBasisValues<DataType>(basis, temp);
#if USE_SSE
                    val.f[0] = static_cast<float>(temp[0]);
                    val.f[1] = static_cast<float>(temp[1]);
                    val.f[2] = static_cast<float>(temp[2]);
                    val.f[3] = static_cast<float>(temp[3]);
                    tempCurrent = val.m;
                    for (a = 0; a < 4; a++) {
                        val.m = _mm_set_ps1(static_cast<float>(zBasis[a]));
                        yzBasis.m[a] = _mm_mul_ps(tempCurrent, val.m);
                    }
#else
                    coord = 0;
                    for (a = 0; a < 4; a++) {
                        yzBasis[coord++] = temp[0] * zBasis[a];
                        yzBasis[coord++] = temp[1] * zBasis[a];
                        yzBasis[coord++] = temp[2] * zBasis[a];
                        yzBasis[coord++] = temp[3] * zBasis[a];
                    }
#endif
                    for (x = 0; x < deformationField->nx; x++) {
                        xPre = static_cast<int>(static_cast<DataType>(x) / gridVoxelSpacing[0]);
                        basis = static_cast<DataType>(x) / gridVoxelSpacing[0] - static_cast<DataType>(xPre);
                        if (basis < 0) basis = 0; //rounding error
                        if (bspline) get_BSplineBasisValues<DataType>(basis, temp);
                        else get_SplineBasisValues<DataType>(basis, temp);
#if USE_SSE
                        val.f[0] = static_cast<float>(temp[0]);
                        val.f[1] = static_cast<float>(temp[1]);
                        val.f[2] = static_cast<float>(temp[2]);
                        val.f[3] = static_cast<float>(temp[3]);
                        tempCurrent = val.m;
                        for (a = 0; a < 16; ++a) {
                            val.m = _mm_set_ps1(static_cast<float>(yzBasis.f[a]));
                            xyzBasis.m[a] = _mm_mul_ps(tempCurrent, val.m);
                        }
#else
                        coord = 0;
                        for (a = 0; a < 16; a++) {
                            xyzBasis[coord++] = temp[0] * yzBasis[a];
                            xyzBasis[coord++] = temp[1] * yzBasis[a];
                            xyzBasis[coord++] = temp[2] * yzBasis[a];
                            xyzBasis[coord++] = temp[3] * yzBasis[a];
                        }
#endif
                        if (basis <= oldBasis || x == 0) {
#ifdef USE_SSE
                            get_GridValues<DataType>(xPre,
                                                     yPre,
                                                     zPre,
                                                     splineControlPoint,
                                                     controlPointPtrX,
                                                     controlPointPtrY,
                                                     controlPointPtrZ,
                                                     xControlPointCoordinates.f,
                                                     yControlPointCoordinates.f,
                                                     zControlPointCoordinates.f,
                                                     false,  // no approximation
                                                     false); // not a deformation field
#else // USE_SSE
                            get_GridValues<DataType>(xPre,
                                                     yPre,
                                                     zPre,
                                                     splineControlPoint,
                                                     controlPointPtrX,
                                                     controlPointPtrY,
                                                     controlPointPtrZ,
                                                     xControlPointCoordinates,
                                                     yControlPointCoordinates,
                                                     zControlPointCoordinates,
                                                     false,  // no approximation
                                                     false); // not a deformation field
#endif // USE_SSE
                        }
                        oldBasis = basis;

                        real[0] = 0;
                        real[1] = 0;
                        real[2] = 0;

                        if (mask[index] > -1) {
#if USE_SSE
                            tempX = _mm_set_ps1(0);
                            tempY = _mm_set_ps1(0);
                            tempZ = _mm_set_ps1(0);
                            //addition and multiplication of the 64 basis value and CP displacement for each axis
                            for (a = 0; a < 16; a++) {
                                tempX = _mm_add_ps(_mm_mul_ps(xyzBasis.m[a], xControlPointCoordinates.m[a]), tempX);
                                tempY = _mm_add_ps(_mm_mul_ps(xyzBasis.m[a], yControlPointCoordinates.m[a]), tempY);
                                tempZ = _mm_add_ps(_mm_mul_ps(xyzBasis.m[a], zControlPointCoordinates.m[a]), tempZ);
                            }
                            //the values stored in SSE variables are transferred to normal float
                            val.m = tempX;
                            real[0] = val.f[0] + val.f[1] + val.f[2] + val.f[3];
                            val.m = tempY;
                            real[1] = val.f[0] + val.f[1] + val.f[2] + val.f[3];
                            val.m = tempZ;
                            real[2] = val.f[0] + val.f[1] + val.f[2] + val.f[3];
#else
                            for (a = 0; a < 64; a++) {
                                real[0] += xControlPointCoordinates[a] * xyzBasis[a];
                                real[1] += yControlPointCoordinates[a] * xyzBasis[a];
                                real[2] += zControlPointCoordinates[a] * xyzBasis[a];
                            }
#endif
                        }// mask
                        fieldPtrX[index] = real[0];
                        fieldPtrY[index] = real[1];
                        fieldPtrZ[index] = real[2];
                        index++;
                    } // x
                } // y
            } // z
        } // else spacing==5
    }// from a deformation field
}
/* *************************************************************** */
void reg_spline_getDeformationField(nifti_image *splineControlPoint,
                                    nifti_image *deformationField,
                                    int *mask,
                                    bool composition,
                                    bool bspline,
                                    bool forceNoLut) {
    if (splineControlPoint->datatype != deformationField->datatype)
        NR_FATAL_ERROR("The spline control point image and the deformation field image are expected to be of the same type");

#if USE_SSE
    if (splineControlPoint->datatype != NIFTI_TYPE_FLOAT32)
        NR_FATAL_ERROR("SSE computation has only been implemented for single precision");
#endif

    unique_ptr<int[]> currentMask;
    if (!mask) {
        // Active voxel are all superior to -1, 0 thus will do !
        currentMask.reset(new int[NiftiImage::calcVoxelNumber(deformationField, 3)]());
        mask = currentMask.get();
    }

    // Check if an affine initialisation is required
    if (splineControlPoint->num_ext > 0) {
        if (splineControlPoint->ext_list[0].edata != nullptr) {
            reg_affine_getDeformationField(reinterpret_cast<mat44*>(splineControlPoint->ext_list[0].edata),
                                           deformationField,
                                           composition,
                                           mask);
            composition = true;
        }
    }

    if (splineControlPoint->intent_p1 == LIN_SPLINE_GRID) {
        if (splineControlPoint->nz == 1) {
            NR_FATAL_ERROR("No 2D implementation yet");
        } else {
            switch (deformationField->datatype) {
            case NIFTI_TYPE_FLOAT32:
                reg_linear_spline_getDeformationField3D<float>(splineControlPoint, deformationField, mask, composition);
                break;
            case NIFTI_TYPE_FLOAT64:
                reg_linear_spline_getDeformationField3D<double>(splineControlPoint, deformationField, mask, composition);
                break;
            default:
                NR_FATAL_ERROR("Only single or double precision is implemented for deformation field");
            }
        }
    } else {
        if (splineControlPoint->nz == 1) {
            switch (deformationField->datatype) {
            case NIFTI_TYPE_FLOAT32:
                reg_cubic_spline_getDeformationField2D<float>(splineControlPoint, deformationField, mask, composition, bspline);
                break;
            case NIFTI_TYPE_FLOAT64:
                reg_cubic_spline_getDeformationField2D<double>(splineControlPoint, deformationField, mask, composition, bspline);
                break;
            default:
                NR_FATAL_ERROR("Only single or double precision is implemented for deformation field");
            }
        } else {
            switch (deformationField->datatype) {
            case NIFTI_TYPE_FLOAT32:
                reg_cubic_spline_getDeformationField3D<float>(splineControlPoint, deformationField, mask, composition, bspline, forceNoLut);
                break;
            case NIFTI_TYPE_FLOAT64:
                reg_cubic_spline_getDeformationField3D<double>(splineControlPoint, deformationField, mask, composition, bspline, forceNoLut);
                break;
            default:
                NR_FATAL_ERROR("Only single or double precision is implemented for deformation field");
            }
        }
    }

    if (splineControlPoint->num_ext > 1) {
        if (splineControlPoint->ext_list[1].edata != nullptr) {
            reg_affine_getDeformationField(reinterpret_cast<mat44*>(splineControlPoint->ext_list[1].edata),
                                           deformationField,
                                           true, // composition
                                           mask);
        }
    }
}
/* *************************************************************** */
template<class DataType>
void reg_voxelCentricToNodeCentric(nifti_image *nodeImage,
                                   nifti_image *voxelImage,
                                   float weight,
                                   bool update,
                                   const mat44 *voxelToMillimetre) {
    const size_t nodeNumber = NiftiImage::calcVoxelNumber(nodeImage, 3);
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(voxelImage, 3);
    DataType *nodePtrX = static_cast<DataType*>(nodeImage->data);
    DataType *nodePtrY = &nodePtrX[nodeNumber];
    DataType *nodePtrZ = nullptr;

    DataType *voxelPtrX = static_cast<DataType*>(voxelImage->data);
    DataType *voxelPtrY = &voxelPtrX[voxelNumber];
    DataType *voxelPtrZ = nullptr;

    if (nodeImage->nz > 1) {
        nodePtrZ = &nodePtrY[nodeNumber];
        voxelPtrZ = &voxelPtrY[voxelNumber];
    }

    // The transformation between the image and the grid is used
    mat44 transformation;
    // voxel to millimetre in the grid image
    if (nodeImage->sform_code > 0)
        transformation = nodeImage->sto_xyz;
    else transformation = nodeImage->qto_xyz;
    // Affine transformation between the grid and the reference image
    if (nodeImage->num_ext > 0) {
        if (nodeImage->ext_list[0].edata != nullptr) {
            mat44 temp = *(reinterpret_cast<mat44*>(nodeImage->ext_list[0].edata));
            temp = nifti_mat44_inverse(temp);
            transformation = reg_mat44_mul(&temp, &transformation);
        }
    }
    // millimetre to voxel in the reference image
    if (voxelImage->sform_code > 0)
        transformation = reg_mat44_mul(&voxelImage->sto_ijk, &transformation);
    else transformation = reg_mat44_mul(&voxelImage->qto_ijk, &transformation);

    // The information has to be reoriented
    mat33 reorientation;
    // Voxel to millimetre contains the orientation of the image that is used
    // to compute the spatial gradient (floating image)
    if (voxelToMillimetre != nullptr) {
        reorientation = reg_mat44_to_mat33(voxelToMillimetre);
        if (nodeImage->num_ext > 0) {
            if (nodeImage->ext_list[0].edata != nullptr) {
                mat33 temp = reg_mat44_to_mat33(reinterpret_cast<mat44*>(nodeImage->ext_list[0].edata));
                temp = nifti_mat33_inverse(temp);
                reorientation = nifti_mat33_mul(temp, reorientation);
            }
        }
    } else reg_mat33_eye(&reorientation);
    // The information has to be weighted
    float ratio[3] = { nodeImage->dx, nodeImage->dy, nodeImage->dz };
    for (int i = 0; i < (nodeImage->nz > 1 ? 3 : 2); ++i) {
        if (nodeImage->sform_code > 0) {
            ratio[i] = sqrt(Square(nodeImage->sto_xyz.m[i][0]) +
                            Square(nodeImage->sto_xyz.m[i][1]) +
                            Square(nodeImage->sto_xyz.m[i][2]));
        }
        ratio[i] /= voxelImage->pixdim[i + 1];
        weight *= ratio[i];
    }
    // For each node, the corresponding voxel is computed
    float nodeCoord[3], voxelCoord[3];
    for (int z = 0; z < nodeImage->nz; z++) {
        nodeCoord[2] = static_cast<float>(z);
        for (int y = 0; y < nodeImage->ny; y++) {
            nodeCoord[1] = static_cast<float>(y);
            for (int x = 0; x < nodeImage->nx; x++) {
                nodeCoord[0] = static_cast<float>(x);
                reg_mat44_mul(&transformation, nodeCoord, voxelCoord);
                // linear interpolation is performed
                DataType basisX[2], basisY[2], basisZ[2] = { 0, 0 };
                int pre[3] = {
                    Floor(voxelCoord[0]),
                    Floor(voxelCoord[1]),
                    Floor(voxelCoord[2])
                };
                basisX[1] = voxelCoord[0] - static_cast<DataType>(pre[0]);
                basisX[0] = static_cast<DataType>(1) - basisX[1];
                basisY[1] = voxelCoord[1] - static_cast<DataType>(pre[1]);
                basisY[0] = static_cast<DataType>(1) - basisY[1];
                if (voxelPtrZ != nullptr) {
                    basisZ[1] = voxelCoord[2] - static_cast<DataType>(pre[2]);
                    basisZ[0] = static_cast<DataType>(1) - basisZ[1];
                }
                DataType interpolatedValue[3] = { 0, 0, 0 };
                for (int c = 0; c < 2; ++c) {
                    int indexZ = pre[2] + c;
                    if (indexZ > -1 && indexZ < voxelImage->nz) {
                        for (int b = 0; b < 2; ++b) {
                            int indexY = pre[1] + b;
                            if (indexY > -1 && indexY < voxelImage->ny) {
                                for (int a = 0; a < 2; ++a) {
                                    int indexX = pre[0] + a;
                                    if (indexX > -1 && indexX < voxelImage->nx) {
                                        size_t index = (indexZ * voxelImage->ny + indexY) *
                                            voxelImage->nx + indexX;
                                        DataType linearWeight = basisX[a] * basisY[b];
                                        if (voxelPtrZ != nullptr) linearWeight *= basisZ[c];
                                        interpolatedValue[0] += linearWeight * voxelPtrX[index];
                                        interpolatedValue[1] += linearWeight * voxelPtrY[index];
                                        if (voxelPtrZ != nullptr)
                                            interpolatedValue[2] += linearWeight * voxelPtrZ[index];
                                    }
                                }
                            }
                        }
                    }
                }
                DataType reorientedValue[3] = { 0, 0, 0 };
                reorientedValue[0] =
                    reorientation.m[0][0] * interpolatedValue[0] +
                    reorientation.m[1][0] * interpolatedValue[1] +
                    reorientation.m[2][0] * interpolatedValue[2];
                reorientedValue[1] =
                    reorientation.m[0][1] * interpolatedValue[0] +
                    reorientation.m[1][1] * interpolatedValue[1] +
                    reorientation.m[2][1] * interpolatedValue[2];
                if (voxelPtrZ != nullptr)
                    reorientedValue[2] =
                    reorientation.m[0][2] * interpolatedValue[0] +
                    reorientation.m[1][2] * interpolatedValue[1] +
                    reorientation.m[2][2] * interpolatedValue[2];
                if (update) {
                    *nodePtrX += reorientedValue[0] * static_cast<DataType>(weight);
                    *nodePtrY += reorientedValue[1] * static_cast<DataType>(weight);
                    if (voxelPtrZ != nullptr)
                        *nodePtrZ += reorientedValue[2] * static_cast<DataType>(weight);
                } else {
                    *nodePtrX = reorientedValue[0] * static_cast<DataType>(weight);
                    *nodePtrY = reorientedValue[1] * static_cast<DataType>(weight);
                    if (voxelPtrZ != nullptr)
                        *nodePtrZ = reorientedValue[2] * static_cast<DataType>(weight);
                }
                ++nodePtrX;
                ++nodePtrY;
                if (voxelPtrZ != nullptr)
                    ++nodePtrZ;
            } // loop over
        } // loop over y
    } // loop over z
}
/* *************************************************************** */
void reg_voxelCentricToNodeCentric(nifti_image *nodeImage,
                                   nifti_image *voxelImage,
                                   float weight,
                                   bool update,
                                   const mat44 *voxelToMillimetre) {
    if (nodeImage->datatype != voxelImage->datatype)
        NR_FATAL_ERROR("Both input images are expected to have the same data type");

    switch (nodeImage->datatype) {
    case NIFTI_TYPE_FLOAT32:
        reg_voxelCentricToNodeCentric<float>(nodeImage, voxelImage, weight, update, voxelToMillimetre);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_voxelCentricToNodeCentric<double>(nodeImage, voxelImage, weight, update, voxelToMillimetre);
        break;
    default:
        NR_FATAL_ERROR("Data type not supported");
    }
}
/* *************************************************************** */
template<class SplineTYPE>
SplineTYPE GetValue(SplineTYPE *array, int *dim, int x, int y, int z) {
    if (x < 0 || x >= dim[1] || y < 0 || y >= dim[2] || z < 0 || z >= dim[3])
        return 0;
    return array[(z * dim[2] + y) * dim[1] + x];
}
/* *************************************************************** */
template<class SplineTYPE>
void SetValue(SplineTYPE *array, int *dim, int x, int y, int z, SplineTYPE value) {
    if (x < 0 || x >= dim[1] || y < 0 || y >= dim[2] || z < 0 || z >= dim[3])
        return;
    array[(z * dim[2] + y) * dim[1] + x] = value;
}
/* *************************************************************** */
template<class SplineTYPE>
void reg_spline_refineControlPointGrid2D(nifti_image *splineControlPoint,
                                         nifti_image *referenceImage) {
    // The input grid is first saved
    SplineTYPE *oldGrid = (SplineTYPE*)malloc(splineControlPoint->nvox * splineControlPoint->nbyper);
    SplineTYPE *gridPtrX = static_cast<SplineTYPE*>(splineControlPoint->data);
    memcpy(oldGrid, gridPtrX, splineControlPoint->nvox * splineControlPoint->nbyper);
    if (splineControlPoint->data != nullptr) free(splineControlPoint->data);
    int oldDim[4];
    oldDim[0] = splineControlPoint->dim[0];
    oldDim[1] = splineControlPoint->dim[1];
    oldDim[2] = splineControlPoint->dim[2];
    oldDim[3] = splineControlPoint->dim[3];

    splineControlPoint->dx = splineControlPoint->pixdim[1] = splineControlPoint->dx / 2.0f;
    splineControlPoint->dy = splineControlPoint->pixdim[2] = splineControlPoint->dy / 2.0f;
    splineControlPoint->dz = 1.0f;
    if (referenceImage != nullptr) {
        splineControlPoint->dim[1] = splineControlPoint->nx = Ceil(referenceImage->nx * referenceImage->dx / splineControlPoint->dx + 3.f);
        splineControlPoint->dim[2] = splineControlPoint->ny = Ceil(referenceImage->ny * referenceImage->dy / splineControlPoint->dy + 3.f);
    } else {
        splineControlPoint->dim[1] = splineControlPoint->nx = (oldDim[1] - 3) * 2 + 3;
        splineControlPoint->dim[2] = splineControlPoint->ny = (oldDim[2] - 3) * 2 + 3;
    }
    splineControlPoint->dim[3] = splineControlPoint->nz = 1;

    splineControlPoint->nvox = NiftiImage::calcVoxelNumber(splineControlPoint, splineControlPoint->ndim);
    splineControlPoint->data = calloc(splineControlPoint->nvox, splineControlPoint->nbyper);
    gridPtrX = static_cast<SplineTYPE*>(splineControlPoint->data);
    SplineTYPE *gridPtrY = &gridPtrX[NiftiImage::calcVoxelNumber(splineControlPoint, 2)];
    SplineTYPE *oldGridPtrX = &oldGrid[0];
    SplineTYPE *oldGridPtrY = &oldGridPtrX[oldDim[1] * oldDim[2]];

    for (int y = 0; y < oldDim[2]; y++) {
        int Y = 2 * y - 1;
        if (Y < splineControlPoint->ny) {
            for (int x = 0; x < oldDim[1]; x++) {
                int X = 2 * x - 1;
                if (X < splineControlPoint->nx) {

                    /* X Axis */
                    // 0 0
                    SetValue(gridPtrX, splineControlPoint->dim, X, Y, 0,
                             (GetValue(oldGridPtrX, oldDim, x - 1, y - 1, 0) + GetValue(oldGridPtrX, oldDim, x + 1, y - 1, 0) +
                              GetValue(oldGridPtrX, oldDim, x - 1, y + 1, 0) + GetValue(oldGridPtrX, oldDim, x + 1, y + 1, 0)
                              + 6.0f * (GetValue(oldGridPtrX, oldDim, x - 1, y, 0) + GetValue(oldGridPtrX, oldDim, x + 1, y, 0) +
                                        GetValue(oldGridPtrX, oldDim, x, y - 1, 0) + GetValue(oldGridPtrX, oldDim, x, y + 1, 0))
                              + 36.0f * GetValue(oldGridPtrX, oldDim, x, y, 0)) / 64.0f);
                    // 1 0
                    SetValue(gridPtrX, splineControlPoint->dim, X + 1, Y, 0,
                             (GetValue(oldGridPtrX, oldDim, x, y - 1, 0) + GetValue(oldGridPtrX, oldDim, x + 1, y - 1, 0) +
                              GetValue(oldGridPtrX, oldDim, x, y + 1, 0) + GetValue(oldGridPtrX, oldDim, x + 1, y + 1, 0)
                              + 6.0f * (GetValue(oldGridPtrX, oldDim, x, y, 0) + GetValue(oldGridPtrX, oldDim, x + 1, y, 0))) / 16.0f);
                    // 0 1
                    SetValue(gridPtrX, splineControlPoint->dim, X, Y + 1, 0,
                             (GetValue(oldGridPtrX, oldDim, x - 1, y, 0) + GetValue(oldGridPtrX, oldDim, x - 1, y + 1, 0) +
                              GetValue(oldGridPtrX, oldDim, x + 1, y, 0) + GetValue(oldGridPtrX, oldDim, x + 1, y + 1, 0)
                              + 6.0f * (GetValue(oldGridPtrX, oldDim, x, y, 0) + GetValue(oldGridPtrX, oldDim, x, y + 1, 0))) / 16.0f);
                    // 1 1
                    SetValue(gridPtrX, splineControlPoint->dim, X + 1, Y + 1, 0,
                             (GetValue(oldGridPtrX, oldDim, x, y, 0) + GetValue(oldGridPtrX, oldDim, x + 1, y, 0) +
                              GetValue(oldGridPtrX, oldDim, x, y + 1, 0) + GetValue(oldGridPtrX, oldDim, x + 1, y + 1, 0)) / 4.0f);

                    /* Y Axis */
                    // 0 0
                    SetValue(gridPtrY, splineControlPoint->dim, X, Y, 0,
                             (GetValue(oldGridPtrY, oldDim, x - 1, y - 1, 0) + GetValue(oldGridPtrY, oldDim, x + 1, y - 1, 0) +
                              GetValue(oldGridPtrY, oldDim, x - 1, y + 1, 0) + GetValue(oldGridPtrY, oldDim, x + 1, y + 1, 0)
                              + 6.0f * (GetValue(oldGridPtrY, oldDim, x - 1, y, 0) + GetValue(oldGridPtrY, oldDim, x + 1, y, 0) +
                                        GetValue(oldGridPtrY, oldDim, x, y - 1, 0) + GetValue(oldGridPtrY, oldDim, x, y + 1, 0))
                              + 36.0f * GetValue(oldGridPtrY, oldDim, x, y, 0)) / 64.0f);
                    // 1 0
                    SetValue(gridPtrY, splineControlPoint->dim, X + 1, Y, 0,
                             (GetValue(oldGridPtrY, oldDim, x, y - 1, 0) + GetValue(oldGridPtrY, oldDim, x + 1, y - 1, 0) +
                              GetValue(oldGridPtrY, oldDim, x, y + 1, 0) + GetValue(oldGridPtrY, oldDim, x + 1, y + 1, 0)
                              + 6.0f * (GetValue(oldGridPtrY, oldDim, x, y, 0) + GetValue(oldGridPtrY, oldDim, x + 1, y, 0))) / 16.0f);
                    // 0 1
                    SetValue(gridPtrY, splineControlPoint->dim, X, Y + 1, 0,
                             (GetValue(oldGridPtrY, oldDim, x - 1, y, 0) + GetValue(oldGridPtrY, oldDim, x - 1, y + 1, 0) +
                              GetValue(oldGridPtrY, oldDim, x + 1, y, 0) + GetValue(oldGridPtrY, oldDim, x + 1, y + 1, 0)
                              + 6.0f * (GetValue(oldGridPtrY, oldDim, x, y, 0) + GetValue(oldGridPtrY, oldDim, x, y + 1, 0))) / 16.0f);
                    // 1 1
                    SetValue(gridPtrY, splineControlPoint->dim, X + 1, Y + 1, 0,
                             (GetValue(oldGridPtrY, oldDim, x, y, 0) + GetValue(oldGridPtrY, oldDim, x + 1, y, 0) +
                              GetValue(oldGridPtrY, oldDim, x, y + 1, 0) + GetValue(oldGridPtrY, oldDim, x + 1, y + 1, 0)) / 4.0f);

                }
            }
        }
    }

    free(oldGrid);
}
/* *************************************************************** */
template<class SplineTYPE>
void reg_spline_refineControlPointGrid3D(nifti_image *splineControlPoint, nifti_image *referenceImage) {
    // The input grid is first saved
    SplineTYPE *oldGrid = (SplineTYPE*)malloc(splineControlPoint->nvox * splineControlPoint->nbyper);
    SplineTYPE *gridPtrX = static_cast<SplineTYPE*>(splineControlPoint->data);
    memcpy(oldGrid, gridPtrX, splineControlPoint->nvox * splineControlPoint->nbyper);
    if (splineControlPoint->data != nullptr) free(splineControlPoint->data);
    int oldDim[4];
    oldDim[0] = splineControlPoint->dim[0];
    oldDim[1] = splineControlPoint->dim[1];
    oldDim[2] = splineControlPoint->dim[2];
    oldDim[3] = splineControlPoint->dim[3];

    splineControlPoint->dx = splineControlPoint->pixdim[1] = splineControlPoint->dx / 2.0f;
    splineControlPoint->dy = splineControlPoint->pixdim[2] = splineControlPoint->dy / 2.0f;
    splineControlPoint->dz = splineControlPoint->pixdim[3] = splineControlPoint->dz / 2.0f;

    if (referenceImage != nullptr) {
        splineControlPoint->dim[1] = splineControlPoint->nx = Ceil(referenceImage->nx * referenceImage->dx / splineControlPoint->dx + 3.f);
        splineControlPoint->dim[2] = splineControlPoint->ny = Ceil(referenceImage->ny * referenceImage->dy / splineControlPoint->dy + 3.f);
        splineControlPoint->dim[3] = splineControlPoint->nz = Ceil(referenceImage->nz * referenceImage->dz / splineControlPoint->dz + 3.f);
    } else {
        splineControlPoint->dim[1] = splineControlPoint->nx = (oldDim[1] - 3) * 2 + 3;
        splineControlPoint->dim[2] = splineControlPoint->ny = (oldDim[2] - 3) * 2 + 3;
        splineControlPoint->dim[3] = splineControlPoint->nz = (oldDim[3] - 3) * 2 + 3;
    }
    splineControlPoint->nvox = NiftiImage::calcVoxelNumber(splineControlPoint, splineControlPoint->ndim);
    splineControlPoint->data = calloc(splineControlPoint->nvox, splineControlPoint->nbyper);

    const size_t splineControlPointVoxelNumber = NiftiImage::calcVoxelNumber(splineControlPoint, 3);
    gridPtrX = static_cast<SplineTYPE*>(splineControlPoint->data);
    SplineTYPE *gridPtrY = &gridPtrX[splineControlPointVoxelNumber];
    SplineTYPE *gridPtrZ = &gridPtrY[splineControlPointVoxelNumber];
    SplineTYPE *oldGridPtrX = &oldGrid[0];
    SplineTYPE *oldGridPtrY = &oldGridPtrX[oldDim[1] * oldDim[2] * oldDim[3]];
    SplineTYPE *oldGridPtrZ = &oldGridPtrY[oldDim[1] * oldDim[2] * oldDim[3]];

    for (int z = 0; z < oldDim[3]; z++) {
        int Z = 2 * z - 1;
        if (Z < splineControlPoint->nz) {
            for (int y = 0; y < oldDim[2]; y++) {
                int Y = 2 * y - 1;
                if (Y < splineControlPoint->ny) {
                    for (int x = 0; x < oldDim[1]; x++) {
                        int X = 2 * x - 1;
                        if (X < splineControlPoint->nx) {

                            /* X Axis */
                            // 0 0 0
                            SetValue(gridPtrX, splineControlPoint->dim, X, Y, Z,
                                     (GetValue(oldGridPtrX, oldDim, x - 1, y - 1, z - 1) + GetValue(oldGridPtrX, oldDim, x + 1, y - 1, z - 1) +
                                      GetValue(oldGridPtrX, oldDim, x - 1, y + 1, z - 1) + GetValue(oldGridPtrX, oldDim, x + 1, y + 1, z - 1) +
                                      GetValue(oldGridPtrX, oldDim, x - 1, y - 1, z + 1) + GetValue(oldGridPtrX, oldDim, x + 1, y - 1, z + 1) +
                                      GetValue(oldGridPtrX, oldDim, x - 1, y + 1, z + 1) + GetValue(oldGridPtrX, oldDim, x + 1, y + 1, z + 1)
                                      + 6.0f * (GetValue(oldGridPtrX, oldDim, x - 1, y - 1, z) + GetValue(oldGridPtrX, oldDim, x - 1, y + 1, z) +
                                                GetValue(oldGridPtrX, oldDim, x + 1, y - 1, z) + GetValue(oldGridPtrX, oldDim, x + 1, y + 1, z) +
                                                GetValue(oldGridPtrX, oldDim, x - 1, y, z - 1) + GetValue(oldGridPtrX, oldDim, x - 1, y, z + 1) +
                                                GetValue(oldGridPtrX, oldDim, x + 1, y, z - 1) + GetValue(oldGridPtrX, oldDim, x + 1, y, z + 1) +
                                                GetValue(oldGridPtrX, oldDim, x, y - 1, z - 1) + GetValue(oldGridPtrX, oldDim, x, y - 1, z + 1) +
                                                GetValue(oldGridPtrX, oldDim, x, y + 1, z - 1) + GetValue(oldGridPtrX, oldDim, x, y + 1, z + 1))
                                      + 36.0f * (GetValue(oldGridPtrX, oldDim, x - 1, y, z) + GetValue(oldGridPtrX, oldDim, x + 1, y, z) +
                                                 GetValue(oldGridPtrX, oldDim, x, y - 1, z) + GetValue(oldGridPtrX, oldDim, x, y + 1, z) +
                                                 GetValue(oldGridPtrX, oldDim, x, y, z - 1) + GetValue(oldGridPtrX, oldDim, x, y, z + 1))
                                      + 216.0f * GetValue(oldGridPtrX, oldDim, x, y, z)) / 512.0f);

                            // 1 0 0
                            SetValue(gridPtrX, splineControlPoint->dim, X + 1, Y, Z,
                                     (GetValue(oldGridPtrX, oldDim, x, y - 1, z - 1) + GetValue(oldGridPtrX, oldDim, x, y - 1, z + 1) +
                                      GetValue(oldGridPtrX, oldDim, x, y + 1, z - 1) + GetValue(oldGridPtrX, oldDim, x, y + 1, z + 1) +
                                      GetValue(oldGridPtrX, oldDim, x + 1, y - 1, z - 1) + GetValue(oldGridPtrX, oldDim, x + 1, y - 1, z + 1) +
                                      GetValue(oldGridPtrX, oldDim, x + 1, y + 1, z - 1) + GetValue(oldGridPtrX, oldDim, x + 1, y + 1, z + 1) +
                                      6.0f * (GetValue(oldGridPtrX, oldDim, x, y - 1, z) + GetValue(oldGridPtrX, oldDim, x, y + 1, z) +
                                              GetValue(oldGridPtrX, oldDim, x, y, z - 1) + GetValue(oldGridPtrX, oldDim, x, y, z + 1) +
                                              GetValue(oldGridPtrX, oldDim, x + 1, y - 1, z) + GetValue(oldGridPtrX, oldDim, x + 1, y + 1, z) +
                                              GetValue(oldGridPtrX, oldDim, x + 1, y, z - 1) + GetValue(oldGridPtrX, oldDim, x + 1, y, z + 1)) +
                                      36.0f * (GetValue(oldGridPtrX, oldDim, x, y, z) + GetValue(oldGridPtrX, oldDim, x + 1, y, z))) / 128.0f);

                            // 0 1 0
                            SetValue(gridPtrX, splineControlPoint->dim, X, Y + 1, Z,
                                     (GetValue(oldGridPtrX, oldDim, x - 1, y, z - 1) + GetValue(oldGridPtrX, oldDim, x - 1, y, z + 1) +
                                      GetValue(oldGridPtrX, oldDim, x + 1, y, z - 1) + GetValue(oldGridPtrX, oldDim, x + 1, y, z + 1) +
                                      GetValue(oldGridPtrX, oldDim, x - 1, y + 1, z - 1) + GetValue(oldGridPtrX, oldDim, x - 1, y + 1, z + 1) +
                                      GetValue(oldGridPtrX, oldDim, x + 1, y + 1, z - 1) + GetValue(oldGridPtrX, oldDim, x + 1, y + 1, z + 1) +
                                      6.0f * (GetValue(oldGridPtrX, oldDim, x - 1, y, z) + GetValue(oldGridPtrX, oldDim, x + 1, y, z) +
                                              GetValue(oldGridPtrX, oldDim, x, y, z - 1) + GetValue(oldGridPtrX, oldDim, x, y, z + 1) +
                                              GetValue(oldGridPtrX, oldDim, x - 1, y + 1, z) + GetValue(oldGridPtrX, oldDim, x + 1, y + 1, z) +
                                              GetValue(oldGridPtrX, oldDim, x, y + 1, z - 1) + GetValue(oldGridPtrX, oldDim, x, y + 1, z + 1)) +
                                      36.0f * (GetValue(oldGridPtrX, oldDim, x, y, z) + GetValue(oldGridPtrX, oldDim, x, y + 1, z))) / 128.0f);

                            // 1 1 0
                            SetValue(gridPtrX, splineControlPoint->dim, X + 1, Y + 1, Z,
                                     (GetValue(oldGridPtrX, oldDim, x, y, z - 1) + GetValue(oldGridPtrX, oldDim, x + 1, y, z - 1) +
                                      GetValue(oldGridPtrX, oldDim, x, y + 1, z - 1) + GetValue(oldGridPtrX, oldDim, x + 1, y + 1, z - 1) +
                                      GetValue(oldGridPtrX, oldDim, x, y, z + 1) + GetValue(oldGridPtrX, oldDim, x + 1, y, z + 1) +
                                      GetValue(oldGridPtrX, oldDim, x, y + 1, z + 1) + GetValue(oldGridPtrX, oldDim, x + 1, y + 1, z + 1) +
                                      6.0f * (GetValue(oldGridPtrX, oldDim, x, y, z) + GetValue(oldGridPtrX, oldDim, x + 1, y, z) +
                                              GetValue(oldGridPtrX, oldDim, x, y + 1, z) + GetValue(oldGridPtrX, oldDim, x + 1, y + 1, z))) / 32.0f);

                            // 0 0 1
                            SetValue(gridPtrX, splineControlPoint->dim, X, Y, Z + 1,
                                     (GetValue(oldGridPtrX, oldDim, x - 1, y - 1, z) + GetValue(oldGridPtrX, oldDim, x - 1, y + 1, z) +
                                      GetValue(oldGridPtrX, oldDim, x + 1, y - 1, z) + GetValue(oldGridPtrX, oldDim, x + 1, y + 1, z) +
                                      GetValue(oldGridPtrX, oldDim, x - 1, y - 1, z + 1) + GetValue(oldGridPtrX, oldDim, x - 1, y + 1, z + 1) +
                                      GetValue(oldGridPtrX, oldDim, x + 1, y - 1, z + 1) + GetValue(oldGridPtrX, oldDim, x + 1, y + 1, z + 1) +
                                      6.0f * (GetValue(oldGridPtrX, oldDim, x - 1, y, z) + GetValue(oldGridPtrX, oldDim, x + 1, y, z) +
                                              GetValue(oldGridPtrX, oldDim, x, y - 1, z) + GetValue(oldGridPtrX, oldDim, x, y + 1, z) +
                                              GetValue(oldGridPtrX, oldDim, x - 1, y, z + 1) + GetValue(oldGridPtrX, oldDim, x + 1, y, z + 1) +
                                              GetValue(oldGridPtrX, oldDim, x, y - 1, z + 1) + GetValue(oldGridPtrX, oldDim, x, y + 1, z + 1)) +
                                      36.0f * (GetValue(oldGridPtrX, oldDim, x, y, z) + GetValue(oldGridPtrX, oldDim, x, y, z + 1))) / 128.0f);

                            // 1 0 1
                            SetValue(gridPtrX, splineControlPoint->dim, X + 1, Y, Z + 1,
                                     (GetValue(oldGridPtrX, oldDim, x, y - 1, z) + GetValue(oldGridPtrX, oldDim, x + 1, y - 1, z) +
                                      GetValue(oldGridPtrX, oldDim, x, y - 1, z + 1) + GetValue(oldGridPtrX, oldDim, x + 1, y - 1, z + 1) +
                                      GetValue(oldGridPtrX, oldDim, x, y + 1, z) + GetValue(oldGridPtrX, oldDim, x + 1, y + 1, z) +
                                      GetValue(oldGridPtrX, oldDim, x, y + 1, z + 1) + GetValue(oldGridPtrX, oldDim, x + 1, y + 1, z + 1) +
                                      6.0f * (GetValue(oldGridPtrX, oldDim, x, y, z) + GetValue(oldGridPtrX, oldDim, x + 1, y, z) +
                                              GetValue(oldGridPtrX, oldDim, x, y, z + 1) + GetValue(oldGridPtrX, oldDim, x + 1, y, z + 1))) / 32.0f);

                            // 0 1 1
                            SetValue(gridPtrX, splineControlPoint->dim, X, Y + 1, Z + 1,
                                     (GetValue(oldGridPtrX, oldDim, x - 1, y, z) + GetValue(oldGridPtrX, oldDim, x - 1, y + 1, z) +
                                      GetValue(oldGridPtrX, oldDim, x - 1, y, z + 1) + GetValue(oldGridPtrX, oldDim, x - 1, y + 1, z + 1) +
                                      GetValue(oldGridPtrX, oldDim, x + 1, y, z) + GetValue(oldGridPtrX, oldDim, x + 1, y + 1, z) +
                                      GetValue(oldGridPtrX, oldDim, x + 1, y, z + 1) + GetValue(oldGridPtrX, oldDim, x + 1, y + 1, z + 1) +
                                      6.0f * (GetValue(oldGridPtrX, oldDim, x, y, z) + GetValue(oldGridPtrX, oldDim, x, y + 1, z) +
                                              GetValue(oldGridPtrX, oldDim, x, y, z + 1) + GetValue(oldGridPtrX, oldDim, x, y + 1, z + 1))) / 32.0f);

                            // 1 1 1
                            SetValue(gridPtrX, splineControlPoint->dim, X + 1, Y + 1, Z + 1,
                                     (GetValue(oldGridPtrX, oldDim, x, y, z) + GetValue(oldGridPtrX, oldDim, x + 1, y, z) +
                                      GetValue(oldGridPtrX, oldDim, x, y + 1, z) + GetValue(oldGridPtrX, oldDim, x + 1, y + 1, z) +
                                      GetValue(oldGridPtrX, oldDim, x, y, z + 1) + GetValue(oldGridPtrX, oldDim, x + 1, y, z + 1) +
                                      GetValue(oldGridPtrX, oldDim, x, y + 1, z + 1) + GetValue(oldGridPtrX, oldDim, x + 1, y + 1, z + 1)) / 8.0f);


                            /* Y Axis */
                            // 0 0 0
                            SetValue(gridPtrY, splineControlPoint->dim, X, Y, Z,
                                     (GetValue(oldGridPtrY, oldDim, x - 1, y - 1, z - 1) + GetValue(oldGridPtrY, oldDim, x + 1, y - 1, z - 1) +
                                      GetValue(oldGridPtrY, oldDim, x - 1, y + 1, z - 1) + GetValue(oldGridPtrY, oldDim, x + 1, y + 1, z - 1) +
                                      GetValue(oldGridPtrY, oldDim, x - 1, y - 1, z + 1) + GetValue(oldGridPtrY, oldDim, x + 1, y - 1, z + 1) +
                                      GetValue(oldGridPtrY, oldDim, x - 1, y + 1, z + 1) + GetValue(oldGridPtrY, oldDim, x + 1, y + 1, z + 1)
                                      + 6.0f * (GetValue(oldGridPtrY, oldDim, x - 1, y - 1, z) + GetValue(oldGridPtrY, oldDim, x - 1, y + 1, z) +
                                                GetValue(oldGridPtrY, oldDim, x + 1, y - 1, z) + GetValue(oldGridPtrY, oldDim, x + 1, y + 1, z) +
                                                GetValue(oldGridPtrY, oldDim, x - 1, y, z - 1) + GetValue(oldGridPtrY, oldDim, x - 1, y, z + 1) +
                                                GetValue(oldGridPtrY, oldDim, x + 1, y, z - 1) + GetValue(oldGridPtrY, oldDim, x + 1, y, z + 1) +
                                                GetValue(oldGridPtrY, oldDim, x, y - 1, z - 1) + GetValue(oldGridPtrY, oldDim, x, y - 1, z + 1) +
                                                GetValue(oldGridPtrY, oldDim, x, y + 1, z - 1) + GetValue(oldGridPtrY, oldDim, x, y + 1, z + 1))
                                      + 36.0f * (GetValue(oldGridPtrY, oldDim, x - 1, y, z) + GetValue(oldGridPtrY, oldDim, x + 1, y, z) +
                                                 GetValue(oldGridPtrY, oldDim, x, y - 1, z) + GetValue(oldGridPtrY, oldDim, x, y + 1, z) +
                                                 GetValue(oldGridPtrY, oldDim, x, y, z - 1) + GetValue(oldGridPtrY, oldDim, x, y, z + 1))
                                      + 216.0f * GetValue(oldGridPtrY, oldDim, x, y, z)) / 512.0f);

                            // 1 0 0
                            SetValue(gridPtrY, splineControlPoint->dim, X + 1, Y, Z,
                                     (GetValue(oldGridPtrY, oldDim, x, y - 1, z - 1) + GetValue(oldGridPtrY, oldDim, x, y - 1, z + 1) +
                                      GetValue(oldGridPtrY, oldDim, x, y + 1, z - 1) + GetValue(oldGridPtrY, oldDim, x, y + 1, z + 1) +
                                      GetValue(oldGridPtrY, oldDim, x + 1, y - 1, z - 1) + GetValue(oldGridPtrY, oldDim, x + 1, y - 1, z + 1) +
                                      GetValue(oldGridPtrY, oldDim, x + 1, y + 1, z - 1) + GetValue(oldGridPtrY, oldDim, x + 1, y + 1, z + 1) +
                                      6.0f * (GetValue(oldGridPtrY, oldDim, x, y - 1, z) + GetValue(oldGridPtrY, oldDim, x, y + 1, z) +
                                              GetValue(oldGridPtrY, oldDim, x, y, z - 1) + GetValue(oldGridPtrY, oldDim, x, y, z + 1) +
                                              GetValue(oldGridPtrY, oldDim, x + 1, y - 1, z) + GetValue(oldGridPtrY, oldDim, x + 1, y + 1, z) +
                                              GetValue(oldGridPtrY, oldDim, x + 1, y, z - 1) + GetValue(oldGridPtrY, oldDim, x + 1, y, z + 1)) +
                                      36.0f * (GetValue(oldGridPtrY, oldDim, x, y, z) + GetValue(oldGridPtrY, oldDim, x + 1, y, z))) / 128.0f);

                            // 0 1 0
                            SetValue(gridPtrY, splineControlPoint->dim, X, Y + 1, Z,
                                     (GetValue(oldGridPtrY, oldDim, x - 1, y, z - 1) + GetValue(oldGridPtrY, oldDim, x - 1, y, z + 1) +
                                      GetValue(oldGridPtrY, oldDim, x + 1, y, z - 1) + GetValue(oldGridPtrY, oldDim, x + 1, y, z + 1) +
                                      GetValue(oldGridPtrY, oldDim, x - 1, y + 1, z - 1) + GetValue(oldGridPtrY, oldDim, x - 1, y + 1, z + 1) +
                                      GetValue(oldGridPtrY, oldDim, x + 1, y + 1, z - 1) + GetValue(oldGridPtrY, oldDim, x + 1, y + 1, z + 1) +
                                      6.0f * (GetValue(oldGridPtrY, oldDim, x - 1, y, z) + GetValue(oldGridPtrY, oldDim, x + 1, y, z) +
                                              GetValue(oldGridPtrY, oldDim, x, y, z - 1) + GetValue(oldGridPtrY, oldDim, x, y, z + 1) +
                                              GetValue(oldGridPtrY, oldDim, x - 1, y + 1, z) + GetValue(oldGridPtrY, oldDim, x + 1, y + 1, z) +
                                              GetValue(oldGridPtrY, oldDim, x, y + 1, z - 1) + GetValue(oldGridPtrY, oldDim, x, y + 1, z + 1)) +
                                      36.0f * (GetValue(oldGridPtrY, oldDim, x, y, z) + GetValue(oldGridPtrY, oldDim, x, y + 1, z))) / 128.0f);

                            // 1 1 0
                            SetValue(gridPtrY, splineControlPoint->dim, X + 1, Y + 1, Z,
                                     (GetValue(oldGridPtrY, oldDim, x, y, z - 1) + GetValue(oldGridPtrY, oldDim, x + 1, y, z - 1) +
                                      GetValue(oldGridPtrY, oldDim, x, y + 1, z - 1) + GetValue(oldGridPtrY, oldDim, x + 1, y + 1, z - 1) +
                                      GetValue(oldGridPtrY, oldDim, x, y, z + 1) + GetValue(oldGridPtrY, oldDim, x + 1, y, z + 1) +
                                      GetValue(oldGridPtrY, oldDim, x, y + 1, z + 1) + GetValue(oldGridPtrY, oldDim, x + 1, y + 1, z + 1) +
                                      6.0f * (GetValue(oldGridPtrY, oldDim, x, y, z) + GetValue(oldGridPtrY, oldDim, x + 1, y, z) +
                                              GetValue(oldGridPtrY, oldDim, x, y + 1, z) + GetValue(oldGridPtrY, oldDim, x + 1, y + 1, z))) / 32.0f);

                            // 0 0 1
                            SetValue(gridPtrY, splineControlPoint->dim, X, Y, Z + 1,
                                     (GetValue(oldGridPtrY, oldDim, x - 1, y - 1, z) + GetValue(oldGridPtrY, oldDim, x - 1, y + 1, z) +
                                      GetValue(oldGridPtrY, oldDim, x + 1, y - 1, z) + GetValue(oldGridPtrY, oldDim, x + 1, y + 1, z) +
                                      GetValue(oldGridPtrY, oldDim, x - 1, y - 1, z + 1) + GetValue(oldGridPtrY, oldDim, x - 1, y + 1, z + 1) +
                                      GetValue(oldGridPtrY, oldDim, x + 1, y - 1, z + 1) + GetValue(oldGridPtrY, oldDim, x + 1, y + 1, z + 1) +
                                      6.0f * (GetValue(oldGridPtrY, oldDim, x - 1, y, z) + GetValue(oldGridPtrY, oldDim, x + 1, y, z) +
                                              GetValue(oldGridPtrY, oldDim, x, y - 1, z) + GetValue(oldGridPtrY, oldDim, x, y + 1, z) +
                                              GetValue(oldGridPtrY, oldDim, x - 1, y, z + 1) + GetValue(oldGridPtrY, oldDim, x + 1, y, z + 1) +
                                              GetValue(oldGridPtrY, oldDim, x, y - 1, z + 1) + GetValue(oldGridPtrY, oldDim, x, y + 1, z + 1)) +
                                      36.0f * (GetValue(oldGridPtrY, oldDim, x, y, z) + GetValue(oldGridPtrY, oldDim, x, y, z + 1))) / 128.0f);

                            // 1 0 1
                            SetValue(gridPtrY, splineControlPoint->dim, X + 1, Y, Z + 1,
                                     (GetValue(oldGridPtrY, oldDim, x, y - 1, z) + GetValue(oldGridPtrY, oldDim, x + 1, y - 1, z) +
                                      GetValue(oldGridPtrY, oldDim, x, y - 1, z + 1) + GetValue(oldGridPtrY, oldDim, x + 1, y - 1, z + 1) +
                                      GetValue(oldGridPtrY, oldDim, x, y + 1, z) + GetValue(oldGridPtrY, oldDim, x + 1, y + 1, z) +
                                      GetValue(oldGridPtrY, oldDim, x, y + 1, z + 1) + GetValue(oldGridPtrY, oldDim, x + 1, y + 1, z + 1) +
                                      6.0f * (GetValue(oldGridPtrY, oldDim, x, y, z) + GetValue(oldGridPtrY, oldDim, x + 1, y, z) +
                                              GetValue(oldGridPtrY, oldDim, x, y, z + 1) + GetValue(oldGridPtrY, oldDim, x + 1, y, z + 1))) / 32.0f);

                            // 0 1 1
                            SetValue(gridPtrY, splineControlPoint->dim, X, Y + 1, Z + 1,
                                     (GetValue(oldGridPtrY, oldDim, x - 1, y, z) + GetValue(oldGridPtrY, oldDim, x - 1, y + 1, z) +
                                      GetValue(oldGridPtrY, oldDim, x - 1, y, z + 1) + GetValue(oldGridPtrY, oldDim, x - 1, y + 1, z + 1) +
                                      GetValue(oldGridPtrY, oldDim, x + 1, y, z) + GetValue(oldGridPtrY, oldDim, x + 1, y + 1, z) +
                                      GetValue(oldGridPtrY, oldDim, x + 1, y, z + 1) + GetValue(oldGridPtrY, oldDim, x + 1, y + 1, z + 1) +
                                      6.0f * (GetValue(oldGridPtrY, oldDim, x, y, z) + GetValue(oldGridPtrY, oldDim, x, y + 1, z) +
                                              GetValue(oldGridPtrY, oldDim, x, y, z + 1) + GetValue(oldGridPtrY, oldDim, x, y + 1, z + 1))) / 32.0f);

                            // 1 1 1
                            SetValue(gridPtrY, splineControlPoint->dim, X + 1, Y + 1, Z + 1,
                                     (GetValue(oldGridPtrY, oldDim, x, y, z) + GetValue(oldGridPtrY, oldDim, x + 1, y, z) +
                                      GetValue(oldGridPtrY, oldDim, x, y + 1, z) + GetValue(oldGridPtrY, oldDim, x + 1, y + 1, z) +
                                      GetValue(oldGridPtrY, oldDim, x, y, z + 1) + GetValue(oldGridPtrY, oldDim, x + 1, y, z + 1) +
                                      GetValue(oldGridPtrY, oldDim, x, y + 1, z + 1) + GetValue(oldGridPtrY, oldDim, x + 1, y + 1, z + 1)) / 8.0f);

                            /* Z Axis */
                            // 0 0 0
                            SetValue(gridPtrZ, splineControlPoint->dim, X, Y, Z,
                                     (GetValue(oldGridPtrZ, oldDim, x - 1, y - 1, z - 1) + GetValue(oldGridPtrZ, oldDim, x + 1, y - 1, z - 1) +
                                      GetValue(oldGridPtrZ, oldDim, x - 1, y + 1, z - 1) + GetValue(oldGridPtrZ, oldDim, x + 1, y + 1, z - 1) +
                                      GetValue(oldGridPtrZ, oldDim, x - 1, y - 1, z + 1) + GetValue(oldGridPtrZ, oldDim, x + 1, y - 1, z + 1) +
                                      GetValue(oldGridPtrZ, oldDim, x - 1, y + 1, z + 1) + GetValue(oldGridPtrZ, oldDim, x + 1, y + 1, z + 1)
                                      + 6.0f * (GetValue(oldGridPtrZ, oldDim, x - 1, y - 1, z) + GetValue(oldGridPtrZ, oldDim, x - 1, y + 1, z) +
                                                GetValue(oldGridPtrZ, oldDim, x + 1, y - 1, z) + GetValue(oldGridPtrZ, oldDim, x + 1, y + 1, z) +
                                                GetValue(oldGridPtrZ, oldDim, x - 1, y, z - 1) + GetValue(oldGridPtrZ, oldDim, x - 1, y, z + 1) +
                                                GetValue(oldGridPtrZ, oldDim, x + 1, y, z - 1) + GetValue(oldGridPtrZ, oldDim, x + 1, y, z + 1) +
                                                GetValue(oldGridPtrZ, oldDim, x, y - 1, z - 1) + GetValue(oldGridPtrZ, oldDim, x, y - 1, z + 1) +
                                                GetValue(oldGridPtrZ, oldDim, x, y + 1, z - 1) + GetValue(oldGridPtrZ, oldDim, x, y + 1, z + 1))
                                      + 36.0f * (GetValue(oldGridPtrZ, oldDim, x - 1, y, z) + GetValue(oldGridPtrZ, oldDim, x + 1, y, z) +
                                                 GetValue(oldGridPtrZ, oldDim, x, y - 1, z) + GetValue(oldGridPtrZ, oldDim, x, y + 1, z) +
                                                 GetValue(oldGridPtrZ, oldDim, x, y, z - 1) + GetValue(oldGridPtrZ, oldDim, x, y, z + 1))
                                      + 216.0f * GetValue(oldGridPtrZ, oldDim, x, y, z)) / 512.0f);

                            // 1 0 0
                            SetValue(gridPtrZ, splineControlPoint->dim, X + 1, Y, Z,
                                     (GetValue(oldGridPtrZ, oldDim, x, y - 1, z - 1) + GetValue(oldGridPtrZ, oldDim, x, y - 1, z + 1) +
                                      GetValue(oldGridPtrZ, oldDim, x, y + 1, z - 1) + GetValue(oldGridPtrZ, oldDim, x, y + 1, z + 1) +
                                      GetValue(oldGridPtrZ, oldDim, x + 1, y - 1, z - 1) + GetValue(oldGridPtrZ, oldDim, x + 1, y - 1, z + 1) +
                                      GetValue(oldGridPtrZ, oldDim, x + 1, y + 1, z - 1) + GetValue(oldGridPtrZ, oldDim, x + 1, y + 1, z + 1) +
                                      6.0f * (GetValue(oldGridPtrZ, oldDim, x, y - 1, z) + GetValue(oldGridPtrZ, oldDim, x, y + 1, z) +
                                              GetValue(oldGridPtrZ, oldDim, x, y, z - 1) + GetValue(oldGridPtrZ, oldDim, x, y, z + 1) +
                                              GetValue(oldGridPtrZ, oldDim, x + 1, y - 1, z) + GetValue(oldGridPtrZ, oldDim, x + 1, y + 1, z) +
                                              GetValue(oldGridPtrZ, oldDim, x + 1, y, z - 1) + GetValue(oldGridPtrZ, oldDim, x + 1, y, z + 1)) +
                                      36.0f * (GetValue(oldGridPtrZ, oldDim, x, y, z) + GetValue(oldGridPtrZ, oldDim, x + 1, y, z))) / 128.0f);

                            // 0 1 0
                            SetValue(gridPtrZ, splineControlPoint->dim, X, Y + 1, Z,
                                     (GetValue(oldGridPtrZ, oldDim, x - 1, y, z - 1) + GetValue(oldGridPtrZ, oldDim, x - 1, y, z + 1) +
                                      GetValue(oldGridPtrZ, oldDim, x + 1, y, z - 1) + GetValue(oldGridPtrZ, oldDim, x + 1, y, z + 1) +
                                      GetValue(oldGridPtrZ, oldDim, x - 1, y + 1, z - 1) + GetValue(oldGridPtrZ, oldDim, x - 1, y + 1, z + 1) +
                                      GetValue(oldGridPtrZ, oldDim, x + 1, y + 1, z - 1) + GetValue(oldGridPtrZ, oldDim, x + 1, y + 1, z + 1) +
                                      6.0f * (GetValue(oldGridPtrZ, oldDim, x - 1, y, z) + GetValue(oldGridPtrZ, oldDim, x + 1, y, z) +
                                              GetValue(oldGridPtrZ, oldDim, x, y, z - 1) + GetValue(oldGridPtrZ, oldDim, x, y, z + 1) +
                                              GetValue(oldGridPtrZ, oldDim, x - 1, y + 1, z) + GetValue(oldGridPtrZ, oldDim, x + 1, y + 1, z) +
                                              GetValue(oldGridPtrZ, oldDim, x, y + 1, z - 1) + GetValue(oldGridPtrZ, oldDim, x, y + 1, z + 1)) +
                                      36.0f * (GetValue(oldGridPtrZ, oldDim, x, y, z) + GetValue(oldGridPtrZ, oldDim, x, y + 1, z))) / 128.0f);

                            // 1 1 0
                            SetValue(gridPtrZ, splineControlPoint->dim, X + 1, Y + 1, Z,
                                     (GetValue(oldGridPtrZ, oldDim, x, y, z - 1) + GetValue(oldGridPtrZ, oldDim, x + 1, y, z - 1) +
                                      GetValue(oldGridPtrZ, oldDim, x, y + 1, z - 1) + GetValue(oldGridPtrZ, oldDim, x + 1, y + 1, z - 1) +
                                      GetValue(oldGridPtrZ, oldDim, x, y, z + 1) + GetValue(oldGridPtrZ, oldDim, x + 1, y, z + 1) +
                                      GetValue(oldGridPtrZ, oldDim, x, y + 1, z + 1) + GetValue(oldGridPtrZ, oldDim, x + 1, y + 1, z + 1) +
                                      6.0f * (GetValue(oldGridPtrZ, oldDim, x, y, z) + GetValue(oldGridPtrZ, oldDim, x + 1, y, z) +
                                              GetValue(oldGridPtrZ, oldDim, x, y + 1, z) + GetValue(oldGridPtrZ, oldDim, x + 1, y + 1, z))) / 32.0f);

                            // 0 0 1
                            SetValue(gridPtrZ, splineControlPoint->dim, X, Y, Z + 1,
                                     (GetValue(oldGridPtrZ, oldDim, x - 1, y - 1, z) + GetValue(oldGridPtrZ, oldDim, x - 1, y + 1, z) +
                                      GetValue(oldGridPtrZ, oldDim, x + 1, y - 1, z) + GetValue(oldGridPtrZ, oldDim, x + 1, y + 1, z) +
                                      GetValue(oldGridPtrZ, oldDim, x - 1, y - 1, z + 1) + GetValue(oldGridPtrZ, oldDim, x - 1, y + 1, z + 1) +
                                      GetValue(oldGridPtrZ, oldDim, x + 1, y - 1, z + 1) + GetValue(oldGridPtrZ, oldDim, x + 1, y + 1, z + 1) +
                                      6.0f * (GetValue(oldGridPtrZ, oldDim, x - 1, y, z) + GetValue(oldGridPtrZ, oldDim, x + 1, y, z) +
                                              GetValue(oldGridPtrZ, oldDim, x, y - 1, z) + GetValue(oldGridPtrZ, oldDim, x, y + 1, z) +
                                              GetValue(oldGridPtrZ, oldDim, x - 1, y, z + 1) + GetValue(oldGridPtrZ, oldDim, x + 1, y, z + 1) +
                                              GetValue(oldGridPtrZ, oldDim, x, y - 1, z + 1) + GetValue(oldGridPtrZ, oldDim, x, y + 1, z + 1)) +
                                      36.0f * (GetValue(oldGridPtrZ, oldDim, x, y, z) + GetValue(oldGridPtrZ, oldDim, x, y, z + 1))) / 128.0f);

                            // 1 0 1
                            SetValue(gridPtrZ, splineControlPoint->dim, X + 1, Y, Z + 1,
                                     (GetValue(oldGridPtrZ, oldDim, x, y - 1, z) + GetValue(oldGridPtrZ, oldDim, x + 1, y - 1, z) +
                                      GetValue(oldGridPtrZ, oldDim, x, y - 1, z + 1) + GetValue(oldGridPtrZ, oldDim, x + 1, y - 1, z + 1) +
                                      GetValue(oldGridPtrZ, oldDim, x, y + 1, z) + GetValue(oldGridPtrZ, oldDim, x + 1, y + 1, z) +
                                      GetValue(oldGridPtrZ, oldDim, x, y + 1, z + 1) + GetValue(oldGridPtrZ, oldDim, x + 1, y + 1, z + 1) +
                                      6.0f * (GetValue(oldGridPtrZ, oldDim, x, y, z) + GetValue(oldGridPtrZ, oldDim, x + 1, y, z) +
                                              GetValue(oldGridPtrZ, oldDim, x, y, z + 1) + GetValue(oldGridPtrZ, oldDim, x + 1, y, z + 1))) / 32.0f);

                            // 0 1 1
                            SetValue(gridPtrZ, splineControlPoint->dim, X, Y + 1, Z + 1,
                                     (GetValue(oldGridPtrZ, oldDim, x - 1, y, z) + GetValue(oldGridPtrZ, oldDim, x - 1, y + 1, z) +
                                      GetValue(oldGridPtrZ, oldDim, x - 1, y, z + 1) + GetValue(oldGridPtrZ, oldDim, x - 1, y + 1, z + 1) +
                                      GetValue(oldGridPtrZ, oldDim, x + 1, y, z) + GetValue(oldGridPtrZ, oldDim, x + 1, y + 1, z) +
                                      GetValue(oldGridPtrZ, oldDim, x + 1, y, z + 1) + GetValue(oldGridPtrZ, oldDim, x + 1, y + 1, z + 1) +
                                      6.0f * (GetValue(oldGridPtrZ, oldDim, x, y, z) + GetValue(oldGridPtrZ, oldDim, x, y + 1, z) +
                                              GetValue(oldGridPtrZ, oldDim, x, y, z + 1) + GetValue(oldGridPtrZ, oldDim, x, y + 1, z + 1))) / 32.0f);

                            // 1 1 1
                            SetValue(gridPtrZ, splineControlPoint->dim, X + 1, Y + 1, Z + 1,
                                     (GetValue(oldGridPtrZ, oldDim, x, y, z) + GetValue(oldGridPtrZ, oldDim, x + 1, y, z) +
                                      GetValue(oldGridPtrZ, oldDim, x, y + 1, z) + GetValue(oldGridPtrZ, oldDim, x + 1, y + 1, z) +
                                      GetValue(oldGridPtrZ, oldDim, x, y, z + 1) + GetValue(oldGridPtrZ, oldDim, x + 1, y, z + 1) +
                                      GetValue(oldGridPtrZ, oldDim, x, y + 1, z + 1) + GetValue(oldGridPtrZ, oldDim, x + 1, y + 1, z + 1)) / 8.0f);
                        }
                    }
                }
            }
        }
    }
    free(oldGrid);
}
/* *************************************************************** */
void reg_spline_refineControlPointGrid(nifti_image *controlPointGrid,
                                       nifti_image *referenceImage) {
    NR_DEBUG("Starting the refine the control point grid");
    if (controlPointGrid->nz == 1) {
        switch (controlPointGrid->datatype) {
        case NIFTI_TYPE_FLOAT32:
            reg_spline_refineControlPointGrid2D<float>(controlPointGrid, referenceImage);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_spline_refineControlPointGrid2D<double>(controlPointGrid, referenceImage);
            break;
        default:
            NR_FATAL_ERROR("Only single or double precision is implemented for the bending energy gradient");
        }
    } else {
        switch (controlPointGrid->datatype) {
        case NIFTI_TYPE_FLOAT32:
            reg_spline_refineControlPointGrid3D<float>(controlPointGrid, referenceImage);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_spline_refineControlPointGrid3D<double>(controlPointGrid, referenceImage);
            break;
        default:
            NR_FATAL_ERROR("Only single or double precision is implemented for the bending energy gradient");
        }
    }
    if (referenceImage != nullptr) {
        // Compute the new control point header
        // The qform (and sform) are set for the control point position image
        controlPointGrid->quatern_b = referenceImage->quatern_b;
        controlPointGrid->quatern_c = referenceImage->quatern_c;
        controlPointGrid->quatern_d = referenceImage->quatern_d;
        controlPointGrid->qoffset_x = referenceImage->qoffset_x;
        controlPointGrid->qoffset_y = referenceImage->qoffset_y;
        controlPointGrid->qoffset_z = referenceImage->qoffset_z;
        controlPointGrid->qfac = referenceImage->qfac;
        controlPointGrid->qto_xyz = nifti_quatern_to_mat44(controlPointGrid->quatern_b,
                                                           controlPointGrid->quatern_c,
                                                           controlPointGrid->quatern_d,
                                                           controlPointGrid->qoffset_x,
                                                           controlPointGrid->qoffset_y,
                                                           controlPointGrid->qoffset_z,
                                                           controlPointGrid->dx,
                                                           controlPointGrid->dy,
                                                           controlPointGrid->dz,
                                                           controlPointGrid->qfac);

        // Origin is shifted from 1 control point in the qform
        float originIndex[3];
        float originReal[3];
        originIndex[0] = -1.0f;
        originIndex[1] = -1.0f;
        originIndex[2] = 0.0f;
        if (referenceImage->nz > 1) originIndex[2] = -1.0f;
        reg_mat44_mul(&(controlPointGrid->qto_xyz), originIndex, originReal);
        if (controlPointGrid->qform_code == 0 && controlPointGrid->sform_code == 0)
            controlPointGrid->qform_code = 1;
        controlPointGrid->qto_xyz.m[0][3] = controlPointGrid->qoffset_x = originReal[0];
        controlPointGrid->qto_xyz.m[1][3] = controlPointGrid->qoffset_y = originReal[1];
        controlPointGrid->qto_xyz.m[2][3] = controlPointGrid->qoffset_z = originReal[2];

        controlPointGrid->qto_ijk = nifti_mat44_inverse(controlPointGrid->qto_xyz);

        if (controlPointGrid->sform_code > 0) {
            float scalingRatio[3];
            scalingRatio[0] = controlPointGrid->dx / referenceImage->dx;
            scalingRatio[1] = controlPointGrid->dy / referenceImage->dy;
            scalingRatio[2] = 1.f;
            if (controlPointGrid->nz > 1)
                scalingRatio[2] = controlPointGrid->dz / referenceImage->dz;

            controlPointGrid->sto_xyz.m[0][0] = referenceImage->sto_xyz.m[0][0] * scalingRatio[0];
            controlPointGrid->sto_xyz.m[1][0] = referenceImage->sto_xyz.m[1][0] * scalingRatio[0];
            controlPointGrid->sto_xyz.m[2][0] = referenceImage->sto_xyz.m[2][0] * scalingRatio[0];
            controlPointGrid->sto_xyz.m[3][0] = 0.f;
            controlPointGrid->sto_xyz.m[0][1] = referenceImage->sto_xyz.m[0][1] * scalingRatio[1];
            controlPointGrid->sto_xyz.m[1][1] = referenceImage->sto_xyz.m[1][1] * scalingRatio[1];
            controlPointGrid->sto_xyz.m[2][1] = referenceImage->sto_xyz.m[2][1] * scalingRatio[1];
            controlPointGrid->sto_xyz.m[3][1] = 0.f;
            controlPointGrid->sto_xyz.m[0][2] = referenceImage->sto_xyz.m[0][2] * scalingRatio[2];
            controlPointGrid->sto_xyz.m[1][2] = referenceImage->sto_xyz.m[1][2] * scalingRatio[2];
            controlPointGrid->sto_xyz.m[2][2] = referenceImage->sto_xyz.m[2][2] * scalingRatio[2];
            controlPointGrid->sto_xyz.m[3][2] = 0.f;
            controlPointGrid->sto_xyz.m[0][3] = referenceImage->sto_xyz.m[0][3];
            controlPointGrid->sto_xyz.m[1][3] = referenceImage->sto_xyz.m[1][3];
            controlPointGrid->sto_xyz.m[2][3] = referenceImage->sto_xyz.m[2][3];
            controlPointGrid->sto_xyz.m[3][3] = 1.f;

            // The origin is shifted by one compare to the reference image
            float originIndex[3];
            originIndex[0] = originIndex[1] = originIndex[2] = -1;
            if (referenceImage->nz <= 1) originIndex[2] = 0;
            reg_mat44_mul(&(controlPointGrid->sto_xyz), originIndex, originReal);
            controlPointGrid->sto_xyz.m[0][3] = originReal[0];
            controlPointGrid->sto_xyz.m[1][3] = originReal[1];
            controlPointGrid->sto_xyz.m[2][3] = originReal[2];
            controlPointGrid->sto_ijk = nifti_mat44_inverse(controlPointGrid->sto_xyz);
        }
    } else {
        // The voxel spacing is reduced by two
        for (unsigned i = 0; i < 3; ++i) {
            controlPointGrid->sto_xyz.m[0][i] /= 2.f;
            controlPointGrid->sto_xyz.m[1][i] /= 2.f;
            if (controlPointGrid->nz > 1)
                controlPointGrid->sto_xyz.m[2][i] /= 2.f;
        }
        // The origin is shifted by one node when compared to the previous origin
        float nodeCoord[3] = { 1, 1, 1 };
        float newOrigin[3];
        reg_mat44_mul(&controlPointGrid->sto_xyz, nodeCoord, newOrigin);
        controlPointGrid->sto_xyz.m[0][3] = newOrigin[0];
        controlPointGrid->sto_xyz.m[1][3] = newOrigin[1];
        if (controlPointGrid->nz > 1)
            controlPointGrid->sto_xyz.m[2][3] = newOrigin[2];
        controlPointGrid->sto_ijk = nifti_mat44_inverse(controlPointGrid->sto_xyz);
    }
    NR_DEBUG("The control point grid has been refined");
}
/* *************************************************************** */
template <class DataType>
void reg_defField_compose2D(const nifti_image *deformationField,
                            nifti_image *dfToUpdate,
                            const int *mask) {
    const size_t dfVoxelNumber = NiftiImage::calcVoxelNumber(deformationField, 2);
#ifdef _WIN32
    long i;
    const long warVoxelNumber = (long)NiftiImage::calcVoxelNumber(dfToUpdate, 2);
#else
    size_t i;
    const size_t warVoxelNumber = NiftiImage::calcVoxelNumber(dfToUpdate, 2);
#endif
    const DataType *defPtrX = static_cast<DataType*>(deformationField->data);
    const DataType *defPtrY = &defPtrX[dfVoxelNumber];

    DataType *resPtrX = static_cast<DataType*>(dfToUpdate->data);
    DataType *resPtrY = &resPtrX[warVoxelNumber];

    const mat44 *df_real2Voxel;
    const mat44 *df_voxel2Real;
    if (deformationField->sform_code > 0) {
        df_real2Voxel = &dfToUpdate->sto_ijk;
        df_voxel2Real = &deformationField->sto_xyz;
    } else {
        df_real2Voxel = &dfToUpdate->qto_ijk;
        df_voxel2Real = &deformationField->qto_xyz;
    }

    size_t index;
    int a, b, pre[2];
    DataType realDefX, realDefY, voxelX, voxelY;
    DataType defX, defY, relX[2], relY[2], basis;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(warVoxelNumber, mask, df_real2Voxel, df_voxel2Real, \
   deformationField, defPtrX, defPtrY, resPtrX, resPtrY) \
   private(a, b, index, pre,realDefX, realDefY, voxelX, voxelY, \
   defX, defY, relX, relY, basis)
#endif
    for (i = 0; i < warVoxelNumber; ++i) {
        if (mask[i] > -1) {
            realDefX = resPtrX[i];
            realDefY = resPtrY[i];

            // Conversion from real to voxel in the deformation field
            voxelX =
                realDefX * df_real2Voxel->m[0][0] +
                realDefY * df_real2Voxel->m[0][1] +
                df_real2Voxel->m[0][3];
            voxelY =
                realDefX * df_real2Voxel->m[1][0] +
                realDefY * df_real2Voxel->m[1][1] +
                df_real2Voxel->m[1][3];

            // Linear interpolation to compute the new deformation
            pre[0] = Floor(voxelX);
            pre[1] = Floor(voxelY);
            relX[1] = voxelX - static_cast<DataType>(pre[0]);
            relX[0] = 1.f - relX[1];
            relY[1] = voxelY - static_cast<DataType>(pre[1]);
            relY[0] = 1.f - relY[1];
            realDefX = realDefY = 0;
            for (b = 0; b < 2; ++b) {
                for (a = 0; a < 2; ++a) {
                    basis = relX[a] * relY[b];
                    if (pre[0] + a > -1 && pre[0] + a < deformationField->nx &&
                        pre[1] + b > -1 && pre[1] + b < deformationField->ny) {
                        // Uses the deformation field if voxel is in its space
                        index = (pre[1] + b) * deformationField->nx + pre[0] + a;
                        defX = defPtrX[index];
                        defY = defPtrY[index];
                    } else {
                        // Uses a sliding effect
                        get_SlidedValues<DataType>(defX,
                                                   defY,
                                                   pre[0] + a,
                                                   pre[1] + b,
                                                   defPtrX,
                                                   defPtrY,
                                                   df_voxel2Real,
                                                   deformationField->dim,
                                                   false); // not a deformation field
                    }
                    realDefX += defX * basis;
                    realDefY += defY * basis;
                }
            }
            resPtrX[i] = realDefX;
            resPtrY[i] = realDefY;
        }// mask
    }// loop over every voxel
}
/* *************************************************************** */
template <class DataType>
void reg_defField_compose3D(const nifti_image *deformationField,
                            nifti_image *dfToUpdate,
                            const int *mask) {
    const size_t dfVoxelNumber = NiftiImage::calcVoxelNumber(deformationField, 3);
#ifdef _WIN32
    long i;
    const long warVoxelNumber = (long)NiftiImage::calcVoxelNumber(dfToUpdate, 3);
#else
    size_t i;
    const size_t warVoxelNumber = NiftiImage::calcVoxelNumber(dfToUpdate, 3);
#endif
    const DataType *defPtrX = static_cast<DataType*>(deformationField->data);
    const DataType *defPtrY = &defPtrX[dfVoxelNumber];
    const DataType *defPtrZ = &defPtrY[dfVoxelNumber];

    DataType *resPtrX = static_cast<DataType*>(dfToUpdate->data);
    DataType *resPtrY = &resPtrX[warVoxelNumber];
    DataType *resPtrZ = &resPtrY[warVoxelNumber];

#ifdef _WIN32
    __declspec(align(16))mat44 df_real2Voxel;
#else
    mat44 df_real2Voxel __attribute__((aligned(16)));
#endif
    const mat44 *df_voxel2Real;
    if (deformationField->sform_code > 0) {
        df_real2Voxel = deformationField->sto_ijk;
        df_voxel2Real = &deformationField->sto_xyz;
    } else {
        df_real2Voxel = deformationField->qto_ijk;
        df_voxel2Real = &deformationField->qto_xyz;
    }

    size_t tempIndex, index;
    int a, b, c, currentX, currentY, currentZ, pre[3];
    DataType realDef[3], voxel[3], basis, tempBasis;
    DataType defX, defY, defZ, relX[2], relY[2], relZ[2];
    bool inY, inZ;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(warVoxelNumber, mask, df_real2Voxel, df_voxel2Real, \
   defPtrX, defPtrY, defPtrZ, resPtrX, resPtrY, resPtrZ, deformationField) \
   private(a, b, c, currentX, currentY, currentZ, index, tempIndex, pre, \
   realDef, voxel, tempBasis, defX, defY, defZ, relX, relY, relZ, basis, inY, inZ)
#endif
    for (i = 0; i < warVoxelNumber; ++i) {
        if (mask[i] > -1) {
            // Conversion from real to voxel in the deformation field
            realDef[0] = resPtrX[i];
            realDef[1] = resPtrY[i];
            realDef[2] = resPtrZ[i];
            voxel[0] =
                df_real2Voxel.m[0][0] * realDef[0] +
                df_real2Voxel.m[0][1] * realDef[1] +
                df_real2Voxel.m[0][2] * realDef[2] +
                df_real2Voxel.m[0][3];
            voxel[1] =
                df_real2Voxel.m[1][0] * realDef[0] +
                df_real2Voxel.m[1][1] * realDef[1] +
                df_real2Voxel.m[1][2] * realDef[2] +
                df_real2Voxel.m[1][3];
            voxel[2] =
                df_real2Voxel.m[2][0] * realDef[0] +
                df_real2Voxel.m[2][1] * realDef[1] +
                df_real2Voxel.m[2][2] * realDef[2] +
                df_real2Voxel.m[2][3];
            //reg_mat44_mul(df_real2Voxel, realDef, voxel);

            // Linear interpolation to compute the new deformation
            pre[0] = Floor(voxel[0]);
            pre[1] = Floor(voxel[1]);
            pre[2] = Floor(voxel[2]);
            relX[1] = voxel[0] - static_cast<DataType>(pre[0]);
            relX[0] = 1.f - relX[1];
            relY[1] = voxel[1] - static_cast<DataType>(pre[1]);
            relY[0] = 1.f - relY[1];
            relZ[1] = voxel[2] - static_cast<DataType>(pre[2]);
            relZ[0] = 1.f - relZ[1];
            realDef[0] = realDef[1] = realDef[2] = 0;
            for (c = 0; c < 2; ++c) {
                currentZ = pre[2] + c;
                tempIndex = currentZ * deformationField->nx * deformationField->ny;
                if (currentZ > -1 && currentZ < deformationField->nz) inZ = true;
                else inZ = false;
                for (b = 0; b < 2; ++b) {
                    currentY = pre[1] + b;
                    index = tempIndex + currentY * deformationField->nx + pre[0];
                    tempBasis = relY[b] * relZ[c];
                    if (currentY > -1 && currentY < deformationField->ny) inY = true;
                    else inY = false;
                    for (a = 0; a < 2; ++a) {
                        currentX = pre[0] + a;
                        if (currentX > -1 && currentX < deformationField->nx && inY && inZ) {
                            // Uses the deformation field if voxel is in its space
                            defX = defPtrX[index];
                            defY = defPtrY[index];
                            defZ = defPtrZ[index];
                        } else {
                            // Uses a sliding effect
                            get_SlidedValues<DataType>(defX,
                                                       defY,
                                                       defZ,
                                                       currentX,
                                                       currentY,
                                                       currentZ,
                                                       defPtrX,
                                                       defPtrY,
                                                       defPtrZ,
                                                       df_voxel2Real,
                                                       deformationField->dim,
                                                       false); // not a displacement field
                        }
                        ++index;
                        basis = relX[a] * tempBasis;
                        realDef[0] += defX * basis;
                        realDef[1] += defY * basis;
                        realDef[2] += defZ * basis;
                    } // a loop
                } // b loop
            } // c loop
            resPtrX[i] = realDef[0];
            resPtrY[i] = realDef[1];
            resPtrZ[i] = realDef[2];
        }// mask
    }// loop over every voxel
}
/* *************************************************************** */
void reg_defField_compose(const nifti_image *deformationField,
                          nifti_image *dfToUpdate,
                          const int *mask) {
    if (deformationField->datatype != dfToUpdate->datatype)
        NR_FATAL_ERROR("Both deformation fields are expected to have the same type");

    unique_ptr<int[]> currentMask;
    if (!mask) {
        currentMask.reset(new int[NiftiImage::calcVoxelNumber(dfToUpdate, 3)]());
        mask = currentMask.get();
    }

    std::visit([&](auto&& defFieldDataType) {
        using DefFieldDataType = std::decay_t<decltype(defFieldDataType)>;
        auto defFieldCompose = dfToUpdate->nu == 2 ? reg_defField_compose2D<DefFieldDataType> : reg_defField_compose3D<DefFieldDataType>;
        defFieldCompose(deformationField, dfToUpdate, mask);
    }, NiftiImage::getFloatingDataType(deformationField));
}
/* *************************************************************** */
/// @brief Internal data structure to pass user data into optimizer that get passed to cost_function
struct ddata {
    nifti_image *deformationField;
    double gx, gy, gz;
    double *arrayy[4];
    double values[4];
};

/* ************************************************************************** */
/* internal routine : deform one point(x, y, x) according to deformationField */
/* returns ERROR when the input point falls outside the deformation field     */
/* ************************************************************************** */

template<class FieldTYPE>
inline static int FastWarp(double x, double y, double z, nifti_image *deformationField, double *px, double *py, double *pz) {
    double wax, wbx, wcx, wdx, wex, wfx, wgx, whx, wf3x;
    FieldTYPE *wpx;
    double way, wby, wcy, wdy, wey, wfy, wgy, why, wf3y;
    FieldTYPE *wpy;
    double waz, wbz, wcz, wdz, wez, wfz, wgz, whz, wf3z;
    FieldTYPE *wpz;
    int   xw, yw, zw, dxw, dyw, dxyw, dxyzw;
    double wxf, wyf, wzf, wyzf;
    double world[4], position[4];

    FieldTYPE *warpdata = static_cast<FieldTYPE*>(deformationField->data);

    const mat44 *deformationFieldIJKMatrix;
    if (deformationField->sform_code > 0)
        deformationFieldIJKMatrix = &deformationField->sto_ijk;
    else deformationFieldIJKMatrix = &deformationField->qto_ijk;

    dxw = deformationField->nx;
    dyw = deformationField->ny;
    dxyw = dxw * dyw;
    dxyzw = dxw * dyw * deformationField->nz;

    // first guess
    *px = x;
    *py = y;
    *pz = z;

    // detect NAN input
    if (x != x || y != y || z != z) return EXIT_FAILURE;

    // convert x, y,z to indices in deformationField
    world[0] = x;
    world[1] = y;
    world[2] = z;
    world[3] = 1;
    reg_mat44_mul(deformationFieldIJKMatrix, world, position);
    x = position[0];
    y = position[1];
    z = position[2];

    xw = (int)x;        /* get indices into DVF */
    yw = (int)y;
    zw = (int)z;

    // if you block out the next three lines the routine will extrapolate indefinitively
#if 0
    if (x < 0 || x >= deformationField->nx - 1) return ERROR;
    if (y < 0 || y >= deformationField->ny - 1) return ERROR;
    if (z < 0 || z >= deformationField->nz - 1) return ERROR;
#else
    if (xw < 0) xw = 0;     /* clip */
    if (yw < 0) yw = 0;
    if (zw < 0) zw = 0;
    if (xw > deformationField->nx - 2) xw = deformationField->nx - 2;
    if (yw > deformationField->ny - 2) yw = deformationField->ny - 2;
    if (zw > deformationField->nz - 2) zw = deformationField->nz - 2;
#endif

    wxf = x - xw;                  /* fractional coordinates */
    wyf = y - yw;
    wzf = z - zw;

    /* cornerstone for warp coordinates */
    wpx = warpdata + zw * dxyw + yw * dxw + xw;
    wpy = wpx + dxyzw;
    wpz = wpy + dxyzw;

    wf3x = wpx[dxw + 1];
    wax = wpx[0];
    wbx = wpx[1] - wax;
    wcx = wpx[dxw] - wax;
    wdx = wpx[dxyw] - wax;
    wex = wpx[dxyw + dxw] - wax - wcx - wdx;
    wfx = wpx[dxyw + 1] - wax - wbx - wdx;
    wgx = wf3x - wax - wbx - wcx;
    whx = wpx[dxyw + dxw + 1] - wf3x - wdx - wex - wfx;

    wf3y = wpy[dxw + 1];
    way = wpy[0];
    wby = wpy[1] - way;
    wcy = wpy[dxw] - way;
    wdy = wpy[dxyw] - way;
    wey = wpy[dxyw + dxw] - way - wcy - wdy;
    wfy = wpy[dxyw + 1] - way - wby - wdy;
    wgy = wf3y - way - wby - wcy;
    why = wpy[dxyw + dxw + 1] - wf3y - wdy - wey - wfy;

    wf3z = wpz[dxw + 1];
    waz = wpz[0];
    wbz = wpz[1] - waz;
    wcz = wpz[dxw] - waz;
    wdz = wpz[dxyw] - waz;
    wez = wpz[dxyw + dxw] - waz - wcz - wdz;
    wfz = wpz[dxyw + 1] - waz - wbz - wdz;
    wgz = wf3z - waz - wbz - wcz;
    whz = wpz[dxyw + dxw + 1] - wf3z - wdz - wez - wfz;

    wyzf = wyf * wzf;                   /* common term in interpolation     */

    /* trilinear interpolation formulae  */
    *px = wax + wbx * wxf + wcx * wyf + wdx * wzf + wex * wyzf + wfx * wxf * wzf + wgx * wxf * wyf + whx * wxf * wyzf;
    *py = way + wby * wxf + wcy * wyf + wdy * wzf + wey * wyzf + wfy * wxf * wzf + wgy * wxf * wyf + why * wxf * wyzf;
    *pz = waz + wbz * wxf + wcz * wyf + wdz * wzf + wez * wyzf + wfz * wxf * wzf + wgz * wxf * wyf + whz * wxf * wyzf;

    return EXIT_SUCCESS;
}

/* Internal square distance cost function; supports NIFTI_TYPE_FLOAT32 and NIFTI_TYPE_FLOAT64 */
static double cost_function(const double *vector, const void *data) {
    struct ddata *dat = (struct ddata*)data;
    double x, y, z;
    if (dat->deformationField->datatype == NIFTI_TYPE_FLOAT64)
        FastWarp<double>(vector[0], vector[1], vector[2], dat->deformationField, &x, &y, &z);
    else
        FastWarp<float>(vector[0], vector[1], vector[2], dat->deformationField, &x, &y, &z);

    return (x - dat->gx) * (x - dat->gx) + (y - dat->gy) * (y - dat->gy) + (z - dat->gz) * (z - dat->gz);
}

/* multimin/simplex.c
 *
 * Copyright (C) 2002 Tuomo Keskitalo, Ivo Alxneit
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

 /*
    - Originally written by Tuomo Keskitalo <tuomo.keskitalo@iki.fi>
    - Corrections to nmsimplex_iterate and other functions
      by Ivo Alxneit <ivo.alxneit@psi.ch>
    - Additional help by Brian Gough <bjg@network-theory.co.uk>

    Modified version by mvh to make it work standalone of GSL
 */

 /* The Simplex method of Nelder and Mead,
    also known as the polytope search alogorithm. Ref:
    Nelder, J.A., Mead, R., Computer Journal 7 (1965) pp. 308-313.

    This implementation uses 4 corner points in the simplex for a 3D search.
 */

typedef struct {
    double x1[12];              /* simplex corner points nsimplex*nvec */
    double y1[4];               /* function value at corner points */
    double ws1[3];              /* workspace 1 for algorithm */
    double ws2[3];              /* workspace 2 for algorithm */
    int    nvec;
    int    nsimplex;
}
nmsimplex_state_t;

typedef double gsl_multimin_function(const double*, const void*);

static double
nmsimplex_move_corner(const double coeff, nmsimplex_state_t *state,
                      size_t corner, double *xc,
                      gsl_multimin_function *f, void *fdata) {
    /* moves a simplex corner scaled by coeff (negative value represents
     mirroring by the middle point of the "other" corner points)
     and gives new corner in xc and function value at xc as a
     return value
    */

    double *x1 = state->x1;

    size_t i, j;
    double newval, mp;

    for (j = 0; j < (size_t)state->nvec; j++) {
        mp = 0;
        for (i = 0; i < (size_t)state->nsimplex; i++) {
            if (i != corner) {
                mp += x1[i * state->nvec + j];
            }
        }
        mp /= (double)(state->nsimplex - 1);
        newval = mp - coeff * (mp - x1[corner * state->nvec + j]);
        xc[j] = newval;
    }

    newval = f(xc, fdata);

    return newval;
}

static void
nmsimplex_contract_by_best(nmsimplex_state_t *state, size_t best,
                           double *xc, gsl_multimin_function *f, void *fdata) {

    /* Function contracts the simplex in respect to
     best valued corner. That is, all corners besides the
     best corner are moved. */

     /* the xc vector is simply work space here */

    double *x1 = state->x1;
    double *y1 = state->y1;

    size_t i, j;
    double newval;

    for (i = 0; i < (size_t)state->nsimplex; i++) {
        if (i != best) {
            for (j = 0; j < (size_t)state->nvec; j++) {
                newval = 0.5 * (x1[i * state->nvec + j] + x1[best * state->nvec + j]);
                x1[i * state->nvec + j] = newval;
            }

            /* evaluate function in the new point */

            xc = x1 + i * state->nvec;
            newval = f(xc, fdata);
            y1[i] = newval;
        }
    }
}

static void
nmsimplex_calc_center(const nmsimplex_state_t *state, double *mp) {
    /* calculates the center of the simplex to mp */

    const double *x1 = state->x1;

    size_t i, j;
    double val;

    for (j = 0; j < (size_t)state->nvec; j++) {
        val = 0;
        for (i = 0; i < (size_t)state->nsimplex; i++) {
            val += x1[i * state->nvec + j];
        }
        val /= state->nsimplex;
        mp[j] = val;
    }
}

static double
nmsimplex_size(nmsimplex_state_t *state) {
    /* calculates simplex size as average sum of length of vectors
     from simplex center to corner points:

     ( sum ( || y - y_middlepoint || ) ) / n
    */

    double *s = state->ws1;
    double *mp = state->ws2;
    double *x1 = state->x1;

    size_t i, j;

    double t, ss = 0;

    /* Calculate middle point */
    nmsimplex_calc_center(state, mp);

    for (i = 0; i < (size_t)state->nsimplex; i++) {
        for (j = 0; j < (size_t)state->nvec; j++) s[j] = x1[i * state->nvec + j] - mp[j];
        t = 0;
        for (j = 0; j < (size_t)state->nvec; j++) t += s[j] * s[j];
        ss += sqrt(t);
    }

    return ss / (double)(state->nsimplex);
}

static void
nmsimplex_set(void *vstate, gsl_multimin_function *f,
              const double *x,
              double *size, const double *step_size, void *fdata) {
    size_t i, j;
    double val;

    nmsimplex_state_t *state = (nmsimplex_state_t*)vstate;

    double *xtemp = state->ws1;

    /* first point is the original x0 */

    val = f(x, fdata);
    for (j = 0; j < (size_t)state->nvec; j++) state->x1[j] = x[j];
    state->y1[0] = val;

    /* following points are initialized to x0 + step_size */

    for (i = 0; i < (size_t)state->nvec; i++) {
        for (j = 0; j < (size_t)state->nvec; j++) xtemp[j] = x[j];

        val = xtemp[i] + step_size[i];
        xtemp[i] = val;
        val = f(xtemp, fdata);
        for (j = 0; j < (size_t)state->nvec; j++)
            state->x1[(i + 1) * state->nvec + j] = xtemp[j];
        state->y1[i + 1] = val;
    }

    /* Initialize simplex size */

    *size = nmsimplex_size(state);
}

static void
nmsimplex_iterate(void *vstate, gsl_multimin_function *f,
                  double *x, double *size, double *fval, void *fdata) {

    /* Simplex iteration tries to minimize function f value */
    /* Includes corrections from Ivo Alxneit <ivo.alxneit@psi.ch> */

    nmsimplex_state_t *state = (nmsimplex_state_t*)vstate;

    /* xc and xc2 vectors store tried corner point coordinates */

    double *xc = state->ws1;
    double *xc2 = state->ws2;
    double *y1 = state->y1;
    double *x1 = state->x1;

    size_t n = state->nsimplex;
    size_t i, j;
    size_t hi = 0, s_hi = 0, lo = 0;
    double dhi, ds_hi, dlo;
    double val, val2;

    /* get index of highest, second highest and lowest point */

    dhi = ds_hi = dlo = y1[0];

    for (i = 1; i < n; i++) {
        val = y1[i];
        if (val < dlo) {
            dlo = val;
            lo = i;
        } else if (val > dhi) {
            ds_hi = dhi;
            s_hi = hi;
            dhi = val;
            hi = i;
        } else if (val > ds_hi) {
            ds_hi = val;
            s_hi = i;
        }
    }

    /* reflect the highest value */

    val = nmsimplex_move_corner(-1.0, state, hi, xc, f, fdata);

    if (val < y1[lo]) {

        /* reflected point becomes lowest point, try expansion */

        val2 = nmsimplex_move_corner(-2.0, state, hi, xc2, f, fdata);

        if (val2 < y1[lo]) {
            for (j = 0; j < (size_t)state->nvec; j++) x1[hi * state->nvec + j] = xc2[j];
            y1[hi] = val2;
        } else {
            for (j = 0; j < (size_t)state->nvec; j++) x1[hi * state->nvec + j] = xc[j];
            y1[hi] = val;
        }
    }

    /* reflection does not improve things enough */

    else if (val > y1[s_hi]) {
        if (val <= y1[hi]) {

            /* if trial point is better than highest point, replace
             highest point */

            for (j = 0; j < (size_t)state->nvec; j++) x1[hi * state->nvec + j] = xc[j];
            y1[hi] = val;
        }

        /* try one dimensional contraction */

        val2 = nmsimplex_move_corner(0.5, state, hi, xc2, f, fdata);

        if (val2 <= y1[hi]) {
            for (j = 0; j < (size_t)state->nvec; j++) x1[hi * state->nvec + j] = xc2[j];
            y1[hi] = val2;
        }

        else {
            /* contract the whole simplex in respect to the best point */
            nmsimplex_contract_by_best(state, lo, xc, f, fdata);
        }
    } else {

        /* trial point is better than second highest point.
         Replace highest point by it */

        for (j = 0; j < (size_t)state->nvec; j++) x1[hi * state->nvec + j] = xc[j];
        y1[hi] = val;
    }

    /* return lowest point of simplex as x */

    lo = 0;
    val = y1[0];
    for (j = 1; j < (size_t)state->nsimplex; j++) if (y1[j] < val) lo = j, val = y1[j];
    for (j = 0; j < (size_t)state->nvec; j++) x[j] = x1[lo * state->nvec + j];
    *fval = y1[lo];


    /* Update simplex size */

    *size = nmsimplex_size(state);
}

/* Internal wrapper for nmsimplex_iterate */
static void optimize(gsl_multimin_function *f, double *start, void *data, double tol) {
    nmsimplex_state_t t;
    double fval[4];
    double offset[3] = { 10, 10, 10 };
    double size;
    int n = 0;
    t.nvec = 3;
    t.nsimplex = 4;
    nmsimplex_set(&t, f, start, &size, offset, data);
    while (size > tol && n < 300) {
        nmsimplex_iterate(&t, f, start, &size, fval, data);
        n++;
    }
    nmsimplex_calc_center(&t, start);
}
/* *************************************************************** */
template <class DataType>
void reg_defFieldInvert3D(nifti_image *inputDeformationField,
                          nifti_image *outputDeformationField,
                          float tolerance) {
    const size_t outputVoxelNumber = NiftiImage::calcVoxelNumber(outputDeformationField, 3);

    const mat44 *OutXYZMatrix;
    if (outputDeformationField->sform_code > 0)
        OutXYZMatrix = &outputDeformationField->sto_xyz;
    else OutXYZMatrix = &outputDeformationField->qto_xyz;

    const mat44 *InXYZMatrix;
    if (inputDeformationField->sform_code > 0)
        InXYZMatrix = &inputDeformationField->sto_xyz;
    else InXYZMatrix = &inputDeformationField->qto_xyz;
    float center[4], center2[4];
    double centerout[4], delta[4];
    center[0] = static_cast<float>(inputDeformationField->nx / 2);
    center[1] = static_cast<float>(inputDeformationField->ny / 2);
    center[2] = static_cast<float>(inputDeformationField->nz / 2);
    center[3] = 1;
    reg_mat44_mul(InXYZMatrix, center, center2);
    FastWarp<float>(center2[0], center2[1], center2[2], inputDeformationField, &centerout[0], &centerout[1], &centerout[2]);
    delta[0] = center2[0] - centerout[0];
    delta[1] = center2[1] - centerout[1];
    delta[2] = center2[2] - centerout[2];
    // end added


    int i, x, y, z;
    double position[4], pars[4], arrayy[4][3];
    struct ddata dat;
    DataType *outData;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(outputDeformationField,tolerance,outputVoxelNumber, \
   inputDeformationField, OutXYZMatrix, delta) \
   private(i, x, y, dat, outData, position, pars, arrayy)
#endif
    for (z = 0; z < outputDeformationField->nz; ++z) {
        dat.deformationField = inputDeformationField;
        for (i = 0; i < 4; ++i)              /* set up 2D array pointers */
            dat.arrayy[i] = arrayy[i];

        outData = (DataType*)(outputDeformationField->data) +
            outputDeformationField->nx * outputDeformationField->ny * z;

        for (y = 0; y < outputDeformationField->ny; ++y) {
            for (x = 0; x < outputDeformationField->nx; ++x) {

                // convert x, y,z to world coordinates
                position[0] = x;
                position[1] = y;
                position[2] = z;
                position[3] = 1;
                reg_mat44_mul(OutXYZMatrix, position, pars);
                dat.gx = pars[0];
                dat.gy = pars[1];
                dat.gz = pars[2];

                // added
                pars[0] += delta[0];
                pars[1] += delta[1];
                pars[2] += delta[2];
                // end added

                optimize(cost_function, pars, &dat, tolerance);
                // output = (warp-1)(input);

                outData[0] = static_cast<DataType>(pars[0]);
                outData[outputVoxelNumber] = static_cast<DataType>(pars[1]);
                outData[outputVoxelNumber * 2] = static_cast<DataType>(pars[2]);
                ++outData;
            }
        }
    }
}
/* *************************************************************** */
void reg_defFieldInvert(nifti_image *inputDeformationField,
                        nifti_image *outputDeformationField,
                        float tolerance) {
    // Check the input image data types
    if (inputDeformationField->datatype != outputDeformationField->datatype)
        NR_FATAL_ERROR("Both deformation fields are expected to have the same data type");

    if (inputDeformationField->nu != 3)
        NR_FATAL_ERROR("The function has only been implemented for 3D deformation field yet");

    switch (inputDeformationField->datatype) {
    case NIFTI_TYPE_FLOAT32:
        reg_defFieldInvert3D<float>
            (inputDeformationField, outputDeformationField, tolerance);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_defFieldInvert3D<double>
            (inputDeformationField, outputDeformationField, tolerance);
    default:
        NR_FATAL_ERROR("Deformation field pixel type is unsupported");
    }
}
/* *************************************************************** */
// TODO: HAVE TO BE CHECKED
template<class DataType>
void reg_spline_cppComposition_2D(nifti_image *grid1,
                                  nifti_image *grid2,
                                  bool displacement1,
                                  bool displacement2,
                                  bool bspline) {
    // REMINDER Grid2(x)=Grid1(Grid2(x))

#if USE_SSE
    union {
        __m128 m;
        float f[4];
    } val;
#endif // USE_SSE

    DataType *outCPPPtrX = static_cast<DataType*>(grid2->data);
    DataType *outCPPPtrY = &outCPPPtrX[NiftiImage::calcVoxelNumber(grid2, 2)];

    DataType *controlPointPtrX = static_cast<DataType*>(grid1->data);
    DataType *controlPointPtrY = &controlPointPtrX[NiftiImage::calcVoxelNumber(grid1, 2)];

    DataType basis;

#ifdef _WIN32
    __declspec(align(16)) DataType xBasis[4];
    __declspec(align(16)) DataType yBasis[4];
#if USE_SSE
    __declspec(align(16)) DataType xyBasis[16];
#endif  //USE_SSE

    __declspec(align(16)) DataType xControlPointCoordinates[16];
    __declspec(align(16)) DataType yControlPointCoordinates[16];
#else // _WIN32
    DataType xBasis[4] __attribute__((aligned(16)));
    DataType yBasis[4] __attribute__((aligned(16)));
#if USE_SSE
    DataType xyBasis[16] __attribute__((aligned(16)));
#endif  //USE_SSE

    DataType xControlPointCoordinates[16] __attribute__((aligned(16)));
    DataType yControlPointCoordinates[16] __attribute__((aligned(16)));
#endif // _WIN32

    size_t coord;

    // read the xyz/ijk sform or qform, as appropriate
    const mat44 *matrix_real_to_voxel1, *matrix_voxel_to_real2;
    if (grid1->sform_code > 0)
        matrix_real_to_voxel1 = &grid1->sto_ijk;
    else matrix_real_to_voxel1 = &grid1->qto_ijk;
    if (grid2->sform_code > 0)
        matrix_voxel_to_real2 = &grid2->sto_xyz;
    else matrix_voxel_to_real2 = &grid2->qto_xyz;

    for (int y = 0; y < grid2->ny; y++) {
        for (int x = 0; x < grid2->nx; x++) {
            // Get the control point actual position
            DataType xReal = *outCPPPtrX;
            DataType yReal = *outCPPPtrY;
            DataType initialX = xReal;
            DataType initialY = yReal;
            if (displacement2) {
                xReal +=
                    matrix_voxel_to_real2->m[0][0] * x
                    + matrix_voxel_to_real2->m[0][1] * y
                    + matrix_voxel_to_real2->m[0][3];
                yReal +=
                    matrix_voxel_to_real2->m[1][0] * x
                    + matrix_voxel_to_real2->m[1][1] * y
                    + matrix_voxel_to_real2->m[1][3];
            }

            // Get the voxel based control point position in grid1
            DataType xVoxel = matrix_real_to_voxel1->m[0][0] * xReal
                + matrix_real_to_voxel1->m[0][1] * yReal
                + matrix_real_to_voxel1->m[0][3];
            DataType yVoxel = matrix_real_to_voxel1->m[1][0] * xReal
                + matrix_real_to_voxel1->m[1][1] * yReal
                + matrix_real_to_voxel1->m[1][3];

            // The spline coefficients are computed
            int xPre = Floor(xVoxel);
            basis = xVoxel - static_cast<DataType>(xPre--);
            if (basis < 0) basis = 0; //rounding error
            if (bspline) get_BSplineBasisValues<DataType>(basis, xBasis);
            else get_SplineBasisValues<DataType>(basis, xBasis);

            int yPre = Floor(yVoxel);
            basis = yVoxel - static_cast<DataType>(yPre--);
            if (basis < 0) basis = 0; //rounding error
            if (bspline) get_BSplineBasisValues<DataType>(basis, yBasis);
            else get_SplineBasisValues<DataType>(basis, yBasis);

            // The control points are stored
            get_GridValues<DataType>(xPre,
                                     yPre,
                                     grid1,
                                     controlPointPtrX,
                                     controlPointPtrY,
                                     xControlPointCoordinates,
                                     yControlPointCoordinates,
                                     false, // no approximation
                                     displacement1); // displacement field?
            xReal = 0;
            yReal = 0;
#if USE_SSE
            coord = 0;
            for (unsigned b = 0; b < 4; b++) {
                for (unsigned a = 0; a < 4; a++) {
                    xyBasis[coord++] = xBasis[a] * yBasis[b];
                }
            }

            __m128 tempX = _mm_set_ps1(0);
            __m128 tempY = _mm_set_ps1(0);
            __m128 *ptrX = (__m128*)&xControlPointCoordinates[0];
            __m128 *ptrY = (__m128*)&yControlPointCoordinates[0];
            __m128 *ptrBasis = (__m128*)&xyBasis[0];
            //addition and multiplication of the 16 basis value and CP position for each axis
            for (unsigned a = 0; a < 4; a++) {
                tempX = _mm_add_ps(_mm_mul_ps(*ptrBasis, *ptrX++), tempX);
                tempY = _mm_add_ps(_mm_mul_ps(*ptrBasis, *ptrY++), tempY);
                ptrBasis++;
            }
            //the values stored in SSE variables are transferred to normal float
            val.m = tempX;
            xReal = val.f[0] + val.f[1] + val.f[2] + val.f[3];
            val.m = tempY;
            yReal = val.f[0] + val.f[1] + val.f[2] + val.f[3];
#else
            coord = 0;
            for (unsigned b = 0; b < 4; b++) {
                for (unsigned a = 0; a < 4; a++) {
                    DataType tempValue = xBasis[a] * yBasis[b];
                    xReal += xControlPointCoordinates[coord] * tempValue;
                    yReal += yControlPointCoordinates[coord] * tempValue;
                    coord++;
                }
            }
#endif
            if (displacement1) {
                xReal += initialX;
                yReal += initialY;
            }
            *outCPPPtrX++ = xReal;
            *outCPPPtrY++ = yReal;
        }
    }
}
/* *************************************************************** */
//HAVE TO BE CHECKED
template<class DataType>
void reg_spline_cppComposition_3D(nifti_image *grid1,
                                  nifti_image *grid2,
                                  bool displacement1,
                                  bool displacement2,
                                  bool bspline) {
    // REMINDER Grid2(x)=Grid1(Grid2(x))
#if USE_SSE
    union {
        __m128 m;
        float f[4];
    } val;
    __m128 _xBasis_sse;
    __m128 tempX;
    __m128 tempY;
    __m128 tempZ;
    __m128 *ptrX;
    __m128 *ptrY;
    __m128 *ptrZ;
    __m128 _yBasis_sse;
    __m128 _zBasis_sse;
    __m128 _temp_basis;
    __m128 _basis;
#else
    int a, b, c;
    size_t coord;
    DataType tempValue;
#endif

    const size_t grid2VoxelNumber = NiftiImage::calcVoxelNumber(grid2, 3);
    DataType *outCPPPtrX = static_cast<DataType*>(grid2->data);
    DataType *outCPPPtrY = &outCPPPtrX[grid2VoxelNumber];
    DataType *outCPPPtrZ = &outCPPPtrY[grid2VoxelNumber];

    const size_t grid1VoxelNumber = NiftiImage::calcVoxelNumber(grid1, 3);
    DataType *controlPointPtrX = static_cast<DataType*>(grid1->data);
    DataType *controlPointPtrY = &controlPointPtrX[grid1VoxelNumber];
    DataType *controlPointPtrZ = &controlPointPtrY[grid1VoxelNumber];

    DataType basis;

#ifdef _WIN32
    __declspec(align(16)) DataType xBasis[4];
    __declspec(align(16)) DataType yBasis[4];
    __declspec(align(16)) DataType zBasis[4];
    __declspec(align(16)) DataType xControlPointCoordinates[64];
    __declspec(align(16)) DataType yControlPointCoordinates[64];
    __declspec(align(16)) DataType zControlPointCoordinates[64];
#else
    DataType xBasis[4] __attribute__((aligned(16)));
    DataType yBasis[4] __attribute__((aligned(16)));
    DataType zBasis[4] __attribute__((aligned(16)));
    DataType xControlPointCoordinates[64] __attribute__((aligned(16)));
    DataType yControlPointCoordinates[64] __attribute__((aligned(16)));
    DataType zControlPointCoordinates[64] __attribute__((aligned(16)));
#endif

    int xPre, xPreOld, yPre, yPreOld, zPre, zPreOld;
    int x, y, z;
    size_t index;
    DataType xReal, yReal, zReal, initialPositionX, initialPositionY, initialPositionZ;
    DataType xVoxel, yVoxel, zVoxel;

    // read the xyz/ijk sform or qform, as appropriate
    const mat44 *matrix_real_to_voxel1, *matrix_voxel_to_real2;
    if (grid1->sform_code > 0)
        matrix_real_to_voxel1 = &grid1->sto_ijk;
    else matrix_real_to_voxel1 = &grid1->qto_ijk;
    if (grid2->sform_code > 0)
        matrix_voxel_to_real2 = &grid2->sto_xyz;
    else matrix_voxel_to_real2 = &grid2->qto_xyz;

#ifdef _OPENMP
#ifdef USE_SSE
#pragma omp parallel for default(none) \
   shared(grid1, grid2, displacement1, displacement2, matrix_voxel_to_real2, matrix_real_to_voxel1, \
   outCPPPtrX, outCPPPtrY, outCPPPtrZ, controlPointPtrX, controlPointPtrY, controlPointPtrZ, bspline) \
   private(xPre, xPreOld, yPre, yPreOld, zPre, zPreOld, val, index, \
   x, y, xVoxel, yVoxel, zVoxel, basis, xBasis, yBasis, zBasis, \
   xReal, yReal, zReal, initialPositionX, initialPositionY, initialPositionZ, \
   _xBasis_sse, tempX, tempY, tempZ, ptrX, ptrY, ptrZ, _yBasis_sse, _zBasis_sse, _temp_basis, _basis, \
   xControlPointCoordinates, yControlPointCoordinates, zControlPointCoordinates)
#else
#pragma omp parallel for default(none) \
   shared(grid1, grid2, displacement1, displacement2, matrix_voxel_to_real2, matrix_real_to_voxel1, \
   outCPPPtrX, outCPPPtrY, outCPPPtrZ, controlPointPtrX, controlPointPtrY, controlPointPtrZ, bspline) \
   private(xPre, xPreOld, yPre, yPreOld, zPre, zPreOld, index, \
   x, y, xVoxel, yVoxel, zVoxel, a, b, c, coord, basis, tempValue, xBasis, yBasis, zBasis, \
   xReal, yReal, zReal, initialPositionX, initialPositionY, initialPositionZ, \
   xControlPointCoordinates, yControlPointCoordinates, zControlPointCoordinates)
#endif
#endif
    for (z = 0; z < grid2->nz; z++) {
        xPreOld = 99999;
        yPreOld = 99999;
        zPreOld = 99999;
        index = z * grid2->nx * grid2->ny;
        for (y = 0; y < grid2->ny; y++) {
            for (x = 0; x < grid2->nx; x++) {
                // Get the control point actual position
                xReal = outCPPPtrX[index];
                yReal = outCPPPtrY[index];
                zReal = outCPPPtrZ[index];
                initialPositionX = 0;
                initialPositionY = 0;
                initialPositionZ = 0;
                if (displacement2) {
                    xReal += initialPositionX =
                        matrix_voxel_to_real2->m[0][0] * x
                        + matrix_voxel_to_real2->m[0][1] * y
                        + matrix_voxel_to_real2->m[0][2] * z
                        + matrix_voxel_to_real2->m[0][3];
                    yReal += initialPositionY =
                        matrix_voxel_to_real2->m[1][0] * x
                        + matrix_voxel_to_real2->m[1][1] * y
                        + matrix_voxel_to_real2->m[1][2] * z
                        + matrix_voxel_to_real2->m[1][3];
                    zReal += initialPositionZ =
                        matrix_voxel_to_real2->m[2][0] * x
                        + matrix_voxel_to_real2->m[2][1] * y
                        + matrix_voxel_to_real2->m[2][2] * z
                        + matrix_voxel_to_real2->m[2][3];
                }

                // Get the voxel based control point position in grid1
                xVoxel =
                    matrix_real_to_voxel1->m[0][0] * xReal
                    + matrix_real_to_voxel1->m[0][1] * yReal
                    + matrix_real_to_voxel1->m[0][2] * zReal
                    + matrix_real_to_voxel1->m[0][3];
                yVoxel =
                    matrix_real_to_voxel1->m[1][0] * xReal
                    + matrix_real_to_voxel1->m[1][1] * yReal
                    + matrix_real_to_voxel1->m[1][2] * zReal
                    + matrix_real_to_voxel1->m[1][3];
                zVoxel =
                    matrix_real_to_voxel1->m[2][0] * xReal
                    + matrix_real_to_voxel1->m[2][1] * yReal
                    + matrix_real_to_voxel1->m[2][2] * zReal
                    + matrix_real_to_voxel1->m[2][3];

                // The spline coefficients are computed
                xPre = Floor(xVoxel);
                basis = xVoxel - static_cast<DataType>(xPre--);
                if (basis < 0) basis = 0; //rounding error
                if (bspline) get_BSplineBasisValues<DataType>(basis, xBasis);
                else get_SplineBasisValues<DataType>(basis, xBasis);

                yPre = Floor(yVoxel);
                basis = yVoxel - static_cast<DataType>(yPre--);
                if (basis < 0) basis = 0; //rounding error
                if (bspline) get_BSplineBasisValues<DataType>(basis, yBasis);
                else get_SplineBasisValues<DataType>(basis, yBasis);

                zPre = Floor(zVoxel);
                basis = zVoxel - static_cast<DataType>(zPre--);
                if (basis < 0) basis = 0; //rounding error
                if (bspline) get_BSplineBasisValues<DataType>(basis, zBasis);
                else get_SplineBasisValues<DataType>(basis, zBasis);

                // The control points are stored
                if (xPre != xPreOld || yPre != yPreOld || zPre != zPreOld) {
                    get_GridValues(xPre,
                                   yPre,
                                   zPre,
                                   grid1,
                                   controlPointPtrX,
                                   controlPointPtrY,
                                   controlPointPtrZ,
                                   xControlPointCoordinates,
                                   yControlPointCoordinates,
                                   zControlPointCoordinates,
                                   false, // no approximation
                                   displacement1); // a displacement field?
                    xPreOld = xPre;
                    yPreOld = yPre;
                    zPreOld = zPre;
                }
                xReal = 0;
                yReal = 0;
                zReal = 0;
#if USE_SSE
                val.f[0] = static_cast<float>(xBasis[0]);
                val.f[1] = static_cast<float>(xBasis[1]);
                val.f[2] = static_cast<float>(xBasis[2]);
                val.f[3] = static_cast<float>(xBasis[3]);
                _xBasis_sse = val.m;

                tempX = _mm_set_ps1(0);
                tempY = _mm_set_ps1(0);
                tempZ = _mm_set_ps1(0);
                ptrX = (__m128*)&xControlPointCoordinates[0];
                ptrY = (__m128*)&yControlPointCoordinates[0];
                ptrZ = (__m128*)&zControlPointCoordinates[0];

                for (unsigned c = 0; c < 4; c++) {
                    for (unsigned b = 0; b < 4; b++) {
                        _yBasis_sse = _mm_set_ps1(static_cast<float>(yBasis[b]));
                        _zBasis_sse = _mm_set_ps1(static_cast<float>(zBasis[c]));
                        _temp_basis = _mm_mul_ps(_yBasis_sse, _zBasis_sse);
                        _basis = _mm_mul_ps(_temp_basis, _xBasis_sse);
                        tempX = _mm_add_ps(_mm_mul_ps(_basis, *ptrX++), tempX);
                        tempY = _mm_add_ps(_mm_mul_ps(_basis, *ptrY++), tempY);
                        tempZ = _mm_add_ps(_mm_mul_ps(_basis, *ptrZ++), tempZ);
                    }
                }
                //the values stored in SSE variables are transferred to normal float
                val.m = tempX;
                xReal = val.f[0] + val.f[1] + val.f[2] + val.f[3];
                val.m = tempY;
                yReal = val.f[0] + val.f[1] + val.f[2] + val.f[3];
                val.m = tempZ;
                zReal = val.f[0] + val.f[1] + val.f[2] + val.f[3];
#else
                coord = 0;
                for (c = 0; c < 4; c++) {
                    for (b = 0; b < 4; b++) {
                        for (a = 0; a < 4; a++) {
                            tempValue = xBasis[a] * yBasis[b] * zBasis[c];
                            xReal += xControlPointCoordinates[coord] * tempValue;
                            yReal += yControlPointCoordinates[coord] * tempValue;
                            zReal += zControlPointCoordinates[coord] * tempValue;
                            coord++;
                        }
                    }
                }
#endif
                if (displacement2) {
                    xReal -= initialPositionX;
                    yReal -= initialPositionY;
                    zReal -= initialPositionZ;
                }
                outCPPPtrX[index] = xReal;
                outCPPPtrY[index] = yReal;
                outCPPPtrZ[index] = zReal;
                index++;
            }
        }
    }
}
/* *************************************************************** */
int reg_spline_cppComposition(nifti_image *grid1,
                              nifti_image *grid2,
                              bool displacement1,
                              bool displacement2,
                              bool bspline) {
    // REMINDER Grid2(x)=Grid1(Grid2(x))

    if (grid1->datatype != grid2->datatype)
        NR_FATAL_ERROR("Both input images are expected to have the same data type");

#if USE_SSE
    if (grid1->datatype != NIFTI_TYPE_FLOAT32)
        NR_FATAL_ERROR("SSE computation has only been implemented for single precision");
#endif

    if (grid1->nz > 1) {
        switch (grid1->datatype) {
        case NIFTI_TYPE_FLOAT32:
            reg_spline_cppComposition_3D<float>(grid1, grid2, displacement1, displacement2, bspline);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_spline_cppComposition_3D<double>(grid1, grid2, displacement1, displacement2, bspline);
            break;
        default:
            NR_FATAL_ERROR("Only implemented for single or double floating images");
        }
    } else {
        switch (grid1->datatype) {
        case NIFTI_TYPE_FLOAT32:
            reg_spline_cppComposition_2D<float>(grid1, grid2, displacement1, displacement2, bspline);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_spline_cppComposition_2D<double>(grid1, grid2, displacement1, displacement2, bspline);
            break;
        default:
            NR_FATAL_ERROR("Only implemented for single or double floating images");
        }
    }
    return EXIT_SUCCESS;
}
/* *************************************************************** */
void reg_spline_getFlowFieldFromVelocityGrid(nifti_image *velocityFieldGrid,
                                             nifti_image *flowField) {
    // Check first if the velocity field is actually a velocity field
    if (velocityFieldGrid->intent_p1 != SPLINE_VEL_GRID)
        NR_FATAL_ERROR("The provided grid is not a velocity field");

    // Initialise the flow field with an identity transformation
    reg_tools_multiplyValueToImage(flowField, flowField, 0.f);
    flowField->intent_p1 = DISP_VEL_FIELD;
    reg_getDeformationFromDisplacement(flowField);

    // Fake the number of extension here to avoid the second half of the affine
    int oldNumExt = velocityFieldGrid->num_ext;
    if (oldNumExt > 1)
        velocityFieldGrid->num_ext = 1;

    // Copy over the number of required squaring steps
    flowField->intent_p2 = velocityFieldGrid->intent_p2;
    // The initial flow field is generated using cubic B-Spline interpolation/approximation
    reg_spline_getDeformationField(velocityFieldGrid,
                                   flowField,
                                   nullptr, // mask
                                   true,  // composition
                                   true); // bspline

    velocityFieldGrid->num_ext = oldNumExt;
}
/* *************************************************************** */
void reg_defField_getDeformationFieldFromFlowField(nifti_image *flowField,
                                                   nifti_image *deformationField,
                                                   const bool updateStepNumber) {
    // Check first if the velocity field is actually a velocity field
    if (flowField->intent_p1 != DEF_VEL_FIELD)
        NR_FATAL_ERROR("The provided field is not a velocity field");

    // Remove the affine component from the flow field
    NiftiImage affineOnly;
    if (flowField->num_ext > 0) {
        if (flowField->ext_list[0].edata != nullptr) {
            // Create a field that contains the affine component only
            affineOnly = NiftiImage(deformationField, NiftiImage::Copy::ImageInfoAndAllocData);
            reg_affine_getDeformationField(reinterpret_cast<mat44*>(flowField->ext_list[0].edata),
                                           affineOnly,
                                           false);
            reg_tools_subtractImageFromImage(flowField, affineOnly, flowField);
        }
    } else reg_getDisplacementFromDeformation(flowField);

    // Compute the number of scaling value to ensure unfolded transformation
    int squaringNumber = 1;
    if (updateStepNumber || flowField->intent_p2 == 0) {
        // Check the largest value
        float extrema = fabsf(reg_tools_getMinValue(flowField, -1));
        float temp = reg_tools_getMaxValue(flowField, -1);
        extrema = extrema > temp ? extrema : temp;
        // Check the values for scaling purpose
        float maxLength;
        if (deformationField->nz > 1)
            maxLength = 0.28f;  // sqrt(0.5^2/3)
        else maxLength = 0.35f; // sqrt(0.5^2/2)
        while (extrema / pow(2.0f, squaringNumber) >= maxLength)
            squaringNumber++;
        // The minimal number of step is set to 6 by default
        squaringNumber = squaringNumber < 6 ? 6 : squaringNumber;
        // Set the number of squaring step in the flow field
        if (fabs(flowField->intent_p2) != squaringNumber) {
            NR_WARN("Changing from " << Round(fabs(flowField->intent_p2)) << " to " << abs(squaringNumber) <<
                    " squaring step (equivalent to scaling down by " << (int)pow(2.0f, squaringNumber) << ")");
        }
        // Update the number of squaring step required
        if (flowField->intent_p2 >= 0)
            flowField->intent_p2 = static_cast<float>(squaringNumber);
        else flowField->intent_p2 = static_cast<float>(-squaringNumber);
    } else squaringNumber = static_cast<int>(fabsf(flowField->intent_p2));

    // The displacement field is scaled
    float scalingValue = pow(2.0f, static_cast<float>(std::abs(squaringNumber)));
    if (flowField->intent_p2 < 0)
        // backward deformation field is scaled down
        reg_tools_divideValueToImage(flowField,
                                     flowField,
                                     -scalingValue); // (/-scalingValue)
    else
        // forward deformation field is scaled down
        reg_tools_divideValueToImage(flowField,
                                     flowField,
                                     scalingValue); // (/scalingValue)

    // Conversion from displacement to deformation
    reg_getDeformationFromDisplacement(flowField);

    // The computed scaled deformation field is copied over
    memcpy(deformationField->data, flowField->data,
           deformationField->nvox * deformationField->nbyper);

    // The deformation field is squared
    for (unsigned short i = 0; i < squaringNumber; ++i) {
        // The deformation field is applied to itself
        reg_defField_compose(deformationField,
                             flowField,
                             nullptr);
        // The computed scaled deformation field is copied over
        memcpy(deformationField->data, flowField->data,
               deformationField->nvox * deformationField->nbyper);
        NR_DEBUG("Squaring (composition) step " << i + 1 << "/" << squaringNumber);
    }
    // The affine conponent of the transformation is restored
    if (affineOnly) {
        reg_getDisplacementFromDeformation(deformationField);
        reg_tools_addImageToImage(deformationField, affineOnly, deformationField);
    }
    deformationField->intent_p1 = DEF_FIELD;
    deformationField->intent_p2 = 0;
    // If required an affine component is composed
    if (flowField->num_ext > 1)
        reg_affine_getDeformationField(reinterpret_cast<mat44*>(flowField->ext_list[1].edata), deformationField, true);
}
/* *************************************************************** */
void reg_spline_getDefFieldFromVelocityGrid(nifti_image *velocityFieldGrid,
                                            nifti_image *deformationField,
                                            const bool updateStepNumber) {
    // Clean any extension in the deformation field as it is unexpected
    nifti_free_extensions(deformationField);

    // Check if the velocity field is actually a velocity field
    if (velocityFieldGrid->intent_p1 == CUB_SPLINE_GRID) {
        // Use the spline approximation to generate the deformation field
        reg_spline_getDeformationField(velocityFieldGrid,
                                       deformationField,
                                       nullptr,
                                       false, // composition
                                       true); // bspline
    } else if (velocityFieldGrid->intent_p1 == SPLINE_VEL_GRID) {
        // Create an image to store the flow field
        NiftiImage flowField(deformationField, NiftiImage::Copy::ImageInfoAndAllocData);
        flowField.setIntentName("NREG_TRANS"s);
        flowField->intent_code = NIFTI_INTENT_VECTOR;
        flowField->intent_p1 = DEF_VEL_FIELD;
        flowField->intent_p2 = velocityFieldGrid->intent_p2;
        if (velocityFieldGrid->num_ext > 0)
            nifti_copy_extensions(flowField, velocityFieldGrid);

        // Generate the velocity field
        reg_spline_getFlowFieldFromVelocityGrid(velocityFieldGrid, flowField);
        // Exponentiate the flow field
        reg_defField_getDeformationFieldFromFlowField(flowField, deformationField, updateStepNumber);
        // Update the number of step required. No action otherwise
        velocityFieldGrid->intent_p2 = flowField->intent_p2;
    } else NR_FATAL_ERROR("The provided input image is not a spline parametrised transformation");
}
/* *************************************************************** */
void reg_spline_getIntermediateDefFieldFromVelGrid(nifti_image *velocityFieldGrid,
                                                   nifti_image **deformationField) {
    // Check if the velocity field is actually a velocity field
    if (velocityFieldGrid->intent_p1 == SPLINE_VEL_GRID) {
        // Create an image to store the flow field
        nifti_image *flowField = nifti_dup(*deformationField[0], false);
        flowField->intent_code = NIFTI_INTENT_VECTOR;
        memset(flowField->intent_name, 0, 16);
        strcpy(flowField->intent_name, "NREG_TRANS");
        flowField->intent_p1 = DEF_VEL_FIELD;
        flowField->intent_p2 = velocityFieldGrid->intent_p2;
        if (velocityFieldGrid->num_ext > 0 && flowField->ext_list == nullptr)
            nifti_copy_extensions(flowField, velocityFieldGrid);

        // Generate the velocity field
        reg_spline_getFlowFieldFromVelocityGrid(velocityFieldGrid, flowField);
        // Remove the affine component from the flow field
        nifti_image *affineOnly = nullptr;
        if (flowField->num_ext > 0) {
            if (flowField->ext_list[0].edata != nullptr) {
                // Create a field that contains the affine component only
                affineOnly = nifti_dup(*deformationField[0], false);
                reg_affine_getDeformationField(reinterpret_cast<mat44*>(flowField->ext_list[0].edata), affineOnly, false);
                reg_tools_subtractImageFromImage(flowField, affineOnly, flowField);
            }
        } else reg_getDisplacementFromDeformation(flowField);

        // Compute the number of scaling value to ensure unfolded transformation
        int squaringNumber = static_cast<int>(fabsf(velocityFieldGrid->intent_p2));

        // The displacement field is scaled
        float scalingValue = pow(2.0f, std::abs((float)squaringNumber));
        if (velocityFieldGrid->intent_p2 < 0)
            // backward deformation field is scaled down
            reg_tools_divideValueToImage(flowField, deformationField[0], -scalingValue);
        else
            // forward deformation field is scaled down
            reg_tools_divideValueToImage(flowField, deformationField[0], scalingValue);

        // Deallocate the allocated flow field
        nifti_image_free(flowField);
        flowField = nullptr;

        // Conversion from displacement to deformation
        reg_getDeformationFromDisplacement(deformationField[0]);

        // The deformation field is squared
        for (unsigned short i = 0; i < squaringNumber; ++i) {
            // The computed scaled deformation field is copied over
            memcpy(deformationField[i + 1]->data, deformationField[i]->data,
                   deformationField[i]->nvox * deformationField[i]->nbyper);
            // The deformation field is applied to itself
            reg_defField_compose(deformationField[i], // to apply
                                 deformationField[i + 1], // to update
                                 nullptr);
            NR_DEBUG("Squaring (composition) step " << i + 1 << "/" << squaringNumber);
        }
        // The affine conponent of the transformation is restored
        if (affineOnly != nullptr) {
            for (unsigned short i = 0; i <= squaringNumber; ++i) {
                reg_getDisplacementFromDeformation(deformationField[i]);
                reg_tools_addImageToImage(deformationField[i], affineOnly, deformationField[i]);
                deformationField[i]->intent_p1 = DEF_FIELD;
                deformationField[i]->intent_p2 = 0;
            }
            nifti_image_free(affineOnly);
            affineOnly = nullptr;
        }
        // If required an affine component is composed
        if (velocityFieldGrid->num_ext > 1) {
            for (unsigned short i = 0; i <= squaringNumber; ++i) {
                reg_affine_getDeformationField(reinterpret_cast<mat44*>(velocityFieldGrid->ext_list[1].edata),
                                               deformationField[i],
                                               true);
            }
        }
    } else NR_FATAL_ERROR("The provided input image is not a spline parametrised transformation");
}
/* *************************************************************** */
template <class DataType>
void compute_lie_bracket(nifti_image *img1,
                         nifti_image *img2,
                         nifti_image *res,
                         bool use_jac) {
    NR_FATAL_ERROR("The compute_lie_bracket function needs updating");
#ifdef _WIN32
    long voxNumber = (long)NiftiImage::calcVoxelNumber(img1, 3);
#else
    size_t voxNumber = NiftiImage::calcVoxelNumber(img1, 3);
#endif
    // Lie bracket using Jacobian for testing
    if (use_jac) {
        mat33 *jacImg1 = (mat33*)malloc(voxNumber * sizeof(mat33));
        mat33 *jacImg2 = (mat33*)malloc(voxNumber * sizeof(mat33));

        reg_getDeformationFromDisplacement(img1);
        reg_getDeformationFromDisplacement(img2);
        // HERE TO DO
        NR_FATAL_ERROR("The function needs updating");
        //        reg_spline_GetJacobianMatrixFull(img1,img1,jacImg1);
        //        reg_spline_GetJacobianMatrixFull(img2,img2,jacImg2);
        reg_getDisplacementFromDeformation(img1);
        reg_getDisplacementFromDeformation(img2);

        DataType *resPtrX = static_cast<DataType*>(res->data);
        DataType *resPtrY = &resPtrX[voxNumber];
        DataType *img1DispPtrX = static_cast<DataType*>(img1->data);
        DataType *img1DispPtrY = &img1DispPtrX[voxNumber];
        DataType *img2DispPtrX = static_cast<DataType*>(img2->data);
        DataType *img2DispPtrY = &img1DispPtrX[voxNumber];
        if (img1->nz > 1) {
            DataType *resPtrZ = &resPtrY[voxNumber];
            DataType *img1DispPtrZ = &img1DispPtrY[voxNumber];
            DataType *img2DispPtrZ = &img1DispPtrY[voxNumber];

            for (size_t i = 0; i < voxNumber; ++i) {
                resPtrX[i] =
                    (jacImg2[i].m[0][0] * img1DispPtrX[i] +
                     jacImg2[i].m[0][1] * img1DispPtrY[i] +
                     jacImg2[i].m[0][2] * img1DispPtrZ[i])
                    -
                    (jacImg1[i].m[0][0] * img2DispPtrX[i] +
                     jacImg1[i].m[0][1] * img2DispPtrY[i] +
                     jacImg1[i].m[0][2] * img2DispPtrZ[i]);
                resPtrY[i] =
                    (jacImg2[i].m[1][0] * img1DispPtrX[i] +
                     jacImg2[i].m[1][1] * img1DispPtrY[i] +
                     jacImg2[i].m[1][2] * img1DispPtrZ[i])
                    -
                    (jacImg1[i].m[1][0] * img2DispPtrX[i] +
                     jacImg1[i].m[1][1] * img2DispPtrY[i] +
                     jacImg1[i].m[1][2] * img2DispPtrZ[i]);
                resPtrZ[i] =
                    (jacImg2[i].m[2][0] * img1DispPtrX[i] +
                     jacImg2[i].m[2][1] * img1DispPtrY[i] +
                     jacImg2[i].m[2][2] * img1DispPtrZ[i])
                    -
                    (jacImg1[i].m[2][0] * img2DispPtrX[i] +
                     jacImg1[i].m[2][1] * img2DispPtrY[i] +
                     jacImg1[i].m[2][2] * img2DispPtrZ[i]);
            }
        } else {
            for (size_t i = 0; i < voxNumber; ++i) {
                resPtrX[i] =
                    (jacImg2[i].m[0][0] * img1DispPtrX[i] +
                     jacImg2[i].m[0][1] * img1DispPtrY[i])
                    -
                    (jacImg1[i].m[0][0] * img2DispPtrX[i] +
                     jacImg1[i].m[0][1] * img2DispPtrY[i]);
                resPtrY[i] =
                    (jacImg2[i].m[1][0] * img1DispPtrX[i] +
                     jacImg2[i].m[1][1] * img1DispPtrY[i])
                    -
                    (jacImg1[i].m[1][0] * img2DispPtrX[i] +
                     jacImg1[i].m[1][1] * img2DispPtrY[i]);
            }
        }
        free(jacImg1);
        free(jacImg2);
        return;
    }


    // Allocate two temporary nifti images and set them to zero displacement
    nifti_image *one_two = nifti_dup(*img2, false);
    nifti_image *two_one = nifti_dup(*img1, false);
    // Compute the displacement from img1
    reg_spline_cppComposition(img1,
                              two_one,
                              true,  // displacement1?
                              true,  // displacement2?
                              true); // bspline?
    // Compute the displacement from img2
    reg_spline_cppComposition(img2,
                              one_two,
                              true,  // displacement1?
                              true,  // displacement2?
                              true); // bspline?
    // Compose both transformations
    reg_spline_cppComposition(img1,
                              one_two,
                              true,  // displacement1?
                              true,  // displacement2?
                              true); // bspline?
    // Compose both transformations
    reg_spline_cppComposition(img2,
                              two_one,
                              true,  // displacement1?
                              true,  // displacement2?
                              true); // bspline?
    // Create the data pointers
    DataType *resPtr = static_cast<DataType*>(res->data);
    DataType *one_twoPtr = static_cast<DataType*>(one_two->data);
    DataType *two_onePtr = static_cast<DataType*>(two_one->data);
    // Compute the lie bracket value using difference of composition

#ifdef _WIN32
    long i;
    voxNumber = (long)res->nvox;
#else
    size_t i;
    voxNumber = res->nvox;
#endif

#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(voxNumber, resPtr, one_twoPtr, two_onePtr)
#endif
    for (i = 0; i < voxNumber; ++i)
        resPtr[i] = two_onePtr[i] - one_twoPtr[i];
    // Free the temporary nifti images
    nifti_image_free(one_two);
    nifti_image_free(two_one);
}
/* *************************************************************** */
template <class DataType>
void compute_BCH_update(nifti_image *img1, // current field
                         nifti_image *img2, // gradient
                         int type) {
    // To update
    NR_FATAL_ERROR("The compute_BCH_update function needs updating");
    DataType *res = (DataType*)malloc(img1->nvox * sizeof(DataType));

#ifdef _WIN32
    long i;
    long voxelNumber = (long)img1->nvox;
#else
    size_t i;
    size_t voxelNumber = img1->nvox;
#endif

    bool use_jac = false;

    // r <- 2 + 1
    DataType *img1Ptr = static_cast<DataType*>(img1->data);
    DataType *img2Ptr = static_cast<DataType*>(img2->data);
#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(voxelNumber,img1Ptr,img2Ptr, res)
#endif
    for (i = 0; i < voxelNumber; ++i)
        res[i] = img1Ptr[i] + img2Ptr[i];

    if (type > 0) {
        // Convert the deformation field into a displacement field
        reg_getDisplacementFromDeformation(img1);

        // r <- 2 + 1 + 0.5[2,1]
        nifti_image *lie_bracket_img2_img1 = nifti_dup(*img1, false);
        compute_lie_bracket<DataType>(img2, img1, lie_bracket_img2_img1, use_jac);
        DataType *lie_bracket_img2_img1Ptr = static_cast<DataType*>(lie_bracket_img2_img1->data);
#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(voxelNumber, res, lie_bracket_img2_img1Ptr)
#endif
        for (i = 0; i < voxelNumber; ++i)
            res[i] += 0.5f * lie_bracket_img2_img1Ptr[i];

        if (type > 1) {
            // r <- 2 + 1 + 0.5[2,1] + [2,[2,1]]/12
            nifti_image *lie_bracket_img2_lie1 = nifti_dup(*lie_bracket_img2_img1, false);
            compute_lie_bracket<DataType>(img2, lie_bracket_img2_img1, lie_bracket_img2_lie1, use_jac);
            DataType *lie_bracket_img2_lie1Ptr = static_cast<DataType*>(lie_bracket_img2_lie1->data);
#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(voxelNumber, res, lie_bracket_img2_lie1Ptr)
#endif
            for (i = 0; i < voxelNumber; ++i)
                res[i] += lie_bracket_img2_lie1Ptr[i] / 12.f;

            if (type > 2) {
                // r <- 2 + 1 + 0.5[2,1] + [2,[2,1]]/12 - [1,[2,1]]/12
                nifti_image *lie_bracket_img1_lie1 = nifti_dup(*lie_bracket_img2_img1, false);
                compute_lie_bracket<DataType>(img1, lie_bracket_img2_img1, lie_bracket_img1_lie1, use_jac);
                DataType *lie_bracket_img1_lie1Ptr = static_cast<DataType*>(lie_bracket_img1_lie1->data);
#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(voxelNumber, res, lie_bracket_img1_lie1Ptr)
#endif
                for (i = 0; i < voxelNumber; ++i)
                    res[i] -= lie_bracket_img1_lie1Ptr[i] / 12.f;
                nifti_image_free(lie_bracket_img1_lie1);

                if (type > 3) {
                    // r <- 2 + 1 + 0.5[2,1] + [2,[2,1]]/12 - [1,[2,1]]/12 - [1,[2,[2,1]]]/24
                    nifti_image *lie_bracket_img1_lie2 = nifti_dup(*lie_bracket_img2_lie1, false);
                    compute_lie_bracket<DataType>(img1, lie_bracket_img2_lie1, lie_bracket_img1_lie2, use_jac);
                    DataType *lie_bracket_img1_lie2Ptr = static_cast<DataType*>(lie_bracket_img1_lie2->data);
#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(voxelNumber, res, lie_bracket_img1_lie2Ptr)
#endif
                    for (i = 0; i < voxelNumber; ++i)
                        res[i] -= lie_bracket_img1_lie2Ptr[i] / 24.f;
                    nifti_image_free(lie_bracket_img1_lie2);
                }// >3
            }// >2
            nifti_image_free(lie_bracket_img2_lie1);
        }// >1
        nifti_image_free(lie_bracket_img2_img1);
    }// >0

    // update the deformation field
    memcpy(img1->data, res, img1->nvox * img1->nbyper);
    free(res);
}
/* *************************************************************** */
void compute_BCH_update(nifti_image *img1, // current field
                        nifti_image *img2, // gradient
                        int type) {
    if (img1->datatype != img2->datatype)
        NR_FATAL_ERROR("Both input images are expected to be of same type");
    switch (img1->datatype) {
    case NIFTI_TYPE_FLOAT32:
        compute_BCH_update<float>(img1, img2, type);
        break;
    case NIFTI_TYPE_FLOAT64:
        compute_BCH_update<double>(img1, img2, type);
        break;
    default:
        NR_FATAL_ERROR("Only implemented for single or double precision images");
    }
}
/* *************************************************************** */
template <class DataType>
void extractLine(int start, int end, int increment, const DataType *image, DataType *values) {
    size_t index = 0;
    for (int i = start; i < end; i += increment) values[index++] = image[i];
}
/* *************************************************************** */
template <class DataType>
void restoreLine(int start, int end, int increment, DataType *image, const DataType *values) {
    size_t index = 0;
    for (int i = start; i < end; i += increment) image[i] = values[index++];
}
/* *************************************************************** */
template <class DataType>
void intensitiesToSplineCoefficients(DataType *values, int number) {
    // Border are set to zero
    DataType pole = sqrt(3.0) - 2.0;
    DataType currentPole = pole;
    DataType currentOpposite = pow(pole, (DataType)(2.0 * (DataType)number - 1.0));
    DataType sum = 0;
    for (int i = 1; i < number; i++) {
        sum += (currentPole - currentOpposite) * values[i];
        currentPole *= pole;
        currentOpposite /= pole;
    }
    values[0] = (DataType)((values[0] - pole * pole * (values[0] + sum)) / (1.0 - pow(pole, (DataType)(2.0 * (double)number + 2.0))));

    //other values forward
    for (int i = 1; i < number; i++) {
        values[i] += pole * values[i - 1];
    }

    DataType ipp = (DataType)(1.0 - pole);
    ipp *= ipp;

    //last value
    values[number - 1] = ipp * values[number - 1];

    //other values backward
    for (int i = number - 2; 0 <= i; i--) {
        values[i] = pole * values[i + 1] + ipp * values[i];
    }
}
/* *************************************************************** */
template <class DataType>
void reg_spline_getDeconvolvedCoefficents(nifti_image *img) {
    double *coeff = (double*)malloc(img->nvox * sizeof(double));
    DataType *imgPtr = static_cast<DataType*>(img->data);
    for (size_t i = 0; i < img->nvox; ++i)
        coeff[i] = imgPtr[i];
    for (int u = 0; u < img->nu; ++u) {
        for (int t = 0; t < img->nt; ++t) {
            double *coeffPtr = &coeff[(u * img->nt + t) * img->nx * img->ny * img->nz];

            // Along the X axis
            int number = img->nx;
            double *values = new double[number];
            int increment = 1;
            for (int i = 0; i < img->ny * img->nz; i++) {
                int start = i * img->nx;
                int end = start + img->nx;
                extractLine<double>(start, end, increment, coeffPtr, values);
                intensitiesToSplineCoefficients<double>(values, number);
                restoreLine<double>(start, end, increment, coeffPtr, values);
            }
            delete[] values;
            values = nullptr;

            // Along the Y axis
            number = img->ny;
            values = new double[number];
            increment = img->nx;
            for (int i = 0; i < img->nx * img->nz; i++) {
                int start = i + i / img->nx * img->nx * (img->ny - 1);
                int end = start + img->nx * img->ny;
                extractLine<double>(start, end, increment, coeffPtr, values);
                intensitiesToSplineCoefficients<double>(values, number);
                restoreLine<double>(start, end, increment, coeffPtr, values);
            }
            delete[] values;
            values = nullptr;

            // Along the Z axis
            if (img->nz > 1) {
                number = img->nz;
                values = new double[number];
                increment = img->nx * img->ny;
                for (int i = 0; i < img->nx * img->ny; i++) {
                    int start = i;
                    int end = start + img->nx * img->ny * img->nz;
                    extractLine<double>(start, end, increment, coeffPtr, values);
                    intensitiesToSplineCoefficients<double>(values, number);
                    restoreLine<double>(start, end, increment, coeffPtr, values);
                }
                delete[] values;
                values = nullptr;
            }
        }//t
    }//u

    for (size_t i = 0; i < img->nvox; ++i)
        imgPtr[i] = static_cast<DataType>(coeff[i]);
    free(coeff);
}
/* *************************************************************** */
void reg_spline_getDeconvolvedCoefficents(nifti_image *img) {
    switch (img->datatype) {
    case NIFTI_TYPE_FLOAT32:
        reg_spline_getDeconvolvedCoefficents<float>(img);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_spline_getDeconvolvedCoefficents<double>(img);
        break;
    default:
        NR_FATAL_ERROR("Only implemented for single or double precision images");
    }
}
/* *************************************************************** */
