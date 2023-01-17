#include "Compute.h"
#include "F3dContent.h"
#include "_reg_resampling.h"
#include "_reg_localTrans_jac.h"
#include "_reg_localTrans_regul.h"

/* *************************************************************** */
void Compute::ResampleImage(int inter, float paddingValue) {
    reg_resampleImage(con.GetFloating(),
                      con.GetWarped(),
                      con.GetDeformationField(),
                      con.GetReferenceMask(),
                      inter,
                      paddingValue);
}
/* *************************************************************** */
double Compute::GetJacobianPenaltyTerm(bool approx) {
    F3dContent& con = dynamic_cast<F3dContent&>(this->con);
    return reg_spline_getJacobianPenaltyTerm(con.GetControlPointGrid(),
                                             con.GetReference(),
                                             approx);
}
/* *************************************************************** */
void Compute::JacobianPenaltyTermGradient(float weight, bool approx) {
    F3dContent& con = dynamic_cast<F3dContent&>(this->con);
    reg_spline_getJacobianPenaltyTermGradient(con.GetControlPointGrid(),
                                              con.GetReference(),
                                              con.GetTransformationGradient(),
                                              weight,
                                              approx);
}
/* *************************************************************** */
double Compute::CorrectFolding(bool approx) {
    F3dContent& con = dynamic_cast<F3dContent&>(this->con);
    return reg_spline_correctFolding(con.GetControlPointGrid(),
                                     con.GetReference(),
                                     approx);
}
/* *************************************************************** */
double Compute::ApproxBendingEnergy() {
    F3dContent& con = dynamic_cast<F3dContent&>(this->con);
    return reg_spline_approxBendingEnergy(con.GetControlPointGrid());
}
/* *************************************************************** */
void Compute::ApproxBendingEnergyGradient(float weight) {
    F3dContent& con = dynamic_cast<F3dContent&>(this->con);
    reg_spline_approxBendingEnergyGradient(con.GetControlPointGrid(),
                                           con.GetTransformationGradient(),
                                           weight);
}
/* *************************************************************** */
double Compute::ApproxLinearEnergy() {
    F3dContent& con = dynamic_cast<F3dContent&>(this->con);
    return reg_spline_approxLinearEnergy(con.GetControlPointGrid());
}
/* *************************************************************** */
void Compute::ApproxLinearEnergyGradient(float weight) {
    F3dContent& con = dynamic_cast<F3dContent&>(this->con);
    reg_spline_approxLinearEnergyGradient(con.F3dContent::GetControlPointGrid(),
                                          con.GetTransformationGradient(),
                                          weight);
}
/* *************************************************************** */
double Compute::GetLandmarkDistance(size_t landmarkNumber, float *landmarkReference, float *landmarkFloating) {
    F3dContent& con = dynamic_cast<F3dContent&>(this->con);
    return reg_spline_getLandmarkDistance(con.F3dContent::GetControlPointGrid(),
                                          landmarkNumber,
                                          landmarkReference,
                                          landmarkFloating);
}
/* *************************************************************** */
void Compute::LandmarkDistanceGradient(size_t landmarkNumber, float *landmarkReference, float *landmarkFloating, float weight) {
    F3dContent& con = dynamic_cast<F3dContent&>(this->con);
    reg_spline_getLandmarkDistanceGradient(con.F3dContent::GetControlPointGrid(),
                                           con.GetTransformationGradient(),
                                           landmarkNumber,
                                           landmarkReference,
                                           landmarkFloating,
                                           weight);
}
/* *************************************************************** */
void Compute::GetDeformationField(bool composition, bool bspline) {
    F3dContent& con = dynamic_cast<F3dContent&>(this->con);
    reg_spline_getDeformationField(con.GetControlPointGrid(),
                                   con.GetDeformationField(),
                                   con.GetReferenceMask(),
                                   composition,
                                   bspline);
}
/* *************************************************************** */
void Compute::UpdateControlPointPosition(float *currentDOF, float *bestDOF, float *gradient, float scale, bool optimiseX, bool optimiseY, bool optimiseZ) {
    nifti_image *controlPointGrid = dynamic_cast<F3dContent&>(con).GetControlPointGrid();
    if (optimiseX && optimiseY && optimiseZ) {
        // Update the values for all axis displacement
        for (size_t i = 0; i < controlPointGrid->nvox; ++i)
            currentDOF[i] = bestDOF[i] + scale * gradient[i];
    } else {
        size_t voxNumber = controlPointGrid->nvox / controlPointGrid->ndim;
        // Update the values for the x-axis displacement
        if (optimiseX) {
            for (size_t i = 0; i < voxNumber; ++i)
                currentDOF[i] = bestDOF[i] + scale * gradient[i];
        }
        // Update the values for the y-axis displacement
        if (optimiseY && controlPointGrid->ndim > 1) {
            float *currentDOFY = &currentDOF[voxNumber];
            float *bestDOFY = &bestDOF[voxNumber];
            float *gradientY = &gradient[voxNumber];
            for (size_t i = 0; i < voxNumber; ++i)
                currentDOFY[i] = bestDOFY[i] + scale * gradientY[i];
        }
        // Update the values for the z-axis displacement
        if (optimiseZ && controlPointGrid->ndim > 2) {
            float *currentDOFZ = &currentDOF[2 * voxNumber];
            float *bestDOFZ = &bestDOF[2 * voxNumber];
            float *gradientZ = &gradient[2 * voxNumber];
            for (size_t i = 0; i < voxNumber; ++i)
                currentDOFZ[i] = bestDOFZ[i] + scale * gradientZ[i];
        }
    }
}
/* *************************************************************** */
void Compute::GetImageGradient(int interpolation, float paddingValue, int activeTimepoint) {
    F3dContent& con = dynamic_cast<F3dContent&>(this->con);
    reg_getImageGradient(con.GetFloating(),
                         con.GetWarpedGradient(),
                         con.GetDeformationField(),
                         con.GetReferenceMask(),
                         interpolation,
                         paddingValue,
                         activeTimepoint);
}
/* *************************************************************** */
void Compute::VoxelCentricToNodeCentric(float weight) {
    F3dContent& con = dynamic_cast<F3dContent&>(this->con);
    mat44 *reorientation = Content::GetIJKMatrix(*con.GetFloating());
    reg_voxelCentric2NodeCentric(con.GetTransformationGradient(),
                                 con.GetVoxelBasedMeasureGradient(),
                                 weight,
                                 false, // no update
                                 reorientation);
}
/* *************************************************************** */
double Compute::GetMaximalLength(size_t nodeNumber, bool optimiseX, bool optimiseY, bool optimiseZ) {
    // TODO Fix reg_getMaximalLength to accept optimiseX, optimiseY, optimiseZ
    nifti_image *transformationGradient = dynamic_cast<F3dContent&>(con).GetTransformationGradient();
    switch (transformationGradient->datatype) {
    case NIFTI_TYPE_FLOAT32:
        return reg_getMaximalLength<float>(transformationGradient);
    case NIFTI_TYPE_FLOAT64:
        return reg_getMaximalLength<double>(transformationGradient);
    }
    return 0;
}
/* *************************************************************** */
void Compute::NormaliseGradient(size_t nodeNumber, double maxGradLength) {
    // TODO Fix reg_tools_multiplyValueToImage to accept optimiseX, optimiseY, optimiseZ
    nifti_image *transformationGradient = dynamic_cast<F3dContent&>(con).GetTransformationGradient();
    reg_tools_multiplyValueToImage(transformationGradient, transformationGradient, 1 / (float)maxGradLength);
}
/* *************************************************************** */
