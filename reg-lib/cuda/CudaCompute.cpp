#include "CudaCompute.h"
#include "CudaF3dContent.h"
#include "CudaNormaliseGradient.hpp"
#include "_reg_resampling_gpu.h"
#include "_reg_localTransformation_gpu.h"
#include "_reg_optimiser_gpu.h"

/* *************************************************************** */
void CudaCompute::ResampleImage(int inter, float paddingValue) {
    CudaContent& con = dynamic_cast<CudaContent&>(this->con);
    reg_resampleImage_gpu(con.Content::GetFloating(),
                          con.GetWarpedCuda(),
                          con.GetFloatingCuda(),
                          con.GetDeformationFieldCuda(),
                          con.GetReferenceMaskCuda(),
                          con.GetActiveVoxelNumber(),
                          paddingValue);
}
/* *************************************************************** */
double CudaCompute::GetJacobianPenaltyTerm(bool approx) {
    CudaF3dContent& con = dynamic_cast<CudaF3dContent&>(this->con);
    return reg_spline_getJacobianPenaltyTerm_gpu(con.F3dContent::GetReference(),
                                                 con.F3dContent::GetControlPointGrid(),
                                                 con.GetControlPointGridCuda(),
                                                 approx);
}
/* *************************************************************** */
void CudaCompute::JacobianPenaltyTermGradient(float weight, bool approx) {
    CudaF3dContent& con = dynamic_cast<CudaF3dContent&>(this->con);
    reg_spline_getJacobianPenaltyTermGradient_gpu(con.F3dContent::GetReference(),
                                                  con.F3dContent::GetControlPointGrid(),
                                                  con.GetControlPointGridCuda(),
                                                  con.GetTransformationGradientCuda(),
                                                  weight,
                                                  approx);
}
/* *************************************************************** */
double CudaCompute::CorrectFolding(bool approx) {
    CudaF3dContent& con = dynamic_cast<CudaF3dContent&>(this->con);
    return reg_spline_correctFolding_gpu(con.F3dContent::GetReference(),
                                         con.F3dContent::GetControlPointGrid(),
                                         con.GetControlPointGridCuda(),
                                         approx);
}
/* *************************************************************** */
double CudaCompute::ApproxBendingEnergy() {
    CudaF3dContent& con = dynamic_cast<CudaF3dContent&>(this->con);
    return reg_spline_approxBendingEnergy_gpu(con.F3dContent::GetControlPointGrid(), con.GetControlPointGridCuda());
}
/* *************************************************************** */
void CudaCompute::ApproxBendingEnergyGradient(float weight) {
    CudaF3dContent& con = dynamic_cast<CudaF3dContent&>(this->con);
    reg_spline_approxBendingEnergyGradient_gpu(con.F3dContent::GetControlPointGrid(),
                                               con.GetControlPointGridCuda(),
                                               con.GetTransformationGradientCuda(),
                                               weight);
}
/* *************************************************************** */
double CudaCompute::ApproxLinearEnergy() {
    // TODO Implement this for CUDA
    // Use CPU temporarily
    return Compute::ApproxLinearEnergy();
}
/* *************************************************************** */
void CudaCompute::ApproxLinearEnergyGradient(float weight) {
    // TODO Implement this for CUDA
    // Use CPU temporarily
    Compute::ApproxLinearEnergyGradient(weight);
    // Transfer the data back to the CUDA device
    dynamic_cast<CudaF3dContent&>(con).UpdateTransformationGradient();
}
/* *************************************************************** */
double CudaCompute::GetLandmarkDistance(size_t landmarkNumber, float *landmarkReference, float *landmarkFloating) {
    // TODO Implement this for CUDA
    // Use CPU temporarily
    return Compute::GetLandmarkDistance(landmarkNumber, landmarkReference, landmarkFloating);
}
/* *************************************************************** */
void CudaCompute::LandmarkDistanceGradient(size_t landmarkNumber, float *landmarkReference, float *landmarkFloating, float weight) {
    // TODO Implement this for CUDA
    // Use CPU temporarily
    Compute::LandmarkDistanceGradient(landmarkNumber, landmarkReference, landmarkFloating, weight);
    // Transfer the data back to the CUDA device
    dynamic_cast<CudaF3dContent&>(con).UpdateTransformationGradient();
}
/* *************************************************************** */
void CudaCompute::GetDeformationField(bool composition, bool bspline) {
    // TODO Fix reg_spline_getDeformationField_gpu to accept composition
    CudaF3dContent& con = dynamic_cast<CudaF3dContent&>(this->con);
    reg_spline_getDeformationField_gpu(con.F3dContent::GetControlPointGrid(),
                                       con.F3dContent::GetReference(),
                                       con.GetControlPointGridCuda(),
                                       con.GetDeformationFieldCuda(),
                                       con.GetReferenceMaskCuda(),
                                       con.GetActiveVoxelNumber(),
                                       bspline);
}
/* *************************************************************** */
void CudaCompute::UpdateControlPointPosition(float *currentDof,
                                             const float *bestDof,
                                             const float *gradient,
                                             const float& scale,
                                             const bool& optimiseX,
                                             const bool& optimiseY,
                                             const bool& optimiseZ) {
    reg_updateControlPointPosition_gpu(NiftiImage::calcVoxelNumber(dynamic_cast<CudaF3dContent&>(con).F3dContent::GetControlPointGrid(), 3),
                                       reinterpret_cast<float4*>(currentDof),
                                       reinterpret_cast<const float4*>(bestDof),
                                       reinterpret_cast<const float4*>(gradient),
                                       scale,
                                       optimiseX,
                                       optimiseY,
                                       optimiseZ);
}
/* *************************************************************** */
void CudaCompute::GetImageGradient(int interpolation, float paddingValue, int activeTimepoint) {
    // TODO Fix reg_getImageGradient_gpu to accept interpolation and activeTimepoint
    CudaF3dContent& con = dynamic_cast<CudaF3dContent&>(this->con);
    reg_getImageGradient_gpu(con.F3dContent::GetFloating(),
                             con.GetFloatingCuda(),
                             con.GetDeformationFieldCuda(),
                             con.GetWarpedGradientCuda(),
                             con.GetActiveVoxelNumber(),
                             paddingValue);
}
/* *************************************************************** */
double CudaCompute::GetMaximalLength(bool optimiseX, bool optimiseY, bool optimiseZ) {
    if (!optimiseX && !optimiseY && !optimiseZ) return 0;
    CudaF3dContent& con = dynamic_cast<CudaF3dContent&>(this->con);
    const size_t voxelsPerVolume = NiftiImage::calcVoxelNumber(con.F3dContent::GetTransformationGradient(), 3);
    return NiftyReg::Cuda::GetMaximalLength(con.GetTransformationGradientCuda(), voxelsPerVolume, optimiseX, optimiseY, optimiseZ);
}
/* *************************************************************** */
void CudaCompute::NormaliseGradient(double maxGradLength, bool optimiseX, bool optimiseY, bool optimiseZ) {
    if (maxGradLength == 0 || (!optimiseX && !optimiseY && !optimiseZ)) return;
    CudaF3dContent& con = dynamic_cast<CudaF3dContent&>(this->con);
    const size_t voxelsPerVolume = NiftiImage::calcVoxelNumber(con.F3dContent::GetTransformationGradient(), 3);
    NiftyReg::Cuda::NormaliseGradient(con.GetTransformationGradientCuda(), voxelsPerVolume, static_cast<float>(maxGradLength), optimiseX, optimiseY, optimiseZ);
}
/* *************************************************************** */
void CudaCompute::SmoothGradient(float sigma) {
    // TODO Implement this for CUDA
    // Use CPU temporarily
    if (sigma != 0) {
        Compute::SmoothGradient(sigma);
        // Update the changes for GPU
        dynamic_cast<CudaF3dContent&>(con).UpdateTransformationGradient();
    }
}
/* *************************************************************** */
void CudaCompute::GetApproximatedGradient(InterfaceOptimiser& opt) {
    // TODO Implement this for CUDA
    // Use CPU temporarily
    Compute::GetApproximatedGradient(opt);
}
/* *************************************************************** */
void CudaCompute::GetDefFieldFromVelocityGrid(bool updateStepNumber) {
    // TODO Implement this for CUDA
    // Use CPU temporarily
    Compute::GetDefFieldFromVelocityGrid(updateStepNumber);
    // Transfer the data back to the CUDA device
    CudaF3dContent& con = dynamic_cast<CudaF3dContent&>(this->con);
    // TODO update only the required ones
    con.UpdateControlPointGrid();
    con.UpdateDeformationField();
}
/* *************************************************************** */
void CudaCompute::ConvolveVoxelBasedMeasureGradient(float weight) {
    // TODO Implement this for CUDA
    // Use CPU temporarily
    CudaF3dContent& con = dynamic_cast<CudaF3dContent&>(this->con);
    Compute::ConvolveImage(con.GetVoxelBasedMeasureGradient());
    // Transfer the data back to the CUDA device
    con.UpdateVoxelBasedMeasureGradient();

    // The node-based NMI gradient is extracted
    reg_voxelCentric2NodeCentric_gpu(con.F3dContent::GetWarped(),
                                     con.F3dContent::GetControlPointGrid(),
                                     con.GetVoxelBasedMeasureGradientCuda(),
                                     con.GetTransformationGradientCuda(),
                                     weight);
}
/* *************************************************************** */
void CudaCompute::ExponentiateGradient(Content& conBwIn) {
    // TODO Implement this for CUDA
    // Use CPU temporarily
    Compute::ExponentiateGradient(conBwIn);
    // Transfer the data back to the CUDA device
    dynamic_cast<CudaF3dContent&>(con).UpdateVoxelBasedMeasureGradient();
}
/* *************************************************************** */
void CudaCompute::UpdateVelocityField(float scale, bool optimiseX, bool optimiseY, bool optimiseZ) {
    // TODO Implement this for CUDA
    // Use CPU temporarily
    Compute::UpdateVelocityField(scale, optimiseX, optimiseY, optimiseZ);
    // Transfer the data back to the CUDA device
    dynamic_cast<CudaF3dContent&>(con).UpdateControlPointGrid();
}
/* *************************************************************** */
void CudaCompute::BchUpdate(float scale, int bchUpdateValue) {
    // TODO Implement this for CUDA
    // Use CPU temporarily
    Compute::BchUpdate(scale, bchUpdateValue);
    // Transfer the data back to the CUDA device
    dynamic_cast<CudaF3dContent&>(con).UpdateControlPointGrid();
}
/* *************************************************************** */
void CudaCompute::SymmetriseVelocityFields(Content& conBwIn) {
    // TODO Implement this for CUDA
    // Use CPU temporarily
    Compute::SymmetriseVelocityFields(conBwIn);
    // Transfer the data back to the CUDA device
    dynamic_cast<CudaF3dContent&>(con).UpdateControlPointGrid();
    dynamic_cast<CudaF3dContent&>(conBwIn).UpdateControlPointGrid();
}
/* *************************************************************** */
