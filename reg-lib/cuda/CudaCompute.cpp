#include "CudaCompute.h"
#include "CudaF3dContent.h"
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
                          con.Content::GetReference()->nvox,
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
    CudaF3dContent& con = dynamic_cast<CudaF3dContent&>(this->con);
    reg_spline_getDeformationField_gpu(con.F3dContent::GetControlPointGrid(),
                                       con.F3dContent::GetReference(),
                                       con.GetControlPointGridCuda(),
                                       con.GetDeformationFieldCuda(),
                                       con.GetReferenceMaskCuda(),
                                       con.F3dContent::GetReference()->nvox,
                                       bspline);
}
/* *************************************************************** */
void CudaCompute::UpdateControlPointPosition(float *currentDOF, float *bestDOF, float *gradient, float scale, bool optimiseX, bool optimiseY, bool optimiseZ) {
    // TODO Fix reg_updateControlPointPosition_gpu to accept optimiseX, optimiseY, optimiseZ
    reg_updateControlPointPosition_gpu(dynamic_cast<CudaF3dContent&>(con).F3dContent::GetControlPointGrid(),
                                       reinterpret_cast<float4*>(currentDOF),
                                       reinterpret_cast<float4*>(bestDOF),
                                       reinterpret_cast<float4*>(gradient),
                                       scale);
}
/* *************************************************************** */
void CudaCompute::GetImageGradient(int interpolation, float paddingValue, int activeTimepoint) {
    CudaF3dContent& con = dynamic_cast<CudaF3dContent&>(this->con);
    reg_getImageGradient_gpu(con.F3dContent::GetFloating(),
                             con.GetFloatingCuda(),
                             con.GetDeformationFieldCuda(),
                             con.GetWarpedGradientCuda(),
                             con.F3dContent::GetReference()->nvox,
                             paddingValue);
}
/* *************************************************************** */
void CudaCompute::VoxelCentricToNodeCentric(float weight) {
    CudaF3dContent& con = dynamic_cast<CudaF3dContent&>(this->con);
    reg_voxelCentric2NodeCentric_gpu(con.F3dContent::GetWarped(),
                                     con.F3dContent::GetControlPointGrid(),
                                     con.GetVoxelBasedMeasureGradientCuda(),
                                     con.GetTransformationGradientCuda(),
                                     weight);
}
/* *************************************************************** */
double CudaCompute::GetMaximalLength(size_t nodeNumber, bool optimiseX, bool optimiseY, bool optimiseZ) {
    // TODO Fix reg_getMaximalLength_gpu to accept optimiseX, optimiseY, optimiseZ
    return reg_getMaximalLength_gpu(dynamic_cast<CudaF3dContent&>(con).GetTransformationGradientCuda(), nodeNumber);
}
/* *************************************************************** */
void CudaCompute::NormaliseGradient(size_t nodeNumber, double maxGradLength) {
    // TODO Fix reg_multiplyValue_gpu to accept optimiseX, optimiseY, optimiseZ
    reg_multiplyValue_gpu(nodeNumber, dynamic_cast<CudaF3dContent&>(con).GetTransformationGradientCuda(), 1 / (float)maxGradLength);
}
/* *************************************************************** */
void CudaCompute::GetApproximatedGradient(InterfaceOptimiser& opt) {
    // TODO Implement this for CUDA
    // Use CPU temporarily
    Compute::GetApproximatedGradient(opt);
}
/* *************************************************************** */
