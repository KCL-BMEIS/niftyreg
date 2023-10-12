#include "CudaCompute.h"
#include "CudaF3dContent.h"
#include "CudaKernelConvolution.hpp"
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
    CudaF3dContent& con = dynamic_cast<CudaF3dContent&>(this->con);
    const nifti_image *controlPointGrid = con.F3dContent::GetControlPointGrid();
    auto approxLinearEnergy = controlPointGrid->nz > 1 ? reg_spline_approxLinearEnergy_gpu<true> :
                                                         reg_spline_approxLinearEnergy_gpu<false>;
    return approxLinearEnergy(controlPointGrid, con.GetControlPointGridCuda());
}
/* *************************************************************** */
void CudaCompute::ApproxLinearEnergyGradient(float weight) {
    CudaF3dContent& con = dynamic_cast<CudaF3dContent&>(this->con);
    const nifti_image *controlPointGrid = con.F3dContent::GetControlPointGrid();
    auto approxLinearEnergyGradient = controlPointGrid->nz > 1 ? reg_spline_approxLinearEnergyGradient_gpu<true> :
                                                                 reg_spline_approxLinearEnergyGradient_gpu<false>;
    approxLinearEnergyGradient(controlPointGrid, con.GetControlPointGridCuda(), con.GetTransformationGradientCuda(), weight);
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
    CudaDefContent& con = dynamic_cast<CudaDefContent&>(this->con);
    reg_getImageGradient_gpu(con.DefContent::GetFloating(),
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
    return Cuda::GetMaximalLength(con.GetTransformationGradientCuda(), voxelsPerVolume, optimiseX, optimiseY, optimiseZ);
}
/* *************************************************************** */
void CudaCompute::NormaliseGradient(double maxGradLength, bool optimiseX, bool optimiseY, bool optimiseZ) {
    if (maxGradLength == 0 || (!optimiseX && !optimiseY && !optimiseZ)) return;
    CudaF3dContent& con = dynamic_cast<CudaF3dContent&>(this->con);
    const size_t voxelsPerVolume = NiftiImage::calcVoxelNumber(con.F3dContent::GetTransformationGradient(), 3);
    Cuda::NormaliseGradient(con.GetTransformationGradientCuda(), voxelsPerVolume, static_cast<float>(maxGradLength), optimiseX, optimiseY, optimiseZ);
}
/* *************************************************************** */
void CudaCompute::SmoothGradient(float sigma) {
    if (sigma == 0) return;
    sigma = fabs(sigma);
    CudaF3dContent& con = dynamic_cast<CudaF3dContent&>(this->con);
    Cuda::KernelConvolution<ConvKernelType::Gaussian>(con.F3dContent::GetTransformationGradient(), con.GetTransformationGradientCuda(), &sigma);
}
/* *************************************************************** */
void CudaCompute::GetApproximatedGradient(InterfaceOptimiser& opt) {
    // TODO Implement this for CUDA
    // Use CPU temporarily
    Compute::GetApproximatedGradient(opt);
}
/* *************************************************************** */
void CudaCompute::GetDefFieldFromVelocityGrid(const bool updateStepNumber) {
    CudaF3dContent& con = dynamic_cast<CudaF3dContent&>(this->con);
    reg_spline_getDefFieldFromVelocityGrid_gpu(con.F3dContent::GetControlPointGrid(),
                                               con.F3dContent::GetDeformationField(),
                                               con.GetControlPointGridCuda(),
                                               con.GetDeformationFieldCuda(),
                                               updateStepNumber);
}
/* *************************************************************** */
void CudaCompute::ConvolveImage(const nifti_image *image, float4 *imageCuda) {
    const nifti_image *controlPointGrid = dynamic_cast<F3dContent&>(con).F3dContent::GetControlPointGrid();
    constexpr ConvKernelType kernelType = ConvKernelType::Cubic;
    float currentNodeSpacing[3];
    currentNodeSpacing[0] = currentNodeSpacing[1] = currentNodeSpacing[2] = controlPointGrid->dx;
    bool activeAxis[3] = { 1, 0, 0 };
    Cuda::KernelConvolution<kernelType>(image,
                                        imageCuda,
                                        currentNodeSpacing,
                                        nullptr, // all volumes are considered as active
                                        activeAxis);
    // Convolution along the y axis
    currentNodeSpacing[0] = currentNodeSpacing[1] = currentNodeSpacing[2] = controlPointGrid->dy;
    activeAxis[0] = 0;
    activeAxis[1] = 1;
    Cuda::KernelConvolution<kernelType>(image,
                                        imageCuda,
                                        currentNodeSpacing,
                                        nullptr, // all volumes are considered as active
                                        activeAxis);
    // Convolution along the z axis if required
    if (image->nz > 1) {
        currentNodeSpacing[0] = currentNodeSpacing[1] = currentNodeSpacing[2] = controlPointGrid->dz;
        activeAxis[1] = 0;
        activeAxis[2] = 1;
        Cuda::KernelConvolution<kernelType>(image,
                                            imageCuda,
                                            currentNodeSpacing,
                                            nullptr, // all volumes are considered as active
                                            activeAxis);
    }
}
/* *************************************************************** */
void CudaCompute::VoxelCentricToNodeCentric(float weight) {
    CudaF3dContent& con = dynamic_cast<CudaF3dContent&>(this->con);
    const mat44 *reorientation = Content::GetIJKMatrix(*con.Content::GetFloating());
    reg_voxelCentric2NodeCentric_gpu(con.F3dContent::GetTransformationGradient(),
                                     con.F3dContent::GetVoxelBasedMeasureGradient(),
                                     con.GetTransformationGradientCuda(),
                                     con.GetVoxelBasedMeasureGradientCuda(),
                                     weight,
                                     reorientation);
}
/* *************************************************************** */
void CudaCompute::ConvolveVoxelBasedMeasureGradient(float weight) {
    CudaDefContent& con = dynamic_cast<CudaDefContent&>(this->con);
    ConvolveImage(con.DefContent::GetVoxelBasedMeasureGradient(), con.GetVoxelBasedMeasureGradientCuda());
    // The node-based NMI gradient is extracted from the voxel-based gradient
    VoxelCentricToNodeCentric(weight);
}
/* *************************************************************** */
void CudaCompute::ExponentiateGradient(Content& conBwIn) {
    // TODO Implement this for CUDA
    // Use CPU temporarily
    Compute::ExponentiateGradient(conBwIn);
    // Transfer the data back to the CUDA device
    dynamic_cast<CudaDefContent&>(con).UpdateVoxelBasedMeasureGradient();
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
void CudaCompute::DefFieldCompose(const nifti_image *defField) {
    CudaContent& con = dynamic_cast<CudaContent&>(this->con);
    const size_t& voxelNumber = NiftiImage::calcVoxelNumber(defField, 3);
    thrust::device_vector<float4> defFieldCuda(voxelNumber);
    Cuda::TransferNiftiToDevice(defFieldCuda.data().get(), defField);
    reg_defField_compose_gpu(defField, defFieldCuda.data().get(), con.GetDeformationFieldCuda());
}
/* *************************************************************** */
