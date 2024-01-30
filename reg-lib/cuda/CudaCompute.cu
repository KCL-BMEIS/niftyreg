#include "CudaCompute.h"
#include "CudaF3dContent.h"
#include "CudaGlobalTransformation.hpp"
#include "CudaKernelConvolution.hpp"
#include "CudaLocalTransformation.hpp"
#include "CudaNormaliseGradient.hpp"
#include "CudaResampling.hpp"
#include "CudaOptimiser.hpp"

/* *************************************************************** */
void CudaCompute::ResampleImage(int interpolation, float paddingValue) {
    CudaContent& con = dynamic_cast<CudaContent&>(this->con);
    const nifti_image *floating = con.Content::GetFloating();
    auto resampleImage = floating->nz > 1 ? Cuda::ResampleImage<true> : Cuda::ResampleImage<false>;
    resampleImage(floating,
                  con.GetFloatingCuda(),
                  con.Content::GetWarped(),
                  con.GetWarpedCuda(),
                  con.Content::GetDeformationField(),
                  con.GetDeformationFieldCuda(),
                  con.GetReferenceMaskCuda(),
                  con.GetActiveVoxelNumber(),
                  interpolation,
                  paddingValue);
}
/* *************************************************************** */
double CudaCompute::GetJacobianPenaltyTerm(bool approx) {
    CudaF3dContent& con = dynamic_cast<CudaF3dContent&>(this->con);
    return Cuda::GetJacobianPenaltyTerm(con.F3dContent::GetReference(),
                                        con.F3dContent::GetControlPointGrid(),
                                        con.GetControlPointGridCuda(),
                                        approx);
}
/* *************************************************************** */
void CudaCompute::JacobianPenaltyTermGradient(float weight, bool approx) {
    CudaF3dContent& con = dynamic_cast<CudaF3dContent&>(this->con);
    Cuda::GetJacobianPenaltyTermGradient(con.F3dContent::GetReference(),
                                         con.F3dContent::GetControlPointGrid(),
                                         con.GetControlPointGridCuda(),
                                         con.GetTransformationGradientCuda(),
                                         weight,
                                         approx);
}
/* *************************************************************** */
double CudaCompute::CorrectFolding(bool approx) {
    CudaF3dContent& con = dynamic_cast<CudaF3dContent&>(this->con);
    return Cuda::CorrectFolding(con.F3dContent::GetReference(),
                                con.F3dContent::GetControlPointGrid(),
                                con.GetControlPointGridCuda(),
                                approx);
}
/* *************************************************************** */
double CudaCompute::ApproxBendingEnergy() {
    CudaF3dContent& con = dynamic_cast<CudaF3dContent&>(this->con);
    const nifti_image *controlPointGrid = con.F3dContent::GetControlPointGrid();
    auto approxBendingEnergy = controlPointGrid->nz > 1 ? Cuda::ApproxBendingEnergy<true> :
                                                          Cuda::ApproxBendingEnergy<false>;
    return approxBendingEnergy(controlPointGrid, con.GetControlPointGridCuda());
}
/* *************************************************************** */
void CudaCompute::ApproxBendingEnergyGradient(float weight) {
    CudaF3dContent& con = dynamic_cast<CudaF3dContent&>(this->con);
    nifti_image *controlPointGrid = con.F3dContent::GetControlPointGrid();
    auto approxBendingEnergyGradient = controlPointGrid->nz > 1 ? Cuda::ApproxBendingEnergyGradient<true> :
                                                                  Cuda::ApproxBendingEnergyGradient<false>;
    approxBendingEnergyGradient(controlPointGrid,
                                con.GetControlPointGridCuda(),
                                con.GetTransformationGradientCuda(),
                                weight);
}
/* *************************************************************** */
double CudaCompute::ApproxLinearEnergy() {
    CudaF3dContent& con = dynamic_cast<CudaF3dContent&>(this->con);
    const nifti_image *controlPointGrid = con.F3dContent::GetControlPointGrid();
    auto approxLinearEnergy = controlPointGrid->nz > 1 ? Cuda::ApproxLinearEnergy<true> :
                                                         Cuda::ApproxLinearEnergy<false>;
    return approxLinearEnergy(controlPointGrid, con.GetControlPointGridCuda());
}
/* *************************************************************** */
void CudaCompute::ApproxLinearEnergyGradient(float weight) {
    CudaF3dContent& con = dynamic_cast<CudaF3dContent&>(this->con);
    const nifti_image *controlPointGrid = con.F3dContent::GetControlPointGrid();
    auto approxLinearEnergyGradient = controlPointGrid->nz > 1 ? Cuda::ApproxLinearEnergyGradient<true> :
                                                                 Cuda::ApproxLinearEnergyGradient<false>;
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
    decltype(Cuda::GetDeformationField<true, true>) *getDeformationField;
    if (composition)
        getDeformationField = bspline ? Cuda::GetDeformationField<true, true> :
                                        Cuda::GetDeformationField<true, false>;
    else
        getDeformationField = bspline ? Cuda::GetDeformationField<false, true> :
                                        Cuda::GetDeformationField<false, false>;
    CudaF3dContent& con = dynamic_cast<CudaF3dContent&>(this->con);
    getDeformationField(con.F3dContent::GetControlPointGrid(),
                        con.F3dContent::GetReference(),
                        con.GetControlPointGridCuda(),
                        con.GetDeformationFieldCuda(),
                        con.GetReferenceMaskCuda(),
                        con.GetActiveVoxelNumber());
}
/* *************************************************************** */
template<bool optimiseX, bool optimiseY, bool optimiseZ>
inline void UpdateControlPointPosition(float4 *currentDofCuda,
                                       cudaTextureObject_t bestDofTexture,
                                       cudaTextureObject_t gradientTexture,
                                       const size_t nVoxels,
                                       const float scale) {
    thrust::for_each_n(thrust::device, thrust::make_counting_iterator(0), nVoxels, [=]__device__(const int index) {
        float4 dofValue = currentDofCuda[index]; scale; // To capture scale
        const float4 bestValue = tex1Dfetch<float4>(bestDofTexture, index);
        const float4 gradValue = tex1Dfetch<float4>(gradientTexture, index);
        if constexpr (optimiseX)
            dofValue.x = bestValue.x + scale * gradValue.x;
        if constexpr (optimiseY)
            dofValue.y = bestValue.y + scale * gradValue.y;
        if constexpr (optimiseZ)
            dofValue.z = bestValue.z + scale * gradValue.z;
        currentDofCuda[index] = dofValue;
    });
}
/* *************************************************************** */
template<bool optimiseX, bool optimiseY>
static inline void UpdateControlPointPosition(float4 *currentDofCuda,
                                              cudaTextureObject_t bestDofTexture,
                                              cudaTextureObject_t gradientTexture,
                                              const size_t nVoxels,
                                              const float scale,
                                              const bool optimiseZ) {
    auto updateControlPointPosition = UpdateControlPointPosition<optimiseX, optimiseY, true>;
    if (!optimiseZ) updateControlPointPosition = UpdateControlPointPosition<optimiseX, optimiseY, false>;
    updateControlPointPosition(currentDofCuda, bestDofTexture, gradientTexture, nVoxels, scale);
}
/* *************************************************************** */
template<bool optimiseX>
static inline void UpdateControlPointPosition(float4 *currentDofCuda,
                                              cudaTextureObject_t bestDofTexture,
                                              cudaTextureObject_t gradientTexture,
                                              const size_t nVoxels,
                                              const float scale,
                                              const bool optimiseY,
                                              const bool optimiseZ) {
    auto updateControlPointPosition = UpdateControlPointPosition<optimiseX, true>;
    if (!optimiseY) updateControlPointPosition = UpdateControlPointPosition<optimiseX, false>;
    updateControlPointPosition(currentDofCuda, bestDofTexture, gradientTexture, nVoxels, scale, optimiseZ);
}
/* *************************************************************** */
void CudaCompute::UpdateControlPointPosition(float *currentDof,
                                             const float *bestDof,
                                             const float *gradient,
                                             const float scale,
                                             const bool optimiseX,
                                             const bool optimiseY,
                                             const bool optimiseZ) {
    const nifti_image *controlPointGrid = dynamic_cast<CudaF3dContent&>(con).F3dContent::GetControlPointGrid();
    const bool is3d = controlPointGrid->nz > 1;
    const size_t nVoxels = NiftiImage::calcVoxelNumber(controlPointGrid, 3);
    auto bestDofTexturePtr = Cuda::CreateTextureObject(reinterpret_cast<const float4*>(bestDof), nVoxels, cudaChannelFormatKindFloat, 4);
    auto gradientTexturePtr = Cuda::CreateTextureObject(reinterpret_cast<const float4*>(gradient), nVoxels, cudaChannelFormatKindFloat, 4);

    auto updateControlPointPosition = ::UpdateControlPointPosition<true>;
    if (!optimiseX) updateControlPointPosition = ::UpdateControlPointPosition<false>;
    updateControlPointPosition(reinterpret_cast<float4*>(currentDof), *bestDofTexturePtr, *gradientTexturePtr,
                               nVoxels, scale, optimiseY, is3d ? optimiseZ : false);
}
/* *************************************************************** */
void CudaCompute::GetImageGradient(int interpolation, float paddingValue, int activeTimePoint) {
    CudaDefContent& con = dynamic_cast<CudaDefContent&>(this->con);
    const nifti_image *floating = con.Content::GetFloating();
    auto getImageGradient = floating->nz > 1 ? Cuda::GetImageGradient<true> : Cuda::GetImageGradient<false>;
    getImageGradient(floating,
                     con.GetFloatingCuda(),
                     con.GetDeformationFieldCuda(),
                     con.DefContent::GetWarpedGradient(),
                     con.GetWarpedGradientCuda(),
                     interpolation,
                     paddingValue,
                     activeTimePoint);
}
/* *************************************************************** */
double CudaCompute::GetMaximalLength(bool optimiseX, bool optimiseY, bool optimiseZ) {
    if (!optimiseX && !optimiseY && !optimiseZ) return 0;
    CudaF3dContent& con = dynamic_cast<CudaF3dContent&>(this->con);
    nifti_image *transGrad = con.F3dContent::GetTransformationGradient();
    const size_t voxelsPerVolume = NiftiImage::calcVoxelNumber(transGrad, 3);
    if (transGrad->nz <= 1) optimiseZ = false;
    return Cuda::GetMaximalLength(con.GetTransformationGradientCuda(), voxelsPerVolume, optimiseX, optimiseY, optimiseZ);
}
/* *************************************************************** */
void CudaCompute::NormaliseGradient(double maxGradLength, bool optimiseX, bool optimiseY, bool optimiseZ) {
    if (maxGradLength == 0 || (!optimiseX && !optimiseY && !optimiseZ)) return;
    CudaF3dContent& con = dynamic_cast<CudaF3dContent&>(this->con);
    nifti_image *transGrad = con.F3dContent::GetTransformationGradient();
    const size_t voxelsPerVolume = NiftiImage::calcVoxelNumber(transGrad, 3);
    if (transGrad->nz <= 1) optimiseZ = false;
    Cuda::NormaliseGradient(con.GetTransformationGradientCuda(), voxelsPerVolume, maxGradLength, optimiseX, optimiseY, optimiseZ);
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
    Cuda::GetDefFieldFromVelocityGrid(con.F3dContent::GetControlPointGrid(),
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
    const nifti_image *transGrad = con.F3dContent::GetTransformationGradient();
    auto voxelCentricToNodeCentric = transGrad->nz > 1 ? Cuda::VoxelCentricToNodeCentric<true> :
                                                         Cuda::VoxelCentricToNodeCentric<false>;
    voxelCentricToNodeCentric(transGrad,
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
    CudaF3dContent& con = dynamic_cast<CudaF3dContent&>(this->con);
    CudaF3dContent& conBw = dynamic_cast<CudaF3dContent&>(conBwIn);
    nifti_image *deformationField = con.Content::GetDeformationField();
    nifti_image *voxelBasedMeasureGradient = con.DefContent::GetVoxelBasedMeasureGradient();
    float4 *voxelBasedMeasureGradientCuda = con.GetVoxelBasedMeasureGradientCuda();
    nifti_image *controlPointGridBw = conBw.F3dContent::GetControlPointGrid();
    float4 *controlPointGridBwCuda = conBw.GetControlPointGridCuda();
    mat44 *affineTransformationBw = conBw.Content::GetTransformationMatrix();
    const int compNum = std::abs(static_cast<int>(controlPointGridBw->intent_p2)); // The number of composition

    /* Allocate a temporary gradient image to store the backward gradient */
    const size_t voxelGradNumber = NiftiImage::calcVoxelNumber(voxelBasedMeasureGradient, 3);
    NiftiImage warped(voxelBasedMeasureGradient, NiftiImage::Copy::ImageInfo);
    thrust::device_vector<float4> warpedCudaVec(voxelGradNumber);

    // Create all deformation field images needed for resampling
    const size_t defFieldNumber = NiftiImage::calcVoxelNumber(deformationField, 3);
    vector<NiftiImage> defFields(compNum + 1, NiftiImage(deformationField, NiftiImage::Copy::ImageInfo));
    vector<thrust::device_vector<float4>> defFieldCudaVecs(compNum + 1, thrust::device_vector<float4>(defFieldNumber));

    // Generate all intermediate deformation fields
    Cuda::GetIntermediateDefFieldFromVelGrid(controlPointGridBw, controlPointGridBwCuda, defFields, defFieldCudaVecs);

    // Remove the affine component
    NiftiImage affineDisp;
    thrust::device_vector<float4> affineDispCudaVec;
    if (affineTransformationBw) {
        affineDisp = NiftiImage(deformationField, NiftiImage::Copy::ImageInfo);
        affineDispCudaVec.resize(defFieldNumber);
        Cuda::GetAffineDeformationField(affineTransformationBw, affineDisp, affineDispCudaVec.data().get());
        Cuda::GetDisplacementFromDeformation(affineDisp, affineDispCudaVec.data().get());
    }

    auto resampleGradient = voxelBasedMeasureGradient->nz > 1 ? Cuda::ResampleGradient<true> : Cuda::ResampleGradient<false>;
    for (int i = 0; i < compNum; i++) {
        if (affineTransformationBw)
            Cuda::SubtractImages(defFields[i], defFieldCudaVecs[i].data().get(), affineDispCudaVec.data().get());
        resampleGradient(voxelBasedMeasureGradient, voxelBasedMeasureGradientCuda,    // Floating
                         warped, warpedCudaVec.data().get(),  // Output
                         defFields[i], defFieldCudaVecs[i].data().get(),
                         con.GetReferenceMaskCuda(),
                         con.GetActiveVoxelNumber(),
                         1,   // Interpolation type - linear
                         0);  // Padding value
        Cuda::AddImages(voxelBasedMeasureGradient, voxelBasedMeasureGradientCuda, warpedCudaVec.data().get());
    }

    // Normalise the forward gradient
    Cuda::MultiplyValue(voxelGradNumber, voxelBasedMeasureGradientCuda, 1.f / powf(2.f, static_cast<float>(compNum)));
}
/* *************************************************************** */
Cuda::UniquePtr<float4> CudaCompute::ScaleGradient(const float4 *transGradCuda, const size_t voxelNumber, const float scale) {
    float4 *scaledGradient;
    Cuda::Allocate(&scaledGradient, voxelNumber);
    Cuda::MultiplyValue(voxelNumber, transGradCuda, scaledGradient, scale);
    return Cuda::UniquePtr<float4>(scaledGradient);
}
/* *************************************************************** */
void CudaCompute::UpdateVelocityField(float scale, bool optimiseX, bool optimiseY, bool optimiseZ) {
    if (!optimiseX && !optimiseY && !optimiseZ) return;

    CudaF3dContent& con = dynamic_cast<CudaF3dContent&>(this->con);
    const nifti_image *controlPointGrid = con.F3dContent::GetControlPointGrid();
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(controlPointGrid, 3);
    auto scaledGradientCudaPtr = ScaleGradient(con.GetTransformationGradientCuda(), voxelNumber, scale);

    // Reset the gradient along the axes if appropriate
    if (controlPointGrid->nu < 3) optimiseZ = true;
    Cuda::SetGradientToZero(scaledGradientCudaPtr.get(), voxelNumber, !optimiseX, !optimiseY, !optimiseZ);

    // Update the velocity field
    Cuda::AddImages(controlPointGrid, con.GetControlPointGridCuda(), scaledGradientCudaPtr.get());
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
    CudaF3dContent& con = dynamic_cast<CudaF3dContent&>(this->con);
    CudaF3dContent& conBw = dynamic_cast<CudaF3dContent&>(conBwIn);

    nifti_image *controlPointGrid = con.F3dContent::GetControlPointGrid();
    nifti_image *controlPointGridBw = conBw.F3dContent::GetControlPointGrid();
    float4 *controlPointGridCuda = con.GetControlPointGridCuda();
    float4 *controlPointGridBwCuda = conBw.GetControlPointGridCuda();
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(controlPointGrid, 3);

    // In order to ensure symmetry, the forward and backward velocity fields
    // are averaged in both image spaces: reference and floating

    // Both parametrisations are converted into displacement
    Cuda::GetDisplacementFromDeformation(controlPointGrid, controlPointGridCuda);
    Cuda::GetDisplacementFromDeformation(controlPointGridBw, controlPointGridBwCuda);

    // Backup the backward displacement field
    thrust::device_ptr<float4> controlPointGridBwCudaPtr(controlPointGridBwCuda);
    thrust::device_vector<float4> controlPointGridBwOrgCudaVec(controlPointGridBwCudaPtr, controlPointGridBwCudaPtr + voxelNumber);

    // Both parametrisations are subtracted (sum and negation)
    Cuda::SubtractImages(controlPointGridBw, controlPointGridBwCuda, controlPointGridCuda);
    Cuda::SubtractImages(controlPointGrid, controlPointGridCuda, controlPointGridBwOrgCudaVec.data().get());

    // Divide by 2
    Cuda::MultiplyValue(voxelNumber, controlPointGridCuda, 0.5f);
    Cuda::MultiplyValue(voxelNumber, controlPointGridBwCuda, 0.5f);

    // Convert the velocity field from displacement to deformation
    Cuda::GetDeformationFromDisplacement(controlPointGrid, controlPointGridCuda);
    Cuda::GetDeformationFromDisplacement(controlPointGridBw, controlPointGridBwCuda);
}
/* *************************************************************** */
void CudaCompute::DefFieldCompose(const nifti_image *defField) {
    CudaContent& con = dynamic_cast<CudaContent&>(this->con);
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(defField, 3);
    thrust::device_vector<float4> defFieldCuda(voxelNumber);
    Cuda::TransferNiftiToDevice(defFieldCuda.data().get(), defField);
    auto defFieldCompose = defField->nz > 1 ? Cuda::DefFieldCompose<true> : Cuda::DefFieldCompose<false>;
    defFieldCompose(defField, defFieldCuda.data().get(), con.GetDeformationFieldCuda());
}
/* *************************************************************** */
NiftiImage CudaCompute::ResampleGradient(int interpolation, float padding) {
    CudaDefContent& con = dynamic_cast<CudaDefContent&>(this->con);
    const nifti_image *voxelBasedMeasureGradient = con.DefContent::GetVoxelBasedMeasureGradient();
    auto resampleGradient = voxelBasedMeasureGradient->nz > 1 ? Cuda::ResampleGradient<true> : Cuda::ResampleGradient<false>;
    resampleGradient(voxelBasedMeasureGradient,
                     con.GetVoxelBasedMeasureGradientCuda(),
                     voxelBasedMeasureGradient,
                     con.GetWarpedGradientCuda(),
                     con.Content::GetDeformationField(),
                     con.GetDeformationFieldCuda(),
                     con.GetReferenceMaskCuda(),
                     con.GetActiveVoxelNumber(),
                     interpolation,
                     padding);
    return NiftiImage(con.GetWarpedGradient(), NiftiImage::Copy::Image);
}
/* *************************************************************** */
void CudaCompute::GetAffineDeformationField(bool compose) {
    CudaContent& con = dynamic_cast<CudaContent&>(this->con);
    auto getAffineDeformationField = compose ? Cuda::GetAffineDeformationField<true> :
                                               Cuda::GetAffineDeformationField<false>;
    getAffineDeformationField(con.Content::GetTransformationMatrix(),
                              con.Content::GetDeformationField(),
                              con.GetDeformationFieldCuda());
}
/* *************************************************************** */
