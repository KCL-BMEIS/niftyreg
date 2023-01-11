/*
 * @file _reg_aladin.h
 * @author Marc Modat
 * @date 08/12/2011
 *
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#pragma once

#define CONVERGENCE_EPS 0.00001
#define RIGID 0
#define AFFINE 1

#include "_reg_macros.h"
#include "_reg_resampling.h"
#include "_reg_blockMatching.h"
#include "_reg_globalTrans.h"
#include "_reg_nmi.h"
#include "_reg_ssd.h"
#include "_reg_tools.h"
#include "_reg_ReadWriteMatrix.h"
#include "_reg_stringFormat.h"
#include "Platform.h"
#include "AffineDeformationFieldKernel.h"
#include "ResampleImageKernel.h"
#include "BlockMatchingKernel.h"
#include "OptimiseKernel.h"
#include "ConvolutionKernel.h"
#include "AladinContent.h"

#ifdef _USE_CUDA
#include "CudaAladinContent.h"
#endif
#ifdef _USE_OPENCL
#include "ClAladinContent.h"
#include "InfoDevice.h"
#endif

/**
 * @brief Block matching registration class
 *
 * Main algorithm of Ourselin et al.
 * The essence of the algorithm is as follows:
 * - Subdivide the reference image into a number of blocks and find
 *   the block in the warped image that is most similar.
 * - Get the point pair between the reference and the warped image block
 *   for the most similar block.
 *
 * reference: Pointer to the nifti reference image.
 * warped: Pointer to the nifti warped image.
 *
 *
 * block_size: Size of the block.
 * block_half_width: Half-width of the search neighborhood.
 * delta_1: Spacing between two consecutive blocks
 * delta_2: Sub-sampling value for a block
 *
 * Possible improvement: Take care of anisotropic data. Right now, we specify
 * the block size, neighborhood and the step sizes in voxels and it would be
 * better to specify it in millimeters and take the voxel size into account.
 * However, it would be more efficient to calculate this once (outside this
 * module) and pass these values for each axes. For the time being, we do this
 * simple implementation.
 */
template<class T>
class reg_aladin {
protected:
    char *executableName;
    nifti_image *inputReference;
    nifti_image *inputFloating;
    nifti_image *inputReferenceMask;
    nifti_image **referencePyramid;
    nifti_image **floatingPyramid;
    int **referenceMaskPyramid;
    int *activeVoxelNumber; ///TODO Needs to be removed

    char *inputTransformName;
    mat44 *transformationMatrix;

    bool verbose;

    unsigned int maxIterations;

    unsigned int currentLevel;
    unsigned int numberOfLevels;
    unsigned int levelsToPerform;

    bool performRigid;
    bool performAffine;
    int captureRangeVox;

    int blockPercentage;
    int inlierLts;
    int blockStepSize;
    _reg_blockMatchingParam *blockMatchingParams;

    bool alignCentre;
    int alignCentreMass;

    int interpolation;

    float floatingSigma;
    float referenceSigma;

    float referenceUpperThreshold;
    float referenceLowerThreshold;
    float floatingUpperThreshold;
    float floatingLowerThreshold;
    float warpedPaddingValue;

    Platform *platform;
    PlatformType platformType;
    unsigned gpuIdx;

    bool TestMatrixConvergence(mat44 *mat);

    virtual void InitialiseRegistration();
    virtual void DeallocateCurrentInputImage();

    virtual void GetDeformationField();
    virtual void GetWarpedImage(int, float padding);
    virtual void UpdateTransformationMatrix(int);

    void (*funcProgressCallback)(float pcntProgress, void *params);
    void *paramsProgressCallback;

    //platform factory methods
    virtual void InitAladinContent(nifti_image *ref,
                                   nifti_image *flo,
                                   int *mask,
                                   mat44 *transMat,
                                   size_t bytes,
                                   unsigned int blockPercentage = 0,
                                   unsigned int inlierLts = 0,
                                   unsigned int blockStepSize = 0);
    virtual void DeinitAladinContent();
    virtual void CreateKernels();
    virtual void DeallocateKernels();

public:
    reg_aladin();
    virtual ~reg_aladin();
    GetStringMacro(ExecutableName, executableName);

    //No allocating of the images here...
    void SetInputReference(nifti_image *input) {
        this->inputReference = input;
    }
    nifti_image* GetInputReference() {
        return this->inputReference;
    }
    void SetInputFloating(nifti_image *input) {
        this->inputFloating = input;
    }
    nifti_image* GetInputFloating() {
        return this->inputFloating;
    }

    void SetInputMask(nifti_image *input) {
        this->inputReferenceMask = input;
    }
    nifti_image* GetInputMask() {
        return this->inputReferenceMask;
    }

    void SetInputTransform(const char *filename);
    char* GetInputTransform() {
        return this->inputTransformName;
    }

    mat44* GetTransformationMatrix() {
        return this->transformationMatrix;
    }
    nifti_image* GetFinalWarpedImage();

    void SetPlatformType(const PlatformType& platformTypeIn) {
        this->platformType = platformTypeIn;
    }
    void SetGpuIdx(unsigned gpuIdxIn) {
        this->gpuIdx = gpuIdxIn;
    }

    SetMacro(MaxIterations, maxIterations, unsigned int);
    GetMacro(MaxIterations, maxIterations, unsigned int);

    SetMacro(NumberOfLevels, numberOfLevels, unsigned int);
    GetMacro(NumberOfLevels, numberOfLevels, unsigned int);

    SetMacro(LevelsToPerform, levelsToPerform, unsigned int);
    GetMacro(LevelsToPerform, levelsToPerform, unsigned int);

    SetMacro(BlockPercentage, blockPercentage, int);
    GetMacro(BlockPercentage, blockPercentage, int);

    SetMacro(BlockStepSize, blockStepSize, int);
    GetMacro(BlockStepSize, blockStepSize, int);

    SetMacro(InlierLts, inlierLts, int);
    GetMacro(InlierLts, inlierLts, int);

    SetMacro(ReferenceSigma, referenceSigma, float);
    GetMacro(ReferenceSigma, referenceSigma, float);

    SetMacro(ReferenceUpperThreshold, referenceUpperThreshold, float);
    GetMacro(ReferenceUpperThreshold, referenceUpperThreshold, float);
    SetMacro(ReferenceLowerThreshold, referenceLowerThreshold, float);
    GetMacro(ReferenceLowerThreshold, referenceLowerThreshold, float);

    SetMacro(FloatingUpperThreshold, floatingUpperThreshold, float);
    GetMacro(FloatingUpperThreshold, floatingUpperThreshold, float);
    SetMacro(FloatingLowerThreshold, floatingLowerThreshold, float);
    GetMacro(FloatingLowerThreshold, floatingLowerThreshold, float);

    SetMacro(WarpedPaddingValue, warpedPaddingValue, float);
    GetMacro(WarpedPaddingValue, warpedPaddingValue, float);

    SetMacro(FloatingSigma, floatingSigma, float);
    GetMacro(FloatingSigma, floatingSigma, float);

    SetMacro(PerformRigid, performRigid, bool);
    GetMacro(PerformRigid, performRigid, bool);
    BooleanMacro(PerformRigid, bool);

    SetMacro(PerformAffine, performAffine, bool);
    GetMacro(PerformAffine, performAffine, bool);
    BooleanMacro(PerformAffine, bool);

    GetMacro(AlignCentre, alignCentre, bool);
    SetMacro(AlignCentre, alignCentre, bool);
    BooleanMacro(AlignCentre, bool);
    GetMacro(AlignCentreMass, alignCentreMass, int);
    SetMacro(AlignCentreMass, alignCentreMass, int);

    SetClampMacro(Interpolation, interpolation, int, 0, 3);
    GetMacro(Interpolation, interpolation, int);

    virtual void SetInputFloatingMask(nifti_image*) {
        reg_print_fct_warn("reg_aladin::SetInputFloatingMask()");
        reg_print_msg_warn("Floating mask not used in the asymmetric global registration");
    }
    void SetInterpolationToNearestNeighbor() {
        this->SetInterpolation(0);
    }
    void SetInterpolationToTrilinear() {
        this->SetInterpolation(1);
    }
    void SetInterpolationToCubic() {
        this->SetInterpolation(3);
    }
    void SetCaptureRangeVox(int captureRangeIn) {
        this->captureRangeVox = captureRangeIn;
    }

    virtual int Check();
    virtual int Print();
    virtual void Run();

    virtual void DebugPrintLevelInfoStart();
    virtual void DebugPrintLevelInfoEnd();
    virtual void SetVerbose(bool _verbose);

    void SetProgressCallbackFunction(void (*funcProgCallback)(float pcntProgress, void *params),
                                     void *paramsProgCallback) {
        funcProgressCallback = funcProgCallback;
        paramsProgressCallback = paramsProgCallback;
    }
    AladinContent *con;

private:
    Kernel *affineTransformation3DKernel, *blockMatchingKernel;
    Kernel *optimiseKernel, *resamplingKernel;
    void ResolveMatrix(unsigned int iterations, const unsigned int optimizationFlag);
};
