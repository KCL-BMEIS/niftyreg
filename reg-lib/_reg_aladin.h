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
#include "Platform.h"
#include "AffineDeformationFieldKernel.h"
#include "ResampleImageKernel.h"
#include "BlockMatchingKernel.h"
#include "LtsKernel.h"
#include "ConvolutionKernel.h"
#include "AladinContent.h"

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
 * better to specify it in millimetres and take the voxel size into account.
 * However, it would be more efficient to calculate this once (outside this
 * module) and pass these values for each axes. For the time being, we do this
 * simple implementation.
 */
template<class T>
class reg_aladin {
protected:
    char *executableName;
    NiftiImage inputReference;
    NiftiImage inputFloating;
    NiftiImage inputReferenceMask;
    vector<NiftiImage> referencePyramid;
    vector<NiftiImage> floatingPyramid;
    vector<unique_ptr<int[]>> referenceMaskPyramid;

    char *inputTransformName;
    unique_ptr<mat44> affineTransformation;

    bool verbose;

    unsigned maxIterations;

    unsigned currentLevel;
    unsigned numberOfLevels;
    unsigned levelsToPerform;

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

    unique_ptr<Platform> platform;
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
                                   unsigned blockPercentage = 0,
                                   unsigned inlierLts = 0,
                                   unsigned blockStepSize = 0);
    virtual void DeinitAladinContent();
    virtual void CreateKernels();
    virtual void DeallocateKernels();

public:
    unique_ptr<AladinContent> con;

    reg_aladin();
    GetStringMacro(ExecutableName, executableName);

    //No allocating of the images here...
    void SetInputReference(NiftiImage input) {
        this->inputReference = input;
    }
    NiftiImage GetInputReference() {
        return this->inputReference;
    }
    void SetInputFloating(NiftiImage input) {
        this->inputFloating = input;
    }
    NiftiImage GetInputFloating() {
        return this->inputFloating;
    }

    void SetInputMask(NiftiImage input) {
        this->inputReferenceMask = input;
    }
    NiftiImage GetInputMask() {
        return this->inputReferenceMask;
    }

    void SetInputTransform(const char *filename);
    char* GetInputTransform() {
        return this->inputTransformName;
    }

    const mat44* GetTransformationMatrix() {
        return this->affineTransformation.get();
    }
    NiftiImage GetFinalWarpedImage();

    void SetPlatformType(const PlatformType platformTypeIn) {
        this->platformType = platformTypeIn;
    }
    void SetGpuIdx(unsigned gpuIdxIn) {
        this->gpuIdx = gpuIdxIn;
    }

    SetMacro(MaxIterations, maxIterations, unsigned);
    GetMacro(MaxIterations, maxIterations, unsigned);

    SetMacro(NumberOfLevels, numberOfLevels, unsigned);
    GetMacro(NumberOfLevels, numberOfLevels, unsigned);

    SetMacro(LevelsToPerform, levelsToPerform, unsigned);
    GetMacro(LevelsToPerform, levelsToPerform, unsigned);

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
        NR_WARN_WFCT("Floating mask not used in the asymmetric global registration");
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
    virtual void Print();
    virtual void Run();

    virtual void DebugPrintLevelInfoStart();
    virtual void DebugPrintLevelInfoEnd();
    virtual void SetVerbose(bool _verbose);

    void SetProgressCallbackFunction(void (*funcProgCallback)(float pcntProgress, void *params),
                                     void *paramsProgCallback) {
        funcProgressCallback = funcProgCallback;
        paramsProgressCallback = paramsProgCallback;
    }

private:
    unique_ptr<Kernel> affineTransformation3DKernel, blockMatchingKernel, ltsKernel, resamplingKernel;
    void ResolveMatrix(unsigned iterations, const unsigned optimizationFlag);
};
