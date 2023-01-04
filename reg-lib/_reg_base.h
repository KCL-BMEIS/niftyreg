/**
 * @file _reg_base.h
 * @author Marc Modat
 * @date 15/11/2012
 *
 *  Copyright (c) 2012-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#pragma once

#include "_reg_resampling.h"
#include "_reg_globalTrans.h"
#include "_reg_localTrans.h"
#include "_reg_localTrans_jac.h"
#include "_reg_localTrans_regul.h"
#include "_reg_nmi.h"
#include "_reg_dti.h"
#include "_reg_ssd.h"
#include "_reg_mind.h"
#include "_reg_kld.h"
#include "_reg_lncc.h"
#include "_reg_tools.h"
#include "_reg_ReadWriteImage.h"
#include "_reg_stringFormat.h"
#include "_reg_optimiser.h"
#include "float.h"
#include "Platform.h"

/// @brief Base registration class
template<class T>
class reg_base: public InterfaceOptimiser {
protected:
    // Platform
    Platform *platform;
    int platformCode;
    unsigned gpuIdx;

    // Content
    Content *con = nullptr;

    // Compute
    Compute *compute = nullptr;

    // Optimiser related variables
    reg_optimiser<T> *optimiser;
    size_t maxIterationNumber;
    size_t perturbationNumber;
    bool optimiseX;
    bool optimiseY;
    bool optimiseZ;

    // Optimiser related function
    virtual void SetOptimiser() = 0;

    // Measure related variables
    reg_ssd *measure_ssd;
    reg_kld *measure_kld;
    reg_dti *measure_dti;
    reg_lncc *measure_lncc;
    reg_nmi *measure_nmi;
    reg_mind *measure_mind;
    reg_mindssc *measure_mindssc;
    nifti_image *localWeightSimInput;
    // nifti_image *localWeightSimCurrent;

    char *executableName;
    int referenceTimePoint;
    int floatingTimePoint;
    nifti_image *inputReference; // pointer to external
    nifti_image *inputFloating; // pointer to external
    nifti_image *maskImage; // pointer to external
    mat44 *affineTransformation; // pointer to external
    int *referenceMask;
    T referenceSmoothingSigma;
    T floatingSmoothingSigma;
    float *referenceThresholdUp;
    float *referenceThresholdLow;
    float *floatingThresholdUp;
    float *floatingThresholdLow;
    bool robustRange;
    float warpedPaddingValue;
    unsigned int levelNumber;
    unsigned int levelToPerform;
    T gradientSmoothingSigma;
    T similarityWeight;
    bool additive_mc_nmi;
    bool useConjGradient;
    bool useApproxGradient;
    bool verbose;
    bool usePyramid;
    int interpolation;

    bool initialised;
    nifti_image **referencePyramid;
    nifti_image **floatingPyramid;
    int **maskPyramid;
    int *activeVoxelNumber;
    // nifti_image *reference;
    // nifti_image *floating;
    // int *currentMask;
    // nifti_image *warped;
    // nifti_image *deformationFieldImage;
    // nifti_image *warpedGradient;
    // nifti_image *voxelBasedMeasureGradient;
    unsigned int currentLevel;

    mat33 *forwardJacobianMatrix;

    double bestWMeasure;
    double currentWMeasure;

    double currentWLand;
    double bestWLand;

    float landmarkRegWeight;
    size_t landmarkRegNumber;
    float *landmarkReference;
    float *landmarkFloating;

    // virtual void AllocateWarped();
    // virtual void DeallocateWarped();
    // virtual void AllocateDeformationField();
    // virtual void DeallocateDeformationField();
    // virtual void AllocateWarpedGradient();
    // virtual void DeallocateWarpedGradient();
    // virtual void AllocateVoxelBasedMeasureGradient();
    // virtual void DeallocateVoxelBasedMeasureGradient();
    // virtual void DeallocateCurrentInputImage();

    virtual void WarpFloatingImage(int);
    virtual double ComputeSimilarityMeasure();
    virtual void GetVoxelBasedGradient();
    virtual void InitialiseSimilarity();

    // Virtual empty functions that have to be filled
    virtual T InitialiseCurrentLevel(nifti_image *reference) = 0;
    virtual void SmoothGradient() = 0;
    virtual void GetDeformationField() = 0;
    // virtual void SetGradientImageToZero() = 0;
    virtual void GetApproximatedGradient() = 0;
    virtual double GetObjectiveFunctionValue() = 0;
    virtual void UpdateParameters(float) = 0;
    virtual T NormaliseGradient() = 0;
    virtual void GetSimilarityMeasureGradient() = 0;
    virtual void GetObjectiveFunctionGradient() = 0;
    virtual void DisplayCurrentLevelParameters() = 0;
    virtual void UpdateBestObjFunctionValue() = 0;
    virtual void PrintCurrentObjFunctionValue(T) = 0;
    virtual void PrintInitialObjFunctionValue() = 0;
    // virtual void AllocateTransformationGradient() = 0;
    // virtual void DeallocateTransformationGradient() = 0;
    virtual void CorrectTransformation() = 0;

    void (*funcProgressCallback)(float pcntProgress, void *params);
    void* paramsProgressCallback;

public:
    reg_base(int refTimePoint, int floTimePoint);
    virtual ~reg_base();

    // Platform
    Platform* GetPlatform();
    void SetPlatformCode(const int platformCodeIn) { platformCode = platformCodeIn; }
    void SetGpuIdx(unsigned gpuIdxIn) { gpuIdx = gpuIdxIn; }

    // Optimisation related functions
    void SetMaximalIterationNumber(unsigned int);
    void NoOptimisationAlongX() { optimiseX = false; }
    void NoOptimisationAlongY() { optimiseY = false; }
    void NoOptimisationAlongZ() { optimiseZ = false; }
    void SetPerturbationNumber(size_t v) { perturbationNumber = v; }
    void UseConjugateGradient();
    void DoNotUseConjugateGradient();
    void UseApproximatedGradient();
    void DoNotUseApproximatedGradient();
    // Measure of similarity related functions
 //    void ApproximateParzenWindow();
 //    void DoNotApproximateParzenWindow();
    virtual void UseNMISetReferenceBinNumber(int, int);
    virtual void UseNMISetFloatingBinNumber(int, int);
    virtual void UseSSD(int timepoint, bool normalize);
    virtual void UseMIND(int timepoint, int offset);
    virtual void UseMINDSSC(int timepoint, int offset);
    virtual void UseKLDivergence(int timepoint);
    virtual void UseDTI(bool *timepoint);
    virtual void UseLNCC(int timepoint, float stdDevKernel);
    virtual void SetLNCCKernelType(int type);
    void SetLocalWeightSim(nifti_image*);

    void SetNMIWeight(int, double);
    void SetSSDWeight(int, double);
    void SetKLDWeight(int, double);
    void SetLNCCWeight(int, double);

    void SetReferenceImage(nifti_image*);
    void SetFloatingImage(nifti_image*);
    void SetReferenceMask(nifti_image*);
    void SetAffineTransformation(mat44*);
    void SetReferenceSmoothingSigma(T);
    void SetFloatingSmoothingSigma(T);
    void SetGradientSmoothingSigma(T);
    void SetReferenceThresholdUp(unsigned int, T);
    void SetReferenceThresholdLow(unsigned int, T);
    void SetFloatingThresholdUp(unsigned int, T);
    void SetFloatingThresholdLow(unsigned int, T);
    void UseRobustRange();
    void DoNotUseRobustRange();
    void SetWarpedPaddingValue(float);
    void SetLevelNumber(unsigned int);
    void SetLevelToPerform(unsigned int);
    void PrintOutInformation();
    void DoNotPrintOutInformation();
    void DoNotUsePyramidalApproach();
    void UseNearestNeighborInterpolation();
    void UseLinearInterpolation();
    void UseCubicSplineInterpolation();
    void SetLandmarkRegularisationParam(size_t, float*, float*, float);

    virtual void CheckParameters();
    void Run();
    virtual void Initialise();
    virtual void InitContent(nifti_image *reference, nifti_image *floating, int *mask) = 0;
    virtual void DeinitContent() = 0;
    virtual nifti_image** GetWarpedImage() = 0;
    virtual char* GetExecutableName() { return executableName; }
    virtual bool GetSymmetricStatus() { return false; }

    // Function required for the NiftyReg plugin in NiftyView
    void SetProgressCallbackFunction(void (*funcProgCallback)(float pcntProgress, void *params),
                                     void *paramsProgCallback) {
        funcProgressCallback = funcProgCallback;
        paramsProgressCallback = paramsProgCallback;
    }

    // Function used for testing
    virtual void reg_test_setOptimiser(reg_optimiser<T> *opt) { optimiser = opt; }
};
