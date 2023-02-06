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
#include "Platform.h"

/// @brief Base registration class
template<class T>
class reg_base: public InterfaceOptimiser {
protected:
    // Platform
    Platform *platform;
    PlatformType platformType;
    unsigned gpuIdx;

    // Content
    Content *con = nullptr;

    // Compute
    Compute *compute = nullptr;

    // Measure
    Measure *measure = nullptr;

    // Optimiser-related variables
    reg_optimiser<T> *optimiser;
    size_t maxIterationNumber;
    size_t perturbationNumber;
    bool optimiseX;
    bool optimiseY;
    bool optimiseZ;

    // Measure-related variables
    reg_ssd *measure_ssd;
    reg_kld *measure_kld;
    reg_dti *measure_dti;
    reg_lncc *measure_lncc;
    reg_nmi *measure_nmi;
    reg_mind *measure_mind;
    reg_mindssc *measure_mindssc;
    nifti_image *localWeightSimInput;

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

    double bestWMeasure;
    double currentWMeasure;

    double currentWLand;
    double bestWLand;

    float landmarkRegWeight;
    size_t landmarkRegNumber;
    float *landmarkReference;
    float *landmarkFloating;

    // For the NiftyReg plugin in NiftyView
    void (*funcProgressCallback)(float pcntProgress, void *params);
    void* paramsProgressCallback;

    virtual void WarpFloatingImage(int);
    virtual double ComputeSimilarityMeasure();
    virtual void GetVoxelBasedGradient();
    virtual void InitialiseSimilarity();
    virtual void CheckParameters();
    virtual void Initialise();

    // Pure virtual functions
    virtual void SetOptimiser() = 0;
    virtual T InitCurrentLevel(int) = 0;
    virtual void DeinitCurrentLevel(int);
    virtual void SmoothGradient() = 0;
    virtual void GetDeformationField() = 0;
    virtual void GetApproximatedGradient() = 0;
    virtual double GetObjectiveFunctionValue() = 0;
    virtual void UpdateParameters(float) = 0;
    virtual T NormaliseGradient() = 0;
    virtual void GetSimilarityMeasureGradient() = 0;
    virtual void GetObjectiveFunctionGradient() = 0;
    virtual void DisplayCurrentLevelParameters(int) = 0;
    virtual void UpdateBestObjFunctionValue() = 0;
    virtual void PrintCurrentObjFunctionValue(T) = 0;
    virtual void PrintInitialObjFunctionValue() = 0;
    virtual void CorrectTransformation() = 0;

public:
    reg_base(int refTimePoint, int floTimePoint);
    virtual ~reg_base();

    virtual void Run();
    virtual nifti_image** GetWarpedImage() = 0;
    virtual char* GetExecutableName() { return executableName; }
    virtual bool GetSymmetricStatus() { return false; }

    // Platform
    virtual void SetPlatformType(const PlatformType& platformTypeIn) { platformType = platformTypeIn; }
    virtual void SetGpuIdx(unsigned gpuIdxIn) { gpuIdx = gpuIdxIn; }

    // Optimisation-related functions
    virtual void SetMaximalIterationNumber(unsigned int);
    virtual void NoOptimisationAlongX() { optimiseX = false; }
    virtual void NoOptimisationAlongY() { optimiseY = false; }
    virtual void NoOptimisationAlongZ() { optimiseZ = false; }
    virtual void SetPerturbationNumber(size_t v) { perturbationNumber = v; }
    virtual void UseConjugateGradient();
    virtual void DoNotUseConjugateGradient();
    virtual void UseApproximatedGradient();
    virtual void DoNotUseApproximatedGradient();
    // Measure of similarity-related functions
    // virtual void ApproximateParzenWindow();
    // virtual void DoNotApproximateParzenWindow();
    virtual void UseNMISetReferenceBinNumber(int, int);
    virtual void UseNMISetFloatingBinNumber(int, int);
    virtual void UseSSD(int, bool);
    virtual void UseMIND(int, int);
    virtual void UseMINDSSC(int, int);
    virtual void UseKLDivergence(int);
    virtual void UseDTI(bool*);
    virtual void UseLNCC(int, float);
    virtual void SetLNCCKernelType(int type);
    virtual void SetLocalWeightSim(nifti_image*);

    virtual void SetNMIWeight(int, double);
    virtual void SetSSDWeight(int, double);
    virtual void SetKLDWeight(int, double);
    virtual void SetLNCCWeight(int, double);

    virtual void SetReferenceImage(nifti_image*);
    virtual void SetFloatingImage(nifti_image*);
    virtual void SetReferenceMask(nifti_image*);
    virtual void SetAffineTransformation(mat44*);
    virtual void SetReferenceSmoothingSigma(T);
    virtual void SetFloatingSmoothingSigma(T);
    virtual void SetGradientSmoothingSigma(T);
    virtual void SetReferenceThresholdUp(unsigned int, T);
    virtual void SetReferenceThresholdLow(unsigned int, T);
    virtual void SetFloatingThresholdUp(unsigned int, T);
    virtual void SetFloatingThresholdLow(unsigned int, T);
    virtual void UseRobustRange();
    virtual void DoNotUseRobustRange();
    virtual void SetWarpedPaddingValue(float);
    virtual void SetLevelNumber(unsigned int);
    virtual void SetLevelToPerform(unsigned int);
    virtual void PrintOutInformation();
    virtual void DoNotPrintOutInformation();
    virtual void DoNotUsePyramidalApproach();
    virtual void UseNearestNeighborInterpolation();
    virtual void UseLinearInterpolation();
    virtual void UseCubicSplineInterpolation();
    virtual void SetLandmarkRegularisationParam(size_t, float*, float*, float);

    // For the NiftyReg plugin in NiftyView
    virtual void SetProgressCallbackFunction(void (*funcProgCallback)(float pcntProgress, void *params),
                                             void *paramsProgCallback) {
        funcProgressCallback = funcProgCallback;
        paramsProgressCallback = paramsProgCallback;
    }

    // For testing
    virtual void reg_test_setOptimiser(reg_optimiser<T> *opt) { optimiser = opt; }
};
