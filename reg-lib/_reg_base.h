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
#include "Optimiser.hpp"
#include "Platform.h"

/// @brief Base registration class
template<class T>
class reg_base: public InterfaceOptimiser {
protected:
    // Platform
    unique_ptr<Platform> platform;

    // Content
    unique_ptr<Content> con;

    // Compute
    unique_ptr<Compute> compute;

    // Measure
    unique_ptr<Measure> measure;

    // Optimiser-related variables
    unique_ptr<Optimiser<T>> optimiser;
    size_t maxIterationNumber;
    size_t perturbationNumber;
    bool optimiseX;
    bool optimiseY;
    bool optimiseZ;

    // Measure-related variables
    unique_ptr<reg_ssd> measure_ssd;
    unique_ptr<reg_kld> measure_kld;
    unique_ptr<reg_dti> measure_dti;
    unique_ptr<reg_lncc> measure_lncc;
    unique_ptr<reg_nmi> measure_nmi;
    unique_ptr<reg_mind> measure_mind;
    unique_ptr<reg_mindssc> measure_mindssc;
    NiftiImage localWeightSimInput;

    char *executableName;
    int referenceTimePoints;
    int floatingTimePoints;
    NiftiImage inputReference;
    NiftiImage inputFloating;
    NiftiImage maskImage;
    unique_ptr<mat44> affineTransformation;
    T referenceSmoothingSigma;
    T floatingSmoothingSigma;
    unique_ptr<T[]> referenceThresholdUp;
    unique_ptr<T[]> referenceThresholdLow;
    unique_ptr<T[]> floatingThresholdUp;
    unique_ptr<T[]> floatingThresholdLow;
    bool robustRange;
    float warpedPaddingValue;
    unsigned levelNumber;
    unsigned levelToPerform;
    T gradientSmoothingSigma;
    T similarityWeight;
    bool useConjGradient;
    bool useApproxGradient;
    bool verbose;
    bool usePyramid;
    int interpolation;

    bool initialised;
    vector<NiftiImage> referencePyramid;
    vector<NiftiImage> floatingPyramid;
    vector<unique_ptr<int[]>> maskPyramid;

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
    reg_base(int refTimePoints, int floTimePoints);

    virtual void Run();
    virtual vector<NiftiImage> GetWarpedImage() = 0;
    virtual char* GetExecutableName() { return executableName; }
    virtual bool GetSymmetricStatus() { return false; }

    // Platform
    virtual void SetPlatformType(const PlatformType platformType) {
        platform.reset(new Platform(platformType));
        measure.reset(platform->CreateMeasure());
    }
    virtual void SetGpuIdx(const unsigned gpuIdx) { platform->SetGpuIdx(gpuIdx); }

    // Optimisation-related functions
    virtual void SetMaximalIterationNumber(unsigned);
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
    virtual void SetLNCCKernelType(ConvKernelType type);
    virtual void SetLocalWeightSim(NiftiImage);

    virtual void SetNMIWeight(int, double);
    virtual void SetSSDWeight(int, double);
    virtual void SetKLDWeight(int, double);
    virtual void SetLNCCWeight(int, double);

    virtual void SetReferenceImage(NiftiImage);
    virtual void SetFloatingImage(NiftiImage);
    virtual void SetReferenceMask(NiftiImage);
    virtual void SetAffineTransformation(const mat44&);
    virtual void SetReferenceSmoothingSigma(T);
    virtual void SetFloatingSmoothingSigma(T);
    virtual void SetGradientSmoothingSigma(T);
    virtual void SetReferenceThresholdUp(unsigned, T);
    virtual void SetReferenceThresholdLow(unsigned, T);
    virtual void SetFloatingThresholdUp(unsigned, T);
    virtual void SetFloatingThresholdLow(unsigned, T);
    virtual void UseRobustRange();
    virtual void DoNotUseRobustRange();
    virtual void SetWarpedPaddingValue(float);
    virtual void SetLevelNumber(unsigned);
    virtual void SetLevelToPerform(unsigned);
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
};
