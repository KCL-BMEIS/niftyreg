/**
 * @file _reg_base.h
 * @author Marc Modat
 * @date 15/11/2012
 *
 * Copyright (c) 2012, University College London. All rights reserved.
 * Centre for Medical Image Computing (CMIC)
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_BASE_H
#define _REG_BASE_H

#include "_reg_resampling.h"
#include "_reg_globalTransformation.h"
#include "_reg_localTransformation.h"
#include "_reg_mutualinformation.h"
#include "_reg_ssd.h"
#include "_reg_KLdivergence.h"
#include "_reg_tools.h"
#include "_reg_ReadWriteImage.h"
#include "_reg_optimiser.h"
#include "float.h"
#include <limits>

template <class T>
class reg_base : public InterfaceOptimiser
{
protected:
    reg_optimiser<T> *optimiser;
    size_t maxiterationNumber;
    size_t perturbationNumber;
    bool optimiseX;
    bool optimiseY;
    bool optimiseZ;
    virtual void SetOptimiser();

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
    T warpedPaddingValue;
    unsigned int levelNumber;
    unsigned int levelToPerform;
    T gradientSmoothingSigma;
    T similarityWeight;
    bool additive_mc_nmi;
    bool useSSD;
    bool useKLD;
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
    nifti_image *currentReference;
    nifti_image *currentFloating;
    int *currentMask;
    nifti_image *warped;
    nifti_image *deformationFieldImage;
    nifti_image *warpedGradientImage;
    nifti_image *voxelBasedMeasureGradientImage;
    unsigned int currentLevel;

    unsigned int *referenceBinNumber;
    unsigned int *floatingBinNumber;
    unsigned int totalBinNumber;
    double *probaJointHistogram;
    double *logJointHistogram;
    double entropies[4];
    bool approxParzenWindow;
    T *maxSSD;
    double bestWMeasure;
    double currentWMeasure;

    virtual void AllocateWarped();
    virtual void ClearWarped();
    virtual void AllocateDeformationField();
    virtual void ClearDeformationField();
    virtual void AllocateWarpedGradient();
    virtual void ClearWarpedGradient();
    virtual void AllocateJointHistogram();
    virtual void ClearJointHistogram();
    virtual void AllocateVoxelBasedMeasureGradient();
    virtual void ClearVoxelBasedMeasureGradient();
    virtual T InitialiseCurrentLevel(){return 0.;}
    virtual void ClearCurrentInputImage();

    virtual void WarpFloatingImage(int);
    virtual double ComputeSimilarityMeasure();
    virtual void GetVoxelBasedGradient();
    virtual void SmoothGradient(){return;}

    // Virtual empty functions that have to be filled
    virtual void GetDeformationField()
        {return;} // Need to be filled
    virtual void SetGradientImageToZero()
        {return;} // Need to be filled
    virtual void GetApproximatedGradient()
        {return;} // Need to be filled
    virtual double GetObjectiveFunctionValue()
        {return std::numeric_limits<double>::quiet_NaN();} // Need to be filled
    virtual void UpdateParameters(float)
        {return;} // Need to be filled
    virtual T NormaliseGradient()
        {return std::numeric_limits<float>::quiet_NaN();} // Need to be filled
    virtual void GetSimilarityMeasureGradient()
        {return;} // Need to be filled
    virtual void GetObjectiveFunctionGradient()
        {return;} // Need to be filled
    virtual void DisplayCurrentLevelParameters()
        {return;} // Need to be filled
    virtual void UpdateBestObjFunctionValue()
        {return;} // Need to be filled
    virtual void PrintCurrentObjFunctionValue(T)
        {return;} // Need to be filled
    virtual void PrintInitialObjFunctionValue()
        {return;} // Need to be filled
    virtual void AllocateTransformationGradient()
        {return;} // Need to be filled
    virtual void ClearTransformationGradient()
        {return;} // Need to be filled
    virtual void CorrectTransformation()
        {return;} // Need to be filled

    void (*funcProgressCallback)(float pcntProgress, void *params);
    void *paramsProgressCallback;

public:
    reg_base(int refTimePoint,int floTimePoint);
    virtual ~reg_base();

    void SetMaximalIterationNumber(unsigned int);
    void NoOptimisationAlongX(){this->optimiseX=false;}
    void NoOptimisationAlongY(){this->optimiseY=false;}
    void NoOptimisationAlongZ(){this->optimiseZ=false;}
    void SetPerturbationNumber(size_t v){this->perturbationNumber=v;}
    void SetReferenceImage(nifti_image *);
    void SetFloatingImage(nifti_image *);
    void SetReferenceMask(nifti_image *);
    void SetAffineTransformation(mat44 *);
    void SetReferenceSmoothingSigma(T);
    void SetFloatingSmoothingSigma(T);
    void SetGradientSmoothingSigma(T);
    void SetReferenceThresholdUp(unsigned int,T);
    void SetReferenceThresholdLow(unsigned int,T);
    void SetFloatingThresholdUp(unsigned int, T);
    void SetFloatingThresholdLow(unsigned int,T);
    void SetWarpedPaddingValue(T);
    void SetLevelNumber(unsigned int);
    void SetLevelToPerform(unsigned int);
    void UseConjugateGradient();
    void DoNotUseConjugateGradient();
    void UseApproximatedGradient();
    void DoNotUseApproximatedGradient();
    void PrintOutInformation();
    void DoNotPrintOutInformation();
    void DoNotUsePyramidalApproach();
    void UseNeareatNeighborInterpolation();
    void UseLinearInterpolation();
    void UseCubicSplineInterpolation();
    void SetReferenceBinNumber(int, unsigned int);
    void SetFloatingBinNumber(int, unsigned int);
    void ApproximateParzenWindow();
    void DoNotApproximateParzenWindow();
    void UseSSD();
    void DoNotUseSSD();
    void UseKLDivergence();
    void DoNotUseKLDivergence();
    void SetAdditiveMC(){this->additive_mc_nmi=true;}

    virtual void CheckParameters();
    void Run();
    virtual void Initisalise();
    virtual nifti_image **GetWarpedImage(){return NULL;} // Need to be filled

    // Function required for the NiftyReg pluggin in NiftyView
    void SetProgressCallbackFunction(void (*funcProgCallback)(float pcntProgress,
                                                              void *params),
                                     void *paramsProgCallback){
        funcProgressCallback = funcProgCallback;
        paramsProgressCallback = paramsProgCallback;
    }
};

#include "_reg_base.cpp"

#endif // _REG_BASE_H
