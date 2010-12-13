/*
 *  _reg_f3d.h
 *
 *
 *  Created by Marc Modat on 19/11/2010.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_F3D_H
#define _REG_F3D_H

#include "_reg_resampling.h"
#include "_reg_affineTransformation.h"
#include "_reg_bspline.h"
#include "_reg_bspline_comp.h"
#include "_reg_mutualinformation.h"
#include "_reg_ssd.h"
#include "_reg_tools.h"
#include "float.h"
#include <limits>


template <class T>
class reg_f3d
{
  protected:

    int referenceTimePoint;
    int floatingTimePoint;
    nifti_image *inputReference; // pointer to external
    nifti_image *inputFloating; // pointer to external
    nifti_image *inputControlPointGrid; // pointer to external
    nifti_image *maskImage; // pointer to external
    mat44 *affineTransformation; // pointer to external
    int *referenceMask;
    nifti_image *controlPointGrid;
	T bendingEnergyWeight;
	bool bendingEnergyApproximation;
	T jacobianLogWeight;
	bool jacobianLogApproximation;
    unsigned int maxiterationNumber;
	T referenceSmoothingSigma;
	T floatingSmoothingSigma;
    float *referenceThresholdUp;
    float *referenceThresholdLow;
    float *floatingThresholdUp;
    float *floatingThresholdLow;
    unsigned int *referenceBinNumber;
    unsigned int *floatingBinNumber;
	T warpedPaddingValue;
	T spacing[3];
	unsigned int levelNumber;
	unsigned int levelToPerform;
	T gradientSmoothingSigma;
    bool useComposition;
    bool useSSD;
    bool useConjGradient;
    bool verbose;
//	int threadNumber;

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
	nifti_image *nodeBasedMeasureGradientImage;
	T *conjugateG;
	T *conjugateH;
	T *bestControlPointPosition;
	double *probaJointHistogram;
	double *logJointHistogram;
    double entropies[4];
	T *maxSSD;
	unsigned int currentLevel;
    unsigned totalBinNumber;

	virtual int AllocateWarped();
	virtual int ClearWarped();
    virtual int AllocateDeformationField();
    virtual int ClearDeformationField();
    virtual int AllocateWarpedGradient();
    virtual int ClearWarpedGradient();
    virtual int AllocateVoxelBasedMeasureGradient();
    virtual int ClearVoxelBasedMeasureGradient();
    virtual int AllocateNodeBasedMeasureGradient();
    virtual int ClearNodeBasedMeasureGradient();
    virtual int AllocateConjugateGradientVariables();
    virtual int ClearConjugateGradientVariables();
    virtual int AllocateBestControlPointArray();
    virtual int ClearBestControlPointArray();
    virtual int AllocateJointHistogram();
    virtual int ClearJointHistogram();
    virtual int AllocateCurrentInputImage();
    virtual int ClearCurrentInputImage();

    virtual int SaveCurrentControlPoint();
    virtual int RestoreCurrentControlPoint();
    virtual double ComputeJacobianBasedPenaltyTerm(int);
    virtual double ComputeBendingEnergyPenaltyTerm();
    virtual int WarpFloatingImage(int);
    virtual double ComputeSimilarityMeasure();
    virtual int GetSimilarityMeasureGradient();
    virtual int GetBendingEnergyGradient();
    virtual int GetJacobianBasedGradient();
    virtual int ComputeConjugateGradient(unsigned int );
    virtual T GetMaximalGradientLength();
    virtual int UpdateControlPointPosition(T);

public:
    reg_f3d(int refTimePoint,int floTimePoint);
    virtual ~reg_f3d();

	int SetReferenceImage(nifti_image *);
	int SetFloatingImage(nifti_image *);
	int SetControlPointGridImage(nifti_image *);
	int SetReferenceMask(nifti_image *);
	int SetAffineTransformation(mat44 *);
	int SetBendingEnergyWeight(T);
	int ApproximateBendingEnergy();
	int DoNotApproximateBendingEnergy();
	int SetJacobianLogWeight(T);
	int ApproximateJacobianLog();
	int DoNotApproximateJacobianLog();
	int SetReferenceSmoothingSigma(T);
	int SetFloatingSmoothingSigma(T);
    int SetReferenceThresholdUp(unsigned int,T);
    int SetReferenceThresholdLow(unsigned int,T);
    int SetFloatingThresholdUp(unsigned int, T);
    int SetFloatingThresholdLow(unsigned int,T);
    int SetWarpedPaddingValue(T);
    int SetSpacing(unsigned int ,T);
    int SetLevelNumber(unsigned int);
    int SetLevelToPerform(unsigned int);
	int SetGradientSmoothingSigma(T);
	int UseComposition();
    int DoNotUseComposition();
    int UseSSD();
    int DoNotUseSSD();
    int UseConjugateGradient();
    int DoNotUseConjugateGradient();
    int PrintOutInformation();
    int DoNotPrintOutInformation();
    int SetMaximalIterationNumber(unsigned int);
    int SetReferenceBinNumber(int, unsigned int);
    int SetFloatingBinNumber(int, unsigned int);
//	int SetThreadNumber(int t);

    int CheckParameters_f3d();
    int Initisalise_f3d();
    int Run_f3d();
    virtual int CheckMemoryMB_f3d(){return 0;};
	nifti_image *GetWarpedImage();
    nifti_image *GetControlPointPositionImage();
};

#include "_reg_f3d.cpp"

#endif
