/*
 * @file _reg_aladin.h
 * @author Marc Modat
 * @date 08/12/2011
 *
 * Copyright (c) 2009, University College London. All rights reserved.
 * Centre for Medical Image Computing (CMIC)
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_ALADIN_H
#define _REG_ALADIN_H
#define CONVERGENCE_EPS 0.00001
#define RIGID 0
#define AFFINE 1
#include "_reg_macros.h"
#include "_reg_resampling.h"
#include "_reg_blockMatching.h"
#include "_reg_globalTransformation.h"
#include "_reg_nmi.h"
#include "_reg_ssd.h"
#include "_reg_tools.h"
#include "float.h"
#include <limits>

class Content;
class Platform;
class Kernel;

template<class T>
class reg_aladin
{
protected:
	char *executableName;
	nifti_image *InputReference;
	nifti_image *InputFloating;
	nifti_image *InputReferenceMask;
	nifti_image **ReferencePyramid;
	nifti_image **FloatingPyramid;
	int **ReferenceMaskPyramid;
	nifti_image *CurrentReference;
	nifti_image *CurrentFloating;
	nifti_image *CurrentWarped;
	nifti_image *deformationFieldImage;
	int *CurrentReferenceMask;
	int *activeVoxelNumber;

	char *InputTransformName;
	mat44 *TransformationMatrix;

	bool Verbose;

	unsigned int MaxIterations;

	unsigned int CurrentLevel;
	unsigned int NumberOfLevels;
	unsigned int LevelsToPerform;

	bool PerformRigid;
	bool PerformAffine;
	bool ils;
	bool cusvd;
	int captureRangeVox;

	int BlockPercentage;
	int InlierLts;
	int BlockStepSize;
	_reg_blockMatchingParam *blockMatchingParams;

	bool AlignCentre;
	bool AlignCentreGravity;

	int Interpolation;

	float FloatingSigma;
	float ReferenceSigma;

	float ReferenceUpperThreshold;
	float ReferenceLowerThreshold;
	float FloatingUpperThreshold;
	float FloatingLowerThreshold;
	int clIdx;

	Platform *platform;

	bool TestMatrixConvergence(mat44 *mat);

	virtual void InitialiseRegistration();
	virtual void SetCurrentImages();
	virtual void ClearCurrentInputImage();
	virtual void AllocateWarpedImage();
	virtual void ClearWarpedImage();

	virtual void GetDeformationField();
	virtual void GetWarpedImage(int);
	virtual void UpdateTransformationMatrix(int);

	void (*funcProgressCallback)(float pcntProgress, void *params);
	void *paramsProgressCallback;

	//platform factory methods
	virtual void initContent(nifti_image *ref,
									 nifti_image *flo,
									 int *mask,
									 mat44 *transMat,
									 size_t bytes,
									 unsigned int blockPercentage,
									 unsigned int inlierLts,
									 unsigned int blockStepSize);
	virtual void initContent(nifti_image *ref,
									 nifti_image *flo,
									 int *mask,
									 mat44 *transMat,
									 size_t bytes);
	virtual void clearContent();
	virtual void createKernels();
	virtual void clearKernels();

public:
	reg_aladin();
	virtual ~reg_aladin();
	GetStringMacro(executableName)

	int platformCode;

	void setPlatformCode(const int platformCodeIn)
	{
		platformCode = platformCodeIn;
	}
	void setIls(const bool ilsIn)
	{
		ils = ilsIn;
	}
	void setCusvd(const bool cusvdIn)
	{
		cusvd = cusvdIn;
	}

	//No allocating of the images here...
	void SetInputReference(nifti_image *input)
	{
		this->InputReference = input;
	}
	nifti_image *GetInputReference()
	{
		return this->InputReference;
	}
	void SetInputFloating(nifti_image *input)
	{
		this->InputFloating = input;
	}
	nifti_image *GetInputFloating()
	{
		return this->InputFloating;
	}

	void SetInputMask(nifti_image *input)
	{
		this->InputReferenceMask = input;
	}
	nifti_image *GetInputMask()
	{
		return this->InputReferenceMask;
	}

	void SetInputTransform(const char *filename);
	mat44 *GetInputTransform()
	{
		return this->InputTransform;
	}

	mat44 *GetTransformationMatrix()
	{
		return this->TransformationMatrix;
	}
	nifti_image *GetFinalWarpedImage();

	SetMacro(MaxIterations,unsigned int)
	GetMacro(MaxIterations,unsigned int)

	SetMacro(NumberOfLevels,unsigned int)
	GetMacro(NumberOfLevels,unsigned int)

	SetMacro(LevelsToPerform,unsigned int)
	GetMacro(LevelsToPerform,unsigned int)

	SetMacro(BlockPercentage,int)
	GetMacro(BlockPercentage,int)

	SetMacro(BlockStepSize,int)
	GetMacro(BlockStepSize,int)

	SetMacro(InlierLts,float)
	GetMacro(InlierLts,float)

	SetMacro(ReferenceSigma,float)
	GetMacro(ReferenceSigma,float)

	SetMacro(ReferenceUpperThreshold,float)
	GetMacro(ReferenceUpperThreshold,float)
	SetMacro(ReferenceLowerThreshold,float)
	GetMacro(ReferenceLowerThreshold,float)

	SetMacro(FloatingUpperThreshold,float)
	GetMacro(FloatingUpperThreshold,float)
	SetMacro(FloatingLowerThreshold,float)
	GetMacro(FloatingLowerThreshold,float)

	SetMacro(FloatingSigma,float)
	GetMacro(FloatingSigma,float)

	SetMacro(PerformRigid,int)
	GetMacro(PerformRigid,int)
	BooleanMacro(PerformRigid, int)

	SetMacro(PerformAffine,int)
	GetMacro(PerformAffine,int)
	BooleanMacro(PerformAffine, int)

	GetMacro(AlignCentre,int)
	SetMacro(AlignCentre,int)
	BooleanMacro(AlignCentre, int)
	GetMacro(AlignCentreGravity,int)
	SetMacro(AlignCentreGravity,int)
	BooleanMacro(AlignCentreGravity, int)

	SetClampMacro(Interpolation,int,0,3)
	GetMacro(Interpolation, int)

	virtual void SetInputFloatingMask(nifti_image*)
	{
		reg_print_fct_warn("reg_aladin::SetInputFloatingMask()");
		reg_print_msg_warn("Floating mask not used in the asymmetric global registration");
	}
	void SetInterpolationToNearestNeighbor()
	{
		this->SetInterpolation(0);
	}
	void SetInterpolationToTrilinear()
	{
		this->SetInterpolation(1);
	}
	void SetInterpolationToCubic()
	{
		this->SetInterpolation(3);
	}
	void setCaptureRangeVox(int captureRangeIn)
	{
		this->captureRangeVox = captureRangeIn;
	}

	void setClIdx(int clIdxIn) {
		this->clIdx = clIdxIn;
	}
	virtual int Check();
	virtual int Print();
	virtual void Run();

	virtual void DebugPrintLevelInfoStart();
	virtual void DebugPrintLevelInfoEnd();
	virtual void SetVerbose(bool _verbose);

	void SetProgressCallbackFunction(void (*funcProgCallback)(float pcntProgress,
																				 void *params),
												void *paramsProgCallback)
	{
		funcProgressCallback = funcProgCallback;
		paramsProgressCallback = paramsProgCallback;
	}
	Content *con;

private:
	Kernel *affineTransformation3DKernel,*blockMatchingKernel;
	Kernel *optimiseKernel, *resamplingKernel;
	void resolveMatrix(unsigned int iterations,
							 const unsigned int optimizationFlag);
};

#include "_reg_aladin.cpp"
#endif // _REG_ALADIN_H
