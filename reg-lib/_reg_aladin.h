/*
 *  _reg_aladin.h
 *
 *
 *  Created by Marc Modat on 08/12/2011.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
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
#include "_reg_mutualinformation.h"
#include "_reg_ssd.h"
#include "_reg_tools.h"
#include "float.h"
#include <limits>

template <class T>
class reg_aladin
{
    protected:
      char *ExecutableName;
      nifti_image* InputReference;
      nifti_image* InputFloating;
      nifti_image *InputMask;
      nifti_image *OutputImage;
      mat44 *InputTransform;
      mat44 *OutputTransform;
      int Verbose;
      int *ActiveVoxelNumber;
      int MaxIterations;
      int BackgroundIndex[3];
      T SourceBackgroundValue;
      float TargetSigma;
      float SourceSigma;
      int NumberOfLevels;
      int LevelsToPerform;
      int BlockPercentage;
      int InlierLts;
      int UseBackgroundIndex;
      int Interpolation; //enumerated
      int PerformRigid; //flag
      int PerformAffine; //flag
      int AlignCentre; //flag
      int ImageDimension; //flag
      int UseInputTransform; //flag
      int UseGpu; //flag
      int UseTargetMask; //flag
      int SmoothTarget;
      int SmoothSource;
      //Maybe need one for Two D Registration

      bool TestMatrixConvergence(mat44 *mat);
      SetMacro(UseBackgroundIndex,int);
      SetMacro(PerformRigid,int);
      SetMacro(PerformAffine,int);
      SetMacro(AlignCentre,int);
      SetMacro(UseGpu,int);
      SetMacro(UseTargetMask,int);
      SetMacro(UseInputTransform,int);
      SetMacro(SmoothTarget,int);
      SetMacro(SmoothSource,int);

    public:
      reg_aladin();
      //reg_aladin(char* reference, char* floating);
~reg_aladin();
      GetStringMacro(ExecutableName);

      //No allocating of the images here...
      void SetInputReference(nifti_image *input) {this->InputReference = input;}
      nifti_image* GetInputReference() {return this->InputReference;}

      void SetInputFloating(nifti_image *input) {this->InputFloating=input;}
      nifti_image *GetInputFloating() {return this->InputFloating;}

      void SetInputMask(nifti_image *input) {this->InputMask=input;}
      nifti_image *GetInputMask() {return this->InputMask;}

      int SetInputTransform(char *filename,int flirtFlag);
      mat44* GetInputTransform() {return this->InputTransform;}

      mat44* GetOutputTransform() {return this->OutputTransform;}
      nifti_image *GetOutputImage() {return this->OutputImage;}

      SetMacro(MaxIterations,int);
      GetMacro(MaxIterations,int);

      SetMacro(NumberOfLevels,int);
      GetMacro(NumberOfLevels,int);

      SetMacro(LevelsToPerform,int);
      GetMacro(LevelsToPerform,int);

      SetMacro(BlockPercentage,int);
      GetMacro(BlockPercentage,int);

      SetMacro(InlierLts,int);
      GetMacro(InlierLts,int);

      SetMacro(TargetSigma,float);
      GetMacro(TargetSigma,float);

      SetMacro(SourceSigma,float);
      GetMacro(SourceSigma,float);

      SetVector3Macro(BackgroundIndex,int);
      GetVector3Macro(BackgroundIndex,int);

      BooleanMacro(UseBackgroundIndex,int);
      GetMacro(UseBackgroundIndex,int);

      BooleanMacro(PerformRigid, int);
      GetMacro(PerformRigid,int);

      BooleanMacro(PerformAffine, int);
      GetMacro(PerformAffine,int);

      BooleanMacro(AlignCentre, int);
      GetMacro(AlignCentre,int);

      BooleanMacro(UseGpu, int);
      GetMacro(UseGpu,int);

      BooleanMacro(UseTargetMask, int);
      GetMacro(UseTargetMask,int);

      BooleanMacro(SmoothTarget,int);
      GetMacro(SmoothTarget,int);

      BooleanMacro(SmoothSource,int);
      GetMacro(SmoothSource,int);

      BooleanMacro(UseInputTransform,int);
      GetMacro(UseInputTransform,int);

      SetClampMacro(ImageDimension,int,2,3);
      GetMacro(ImageDimension,int);

      SetClampMacro(Interpolation,int,0,2);
      GetMacro(Interpolation, int);
      void SetInterpolationToNearestNeighbor() {this->SetInterpolation(0);}
      void SetInterpolationToTrilinear() {this->SetInterpolation(1);}
      void SetInterpolationToCubic() {this->SetInterpolation(2);}

      int Check();
      int Print();
      void Run();


};

#include "_reg_aladin.cpp"
#endif // _REG_ALADIN_H
