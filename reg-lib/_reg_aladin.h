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
#include "_reg_globalTrans.h"
#include "_reg_nmi.h"
#include "_reg_ssd.h"
#include "_reg_tools.h"
#include "float.h"
#include <limits>

class AladinContent;
class Platform;
class Kernel;

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
class reg_aladin
{
public:
    reg_aladin(int platformCodeIn);
    virtual ~reg_aladin();
    GetStringMacro(executableName)

    //No allocating of the images here...
    void SetInputReference(nifti_image *inputRefIn);
    void SetInputFloating(nifti_image *inputFloIn);
    void SetInputReferenceMask(nifti_image *input);

    void SetNumberOfLevels(unsigned levelNumber);
    unsigned GetNumberOfLevels();

    void SetLevelsToPerform(unsigned lp);
    unsigned GetLevelsToPerform();

    void SetReferenceSigma(float sigma);
    void SetFloatingSigma(float sigma);

    mat44* GetTransformationMatrix();

    void SetReferenceLowerThreshold(float th);
    void SetReferenceUpperThreshold(float th);
    void SetFloatingLowerThreshold(float th);
    void SetFloatingUpperThreshold(float th);

    void SetBlockStepSize(int bss);
    void SetBlockPercentage(unsigned bss);
    void SetInlierLts(unsigned bss);
    void SetInputTransform(mat44* mat44In);
    void SetGpuIdx(unsigned gpuIdxIn);
    nifti_image *GetFinalWarpedImage();

    SetMacro(maxIterations,unsigned int)
    GetMacro(maxIterations,unsigned int)

    SetMacro(performRigid,bool)
    GetMacro(performRigid,bool)
    BooleanMacro(performRigid, bool)

    SetMacro(performAffine,bool)
    GetMacro(performAffine,bool)
    BooleanMacro(performAffine, bool)

    GetMacro(alignCentre,bool)
    SetMacro(alignCentre,bool)
    BooleanMacro(alignCentre, bool)
    GetMacro(alignCentreGravity,bool)
    SetMacro(alignCentreGravity,bool)
    BooleanMacro(alignCentreGravity, bool)

    SetClampMacro(interpolation,int,0,3)
    GetMacro(interpolation, int)

    virtual void SetInputFloatingMask(nifti_image*);

    void SetCaptureRangeVox(int captureRangeIn);

    virtual int Check();
    virtual int Print();
    void Run();

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

protected:
    char *executableName;

    bool verbose;

    unsigned int maxIterations;

    unsigned int currentLevel;

    bool performRigid;
    bool performAffine;
    int captureRangeVox;

    bool alignCentre;
    bool alignCentreGravity;

    int interpolation;

    AladinContent *con;
    ////////////////////////////////////
    bool TestMatrixConvergence(mat44 *mat);

    virtual void InitialiseRegistration();
    //virtual void ClearCurrentImagePyramid();
    virtual void ClearPyramid();
    virtual void ClearBlockMatchingParams();

    virtual void GetDeformationField();
    virtual void GetWarpedImage(int interp = 1);
    virtual void UpdateTransformationMatrix(int);

    void (*funcProgressCallback)(float pcntProgress, void *params);
    void *paramsProgressCallback;

    //platform factory methods
    virtual void InitCurrentLevel(unsigned int cl);

    //virtual void clearAladinContent();
    virtual void AllocateImages();
    virtual void ClearAllocatedImages();

    virtual void CreateKernels();
    virtual void ClearKernels();

private:
    Kernel *affineTransformation3DKernel,*blockMatchingKernel;
    Kernel *optimiseKernel, *resamplingKernel;
    void ResolveMatrix(unsigned int iterations,
                       const unsigned int optimizationFlag);
};

#include "_reg_aladin.cpp"
#endif // _REG_ALADIN_H
