#ifndef GLOBALCONTENT_H_
#define GLOBALCONTENT_H_

#include <ctime>
#include <iosfwd>
#include <map>
#include <string>
#include <vector>
#include "Kernel.h"
#include "nifti1_io.h"
#include "_reg_maths.h"
#include "_reg_tools.h"
#include "Platform.h"
#include "Kernel.h"
#include "ResampleImageKernel.h"
#include "ConvolutionKernel.h"

/// @brief shared content between reg_aladin and reg_f3d

class GlobalContent {

public:
    GlobalContent();
    GlobalContent(int platformCodeIn);
    GlobalContent(int platformCodeIn, int refTime, int floTime);
    virtual ~GlobalContent();

    void InitialiseGlobalContent();

    virtual void AllocateWarped();
    virtual void ClearWarped();
    virtual void AllocateDeformationField();
    virtual void ClearDeformationField();

    void AllocateMaskPyramid();
    void ClearMaskPyramid();
    void AllocateActiveVoxelNumber();
    void ClearActiveVoxelNumber();
    void ClearThresholds();

    virtual void ClearCurrentInputImages();
    virtual void ClearCurrentImagePyramid(int currentPyramidLevel);
    virtual void ClearPyramid();

    void CheckParameters();
    //
    //virtual void WarpFloatingImage(int interp = 1);

    //getters
    nifti_image* getInputReference();
    nifti_image* getInputReferenceMask();
    nifti_image* getInputFloating();
    mat44* getAffineTransformation();

    unsigned getNbRefTimePoint();
    unsigned getNbFloTimePoint();

    float getReferenceSmoothingSigma();
    float getFloatingSmoothingSigma();

    bool getRobustRange();

    float* getReferenceThresholdUp();
    float* getReferenceThresholdLow();
    float* getFloatingThresholdUp();
    float* getFloatingThresholdLow();

    nifti_image* getCurrentReference();
    mat44* getCurrentReferenceMatrix_xyz();
    nifti_image* getCurrentFloating();
    mat44* getCurrentFloatingMatrix_xyz();
    float getWarpedPaddingValue();
    virtual int* getCurrentReferenceMask();
    //mat44* getTransformationMatrix();
    virtual nifti_image* getCurrentWarped(int datatype = NIFTI_TYPE_FLOAT32);
    virtual nifti_image* getCurrentDeformationField();

    bool isPyramidUsed();
    unsigned int getLevelNumber();
    unsigned int getLevelToPerform();
    //
    nifti_image** getReferencePyramid();
    nifti_image** getFloatingPyramid();
    int** getMaskPyramid();
    int* getActiveVoxelNumber();
    //Platform
    Platform* getPlatform();

    //setters
    void setInputReference(nifti_image *r);
    void setInputReferenceMask(nifti_image *m);
    void setInputFloating(nifti_image *f);
    void setAffineTransformation(mat44 *a);

    void setNbRefTimePoint(unsigned ntp);
    void setNbFloTimePoint(unsigned ntp);

    void setRobustRange(bool rr);

    void setReferenceSmoothingSigma(float s);
    void setFloatingSmoothingSigma(float s);
    void setReferenceThresholdUp(unsigned int i, float t);
    void setReferenceThresholdLow(unsigned int i, float t);
    void setFloatingThresholdUp(unsigned int i, float t);
    void setFloatingThresholdLow(unsigned int i, float t);
    void setLevelNumber(unsigned int l);
    void setLevelToPerform(unsigned int l);
    void useRobustRange();
    void doNotUseRobustRange();
    void setWarpedPaddingValue(float p);
    void doNotUsePyramidalApproach();
    //
    void setReferencePyramid(nifti_image** rp);
    void setFloatingPyramid(nifti_image** fp);
    void setMaskPyramid(int** mp);
    void setActiveVoxelNumber(int py, int avn);
    //
    virtual void setCurrentReference(nifti_image* currentRefIn);
    virtual void setCurrentReferenceMask(int * currentRefMaskIn, size_t nvox);
    virtual void setCurrentFloating(nifti_image* currentFloIn);
    //void setCurrentTransformationMatrix(mat44 *transformationMatrixIn);
    virtual void setCurrentWarped(nifti_image* currentWarpedIn);
    virtual void setCurrentDeformationField(nifti_image* currentDeformationFieldIn);
    //
    virtual bool isCurrentComputationDoubleCapable();

protected:
    nifti_image *inputReference; // pointer to external
    nifti_image *inputFloating; // pointer to external
    nifti_image *maskImage; // pointer to external
    mat44 *affineTransformation; // pointer to external

    unsigned nbRefTimePoint;
    unsigned nbFloTimePoint;
    float referenceSmoothingSigma;
    float floatingSmoothingSigma;
    bool robustRange;
    float *referenceThresholdUp;
    float *referenceThresholdLow;
    float *floatingThresholdUp;
    float *floatingThresholdLow;

    bool usePyramid;
    unsigned int levelNumber;
    unsigned int levelToPerform;

    nifti_image **referencePyramid;
    nifti_image **floatingPyramid;
    int **maskPyramid;
    int *activeVoxelNumber;

    nifti_image *currentReference;
    mat44* refMatrix_xyz;
    mat44* refMatrix_ijk;
    int *currentReferenceMask;
    nifti_image *currentFloating;
    mat44* floMatrix_xyz;
    mat44* floMatrix_ijk;
    nifti_image *currentDeformationField;
    nifti_image *currentWarped;
    float warpedPaddingValue;

    Platform* platform;
};

#endif //GLOBALCONTENT_H_
