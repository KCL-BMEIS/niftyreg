#pragma once

#include "nifti1_io.h"

class Content {
public:
    Content() = delete; // Can't be initialised without reference and floating images
    Content(nifti_image *currentReferenceIn,
            nifti_image *currentFloatingIn,
            int *currentReferenceMaskIn = nullptr,
            mat44 *transformationMatrixIn = nullptr,
            size_t bytesIn = sizeof(float));
    virtual ~Content();

    // Getters
    virtual nifti_image* GetCurrentDeformationField() { return currentDeformationField; }
    virtual nifti_image* GetCurrentReference() { return currentReference; }
    virtual nifti_image* GetCurrentFloating() { return currentFloating; }
    virtual nifti_image* GetCurrentWarped(int = 0) { return currentWarped; }
    virtual int* GetCurrentReferenceMask() { return currentReferenceMask; }
    virtual mat44* GetTransformationMatrix() { return transformationMatrix; }

    // Setters
    virtual void SetTransformationMatrix(mat44 *transformationMatrixIn) {
        transformationMatrix = transformationMatrixIn;
    }
    virtual void SetCurrentDeformationField(nifti_image *currentDeformationFieldIn) {
        ClearDeformationField();
        currentDeformationField = currentDeformationFieldIn;
    }
    virtual void SetCurrentWarped(nifti_image *currentWarpedImageIn) {
        ClearWarpedImage();
        currentWarped = currentWarpedImageIn;
    }
    virtual void SetCurrentReferenceMask(int *currentReferenceMaskIn) {
        free(currentReferenceMask);
        currentReferenceMask = currentReferenceMaskIn;
    }

    virtual bool IsCurrentComputationDoubleCapable() { return true; }

    static mat44* GetXYZMatrix(nifti_image *image) {
        return image->sform_code > 0 ? &image->sto_xyz : &image->qto_xyz;
    }
    static mat44* GetIJKMatrix(nifti_image *image) {
        return image->sform_code > 0 ? &image->sto_ijk : &image->qto_ijk;
    }

protected:
    virtual void AllocateWarpedImage();
    virtual void ClearWarpedImage();
    virtual void AllocateDeformationField(size_t bytes);
    virtual void ClearDeformationField();

    nifti_image *currentReference;
    nifti_image *currentFloating;
    int *currentReferenceMask;
    nifti_image *currentDeformationField;
    nifti_image *currentWarped;
    mat44 *transformationMatrix;
};
