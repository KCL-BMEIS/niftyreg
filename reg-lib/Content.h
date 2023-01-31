#pragma once

#include "_reg_maths.h"

class Content {
public:
    Content() = delete; // Can't be initialised without reference and floating images
    Content(nifti_image *referenceIn,
            nifti_image *floatingIn,
            int *referenceMaskIn = nullptr,
            mat44 *transformationMatrixIn = nullptr,
            size_t bytesIn = sizeof(float));
    virtual ~Content();

    virtual bool IsCurrentComputationDoubleCapable() { return true; }

    // Getters
    virtual nifti_image* GetReference() { return reference; }
    virtual nifti_image* GetFloating() { return floating; }
    virtual nifti_image* GetDeformationField() { return deformationField; }
    virtual int* GetReferenceMask() { return referenceMask; }
    virtual mat44* GetTransformationMatrix() { return transformationMatrix; }
    virtual nifti_image* GetWarped() { return warped; }

    // Setters
    virtual void SetDeformationField(nifti_image *deformationFieldIn) { deformationField = deformationFieldIn; }
    virtual void SetReferenceMask(int *referenceMaskIn) { referenceMask = referenceMaskIn; }
    virtual void SetTransformationMatrix(mat44 *transformationMatrixIn) { transformationMatrix = transformationMatrixIn; }
    virtual void SetWarped(nifti_image *warpedIn) { warped = warpedIn; }

    // Auxiliary methods
    static mat44* GetXYZMatrix(nifti_image& image) {
        return image.sform_code > 0 ? &image.sto_xyz : &image.qto_xyz;
    }
    static mat44* GetIJKMatrix(nifti_image& image) {
        return image.sform_code > 0 ? &image.sto_ijk : &image.qto_ijk;
    }

protected:
    nifti_image *reference = nullptr;
    nifti_image *floating = nullptr;
    nifti_image *deformationField = nullptr;
    int *referenceMask = nullptr;
    mat44 *transformationMatrix = nullptr;
    nifti_image *warped = nullptr;

private:
    void AllocateWarped();
    void DeallocateWarped();
    void AllocateDeformationField(size_t bytes);
    void DeallocateDeformationField();
};
