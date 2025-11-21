#pragma once

#include "_reg_tools.h"

class Content {
public:
    Content() = delete; // Can't be initialised without reference and floating images
    Content(NiftiImage& referenceIn,
            NiftiImage& floatingIn,
            int *referenceMaskIn = nullptr,
            mat44 *transformationMatrixIn = nullptr,
            size_t bytesIn = sizeof(float));
    virtual ~Content() = default;

    virtual bool IsCurrentComputationDoubleCapable() { return true; }

    // Getters
    virtual size_t GetActiveVoxelNumber() { return activeVoxelNumber; }
    virtual NiftiImage& GetReference() { return reference; }
    virtual NiftiImage& GetFloating() { return floating; }
    virtual NiftiImage& GetDeformationField() { return deformationField; }
    virtual int* GetReferenceMask() { return referenceMask; }
    virtual mat44* GetTransformationMatrix() { return transformationMatrix; }
    virtual NiftiImage& GetWarped() { return warped; }

    // Methods for transferring data from nifti to device
    virtual void UpdateDeformationField() {}
    virtual void UpdateWarped() {}

    // Auxiliary methods
    static mat44* GetXYZMatrix(nifti_image& image) {
        return image.sform_code > 0 ? &image.sto_xyz : &image.qto_xyz;
    }
    static mat44* GetIJKMatrix(nifti_image& image) {
        return image.sform_code > 0 ? &image.sto_ijk : &image.qto_ijk;
    }

protected:
    size_t activeVoxelNumber = 0;
    NiftiImage reference;
    NiftiImage floating;
    NiftiImage deformationField;
    int *referenceMask = nullptr;
    unique_ptr<int[]> referenceMaskManaged;
    mat44 *transformationMatrix = nullptr;
    NiftiImage warped;

private:
    void AllocateWarped();
    void AllocateDeformationField(size_t bytes);

#ifdef NR_TESTING
public:
#else
protected:
#endif
    // Functions for testing
    virtual void SetDeformationField(NiftiImage&& deformationFieldIn) { deformationField = std::move(deformationFieldIn); }
    virtual void SetReferenceMask(int *referenceMaskIn) { referenceMask = referenceMaskIn; }
    virtual void SetTransformationMatrix(mat44 *transformationMatrixIn) { transformationMatrix = transformationMatrixIn; }
    virtual void SetWarped(NiftiImage&& warpedIn) { warped = std::move(warpedIn); }
};
