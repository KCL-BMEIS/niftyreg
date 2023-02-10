#pragma once

#include "ContentCreator.h"
#include "CudaContent.h"

class CudaContentCreator: public ContentCreator {
public:
    virtual Content* Create(nifti_image *reference,
                            nifti_image *floating,
                            int *referenceMask = nullptr,
                            mat44 *transformationMatrix = nullptr,
                            size_t bytes = sizeof(float)) override {
        return new CudaContent(reference, floating, referenceMask, transformationMatrix, bytes);
    }
};
