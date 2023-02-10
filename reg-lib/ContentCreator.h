#pragma once

#include "Content.h"

class ContentCreator {
public:
    virtual Content* Create(nifti_image *reference,
                            nifti_image *floating,
                            int *referenceMask = nullptr,
                            mat44 *transformationMatrix = nullptr,
                            size_t bytes = sizeof(float)) {
        return new Content(reference, floating, referenceMask, transformationMatrix, bytes);
    }
};
