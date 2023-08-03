#pragma once

#include "ContentCreator.h"
#include "DefContent.h"

class DefContentCreator: public ContentCreator {
public:
    virtual DefContent* Create(nifti_image *reference,
                               nifti_image *floating,
                               nifti_image *localWeightSim = nullptr,
                               int *referenceMask = nullptr,
                               mat44 *transformationMatrix = nullptr,
                               size_t bytes = sizeof(float)) {
        return new DefContent(reference, floating, localWeightSim, referenceMask, transformationMatrix, bytes);
    }
};
