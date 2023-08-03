#pragma once

#include "DefContentCreator.h"
#include "CudaDefContent.h"

class CudaDefContentCreator: public DefContentCreator {
public:
    virtual DefContent* Create(nifti_image *reference,
                               nifti_image *floating,
                               nifti_image *localWeightSim = nullptr,
                               int *referenceMask = nullptr,
                               mat44 *transformationMatrix = nullptr,
                               size_t bytes = sizeof(float)) override {
        return new CudaDefContent(reference, floating, localWeightSim, referenceMask, transformationMatrix, bytes);
    }
};
