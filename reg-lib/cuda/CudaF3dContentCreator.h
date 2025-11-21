#pragma once

#include "F3dContentCreator.h"
#include "CudaF3dContent.h"

class CudaF3dContentCreator: public F3dContentCreator {
public:
    virtual F3dContent* Create(NiftiImage& reference,
                               NiftiImage& floating,
                               NiftiImage& controlPointGrid,
                               NiftiImage *localWeightSim = nullptr,
                               int *referenceMask = nullptr,
                               mat44 *transformationMatrix = nullptr,
                               size_t bytes = sizeof(float)) override {
        return new CudaF3dContent(reference, floating, controlPointGrid, localWeightSim, referenceMask, transformationMatrix, bytes);
    }
};
