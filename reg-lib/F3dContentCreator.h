#pragma once

#include "ContentCreator.h"
#include "F3dContent.h"

class F3dContentCreator: public ContentCreator {
public:
    virtual F3dContent* Create(NiftiImage& reference,
                               NiftiImage& floating,
                               NiftiImage& controlPointGrid,
                               NiftiImage *localWeightSim = nullptr,
                               int *referenceMask = nullptr,
                               mat44 *transformationMatrix = nullptr,
                               size_t bytes = sizeof(float)) {
        return new F3dContent(reference, floating, controlPointGrid, localWeightSim, referenceMask, transformationMatrix, bytes);
    }
};
