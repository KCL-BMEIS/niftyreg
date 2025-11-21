#pragma once

#include "ContentCreator.h"
#include "F3dContent.h"

class F3d2ContentCreator: public ContentCreator {
public:
    virtual std::pair<F3dContent*, F3dContent*> Create(NiftiImage& reference,
                                                       NiftiImage& floating,
                                                       NiftiImage& controlPointGrid,
                                                       NiftiImage& controlPointGridBw,
                                                       NiftiImage *localWeightSim = nullptr,
                                                       int *referenceMask = nullptr,
                                                       int *floatingMask = nullptr,
                                                       mat44 *transformationMatrix = nullptr,
                                                       mat44 *transformationMatrixBw = nullptr,
                                                       size_t bytes = sizeof(float)) {
        auto con = new F3dContent(reference, floating, controlPointGrid, localWeightSim, referenceMask, transformationMatrix, bytes);
        auto conBw = new F3dContent(floating, reference, controlPointGridBw, nullptr, floatingMask, transformationMatrixBw, bytes);
        return { con, conBw };
    }
};
