#pragma once

#include "AladinContentCreator.h"
#include "ClAladinContent.h"

class ClAladinContentCreator: public AladinContentCreator {
public:
    virtual AladinContent* Create(nifti_image *reference,
                                  nifti_image *floating,
                                  int *referenceMask = nullptr,
                                  mat44 *transformationMatrix = nullptr,
                                  size_t bytes = sizeof(float),
                                  const unsigned int percentageOfBlocks = 0,
                                  const unsigned int inlierLts = 0,
                                  int blockStepSize = 0) override {
        return new ClAladinContent(reference, floating, referenceMask, transformationMatrix, bytes, percentageOfBlocks, inlierLts, blockStepSize);
    }
};