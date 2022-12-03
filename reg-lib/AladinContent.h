#pragma once

#include <ctime>
#include <iosfwd>
#include <map>
#include <string>
#include <vector>
#include "Kernel.h"
#include "Content.h"
#include "_reg_blockMatching.h"

class AladinContent: public Content {
public:
    AladinContent(nifti_image *referenceIn,
                  nifti_image *floatingIn,
                  int *referenceMaskIn = nullptr,
                  mat44 *transformationMatrixIn = nullptr,
                  size_t bytesIn = sizeof(float),
                  const unsigned int percentageOfBlocks = 0,
                  const unsigned int inlierLts = 0,
                  int blockStepSize = 0);

    virtual ~AladinContent();

    // Getters
    virtual _reg_blockMatchingParam* GetBlockMatchingParams() { return blockMatchingParams; }

    // Setters
    void SetCaptureRange(const int captureRangeIn);
    virtual void SetBlockMatchingParams(_reg_blockMatchingParam *bmp) { blockMatchingParams = bmp; }

protected:
    _reg_blockMatchingParam* blockMatchingParams;
    unsigned int currentPercentageOfBlockToUse;
    unsigned int inlierLts;
    int stepSizeBlock;
};
