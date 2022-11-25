#include "AladinContent.h"

using namespace std;

/* *************************************************************** */
AladinContent::AladinContent(nifti_image *currentReferenceIn,
                             nifti_image *currentFloatingIn,
                             int *currentReferenceMaskIn,
                             mat44 *transformationMatrixIn,
                             size_t bytesIn,
                             const unsigned int currentPercentageOfBlockToUseIn,
                             const unsigned int inlierLtsIn,
                             int stepSizeBlockIn) :
    Content(currentReferenceIn, currentFloatingIn, currentReferenceMaskIn, transformationMatrixIn, bytesIn),
    currentPercentageOfBlockToUse(currentPercentageOfBlockToUseIn),
    inlierLts(inlierLtsIn),
    stepSizeBlock(stepSizeBlockIn) {
    if (currentPercentageOfBlockToUseIn || inlierLtsIn || stepSizeBlockIn) {
        blockMatchingParams = new _reg_blockMatchingParam();
        initialise_block_matching_method(currentReference,
                                         blockMatchingParams,
                                         currentPercentageOfBlockToUse,
                                         inlierLts,
                                         stepSizeBlock,
                                         currentReferenceMask,
                                         false);
    } else {
        blockMatchingParams = nullptr;
    }
}
/* *************************************************************** */
AladinContent::~AladinContent() {
    if (blockMatchingParams != nullptr)
        delete blockMatchingParams;
}
/* *************************************************************** */
void AladinContent::SetCaptureRange(const int voxelCaptureRangeIn) {
    blockMatchingParams->voxelCaptureRange = voxelCaptureRangeIn;
}
/* *************************************************************** */
