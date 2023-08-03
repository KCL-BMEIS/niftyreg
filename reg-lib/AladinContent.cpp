#include "AladinContent.h"

using namespace std;

/* *************************************************************** */
AladinContent::AladinContent(nifti_image *referenceIn,
                             nifti_image *floatingIn,
                             int *referenceMaskIn,
                             mat44 *transformationMatrixIn,
                             size_t bytesIn,
                             const unsigned currentPercentageOfBlockToUseIn,
                             const unsigned inlierLtsIn,
                             int stepSizeBlockIn) :
    Content(referenceIn, floatingIn, referenceMaskIn, transformationMatrixIn, bytesIn),
    currentPercentageOfBlockToUse(currentPercentageOfBlockToUseIn),
    inlierLts(inlierLtsIn),
    stepSizeBlock(stepSizeBlockIn) {
    if (currentPercentageOfBlockToUseIn || inlierLtsIn || stepSizeBlockIn) {
        blockMatchingParams = new _reg_blockMatchingParam();
        initialise_block_matching_method(reference,
                                         blockMatchingParams,
                                         currentPercentageOfBlockToUse,
                                         inlierLts,
                                         stepSizeBlock,
                                         referenceMask,
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
