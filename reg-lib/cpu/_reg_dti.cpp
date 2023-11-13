/*
 *  _reg_dti.cpp
 *
 *
 *  Created by Ivor Simpson on 22/10/2013.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_dti.h"

/* *************************************************************** */
reg_dti::reg_dti(): reg_measure() {
    NR_FUNC_CALLED();
}
/* *************************************************************** */
// This function is directly the same as that used for reg_ssd
void reg_dti::InitialiseMeasure(nifti_image *refImg,
                                nifti_image *floImg,
                                int *refMask,
                                nifti_image *warpedImg,
                                nifti_image *warpedGrad,
                                nifti_image *voxelBasedGrad,
                                nifti_image *localWeightSim,
                                int *floMask,
                                nifti_image *warpedImgBw,
                                nifti_image *warpedGradBw,
                                nifti_image *voxelBasedGradBw) {
    // Set the pointers using the parent class function
    reg_measure::InitialiseMeasure(refImg,
                                   floImg,
                                   refMask,
                                   warpedImg,
                                   warpedGrad,
                                   voxelBasedGrad,
                                   localWeightSim,
                                   floMask,
                                   warpedImgBw,
                                   warpedGradBw,
                                   voxelBasedGradBw);

    // Check that the input images have the same number of time point
    if (this->referenceImage->nt != this->floatingImage->nt)
        NR_FATAL_ERROR("This number of time point should be the same for both input images");

    int j = 0;
    for (int i = 0; i < refImg->nt; ++i) {
        // JM - note, the specific value of timePointWeights is not used for DTI images
        // any value > 0 indicates the 'time point' is active
        if (this->timePointWeights[i] > 0) {
            this->dtIndicies[j++] = i;
            NR_DEBUG("Active time point: " << i);
        }
    }
    if ((refImg->nz > 1 && j != 6) && (refImg->nz == 1 && j != 3))
        NR_FATAL_ERROR("Unexpected number of DTI components");

    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class DataType>
double reg_getDtiMeasureValue(const nifti_image *referenceImage,
                              const nifti_image *warpedImage,
                              const int *mask,
                              const unsigned *dtIndicies) {
#ifdef _WIN32
    long voxel;
    const long voxelNumber = (long)NiftiImage::calcVoxelNumber(referenceImage, 3);
#else
    size_t voxel;
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);
#endif
    // As the tensor has 6 unique components that we need to worry about
    // Read them out for the floating and reference images
    const DataType *firstWarpedVox = static_cast<DataType*>(warpedImage->data);
    const DataType *warpedIntensityXX = &firstWarpedVox[voxelNumber * dtIndicies[0]];
    const DataType *warpedIntensityXY = &firstWarpedVox[voxelNumber * dtIndicies[1]];
    const DataType *warpedIntensityYY = &firstWarpedVox[voxelNumber * dtIndicies[2]];
    const DataType *warpedIntensityXZ = &firstWarpedVox[voxelNumber * dtIndicies[3]];
    const DataType *warpedIntensityYZ = &firstWarpedVox[voxelNumber * dtIndicies[4]];
    const DataType *warpedIntensityZZ = &firstWarpedVox[voxelNumber * dtIndicies[5]];

    const DataType *firstRefVox = static_cast<DataType*>(referenceImage->data);
    const DataType *referenceIntensityXX = &firstRefVox[voxelNumber * dtIndicies[0]];
    const DataType *referenceIntensityXY = &firstRefVox[voxelNumber * dtIndicies[1]];
    const DataType *referenceIntensityYY = &firstRefVox[voxelNumber * dtIndicies[2]];
    const DataType *referenceIntensityXZ = &firstRefVox[voxelNumber * dtIndicies[3]];
    const DataType *referenceIntensityYZ = &firstRefVox[voxelNumber * dtIndicies[4]];
    const DataType *referenceIntensityZZ = &firstRefVox[voxelNumber * dtIndicies[5]];

    double dtiCost = 0, n = 0;
    constexpr double twoThirds = 2.0 / 3.0;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(referenceImage, referenceIntensityXX, referenceIntensityXY, referenceIntensityXZ, \
          referenceIntensityYY, referenceIntensityYZ, referenceIntensityZZ, \
          warpedIntensityXX, warpedIntensityXY, warpedIntensityXZ, \
          warpedIntensityYY, warpedIntensityYZ, warpedIntensityZZ, mask, voxelNumber) \
   reduction(+:dtiCost, n)
#endif
    for (voxel = 0; voxel < voxelNumber; ++voxel) {
        // Check if the current voxel belongs to the mask and the intensities are not nans
        if (mask[voxel] > -1) {
            if (referenceIntensityXX[voxel] == referenceIntensityXX[voxel] &&
                warpedIntensityXX[voxel] == warpedIntensityXX[voxel]) {
                // Calculate the elementwise residual of the diffusion tensor components
                const DataType rXX = referenceIntensityXX[voxel] - warpedIntensityXX[voxel];
                const DataType rXY = referenceIntensityXY[voxel] - warpedIntensityXY[voxel];
                const DataType rYY = referenceIntensityYY[voxel] - warpedIntensityYY[voxel];
                const DataType rXZ = referenceIntensityXZ[voxel] - warpedIntensityXZ[voxel];
                const DataType rYZ = referenceIntensityYZ[voxel] - warpedIntensityYZ[voxel];
                const DataType rZZ = referenceIntensityZZ[voxel] - warpedIntensityZZ[voxel];
                dtiCost -= twoThirds * (Square(rXX) + Square(rYY) + Square(rZZ))
                    + 2.0 * (Square(rXY) + Square(rXZ) + Square(rYZ))
                    - twoThirds * (rXX * rYY + rXX * rZZ + rYY * rZZ);
                n++;
            } // check if values are defined
        } // check if voxel belongs mask
    } // loop over voxels
    return dtiCost / n;
}
/* *************************************************************** */
double GetSimilarityMeasureValue(const nifti_image *referenceImage,
                                 const nifti_image *warpedImage,
                                 const int *mask,
                                 const unsigned *dtIndicies) {
    return std::visit([&](auto&& refImgDataType) {
        using RefImgDataType = std::decay_t<decltype(refImgDataType)>;
        return reg_getDtiMeasureValue<RefImgDataType>(referenceImage,
                                                      warpedImage,
                                                      mask,
                                                      dtIndicies);
    }, NiftiImage::getFloatingDataType(referenceImage));
}
/* *************************************************************** */
double reg_dti::GetSimilarityMeasureValueFw() {
    return ::GetSimilarityMeasureValue(this->referenceImage,
                                       this->warpedImage,
                                       this->referenceMask,
                                       this->dtIndicies);
}
/* *************************************************************** */
double reg_dti::GetSimilarityMeasureValueBw() {
    return ::GetSimilarityMeasureValue(this->floatingImage,
                                       this->warpedImageBw,
                                       this->floatingMask,
                                       this->dtIndicies);
}
/* *************************************************************** */
template <class DataType>
void reg_getVoxelBasedDtiMeasureGradient(const nifti_image *referenceImage,
                                         const nifti_image *warpedImage,
                                         const nifti_image *warpedGradient,
                                         nifti_image *dtiMeasureGradientImage,
                                         const int *mask,
                                         const unsigned *dtIndicies) {
#ifdef _WIN32
    long voxel;
    const long voxelNumber = (long)NiftiImage::calcVoxelNumber(referenceImage, 3);
#else
    size_t voxel;
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);
#endif
    // As the tensor has 6 unique components that we need to worry about
    // Read them out for the floating and reference images
    const DataType *firstWarpedVox = static_cast<DataType*>(warpedImage->data);
    const DataType *warpedIntensityXX = &firstWarpedVox[voxelNumber * dtIndicies[0]];
    const DataType *warpedIntensityXY = &firstWarpedVox[voxelNumber * dtIndicies[1]];
    const DataType *warpedIntensityYY = &firstWarpedVox[voxelNumber * dtIndicies[2]];
    const DataType *warpedIntensityXZ = &firstWarpedVox[voxelNumber * dtIndicies[3]];
    const DataType *warpedIntensityYZ = &firstWarpedVox[voxelNumber * dtIndicies[4]];
    const DataType *warpedIntensityZZ = &firstWarpedVox[voxelNumber * dtIndicies[5]];

    const DataType *firstRefVox = static_cast<DataType*>(referenceImage->data);
    const DataType *referenceIntensityXX = &firstRefVox[voxelNumber * dtIndicies[0]];
    const DataType *referenceIntensityXY = &firstRefVox[voxelNumber * dtIndicies[1]];
    const DataType *referenceIntensityYY = &firstRefVox[voxelNumber * dtIndicies[2]];
    const DataType *referenceIntensityXZ = &firstRefVox[voxelNumber * dtIndicies[3]];
    const DataType *referenceIntensityYZ = &firstRefVox[voxelNumber * dtIndicies[4]];
    const DataType *referenceIntensityZZ = &firstRefVox[voxelNumber * dtIndicies[5]];

    // THE FOLLOWING IS WRONG
    NR_FATAL_ERROR("ERROR IN THE DTI GRADIENT COMPUTATION - TO FIX");
    const size_t gradientVoxels = (size_t)warpedGradient->nu * voxelNumber;
    const DataType *firstGradVox = static_cast<DataType*>(warpedGradient->data);
    const DataType *spatialGradXX = &firstGradVox[gradientVoxels * dtIndicies[0]];
    const DataType *spatialGradXY = &firstGradVox[gradientVoxels * dtIndicies[1]];
    const DataType *spatialGradYY = &firstGradVox[gradientVoxels * dtIndicies[2]];
    const DataType *spatialGradXZ = &firstGradVox[gradientVoxels * dtIndicies[3]];
    const DataType *spatialGradYZ = &firstGradVox[gradientVoxels * dtIndicies[4]];
    const DataType *spatialGradZZ = &firstGradVox[gradientVoxels * dtIndicies[5]];

    // Create an array to store the computed gradient per time point
    DataType *dtiMeasureGradPtrX = static_cast<DataType*>(dtiMeasureGradientImage->data);
    DataType *dtiMeasureGradPtrY = &dtiMeasureGradPtrX[voxelNumber];
    DataType *dtiMeasureGradPtrZ = &dtiMeasureGradPtrY[voxelNumber];

    constexpr double twoThirds = 2.0 / 3.0;
    constexpr double fourThirds = 4.0 / 3.0;

#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(referenceIntensityXX, referenceIntensityXY, referenceIntensityXZ, \
          referenceIntensityYY, referenceIntensityYZ, referenceIntensityZZ,warpedIntensityXX, \
          warpedIntensityXY,warpedIntensityXZ ,warpedIntensityYY,warpedIntensityYZ, warpedIntensityZZ, \
          mask, spatialGradXX, spatialGradXY, spatialGradXZ, spatialGradYY, spatialGradYZ, spatialGradZZ, \
          dtiMeasureGradPtrX, dtiMeasureGradPtrY, dtiMeasureGradPtrZ, voxelNumber)
#endif
    for (voxel = 0; voxel < voxelNumber; voxel++) {
        if (mask[voxel] > -1) {
            if (referenceIntensityXX[voxel] == referenceIntensityXX[voxel] &&
                warpedIntensityXX[voxel] == warpedIntensityXX[voxel]) {
                const DataType rXX = referenceIntensityXX[voxel] - warpedIntensityXX[voxel];
                const DataType rXY = referenceIntensityXY[voxel] - warpedIntensityXY[voxel];
                const DataType rYY = referenceIntensityYY[voxel] - warpedIntensityYY[voxel];
                const DataType rXZ = referenceIntensityXZ[voxel] - warpedIntensityXZ[voxel];
                const DataType rYZ = referenceIntensityYZ[voxel] - warpedIntensityYZ[voxel];
                const DataType rZZ = referenceIntensityZZ[voxel] - warpedIntensityZZ[voxel];

                const DataType xxGrad = static_cast<DataType>(fourThirds * rXX - twoThirds * (rYY + rZZ));
                const DataType yyGrad = static_cast<DataType>(fourThirds * rYY - twoThirds * (rXX + rZZ));
                const DataType zzGrad = static_cast<DataType>(fourThirds * rZZ - twoThirds * (rYY + rXX));
                const DataType xyGrad = 4.f * rXY;
                const DataType xzGrad = 4.f * rXZ;
                const DataType yzGrad = 4.f * rYZ;

                dtiMeasureGradPtrX[voxel] -= (spatialGradXX[voxel] * xxGrad + spatialGradYY[voxel] * yyGrad + spatialGradZZ[voxel] * zzGrad
                                              + spatialGradXY[voxel] * xyGrad + spatialGradXZ[voxel] * xzGrad + spatialGradYZ[voxel] * yzGrad);

                dtiMeasureGradPtrY[voxel] -= (spatialGradXX[voxel + voxelNumber] * xxGrad + spatialGradYY[voxel + voxelNumber] * yyGrad + spatialGradZZ[voxel + voxelNumber] * zzGrad
                                              + spatialGradXY[voxel + voxelNumber] * xyGrad + spatialGradXZ[voxel + voxelNumber] * xzGrad + spatialGradYZ[voxel + voxelNumber] * yzGrad);

                dtiMeasureGradPtrZ[voxel] -= (spatialGradXX[voxel + 2 * voxelNumber] * xxGrad + spatialGradYY[voxel + 2 * voxelNumber] * yyGrad
                                              + spatialGradZZ[voxel + 2 * voxelNumber] * zzGrad + spatialGradXY[voxel + 2 * voxelNumber] * xyGrad
                                              + spatialGradXZ[voxel + 2 * voxelNumber] * xzGrad + spatialGradYZ[voxel + 2 * voxelNumber] * yzGrad);
            }
        }
    }
}
/* *************************************************************** */
void GetVoxelBasedSimilarityMeasureGradient(const nifti_image *referenceImage,
                                            const nifti_image *warpedImage,
                                            const nifti_image *warpedGradient,
                                            nifti_image *voxelBasedGradient,
                                            const int *referenceMask,
                                            const unsigned *dtIndicies) {
    std::visit([&](auto&& refImgDataType) {
        using RefImgDataType = std::decay_t<decltype(refImgDataType)>;
        reg_getVoxelBasedDtiMeasureGradient<RefImgDataType>(referenceImage,
                                                            warpedImage,
                                                            warpedGradient,
                                                            voxelBasedGradient,
                                                            referenceMask,
                                                            dtIndicies);
    }, NiftiImage::getFloatingDataType(referenceImage));
}
/* *************************************************************** */
void reg_dti::GetVoxelBasedSimilarityMeasureGradientFw(int currentTimePoint) {
    ::GetVoxelBasedSimilarityMeasureGradient(this->referenceImage,
                                             this->warpedImage,
                                             this->warpedGradient,
                                             this->voxelBasedGradient,
                                             this->referenceMask,
                                             this->dtIndicies);
}
/* *************************************************************** */
void reg_dti::GetVoxelBasedSimilarityMeasureGradientBw(int currentTimePoint) {
    ::GetVoxelBasedSimilarityMeasureGradient(this->floatingImage,
                                             this->warpedImageBw,
                                             this->warpedGradientBw,
                                             this->voxelBasedGradientBw,
                                             this->floatingMask,
                                             this->dtIndicies);
}
/* *************************************************************** */
