/*
 *  _reg_nmi.cpp
 *
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_nmi.h"

/* *************************************************************** */
reg_nmi::reg_nmi(): reg_measure() {
    this->jointHistogramPro = nullptr;
    this->jointHistogramLog = nullptr;
    this->entropyValues = nullptr;
    this->jointHistogramProBw = nullptr;
    this->jointHistogramLogBw = nullptr;
    this->entropyValuesBw = nullptr;

    for (int i = 0; i < 255; ++i) {
        this->referenceBinNumber[i] = 68;
        this->floatingBinNumber[i] = 68;
    }
#ifndef NDEBUG
    reg_print_msg_debug("reg_nmi constructor called");
#endif
}
/* *************************************************************** */
reg_nmi::~reg_nmi() {
    this->DeallocateHistogram();
#ifndef NDEBUG
    reg_print_msg_debug("reg_nmi destructor called");
#endif
}
/* *************************************************************** */
void reg_nmi::DeallocateHistogram() {
    int timepoint = this->referenceTimePoint;
    // Free the joint histograms and the entropy arrays
    if (this->jointHistogramPro != nullptr) {
        for (int i = 0; i < timepoint; ++i) {
            if (this->jointHistogramPro[i] != nullptr)
                free(this->jointHistogramPro[i]);
            this->jointHistogramPro[i] = nullptr;
        }
        free(this->jointHistogramPro);
    }
    this->jointHistogramPro = nullptr;
    if (this->jointHistogramProBw != nullptr) {
        for (int i = 0; i < timepoint; ++i) {
            if (this->jointHistogramProBw[i] != nullptr)
                free(this->jointHistogramProBw[i]);
            this->jointHistogramProBw[i] = nullptr;
        }
        free(this->jointHistogramProBw);
    }
    this->jointHistogramProBw = nullptr;

    if (this->jointHistogramLog != nullptr) {
        for (int i = 0; i < timepoint; ++i) {
            if (this->jointHistogramLog[i] != nullptr)
                free(this->jointHistogramLog[i]);
            this->jointHistogramLog[i] = nullptr;
        }
        free(this->jointHistogramLog);
    }
    this->jointHistogramLog = nullptr;
    if (this->jointHistogramLogBw != nullptr) {
        for (int i = 0; i < timepoint; ++i) {
            if (this->jointHistogramLogBw[i] != nullptr)
                free(this->jointHistogramLogBw[i]);
            this->jointHistogramLogBw[i] = nullptr;
        }
        free(this->jointHistogramLogBw);
    }
    this->jointHistogramLogBw = nullptr;

    if (this->entropyValues != nullptr) {
        for (int i = 0; i < timepoint; ++i) {
            if (this->entropyValues[i] != nullptr)
                free(this->entropyValues[i]);
            this->entropyValues[i] = nullptr;
        }
        free(this->entropyValues);
    }
    this->entropyValues = nullptr;
    if (this->entropyValuesBw != nullptr) {
        for (int i = 0; i < timepoint; ++i) {
            if (this->entropyValuesBw[i] != nullptr)
                free(this->entropyValuesBw[i]);
            this->entropyValuesBw[i] = nullptr;
        }
        free(this->entropyValuesBw);
    }
    this->entropyValuesBw = nullptr;
#ifndef NDEBUG
    reg_print_msg_debug("reg_nmi::DeallocateHistogram called");
#endif
}
/* *************************************************************** */
void reg_nmi::InitialiseMeasure(nifti_image *refImg,
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

    // Deallocate all allocated arrays
    this->DeallocateHistogram();
    // Reference and floating are resampled between 2 and bin-3
    for (int i = 0; i < this->referenceTimePoint; ++i) {
        if (this->timePointWeight[i] > 0) {
            reg_intensityRescale(this->referenceImage,
                                 i,
                                 2.f,
                                 this->referenceBinNumber[i] - 3.f);
            reg_intensityRescale(this->floatingImage,
                                 i,
                                 2.f,
                                 this->floatingBinNumber[i] - 3.f);
        }
    }
    // Create the joint histograms
    this->jointHistogramPro = (double**)calloc(255, sizeof(double*));
    this->jointHistogramLog = (double**)calloc(255, sizeof(double*));
    this->entropyValues = (double**)calloc(255, sizeof(double*));
    if (this->isSymmetric) {
        this->jointHistogramProBw = (double**)calloc(255, sizeof(double*));
        this->jointHistogramLogBw = (double**)calloc(255, sizeof(double*));
        this->entropyValuesBw = (double**)calloc(255, sizeof(double*));
    }
    for (int i = 0; i < this->referenceTimePoint; ++i) {
        if (this->timePointWeight[i] > 0) {
            // Compute the total number of bin
            this->totalBinNumber[i] = this->referenceBinNumber[i] * this->floatingBinNumber[i] +
                this->referenceBinNumber[i] + this->floatingBinNumber[i];
            this->jointHistogramLog[i] = (double*)calloc(this->totalBinNumber[i], sizeof(double));
            this->jointHistogramPro[i] = (double*)calloc(this->totalBinNumber[i], sizeof(double));
            this->entropyValues[i] = (double*)calloc(4, sizeof(double));
            if (this->isSymmetric) {
                this->jointHistogramLogBw[i] = (double*)calloc(this->totalBinNumber[i], sizeof(double));
                this->jointHistogramProBw[i] = (double*)calloc(this->totalBinNumber[i], sizeof(double));
                this->entropyValuesBw[i] = (double*)calloc(4, sizeof(double));
            }
        }
    }
#ifndef NDEBUG
    char text[255];
    reg_print_msg_debug("reg_nmi::InitialiseMeasure()");
    for (int i = 0; i < this->referenceImage->nt; ++i) {
        sprintf(text, "Weight for timepoint %i: %f", i, this->timePointWeight[i]);
        reg_print_msg_debug(text);
    }
#endif
}
/* *************************************************************** */
static double GetBasisSplineValue(double x) {
    x = fabs(x);
    double value = 0;
    if (x < 2.0) {
        if (x < 1.0)
            value = 2.0 / 3.0 + (0.5 * x - 1.0) * x * x;
        else {
            x -= 2.0;
            value = -x * x * x / 6.0;
        }
    }
    return value;
}
/* *************************************************************** */
static double GetBasisSplineDerivativeValue(double ori) {
    double x = fabs(ori), value = 0;
    if (x < 2.0) {
        if (x < 1.0)
            value = (1.5 * x - 2.0) * ori;
        else {
            x -= 2.0;
            value = -0.5 * x * x;
            if (ori < 0.0) value = -value;
        }
    }
    return value;
}
/* *************************************************************** */
template <class DataType>
void reg_getNMIValue(const nifti_image *referenceImage,
                     const nifti_image *warpedImage,
                     const double *timePointWeight,
                     const unsigned short *referenceBinNumber,
                     const unsigned short *floatingBinNumber,
                     const unsigned short *totalBinNumber,
                     double **jointHistogramLog,
                     double **jointHistogramPro,
                     double **entropyValues,
                     const int *referenceMask) {
    // Create pointers to the image data arrays
    const DataType *refImagePtr = static_cast<DataType*>(referenceImage->data);
    const DataType *warImagePtr = static_cast<DataType*>(warpedImage->data);
    // Useful variable
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);
    // Iterate over all active time points
    for (int t = 0; t < referenceImage->nt; ++t) {
        if (timePointWeight[t] > 0) {
#ifndef NDEBUG
            char text[255];
            sprintf(text, "Computing NMI for time point %i", t);
            reg_print_msg_debug(text);
#endif
            // Define some pointers to the current histograms
            double *jointHistoProPtr = jointHistogramPro[t];
            double *jointHistoLogPtr = jointHistogramLog[t];
            // Empty the joint histogram
            memset(jointHistoProPtr, 0, totalBinNumber[t] * sizeof(double));
            // Fill the joint histograms using an approximation
            const DataType *refPtr = &refImagePtr[t * voxelNumber];
            const DataType *warPtr = &warImagePtr[t * voxelNumber];
            for (size_t voxel = 0; voxel < voxelNumber; ++voxel) {
                if (referenceMask[voxel] > -1) {
                    const DataType refValue = refPtr[voxel];
                    const DataType warValue = warPtr[voxel];
                    if (refValue == refValue && warValue == warValue){
                        for(int r = int(refValue-1); r < int(refValue+3); ++r){
                            if( 0 <= r && r < referenceBinNumber[t]){
                                const double refBasis = GetBasisSplineValue(refValue - r);
                                for(int w = int(warValue-1); w < int(warValue+3); ++w){
                                    if( 0 <= w && w < floatingBinNumber[t]){
                                        const double warBasis = GetBasisSplineValue(warValue - w);
                                        jointHistoProPtr[r + w * referenceBinNumber[t]] += refBasis * warBasis;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Normalise the histogram
            double activeVoxel = 0.f;
            for (int i = 0; i < totalBinNumber[t]; ++i)
                activeVoxel += jointHistoProPtr[i];
            entropyValues[t][3] = activeVoxel;
            for (int i = 0; i < totalBinNumber[t]; ++i)
                jointHistoProPtr[i] /= activeVoxel;
            // Marginalise over the reference axis
            for (int r = 0; r < referenceBinNumber[t]; ++r) {
                double sum = 0.;
                int index = r;
                for (int f = 0; f < floatingBinNumber[t]; ++f) {
                    sum += jointHistoProPtr[index];
                    index += referenceBinNumber[t];
                }
                jointHistoProPtr[referenceBinNumber[t] *
                    floatingBinNumber[t] + r] = sum;
            }
            // Marginalise over the warped floating axis
            for (int f = 0; f < floatingBinNumber[t]; ++f) {
                double sum = 0.;
                int index = referenceBinNumber[t] * f;
                for (int r = 0; r < referenceBinNumber[t]; ++r) {
                    sum += jointHistoProPtr[index];
                    ++index;
                }
                jointHistoProPtr[referenceBinNumber[t] * floatingBinNumber[t] + referenceBinNumber[t] + f] = sum;
            }
            // Set the log values to zero
            memset(jointHistoLogPtr, 0, totalBinNumber[t] * sizeof(double));
            // Compute the entropy of the reference image
            double referenceEntropy = 0.;
            for (int r = 0; r < referenceBinNumber[t]; ++r) {
                double valPro = jointHistoProPtr[referenceBinNumber[t] * floatingBinNumber[t] + r];
                if (valPro > 0) {
                    double valLog = log(valPro);
                    referenceEntropy -= valPro * valLog;
                    jointHistoLogPtr[referenceBinNumber[t] * floatingBinNumber[t] + r] = valLog;
                }
            }
            entropyValues[t][0] = referenceEntropy;
            // Compute the entropy of the warped floating image
            double warpedEntropy = 0.;
            for (int f = 0; f < floatingBinNumber[t]; ++f) {
                double valPro = jointHistoProPtr[referenceBinNumber[t] * floatingBinNumber[t] +
                    referenceBinNumber[t] + f];
                if (valPro > 0) {
                    double valLog = log(valPro);
                    warpedEntropy -= valPro * valLog;
                    jointHistoLogPtr[referenceBinNumber[t] * floatingBinNumber[t] + referenceBinNumber[t] + f] = valLog;
                }
            }
            entropyValues[t][1] = warpedEntropy;
            // Compute the joint entropy
            double jointEntropy = 0.;
            for (int i = 0; i < referenceBinNumber[t] * floatingBinNumber[t]; ++i) {
                double valPro = jointHistoProPtr[i];
                if (valPro > 0) {
                    double valLog = log(valPro);
                    jointEntropy -= valPro * valLog;
                    jointHistoLogPtr[i] = valLog;
                }
            }
            entropyValues[t][2] = jointEntropy;
        } // if active time point
    } // iterate over all time point in the reference image
}
/* *************************************************************** */
double GetSimilarityMeasureValue(const nifti_image *referenceImage,
                                 const nifti_image *warpedImage,
                                 const double *timePointWeight,
                                 const unsigned short *referenceBinNumber,
                                 const unsigned short *floatingBinNumber,
                                 const unsigned short *totalBinNumber,
                                 double **jointHistogramLog,
                                 double **jointHistogramPro,
                                 double **entropyValues,
                                 const int *referenceMask,
                                 const int& referenceTimePoint) {
    std::visit([&](auto&& refImgDataType) {
        using RefImgDataType = std::decay_t<decltype(refImgDataType)>;
        reg_getNMIValue<RefImgDataType>(referenceImage,
                                        warpedImage,
                                        timePointWeight,
                                        referenceBinNumber,
                                        floatingBinNumber,
                                        totalBinNumber,
                                        jointHistogramLog,
                                        jointHistogramPro,
                                        entropyValues,
                                        referenceMask);
    }, NiftiImage::getFloatingDataType(referenceImage));

    double nmi = 0;
    for (int t = 0; t < referenceTimePoint; ++t) {
        if (timePointWeight[t] > 0)
            nmi += timePointWeight[t] * (entropyValues[t][0] + entropyValues[t][1]) / entropyValues[t][2];
    }
    return nmi;
}
/* *************************************************************** */
double reg_nmi::GetSimilarityMeasureValueFw() {
    return ::GetSimilarityMeasureValue(this->referenceImage,
                                       this->warpedImage,
                                       this->timePointWeight,
                                       this->referenceBinNumber,
                                       this->floatingBinNumber,
                                       this->totalBinNumber,
                                       this->jointHistogramLog,
                                       this->jointHistogramPro,
                                       this->entropyValues,
                                       this->referenceMask,
                                       this->referenceTimePoint);
}
/* *************************************************************** */
double reg_nmi::GetSimilarityMeasureValueBw() {
    return ::GetSimilarityMeasureValue(this->floatingImage,
                                       this->warpedImageBw,
                                       this->timePointWeight,
                                       this->floatingBinNumber,
                                       this->referenceBinNumber,
                                       this->totalBinNumber,
                                       this->jointHistogramLogBw,
                                       this->jointHistogramProBw,
                                       this->entropyValuesBw,
                                       this->floatingMask,
                                       this->referenceTimePoint);
}
/* *************************************************************** */
template <class DataType>
void reg_getVoxelBasedNmiGradient2d(const nifti_image *referenceImage,
                                    const nifti_image *warpedImage,
                                    const unsigned short *referenceBinNumber,
                                    const unsigned short *floatingBinNumber,
                                    const double *const *jointHistogramLog,
                                    const double *const *entropyValues,
                                    const nifti_image *warpedGradient,
                                    nifti_image *measureGradientImage,
                                    const int *referenceMask,
                                    const int& currentTimepoint,
                                    const double& timepointWeight) {
#ifdef WIN32
    long i;
    const long voxelNumber = (long)NiftiImage::calcVoxelNumber(referenceImage, 2);
#else
    size_t i;
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 2);
#endif
    // Pointers to the image data
    const DataType *refImagePtr = static_cast<DataType*>(referenceImage->data);
    const DataType *refPtr = &refImagePtr[currentTimepoint * voxelNumber];
    const DataType *warImagePtr = static_cast<DataType*>(warpedImage->data);
    const DataType *warPtr = &warImagePtr[currentTimepoint * voxelNumber];

    // Pointers to the spatial gradient of the warped image
    const DataType *warGradPtrX = static_cast<DataType*>(warpedGradient->data);
    const DataType *warGradPtrY = &warGradPtrX[voxelNumber];

    // Pointers to the measure of similarity gradient
    DataType *measureGradPtrX = static_cast<DataType*>(measureGradientImage->data);
    DataType *measureGradPtrY = &measureGradPtrX[voxelNumber];

    // Create pointers to the current joint histogram
    const double *logHistoPtr = jointHistogramLog[currentTimepoint];
    const double *entropyPtr = entropyValues[currentTimepoint];
    const double nmi = (entropyPtr[0] + entropyPtr[1]) / entropyPtr[2];
    const size_t referenceOffset = referenceBinNumber[currentTimepoint] * floatingBinNumber[currentTimepoint];
    const size_t floatingOffset = referenceOffset + referenceBinNumber[currentTimepoint];

    // Iterate over all voxel
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(voxelNumber,referenceMask,refPtr,warPtr,referenceBinNumber,floatingBinNumber, \
    logHistoPtr,referenceOffset,floatingOffset,measureGradPtrX,measureGradPtrY, \
    warGradPtrX,warGradPtrY,entropyPtr,nmi,currentTimepoint,timepointWeight)
#endif // _OPENMP
    for (i = 0; i < voxelNumber; ++i) {
        // Check if the voxel belongs to the image mask
        if (referenceMask[i] > -1) {
            DataType refValue = refPtr[i], warValue = warPtr[i];
            if (refValue == refValue && warValue == warValue) {
                DataType gradX = warGradPtrX[i], gradY = warGradPtrY[i];
                double jointDeriv[2]{}, refDeriv[2]{}, warDeriv[2]{};
                for (int r = int(refValue - 1.f); r < int(refValue + 3.f); ++r) {
                    if (-1 < r && r < referenceBinNumber[currentTimepoint]) {
                        for (int w = int(warValue - 1.f); w < int(warValue + 3.f); ++w) {
                            if (-1 < w && w < floatingBinNumber[currentTimepoint]) {
                                const double commun = GetBasisSplineValue(refValue - r) *
                                    GetBasisSplineDerivativeValue(warValue - w);
                                const double &jointLog = logHistoPtr[r + w * referenceBinNumber[currentTimepoint]];
                                const double &refLog = logHistoPtr[r + referenceOffset];
                                const double &warLog = logHistoPtr[w + floatingOffset];
                                if (gradX == gradX) {
                                    jointDeriv[0] += commun * gradX * jointLog;
                                    refDeriv[0] += commun * gradX * refLog;
                                    warDeriv[0] += commun * gradX * warLog;
                                }
                                if (gradY == gradY) {
                                    jointDeriv[1] += commun * gradY * jointLog;
                                    refDeriv[1] += commun * gradY * refLog;
                                    warDeriv[1] += commun * gradY * warLog;
                                }
                            }
                        }
                    }
                }
                measureGradPtrX[i] += static_cast<DataType>(timepointWeight * (refDeriv[0] + warDeriv[0] -
                                                                               nmi * jointDeriv[0]) / (entropyPtr[2] * entropyPtr[3]));
                measureGradPtrY[i] += static_cast<DataType>(timepointWeight * (refDeriv[1] + warDeriv[1] -
                                                                               nmi * jointDeriv[1]) / (entropyPtr[2] * entropyPtr[3]));
            }// Check that the values are defined
        } // mask
    } // loop over all voxel
}
/* *************************************************************** */
template <class DataType>
void reg_getVoxelBasedNmiGradient3d(const nifti_image *referenceImage,
                                    const nifti_image *warpedImage,
                                    const unsigned short *referenceBinNumber,
                                    const unsigned short *floatingBinNumber,
                                    const double *const *jointHistogramLog,
                                    const double *const *entropyValues,
                                    const nifti_image *warpedGradient,
                                    nifti_image *measureGradientImage,
                                    const int *referenceMask,
                                    const int& currentTimepoint,
                                    const double& timepointWeight) {
#ifdef WIN32
    long i;
    const long voxelNumber = (long)NiftiImage::calcVoxelNumber(referenceImage, 3);
#else
    size_t i;
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);
#endif
    // Pointers to the image data
    const DataType *refImagePtr = static_cast<DataType*>(referenceImage->data);
    const DataType *refPtr = &refImagePtr[currentTimepoint * voxelNumber];
    const DataType *warImagePtr = static_cast<DataType*>(warpedImage->data);
    const DataType *warPtr = &warImagePtr[currentTimepoint * voxelNumber];

    // Pointers to the spatial gradient of the warped image
    const DataType *warGradPtrX = static_cast<DataType*>(warpedGradient->data);
    const DataType *warGradPtrY = &warGradPtrX[voxelNumber];
    const DataType *warGradPtrZ = &warGradPtrY[voxelNumber];

    // Pointers to the measure of similarity gradient
    DataType *measureGradPtrX = static_cast<DataType*>(measureGradientImage->data);
    DataType *measureGradPtrY = &measureGradPtrX[voxelNumber];
    DataType *measureGradPtrZ = &measureGradPtrY[voxelNumber];

    // Create pointers to the current joint histogram
    const double *logHistoPtr = jointHistogramLog[currentTimepoint];
    const double *entropyPtr = entropyValues[currentTimepoint];
    const double nmi = (entropyPtr[0] + entropyPtr[1]) / entropyPtr[2];
    const size_t referenceOffset = referenceBinNumber[currentTimepoint] * floatingBinNumber[currentTimepoint];
    const size_t floatingOffset = referenceOffset + referenceBinNumber[currentTimepoint];
    // Iterate over all voxel
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(voxelNumber,referenceMask,refPtr,warPtr,referenceBinNumber,floatingBinNumber, \
    logHistoPtr,referenceOffset,floatingOffset,measureGradPtrX,measureGradPtrY,measureGradPtrZ, \
    warGradPtrX,warGradPtrY,warGradPtrZ,entropyPtr,nmi,currentTimepoint,timepointWeight)
#endif // _OPENMP
    for (i = 0; i < voxelNumber; ++i) {
        // Check if the voxel belongs to the image mask
        if (referenceMask[i] > -1) {
            DataType refValue = refPtr[i], warValue = warPtr[i];
            if (refValue == refValue && warValue == warValue) {
                DataType gradX = warGradPtrX[i], gradY = warGradPtrY[i], gradZ = warGradPtrZ[i];
                double jointDeriv[3]{}, refDeriv[3]{}, warDeriv[3]{};
                for (int r = int(refValue - 1.f); r < int(refValue + 3.f); ++r) {
                    if (-1 < r && r < referenceBinNumber[currentTimepoint]) {
                        for (int w = int(warValue - 1.f); w < int(warValue + 3.f); ++w) {
                            if (-1 < w && w < floatingBinNumber[currentTimepoint]) {
                                const double commun = GetBasisSplineValue(refValue - r) *
                                    GetBasisSplineDerivativeValue(warValue - w);
                                const double& jointLog = logHistoPtr[r + w * referenceBinNumber[currentTimepoint]];
                                const double& refLog = logHistoPtr[r + referenceOffset];
                                const double& warLog = logHistoPtr[w + floatingOffset];
                                if (gradX == gradX) {
                                    refDeriv[0] += commun * gradX * refLog;
                                    warDeriv[0] += commun * gradX * warLog;
                                    jointDeriv[0] += commun * gradX * jointLog;
                                }
                                if (gradY == gradY) {
                                    refDeriv[1] += commun * gradY * refLog;
                                    warDeriv[1] += commun * gradY * warLog;
                                    jointDeriv[1] += commun * gradY * jointLog;
                                }
                                if (gradZ == gradZ) {
                                    refDeriv[2] += commun * gradZ * refLog;
                                    warDeriv[2] += commun * gradZ * warLog;
                                    jointDeriv[2] += commun * gradZ * jointLog;
                                }
                            }
                        }
                    }
                }
                measureGradPtrX[i] += static_cast<DataType>(timepointWeight * (refDeriv[0] + warDeriv[0] -
                                                                               nmi * jointDeriv[0]) / (entropyPtr[2] * entropyPtr[3]));
                measureGradPtrY[i] += static_cast<DataType>(timepointWeight * (refDeriv[1] + warDeriv[1] -
                                                                               nmi * jointDeriv[1]) / (entropyPtr[2] * entropyPtr[3]));
                measureGradPtrZ[i] += static_cast<DataType>(timepointWeight * (refDeriv[2] + warDeriv[2] -
                                                                               nmi * jointDeriv[2]) / (entropyPtr[2] * entropyPtr[3]));
            }// Check that the values are defined
        } // mask
    } // loop over all voxel
}
/* *************************************************************** */
void GetVoxelBasedSimilarityMeasureGradient(const nifti_image *referenceImage,
                                            const nifti_image *warpedImage,
                                            const unsigned short *referenceBinNumber,
                                            const unsigned short *floatingBinNumber,
                                            const double *const *jointHistogramLog,
                                            const double *const *entropyValues,
                                            const nifti_image *warpedGradient,
                                            nifti_image *voxelBasedGradient,
                                            const int *referenceMask,
                                            const int& currentTimepoint,
                                            const double& timepointWeight) {
    std::visit([&](auto&& refImgDataType) {
        using RefImgDataType = std::decay_t<decltype(refImgDataType)>;
        auto GetVoxelBasedNmiGradient = referenceImage->nz > 1 ? reg_getVoxelBasedNmiGradient3d<RefImgDataType> : reg_getVoxelBasedNmiGradient2d<RefImgDataType>;
        GetVoxelBasedNmiGradient(referenceImage,
                                 warpedImage,
                                 referenceBinNumber,
                                 floatingBinNumber,
                                 jointHistogramLog,
                                 entropyValues,
                                 warpedGradient,
                                 voxelBasedGradient,
                                 referenceMask,
                                 currentTimepoint,
                                 timepointWeight);
    }, NiftiImage::getFloatingDataType(referenceImage));
}
/* *************************************************************** */
void reg_nmi::GetVoxelBasedSimilarityMeasureGradientFw(int currentTimepoint) {
    // Call compute similarity measure to calculate joint histogram
    this->GetSimilarityMeasureValue();

    ::GetVoxelBasedSimilarityMeasureGradient(this->referenceImage,
                                             this->warpedImage,
                                             this->referenceBinNumber,
                                             this->floatingBinNumber,
                                             this->jointHistogramLog,
                                             this->entropyValues,
                                             this->warpedGradient,
                                             this->voxelBasedGradient,
                                             this->referenceMask,
                                             currentTimepoint,
                                             this->timePointWeight[currentTimepoint]);
}
/* *************************************************************** */
void reg_nmi::GetVoxelBasedSimilarityMeasureGradientBw(int currentTimepoint) {
    ::GetVoxelBasedSimilarityMeasureGradient(this->floatingImage,
                                             this->warpedImageBw,
                                             this->floatingBinNumber,
                                             this->referenceBinNumber,
                                             this->jointHistogramLogBw,
                                             this->entropyValuesBw,
                                             this->warpedGradientBw,
                                             this->voxelBasedGradientBw,
                                             this->floatingMask,
                                             currentTimepoint,
                                             this->timePointWeight[currentTimepoint]);
}
/* *************************************************************** */
