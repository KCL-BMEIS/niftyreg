/*
 *  _reg_nmi.h
 *
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#pragma once

#include "_reg_measure.h"
#include <vector>
#ifdef _OPENMP
#include "omp.h"
#endif

/* *************************************************************** */
/// @brief NMI measure of similarity class
class reg_nmi: public reg_measure {
public:
    /// @brief reg_nmi class constructor
    reg_nmi();
    /// @brief reg_nmi class destructor
    virtual ~reg_nmi();

    virtual void InitialiseMeasure(nifti_image *refImg,
                                   nifti_image *floImg,
                                   int *refMask,
                                   nifti_image *warpedImg,
                                   nifti_image *warpedGrad,
                                   nifti_image *voxelBasedGrad,
                                   nifti_image *localWeightSim = nullptr,
                                   int *floMask = nullptr,
                                   nifti_image *warpedImgBw = nullptr,
                                   nifti_image *warpedGradBw = nullptr,
                                   nifti_image *voxelBasedGradBw = nullptr) override;
    /// @brief Returns the nmi value forwards
    virtual double GetSimilarityMeasureValueFw() override;
    /// @brief Returns the nmi value backwards
    virtual double GetSimilarityMeasureValueBw() override;
    /// @brief Compute the voxel-based nmi gradient forwards
    virtual void GetVoxelBasedSimilarityMeasureGradientFw(int currentTimepoint) override;
    /// @brief Compute the voxel-based nmi gradient backwards
    virtual void GetVoxelBasedSimilarityMeasureGradientBw(int currentTimepoint) override;

    virtual void SetRefAndFloatBinNumbers(unsigned short refBinNumber,
                                          unsigned short floBinNumber,
                                          int timepoint) {
        this->referenceBinNumber[timepoint] = refBinNumber;
        this->floatingBinNumber[timepoint] = floBinNumber;
    }
    virtual void SetReferenceBinNumber(int b, int t) {
        this->referenceBinNumber[t] = b;
    }
    virtual void SetFloatingBinNumber(int b, int t) {
        this->floatingBinNumber[t] = b;
    }
    virtual unsigned short* GetReferenceBinNumber() {
        return this->referenceBinNumber;
    }
    virtual unsigned short* GetFloatingBinNumber() {
        return this->floatingBinNumber;
    }
    virtual void ApproximatePW() {
        this->approximatePW = true;
    }
    virtual void DoNotApproximatePW() {
        this->approximatePW = false;
    }

protected:
    bool approximatePW;
    unsigned short referenceBinNumber[255];
    unsigned short floatingBinNumber[255];
    unsigned short totalBinNumber[255];
    double **jointHistogramPro;
    double **jointHistogramLog;
    double **entropyValues;
    double **jointHistogramProBw;
    double **jointHistogramLogBw;
    double **entropyValuesBw;

    void DeallocateHistogram();
};
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
                     const int *referenceMask,
                     const bool approximation);
/* *************************************************************** */
// Simple class to dynamically manage an array of pointers
// Needed for multi channel NMI
template<class DataTYPE>
class SafeArray {
public:
    /// Constructor
    SafeArray(int items) {
        data = new DataTYPE[items];
    }

    /// Destructor
    ~SafeArray() {
        delete[] data;
    }

    /// Implicit conversion
    operator DataTYPE *() {
        return data;
    }

private:
    void operator=(const SafeArray &) {};
    SafeArray(const SafeArray &) {};

    DataTYPE *data;
};

//-----------------------------------------------------------------------------
// Template for emulating nested multiple loops, where the number of nested loops
// is only known at runtime.
// The index type may be any incrementable type, including pointers and iterators.
// 'end' values are like the STL ranges, where they signify one past the last value.
//-----------------------------------------------------------------------------
template<typename T>
class Multi_Loop {
public:
    /// Add a for loop to the list
    void Add(T begin_value, T end_value) {
        begin.push_back(begin_value);
        end.push_back(end_value);
    }

    // Initialises the loops before use.
    void Initialise() {
        current.resize(Count());
        std::copy(begin.begin(), begin.end(), current.begin());
    }

    /// Gets the index or iterator for the specified loop.
    T Index(int index) const {
        return (current[index]);
    }

    /// Gets the index or iterator for the specified loop.
    const T &operator [](int index) const {
        return (current[index]);
    }

    /// Tests to see if the loops continue.
    bool Continue() const {
        return (current[0] != end[0]);
    }

    /// Compute the next set of indexes or iterators in the sequence.
    void Next() {
        int position = begin.size() - 1;
        bool finished = false;

        while (!finished) {
            ++current[position];
            // Finished incrementing?
            if ((current[position] != end[position]) || (position == 0)) {
                finished = true;
            } else {
                // Reset this index, and move on to the previous one.
                current[position] = begin[position];
                --position;
            }
        }
    }

    /// Returns the number of 'for' loops added.
    int Count() const {
        return (static_cast<int>(begin.size()));
    }

private:
    std::vector<T> begin;   // Start for each loop.
    std::vector<T> end;     // End for each loop.
    std::vector<T> current; // Current position of each loop
};

/// Some methods that will be needed for generating the multi-channel histogram
/// Needed for multi channel NMI
inline int calculate_product(int dim, int *dimensions) {
    int product = 1;
    for (int i = 0; i < dim; ++i)
        product *= dimensions[i];

    return product;
}

inline int calculate_index(int num_dims, int *dimensions, int *indices) {
    int index = 0;
    for (int i = 0; i < num_dims; ++i)
        index += indices[i] * calculate_product(i, dimensions);

    return index;
}

inline int previous(int current, int num_dims) {
    if (current > 0)
        return current - 1;

    return num_dims - 1;
}
/* *************************************************************** */
/// @brief NMI measure of similarity class
class reg_multichannel_nmi: public reg_measure {
public:
    /// @brief reg_multichannel_nmi class constructor
    reg_multichannel_nmi() {}
    /// @brief reg_multichannel_nmi class destructor
    virtual ~reg_multichannel_nmi() {}

    /// @brief Returns the nmi value forwards
    virtual double GetSimilarityMeasureValueFw() override { return 0; }
    /// @brief Returns the nmi value backwards
    virtual double GetSimilarityMeasureValueBw() override { return 0; }

    /// @brief Compute the voxel-based nmi gradient forwards
    virtual void GetVoxelBasedSimilarityMeasureGradientFw(int currentTimepoint) override {}
    /// @brief Compute the voxel-based nmi gradient backwards
    virtual void GetVoxelBasedSimilarityMeasureGradientBw(int currentTimepoint) override {}

protected:
    unsigned short referenceBinNumber[255];
    unsigned short floatingBinNumber[255];
    unsigned short totalBinNumber[255];
    double *jointHistogramProp;
    double *jointHistogramLog;
    double *entropyValues;
    double *jointHistogramPropBw;
    double *jointHistogramLogBw;
    double *entropyValuesBw;
};
/* *************************************************************** */
/// Multi channel NMI version - Entropy
void reg_getMultiChannelNmiValue(nifti_image *referenceImages,
                                 nifti_image *warpedImages,
                                 unsigned *referenceBins, // should be an array of size num_reference_volumes
                                 unsigned *warpedBins, // should be an array of size num_warped_volumes
                                 double *probaJointHistogram,
                                 double *logJointHistogram,
                                 double *entropies,
                                 int *mask,
                                 bool approx);
/* *************************************************************** */
/// Multi channel NMI version - Gradient
void reg_getVoxelBasedMultiChannelNmiGradient2D(nifti_image *referenceImages,
                                                nifti_image *warpedImages,
                                                nifti_image *warpedImageGradient,
                                                unsigned *referenceBins,
                                                unsigned *warpedBins,
                                                double *logJointHistogram,
                                                double *entropies,
                                                nifti_image *nmiGradientImage,
                                                int *mask,
                                                bool approx);
/* *************************************************************** */
/// Multi channel NMI version - Gradient
void reg_getVoxelBasedMultiChannelNmiGradient3D(nifti_image *referenceImages,
                                                nifti_image *warpedImages,
                                                nifti_image *warpedImageGradient,
                                                unsigned *referenceBins,
                                                unsigned *warpedBins,
                                                double *logJointHistogram,
                                                double *entropies,
                                                nifti_image *nmiGradientImage,
                                                int *mask,
                                                bool approx);
/* *************************************************************** */
