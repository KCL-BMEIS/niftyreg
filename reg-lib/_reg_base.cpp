/**
 * @file _reg_base.cpp
 * @author Marc Modat
 * @date 15/11/2012
 *
 *  Copyright (c) 2012-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_base.h"

/* *************************************************************** */
template<class T>
reg_base<T>::reg_base(int refTimePoints, int floTimePoints) {
    SetPlatformType(PlatformType::Cpu);

    maxIterationNumber = 150;
    optimiseX = true;
    optimiseY = true;
    optimiseZ = true;
    perturbationNumber = 0;
    useConjGradient = true;
    useApproxGradient = false;

    similarityWeight = 0; // automatically set depending of the penalty term weights

    executableName = (char*)"NiftyReg BASE";
    referenceTimePoints = refTimePoints;
    floatingTimePoints = floTimePoints;
    referenceSmoothingSigma = 0;
    floatingSmoothingSigma = 0;

    referenceThresholdLow.reset(new T[referenceTimePoints]);
    std::fill(referenceThresholdLow.get(), referenceThresholdLow.get() + referenceTimePoints, std::numeric_limits<T>::lowest());
    referenceThresholdUp.reset(new T[referenceTimePoints]);
    std::fill(referenceThresholdUp.get(), referenceThresholdUp.get() + referenceTimePoints, std::numeric_limits<T>::max());
    floatingThresholdLow.reset(new T[floatingTimePoints]);
    std::fill(floatingThresholdLow.get(), floatingThresholdLow.get() + floatingTimePoints, std::numeric_limits<T>::lowest());
    floatingThresholdUp.reset(new T[floatingTimePoints]);
    std::fill(floatingThresholdUp.get(), floatingThresholdUp.get() + floatingTimePoints, std::numeric_limits<T>::max());

    robustRange = false;
    warpedPaddingValue = std::numeric_limits<T>::quiet_NaN();
    levelNumber = 3;
    levelToPerform = 0;
    gradientSmoothingSigma = 0;
    verbose = true;
    usePyramid = true;

    initialised = false;

    interpolation = 1;  // linear

    landmarkRegWeight = 0;
    landmarkRegNumber = 0;
    landmarkReference = nullptr;
    landmarkFloating = nullptr;

    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetReferenceImage(NiftiImage inputReferenceIn) {
    inputReference = inputReferenceIn;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetFloatingImage(NiftiImage inputFloatingIn) {
    inputFloating = inputFloatingIn;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetMaximalIterationNumber(unsigned iter) {
    maxIterationNumber = iter;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetReferenceMask(NiftiImage maskImageIn) {
    maskImage = maskImageIn;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetAffineTransformation(const mat44& affineTransformationIn) {
    affineTransformation.reset(new mat44(affineTransformationIn));
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetReferenceSmoothingSigma(T referenceSmoothingSigmaIn) {
    referenceSmoothingSigma = referenceSmoothingSigmaIn;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetFloatingSmoothingSigma(T floatingSmoothingSigmaIn) {
    floatingSmoothingSigma = floatingSmoothingSigmaIn;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetReferenceThresholdUp(unsigned i, T t) {
    referenceThresholdUp[i] = t;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetReferenceThresholdLow(unsigned i, T t) {
    referenceThresholdLow[i] = t;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetFloatingThresholdUp(unsigned i, T t) {
    floatingThresholdUp[i] = t;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetFloatingThresholdLow(unsigned i, T t) {
    floatingThresholdLow[i] = t;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseRobustRange() {
    robustRange = true;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::DoNotUseRobustRange() {
    robustRange = false;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetWarpedPaddingValue(float warpedPaddingValueIn) {
    warpedPaddingValue = warpedPaddingValueIn;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetLevelNumber(unsigned levelNumberIn) {
    if (levelNumberIn > 0)
        levelNumber = levelNumberIn;
    else
        NR_FATAL_ERROR("The number of level is expected to be strictly positive!");
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetLevelToPerform(unsigned levelToPerformIn) {
    levelToPerform = levelToPerformIn;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetGradientSmoothingSigma(T gradientSmoothingSigmaIn) {
    gradientSmoothingSigma = gradientSmoothingSigmaIn;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseConjugateGradient() {
    useConjGradient = true;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::DoNotUseConjugateGradient() {
    useConjGradient = false;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseApproximatedGradient() {
    useApproxGradient = true;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::DoNotUseApproximatedGradient() {
    useApproxGradient = false;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::PrintOutInformation() {
    verbose = true;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::DoNotPrintOutInformation() {
    verbose = false;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::DoNotUsePyramidalApproach() {
    usePyramid = false;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseNearestNeighborInterpolation() {
    interpolation = 0;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseLinearInterpolation() {
    interpolation = 1;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseCubicSplineInterpolation() {
    interpolation = 3;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetLandmarkRegularisationParam(size_t n, float *r, float *f, float w) {
    landmarkRegNumber = n;
    landmarkReference = r;
    landmarkFloating = f;
    landmarkRegWeight = w;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::CheckParameters() {
    // Check if both input images are defined
    if (!inputReference)
        NR_FATAL_ERROR("The reference image is not defined");
    if (!inputFloating)
        NR_FATAL_ERROR("The floating image is not defined");

    // Check the mask dimension if it is defined
    if (maskImage && (inputReference->nx != maskImage->nx ||
                      inputReference->ny != maskImage->ny ||
                      inputReference->nz != maskImage->nz))
        NR_FATAL_ERROR("The reference and mask images have different dimension");

    // Check the number of level to perform
    if (levelToPerform > 0) {
        levelToPerform = levelToPerform < levelNumber ? levelToPerform : levelNumber;
    } else levelToPerform = levelNumber;
    if (levelToPerform == 0 || levelToPerform > levelNumber)
        levelToPerform = levelNumber;

    // Set the default similarity measure if none has been set
    if (!measure_nmi && !measure_ssd && !measure_dti && !measure_lncc &&
        !measure_kld && !measure_mind && !measure_mindssc) {
        measure_nmi.reset(dynamic_cast<reg_nmi*>(measure->Create(MeasureType::Nmi)));
        for (int i = 0; i < inputReference->nt; ++i)
            measure_nmi->SetTimePointWeight(i, 1.0);
    }

    // Check that images have same number of channels (time points)
    // that each channel has at least one similarity measure assigned
    // and that each similarity measure is used for at least one channel
    // Normalise channel and similarity weights so total = 1
    //
    // NOTE - DTI currently ignored as needs fixing
    //
    // Tests are ignored if using MIND or MINDSSC as they are not implemented for multi-channel or weighting
    if (!measure_mind && !measure_mindssc) {
        if (inputFloating->nt != inputReference->nt)
            NR_FATAL_ERROR("The reference and floating images have different numbers of channels (time points)");
        unique_ptr<double[]> chanWeightSum(new double[inputReference->nt]());
        double simWeightSum, totWeightSum = 0.;
        double *nmiWeights = nullptr, *ssdWeights = nullptr, *kldWeights = nullptr, *lnccWeights = nullptr;
        if (measure_nmi) {
            nmiWeights = measure_nmi->GetTimePointWeights();
            simWeightSum = 0;
            for (int n = 0; n < inputReference->nt; n++) {
                if (nmiWeights[n] < 0)
                    NR_FATAL_ERROR("The NMI weight for time point " + std::to_string(n) + " has a negative value - weights must be positive");
                chanWeightSum[n] += nmiWeights[n];
                simWeightSum += nmiWeights[n];
                totWeightSum += nmiWeights[n];
            }
            if (simWeightSum == 0)
                NR_WARN_WFCT("The NMI similarity measure has a weight of 0 for all channels so will be ignored");
        }
        if (measure_ssd) {
            ssdWeights = measure_ssd->GetTimePointWeights();
            simWeightSum = 0;
            for (int n = 0; n < inputReference->nt; n++) {
                if (ssdWeights[n] < 0)
                    NR_FATAL_ERROR("The SSD weight for time point " + std::to_string(n) + " has a negative value - weights must be positive");
                chanWeightSum[n] += ssdWeights[n];
                simWeightSum += ssdWeights[n];
                totWeightSum += ssdWeights[n];
            }
            if (simWeightSum == 0)
                NR_WARN_WFCT("The SSD similarity measure has a weight of 0 for all channels so will be ignored");
        }
        if (measure_kld) {
            kldWeights = measure_kld->GetTimePointWeights();
            simWeightSum = 0;
            for (int n = 0; n < inputReference->nt; n++) {
                if (kldWeights[n] < 0)
                    NR_FATAL_ERROR("The KLD weight for time point " + std::to_string(n) + " has a negative value - weights must be positive");
                chanWeightSum[n] += kldWeights[n];
                simWeightSum += kldWeights[n];
                totWeightSum += kldWeights[n];
            }
            if (simWeightSum == 0)
                NR_WARN_WFCT("The KLD similarity measure has a weight of 0 for all channels so will be ignored");
        }
        if (measure_lncc) {
            lnccWeights = measure_lncc->GetTimePointWeights();
            simWeightSum = 0;
            for (int n = 0; n < inputReference->nt; n++) {
                if (lnccWeights[n] < 0)
                    NR_FATAL_ERROR("The LNCC weight for time point " + std::to_string(n) + " has a negative value - weights must be positive");
                chanWeightSum[n] += lnccWeights[n];
                simWeightSum += lnccWeights[n];
                totWeightSum += lnccWeights[n];
            }
            if (simWeightSum == 0)
                NR_WARN_WFCT("The LNCC similarity measure has a weight of 0 for all channels so will be ignored");
        }
        for (int n = 0; n < inputReference->nt; n++) {
            if (chanWeightSum[n] == 0)
                NR_WARN_WFCT("Channel " << n << " has a weight of 0 for all similarity measures so will be ignored");
            if (measure_nmi)
                measure_nmi->SetTimePointWeight(n, nmiWeights[n] / totWeightSum);
            if (measure_ssd)
                measure_ssd->SetTimePointWeight(n, ssdWeights[n] / totWeightSum);
            if (measure_kld)
                measure_kld->SetTimePointWeight(n, kldWeights[n] / totWeightSum);
            if (measure_lncc)
                measure_lncc->SetTimePointWeight(n, lnccWeights[n] / totWeightSum);
        }
    }

    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::InitialiseSimilarity() {
    DefContent& con = dynamic_cast<DefContent&>(*this->con);

    if (measure_nmi)
        measure->Initialise(*measure_nmi, con);

    if (measure_ssd)
        measure->Initialise(*measure_ssd, con);

    if (measure_kld)
        measure->Initialise(*measure_kld, con);

    if (measure_lncc)
        measure->Initialise(*measure_lncc, con);

    if (measure_dti)
        measure->Initialise(*measure_dti, con);

    if (measure_mind)
        measure->Initialise(*measure_mind, con);

    if (measure_mindssc)
        measure->Initialise(*measure_mindssc, con);

    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::Initialise() {
    if (initialised) return;

    CheckParameters();

    // CREATE THE PYRAMID IMAGES
    const unsigned imageCount = usePyramid ? levelToPerform : 1;
    referencePyramid = vector<NiftiImage>(imageCount);
    floatingPyramid = vector<NiftiImage>(imageCount);
    maskPyramid = vector<unique_ptr<int[]>>(imageCount);

    // Update the input images threshold if required
    if (robustRange) {
        // Create a copy of the reference image to extract the robust range
        NiftiImage tmpReference = inputReference;
        reg_tools_changeDatatype<T>(tmpReference);
        // Extract the robust range of the reference image
        T *refDataPtr = static_cast<T *>(tmpReference->data);
        reg_heapSort(refDataPtr, tmpReference->nvox);
        // Update the reference threshold values if no value has been setup by the user
        if (referenceThresholdLow[0] == std::numeric_limits<T>::lowest())
            referenceThresholdLow[0] = refDataPtr[Round((float)tmpReference->nvox * 0.02f)];
        if (referenceThresholdUp[0] == std::numeric_limits<T>::max())
            referenceThresholdUp[0] = refDataPtr[Round((float)tmpReference->nvox * 0.98f)];

        // Create a copy of the floating image to extract the robust range
        NiftiImage tmpFloating = inputFloating;
        reg_tools_changeDatatype<T>(tmpFloating);
        // Extract the robust range of the floating image
        T *floDataPtr = static_cast<T *>(tmpFloating->data);
        reg_heapSort(floDataPtr, tmpFloating->nvox);
        // Update the floating threshold values if no value has been setup by the user
        if (floatingThresholdLow[0] == std::numeric_limits<T>::lowest())
            floatingThresholdLow[0] = floDataPtr[Round((float)tmpFloating->nvox * 0.02f)];
        if (floatingThresholdUp[0] == std::numeric_limits<T>::max())
            floatingThresholdUp[0] = floDataPtr[Round((float)tmpFloating->nvox * 0.98f)];
    }

    // FINEST LEVEL OF REGISTRATION
    const unsigned levelCount = usePyramid ? levelNumber : 1;
    reg_createImagePyramid<T>(inputReference, referencePyramid, levelCount, imageCount);
    reg_createImagePyramid<T>(inputFloating, floatingPyramid, levelCount, imageCount);
    if (maskImage)
        reg_createMaskPyramid<T>(maskImage, maskPyramid, levelCount, imageCount);
    else
        for (unsigned l = 0; l < imageCount; ++l)
            maskPyramid[l].reset(new int[referencePyramid[l].nVoxelsPerVolume()]());

    // SMOOTH THE INPUT IMAGES IF REQUIRED
    for (unsigned l = 0; l < levelToPerform; l++) {
        if (referenceSmoothingSigma != 0) {
            unique_ptr<bool[]> active(new bool[referencePyramid[l]->nt]);
            unique_ptr<float[]> sigma(new float[referencePyramid[l]->nt]);
            active[0] = true;
            for (int i = 1; i < referencePyramid[l]->nt; ++i)
                active[i] = false;
            sigma[0] = referenceSmoothingSigma;
            reg_tools_kernelConvolution(referencePyramid[l], sigma.get(), ConvKernelType::Gaussian, nullptr, active.get());
        }
        if (floatingSmoothingSigma != 0) {
            // Only the first image is smoothed
            unique_ptr<bool[]> active(new bool[floatingPyramid[l]->nt]);
            unique_ptr<float[]> sigma(new float[floatingPyramid[l]->nt]);
            active[0] = true;
            for (int i = 1; i < floatingPyramid[l]->nt; ++i)
                active[i] = false;
            sigma[0] = floatingSmoothingSigma;
            reg_tools_kernelConvolution(floatingPyramid[l], sigma.get(), ConvKernelType::Gaussian, nullptr, active.get());
        }
    }

    // THRESHOLD THE INPUT IMAGES IF REQUIRED
    for (unsigned l = 0; l < imageCount; l++) {
        reg_thresholdImage<T>(referencePyramid[l], referenceThresholdLow[0], referenceThresholdUp[0]);
        reg_thresholdImage<T>(floatingPyramid[l], referenceThresholdLow[0], referenceThresholdUp[0]);
    }

    initialised = true;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
double reg_base<T>::ComputeSimilarityMeasure() {
    double measure = 0;
    if (measure_nmi)
        measure += measure_nmi->GetSimilarityMeasureValue();

    if (measure_ssd)
        measure += measure_ssd->GetSimilarityMeasureValue();

    if (measure_kld)
        measure += measure_kld->GetSimilarityMeasureValue();

    if (measure_lncc)
        measure += measure_lncc->GetSimilarityMeasureValue();

    if (measure_dti)
        measure += measure_dti->GetSimilarityMeasureValue();

    if (measure_mind)
        measure += measure_mind->GetSimilarityMeasureValue();

    if (measure_mindssc)
        measure += measure_mindssc->GetSimilarityMeasureValue();

    NR_FUNC_CALLED();
    return similarityWeight * measure;
}
/* *************************************************************** */
template<class T>
void reg_base<T>::GetVoxelBasedGradient() {
    // The voxel based gradient image is filled with zeros
    dynamic_cast<DefContent&>(*con).ZeroVoxelBasedMeasureGradient();

    // The intensity gradient is first computed
    //   if(measure_nmi || measure_ssd ||
    //         measure_kld || measure_lncc ||
    //         measure_dti)
    //   {
    //    if(measure_dti){
    //        reg_getImageGradient(floating,
    //                             warpedGradient,
    //                             deformationFieldImage,
    //                             currentMask,
    //                             interpolation,
    //                             warpedPaddingValue,
    //                             measure_dti->GetActiveTimePoints(),
    //		 					   forwardJacobianMatrix,
    //							   warped);
    //    }
    //    else{
    //    }
    //   }

    //   if(measure_dti)
    //      measure_dti->GetVoxelBasedSimilarityMeasureGradient();

    for (int t = 0; t < con->Content::GetReference()->nt; ++t) {
        compute->GetImageGradient(interpolation, warpedPaddingValue, t);

        // The gradient of the various measures of similarity are computed
        if (measure_nmi)
            measure_nmi->GetVoxelBasedSimilarityMeasureGradient(t);

        if (measure_ssd)
            measure_ssd->GetVoxelBasedSimilarityMeasureGradient(t);

        if (measure_kld)
            measure_kld->GetVoxelBasedSimilarityMeasureGradient(t);

        if (measure_lncc)
            measure_lncc->GetVoxelBasedSimilarityMeasureGradient(t);

        if (measure_mind)
            measure_mind->GetVoxelBasedSimilarityMeasureGradient(t);

        if (measure_mindssc)
            measure_mindssc->GetVoxelBasedSimilarityMeasureGradient(t);
    }

    NR_FUNC_CALLED();
}
/* *************************************************************** */
//template<class T>
//void reg_base<T>::ApproximateParzenWindow()
//{
//    if(!measure_nmi)
//        measure_nmi.reset(dynamic_cast<reg_nmi*>(measure->Create(MeasureType::Nmi)));
//    measure_nmi=approxParzenWindow = true;
//}
///* *************************************************************** */
//template<class T>
//void reg_base<T>::DoNotApproximateParzenWindow()
//{
//    if(!measure_nmi)
//        measure_nmi.reset(dynamic_cast<reg_nmi*>(measure->Create(MeasureType::Nmi)));
//    measure_nmi=approxParzenWindow = false;
//}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseNMISetReferenceBinNumber(int timePoint, int refBinNumber) {
    if (!measure_nmi)
        measure_nmi.reset(dynamic_cast<reg_nmi*>(measure->Create(MeasureType::Nmi)));
    measure_nmi->SetTimePointWeight(timePoint, 1.0);//weight initially set to default value of 1.0
    // I am here adding 4 to the specified bin number to accommodate for
    // the spline support
    measure_nmi->SetReferenceBinNumber(refBinNumber + 4, timePoint);
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseNMISetFloatingBinNumber(int timePoint, int floBinNumber) {
    if (!measure_nmi)
        measure_nmi.reset(dynamic_cast<reg_nmi*>(measure->Create(MeasureType::Nmi)));
    measure_nmi->SetTimePointWeight(timePoint, 1.0);//weight initially set to default value of 1.0
    // I am here adding 4 to the specified bin number to accommodate for
    // the spline support
    measure_nmi->SetFloatingBinNumber(floBinNumber + 4, timePoint);
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseSSD(int timePoint, bool normalise) {
    if (!measure_ssd)
        measure_ssd.reset(dynamic_cast<reg_ssd*>(measure->Create(MeasureType::Ssd)));
    measure_ssd->SetTimePointWeight(timePoint, 1.0);//weight initially set to default value of 1.0
    measure_ssd->SetNormaliseTimePoint(timePoint, normalise);
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseMIND(int timePoint, int offset) {
    if (!measure_mind)
        measure_mind.reset(dynamic_cast<reg_mind*>(measure->Create(MeasureType::Mind)));
    measure_mind->SetTimePointWeight(timePoint, 1.0);//weight set to 1.0 to indicate time point is active
    measure_mind->SetDescriptorOffset(offset);
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseMINDSSC(int timePoint, int offset) {
    if (!measure_mindssc)
        measure_mindssc.reset(dynamic_cast<reg_mindssc*>(measure->Create(MeasureType::MindSsc)));
    measure_mindssc->SetTimePointWeight(timePoint, 1.0);//weight set to 1.0 to indicate time point is active
    measure_mindssc->SetDescriptorOffset(offset);
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseKLDivergence(int timePoint) {
    if (!measure_kld)
        measure_kld.reset(dynamic_cast<reg_kld*>(measure->Create(MeasureType::Kld)));
    measure_kld->SetTimePointWeight(timePoint, 1.0);//weight initially set to default value of 1.0
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseLNCC(int timePoint, float stddev) {
    if (!measure_lncc)
        measure_lncc.reset(dynamic_cast<reg_lncc*>(measure->Create(MeasureType::Lncc)));
    measure_lncc->SetKernelStandardDeviation(timePoint, stddev);
    measure_lncc->SetTimePointWeight(timePoint, 1.0); // weight initially set to default value of 1.0
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetLNCCKernelType(ConvKernelType type) {
    if (!measure_lncc)
        NR_FATAL_ERROR("The LNCC object has to be created first");
    measure_lncc->SetKernelType(type);
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::UseDTI(bool *timePoint) {
    NR_FATAL_ERROR("The use of DTI has been deactivated as it requires some refactoring");

    if (!measure_dti)
        measure_dti.reset(dynamic_cast<reg_dti*>(measure->Create(MeasureType::Dti)));
    for (int i = 0; i < inputReference->nt; ++i) {
        if (timePoint[i])
            measure_dti->SetTimePointWeight(i, 1.0);  // weight set to 1.0 to indicate time point is active
    }
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetNMIWeight(int timePoint, double weight) {
    if (!measure_nmi)
        NR_FATAL_ERROR("The NMI object has to be created before the time point weights can be set");
    measure_nmi->SetTimePointWeight(timePoint, weight);
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetLNCCWeight(int timePoint, double weight) {
    if (!measure_lncc)
        NR_FATAL_ERROR("The LNCC object has to be created before the time point weights can be set");
    measure_lncc->SetTimePointWeight(timePoint, weight);
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetSSDWeight(int timePoint, double weight) {
    if (!measure_ssd)
        NR_FATAL_ERROR("The SSD object has to be created before the time point weights can be set");
    measure_ssd->SetTimePointWeight(timePoint, weight);
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetKLDWeight(int timePoint, double weight) {
    if (!measure_kld)
        NR_FATAL_ERROR("The KLD object has to be created before the time point weights can be set");
    measure_kld->SetTimePointWeight(timePoint, weight);
}
/* *************************************************************** */
template<class T>
void reg_base<T>::SetLocalWeightSim(NiftiImage localWeightSimInputIn) {
    localWeightSimInput = localWeightSimInputIn;
    reg_tools_changeDatatype<T>(localWeightSimInput);
}
/* *************************************************************** */
template<class T>
void reg_base<T>::WarpFloatingImage(int inter) {
    // Compute the deformation field
    GetDeformationField();

    if (!measure_dti) {
        // Resample the floating image
        compute->ResampleImage(inter, warpedPaddingValue);
    } else {
        // reg_defField_getJacobianMatrix(deformationFieldImage, forwardJacobianMatrix);
        /*DTI needs fixing!
       reg_resampleImage(floating,
                          warped,
                          deformationFieldImage,
                          currentMask,
                          inter,
                          warpedPaddingValue,
                          measure_dti->GetActiveTimePoints(),
                          forwardJacobianMatrix);*/
    }
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_base<T>::DeinitCurrentLevel(int currentLevel) {
    optimiser = nullptr;
    if (currentLevel >= 0) {
        if (usePyramid) {
            referencePyramid[currentLevel] = nullptr;
            floatingPyramid[currentLevel] = nullptr;
            maskPyramid[currentLevel] = nullptr;
        } else if (currentLevel == levelToPerform - 1) {
            referencePyramid[0] = nullptr;
            floatingPyramid[0] = nullptr;
            maskPyramid[0] = nullptr;
        }
    }
}
/* *************************************************************** */
template<class T>
void reg_base<T>::Run() {
    NR_DEBUG(executableName << "::Run() called");

    Initialise();

    NR_VERBOSE("***********************************************************");

    // Update the maximal number of iteration to perform per level
    maxIterationNumber *= pow(2, levelToPerform - 1);

    // Loop over the different resolution level to perform
    for (int currentLevel = 0; currentLevel < levelToPerform; currentLevel++) {
        // The grid is refined if necessary
        T maxStepSize = InitCurrentLevel(currentLevel);
        T currentSize = maxStepSize;
        T smallestSize = maxStepSize / (T)100.0;

        DisplayCurrentLevelParameters(currentLevel);

        // Initialise the measures of similarity
        InitialiseSimilarity();

        // Initialise the optimiser
        SetOptimiser();

        // Loop over the number of perturbation to do
        for (size_t perturbation = 0; perturbation <= perturbationNumber; ++perturbation) {
            // Evaluate the objective function value
            UpdateBestObjFunctionValue();
            PrintInitialObjFunctionValue();

            // Iterate until convergence or until the max number of iteration is reach
            while (currentSize) {
                if (optimiser->GetCurrentIterationNumber() >= optimiser->GetMaxIterationNumber()) {
                    NR_WARN("The current level reached the maximum number of iteration");
                    break;
                }

                // Compute the objective function gradient
                GetObjectiveFunctionGradient();

                // Normalise the gradient
                NormaliseGradient();

                // Initialise the line search initial step size
                currentSize = std::min(currentSize, maxStepSize);

                // A line search is performed
                optimiser->Optimise(maxStepSize, smallestSize, currentSize);

                // Update the objective function variables and print some information
                PrintCurrentObjFunctionValue(currentSize);
            }

            if (perturbation < perturbationNumber) {
                optimiser->Perturbation(smallestSize);
                currentSize = maxStepSize;
                NR_VERBOSE("Perturbation Step - The number of iteration is reset to 0");
                NR_VERBOSE("Perturbation Step - Every control point positions is altered by [-" << smallestSize << " " << smallestSize << "]");
            }
        } // perturbation loop

        // Final folding correction
        CorrectTransformation();

        // Some cleaning is performed
        DeinitCurrentLevel(currentLevel);

        NR_VERBOSE("Current registration level done");
        NR_VERBOSE("***********************************************************");

        // Update the number of level for the next level
        maxIterationNumber /= 2;
    } // level levelToPerform

    NR_FUNC_CALLED();
}
/* *************************************************************** */
template class reg_base<float>;
