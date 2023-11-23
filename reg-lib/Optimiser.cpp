/** @file Optimiser.cpp
 * @author Marc Modat
 * @date 20/07/2012
 */

#include "Optimiser.hpp"

/* *************************************************************** */
namespace NiftyReg {
/* *************************************************************** */
template <class T>
Optimiser<T>::Optimiser() {
    this->dofNumber = 0;
    this->dofNumberBw = 0;
    this->ndim = 3;
    this->optimiseX = true;
    this->optimiseY = true;
    this->optimiseZ = true;
    this->currentDof = nullptr;
    this->currentDofBw = nullptr;
    this->bestDof = nullptr;
    this->bestDofBw = nullptr;
    this->isSymmetric = false;
    this->gradient = nullptr;
    this->currentIterationNumber = 0;
    this->currentObjFunctionValue = 0;
    this->maxIterationNumber = 0;
    this->bestObjFunctionValue = 0;
    this->intOpt = nullptr;
    this->gradientBw = nullptr;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template <class T>
Optimiser<T>::~Optimiser() {
    if (this->bestDof) {
        free(this->bestDof);
        this->bestDof = nullptr;
    }
    if (this->bestDofBw) {
        free(this->bestDofBw);
        this->bestDofBw = nullptr;
    }
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template <class T>
void Optimiser<T>::Initialise(size_t nvox,
                              int ndim,
                              bool optX,
                              bool optY,
                              bool optZ,
                              size_t maxIt,
                              size_t startIt,
                              InterfaceOptimiser *intOpt,
                              T *cppData,
                              T *gradData,
                              size_t nvoxBw,
                              T *cppDataBw,
                              T *gradDataBw) {
    this->dofNumber = nvox;
    this->ndim = ndim;
    this->optimiseX = optX;
    this->optimiseY = optY;
    this->optimiseZ = optZ;
    this->maxIterationNumber = maxIt;
    this->currentIterationNumber = startIt;
    this->currentDof = cppData;
    this->gradient = gradData;

    if (this->bestDof) free(this->bestDof);
    this->bestDof = (T*)malloc(this->dofNumber * sizeof(T));

    this->isSymmetric = nvoxBw > 0 && cppDataBw && gradDataBw;
    if (this->isSymmetric) {
        this->dofNumberBw = nvoxBw;
        this->currentDofBw = cppDataBw;
        this->gradientBw = gradDataBw;
        if (this->bestDofBw) free(this->bestDofBw);
        this->bestDofBw = (T*)malloc(this->dofNumberBw * sizeof(T));
    }

    this->StoreCurrentDof();

    this->intOpt = intOpt;
    this->bestObjFunctionValue = this->currentObjFunctionValue = this->intOpt->GetObjectiveFunctionValue();

    NR_FUNC_CALLED();
}
/* *************************************************************** */
template <class T>
void Optimiser<T>::RestoreBestDof() {
    // Restore forward transformation
    memcpy(this->currentDof, this->bestDof, this->dofNumber * sizeof(T));
    // Restore backward transformation if required
    if (this->isSymmetric)
        memcpy(this->currentDofBw, this->bestDofBw, this->dofNumberBw * sizeof(T));
}
/* *************************************************************** */
template <class T>
void Optimiser<T>::StoreCurrentDof() {
    // Save forward transformation
    memcpy(this->bestDof, this->currentDof, this->dofNumber * sizeof(T));
    // Save backward transformation if required
    if (this->isSymmetric)
        memcpy(this->bestDofBw, this->currentDofBw, this->dofNumberBw * sizeof(T));
}
/* *************************************************************** */
template <class T>
void Optimiser<T>::Perturbation(float length) {
    // Initialise the randomiser
    srand((unsigned)time(nullptr));
    // Reset the number of iteration
    this->currentIterationNumber = 0;
    // Create some perturbation for degree of freedom
    for (size_t i = 0; i < this->dofNumber; ++i) {
        this->currentDof[i] = this->bestDof[i] + length * (float)(rand() - RAND_MAX / 2) / ((float)RAND_MAX / 2.0f);
    }
    if (this->isSymmetric) {
        for (size_t i = 0; i < this->dofNumberBw; ++i) {
            this->currentDofBw[i] = this->bestDofBw[i] + length * (float)(rand() % 2001 - 1000) / 1000.f;
        }
    }
    this->StoreCurrentDof();
    this->currentObjFunctionValue = this->bestObjFunctionValue = this->intOpt->GetObjectiveFunctionValue();
}
/* *************************************************************** */
template <class T>
void Optimiser<T>::Optimise(T maxLength, T smallLength, T& startLength) {
    size_t lineIteration = 0;
    float addedLength = 0;
    float currentLength = static_cast<float>(startLength);

    // Start performing the line search
    while (currentLength > smallLength &&
           lineIteration < 12 &&
           this->currentIterationNumber < this->maxIterationNumber) {

        // Compute the gradient normalisation value
        float normValue = -currentLength;

        this->intOpt->UpdateParameters(normValue);

        // Compute the new value
        this->currentObjFunctionValue = this->intOpt->GetObjectiveFunctionValue();

        // Check if the update lead to an improvement of the objective function
        const bool isImproved = this->currentObjFunctionValue > this->bestObjFunctionValue;
        NR_DEBUG("[" << this->currentIterationNumber << "] objective function: " << this->currentObjFunctionValue <<
                 " | Increment " << currentLength << " | " << (isImproved ? "ACCEPTED" : "REJECTED"));
        if (isImproved) {
            // Improvement - Save the new objective function value
            this->intOpt->UpdateBestObjFunctionValue();
            this->bestObjFunctionValue = this->currentObjFunctionValue;
            // Update the total added length
            addedLength += currentLength;
            // Increase the step size
            currentLength *= 1.1f;
            currentLength = std::min(currentLength, static_cast<float>(maxLength));
            // Save the current deformation parametrisation
            this->StoreCurrentDof();
        } else {
            // No improvement - Decrease the step size
            currentLength *= 0.5;
        }
        this->IncrementCurrentIterationNumber();
        ++lineIteration;
    }
    // update the current size for the next iteration
    startLength = addedLength;
    // Restore the last best deformation parametrisation
    this->RestoreBestDof();
}
/* *************************************************************** */
template class Optimiser<float>;
template class Optimiser<double>;
/* *************************************************************** */
template <class T>
ConjugateGradient<T>::ConjugateGradient(): Optimiser<T>::Optimiser() {
    this->array1 = nullptr;
    this->array1Bw = nullptr;
    this->array2 = nullptr;
    this->array2Bw = nullptr;
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template <class T>
ConjugateGradient<T>::~ConjugateGradient() {
    if (this->array1) {
        free(this->array1);
        this->array1 = nullptr;
    }
    if (this->array1Bw) {
        free(this->array1Bw);
        this->array1Bw = nullptr;
    }
    if (this->array2) {
        free(this->array2);
        this->array2 = nullptr;
    }
    if (this->array2Bw) {
        free(this->array2Bw);
        this->array2Bw = nullptr;
    }
    NR_FUNC_CALLED();
}
/* *************************************************************** */
template <class T>
void ConjugateGradient<T>::Initialise(size_t nvox,
                                      int ndim,
                                      bool optX,
                                      bool optY,
                                      bool optZ,
                                      size_t maxIt,
                                      size_t startIt,
                                      InterfaceOptimiser *intOpt,
                                      T *cppData,
                                      T *gradData,
                                      size_t nvoxBw,
                                      T *cppDataBw,
                                      T *gradDataBw) {
    Optimiser<T>::Initialise(nvox, ndim, optX, optY, optZ, maxIt, startIt, intOpt, cppData, gradData, nvoxBw, cppDataBw, gradDataBw);
    this->firstCall = true;
    if (this->array1) free(this->array1);
    if (this->array2) free(this->array2);
    this->array1 = (T*)malloc(this->dofNumber * sizeof(T));
    this->array2 = (T*)malloc(this->dofNumber * sizeof(T));

    if (this->isSymmetric) {
        if (this->array1Bw) free(this->array1Bw);
        if (this->array2Bw) free(this->array2Bw);
        this->array1Bw = (T*)malloc(this->dofNumberBw * sizeof(T));
        this->array2Bw = (T*)malloc(this->dofNumberBw * sizeof(T));
    }

    NR_FUNC_CALLED();
}
/* *************************************************************** */
template <class T>
void ConjugateGradient<T>::UpdateGradientValues() {
#ifdef WIN32
    long i;
    long num = (long)this->dofNumber;
    long numBw = (long)this->dofNumberBw;
#else
    size_t i;
    size_t num = (size_t)this->dofNumber;
    size_t numBw = (size_t)this->dofNumberBw;
#endif

    T *gradientPtr = this->gradient;
    T *array1Ptr = this->array1;
    T *array2Ptr = this->array2;

    T *gradientPtrBw = this->gradientBw;
    T *array1PtrBw = this->array1Bw;
    T *array2PtrBw = this->array2Bw;

    if (this->firstCall) {
        NR_DEBUG("Conjugate gradient initialisation");
        // first conjugate gradient iteration
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(num,array1Ptr,array2Ptr,gradientPtr)
#endif
        for (i = 0; i < num; i++)
            array2Ptr[i] = array1Ptr[i] = -gradientPtr[i];
        if (this->isSymmetric) {
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(numBw,array1PtrBw,array2PtrBw,gradientPtrBw)
#endif
            for (i = 0; i < numBw; i++)
                array2PtrBw[i] = array1PtrBw[i] = -gradientPtrBw[i];
        }
        this->firstCall = false;
    } else {
        NR_DEBUG("Conjugate gradient update");
        double dgg = 0, gg = 0;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(num,array1Ptr,array2Ptr,gradientPtr) \
    reduction(+:gg, dgg)
#endif
        for (i = 0; i < num; i++) {
            gg += array2Ptr[i] * array1Ptr[i];
            dgg += (gradientPtr[i] + array1Ptr[i]) * gradientPtr[i];
        }
        double gam = dgg / gg;

        if (this->isSymmetric) {
            double dggBw = 0, ggBw = 0;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(numBw,array1PtrBw,array2PtrBw,gradientPtrBw) \
    reduction(+:ggBw) \
    reduction(+:dggBw)
#endif
            for (i = 0; i < numBw; i++) {
                ggBw += array2PtrBw[i] * array1PtrBw[i];
                dggBw += (gradientPtrBw[i] + array1PtrBw[i]) * gradientPtrBw[i];
            }
            gam = (dgg + dggBw) / (gg + ggBw);
        }
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(num,array1Ptr,array2Ptr,gradientPtr,gam)
#endif
        for (i = 0; i < num; i++) {
            array1Ptr[i] = -gradientPtr[i];
            array2Ptr[i] = static_cast<T>(array1Ptr[i] + gam * array2Ptr[i]);
            gradientPtr[i] = -array2Ptr[i];
        }
        if (this->isSymmetric) {
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(numBw,array1PtrBw,array2PtrBw,gradientPtrBw,gam)
#endif
            for (i = 0; i < numBw; i++) {
                array1PtrBw[i] = -gradientPtrBw[i];
                array2PtrBw[i] = static_cast<T>(array1PtrBw[i] + gam * array2PtrBw[i]);
                gradientPtrBw[i] = -array2PtrBw[i];
            }
        }
    }
}
/* *************************************************************** */
template <class T>
void ConjugateGradient<T>::Optimise(T maxLength, T smallLength, T& startLength) {
    this->UpdateGradientValues();
    Optimiser<T>::Optimise(maxLength, smallLength, startLength);
}
/* *************************************************************** */
template <class T>
void ConjugateGradient<T>::Perturbation(float length) {
    Optimiser<T>::Perturbation(length);
    this->firstCall = true;
}
/* *************************************************************** */
template class ConjugateGradient<float>;
template class ConjugateGradient<double>;
/* *************************************************************** */
template <class T>
Lbfgs<T>::Lbfgs(): Optimiser<T>::Optimiser() {
    this->stepToKeep = 5;
    this->oldDof = nullptr;
    this->oldGrad = nullptr;
    this->diffDof = nullptr;
    this->diffGrad = nullptr;
}
/* *************************************************************** */
template <class T>
Lbfgs<T>::~Lbfgs() {
    if (this->oldDof) {
        free(this->oldDof);
        this->oldDof = nullptr;
    }
    if (this->oldGrad) {
        free(this->oldGrad);
        this->oldGrad = nullptr;
    }
    for (size_t i = 0; i < this->stepToKeep; ++i) {
        if (this->diffDof[i]) {
            free(this->diffDof[i]);
            this->diffDof[i] = nullptr;
        }
        if (this->diffGrad[i]) {
            free(this->diffGrad[i]);
            this->diffGrad[i] = nullptr;
        }
    }
    if (this->diffDof) {
        free(this->diffDof);
        this->diffDof = nullptr;
    }
    if (this->diffGrad) {
        free(this->diffGrad);
        this->diffGrad = nullptr;
    }
}
/* *************************************************************** */
template <class T>
void Lbfgs<T>::Initialise(size_t nvox,
                          int ndim,
                          bool optX,
                          bool optY,
                          bool optZ,
                          size_t maxIt,
                          size_t startIt,
                          InterfaceOptimiser *intOpt,
                          T *cppData,
                          T *gradData,
                          size_t nvoxBw,
                          T *cppDataBw,
                          T *gradDataBw) {
    Optimiser<T>::Initialise(nvox, ndim, optX, optY, optZ, maxIt, startIt, intOpt, cppData, gradData, nvoxBw, cppDataBw, gradDataBw);
    this->stepToKeep = 5;
    this->diffDof = (T**)malloc(this->stepToKeep * sizeof(T*));
    this->diffGrad = (T**)malloc(this->stepToKeep * sizeof(T*));
    for (size_t i = 0; i < this->stepToKeep; ++i) {
        this->diffDof[i] = (T*)malloc(this->dofNumber * sizeof(T));
        this->diffGrad[i] = (T*)malloc(this->dofNumber * sizeof(T));
        if (this->diffDof[i] == nullptr || this->diffGrad[i] == nullptr)
            NR_FATAL_ERROR("Out of memory");
    }
    this->oldDof = (T*)malloc(this->dofNumber * sizeof(T));
    this->oldGrad = (T*)malloc(this->dofNumber * sizeof(T));
    if (this->oldDof == nullptr || this->oldGrad == nullptr)
        NR_FATAL_ERROR("Out of memory");
}
/* *************************************************************** */
template <class T>
void Lbfgs<T>::UpdateGradientValues() {
    NR_FATAL_ERROR("Not implemented");
}
/* *************************************************************** */
template <class T>
void Lbfgs<T>::Optimise(T maxLength, T smallLength, T& startLength) {
    this->UpdateGradientValues();
    Optimiser<T>::Optimise(maxLength, smallLength, startLength);
}
/* *************************************************************** */
} // namespace NiftyReg
/* *************************************************************** */
