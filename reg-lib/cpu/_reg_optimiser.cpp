/** @file _reg_optimiser.cpp
 * @author Marc Modat
 * @date 20/07/2012
 */

#include "_reg_optimiser.h"

/* *************************************************************** */
template <class T>
reg_optimiser<T>::reg_optimiser() {
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
    this->isBackwards = false;
    this->gradient = nullptr;
    this->currentIterationNumber = 0;
    this->currentObjFunctionValue = 0;
    this->maxIterationNumber = 0;
    this->bestObjFunctionValue = 0;
    this->intOpt = nullptr;
    this->gradientBw = nullptr;

#ifndef NDEBUG
    reg_print_msg_debug("reg_optimiser<T>::reg_optimiser() called");
#endif
}
/* *************************************************************** */
template <class T>
reg_optimiser<T>::~reg_optimiser() {
    if (this->bestDof) {
        free(this->bestDof);
        this->bestDof = nullptr;
    }
    if (this->bestDofBw) {
        free(this->bestDofBw);
        this->bestDofBw = nullptr;
    }
#ifndef NDEBUG
    reg_print_msg_debug("reg_optimiser<T>::~reg_optimiser() called");
#endif
}
/* *************************************************************** */
template <class T>
void reg_optimiser<T>::Initialise(size_t nvox,
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
    if (this->bestDof != nullptr) free(this->bestDof);
    this->bestDof = (T*)malloc(this->dofNumber * sizeof(T));
    memcpy(this->bestDof, this->currentDof, this->dofNumber * sizeof(T));
    if (gradData)
        this->gradient = gradData;

    if (nvoxBw > 0)
        this->dofNumberBw = nvoxBw;
    if (cppDataBw) {
        this->currentDofBw = cppDataBw;
        this->isBackwards = true;
        if (this->bestDofBw != nullptr) free(this->bestDofBw);
        this->bestDofBw = (T*)malloc(this->dofNumberBw * sizeof(T));
        memcpy(this->bestDofBw, this->currentDofBw, this->dofNumberBw * sizeof(T));
    }
    if (gradDataBw)
        this->gradientBw = gradDataBw;

    this->intOpt = intOpt;
    this->bestObjFunctionValue = this->currentObjFunctionValue = this->intOpt->GetObjectiveFunctionValue();

#ifndef NDEBUG
    reg_print_msg_debug("reg_optimiser<T>::Initialise called");
#endif
}
/* *************************************************************** */
template <class T>
void reg_optimiser<T>::RestoreBestDof() {
    // restore forward transformation
    memcpy(this->currentDof, this->bestDof, this->dofNumber * sizeof(T));
    // restore backward transformation if required
    if (this->currentDofBw && this->bestDofBw && this->dofNumberBw > 0)
        memcpy(this->currentDofBw, this->bestDofBw, this->dofNumberBw * sizeof(T));
}
/* *************************************************************** */
template <class T>
void reg_optimiser<T>::StoreCurrentDof() {
    // save forward transformation
    memcpy(this->bestDof, this->currentDof, this->dofNumber * sizeof(T));
    // save backward transformation if required
    if (this->currentDofBw && this->bestDofBw && this->dofNumberBw > 0)
        memcpy(this->bestDofBw, this->currentDofBw, this->dofNumberBw * sizeof(T));
}
/* *************************************************************** */
template <class T>
void reg_optimiser<T>::Perturbation(float length) {
    // initialise the randomiser
    srand((unsigned)time(nullptr));
    // Reset the number of iteration
    this->currentIterationNumber = 0;
    // Create some perturbation for degree of freedom
    for (size_t i = 0; i < this->dofNumber; ++i) {
        this->currentDof[i] = this->bestDof[i] + length * (float)(rand() - RAND_MAX / 2) / ((float)RAND_MAX / 2.0f);
    }
    if (this->isBackwards) {
        for (size_t i = 0; i < this->dofNumberBw; ++i) {
            this->currentDofBw[i] = this->bestDofBw[i] + length * (float)(rand() % 2001 - 1000) / 1000.f;
        }
    }
    this->StoreCurrentDof();
    this->currentObjFunctionValue = this->bestObjFunctionValue = this->intOpt->GetObjectiveFunctionValue();
}
/* *************************************************************** */
template <class T>
void reg_optimiser<T>::Optimise(T maxLength, T smallLength, T& startLength) {
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
        if (this->currentObjFunctionValue > this->bestObjFunctionValue) {
#ifndef NDEBUG
            char text[255];
            sprintf(text, "[%i] objective function: %g | Increment %g | ACCEPTED",
                    (int)this->currentIterationNumber,
                    this->currentObjFunctionValue,
                    currentLength);
            reg_print_msg_debug(text);
#endif
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
#ifndef NDEBUG
            char text[255];
            sprintf(text, "[%i] objective function: %g | Increment %g | REJECTED",
                    (int)this->currentIterationNumber,
                    this->currentObjFunctionValue,
                    currentLength);
            reg_print_msg_debug(text);
#endif
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
template <class T>
reg_conjugateGradient<T>::reg_conjugateGradient(): reg_optimiser<T>::reg_optimiser() {
    this->array1 = nullptr;
    this->array2 = nullptr;
    this->array1Bw = nullptr;
    this->array2Bw = nullptr;

#ifndef NDEBUG
    reg_print_msg_debug("reg_conjugateGradient<T>::reg_conjugateGradient() called");
#endif
}
/* *************************************************************** */
template <class T>
reg_conjugateGradient<T>::~reg_conjugateGradient() {
    if (this->array1) {
        free(this->array1);
        this->array1 = nullptr;
    }

    if (this->array2) {
        free(this->array2);
        this->array2 = nullptr;
    }

    if (this->array1Bw) {
        free(this->array1Bw);
        this->array1Bw = nullptr;
    }

    if (this->array2Bw) {
        free(this->array2Bw);
        this->array2Bw = nullptr;
    }

#ifndef NDEBUG
    reg_print_msg_debug("reg_conjugateGradient<T>::~reg_conjugateGradient() called");
#endif
}
/* *************************************************************** */
template <class T>
void reg_conjugateGradient<T>::Initialise(size_t nvox,
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
    reg_optimiser<T>::Initialise(nvox, ndim, optX, optY, optZ, maxIt, startIt, intOpt, cppData, gradData, nvoxBw, cppDataBw, gradDataBw);
    this->firstCall = true;
    if (this->array1) free(this->array1);
    if (this->array2) free(this->array2);
    this->array1 = (T*)malloc(this->dofNumber * sizeof(T));
    this->array2 = (T*)malloc(this->dofNumber * sizeof(T));

    if (cppDataBw && gradDataBw && nvoxBw > 0) {
        if (this->array1Bw) free(this->array1Bw);
        if (this->array2Bw) free(this->array2Bw);
        this->array1Bw = (T*)malloc(this->dofNumberBw * sizeof(T));
        this->array2Bw = (T*)malloc(this->dofNumberBw * sizeof(T));
    }

#ifndef NDEBUG
    reg_print_msg_debug("reg_conjugateGradient<T>::Initialise called");
#endif
}
/* *************************************************************** */
template <class T>
void reg_conjugateGradient<T>::UpdateGradientValues() {
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
#ifndef NDEBUG
        reg_print_msg_debug("Conjugate gradient initialisation");
#endif
        // first conjugate gradient iteration
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(num,array1Ptr,array2Ptr,gradientPtr)
#endif
        for (i = 0; i < num; i++) {
            array2Ptr[i] = array1Ptr[i] = -gradientPtr[i];
        }
        if (this->dofNumberBw > 0) {
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(numBw,array1PtrBw,array2PtrBw,gradientPtrBw)
#endif
            for (i = 0; i < numBw; i++) {
                array2PtrBw[i] = array1PtrBw[i] = -gradientPtrBw[i];
            }
        }
        this->firstCall = false;
    } else {
#ifndef NDEBUG
        reg_print_msg_debug("Conjugate gradient update");
#endif
        double dgg = 0, gg = 0;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(num,array1Ptr,array2Ptr,gradientPtr) \
    reduction(+:gg) \
    reduction(+:dgg)
#endif
        for (i = 0; i < num; i++) {
            gg += array2Ptr[i] * array1Ptr[i];
            dgg += (gradientPtr[i] + array1Ptr[i]) * gradientPtr[i];
        }
        double gam = dgg / gg;

        if (this->dofNumberBw > 0) {
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
        if (this->dofNumberBw > 0) {
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
void reg_conjugateGradient<T>::Optimise(T maxLength,
                                        T smallLength,
                                        T &startLength) {
    this->UpdateGradientValues();
    reg_optimiser<T>::Optimise(maxLength,
                               smallLength,
                               startLength);
}
/* *************************************************************** */
template <class T>
void reg_conjugateGradient<T>::Perturbation(float length) {
    reg_optimiser<T>::Perturbation(length);
    this->firstCall = true;
}
/* *************************************************************** */
template <class T>
reg_lbfgs<T>::reg_lbfgs()
    :reg_optimiser<T>::reg_optimiser() {
    this->stepToKeep = 5;
    this->oldDof = nullptr;
    this->oldGrad = nullptr;
    this->diffDof = nullptr;
    this->diffGrad = nullptr;
}
/* *************************************************************** */
template <class T>
reg_lbfgs<T>::~reg_lbfgs() {
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
void reg_lbfgs<T>::Initialise(size_t nvox,
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
    reg_optimiser<T>::Initialise(nvox, ndim, optX, optY, optZ, maxIt, startIt, intOpt, cppData, gradData, nvoxBw, cppDataBw, gradDataBw);
    this->stepToKeep = 5;
    this->diffDof = (T**)malloc(this->stepToKeep * sizeof(T*));
    this->diffGrad = (T**)malloc(this->stepToKeep * sizeof(T*));
    for (size_t i = 0; i < this->stepToKeep; ++i) {
        this->diffDof[i] = (T*)malloc(this->dofNumber * sizeof(T));
        this->diffGrad[i] = (T*)malloc(this->dofNumber * sizeof(T));
        if (this->diffDof[i] == nullptr || this->diffGrad[i] == nullptr) {
            reg_print_fct_error("reg_lbfgs<T>::Initialise");
            reg_print_msg_error("Out of memory");
            reg_exit();
        }
    }
    this->oldDof = (T*)malloc(this->dofNumber * sizeof(T));
    this->oldGrad = (T*)malloc(this->dofNumber * sizeof(T));
    if (this->oldDof == nullptr || this->oldGrad == nullptr) {
        reg_print_fct_error("reg_lbfgs<T>::Initialise");
        reg_print_msg_error("Out of memory");
        reg_exit();
    }
}
/* *************************************************************** */
template <class T>
void reg_lbfgs<T>::UpdateGradientValues() {

}
/* *************************************************************** */
template <class T>
void reg_lbfgs<T>::Optimise(T maxLength,
                            T smallLength,
                            T &startLength) {
    this->UpdateGradientValues();
    reg_optimiser<T>::Optimise(maxLength,
                               smallLength,
                               startLength);
}
/* *************************************************************** */
