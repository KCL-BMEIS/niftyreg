/** @file Optimiser.hpp
 * @author Marc Modat
 * @date 20/07/2012
 */

#pragma once

#include "_reg_tools.h"

/* *************************************************************** */
namespace NiftyReg {
/* *************************************************************** */
/** @brief Interface between the registration class and the optimiser
 */
class InterfaceOptimiser {
public:
    /// @brief Returns the registration current objective function value
    virtual double GetObjectiveFunctionValue() = 0;
    /// @brief The transformation parameters are optimised
    virtual void UpdateParameters(float) = 0;
    /// @brief The best objective function values are stored
    virtual void UpdateBestObjFunctionValue() = 0;
};
/* *************************************************************** */
/** @class Optimiser
 * @brief Standard gradient ascent optimisation
 */
template <class T>
class Optimiser {
protected:
    bool isSymmetric;
    size_t dofNumber;
    size_t dofNumberBw;
    size_t ndim;
    T *currentDof; // pointer to the cpp nifti image array
    T *currentDofBw; // pointer to the cpp nifti image array (backwards)
    T *bestDof;
    T *bestDofBw;
    T *gradient;
    T *gradientBw;
    bool optimiseX;
    bool optimiseY;
    bool optimiseZ;
    size_t maxIterationNumber;
    size_t currentIterationNumber;
    double bestObjFunctionValue;
    double currentObjFunctionValue;
    InterfaceOptimiser *intOpt;

#ifdef NR_TESTING
public:
#endif
    /// @brief Update the gradient array
    virtual void UpdateGradientValues() {}

public:
    Optimiser();
    virtual ~Optimiser();
    virtual void StoreCurrentDof();
    virtual void RestoreBestDof();
    virtual size_t GetDofNumber() {
        return this->dofNumber;
    }
    virtual size_t GetDofNumberBw() {
        return this->dofNumberBw;
    }
    virtual size_t GetNDim() {
        return this->ndim;
    }
    virtual size_t GetVoxNumber() {
        return this->dofNumber / this->ndim;
    }
    virtual size_t GetVoxNumberBw() {
        return this->dofNumberBw / this->ndim;
    }
    virtual T* GetBestDof() {
        return this->bestDof;
    }
    virtual T* GetBestDofBw() {
        return this->bestDofBw;
    }
    virtual T* GetCurrentDof() {
        return this->currentDof;
    }
    virtual T* GetCurrentDofBw() {
        return this->currentDofBw;
    }
    virtual T* GetGradient() {
        return this->gradient;
    }
    virtual T* GetGradientBw() {
        return this->gradientBw;
    }
    virtual bool GetOptimiseX() {
        return this->optimiseX;
    }
    virtual bool GetOptimiseY() {
        return this->optimiseY;
    }
    virtual bool GetOptimiseZ() {
        return this->optimiseZ;
    }
    virtual size_t GetMaxIterationNumber() {
        return this->maxIterationNumber;
    }
    virtual size_t GetCurrentIterationNumber() {
        return this->currentIterationNumber;
    }
    virtual size_t ResetCurrentIterationNumber() {
        return this->currentIterationNumber = 0;
    }
    virtual double GetBestObjFunctionValue() {
        return this->bestObjFunctionValue;
    }
    virtual void SetBestObjFunctionValue(double i) {
        this->bestObjFunctionValue = i;
    }
    virtual double GetCurrentObjFunctionValue() {
        return this->currentObjFunctionValue;
    }
    virtual void IncrementCurrentIterationNumber() {
        this->currentIterationNumber++;
    }
    virtual void Initialise(size_t nvox,
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
                            T *gradDataBw);
    virtual void Optimise(T maxLength,
                          T smallLength,
                          T& startLength);
    virtual void Perturbation(float length);
};
/* *************************************************************** */
/** @class ConjugateGradient
 * @brief Conjugate gradient ascent optimisation
 */
template <class T>
class ConjugateGradient: public Optimiser<T> {
protected:
    T *array1;
    T *array1Bw;
    T *array2;
    T *array2Bw;
    bool firstCall;

#ifdef NR_TESTING
public:
#endif
    virtual void UpdateGradientValues() override;

public:
    ConjugateGradient();
    virtual ~ConjugateGradient();
    virtual void Initialise(size_t nvox,
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
                            T *gradDataBw) override;
    virtual void Optimise(T maxLength,
                          T smallLength,
                          T& startLength) override;
    virtual void Perturbation(float length) override;
};
/* *************************************************************** */
/** @class Global optimisation class
 * @brief
 */
template <class T>
class Lbfgs: public Optimiser<T> {
protected:
    size_t stepToKeep;
    T *oldDof;
    T *oldGrad;
    T **diffDof;
    T **diffGrad;

#ifdef NR_TESTING
public:
#endif
    virtual void UpdateGradientValues() override;

public:
    Lbfgs();
    virtual ~Lbfgs();
    virtual void Initialise(size_t nvox,
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
                            T *gradDataBw) override;
    virtual void Optimise(T maxLength,
                          T smallLength,
                          T& startLength) override;
};
/* *************************************************************** */
} // namespace NiftyReg
/* *************************************************************** */
