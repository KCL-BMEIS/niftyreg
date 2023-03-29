/** @file _reg_optimiser.h
 * @author Marc Modat
 * @date 20/07/2012
 */

#pragma once

#include "_reg_maths.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

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
/** @class reg_optimiser
 * @brief Standard gradient ascent optimisation
 */
template <class T>
class reg_optimiser {
protected:
    bool isBackwards;
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

public:
    reg_optimiser();
    virtual ~reg_optimiser();
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
                            T *gradData = nullptr,
                            size_t nvoxBw = 0,
                            T *cppDataBw = nullptr,
                            T *gradDataBw = nullptr);
    virtual void Optimise(T maxLength,
                          T smallLength,
                          T &startLength);
    virtual void Perturbation(float length);
};
/* *************************************************************** */
/** @class reg_conjugateGradient
 * @brief Conjugate gradient ascent optimisation
 */
template <class T>
class reg_conjugateGradient: public reg_optimiser<T> {
protected:
    T *array1;
    T *array1Bw;
    T *array2;
    T *array2Bw;
    bool firstCall;

    void UpdateGradientValues(); /// @brief Update the gradient array

public:
    reg_conjugateGradient();
    virtual ~reg_conjugateGradient();
    virtual void Initialise(size_t nvox,
                            int ndim,
                            bool optX,
                            bool optY,
                            bool optZ,
                            size_t maxIt,
                            size_t startIt,
                            InterfaceOptimiser *intOpt,
                            T *cppData = nullptr,
                            T *gradData = nullptr,
                            size_t nvoxBw = 0,
                            T *cppDataBw = nullptr,
                            T *gradDataBw = nullptr) override;
    virtual void Optimise(T maxLength,
                          T smallLength,
                          T &startLength) override;
    virtual void Perturbation(float length) override;
};
/* *************************************************************** */
/** @class Global optimisation class
 * @brief
 */
template <class T>
class reg_lbfgs: public reg_optimiser<T> {
protected:
    size_t stepToKeep;
    T *oldDof;
    T *oldGrad;
    T **diffDof;
    T **diffGrad;

public:
    reg_lbfgs();
    virtual ~reg_lbfgs();
    virtual void Initialise(size_t nvox,
                            int ndim,
                            bool optX,
                            bool optY,
                            bool optZ,
                            size_t maxIt,
                            size_t startIt,
                            InterfaceOptimiser *intOpt,
                            T *cppData = nullptr,
                            T *gradData = nullptr,
                            size_t nvoxBw = 0,
                            T *cppDataBw = nullptr,
                            T *gradDataBw = nullptr) override;
    virtual void Optimise(T maxLength,
                          T smallLength,
                          T &startLength) override;
    virtual void UpdateGradientValues() override;
};
/* *************************************************************** */
#include "_reg_optimiser.cpp"
