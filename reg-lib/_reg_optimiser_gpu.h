#ifndef _REG_OPTIMISER_GPU_H
#define _REG_OPTIMISER_GPU_H

#include "_reg_optimiser.h"
#include "_reg_blocksize_gpu.h"
#include "_reg_cudaCommon.h"
#include "_reg_tools_gpu.h"

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/** @class Global optimisation class for GPU
 * @brief Standard gradient acent optimisation
 */
class reg_optimiser_gpu
{
protected:
  bool backward;
  size_t dofNumber;
  size_t ndim;
  float4 *currentDOF; // pointer to the cpp nifti image
  float4 *bestDOF;
  float4 *gradient;
  bool optimiseX;
  bool optimiseY;
  bool optimiseZ;
  size_t maxIterationNumber;
  size_t currentIterationNumber;
  double bestObjFunctionValue;
  double currentObjFunctionValue;
  InterfaceOptimiser *objFunc;

public:
  reg_optimiser_gpu();
  ~reg_optimiser_gpu();
  virtual void StoreCurrentDOF();
  virtual void RestoreBestDOF();
  virtual size_t GetDOFNumber(){return this->dofNumber;}
  virtual size_t GetNDim(){return this->ndim;}
  virtual size_t GetVoxNumber(){return this->dofNumber/this->ndim;}
  virtual float4* GetBestDOF(){return this->bestDOF;}
  virtual float4* GetCurrentDOF(){return this->currentDOF;}
  virtual float4* GetGradient(){return this->gradient;}
  virtual bool GetOptimiseX(){return this->optimiseX;}
  virtual bool GetOptimiseY(){return this->optimiseY;}
  virtual bool GetOptimiseZ(){return this->optimiseZ;}
  virtual size_t GetMaxIterationNumber(){return this->maxIterationNumber;}
  virtual size_t GetCurrentIterationNumber(){return this->currentIterationNumber;}
  virtual double GetBestObjFunctionValue(){return this->bestObjFunctionValue;}
  virtual double GetCurrentObjFunctionValue(){return this->currentObjFunctionValue;}
  virtual void IncrementCurrentIterationNumber(){this->currentIterationNumber++;}

  virtual void Initialise(size_t nvox,
                          int dim,
                          bool optX,
                          bool optY,
                          bool optZ,
                          size_t maxit,
                          size_t start,
                          InterfaceOptimiser *o,
                          float4 *cppData,
                          float4 *gradData);
  virtual void Optimise(float maxLength,
                        float smallLength,
                        float &startLength);
  virtual void Perturbation(float length);
  virtual void NormaliseGradient();
};
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/** @class Conjugate gradient optimisation class for GPU
 * @brief
 */
class reg_conjugateGradient_gpu : public reg_optimiser_gpu
{
  protected:
    float4 *array1;
    float4 *array2;
    bool firstcall;
    void UpdateGradientValues(); /// @brief Update the gradient array

  public:
    reg_conjugateGradient_gpu();
    ~reg_conjugateGradient_gpu();

    virtual void Initialise(size_t nvox,
                            int dim,
                            bool optX,
                            bool optY,
                            bool optZ,
                            size_t maxit,
                            size_t start,
                            InterfaceOptimiser *o,
                            float4 *cppData,
                            float4 *gradData);
    virtual void Optimise(float maxLength,
                          float smallLength,
                          float &startLength);
    virtual void Perturbation(float length);
};
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/** @brief
 */
extern "C++"
void reg_initialiseConjugateGradient(float4 **nodeNMIGradientArray_d,
                                     float4 **conjugateG_d,
                                     float4 **conjugateH_d,
                                     int nodeNumber);

/** @brief
 */
extern "C++"
void reg_GetConjugateGradient(float4 **nodeNMIGradientArray_d,
                              float4 **conjugateG_d,
                              float4 **conjugateH_d,
                              int nodeNumber);

/** @brief
 */
extern "C++"
float reg_getMaximalLength_gpu(float4 **nodeNMIGradientArray_d,
                               int nodeNumber);

/** @brief
 */
extern "C++"
void reg_updateControlPointPosition_gpu(nifti_image *controlPointImage,
                                        float4 **controlPointImageArray_d,
                                        float4 **bestControlPointPosition_d,
                                        float4 **nodeNMIGradientArray_d,
                                        float currentLength);

#endif // _REG_OPTIMISER_GPU_H
