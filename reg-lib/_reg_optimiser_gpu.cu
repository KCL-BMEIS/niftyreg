#ifndef _reg_optimiser_CU
#define _reg_optimiser_CU

#include "_reg_optimiser_gpu.h"
#include "_reg_optimiser_kernels.cu"

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
reg_optimiser_gpu::reg_optimiser_gpu()
{
    this->dofNumber=0;
    this->ndim=3;
    this->optimiseX=true;
    this->optimiseY=true;
    this->optimiseZ=true;
    this->currentDOF=NULL;
    this->bestDOF=NULL;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
reg_optimiser_gpu::~reg_optimiser_gpu()
{
    if(this->bestDOF!=NULL)
        cudaCommon_free<float4>(&this->bestDOF);;
    this->bestDOF=NULL;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_optimiser_gpu::Initialise(size_t nvox,
                                   int dim,
                                   bool optX,
                                   bool optY,
                                   bool optZ,
                                   size_t maxit,
                                   size_t start,
                                   InterfaceOptimiser *obj,
                                   float4 *cppData,
                                   float4 *gradData
                                   )
{
    this->dofNumber=nvox;
    this->ndim=dim;
    this->optimiseX=optX;
    this->optimiseY=optY;
    this->optimiseZ=optZ;
    this->maxIterationNumber=maxit;
    this->currentIterationNumber=start;

    this->currentDOF=cppData;

    if(gradient!=NULL)
        this->gradient=gradData;

    if(this->bestDOF!=NULL)
        cudaCommon_free<float4>(&this->bestDOF);

    if(cudaCommon_allocateArrayToDevice(&this->bestDOF,
                                        (int)(this->dofNumber/this->ndim))){
        printf("[NiftyReg ERROR] Error when allocating the best control point array on the GPU.\n");
        exit(1);
    }

    reg_optimiser_gpu::StoreCurrentDOF();

    this->objFunc=obj;
    this->bestObjFunctionValue = this->currentObjFunctionValue =
            this->objFunc->GetObjectiveFunctionValue();
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_optimiser_gpu::RestoreBestDOF()
{
    // restore forward transformation
    NR_CUDA_SAFE_CALL(
        cudaMemcpy(this->currentDOF,
                   this->bestDOF,
                   this->GetVoxNumber()*sizeof(float4),
                   cudaMemcpyDeviceToDevice))
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_optimiser_gpu::StoreCurrentDOF()
{
    // save forward transformation
    NR_CUDA_SAFE_CALL(
        cudaMemcpy(this->bestDOF,
                   this->currentDOF,
                   this->GetVoxNumber()*sizeof(float4),
                   cudaMemcpyDeviceToDevice))
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_optimiser_gpu::Perturbation(float length)
{
    /// @todo
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_optimiser_gpu::NormaliseGradient()
{
    // First compute the gradienfloat max length for normalisation purpose
    float maxGradValue=reg_getMaximalLength_gpu(&this->gradient,
                                                (int)(this->dofNumber / this->ndim));
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] Objective function gradient maximal length: %g\n",maxGradValue);
#endif

    reg_multiplyValue_gpu((int)(this->dofNumber / this->ndim),
                          &this->gradient,
                          1.f/maxGradValue);
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
reg_conjugateGradient_gpu::reg_conjugateGradient_gpu()
    :reg_optimiser_gpu::reg_optimiser_gpu()
{
    this->array1=NULL;
    this->array2=NULL;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
reg_conjugateGradient_gpu::~reg_conjugateGradient_gpu()
{
    if(this->array1!=NULL)
        cudaCommon_free<float4>(&this->array1);
    this->array1=NULL;

    if(this->array2!=NULL)
        cudaCommon_free<float4>(&this->array2);
    this->array2=NULL;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_conjugateGradient_gpu::Initialise(size_t nvox,
                                           int dim,
                                           bool optX,
                                           bool optY,
                                           bool optZ,
                                           size_t maxit,
                                           size_t start,
                                           InterfaceOptimiser *obj,
                                           float4 *cppData,
                                           float4 *gradData
                                           )
{
    reg_optimiser_gpu::Initialise(nvox,
                                  dim,
                                  optX,
                                  optY,
                                  optZ,
                                  maxit,
                                  start,
                                  obj,
                                  cppData,
                                  gradData
                                  );
    this->firstcall=true;
    reg_optimiser_gpu *super = reinterpret_cast<reg_optimiser_gpu *>(this);
    if(cudaCommon_allocateArrayToDevice(&this->array1,
                                        (int)(super->GetVoxNumber()))){
        printf("[NiftyReg ERROR] Error when allocating the first conjugate gradient array on the GPU.\n");
        exit(1);
    }
    if(cudaCommon_allocateArrayToDevice(&this->array2,
                                        (int)(this->GetVoxNumber()))){
        printf("[NiftyReg ERROR] Error when allocating the second conjugate gradient array on the GPU.\n");
        exit(1);
    }
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_conjugateGradient_gpu::UpdateGradientValues()
{
    if(this->firstcall==true){
        reg_initialiseConjugateGradient(&this->gradient,
                                        &this->array1,
                                        &this->array2,
                                        (int)(this->GetVoxNumber()));
        this->firstcall=false;
    }
    else{
        reg_GetConjugateGradient(&this->gradient,
                                 &this->array1,
                                 &this->array2,
                                 (int)(this->GetVoxNumber()));
    }
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_optimiser_gpu::Optimise(float maxLength,
                                 float smallLength,
                                 float &startLength)
{
    size_t lineIteration=0;
    float addedLength=0;
    float currentLength=startLength;

    this->NormaliseGradient();

    // Start performing the line search
    while(currentLength>smallLength &&
          lineIteration<12 &&
          this->currentIterationNumber<this->maxIterationNumber){

        // Compute the gradient normalisation value
        float normValue = -currentLength;

        this->objFunc->UpdateParameters(normValue);

        // Compute the new value
        this->currentObjFunctionValue=this->objFunc->GetObjectiveFunctionValue();

        // Check if the update lead to an improvement of the objective function
        if(this->currentObjFunctionValue > this->bestObjFunctionValue){
#ifndef NDEBUG
            printf("[NiftyReg DEBUG] [%i] objective function: %g | Increment %g | ACCEPTED\n",
                   (int)this->currentIterationNumber,
                   this->currentObjFunctionValue,
                   currentLength);
#endif
            // Improvement - Save the new objective function value
            this->bestObjFunctionValue=this->currentObjFunctionValue;
            // Update the total added length
            addedLength += currentLength;
            // Increase the step size
            currentLength *= 1.1f;
            currentLength = (currentLength<maxLength)?currentLength:maxLength;
            // Save the current deformation parametrisation
            this->StoreCurrentDOF();
        }
        else{
#ifndef NDEBUG
            printf("[NiftyReg DEBUG] [%i] objective function: %g | Increment %g | REJECTED\n",
                   (int)this->currentIterationNumber,
                   this->currentObjFunctionValue,
                   currentLength);
#endif
            // No improvement - Decrease the step size
            currentLength*=0.5;
        }
        this->IncrementCurrentIterationNumber();
        ++lineIteration;
    }
    // update the current size for the next iteration
    startLength=addedLength;
    // Restore the last best deformation parametrisation
    this->RestoreBestDOF();
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_conjugateGradient_gpu::Optimise(float maxLength,
                                         float smallLength,
                                         float &startLength)
{
    this->UpdateGradientValues();
    reg_optimiser_gpu::Optimise(maxLength,
                                smallLength,
                                startLength);
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_conjugateGradient_gpu::Perturbation(float length)
{
    reg_optimiser_gpu::Perturbation(length);
    this->firstcall=true;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_initialiseConjugateGradient(float4 **nodeNMIGradientArray_d,
                                     float4 **conjugateG_d,
                                     float4 **conjugateH_d,
                                     int nodeNumber)
{
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_NodeNumber,&nodeNumber,sizeof(int)))
            NR_CUDA_SAFE_CALL(cudaBindTexture(0, gradientImageTexture, *nodeNMIGradientArray_d, nodeNumber*sizeof(float4)))

            const unsigned int Grid_reg_initialiseConjugateGradient =
            (unsigned int)ceil(sqrtf((float)nodeNumber/(float)Block_reg_initialiseConjugateGradient));
    dim3 G1(Grid_reg_initialiseConjugateGradient,Grid_reg_initialiseConjugateGradient,1);
    dim3 B1(Block_reg_initialiseConjugateGradient,1,1);

    reg_initialiseConjugateGradient_kernel <<< G1, B1 >>> (*conjugateG_d);
    NR_CUDA_CHECK_KERNEL(G1,B1)
            NR_CUDA_SAFE_CALL(cudaUnbindTexture(gradientImageTexture))
            NR_CUDA_SAFE_CALL(cudaMemcpy(*conjugateH_d, *conjugateG_d, nodeNumber*sizeof(float4), cudaMemcpyDeviceToDevice))
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_GetConjugateGradient(float4 **nodeNMIGradientArray_d,
                              float4 **conjugateG_d,
                              float4 **conjugateH_d,
                              int nodeNumber)
{
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_NodeNumber,&nodeNumber,sizeof(int)))
            NR_CUDA_SAFE_CALL(cudaBindTexture(0, conjugateGTexture, *conjugateG_d, nodeNumber*sizeof(float4)))
            NR_CUDA_SAFE_CALL(cudaBindTexture(0, conjugateHTexture, *conjugateH_d, nodeNumber*sizeof(float4)))
            NR_CUDA_SAFE_CALL(cudaBindTexture(0, gradientImageTexture, *nodeNMIGradientArray_d, nodeNumber*sizeof(float4)))

            // gam = sum((grad+g)*grad)/sum(HxG);
            const unsigned int Grid_reg_GetConjugateGradient1 = (unsigned int)ceil(sqrtf((float)nodeNumber/(float)Block_reg_GetConjugateGradient1));
    dim3 B1(Block_reg_GetConjugateGradient1,1,1);
    dim3 G1(Grid_reg_GetConjugateGradient1,Grid_reg_GetConjugateGradient1,1);

    float2 *sum_d;
    NR_CUDA_SAFE_CALL(cudaMalloc(&sum_d, nodeNumber*sizeof(float2)))
            reg_GetConjugateGradient1_kernel <<< G1, B1 >>> (sum_d);
    NR_CUDA_CHECK_KERNEL(G1,B1)
            float2 *sum_h;NR_CUDA_SAFE_CALL(cudaMallocHost(&sum_h, nodeNumber*sizeof(float2)))
            NR_CUDA_SAFE_CALL(cudaMemcpy(sum_h,sum_d, nodeNumber*sizeof(float2),cudaMemcpyDeviceToHost))
            NR_CUDA_SAFE_CALL(cudaFree(sum_d))
            double dgg = 0.0;
    double gg = 0.0;
    for(int i=0; i<nodeNumber; i++){
        dgg += sum_h[i].x;
        gg += sum_h[i].y;
    }
    float gam = (float)(dgg / gg);
    NR_CUDA_SAFE_CALL(cudaFreeHost((void *)sum_h))

            NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ScalingFactor,&gam,sizeof(float)))
            const unsigned int Grid_reg_GetConjugateGradient2 = (unsigned int)ceil(sqrtf((float)nodeNumber/(float)Block_reg_GetConjugateGradient2));
    dim3 B2(Block_reg_GetConjugateGradient2,1,1);
    dim3 G2(Grid_reg_GetConjugateGradient2,Grid_reg_GetConjugateGradient2,1);
    reg_GetConjugateGradient2_kernel <<< G2, B2 >>> (*nodeNMIGradientArray_d, *conjugateG_d, *conjugateH_d);
    NR_CUDA_CHECK_KERNEL(G1,B1)

            NR_CUDA_SAFE_CALL(cudaUnbindTexture(conjugateGTexture))
            NR_CUDA_SAFE_CALL(cudaUnbindTexture(conjugateHTexture))
            NR_CUDA_SAFE_CALL(cudaUnbindTexture(gradientImageTexture))

}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
float reg_getMaximalLength_gpu(	float4 **nodeNMIGradientArray_d,
                               int nodeNumber)
{

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_NodeNumber,&nodeNumber,sizeof(int)))
            NR_CUDA_SAFE_CALL(cudaBindTexture(0, gradientImageTexture, *nodeNMIGradientArray_d, nodeNumber*sizeof(float4)))

            // each thread extract the maximal value out of 128
            const int threadNumber = (int)ceil((float)nodeNumber/128.0f);
    const unsigned int Grid_reg_getMaximalLength = (unsigned int)ceil(sqrtf((float)threadNumber/(float)Block_reg_getMaximalLength));
    dim3 B1(Block_reg_getMaximalLength,1,1);
    dim3 G1(Grid_reg_getMaximalLength,Grid_reg_getMaximalLength,1);

    float *all_d;
    NR_CUDA_SAFE_CALL(cudaMalloc(&all_d, threadNumber*sizeof(float)))
            reg_getMaximalLength_kernel <<< G1, B1 >>> (all_d);
    NR_CUDA_CHECK_KERNEL(G1,B1)

            float *all_h;NR_CUDA_SAFE_CALL(cudaMallocHost(&all_h, nodeNumber*sizeof(float)))
            NR_CUDA_SAFE_CALL(cudaMemcpy(all_h, all_d, threadNumber*sizeof(float),cudaMemcpyDeviceToHost))
            NR_CUDA_SAFE_CALL(cudaFree(all_d))
            double maxDistance = 0.0f;
    for(int i=0; i<threadNumber; i++) maxDistance = all_h[i]>maxDistance?all_h[i]:maxDistance;
    NR_CUDA_SAFE_CALL(cudaFreeHost((void *)all_h))

            NR_CUDA_SAFE_CALL(cudaUnbindTexture(gradientImageTexture))
            return (float)maxDistance;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_updateControlPointPosition_gpu(nifti_image *controlPointImage,
                                        float4 **controlPointImageArray_d,
                                        float4 **bestControlPointPosition_d,
                                        float4 **nodeNMIGradientArray_d,
                                        float currentLength)
{
    const int nodeNumber = controlPointImage->nx * controlPointImage->ny * controlPointImage->nz;
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_NodeNumber,&nodeNumber,sizeof(int)))
            NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ScalingFactor,&currentLength,sizeof(float)))

            NR_CUDA_SAFE_CALL(cudaBindTexture(0, controlPointTexture, *bestControlPointPosition_d, nodeNumber*sizeof(float4)))
            NR_CUDA_SAFE_CALL(cudaBindTexture(0, gradientImageTexture, *nodeNMIGradientArray_d, nodeNumber*sizeof(float4)))

            const unsigned int Grid_reg_updateControlPointPosition = (unsigned int)ceil(sqrtf((float)nodeNumber/(float)Block_reg_updateControlPointPosition));
    dim3 B1(Block_reg_updateControlPointPosition,1,1);
    dim3 G1(Grid_reg_updateControlPointPosition,Grid_reg_updateControlPointPosition,1);

    reg_updateControlPointPosition_kernel <<< G1, B1 >>> (*controlPointImageArray_d);
    NR_CUDA_CHECK_KERNEL(G1,B1)
            NR_CUDA_SAFE_CALL(cudaUnbindTexture(controlPointTexture))
            NR_CUDA_SAFE_CALL(cudaUnbindTexture(gradientImageTexture))
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

#endif // _reg_optimiser_CU
