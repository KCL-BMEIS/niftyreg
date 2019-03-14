// Author:  Lucas Fidon
// Class for F3D registration using the Ipopt optimisation solver
//

#ifndef NIFTYREG_REG_F3D_IPOPT_H
#define NIFTYREG_REG_F3D_IPOPT_H


#include "_reg_f3d.h"
#include "IpTNLP.hpp"
#include "IpIpoptCalculatedQuantities.hpp"
#include "IpIpoptData.hpp"
#include "IpTNLPAdapter.hpp"
#include "IpOrigIpoptNLP.hpp"
//#include "exception.h"
#include <cassert>
#include <cstdio>
#include <limits>

using namespace Ipopt;


// This inherits from NiftyReg reg_f3d class and extends its API
// so as to perform the optimisation using Ipopt library
template <class T>
class reg_f3d_ipopt : public reg_f3d<T>, public TNLP
{
public:
    /** constructor that takes in problem data */
    reg_f3d_ipopt(int refTimePoint, int floTimePoint);

    /** default destructor */
    virtual ~reg_f3d_ipopt();

    void initLevel(int level);

    void clearLevel(int level);

    void setNewControlPointGrid(const Number *x, Index n);

    void setScale(float scale);

    void printConfigInfo();

    void gradientCheck();

    void voxelBasedGradientCheck();

    void printImgStat();

    /**@name Overloaded from TNLP */
    //@{
    /** Method to return some info about the nlp */
    virtual bool get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
                              Index& nnz_h_lag, IndexStyleEnum& index_style);

    /** Method to return the bounds for my problem */
    virtual bool get_bounds_info(Index n, Number* x_l, Number* x_u,
                                 Index m, Number* g_l, Number* g_u);

    /** Method to return the starting point for the algorithm */
    virtual bool get_starting_point(Index n, bool init_x, Number* x,
                                    bool init_z, Number* z_L, Number* z_U,
                                    Index m, bool init_lambda,
                                    Number* lambda);

    /** Method to return the objective value */
    virtual bool eval_f(Index n, const Number* x, bool new_x, Number& obj_value);

    /** Method to return the gradient of the objective */
    virtual bool eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f);

    /** Method to return the constraint residuals */
    virtual bool eval_g(Index n, const Number* x, bool new_x, Index m, Number* g);

    /** Method to return:
     *   1) The structure of the jacobian (if "values" is NULL)
     *   2) The values of the jacobian (if "values" is not NULL)
     */
    virtual bool eval_jac_g(Index n, const Number* x, bool new_x,
                            Index m, Index nele_jac, Index* iRow, Index *jCol,
                            Number* values);

    /** Method to return:
     *   1) The structure of the hessian of the lagrangian (if "values" is NULL)
     *   2) The values of the hessian of the lagrangian (if "values" is not NULL)
     */
    virtual bool eval_h(Index n, const Number* x, bool new_x,
                        Number obj_factor, Index m, const Number* lambda,
                        bool new_lambda, Index nele_hess, Index* iRow,
                        Index* jCol, Number* values);

    /** (Optional) Method to obtain information about the optimisation at each iteration.
     *    This is used to track the values of the primal variables hat gave the best objective function.
     */
    virtual bool intermediate_callback(AlgorithmMode mode,
                                       Index iter, Number obj_value,
                                       Number inf_pr, Number inf_du,
                                       Number mu, Number d_norm,
                                       Number regularization_size,
                                       Number alpha_du, Number alpha_pr,
                                       Index ls_trials,
                                       const IpoptData* ip_data,
                                       IpoptCalculatedQuantities* ip_cq);
    //@}

    /** @name Solution Methods */
    //@{
    /** This method is called when the algorithm is complete so the TNLP can store/write the solution */
    virtual void finalize_solution(SolverReturn status,
                                   Index n, const Number* x, const Number* z_L, const Number* z_U,
                                   Index m, const Number* g, const Number* lambda,
                                   Number obj_value,
                                   const IpoptData* ip_data,
                                   IpoptCalculatedQuantities* ip_cq);
    //@}

protected:
    double scalingCst;  // scaling factor for the loss function (make sure that typical gradient values are of order 0.1-10)
    T bestObj;  // best objective function value
    nifti_image *bestControlPointGrid;  // controlPointGrid values corresponding to the best objective function

private:
    /**@name Methods to block default compiler methods.
     * The compiler automatically generates the following three methods.
     *  Since the default compiler implementation is generally not what
     *  you want (for all but the most simple classes), we usually
     *  put the declarations of these methods in the private section
     *  and never implement them. This prevents the compiler from
     *  implementing an incorrect "default" behavior without us
     *  knowing. (See Scott Meyers book, "Effective C++")
     *
     */
    //@{
    // Reg_TNLP();
    // Reg_TNLP(const Reg_TNLP&);
    // Reg_TNLP& operator=(const Reg_TNLP&);
    //@}

    /** @name reg_f3d_ipopt data */
    //@{
    //@}
};

template <class T>
reg_f3d_ipopt<T>::reg_f3d_ipopt(int refTimePoint, int floTimePoint)
        : reg_f3d<T>::reg_f3d(refTimePoint, floTimePoint) {
    this->scalingCst = 1.;
}

template <class T>
reg_f3d_ipopt<T>::~reg_f3d_ipopt(){
#ifndef NDEBUG
    reg_print_msg_debug("reg_f3d_ipopt destructor called");
#endif
}

template <class T>
void reg_f3d_ipopt<T>::initLevel(int level){
    if(!this->initialised) {
        this->Initialise();
    }
    // forward and backward velocity grids are initialised to identity deformation field
    // see /cpu/_reg_localTrans.cpp l.410 (reg_getDeformationFromDisplacement is used)

    // Initialise best objective value avd corresponding best control point grid
    this->bestObj = std::numeric_limits<T>::max();
    this->bestControlPointGrid = nifti_copy_nim_info(this->controlPointGrid);
    this->bestControlPointGrid->data = (void *)calloc(this->bestControlPointGrid->nvox,
                                                      this->bestControlPointGrid->nbyper);

    this->currentLevel = (unsigned int) level;

    // Set the current input images
    this->currentReference = this->referencePyramid[level];
    this->currentFloating = this->floatingPyramid[level];
    this->currentMask = this->maskPyramid[level];

    // Allocate image that depends on the reference image
    this->AllocateWarped();
    this->AllocateDeformationField();
    this->AllocateWarpedGradient();

    // The grid is refined if necessary and the floatingMask is set
    T maxStepSize=this->InitialiseCurrentLevel();  // This has been put in updateInputControlPoint
//  T currentSize = maxStepSize;
//  T smallestSize = maxStepSize / (T)100.0;

    this->DisplayCurrentLevelParameters();

    // Allocate image that are required to compute the gradient
    this->AllocateVoxelBasedMeasureGradient();
    this->AllocateTransformationGradient();

    // Initialise the measures of similarity
    this->InitialiseSimilarity();

    // initialise the optimiser
    this->SetOptimiser();

    // Evaluate the initial objective function value and print it
    this->UpdateBestObjFunctionValue();
    this->PrintInitialObjFunctionValue();
}

template <class T>
void reg_f3d_ipopt<T>::clearLevel(int level) {
    delete this->optimiser;
    this->optimiser = NULL;
    this->ClearWarped();
    this->ClearDeformationField();
    this->ClearWarpedGradient();
    this->ClearVoxelBasedMeasureGradient();
    this->ClearTransformationGradient();
    // level specific variables are cleaned
    nifti_image_free(this->referencePyramid[level]);
    this->referencePyramid[level] = NULL;
    nifti_image_free(this->floatingPyramid[level]);
    this->floatingPyramid[level] = NULL;
    free(this->maskPyramid[level]);
    this->maskPyramid[level] = NULL;
    this->ClearCurrentInputImage();
}

template <class T>
void reg_f3d_ipopt<T>::setScale(float scale) {
    this->scalingCst = (T) scale;
}

template <class T>
void reg_f3d_ipopt<T>::printConfigInfo() {
    std::cout << std::endl;
    std::cout << "#################" << std::endl;
    std::cout << "REG_F3D Config Info" << std::endl;
    std::cout << "Scaling factor used for the loss function = " << std::scientific << this->scalingCst << std::endl;
    std::cout << "#################" << std::endl;
    std::cout << std::endl;
}

template <class T>
void reg_f3d_ipopt<T>::setNewControlPointGrid(const Number *x, Index n) {
    assert(n == this->controlPointGrid->nvox);
    T *controlPointGridPtr = static_cast<T *>(this->controlPointGrid->data);
    for (int i=0; i<n; i++) {
        controlPointGridPtr[i] = (T) x[i];
    }
    // as x is a displacement/velocity grid, but NiftyReg works with deformation grid
    // it is necessary to convert controlPointGrid to a deformation
    reg_getDeformationFromDisplacement(this->controlPointGrid);  // add identity deformation
};

template <class T>
// returns the size of the problem.
// IPOPT uses this information when allocating the arrays
// that it will later ask you to fill with values.
bool reg_f3d_ipopt<T>::get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
                                     Index& nnz_h_lag, IndexStyleEnum& index_style) {
//  std::cout << "Call get_nlp_info" << std::endl;
    // number of variables (forward + (optional) backward displacement grid)
    n = (int)(this->controlPointGrid->nvox);
    // number of constraints (null divergence of the velocity vector field)
    m = 0; // no constraint
    // number of non-zeros values in the jacobian of the constraint function
    nnz_jac_g = n * m; //full matrix
    // number of non-zeros values in the hessian
    nnz_h_lag = n * n; // full matrix
    // use the C style indexing (0-based) for the matrices
    index_style = TNLP::C_STYLE;
    return true;
}

template <class T>
// returns the variable bounds
bool reg_f3d_ipopt<T>::get_bounds_info(Index n, // number of variables (dim of x)
                                        Number* x_l, // lower bounds for x
                                        Number* x_u, // upperbound for x
                                        Index m, // number of constraints (dim of g(x))
                                        Number* g_l, // lower bounds for g(x)
                                        Number* g_u) { // upper bounds for g(x)
//  std::cout <<"Call get_bound_info" << std::endl;
    // lower and upper bounds for the primal variables
    for (Index i=0; i<n; i++) {
//    x_l[i] = -1e20;  // -infty
        x_l[i] = -1e2;  // in mm
//    x_u[i] = 1e20;  // +infty
        x_u[i] = 1e2;  // in mm
    }
    // lower and upper bounds for the inequality constraints
    for (Index i=0; i<m; i++) {
        g_l[i] = 0.;
        g_u[i] = 0.;
    }
    return true;
}

template <class T>
// returns the initial point for the problem
bool reg_f3d_ipopt<T>::get_starting_point(Index n, bool init_x, Number* x,
                                           bool init_z, Number* z_L, Number* z_U,
                                           Index m, bool init_lambda,
                                           Number* lambda) {
//  std::cout <<"Call get_starting_point" << std::endl;
    // Here, we assume we only have starting values for x, if you code
    // your own NLP, you can provide starting values for the dual variables if you wish
    assert(init_x == true);
    assert(init_z == false);
    assert(init_lambda == false);

    // make sure there is as many elements in x and the control point grid
    assert(n == this->controlPointGrid->nvox);

    // x is a displacement grid
    // so the initial deformations are converted into displacements
//    reg_getDisplacementFromDeformation(this->controlPointGrid);

    // initialize the displacement field associated to the current control point grid
    T *controlPointPtr = static_cast<T *>(this->controlPointGrid->data);
    for (Index i = 0; i < n; i++) {
        x[i] = 0;  // initialise to identity transformation
//        x[i] = static_cast<Number>(controlPointPtr[i]);
    }
    // set the controlPointGrid as a deformation grid again
//    reg_getDeformationFromDisplacement(this->controlPointGrid);

#ifdef skip_me
    /* If checking derivatives, if is useful to choose different values */
#endif

    return true;
}

template <class T>
// returns the value of the objective function
bool reg_f3d_ipopt<T>::eval_f(Index n, const Number* x,
                               bool new_x, Number& obj_value){
#ifndef NDEBUG
    std::cout <<"Call eval_f" << std::endl;
#endif

    // make sure there is as many elements in x and the control point grid
    assert(n == this->controlPointGrid->nvox);

    // set the current velocity vector field to x
    this->setNewControlPointGrid(x, n);

    // take the opposite of objective value to maximise because ipopt performs minimisation
    obj_value = -this->scalingCst * this->GetObjectiveFunctionValue();
#ifndef NDEBUG
    std::cout << "COST FONCTION = " << obj_value << std::endl;
#endif

    return true;
}

template <class T>
// return the gradient of the objective function grad_{x} f(x)
// it sets all values of gradient in grad_f
bool reg_f3d_ipopt<T>::eval_grad_f(Index n, const Number* x,
                                    bool new_x, Number* grad_f){
#ifndef NDEBUG
    std::cout << "Call eval_grad_f" << std::endl;

    printf("Grid: %d dimensions (%dx%dx%dx%dx%dx%dx%d)\n", this->controlPointGrid->ndim,
           this->controlPointGrid->nx, this->controlPointGrid->ny, this->controlPointGrid->nz,
           this->controlPointGrid->nt, this->controlPointGrid->nu, this->controlPointGrid->nv,
           this->controlPointGrid->nw);
#endif
    // make sure there is as many elements the displacement grid and in x
    assert(n == this->controlPointGrid->nvox);
    // set the current velocity field to x
    this->setNewControlPointGrid(x, n);

    // compute the objective function gradient value
//    this->GetObjectiveFunctionValue();  // this is just to make sure it is up to date
    this->GetObjectiveFunctionGradient();
//  this->NormaliseGradient();
    // update the Ipopt gradient
    T *gradient = static_cast<T *>(this->transformationGradient->data);

    for (int i = 0; i < n; i++) {
#ifndef NDEBUG
        if (gradient[i] != gradient[i]) {
            std::cout << "Nan value found at voxel " << i << std::endl;
//            throw NaNValueInGradientException();
        }
#endif
        grad_f[i] = (Number) (this->scalingCst*gradient[i]);
//        std::cout << "grad_f[" << i << "] = " << grad_f[i] << std::endl;
    }

    return true;
}

template <class T>
// return the value of the constraints: g(x)
// Divergence fo the velocity vector field
bool reg_f3d_ipopt<T>::eval_g(Index n, const Number* x,
                               bool new_x, Index m, Number* g)
{
    std::cout << "Call eval_g" << std::endl;
    // HERE COMPUTE VALUE OF ALL CONSTRAINTS IN g
//  for (Index j=0; j<m; j++){
//   g[j] = 0.;
//  }
//  g[0] = x[0];
    return true;
}

template <class T>
// return the structure or values of the jacobian
bool reg_f3d_ipopt<T>::eval_jac_g(Index n, const Number* x, bool new_x,
                                   Index m, Index nele_jac, Index* iRow,
                                   Index *jCol, Number* values){
    std::cout <<"Call eval_jac_g" << std::endl;
    if (values == NULL) {
        // return the structure of the jacobian
//    iRow[0] = 0;
//    jCol[0] = 0;
    }
    else {
        // return the values of the jacobian of the constraints
//    values[0] = 1;
    }

    return true;
}

template <class T>
//return the structure or values of the hessian
bool reg_f3d_ipopt<T>::eval_h(Index n, const Number* x, bool new_x,
                               Number obj_factor, Index m, const Number* lambda,
                               bool new_lambda, Index nele_hess, Index* iRow,
                               Index* jCol, Number* values){
    std::cout << "Call eval_h" << std::endl;
    if (values == NULL) {
        // return the structure (lower or upper triangular part) of the
        // Hessian of the Lagrangian function
    }
    else {
        // return the values of the Hessian of hte Lagrangian function
        // HERE FILL values
    }

    return true;
}

template <class T>
bool reg_f3d_ipopt<T>::intermediate_callback(AlgorithmMode mode,
                                             Index iter, Number obj_value,
                                             Number inf_pr, Number inf_du,
                                             Number mu, Number d_norm,
                                             Number regularization_size,
                                             Number alpha_du, Number alpha_pr,
                                             Index ls_trials,
                                             const IpoptData* ip_data,
                                             IpoptCalculatedQuantities* ip_cq) {
    // update best objective function and corresponding control point grid
    // if an improvement has been observed during this iteration
    if (obj_value < this->bestObj) {
        // get access to the primal variables
        TNLPAdapter *tnlp_adapter = NULL;
        if (ip_cq != NULL) {
            OrigIpoptNLP *orignlp;
            orignlp = dynamic_cast<OrigIpoptNLP *>(GetRawPtr(ip_cq->GetIpoptNLP()));
            if (orignlp != NULL)
                tnlp_adapter = dynamic_cast<TNLPAdapter *>(GetRawPtr(orignlp->nlp()));
        }
        T *primals = new T[this->bestControlPointGrid->nvox];
        tnlp_adapter->ResortX(*ip_data->curr()->x(), primals);
        // update best control point grid
        T *bestControlPointGridPtr = static_cast<T *>(this->bestControlPointGrid->data);
        for (int i=0; i<this->bestControlPointGrid->nvox; i++) {
            bestControlPointGridPtr[i] = (T) primals[i];
        }
        // as x is a displacement/velocity grid, but NiftyReg works with deformation grid
        // it is necessary to convert controlPointGrid to a deformation
        reg_getDeformationFromDisplacement(this->bestControlPointGrid);  // add identity deformation
        // update best objective function value
        this->bestObj = (T)obj_value;
    }
    return true;
}

template <class T>
void reg_f3d_ipopt<T>::finalize_solution(SolverReturn status,
                                          Index n, const Number* x,
                                          const Number* z_L, const Number* z_U,
                                          Index m, const Number* g,
                                          const Number* lambda,
                                          Number obj_value,
                                          const IpoptData* ip_data,
                                          IpoptCalculatedQuantities* ip_cq) {
    // update the current velocity vector field to the best found during optimisation
//    this->setNewControlPointGrid(x, n);
    memcpy(this->controlPointGrid->data, this->bestControlPointGrid,
           this->controlPointGrid->nvox * this->controlPointGrid->nbyper);

//  this->GetObjectiveFunctionValue();  // make sure all the variables are up-to-date

    // Save the control point image
    nifti_image *outputControlPointGrid = this->GetControlPointPositionImage();
//    nifti_image *outputControlPointGrid = nifti_copy_nim_info(this->bestControlPointGrid);
//    outputControlPointGrid->data=(void *)malloc(outputControlPointGrid->nvox*outputControlPointGrid->nbyper);
//    memcpy(outputControlPointGrid->data, this->bestControlPointGrid->data,
//           outputControlPointGrid->nvox*outputControlPointGrid->nbyper);
    std::string outputCPPImageName=stringFormat("outputCPP_level%d.nii", this->currentLevel+1);
    memset(outputControlPointGrid->descrip, 0, 80);
    strcpy (outputControlPointGrid->descrip,"Deformation field grid from NiftyReg");
    reg_io_WriteImageFile(outputControlPointGrid, outputCPPImageName.c_str());
    nifti_image_free(outputControlPointGrid);

    // Save the warped image(s)
    // allocate memory for two images for the symmetric case
    nifti_image **outputWarpedImage=(nifti_image **)malloc(2*sizeof(nifti_image *));
    outputWarpedImage[0] = NULL;
    outputWarpedImage[1] = NULL;
    outputWarpedImage = this->GetWarpedImage();
    std::string fileName = stringFormat("out_level%d.nii", this->currentLevel+1);
    memset(outputWarpedImage[0]->descrip, 0, 80);
    strcpy (outputWarpedImage[0]->descrip, "Warped image using NiftyReg");
    reg_io_WriteImageFile(outputWarpedImage[0], fileName.c_str());
    if (outputWarpedImage[1] != NULL) {
        fileName = stringFormat("out_backward_level%d.nii", this->currentLevel+1);
        memset(outputWarpedImage[1]->descrip, 0, 80);
        strcpy (outputWarpedImage[1]->descrip, "Warped backward image using NiftyReg");
        reg_io_WriteImageFile(outputWarpedImage[1], fileName.c_str());
    }
    // Compute and save absolute error map
    reg_tools_substractImageToImage(outputWarpedImage[0],
                                    this->currentReference, outputWarpedImage[0]);
    reg_tools_abs_image(outputWarpedImage[0]);
    fileName = stringFormat("abs_error_level%d.nii", this->currentLevel+1);
    reg_io_WriteImageFile(outputWarpedImage[0], fileName.c_str());
    // free allocated memory
    if(outputWarpedImage[0]!=NULL)
        nifti_image_free(outputWarpedImage[0]);
    outputWarpedImage[0]=NULL;
    if(outputWarpedImage[1]!=NULL)
        nifti_image_free(outputWarpedImage[1]);
    outputWarpedImage[1]=NULL;
    free(outputWarpedImage);

    // Compute and save the jacobian map
    size_t nvox = (size_t) this->currentReference->nx * this->currentReference->ny * this->currentReference->nz;
    nifti_image *jacobianDeterminantArray = nifti_copy_nim_info(this->currentReference);
    jacobianDeterminantArray->nbyper = this->controlPointGrid->nbyper;
    jacobianDeterminantArray->datatype = this->controlPointGrid->datatype;
    jacobianDeterminantArray->data = malloc(nvox*this->controlPointGrid->nbyper);
    jacobianDeterminantArray->cal_min=0;
    jacobianDeterminantArray->cal_max=0;
    jacobianDeterminantArray->scl_slope = 1.0f;
    jacobianDeterminantArray->scl_inter = 0.0f;
    // Jacobian map for cubic spline
    reg_spline_GetJacobianMap(this->controlPointGrid, jacobianDeterminantArray);
    fileName = stringFormat("jacobian_map_level%d.nii", this->currentLevel+1);
    reg_io_WriteImageFile(jacobianDeterminantArray, fileName.c_str());
    nifti_image_free(jacobianDeterminantArray);

//  std::cout << std::endl << "Writing solution file solution.txt" << std::endl;
    FILE* fp = fopen("solution.txt", "w");

    fprintf(fp, "\n\nObjective value\n");
    fprintf(fp, "f(x*) = %e\n", obj_value);
    fclose(fp);
}

template <class T>
void reg_f3d_ipopt<T>::gradientCheck() {
    T eps = 1e-4;
    std::cout.precision(17);
    // set control point at which we will compute the gradient
    // set the transformation to identity
    reg_tools_multiplyValueToImage(this->controlPointGrid, this->controlPointGrid, 0.f);
    reg_getDeformationFromDisplacement(this->controlPointGrid);
    // compute the gradient in this->transformationGradient
    this->GetObjectiveFunctionGradient();
    // save the gradient
    reg_io_WriteImageFile(this->transformationGradient, "full_grad_ipopt.nii");
    // allocate a copy of the param / control point grid
//    nifti_image *paramCopy = nifti_copy_nim_info(this->controlPointGrid);
//    paramCopy->data = (void *)calloc(paramCopy->nvox, paramCopy->nbyper);

    // compute approximation of the gradient by finite difference
    nifti_image *approxGrad = nifti_copy_nim_info(this->transformationGradient);
    approxGrad->data = (void *)calloc(approxGrad->nvox, approxGrad->nbyper);
    nifti_image *errorGrad = nifti_copy_nim_info(this->transformationGradient);
    errorGrad->data = (void *)calloc(errorGrad->nvox, errorGrad->nbyper);
    T* gradPtr = static_cast<T*>(this->transformationGradient->data);
    T* approxGradPtr = static_cast<T*>(approxGrad->data);
    T* errorGradPtr = static_cast<T*>(errorGrad->data);
    T* paramPtr = static_cast<T*>(this->controlPointGrid->data);
    T pre = 0;
    T post = 0;
    T currentParamVal = 0;
    assert(approxGrad->nvox == this->transformationGradient->nvox);
    for (int i=0; i<approxGrad->nvox; ++i) {
        currentParamVal = paramPtr[i];
        paramPtr[i] = currentParamVal - eps;
        pre = -this->GetObjectiveFunctionValue();
        paramPtr[i] = currentParamVal + eps;
        post = -this->GetObjectiveFunctionValue();
        paramPtr[i] = currentParamVal;
        approxGradPtr[i] = (post - pre) / (2.*eps);
        T error = fabs(approxGradPtr[i] - gradPtr[i]);
        errorGradPtr[i] = error;
//        if (approxGradPtr[i] != 0) {
//            errorGradPtr[i] = error / fabs(approxGradPtr[i]);
//        }
//        else {
//            errorGradPtr[i] = error;
//        }
        std::cout << "grad[" << i << "] = " << std::fixed << gradPtr[i]
            << "   ~   "
            << std::fixed << approxGradPtr[i]
            << "  [" << std::fixed << error << "]" << std::endl;
    }
    // save approximated gradient
    reg_io_WriteImageFile(approxGrad, "full_finite_diff_grad_ipopt.nii");
    reg_io_WriteImageFile(errorGrad, "absolute_error_grad.nii");
}

template <class T>
void reg_f3d_ipopt<T>::voxelBasedGradientCheck() {
    T eps = 1e-4;
    std::cout.precision(17);
    // set control point at which we will compute the gradient
    // set the transformation to identity
    reg_tools_multiplyValueToImage(this->controlPointGrid, this->controlPointGrid, 0.f);
    reg_getDeformationFromDisplacement(this->controlPointGrid);
    this->GetDeformationField();
    // compute the gradient in this->transformationGradient and this->voxelBasedMeasureGradient
    this->GetObjectiveFunctionGradient();
    // save the gradient
    reg_io_WriteImageFile(this->voxelBasedMeasureGradient, "voxel_based_grad_ipopt.nii");
    // allocate a copy of the param / control point grid
//    nifti_image *paramCopy = nifti_copy_nim_info(this->controlPointGrid);
//    paramCopy->data = (void *)calloc(paramCopy->nvox, paramCopy->nbyper);

    // compute approximation of the gradient by finite difference
    nifti_image *approxGrad = nifti_copy_nim_info(this->voxelBasedMeasureGradient);
    approxGrad->data = (void *)calloc(approxGrad->nvox, approxGrad->nbyper);
    nifti_image *errorGrad = nifti_copy_nim_info(this->voxelBasedMeasureGradient);
    errorGrad->data = (void *)calloc(errorGrad->nvox, errorGrad->nbyper);
    T* gradPtr = static_cast<T*>(this->voxelBasedMeasureGradient->data);
    T* approxGradPtr = static_cast<T*>(approxGrad->data);
    T* errorGradPtr = static_cast<T*>(errorGrad->data);
    T* paramPtr = static_cast<T*>(this->deformationFieldImage->data);
    T pre = 0;
    T post = 0;
    T currentParamVal = 0;
    assert(approxGrad->nvox == this->voxelBasedMeasureGradient->nvox);
    for (int i=0; i<approxGrad->nvox; ++i) {
        currentParamVal = paramPtr[i];
        paramPtr[i] = currentParamVal - eps;
        // Compute gradient wrt deformation field (and not wrt control points)
        // warp image
        reg_resampleImage(this->currentFloating,
                          this->warped,
                          this->deformationFieldImage,
                          this->currentMask,
                          this->interpolation,
                          this->warpedPaddingValue);
        pre = this->ComputeSimilarityMeasure();
        paramPtr[i] = currentParamVal + eps;
        // warp image
        reg_resampleImage(this->currentFloating,
                          this->warped,
                          this->deformationFieldImage,
                          this->currentMask,
                          this->interpolation,
                          this->warpedPaddingValue);
        post = this->ComputeSimilarityMeasure();
        // set param value to its original value
        paramPtr[i] = currentParamVal;
        // finite difference
        approxGradPtr[i] = (post - pre) / (2.*eps);
        T error = fabs(approxGradPtr[i] - gradPtr[i]);
        errorGradPtr[i] = error;
//        if (approxGradPtr[i] != 0) {
//            errorGradPtr[i] = error / fabs(approxGradPtr[i]);
//        }
//        else {
//            errorGradPtr[i] = error;
//        }
        std::cout << "grad[" << i << "] = " << std::fixed << gradPtr[i]
                  << "   ~   "
                  << std::fixed << approxGradPtr[i]
                  << "  [" << std::fixed << error << "]" << std::endl;
    }
    // save approximated gradient
    reg_io_WriteImageFile(approxGrad, "voxel_based_grad_finite_diff_ipopt.nii");
    reg_io_WriteImageFile(errorGrad, "voxel_based_grad_absolute_error_grad.nii");
}

template <class T>
void reg_f3d_ipopt<T>::printImgStat() {
    std::cout.precision(17);
    T *refImg = static_cast<T *>(this->currentReference->data);
    T *floImg = static_cast<T *>(this->currentFloating->data);
    T minRef = reg_tools_getMinValue(this->currentReference, 0);
    T minFlo = reg_tools_getMinValue(this->currentFloating, 0);
    T maxRef = reg_tools_getMaxValue(this->currentReference, 0);
    T maxFlo = reg_tools_getMaxValue(this->currentFloating, 0);
    T maxAbsDiff = fabs(refImg[0] - floImg[0]);
    for (int i=1; i < this->currentFloating->nvox; i++) {
        if (fabs(refImg[i] - floImg[i]) > maxAbsDiff) {
            maxAbsDiff = fabs(refImg[i] - floImg[i]);
        }
    }
    std::cout << std::endl;
    std::cout << "Images stats:" << std::endl;
    std::cout << "Ref Image: min = " << minRef << ", max = " << maxRef << std::endl;
    std::cout << "Flo Image: min = " << minFlo << ", max = " << maxFlo << std::endl;
    std::cout << "Max absolute difference = " << maxAbsDiff << std::endl;
    std::cout << std::endl;
}

#endif //NIFTYREG_REG_F3D_IPOPT_H
