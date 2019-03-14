// Author:  Lucas Fidon
// Class for F3D2 registration using the Ipopt optimisation solver
//

#ifndef _REG_F3D2_IPOPT_H
#define _REG_F3D2_IPOPT_H

#include "_reg_f3d2.h"
#include "IpTNLP.hpp"
#include "IpIpoptCalculatedQuantities.hpp"
#include "IpIpoptData.hpp"
#include "IpTNLPAdapter.hpp"
#include "IpOrigIpoptNLP.hpp"
//#include "exception.h"
#include <cassert>
#include <cstdio>

using namespace Ipopt;


// This inherits from NiftyReg reg_f3d2 class and extends its API
// so as to perform the optimisation using Ipopt library
template <class T>
class reg_f3d2_ipopt : public reg_f3d2<T>, public TNLP
{
public:
  /** constructor that takes in problem data */
  reg_f3d2_ipopt(int refTimePoint, int floTimePoint);

  /** default destructor */
  virtual ~reg_f3d2_ipopt();

  void initLevel(int level);

  void clearLevel(int level);

  void setNewControlPointGrid(const Number *x, Index n);

  void setDivergenceConstraint(bool state);

  void setScale(float scale);

  void printConfigInfo();

  void gradientCheck();

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

  /** Override parent method to deal with divergence-conforming B-spline used for constraints*/
  virtual void GetSimilarityMeasureGradient();

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
   *
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
    int n;  // number of variables in ipopt
    bool optimiseBackwardTransform;  // if false, the backward trans is set to be the inverse of the Forward trans
    double scalingCst;
    T bestObj;  // best objective function value
    Number *bestX;  // variable values corresponding to the best objective function
    bool useDivergenceConstraint;


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

  /** @name reg_f3d2_ipopt data */
  //@{
  //@}
};

template <class T>
reg_f3d2_ipopt<T>::reg_f3d2_ipopt(int refTimePoint, int floTimePoint)
    : reg_f3d2<T>::reg_f3d2(refTimePoint, floTimePoint) {
    this->optimiseBackwardTransform = false;
    this->useGradientCumulativeExp = false; // approximate gradient of the exponential by displacement gradient
    this->useLucasExpGradient = false;
    this->useDivergenceConstraint = false;
    this->scalingCst = 1.;
}

template <class T>
reg_f3d2_ipopt<T>::~reg_f3d2_ipopt() {
#ifndef NDEBUG
   reg_print_msg_debug("reg_f3d2_ipopt destructor called");
#endif
}

template <class T>
void reg_f3d2_ipopt<T>::initLevel(int level) {
#ifndef NDEBUG
  std::cout <<"Call initLevel" << std::endl;
#endif
    if(!this->initialised) {
        this->Initialise();
        if (this->useDivergenceConstraint) {
          this->controlPointGrid->intent_p1 = DIV_CONFORMING_VEL_GRID;
          this->backwardControlPointGrid->intent_p1 = DIV_CONFORMING_VEL_GRID;
        }
    }
    // forward and backward velocity grids are initialised to identity deformation field
    // see /cpu/_reg_localTrans.cpp l.410 (reg_getDeformationFromDisplacement is used)

    // Initialise best objective value and corresponding best control point grid
    this->bestObj = std::numeric_limits<T>::max();

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

    // Evalulate the initial objective function value
    this->UpdateBestObjFunctionValue();
    this->PrintInitialObjFunctionValue();
}

template <class T>
void reg_f3d2_ipopt<T>::clearLevel(int level) {
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
void reg_f3d2_ipopt<T>::setDivergenceConstraint(bool state) {
    this->useDivergenceConstraint = state;
}

template <class T>
void reg_f3d2_ipopt<T>::setScale(float scale) {
    this->scalingCst = (T) scale;
}

template <class T>
void reg_f3d2_ipopt<T>::printConfigInfo() {
    std::cout << std::endl;
    std::cout << "#################" << std::endl;
    std::cout << "REG_F3D2 Config Info" << std::endl;
    std::cout << "Scaling factor used for the loss function = " << std::scientific << this->scalingCst << std::endl;
    if (this->useLucasExpGradient) {
        std::cout << "True gradient of the approximated exponential mapping is used." << std::endl;
    }
    if (this->useDivergenceConstraint) {
      std::cout << "Divergence conforming B-spline with hard constraint on its divergence is used." << std::endl;
    }
    std::cout << "#################" << std::endl;
    std::cout << std::endl;
}

template <class T>
void reg_f3d2_ipopt<T>::setNewControlPointGrid(const Number *x, Index n) {
#ifndef NDEBUG
  std::cout <<"Call setNewControlPointGrid" << std::endl;
#endif
  if (!optimiseBackwardTransform) {
    assert(n == this->controlPointGrid->nvox);
    T *controlPointGridPtr = static_cast<T *>(this->controlPointGrid->data);
    T *backwardControlPointGridPtr = static_cast<T *>(this->backwardControlPointGrid->data);
    for (int i=0; i<n; i++) {
      controlPointGridPtr[i] = (T) x[i];
      backwardControlPointGridPtr[i] = (T) (-x[i]);  // velocity field of the inverse transformation
    }
  }
  else {
    size_t nvox_for = this->controlPointGrid->nvox;
    size_t nvox_back = this->backwardControlPointGrid->nvox;
    assert(n == nvox_for + nvox_back);
    T *controlPointGridPtr = static_cast<T *>(this->controlPointGrid->data);
    T *backwardControlPointGridPtr = static_cast<T *>(this->backwardControlPointGrid->data);
    for (int i = 0; i < n; i++) {
      if (i < nvox_for) {
        controlPointGridPtr[i] = (T) x[i];
      } else {
        backwardControlPointGridPtr[i - nvox_for] = (T) x[i];
      }
    }
  }
  // as x is a displacement/velocity grid, but NiftyReg works with deformation grid
  // it is necessary to convert controlPointGrid to a deformation
  reg_getDeformationFromDisplacement(this->controlPointGrid);  // add identity deformation
  reg_getDeformationFromDisplacement(this->backwardControlPointGrid);  // add identity deformation
};

template <class T>
// returns the size of the problem.
// IPOPT uses this information when allocating the arrays
// that it will later ask you to fill with values.
bool reg_f3d2_ipopt<T>::get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
                   Index& nnz_h_lag, IndexStyleEnum& index_style) {
#ifndef NDEBUG
  std::cout <<"Call get_nlp_info" << std::endl;
#endif
  // number of variables (forward + (optional) backward displacement grid)
  n = (int)(this->controlPointGrid->nvox);
  if (optimiseBackwardTransform) {
    n += (int)(this->backwardControlPointGrid->nvox);
  }
  this->n = n;
  // initialise tracking of the primal variables values that gives the best ojective function value
  this->bestX = new Number[n];
  // set the number of constraints m (null divergence of the velocity vector field)
  if (this->useDivergenceConstraint) {  // 3D
      // ignore border
      assert(this->controlPointGrid->nx > 2);
      assert(this->controlPointGrid->ny > 2);
      // m: number of constraints
      m = (int)((this->controlPointGrid->nx - 2) * (this->controlPointGrid->ny - 2));
      if (this->controlPointGrid->nz > 1) {
          assert(this->controlPointGrid->nz > 2);
          m *= (int)(this->controlPointGrid->nz - 2);
          // nnz_jac_g: number of non-zeros values in the jacobian of the constraint function
          nnz_jac_g = m * 6;  // 2 non zero values per constraint and dimension
      }  // 3D
      else {  // 2D
          nnz_jac_g = m * 4;
      } // 2D
  }
  else {
      m = 0;  // no constraint
      nnz_jac_g = 0;
  }
  // number of non-zeros values in the hessian
  nnz_h_lag = n * n; // full matrix in general
  // use the C style indexing (0-based) for the matrices
  index_style = TNLP::C_STYLE;
  return true;
}

template <class T>
// returns the variable bounds
bool reg_f3d2_ipopt<T>::get_bounds_info(Index n, // number of variables (dim of x)
                                        Number* x_l, // lower bounds for x
                                        Number* x_u, // upperbound for x
                                        Index m, // number of constraints (dim of g(x))
                                        Number* g_l, // lower bounds for g(x)
                                        Number* g_u) { // upper bounds for g(x)
#ifndef NDEBUG
  std::cout <<"Call get_bounds_info" << std::endl;
#endif
  // lower and upper bounds for the primal variables
  for (Index i=0; i<n; i++) {
//    x_l[i] = -1e20;  // -infty
//    x_l[i] = -1e2;  // in mm
    x_l[i] = -50.;  // in mm
//    x_u[i] = 1e20;  // +infty
//    x_u[i] = 1e2;  // in mm
    x_u[i] = 50.;  // in mm
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
bool reg_f3d2_ipopt<T>::get_starting_point(Index n, bool init_x, Number* x,
                     bool init_z, Number* z_L, Number* z_U,
                     Index m, bool init_lambda,
                     Number* lambda) {
#ifndef NDEBUG
  std::cout <<"Call get_starting_point" << std::endl;
#endif
  // Here, we assume we only have starting values for x, if you code
  // your own NLP, you can provide starting values for the dual variables if you wish
  assert(init_x == true);
  assert(init_z == false);
  assert(init_lambda == false);

  // make sure there is as many elements in x and the control point grid
  if (optimiseBackwardTransform) {
    assert(n == this->controlPointGrid->nvox + this->backwardControlPointGrid->nvox);
  }
  else {
    assert(n == this->controlPointGrid->nvox);
  }

  // x is a displacement grid
  // so the initial deformations are converted into displacements
  reg_getDisplacementFromDeformation(this->controlPointGrid);
  if (optimiseBackwardTransform) {
    reg_getDisplacementFromDeformation(this->backwardControlPointGrid);
  }

  // generator of random integer for random perturbation of the input
//    std::random_device rd;  //Will be used to obtain a seed for the random number engine
//    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
//    std::uniform_int_distribution<> dis(1, 20);

  // initialize the displacement field associated to the current control point grid
  T *controlPointPtr = static_cast<T *>(this->controlPointGrid->data);
  if (!optimiseBackwardTransform) {
    for (Index i = 0; i < n; i++) {
      x[i] = static_cast<Number>(controlPointPtr[i]);
//      x[i] += (Number)((dis(gen) - 10)*0.05);  // add random small perturbation (in mm)
    }
  }
  else {
    T *backwardControlPointPtr = static_cast<T *>(this->backwardControlPointGrid->data);
    size_t nvox_for = this->controlPointGrid->nvox;
    for (Index i = 0; i < n; i++) {
      if (i < nvox_for) {
        x[i] = static_cast<Number>(controlPointPtr[i]);
      } else {
        x[i] = static_cast<Number>(backwardControlPointPtr[i - nvox_for]);
      }
    }
  }
  // set the controlPointGrid as a deformation grid again
  reg_getDeformationFromDisplacement(this->controlPointGrid);
  if (optimiseBackwardTransform) {
    reg_getDeformationFromDisplacement(this->backwardControlPointGrid);
  }

#ifdef skip_me
  /* If checking derivatives, if is useful to choose different values */
#endif

  return true;
}

template <class T>
// returns the value of the objective function
bool reg_f3d2_ipopt<T>::eval_f(Index n, const Number* x,
                 bool new_x, Number& obj_value){
  #ifndef NDEBUG
    std::cout <<"Call eval_f" << std::endl;
  #endif

  // make sure there is as many elements in x and the control point grid
  if (optimiseBackwardTransform) {
    assert(n == this->controlPointGrid->nvox + this->backwardControlPointGrid->nvox);
  }
  else {
    assert(n == this->controlPointGrid->nvox);
  }

  // set the current velocity vector field to x
  this->setNewControlPointGrid(x, n);

  // take the opposite of objective value to maximise because ipopt performs minimisation
  obj_value = -this->scalingCst * this->GetObjectiveFunctionValue();
  #ifndef NDEBUG
    std::cout << "FONCTION DE COUT = " << obj_value << std::endl;
  #endif

  return true;
}

template <class T>
void reg_f3d2_ipopt<T>::GetSimilarityMeasureGradient() {
  if (!this->useDivergenceConstraint) {  // no constraint; can use parent method
    assert(this->controlPointGrid->intent_p1 != DIV_CONFORMING_VEL_GRID);
    reg_f3d2<T>::GetSimilarityMeasureGradient();
  }
  else { // use specific implementation for divergence conforming B-spline
#ifndef NDEBUG
      std::cout << "Divergence-conforming B-spline GetSimilarityMeasureGradient is called" << std::endl;
#endif
    this->GetVoxelBasedGradient();

    int kernel_type = DIV_CONFORMING_SPLINE_KERNEL;
    // The voxel based sim measure gradient is convolved with a spline kernel
    // Convolution along the x axis
    float currentNodeSpacing[3];
    currentNodeSpacing[0] = currentNodeSpacing[1] = currentNodeSpacing[2] = this->controlPointGrid->dx;
    bool activeAxis[3]= {true, false, false};
    reg_tools_kernelConvolution(this->voxelBasedMeasureGradient, // in and out
                                currentNodeSpacing,  // sigma
                                kernel_type,
                                NULL, // mask
                                NULL, // all volumes are considered as active
                                activeAxis);
    reg_tools_kernelConvolution(this->backwardVoxelBasedMeasureGradientImage, // in and out
                                currentNodeSpacing,  // sigma
                                kernel_type,
                                NULL, // mask
                                NULL, // all volumes are considered as active
                                activeAxis);
    // Convolution along the y axis
    currentNodeSpacing[0] = currentNodeSpacing[1] = currentNodeSpacing[2] = this->controlPointGrid->dy;
    activeAxis[0] = false;
    activeAxis[1] = true;
    reg_tools_kernelConvolution(this->voxelBasedMeasureGradient,
                                currentNodeSpacing,
                                kernel_type,
                                NULL, // mask
                                NULL, // all volumes are considered as active
                                activeAxis);
    reg_tools_kernelConvolution(this->backwardVoxelBasedMeasureGradientImage,
                                currentNodeSpacing,  // sigma
                                kernel_type,
                                NULL, // mask
                                NULL, // all volumes are considered as active
                                activeAxis);
    // Convolution along the z axis if required
    if(this->voxelBasedMeasureGradient->nz>1) {
      currentNodeSpacing[0] = currentNodeSpacing[1] = currentNodeSpacing[2] = this->controlPointGrid->dz;
      activeAxis[1] = false;
      activeAxis[2] = true;
      reg_tools_kernelConvolution(this->voxelBasedMeasureGradient,
                                  currentNodeSpacing,
                                  kernel_type,
                                  NULL, // mask
                                  NULL, // all volumes are considered as active
                                  activeAxis);
      reg_tools_kernelConvolution(this->backwardVoxelBasedMeasureGradientImage,
                                  currentNodeSpacing,  // sigma
                                  kernel_type,
                                  NULL, // mask
                                  NULL, // all volumes are considered as active
                                  activeAxis);
    }

    // The node based sim measure gradients are extracted
    mat44 reorientation;
    if(this->currentFloating->sform_code>0) {
        reorientation = this->currentFloating->sto_ijk;
    }
    else {
        reorientation = this->currentFloating->qto_ijk;
    }
    reg_voxelCentric2NodeCentric(this->transformationGradient,
                                 this->voxelBasedMeasureGradient,
                                 this->similarityWeight,
                                 false, // no update
                                 &reorientation);
    if(this->currentReference->sform_code>0) {
        reorientation = this->currentReference->sto_ijk;
    }
    else {
        reorientation = this->currentReference->qto_ijk;
    }
    reg_voxelCentric2NodeCentric(this->backwardTransformationGradient,
                                 this->backwardVoxelBasedMeasureGradientImage,
                                 this->similarityWeight,
                                 false, // no update
                                 &reorientation); // voxel to mm conversion

  } // divergence-conforming B-spline
}

template <class T>
// return the gradient of the objective function grad_{x} f(x)
// it sets all values of gradient in grad_f
bool reg_f3d2_ipopt<T>::eval_grad_f(Index n, const Number* x,
                  bool new_x, Number* grad_f){
  #ifndef NDEBUG
    std::cout << "Call eval_grad_f" << std::endl;

    printf("Grid: %d dimensions (%dx%dx%dx%dx%dx%dx%d)\n", this->controlPointGrid->ndim,
         this->controlPointGrid->nx, this->controlPointGrid->ny, this->controlPointGrid->nz,
         this->controlPointGrid->nt, this->controlPointGrid->nu, this->controlPointGrid->nv,
         this->controlPointGrid->nw);
  #endif
  // make sure there is as many elements the the vel and in x
  if (optimiseBackwardTransform) {
    assert(n == this->controlPointGrid->nvox + this->backwardControlPointGrid->nvox);
  }
  else {
    assert(n == this->controlPointGrid->nvox);
  }
  // set the current velocity field to x
  this->setNewControlPointGrid(x, n);

  // compute the objective function gradient value
  this->GetObjectiveFunctionValue();  // this is just to make sure it is up to date
  this->GetObjectiveFunctionGradient();
//  this->NormaliseGradient();
  // update the Ipopt gradient
  T *gradient = static_cast<T *>(this->transformationGradient->data);
  T *backwardGradient = static_cast<T *>(this->backwardTransformationGradient->data);
  if (!optimiseBackwardTransform) {
    for (int i = 0; i < n; i++) {
      #ifndef NDEBUG
      if (gradient[i] != gradient[i] || backwardGradient[i] != backwardGradient[i]) {
        std::cout << "Nan value found at voxel " << i << std::endl;
//        throw NaNValueInGradientException();
      }
      #endif
      // Gradients of the forward and backward objective function terms are added.
      // v_back = -v_for
      // so we have to substract the gradient to take into account the composition by -Id.
      grad_f[i] = (Number) (this->scalingCst*(gradient[i] - backwardGradient[i]));
      // prints to remove after debugging
//      std::cout << "grad_f[" << i << "] = " << std::scientific << grad_f[i] << std::endl;
//      std::cout << "forward_grad[" << i << "] = " << std::scientific << gradient[i] << std::endl;
//      std::cout << "backward_grad[" << i << "] = " << std::scientific << backwardGradient[i] << std::endl;
    }
  }
  else {
//    T *backwardGradient = static_cast<T *>(this->backwardTransformationGradient->data);
    size_t nvox_for = this->controlPointGrid->nvox;
    for (int i = 0; i < n; i++) {
      if (i < nvox_for) {
        grad_f[i] = (Number) (this->scalingCst * gradient[i]); // no need to multiply by -1 because NiftyReg does gradient descent
      } else {
        grad_f[i] = (Number) (this->scalingCst * backwardGradient[i - nvox_for]);
      }
    }
  }

  return true;
}

template <class T>
// return the value of the constraints: g(x)
// Divergence fo the velocity vector field
bool reg_f3d2_ipopt<T>::eval_g(Index n, const Number* x, bool new_x, Index m, Number* g) {
#ifndef NDEBUG
  std::cout <<"Call eval_g" << std::endl;
#endif
  if (m > 0) {  // constraint
    int numGridPoint = this->controlPointGrid->nx * this->controlPointGrid->ny * this->controlPointGrid->nz;
    T gridVoxelSpacing[3];
    gridVoxelSpacing[0] = this->controlPointGrid->dx / this->currentFloating->dx;
    gridVoxelSpacing[1] = this->controlPointGrid->dy / this->currentFloating->dy;
    // index for g
    int index = 0;
    // index for controlPointGrid
    int tempIndexX = 0;
    int tempIndexY = 0;
    int tempIndexZ = 0;
    int tempIndexPrevX = 0;
    int tempIndexPrevY = 0;
    int tempIndexPrevZ = 0;
    if (this->controlPointGrid->nz > 1) {  // 3D
      gridVoxelSpacing[2] = this->controlPointGrid->dz / this->currentFloating->dz;
      for (int k=1; k < (this->controlPointGrid->nz - 1); ++k) {
        for (int j=1; j < (this->controlPointGrid->ny - 1); ++j) {
          for (int i=1; i < (this->controlPointGrid->nx - 1); ++i) {
            tempIndexX = k * this->controlPointGrid->nx * this->controlPointGrid->ny
                    + j * this->controlPointGrid->nx + i;
            tempIndexY = tempIndexX + numGridPoint;
            tempIndexZ = tempIndexY + numGridPoint;
            tempIndexPrevZ = tempIndexZ - this->controlPointGrid->nx * this->controlPointGrid->ny;
            tempIndexPrevY = tempIndexY - this->controlPointGrid->nx;
            tempIndexPrevX = tempIndexX - 1;
            g[index] = (x[tempIndexX] - x[tempIndexPrevX]) / gridVoxelSpacing[0]
                       + (x[tempIndexY] - x[tempIndexPrevY]) / gridVoxelSpacing[1]
                       + (x[tempIndexZ] - x[tempIndexPrevZ]) / gridVoxelSpacing[2];
            ++index;
          }
        }
      }
    }  // 3D
    else {  // 2D
      for (int j=1; j < (this->controlPointGrid->ny - 1); ++j) {
        for (int i=1; i < (this->controlPointGrid->nx - 1); ++i) {
          tempIndexX = j * this->controlPointGrid->nx + i;
          tempIndexY = tempIndexX + numGridPoint;
          tempIndexPrevY = tempIndexY - this->controlPointGrid->nx;
          tempIndexPrevX = tempIndexX - 1;
          g[index] = (x[tempIndexX] - x[tempIndexPrevX]) / gridVoxelSpacing[0]
                     + (x[tempIndexY] - x[tempIndexPrevY]) / gridVoxelSpacing[1];
          ++index;
        }
      }
    }  // 2D
  }  // constraint

  return true;
}

template <class T>
// return the structure or values of the jacobian
bool reg_f3d2_ipopt<T>::eval_jac_g(Index n, const Number* x, bool new_x,
                 Index m, Index nele_jac, Index* iRow,
                 Index *jCol, Number* values) {
#ifndef NDEBUG
  std::cout <<"Call eval_jac_g" << std::endl;
#endif
  if (m > 0) {  // constraint
    int numGridPoint = this->controlPointGrid->nx * this->controlPointGrid->ny * this->controlPointGrid->nz;
    Number gridVoxelSpacing[3];
    gridVoxelSpacing[0] = (Number) (this->controlPointGrid->dx / this->currentFloating->dx);
    gridVoxelSpacing[1] = (Number) (this->controlPointGrid->dy / this->currentFloating->dy);
    // constraint index (rows)
    int index = 0;
    // variable index (columns)
    int tempIndex = 0;
    int tempIndexPrevX = 0;
    int tempIndexPrevY = 0;
    int tempIndexPrevZ = 0;
    if (this->controlPointGrid->nz > 1) {  // 3D
      gridVoxelSpacing[2] = (Number) (this->controlPointGrid->dz / this->currentFloating->dz);
      for (int k = 1; k < (this->controlPointGrid->nz - 1); ++k) {
        for (int j = 1; j < (this->controlPointGrid->ny - 1); ++j) {
          for (int i = 1; i < (this->controlPointGrid->nx - 1); ++i) {
            tempIndex = k * this->controlPointGrid->nx * this->controlPointGrid->ny
                        + j * this->controlPointGrid->nx + i;
            tempIndexPrevZ = tempIndex - this->controlPointGrid->nx * this->controlPointGrid->ny;
            tempIndexPrevY = tempIndex - this->controlPointGrid->nx;
            tempIndexPrevX = tempIndex - 1;
            if (values == NULL) {  // return the structure of the constraint jacobian
              // iRow and jCol contain the index of non zero entries of the jacobian of the constraint
              // index of the columns correspond to the 1D-array x that contains the problem variables
              // index of the rows correspond to the constraint number
              iRow[6 * index] = index;
              jCol[6 * index] = tempIndex;
              iRow[6 * index + 1] = index;
              jCol[6 * index + 1] = tempIndexPrevX;
              iRow[6 * index + 2] = index;
              jCol[6 * index + 2] = numGridPoint + tempIndex;
              iRow[6 * index + 3] = index;
              jCol[6 * index + 3] = numGridPoint + tempIndexPrevY;
              iRow[6 * index + 4] = index;
              jCol[6 * index + 4] = 2 * numGridPoint + tempIndex;
              iRow[6 * index + 5] = index;
              jCol[6 * index + 5] = 2 * numGridPoint + tempIndexPrevZ;
            }  // jacobian structure
            else {  // return the values of the jacobian of the constraints
              values[6 * index] = 1. / gridVoxelSpacing[0];
              values[6 * index + 1] = -1. / gridVoxelSpacing[0];
              values[6 * index + 2] = 1. / gridVoxelSpacing[1];
              values[6 * index + 3] = -1. / gridVoxelSpacing[1];
              values[6 * index + 4] = 1. / gridVoxelSpacing[2];
              values[6 * index + 5] = -1. / gridVoxelSpacing[2];
            }  // jacobian values
            ++index;
          }
        }
      }
    }  // 3D
    else {  // 2D
      for (int j = 1; j < (this->controlPointGrid->ny - 1); ++j) {
        for (int i = 1; i < (this->controlPointGrid->nx - 1); ++i) {
          tempIndex = j * this->controlPointGrid->nx + i;
          tempIndexPrevY = tempIndex - this->controlPointGrid->nx;
          tempIndexPrevX = tempIndex - 1;
          if (values == NULL) {  // return the structure of the constraint jacobian
            // iRow and jCol contain the index of non zero entries of the jacobian of the constraint
            // index of the columns correspond to the 1D-array x that contains the problem variables
            // index of the rows correspond to the constraint number
            iRow[4 * index] = index;
            jCol[4 * index] = tempIndex;
            iRow[4 * index + 1] = index;
            jCol[4 * index + 1] = tempIndexPrevX;
            iRow[4 * index + 2] = index;
            jCol[4 * index + 2] = numGridPoint + tempIndex;
            iRow[4 * index + 3] = index;
            jCol[4 * index + 3] = numGridPoint + tempIndexPrevY;
          }  // jacobian structure
          else {  // return the values of the jacobian of the constraints
            values[4 * index] = 1. / gridVoxelSpacing[0];
            values[4 * index + 1] = -1. / gridVoxelSpacing[0];
            values[4 * index + 2] = 1. / gridVoxelSpacing[1];
            values[4 * index + 3] = -1. / gridVoxelSpacing[1];
          }  // jacobian values
          ++index;
        }
      }
    }  // 2D
  }  // constraint

  return true;
}

template <class T>
// return the structure or values of the hessian
// not implemented: only Quasi-Newton methods are supported
bool reg_f3d2_ipopt<T>::eval_h(Index n, const Number* x, bool new_x,
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
bool reg_f3d2_ipopt<T>::intermediate_callback(AlgorithmMode mode,
                                             Index iter, Number obj_value,
                                             Number inf_pr, Number inf_du,
                                             Number mu, Number d_norm,
                                             Number regularization_size,
                                             Number alpha_du, Number alpha_pr,
                                             Index ls_trials,
                                             const IpoptData* ip_data,
                                             IpoptCalculatedQuantities* ip_cq) {
#ifndef NDEBUG
  std::cout <<"Call intermediate_callback" << std::endl;
#endif
  // update best objective function and corresponding control point grid
  // if an improvement has been observed during the current iteration
  if (obj_value < this->bestObj) {
    // get access to the primal variables
    TNLPAdapter *tnlp_adapter = NULL;
    if (ip_cq != NULL) {
      OrigIpoptNLP *orignlp;
      orignlp = dynamic_cast<OrigIpoptNLP *>(GetRawPtr(ip_cq->GetIpoptNLP()));
      if (orignlp != NULL) {
        tnlp_adapter = dynamic_cast<TNLPAdapter *>(GetRawPtr(orignlp->nlp()));
        // copy values of the varibles giving the new best objective function value
        tnlp_adapter->ResortX(*ip_data->curr()->x(), this->bestX);
        // update best objective function value
        this->bestObj = (T) obj_value;
      }
    }
  }
  return true;
}

template <class T>
void reg_f3d2_ipopt<T>::finalize_solution(SolverReturn status,
                    Index n, const Number* x,
                    const Number* z_L, const Number* z_U,
                    Index m, const Number* g,
                    const Number* lambda,
                    Number obj_value,
                    const IpoptData* ip_data,
                    IpoptCalculatedQuantities* ip_cq) {
  std::string fileName;
  // update the current velocity vector field
  this->setNewControlPointGrid(this->bestX, n);
//  this->setNewControlPointGrid(x, n);

//  this->GetObjectiveFunctionValue();  // make sure all the variables are up-to-date

  // Compute and save the jacobian map fo the forward transformation
  size_t nvox = (size_t) this->currentReference->nx * this->currentReference->ny * this->currentReference->nz;
  nifti_image *jacobianDeterminantArray = nifti_copy_nim_info(this->currentReference);
  jacobianDeterminantArray->nbyper = this->controlPointGrid->nbyper;
  jacobianDeterminantArray->datatype = this->controlPointGrid->datatype;
  jacobianDeterminantArray->data = (void *)calloc(nvox, this->controlPointGrid->nbyper);
  // initialise the jacobian array values to 1
  reg_tools_addValueToImage(jacobianDeterminantArray,
                            jacobianDeterminantArray,
                            1);
//  memset(jacobianDeterminantArray->data, 1, nvox * this->controlPointGrid->nbyper);
  jacobianDeterminantArray->cal_min=0;
  jacobianDeterminantArray->cal_max=0;
  jacobianDeterminantArray->scl_slope = 1.0f;
  jacobianDeterminantArray->scl_inter = 0.0f;
  // get the transformation
  this->GetDeformationField();
  this->deformationFieldImage->intent_p1 = LIN_SPLINE_GRID;
  // Jacobian map for deformation field
  reg_spline_GetJacobianMap(this->deformationFieldImage, jacobianDeterminantArray);
  // Jacobian map for velocity grid - Marc version
//  reg_spline_GetJacobianDetFromVelocityGrid(jacobianDeterminantArray, this->controlPointGrid);
  fileName = stringFormat("jacobian_map_ss_level%d.nii", this->currentLevel+1);
  reg_io_WriteImageFile(jacobianDeterminantArray, fileName.c_str());

  //TODO: compute the Jacobian of the integrator

  // free nifti_image instance
  nifti_image_free(jacobianDeterminantArray);

  // Save the control point image
  nifti_image *outputControlPointGridImage = this->GetControlPointPositionImage();
  std::string outputCPPImageName=stringFormat("outputCPP_level%d.nii", this->currentLevel+1);
  memset(outputControlPointGridImage->descrip, 0, 80);
  strcpy (outputControlPointGridImage->descrip,"Velocity field grid from NiftyReg");
  reg_io_WriteImageFile(outputControlPointGridImage,outputCPPImageName.c_str());
  nifti_image_free(outputControlPointGridImage);

  // Save the warped image(s)
  // allocate memory for two images for the symmetric case
  nifti_image **outputWarpedImage=(nifti_image **)malloc(2*sizeof(nifti_image *));
  outputWarpedImage[0] = NULL;
  outputWarpedImage[1] = NULL;
  outputWarpedImage = this->GetWarpedImage();
  fileName = stringFormat("out_level%d.nii", this->currentLevel+1);
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

  std::cout << "Objective value = "  << this->bestObj << std::endl;

//  std::cout << std::endl << "Writing solution file solution.txt" << std::endl;
  FILE* fp = fopen("solution.txt", "w");

  fprintf(fp, "\n\nObjective value\n");
  fprintf(fp, "f(x*) = %e\n", this->bestObj);
  fclose(fp);
}

template <class T>
void reg_f3d2_ipopt<T>::gradientCheck() {
  T eps = 1e-4;
  std::cout.precision(17);
  // set control point at which we will compute the gradient
  // set the transformation to identity
  reg_tools_multiplyValueToImage(this->controlPointGrid, this->controlPointGrid, 0.f);
  reg_getDeformationFromDisplacement(this->controlPointGrid);
  reg_tools_multiplyValueToImage(this->backwardControlPointGrid, this->backwardControlPointGrid, 0.f);
  reg_getDeformationFromDisplacement(this->backwardControlPointGrid);
  // compute the gradient in this->transformationGradient
  this->GetObjectiveFunctionGradient();
  // save the gradient
  nifti_image *grad = nifti_copy_nim_info(this->transformationGradient);
  grad->data = (void *)calloc(grad->nvox, grad->nbyper);
  reg_tools_addImageToImage(grad, this->transformationGradient, grad);
  reg_tools_substractImageToImage(grad, this->backwardTransformationGradient, grad);
  if (this->useGradientCumulativeExp) {
    reg_io_WriteImageFile(grad, "grad_f3d2_ipopt_gce.nii");
  }
  else {
    if (this->useLucasExpGradient) {
      reg_io_WriteImageFile(grad, "grad_f3d2_ipopt_exp.nii");
    }
    else {
      reg_io_WriteImageFile(grad, "grad_f3d2_ipopt.nii");
    }
  }
  // compute approximation of the gradient by finite difference
  nifti_image *approxGrad = nifti_copy_nim_info(this->transformationGradient);
  approxGrad->data = (void *)calloc(approxGrad->nvox, approxGrad->nbyper);
  nifti_image *errorGrad = nifti_copy_nim_info(this->transformationGradient);
  errorGrad->data = (void *)calloc(errorGrad->nvox, errorGrad->nbyper);
  T* gradPtr = static_cast<T*>(this->transformationGradient->data);
  T* approxGradPtr = static_cast<T*>(approxGrad->data);
  T* errorGradPtr = static_cast<T*>(errorGrad->data);
  T* paramPtr = static_cast<T*>(this->controlPointGrid->data);
  T* backParamPtr = static_cast<T*>(this->backwardControlPointGrid->data);
  T pre = 0;
  T post = 0;
  T currentParamVal = 0;
  T currentBackParamVal = 0;
  for (int i=0; i<approxGrad->nvox; ++i) {
    currentParamVal = paramPtr[i];
    currentBackParamVal = backParamPtr[i];
    paramPtr[i] = currentParamVal - eps;
    backParamPtr[i] = currentBackParamVal + eps;
    pre = -this->GetObjectiveFunctionValue();
    paramPtr[i] = currentParamVal + eps;
    backParamPtr[i] = currentBackParamVal - eps;
    post = -this->GetObjectiveFunctionValue();
    paramPtr[i] = currentParamVal;
    backParamPtr[i] = currentBackParamVal;
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
  reg_io_WriteImageFile(approxGrad, "finite_diff_grad_f3d2_ipopt.nii");
//  reg_io_WriteImageFile(errorGrad, "absolute_error_grad.nii");
}

template <class T>
void reg_f3d2_ipopt<T>::printImgStat() {
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

#endif
