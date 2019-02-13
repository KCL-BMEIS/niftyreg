// Author:  Lucas Fidon
// Class for F3D2 registration using the Ipopt optimisation solver
//

#ifndef _REG_F3D2_IPOPT_H
#define _REG_F3D2_IPOPT_H

#include "_reg_f3d2.h"
#include "IpTNLP.hpp"
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
    bool optimiseBackwardTransform;  // if false, the backward trans is set to be the inverse of the Forward trans
//    nifti_image ** optimInitControlPointGridPyramid;
//    nifti_image * currentOptimInitControlPointGrid;

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
}

template <class T>
reg_f3d2_ipopt<T>::~reg_f3d2_ipopt(){
#ifndef NDEBUG
   reg_print_msg_debug("reg_f3d2_ipopt destructor called");
#endif
}

template <class T>
void reg_f3d2_ipopt<T>::initLevel(int level){
  if(!this->initialised) {
    this->Initialise();
  }
  // forward and backward velocity grids are initialised to identity deformation field
  // see /cpu/_reg_localTrans.cpp l.410 (reg_getDeformationFromDisplacement is used)
  // which is very weired in the case of velocity grids...

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
void reg_f3d2_ipopt<T>::setNewControlPointGrid(const Number *x, Index n) {
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
//  std::cout << "Call get_nlp_info" << std::endl;
  // number of variables (forward + (optional) backward displacement grid)
  n = (int)(this->controlPointGrid->nvox);
  if (optimiseBackwardTransform) {
    n += (int)(this->backwardControlPointGrid->nvox);
  }
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
bool reg_f3d2_ipopt<T>::get_bounds_info(Index n, // number of variables (dim of x)
                                        Number* x_l, // lower bounds for x
                                        Number* x_u, // upperbound for x
                                        Index m, // number of constraints (dim of g(x))
                                        Number* g_l, // lower bounds for g(x)
                                        Number* g_u) { // upper bounds for g(x)
//  std::cout <<"Call get_bound_info" << std::endl;
  // lower and upper bounds for the primal variables
  for (Index i=0; i<n; i++) {
//    x_l[i] = -1e19;  // -infty
    x_l[i] = -1e2;  // in mm
//    x_u[i] = 1e19;  // +infty
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
bool reg_f3d2_ipopt<T>::get_starting_point(Index n, bool init_x, Number* x,
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

  // initialize the displacement field associated to the current control point grid
  T *controlPointPtr = static_cast<T *>(this->controlPointGrid->data);
  if (!optimiseBackwardTransform) {
    for (Index i = 0; i < n; i++) {
      x[i] = static_cast<Number>(controlPointPtr[i]);
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
  obj_value = -this->GetObjectiveFunctionValue();
  #ifndef NDEBUG
    std::cout << "FONCTION DE COUT = " << obj_value << std::endl;
  #endif

  return true;
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
      // Gradients of the forward and backward objective function terms are added.
      // v_back = -v_for
      // so we havr to substract the gradient to take into account the composition by -Id.
      grad_f[i] = (Number) (gradient[i] - backwardGradient[i]);
    }
  }
  else {
//    T *backwardGradient = static_cast<T *>(this->backwardTransformationGradient->data);
    size_t nvox_for = this->controlPointGrid->nvox;
    for (int i = 0; i < n; i++) {
      if (i < nvox_for) {
        grad_f[i] = (Number) gradient[i]; // no need to multiply by -1 because NiftyReg does gradient descent
      } else {
        grad_f[i] = (Number) backwardGradient[i - nvox_for];
      }
    }
  }

  return true;
}

template <class T>
// return the value of the constraints: g(x)
// Divergence fo the velocity vector field
bool reg_f3d2_ipopt<T>::eval_g(Index n, const Number* x,
                 bool new_x, Index m, Number* g)
{
  std::cout << "Call eval_g" << std::endl;
  // HERE COMPUTE VALUE OF ALL CONSTRAINTS IN g
  for (Index j=0; j<m; j++){
   g[j] = 0.;
  }

  return true;
}

template <class T>
// return the structure or values of the jacobian
bool reg_f3d2_ipopt<T>::eval_jac_g(Index n, const Number* x, bool new_x,
                 Index m, Index nele_jac, Index* iRow,
                 Index *jCol, Number* values){
  std::cout <<"Call eval_jac_g" << std::endl;
  if (values == NULL) {
    // return the structure of the jacobian
  }
  else {
    // return the values of the jacobian of the constraints
  }

  return true;
}

template <class T>
//return the structure or values of the hessian
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
void reg_f3d2_ipopt<T>::finalize_solution(SolverReturn status,
                    Index n, const Number* x,
                    const Number* z_L, const Number* z_U,
                    Index m, const Number* g,
                    const Number* lambda,
                    Number obj_value,
                    const IpoptData* ip_data,
                    IpoptCalculatedQuantities* ip_cq) {
  // update the current velocity vector field
  this->setNewControlPointGrid(x, n);

  this->GetObjectiveFunctionValue();  // make sure all the variables are up-to-date

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
  // Jacobian map for velocity grid
  reg_spline_GetJacobianDetFromVelocityGrid(jacobianDeterminantArray, this->controlPointGrid);
  fileName = stringFormat("jacobian_map_level%d.nii", this->currentLevel+1);
  reg_io_WriteImageFile(jacobianDeterminantArray, fileName.c_str());
  nifti_image_free(jacobianDeterminantArray);

//  std::cout << std::endl << "Writing solution file solution.txt" << std::endl;
  FILE* fp = fopen("solution.txt", "w");

  fprintf(fp, "\n\nObjective value\n");
  fprintf(fp, "f(x*) = %e\n", obj_value);
  fclose(fp);
}

#endif
