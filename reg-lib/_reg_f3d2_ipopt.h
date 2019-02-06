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
class reg_f3d2_ipopt : public reg_f3d<T>, public TNLP
{
public:
  /** constructor that takes in problem data */
  reg_f3d2_ipopt(int refTimePoint, int floTimePoint);

  /** default destructor */
  virtual ~reg_f3d2_ipopt();

  void initLevel(int level);

  void clearLevel(int level);

  void updateOptimInitControlPoint(int level);

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
  : reg_f3d2<T>::reg_f3d(refTimePoint, floTimePoint){
  // Set the initial forward transformation to identity
//  memset(this->inputControlPointGrid->data, 0,
//         this->inputControlPointGrid->nvox*this->inputControlPointGrid->nbyper);
//  reg_getDeformationFromDisplacement(this->inputControlPointGrid);
//  std::cout << " Call reg_ipopt constructor" << std::endl;
//  if (this->usePyramid) {
//    this->optimInitControlPointGridPyramid = (nifti_image **)malloc(this->levelToPerform*sizeof(nifti_image *));
//  }
//  else {
//    this->optimInitControlPointGridPyramid = (nifti_image **)malloc(sizeof(nifti_image *));
//  }
//  std::cout << "init control point grid created" << std::endl;
//   Set the initial control point grid for the first level to identity
//  memset(this->optimInitControlPointGridPyramid[0]->data, 0,
//         this->optimInitControlPointGridPyramid[0]->nvox*this->optimInitControlPointGridPyramid[0]->nbyper);
//  reg_getDeformationFromDisplacement(this->optimInitControlPointGridPyramid[0]);
//  std::cout << "init control point grid initialised" << std::endl;
}

template <class T>
reg_f3d2_ipopt<T>::~reg_f3d2_ipopt(){
#ifndef NDEBUG
   reg_print_msg_debug("reg_f3d2_ipopt destructor called");
#endif
}

template <class T>
void reg_f3d2_ipopt<T>::initLevel(int level){
  if(!this->initialised) this->Initialise();
  this->currentLevel = level;
  // Set the current input images
  this->currentReference = this->referencePyramid[level];
  this->currentFloating = this->floatingPyramid[level];
//  this->currentOptimInitControlPointGrid = this->optimInitControlPointGridPyramid[level];
  this->currentMask = this->maskPyramid[level];
  // Allocate image that depends on the reference image
  this->AllocateWarped();
  this->AllocateDeformationField();
  this->AllocateWarpedGradient();
  // Set penalisation weights for current level
  if(level==0) {
    this->bendingEnergyWeight = this->bendingEnergyWeight / static_cast<T>(powf(16.0f, this->levelNumber-1));
    this->linearEnergyWeight = this->linearEnergyWeight / static_cast<T>(powf(3.0f, this->levelNumber-1));
  }
  else {
    this->bendingEnergyWeight = this->bendingEnergyWeight * static_cast<T>(16);
    this->linearEnergyWeight = this->linearEnergyWeight * static_cast<T>(3);
  }
  // The grid is refined if necessary
//  T maxStepSize=this->InitialiseCurrentLevel();  // This has been put in updateInputControlPoint
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
//  nifti_image_free(this->optimInitControlPointGridPyramid[level]);
//  this->optimInitControlPointGridPyramid[level] = NULL;
  free(this->maskPyramid[level]);
  this->maskPyramid[level] = NULL;
  this->ClearCurrentInputImage();
}

template <class T>
void reg_f3d2_ipopt<T>::updateOptimInitControlPoint(int level) {
  assert(level < this->levelToPerform - 1);
  // controlPointGrid is refined by dividing the control point spacing by a factor of 2
  reg_spline_refineControlPointGrid(this->controlPointGrid,
          this->referencePyramid[level+1]);
//  assert(this->controlPointGrid->nvox == this->optimInitControlPointGridPyramid[level+1]->nvox);
//  T * refinedPtr = static_cast<T *>(this->controlPointGrid->data);
//  T * nextOptimInitPtr = static_cast<T *>(this->optimInitControlPointGridPyramid[level+1]->data);
//  for (int i = 0; i < this->controlPointGrid->nvox; i++) {
//    nextOptimInitPtr[i] = refinedPtr[i];
//  }
}

template <class T>
// returns the size of the problem.
// IPOPT uses this information when allocating the arrays
// that it will later ask you to fill with values.
bool reg_f3d2_ipopt<T>::get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
                   Index& nnz_h_lag, IndexStyleEnum& index_style) {
//  std::cout << "Call get_nlp_info" << std::endl;
  // number of variables
  n = (int)this->controlPointGrid->nvox;
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
    x_l[i] = -1e19;  // -infty
    x_u[i] = 1e19;  // +infty
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
  // your own NLP, you can provide starting values for the dual variables
  // if you wish
  assert(init_x == true);
  assert(init_z == false);
  assert(init_lambda == false);

  // make sure there is as many elements in x and the control point grid
//  assert(this->currentOptimInitControlPointGrid->nvox == n);
  assert(this->controlPointGrid->nvox == n);

  // initialize to the given starting point
//  T *controlPointPtr = static_cast<T *>(this->currentOptimInitControlPointGrid->data);
  T *controlPointPtr = static_cast<T *>(this->controlPointGrid->data);
  for (Index i=0; i<n; i++) {
    x[i] = static_cast<Number>(controlPointPtr[i]); // start with an identity control point field
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
  assert(this->controlPointGrid->nvox == n);

  // set the current velocity vector field to x
  T *controlPointGridPtr = static_cast<T *>(this->controlPointGrid->data);
  for (int i=0; i<n; i++) {
    controlPointGridPtr[i] = (T) x[i];
  }

//    backwardControlPointGridPtr[i] = (T) (-x[i]);]
//  T *backwardControlPointGridPtr = static_cast<T *>(this->backwardControlPointGrid->data);
//    for (int i=0; i<n; i++){
//      backwardControlPointGridPtr[i] = 0;  // no sym
//  }
  // Set the backward transformation to identity
//  memset(this->backwardControlPointGrid->data, 0,
//          this->backwardControlPointGrid->nvox*
//          this->backwardControlPointGrid->nbyper);
//  reg_getDeformationFromDisplacement(this->backwardControlPointGrid);
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
  assert(this->controlPointGrid->nvox == n);
  // set the current velocity field to x
  T *controlPointGridPtr = static_cast<T *>(this->controlPointGrid->data);
//  T *backwardControlPointGridPtr = static_cast<T *>(this->backwardControlPointGrid->data);
  //std::shared_ptr<T> backwardControlPointGridPtr = static_cast<T *>(this->backwardControlPointGrid->data);
  for (int i=0; i<n; i++){
    controlPointGridPtr[i] = (T)x[i];
//    backwardControlPointGridPtr[i] = (T)(-x[i]);
//    backwardControlPointGridPtr[i] = 0;  // no sym
  }
  // Set the backward transformation to identity
//  memset(this->backwardControlPointGrid->data, 0,
//         this->backwardControlPointGrid->nvox*
//         this->backwardControlPointGrid->nbyper);
//  reg_getDeformationFromDisplacement(this->backwardControlPointGrid);
  // compute the objective function gradient value
  this->GetObjectiveFunctionValue();  // this is just to make sure it is up to date
  this->GetObjectiveFunctionGradient();
//  this->NormaliseGradient();
  // combine forward and backward gradients
//  reg_tools_addImageToImage(this->transformationGradient, // in1
//                            this->backwardTransformationGradient, // in2
//                            this->transformationGradient); // out
  // update the Ipopt gradient
  T *gradient = static_cast<T *>(this->transformationGradient->data);
//  T *gradient = this->optimiser->GetGradient();
  for(int i=0; i<n; i++){
//    grad_f[i] = (Number)(-gradient[i]);  // multiply by -1 because we minimise the objective
    grad_f[i] = (Number)gradient[i];
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
    //}
  }
  else {
    // return the values of the jacobian of the constraints
    //}
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
  T *controlPointGridPtr = static_cast<T *>(this->controlPointGrid->data);
  for (int i=0; i<n; i++) {
    controlPointGridPtr[i] = (T) x[i];
  }
  this->GetObjectiveFunctionValue();  // make sure all the variables are up-to-date

  // Save the warped image(s)
  // allocate memory for two images for the symmetric case
  nifti_image **outputWarpedImage=(nifti_image **)malloc(2*sizeof(nifti_image *));
  outputWarpedImage[0]=NULL;
  outputWarpedImage[1]=NULL;
  outputWarpedImage = this->GetWarpedImage();
  std::string fileName = stringFormat("out_level%d.nii", this->currentLevel+1);
  memset(outputWarpedImage[0]->descrip, 0, 80);
  strcpy (outputWarpedImage[0]->descrip, "Warped image using NiftyReg");
  reg_io_WriteImageFile(outputWarpedImage[0], fileName.c_str());
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

#endif
