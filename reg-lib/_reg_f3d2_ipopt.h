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

  void setConstraintMask(nifti_image *m);

  void initLevel(int level);

  void clearLevel(int level);

  void setNewControlPointGrid(const Number *x, Index n);

  void setDivergenceConstraint(bool state);

  void setScale(float scale);

  void setSaveDir(std::string saveDir);

  void SaveStatInfo(std::string path);

  void printConfigInfo();

  void gradientCheck();

  void printImgStat();

  void GetDeformationField();

  void GetDeformationFieldEuler();

  void WarpFloatingImageEuler(int inter);

  nifti_image **GetWarpedImageEuler();

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
    nifti_image *constraintMask;  // mask indicating where to impose the incompressibility constraint
    int *currentConstraintMask;
    int *currentConstraintMaskGrid;  // projection of the constraint mask on the grid
    int **constraintMaskPyramid;
    std::string saveDir;


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
    this->constraintMask = NULL;
    this->constraintMaskPyramid = NULL;
    this->currentConstraintMask = NULL;
    this->currentConstraintMaskGrid = NULL;
    this->scalingCst = 1.;
}

template <class T>
reg_f3d2_ipopt<T>::~reg_f3d2_ipopt() {
    if (this->constraintMask != NULL) {
        nifti_image_free(this->constraintMask);
        this->constraintMask = NULL;
    }
    if (this->constraintMaskPyramid != NULL) {
        free(this->constraintMaskPyramid);
        this->constraintMaskPyramid = NULL;
    }
    if (this->currentConstraintMask != NULL) {
        free(this->currentConstraintMask);
        this->currentConstraintMask = NULL;
    }
    if (this->currentConstraintMaskGrid != NULL) {
        free(this->currentConstraintMaskGrid);
        this->currentConstraintMaskGrid = NULL;
    }
#ifndef NDEBUG
   reg_print_msg_debug("reg_f3d2_ipopt destructor called");
#endif
}

template<class T>
void reg_f3d2_ipopt<T>::setConstraintMask(nifti_image *m) {
    this->constraintMask = m;
#ifndef NDEBUG
    reg_print_fct_debug("reg_f3d2_ipopt<T>::setConstraintMask");
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
            // use divergence-conforming B-splines
            this->controlPointGrid->intent_p1 = DIV_CONFORMING_VEL_GRID;
            this->backwardControlPointGrid->intent_p1 = DIV_CONFORMING_VEL_GRID;
            // set the number of steps in the scaling and squaring
            this->controlPointGrid->intent_p2 = 8;
            this->backwardControlPointGrid->intent_p2 = 8;
            // create constraint mask pyramid
            if (this->constraintMask != NULL) {
                this->constraintMaskPyramid = (int **)calloc(this->levelToPerform, sizeof(int *));
                reg_createMaskPyramid<T>(this->constraintMask, this->constraintMaskPyramid, level,
                                         this->levelToPerform, this->activeVoxelNumber);
            }
        } // use divergence constraint
    }
    // forward and backward velocity grids are initialised to identity deformation field
    // see /cpu/_reg_localTrans.cpp l.410 (reg_getDeformationFromDisplacement is used)

    // Initialise best objective value and corresponding best control point grid
    this->bestObj = std::numeric_limits<T>::max();

    this->currentLevel = (unsigned int) level;

    // Set the current input images
    this->currentReference = this->referencePyramid[level];
    this->currentFloating = this->floatingPyramid[level];

    // The grid is refined if necessary and the reference and floating masks are set
    // and the weights for the regularisation terms are adapted
    T maxStepSize = this->InitialiseCurrentLevel();
//    this->currentMask = this->maskPyramid[level];

    // set the current constraint mask that is used to know where to apply the incompressibility constraint
    // on the control point grid.
    // We need to project the currentContraintMask that is defined on the space of the currentFloating image
    // into a currentConstraintMaskGrid that is defined on the space of the controlPointGrid
    if (this->constraintMask != NULL) {
        this->currentConstraintMask = this->constraintMaskPyramid[level];
        // project the constraint mask on the grid
        mat44 referenceMatrix_voxel_grid_to_real;
        if (this->controlPointGrid->sform_code > 0)
            referenceMatrix_voxel_grid_to_real = (this->controlPointGrid->sto_xyz);
        else referenceMatrix_voxel_grid_to_real = (this->controlPointGrid->qto_xyz);
        // read the ijk sform or qform, as appropriate
        mat44 referenceMatrix_real_to_voxel_dense;
        if (this->currentFloating->sform_code > 0)
            referenceMatrix_real_to_voxel_dense = (this->currentFloating->sto_ijk);
        else referenceMatrix_real_to_voxel_dense = (this->currentFloating->qto_ijk);
        int numVoxGrid = this->controlPointGrid->nx * this->controlPointGrid->ny * this->controlPointGrid->nz;
        this->currentConstraintMaskGrid = (int *)calloc(numVoxGrid, sizeof(int));
        float real[3] = {0.f, 0.f, 0.f};  // homogeneous coordinates in real space
        float voxelGrid[3] = {0.f, 0.f, 0.f};  // homogeneous coordinates in voxel grid space
        int voxelDense[3] = {0, 0, 0};  // homogeneous coordinates in voxel dense space
        int indexGrid = 0;
        int indexMask = 0;
        bool isOutside = false;  // state if the current control point is inside the dense image domain
        for (int k=0; k < this->controlPointGrid->nz; ++k) {
            for (int j=0; j < this->controlPointGrid->ny; ++j) {
                for (int i=0; i < this->controlPointGrid->nx; ++i) {
                    // set current voxel grid coordinates
                    voxelGrid[0] = i;
                    voxelGrid[1] = j;
                    voxelGrid[2] = k;
                    // go from grid voxel space into real world space
                    reg_mat44_mul(&referenceMatrix_voxel_grid_to_real,
                                  voxelGrid,  // in
                                  real);  // out
                    // go from real world space into dense image voxel space
                    reg_mat44_mul(&referenceMatrix_real_to_voxel_dense,
                                  real,  // in
                                  voxelDense);  // out
                    isOutside = false;
                    if (voxelDense[0] < 0 || voxelDense[1] < 0 || voxelDense[2] < 0 ||
                        voxelDense[0] >= this->currentFloating->nx || voxelDense[1] >= this->currentFloating->ny ||
                        voxelDense[2] >= this->currentFloating->nz) {
                        isOutside = true;
                    }
                    if (!isOutside) {
                        // currentConstraintMask is defined on the same space as the current floating image
                        indexMask = (int) (voxelDense[2] * this->currentFloating->nx * this->currentFloating->ny
                                           + voxelDense[1] * this->currentFloating->nx + voxelDense[0]);
                        // we only impose incompressibility constraints to the grid points inside the mask
                        if (this->currentConstraintMask[indexMask] > 0) {
                            indexGrid = k * this->controlPointGrid->nx * this->controlPointGrid->ny
                                        + j * this->controlPointGrid->nx
                                        + i;
                            this->currentConstraintMaskGrid[indexGrid] = 1;
                        }
                    }
                }  // i
            }  // j
        }  // k
    }  // set currentConstraintMaskGrid

    // Allocate image that depends on the reference image
    this->AllocateWarped();
    this->AllocateDeformationField();
    this->AllocateWarpedGradient();
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
    if (this->constraintMaskPyramid != NULL) {
        free(this->constraintMaskPyramid[level]);
        this->constraintMaskPyramid[level] = NULL;
    }
    if (this->currentConstraintMask != NULL) {
        this->currentConstraintMask = NULL;
    }
    if (this->currentConstraintMaskGrid != NULL) {
        free(this->currentConstraintMaskGrid);
        this->currentConstraintMaskGrid = NULL;
    }
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
void reg_f3d2_ipopt<T>::setSaveDir(std::string saveDir) {
    this->saveDir = saveDir;
}

template <class T>
void reg_f3d2_ipopt<T>::GetDeformationField() {
#ifndef NDEBUG
    std::cout << "reg_f3d2_ipopt<T>::GetDeformationField is called" << std::endl;
#endif
    // I don't want the number of steps to be updated
    bool updateStepNumber = false;
#ifndef NDEBUG
    char text[255];
   sprintf(text, "Velocity integration forward. Step number update=%i",updateStepNumber);
   reg_print_msg_debug(text);
#endif
    // The forward transformation is computed using the scaling-and-squaring approach
    reg_spline_getDefFieldFromVelocityGrid(this->controlPointGrid,  // in
                                           this->deformationFieldImage,  // out
                                           updateStepNumber);
#ifndef NDEBUG
    sprintf(text, "Velocity integration backward. Step number update=%i",updateStepNumber);
   reg_print_msg_debug(text);
#endif
    // The backward transformation is computed using the scaling-and-squaring approach
    reg_spline_getDefFieldFromVelocityGrid(this->backwardControlPointGrid,
                                           this->backwardDeformationFieldImage,
                                           false);
}

template <class T>
void reg_f3d2_ipopt<T>::GetDeformationFieldEuler() {
#ifndef NDEBUG
    std::cout << "GetDeformationFieldEuler is called" << std::endl;
#endif
    reg_spline_getDefFieldFromVelocityGridEuler(this->controlPointGrid,  // in
                                                this->deformationFieldImage);  // out
    reg_spline_getDefFieldFromVelocityGridEuler(this->backwardControlPointGrid,  // in
                                                this->backwardDeformationFieldImage);  // out
}

template <class T>
void reg_f3d2_ipopt<T>::WarpFloatingImageEuler(int inter) {
    // Compute the deformation fields
    this->GetDeformationFieldEuler();

    // Resample the floating image
    reg_resampleImage(this->currentFloating,
                      this->warped,
                      this->deformationFieldImage,
                      this->currentMask,
                      inter,
                      this->warpedPaddingValue);

    // Resample the reference image
    reg_resampleImage(this->currentReference, // input image
                      this->backwardWarped, // warped input image
                      this->backwardDeformationFieldImage, // deformation field
                      this->currentFloatingMask, // mask
                      inter, // interpolation type
                      this->warpedPaddingValue); // padding value
}

template<class T>
nifti_image **reg_f3d2_ipopt<T>::GetWarpedImageEuler() {
    // The initial images are used
    if(this->inputReference==NULL ||
       this->inputFloating==NULL ||
       this->controlPointGrid==NULL ||
       this->backwardControlPointGrid==NULL) {
        reg_print_fct_error("reg_f3d2_ipopt<T>::GetWarpedImageEuler()");
        reg_print_msg_error("The reference, floating and control point grid images have to be defined");
        reg_exit();
    }

    // Set the input images
    reg_f3d2<T>::currentReference = this->inputReference;
    reg_f3d2<T>::currentFloating = this->inputFloating;
    // No mask is used to perform the final resampling
    reg_f3d2<T>::currentMask = NULL;
    reg_f3d2<T>::currentFloatingMask = NULL;

    // Allocate the forward and backward warped images
    reg_f3d2<T>::AllocateWarped();
    // Allocate the forward and backward dense deformation field
    reg_f3d2<T>::AllocateDeformationField();

    // Warp the floating images into the reference spaces using a cubic spline interpolation
    this->WarpFloatingImageEuler(3); // cubic spline interpolation

    // Clear the deformation field
    reg_f3d2<T>::ClearDeformationField();

    // Allocate and save the forward transformation warped image
    nifti_image **warpedImage=(nifti_image **)malloc(2*sizeof(nifti_image *));
    warpedImage[0] = nifti_copy_nim_info(this->warped);
    warpedImage[0]->cal_min=this->inputFloating->cal_min;
    warpedImage[0]->cal_max=this->inputFloating->cal_max;
    warpedImage[0]->scl_slope=this->inputFloating->scl_slope;
    warpedImage[0]->scl_inter=this->inputFloating->scl_inter;
    warpedImage[0]->data=(void *)malloc(warpedImage[0]->nvox*warpedImage[0]->nbyper);
    memcpy(warpedImage[0]->data, this->warped->data, warpedImage[0]->nvox*warpedImage[0]->nbyper);

    // Allocate and save the backward transformation warped image
    warpedImage[1] = nifti_copy_nim_info(this->backwardWarped);
    warpedImage[1]->cal_min=this->inputReference->cal_min;
    warpedImage[1]->cal_max=this->inputReference->cal_max;
    warpedImage[1]->scl_slope=this->inputReference->scl_slope;
    warpedImage[1]->scl_inter=this->inputReference->scl_inter;
    warpedImage[1]->data=(void *)malloc(warpedImage[1]->nvox*warpedImage[1]->nbyper);
    memcpy(warpedImage[1]->data, this->backwardWarped->data, warpedImage[1]->nvox*warpedImage[1]->nbyper);

    // Clear the warped images
    reg_f3d2<T>::ClearWarped();

    // Return the two final warped images
    return warpedImage;
}

template<class T>
void reg_f3d2_ipopt<T>::SaveStatInfo(std::string path) {
    std::ofstream file(path);

    file << "Objective value = " << this->bestObj << std::endl;
    file << "(wMeasure) " << std::scientific << this->bestWMeasure
         << " | (wBE) " << std::scientific << this->bestWBE
         << " | (wLE) " << std::scientific << this->bestWLE
         << " | (wJac) " << std::scientific << this->bestWJac
         << " | (wLan)" << std::scientific << this->bestWLand
         << " | (wIC) " << std::scientific << this->bestIC << std::endl;

    file << std::endl;
    file << "Number of objective function evaluations = " << this->NumObjFctEval << std::endl;
    file << "Number of objective gradient evaluations = " << this->NumObjGradFctEval << std::endl;
    file.close();
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
        // set the number of constraints m
        m = 0;
        // we just have to count the number of non zero values in currentMaskGrid
        int numVoxGrid = this->controlPointGrid->nx * this->controlPointGrid->ny * this->controlPointGrid->nz;
        for (int i=0; i < numVoxGrid; ++i) {
            if (this->currentConstraintMaskGrid[i] > 0) {
                m += 1;
            }
        }
        // set the number of non zero values in the jacobian matrix of the constraint
        if (this->controlPointGrid->nz > 1) {
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
  // set lower and upper bounds for the primal variables (displacement vector field)
  for (Index i=0; i<n; i++) {
//    x_l[i] = -1e20;  // -infty
    x_l[i] = -10.;  // in mm
//    x_u[i] = 1e20;  // +infty
    x_u[i] = 10.;  // in mm
  }
  // set lower and upper bounds for the inequality constraints
  for (Index i=0; i<m; i++) {
      g_l[i] = 0.;  // zero for incompressibility
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
  obj_value = -this->scalingCst * this->GetObjectiveFunctionValue();
  #ifndef NDEBUG
    std::cout << "Objective function value = " << obj_value << std::endl;
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
      // so we have to substract the gradient to take into account the composition by -Id.
      grad_f[i] = (Number) (this->scalingCst*(gradient[i] - backwardGradient[i]));
    }
  }
  else {
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
    // compute the grid size which appears in the formula of the divergence
    T gridVoxelSpacing[3];
    gridVoxelSpacing[0] = this->controlPointGrid->dx / this->currentFloating->dx;
    gridVoxelSpacing[1] = this->controlPointGrid->dy / this->currentFloating->dy;
    // index for g
    int index = 0;
    // index for controlPointGrid
    int tempIndexX = 0;  // index for the first component of the vector field
    int tempIndexY = 0;  // index for the second component of the vector field
    int tempIndexZ = 0;  // index for the third component of the vector field
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
            // only constrain in the mask
            if (this->currentConstraintMaskGrid[tempIndexX] > 0) {
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
      }
    }  // 3D
    else {  // 2D
      for (int j=1; j < (this->controlPointGrid->ny - 1); ++j) {
        for (int i=1; i < (this->controlPointGrid->nx - 1); ++i) {
          tempIndexX = j * this->controlPointGrid->nx + i;
          // only constrain in the mask
          if (this->currentConstraintMaskGrid[tempIndexX] > 0) {
              tempIndexY = tempIndexX + numGridPoint;
              tempIndexPrevY = tempIndexY - this->controlPointGrid->nx;
              tempIndexPrevX = tempIndexX - 1;
              g[index] = (x[tempIndexX] - x[tempIndexPrevX]) / gridVoxelSpacing[0]
                         + (x[tempIndexY] - x[tempIndexPrevY]) / gridVoxelSpacing[1];
              ++index;
          }
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
            // only constraint inside the mask
            if (this->currentConstraintMaskGrid[tempIndex] > 0) {
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
      }
    }  // 3D
    else {  // 2D
      for (int j = 1; j < (this->controlPointGrid->ny - 2); ++j) {
        for (int i = 1; i < (this->controlPointGrid->nx - 2); ++i) {
          tempIndex = j * this->controlPointGrid->nx + i;
          // only constrain in the mask
          if (this->currentConstraintMaskGrid[tempIndex] > 0) {  // only constraint inside the mask
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
        this->bestWMeasure = this->currentWMeasure;
        this->bestWBE = this->currentWBE;
        this->bestWLE = this->currentWLE;
        this->bestWJac = this->currentWJac;
        this->bestWLand = this->currentWLand;
        this->bestIC = this->currentIC;
      }
    }
  }
  // print values of each individual objective function term
  std::cout << "(totalObj) " << std::scientific << obj_value
            << " | (wMeasure) " << std::scientific << this->currentWMeasure
            << " | (wBE) " << std::scientific << this->currentWBE
            << " | (wLE) " << std::scientific << this->currentWLE
            << " | (wJac) " << std::scientific << this->currentWJac
            << " | (wLan)" << std::scientific << this->currentWLand
            << " | (wIC) " << std::scientific << this->currentIC << std::endl;
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

    // save floating and reference images
//    fileName = stringFormat("%s/input_flo_level%d.nii.gz", this->saveDir.c_str(), this->currentLevel+1);
//    reg_io_WriteImageFile(this->currentFloating, fileName.c_str());
//    fileName = stringFormat("%s/input_ref_level%d.nii.gz", this->saveDir.c_str(), this->currentLevel+1);
//    reg_io_WriteImageFile(this->currentReference, fileName.c_str());

//    if (this->useDivergenceConstraint and this->currentConstraintMask != NULL) {
//        // save currentMask
//        nifti_image *currMask = nifti_copy_nim_info(this->currentReference);
//        currMask->data = (void *)malloc(currMask->nvox*currMask->nbyper);
//        T *currMaskPtr = static_cast<T *>(currMask->data);
//        for (int i=0; i<currMask->nvox; ++i) {
//            currMaskPtr[i] = this->currentConstraintMask[i];
//        }
//        fileName = stringFormat("%s/input_mask_level%d.nii.gz", this->saveDir.c_str(), this->currentLevel+1);
//        reg_io_WriteImageFile(currMask, fileName.c_str());
//        nifti_image_free(currMask);
//        currMask = NULL;
//
//        // save currentMaskGrid
//        nifti_image *currMaskGrid = nifti_copy_nim_info(this->controlPointGrid);
//        currMaskGrid->nu = 1;
//        currMaskGrid->nvox = currMaskGrid->nx * currMaskGrid->ny * currMaskGrid->nz;
//        currMaskGrid->data = (void *)malloc(currMaskGrid->nvox*currMaskGrid->nbyper);
//        T *currMaskGridPtr = static_cast<T *>(currMaskGrid->data);
//        for (int i=0; i<currMaskGrid->nvox; ++i) {
//            currMaskGridPtr[i] = this->currentConstraintMaskGrid[i];
//        }
//        fileName = stringFormat("%s/input_mask_grid_level%d.nii.gz", this->saveDir.c_str(), this->currentLevel+1);
//        reg_io_WriteImageFile(currMaskGrid, fileName.c_str());
//        nifti_image_free(currMaskGrid);
//    }

//    // Compute and save the jacobian map fo the forward transformation
//    size_t nvox = (size_t) this->currentFloating->nx * this->currentFloating->ny * this->currentFloating->nz;
//    nifti_image *jacobianDeterminantArray = nifti_copy_nim_info(this->currentFloating);
//    jacobianDeterminantArray->nbyper = this->controlPointGrid->nbyper;
//    jacobianDeterminantArray->datatype = this->controlPointGrid->datatype;
//    jacobianDeterminantArray->data = (void *)calloc(nvox, this->controlPointGrid->nbyper);
//    // initialise the jacobian array values to 1
//    reg_tools_addValueToImage(jacobianDeterminantArray,
//                              jacobianDeterminantArray,
//                              1);
//    jacobianDeterminantArray->cal_min=0;
//    jacobianDeterminantArray->cal_max=0;
//    jacobianDeterminantArray->scl_slope = 1.0f;
//    jacobianDeterminantArray->scl_inter = 0.0f;
//
//    // original niftyreg jacobian for f3d2
//    // re-initialise the jacobian determinant map
//    reg_tools_multiplyValueToImage(jacobianDeterminantArray, jacobianDeterminantArray, 0);
//    reg_tools_addValueToImage(jacobianDeterminantArray,
//                              jacobianDeterminantArray,
//                              1);
//    reg_spline_GetJacobianDetFromVelocityGrid(jacobianDeterminantArray, this->controlPointGrid);
//    fileName = stringFormat("%s/output_jacobian_map_Marc_level%d.nii.gz",
//                            this->saveDir.c_str(), this->currentLevel+1);
//    reg_io_WriteImageFile(jacobianDeterminantArray, fileName.c_str());

    // True jacobian of the integrator
    nifti_image *jac = reg_spline_GetJacobianFromVelocityGrid(this->deformationFieldImage,
                                                              this->controlPointGrid);
    fileName = stringFormat("%s/output_jacobian_integrator_level%d.nii.gz",
                            this->saveDir.c_str(), this->currentLevel+1);
    reg_io_WriteImageFile(jac, fileName.c_str());

    // free nifti_image instance
//    nifti_image_free(jacobianDeterminantArray);
    nifti_image_free(jac);
//    this->deformationFieldImage->intent_p1 = defIntentP1;

    // compute and save the divergence of the stationary velocity field
    nifti_image *divergenceImage = reg_spline_GetDivergenceFromVelocityGrid(this->deformationFieldImage,
                                                                            this->controlPointGrid);
    fileName = stringFormat("%s/output_divergence_map_level%d.nii.gz",
                            this->saveDir.c_str(), this->currentLevel+1);
    reg_io_WriteImageFile(divergenceImage, fileName.c_str());
    nifti_image_free(divergenceImage);

    // Save the control point image
    nifti_image *outputControlPointGridImage = this->GetControlPointPositionImage();
    std::string outputCPPImageName=stringFormat("%s/outputCPP_level%d.nii.gz",
            this->saveDir.c_str(), this->currentLevel+1);
    memset(outputControlPointGridImage->descrip, 0, 80);
    strcpy (outputControlPointGridImage->descrip, "Velocity field grid from NiftyReg");
    reg_io_WriteImageFile(outputControlPointGridImage, outputCPPImageName.c_str());
//    // save the displacement control point grid
//    reg_getDisplacementFromDeformation(outputControlPointGridImage);
//    outputCPPImageName=stringFormat("%s/outputCPPdisplacement_level%d.nii.gz",
//                                    this->saveDir.c_str(), this->currentLevel+1);
//    reg_io_WriteImageFile(outputControlPointGridImage, outputCPPImageName.c_str());
    nifti_image_free(outputControlPointGridImage);

//    // Save the warped image(s)
//    // allocate memory for two images for the symmetric case
//    nifti_image **outputWarpedImage=(nifti_image **)malloc(2*sizeof(nifti_image *));
//    outputWarpedImage[0] = NULL;
//    outputWarpedImage[1] = NULL;
//    outputWarpedImage = this->GetWarpedImage();
//    fileName = stringFormat("%s/output_warped_flo_level%d.nii.gz", this->saveDir.c_str(), this->currentLevel+1);
//    memset(outputWarpedImage[0]->descrip, 0, 80);
//    strcpy (outputWarpedImage[0]->descrip, "Warped image using NiftyReg");
//    reg_io_WriteImageFile(outputWarpedImage[0], fileName.c_str());
//    if (outputWarpedImage[1] != NULL) {
//        fileName = stringFormat("%s/output_warped_ref_level%d.nii.gz", this->saveDir.c_str(), this->currentLevel+1);
//        memset(outputWarpedImage[1]->descrip, 0, 80);
//        strcpy (outputWarpedImage[1]->descrip, "Warped backward image using NiftyReg");
//        reg_io_WriteImageFile(outputWarpedImage[1], fileName.c_str());
//    }
    // Compute and save absolute error map
//    reg_tools_substractImageToImage(outputWarpedImage[0],
//            this->currentReference, outputWarpedImage[0]);
//    reg_tools_abs_image(outputWarpedImage[0]);
//    fileName = stringFormat("%s/output_abs_error_level%d.nii.gz", this->saveDir.c_str(), this->currentLevel+1);
//    reg_io_WriteImageFile(outputWarpedImage[0], fileName.c_str());
//    // free allocated memory
//    if(outputWarpedImage[0]!=NULL)
//        nifti_image_free(outputWarpedImage[0]);
//    outputWarpedImage[0]=NULL;
//    if(outputWarpedImage[1]!=NULL)
//        nifti_image_free(outputWarpedImage[1]);
//    outputWarpedImage[1]=NULL;
//    free(outputWarpedImage);

    // Save the warped image(s) with an Euler integration
    // allocate memory for two images for the symmetric case
//    nifti_image **outputWarpedImageEuler=(nifti_image **)malloc(2*sizeof(nifti_image *));
//    outputWarpedImageEuler[0] = NULL;
//    outputWarpedImageEuler[1] = NULL;
//    outputWarpedImageEuler = this->GetWarpedImageEuler();
//    fileName = stringFormat("%s/output_warped_euler_flo_level%d.nii.gz", this->saveDir.c_str(), this->currentLevel+1);
//    memset(outputWarpedImageEuler[0]->descrip, 0, 80);
//    strcpy (outputWarpedImageEuler[0]->descrip, "Warped image using NiftyReg");
//    reg_io_WriteImageFile(outputWarpedImageEuler[0], fileName.c_str());
//    if (outputWarpedImageEuler[1] != NULL) {
//        fileName = stringFormat("%s/output_warped_euler_ref_level%d.nii.gz", this->saveDir.c_str(), this->currentLevel+1);
//        memset(outputWarpedImageEuler[1]->descrip, 0, 80);
//        strcpy (outputWarpedImageEuler[1]->descrip, "Warped backward image using NiftyReg");
//        reg_io_WriteImageFile(outputWarpedImageEuler[1], fileName.c_str());
//    }
//    // Compute and save absolute error map
//    reg_tools_substractImageToImage(outputWarpedImageEuler[0],
//                                    this->currentReference, outputWarpedImageEuler[0]);
//    reg_tools_abs_image(outputWarpedImageEuler[0]);
//    fileName = stringFormat("%s/output_abs_error_euler_level%d.nii.gz", this->saveDir.c_str(), this->currentLevel+1);
//    reg_io_WriteImageFile(outputWarpedImageEuler[0], fileName.c_str());
//    // free allocated memory
//    if(outputWarpedImageEuler[0]!=NULL)
//        nifti_image_free(outputWarpedImageEuler[0]);
//    outputWarpedImageEuler[0]=NULL;
//    if(outputWarpedImageEuler[1]!=NULL)
//        nifti_image_free(outputWarpedImageEuler[1]);
//    outputWarpedImageEuler[1]=NULL;
//    free(outputWarpedImageEuler);

    std::cout << "Objective value = "  << this->bestObj << std::endl;

    std::string sol_path = this->saveDir;
    sol_path += "/final_objective.txt";
    this->SaveStatInfo(sol_path);
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
    reg_io_WriteImageFile(grad, "grad_f3d2_ipopt_gce.nii.gz");
  }
  else {
    if (this->useLucasExpGradient) {
      reg_io_WriteImageFile(grad, "grad_f3d2_ipopt_exp.nii.gz");
    }
    else {
      reg_io_WriteImageFile(grad, "grad_f3d2_ipopt.nii.gz");
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
    std::cout << "grad[" << i << "] = " << std::fixed << gradPtr[i]
              << "   ~   "
              << std::fixed << approxGradPtr[i]
              << "  [" << std::fixed << error << "]" << std::endl;
  }
  // save approximated gradient
  reg_io_WriteImageFile(approxGrad, "finite_diff_grad_f3d2_ipopt.nii.gz");
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
