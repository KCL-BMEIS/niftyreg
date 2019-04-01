//
// Created by lf18 on 29/03/19.
//

#ifndef NIFTYREG_DIV_FREE_PROJECTION_H
#define NIFTYREG_DIV_FREE_PROJECTION_H

#include "IpTNLP.hpp"
#include "IpIpoptCalculatedQuantities.hpp"
#include "IpIpoptData.hpp"
#include "IpTNLPAdapter.hpp"
#include "IpOrigIpoptNLP.hpp"
#include "_reg_tools.h"
#include "_reg_ReadWriteImage.h"
#include "nifti1_io.h"
#include "nifti1.h"
#include <cassert>

using namespace Ipopt;

/** This inherit from the optimisation API for IPOPT
    It aims at projecting a velocity vector field parameterised as a cubic B-spline
    on the space of (approximated) divergence-free.
    This is only an approximation because the divergence operator has to be discretised
    on the grid. The finner the grid the more accurate is the projection.
    Here we use a symmetric 2nd order approximation of the divergence operator.
 **/
template <class T>
class div_free_projection : public TNLP {
public:
    div_free_projection(nifti_image *vel);

    virtual ~div_free_projection();

    void set_save_path(std::string save_path);

//    void updateDivFreeVelocityField(Number *x);

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
     *
     */
//    virtual bool intermediate_callback(AlgorithmMode mode,
//                                       Index iter, Number obj_value,
//                                       Number inf_pr, Number inf_du,
//                                       Number mu, Number d_norm,
//                                       Number regularization_size,
//                                       Number alpha_du, Number alpha_pr,
//                                       Index ls_trials,
//                                       const IpoptData* ip_data,
//                                       IpoptCalculatedQuantities* ip_cq);

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
    // input velocity vector field to make incompressible
    nifti_image *velocityControlPointGrid;
    std::string savePath;
    // output divergence-free velocity vector field
//    nifti_image *divFreeVelocityControlPointGrid;

private:
    // delete the default constructor
    div_free_projection();
    // delete setter for the velocity field to project
    void setVelocityField(nifti_image vel);

};

template <class T>
div_free_projection<T>::div_free_projection(nifti_image *vel) {
    this->velocityControlPointGrid = vel;
    // change the data type of the velocity field to T
    reg_tools_changeDatatype<T>(this->velocityControlPointGrid);
    // everything is done with displacements in this class
    reg_getDisplacementFromDeformation(this->velocityControlPointGrid);
    // initialise the velocity grid to store intermediate best solutions
//    this->divFreeVelocityControlPointGrid = nifti_copy_nim_info(vel);
//    this->divFreeVelocityControlPointGrid->data = (void *)calloc(vel->nvox, vel->nbyper);
}

template <class T>
div_free_projection<T>::~div_free_projection() {
    nifti_image_free(this->velocityControlPointGrid);
//    nifti_image_free(this->divFreeVelocityControlPointGrid);
}

template <class T>
void div_free_projection<T>::set_save_path(std::string save_path) {
    this->savePath = save_path;
}

template <class T>
// returns the size of the problem.
// IPOPT uses this information when allocating the arrays
// that it will later ask you to fill with values.
bool div_free_projection<T>::get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
                                          Index& nnz_h_lag, IndexStyleEnum& index_style) {
#ifndef NDEBUG
    std::cout <<"Call get_nlp_info" << std::endl;
#endif
    // number of variables
    n = this->velocityControlPointGrid->nvox;

    // number of constraints: one per grid point with a margin of 1
    m = (this->velocityControlPointGrid->nx - 2) * (this->velocityControlPointGrid->ny - 2);
    if (this->velocityControlPointGrid->nz > 2) { // 3D
        m *= this->velocityControlPointGrid->nz - 2;
    }

    // number of non zero values in the the constraint jacobian matrix
    nnz_jac_g = m * 2 * this->velocityControlPointGrid->nu;  // 2 non-zero values per constraint and per dimension

    // number of non zero values in the hessian
    nnz_h_lag = n;  // the Hessian is identity for an euclidian projection

    // use the C style indexing (0-based) for the matrices
    index_style = TNLP::C_STYLE;
    return true;
}

template <class T>
// returns the variable bounds
bool div_free_projection<T>::get_bounds_info(Index n, // number of variables (dim of x)
                                             Number* x_l, // lower bounds for x
                                             Number* x_u, // upperbound for x
                                             Index m, // number of constraints (dim of g(x))
                                             Number* g_l, // lower bounds for g(x)
                                             Number* g_u) { // upper bounds for g(x)
#ifndef NDEBUG
    std::cout <<"Call get_bounds_info" << std::endl;
#endif
    T minVel = reg_tools_getMinValue(this->velocityControlPointGrid, 0);
    T maxVel = reg_tools_getMaxValue(this->velocityControlPointGrid, 0);
    // set lower and upper bounds for the primal variables (displacement vector field)
    // It helps for the stability of the optimisation
    // add 1 to make sure we can reach the best solution with interior point method
    for (Index i = 0; i < n; i++) {
        x_l[i] = minVel - 1;  // in mm
        x_u[i] = maxVel + 1;  // in mm
    }
    // set lower and upper bounds for the inequality constraints
    for (Index i = 0; i < m; i++) {
        g_l[i] = 0.;
        g_u[i] = 0.;
    }
    return true;
}

template <class T>
// returns the initial point for the problem
bool div_free_projection<T>::get_starting_point(Index n, bool init_x, Number* x,
                                                bool init_z, Number* z_L, Number* z_U,
                                                Index m, bool init_lambda, Number* lambda) {
    // Here, we assume we only have starting values for x, if you code
    // your own NLP, you can provide starting values for the dual variables if you wish
    assert(init_x == true);
    assert(init_z == false);  // no warm start for the dual variables
    assert(init_lambda == false);  // no warm start for the interior point parameter

    T *controlPointPtr = static_cast<T *>(this->velocityControlPointGrid->data);
    for (int i = 0; i < n; ++i) {
        x[i] = static_cast<Number>(controlPointPtr[i]);
    }
    return true;
}

template <class T>
// return the objective function: 1/2*SSD
bool div_free_projection<T>::eval_f(Index n, const Number *x, bool new_x, Number &obj_value) {
    obj_value = 0;
    T *controlPointPtr = static_cast<T *>(this->velocityControlPointGrid->data);
    for (int i = 0; i < n; ++i) {
        obj_value += 0.5 * (x[i] - static_cast<Number>(controlPointPtr[i])) * (x[i] - static_cast<Number>(controlPointPtr[i]));
    }
    return true;
}

template <class T>
// return the gradient of the objective function
bool div_free_projection<T>::eval_grad_f(Index n, const Number *x, bool new_x, Number *grad_f) {
    T *controlPointPtr = static_cast<T *>(this->velocityControlPointGrid->data);
    for (int i = 0; i < n; ++i) {
        grad_f[i] = x[i] - static_cast<Number>(controlPointPtr[i]);
    }
    return true;
}

template <class T>
// return the value of the approximated divergence-free constraint
bool div_free_projection<T>::eval_g(Index n, const Number *x, bool new_x, Index m, Number *g) {
    // we use a second order approximation for first order partial derivative:
    // f'(x) = (f(x+1) - f(x-1)) / 2*spacing + O(spacing^2)
    if (m > 0) {  // constraint
        int numGridPoint = this->velocityControlPointGrid->nx * this->velocityControlPointGrid->ny * this->velocityControlPointGrid->nz;
        // compute the grid size which appears in the formula of the divergence
        T gridVoxelSpacing[3];
        gridVoxelSpacing[0] = this->velocityControlPointGrid->dx;
        gridVoxelSpacing[1] = this->velocityControlPointGrid->dy;
        // index for g
        int index = 0;
        // index for velocityControlPointGrid
        int tempIndexNextX, tempIndexPrevX = 0;  // index for the first component of the vector field
        int tempIndexNextY, tempIndexPrevY = 0;  // index for the second component of the vector field
        int tempIndexNextZ, tempIndexPrevZ = 0;  // index for the third component of the vector field
//        int tempIndexPrevX = 0;
//        int tempIndexPrevY = 0;
//        int tempIndexPrevZ = 0;
        if (this->velocityControlPointGrid->nz > 1) {  // 3D
            gridVoxelSpacing[2] = this->velocityControlPointGrid->dz;
            // the constraint is not imposed in marginal points
            for (int k=1; k < (this->velocityControlPointGrid->nz - 1); ++k) {
                for (int j=1; j < (this->velocityControlPointGrid->ny - 1); ++j) {
                    for (int i=1; i < (this->velocityControlPointGrid->nx - 1); ++i) {
                        // prepare index for active values
                        tempIndexNextX = k * this->velocityControlPointGrid->nx * this->velocityControlPointGrid->ny
                                         + j * this->velocityControlPointGrid->nx + i + 1;
                        tempIndexPrevX = k * this->velocityControlPointGrid->nx * this->velocityControlPointGrid->ny
                                         + j * this->velocityControlPointGrid->nx + i - 1;
                        tempIndexNextY = numGridPoint + k * this->velocityControlPointGrid->nx * this->velocityControlPointGrid->ny
                                         + (j+1) * this->velocityControlPointGrid->nx + i;
                        tempIndexPrevY = numGridPoint + k * this->velocityControlPointGrid->nx * this->velocityControlPointGrid->ny
                                         + (j-1) * this->velocityControlPointGrid->nx + i;
                        tempIndexNextZ = 2*numGridPoint + (k+1) * this->velocityControlPointGrid->nx * this->velocityControlPointGrid->ny
                                         + j * this->velocityControlPointGrid->nx + i;
                        tempIndexPrevZ = 2*numGridPoint + (k-1) * this->velocityControlPointGrid->nx * this->velocityControlPointGrid->ny
                                         + j * this->velocityControlPointGrid->nx + i;
                        // set the value of the constraint function for the current index
                        g[index] = (x[tempIndexNextX] - x[tempIndexPrevX]) / (2*gridVoxelSpacing[0])
                                   + (x[tempIndexNextY] - x[tempIndexPrevY]) / (2*gridVoxelSpacing[1])
                                   + (x[tempIndexNextZ] - x[tempIndexPrevZ]) / (2*gridVoxelSpacing[2]);
                        ++index;
                    }
                }
            }
        }  // 3D
        else {  // 2D
            for (int j=1; j < (this->velocityControlPointGrid->ny - 1); ++j) {
                for (int i=1; i < (this->velocityControlPointGrid->nx - 1); ++i) {
                    // prepare index for active values
                    tempIndexNextX = j * this->velocityControlPointGrid->nx + i + 1;
                    tempIndexPrevX = j * this->velocityControlPointGrid->nx + i - 1;
                    tempIndexNextY = (j+1) * this->velocityControlPointGrid->nx + i;
                    tempIndexPrevY = (j-1) * this->velocityControlPointGrid->nx + i;
                    // set the value of the constraint function for the current index
                    g[index] = (x[tempIndexNextX] - x[tempIndexPrevX]) / (2*gridVoxelSpacing[0])
                               + (x[tempIndexNextY] - x[tempIndexPrevY]) / (2*gridVoxelSpacing[1]);
                    ++index;
                }
            }
        }  // 2D
    }  // constraint

    return true;
}

template <class T>
// return the structure or values of the jacobian
bool div_free_projection<T>::eval_jac_g(Index n, const Number* x, bool new_x,
                                   Index m, Index nele_jac, Index* iRow,
                                   Index *jCol, Number* values) {
#ifndef NDEBUG
    std::cout <<"Call eval_jac_g" << std::endl;
#endif
    if (m > 0) {  // constraint
        int numGridPoint = this->velocityControlPointGrid->nx * this->velocityControlPointGrid->ny * this->velocityControlPointGrid->nz;
        Number gridVoxelSpacing[3];
        gridVoxelSpacing[0] = (Number) (this->velocityControlPointGrid->dx);
        gridVoxelSpacing[1] = (Number) (this->velocityControlPointGrid->dy);
        // constraint index (rows)
        int index = 0;
        // variable index (columns)
//        int tempIndex = 0;
        int tempIndexPrevX, tempIndexNextX = 0;
        int tempIndexPrevY, tempIndexNextY = 0;
        int tempIndexPrevZ, tempIndexNextZ = 0;
        if (this->velocityControlPointGrid->nz > 1) {  // 3D
            gridVoxelSpacing[2] = (Number) (this->velocityControlPointGrid->dz);
            // the border os the grid is excluded from the constraint
            for (int k = 1; k < (this->velocityControlPointGrid->nz - 1); ++k) {
                for (int j = 1; j < (this->velocityControlPointGrid->ny - 1); ++j) {
                    for (int i = 1; i < (this->velocityControlPointGrid->nx - 1); ++i) {
                        // prepare index for active values
                        tempIndexNextX = k * this->velocityControlPointGrid->nx * this->velocityControlPointGrid->ny
                                         + j * this->velocityControlPointGrid->nx + i + 1;
                        tempIndexPrevX = k * this->velocityControlPointGrid->nx * this->velocityControlPointGrid->ny
                                         + j * this->velocityControlPointGrid->nx + i - 1;
                        tempIndexNextY = numGridPoint + k * this->velocityControlPointGrid->nx * this->velocityControlPointGrid->ny
                                         + (j+1) * this->velocityControlPointGrid->nx + i;
                        tempIndexPrevY = numGridPoint + k * this->velocityControlPointGrid->nx * this->velocityControlPointGrid->ny
                                         + (j-1) * this->velocityControlPointGrid->nx + i;
                        tempIndexNextZ = 2*numGridPoint + (k+1) * this->velocityControlPointGrid->nx * this->velocityControlPointGrid->ny
                                         + j * this->velocityControlPointGrid->nx + i;
                        tempIndexPrevZ = 2*numGridPoint + (k-1) * this->velocityControlPointGrid->nx * this->velocityControlPointGrid->ny
                                         + j * this->velocityControlPointGrid->nx + i;
                        if (values == NULL) {  // return the structure of the constraint jacobian
                            // iRow and jCol contain the index of non zero entries of the jacobian of the constraint
                            // index of the columns correspond to the 1D-array x that contains the primal variables
                            // index of the rows correspond to the constraint index
                            iRow[6 * index] = index;
                            jCol[6 * index] = tempIndexNextX;
                            iRow[6 * index + 1] = index;
                            jCol[6 * index + 1] = tempIndexPrevX;
                            iRow[6 * index + 2] = index;
                            jCol[6 * index + 2] = tempIndexNextY;
                            iRow[6 * index + 3] = index;
                            jCol[6 * index + 3] = tempIndexPrevY;
                            iRow[6 * index + 4] = index;
                            jCol[6 * index + 4] = tempIndexNextZ;
                            iRow[6 * index + 5] = index;
                            jCol[6 * index + 5] = tempIndexPrevZ;
                            }  // jacobian structure
                        else {  // return the values of the jacobian of the constraints
                            values[6 * index] = 0.5 / gridVoxelSpacing[0];
                            values[6 * index + 1] = -0.5 / gridVoxelSpacing[0];
                            values[6 * index + 2] = 0.5 / gridVoxelSpacing[1];
                            values[6 * index + 3] = -0.5 / gridVoxelSpacing[1];
                            values[6 * index + 4] = 0.5 / gridVoxelSpacing[2];
                            values[6 * index + 5] = -0.5 / gridVoxelSpacing[2];
                        }  // jacobian values
                        ++index;
                    }
                }
            }
        }  // 3D
        else {  // 2D
            for (int j = 1; j < (this->velocityControlPointGrid->ny - 1); ++j) {
                for (int i = 1; i < (this->velocityControlPointGrid->nx - 1); ++i) {
                    // prepare index for active values
                    tempIndexNextX = j * this->velocityControlPointGrid->nx + i + 1;
                    tempIndexPrevX = j * this->velocityControlPointGrid->nx + i - 1;
                    tempIndexNextY = numGridPoint + (j + 1) * this->velocityControlPointGrid->nx + i;
                    tempIndexPrevY = numGridPoint + (j - 1) * this->velocityControlPointGrid->nx + i;
                    if (values == NULL) {  // return the structure of the constraint jacobian
                        // iRow and jCol contain the index of non zero entries of the jacobian of the constraint
                        // index of the columns correspond to the 1D-array x that contains the primal variables
                        // index of the rows correspond to the constraint index
                        iRow[4 * index] = index;
                        jCol[4 * index] = tempIndexNextX;
                        iRow[4 * index + 1] = index;
                        jCol[4 * index + 1] = tempIndexPrevX;
                        iRow[4 * index + 2] = index;
                        jCol[4 * index + 2] = tempIndexNextY;
                        iRow[4 * index + 3] = index;
                        jCol[4 * index + 3] = tempIndexPrevY;
                    }  // jacobian structure
                    else {  // return the values of the jacobian of the constraints
                        values[4 * index] = 0.5 / gridVoxelSpacing[0];
                        values[4 * index + 1] = -0.5 / gridVoxelSpacing[0];
                        values[4 * index + 2] = 0.5 / gridVoxelSpacing[1];
                        values[4 * index + 3] = -0.5 / gridVoxelSpacing[1];
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
// that is identity for an euclidian projection
bool div_free_projection<T>::eval_h(Index n, const Number* x, bool new_x,
                               Number obj_factor, Index m, const Number* lambda,
                               bool new_lambda, Index nele_hess, Index* iRow,
                               Index* jCol, Number* values){
    for (Index i = 0; i < n; ++i) {
        if (values == NULL) {
            // return the structure (lower or upper triangular part) of the
            // Hessian of the Lagrangian function
            iRow[i] = i;
            jCol[i] = i;
        } else {
            // return the values of the Hessian of the Lagrangian function
            values[i] = 1;
        }
    }

    return true;
}

template <class T>
void div_free_projection<T>::finalize_solution(SolverReturn status, Index n, const Number *x, const Number *z_L,
                                               const Number *z_U, Index m, const Number *g, const Number *lambda,
                                               Number obj_value, const IpoptData *ip_data,
                                               IpoptCalculatedQuantities *ip_cq) {
    T *controlPointGridPtr = static_cast<T *>(this->velocityControlPointGrid->data);
    for (int i = 0; i < n; ++i) {
        controlPointGridPtr[i] = (T)(x[i]);
    }
    // convert the projection back to a deforamtion field
    // because NiftyReg uses deformations by default...
    reg_getDeformationFromDisplacement(this->velocityControlPointGrid);
    reg_io_WriteImageFile(this->velocityControlPointGrid, this->savePath.c_str());
}
template class div_free_projection<double>;
template class div_free_projection<float>;
#endif //NIFTYREG_DIV_FREE_PROJECTION_H
