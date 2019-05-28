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

/** This inherit from the optimisation API for IPOPT.
    It aims at projecting a velocity vector field parameterised as a cubic B-spline
    on the space of (approximatly) divergence-free velocity field.
    This is only an approximation because the divergence of the velocity field can
    only be imposed to be zero on the grid.
    As a result, the finner the grid the more accurate is the projection.
 **/
template <class T>
class div_free_projection : public TNLP {
public:
    div_free_projection(nifti_image *vel);

    virtual ~div_free_projection();

    int voxel_to_index(int i_x, int i_y, int i_z, int i_d);

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
    int numGridPoints;
    float divCoef [3] = {1./6., 2./3., 1./6.};

private:
    // delete the default constructor
    div_free_projection();
    // delete setter for the velocity field to project
    void setVelocityField(nifti_image vel);

};

template <class T>
div_free_projection<T>::div_free_projection(nifti_image *vel) {
    if (vel->intent_p1 != SPLINE_VEL_GRID) {
        reg_print_msg_error("Only cubic B-splines velocity fields are supported");
        reg_exit();
    }
    this->velocityControlPointGrid = vel;
    // change the data type of the velocity field to T
    reg_tools_changeDatatype<T>(this->velocityControlPointGrid);
    // everything is done with displacements in this class
    reg_getDisplacementFromDeformation(this->velocityControlPointGrid);
    this->numGridPoints = this->velocityControlPointGrid->nx * this->velocityControlPointGrid->ny
            * this->velocityControlPointGrid->nz;
}

template <class T>
div_free_projection<T>::~div_free_projection() {
    nifti_image_free(this->velocityControlPointGrid);
}

template <class T>
int div_free_projection<T>::voxel_to_index(int i_x, int i_y, int i_z, int i_d) {
    // the forth dimension correspond to the dimension of the vector of the velocity field
    assert(id < 3);
    int index = i_d * this->numGridPoints
            + i_z * this->velocityControlPointGrid->nx * this->velocityControlPointGrid->ny
            + i_y * this->velocityControlPointGrid->nx
            + i_x;
    return index;
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
    if (this->velocityControlPointGrid->nu == 2) {
        nnz_jac_g = m * 6 * this->velocityControlPointGrid->nu;  // 2 * 3 non-zero values per constraint and per dimension
    }
    else {  // 3D
        nnz_jac_g = m * 18 * this->velocityControlPointGrid->nu;  // 2 * 3 * 3 non-zero values per constraint and per dimension
    }
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
    // we apply constraint on the (true) divergence of the cubic B-spline
    // but only on the grid
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
    // b-spline coefficient for the divergence
    float coef = 0.;
    if (this->velocityControlPointGrid->nz > 1) {  // 3D
        gridVoxelSpacing[2] = this->velocityControlPointGrid->dz;
        // the constraint is imposed everywhere except in marginal points
        // go over all knots (i, j k) where a constraint must be imposed
        for (int k=1; k < (this->velocityControlPointGrid->nz - 1); ++k) {
            for (int j=1; j < (this->velocityControlPointGrid->ny - 1); ++j) {
                for (int i=1; i < (this->velocityControlPointGrid->nx - 1); ++i) {
                    g[index] = 0.;
                    // for a given knot (i, j, k) go over all neighbor knots parameters and all channels
                    // contributing to the divergence of (i, j, k)
                    for (int u=0; u < 3; ++u) {
                        for (int v=0; v < 3; ++v) {
                            coef = this->divCoef[u] * this->divCoef[v];
                            // update for 1st component of the velocity field param
                            tempIndexNextX = this->voxel_to_index(i + 1, j + u - 1, k + v - 1, 0);
                            tempIndexPrevX = this->voxel_to_index(i - 1, j + u - 1, k + v - 1, 0);
                            g[index] += coef * (x[tempIndexNextX] - x[tempIndexPrevX]) / (2*gridVoxelSpacing[0]);
                            // update for 2nd component of the velocity field param
                            tempIndexNextY = this->voxel_to_index(i + u - 1, j + 1, k + v - 1, 1);
                            tempIndexPrevY = this->voxel_to_index(i + u - 1, j - 1, k + v - 1, 1);
                            g[index] += coef * (x[tempIndexNextY] - x[tempIndexPrevY]) / (2*gridVoxelSpacing[1]);
                            // update for 3rd component of the velocity field param
                            tempIndexNextZ = this->voxel_to_index(i + u - 1, j + v - 1, k + 1, 2);
                            tempIndexPrevZ = this->voxel_to_index(i + u - 1, j + v - 1, k - 1, 2);
                            g[index] += coef * (x[tempIndexNextZ] - x[tempIndexPrevZ]) / (2*gridVoxelSpacing[2]);
                        }
                    }  // neighbors
                    ++index;
                }
            }
        }  // knots with a constraint
    }  // 3D
    else {  // 2D
        for (int j=1; j < (this->velocityControlPointGrid->ny - 1); ++j) {
            for (int i=1; i < (this->velocityControlPointGrid->nx - 1); ++i) {
                g[index] = 0.;
                // for a given knot (i, j, k) go over all neighbor knots parameters and all channels
                // contributing to the divergence of (i, j, k)
                for (int u=0; u < 3; ++u) {
                    coef = this->divCoef[u];
                    // update for 1st component of the velocity field param
                    tempIndexNextX = this->voxel_to_index(i + 1, j + u - 1, 0, 0);
                    tempIndexPrevX = this->voxel_to_index(i - 1, j + u - 1, 0, 0);
                    g[index] += coef * (x[tempIndexNextX] - x[tempIndexPrevX]) / (2*gridVoxelSpacing[0]);
                    // update for 2nd component of the velocity field param
                    tempIndexNextY = this->voxel_to_index(i + u - 1, j + 1, 0, 1);
                    tempIndexPrevY = this->voxel_to_index(i + u - 1, j - 1, 0, 1);
                    g[index] += coef * (x[tempIndexNextY] - x[tempIndexPrevY]) / (2*gridVoxelSpacing[1]);
                }  // neighbors
                ++index;
            }
        }
    }  // 2D

    return true;
}

template <class T>
// return the structure or values of the jacobian
// IPOPT use a sparse representation for the jacobian matrix of the constraints
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
        float coef = 0;
        // constraint index (rows)
        int index = 0;
        // index for the sparse representation of the constraints Jacobian matrix
        int indexSparse = 0;
        // variable index (columns)
        int tempIndexPrevX, tempIndexNextX = 0;
        int tempIndexPrevY, tempIndexNextY = 0;
        int tempIndexPrevZ, tempIndexNextZ = 0;
        if (this->velocityControlPointGrid->nz > 1) {  // 3D
            gridVoxelSpacing[2] = (Number) (this->velocityControlPointGrid->dz);
            // the border os the grid is excluded from the constraint
            for (int k = 1; k < (this->velocityControlPointGrid->nz - 1); ++k) {
                for (int j = 1; j < (this->velocityControlPointGrid->ny - 1); ++j) {
                    for (int i = 1; i < (this->velocityControlPointGrid->nx - 1); ++i) {
                        // for a given knot (i, j, k) go over all neighbor knots parameters and all channels
                        // contributing to the divergence of (i, j, k)
                        for (int u=0; u < 3; ++u) {
                            for (int v=0; v < 3; ++v) {
                                coef = this->divCoef[u] * this->divCoef[v];
                                // update for 1st component of the velocity field param
                                tempIndexNextX = this->voxel_to_index(i + 1, j + u - 1, k + v - 1, 0);
                                tempIndexPrevX = this->voxel_to_index(i - 1, j + u - 1, k + v - 1, 0);
                                // update for 2nd component of the velocity field param
                                tempIndexNextY = this->voxel_to_index(i + u - 1, j + 1, k + v - 1, 1);
                                tempIndexPrevY = this->voxel_to_index(i + u - 1, j - 1, k + v - 1, 1);
                                // update for 3rd component of the velocity field param
                                tempIndexNextZ = this->voxel_to_index(i + u - 1, j + v - 1, k + 1, 2);
                                tempIndexPrevZ = this->voxel_to_index(i + u - 1, j + v - 1, k - 1, 2);
                                if (values == NULL) {  // return the (sparse) structure of the constraint jacobian
                                    // iRow and jCol contain the index of non zero entries of the jacobian of the constraint
                                    // index of the columns correspond to the 1D-array x that contains the primal variables
                                    // index of the rows correspond to the constraint index
                                    iRow[indexSparse] = index;
                                    jCol[indexSparse] = tempIndexNextX;
                                    indexSparse += 1;
                                    iRow[indexSparse] = index;
                                    jCol[indexSparse] = tempIndexPrevX;
                                    indexSparse += 1;
                                    iRow[indexSparse] = index;
                                    jCol[indexSparse] = tempIndexNextY;
                                    indexSparse += 1;
                                    iRow[indexSparse] = index;
                                    jCol[indexSparse] = tempIndexPrevY;
                                    indexSparse += 1;
                                    iRow[indexSparse] = index;
                                    jCol[indexSparse] = tempIndexNextZ;
                                    indexSparse += 1;
                                    iRow[indexSparse] = index;
                                    jCol[indexSparse] = tempIndexPrevZ;
                                    indexSparse += 1;
                                }  // jacobian structure
                                else {  // return the values of the jacobian of the constraints
                                    values[indexSparse] = coef * 0.5 / gridVoxelSpacing[0];
                                    indexSparse += 1;
                                    values[indexSparse] = -coef * 0.5 / gridVoxelSpacing[0];
                                    indexSparse += 1;
                                    values[indexSparse] = coef * 0.5 / gridVoxelSpacing[1];
                                    indexSparse += 1;
                                    values[indexSparse] = -coef * 0.5 / gridVoxelSpacing[1];
                                    indexSparse += 1;
                                    values[indexSparse] = coef * 0.5 / gridVoxelSpacing[2];
                                    indexSparse += 1;
                                    values[indexSparse] = -coef * 0.5 / gridVoxelSpacing[2];
                                    indexSparse += 1;
                                }  // jacobian values
                            }
                        }  // neighbors
                        ++index;
                    }
                }
            }
        }  // 3D
        else {  // 2D
            for (int j = 1; j < (this->velocityControlPointGrid->ny - 1); ++j) {
                for (int i = 1; i < (this->velocityControlPointGrid->nx - 1); ++i) {
                    // for a given knot (i, j, k) go over all neighbor knots parameters and all channels
                    // contributing to the divergence of (i, j, k)
                    for (int u=0; u < 3; ++u) {
                        coef = this->divCoef[u];
                        // update for 1st component of the velocity field param
                        tempIndexNextX = this->voxel_to_index(i + 1, j + u - 1, 0, 0);
                        tempIndexPrevX = this->voxel_to_index(i - 1, j + u - 1, 0, 0);
                        // update for 2nd component of the velocity field param
                        tempIndexNextY = this->voxel_to_index(i + u - 1, j + 1, 0, 1);
                        tempIndexPrevY = this->voxel_to_index(i + u - 1, j - 1, 0, 1);
                        if (values == NULL) {  // return the (sparse) structure of the constraint jacobian
                            // iRow and jCol contain the index of non zero entries of the jacobian of the constraint
                            // index of the columns correspond to the 1D-array x that contains the primal variables
                            // index of the rows correspond to the constraint index
                            iRow[indexSparse] = index;
                            jCol[indexSparse] = tempIndexNextX;
                            indexSparse += 1;
                            iRow[indexSparse] = index;
                            jCol[indexSparse] = tempIndexPrevX;
                            indexSparse += 1;
                            iRow[indexSparse] = index;
                            jCol[indexSparse] = tempIndexNextY;
                            indexSparse += 1;
                            iRow[indexSparse] = index;
                            jCol[indexSparse] = tempIndexPrevY;
                            indexSparse += 1;
                        }  // jacobian structure
                        else {  // return the values of the jacobian of the constraints
                            values[indexSparse] = coef * 0.5 / gridVoxelSpacing[0];
                            indexSparse += 1;
                            values[indexSparse] = -coef * 0.5 / gridVoxelSpacing[0];
                            indexSparse += 1;
                            values[indexSparse] = coef * 0.5 / gridVoxelSpacing[1];
                            indexSparse += 1;
                            values[indexSparse] = -coef * 0.5 / gridVoxelSpacing[1];
                            indexSparse += 1;
                        }  // jacobian values
                    }  // neighbors

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
    // convert the projection back to a deformation field
    // because NiftyReg uses deformations by default...
    reg_getDeformationFromDisplacement(this->velocityControlPointGrid);
    reg_io_WriteImageFile(this->velocityControlPointGrid, this->savePath.c_str());

    // save the backward velocity grid
    std::string b(this->savePath);
    if(b.find( ".nii.gz") != std::string::npos)
        b.replace(b.find( ".nii.gz"),7,"_backward.nii.gz");
    reg_getDisplacementFromDeformation(this->velocityControlPointGrid);
    reg_tools_multiplyValueToImage(this->velocityControlPointGrid,
            this->velocityControlPointGrid, -1.f);
    reg_getDeformationFromDisplacement(this->velocityControlPointGrid);
    reg_io_WriteImageFile(this->velocityControlPointGrid, b.c_str());
}
template class div_free_projection<double>;
template class div_free_projection<float>;
#endif //NIFTYREG_DIV_FREE_PROJECTION_H
