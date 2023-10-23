/**
 * @file _reg_maths.h
 * @brief Library that contains small math routines
 * @author Marc Modat
 * @date 25/03/2009
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#pragma once

#include "RNifti.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#if USE_SSE
#include <emmintrin.h>
#include <xmmintrin.h>
#ifdef __SSE3__
#include <pmmintrin.h>
#endif
#endif

#define _USE_MATH_DEFINES
#include <math.h>

#ifdef __CUDACC__
#define DEVICE  __host__ __device__
#else
#define DEVICE
#endif

typedef enum {
    DEF_FIELD,
    DISP_FIELD,
    CUB_SPLINE_GRID,
    DEF_VEL_FIELD,
    DISP_VEL_FIELD,
    SPLINE_VEL_GRID,
    LIN_SPLINE_GRID
} NREG_TRANS_TYPE;

/* *************************************************************** */
namespace NiftyReg {
/* *************************************************************** */
// The functions in the standard library are slower; so, these are implemented
template<typename T>
DEVICE inline T Square(const T& x) {
    return x * x;
}
template<typename T>
DEVICE inline T Cube(const T& x) {
    return x * x * x;
}
template<typename T>
DEVICE inline int Floor(const T& x) {
    const int i = static_cast<int>(x);
    return i - (x < i);
}
template<typename T>
DEVICE inline int Ceil(const T& x) {
    const int i = static_cast<int>(x);
    return i + (x > i);
}
template<typename T>
DEVICE inline int Round(const T& x) {
    return static_cast<int>(x + (x >= 0 ? 0.5 : -0.5));
}
/* *************************************************************** */
} // namespace NiftyReg
/* *************************************************************** */
template <class T>
void reg_LUdecomposition(T *inputMatrix,
                         size_t dim,
                         size_t *index);
/* *************************************************************** */
template <class T>
void reg_matrixMultiply(T *mat1,
                        T *mat2,
                        size_t *dim1,
                        size_t *dim2,
                        T * &res);
/* *************************************************************** */
template <class T>
void reg_matrixInvertMultiply(T *mat,
                              size_t dim,
                              size_t *index,
                              T *vec);
/* *************************************************************** */
template<class T>
T* reg_matrix1DAllocate(size_t arraySize);
/* *************************************************************** */
template<class T>
T* reg_matrix1DAllocateAndInitToZero(size_t arraySize);
/* *************************************************************** */
template<class T>
void reg_matrix1DDeallocate(T* mat);
/* *************************************************************** */
template<class T>
T** reg_matrix2DAllocate(size_t arraySizeX, size_t arraySizeY);
/* *************************************************************** */
template<class T>
T** reg_matrix2DAllocateAndInitToZero(size_t arraySizeX, size_t arraySizeY);
/* *************************************************************** */
template<class T>
void reg_matrix2DDeallocate(size_t arraySizeX, T** mat);
/* *************************************************************** */
template<class T>
T** reg_matrix2DTranspose(T** mat, size_t arraySizeX, size_t arraySizeY);
/* *************************************************************** */
template<class T>
T** reg_matrix2DMultiply(T** mat1, size_t mat1X, size_t mat1Y, T** mat2, size_t mat2X, size_t mat2Y, bool transposeMat2);
template<class T>
void reg_matrix2DMultiply(T** mat1, size_t mat1X, size_t mat1Y, T** mat2, size_t mat2X, size_t mat2Y, T** res, bool transposeMat2);
/* *************************************************************** */
template<class T>
T* reg_matrix2DVectorMultiply(T** mat, size_t m, size_t n, T* vect);
template<class T>
void reg_matrix2DVectorMultiply(T** mat, size_t m, size_t n, T* vect, T* res);
/* *************************************************************** */
/** @brief Add two 3-by-3 matrices
*/
mat33 reg_mat33_add(mat33 const* A, mat33 const* B);
mat33 operator+(mat33 A, mat33 B);
/* *************************************************************** */
/** @brief Multiply two 3-by-3 matrices
*/
mat33 reg_mat33_mul(mat33 const* A,
    mat33 const* B);
mat33 operator*(mat33 A,
    mat33 B);
/* *************************************************************** */
//The mat33 represent a 3x3 matrix
void reg_mat33_mul(mat44 const* mat, float const* in, float *out);
void reg_mat33_mul(mat33 const* mat, float const* in, float *out);
/* *************************************************************** */
/** @brief Subtract two 3-by-3 matrices
*/
mat33 reg_mat33_minus(mat33 const* A, mat33 const* B);
mat33 operator-(mat33 A, mat33 B);
/* *************************************************************** */
/** @brief Transpose a 3-by-3 matrix
*/
mat33 reg_mat33_trans(mat33 A);
/* *************************************************************** */
/** @brief Diagonalize a 3-by-3 matrix
*/
void reg_mat33_diagonalize(mat33 const* A, mat33 * Q, mat33 * D);
/* *************************************************************** */
/** @brief Set up a 3-by-3 matrix with an identity
*/
void reg_mat33_eye(mat33 *mat);
/* *************************************************************** */
/** @brief Compute the determinant of a 3-by-3 matrix
*/
template<class T> T reg_mat33_det(mat33 const* A);
/* *************************************************************** */
/** @brief Compute the determinant of a 3-by-3 matrix
*/
void reg_mat33_to_nan(mat33 *A);
/* *************************************************************** */
/** @brief Transform a mat44 to a mat33 matrix
*/
mat33 reg_mat44_to_mat33(mat44 const* A);
void reg_heapSort(float *array_tmp, int *index_tmp, int blockNum);
/* *************************************************************** */
template <class T>
void reg_heapSort(T *array_tmp,int blockNum);
/* *************************************************************** */
bool operator==(mat44 A,mat44 B);
/* *************************************************************** */
bool operator!=(mat44 A,mat44 B);
/* *************************************************************** */
/** @brief Multiply two 4-by-4 matrices
 */
mat44 reg_mat44_mul(mat44 const* A,
                    mat44 const* B);
mat44 operator*(mat44 A,
                mat44 B);
/* *************************************************************** */
/** @brief Multiply a vector with a 4-by-4 matrix
 */
void reg_mat44_mul(mat44 const* mat,
                   float const* in,
                   float *out);

void reg_mat44_mul(mat44 const* mat,
                   double const* in,
                   double *out);
/* *************************************************************** */
/** @brief Multiply a 4-by-4 matrix with a scalar
 */
mat44 reg_mat44_mul(mat44 const* mat,
                    double scalar);
/* *************************************************************** */
/** @brief Add two 4-by-4 matrices
 */
mat44 reg_mat44_add(mat44 const* A, mat44 const* B);
mat44 operator+(mat44 A,mat44 B);
/* *************************************************************** */
/** @brief Subtract two 4-by-4 matrices
 */
mat44 reg_mat44_minus(mat44 const* A, mat44 const* B);
mat44 operator-(mat44 A,mat44 B);
/* *************************************************************** */
/** @brief Set up a 4-by-4 matrix with an identity
 */
void reg_mat44_eye(mat44 *mat);
/* *************************************************************** */
/** @brief Compute the determinant of a 4-by-4 matrix
 */
template<class T> T reg_mat44_det(mat44 const* A);
/* *************************************************************** */
float reg_mat44_norm_inf(mat44 const* mat);
/* *************************************************************** */
/** @brief Display a mat44 matrix
 */
void reg_mat44_disp(const mat44& mat, const std::string& title);
/* *************************************************************** */
/** @brief Display a mat33 matrix
 */
void reg_mat33_disp(const mat33& mat, const std::string& title);
/* *************************************************************** */
double get_square_distance3D(float * first_point3D, float * second_point3D);
/* *************************************************************** */
double get_square_distance2D(float * first_point2D, float * second_point2D);
/* *************************************************************** */
