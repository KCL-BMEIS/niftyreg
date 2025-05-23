/**
 * @file Maths.hpp
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
template<typename RetT>
DEVICE inline RetT Floor(const float x) {
    const int i = static_cast<int>(x);
    return static_cast<RetT>(i - (x < i));
}
template<typename RetT>
DEVICE inline RetT Floor(const double x) {
    const int64_t i = static_cast<int64_t>(x);
    return static_cast<RetT>(i - (x < i));
}
template<typename RetT>
DEVICE inline RetT Ceil(const float x) {
    const int i = static_cast<int>(x);
    return static_cast<RetT>(i + (x > i));
}
template<typename RetT>
DEVICE inline RetT Ceil(const double x) {
    const int64_t i = static_cast<int64_t>(x);
    return static_cast<RetT>(i + (x > i));
}
template<typename RetT>
DEVICE inline RetT Round(const float x) {
    return static_cast<RetT>(static_cast<int>(x + (x >= 0 ? 0.5f : -0.5f)));
}
template<typename RetT>
DEVICE inline RetT Round(const double x) {
    return static_cast<RetT>(static_cast<int64_t>(x + (x >= 0 ? 0.5 : -0.5)));
}
/* *************************************************************** */
DEVICE inline void Divide(const int num, const int denom, int& quot, int& rem) {
    // This will be optimised by the compiler into a single div instruction
    quot = num / denom;
    rem = num % denom;
}
/* *************************************************************** */
template<class T>
DEVICE inline T* Matrix1dAlloc(const size_t arraySize) {
    return static_cast<T*>(malloc(arraySize * sizeof(T)));
}
/* *************************************************************** */
template<class T>
DEVICE inline void Matrix1dDealloc(T *mat) {
    free(mat);
}
/* *************************************************************** */
template<class T>
DEVICE inline T** Matrix2dAlloc(const size_t arraySizeX, const size_t arraySizeY) {
    T **res;
    res = static_cast<T**>(malloc(arraySizeX * sizeof(T*)));
    for (size_t i = 0; i < arraySizeX; i++)
        res[i] = static_cast<T*>(malloc(arraySizeY * sizeof(T)));
    return res;
}
/* *************************************************************** */
template<class T>
DEVICE inline void Matrix2dDealloc(const size_t arraySizeX, T **mat) {
    for (size_t i = 0; i < arraySizeX; i++)
        free(mat[i]);
    free(mat);
}
/* *************************************************************** */
template<class T>
DEVICE inline T** Matrix2dTranspose(T **mat, const size_t arraySizeX, const size_t arraySizeY) {
    T **res;
    res = static_cast<T**>(malloc(arraySizeY * sizeof(T*)));
    for (size_t i = 0; i < arraySizeY; i++)
        res[i] = static_cast<T*>(malloc(arraySizeX * sizeof(T)));
    for (size_t i = 0; i < arraySizeX; i++)
        for (size_t j = 0; j < arraySizeY; j++)
            res[j][i] = mat[i][j];
    return res;
}
/* *************************************************************** */
template<class T>
T** Matrix2dMultiply(T **mat1, const size_t mat1X, const size_t mat1Y, T **mat2, const size_t mat2X, const size_t mat2Y, const bool transposeMat2);
template<class T>
void Matrix2dMultiply(T **mat1, const size_t mat1X, const size_t mat1Y, T **mat2, const size_t mat2X, const size_t mat2Y, T **res, const bool transposeMat2);
/* *************************************************************** */
template<class T>
T* Matrix2dVectorMultiply(T **mat, const size_t m, const size_t n, T *vect);
template<class T>
void Matrix2dVectorMultiply(T **mat, const size_t m, const size_t n, T *vect, T *res);
/* *************************************************************** */
/// @brief Subtract two 3-by-3 matrices
DEVICE inline mat33 operator-(const mat33& A, const mat33& B) {
    mat33 R;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R.m[i][j] = static_cast<float>(static_cast<double>(A.m[i][j]) - static_cast<double>(B.m[i][j]));
    return R;
}
/* *************************************************************** */
/// @brief Multiply two 3-by-3 matrices
DEVICE inline mat33 operator*(const mat33& A, const mat33& B) {
    mat33 R;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R.m[i][j] = static_cast<float>(static_cast<double>(A.m[i][0]) * static_cast<double>(B.m[0][j]) +
                                           static_cast<double>(A.m[i][1]) * static_cast<double>(B.m[1][j]) +
                                           static_cast<double>(A.m[i][2]) * static_cast<double>(B.m[2][j]));
    return R;
}
/* *************************************************************** */
/// @brief Multiply a vector with a 3-by-3 matrix
DEVICE inline void Mat33Mul(const mat33& mat, const float (&in)[2], float (&out)[2]) {
    out[0] = static_cast<float>(static_cast<double>(in[0]) * static_cast<double>(mat.m[0][0]) +
                                static_cast<double>(in[1]) * static_cast<double>(mat.m[0][1]) +
                                static_cast<double>(mat.m[0][2]));
    out[1] = static_cast<float>(static_cast<double>(in[0]) * static_cast<double>(mat.m[1][0]) +
                                static_cast<double>(in[1]) * static_cast<double>(mat.m[1][1]) +
                                static_cast<double>(mat.m[1][2]));
}
/* *************************************************************** */
/// @brief Multiply a vector with a 3-by-3 matrix
DEVICE inline void Mat33Mul(const mat44& mat, const float (&in)[2], float (&out)[2]) {
    out[0] = static_cast<float>(static_cast<double>(in[0]) * static_cast<double>(mat.m[0][0]) +
                                static_cast<double>(in[1]) * static_cast<double>(mat.m[0][1]) +
                                static_cast<double>(mat.m[0][3]));
    out[1] = static_cast<float>(static_cast<double>(in[0]) * static_cast<double>(mat.m[1][0]) +
                                static_cast<double>(in[1]) * static_cast<double>(mat.m[1][1]) +
                                static_cast<double>(mat.m[1][3]));
}
/* *************************************************************** */
/// @brief Multiply a scalar with a 3-by-3 matrix multiplied by a vector
template<bool is3d>
DEVICE inline void Mat33Mul(const mat33& mat, const float (&in)[3], const float weight, float (&out)[3]) {
    out[0] = weight * (mat.m[0][0] * in[0] + mat.m[1][0] * in[1] + mat.m[2][0] * in[2]);
    out[1] = weight * (mat.m[0][1] * in[0] + mat.m[1][1] * in[1] + mat.m[2][1] * in[2]);
    if constexpr (is3d)
        out[2] = weight * (mat.m[0][2] * in[0] + mat.m[1][2] * in[1] + mat.m[2][2] * in[2]);
}
/* *************************************************************** */
/// @brief Transpose a 3-by-3 matrix
DEVICE inline mat33 Mat33Trans(const mat33 A) {
    mat33 R;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R.m[j][i] = A.m[i][j];
    return R;
}
/* *************************************************************** */
/// @brief Diagonalize a 3-by-3 matrix
void Mat33Diagonalize(const mat33 *A, mat33 *Q, mat33 *D);
/* *************************************************************** */
/// @brief Set up a 3-by-3 matrix with an identity
DEVICE inline void Mat33Eye(mat33 *mat) {
    mat->m[0][0] = 1.f;
    mat->m[0][1] = mat->m[0][2] = 0.f;
    mat->m[1][1] = 1.f;
    mat->m[1][0] = mat->m[1][2] = 0.f;
    mat->m[2][2] = 1.f;
    mat->m[2][0] = mat->m[2][1] = 0.f;
}
/* *************************************************************** */
DEVICE inline void Mat33ToNan(mat33 *A) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            A->m[i][j] = std::numeric_limits<float>::quiet_NaN();
}
/* *************************************************************** */
/// @brief Transform a mat44 to a mat33 matrix
DEVICE inline mat33 Mat44ToMat33(const mat44 *A) {
    mat33 out;
    out.m[0][0] = A->m[0][0];
    out.m[0][1] = A->m[0][1];
    out.m[0][2] = A->m[0][2];
    out.m[1][0] = A->m[1][0];
    out.m[1][1] = A->m[1][1];
    out.m[1][2] = A->m[1][2];
    out.m[2][0] = A->m[2][0];
    out.m[2][1] = A->m[2][1];
    out.m[2][2] = A->m[2][2];
    return out;
}
/* *************************************************************** */
template<class T>
void HeapSort(T *array_tmp, int blockNum);
void HeapSort(float *array_tmp, int *index_tmp, int blockNum);
/* *************************************************************** */
DEVICE inline bool operator==(const mat44& A, const mat44& B) {
    for (char i = 0; i < 4; ++i)
        for (char j = 0; j < 4; ++j)
            if (A.m[i][j] != B.m[i][j])
                return false;
    return true;
}
/* *************************************************************** */
DEVICE inline bool operator!=(const mat44& A, const mat44& B) {
    return !(A == B);
}
/* *************************************************************** */
/// @brief Multiply two 4-by-4 matrices
DEVICE inline mat44 operator*(const mat44& A, const mat44& B) {
    mat44 R;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            R.m[i][j] = static_cast<float>(static_cast<double>(A.m[i][0]) * static_cast<double>(B.m[0][j]) +
                                           static_cast<double>(A.m[i][1]) * static_cast<double>(B.m[1][j]) +
                                           static_cast<double>(A.m[i][2]) * static_cast<double>(B.m[2][j]) +
                                           static_cast<double>(A.m[i][3]) * static_cast<double>(B.m[3][j]));
    return R;
}
/* *************************************************************** */
/// @brief Multiply a 4-by-4 matrix with a scalar
DEVICE inline mat44 operator*(const mat44& mat, const double scalar) {
    mat44 out;
    out.m[0][0] = mat.m[0][0] * scalar;
    out.m[0][1] = mat.m[0][1] * scalar;
    out.m[0][2] = mat.m[0][2] * scalar;
    out.m[0][3] = mat.m[0][3] * scalar;
    out.m[1][0] = mat.m[1][0] * scalar;
    out.m[1][1] = mat.m[1][1] * scalar;
    out.m[1][2] = mat.m[1][2] * scalar;
    out.m[1][3] = mat.m[1][3] * scalar;
    out.m[2][0] = mat.m[2][0] * scalar;
    out.m[2][1] = mat.m[2][1] * scalar;
    out.m[2][2] = mat.m[2][2] * scalar;
    out.m[2][3] = mat.m[2][3] * scalar;
    out.m[3][0] = mat.m[3][0] * scalar;
    out.m[3][1] = mat.m[3][1] * scalar;
    out.m[3][2] = mat.m[3][2] * scalar;
    out.m[3][3] = mat.m[3][3] * scalar;
    return out;
}
/* *************************************************************** */
/// @brief Multiply a vector with a 4-by-4 matrix
template<class T, bool is3d=true>
DEVICE inline void Mat44Mul(const mat44& mat, const T (&in)[3], T (&out)[3]) {
    out[0] = static_cast<T>(static_cast<double>(mat.m[0][0]) * static_cast<double>(in[0]) +
                            static_cast<double>(mat.m[0][1]) * static_cast<double>(in[1]) +
                            static_cast<double>(mat.m[0][2]) * static_cast<double>(in[2]) +
                            static_cast<double>(mat.m[0][3]));
    out[1] = static_cast<T>(static_cast<double>(mat.m[1][0]) * static_cast<double>(in[0]) +
                            static_cast<double>(mat.m[1][1]) * static_cast<double>(in[1]) +
                            static_cast<double>(mat.m[1][2]) * static_cast<double>(in[2]) +
                            static_cast<double>(mat.m[1][3]));
    if constexpr (is3d)
        out[2] = static_cast<T>(static_cast<double>(mat.m[2][0]) * static_cast<double>(in[0]) +
                                static_cast<double>(mat.m[2][1]) * static_cast<double>(in[1]) +
                                static_cast<double>(mat.m[2][2]) * static_cast<double>(in[2]) +
                                static_cast<double>(mat.m[2][3]));
}
/* *************************************************************** */
/// @brief Add two 4-by-4 matrices
DEVICE inline mat44 operator+(const mat44& A, const mat44& B) {
    mat44 R;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            R.m[i][j] = static_cast<float>(static_cast<double>(A.m[i][j]) + static_cast<double>(B.m[i][j]));
    return R;
}
/* *************************************************************** */
/// @brief Subtract two 4-by-4 matrices
DEVICE inline mat44 operator-(const mat44& A, const mat44& B) {
    mat44 R;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            R.m[i][j] = static_cast<float>(static_cast<double>(A.m[i][j]) - static_cast<double>(B.m[i][j]));
    return R;
}
/* *************************************************************** */
/// @brief Set up a 4-by-4 matrix with an identity
DEVICE inline void Mat44Eye(mat44 *mat) {
    mat->m[0][0] = 1.f;
    mat->m[0][1] = mat->m[0][2] = mat->m[0][3] = 0.f;
    mat->m[1][1] = 1.f;
    mat->m[1][0] = mat->m[1][2] = mat->m[1][3] = 0.f;
    mat->m[2][2] = 1.f;
    mat->m[2][0] = mat->m[2][1] = mat->m[2][3] = 0.f;
    mat->m[3][3] = 1.f;
    mat->m[3][0] = mat->m[3][1] = mat->m[3][2] = 0.f;
}
/* *************************************************************** */
/// @brief Compute the determinant of a 4-by-4 matrix
template<class T>
T Mat44Det(const mat44 *A);
/* *************************************************************** */
/// @brief Display a mat44 matrix
void Mat44Disp(const mat44& mat, const std::string& title);
/* *************************************************************** */
//is it square distance or just distance?
DEVICE inline double SquareDistance2d(const float *first_point2D, const float *second_point2D) {
    return sqrt(Square(first_point2D[0] - second_point2D[0]) +
                Square(first_point2D[1] - second_point2D[1]));
}
/* *************************************************************** */
//is it square distance or just distance?
DEVICE inline double SquareDistance3d(const float *first_point3D, const float *second_point3D) {
    return sqrt(Square(first_point3D[0] - second_point3D[0]) +
                Square(first_point3D[1] - second_point3D[1]) +
                Square(first_point3D[2] - second_point3D[2]));
}
/* *************************************************************** */
template<class T>
void Svd(T **in, const size_t m, const size_t n, T *w, T **v);
/* *************************************************************** */
template<class T>
T Matrix2dDet(T **mat, const size_t m, const size_t n);
/* *************************************************************** */
/// @brief Compute the log of a 3-by-3 matrix
void Mat33Expm(mat33 *tensorIn);
/* *************************************************************** */
/// @brief Compute the exp of a 4-by-4 matrix
mat44 Mat44Expm(const mat44 *mat);
/* *************************************************************** */
/// @brief Compute the log of a 3-by-3 matrix
void Mat33Logm(mat33 *tensorIn);
/* *************************************************************** */
/// @brief Compute the log of a 4-by-4 matrix
mat44 Mat44Logm(const mat44 *mat);
/* *************************************************************** */
/// @brief Compute the average of two matrices using a log-euclidean framework
mat44 Mat44Avg2(const mat44 *A, const mat44 *b);
/* *************************************************************** */
} // namespace NiftyReg
/* *************************************************************** */
