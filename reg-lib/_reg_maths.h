/**
 * @file _reg_maths.h
 * @brief Library that contains small math routines
 * @author Marc Modat
 * @date 25/03/2009
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */
#ifndef _REG_MATHS_H
#define _REG_MATHS_H

#include <stdio.h>
#include <math.h>
#include <iostream>
#include "nifti1_io.h"

typedef enum{
    DEF_FIELD,
    DISP_FIELD,
    SPLINE_GRID,
    DEF_VEL_FIELD,
    DISP_VEL_FIELD,
    SPLINE_VEL_GRID
}NREG_TRANS_TYPE;

/* *************************************************************** */
#define reg_pow2(a) ((a)*(a))
#define reg_ceil(a) (ceil(a))
#define reg_round(a) ((a)>0.0 ?(int)((a)+0.5):(int)((a)-0.5))
#ifdef _WIN32
    #define reg_floor(a) ((a)>0?(int)(a):(int)((a)-1))
#else
    #define reg_floor(a) ((a)>=0?(int)(a):floor(a))
#endif
/* *************************************************************** */
#define reg_exit(val){ \
	fprintf(stderr,"[NiftyReg] Exit here. File: %s. Line: %i\n",__FILE__, __LINE__); \
	exit(val); \
}
/* *************************************************************** */
#if defined(_WIN32) && !defined(__CYGWIN__)
    #include <limits>
    #include <float.h>
    #include <time.h>
    #ifndef M_PI
        #define M_PI 3.14159265358979323846
    #endif
    #ifndef isnan
        #define isnan(_X) _isnan(_X)
    #endif
    #ifndef strtof
        #define strtof(_s, _t) (float) strtod(_s, _t)
    #endif
    template<class PrecisionType> inline int round(PrecisionType x){ return int(x > 0.0 ? (x + 0.5) : (x - 0.5));}
    template<typename T>inline bool isinf(T value){ return std::numeric_limits<T>::has_infinity && value == std::numeric_limits<T>::infinity();}
    inline int fabs(int _x){ return (int)fabs((float)(_x)); }
#endif // If on windows...

/* *************************************************************** */
typedef struct{
    double m[4][4];
} reg_mat44d;
/* *************************************************************** */
extern "C++" template <class T>
void reg_LUdecomposition(T *inputMatrix,
                         size_t dim,
                         size_t *index);
/* *************************************************************** */
extern "C++" template <class T>
void reg_matrixMultiply(T *mat1,
                        T *mat2,
                        int *dim1,
                        int *dim2,
                        T *&res);
/* *************************************************************** */
extern "C++" template <class T>
void reg_matrixInverse(T *mat,
                       int *dim);
/* *************************************************************** */
extern "C++" template <class T>
void reg_matrixInvertMultiply(T *mat,
                              size_t dim,
                              size_t *index,
                              T *vec);
/* *************************************************************** */
extern "C++"
void reg_heapSort(float *array_tmp, int *index_tmp, int blockNum);
/* *************************************************************** */
extern "C++"
void reg_heapSort(float *array_tmp,int blockNum);
/* *************************************************************** */
extern "C++"
reg_mat44d reg_mat44_singleToDouble(mat44 const *mat);
/* *************************************************************** */
extern "C++"
mat44 reg_mat44_doubleToSingle(reg_mat44d const *mat);
/* *************************************************************** */
/** @brief here
 */
extern "C++"
mat33 reg_mat44_to_mat33(mat44 const* A);
/* *************************************************************** */
/** @brief Multipy two 4-by-4 matrices
 */
extern "C++" template <class MTYPE>
MTYPE reg_mat44_mul(MTYPE const* A,MTYPE const* B);
mat44 operator*(mat44 A,mat44 B);
reg_mat44d operator*(reg_mat44d A,reg_mat44d B);
/* *************************************************************** */
/** @brief Multipy a vector with a 4-by-4 matrix
 */
extern "C++" template <class DTYPE, class MTYPE>
void reg_mat44_mul(MTYPE const* mat,
                   DTYPE const* in,
                   DTYPE *out);
/* *************************************************************** */
/** @brief Multipy a 4-by-4 matrix with a scalar
 */
extern "C++" template <class MTYPE>
MTYPE reg_mat44_mul(MTYPE const* mat,
                    double scalar);
/* *************************************************************** */
/** @brief Add two 4-by-4 matrices
 */
extern "C++" template <class MTYPE>
MTYPE reg_mat44_add(MTYPE const* A, MTYPE const* B);
mat44 operator+(mat44 A,mat44 B);
reg_mat44d operator+(reg_mat44d A,reg_mat44d B);
/* *************************************************************** */
/** @brief Substract two 4-by-4 matrices
 */
extern "C++" template <class MTYPE>
MTYPE reg_mat44_minus(MTYPE const* A, MTYPE const* B);
mat44 operator-(mat44 A,mat44 B);
reg_mat44d operator-(reg_mat44d A,reg_mat44d B);
/* *************************************************************** */
extern "C++" template <class MTYPE>
MTYPE reg_mat44_to_MTYPE(mat44 *M);
/* *************************************************************** */
extern "C++" template <class MTYPE>
mat44 reg_MTYPE_to_mat44(MTYPE *M);
/* *************************************************************** */
/** @brief Set up a 3-by-3 matrix with an identity
 */
extern "C++"
void reg_mat33_eye (mat33 *mat);
/* *************************************************************** */
/** @brief Set up a 4-by-4 matrix with an identity
 */
extern "C++" template <class MTYPE>
void reg_mat44_eye(MTYPE *mat);
/* *************************************************************** */
/** @brief Compute the determinant of a 4-by-4 matrix
 */
extern "C++" template <class MTYPE>
float reg_mat44_det(MTYPE const* A);
/* *************************************************************** */
/** @brief Compute the inverse of a 4-by-4 matrix
 */
extern "C++" template <class MTYPE>
MTYPE reg_mat44_inv(MTYPE const* A);
/* *************************************************************** */
extern "C++" template <class MTYPE>
float reg_mat44_norm_inf(MTYPE const* mat);
/* *************************************************************** */
/** @brief Compute the square root of a 4-by-4 matrix
 */
extern "C++" template <class MTYPE>
MTYPE reg_mat44_sqrt(MTYPE const* mat);
/* *************************************************************** */
/** @brief Compute the exp of a 4-by-4 matrix
 */
extern "C++" template <class MTYPE>
MTYPE reg_mat44_expm(MTYPE const* mat, int maxit=6);
/* *************************************************************** */
/** @brief Compute the log of a 4-by-4 matrix
 */
extern "C++" template <class MTYPE>
MTYPE reg_mat44_logm(MTYPE const* mat);
/* *************************************************************** */
/** @brief Compute the average of two matrices using a log-euclidean
 * framework
 */
extern "C++" template <class MTYPE>
MTYPE reg_mat44_avg2(MTYPE const* A, MTYPE const* b);
/* *************************************************************** */
/** @brief Display a mat44 matrix
 */
extern "C++" template <class MTYPE>
void reg_mat44_disp(MTYPE *mat,
                    char * title);
/* *************************************************************** */
/** @brief Display a mat33 matrix
 */
extern "C++"
void reg_mat33_disp(mat33 *mat,
                    char * title);
/* *************************************************************** */
/** @brief Compute the transformation matrix to diagonalise the input matrix
 */
extern "C++"
void reg_getReorientationMatrix(nifti_image *splineControlPoint,
                                mat33 *reorient);
/* *************************************************************** */
extern "C++" template <class T>
void svd(T ** in, size_t m, size_t n, T * w, T ** v);
/* *************************************************************** */
#endif // _REG_MATHS_H
