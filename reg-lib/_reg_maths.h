#ifndef _REG_MATHS_H
#define _REG_MATHS_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include "nifti1_io.h"

/* *************************************************************** */
#define reg_pow2(a) ((a)*(a))
#define reg_ceil(a) (ceil(a))
#define reg_round(a) ((a)>0.0 ?(int)((a)+0.5):(int)((a)-0.5))
#ifdef _WINDOWS
    #define reg_floor(a) ((a)>0?(int)(a):(int)((a)-1))
#else
    #define reg_floor(a) ((a)>=0?(int)(a):floor(a))
#endif
/* *************************************************************** */

#if defined(_WIN32) && !defined(__CYGWIN__)
    #include <float.h>
    #include <time.h>
    #ifndef M_PI
        #define M_PI (3.14159265358979323846)
    #endif
    #ifndef isnan(_X)
        #define isnan(_X) _isnan(_X)
    #endif
    #ifndef strtof(_s, _t)
        #define strtof(_s, _t) (float) strtod(_s, _t)
    #endif
    template<class PrecisionType> inline int round(PrecisionType x) { return int(x > 0.0 ? (x + 0.5) : (x - 0.5)); }
    template<typename T> inline bool isinf(T value) { return std::numeric_limits<T>::has_infinity && value == std::numeric_limits<T>::infinity(); }
    inline int fabs(int _x) { return (int)fabs((float)(_x)); }
#endif // If on windows...

/* *************************************************************** */
extern "C++" template <class T>
void reg_LUdecomposition(T *inputMatrix,
                         size_t dim,
                         size_t *index);
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
mat44 reg_mat44_mul(mat44 const* A,
                     mat44 const* B);
/* *************************************************************** */
extern "C++" template <class DTYPE>
void reg_mat44_mul(mat44 const* mat,
                   DTYPE const* in,
                   DTYPE *out);
/* *************************************************************** */
extern "C++"
mat44 reg_mat44_mul(mat44 const* mat,
                    double scalar);
/* *************************************************************** */
extern "C++"
mat44 reg_mat44_add(mat44 const* A, mat44 const* B);
/* *************************************************************** */
extern "C++"
mat44 reg_mat44_minus(mat44 const* A, mat44 const* B);
/* *************************************************************** */
extern "C++"
void reg_mat33_eye (mat33 *mat);
/* *************************************************************** */
extern "C++"
void reg_mat44_eye (mat44 *mat);
/* *************************************************************** */
extern "C++"
float reg_mat44_det(mat44 const* A);
/* *************************************************************** */
extern "C++"
mat44 reg_mat44_inv(mat44 const* A);
/* *************************************************************** */
extern "C++"
double reg_mat44_norm_inf(mat44 const* mat);
/* *************************************************************** */
extern "C++"
mat44 reg_mat44_sqrt(mat44 const* mat);
/* *************************************************************** */
extern "C++"
mat44 reg_mat44_expm(mat44 const* mat, int maxit=6);
/* *************************************************************** */
extern "C++"
mat44 reg_mat44_logm(mat44 const* mat);
/* *************************************************************** */
extern "C++"
mat44 reg_mat44_avg2(mat44 const* A, mat44 const* b);
/* *************************************************************** */
extern "C++"
void reg_mat44_disp(mat44 *mat,
                    char * title);
/* *************************************************************** */
extern "C++"
void reg_mat33_disp(mat33 *mat,
                    char * title);
/* *************************************************************** */
/** getReorientationMatrix
 * Compute the transformation matrix to diagonalise the input matrix
 */
extern "C++"
void reg_getReorientationMatrix(nifti_image *splineControlPoint,
                                mat33 *desorient,
                                mat33 *reorient);
/* *************************************************************** */
extern "C++" template <class T>
void svd(T ** in, size_t m, size_t n, T * w, T ** v);
/* *************************************************************** */
#endif // _REG_MATHS_H
