#ifndef _REG_MATHS_H
#define _REG_MATHS_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "nifti1_io.h"

#define POW2(a) ((a)*(a))

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
mat44 reg_mat44_mul(mat44 *A,
                     mat44 *B);
/* *************************************************************** */
extern "C++" template <class DTYPE>
void reg_mat44_mul(mat44 *mat,
                   DTYPE *in,
                   DTYPE *out);
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
