#ifndef _REG_MATHS_H
#define _REG_MATHS_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "nifti1_io.h"

#define POW2(a) ((a)*(a))

/* *************************************************************** */
extern "C++"
void reg_LUdecomposition(float *inputMatrix,
                         int dim,
                         int *index);
/* *************************************************************** */
extern "C++"
void reg_matrixInvertMultiply(float *mat,
                              int dim,
                              int *index,
                              float *vec);
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
#endif // _REG_MATHS_H
