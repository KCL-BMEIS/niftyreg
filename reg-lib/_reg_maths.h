#ifndef _REG_MATHS_H
#define _REG_MATHS_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* *************************************************************** */
/* *************************************************************** */
extern "C++"
void reg_LUdecomposition(float *inputMatrix,
                         int dim,
                         int *index);
/* *************************************************************** */
/* *************************************************************** */
extern "C++"
void reg_matrixInvertMultiply(float *mat,
                              int dim,
                              int *index,
                              float *vec);
/* *************************************************************** */
/* *************************************************************** */
extern "C++"
void reg_heapSort(float *array_tmp, int *index_tmp, int blockNum);
/* *************************************************************** */
/* *************************************************************** */

#endif // _REG_MATHS_H
