/**
 * @file _reg_splineBasis.h
 * @brief Library that contains local deformation related functions
 * @author Marc Modat
 * @date 23/12/2015
 *
 * Centre for Medical Image Computing (CMIC)
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_SPLINE_H
#define _REG_SPLINE_H

#include "_reg_tools.h"


extern "C++" template<class DTYPE>
void get_BSplineBasisValues(DTYPE basis,
                            DTYPE *values);//,
//                            int order=3);  // default is cubic B-Spline
extern "C++" template<class DTYPE>
void get_BSplineBasisValues(DTYPE basis,
                            DTYPE *values,
                            DTYPE *first);
extern "C++" template<class DTYPE>
void get_BSplineBasisValues(DTYPE basis,
                            DTYPE *values,
                            DTYPE *first,
                            DTYPE *second);
// add the order as a parameter of existing functions instead
/** @brief Refactoring of get_BSplineBasisValues that works for any B-spline order.
 * Compute the weights for a 1D B-spline of a given order.
 * @param coord centered coordinate
 * @param values output array of the corresponding weights
 * @param order order of the B-spline basis to use (default is 3: cubic B-spline)
 */
extern "C++" template<class DTYPE>
void get_BSplineBasisWeights(DTYPE coord,
                             DTYPE *values,
                             int order=3);

extern "C++" template<class DTYPE>
void get_BSplineDerivativeBasisWeights(DTYPE coord,
                             DTYPE *values,
                             int order=3);


extern "C++" template<class DTYPE>
void get_BSplineBasisValue(DTYPE basis,
                           int index,
                           DTYPE &value);
extern "C++" template<class DTYPE>
void get_BSplineBasisValue(DTYPE basis,
                           int index,
                           DTYPE &value,
                           DTYPE &first);
extern "C++" template<class DTYPE>
void get_BSplineBasisValue(DTYPE basis,
                           int index,
                           DTYPE &value,
                           DTYPE &first,
                           DTYPE &second);

extern "C++" template <class DTYPE>
void set_first_order_basis_values(DTYPE *basisX,
                                  DTYPE *basisY);

extern "C++" template <class DTYPE>
void set_first_order_basis_values(DTYPE *basisX,
                                  DTYPE *basisY,
                                  DTYPE *basisZ);

extern "C++" template <class DTYPE>
void set_second_order_bspline_basis_values(DTYPE *basisXX,
                                           DTYPE *basisYY,
                                           DTYPE *basisXY);
extern "C++" template <class DTYPE>
void set_second_order_bspline_basis_values(DTYPE *basisXX,
                                           DTYPE *basisYY,
                                           DTYPE *basisZZ,
                                           DTYPE *basisXY,
                                           DTYPE *basisYZ,
                                           DTYPE *basisXZ);


extern "C++" template<class DTYPE>
void get_SplineBasisValues(DTYPE basis,
                           DTYPE *values);
extern "C++" template<class DTYPE>
void get_SplineBasisValues(DTYPE basis,
                           DTYPE *values,
                           DTYPE *first);
extern "C++" template<class DTYPE>
void get_SplineBasisValues(DTYPE basis,
                           DTYPE *values,
                           DTYPE *first,
                           DTYPE *second);

extern "C++" template <class DTYPE>
void get_SlidedValues(DTYPE &defX,
                      DTYPE &defY,
                      int X,
                      int Y,
                      DTYPE *defPtrX,
                      DTYPE *defPtrY,
                      mat44 *df_voxel2Real,
                      int *dim,
                      bool displacement);
extern "C++" template <class DTYPE>
void get_SlidedValues(DTYPE &defX,
                      DTYPE &defY,
                      DTYPE &defZ,
                      int X,
                      int Y,
                      int Z,
                      DTYPE *defPtrX,
                      DTYPE *defPtrY,
                      DTYPE *defPtrZ,
                      mat44 *df_voxel2Real,
                      int *dim,
                      bool displacement);


extern "C++" template <class DTYPE>
void get_GridValues(int startX,
                    int startY,
                    nifti_image *splineControlPoint,
                    DTYPE *splineX,
                    DTYPE *splineY,
                    DTYPE *dispX,
                    DTYPE *dispY,
                    bool approx,
                    bool displacement);
extern "C++" template <class DTYPE>
void get_GridValues(int startX,
                    int startY,
                    int startZ,
                    nifti_image *splineControlPoint,
                    DTYPE *splineX,
                    DTYPE *splineY,
                    DTYPE *splineZ,
                    DTYPE *dispX,
                    DTYPE *dispY,
                    DTYPE *dispZ,
                    bool approx,
                    bool displacement);

#endif
