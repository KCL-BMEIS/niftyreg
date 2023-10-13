/**
 * @file _reg_splineBasis.h
 * @brief Library that contains local deformation related functions
 * @author Marc Modat
 * @date 23/12/2015
 *
 * Copyright (c) 2015-2018, University College London
 * Copyright (c) 2018, NiftyReg Developers.
 * All rights reserved.
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#pragma once

#include "_reg_tools.h"


template<class DataType>
void get_BSplineBasisValues(DataType basis,
                            DataType *values);
template<class DataType>
void get_BSplineBasisValues(DataType basis,
                            DataType *values,
                            DataType *first);
template<class DataType>
void get_BSplineBasisValues(DataType basis,
                            DataType *values,
                            DataType *first,
                            DataType *second);


template<class DataType>
void get_BSplineBasisValue(DataType basis,
                           int index,
                           DataType &value);
template<class DataType>
void get_BSplineBasisValue(DataType basis,
                           int index,
                           DataType &value,
                           DataType &first);
template<class DataType>
void get_BSplineBasisValue(DataType basis,
                           int index,
                           DataType &value,
                           DataType &first,
                           DataType &second);

template <class DataType>
void set_first_order_basis_values(DataType *basisX,
                                  DataType *basisY);

template <class DataType>
void set_first_order_basis_values(DataType *basisX,
                                  DataType *basisY,
                                  DataType *basisZ);

template <class DataType>
void set_second_order_bspline_basis_values(DataType *basisXX,
                                           DataType *basisYY,
                                           DataType *basisXY);
template <class DataType>
void set_second_order_bspline_basis_values(DataType *basisXX,
                                           DataType *basisYY,
                                           DataType *basisZZ,
                                           DataType *basisXY,
                                           DataType *basisYZ,
                                           DataType *basisXZ);


template<class DataType>
void get_SplineBasisValues(DataType basis,
                           DataType *values);
template<class DataType>
void get_SplineBasisValues(DataType basis,
                           DataType *values,
                           DataType *first);
template<class DataType>
void get_SplineBasisValues(DataType basis,
                           DataType *values,
                           DataType *first,
                           DataType *second);

template <class DataType>
void get_SlidedValues(DataType &defX,
                      DataType &defY,
                      const int x,
                      const int y,
                      const DataType *defPtrX,
                      const DataType *defPtrY,
                      const mat44 *dfVoxel2Real,
                      const int *dim,
                      const bool displacement);
template <class DataType>
void get_SlidedValues(DataType &defX,
                      DataType &defY,
                      DataType &defZ,
                      const int x,
                      const int y,
                      const int z,
                      const DataType *defPtrX,
                      const DataType *defPtrY,
                      const DataType *defPtrZ,
                      const mat44 *dfVoxel2Real,
                      const int *dim,
                      const bool displacement);


template <class DataType>
void get_GridValues(int startX,
                    int startY,
                    nifti_image *splineControlPoint,
                    DataType *splineX,
                    DataType *splineY,
                    DataType *dispX,
                    DataType *dispY,
                    bool approx,
                    bool displacement);
template <class DataType>
void get_GridValues(int startX,
                    int startY,
                    int startZ,
                    nifti_image *splineControlPoint,
                    DataType *splineX,
                    DataType *splineY,
                    DataType *splineZ,
                    DataType *dispX,
                    DataType *dispY,
                    DataType *dispZ,
                    bool approx,
                    bool displacement);
