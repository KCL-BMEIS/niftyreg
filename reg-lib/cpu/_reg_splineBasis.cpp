/**
 * @file _reg_splineBasis.cpp
 * @brief Library that contains local deformation related functions
 * @author Marc Modat
 * @date 23/12/2015
 *
 *  Copyright (c) 2015-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_splineBasis.h"

/* *************************************************************** */
// Evaluated entirely in DataType, with no double intermediates: the float instantiation then rounds
// identically to the CUDA kernels (Cuda::GetBasisSplineValues), which is what keeps the CPU and
// CUDA platforms bit-exact. The double instantiation is unaffected.
template<class DataType>
void get_BSplineBasisValues(DataType basis, DataType *values) {
    const DataType ff = basis * basis;
    const DataType fff = ff * basis;
    const DataType mf = DataType(1) - basis;
    values[0] = mf * mf * mf / DataType(6);
    values[1] = (DataType(3) * fff - DataType(6) * ff + DataType(4)) / DataType(6);
    values[2] = (DataType(-3) * fff + DataType(3) * ff + DataType(3) * basis + DataType(1)) / DataType(6);
    values[3] = fff / DataType(6);
}
template void get_BSplineBasisValues<float>(float, float *);
template void get_BSplineBasisValues<double>(double, double *);
/* *************************************************************** */
template<class DataType>
void get_BSplineBasisValues(DataType basis, DataType *values, DataType *first) {
    get_BSplineBasisValues<DataType>(basis, values);
    first[3] = basis * basis / DataType(2);
    first[0] = basis - DataType(1) / DataType(2) - first[3];
    first[2] = DataType(1) + first[0] - DataType(2) * first[3];
    first[1] = -first[0] - first[2] - first[3];
}
template void get_BSplineBasisValues<float>(float, float*, float *);
template void get_BSplineBasisValues<double>(double, double*, double *);
/* *************************************************************** */
template<class DataType>
void get_BSplineBasisValues(DataType basis, DataType *values, DataType *first, DataType *second) {
    get_BSplineBasisValues<DataType>(basis, values, first);
    second[3] = basis;
    second[0] = DataType(1) - second[3];
    second[2] = second[0] - DataType(2) * second[3];
    second[1] = -second[0] - second[2] - second[3];
}
template void get_BSplineBasisValues<float>(float, float*, float*, float *);
template void get_BSplineBasisValues<double>(double, double*, double*, double *);
/* *************************************************************** */
template<class DataType>
void get_BSplineBasisValue(DataType basis, int index, DataType& value) {
    switch (index) {
    case 0:
        value = (DataType(1) - basis) * (DataType(1) - basis) * (DataType(1) - basis) / DataType(6);
        break;
    case 1:
        value = (DataType(3) * basis * basis * basis - DataType(6) * basis * basis + DataType(4)) / DataType(6);
        break;
    case 2:
        value = (DataType(3) * basis * basis - DataType(3) * basis * basis * basis + DataType(3) * basis + DataType(1)) / DataType(6);
        break;
    case 3:
        value = basis * basis * basis / DataType(6);
        break;
    default:
        value = (DataType)0;
        break;
    }
}
template void get_BSplineBasisValue<float>(float, int, float&);
template void get_BSplineBasisValue<double>(double, int, double&);
/* *************************************************************** */
template<class DataType>
void get_BSplineBasisValue(DataType basis, int index, DataType& value, DataType& first) {
    get_BSplineBasisValue<DataType>(basis, index, value);
    switch (index) {
    case 0:
        first = (DataType(2) * basis - basis * basis - DataType(1)) / DataType(2);
        break;
    case 1:
        first = (DataType(3) * basis * basis - DataType(4) * basis) / DataType(2);
        break;
    case 2:
        first = (DataType(2) * basis - DataType(3) * basis * basis + DataType(1)) / DataType(2);
        break;
    case 3:
        first = basis * basis / DataType(2);
        break;
    default:
        first = (DataType)0;
        break;
    }
}
template void get_BSplineBasisValue<float>(float, int, float&, float&);
template void get_BSplineBasisValue<double>(double, int, double&, double&);
/* *************************************************************** */
template<class DataType>
void get_BSplineBasisValue(DataType basis, int index, DataType& value, DataType& first, DataType& second) {
    get_BSplineBasisValue<DataType>(basis, index, value, first);
    switch (index) {
    case 0:
        second = DataType(1) - basis;
        break;
    case 1:
        second = DataType(3) * basis - DataType(2);
        break;
    case 2:
        second = DataType(1) - DataType(3) * basis;
        break;
    case 3:
        second = basis;
        break;
    default:
        second = (DataType)0;
        break;
    }
}
template void get_BSplineBasisValue<float>(float, int, float&, float&, float&);
template void get_BSplineBasisValue<double>(double, int, double&, double&, double&);
/* *************************************************************** */
// Evaluated entirely in DataType, for the reason given on get_BSplineBasisValues above, and kept
// structurally identical to Cuda::GetBasisSplineValues<false>.
template<class DataType>
void get_SplineBasisValues(DataType basis, DataType *values) {
    const DataType ff = basis * basis;
    values[0] = (basis * ((DataType(2) - basis) * basis - DataType(1))) / DataType(2);
    values[1] = (ff * (DataType(3) * basis - DataType(5)) + DataType(2)) / DataType(2);
    values[2] = (basis * ((DataType(4) - DataType(3) * basis) * basis + DataType(1))) / DataType(2);
    values[3] = (basis - DataType(1)) * ff / DataType(2);
}
template void get_SplineBasisValues<float>(float, float *);
template void get_SplineBasisValues<double>(double, double *);
/* *************************************************************** */
template<class DataType>
void get_SplineBasisValues(DataType basis, DataType *values, DataType *first) {
    get_SplineBasisValues<DataType>(basis, values);
    const DataType ff = basis * basis;
    first[0] = (DataType(4) * basis - DataType(3) * ff - DataType(1)) / DataType(2);
    first[1] = (DataType(9) * basis - DataType(10)) * basis / DataType(2);
    first[2] = (DataType(8) * basis - DataType(9) * ff + DataType(1)) / DataType(2);
    first[3] = (DataType(3) * basis - DataType(2)) * basis / DataType(2);
}
template void get_SplineBasisValues<float>(float, float*, float *);
template void get_SplineBasisValues<double>(double, double*, double *);
/* *************************************************************** */
template<class DataType>
void get_SplineBasisValues(DataType basis, DataType *values, DataType *first, DataType *second) {
    get_SplineBasisValues<DataType>(basis, values, first);
    second[0] = DataType(2) - DataType(3) * basis;
    second[1] = DataType(9) * basis - DataType(5);
    second[2] = DataType(4) - DataType(9) * basis;
    second[3] = DataType(3) * basis - DataType(1);
}
template void get_SplineBasisValues<float>(float, float*, float*, float *);
template void get_SplineBasisValues<double>(double, double*, double*, double *);
/* *************************************************************** */
template <class DataType>
void set_first_order_basis_values(DataType *basisX, DataType *basisY) {
    double BASIS[4], FIRST[4]; get_BSplineBasisValues<double>(0, BASIS, FIRST);
    int index = 0;
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++, index++) {
            basisX[index] = static_cast<DataType>(FIRST[x] * BASIS[y]);
            basisY[index] = static_cast<DataType>(BASIS[x] * FIRST[y]);
        }
    }
}
template void set_first_order_basis_values<float>(float*, float *);
template void set_first_order_basis_values<double>(double*, double *);
/* *************************************************************** */
template <class DataType>
void set_first_order_basis_values(DataType *basisX, DataType *basisY, DataType *basisZ) {
    basisX[0] = static_cast<DataType>(-0.0138889);
    basisY[0] = static_cast<DataType>(-0.0138889);
    basisZ[0] = static_cast<DataType>(-0.0138889);
    basisX[1] = static_cast<DataType>(0);
    basisY[1] = static_cast<DataType>(-0.0555556);
    basisZ[1] = static_cast<DataType>(-0.0555556);
    basisX[2] = static_cast<DataType>(0.0138889);
    basisY[2] = static_cast<DataType>(-0.0138889);
    basisZ[2] = static_cast<DataType>(-0.0138889);
    basisX[3] = static_cast<DataType>(-0.0555556);
    basisY[3] = static_cast<DataType>(0);
    basisZ[3] = static_cast<DataType>(-0.0555556);
    basisX[4] = static_cast<DataType>(0);
    basisY[4] = static_cast<DataType>(0);
    basisZ[4] = static_cast<DataType>(-0.222222);
    basisX[5] = static_cast<DataType>(0.0555556);
    basisY[5] = static_cast<DataType>(0);
    basisZ[5] = static_cast<DataType>(-0.0555556);
    basisX[6] = static_cast<DataType>(-0.0138889);
    basisY[6] = static_cast<DataType>(0.0138889);
    basisZ[6] = static_cast<DataType>(-0.0138889);
    basisX[7] = static_cast<DataType>(0);
    basisY[7] = static_cast<DataType>(0.0555556);
    basisZ[7] = static_cast<DataType>(-0.0555556);
    basisX[8] = static_cast<DataType>(0.0138889);
    basisY[8] = static_cast<DataType>(0.0138889);
    basisZ[8] = static_cast<DataType>(-0.0138889);
    basisX[9] = static_cast<DataType>(-0.0555556);
    basisY[9] = static_cast<DataType>(-0.0555556);
    basisZ[9] = static_cast<DataType>(0);
    basisX[10] = static_cast<DataType>(0);
    basisY[10] = static_cast<DataType>(-0.222222);
    basisZ[10] = static_cast<DataType>(0);
    basisX[11] = static_cast<DataType>(0.0555556);
    basisY[11] = static_cast<DataType>(-0.0555556);
    basisZ[11] = static_cast<DataType>(0);
    basisX[12] = static_cast<DataType>(-0.222222);
    basisY[12] = static_cast<DataType>(0);
    basisZ[12] = static_cast<DataType>(0);
    basisX[13] = static_cast<DataType>(0);
    basisY[13] = static_cast<DataType>(0);
    basisZ[13] = static_cast<DataType>(0);
    basisX[14] = static_cast<DataType>(0.222222);
    basisY[14] = static_cast<DataType>(0);
    basisZ[14] = static_cast<DataType>(0);
    basisX[15] = static_cast<DataType>(-0.0555556);
    basisY[15] = static_cast<DataType>(0.0555556);
    basisZ[15] = static_cast<DataType>(0);
    basisX[16] = static_cast<DataType>(0);
    basisY[16] = static_cast<DataType>(0.222222);
    basisZ[16] = static_cast<DataType>(0);
    basisX[17] = static_cast<DataType>(0.0555556);
    basisY[17] = static_cast<DataType>(0.0555556);
    basisZ[17] = static_cast<DataType>(0);
    basisX[18] = static_cast<DataType>(-0.0138889);
    basisY[18] = static_cast<DataType>(-0.0138889);
    basisZ[18] = static_cast<DataType>(0.0138889);
    basisX[19] = static_cast<DataType>(0);
    basisY[19] = static_cast<DataType>(-0.0555556);
    basisZ[19] = static_cast<DataType>(0.0555556);
    basisX[20] = static_cast<DataType>(0.0138889);
    basisY[20] = static_cast<DataType>(-0.0138889);
    basisZ[20] = static_cast<DataType>(0.0138889);
    basisX[21] = static_cast<DataType>(-0.0555556);
    basisY[21] = static_cast<DataType>(0);
    basisZ[21] = static_cast<DataType>(0.0555556);
    basisX[22] = static_cast<DataType>(0);
    basisY[22] = static_cast<DataType>(0);
    basisZ[22] = static_cast<DataType>(0.222222);
    basisX[23] = static_cast<DataType>(0.0555556);
    basisY[23] = static_cast<DataType>(0);
    basisZ[23] = static_cast<DataType>(0.0555556);
    basisX[24] = static_cast<DataType>(-0.0138889);
    basisY[24] = static_cast<DataType>(0.0138889);
    basisZ[24] = static_cast<DataType>(0.0138889);
    basisX[25] = static_cast<DataType>(0);
    basisY[25] = static_cast<DataType>(0.0555556);
    basisZ[25] = static_cast<DataType>(0.0555556);
    basisX[26] = static_cast<DataType>(0.0138889);
    basisY[26] = static_cast<DataType>(0.0138889);
    basisZ[26] = static_cast<DataType>(0.0138889);
}
template void set_first_order_basis_values<float>(float*, float*, float *);
template void set_first_order_basis_values<double>(double*, double*, double *);
/* *************************************************************** */
template <class DataType>
void set_second_order_bspline_basis_values(DataType *basisXX, DataType *basisYY, DataType *basisXY) {
    basisXX[0] = 0.166667f;
    basisYY[0] = 0.166667f;
    basisXY[0] = 0.25f;
    basisXX[1] = -0.333333f;
    basisYY[1] = 0.666667f;
    basisXY[1] = -0.f;
    basisXX[2] = 0.166667f;
    basisYY[2] = 0.166667f;
    basisXY[2] = -0.25f;
    basisXX[3] = 0.666667f;
    basisYY[3] = -0.333333f;
    basisXY[3] = -0.f;
    basisXX[4] = -1.33333f;
    basisYY[4] = -1.33333f;
    basisXY[4] = 0.f;
    basisXX[5] = 0.666667f;
    basisYY[5] = -0.333333f;
    basisXY[5] = 0.f;
    basisXX[6] = 0.166667f;
    basisYY[6] = 0.166667f;
    basisXY[6] = -0.25f;
    basisXX[7] = -0.333333f;
    basisYY[7] = 0.666667f;
    basisXY[7] = 0.f;
    basisXX[8] = 0.166667f;
    basisYY[8] = 0.166667f;
    basisXY[8] = 0.25f;
}
template void set_second_order_bspline_basis_values<float>(float*, float*, float *);
template void set_second_order_bspline_basis_values<double>(double*, double*, double *);
/* *************************************************************** */
template <class DataType>
void set_second_order_bspline_basis_values(DataType *basisXX, DataType *basisYY, DataType *basisZZ, DataType *basisXY, DataType *basisYZ, DataType *basisXZ) {
    basisXX[0] = 0.027778f;
    basisYY[0] = 0.027778f;
    basisZZ[0] = 0.027778f;
    basisXY[0] = 0.041667f;
    basisYZ[0] = 0.041667f;
    basisXZ[0] = 0.041667f;
    basisXX[1] = -0.055556f;
    basisYY[1] = 0.111111f;
    basisZZ[1] = 0.111111f;
    basisXY[1] = -0.000000f;
    basisYZ[1] = 0.166667f;
    basisXZ[1] = -0.000000f;
    basisXX[2] = 0.027778f;
    basisYY[2] = 0.027778f;
    basisZZ[2] = 0.027778f;
    basisXY[2] = -0.041667f;
    basisYZ[2] = 0.041667f;
    basisXZ[2] = -0.041667f;
    basisXX[3] = 0.111111f;
    basisYY[3] = -0.055556f;
    basisZZ[3] = 0.111111f;
    basisXY[3] = -0.000000f;
    basisYZ[3] = -0.000000f;
    basisXZ[3] = 0.166667f;
    basisXX[4] = -0.222222f;
    basisYY[4] = -0.222222f;
    basisZZ[4] = 0.444444f;
    basisXY[4] = 0.000000f;
    basisYZ[4] = -0.000000f;
    basisXZ[4] = -0.000000f;
    basisXX[5] = 0.111111f;
    basisYY[5] = -0.055556f;
    basisZZ[5] = 0.111111f;
    basisXY[5] = 0.000000f;
    basisYZ[5] = -0.000000f;
    basisXZ[5] = -0.166667f;
    basisXX[6] = 0.027778f;
    basisYY[6] = 0.027778f;
    basisZZ[6] = 0.027778f;
    basisXY[6] = -0.041667f;
    basisYZ[6] = -0.041667f;
    basisXZ[6] = 0.041667f;
    basisXX[7] = -0.055556f;
    basisYY[7] = 0.111111f;
    basisZZ[7] = 0.111111f;
    basisXY[7] = 0.000000f;
    basisYZ[7] = -0.166667f;
    basisXZ[7] = -0.000000f;
    basisXX[8] = 0.027778f;
    basisYY[8] = 0.027778f;
    basisZZ[8] = 0.027778f;
    basisXY[8] = 0.041667f;
    basisYZ[8] = -0.041667f;
    basisXZ[8] = -0.041667f;
    basisXX[9] = 0.111111f;
    basisYY[9] = 0.111111f;
    basisZZ[9] = -0.055556f;
    basisXY[9] = 0.166667f;
    basisYZ[9] = -0.000000f;
    basisXZ[9] = -0.000000f;
    basisXX[10] = -0.222222f;
    basisYY[10] = 0.444444f;
    basisZZ[10] = -0.222222f;
    basisXY[10] = -0.000000f;
    basisYZ[10] = -0.000000f;
    basisXZ[10] = 0.000000f;
    basisXX[11] = 0.111111f;
    basisYY[11] = 0.111111f;
    basisZZ[11] = -0.055556f;
    basisXY[11] = -0.166667f;
    basisYZ[11] = -0.000000f;
    basisXZ[11] = 0.000000f;
    basisXX[12] = 0.444444f;
    basisYY[12] = -0.222222f;
    basisZZ[12] = -0.222222f;
    basisXY[12] = -0.000000f;
    basisYZ[12] = 0.000000f;
    basisXZ[12] = -0.000000f;
    basisXX[13] = -0.888889f;
    basisYY[13] = -0.888889f;
    basisZZ[13] = -0.888889f;
    basisXY[13] = 0.000000f;
    basisYZ[13] = 0.000000f;
    basisXZ[13] = 0.000000f;
    basisXX[14] = 0.444444f;
    basisYY[14] = -0.222222f;
    basisZZ[14] = -0.222222f;
    basisXY[14] = 0.000000f;
    basisYZ[14] = 0.000000f;
    basisXZ[14] = 0.000000f;
    basisXX[15] = 0.111111f;
    basisYY[15] = 0.111111f;
    basisZZ[15] = -0.055556f;
    basisXY[15] = -0.166667f;
    basisYZ[15] = 0.000000f;
    basisXZ[15] = -0.000000f;
    basisXX[16] = -0.222222f;
    basisYY[16] = 0.444444f;
    basisZZ[16] = -0.222222f;
    basisXY[16] = 0.000000f;
    basisYZ[16] = 0.000000f;
    basisXZ[16] = 0.000000f;
    basisXX[17] = 0.111111f;
    basisYY[17] = 0.111111f;
    basisZZ[17] = -0.055556f;
    basisXY[17] = 0.166667f;
    basisYZ[17] = 0.000000f;
    basisXZ[17] = 0.000000f;
    basisXX[18] = 0.027778f;
    basisYY[18] = 0.027778f;
    basisZZ[18] = 0.027778f;
    basisXY[18] = 0.041667f;
    basisYZ[18] = -0.041667f;
    basisXZ[18] = -0.041667f;
    basisXX[19] = -0.055556f;
    basisYY[19] = 0.111111f;
    basisZZ[19] = 0.111111f;
    basisXY[19] = -0.000000f;
    basisYZ[19] = -0.166667f;
    basisXZ[19] = 0.000000f;
    basisXX[20] = 0.027778f;
    basisYY[20] = 0.027778f;
    basisZZ[20] = 0.027778f;
    basisXY[20] = -0.041667f;
    basisYZ[20] = -0.041667f;
    basisXZ[20] = 0.041667f;
    basisXX[21] = 0.111111f;
    basisYY[21] = -0.055556f;
    basisZZ[21] = 0.111111f;
    basisXY[21] = -0.000000f;
    basisYZ[21] = 0.000000f;
    basisXZ[21] = -0.166667f;
    basisXX[22] = -0.222222f;
    basisYY[22] = -0.222222f;
    basisZZ[22] = 0.444444f;
    basisXY[22] = 0.000000f;
    basisYZ[22] = 0.000000f;
    basisXZ[22] = 0.000000f;
    basisXX[23] = 0.111111f;
    basisYY[23] = -0.055556f;
    basisZZ[23] = 0.111111f;
    basisXY[23] = 0.000000f;
    basisYZ[23] = 0.000000f;
    basisXZ[23] = 0.166667f;
    basisXX[24] = 0.027778f;
    basisYY[24] = 0.027778f;
    basisZZ[24] = 0.027778f;
    basisXY[24] = -0.041667f;
    basisYZ[24] = 0.041667f;
    basisXZ[24] = -0.041667f;
    basisXX[25] = -0.055556f;
    basisYY[25] = 0.111111f;
    basisZZ[25] = 0.111111f;
    basisXY[25] = 0.000000f;
    basisYZ[25] = 0.166667f;
    basisXZ[25] = 0.000000f;
    basisXX[26] = 0.027778f;
    basisYY[26] = 0.027778f;
    basisZZ[26] = 0.027778f;
    basisXY[26] = 0.041667f;
    basisYZ[26] = 0.041667f;
    basisXZ[26] = 0.041667f;
}
template void set_second_order_bspline_basis_values<float>(float*, float*, float*, float*, float*, float*);
template void set_second_order_bspline_basis_values<double>(double*, double*, double*, double*, double*, double*);
/* *************************************************************** */
template <class DataType>
void get_SlidedValues(DataType& defX,
                      DataType& defY,
                      const int x,
                      const int y,
                      const DataType *defPtrX,
                      const DataType *defPtrY,
                      const mat44 *dfVoxelToReal,
                      const int *dim,
                      const bool displacement) {
    int newX = x;
    if (x < 0)
        newX = 0;
    else if (x >= dim[1])
        newX = dim[1] - 1;

    int newY = y;
    if (y < 0)
        newY = 0;
    else if (y >= dim[2])
        newY = dim[2] - 1;

    DataType shiftValueX = 0;
    DataType shiftValueY = 0;
    if (!displacement) {
        const int shiftIndexX = x - newX;
        const int shiftIndexY = y - newY;
        shiftValueX = shiftIndexX * dfVoxelToReal->m[0][0] + shiftIndexY * dfVoxelToReal->m[0][1];
        shiftValueY = shiftIndexX * dfVoxelToReal->m[1][0] + shiftIndexY * dfVoxelToReal->m[1][1];
    }
    const int index = newY * dim[1] + newX;
    defX = defPtrX[index] + shiftValueX;
    defY = defPtrY[index] + shiftValueY;
}
template void get_SlidedValues<float>(float&, float&, const int, const int, const float*, const float*, const mat44*, const int*, const bool);
template void get_SlidedValues<double>(double&, double&, const int, const int, const double*, const double*, const mat44*, const int*, const bool);
/* *************************************************************** */
template <class DataType>
void get_SlidedValues(DataType& defX,
                      DataType& defY,
                      DataType& defZ,
                      const int x,
                      const int y,
                      const int z,
                      const DataType *defPtrX,
                      const DataType *defPtrY,
                      const DataType *defPtrZ,
                      const mat44 *dfVoxelToReal,
                      const int *dim,
                      const bool displacement) {
    int newX = x;
    if (x < 0)
        newX = 0;
    else if (x >= dim[1])
        newX = dim[1] - 1;

    int newY = y;
    if (y < 0)
        newY = 0;
    else if (y >= dim[2])
        newY = dim[2] - 1;

    int newZ = z;
    if (z < 0)
        newZ = 0;
    else if (z >= dim[3])
        newZ = dim[3] - 1;

    DataType shiftValueX = 0;
    DataType shiftValueY = 0;
    DataType shiftValueZ = 0;
    if (!displacement) {
        const int shiftIndexX = x - newX;
        const int shiftIndexY = y - newY;
        const int shiftIndexZ = z - newZ;
        shiftValueX =
            shiftIndexX * dfVoxelToReal->m[0][0] +
            shiftIndexY * dfVoxelToReal->m[0][1] +
            shiftIndexZ * dfVoxelToReal->m[0][2];
        shiftValueY =
            shiftIndexX * dfVoxelToReal->m[1][0] +
            shiftIndexY * dfVoxelToReal->m[1][1] +
            shiftIndexZ * dfVoxelToReal->m[1][2];
        shiftValueZ =
            shiftIndexX * dfVoxelToReal->m[2][0] +
            shiftIndexY * dfVoxelToReal->m[2][1] +
            shiftIndexZ * dfVoxelToReal->m[2][2];
    }
    const int index = (newZ * dim[2] + newY) * dim[1] + newX;
    defX = defPtrX[index] + shiftValueX;
    defY = defPtrY[index] + shiftValueY;
    defZ = defPtrZ[index] + shiftValueZ;
}
template void get_SlidedValues<float>(float&, float&, float&, const int, const int, const int, const float*, const float*, const float*, const mat44*, const int*, const bool);
template void get_SlidedValues<double>(double&, double&, double&, const int, const int, const int, const double*, const double*, const double*, const mat44*, const int*, const bool);
/* *************************************************************** */
template <class DataType>
void get_GridValues(int startX,
                    int startY,
                    nifti_image *splineControlPoint,
                    DataType *splineX,
                    DataType *splineY,
                    DataType *dispX,
                    DataType *dispY,
                    bool approx,
                    bool displacement) {
    int range = 4;
    if (approx) range = 3;

    size_t index;
    size_t coord = 0;
    DataType *xxPtr = nullptr, *yyPtr = nullptr;

    const mat44 *voxelToReal = splineControlPoint->sform_code > 0 ? &splineControlPoint->sto_xyz : &splineControlPoint->qto_xyz;

    for (int Y = startY; Y < startY + range; Y++) {
        bool out = false;
        if (Y > -1 && Y < splineControlPoint->ny) {
            index = Y * splineControlPoint->nx;
            xxPtr = &splineX[index];
            yyPtr = &splineY[index];
        } else out = true;
        for (int X = startX; X < startX + range; X++, coord++) {
            if (X > -1 && X < splineControlPoint->nx && out == false) {
                dispX[coord] = xxPtr[X];
                dispY[coord] = yyPtr[X];
            } else {
                get_SlidedValues<DataType>(dispX[coord],
                                           dispY[coord],
                                           X,
                                           Y,
                                           splineX,
                                           splineY,
                                           voxelToReal,
                                           splineControlPoint->dim,
                                           displacement);
            }
        }
    }
}
template void get_GridValues<float>(int, int, nifti_image*, float*, float*, float*, float*, bool, bool);
template void get_GridValues<double>(int, int, nifti_image*, double*, double*, double*, double*, bool, bool);
/* *************************************************************** */
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
                    bool displacement) {
    int range = 4;
    if (approx)
        range = 3;

    size_t index;
    size_t coord = 0;
    DataType *xPtr = nullptr, *yPtr = nullptr, *zPtr = nullptr;
    DataType *xxPtr = nullptr, *yyPtr = nullptr, *zzPtr = nullptr;

    const mat44 *voxelToReal = splineControlPoint->sform_code > 0 ? &splineControlPoint->sto_xyz : &splineControlPoint->qto_xyz;

    for (int Z = startZ; Z < startZ + range; Z++) {
        bool out = false;
        if (Z > -1 && Z < splineControlPoint->nz) {
            index = Z * splineControlPoint->nx * splineControlPoint->ny;
            xPtr = &splineX[index];
            yPtr = &splineY[index];
            zPtr = &splineZ[index];
        } else out = true;
        for (int Y = startY; Y < startY + range; Y++) {
            if (Y > -1 && Y < splineControlPoint->ny && out == false) {
                index = Y * splineControlPoint->nx;
                xxPtr = &xPtr[index];
                yyPtr = &yPtr[index];
                zzPtr = &zPtr[index];
            } else out = true;
            for (int X = startX; X < startX + range; X++, coord++) {
                if (X > -1 && X < splineControlPoint->nx && out == false) {
                    dispX[coord] = xxPtr[X];
                    dispY[coord] = yyPtr[X];
                    dispZ[coord] = zzPtr[X];
                } else {
                    get_SlidedValues<DataType>(dispX[coord],
                                               dispY[coord],
                                               dispZ[coord],
                                               X,
                                               Y,
                                               Z,
                                               splineX,
                                               splineY,
                                               splineZ,
                                               voxelToReal,
                                               splineControlPoint->dim,
                                               displacement);
                }
            } // X
        } // Y
    } // Z
}
template void get_GridValues<float>(int, int, int, nifti_image*, float*, float*, float*, float*, float*, float*, bool, bool);
template void get_GridValues<double>(int, int, int, nifti_image*, double*, double*, double*, double*, double*, double*, bool, bool);
/* *************************************************************** */
