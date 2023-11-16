/*
 *  _reg_localTrans_regul.cpp
 *
 *
 *  Created by Marc Modat on 10/05/2011.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_localTrans_regul.h"

/* *************************************************************** */
template<class DataType>
double reg_spline_approxBendingEnergyValue2D(const nifti_image *splineControlPoint) {
    const size_t nodeNumber = NiftiImage::calcVoxelNumber(splineControlPoint, 2);
    int a, b, x, y, index, i;

    // Create pointers to the spline coefficients
    const DataType *splinePtrX = static_cast<DataType*>(splineControlPoint->data);
    const DataType *splinePtrY = &splinePtrX[nodeNumber];

    // get the constant basis values
    DataType basisXX[9], basisYY[9], basisXY[9];
    set_second_order_bspline_basis_values(basisXX, basisYY, basisXY);

    double constraintValue = 0;

    DataType splineCoeffX, splineCoeffY;
    DataType XX_x, YY_x, XY_x;
    DataType XX_y, YY_y, XY_y;

#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(splineControlPoint, splinePtrX, splinePtrY, \
    basisXX, basisYY, basisXY) \
    private(XX_x, YY_x, XY_x, XX_y, YY_y, XY_y, \
    x, a, b, index, i, splineCoeffX, splineCoeffY) \
    reduction(+:constraintValue)
#endif
    for (y = 1; y < splineControlPoint->ny - 1; ++y) {
        for (x = 1; x < splineControlPoint->nx - 1; ++x) {
            XX_x = 0, YY_x = 0, XY_x = 0;
            XX_y = 0, YY_y = 0, XY_y = 0;

            i = 0;
            for (b = -1; b < 2; b++) {
                for (a = -1; a < 2; a++) {
                    index = (y + b) * splineControlPoint->nx + x + a;
                    splineCoeffX = splinePtrX[index];
                    splineCoeffY = splinePtrY[index];
                    XX_x += basisXX[i] * splineCoeffX;
                    YY_x += basisYY[i] * splineCoeffX;
                    XY_x += basisXY[i] * splineCoeffX;

                    XX_y += basisXX[i] * splineCoeffY;
                    YY_y += basisYY[i] * splineCoeffY;
                    XY_y += basisXY[i] * splineCoeffY;
                    ++i;
                }
            }

            constraintValue += double(XX_x * XX_x + YY_x * YY_x + 2.0 * XY_x * XY_x +
                                      XX_y * XX_y + YY_y * YY_y + 2.0 * XY_y * XY_y);
        }
    }
    return constraintValue / (double)splineControlPoint->nvox;
}
/* *************************************************************** */
template<class DataType>
double reg_spline_approxBendingEnergyValue3D(const nifti_image *splineControlPoint) {
    const size_t nodeNumber = NiftiImage::calcVoxelNumber(splineControlPoint, 3);
    int a, b, c, x, y, z, index, i;

    // Create pointers to the spline coefficients
    const DataType *splinePtrX = static_cast<DataType*>(splineControlPoint->data);
    const DataType *splinePtrY = &splinePtrX[nodeNumber];
    const DataType *splinePtrZ = &splinePtrY[nodeNumber];

    // get the constant basis values
    DataType basisXX[27], basisYY[27], basisZZ[27], basisXY[27], basisYZ[27], basisXZ[27];
    set_second_order_bspline_basis_values(basisXX, basisYY, basisZZ, basisXY, basisYZ, basisXZ);

    double constraintValue = 0;

    DataType splineCoeffX, splineCoeffY, splineCoeffZ;
    DataType XX_x, YY_x, ZZ_x, XY_x, YZ_x, XZ_x;
    DataType XX_y, YY_y, ZZ_y, XY_y, YZ_y, XZ_y;
    DataType XX_z, YY_z, ZZ_z, XY_z, YZ_z, XZ_z;

#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(splineControlPoint, splinePtrX, splinePtrY, splinePtrZ, \
    basisXX, basisYY, basisZZ, basisXY, basisYZ, basisXZ) \
    private(XX_x, YY_x, ZZ_x, XY_x, YZ_x, XZ_x, XX_y, YY_y, ZZ_y, XY_y, YZ_y, XZ_y, \
    XX_z, YY_z, ZZ_z, XY_z, YZ_z, XZ_z, x, y, a, b, c, index, i, \
    splineCoeffX, splineCoeffY, splineCoeffZ) \
    reduction(+:constraintValue)
#endif
    for (z = 1; z < splineControlPoint->nz - 1; ++z) {
        for (y = 1; y < splineControlPoint->ny - 1; ++y) {
            for (x = 1; x < splineControlPoint->nx - 1; ++x) {
                XX_x = 0, YY_x = 0, ZZ_x = 0;
                XY_x = 0, YZ_x = 0, XZ_x = 0;
                XX_y = 0, YY_y = 0, ZZ_y = 0;
                XY_y = 0, YZ_y = 0, XZ_y = 0;
                XX_z = 0, YY_z = 0, ZZ_z = 0;
                XY_z = 0, YZ_z = 0, XZ_z = 0;

                i = 0;
                for (c = -1; c < 2; c++) {
                    for (b = -1; b < 2; b++) {
                        for (a = -1; a < 2; a++) {
                            index = ((z + c) * splineControlPoint->ny + y + b) * splineControlPoint->nx + x + a;
                            splineCoeffX = splinePtrX[index];
                            splineCoeffY = splinePtrY[index];
                            splineCoeffZ = splinePtrZ[index];
                            XX_x += basisXX[i] * splineCoeffX;
                            YY_x += basisYY[i] * splineCoeffX;
                            ZZ_x += basisZZ[i] * splineCoeffX;
                            XY_x += basisXY[i] * splineCoeffX;
                            YZ_x += basisYZ[i] * splineCoeffX;
                            XZ_x += basisXZ[i] * splineCoeffX;

                            XX_y += basisXX[i] * splineCoeffY;
                            YY_y += basisYY[i] * splineCoeffY;
                            ZZ_y += basisZZ[i] * splineCoeffY;
                            XY_y += basisXY[i] * splineCoeffY;
                            YZ_y += basisYZ[i] * splineCoeffY;
                            XZ_y += basisXZ[i] * splineCoeffY;

                            XX_z += basisXX[i] * splineCoeffZ;
                            YY_z += basisYY[i] * splineCoeffZ;
                            ZZ_z += basisZZ[i] * splineCoeffZ;
                            XY_z += basisXY[i] * splineCoeffZ;
                            YZ_z += basisYZ[i] * splineCoeffZ;
                            XZ_z += basisXZ[i] * splineCoeffZ;
                            ++i;
                        }
                    }
                }

                constraintValue += double(
                    XX_x * XX_x + YY_x * YY_x + ZZ_x * ZZ_x + 2.0 * (XY_x * XY_x + YZ_x * YZ_x + XZ_x * XZ_x) +
                    XX_y * XX_y + YY_y * YY_y + ZZ_y * ZZ_y + 2.0 * (XY_y * XY_y + YZ_y * YZ_y + XZ_y * XZ_y) +
                    XX_z * XX_z + YY_z * YY_z + ZZ_z * ZZ_z + 2.0 * (XY_z * XY_z + YZ_z * YZ_z + XZ_z * XZ_z));
            }
        }
    }
    return constraintValue / (double)splineControlPoint->nvox;
}
/* *************************************************************** */
double reg_spline_approxBendingEnergy(const nifti_image *splineControlPoint) {
    if (splineControlPoint->nz == 1) {
        switch (splineControlPoint->datatype) {
        case NIFTI_TYPE_FLOAT32:
            return reg_spline_approxBendingEnergyValue2D<float>(splineControlPoint);
        case NIFTI_TYPE_FLOAT64:
            return reg_spline_approxBendingEnergyValue2D<double>(splineControlPoint);
        default:
            NR_FATAL_ERROR("Only implemented for single or double precision images");
            return 0;
        }
    } else {
        switch (splineControlPoint->datatype) {
        case NIFTI_TYPE_FLOAT32:
            return reg_spline_approxBendingEnergyValue3D<float>(splineControlPoint);
        case NIFTI_TYPE_FLOAT64:
            return reg_spline_approxBendingEnergyValue3D<double>(splineControlPoint);
        default:
            NR_FATAL_ERROR("Only implemented for single or double precision images");
            return 0;
        }
    }
}
/* *************************************************************** */
template<class DataType>
void reg_spline_approxBendingEnergyGradient2D(nifti_image *splineControlPoint,
                                              nifti_image *gradientImage,
                                              float weight) {
    const size_t nodeNumber = NiftiImage::calcVoxelNumber(splineControlPoint, 2);
    int a, b, x, y, X, Y, index, i;

    // Create pointers to the spline coefficients
    const DataType *splinePtrX = static_cast<DataType*>(splineControlPoint->data);
    const DataType *splinePtrY = &splinePtrX[nodeNumber];

    // get the constant basis values
    DataType basisXX[9], basisYY[9], basisXY[9];
    set_second_order_bspline_basis_values(basisXX, basisYY, basisXY);

    DataType splineCoeffX;
    DataType splineCoeffY;
    DataType XX_x, YY_x, XY_x;
    DataType XX_y, YY_y, XY_y;

    DataType *derivativeValues = (DataType*)calloc(6 * nodeNumber, sizeof(DataType));
    DataType *derivativeValuesPtr;

    reg_getDisplacementFromDeformation(splineControlPoint);

    // Compute the bending energy values everywhere but at the boundary
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(splineControlPoint,splinePtrX,splinePtrY, derivativeValues, \
    basisXX, basisYY, basisXY) \
    private(a, b, i, index, x, derivativeValuesPtr, splineCoeffX, splineCoeffY, \
    XX_x, YY_x, XY_x, XX_y, YY_y, XY_y)
#endif
    for (y = 0; y < splineControlPoint->ny; y++) {
        derivativeValuesPtr = &derivativeValues[6 * y * splineControlPoint->nx];
        for (x = 0; x < splineControlPoint->nx; x++) {
            XX_x = 0, YY_x = 0, XY_x = 0;
            XX_y = 0, YY_y = 0, XY_y = 0;

            i = 0;
            for (b = -1; b < 2; b++) {
                for (a = -1; a < 2; a++) {
                    if (-1 < (x + a) && -1 < (y + b) && (x + a) < splineControlPoint->nx && (y + b) < splineControlPoint->ny) {
                        index = (y + b) * splineControlPoint->nx + x + a;
                        splineCoeffX = splinePtrX[index];
                        splineCoeffY = splinePtrY[index];
                        XX_x += basisXX[i] * splineCoeffX;
                        YY_x += basisYY[i] * splineCoeffX;
                        XY_x += basisXY[i] * splineCoeffX;

                        XX_y += basisXX[i] * splineCoeffY;
                        YY_y += basisYY[i] * splineCoeffY;
                        XY_y += basisXY[i] * splineCoeffY;
                    }
                    ++i;
                }
            }
            *derivativeValuesPtr++ = XX_x;
            *derivativeValuesPtr++ = XX_y;
            *derivativeValuesPtr++ = YY_x;
            *derivativeValuesPtr++ = YY_y;
            *derivativeValuesPtr++ = 2.f * XY_x;
            *derivativeValuesPtr++ = 2.f * XY_y;
        }
    }

    DataType *gradientXPtr = static_cast<DataType*>(gradientImage->data);
    DataType *gradientYPtr = &gradientXPtr[nodeNumber];

    DataType approxRatio = weight / static_cast<DataType>(nodeNumber);
    DataType gradientValue[2];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(splineControlPoint, derivativeValues, gradientXPtr, gradientYPtr, \
    basisXX, basisYY, basisXY, approxRatio) \
    private(index, a, X, Y, x, derivativeValuesPtr, gradientValue)
#endif
    for (y = 0; y < splineControlPoint->ny; y++) {
        index = y * splineControlPoint->nx;
        for (x = 0; x < splineControlPoint->nx; x++) {
            gradientValue[0] = gradientValue[1] = 0;
            a = 0;
            for (Y = y - 1; Y < y + 2; Y++) {
                for (X = x - 1; X < x + 2; X++) {
                    if (-1 < X && -1 < Y && X < splineControlPoint->nx && Y < splineControlPoint->ny) {
                        derivativeValuesPtr = &derivativeValues[6 * (Y * splineControlPoint->nx + X)];
                        gradientValue[0] += (*derivativeValuesPtr++) * basisXX[a];
                        gradientValue[1] += (*derivativeValuesPtr++) * basisXX[a];

                        gradientValue[0] += (*derivativeValuesPtr++) * basisYY[a];
                        gradientValue[1] += (*derivativeValuesPtr++) * basisYY[a];

                        gradientValue[0] += (*derivativeValuesPtr++) * basisXY[a];
                        gradientValue[1] += (*derivativeValuesPtr++) * basisXY[a];
                    }
                    a++;
                }
            }
            gradientXPtr[index] += approxRatio * gradientValue[0];
            gradientYPtr[index] += approxRatio * gradientValue[1];
            index++;
        }
    }
    reg_getDeformationFromDisplacement(splineControlPoint);
    free(derivativeValues);
}
/* *************************************************************** */
template<class DataType>
void reg_spline_approxBendingEnergyGradient3D(nifti_image *splineControlPoint,
                                              nifti_image *gradientImage,
                                              float weight) {
    const size_t nodeNumber = NiftiImage::calcVoxelNumber(splineControlPoint, 3);
    int a, b, c, x, y, z, X, Y, Z, index, i;

    // Create pointers to the spline coefficients
    DataType *splinePtrX = static_cast<DataType*>(splineControlPoint->data);
    DataType *splinePtrY = &splinePtrX[nodeNumber];
    DataType *splinePtrZ = &splinePtrY[nodeNumber];

    // get the constant basis values
    DataType basisXX[27], basisYY[27], basisZZ[27], basisXY[27], basisYZ[27], basisXZ[27];
    set_second_order_bspline_basis_values(basisXX, basisYY, basisZZ, basisXY, basisYZ, basisXZ);

    DataType splineCoeffX;
    DataType splineCoeffY;
    DataType splineCoeffZ;
    DataType XX_x, YY_x, ZZ_x, XY_x, YZ_x, XZ_x;
    DataType XX_y, YY_y, ZZ_y, XY_y, YZ_y, XZ_y;
    DataType XX_z, YY_z, ZZ_z, XY_z, YZ_z, XZ_z;

    DataType *derivativeValues = (DataType*)calloc(18 * nodeNumber, sizeof(DataType));
    DataType *derivativeValuesPtr;

    reg_getDisplacementFromDeformation(splineControlPoint);

    // Compute the bending energy values everywhere but at the boundary
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(splineControlPoint,splinePtrX,splinePtrY,splinePtrZ, derivativeValues, \
    basisXX, basisYY, basisZZ, basisXY, basisYZ, basisXZ) \
    private(a, b, c, i, index, x, y, derivativeValuesPtr, splineCoeffX, splineCoeffY, \
    splineCoeffZ, XX_x, YY_x, ZZ_x, XY_x, YZ_x, XZ_x, XX_y, YY_y, \
    ZZ_y, XY_y, YZ_y, XZ_y, XX_z, YY_z, ZZ_z, XY_z, YZ_z, XZ_z)
#endif
    for (z = 0; z < splineControlPoint->nz; z++) {
        derivativeValuesPtr = &derivativeValues[18 * z * splineControlPoint->ny * splineControlPoint->nx];
        for (y = 0; y < splineControlPoint->ny; y++) {
            for (x = 0; x < splineControlPoint->nx; x++) {
                XX_x = 0, YY_x = 0, ZZ_x = 0;
                XY_x = 0, YZ_x = 0, XZ_x = 0;
                XX_y = 0, YY_y = 0, ZZ_y = 0;
                XY_y = 0, YZ_y = 0, XZ_y = 0;
                XX_z = 0, YY_z = 0, ZZ_z = 0;
                XY_z = 0, YZ_z = 0, XZ_z = 0;

                i = 0;
                for (c = -1; c < 2; c++) {
                    for (b = -1; b < 2; b++) {
                        for (a = -1; a < 2; a++) {
                            if (-1 < (x + a) && -1 < (y + b) && -1 < (z + c) && (x + a) < splineControlPoint->nx &&
                                (y + b) < splineControlPoint->ny && (z + c) < splineControlPoint->nz) {
                                index = ((z + c) * splineControlPoint->ny + y + b) * splineControlPoint->nx + x + a;
                                splineCoeffX = splinePtrX[index];
                                splineCoeffY = splinePtrY[index];
                                splineCoeffZ = splinePtrZ[index];
                                XX_x += basisXX[i] * splineCoeffX;
                                YY_x += basisYY[i] * splineCoeffX;
                                ZZ_x += basisZZ[i] * splineCoeffX;
                                XY_x += basisXY[i] * splineCoeffX;
                                YZ_x += basisYZ[i] * splineCoeffX;
                                XZ_x += basisXZ[i] * splineCoeffX;

                                XX_y += basisXX[i] * splineCoeffY;
                                YY_y += basisYY[i] * splineCoeffY;
                                ZZ_y += basisZZ[i] * splineCoeffY;
                                XY_y += basisXY[i] * splineCoeffY;
                                YZ_y += basisYZ[i] * splineCoeffY;
                                XZ_y += basisXZ[i] * splineCoeffY;

                                XX_z += basisXX[i] * splineCoeffZ;
                                YY_z += basisYY[i] * splineCoeffZ;
                                ZZ_z += basisZZ[i] * splineCoeffZ;
                                XY_z += basisXY[i] * splineCoeffZ;
                                YZ_z += basisYZ[i] * splineCoeffZ;
                                XZ_z += basisXZ[i] * splineCoeffZ;
                            }
                            ++i;
                        }
                    }
                }
                *derivativeValuesPtr++ = XX_x;
                *derivativeValuesPtr++ = XX_y;
                *derivativeValuesPtr++ = XX_z;
                *derivativeValuesPtr++ = YY_x;
                *derivativeValuesPtr++ = YY_y;
                *derivativeValuesPtr++ = YY_z;
                *derivativeValuesPtr++ = ZZ_x;
                *derivativeValuesPtr++ = ZZ_y;
                *derivativeValuesPtr++ = ZZ_z;
                *derivativeValuesPtr++ = (DataType)(2.0 * XY_x);
                *derivativeValuesPtr++ = (DataType)(2.0 * XY_y);
                *derivativeValuesPtr++ = (DataType)(2.0 * XY_z);
                *derivativeValuesPtr++ = (DataType)(2.0 * YZ_x);
                *derivativeValuesPtr++ = (DataType)(2.0 * YZ_y);
                *derivativeValuesPtr++ = (DataType)(2.0 * YZ_z);
                *derivativeValuesPtr++ = (DataType)(2.0 * XZ_x);
                *derivativeValuesPtr++ = (DataType)(2.0 * XZ_y);
                *derivativeValuesPtr++ = (DataType)(2.0 * XZ_z);
            }
        }
    }

    DataType *gradientXPtr = static_cast<DataType*>(gradientImage->data);
    DataType *gradientYPtr = &gradientXPtr[nodeNumber];
    DataType *gradientZPtr = &gradientYPtr[nodeNumber];

    DataType approxRatio = weight / static_cast<DataType>(nodeNumber);
    DataType gradientValue[3];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(splineControlPoint, derivativeValues, gradientXPtr, gradientYPtr, gradientZPtr, \
    basisXX, basisYY, basisZZ, basisXY, basisYZ, basisXZ, approxRatio) \
    private(index, a, X, Y, Z, x, y, derivativeValuesPtr, gradientValue)
#endif
    for (z = 0; z < splineControlPoint->nz; z++) {
        index = z * splineControlPoint->nx * splineControlPoint->ny;
        for (y = 0; y < splineControlPoint->ny; y++) {
            for (x = 0; x < splineControlPoint->nx; x++) {
                gradientValue[0] = gradientValue[1] = gradientValue[2] = 0;
                a = 0;
                for (Z = z - 1; Z < z + 2; Z++) {
                    for (Y = y - 1; Y < y + 2; Y++) {
                        for (X = x - 1; X < x + 2; X++) {
                            if (-1 < X && -1 < Y && -1 < Z && X < splineControlPoint->nx && Y < splineControlPoint->ny && Z < splineControlPoint->nz) {
                                derivativeValuesPtr = &derivativeValues[18 * ((Z * splineControlPoint->ny + Y) * splineControlPoint->nx + X)];
                                gradientValue[0] += (*derivativeValuesPtr++) * basisXX[a];
                                gradientValue[1] += (*derivativeValuesPtr++) * basisXX[a];
                                gradientValue[2] += (*derivativeValuesPtr++) * basisXX[a];

                                gradientValue[0] += (*derivativeValuesPtr++) * basisYY[a];
                                gradientValue[1] += (*derivativeValuesPtr++) * basisYY[a];
                                gradientValue[2] += (*derivativeValuesPtr++) * basisYY[a];

                                gradientValue[0] += (*derivativeValuesPtr++) * basisZZ[a];
                                gradientValue[1] += (*derivativeValuesPtr++) * basisZZ[a];
                                gradientValue[2] += (*derivativeValuesPtr++) * basisZZ[a];

                                gradientValue[0] += (*derivativeValuesPtr++) * basisXY[a];
                                gradientValue[1] += (*derivativeValuesPtr++) * basisXY[a];
                                gradientValue[2] += (*derivativeValuesPtr++) * basisXY[a];

                                gradientValue[0] += (*derivativeValuesPtr++) * basisYZ[a];
                                gradientValue[1] += (*derivativeValuesPtr++) * basisYZ[a];
                                gradientValue[2] += (*derivativeValuesPtr++) * basisYZ[a];

                                gradientValue[0] += (*derivativeValuesPtr++) * basisXZ[a];
                                gradientValue[1] += (*derivativeValuesPtr++) * basisXZ[a];
                                gradientValue[2] += (*derivativeValuesPtr++) * basisXZ[a];
                            }
                            a++;
                        }
                    }
                }
                gradientXPtr[index] += approxRatio * gradientValue[0];
                gradientYPtr[index] += approxRatio * gradientValue[1];
                gradientZPtr[index] += approxRatio * gradientValue[2];
                index++;
            }
        }
    }
    free(derivativeValues);
    reg_getDeformationFromDisplacement(splineControlPoint);
}
/* *************************************************************** */
void reg_spline_approxBendingEnergyGradient(nifti_image *splineControlPoint,
                                            nifti_image *gradientImage,
                                            float weight) {
    if (splineControlPoint->datatype != gradientImage->datatype)
        NR_FATAL_ERROR("The input images are expected to have the same type");

    if (splineControlPoint->nz == 1) {
        switch (splineControlPoint->datatype) {
        case NIFTI_TYPE_FLOAT32:
            reg_spline_approxBendingEnergyGradient2D<float>(splineControlPoint, gradientImage, weight);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_spline_approxBendingEnergyGradient2D<double>(splineControlPoint, gradientImage, weight);
            break;
        default:
            NR_FATAL_ERROR("Only implemented for single or double precision images");
        }
    } else {
        switch (splineControlPoint->datatype) {
        case NIFTI_TYPE_FLOAT32:
            reg_spline_approxBendingEnergyGradient3D<float>(splineControlPoint, gradientImage, weight);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_spline_approxBendingEnergyGradient3D<double>(splineControlPoint, gradientImage, weight);
            break;
        default:
            NR_FATAL_ERROR("Only implemented for single or double precision images");
        }
    }
}
/* *************************************************************** */
template <class DataType>
double reg_spline_approxLinearEnergyValue2D(const nifti_image *splineControlPoint) {
    const size_t nodeNumber = NiftiImage::calcVoxelNumber(splineControlPoint, 2);
    int a, b, x, y, i, index;

    double constraintValue = 0;
    double currentValue;

    // Create pointers to the spline coefficients
    const DataType *splinePtrX = static_cast<DataType*>(splineControlPoint->data);
    const DataType *splinePtrY = &splinePtrX[nodeNumber];

    // Store the basis values since they are constant as the value is approximated
    // at the control point positions only
    DataType basisX[9], basisY[9];
    set_first_order_basis_values(basisX, basisY);

    DataType splineCoeffX;
    DataType splineCoeffY;

    mat33 matrix, r;

    // Matrix to use to convert the gradient from mm to voxel
    mat33 reorientation;
    if (splineControlPoint->sform_code > 0)
        reorientation = reg_mat44_to_mat33(&splineControlPoint->sto_ijk);
    else reorientation = reg_mat44_to_mat33(&splineControlPoint->qto_ijk);

#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(splinePtrX, splinePtrY, splineControlPoint, \
    basisX, basisY, reorientation) \
    private(x, a, b, i, index, matrix, r, \
    splineCoeffX, splineCoeffY, currentValue) \
    reduction(+:constraintValue)
#endif
    for (y = 1; y < splineControlPoint->ny - 1; ++y) {
        for (x = 1; x < splineControlPoint->nx - 1; ++x) {
            memset(&matrix, 0, sizeof(mat33));
            matrix.m[2][2] = 1;

            i = 0;
            for (b = -1; b < 2; b++) {
                for (a = -1; a < 2; a++) {
                    index = (y + b) * splineControlPoint->nx + x + a;
                    splineCoeffX = splinePtrX[index];
                    splineCoeffY = splinePtrY[index];
                    matrix.m[0][0] += static_cast<float>(basisX[i] * splineCoeffX);
                    matrix.m[1][0] += static_cast<float>(basisY[i] * splineCoeffX);
                    matrix.m[0][1] += static_cast<float>(basisX[i] * splineCoeffY);
                    matrix.m[1][1] += static_cast<float>(basisY[i] * splineCoeffY);
                    ++i;
                }
            }
            // Convert from mm to voxel
            matrix = nifti_mat33_mul(reorientation, matrix);
            // Removing the rotation component
            r = nifti_mat33_inverse(nifti_mat33_polar(matrix));
            matrix = nifti_mat33_mul(r, matrix);
            // Convert to displacement
            --matrix.m[0][0];
            --matrix.m[1][1];

            currentValue = 0;
            for (b = 0; b < 2; b++) {
                for (a = 0; a < 2; a++) {
                    currentValue += Square(0.5 * (matrix.m[a][b] + matrix.m[b][a])); // symmetric part
                }
            }
            constraintValue += currentValue;
        }
    }
    return constraintValue / static_cast<double>(splineControlPoint->nvox);
}
/* *************************************************************** */
template <class DataType>
double reg_spline_approxLinearEnergyValue3D(const nifti_image *splineControlPoint) {
    const size_t nodeNumber = NiftiImage::calcVoxelNumber(splineControlPoint, 3);
    int a, b, c, x, y, z, i, index;

    double constraintValue = 0;
    double currentValue;

    // Create pointers to the spline coefficients
    const DataType *splinePtrX = static_cast<DataType*>(splineControlPoint->data);
    const DataType *splinePtrY = &splinePtrX[nodeNumber];
    const DataType *splinePtrZ = &splinePtrY[nodeNumber];

    // Store the basis values since they are constant as the value is approximated
    // at the control point positions only
    DataType basisX[27], basisY[27], basisZ[27];
    set_first_order_basis_values(basisX, basisY, basisZ);

    DataType splineCoeffX;
    DataType splineCoeffY;
    DataType splineCoeffZ;

    mat33 matrix, r;

    // Matrix to use to convert the gradient from mm to voxel
    mat33 reorientation;
    if (splineControlPoint->sform_code > 0)
        reorientation = reg_mat44_to_mat33(&splineControlPoint->sto_ijk);
    else reorientation = reg_mat44_to_mat33(&splineControlPoint->qto_ijk);

#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(splinePtrX, splinePtrY, splinePtrZ, splineControlPoint, \
    basisX, basisY, basisZ, reorientation) \
    private(x, y, a, b, c, i, index, matrix, r, \
    splineCoeffX, splineCoeffY, splineCoeffZ, currentValue) \
    reduction(+:constraintValue)
#endif
    for (z = 1; z < splineControlPoint->nz - 1; ++z) {
        for (y = 1; y < splineControlPoint->ny - 1; ++y) {
            for (x = 1; x < splineControlPoint->nx - 1; ++x) {
                memset(&matrix, 0, sizeof(mat33));

                i = 0;
                for (c = -1; c < 2; c++) {
                    for (b = -1; b < 2; b++) {
                        for (a = -1; a < 2; a++) {
                            index = ((z + c) * splineControlPoint->ny + y + b) * splineControlPoint->nx + x + a;
                            splineCoeffX = splinePtrX[index];
                            splineCoeffY = splinePtrY[index];
                            splineCoeffZ = splinePtrZ[index];

                            matrix.m[0][0] += static_cast<float>(basisX[i] * splineCoeffX);
                            matrix.m[1][0] += static_cast<float>(basisY[i] * splineCoeffX);
                            matrix.m[2][0] += static_cast<float>(basisZ[i] * splineCoeffX);

                            matrix.m[0][1] += static_cast<float>(basisX[i] * splineCoeffY);
                            matrix.m[1][1] += static_cast<float>(basisY[i] * splineCoeffY);
                            matrix.m[2][1] += static_cast<float>(basisZ[i] * splineCoeffY);

                            matrix.m[0][2] += static_cast<float>(basisX[i] * splineCoeffZ);
                            matrix.m[1][2] += static_cast<float>(basisY[i] * splineCoeffZ);
                            matrix.m[2][2] += static_cast<float>(basisZ[i] * splineCoeffZ);
                            ++i;
                        }
                    }
                }
                // Convert from mm to voxel
                matrix = nifti_mat33_mul(reorientation, matrix);
                // Removing the rotation component
                r = nifti_mat33_inverse(nifti_mat33_polar(matrix));
                matrix = nifti_mat33_mul(r, matrix);
                // Convert to displacement
                --matrix.m[0][0];
                --matrix.m[1][1];
                --matrix.m[2][2];

                currentValue = 0;
                for (b = 0; b < 3; b++) {
                    for (a = 0; a < 3; a++) {
                        currentValue += Square(0.5 * (matrix.m[a][b] + matrix.m[b][a])); // symmetric part
                    }
                }
                constraintValue += currentValue;
            }
        }
    }
    return constraintValue / static_cast<double>(splineControlPoint->nvox);
}
/* *************************************************************** */
double reg_spline_approxLinearEnergy(const nifti_image *splineControlPoint) {
    if (splineControlPoint->nz > 1) {
        switch (splineControlPoint->datatype) {
        case NIFTI_TYPE_FLOAT32:
            return reg_spline_approxLinearEnergyValue3D<float>(splineControlPoint);
        case NIFTI_TYPE_FLOAT64:
            return reg_spline_approxLinearEnergyValue3D<double>(splineControlPoint);
        default:
            NR_FATAL_ERROR("Only implemented for single or double precision images");
            return 0;
        }
    } else {
        switch (splineControlPoint->datatype) {
        case NIFTI_TYPE_FLOAT32:
            return reg_spline_approxLinearEnergyValue2D<float>(splineControlPoint);
        case NIFTI_TYPE_FLOAT64:
            return reg_spline_approxLinearEnergyValue2D<double>(splineControlPoint);
        default:
            NR_FATAL_ERROR("Only implemented for single or double precision images");
            return 0;
        }
    }
}
/* *************************************************************** */
template <class DataType>
double reg_spline_linearEnergyValue2D(const nifti_image *referenceImage,
                                      const nifti_image *splineControlPoint) {
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 2);
    int a, b, x, y, index, xPre, yPre;
    DataType basis;

    const DataType gridVoxelSpacing[2] = {
        splineControlPoint->dx / referenceImage->dx,
        splineControlPoint->dy / referenceImage->dy
    };

    double constraintValue = 0;
    double currentValue;

    // Create pointers to the spline coefficients
    const size_t nodeNumber = NiftiImage::calcVoxelNumber(splineControlPoint, 3);
    const DataType *splinePtrX = static_cast<DataType*>(splineControlPoint->data);
    const DataType *splinePtrY = &splinePtrX[nodeNumber];
    DataType splineCoeffX, splineCoeffY;

    // Store the basis values since they are constant as the value is approximated
    // at the control point positions only
    DataType basisX[4], basisY[4];
    DataType firstX[4], firstY[4];

    mat33 matrix, r;

    // Matrix to use to convert the gradient from mm to voxel
    mat33 reorientation;
    if (splineControlPoint->sform_code > 0)
        reorientation = reg_mat44_to_mat33(&splineControlPoint->sto_ijk);
    else reorientation = reg_mat44_to_mat33(&splineControlPoint->qto_ijk);


    for (y = 0; y < referenceImage->ny; ++y) {
        yPre = static_cast<int>(static_cast<DataType>(y) / gridVoxelSpacing[1]);
        basis = static_cast<DataType>(y) / gridVoxelSpacing[1] - static_cast<DataType>(yPre);
        if (basis < 0) basis = 0; //rounding error
        get_BSplineBasisValues<DataType>(basis, basisY, firstY);

        for (x = 0; x < referenceImage->nx; ++x) {
            xPre = static_cast<int>(static_cast<DataType>(x) / gridVoxelSpacing[0]);
            basis = static_cast<DataType>(x) / gridVoxelSpacing[0] - static_cast<DataType>(xPre);
            if (basis < 0) basis = 0; //rounding error
            get_BSplineBasisValues<DataType>(basis, basisX, firstX);

            memset(&matrix, 0, sizeof(mat33));

            for (b = 0; b < 4; b++) {
                for (a = 0; a < 4; a++) {
                    index = (yPre + b) * splineControlPoint->nx + xPre + a;
                    splineCoeffX = splinePtrX[index];
                    splineCoeffY = splinePtrY[index];

                    matrix.m[0][0] += static_cast<float>(firstX[a] * basisY[b] * splineCoeffX);
                    matrix.m[1][0] += static_cast<float>(basisX[a] * firstY[b] * splineCoeffX);

                    matrix.m[0][1] += static_cast<float>(firstX[a] * basisY[b] * splineCoeffY);
                    matrix.m[1][1] += static_cast<float>(basisX[a] * firstY[b] * splineCoeffY);
                }
            }
            // Convert from mm to voxel
            matrix = nifti_mat33_mul(reorientation, matrix);
            // Removing the rotation component
            r = nifti_mat33_inverse(nifti_mat33_polar(matrix));
            matrix = nifti_mat33_mul(r, matrix);
            // Convert to displacement
            --matrix.m[0][0];
            --matrix.m[1][1];

            currentValue = 0;
            for (b = 0; b < 2; b++) {
                for (a = 0; a < 2; a++) {
                    currentValue += Square(0.5 * (matrix.m[a][b] + matrix.m[b][a])); // symmetric part
                }
            }
            constraintValue += currentValue;
        }
    }
    return constraintValue / static_cast<double>(voxelNumber * 2);
}
/* *************************************************************** */
template <class DataType>
double reg_spline_linearEnergyValue3D(const nifti_image *referenceImage,
                                      const nifti_image *splineControlPoint) {
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);
    int a, b, c, x, y, z, index, xPre, yPre, zPre;
    DataType basis;

    const DataType gridVoxelSpacing[3] = {
        splineControlPoint->dx / referenceImage->dx,
        splineControlPoint->dy / referenceImage->dy,
        splineControlPoint->dz / referenceImage->dz
    };

    double constraintValue = 0;
    double currentValue;

    // Create pointers to the spline coefficients
    const size_t nodeNumber = NiftiImage::calcVoxelNumber(splineControlPoint, 3);
    const DataType *splinePtrX = static_cast<DataType*>(splineControlPoint->data);
    const DataType *splinePtrY = &splinePtrX[nodeNumber];
    const DataType *splinePtrZ = &splinePtrY[nodeNumber];
    DataType splineCoeffX, splineCoeffY, splineCoeffZ;

    // Store the basis values since they are constant as the value is approximated
    // at the control point positions only
    DataType basisX[4], basisY[4], basisZ[4];
    DataType firstX[4], firstY[4], firstZ[4];

    mat33 matrix, r;

    // Matrix to use to convert the gradient from mm to voxel
    mat33 reorientation;
    if (splineControlPoint->sform_code > 0)
        reorientation = reg_mat44_to_mat33(&splineControlPoint->sto_ijk);
    else reorientation = reg_mat44_to_mat33(&splineControlPoint->qto_ijk);

    for (z = 0; z < referenceImage->nz; ++z) {
        zPre = static_cast<int>(static_cast<DataType>(z) / gridVoxelSpacing[2]);
        basis = static_cast<DataType>(z) / gridVoxelSpacing[2] - static_cast<DataType>(zPre);
        if (basis < 0) basis = 0; //rounding error
        get_BSplineBasisValues<DataType>(basis, basisZ, firstZ);

        for (y = 0; y < referenceImage->ny; ++y) {
            yPre = static_cast<int>(static_cast<DataType>(y) / gridVoxelSpacing[1]);
            basis = static_cast<DataType>(y) / gridVoxelSpacing[1] - static_cast<DataType>(yPre);
            if (basis < 0) basis = 0; //rounding error
            get_BSplineBasisValues<DataType>(basis, basisY, firstY);

            for (x = 0; x < referenceImage->nx; ++x) {
                xPre = static_cast<int>(static_cast<DataType>(x) / gridVoxelSpacing[0]);
                basis = static_cast<DataType>(x) / gridVoxelSpacing[0] - static_cast<DataType>(xPre);
                if (basis < 0) basis = 0; //rounding error
                get_BSplineBasisValues<DataType>(basis, basisX, firstX);

                memset(&matrix, 0, sizeof(mat33));

                for (c = 0; c < 4; c++) {
                    for (b = 0; b < 4; b++) {
                        for (a = 0; a < 4; a++) {
                            index = ((zPre + c) * splineControlPoint->ny + yPre + b) * splineControlPoint->nx + xPre + a;
                            splineCoeffX = splinePtrX[index];
                            splineCoeffY = splinePtrY[index];
                            splineCoeffZ = splinePtrZ[index];

                            matrix.m[0][0] += static_cast<float>(firstX[a] * basisY[b] * basisZ[c] * splineCoeffX);
                            matrix.m[1][0] += static_cast<float>(basisX[a] * firstY[b] * basisZ[c] * splineCoeffX);
                            matrix.m[2][0] += static_cast<float>(basisX[a] * basisY[b] * firstZ[c] * splineCoeffX);

                            matrix.m[0][1] += static_cast<float>(firstX[a] * basisY[b] * basisZ[c] * splineCoeffY);
                            matrix.m[1][1] += static_cast<float>(basisX[a] * firstY[b] * basisZ[c] * splineCoeffY);
                            matrix.m[2][1] += static_cast<float>(basisX[a] * basisY[b] * firstZ[c] * splineCoeffY);

                            matrix.m[0][2] += static_cast<float>(firstX[a] * basisY[b] * basisZ[c] * splineCoeffZ);
                            matrix.m[1][2] += static_cast<float>(basisX[a] * firstY[b] * basisZ[c] * splineCoeffZ);
                            matrix.m[2][2] += static_cast<float>(basisX[a] * basisY[b] * firstZ[c] * splineCoeffZ);
                        }
                    }
                }
                // Convert from mm to voxel
                matrix = nifti_mat33_mul(reorientation, matrix);
                // Removing the rotation component
                r = nifti_mat33_inverse(nifti_mat33_polar(matrix));
                matrix = nifti_mat33_mul(r, matrix);
                // Convert to displacement
                --matrix.m[0][0];
                --matrix.m[1][1];
                --matrix.m[2][2];

                currentValue = 0;
                for (b = 0; b < 3; b++) {
                    for (a = 0; a < 3; a++) {
                        currentValue += Square(0.5 * (matrix.m[a][b] + matrix.m[b][a])); // symmetric part
                    }
                }
                constraintValue += currentValue;
            }
        }
    }
    return constraintValue / static_cast<double>(voxelNumber * 3);
}
/* *************************************************************** */
double reg_spline_linearEnergy(const nifti_image *referenceImage,
                               const nifti_image *splineControlPoint) {
    if (splineControlPoint->nz > 1) {
        switch (splineControlPoint->datatype) {
        case NIFTI_TYPE_FLOAT32:
            return reg_spline_linearEnergyValue3D<float>(referenceImage, splineControlPoint);
        case NIFTI_TYPE_FLOAT64:
            return reg_spline_linearEnergyValue3D<double>(referenceImage, splineControlPoint);
        default:
            NR_FATAL_ERROR("Only implemented for single or double precision images");
            return 0;
        }
    } else {
        switch (splineControlPoint->datatype) {
        case NIFTI_TYPE_FLOAT32:
            return reg_spline_linearEnergyValue2D<float>(referenceImage, splineControlPoint);
        case NIFTI_TYPE_FLOAT64:
            return reg_spline_linearEnergyValue2D<double>(referenceImage, splineControlPoint);
        default:
            NR_FATAL_ERROR("Only implemented for single or double precision images");
            return 0;
        }
    }
}
/* *************************************************************** */
template <class DataType>
void reg_spline_linearEnergyGradient2D(const nifti_image *referenceImage,
                                       const nifti_image *splineControlPoint,
                                       nifti_image *gradientImage,
                                       float weight) {
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 2);
    int a, b, x, y, index, xPre, yPre;
    DataType basis;

    const DataType gridVoxelSpacing[2] = {
        splineControlPoint->dx / referenceImage->dx,
        splineControlPoint->dy / referenceImage->dy
    };

    // Create pointers to the spline coefficients
    const size_t nodeNumber = NiftiImage::calcVoxelNumber(splineControlPoint, 3);
    const DataType *splinePtrX = static_cast<DataType*>(splineControlPoint->data);
    const DataType *splinePtrY = &splinePtrX[nodeNumber];
    DataType splineCoeffX, splineCoeffY;

    // Store the basis values since they are constant as the value is approximated
    // at the control point positions only
    DataType basisX[4], basisY[4];
    DataType firstX[4], firstY[4];

    mat33 matrix, r;

    DataType *gradientXPtr = static_cast<DataType*>(gradientImage->data);
    DataType *gradientYPtr = &gradientXPtr[nodeNumber];

    DataType approxRatio = weight / static_cast<DataType>(voxelNumber);
    DataType gradValues[2];

    // Matrix to use to convert the gradient from mm to voxel
    mat33 reorientation;
    if (splineControlPoint->sform_code > 0)
        reorientation = reg_mat44_to_mat33(&splineControlPoint->sto_ijk);
    else reorientation = reg_mat44_to_mat33(&splineControlPoint->qto_ijk);
    mat33 invReorientation = nifti_mat33_inverse(reorientation);

    // Loop over all voxels
    for (y = 0; y < referenceImage->ny; ++y) {
        yPre = static_cast<int>(static_cast<DataType>(y) / gridVoxelSpacing[1]);
        basis = static_cast<DataType>(y) / gridVoxelSpacing[1] - static_cast<DataType>(yPre);
        if (basis < 0) basis = 0; //rounding error
        get_BSplineBasisValues<DataType>(basis, basisY, firstY);

        for (x = 0; x < referenceImage->nx; ++x) {
            xPre = static_cast<int>(static_cast<DataType>(x) / gridVoxelSpacing[0]);
            basis = static_cast<DataType>(x) / gridVoxelSpacing[0] - static_cast<DataType>(xPre);
            if (basis < 0) basis = 0; //rounding error
            get_BSplineBasisValues<DataType>(basis, basisX, firstX);

            memset(&matrix, 0, sizeof(mat33));

            for (b = 0; b < 4; b++) {
                for (a = 0; a < 4; a++) {
                    index = (yPre + b) * splineControlPoint->nx + xPre + a;
                    splineCoeffX = splinePtrX[index];
                    splineCoeffY = splinePtrY[index];

                    matrix.m[0][0] += static_cast<float>(firstX[a] * basisY[b] * splineCoeffX);
                    matrix.m[1][0] += static_cast<float>(basisX[a] * firstY[b] * splineCoeffX);

                    matrix.m[0][1] += static_cast<float>(firstX[a] * basisY[b] * splineCoeffY);
                    matrix.m[1][1] += static_cast<float>(basisX[a] * firstY[b] * splineCoeffY);
                }
            }
            // Convert from mm to voxel
            matrix = nifti_mat33_mul(reorientation, matrix);
            // Removing the rotation component
            r = nifti_mat33_inverse(nifti_mat33_polar(matrix));
            matrix = nifti_mat33_mul(r, matrix);
            // Convert to displacement
            --matrix.m[0][0];
            --matrix.m[1][1];
            for (b = 0; b < 4; b++) {
                for (a = 0; a < 4; a++) {
                    index = (yPre + b) * splineControlPoint->nx + xPre + a;
                    gradValues[0] = -2.f * matrix.m[0][0] * firstX[3 - a] * basisY[3 - b];
                    gradValues[1] = -2.f * matrix.m[1][1] * basisX[3 - a] * firstY[3 - b];
                    gradientXPtr[index] += approxRatio * (invReorientation.m[0][0] * gradValues[0] +
                                                          invReorientation.m[0][1] * gradValues[1]);
                    gradientYPtr[index] += approxRatio * (invReorientation.m[1][0] * gradValues[0] +
                                                          invReorientation.m[1][1] * gradValues[1]);
                } // a
            } // b
        }
    }
}
/* *************************************************************** */
template <class DataType>
void reg_spline_linearEnergyGradient3D(const nifti_image *referenceImage,
                                       const nifti_image *splineControlPoint,
                                       nifti_image *gradientImage,
                                       float weight) {
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(referenceImage, 3);
    int a, b, c, x, y, z, index, xPre, yPre, zPre;
    DataType basis;

    const DataType gridVoxelSpacing[3] = {
        splineControlPoint->dx / referenceImage->dx,
        splineControlPoint->dy / referenceImage->dy,
        splineControlPoint->dz / referenceImage->dz
    };

    // Create pointers to the spline coefficients
    const size_t nodeNumber = NiftiImage::calcVoxelNumber(splineControlPoint, 3);
    const DataType *splinePtrX = static_cast<DataType*>(splineControlPoint->data);
    const DataType *splinePtrY = &splinePtrX[nodeNumber];
    const DataType *splinePtrZ = &splinePtrY[nodeNumber];
    DataType splineCoeffX, splineCoeffY, splineCoeffZ;

    // Store the basis values since they are constant as the value is approximated
    // at the control point positions only
    DataType basisX[4], basisY[4], basisZ[4];
    DataType firstX[4], firstY[4], firstZ[4];

    mat33 matrix, r;

    DataType *gradientXPtr = static_cast<DataType*>(gradientImage->data);
    DataType *gradientYPtr = &gradientXPtr[nodeNumber];
    DataType *gradientZPtr = &gradientYPtr[nodeNumber];

    DataType approxRatio = weight / static_cast<DataType>(voxelNumber);
    DataType gradValues[3];

    // Matrix to use to convert the gradient from mm to voxel
    mat33 reorientation;
    if (splineControlPoint->sform_code > 0)
        reorientation = reg_mat44_to_mat33(&splineControlPoint->sto_ijk);
    else reorientation = reg_mat44_to_mat33(&splineControlPoint->qto_ijk);
    mat33 invReorientation = nifti_mat33_inverse(reorientation);

    // Loop over all voxels
    for (z = 0; z < referenceImage->nz; ++z) {
        zPre = static_cast<int>(static_cast<DataType>(z) / gridVoxelSpacing[2]);
        basis = static_cast<DataType>(z) / gridVoxelSpacing[2] - static_cast<DataType>(zPre);
        if (basis < 0) basis = 0; //rounding error
        get_BSplineBasisValues<DataType>(basis, basisZ, firstZ);

        for (y = 0; y < referenceImage->ny; ++y) {
            yPre = static_cast<int>(static_cast<DataType>(y) / gridVoxelSpacing[1]);
            basis = static_cast<DataType>(y) / gridVoxelSpacing[1] - static_cast<DataType>(yPre);
            if (basis < 0) basis = 0; //rounding error
            get_BSplineBasisValues<DataType>(basis, basisY, firstY);

            for (x = 0; x < referenceImage->nx; ++x) {
                xPre = static_cast<int>(static_cast<DataType>(x) / gridVoxelSpacing[0]);
                basis = static_cast<DataType>(x) / gridVoxelSpacing[0] - static_cast<DataType>(xPre);
                if (basis < 0) basis = 0; //rounding error
                get_BSplineBasisValues<DataType>(basis, basisX, firstX);

                memset(&matrix, 0, sizeof(mat33));

                for (c = 0; c < 4; c++) {
                    for (b = 0; b < 4; b++) {
                        for (a = 0; a < 4; a++) {
                            index = ((zPre + c) * splineControlPoint->ny + yPre + b) * splineControlPoint->nx + xPre + a;
                            splineCoeffX = splinePtrX[index];
                            splineCoeffY = splinePtrY[index];
                            splineCoeffZ = splinePtrZ[index];

                            matrix.m[0][0] += static_cast<float>(firstX[a] * basisY[b] * basisZ[c] * splineCoeffX);
                            matrix.m[1][0] += static_cast<float>(basisX[a] * firstY[b] * basisZ[c] * splineCoeffX);
                            matrix.m[2][0] += static_cast<float>(basisX[a] * basisY[b] * firstZ[c] * splineCoeffX);

                            matrix.m[0][1] += static_cast<float>(firstX[a] * basisY[b] * basisZ[c] * splineCoeffY);
                            matrix.m[1][1] += static_cast<float>(basisX[a] * firstY[b] * basisZ[c] * splineCoeffY);
                            matrix.m[2][1] += static_cast<float>(basisX[a] * basisY[b] * firstZ[c] * splineCoeffY);

                            matrix.m[0][2] += static_cast<float>(firstX[a] * basisY[b] * basisZ[c] * splineCoeffZ);
                            matrix.m[1][2] += static_cast<float>(basisX[a] * firstY[b] * basisZ[c] * splineCoeffZ);
                            matrix.m[2][2] += static_cast<float>(basisX[a] * basisY[b] * firstZ[c] * splineCoeffZ);
                        }
                    }
                }
                // Convert from mm to voxel
                matrix = nifti_mat33_mul(reorientation, matrix);
                // Removing the rotation component
                r = nifti_mat33_inverse(nifti_mat33_polar(matrix));
                matrix = nifti_mat33_mul(r, matrix);
                // Convert to displacement
                --matrix.m[0][0];
                --matrix.m[1][1];
                --matrix.m[2][2];
                for (c = 0; c < 4; c++) {
                    for (b = 0; b < 4; b++) {
                        for (a = 0; a < 4; a++) {
                            index = ((zPre + c) * splineControlPoint->ny + yPre + b) * splineControlPoint->nx + xPre + a;
                            gradValues[0] = -2.f * matrix.m[0][0] * firstX[3 - a] * basisY[3 - b] * basisZ[3 - c];
                            gradValues[1] = -2.f * matrix.m[1][1] * basisX[3 - a] * firstY[3 - b] * basisZ[3 - c];
                            gradValues[2] = -2.f * matrix.m[2][2] * basisX[3 - a] * basisY[3 - b] * firstZ[3 - c];
                            gradientXPtr[index] += approxRatio * (invReorientation.m[0][0] * gradValues[0] +
                                                                  invReorientation.m[0][1] * gradValues[1] +
                                                                  invReorientation.m[0][2] * gradValues[2]);
                            gradientYPtr[index] += approxRatio * (invReorientation.m[1][0] * gradValues[0] +
                                                                  invReorientation.m[1][1] * gradValues[1] +
                                                                  invReorientation.m[1][2] * gradValues[2]);
                            gradientZPtr[index] += approxRatio * (invReorientation.m[2][0] * gradValues[0] +
                                                                  invReorientation.m[2][1] * gradValues[1] +
                                                                  invReorientation.m[2][2] * gradValues[2]);
                        } // a
                    } // b
                } // c
            } // x
        } // y
    } // z
}
/* *************************************************************** */
void reg_spline_linearEnergyGradient(const nifti_image *referenceImage,
                                     const nifti_image *splineControlPoint,
                                     nifti_image *gradientImage,
                                     float weight) {
    if (splineControlPoint->datatype != gradientImage->datatype)
        NR_FATAL_ERROR("Input images are expected to have the same datatype");

    if (splineControlPoint->nz > 1) {
        switch (splineControlPoint->datatype) {
        case NIFTI_TYPE_FLOAT32:
            reg_spline_linearEnergyGradient3D<float>(referenceImage, splineControlPoint, gradientImage, weight);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_spline_linearEnergyGradient3D<double>(referenceImage, splineControlPoint, gradientImage, weight);
            break;
        default:
            NR_FATAL_ERROR("Only implemented for single or double precision images");
        }
    } else {
        switch (splineControlPoint->datatype) {
        case NIFTI_TYPE_FLOAT32:
            reg_spline_linearEnergyGradient2D<float>(referenceImage, splineControlPoint, gradientImage, weight);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_spline_linearEnergyGradient2D<double>(referenceImage, splineControlPoint, gradientImage, weight);
            break;
        default:
            NR_FATAL_ERROR("Only implemented for single or double precision images");
        }
    }
}
/* *************************************************************** */
template <class DataType>
void reg_spline_approxLinearEnergyGradient2D(const nifti_image *splineControlPoint,
                                             nifti_image *gradientImage,
                                             float weight) {
    const size_t nodeNumber = NiftiImage::calcVoxelNumber(splineControlPoint, 2);

    // Create the pointers
    const DataType *splinePtrX = static_cast<DataType*>(splineControlPoint->data);
    const DataType *splinePtrY = &splinePtrX[nodeNumber];
    DataType *gradientXPtr = static_cast<DataType*>(gradientImage->data);
    DataType *gradientYPtr = &gradientXPtr[nodeNumber];

    // Store the basis values since they are constant as the value is approximated
    // at the control point positions only
    DataType basisX[9], basisY[9];
    set_first_order_basis_values(basisX, basisY);

    // Matrix to use to convert the gradient from mm to voxel
    const mat33 reorientation = reg_mat44_to_mat33(splineControlPoint->sform_code > 0 ? &splineControlPoint->sto_ijk : &splineControlPoint->qto_ijk);
    const mat33 invReorientation = nifti_mat33_inverse(reorientation);

    const DataType approxRatio = weight / static_cast<DataType>(nodeNumber);

    for (int y = 1; y < splineControlPoint->ny - 1; y++) {
        for (int x = 1; x < splineControlPoint->nx - 1; x++) {
            mat33 matrix{ 0, 0, 0, 0, 0, 0, 0, 0, 1 };

            int i = 0;
            for (int b = -1; b < 2; b++) {
                for (int a = -1; a < 2; a++) {
                    const int index = (y + b) * splineControlPoint->nx + x + a;
                    const DataType splineCoeffX = splinePtrX[index];
                    const DataType splineCoeffY = splinePtrY[index];

                    matrix.m[0][0] += static_cast<float>(basisX[i] * splineCoeffX);
                    matrix.m[1][0] += static_cast<float>(basisY[i] * splineCoeffX);

                    matrix.m[0][1] += static_cast<float>(basisX[i] * splineCoeffY);
                    matrix.m[1][1] += static_cast<float>(basisY[i] * splineCoeffY);
                    ++i;
                } // a
            } // b
            // Convert from mm to voxel
            matrix = nifti_mat33_mul(reorientation, matrix);
            // Removing the rotation component
            const mat33 r = nifti_mat33_inverse(nifti_mat33_polar(matrix));
            matrix = nifti_mat33_mul(r, matrix);
            // Convert to displacement
            matrix.m[0][0]--; matrix.m[1][1]--;
            i = 8;
            for (int b = -1; b < 2; b++) {
                for (int a = -1; a < 2; a++) {
                    const DataType gradValues[2]{ -2.f * matrix.m[0][0] * basisX[i], -2.f * matrix.m[1][1] * basisY[i] };
                    const int index = (y + b) * splineControlPoint->nx + x + a;

                    gradientXPtr[index] += approxRatio * (invReorientation.m[0][0] * gradValues[0] +
                                                          invReorientation.m[0][1] * gradValues[1]);
                    gradientYPtr[index] += approxRatio * (invReorientation.m[1][0] * gradValues[0] +
                                                          invReorientation.m[1][1] * gradValues[1]);
                    --i;
                } // a
            } // b
        } // x
    } // y
}
/* *************************************************************** */
template <class DataType>
void reg_spline_approxLinearEnergyGradient3D(const nifti_image *splineControlPoint,
                                             nifti_image *gradientImage,
                                             float weight) {
    const size_t nodeNumber = NiftiImage::calcVoxelNumber(splineControlPoint, 3);

    // Create the pointers
    const DataType *splinePtrX = static_cast<DataType*>(splineControlPoint->data);
    const DataType *splinePtrY = &splinePtrX[nodeNumber];
    const DataType *splinePtrZ = &splinePtrY[nodeNumber];
    DataType *gradientXPtr = static_cast<DataType*>(gradientImage->data);
    DataType *gradientYPtr = &gradientXPtr[nodeNumber];
    DataType *gradientZPtr = &gradientYPtr[nodeNumber];

    // Store the basis values since they are constant as the value is approximated
    // at the control point positions only
    DataType basisX[27], basisY[27], basisZ[27];
    set_first_order_basis_values(basisX, basisY, basisZ);

    // Matrix to use to convert the gradient from mm to voxel
    const mat33 reorientation = reg_mat44_to_mat33(splineControlPoint->sform_code > 0 ? &splineControlPoint->sto_ijk : &splineControlPoint->qto_ijk);
    const mat33 invReorientation = nifti_mat33_inverse(reorientation);

    const DataType approxRatio = weight / static_cast<DataType>(nodeNumber);

    for (int z = 1; z < splineControlPoint->nz - 1; z++) {
        for (int y = 1; y < splineControlPoint->ny - 1; y++) {
            for (int x = 1; x < splineControlPoint->nx - 1; x++) {
                mat33 matrix{};
                int i = 0;
                for (int c = -1; c < 2; c++) {
                    for (int b = -1; b < 2; b++) {
                        for (int a = -1; a < 2; a++) {
                            const int index = ((z + c) * splineControlPoint->ny + y + b) * splineControlPoint->nx + x + a;
                            const DataType splineCoeffX = splinePtrX[index];
                            const DataType splineCoeffY = splinePtrY[index];
                            const DataType splineCoeffZ = splinePtrZ[index];

                            matrix.m[0][0] += static_cast<float>(basisX[i] * splineCoeffX);
                            matrix.m[1][0] += static_cast<float>(basisY[i] * splineCoeffX);
                            matrix.m[2][0] += static_cast<float>(basisZ[i] * splineCoeffX);

                            matrix.m[0][1] += static_cast<float>(basisX[i] * splineCoeffY);
                            matrix.m[1][1] += static_cast<float>(basisY[i] * splineCoeffY);
                            matrix.m[2][1] += static_cast<float>(basisZ[i] * splineCoeffY);

                            matrix.m[0][2] += static_cast<float>(basisX[i] * splineCoeffZ);
                            matrix.m[1][2] += static_cast<float>(basisY[i] * splineCoeffZ);
                            matrix.m[2][2] += static_cast<float>(basisZ[i] * splineCoeffZ);
                            ++i;
                        }
                    }
                }
                // Convert from mm to voxel
                matrix = nifti_mat33_mul(reorientation, matrix);
                // Removing the rotation component
                const mat33 r = nifti_mat33_inverse(nifti_mat33_polar(matrix));
                matrix = nifti_mat33_mul(r, matrix);
                // Convert to displacement
                matrix.m[0][0]--; matrix.m[1][1]--; matrix.m[2][2]--;
                i = 26;
                for (int c = -1; c < 2; c++) {
                    for (int b = -1; b < 2; b++) {
                        for (int a = -1; a < 2; a++) {
                            const int index = ((z + c) * splineControlPoint->ny + y + b) * splineControlPoint->nx + x + a;
                            const DataType gradValues[3]{ -2.f * matrix.m[0][0] * basisX[i],
                                                          -2.f * matrix.m[1][1] * basisY[i],
                                                          -2.f * matrix.m[2][2] * basisZ[i] };

                            gradientXPtr[index] += approxRatio * (invReorientation.m[0][0] * gradValues[0] +
                                                                  invReorientation.m[0][1] * gradValues[1] +
                                                                  invReorientation.m[0][2] * gradValues[2]);
                            gradientYPtr[index] += approxRatio * (invReorientation.m[1][0] * gradValues[0] +
                                                                  invReorientation.m[1][1] * gradValues[1] +
                                                                  invReorientation.m[1][2] * gradValues[2]);
                            gradientZPtr[index] += approxRatio * (invReorientation.m[2][0] * gradValues[0] +
                                                                  invReorientation.m[2][1] * gradValues[1] +
                                                                  invReorientation.m[2][2] * gradValues[2]);
                            --i;
                        } // a
                    } // b
                } // c
            } // x
        } // y
    } // z
}
/* *************************************************************** */
void reg_spline_approxLinearEnergyGradient(const nifti_image *splineControlPoint,
                                           nifti_image *gradientImage,
                                           float weight) {
    if (splineControlPoint->datatype != gradientImage->datatype)
        NR_FATAL_ERROR("Input images are expected to have the same datatype");

    if (splineControlPoint->nz > 1) {
        switch (splineControlPoint->datatype) {
        case NIFTI_TYPE_FLOAT32:
            reg_spline_approxLinearEnergyGradient3D<float>(splineControlPoint, gradientImage, weight);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_spline_approxLinearEnergyGradient3D<double>(splineControlPoint, gradientImage, weight);
            break;
        default:
            NR_FATAL_ERROR("Only implemented for single or double precision images");
        }
    } else {
        switch (splineControlPoint->datatype) {
        case NIFTI_TYPE_FLOAT32:
            reg_spline_approxLinearEnergyGradient2D<float>(splineControlPoint, gradientImage, weight);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_spline_approxLinearEnergyGradient2D<double>(splineControlPoint, gradientImage, weight);
            break;
        default:
            NR_FATAL_ERROR("Only implemented for single or double precision images");
        }
    }
}
/* *************************************************************** */
template <class DataType>
double reg_defField_linearEnergyValue2D(const nifti_image *deformationField) {
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(deformationField, 2);
    int a, b, x, y, X, Y, index;
    DataType basis[2] = {1, 0};
    DataType first[2] = {-1, 1};

    double constraintValue = 0;
    double currentValue;

    // Create pointers to the deformation field
    const DataType *defPtrX = static_cast<DataType*>(deformationField->data);
    const DataType *defPtrY = &defPtrX[voxelNumber];
    DataType defX, defY;

    mat33 matrix, r;

    // Matrix to use to convert the gradient from mm to voxel
    mat33 reorientation;
    if (deformationField->sform_code > 0)
        reorientation = reg_mat44_to_mat33(&deformationField->sto_ijk);
    else reorientation = reg_mat44_to_mat33(&deformationField->qto_ijk);

    for (y = 0; y < deformationField->ny; ++y) {
        Y = (y != deformationField->ny - 1) ? y : y - 1;
        for (x = 0; x < deformationField->nx; ++x) {
            X = (x != deformationField->nx - 1) ? x : x - 1;

            memset(&matrix, 0, sizeof(mat33));

            for (b = 0; b < 2; b++) {
                for (a = 0; a < 2; a++) {
                    index = (Y + b) * deformationField->nx + X + a;
                    defX = defPtrX[index];
                    defY = defPtrY[index];

                    matrix.m[0][0] += static_cast<float>(first[a] * basis[b] * defX);
                    matrix.m[1][0] += static_cast<float>(basis[a] * first[b] * defX);
                    matrix.m[0][1] += static_cast<float>(first[a] * basis[b] * defY);
                    matrix.m[1][1] += static_cast<float>(basis[a] * first[b] * defY);
                }
            }
            // Convert from mm to voxel
            matrix = nifti_mat33_mul(reorientation, matrix);
            // Removing the rotation component
            r = nifti_mat33_inverse(nifti_mat33_polar(matrix));
            matrix = nifti_mat33_mul(r, matrix);
            // Convert to displacement
            --matrix.m[0][0];
            --matrix.m[1][1];

            currentValue = 0;
            for (b = 0; b < 2; b++) {
                for (a = 0; a < 2; a++) {
                    currentValue += Square(0.5 * (matrix.m[a][b] + matrix.m[b][a])); // symmetric part
                }
            }
            constraintValue += currentValue;
        }
    }
    return constraintValue / static_cast<double>(deformationField->nvox);
}
/* *************************************************************** */
template <class DataType>
double reg_defField_linearEnergyValue3D(const nifti_image *deformationField) {
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(deformationField, 3);
    int a, b, c, x, y, z, X, Y, Z, index;
    DataType basis[2] = {1, 0};
    DataType first[2] = {-1, 1};

    double constraintValue = 0;
    double currentValue;

    // Create pointers to the deformation field
    const DataType *defPtrX = static_cast<DataType*>(deformationField->data);
    const DataType *defPtrY = &defPtrX[voxelNumber];
    const DataType *defPtrZ = &defPtrY[voxelNumber];
    DataType defX, defY, defZ;

    mat33 matrix, r;

    // Matrix to use to convert the gradient from mm to voxel
    mat33 reorientation;
    if (deformationField->sform_code > 0)
        reorientation = reg_mat44_to_mat33(&deformationField->sto_ijk);
    else reorientation = reg_mat44_to_mat33(&deformationField->qto_ijk);

    for (z = 0; z < deformationField->nz; ++z) {
        Z = (z != deformationField->nz - 1) ? z : z - 1;
        for (y = 0; y < deformationField->ny; ++y) {
            Y = (y != deformationField->ny - 1) ? y : y - 1;
            for (x = 0; x < deformationField->nx; ++x) {
                X = (x != deformationField->nx - 1) ? x : x - 1;

                memset(&matrix, 0, sizeof(mat33));

                for (c = 0; c < 2; c++) {
                    for (b = 0; b < 2; b++) {
                        for (a = 0; a < 2; a++) {
                            index = ((Z + c) * deformationField->ny + Y + b) * deformationField->nx + X + a;
                            defX = defPtrX[index];
                            defY = defPtrY[index];
                            defZ = defPtrZ[index];

                            matrix.m[0][0] += static_cast<float>(first[a] * basis[b] * basis[c] * defX);
                            matrix.m[1][0] += static_cast<float>(basis[a] * first[b] * basis[c] * defX);
                            matrix.m[2][0] += static_cast<float>(basis[a] * basis[b] * first[c] * defX);

                            matrix.m[0][1] += static_cast<float>(first[a] * basis[b] * basis[c] * defY);
                            matrix.m[1][1] += static_cast<float>(basis[a] * first[b] * basis[c] * defY);
                            matrix.m[2][1] += static_cast<float>(basis[a] * basis[b] * first[c] * defY);

                            matrix.m[0][2] += static_cast<float>(first[a] * basis[b] * basis[c] * defZ);
                            matrix.m[1][2] += static_cast<float>(basis[a] * first[b] * basis[c] * defZ);
                            matrix.m[2][2] += static_cast<float>(basis[a] * basis[b] * first[c] * defZ);
                        }
                    }
                }
                // Convert from mm to voxel
                matrix = nifti_mat33_mul(reorientation, matrix);
                // Removing the rotation component
                r = nifti_mat33_inverse(nifti_mat33_polar(matrix));
                matrix = nifti_mat33_mul(r, matrix);
                // Convert to displacement
                --matrix.m[0][0];
                --matrix.m[1][1];
                --matrix.m[2][2];

                currentValue = 0;
                for (b = 0; b < 3; b++) {
                    for (a = 0; a < 3; a++) {
                        currentValue += Square(0.5 * (matrix.m[a][b] + matrix.m[b][a])); // symmetric part
                    }
                }
                constraintValue += currentValue;
            }
        }
    }
    return constraintValue / static_cast<double>(deformationField->nvox);
}
/* *************************************************************** */
double reg_defField_linearEnergy(const nifti_image *deformationField) {
    if (deformationField->nz > 1) {
        switch (deformationField->datatype) {
        case NIFTI_TYPE_FLOAT32:
            return reg_defField_linearEnergyValue3D<float>(deformationField);
        case NIFTI_TYPE_FLOAT64:
            return reg_defField_linearEnergyValue3D<double>(deformationField);
        default:
            NR_FATAL_ERROR("Only implemented for single or double precision images");
            return 0;
        }
    } else {
        switch (deformationField->datatype) {
        case NIFTI_TYPE_FLOAT32:
            return reg_defField_linearEnergyValue2D<float>(deformationField);
        case NIFTI_TYPE_FLOAT64:
            return reg_defField_linearEnergyValue2D<double>(deformationField);
        default:
            NR_FATAL_ERROR("Only implemented for single or double precision images");
            return 0;
        }
    }
}
/* *************************************************************** */
template <class DataType>
void reg_defField_linearEnergyGradient2D(const nifti_image *deformationField,
                                         nifti_image *gradientImage,
                                         float weight) {
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(deformationField, 2);
    int a, b, x, y, X, Y, index;
    DataType basis[2] = {1, 0};
    DataType first[2] = {-1, 1};

    // Create pointers to the deformation field
    const DataType *defPtrX = static_cast<DataType*>(deformationField->data);
    const DataType *defPtrY = &defPtrX[voxelNumber];
    DataType defX, defY;

    mat33 matrix, r;

    DataType *gradientXPtr = static_cast<DataType*>(gradientImage->data);
    DataType *gradientYPtr = &gradientXPtr[voxelNumber];

    DataType approxRatio = weight / static_cast<DataType>(voxelNumber);
    DataType gradValues[2];

    // Matrix to use to convert the gradient from mm to voxel
    mat33 reorientation;
    if (deformationField->sform_code > 0)
        reorientation = reg_mat44_to_mat33(&deformationField->sto_ijk);
    else reorientation = reg_mat44_to_mat33(&deformationField->qto_ijk);
    mat33 invReorientation = nifti_mat33_inverse(reorientation);

    for (y = 0; y < deformationField->ny; ++y) {
        Y = (y != deformationField->ny - 1) ? y : y - 1;
        for (x = 0; x < deformationField->nx; ++x) {
            X = (x != deformationField->nx - 1) ? x : x - 1;

            memset(&matrix, 0, sizeof(mat33));

            for (b = 0; b < 2; b++) {
                for (a = 0; a < 2; a++) {
                    index = (Y + b) * deformationField->nx + X + a;
                    defX = defPtrX[index];
                    defY = defPtrY[index];

                    matrix.m[0][0] += static_cast<float>(first[a] * basis[b] * defX);
                    matrix.m[1][0] += static_cast<float>(basis[a] * first[b] * defX);
                    matrix.m[0][1] += static_cast<float>(first[a] * basis[b] * defY);
                    matrix.m[1][1] += static_cast<float>(basis[a] * first[b] * defY);
                }
            }
            // Convert from mm to voxel
            matrix = nifti_mat33_mul(reorientation, matrix);
            // Removing the rotation component
            r = nifti_mat33_inverse(nifti_mat33_polar(matrix));
            matrix = nifti_mat33_mul(r, matrix);
            // Convert to displacement
            --matrix.m[0][0];
            --matrix.m[1][1];

            for (b = 0; b < 2; b++) {
                for (a = 0; a < 2; a++) {
                    index = (Y + b) * deformationField->nx + X + a;
                    gradValues[0] = -2.f * matrix.m[0][0] * first[1 - a] * basis[1 - b];
                    gradValues[1] = -2.f * matrix.m[1][1] * basis[1 - a] * first[1 - b];
                    gradientXPtr[index] += approxRatio * (invReorientation.m[0][0] * gradValues[0] +
                                                          invReorientation.m[0][1] * gradValues[1]);
                    gradientYPtr[index] += approxRatio * (invReorientation.m[1][0] * gradValues[0] +
                                                          invReorientation.m[1][1] * gradValues[1]);
                } // a
            } // b
        }
    }
}
/* *************************************************************** */
template <class DataType>
void reg_defField_linearEnergyGradient3D(const nifti_image *deformationField,
                                         nifti_image *gradientImage,
                                         float weight) {
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(deformationField, 3);
    int a, b, c, x, y, z, X, Y, Z, index;
    DataType basis[2] = {1, 0};
    DataType first[2] = {-1, 1};

    // Create pointers to the deformation field
    const DataType *defPtrX = static_cast<DataType*>(deformationField->data);
    const DataType *defPtrY = &defPtrX[voxelNumber];
    const DataType *defPtrZ = &defPtrY[voxelNumber];
    DataType defX, defY, defZ;

    mat33 matrix, r;

    DataType *gradientXPtr = static_cast<DataType*>(gradientImage->data);
    DataType *gradientYPtr = &gradientXPtr[voxelNumber];
    DataType *gradientZPtr = &gradientYPtr[voxelNumber];

    DataType approxRatio = weight / static_cast<DataType>(voxelNumber);
    DataType gradValues[3];

    // Matrix to use to convert the gradient from mm to voxel
    mat33 reorientation;
    if (deformationField->sform_code > 0)
        reorientation = reg_mat44_to_mat33(&deformationField->sto_ijk);
    else reorientation = reg_mat44_to_mat33(&deformationField->qto_ijk);
    mat33 invReorientation = nifti_mat33_inverse(reorientation);

    for (z = 0; z < deformationField->nz; ++z) {
        Z = (z != deformationField->nz - 1) ? z : z - 1;
        for (y = 0; y < deformationField->ny; ++y) {
            Y = (y != deformationField->ny - 1) ? y : y - 1;
            for (x = 0; x < deformationField->nx; ++x) {
                X = (x != deformationField->nx - 1) ? x : x - 1;

                memset(&matrix, 0, sizeof(mat33));

                for (c = 0; c < 2; c++) {
                    for (b = 0; b < 2; b++) {
                        for (a = 0; a < 2; a++) {
                            index = ((Z + c) * deformationField->ny + Y + b) * deformationField->nx + X + a;
                            defX = defPtrX[index];
                            defY = defPtrY[index];
                            defZ = defPtrZ[index];

                            matrix.m[0][0] += static_cast<float>(first[a] * basis[b] * basis[c] * defX);
                            matrix.m[1][0] += static_cast<float>(basis[a] * first[b] * basis[c] * defX);
                            matrix.m[2][0] += static_cast<float>(basis[a] * basis[b] * first[c] * defX);

                            matrix.m[0][1] += static_cast<float>(first[a] * basis[b] * basis[c] * defY);
                            matrix.m[1][1] += static_cast<float>(basis[a] * first[b] * basis[c] * defY);
                            matrix.m[2][1] += static_cast<float>(basis[a] * basis[b] * first[c] * defY);

                            matrix.m[0][2] += static_cast<float>(first[a] * basis[b] * basis[c] * defZ);
                            matrix.m[1][2] += static_cast<float>(basis[a] * first[b] * basis[c] * defZ);
                            matrix.m[2][2] += static_cast<float>(basis[a] * basis[b] * first[c] * defZ);
                        }
                    }
                }
                // Convert from mm to voxel
                matrix = nifti_mat33_mul(reorientation, matrix);
                // Removing the rotation component
                r = nifti_mat33_inverse(nifti_mat33_polar(matrix));
                matrix = nifti_mat33_mul(r, matrix);
                // Convert to displacement
                --matrix.m[0][0];
                --matrix.m[1][1];
                --matrix.m[2][2];
                for (c = 0; c < 2; c++) {
                    for (b = 0; b < 2; b++) {
                        for (a = 0; a < 2; a++) {
                            index = ((Z + c) * deformationField->ny + Y + b) * deformationField->nx + X + a;
                            gradValues[0] = -2.f * matrix.m[0][0] * first[1 - a] * basis[1 - b] * basis[1 - c];
                            gradValues[1] = -2.f * matrix.m[1][1] * basis[1 - a] * first[1 - b] * basis[1 - c];
                            gradValues[2] = -2.f * matrix.m[2][2] * basis[1 - a] * basis[1 - b] * first[1 - c];
                            gradientXPtr[index] += approxRatio * (invReorientation.m[0][0] * gradValues[0] +
                                                                  invReorientation.m[0][1] * gradValues[1] +
                                                                  invReorientation.m[0][2] * gradValues[2]);
                            gradientYPtr[index] += approxRatio * (invReorientation.m[1][0] * gradValues[0] +
                                                                  invReorientation.m[1][1] * gradValues[1] +
                                                                  invReorientation.m[1][2] * gradValues[2]);
                            gradientZPtr[index] += approxRatio * (invReorientation.m[2][0] * gradValues[0] +
                                                                  invReorientation.m[2][1] * gradValues[1] +
                                                                  invReorientation.m[2][2] * gradValues[2]);
                        } // a
                    } // b
                } // c
            }
        }
    }
}
/* *************************************************************** */
void reg_defField_linearEnergyGradient(const nifti_image *deformationField,
                                       nifti_image *gradientImage,
                                       float weight) {
    if (deformationField->nz > 1) {
        switch (deformationField->datatype) {
        case NIFTI_TYPE_FLOAT32:
            reg_defField_linearEnergyGradient3D<float>(deformationField, gradientImage, weight);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_defField_linearEnergyGradient3D<double>(deformationField, gradientImage, weight);
            break;
        default:
            NR_FATAL_ERROR("Only implemented for single or double precision images");
        }
    } else {
        switch (deformationField->datatype) {
        case NIFTI_TYPE_FLOAT32:
            reg_defField_linearEnergyGradient2D<float>(deformationField, gradientImage, weight);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_defField_linearEnergyGradient2D<double>(deformationField, gradientImage, weight);
            break;
        default:
            NR_FATAL_ERROR("Only implemented for single or double precision images");
        }
    }
}
/* *************************************************************** */
template <class DataType>
double reg_spline_getLandmarkDistance_core(const nifti_image *controlPointImage,
                                           size_t landmarkNumber,
                                           float *landmarkReference,
                                           float *landmarkFloating) {
    const int imageDim = controlPointImage->nz > 1 ? 3 : 2;
    const size_t controlPointNumber = NiftiImage::calcVoxelNumber(controlPointImage, 3);
    double constraintValue = 0;
    size_t l, index;
    float refPosition[4];
    float defPosition[4];
    float floPosition[4];
    int previous[3], a, b, c;
    DataType basisX[4], basisY[4], basisZ[4], basis;
    const mat44 *gridRealToVox = &(controlPointImage->qto_ijk);
    if (controlPointImage->sform_code > 0)
        gridRealToVox = &(controlPointImage->sto_ijk);
    const DataType *gridPtrX = static_cast<DataType*>(controlPointImage->data);
    const DataType *gridPtrY = &gridPtrX[controlPointNumber];
    const DataType *gridPtrZ = nullptr;
    if (imageDim > 2)
        gridPtrZ = &gridPtrY[controlPointNumber];

    // Loop over all landmarks
    for (l = 0; l < landmarkNumber; ++l) {
        // fetch the initial positions
        refPosition[0] = landmarkReference[l * imageDim];
        floPosition[0] = landmarkFloating[l * imageDim];
        refPosition[1] = landmarkReference[l * imageDim + 1];
        floPosition[1] = landmarkFloating[l * imageDim + 1];
        if (imageDim > 2) {
            refPosition[2] = landmarkReference[l * imageDim + 2];
            floPosition[2] = landmarkFloating[l * imageDim + 2];
        } else refPosition[2] = floPosition[2] = 0;
        refPosition[3] = floPosition[3] = 1;
        // Convert the reference position to voxel in the control point grid space
        reg_mat44_mul(gridRealToVox, refPosition, defPosition);

        // Extract the corresponding nodes
        previous[0] = Floor(defPosition[0]) - 1;
        previous[1] = Floor(defPosition[1]) - 1;
        previous[2] = Floor(defPosition[2]) - 1;
        // Check that the specified landmark belongs to the input image
        if (previous[0] > -1 && previous[0] + 3 < controlPointImage->nx &&
            previous[1] > -1 && previous[1] + 3 < controlPointImage->ny &&
            ((previous[2] > -1 && previous[2] + 3 < controlPointImage->nz) || imageDim == 2)) {
            // Extract the corresponding basis values
            get_BSplineBasisValues<DataType>(defPosition[0] - 1 - (DataType)previous[0], basisX);
            get_BSplineBasisValues<DataType>(defPosition[1] - 1 - (DataType)previous[1], basisY);
            get_BSplineBasisValues<DataType>(defPosition[2] - 1 - (DataType)previous[2], basisZ);
            defPosition[0] = 0;
            defPosition[1] = 0;
            defPosition[2] = 0;
            if (imageDim > 2) {
                for (c = 0; c < 4; ++c) {
                    for (b = 0; b < 4; ++b) {
                        for (a = 0; a < 4; ++a) {
                            index = ((previous[2] + c) * controlPointImage->ny + previous[1] + b) *
                                controlPointImage->nx + previous[0] + a;
                            basis = basisX[a] * basisY[b] * basisZ[c];
                            defPosition[0] += static_cast<float>(gridPtrX[index] * basis);
                            defPosition[1] += static_cast<float>(gridPtrY[index] * basis);
                            defPosition[2] += static_cast<float>(gridPtrZ[index] * basis);
                        }
                    }
                }
            } else {
                for (b = 0; b < 4; ++b) {
                    for (a = 0; a < 4; ++a) {
                        index = (previous[1] + b) * controlPointImage->nx + previous[0] + a;
                        basis = basisX[a] * basisY[b];
                        defPosition[0] += static_cast<float>(gridPtrX[index] * basis);
                        defPosition[1] += static_cast<float>(gridPtrY[index] * basis);
                    }
                }
            }
            constraintValue += Square(floPosition[0] - defPosition[0]);
            constraintValue += Square(floPosition[1] - defPosition[1]);
            if (imageDim > 2)
                constraintValue += Square(floPosition[2] - defPosition[2]);
        } else {
            NR_WARN("The current landmark at position " << refPosition[0] << " " <<
                    refPosition[1] << (imageDim > 2 ? " "s + std::to_string(refPosition[2]) : "") <<
                    " is ignored as it is not in the space of the reference image");
        }
    }
    return constraintValue;
}
/* *************************************************************** */
double reg_spline_getLandmarkDistance(const nifti_image *controlPointImage,
                                      size_t landmarkNumber,
                                      float *landmarkReference,
                                      float *landmarkFloating) {
    if (controlPointImage->intent_p1 != CUB_SPLINE_GRID)
        NR_FATAL_ERROR("This function is only implemented for control point grid within an Euclidean setting for now");
    switch (controlPointImage->datatype) {
    case NIFTI_TYPE_FLOAT32:
        return reg_spline_getLandmarkDistance_core<float>(controlPointImage, landmarkNumber, landmarkReference, landmarkFloating);
        break;
    case NIFTI_TYPE_FLOAT64:
        return reg_spline_getLandmarkDistance_core<double>(controlPointImage, landmarkNumber, landmarkReference, landmarkFloating);
        break;
    default:
        NR_FATAL_ERROR("Only implemented for single or double precision images");
        return 0;
    }
}
/* *************************************************************** */
template <class DataType>
void reg_spline_getLandmarkDistanceGradient_core(const nifti_image *controlPointImage,
                                                 nifti_image *gradientImage,
                                                 size_t landmarkNumber,
                                                 float *landmarkReference,
                                                 float *landmarkFloating,
                                                 float weight) {
    const int imageDim = controlPointImage->nz > 1 ? 3 : 2;
    const size_t controlPointNumber = NiftiImage::calcVoxelNumber(controlPointImage, 3);
    size_t l, index;
    float refPosition[3];
    float defPosition[3];
    float floPosition[3];
    int previous[3], a, b, c;
    DataType basisX[4], basisY[4], basisZ[4], basis;
    const mat44 *gridRealToVox = &(controlPointImage->qto_ijk);
    if (controlPointImage->sform_code > 0)
        gridRealToVox = &(controlPointImage->sto_ijk);
    const DataType *gridPtrX = static_cast<DataType*>(controlPointImage->data);
    DataType *gradPtrX = static_cast<DataType*>(gradientImage->data);
    const DataType *gridPtrY = &gridPtrX[controlPointNumber];
    DataType *gradPtrY = &gradPtrX[controlPointNumber];
    const DataType *gridPtrZ = nullptr;
    DataType *gradPtrZ = nullptr;
    if (imageDim > 2) {
        gridPtrZ = &gridPtrY[controlPointNumber];
        gradPtrZ = &gradPtrY[controlPointNumber];
    }

    // Loop over all landmarks
    for (l = 0; l < landmarkNumber; ++l) {
        // fetch the initial positions
        refPosition[0] = landmarkReference[l * imageDim];
        floPosition[0] = landmarkFloating[l * imageDim];
        refPosition[1] = landmarkReference[l * imageDim + 1];
        floPosition[1] = landmarkFloating[l * imageDim + 1];
        if (imageDim > 2) {
            refPosition[2] = landmarkReference[l * imageDim + 2];
            floPosition[2] = landmarkFloating[l * imageDim + 2];
        } else refPosition[2] = floPosition[2] = 0;
        // Convert the reference position to voxel in the control point grid space
        reg_mat44_mul(gridRealToVox, refPosition, defPosition);
        if (imageDim == 2) defPosition[2] = 0;
        // Extract the corresponding nodes
        previous[0] = Floor(defPosition[0]) - 1;
        previous[1] = Floor(defPosition[1]) - 1;
        previous[2] = Floor(defPosition[2]) - 1;
        // Check that the specified landmark belongs to the input image
        if (previous[0] > -1 && previous[0] + 3 < controlPointImage->nx &&
            previous[1] > -1 && previous[1] + 3 < controlPointImage->ny &&
            ((previous[2] > -1 && previous[2] + 3 < controlPointImage->nz) || imageDim == 2)) {
            // Extract the corresponding basis values
            get_BSplineBasisValues<DataType>(defPosition[0] - 1 - (DataType)previous[0], basisX);
            get_BSplineBasisValues<DataType>(defPosition[1] - 1 - (DataType)previous[1], basisY);
            get_BSplineBasisValues<DataType>(defPosition[2] - 1 - (DataType)previous[2], basisZ);
            defPosition[0] = 0;
            defPosition[1] = 0;
            defPosition[2] = 0;
            if (imageDim > 2) {
                for (c = 0; c < 4; ++c) {
                    for (b = 0; b < 4; ++b) {
                        for (a = 0; a < 4; ++a) {
                            index = ((previous[2] + c) * controlPointImage->ny + previous[1] + b) *
                                controlPointImage->nx + previous[0] + a;
                            basis = basisX[a] * basisY[b] * basisZ[c];
                            defPosition[0] += static_cast<float>(gridPtrX[index] * basis);
                            defPosition[1] += static_cast<float>(gridPtrY[index] * basis);
                            defPosition[2] += static_cast<float>(gridPtrZ[index] * basis);
                        }
                    }
                }
            } else {
                for (b = 0; b < 4; ++b) {
                    for (a = 0; a < 4; ++a) {
                        index = (previous[1] + b) * controlPointImage->nx + previous[0] + a;
                        basis = basisX[a] * basisY[b];
                        defPosition[0] += static_cast<float>(gridPtrX[index] * basis);
                        defPosition[1] += static_cast<float>(gridPtrY[index] * basis);
                    }
                }
            }
            defPosition[0] = floPosition[0] - defPosition[0];
            defPosition[1] = floPosition[1] - defPosition[1];
            if (imageDim > 2)
                defPosition[2] = floPosition[2] - defPosition[2];
            if (imageDim > 2) {
                for (c = 0; c < 4; ++c) {
                    for (b = 0; b < 4; ++b) {
                        for (a = 0; a < 4; ++a) {
                            index = ((previous[2] + c) * controlPointImage->ny + previous[1] + b) *
                                controlPointImage->nx + previous[0] + a;
                            basis = basisX[a] * basisY[b] * basisZ[c] * weight;
                            gradPtrX[index] -= defPosition[0] * basis;
                            gradPtrY[index] -= defPosition[1] * basis;
                            gradPtrZ[index] -= defPosition[2] * basis;
                        }
                    }
                }
            } else {
                for (b = 0; b < 4; ++b) {
                    for (a = 0; a < 4; ++a) {
                        index = (previous[1] + b) * controlPointImage->nx + previous[0] + a;
                        basis = basisX[a] * basisY[b] * weight;
                        gradPtrX[index] -= defPosition[0] * basis;
                        gradPtrY[index] -= defPosition[1] * basis;
                    }
                }
            }
        } else {
            NR_WARN("The current landmark at position " << refPosition[0] << " " <<
                    refPosition[1] << (imageDim > 2 ? " "s + std::to_string(refPosition[2]) : "") <<
                    " is ignored as it is not in the space of the reference image");
        }
    }
}
/* *************************************************************** */
void reg_spline_getLandmarkDistanceGradient(const nifti_image *controlPointImage,
                                            nifti_image *gradientImage,
                                            size_t landmarkNumber,
                                            float *landmarkReference,
                                            float *landmarkFloating,
                                            float weight) {
    if (controlPointImage->intent_p1 != CUB_SPLINE_GRID)
        NR_FATAL_ERROR("This function is only implemented for control point grid within an Euclidean setting for now");

    switch (controlPointImage->datatype) {
    case NIFTI_TYPE_FLOAT32:
        reg_spline_getLandmarkDistanceGradient_core<float>
            (controlPointImage, gradientImage, landmarkNumber, landmarkReference, landmarkFloating, weight);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_spline_getLandmarkDistanceGradient_core<double>
            (controlPointImage, gradientImage, landmarkNumber, landmarkReference, landmarkFloating, weight);
        break;
    default:
        NR_FATAL_ERROR("Only implemented for single or double precision images");
    }
}
/* *************************************************************** */
template <class DataType>
double reg_spline_approxLinearPairwise3D(nifti_image *splineControlPoint) {
    const size_t nodeNumber = NiftiImage::calcVoxelNumber(splineControlPoint, 3);
    int x, y, z, index;

    // Create pointers to the spline coefficients
    reg_getDisplacementFromDeformation(splineControlPoint);
    DataType *splinePtrX = static_cast<DataType*>(splineControlPoint->data);
    DataType *splinePtrY = &splinePtrX[nodeNumber];
    DataType *splinePtrZ = &splinePtrY[nodeNumber];

    DataType centralCP[3], neigbCP[3];

    double constraintValue = 0;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, x, y, centralCP, neigbCP) \
    shared(splineControlPoint, splinePtrX, splinePtrY, splinePtrZ) \
    reduction(+:constraintValue)
#endif // _OPENMP
    for (z = 0; z < splineControlPoint->nz; ++z) {
        index = z * splineControlPoint->nx * splineControlPoint->ny;
        for (y = 0; y < splineControlPoint->ny; ++y) {
            for (x = 0; x < splineControlPoint->nx; ++x) {
                centralCP[0] = splinePtrX[index];
                centralCP[1] = splinePtrY[index];
                centralCP[2] = splinePtrZ[index];

                if (x > 0) {
                    neigbCP[0] = splinePtrX[index - 1];
                    neigbCP[1] = splinePtrY[index - 1];
                    neigbCP[2] = splinePtrZ[index - 1];
                    constraintValue += (Square(centralCP[0] - neigbCP[0]) + Square(centralCP[1] - neigbCP[1]) +
                                        Square(centralCP[2] - neigbCP[2])) / splineControlPoint->dx;
                }
                if (x < splineControlPoint->nx - 1) {
                    neigbCP[0] = splinePtrX[index + 1];
                    neigbCP[1] = splinePtrY[index + 1];
                    neigbCP[2] = splinePtrZ[index + 1];
                    constraintValue += (Square(centralCP[0] - neigbCP[0]) + Square(centralCP[1] - neigbCP[1]) +
                                        Square(centralCP[2] - neigbCP[2])) / splineControlPoint->dx;
                }

                if (y > 0) {
                    neigbCP[0] = splinePtrX[index - splineControlPoint->nx];
                    neigbCP[1] = splinePtrY[index - splineControlPoint->nx];
                    neigbCP[2] = splinePtrZ[index - splineControlPoint->nx];
                    constraintValue += (Square(centralCP[0] - neigbCP[0]) + Square(centralCP[1] - neigbCP[1]) +
                                        Square(centralCP[2] - neigbCP[2])) / splineControlPoint->dy;
                }
                if (y < splineControlPoint->ny - 1) {
                    neigbCP[0] = splinePtrX[index + splineControlPoint->nx];
                    neigbCP[1] = splinePtrY[index + splineControlPoint->nx];
                    neigbCP[2] = splinePtrZ[index + splineControlPoint->nx];
                    constraintValue += (Square(centralCP[0] - neigbCP[0]) + Square(centralCP[1] - neigbCP[1]) +
                                        Square(centralCP[2] - neigbCP[2])) / splineControlPoint->dy;
                }

                if (z > 0) {
                    neigbCP[0] = splinePtrX[index - splineControlPoint->nx * splineControlPoint->ny];
                    neigbCP[1] = splinePtrY[index - splineControlPoint->nx * splineControlPoint->ny];
                    neigbCP[2] = splinePtrZ[index - splineControlPoint->nx * splineControlPoint->ny];
                    constraintValue += (Square(centralCP[0] - neigbCP[0]) + Square(centralCP[1] - neigbCP[1]) +
                                        Square(centralCP[2] - neigbCP[2])) / splineControlPoint->dz;
                }
                if (z < splineControlPoint->nz - 1) {
                    neigbCP[0] = splinePtrX[index + splineControlPoint->nx * splineControlPoint->ny];
                    neigbCP[1] = splinePtrY[index + splineControlPoint->nx * splineControlPoint->ny];
                    neigbCP[2] = splinePtrZ[index + splineControlPoint->nx * splineControlPoint->ny];
                    constraintValue += (Square(centralCP[0] - neigbCP[0]) + Square(centralCP[1] - neigbCP[1]) +
                                        Square(centralCP[2] - neigbCP[2])) / splineControlPoint->dz;
                }
                index++;
            } // x
        } // y
    } // z
    reg_getDeformationFromDisplacement(splineControlPoint);
    return constraintValue / nodeNumber;
}
/* *************************************************************** */
double reg_spline_approxLinearPairwise(nifti_image *splineControlPoint) {
    if (splineControlPoint->nz > 1) {
        switch (splineControlPoint->datatype) {
        case NIFTI_TYPE_FLOAT32:
            return reg_spline_approxLinearPairwise3D<float>(splineControlPoint);
        case NIFTI_TYPE_FLOAT64:
            return reg_spline_approxLinearPairwise3D<double>(splineControlPoint);
        default:
            NR_FATAL_ERROR("Only implemented for single or double precision images");
            return 0;
        }
    } else {
        NR_FATAL_ERROR("Not implemented in 2D yet");
        return 0;
    }
}
/* *************************************************************** */
template <class DataType>
void reg_spline_approxLinearPairwiseGradient3D(nifti_image *splineControlPoint,
                                               nifti_image *gradientImage,
                                               float weight) {
    const size_t nodeNumber = NiftiImage::calcVoxelNumber(splineControlPoint, 3);
    int x, y, z, index;

    // Create pointers to the spline coefficients
    reg_getDisplacementFromDeformation(splineControlPoint);
    DataType *splinePtrX = static_cast<DataType*>(splineControlPoint->data);
    DataType *splinePtrY = &splinePtrX[nodeNumber];
    DataType *splinePtrZ = &splinePtrY[nodeNumber];

    // Pointers to the gradient image
    DataType *gradPtrX = static_cast<DataType*>(gradientImage->data);
    DataType *gradPtrY = &gradPtrX[nodeNumber];
    DataType *gradPtrZ = &gradPtrY[nodeNumber];

    DataType centralCP[3], neigbCP[3];

    double grad_values[3];

    DataType approxRatio = weight / static_cast<DataType>(nodeNumber);
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, x, y, centralCP, neigbCP, grad_values) \
    shared(splineControlPoint, splinePtrX, splinePtrY, splinePtrZ, approxRatio, \
    gradPtrX, gradPtrY, gradPtrZ)
#endif // _OPENMP
    for (z = 0; z < splineControlPoint->nz; ++z) {
        index = z * splineControlPoint->nx * splineControlPoint->ny;
        for (y = 0; y < splineControlPoint->ny; ++y) {
            for (x = 0; x < splineControlPoint->nx; ++x) {
                centralCP[0] = splinePtrX[index];
                centralCP[1] = splinePtrY[index];
                centralCP[2] = splinePtrZ[index];
                grad_values[0] = 0;
                grad_values[1] = 0;
                grad_values[2] = 0;

                if (x > 0) {
                    neigbCP[0] = splinePtrX[index - 1];
                    neigbCP[1] = splinePtrY[index - 1];
                    neigbCP[2] = splinePtrZ[index - 1];
                    grad_values[0] += 2. * (centralCP[0] - neigbCP[0]) / splineControlPoint->dx;
                    grad_values[1] += 2. * (centralCP[1] - neigbCP[1]) / splineControlPoint->dx;
                    grad_values[2] += 2. * (centralCP[2] - neigbCP[2]) / splineControlPoint->dx;
                }
                if (x < splineControlPoint->nx - 1) {
                    neigbCP[0] = splinePtrX[index + 1];
                    neigbCP[1] = splinePtrY[index + 1];
                    neigbCP[2] = splinePtrZ[index + 1];
                    grad_values[0] += 2. * (centralCP[0] - neigbCP[0]) / splineControlPoint->dx;
                    grad_values[1] += 2. * (centralCP[1] - neigbCP[1]) / splineControlPoint->dx;
                    grad_values[2] += 2. * (centralCP[2] - neigbCP[2]) / splineControlPoint->dx;
                }

                if (y > 0) {
                    neigbCP[0] = splinePtrX[index - splineControlPoint->nx];
                    neigbCP[1] = splinePtrY[index - splineControlPoint->nx];
                    neigbCP[2] = splinePtrZ[index - splineControlPoint->nx];
                    grad_values[0] += 2. * (centralCP[0] - neigbCP[0]) / splineControlPoint->dy;
                    grad_values[1] += 2. * (centralCP[1] - neigbCP[1]) / splineControlPoint->dy;
                    grad_values[2] += 2. * (centralCP[2] - neigbCP[2]) / splineControlPoint->dy;
                }
                if (y < splineControlPoint->ny - 1) {
                    neigbCP[0] = splinePtrX[index + splineControlPoint->nx];
                    neigbCP[1] = splinePtrY[index + splineControlPoint->nx];
                    neigbCP[2] = splinePtrZ[index + splineControlPoint->nx];
                    grad_values[0] += 2. * (centralCP[0] - neigbCP[0]) / splineControlPoint->dy;
                    grad_values[1] += 2. * (centralCP[1] - neigbCP[1]) / splineControlPoint->dy;
                    grad_values[2] += 2. * (centralCP[2] - neigbCP[2]) / splineControlPoint->dy;
                }

                if (z > 0) {
                    neigbCP[0] = splinePtrX[index - splineControlPoint->nx * splineControlPoint->ny];
                    neigbCP[1] = splinePtrY[index - splineControlPoint->nx * splineControlPoint->ny];
                    neigbCP[2] = splinePtrZ[index - splineControlPoint->nx * splineControlPoint->ny];
                    grad_values[0] += 2. * (centralCP[0] - neigbCP[0]) / splineControlPoint->dz;
                    grad_values[1] += 2. * (centralCP[1] - neigbCP[1]) / splineControlPoint->dz;
                    grad_values[2] += 2. * (centralCP[2] - neigbCP[2]) / splineControlPoint->dz;
                }
                if (z < splineControlPoint->nz - 1) {
                    neigbCP[0] = splinePtrX[index + splineControlPoint->nx * splineControlPoint->ny];
                    neigbCP[1] = splinePtrY[index + splineControlPoint->nx * splineControlPoint->ny];
                    neigbCP[2] = splinePtrZ[index + splineControlPoint->nx * splineControlPoint->ny];
                    grad_values[0] += 2. * (centralCP[0] - neigbCP[0]) / splineControlPoint->dz;
                    grad_values[1] += 2. * (centralCP[1] - neigbCP[1]) / splineControlPoint->dz;
                    grad_values[2] += 2. * (centralCP[2] - neigbCP[2]) / splineControlPoint->dz;
                }
                gradPtrX[index] += approxRatio * static_cast<DataType>(grad_values[0]);
                gradPtrY[index] += approxRatio * static_cast<DataType>(grad_values[1]);
                gradPtrZ[index] += approxRatio * static_cast<DataType>(grad_values[2]);

                index++;
            } // x
        } // y
    } // z
    reg_getDeformationFromDisplacement(splineControlPoint);
}
/* *************************************************************** */
void reg_spline_approxLinearPairwiseGradient(nifti_image *splineControlPoint,
                                             nifti_image *gradientImage,
                                             float weight) {
    if (splineControlPoint->datatype != gradientImage->datatype)
        NR_FATAL_ERROR("Input images are expected to have the same datatype");

    if (splineControlPoint->nz > 1) {
        switch (splineControlPoint->datatype) {
        case NIFTI_TYPE_FLOAT32:
            reg_spline_approxLinearPairwiseGradient3D<float>(splineControlPoint, gradientImage, weight);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_spline_approxLinearPairwiseGradient3D<double>(splineControlPoint, gradientImage, weight);
            break;
        default:
            NR_FATAL_ERROR("Only implemented for single or double precision images");
        }
    } else {
        NR_FATAL_ERROR("Not implemented for 2D images yet");
    }
}
/* *************************************************************** */
