/**
 * @file _reg_tools.cpp
 * @author Marc Modat
 * @date 25/03/2009
 * @brief Set of useful functions
 *
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_tools.h"

/* *************************************************************** */
void reg_checkAndCorrectDimension(nifti_image *image) {
    // Ensure that no dimension is set to zero
    if (image->nx < 1 || image->dim[1] < 1) image->dim[1] = image->nx = 1;
    if (image->ny < 1 || image->dim[2] < 1) image->dim[2] = image->ny = 1;
    if (image->nz < 1 || image->dim[3] < 1) image->dim[3] = image->nz = 1;
    if (image->nt < 1 || image->dim[4] < 1) image->dim[4] = image->nt = 1;
    if (image->nu < 1 || image->dim[5] < 1) image->dim[5] = image->nu = 1;
    if (image->nv < 1 || image->dim[6] < 1) image->dim[6] = image->nv = 1;
    if (image->nw < 1 || image->dim[7] < 1) image->dim[7] = image->nw = 1;
    //Correcting the dim of the images
    for (int i = 1; i < 8; ++i) {
        if (image->dim[i] > 1) {
            image->dim[0] = image->ndim = i;
        }
    }
    // Set the slope to 1 if undefined
    if (image->scl_slope == 0) image->scl_slope = 1.f;
    // Ensure that no spacing is set to zero
    if (image->ny == 1 && (image->dy == 0 || image->pixdim[2] == 0))
        image->dy = image->pixdim[2] = 1;
    if (image->nz == 1 && (image->dz == 0 || image->pixdim[3] == 0))
        image->dz = image->pixdim[3] = 1;
    // Create the qform matrix if required
    if (image->qform_code == 0 && image->sform_code == 0) {
        image->qto_xyz = nifti_quatern_to_mat44(image->quatern_b,
                                                image->quatern_c,
                                                image->quatern_d,
                                                image->qoffset_x,
                                                image->qoffset_y,
                                                image->qoffset_z,
                                                image->dx,
                                                image->dy,
                                                image->dz,
                                                image->qfac);
        image->qto_ijk = nifti_mat44_inverse(image->qto_xyz);
    }
    // Set the voxel spacing to millimetres
    if (image->xyz_units == NIFTI_UNITS_MICRON) {
        for (int d = 1; d <= image->ndim; ++d)
            image->pixdim[d] /= 1000.f;
        image->xyz_units = NIFTI_UNITS_MM;
    }
    if (image->xyz_units == NIFTI_UNITS_METER) {
        for (int d = 1; d <= image->ndim; ++d)
            image->pixdim[d] *= 1000.f;
        image->xyz_units = NIFTI_UNITS_MM;
    }
    image->dx = image->pixdim[1];
    image->dy = image->pixdim[2];
    image->dz = image->pixdim[3];
    image->dt = image->pixdim[4];
    image->du = image->pixdim[5];
    image->dv = image->pixdim[6];
    image->dw = image->pixdim[7];
}
/* *************************************************************** */
bool reg_isAnImageFileName(const char *name) {
    const std::string n(name);
    if (n.find(".nii") != std::string::npos)
        return true;
    if (n.find(".nii.gz") != std::string::npos)
        return true;
    if (n.find(".hdr") != std::string::npos)
        return true;
    if (n.find(".img") != std::string::npos)
        return true;
    if (n.find(".img.gz") != std::string::npos)
        return true;
    if (n.find(".nrrd") != std::string::npos)
        return true;
    if (n.find(".png") != std::string::npos)
        return true;
    return false;
}
/* *************************************************************** */
template<class DataType>
void reg_intensityRescale_core(nifti_image *image,
                               int timePoint,
                               float newMin,
                               float newMax) {
    DataType *imagePtr = static_cast<DataType*>(image->data);
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(image, 3);

    // The rescaling is done for each volume independently
    DataType *volumePtr = &imagePtr[timePoint * voxelNumber];
    DataType currentMin = 0;
    DataType currentMax = 0;
    switch (image->datatype) {
    case NIFTI_TYPE_UINT8:
        currentMin = (DataType)std::numeric_limits<unsigned char>::max();
        currentMax = (DataType)std::numeric_limits<unsigned char>::lowest();
        break;
    case NIFTI_TYPE_INT8:
        currentMin = (DataType)std::numeric_limits<char>::max();
        currentMax = (DataType)std::numeric_limits<char>::lowest();
        break;
    case NIFTI_TYPE_UINT16:
        currentMin = (DataType)std::numeric_limits<unsigned short>::max();
        currentMax = (DataType)std::numeric_limits<unsigned short>::lowest();
        break;
    case NIFTI_TYPE_INT16:
        currentMin = (DataType)std::numeric_limits<short>::max();
        currentMax = (DataType)std::numeric_limits<short>::lowest();
        break;
    case NIFTI_TYPE_UINT32:
        currentMin = (DataType)std::numeric_limits<unsigned>::max();
        currentMax = (DataType)std::numeric_limits<unsigned>::lowest();
        break;
    case NIFTI_TYPE_INT32:
        currentMin = (DataType)std::numeric_limits<int>::max();
        currentMax = (DataType)std::numeric_limits<int>::lowest();
        break;
    case NIFTI_TYPE_FLOAT32:
        currentMin = (DataType)std::numeric_limits<float>::max();
        currentMax = (DataType)std::numeric_limits<float>::lowest();
        break;
    case NIFTI_TYPE_FLOAT64:
        currentMin = (DataType)std::numeric_limits<double>::max();
        currentMax = (DataType)std::numeric_limits<double>::lowest();
        break;
    }

    // Extract the minimal and maximal values from the current volume
    if (image->scl_slope == 0) image->scl_slope = 1.0f;
    for (size_t index = 0; index < voxelNumber; index++) {
        DataType value = (DataType)(*volumePtr++ * image->scl_slope + image->scl_inter);
        if (value == value) {
            currentMin = std::min(currentMin, value);
            currentMax = std::max(currentMax, value);
        }
    }

    // Compute constant values to rescale image intensities
    double currentDiff = (double)(currentMax - currentMin);
    double newDiff = (double)(newMax - newMin);

    // Set the image header information for appropriate display
    image->cal_min = newMin;
    image->cal_max = newMax;

    // Reset the volume pointer to the start of the current volume
    volumePtr = &imagePtr[timePoint * voxelNumber];

    // Iterates over all voxels in the current volume
    for (size_t index = 0; index < voxelNumber; index++) {
        double value = (double)*volumePtr * image->scl_slope + image->scl_inter;
        // Check if the value is defined
        if (value == value) {
            // Normalise the value between 0 and 1
            value = (value - (double)currentMin) / currentDiff;
            // Rescale the value using the specified range
            value = value * newDiff + newMin;
        }
        *volumePtr++ = (DataType)value;
    }
    image->scl_slope = 1.f;
    image->scl_inter = 0.f;
}
/* *************************************************************** */
void reg_intensityRescale(nifti_image *image,
                          int timePoint,
                          float newMin,
                          float newMax) {
    switch (image->datatype) {
    case NIFTI_TYPE_UINT8:
        reg_intensityRescale_core<unsigned char>(image, timePoint, newMin, newMax);
        break;
    case NIFTI_TYPE_INT8:
        reg_intensityRescale_core<char>(image, timePoint, newMin, newMax);
        break;
    case NIFTI_TYPE_UINT16:
        reg_intensityRescale_core<unsigned short>(image, timePoint, newMin, newMax);
        break;
    case NIFTI_TYPE_INT16:
        reg_intensityRescale_core<short>(image, timePoint, newMin, newMax);
        break;
    case NIFTI_TYPE_UINT32:
        reg_intensityRescale_core<unsigned>(image, timePoint, newMin, newMax);
        break;
    case NIFTI_TYPE_INT32:
        reg_intensityRescale_core<int>(image, timePoint, newMin, newMax);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_intensityRescale_core<float>(image, timePoint, newMin, newMax);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_intensityRescale_core<double>(image, timePoint, newMin, newMax);
        break;
    default:
        NR_FATAL_ERROR("The image data type is not supported");
    }
}
/* *************************************************************** */
template<class DataType>
void reg_tools_removeSCLInfo(nifti_image *image) {
    if (image->scl_slope == 1.f && image->scl_inter == 0.f)
        return;
    DataType *imgPtr = static_cast<DataType*>(image->data);
    for (size_t i = 0; i < image->nvox; ++i) {
        imgPtr[i] = imgPtr[i] * (DataType)image->scl_slope + (DataType)image->scl_inter;
    }
    image->scl_slope = 1.f;
    image->scl_inter = 0.f;
}
/* *************************************************************** */
void reg_tools_removeSCLInfo(nifti_image *image) {
    switch (image->datatype) {
    case NIFTI_TYPE_UINT8:
        reg_tools_removeSCLInfo<unsigned char>(image);
        break;
    case NIFTI_TYPE_INT8:
        reg_tools_removeSCLInfo<char>(image);
        break;
    case NIFTI_TYPE_UINT16:
        reg_tools_removeSCLInfo<unsigned short>(image);
        break;
    case NIFTI_TYPE_INT16:
        reg_tools_removeSCLInfo<short>(image);
        break;
    case NIFTI_TYPE_UINT32:
        reg_tools_removeSCLInfo<unsigned>(image);
        break;
    case NIFTI_TYPE_INT32:
        reg_tools_removeSCLInfo<int>(image);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_tools_removeSCLInfo<float>(image);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_tools_removeSCLInfo<double>(image);
        break;
    default:
        NR_FATAL_ERROR("The image data type is not supported");
    }
}
/* *************************************************************** */
void reg_getRealImageSpacing(nifti_image *image, float *spacingValues) {
    float indexVoxel1[3] = { 0, 0, 0 };
    float indexVoxel2[3], realVoxel1[3], realVoxel2[3];
    reg_mat44_mul(&(image->sto_xyz), indexVoxel1, realVoxel1);

    indexVoxel2[1] = indexVoxel2[2] = 0;
    indexVoxel2[0] = 1;
    reg_mat44_mul(&(image->sto_xyz), indexVoxel2, realVoxel2);
    spacingValues[0] = sqrtf(Square(realVoxel1[0] - realVoxel2[0]) + Square(realVoxel1[1] - realVoxel2[1]) + Square(realVoxel1[2] - realVoxel2[2]));

    indexVoxel2[0] = indexVoxel2[2] = 0;
    indexVoxel2[1] = 1;
    reg_mat44_mul(&(image->sto_xyz), indexVoxel2, realVoxel2);
    spacingValues[1] = sqrtf(Square(realVoxel1[0] - realVoxel2[0]) + Square(realVoxel1[1] - realVoxel2[1]) + Square(realVoxel1[2] - realVoxel2[2]));

    if (image->nz > 1) {
        indexVoxel2[0] = indexVoxel2[1] = 0;
        indexVoxel2[2] = 1;
        reg_mat44_mul(&(image->sto_xyz), indexVoxel2, realVoxel2);
        spacingValues[2] = sqrtf(Square(realVoxel1[0] - realVoxel2[0]) + Square(realVoxel1[1] - realVoxel2[1]) + Square(realVoxel1[2] - realVoxel2[2]));
    }
}
/* *************************************************************** */
//this function will threshold an image to the values provided,
//set the scl_slope and sct_inter of the image to 1 and 0 (SSD uses actual image data values),
//and sets cal_min and cal_max to have the min/max image data values
template<class T, class DataType>
void reg_thresholdImage(nifti_image *image, T lowThr, T upThr) {
    DataType *imagePtr = static_cast<DataType*>(image->data);
    T currentMin = std::numeric_limits<T>::max();
    T currentMax = std::numeric_limits<T>::lowest();

    if (image->scl_slope == 0)image->scl_slope = 1.0;

    for (size_t i = 0; i < image->nvox; i++) {
        T value = (T)(imagePtr[i] * image->scl_slope + image->scl_inter);
        if (value == value) {
            value = std::clamp(value, lowThr, upThr);
            currentMin = std::min(currentMin, value);
            currentMax = std::max(currentMax, value);
        }
        imagePtr[i] = (DataType)value;
    }

    image->cal_min = static_cast<float>(currentMin);
    image->cal_max = static_cast<float>(currentMax);
}
/* *************************************************************** */
template<class T>
void reg_thresholdImage(nifti_image *image, T lowThr, T upThr) {
    switch (image->datatype) {
    case NIFTI_TYPE_UINT8:
        reg_thresholdImage<T, unsigned char>(image, lowThr, upThr);
        break;
    case NIFTI_TYPE_INT8:
        reg_thresholdImage<T, char>(image, lowThr, upThr);
        break;
    case NIFTI_TYPE_UINT16:
        reg_thresholdImage<T, unsigned short>(image, lowThr, upThr);
        break;
    case NIFTI_TYPE_INT16:
        reg_thresholdImage<T, short>(image, lowThr, upThr);
        break;
    case NIFTI_TYPE_UINT32:
        reg_thresholdImage<T, unsigned>(image, lowThr, upThr);
        break;
    case NIFTI_TYPE_INT32:
        reg_thresholdImage<T, int>(image, lowThr, upThr);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_thresholdImage<T, float>(image, lowThr, upThr);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_thresholdImage<T, double>(image, lowThr, upThr);
        break;
    default:
        NR_FATAL_ERROR("The image data type is not supported");
    }
}
template void reg_thresholdImage<float>(nifti_image*, float, float);
template void reg_thresholdImage<double>(nifti_image*, double, double);
/* *************************************************************** */
template <class PrecisionType, class DataType>
PrecisionType reg_getMaximalLength(const nifti_image *image,
                                   const bool optimiseX,
                                   const bool optimiseY,
                                   const bool optimiseZ) {
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(image, 3);
    const DataType *dataPtrX = static_cast<DataType*>(image->data);
    const DataType *dataPtrY = &dataPtrX[voxelNumber];
    const DataType *dataPtrZ = &dataPtrY[voxelNumber];
    PrecisionType max = 0;
    for (size_t i = 0; i < voxelNumber; i++) {
        const PrecisionType valX = optimiseX ? static_cast<PrecisionType>(*dataPtrX++) : 0;
        const PrecisionType valY = optimiseY ? static_cast<PrecisionType>(*dataPtrY++) : 0;
        const PrecisionType valZ = optimiseZ ? static_cast<PrecisionType>(*dataPtrZ++) : 0;
        const PrecisionType length = static_cast<PrecisionType>(sqrt(valX * valX + valY * valY + valZ * valZ));
        max = std::max(length, max);
    }
    return max;
}
/* *************************************************************** */
template <class PrecisionType>
PrecisionType reg_getMaximalLength(const nifti_image *image,
                                   const bool optimiseX,
                                   const bool optimiseY,
                                   const bool optimiseZ) {
    switch (image->datatype) {
    case NIFTI_TYPE_FLOAT32:
        return reg_getMaximalLength<PrecisionType, float>(image, optimiseX, optimiseY, image->nz > 1 ? optimiseZ : false);
        break;
    case NIFTI_TYPE_FLOAT64:
        return reg_getMaximalLength<PrecisionType, double>(image, optimiseX, optimiseY, image->nz > 1 ? optimiseZ : false);
        break;
    }
    return EXIT_SUCCESS;
}
template float reg_getMaximalLength<float>(const nifti_image*, const bool, const bool, const bool);
template double reg_getMaximalLength<double>(const nifti_image*, const bool, const bool, const bool);
/* *************************************************************** */
template <class NewType, class DataType>
void reg_tools_changeDatatype(nifti_image *image, int type) {
    // the initial array is saved and freed
    DataType *initialValue = (DataType*)malloc(image->nvox * sizeof(DataType));
    memcpy(initialValue, image->data, image->nvox * sizeof(DataType));

    // the new array is allocated and then filled
    if (type > -1) {
        image->datatype = type;
    } else {
        if (sizeof(NewType) == sizeof(unsigned char)) {
            image->datatype = NIFTI_TYPE_UINT8;
            NR_DEBUG("new datatype is NIFTI_TYPE_UINT8");
        } else if (sizeof(NewType) == sizeof(float)) {
            image->datatype = NIFTI_TYPE_FLOAT32;
            NR_DEBUG("new datatype is NIFTI_TYPE_FLOAT32");
        } else if (sizeof(NewType) == sizeof(double)) {
            image->datatype = NIFTI_TYPE_FLOAT64;
            NR_DEBUG("new datatype is NIFTI_TYPE_FLOAT64");
        } else {
            NR_FATAL_ERROR("Only change to unsigned char, float or double are supported");
        }
    }
    free(image->data);
    image->nbyper = sizeof(NewType);
    image->data = calloc(image->nvox, sizeof(NewType));
    NewType *dataPtr = static_cast<NewType*>(image->data);
    for (size_t i = 0; i < image->nvox; i++)
        dataPtr[i] = static_cast<NewType>(initialValue[i]);

    free(initialValue);
}
/* *************************************************************** */
template <class NewType>
void reg_tools_changeDatatype(nifti_image *image, int type) {
    switch (image->datatype) {
    case NIFTI_TYPE_UINT8:
        reg_tools_changeDatatype<NewType, unsigned char>(image, type);
        break;
    case NIFTI_TYPE_INT8:
        reg_tools_changeDatatype<NewType, char>(image, type);
        break;
    case NIFTI_TYPE_UINT16:
        reg_tools_changeDatatype<NewType, unsigned short>(image, type);
        break;
    case NIFTI_TYPE_INT16:
        reg_tools_changeDatatype<NewType, short>(image, type);
        break;
    case NIFTI_TYPE_UINT32:
        reg_tools_changeDatatype<NewType, unsigned>(image, type);
        break;
    case NIFTI_TYPE_INT32:
        reg_tools_changeDatatype<NewType, int>(image, type);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_tools_changeDatatype<NewType, float>(image, type);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_tools_changeDatatype<NewType, double>(image, type);
        break;
    default:
        NR_FATAL_ERROR("Unsupported datatype");
    }
}
template void reg_tools_changeDatatype<unsigned char>(nifti_image*, int);
template void reg_tools_changeDatatype<unsigned short>(nifti_image*, int);
template void reg_tools_changeDatatype<unsigned>(nifti_image*, int);
template void reg_tools_changeDatatype<char>(nifti_image*, int);
template void reg_tools_changeDatatype<short>(nifti_image*, int);
template void reg_tools_changeDatatype<int>(nifti_image*, int);
template void reg_tools_changeDatatype<float>(nifti_image*, int);
template void reg_tools_changeDatatype<double>(nifti_image*, int);
/* *************************************************************** */
struct Operation {
    enum class Type { Add, Subtract, Multiply, Divide } type;
    Operation(Type type) : type(type) {}
    double operator()(const double lhs, const double rhs) const {
        switch (type) {
        case Type::Add:
            return lhs + rhs;
        case Type::Subtract:
            return lhs - rhs;
        case Type::Multiply:
            return lhs * rhs;
        case Type::Divide:
            return lhs / rhs;
        default:
            NR_FATAL_ERROR("Unsupported operation");
            return 0;
        }
    }
};
/* *************************************************************** */
template <class Type>
void reg_tools_operationImageToImage(const nifti_image *img1,
                                     const nifti_image *img2,
                                     nifti_image *res,
                                     const Operation& operation) {
    const Type *img1Ptr = static_cast<Type*>(img1->data);
    const Type *img2Ptr = static_cast<Type*>(img2->data);
    Type *resPtr = static_cast<Type*>(res->data);

    const float sclSlope1 = img1->scl_slope == 0 ? 1 : img1->scl_slope;
    const float sclSlope2 = img2->scl_slope == 0 ? 1 : img2->scl_slope;

    res->scl_slope = sclSlope1;
    res->scl_inter = img1->scl_inter;

#ifdef _WIN32
    long i;
    const long voxelNumber = (long)res->nvox;
#else
    size_t i;
    const size_t voxelNumber = res->nvox;
#endif

#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(voxelNumber,resPtr,img1Ptr,img2Ptr,img1,img2,sclSlope1,sclSlope2,operation)
#endif
    for (i = 0; i < voxelNumber; i++)
        resPtr[i] = static_cast<Type>((operation(img1Ptr[i] * sclSlope1 + img1->scl_inter,
                                                 img2Ptr[i] * sclSlope2 + img2->scl_inter) - img1->scl_inter) / sclSlope1);
}
/* *************************************************************** */
void reg_tools_addImageToImage(const nifti_image *img1,
                               const nifti_image *img2,
                               nifti_image *res) {
    if (img1->datatype != res->datatype || img2->datatype != res->datatype)
        NR_FATAL_ERROR("Input images are expected to be of the same type");
    if (img1->nvox != res->nvox || img2->nvox != res->nvox)
        NR_FATAL_ERROR("Input images are expected to have the same size");
    Operation operation(Operation::Type::Add);
    switch (img1->datatype) {
    case NIFTI_TYPE_UINT8:
        reg_tools_operationImageToImage<unsigned char>(img1, img2, res, operation);
        break;
    case NIFTI_TYPE_INT8:
        reg_tools_operationImageToImage<char>(img1, img2, res, operation);
        break;
    case NIFTI_TYPE_UINT16:
        reg_tools_operationImageToImage<unsigned short>(img1, img2, res, operation);
        break;
    case NIFTI_TYPE_INT16:
        reg_tools_operationImageToImage<short>(img1, img2, res, operation);
        break;
    case NIFTI_TYPE_UINT32:
        reg_tools_operationImageToImage<unsigned>(img1, img2, res, operation);
        break;
    case NIFTI_TYPE_INT32:
        reg_tools_operationImageToImage<int>(img1, img2, res, operation);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_tools_operationImageToImage<float>(img1, img2, res, operation);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_tools_operationImageToImage<double>(img1, img2, res, operation);
        break;
    default:
        NR_FATAL_ERROR("Unsupported datatype");
    }
}
/* *************************************************************** */
void reg_tools_subtractImageFromImage(const nifti_image *img1,
                                      const nifti_image *img2,
                                      nifti_image *res) {
    if (img1->datatype != res->datatype || img2->datatype != res->datatype)
        NR_FATAL_ERROR("Input images are expected to be of the same type");
    if (img1->nvox != res->nvox || img2->nvox != res->nvox)
        NR_FATAL_ERROR("Input images are expected to have the same size");
    Operation operation(Operation::Type::Subtract);
    switch (img1->datatype) {
    case NIFTI_TYPE_UINT8:
        reg_tools_operationImageToImage<unsigned char>(img1, img2, res, operation);
        break;
    case NIFTI_TYPE_INT8:
        reg_tools_operationImageToImage<char>(img1, img2, res, operation);
        break;
    case NIFTI_TYPE_UINT16:
        reg_tools_operationImageToImage<unsigned short>(img1, img2, res, operation);
        break;
    case NIFTI_TYPE_INT16:
        reg_tools_operationImageToImage<short>(img1, img2, res, operation);
        break;
    case NIFTI_TYPE_UINT32:
        reg_tools_operationImageToImage<unsigned>(img1, img2, res, operation);
        break;
    case NIFTI_TYPE_INT32:
        reg_tools_operationImageToImage<int>(img1, img2, res, operation);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_tools_operationImageToImage<float>(img1, img2, res, operation);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_tools_operationImageToImage<double>(img1, img2, res, operation);
        break;
    default:
        NR_FATAL_ERROR("Unsupported datatype");
    }
}
/* *************************************************************** */
void reg_tools_multiplyImageToImage(const nifti_image *img1,
                                    const nifti_image *img2,
                                    nifti_image *res) {
    if (img1->datatype != res->datatype || img2->datatype != res->datatype)
        NR_FATAL_ERROR("Input images are expected to be of the same type");
    if (img1->nvox != res->nvox || img2->nvox != res->nvox)
        NR_FATAL_ERROR("Input images are expected to have the same size");
    Operation operation(Operation::Type::Multiply);
    switch (img1->datatype) {
    case NIFTI_TYPE_UINT8:
        reg_tools_operationImageToImage<unsigned char>(img1, img2, res, operation);
        break;
    case NIFTI_TYPE_INT8:
        reg_tools_operationImageToImage<char>(img1, img2, res, operation);
        break;
    case NIFTI_TYPE_UINT16:
        reg_tools_operationImageToImage<unsigned short>(img1, img2, res, operation);
        break;
    case NIFTI_TYPE_INT16:
        reg_tools_operationImageToImage<short>(img1, img2, res, operation);
        break;
    case NIFTI_TYPE_UINT32:
        reg_tools_operationImageToImage<unsigned>(img1, img2, res, operation);
        break;
    case NIFTI_TYPE_INT32:
        reg_tools_operationImageToImage<int>(img1, img2, res, operation);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_tools_operationImageToImage<float>(img1, img2, res, operation);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_tools_operationImageToImage<double>(img1, img2, res, operation);
        break;
    default:
        NR_FATAL_ERROR("Unsupported datatype");
    }
}
/* *************************************************************** */
void reg_tools_divideImageToImage(const nifti_image *img1,
                                  const nifti_image *img2,
                                  nifti_image *res) {
    if (img1->datatype != res->datatype || img2->datatype != res->datatype)
        NR_FATAL_ERROR("Input images are expected to be of the same type");
    if (img1->nvox != res->nvox || img2->nvox != res->nvox)
        NR_FATAL_ERROR("Input images are expected to have the same size");
    Operation operation(Operation::Type::Divide);
    switch (img1->datatype) {
    case NIFTI_TYPE_UINT8:
        reg_tools_operationImageToImage<unsigned char>(img1, img2, res, operation);
        break;
    case NIFTI_TYPE_INT8:
        reg_tools_operationImageToImage<char>(img1, img2, res, operation);
        break;
    case NIFTI_TYPE_UINT16:
        reg_tools_operationImageToImage<unsigned short>(img1, img2, res, operation);
        break;
    case NIFTI_TYPE_INT16:
        reg_tools_operationImageToImage<short>(img1, img2, res, operation);
        break;
    case NIFTI_TYPE_UINT32:
        reg_tools_operationImageToImage<unsigned>(img1, img2, res, operation);
        break;
    case NIFTI_TYPE_INT32:
        reg_tools_operationImageToImage<int>(img1, img2, res, operation);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_tools_operationImageToImage<float>(img1, img2, res, operation);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_tools_operationImageToImage<double>(img1, img2, res, operation);
        break;
    default:
        NR_FATAL_ERROR("Unsupported datatype");
    }
}
/* *************************************************************** */
template <class Type>
void reg_tools_operationValueToImage(const nifti_image *img,
                                     nifti_image *res,
                                     const double val,
                                     const Operation& operation) {
    const Type *imgPtr = static_cast<Type*>(img->data);
    Type *resPtr = static_cast<Type*>(res->data);

    const float sclSlope = img->scl_slope == 0 ? 1 : img->scl_slope;

    res->scl_slope = sclSlope;
    res->scl_inter = img->scl_inter;

#ifdef _WIN32
    long i;
    const long voxelNumber = (long)res->nvox;
#else
    size_t i;
    const size_t voxelNumber = res->nvox;
#endif

#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(voxelNumber,resPtr,imgPtr,img,val,sclSlope,operation)
#endif
    for (i = 0; i < voxelNumber; i++)
        resPtr[i] = static_cast<Type>((operation(imgPtr[i] * sclSlope + img->scl_inter, val) - img->scl_inter) / sclSlope);
}
/* *************************************************************** */
void reg_tools_addValueToImage(const nifti_image *img,
                               nifti_image *res,
                               const double val) {
    if (img->datatype != res->datatype)
        NR_FATAL_ERROR("Input and output image are expected to be of the same type");
    if (img->nvox != res->nvox)
        NR_FATAL_ERROR("Input images are expected to have the same size");
    Operation operation(Operation::Type::Add);
    switch (img->datatype) {
    case NIFTI_TYPE_UINT8:
        reg_tools_operationValueToImage<unsigned char>(img, res, val, operation);
        break;
    case NIFTI_TYPE_INT8:
        reg_tools_operationValueToImage<char>(img, res, val, operation);
        break;
    case NIFTI_TYPE_UINT16:
        reg_tools_operationValueToImage<unsigned short>(img, res, val, operation);
        break;
    case NIFTI_TYPE_INT16:
        reg_tools_operationValueToImage<short>(img, res, val, operation);
        break;
    case NIFTI_TYPE_UINT32:
        reg_tools_operationValueToImage<unsigned>(img, res, val, operation);
        break;
    case NIFTI_TYPE_INT32:
        reg_tools_operationValueToImage<int>(img, res, val, operation);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_tools_operationValueToImage<float>(img, res, val, operation);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_tools_operationValueToImage<double>(img, res, val, operation);
        break;
    default:
        NR_FATAL_ERROR("Image data type is not supported");
    }
}
/* *************************************************************** */
void reg_tools_subtractValueFromImage(const nifti_image *img,
                                      nifti_image *res,
                                      const double val) {
    if (img->datatype != res->datatype)
        NR_FATAL_ERROR("Input and output image are expected to be of the same type");
    if (img->nvox != res->nvox)
        NR_FATAL_ERROR("Input images are expected to have the same size");
    Operation operation(Operation::Type::Subtract);
    switch (img->datatype) {
    case NIFTI_TYPE_UINT8:
        reg_tools_operationValueToImage<unsigned char>(img, res, val, operation);
        break;
    case NIFTI_TYPE_INT8:
        reg_tools_operationValueToImage<char>(img, res, val, operation);
        break;
    case NIFTI_TYPE_UINT16:
        reg_tools_operationValueToImage<unsigned short>(img, res, val, operation);
        break;
    case NIFTI_TYPE_INT16:
        reg_tools_operationValueToImage<short>(img, res, val, operation);
        break;
    case NIFTI_TYPE_UINT32:
        reg_tools_operationValueToImage<unsigned>(img, res, val, operation);
        break;
    case NIFTI_TYPE_INT32:
        reg_tools_operationValueToImage<int>(img, res, val, operation);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_tools_operationValueToImage<float>(img, res, val, operation);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_tools_operationValueToImage<double>(img, res, val, operation);
        break;
    default:
        NR_FATAL_ERROR("Image data type is not supported");
    }
}
/* *************************************************************** */
void reg_tools_multiplyValueToImage(const nifti_image *img,
                                    nifti_image *res,
                                    const double val) {
    if (img->datatype != res->datatype)
        NR_FATAL_ERROR("Input and output image are expected to be of the same type");
    if (img->nvox != res->nvox)
        NR_FATAL_ERROR("Input images are expected to have the same size");
    Operation operation(Operation::Type::Multiply);
    switch (img->datatype) {
    case NIFTI_TYPE_UINT8:
        reg_tools_operationValueToImage<unsigned char>(img, res, val, operation);
        break;
    case NIFTI_TYPE_INT8:
        reg_tools_operationValueToImage<char>(img, res, val, operation);
        break;
    case NIFTI_TYPE_UINT16:
        reg_tools_operationValueToImage<unsigned short>(img, res, val, operation);
        break;
    case NIFTI_TYPE_INT16:
        reg_tools_operationValueToImage<short>(img, res, val, operation);
        break;
    case NIFTI_TYPE_UINT32:
        reg_tools_operationValueToImage<unsigned>(img, res, val, operation);
        break;
    case NIFTI_TYPE_INT32:
        reg_tools_operationValueToImage<int>(img, res, val, operation);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_tools_operationValueToImage<float>(img, res, val, operation);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_tools_operationValueToImage<double>(img, res, val, operation);
        break;
    default:
        NR_FATAL_ERROR("Image data type is not supported");
    }
}
/* *************************************************************** */
void reg_tools_divideValueToImage(const nifti_image *img,
                                  nifti_image *res,
                                  const double val) {
    if (img->datatype != res->datatype)
        NR_FATAL_ERROR("Input and output image are expected to be of the same type");
    if (img->nvox != res->nvox)
        NR_FATAL_ERROR("Input images are expected to have the same size");
    Operation operation(Operation::Type::Divide);
    switch (img->datatype) {
    case NIFTI_TYPE_UINT8:
        reg_tools_operationValueToImage<unsigned char>(img, res, val, operation);
        break;
    case NIFTI_TYPE_INT8:
        reg_tools_operationValueToImage<char>(img, res, val, operation);
        break;
    case NIFTI_TYPE_UINT16:
        reg_tools_operationValueToImage<unsigned short>(img, res, val, operation);
        break;
    case NIFTI_TYPE_INT16:
        reg_tools_operationValueToImage<short>(img, res, val, operation);
        break;
    case NIFTI_TYPE_UINT32:
        reg_tools_operationValueToImage<unsigned>(img, res, val, operation);
        break;
    case NIFTI_TYPE_INT32:
        reg_tools_operationValueToImage<int>(img, res, val, operation);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_tools_operationValueToImage<float>(img, res, val, operation);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_tools_operationValueToImage<double>(img, res, val, operation);
        break;
    default:
        NR_FATAL_ERROR("Image data type is not supported");
    }
}
/* *************************************************************** */
template <class DataType>
void reg_tools_kernelConvolution(nifti_image *image,
                                 const float *sigma,
                                 const ConvKernelType kernelType,
                                 const int *mask,
                                 const bool *timePoints,
                                 const bool *axes) {
    if (image->nx > 2048 || image->ny > 2048 || image->nz > 2048)
        NR_FATAL_ERROR("This function does not support images with dimensions larger than 2048");

#ifdef WIN32
    long index;
    const long voxelNumber = (long)NiftiImage::calcVoxelNumber(image, 3);
#else
    size_t index;
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(image, 3);
#endif

    DataType *imagePtr = static_cast<DataType*>(image->data);
    const int imageDims[3]{ image->nx, image->ny, image->nz };

    unique_ptr<bool[]> nanImagePtr{ new bool[voxelNumber]() };
    unique_ptr<float[]> densityPtr{ new float[voxelNumber]() };

    // Loop over the dimension higher than 3
    for (int t = 0; t < image->nt * image->nu; t++) {
        if (timePoints[t]) {
            DataType *intensityPtr = &imagePtr[t * voxelNumber];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(densityPtr, intensityPtr, mask, nanImagePtr, voxelNumber)
#endif
            for (index = 0; index < voxelNumber; index++) {
                densityPtr[index] = mask[index] >= 0 && intensityPtr[index] == intensityPtr[index] ? 1.f : 0;
                nanImagePtr[index] = !static_cast<bool>(densityPtr[index]);
                if (nanImagePtr[index]) intensityPtr[index] = 0;
            }
            // Loop over the x, y and z dimensions
            for (int n = 0; n < 3; n++) {
                if (axes[n] && image->dim[n] > 1) {
                    double temp;
                    if (sigma[t] > 0) temp = sigma[t] / image->pixdim[n + 1]; // mm to voxel
                    else temp = fabs(sigma[t]); // voxel-based if negative value
                    int radius = 0;
                    // Define the kernel size
                    if (kernelType == ConvKernelType::Mean || kernelType == ConvKernelType::Linear) {
                        // Mean or linear filtering
                        radius = static_cast<int>(temp);
                    } else if (kernelType == ConvKernelType::Gaussian) {
                        // Gaussian kernel
                        radius = static_cast<int>(temp * 3.0f);
                    } else if (kernelType == ConvKernelType::Cubic) {
                        // Spline kernel
                        radius = static_cast<int>(temp * 2.0f);
                    } else {
                        NR_FATAL_ERROR("Unknown kernel type");
                    }
                    if (radius > 0) {
                        // Allocate the kernel
                        float kernel[4096];
                        double kernelSum = 0;
                        // Fill the kernel
                        if (kernelType == ConvKernelType::Cubic) {
                            // Compute the Cubic Spline kernel
                            for (int i = -radius; i <= radius; i++) {
                                // temp contains the kernel node spacing
                                double relative = fabs(i / temp);
                                if (relative < 1.0)
                                    kernel[i + radius] = static_cast<float>(2.0 / 3.0 - Square(relative) + 0.5 * Cube(relative));
                                else if (relative < 2.0)
                                    kernel[i + radius] = static_cast<float>(-Cube(relative - 2.0) / 6.0);
                                else kernel[i + radius] = 0;
                                kernelSum += kernel[i + radius];
                            }
                        } else if (kernelType == ConvKernelType::Gaussian) {
                            // Compute the Gaussian kernel
                            for (int i = -radius; i <= radius; i++) {
                                // 2.506... = sqrt(2*pi)
                                // temp contains the sigma in voxel
                                kernel[radius + i] = static_cast<float>(exp(-Square(i) / (2.0 * Square(temp))) / (temp * 2.506628274631));
                                kernelSum += kernel[radius + i];
                            }
                        } else if (kernelType == ConvKernelType::Linear) {
                            // Compute the linear kernel
                            for (int i = -radius; i <= radius; i++) {
                                kernel[radius + i] = 1.f - fabs(i / static_cast<float>(radius));
                                kernelSum += kernel[radius + i];
                            }
                        } else if (kernelType == ConvKernelType::Mean && imageDims[2] == 1) {
                            // Compute the mean kernel
                            for (int i = -radius; i <= radius; i++) {
                                kernel[radius + i] = 1.f;
                                kernelSum += kernel[radius + i];
                            }
                        }
                        // No kernel is required for the mean filtering
                        // No need for kernel normalisation as this is handled by the density function
                        NR_DEBUG("Convolution type[" << int(kernelType) << "] dim[" << n << "] tp[" << t << "] radius[" << radius << "] kernelSum[" << kernelSum << "]");

                        int planeNumber, planeIndex, lineOffset;
                        int lineIndex, shiftPre, shiftPst, k;
                        switch (n) {
                        case 0:
                            planeNumber = imageDims[1] * imageDims[2];
                            lineOffset = 1;
                            break;
                        case 1:
                            planeNumber = imageDims[0] * imageDims[2];
                            lineOffset = imageDims[0];
                            break;
                        case 2:
                            planeNumber = imageDims[0] * imageDims[1];
                            lineOffset = planeNumber;
                            break;
                        }

                        size_t realIndex;
                        float *kernelPtr, kernelValue;
                        double densitySum, intensitySum;
                        DataType *currentIntensityPtr = nullptr;
                        float *currentDensityPtr = nullptr;
                        DataType bufferIntensity[2048];
                        float bufferDensity[2048];
                        double bufferIntensityCur = 0;
                        double bufferDensityCur = 0;

#ifdef USE_SSE
                        union {
                            __m128 m;
                            float f[4];
                        } intensity_sum_sse, density_sum_sse;
                        __m128 kernel_sse, intensity_sse, density_sse;
#endif

#ifdef _OPENMP
#ifdef USE_SSE
#pragma omp parallel for default(none) \
   shared(imageDims, intensityPtr, densityPtr, radius, kernel, lineOffset, n, planeNumber, kernelSum) \
   private(realIndex, currentIntensityPtr, currentDensityPtr, lineIndex, bufferIntensity, \
   bufferDensity, shiftPre, shiftPst, kernelPtr, kernelValue, densitySum, intensitySum, \
   k, bufferIntensityCur, bufferDensityCur, \
   kernel_sse, intensity_sse, density_sse, intensity_sum_sse, density_sum_sse)
#else
#pragma omp parallel for default(none) \
   shared(imageDims, intensityPtr, densityPtr, radius, kernel, lineOffset, n, planeNumber, kernelSum) \
   private(realIndex, currentIntensityPtr, currentDensityPtr, lineIndex, bufferIntensity, \
   bufferDensity, shiftPre, shiftPst, kernelPtr, kernelValue, densitySum, intensitySum, \
   k, bufferIntensityCur, bufferDensityCur)
#endif
#endif // _OPENMP
                        // Loop over the different voxel
                        for (planeIndex = 0; planeIndex < planeNumber; ++planeIndex) {
                            switch (n) {
                            case 0:
                                realIndex = planeIndex * imageDims[0];
                                break;
                            case 1:
                                realIndex = (planeIndex / imageDims[0]) * imageDims[0] * imageDims[1] + planeIndex % imageDims[0];
                                break;
                            case 2:
                                realIndex = planeIndex;
                                break;
                            default:
                                realIndex = 0;
                            }
                            // Fetch the current line into a stack buffer
                            currentIntensityPtr = &intensityPtr[realIndex];
                            currentDensityPtr = &densityPtr[realIndex];
                            for (lineIndex = 0; lineIndex < imageDims[n]; ++lineIndex) {
                                bufferIntensity[lineIndex] = *currentIntensityPtr;
                                bufferDensity[lineIndex] = *currentDensityPtr;
                                currentIntensityPtr += lineOffset;
                                currentDensityPtr += lineOffset;
                            }
                            if (kernelSum > 0) {
                                // Perform the kernel convolution along one line
                                for (lineIndex = 0; lineIndex < imageDims[n]; ++lineIndex) {
                                    // Define the kernel boundaries
                                    shiftPre = lineIndex - radius;
                                    shiftPst = lineIndex + radius + 1;
                                    if (shiftPre < 0) {
                                        kernelPtr = &kernel[-shiftPre];
                                        shiftPre = 0;
                                    } else kernelPtr = &kernel[0];
                                    if (shiftPst > imageDims[n]) shiftPst = imageDims[n];
                                    // Set the current values to zero
                                    // Increment the current value by performing the weighted sum
#ifdef USE_SSE
                                    intensity_sum_sse.m = _mm_set_ps1(0);
                                    density_sum_sse.m = _mm_set_ps1(0);
                                    k = shiftPre;
                                    while (k < shiftPst - 3) {
                                        kernel_sse = _mm_set_ps(kernelPtr[0], kernelPtr[1], kernelPtr[2], kernelPtr[3]);
                                        kernelPtr += 4;
                                        intensity_sse = _mm_set_ps(static_cast<float>(bufferIntensity[k]),
                                                                   static_cast<float>(bufferIntensity[k + 1]),
                                                                   static_cast<float>(bufferIntensity[k + 2]),
                                                                   static_cast<float>(bufferIntensity[k + 3]));
                                        density_sse = _mm_set_ps(bufferDensity[k],
                                                                 bufferDensity[k + 1],
                                                                 bufferDensity[k + 2],
                                                                 bufferDensity[k + 3]);
                                        k += 4;
                                        intensity_sum_sse.m = _mm_add_ps(_mm_mul_ps(kernel_sse, intensity_sse), intensity_sum_sse.m);
                                        density_sum_sse.m = _mm_add_ps(_mm_mul_ps(kernel_sse, density_sse), density_sum_sse.m);
                                    }
#ifdef __SSE3__
                                    intensity_sum_sse.m = _mm_hadd_ps(intensity_sum_sse.m, density_sum_sse.m);
                                    intensity_sum_sse.m = _mm_hadd_ps(intensity_sum_sse.m, intensity_sum_sse.m);
                                    intensitySum = intensity_sum_sse.f[0];
                                    densitySum = intensity_sum_sse.f[1];
#else
                                    intensitySum = intensity_sum_sse.f[0] + intensity_sum_sse.f[1] + intensity_sum_sse.f[2] + intensity_sum_sse.f[3];
                                    densitySum = density_sum_sse.f[0] + density_sum_sse.f[1] + density_sum_sse.f[2] + density_sum_sse.f[3];
#endif
                                    while (k < shiftPst) {
                                        kernelValue = *kernelPtr++;
                                        intensitySum += kernelValue * bufferIntensity[k];
                                        densitySum += kernelValue * bufferDensity[k++];
                                    }
#else
                                    intensitySum = 0;
                                    densitySum = 0;
                                    for (k = shiftPre; k < shiftPst; ++k) {
                                        kernelValue = *kernelPtr++;
                                        intensitySum += kernelValue * bufferIntensity[k];
                                        densitySum += kernelValue * bufferDensity[k];
                                    }
#endif
                                    // Store the computed value inplace
                                    intensityPtr[realIndex] = static_cast<DataType>(intensitySum);
                                    densityPtr[realIndex] = static_cast<float>(densitySum);
                                    realIndex += lineOffset;
                                } // line convolution
                            } // kernel sum
                            else {
                                for (lineIndex = 1; lineIndex < imageDims[n]; ++lineIndex) {
                                    bufferIntensity[lineIndex] += bufferIntensity[lineIndex - 1];
                                    bufferDensity[lineIndex] += bufferDensity[lineIndex - 1];
                                }
                                shiftPre = -radius - 1;
                                shiftPst = radius;
                                for (lineIndex = 0; lineIndex < imageDims[n]; ++lineIndex, ++shiftPre, ++shiftPst) {
                                    if (shiftPre > -1) {
                                        if (shiftPst < imageDims[n]) {
                                            bufferIntensityCur = bufferIntensity[shiftPre] - bufferIntensity[shiftPst];
                                            bufferDensityCur = bufferDensity[shiftPre] - bufferDensity[shiftPst];
                                        } else {
                                            bufferIntensityCur = bufferIntensity[shiftPre] - bufferIntensity[imageDims[n] - 1];
                                            bufferDensityCur = bufferDensity[shiftPre] - bufferDensity[imageDims[n] - 1];
                                        }
                                    } else {
                                        if (shiftPst < imageDims[n]) {
                                            bufferIntensityCur = -bufferIntensity[shiftPst];
                                            bufferDensityCur = -bufferDensity[shiftPst];
                                        } else {
                                            bufferIntensityCur = 0;
                                            bufferDensityCur = 0;
                                        }
                                    }
                                    intensityPtr[realIndex] = static_cast<DataType>(bufferIntensityCur);
                                    densityPtr[realIndex] = static_cast<float>(bufferDensityCur);
                                    realIndex += lineOffset;
                                } // line convolution of mean filter
                            } // No kernel computation
                        } // pixel in starting plane
                    } // radius > 0
                } // active axis
            } // axes
            // Normalise per time point
#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(voxelNumber, intensityPtr, densityPtr, nanImagePtr)
#endif
            for (index = 0; index < voxelNumber; ++index) {
                if (nanImagePtr[index])
                    intensityPtr[index] = std::numeric_limits<DataType>::quiet_NaN();
                else intensityPtr[index] = static_cast<DataType>(intensityPtr[index] / densityPtr[index]);
            }
        } // check if the time point is active
    } // loop over the time points
}
/* *************************************************************** */
template <class DataType>
void reg_tools_labelKernelConvolution_core(nifti_image *image,
                                           float varianceX,
                                           float varianceY,
                                           float varianceZ,
                                           int *mask,
                                           bool *timePoints) {
    if (image->nx > 2048 || image->ny > 2048 || image->nz > 2048)
        NR_FATAL_ERROR("This function does not support images with dimension > 2048");
#ifdef WIN32
    long index;
    const long voxelNumber = (long)NiftiImage::calcVoxelNumber(image, 3);
#else
    size_t index;
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(image, 3);
#endif
    DataType *imagePtr = static_cast<DataType*>(image->data);

    const int activeTimePointCount = image->nt * image->nu;
    bool *activeTimePoints = (bool*)calloc(activeTimePointCount, sizeof(bool));
    // Check if input time points and masks are nullptr
    if (timePoints == nullptr) {
        // All time points are considered as active
        for (int i = 0; i < activeTimePointCount; i++) activeTimePoints[i] = true;
    } else for (int i = 0; i < activeTimePointCount; i++) activeTimePoints[i] = timePoints[i];

    int *currentMask = nullptr;
    if (mask == nullptr) {
        currentMask = (int*)calloc(voxelNumber, sizeof(int));
    } else currentMask = mask;


    bool *nanImagePtr = (bool*)calloc(voxelNumber, sizeof(bool));
    DataType *tmpImagePtr = (DataType*)calloc(voxelNumber, sizeof(DataType));

    typedef std::map<DataType, float> DataPointMap;
    typedef std::pair<DataType, float> DataPointPair;
    typedef typename std::map<DataType, float>::iterator DataPointMapIt;

    // Loop over the dimension higher than 3
    for (int t = 0; t < activeTimePointCount; t++) {
        if (activeTimePoints[t]) {
            DataType *intensityPtr = &imagePtr[t * voxelNumber];
            for (index = 0; index < voxelNumber; index++) {
                nanImagePtr[index] = (intensityPtr[index] == intensityPtr[index]) ? true : false;
                nanImagePtr[index] = (currentMask[index] >= 0) ? nanImagePtr[index] : false;
            }
            float gaussX_var = varianceX;
            float gaussY_var = varianceY;
            float gaussZ_var = varianceZ;
            index = 0;
            int currentXYZposition[3] = { 0 };
            int dim_array[3] = { image->nx, image->ny, image->nz };
            int shiftdirection[3] = { 1, image->nx, image->nx * image->ny };

            int kernelXsize, kernelXshift, shiftXstart, shiftXstop;
            int kernelYsize, kernelYshift, shiftYstart, shiftYstop;
            int kernelZsize, kernelZshift, shiftZstart, shiftZstop;
            int shiftx, shifty, shiftz;
            int indexNeighbour;
            float kernelval;
            DataType maxindex;
            double maxval;
            DataPointMapIt location, currIterator;
            DataPointMap tmp_lab;

            for (int currentZposition = 0; currentZposition < dim_array[2]; currentZposition++) {
                currentXYZposition[2] = currentZposition;
                for (currentXYZposition[1] = 0; currentXYZposition[1] < dim_array[1]; currentXYZposition[1]++) {
                    for (currentXYZposition[0] = 0; currentXYZposition[0] < dim_array[0]; currentXYZposition[0]++) {

                        tmp_lab.clear();
                        index = currentXYZposition[0] + (currentXYZposition[1] + currentXYZposition[2] * dim_array[1]) * dim_array[0];

                        // Calculate allowed kernel shifts
                        kernelXsize = (int)(sqrtf(gaussX_var) * 6.0f) % 2 != 0 ?
                            (int)(sqrtf(gaussX_var) * 6.0f) : (int)(sqrtf(gaussX_var) * 6.0f) + 1;
                        kernelXshift = (int)(kernelXsize / 2.0f);
                        shiftXstart = ((currentXYZposition[0] < kernelXshift) ?
                                       -currentXYZposition[0] : -kernelXshift);
                        shiftXstop = ((currentXYZposition[0] >= (dim_array[0] - kernelXshift)) ?
                                      (int)dim_array[0] - currentXYZposition[0] - 1 : kernelXshift);

                        kernelYsize = (int)(sqrtf(gaussY_var) * 6.0f) % 2 != 0 ?
                            (int)(sqrtf(gaussY_var) * 6.0f) : (int)(sqrtf(gaussY_var) * 6.0f) + 1;
                        kernelYshift = (int)(kernelYsize / 2.0f);
                        shiftYstart = ((currentXYZposition[1] < kernelYshift) ?
                                       -currentXYZposition[1] : -kernelYshift);
                        shiftYstop = ((currentXYZposition[1] >= (dim_array[1] - kernelYshift)) ?
                                      (int)dim_array[1] - currentXYZposition[1] - 1 : kernelYshift);

                        kernelZsize = (int)(sqrtf(gaussZ_var) * 6.0f) % 2 != 0 ?
                            (int)(sqrtf(gaussZ_var) * 6.0f) : (int)(sqrtf(gaussZ_var) * 6.0f) + 1;
                        kernelZshift = (int)(kernelZsize / 2.0f);
                        shiftZstart = ((currentXYZposition[2] < kernelZshift) ?
                                       -currentXYZposition[2] : -kernelZshift);
                        shiftZstop = ((currentXYZposition[2] >= (dim_array[2] - kernelZshift)) ?
                                      (int)dim_array[2] - currentXYZposition[2] - 1 : kernelZshift);

                        if (nanImagePtr[index] != 0) {
                            for (shiftx = shiftXstart; shiftx <= shiftXstop; shiftx++) {
                                for (shifty = shiftYstart; shifty <= shiftYstop; shifty++) {
                                    for (shiftz = shiftZstart; shiftz <= shiftZstop; shiftz++) {

                                        // Data Blur
                                        indexNeighbour = index + (shiftx * shiftdirection[0]) +
                                            (shifty * shiftdirection[1]) + (shiftz * shiftdirection[2]);
                                        if (nanImagePtr[indexNeighbour] != 0) {
                                            kernelval = expf((float)(-0.5f * (pow(shiftx, 2) / gaussX_var
                                                                              + pow(shifty, 2) / gaussY_var
                                                                              + pow(shiftz, 2) / gaussZ_var))) /
                                                (sqrtf(2.f * 3.14159265f * pow(gaussX_var * gaussY_var * gaussZ_var, 2.f)));

                                            location = tmp_lab.find(intensityPtr[indexNeighbour]);
                                            if (location != tmp_lab.end()) {
                                                location->second = location->second + kernelval;
                                            } else {
                                                tmp_lab.insert(DataPointPair(intensityPtr[indexNeighbour], kernelval));
                                            }
                                        }
                                    }
                                }
                            }
                            currIterator = tmp_lab.begin();
                            maxindex = 0;
                            maxval = std::numeric_limits<float>::lowest();
                            while (currIterator != tmp_lab.end()) {
                                if (currIterator->second > maxval) {
                                    maxindex = currIterator->first;
                                    maxval = currIterator->second;
                                }
                                currIterator++;
                            }
                            tmpImagePtr[index] = maxindex;
                        } else {
                            tmpImagePtr[index] = std::numeric_limits<DataType>::quiet_NaN();
                        }
                    }
                }
            }
            // Normalise per time point
            for (index = 0; index < voxelNumber; ++index) {
                if (nanImagePtr[index] == 0)
                    intensityPtr[index] = std::numeric_limits<DataType>::quiet_NaN();
                else
                    intensityPtr[index] = tmpImagePtr[index];
            }
        } // check if the time point is active
    } // loop over the time points

    free(tmpImagePtr);
    free(currentMask);
    free(activeTimePoints);
    free(nanImagePtr);
}
/* *************************************************************** */
void reg_tools_labelKernelConvolution(nifti_image *image,
                                      float varianceX,
                                      float varianceY,
                                      float varianceZ,
                                      int *mask,
                                      bool *timePoints) {
    switch (image->datatype) {
    case NIFTI_TYPE_UINT8:
        reg_tools_labelKernelConvolution_core<unsigned char>(image, varianceX, varianceY, varianceZ, mask, timePoints);
        break;
    case NIFTI_TYPE_INT8:
        reg_tools_labelKernelConvolution_core<char>(image, varianceX, varianceY, varianceZ, mask, timePoints);
        break;
    case NIFTI_TYPE_UINT16:
        reg_tools_labelKernelConvolution_core<unsigned short>(image, varianceX, varianceY, varianceZ, mask, timePoints);
        break;
    case NIFTI_TYPE_INT16:
        reg_tools_labelKernelConvolution_core<short>(image, varianceX, varianceY, varianceZ, mask, timePoints);
        break;
    case NIFTI_TYPE_UINT32:
        reg_tools_labelKernelConvolution_core<unsigned>(image, varianceX, varianceY, varianceZ, mask, timePoints);
        break;
    case NIFTI_TYPE_INT32:
        reg_tools_labelKernelConvolution_core<int>(image, varianceX, varianceY, varianceZ, mask, timePoints);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_tools_labelKernelConvolution_core<float>(image, varianceX, varianceY, varianceZ, mask, timePoints);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_tools_labelKernelConvolution_core<double>(image, varianceX, varianceY, varianceZ, mask, timePoints);
        break;
    default:
        NR_FATAL_ERROR("The image data type is not supported");
    }
}
/* *************************************************************** */
void reg_tools_kernelConvolution(nifti_image *image,
                                 const float *sigma,
                                 const ConvKernelType kernelType,
                                 const int *mask,
                                 const bool *timePoints,
                                 const bool *axes) {
    if (image->datatype != NIFTI_TYPE_FLOAT32 && image->datatype != NIFTI_TYPE_FLOAT64)
        NR_FATAL_ERROR("The image is expected to be of floating precision type");

    if (image->nt <= 0) image->nt = image->dim[4] = 1;
    if (image->nu <= 0) image->nu = image->dim[5] = 1;

    bool axesToSmooth[3];
    if (axes == nullptr) {
        // All axes are smoothed by default
        axesToSmooth[0] = axesToSmooth[1] = axesToSmooth[2] = true;
    } else for (int i = 0; i < 3; i++) axesToSmooth[i] = axes[i];

    const int activeTimePointCount = image->nt * image->nu;
    unique_ptr<bool[]> activeTimePoints{ new bool[activeTimePointCount] };
    if (timePoints == nullptr) {
        // All time points are considered as active
        for (int i = 0; i < activeTimePointCount; i++) activeTimePoints[i] = true;
    } else for (int i = 0; i < activeTimePointCount; i++) activeTimePoints[i] = timePoints[i];

    unique_ptr<int[]> currentMask;
    if (!mask) {
        currentMask.reset(new int[NiftiImage::calcVoxelNumber(image, 3)]());
        mask = currentMask.get();
    }

    std::visit([&](auto&& imgDataType) {
        using ImgDataType = std::decay_t<decltype(imgDataType)>;
        reg_tools_kernelConvolution<ImgDataType>(image, sigma, kernelType, mask, activeTimePoints.get(), axesToSmooth);
    }, NiftiImage::getFloatingDataType(image));
}
/* *************************************************************** */
template <class PrecisionType, class ImageType>
void reg_downsampleImage(nifti_image *image, int type, bool *downsampleAxis) {
    if (type == 1) {
        /* the input image is first smooth */
        float *sigma = new float[image->nt];
        for (int i = 0; i < image->nt; ++i) sigma[i] = -0.7355f;
        reg_tools_kernelConvolution(image, sigma, ConvKernelType::Gaussian);
        delete[] sigma;
    }

    /* the values are copied */
    ImageType *oldValues = (ImageType*)malloc(image->nvox * image->nbyper);
    ImageType *imagePtr = static_cast<ImageType*>(image->data);
    memcpy(oldValues, imagePtr, image->nvox * image->nbyper);
    free(image->data);

    // Keep the previous real to voxel qform
    mat44 real2Voxel_qform;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            real2Voxel_qform.m[i][j] = image->qto_ijk.m[i][j];
        }
    }

    // Update the axis dimension
    int oldDim[4];
    for (int i = 1; i < 4; i++) {
        oldDim[i] = image->dim[i];
        if (image->dim[i] > 1 && downsampleAxis[i]) image->dim[i] = Ceil(image->dim[i] / 2.0);
        if (image->pixdim[i] > 0 && downsampleAxis[i]) image->pixdim[i] = image->pixdim[i] * 2.0f;
    }
    image->nx = image->dim[1];
    image->ny = image->dim[2];
    image->nz = image->dim[3];
    image->dx = image->pixdim[1];
    image->dy = image->pixdim[2];
    image->dz = image->pixdim[3];
    if (image->nt < 1 || image->dim[4] < 1) image->nt = image->dim[4] = 1;
    if (image->nu < 1 || image->dim[5] < 1) image->nu = image->dim[5] = 1;
    if (image->nv < 1 || image->dim[6] < 1) image->nv = image->dim[6] = 1;
    if (image->nw < 1 || image->dim[7] < 1) image->nw = image->dim[7] = 1;

    // update the qform matrix
    image->qto_xyz = nifti_quatern_to_mat44(image->quatern_b,
                                            image->quatern_c,
                                            image->quatern_d,
                                            image->qoffset_x,
                                            image->qoffset_y,
                                            image->qoffset_z,
                                            image->dx,
                                            image->dy,
                                            image->dz,
                                            image->qfac);
    image->qto_ijk = nifti_mat44_inverse(image->qto_xyz);

    // update the sform matrix
    if (downsampleAxis[1]) {
        image->sto_xyz.m[0][0] *= 2.f;
        image->sto_xyz.m[1][0] *= 2.f;
        image->sto_xyz.m[2][0] *= 2.f;
    }
    if (downsampleAxis[2]) {
        image->sto_xyz.m[0][1] *= 2.f;
        image->sto_xyz.m[1][1] *= 2.f;
        image->sto_xyz.m[2][1] *= 2.f;
    }
    if (downsampleAxis[3]) {
        image->sto_xyz.m[0][2] *= 2.f;
        image->sto_xyz.m[1][2] *= 2.f;
        image->sto_xyz.m[2][2] *= 2.f;
    }
    float origin_sform[3] = { image->sto_xyz.m[0][3], image->sto_xyz.m[1][3], image->sto_xyz.m[2][3] };
    image->sto_xyz.m[0][3] = origin_sform[0];
    image->sto_xyz.m[1][3] = origin_sform[1];
    image->sto_xyz.m[2][3] = origin_sform[2];
    image->sto_ijk = nifti_mat44_inverse(image->sto_xyz);

    // Reallocate the image
    image->nvox = NiftiImage::calcVoxelNumber(image, 7);
    image->data = calloc(image->nvox, image->nbyper);
    imagePtr = static_cast<ImageType*>(image->data);

    PrecisionType real[3];
    ImageType intensity;
    int position[3];

    // qform is used for resampling
    for (size_t tuvw = 0; tuvw < (size_t)image->nt * image->nu * image->nv * image->nw; tuvw++) {
        ImageType *valuesPtrTUVW = &oldValues[tuvw * oldDim[1] * oldDim[2] * oldDim[3]];
        for (int z = 0; z < image->nz; z++) {
            for (int y = 0; y < image->ny; y++) {
                for (int x = 0; x < image->nx; x++) {
                    // Extract the voxel coordinate in mm
                    real[0] = x * image->qto_xyz.m[0][0] +
                        y * image->qto_xyz.m[0][1] +
                        z * image->qto_xyz.m[0][2] +
                        image->qto_xyz.m[0][3];
                    real[1] = x * image->qto_xyz.m[1][0] +
                        y * image->qto_xyz.m[1][1] +
                        z * image->qto_xyz.m[1][2] +
                        image->qto_xyz.m[1][3];
                    real[2] = x * image->qto_xyz.m[2][0] +
                        y * image->qto_xyz.m[2][1] +
                        z * image->qto_xyz.m[2][2] +
                        image->qto_xyz.m[2][3];
                    // Extract the position in voxel in the old image;
                    position[0] = Round(real[0] * real2Voxel_qform.m[0][0] + real[1] * real2Voxel_qform.m[0][1] + real[2] * real2Voxel_qform.m[0][2] + real2Voxel_qform.m[0][3]);
                    position[1] = Round(real[0] * real2Voxel_qform.m[1][0] + real[1] * real2Voxel_qform.m[1][1] + real[2] * real2Voxel_qform.m[1][2] + real2Voxel_qform.m[1][3]);
                    position[2] = Round(real[0] * real2Voxel_qform.m[2][0] + real[1] * real2Voxel_qform.m[2][1] + real[2] * real2Voxel_qform.m[2][2] + real2Voxel_qform.m[2][3]);
                    if (oldDim[3] == 1) position[2] = 0;
                    // Nearest neighbour is used as downsampling ratio is constant
                    intensity = std::numeric_limits<ImageType>::quiet_NaN();
                    if (-1 < position[0] && position[0] < oldDim[1] &&
                        -1 < position[1] && position[1] < oldDim[2] &&
                        -1 < position[2] && position[2] < oldDim[3]) {
                        intensity = valuesPtrTUVW[(position[2] * oldDim[2] + position[1]) * oldDim[1] + position[0]];
                    }
                    *imagePtr = intensity;
                    imagePtr++;
                }
            }
        }
    }
    free(oldValues);
}
/* *************************************************************** */
template <class PrecisionType>
void reg_downsampleImage(nifti_image *image, int type, bool *downsampleAxis) {
    switch (image->datatype) {
    case NIFTI_TYPE_UINT8:
        reg_downsampleImage<PrecisionType, unsigned char>(image, type, downsampleAxis);
        break;
    case NIFTI_TYPE_INT8:
        reg_downsampleImage<PrecisionType, char>(image, type, downsampleAxis);
        break;
    case NIFTI_TYPE_UINT16:
        reg_downsampleImage<PrecisionType, unsigned short>(image, type, downsampleAxis);
        break;
    case NIFTI_TYPE_INT16:
        reg_downsampleImage<PrecisionType, short>(image, type, downsampleAxis);
        break;
    case NIFTI_TYPE_UINT32:
        reg_downsampleImage<PrecisionType, unsigned>(image, type, downsampleAxis);
        break;
    case NIFTI_TYPE_INT32:
        reg_downsampleImage<PrecisionType, int>(image, type, downsampleAxis);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_downsampleImage<PrecisionType, float>(image, type, downsampleAxis);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_downsampleImage<PrecisionType, double>(image, type, downsampleAxis);
        break;
    default:
        NR_FATAL_ERROR("The image data type is not supported");
    }
}
template void reg_downsampleImage<float>(nifti_image*, int, bool*);
template void reg_downsampleImage<double>(nifti_image*, int, bool*);
/* *************************************************************** */
template <class DataType>
void reg_tools_binarise_image(nifti_image *image) {
    DataType *dataPtr = static_cast<DataType*>(image->data);
    image->scl_inter = 0.f;
    image->scl_slope = 1.f;
    for (size_t i = 0; i < image->nvox; i++)
        dataPtr[i] = dataPtr[i] != 0 ? (DataType)1 : (DataType)0;
}
/* *************************************************************** */
void reg_tools_binarise_image(nifti_image *image) {
    switch (image->datatype) {
    case NIFTI_TYPE_UINT8:
        reg_tools_binarise_image<unsigned char>(image);
        break;
    case NIFTI_TYPE_INT8:
        reg_tools_binarise_image<char>(image);
        break;
    case NIFTI_TYPE_UINT16:
        reg_tools_binarise_image<unsigned short>(image);
        break;
    case NIFTI_TYPE_INT16:
        reg_tools_binarise_image<short>(image);
        break;
    case NIFTI_TYPE_UINT32:
        reg_tools_binarise_image<unsigned>(image);
        break;
    case NIFTI_TYPE_INT32:
        reg_tools_binarise_image<int>(image);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_tools_binarise_image<float>(image);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_tools_binarise_image<double>(image);
        break;
    default:
        NR_FATAL_ERROR("The image data type is not supported");
    }
}
/* *************************************************************** */
template <class DataType>
void reg_tools_binarise_image(nifti_image *image, float threshold) {
    DataType *dataPtr = static_cast<DataType*>(image->data);
    for (size_t i = 0; i < image->nvox; i++)
        dataPtr[i] = dataPtr[i] < threshold ? (DataType)0 : (DataType)1;
}
/* *************************************************************** */
void reg_tools_binarise_image(nifti_image *image, float threshold) {
    switch (image->datatype) {
    case NIFTI_TYPE_UINT8:
        reg_tools_binarise_image<unsigned char>(image, threshold);
        break;
    case NIFTI_TYPE_INT8:
        reg_tools_binarise_image<char>(image, threshold);
        break;
    case NIFTI_TYPE_UINT16:
        reg_tools_binarise_image<unsigned short>(image, threshold);
        break;
    case NIFTI_TYPE_INT16:
        reg_tools_binarise_image<short>(image, threshold);
        break;
    case NIFTI_TYPE_UINT32:
        reg_tools_binarise_image<unsigned>(image, threshold);
        break;
    case NIFTI_TYPE_INT32:
        reg_tools_binarise_image<int>(image, threshold);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_tools_binarise_image<float>(image, threshold);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_tools_binarise_image<double>(image, threshold);
        break;
    default:
        NR_FATAL_ERROR("The image data type is not supported");
    }
}
/* *************************************************************** */
template <class DataType>
void reg_tools_binaryImage2int(const nifti_image *image, int *array) {
    const DataType *dataPtr = static_cast<DataType*>(image->data);
    for (size_t i = 0; i < NiftiImage::calcVoxelNumber(image, 3); i++)
        array[i] = dataPtr[i] != 0 ? 1 : -1;
}
/* *************************************************************** */
void reg_tools_binaryImage2int(const nifti_image *image, int *array) {
    switch (image->datatype) {
    case NIFTI_TYPE_UINT8:
        reg_tools_binaryImage2int<unsigned char>(image, array);
        break;
    case NIFTI_TYPE_INT8:
        reg_tools_binaryImage2int<char>(image, array);
        break;
    case NIFTI_TYPE_UINT16:
        reg_tools_binaryImage2int<unsigned short>(image, array);
        break;
    case NIFTI_TYPE_INT16:
        reg_tools_binaryImage2int<short>(image, array);
        break;
    case NIFTI_TYPE_UINT32:
        reg_tools_binaryImage2int<unsigned>(image, array);
        break;
    case NIFTI_TYPE_INT32:
        reg_tools_binaryImage2int<int>(image, array);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_tools_binaryImage2int<float>(image, array);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_tools_binaryImage2int<double>(image, array);
        break;
    default:
        NR_FATAL_ERROR("The image data type is not supported");
    }
}
/* *************************************************************** */
template <class AType, class BType>
double reg_tools_getMeanRMS(const nifti_image *imageA, const nifti_image *imageB) {
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(imageA, 3);
    const AType *imageAPtrX = static_cast<AType*>(imageA->data);
    const BType *imageBPtrX = static_cast<BType*>(imageB->data);
    const AType *imageAPtrY = nullptr;
    const BType *imageBPtrY = nullptr;
    const AType *imageAPtrZ = nullptr;
    const BType *imageBPtrZ = nullptr;
    if (imageA->dim[5] > 1) {
        imageAPtrY = &imageAPtrX[voxelNumber];
        imageBPtrY = &imageBPtrX[voxelNumber];
    }
    if (imageA->dim[5] > 2) {
        imageAPtrZ = &imageAPtrY[voxelNumber];
        imageBPtrZ = &imageBPtrY[voxelNumber];
    }
    double sum = 0;
    double rms;
    double diff;
    for (size_t i = 0; i < voxelNumber; i++) {
        diff = (double)*imageAPtrX++ - (double)*imageBPtrX++;
        rms = diff * diff;
        if (imageA->dim[5] > 1) {
            diff = (double)*imageAPtrY++ - (double)*imageBPtrY++;
            rms += diff * diff;
        }
        if (imageA->dim[5] > 2) {
            diff = (double)*imageAPtrZ++ - (double)*imageBPtrZ++;
            rms += diff * diff;
        }
        if (rms == rms)
            sum += sqrt(rms);
    }
    return sum / static_cast<double>(voxelNumber);
}
/* *************************************************************** */
template <class AType>
double reg_tools_getMeanRMS(const nifti_image *imageA, const nifti_image *imageB) {
    switch (imageB->datatype) {
    case NIFTI_TYPE_UINT8:
        return reg_tools_getMeanRMS<AType, unsigned char>(imageA, imageB);
    case NIFTI_TYPE_INT8:
        return reg_tools_getMeanRMS<AType, char>(imageA, imageB);
    case NIFTI_TYPE_UINT16:
        return reg_tools_getMeanRMS<AType, unsigned short>(imageA, imageB);
    case NIFTI_TYPE_INT16:
        return reg_tools_getMeanRMS<AType, short>(imageA, imageB);
    case NIFTI_TYPE_UINT32:
        return reg_tools_getMeanRMS<AType, unsigned>(imageA, imageB);
    case NIFTI_TYPE_INT32:
        return reg_tools_getMeanRMS<AType, int>(imageA, imageB);
    case NIFTI_TYPE_FLOAT32:
        return reg_tools_getMeanRMS<AType, float>(imageA, imageB);
    case NIFTI_TYPE_FLOAT64:
        return reg_tools_getMeanRMS<AType, double>(imageA, imageB);
    default:
        NR_FATAL_ERROR("The image data type is not supported");
        return 0;
    }
}
/* *************************************************************** */
double reg_tools_getMeanRMS(const nifti_image *imageA, const nifti_image *imageB) {
    switch (imageA->datatype) {
    case NIFTI_TYPE_UINT8:
        return reg_tools_getMeanRMS<unsigned char>(imageA, imageB);
    case NIFTI_TYPE_INT8:
        return reg_tools_getMeanRMS<char>(imageA, imageB);
    case NIFTI_TYPE_UINT16:
        return reg_tools_getMeanRMS<unsigned short>(imageA, imageB);
    case NIFTI_TYPE_INT16:
        return reg_tools_getMeanRMS<short>(imageA, imageB);
    case NIFTI_TYPE_UINT32:
        return reg_tools_getMeanRMS<unsigned>(imageA, imageB);
    case NIFTI_TYPE_INT32:
        return reg_tools_getMeanRMS<int>(imageA, imageB);
    case NIFTI_TYPE_FLOAT32:
        return reg_tools_getMeanRMS<float>(imageA, imageB);
    case NIFTI_TYPE_FLOAT64:
        return reg_tools_getMeanRMS<double>(imageA, imageB);
    default:
        NR_FATAL_ERROR("The image data type is not supported");
        return 0;
    }
}
/* *************************************************************** */
template <class DataType>
void reg_createImagePyramid(const NiftiImage& inputImage, vector<NiftiImage>& pyramid, unsigned levelNumber, unsigned levelToPerform) {
    // FINEST LEVEL OF REGISTRATION
    pyramid[levelToPerform - 1] = inputImage;
    reg_tools_changeDatatype<DataType>(pyramid[levelToPerform - 1]);
    reg_tools_removeSCLInfo(pyramid[levelToPerform - 1]);

    // Images are downsampled if appropriate
    for (unsigned l = levelToPerform; l < levelNumber; l++) {
        bool downsampleAxis[8] = { false, true, true, true, false, false, false, false };
        if ((pyramid[levelToPerform - 1]->nx / 2) < 32) downsampleAxis[1] = false;
        if ((pyramid[levelToPerform - 1]->ny / 2) < 32) downsampleAxis[2] = false;
        if ((pyramid[levelToPerform - 1]->nz / 2) < 32) downsampleAxis[3] = false;
        reg_downsampleImage<DataType>(pyramid[levelToPerform - 1], 1, downsampleAxis);
    }

    // Images for each subsequent levels are allocated and downsampled if appropriate
    for (int l = levelToPerform - 2; l >= 0; l--) {
        // Allocation of the image
        pyramid[l] = pyramid[l + 1];

        // Downsample the image if appropriate
        bool downsampleAxis[8] = { false, true, true, true, false, false, false, false };
        if ((pyramid[l]->nx / 2) < 32) downsampleAxis[1] = false;
        if ((pyramid[l]->ny / 2) < 32) downsampleAxis[2] = false;
        if ((pyramid[l]->nz / 2) < 32) downsampleAxis[3] = false;
        reg_downsampleImage<DataType>(pyramid[l], 1, downsampleAxis);
    }
}
template void reg_createImagePyramid<float>(const NiftiImage&, vector<NiftiImage>&, unsigned, unsigned);
template void reg_createImagePyramid<double>(const NiftiImage&, vector<NiftiImage>&, unsigned, unsigned);
/* *************************************************************** */
template <class DataType>
void reg_createMaskPyramid(const NiftiImage& inputMaskImage, vector<unique_ptr<int[]>>& maskPyramid, unsigned levelNumber, unsigned levelToPerform) {
    // FINEST LEVEL OF REGISTRATION
    vector<NiftiImage> tempMaskImagePyramid(levelToPerform);
    tempMaskImagePyramid[levelToPerform - 1] = inputMaskImage;
    reg_tools_binarise_image(tempMaskImagePyramid[levelToPerform - 1]);
    reg_tools_changeDatatype<unsigned char>(tempMaskImagePyramid[levelToPerform - 1]);

    // Image is downsampled if appropriate
    for (unsigned l = levelToPerform; l < levelNumber; l++) {
        bool downsampleAxis[8] = { false, true, true, true, false, false, false, false };
        if ((tempMaskImagePyramid[levelToPerform - 1]->nx / 2) < 32) downsampleAxis[1] = false;
        if ((tempMaskImagePyramid[levelToPerform - 1]->ny / 2) < 32) downsampleAxis[2] = false;
        if ((tempMaskImagePyramid[levelToPerform - 1]->nz / 2) < 32) downsampleAxis[3] = false;
        reg_downsampleImage<DataType>(tempMaskImagePyramid[levelToPerform - 1], 0, downsampleAxis);
    }
    size_t voxelNumber = tempMaskImagePyramid[levelToPerform - 1].nVoxelsPerVolume();
    maskPyramid[levelToPerform - 1] = std::make_unique<int[]>(voxelNumber);
    reg_tools_binaryImage2int(tempMaskImagePyramid[levelToPerform - 1], maskPyramid[levelToPerform - 1].get());

    // Images for each subsequent levels are allocated and downsampled if appropriate
    for (int l = (int)levelToPerform - 2; l >= 0; l--) {
        // Allocation of the reference image
        tempMaskImagePyramid[l] = tempMaskImagePyramid[l + 1];

        // Downsample the image if appropriate
        bool downsampleAxis[8] = { false, true, true, true, false, false, false, false };
        if ((tempMaskImagePyramid[l]->nx / 2) < 32) downsampleAxis[1] = false;
        if ((tempMaskImagePyramid[l]->ny / 2) < 32) downsampleAxis[2] = false;
        if ((tempMaskImagePyramid[l]->nz / 2) < 32) downsampleAxis[3] = false;
        reg_downsampleImage<DataType>(tempMaskImagePyramid[l], 0, downsampleAxis);

        voxelNumber = tempMaskImagePyramid[l].nVoxelsPerVolume();
        maskPyramid[l] = std::make_unique<int[]>(voxelNumber);
        reg_tools_binaryImage2int(tempMaskImagePyramid[l], maskPyramid[l].get());
    }
}
template void reg_createMaskPyramid<float>(const NiftiImage&, vector<unique_ptr<int[]>>&, unsigned, unsigned);
template void reg_createMaskPyramid<double>(const NiftiImage&, vector<unique_ptr<int[]>>&, unsigned, unsigned);
/* *************************************************************** */
template <class ImageType, class MaskType>
int reg_tools_nanMask_image(const nifti_image *image, const nifti_image *maskImage, nifti_image *outputImage) {
    const ImageType *imagePtr = static_cast<ImageType*>(image->data);
    const MaskType *maskPtr = static_cast<MaskType*>(maskImage->data);
    ImageType *resPtr = static_cast<ImageType*>(outputImage->data);
    for (size_t i = 0; i < image->nvox; ++i) {
        if (*maskPtr == 0)
            *resPtr = std::numeric_limits<ImageType>::quiet_NaN();
        else *resPtr = *imagePtr;
        maskPtr++;
        imagePtr++;
        resPtr++;
    }
    return EXIT_SUCCESS;
}
/* *************************************************************** */
template <class ImageType>
int reg_tools_nanMask_image(const nifti_image *image, const nifti_image *maskImage, nifti_image *outputImage) {
    switch (maskImage->datatype) {
    case NIFTI_TYPE_UINT8:
        return reg_tools_nanMask_image<ImageType, unsigned char>(image, maskImage, outputImage);
    case NIFTI_TYPE_INT8:
        return reg_tools_nanMask_image<ImageType, char>(image, maskImage, outputImage);
    case NIFTI_TYPE_UINT16:
        return reg_tools_nanMask_image<ImageType, unsigned short>(image, maskImage, outputImage);
    case NIFTI_TYPE_INT16:
        return reg_tools_nanMask_image<ImageType, short>(image, maskImage, outputImage);
    case NIFTI_TYPE_UINT32:
        return reg_tools_nanMask_image<ImageType, unsigned>(image, maskImage, outputImage);
    case NIFTI_TYPE_INT32:
        return reg_tools_nanMask_image<ImageType, int>(image, maskImage, outputImage);
    case NIFTI_TYPE_FLOAT32:
        return reg_tools_nanMask_image<ImageType, float>(image, maskImage, outputImage);
    case NIFTI_TYPE_FLOAT64:
        return reg_tools_nanMask_image<ImageType, double>(image, maskImage, outputImage);
    default:
        NR_FATAL_ERROR("The image data type is not supported");
        return 0;
    }
}
/* *************************************************************** */
int reg_tools_nanMask_image(const nifti_image *image, const nifti_image *maskImage, nifti_image *outputImage) {
    // Check dimension
    if (image->nvox != maskImage->nvox || image->nvox != outputImage->nvox)
        NR_FATAL_ERROR("Input images have different size");
    // Check output data type
    if (image->datatype != outputImage->datatype)
        NR_FATAL_ERROR("Input and output images have different data type");
    switch (image->datatype) {
    case NIFTI_TYPE_UINT8:
        return reg_tools_nanMask_image<unsigned char>(image, maskImage, outputImage);
    case NIFTI_TYPE_INT8:
        return reg_tools_nanMask_image<char>(image, maskImage, outputImage);
    case NIFTI_TYPE_UINT16:
        return reg_tools_nanMask_image<unsigned short>(image, maskImage, outputImage);
    case NIFTI_TYPE_INT16:
        return reg_tools_nanMask_image<short>(image, maskImage, outputImage);
    case NIFTI_TYPE_UINT32:
        return reg_tools_nanMask_image<unsigned>(image, maskImage, outputImage);
    case NIFTI_TYPE_INT32:
        return reg_tools_nanMask_image<int>(image, maskImage, outputImage);
    case NIFTI_TYPE_FLOAT32:
        return reg_tools_nanMask_image<float>(image, maskImage, outputImage);
    case NIFTI_TYPE_FLOAT64:
        return reg_tools_nanMask_image<double>(image, maskImage, outputImage);
    default:
        NR_FATAL_ERROR("The image data type is not supported");
        return 0;
    }
}
/* *************************************************************** */
template <class DataType>
int reg_tools_removeNanFromMask_core(const nifti_image *image, int *mask) {
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(image, 3);
    const DataType *imagePtr = static_cast<DataType*>(image->data);
    for (int t = 0; t < image->nt; ++t) {
        for (size_t i = 0; i < voxelNumber; ++i) {
            DataType value = *imagePtr++;
            if (value != value)
                mask[i] = -1;
        }
    }
    return EXIT_SUCCESS;
}
/* *************************************************************** */
int reg_tools_removeNanFromMask(const nifti_image *image, int *mask) {
    switch (image->datatype) {
    case NIFTI_TYPE_FLOAT32:
        return reg_tools_removeNanFromMask_core<float>(image, mask);
    case NIFTI_TYPE_FLOAT64:
        return reg_tools_removeNanFromMask_core<double>(image, mask);
    default:
        NR_FATAL_ERROR("The image data type is not supported");
        return 0;
    }
}
/* *************************************************************** */
template <class DataType>
DataType reg_tools_getMinMaxValue(const nifti_image *image, int timePoint, bool isMin = true) {
    if (timePoint < -1 || timePoint >= image->nt)
        NR_FATAL_ERROR("The required time point does not exist");

    const DataType *imgPtr = static_cast<DataType*>(image->data);
    DataType retValue = isMin ? std::numeric_limits<DataType>::max() : std::numeric_limits<DataType>::lowest();
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(image, 3);
    const float sclSlope = image->scl_slope == 0 ? 1 : image->scl_slope;

    // The min/max function
    const DataType& (*minMax)(const DataType&, const DataType&);
    if (isMin) minMax = std::min<DataType>;
    else minMax = std::max<DataType>;

    for (int time = 0; time < image->nt; ++time) {
        if (time == timePoint || timePoint == -1) {
            for (int u = 0; u < image->nu; ++u) {
                const DataType *currentVolumePtr = &imgPtr[(u * image->nt + time) * voxelNumber];
                for (size_t i = 0; i < voxelNumber; ++i) {
                    DataType currentVal = (DataType)((float)currentVolumePtr[i] * sclSlope + image->scl_inter);
                    retValue = minMax(currentVal, retValue);
                }
            }
        }
    }
    return retValue;
}
/* *************************************************************** */
float reg_tools_getMinValue(const nifti_image *image, int timePoint) {
    // Check the image data type
    switch (image->datatype) {
    case NIFTI_TYPE_UINT8:
        return reg_tools_getMinMaxValue<unsigned char>(image, timePoint);
    case NIFTI_TYPE_INT8:
        return reg_tools_getMinMaxValue<char>(image, timePoint);
    case NIFTI_TYPE_UINT16:
        return reg_tools_getMinMaxValue<unsigned short>(image, timePoint);
    case NIFTI_TYPE_INT16:
        return reg_tools_getMinMaxValue<short>(image, timePoint);
    case NIFTI_TYPE_UINT32:
        return (float)reg_tools_getMinMaxValue<unsigned>(image, timePoint);
    case NIFTI_TYPE_INT32:
        return (float)reg_tools_getMinMaxValue<int>(image, timePoint);
    case NIFTI_TYPE_FLOAT32:
        return reg_tools_getMinMaxValue<float>(image, timePoint);
    case NIFTI_TYPE_FLOAT64:
        return (float)reg_tools_getMinMaxValue<double>(image, timePoint);
    default:
        NR_FATAL_ERROR("The image data type is not supported");
        return 0;
    }
}
/* *************************************************************** */
float reg_tools_getMaxValue(const nifti_image *image, int timePoint) {
    // Check the image data type
    switch (image->datatype) {
    case NIFTI_TYPE_UINT8:
        return reg_tools_getMinMaxValue<unsigned char>(image, timePoint, false);
    case NIFTI_TYPE_INT8:
        return reg_tools_getMinMaxValue<char>(image, timePoint, false);
    case NIFTI_TYPE_UINT16:
        return reg_tools_getMinMaxValue<unsigned short>(image, timePoint, false);
    case NIFTI_TYPE_INT16:
        return reg_tools_getMinMaxValue<short>(image, timePoint, false);
    case NIFTI_TYPE_UINT32:
        return (float)reg_tools_getMinMaxValue<unsigned>(image, timePoint, false);
    case NIFTI_TYPE_INT32:
        return (float)reg_tools_getMinMaxValue<int>(image, timePoint, false);
    case NIFTI_TYPE_FLOAT32:
        return reg_tools_getMinMaxValue<float>(image, timePoint, false);
    case NIFTI_TYPE_FLOAT64:
        return (float)reg_tools_getMinMaxValue<double>(image, timePoint, false);
    default:
        NR_FATAL_ERROR("The image data type is not supported");
        return 0;
    }
}
/* *************************************************************** */
template <class DataType>
float reg_tools_getMeanValue(const nifti_image *image) {
    const DataType *imgPtr = static_cast<DataType*>(image->data);
    float meanValue = 0;
    const float sclSlope = image->scl_slope == 0 ? 1 : image->scl_slope;
    for (size_t i = 0; i < image->nvox; ++i) {
        const float currentVal = static_cast<float>(imgPtr[i]) * sclSlope + image->scl_inter;
        meanValue += currentVal;
    }
    meanValue = float(meanValue / image->nvox);
    return meanValue;
}
/* *************************************************************** */
float reg_tools_getMeanValue(const nifti_image *image) {
    // Check the image data type
    switch (image->datatype) {
    case NIFTI_TYPE_UINT8:
        return reg_tools_getMeanValue<unsigned char>(image);
    case NIFTI_TYPE_INT8:
        return reg_tools_getMeanValue<char>(image);
    case NIFTI_TYPE_UINT16:
        return reg_tools_getMeanValue<unsigned short>(image);
    case NIFTI_TYPE_INT16:
        return reg_tools_getMeanValue<short>(image);
    case NIFTI_TYPE_UINT32:
        return reg_tools_getMeanValue<unsigned>(image);
    case NIFTI_TYPE_INT32:
        return reg_tools_getMeanValue<int>(image);
    case NIFTI_TYPE_FLOAT32:
        return reg_tools_getMeanValue<float>(image);
    case NIFTI_TYPE_FLOAT64:
        return reg_tools_getMeanValue<double>(image);
    default:
        NR_FATAL_ERROR("The image data type is not supported");
        return 0;
    }
}
/* *************************************************************** */
template <class DataType>
float reg_tools_getSTDValue(const nifti_image *image) {
    const DataType *imgPtr = static_cast<DataType*>(image->data);
    const float meanValue = reg_tools_getMeanValue(image);
    float stdValue = 0;
    const float sclSlope = image->scl_slope == 0 ? 1 : image->scl_slope;
    for (size_t i = 0; i < image->nvox; ++i) {
        const float currentVal = static_cast<float>(imgPtr[i]) * sclSlope + image->scl_inter;
        stdValue += (currentVal - meanValue) * (currentVal - meanValue);
    }
    stdValue = std::sqrt(stdValue / image->nvox);
    return stdValue;
}
/* *************************************************************** */
float reg_tools_getSTDValue(const nifti_image *image) {
    // Check the image data type
    switch (image->datatype) {
    case NIFTI_TYPE_UINT8:
        return reg_tools_getSTDValue<unsigned char>(image);
    case NIFTI_TYPE_INT8:
        return reg_tools_getSTDValue<char>(image);
    case NIFTI_TYPE_UINT16:
        return reg_tools_getSTDValue<unsigned short>(image);
    case NIFTI_TYPE_INT16:
        return reg_tools_getSTDValue<short>(image);
    case NIFTI_TYPE_UINT32:
        return reg_tools_getSTDValue<unsigned>(image);
    case NIFTI_TYPE_INT32:
        return reg_tools_getSTDValue<int>(image);
    case NIFTI_TYPE_FLOAT32:
        return reg_tools_getSTDValue<float>(image);
    case NIFTI_TYPE_FLOAT64:
        return reg_tools_getSTDValue<double>(image);
    default:
        NR_FATAL_ERROR("The image data type is not supported");
        return 0;
    }
}
/* *************************************************************** */
template <class DataType>
void reg_flipAxis(const nifti_image *image, void **outputArray, const std::string& cmd) {
    // Allocate the outputArray if it is not allocated yet
    if (*outputArray == nullptr)
        *outputArray = malloc(NiftiImage::calcVoxelNumber(image, 7) * sizeof(DataType));

    // Parse the cmd to check which axis have to be flipped
    const char *axisName = "x\0y\0z\0t\0u\0v\0w\0";
    int increment[7] = { 1, 1, 1, 1, 1, 1, 1 };
    int start[7] = { 0, 0, 0, 0, 0, 0, 0 };
    const int end[7] = { image->nx, image->ny, image->nz, image->nt, image->nu, image->nv, image->nw };
    for (int i = 0; i < 7; ++i) {
        if (cmd.find(axisName[i * 2]) != std::string::npos) {
            increment[i] = -1;
            start[i] = end[i] - 1;
        }
    }

    // Define the reading and writing pointers
    const DataType *inputPtr = static_cast<const DataType*>(image->data);
    DataType *outputPtr = static_cast<DataType*>(*outputArray);

    // Copy the data and flip axis if required
    for (int w = 0, w2 = start[6]; w < image->nw; ++w, w2 += increment[6]) {
        size_t index_w = w2 * image->nx * image->ny * image->nz * image->nt * image->nu * image->nv;
        for (int v = 0, v2 = start[5]; v < image->nv; ++v, v2 += increment[5]) {
            size_t index_v = index_w + v2 * image->nx * image->ny * image->nz * image->nt * image->nu;
            for (int u = 0, u2 = start[4]; u < image->nu; ++u, u2 += increment[4]) {
                size_t index_u = index_v + u2 * image->nx * image->ny * image->nz * image->nt;
                for (int t = 0, t2 = start[3]; t < image->nt; ++t, t2 += increment[3]) {
                    size_t index_t = index_u + t2 * image->nx * image->ny * image->nz;
                    for (int z = 0, z2 = start[2]; z < image->nz; ++z, z2 += increment[2]) {
                        size_t index_z = index_t + z2 * image->nx * image->ny;
                        for (int y = 0, y2 = start[1]; y < image->ny; ++y, y2 += increment[1]) {
                            size_t index_y = index_z + y2 * image->nx;
                            for (int x = 0, x2 = start[0]; x < image->nx; ++x, x2 += increment[0]) {
                                size_t index = index_y + x2;
                                *outputPtr++ = inputPtr[index];
                            }
                        }
                    }
                }
            }
        }
    }
}
/* *************************************************************** */
void reg_flipAxis(const nifti_image *image, void **outputArray, const std::string& cmd) {
    // Check the image data type
    switch (image->datatype) {
    case NIFTI_TYPE_UINT8:
        reg_flipAxis<unsigned char>(image, outputArray, cmd);
        break;
    case NIFTI_TYPE_INT8:
        reg_flipAxis<char>(image, outputArray, cmd);
        break;
    case NIFTI_TYPE_UINT16:
        reg_flipAxis<unsigned short>(image, outputArray, cmd);
        break;
    case NIFTI_TYPE_INT16:
        reg_flipAxis<short>(image, outputArray, cmd);
        break;
    case NIFTI_TYPE_UINT32:
        reg_flipAxis<unsigned>(image, outputArray, cmd);
        break;
    case NIFTI_TYPE_INT32:
        reg_flipAxis<int>(image, outputArray, cmd);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_flipAxis<float>(image, outputArray, cmd);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_flipAxis<double>(image, outputArray, cmd);
        break;
    default:
        NR_FATAL_ERROR("The image data type is not supported");
    }
}
/* *************************************************************** */
template<class DataType>
void reg_getDisplacementFromDeformation_2D(nifti_image *field) {
    DataType *ptrX = static_cast<DataType*>(field->data);
    DataType *ptrY = &ptrX[NiftiImage::calcVoxelNumber(field, 2)];

    mat44 matrix;
    if (field->sform_code > 0)
        matrix = field->sto_xyz;
    else matrix = field->qto_xyz;

    int x, y, index;
    DataType xInit, yInit;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(field, matrix, ptrX, ptrY) \
   private(x, index, xInit, yInit)
#endif
    for (y = 0; y < field->ny; y++) {
        index = y * field->nx;
        for (x = 0; x < field->nx; x++) {
            // Get the initial control point position
            xInit = matrix.m[0][0] * (DataType)x
                + matrix.m[0][1] * (DataType)y
                + matrix.m[0][3];
            yInit = matrix.m[1][0] * (DataType)x
                + matrix.m[1][1] * (DataType)y
                + matrix.m[1][3];

            // The initial position is subtracted from every values
            ptrX[index] -= xInit;
            ptrY[index] -= yInit;
            index++;
        }
    }
}
/* *************************************************************** */
template<class DataType>
void reg_getDisplacementFromDeformation_3D(nifti_image *field) {
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(field, 3);
    DataType *ptrX = static_cast<DataType*>(field->data);
    DataType *ptrY = &ptrX[voxelNumber];
    DataType *ptrZ = &ptrY[voxelNumber];

    mat44 matrix;
    if (field->sform_code > 0)
        matrix = field->sto_xyz;
    else matrix = field->qto_xyz;

    int x, y, z, index;
    float xInit, yInit, zInit;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(field, matrix, ptrX, ptrY, ptrZ) \
   private(x, y, index, xInit, yInit, zInit)
#endif
    for (z = 0; z < field->nz; z++) {
        index = z * field->nx * field->ny;
        for (y = 0; y < field->ny; y++) {
            for (x = 0; x < field->nx; x++) {
                // Get the initial control point position
                xInit = matrix.m[0][0] * static_cast<float>(x)
                    + matrix.m[0][1] * static_cast<float>(y)
                    + matrix.m[0][2] * static_cast<float>(z)
                    + matrix.m[0][3];
                yInit = matrix.m[1][0] * static_cast<float>(x)
                    + matrix.m[1][1] * static_cast<float>(y)
                    + matrix.m[1][2] * static_cast<float>(z)
                    + matrix.m[1][3];
                zInit = matrix.m[2][0] * static_cast<float>(x)
                    + matrix.m[2][1] * static_cast<float>(y)
                    + matrix.m[2][2] * static_cast<float>(z)
                    + matrix.m[2][3];

                // The initial position is subtracted from every values
                ptrX[index] -= static_cast<DataType>(xInit);
                ptrY[index] -= static_cast<DataType>(yInit);
                ptrZ[index] -= static_cast<DataType>(zInit);
                index++;
            }
        }
    }
}
/* *************************************************************** */
int reg_getDisplacementFromDeformation(nifti_image *field) {
    if (field->datatype == NIFTI_TYPE_FLOAT32) {
        switch (field->nu) {
        case 2:
            reg_getDisplacementFromDeformation_2D<float>(field);
            break;
        case 3:
            reg_getDisplacementFromDeformation_3D<float>(field);
            break;
        default:
            NR_FATAL_ERROR("Only implemented for 5D image with 2 or 3 components in the fifth dimension");
        }
    } else if (field->datatype == NIFTI_TYPE_FLOAT64) {
        switch (field->nu) {
        case 2:
            reg_getDisplacementFromDeformation_2D<double>(field);
            break;
        case 3:
            reg_getDisplacementFromDeformation_3D<double>(field);
            break;
        default:
            NR_FATAL_ERROR("Only implemented for 5D image with 2 or 3 components in the fifth dimension");
        }
    } else {
        NR_FATAL_ERROR("Only single or double floating precision have been implemented");
    }
    field->intent_code = NIFTI_INTENT_VECTOR;
    memset(field->intent_name, 0, 16);
    strcpy(field->intent_name, "NREG_TRANS");
    if (field->intent_p1 == DEF_FIELD)
        field->intent_p1 = DISP_FIELD;
    if (field->intent_p1 == DEF_VEL_FIELD)
        field->intent_p1 = DISP_VEL_FIELD;
    return EXIT_SUCCESS;
}
/* *************************************************************** */
template<class DataType>
void reg_getDeformationFromDisplacement_2D(nifti_image *field) {
    DataType *ptrX = static_cast<DataType*>(field->data);
    DataType *ptrY = &ptrX[NiftiImage::calcVoxelNumber(field, 2)];

    mat44 matrix;
    if (field->sform_code > 0)
        matrix = field->sto_xyz;
    else matrix = field->qto_xyz;

    int x, y, index;
    DataType xInit, yInit;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(field, matrix, ptrX, ptrY) \
   private(x, index, xInit, yInit)
#endif
    for (y = 0; y < field->ny; y++) {
        index = y * field->nx;
        for (x = 0; x < field->nx; x++) {
            // Get the initial control point position
            xInit = matrix.m[0][0] * (DataType)x
                + matrix.m[0][1] * (DataType)y
                + matrix.m[0][3];
            yInit = matrix.m[1][0] * (DataType)x
                + matrix.m[1][1] * (DataType)y
                + matrix.m[1][3];

            // The initial position is added from every values
            ptrX[index] += xInit;
            ptrY[index] += yInit;
            index++;
        }
    }
}
/* *************************************************************** */
template<class DataType>
void reg_getDeformationFromDisplacement_3D(nifti_image *field) {
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(field, 3);
    DataType *ptrX = static_cast<DataType*>(field->data);
    DataType *ptrY = &ptrX[voxelNumber];
    DataType *ptrZ = &ptrY[voxelNumber];

    mat44 matrix;
    if (field->sform_code > 0)
        matrix = field->sto_xyz;
    else matrix = field->qto_xyz;

    int x, y, z, index;
    float xInit, yInit, zInit;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(field, matrix, ptrX, ptrY, ptrZ) \
   private(x, y, index, xInit, yInit, zInit)
#endif
    for (z = 0; z < field->nz; z++) {
        index = z * field->nx * field->ny;
        for (y = 0; y < field->ny; y++) {
            for (x = 0; x < field->nx; x++) {
                // Get the initial control point position
                xInit = matrix.m[0][0] * static_cast<float>(x)
                    + matrix.m[0][1] * static_cast<float>(y)
                    + matrix.m[0][2] * static_cast<float>(z)
                    + matrix.m[0][3];
                yInit = matrix.m[1][0] * static_cast<float>(x)
                    + matrix.m[1][1] * static_cast<float>(y)
                    + matrix.m[1][2] * static_cast<float>(z)
                    + matrix.m[1][3];
                zInit = matrix.m[2][0] * static_cast<float>(x)
                    + matrix.m[2][1] * static_cast<float>(y)
                    + matrix.m[2][2] * static_cast<float>(z)
                    + matrix.m[2][3];

                // The initial position is subtracted from every values
                ptrX[index] += static_cast<DataType>(xInit);
                ptrY[index] += static_cast<DataType>(yInit);
                ptrZ[index] += static_cast<DataType>(zInit);
                index++;
            }
        }
    }
}
/* *************************************************************** */
int reg_getDeformationFromDisplacement(nifti_image *field) {
    if (field->datatype == NIFTI_TYPE_FLOAT32) {
        switch (field->nu) {
        case 2:
            reg_getDeformationFromDisplacement_2D<float>(field);
            break;
        case 3:
            reg_getDeformationFromDisplacement_3D<float>(field);
            break;
        default:
            NR_FATAL_ERROR("Only implemented for 2D or 3D deformation fields");
        }
    } else if (field->datatype == NIFTI_TYPE_FLOAT64) {
        switch (field->nu) {
        case 2:
            reg_getDeformationFromDisplacement_2D<double>(field);
            break;
        case 3:
            reg_getDeformationFromDisplacement_3D<double>(field);
            break;
        default:
            NR_FATAL_ERROR("Only implemented for 2D or 3D deformation fields");
        }
    } else {
        NR_FATAL_ERROR("Only single or double floating precision have been implemented");
    }

    field->intent_code = NIFTI_INTENT_VECTOR;
    memset(field->intent_name, 0, 16);
    strcpy(field->intent_name, "NREG_TRANS");
    if (field->intent_p1 == DISP_FIELD)
        field->intent_p1 = DEF_FIELD;
    if (field->intent_p1 == DISP_VEL_FIELD)
        field->intent_p1 = DEF_VEL_FIELD;
    return EXIT_SUCCESS;
}
/* *************************************************************** */
template <class DataType>
void reg_setGradientToZero_core(nifti_image *image,
                                bool xAxis,
                                bool yAxis,
                                bool zAxis) {
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(image, 3);
    DataType *ptr = static_cast<DataType*>(image->data);
    if (xAxis) {
        for (size_t i = 0; i < voxelNumber; ++i)
            *ptr++ = 0;
    } else ptr += voxelNumber;
    if (yAxis) {
        for (size_t i = 0; i < voxelNumber; ++i)
            *ptr++ = 0;
    } else ptr += voxelNumber;
    if (zAxis && image->nu > 2) {
        for (size_t i = 0; i < voxelNumber; ++i)
            *ptr++ = 0;
    }
}
/* *************************************************************** */
void reg_setGradientToZero(nifti_image *image,
                           bool xAxis,
                           bool yAxis,
                           bool zAxis = false) {
    // Ensure that the specified image is a 5D image
    if (image->ndim != 5)
        NR_FATAL_ERROR("Input image is expected to be a 5D image");
    switch (image->datatype) {
    case NIFTI_TYPE_FLOAT32:
        reg_setGradientToZero_core<float>(image, xAxis, yAxis, zAxis);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_setGradientToZero_core<double>(image, xAxis, yAxis, zAxis);
        break;
    default:
        NR_FATAL_ERROR("Input image is expected to be float or double");
    }
}
/* *************************************************************** */
template <class DataType>
double reg_test_compare_arrays(const DataType *ptrA,
                               const DataType *ptrB,
                               size_t nvox) {
    double maxDifference = 0;

    for (size_t i = 0; i < nvox; ++i) {
        const double valA = (double)ptrA[i];
        const double valB = (double)ptrB[i];
        if (valA != valA || valB != valB) {
            if (valA == valA || valB == valB) {
                NR_WARN_WFCT("Unexpected NaN in only one of the array");
                return std::numeric_limits<float>::max();
            }
        } else {
            if (valA != 0 && valB != 0) {
                double diffRatio = valA / valB;
                if (diffRatio < 0) {
                    diffRatio = std::abs(valA - valB);
                    maxDifference = maxDifference > diffRatio ? maxDifference : diffRatio;
                }
                diffRatio -= 1.0;
                maxDifference = maxDifference > diffRatio ? maxDifference : diffRatio;
            } else {
                double diffRatio = std::abs(valA - valB);
                maxDifference = maxDifference > diffRatio ? maxDifference : diffRatio;
            }
        }
    }
    return maxDifference;
}
template double reg_test_compare_arrays<float>(const float*, const float*, size_t);
template double reg_test_compare_arrays<double>(const double*, const double*, size_t);
/* *************************************************************** */
template <class DataType>
double reg_test_compare_images(const nifti_image *imgA, const nifti_image *imgB) {
    const DataType *imgAPtr = static_cast<DataType*>(imgA->data);
    const DataType *imgBPtr = static_cast<DataType*>(imgB->data);
    return reg_test_compare_arrays<DataType>(imgAPtr, imgBPtr, imgA->nvox);
}
/* *************************************************************** */
double reg_test_compare_images(const nifti_image *imgA, const nifti_image *imgB) {
    if (imgA->datatype != imgB->datatype)
        NR_FATAL_ERROR("Input images have different datatype");
    if (imgA->nvox != imgB->nvox)
        NR_FATAL_ERROR("Input images have different size");
    switch (imgA->datatype) {
    case NIFTI_TYPE_UINT8:
        return reg_test_compare_images<unsigned char>(imgA, imgB);
    case NIFTI_TYPE_UINT16:
        return reg_test_compare_images<unsigned short>(imgA, imgB);
    case NIFTI_TYPE_UINT32:
        return reg_test_compare_images<unsigned>(imgA, imgB);
    case NIFTI_TYPE_INT8:
        return reg_test_compare_images<char>(imgA, imgB);
    case NIFTI_TYPE_INT16:
        return reg_test_compare_images<short>(imgA, imgB);
    case NIFTI_TYPE_INT32:
        return reg_test_compare_images<int>(imgA, imgB);
    case NIFTI_TYPE_FLOAT32:
        return reg_test_compare_images<float>(imgA, imgB);
    case NIFTI_TYPE_FLOAT64:
        return reg_test_compare_images<double>(imgA, imgB);
    default:
        NR_FATAL_ERROR("Unsupported data type");
        return 0;
    }
}
/* *************************************************************** */
template <class DataType>
void reg_tools_abs_image(nifti_image *img) {
    DataType *ptr = static_cast<DataType*>(img->data);
    for (size_t i = 0; i < img->nvox; ++i)
        ptr[i] = static_cast<DataType>(fabs(static_cast<double>(ptr[i])));
}
/* *************************************************************** */
void reg_tools_abs_image(nifti_image *img) {
    switch (img->datatype) {
    case NIFTI_TYPE_UINT8:
        reg_tools_abs_image<unsigned char>(img);
        break;
    case NIFTI_TYPE_UINT16:
        reg_tools_abs_image<unsigned short>(img);
        break;
    case NIFTI_TYPE_UINT32:
        reg_tools_abs_image<unsigned>(img);
        break;
    case NIFTI_TYPE_INT8:
        reg_tools_abs_image<char>(img);
        break;
    case NIFTI_TYPE_INT16:
        reg_tools_abs_image<short>(img);
        break;
    case NIFTI_TYPE_INT32:
        reg_tools_abs_image<int>(img);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_tools_abs_image<float>(img);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_tools_abs_image<double>(img);
        break;
    default:
        NR_FATAL_ERROR("Unsupported data type");
    }
}
/* *************************************************************** */
void mat44ToCptr(const mat44& mat, float *cMat) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            cMat[i * 4 + j] = mat.m[i][j];
        }
    }
}
/* *************************************************************** */
void cPtrToMat44(mat44 *mat, const float *cMat) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            mat->m[i][j] = cMat[i * 4 + j];
        }
    }
}
/* *************************************************************** */
void mat33ToCptr(const mat33 *mat, float *cMat, const unsigned numMats) {
    for (size_t k = 0; k < numMats; k++) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                cMat[9 * k + i * 3 + j] = mat[k].m[i][j];
            }
        }
    }
}
/* *************************************************************** */
void cPtrToMat33(mat33 *mat, const float *cMat) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            mat->m[i][j] = cMat[i * 3 + j];
        }
    }
}
/* *************************************************************** */
template<typename T>
void matmnToCptr(const T **mat, T *cMat, unsigned m, unsigned n) {
    for (unsigned i = 0; i < m; i++) {
        for (unsigned j = 0; j < n; j++) {
            cMat[i * n + j] = mat[i][j];
        }
    }
}
template void matmnToCptr<float>(const float**, float*, unsigned, unsigned);
template void matmnToCptr<double>(const double**, double*, unsigned, unsigned);
/* *************************************************************** */
template<typename T>
void cPtrToMatmn(T **mat, const T *cMat, unsigned m, unsigned n) {
    for (unsigned i = 0; i < m; i++) {
        for (unsigned j = 0; j < n; j++) {
            mat[i][j] = cMat[i * n + j];
        }
    }
}
template void cPtrToMatmn<float>(float**, const float*, unsigned, unsigned);
template void cPtrToMatmn<double>(double**, const double*, unsigned, unsigned);
/* *************************************************************** */
void coordinateFromLinearIndex(int index, int maxValue_x, int maxValue_y, int& x, int& y, int& z) {
    x = index % (maxValue_x + 1);
    index /= (maxValue_x + 1);
    y = index % (maxValue_y + 1);
    index /= (maxValue_y + 1);
    z = index;
}
/* *************************************************************** */
nifti_image* nifti_dup(const nifti_image& image, const bool copyData) {
    nifti_image *newImage = nifti_copy_nim_info(&image);
    newImage->data = calloc(image.nvox, image.nbyper);
    if (copyData)
        memcpy(newImage->data, image.data, image.nvox * image.nbyper);
    return newImage;
}
/* *************************************************************** */
void PrintCmdLine(const int argc, const char *const *argv, const bool verbose) {
    // Print the version
    NR_INFO(argv[0] << " v" << NR_VERSION);
    NR_INFO("");
#ifdef NDEBUG
    if (!verbose) return;
#endif
    NR_INFO("Command line:");
    std::string text("\t");
    for (int i = 0; i < argc; i++)
        text += " "s + argv[i];
    NR_INFO(text);
    NR_INFO("");
}
/* *************************************************************** */
