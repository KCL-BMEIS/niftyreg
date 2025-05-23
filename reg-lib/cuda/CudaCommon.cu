/**
 * @file CudaCommon.cu
 * @author Marc Modat
 * @date 25/03/2009
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "CudaCommon.hpp"

/* *************************************************************** */
namespace NiftyReg::Cuda {
/* *************************************************************** */
template<class DataType>
void Allocate(DataType **arrayCuda, const size_t nVoxels) {
    NR_CUDA_SAFE_CALL(cudaMalloc(arrayCuda, nVoxels * sizeof(DataType)));
}
template void Allocate<int>(int**, const size_t);
template void Allocate<float>(float**, const size_t);
template void Allocate<double>(double**, const size_t);
template void Allocate<float4>(float4**, const size_t);
/* *************************************************************** */
template<class DataType>
void Allocate(DataType **arrayCuda, const int *dim) {
    const size_t memSize = (size_t)std::abs(dim[1]) * (size_t)std::abs(dim[2]) * (size_t)std::abs(dim[3]) * sizeof(DataType);
    NR_CUDA_SAFE_CALL(cudaMalloc(arrayCuda, memSize));
}
template void Allocate<int>(int**, const int*);
template void Allocate<float>(float**, const int*);
template void Allocate<double>(double**, const int*);
template void Allocate<float4>(float4**, const int*);
/* *************************************************************** */
template<class DataType>
void Allocate(DataType **array1Cuda, DataType **array2Cuda, const int *dim) {
    const size_t memSize = (size_t)std::abs(dim[1]) * (size_t)std::abs(dim[2]) * (size_t)std::abs(dim[3]) * sizeof(DataType);
    NR_CUDA_SAFE_CALL(cudaMalloc(array1Cuda, memSize));
    NR_CUDA_SAFE_CALL(cudaMalloc(array2Cuda, memSize));
}
template void Allocate<float>(float**, float**, const int*);
template void Allocate<double>(double**, double**, const int*);
template void Allocate<float4>(float4**, float4**, const int*);
/* *************************************************************** */
template<class DataType, class NiftiType>
void TransferNiftiToDevice(DataType *arrayCuda, const nifti_image *img) {
    if (sizeof(DataType) != sizeof(NiftiType))
        NR_FATAL_ERROR("The host and device arrays are of different types");
    NR_CUDA_SAFE_CALL(cudaMemcpy(arrayCuda, img->data, img->nvox * sizeof(NiftiType), cudaMemcpyHostToDevice));
}
/* *************************************************************** */
template<class DataType>
void TransferNiftiToDevice(DataType *arrayCuda, const nifti_image *img) {
    if (sizeof(DataType) == sizeof(float4)) {
        if (img->datatype != NIFTI_TYPE_FLOAT32)
            NR_FATAL_ERROR("The specified image is not a single precision image");
        const float *niftiImgValues = static_cast<float*>(img->data);
        const size_t voxelNumber = NiftiImage::calcVoxelNumber(img, 3);
        const auto timePointCount = img->dim[4] * img->dim[5];
        unique_ptr<float4[]> array(new float4[voxelNumber]());
        for (size_t i = 0; i < voxelNumber; i++)
            array[i].x = *niftiImgValues++;
        if (timePointCount >= 2) {
            for (size_t i = 0; i < voxelNumber; i++)
                array[i].y = *niftiImgValues++;
        }
        if (timePointCount >= 3) {
            for (size_t i = 0; i < voxelNumber; i++)
                array[i].z = *niftiImgValues++;
        }
        if (timePointCount >= 4) {
            for (size_t i = 0; i < voxelNumber; i++)
                array[i].w = *niftiImgValues++;
        }
        NR_CUDA_SAFE_CALL(cudaMemcpy(arrayCuda, array.get(), voxelNumber * sizeof(float4), cudaMemcpyHostToDevice));
    } else {
        switch (img->datatype) {
        case NIFTI_TYPE_FLOAT32:
            TransferNiftiToDevice<DataType, float>(arrayCuda, img);
            break;
        default:
            NR_FATAL_ERROR("The image data type is not supported");
        }
    }
}
template void TransferNiftiToDevice<int>(int*, const nifti_image*);
template void TransferNiftiToDevice<float>(float*, const nifti_image*);
template void TransferNiftiToDevice<double>(double*, const nifti_image*);
template void TransferNiftiToDevice<float4>(float4*, const nifti_image*);
/* *************************************************************** */
template<class DataType, class NiftiType>
void TransferNiftiToDevice(DataType *array1Cuda, DataType *array2Cuda, const nifti_image *img) {
    if (sizeof(DataType) != sizeof(NiftiType))
        NR_FATAL_ERROR("The host and device arrays are of different types");
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(img, 3);
    const size_t memSize = voxelNumber * sizeof(DataType);
    const NiftiType *array1 = static_cast<NiftiType*>(img->data);
    const NiftiType *array2 = &array1[voxelNumber];
    NR_CUDA_SAFE_CALL(cudaMemcpy(array1Cuda, array1, memSize, cudaMemcpyHostToDevice));
    NR_CUDA_SAFE_CALL(cudaMemcpy(array2Cuda, array2, memSize, cudaMemcpyHostToDevice));
}
/* *************************************************************** */
template<class DataType>
void TransferNiftiToDevice(DataType *array1Cuda, DataType *array2Cuda, const nifti_image *img) {
    if (sizeof(DataType) == sizeof(float4)) {
        if (img->datatype != NIFTI_TYPE_FLOAT32)
            NR_FATAL_ERROR("The specified image is not a single precision image");
        const float *niftiImgValues = static_cast<float*>(img->data);
        const size_t voxelNumber = NiftiImage::calcVoxelNumber(img, 3);
        const auto timePointCount = img->dim[4] * img->dim[5];
        unique_ptr<float4[]> array1(new float4[voxelNumber]());
        unique_ptr<float4[]> array2(new float4[voxelNumber]());
        for (size_t i = 0; i < voxelNumber; i++)
            array1[i].x = *niftiImgValues++;
        for (size_t i = 0; i < voxelNumber; i++)
            array2[i].x = *niftiImgValues++;
        if (timePointCount >= 2) {
            for (size_t i = 0; i < voxelNumber; i++)
                array1[i].y = *niftiImgValues++;
            for (size_t i = 0; i < voxelNumber; i++)
                array2[i].y = *niftiImgValues++;
        }
        if (timePointCount >= 3) {
            for (size_t i = 0; i < voxelNumber; i++)
                array1[i].z = *niftiImgValues++;
            for (size_t i = 0; i < voxelNumber; i++)
                array2[i].z = *niftiImgValues++;
        }
        if (timePointCount >= 4) {
            for (size_t i = 0; i < voxelNumber; i++)
                array1[i].w = *niftiImgValues++;
            for (size_t i = 0; i < voxelNumber; i++)
                array2[i].w = *niftiImgValues++;
        }
        NR_CUDA_SAFE_CALL(cudaMemcpy(array1Cuda, array1.get(), voxelNumber * sizeof(float4), cudaMemcpyHostToDevice));
        NR_CUDA_SAFE_CALL(cudaMemcpy(array2Cuda, array2.get(), voxelNumber * sizeof(float4), cudaMemcpyHostToDevice));
    } else {
        switch (img->datatype) {
        case NIFTI_TYPE_FLOAT32:
            TransferNiftiToDevice<DataType, float>(array1Cuda, array2Cuda, img);
            break;
        default:
            NR_FATAL_ERROR("The image data type is not supported");
        }
    }
}
template void TransferNiftiToDevice<float>(float*, float*, const nifti_image*);
template void TransferNiftiToDevice<double>(double*, double*, const nifti_image*);
template void TransferNiftiToDevice<float4>(float4*, float4*, const nifti_image*);
/* *************************************************************** */
template<class DataType>
void TransferNiftiToDevice(DataType *arrayCuda, const DataType *img, const size_t nvox) {
    NR_CUDA_SAFE_CALL(cudaMemcpy(arrayCuda, img, nvox * sizeof(DataType), cudaMemcpyHostToDevice));
}
template void TransferNiftiToDevice<int>(int*, const int*, const size_t);
template void TransferNiftiToDevice<float>(float*, const float*, const size_t);
template void TransferNiftiToDevice<double>(double*, const double*, const size_t);
/* *************************************************************** */
template<class DataType, class NiftiType>
void TransferFromDeviceToNifti(nifti_image *img, const DataType *arrayCuda) {
    if (sizeof(DataType) != sizeof(NiftiType))
        NR_FATAL_ERROR("The host and device arrays are of different types");
    NR_CUDA_SAFE_CALL(cudaMemcpy(img->data, arrayCuda, img->nvox * sizeof(DataType), cudaMemcpyDeviceToHost));
}
/* *************************************************************** */
template<class DataType>
void TransferFromDeviceToNifti(nifti_image *img, const DataType *arrayCuda) {
    if (sizeof(DataType) == sizeof(float4)) {
        // A nifti 5D volume is expected
        if (img->datatype != NIFTI_TYPE_FLOAT32)
            NR_FATAL_ERROR("The specified image is not a single precision image");
        const size_t voxelNumber = NiftiImage::calcVoxelNumber(img, 3);
        const auto timePointCount = img->dim[4] * img->dim[5];
        thrust::device_ptr<const float4> arrayCudaPtr(reinterpret_cast<const float4*>(arrayCuda));
        const thrust::host_vector<float4> array(arrayCudaPtr, arrayCudaPtr + voxelNumber);
        float *niftiImgValues = static_cast<float*>(img->data);
        for (size_t i = 0; i < voxelNumber; i++)
            *niftiImgValues++ = array[i].x;
        if (timePointCount >= 2) {
            for (size_t i = 0; i < voxelNumber; i++)
                *niftiImgValues++ = array[i].y;
        }
        if (timePointCount >= 3) {
            for (size_t i = 0; i < voxelNumber; i++)
                *niftiImgValues++ = array[i].z;
        }
        if (timePointCount >= 4) {
            for (size_t i = 0; i < voxelNumber; i++)
                *niftiImgValues++ = array[i].w;
        }
    } else {
        switch (img->datatype) {
        case NIFTI_TYPE_FLOAT32:
            TransferFromDeviceToNifti<DataType, float>(img, arrayCuda);
            break;
        default:
            NR_FATAL_ERROR("The image data type is not supported");
        }
    }
}
template void TransferFromDeviceToNifti<float>(nifti_image*, const float*);
template void TransferFromDeviceToNifti<double>(nifti_image*, const double*);
template void TransferFromDeviceToNifti<float4>(nifti_image*, const float4*);
/* *************************************************************** */
template<class DataType, class NiftiType>
void TransferFromDeviceToNifti(nifti_image *img, const DataType *array1Cuda, const DataType *array2Cuda) {
    if (sizeof(DataType) != sizeof(NiftiType))
        NR_FATAL_ERROR("The host and device arrays are of different types");
    const size_t voxelNumber = NiftiImage::calcVoxelNumber(img, 3);
    NiftiType *array1 = static_cast<NiftiType*>(img->data);
    NiftiType *array2 = &array1[voxelNumber];
    NR_CUDA_SAFE_CALL(cudaMemcpy(array1, array1Cuda, voxelNumber * sizeof(DataType), cudaMemcpyDeviceToHost));
    NR_CUDA_SAFE_CALL(cudaMemcpy(array2, array2Cuda, voxelNumber * sizeof(DataType), cudaMemcpyDeviceToHost));
}
/* *************************************************************** */
template<class DataType>
void TransferFromDeviceToNifti(nifti_image *img, const DataType *array1Cuda, const DataType *array2Cuda) {
    if (sizeof(DataType) == sizeof(float4)) {
        // A nifti 5D volume is expected
        if (img->datatype != NIFTI_TYPE_FLOAT32)
            NR_FATAL_ERROR("The specified image is not a single precision image");
        const size_t voxelNumber = NiftiImage::calcVoxelNumber(img, 3);
        const auto timePointCount = img->dim[4] * img->dim[5];
        thrust::device_ptr<const float4> array1CudaPtr(reinterpret_cast<const float4*>(array1Cuda));
        thrust::device_ptr<const float4> array2CudaPtr(reinterpret_cast<const float4*>(array2Cuda));
        const thrust::host_vector<float4> array1(array1CudaPtr, array1CudaPtr + voxelNumber);
        const thrust::host_vector<float4> array2(array2CudaPtr, array2CudaPtr + voxelNumber);
        float *niftiImgValues = static_cast<float*>(img->data);
        for (size_t i = 0; i < voxelNumber; i++)
            *niftiImgValues++ = array1[i].x;
        for (size_t i = 0; i < voxelNumber; i++)
            *niftiImgValues++ = array2[i].x;
        if (timePointCount >= 2) {
            for (size_t i = 0; i < voxelNumber; i++)
                *niftiImgValues++ = array1[i].y;
            for (size_t i = 0; i < voxelNumber; i++)
                *niftiImgValues++ = array2[i].y;
        }
        if (timePointCount >= 3) {
            for (size_t i = 0; i < voxelNumber; i++)
                *niftiImgValues++ = array1[i].z;
            for (size_t i = 0; i < voxelNumber; i++)
                *niftiImgValues++ = array2[i].z;
        }
        if (timePointCount >= 4) {
            for (size_t i = 0; i < voxelNumber; i++)
                *niftiImgValues++ = array1[i].w;
            for (size_t i = 0; i < voxelNumber; i++)
                *niftiImgValues++ = array2[i].w;
        }
    } else {
        switch (img->datatype) {
        case NIFTI_TYPE_FLOAT32:
            TransferFromDeviceToNifti<DataType, float>(img, array1Cuda, array2Cuda);
            break;
        default:
            NR_FATAL_ERROR("The image data type is not supported");
        }
    }
}
template void TransferFromDeviceToNifti<float>(nifti_image*, const float*, const float*);
template void TransferFromDeviceToNifti<double>(nifti_image*, const double*, const double*);
template void TransferFromDeviceToNifti<float4>(nifti_image*, const float4*, const float4*);
/* *************************************************************** */
template<class DataType>
void TransferFromDeviceToHost(DataType *array, const DataType *arrayCuda, const size_t nElements) {
    NR_CUDA_SAFE_CALL(cudaMemcpy(array, arrayCuda, nElements * sizeof(DataType), cudaMemcpyDeviceToHost));
}
template void TransferFromDeviceToHost<float>(float*, const float*, const size_t);
template void TransferFromDeviceToHost<double>(double*, const double*, const size_t);
/* *************************************************************** */
template<class DataType>
void TransferFromHostToDevice(DataType *arrayCuda, const DataType *array, const size_t nElements) {
    NR_CUDA_SAFE_CALL(cudaMemcpy(arrayCuda, array, nElements * sizeof(DataType), cudaMemcpyHostToDevice));
}
template void TransferFromHostToDevice<int>(int*, const int*, const size_t);
template void TransferFromHostToDevice<float>(float*, const float*, const size_t);
template void TransferFromHostToDevice<double>(double*, const double*, const size_t);
/* *************************************************************** */
template<class DataType>
void Free(DataType *arrayCuda) {
    if (arrayCuda != nullptr)
        NR_CUDA_SAFE_CALL(cudaFree(arrayCuda));
}
template void Free<int>(int*);
template void Free<float>(float*);
template void Free<double>(double*);
template void Free<float4>(float4*);
/* *************************************************************** */
template<>
void Free(cudaTextureObject_t *texObj) {
    NR_CUDA_SAFE_CALL(cudaDestroyTextureObject(*texObj));
    delete texObj;
}
/* *************************************************************** */
template<class DataType>
UniqueTextureObjectPtr CreateTextureObject(const DataType *devPtr,
                                           const size_t count,
                                           const cudaChannelFormatKind channelFormat,
                                           const unsigned channelCount) {
    // Specify texture
    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = const_cast<DataType*>(devPtr);
    resDesc.res.linear.desc.f = channelFormat;
    resDesc.res.linear.desc.x = 32;
    if (channelCount > 1)
        resDesc.res.linear.desc.y = 32;
    if (channelCount > 2)
        resDesc.res.linear.desc.z = 32;
    if (channelCount > 3)
        resDesc.res.linear.desc.w = 32;
    resDesc.res.linear.sizeInBytes = count * sizeof(DataType);

    // Specify texture object parameters
    cudaTextureDesc texDesc{};
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.addressMode[2] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;

    // Create texture object
    UniqueTextureObjectPtr texObj(new cudaTextureObject_t());
    NR_CUDA_SAFE_CALL(cudaCreateTextureObject(texObj.get(), &resDesc, &texDesc, nullptr));

    return texObj;
}
template UniqueTextureObjectPtr CreateTextureObject<bool>(const bool*, const size_t, const cudaChannelFormatKind, const unsigned);
template UniqueTextureObjectPtr CreateTextureObject<int>(const int*, const size_t, const cudaChannelFormatKind, const unsigned);
template UniqueTextureObjectPtr CreateTextureObject<float>(const float*, const size_t, const cudaChannelFormatKind, const unsigned);
template UniqueTextureObjectPtr CreateTextureObject<float2>(const float2*, const size_t, const cudaChannelFormatKind, const unsigned);
template UniqueTextureObjectPtr CreateTextureObject<float4>(const float4*, const size_t, const cudaChannelFormatKind, const unsigned);
template UniqueTextureObjectPtr CreateTextureObject<mat33>(const mat33*, const size_t, const cudaChannelFormatKind, const unsigned);
/* *************************************************************** */
} // namespace NiftyReg::Cuda
/* *************************************************************** */
