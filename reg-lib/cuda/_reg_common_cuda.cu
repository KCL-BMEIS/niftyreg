/**
 * @file _reg_common_cuda.cu
 * @author Marc Modat
 * @date 25/03/2009
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_common_cuda.h"
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>

/* *************************************************************** */
template <class NiftiType>
int cudaCommon_transferNiftiToNiftiOnDevice1(nifti_image *imageCuda, const nifti_image *img) {
    const size_t memSize = NiftiImage::calcVoxelNumber(img, 3) * sizeof(NiftiType);
    NR_CUDA_SAFE_CALL(cudaMemcpy(imageCuda, img, sizeof(nifti_image), cudaMemcpyHostToDevice));
    NR_CUDA_SAFE_CALL(cudaMemcpy(imageCuda->data, img->data, memSize, cudaMemcpyHostToDevice));
    NR_CUDA_SAFE_CALL(cudaMemcpy(imageCuda->dim, img->dim, 8 * sizeof(int), cudaMemcpyHostToDevice));
    NR_CUDA_SAFE_CALL(cudaMemcpy(imageCuda->pixdim, img->pixdim, 8 * sizeof(float), cudaMemcpyHostToDevice));
    return EXIT_SUCCESS;
}
/* *************************************************************** */
template <class DataType, class NiftiType>
int cudaCommon_transferNiftiToArrayOnDevice1(DataType *arrayCuda, const nifti_image *img) {
    if (sizeof(DataType) != sizeof(NiftiType)) {
        reg_print_fct_error("cudaCommon_transferNiftiToArrayOnDevice1");
        reg_print_msg_error("The host and device arrays are of different types");
        return EXIT_FAILURE;
    } else {
        const size_t memSize = NiftiImage::calcVoxelNumber(img, 3) * sizeof(NiftiType);
        NR_CUDA_SAFE_CALL(cudaMemcpy(arrayCuda, img->data, memSize, cudaMemcpyHostToDevice));
    }
    return EXIT_SUCCESS;
}
/* *************************************************************** */
template <class DataType>
int cudaCommon_transferNiftiToArrayOnDevice(DataType *arrayCuda, const nifti_image *img) {
    if (sizeof(DataType) == sizeof(float4)) {
        if ((img->datatype != NIFTI_TYPE_FLOAT32) || (img->dim[5] < 2) || (img->dim[4] > 1)) {
            reg_print_fct_error("cudaCommon_transferNiftiToArrayOnDevice");
            reg_print_msg_error("The specified image is not a single precision deformation field image");
            return EXIT_FAILURE;
        }
        const float *niftiImgValues = static_cast<float*>(img->data);
        const size_t voxelNumber = NiftiImage::calcVoxelNumber(img, 3);
        unique_ptr<float4[]> array(new float4[voxelNumber]());
        for (size_t i = 0; i < voxelNumber; i++)
            array[i].x = *niftiImgValues++;
        if (img->dim[5] >= 2) {
            for (size_t i = 0; i < voxelNumber; i++)
                array[i].y = *niftiImgValues++;
        }
        if (img->dim[5] >= 3) {
            for (size_t i = 0; i < voxelNumber; i++)
                array[i].z = *niftiImgValues++;
        }
        if (img->dim[5] >= 4) {
            for (size_t i = 0; i < voxelNumber; i++)
                array[i].w = *niftiImgValues++;
        }
        NR_CUDA_SAFE_CALL(cudaMemcpy(arrayCuda, array.get(), voxelNumber * sizeof(float4), cudaMemcpyHostToDevice));
    } else { // All these else could be removed but the nvcc compiler would warn for unreachable statement
        switch (img->datatype) {
        case NIFTI_TYPE_FLOAT32:
            return cudaCommon_transferNiftiToArrayOnDevice1<DataType, float>(arrayCuda, img);
        default:
            reg_print_fct_error("cudaCommon_transferNiftiToArrayOnDevice");
            reg_print_msg_error("The image data type is not supported");
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
template int cudaCommon_transferNiftiToArrayOnDevice<double>(double*, const nifti_image*);
template int cudaCommon_transferNiftiToArrayOnDevice<float>(float*, const nifti_image*);
template int cudaCommon_transferNiftiToArrayOnDevice<int>(int*, const nifti_image*);
template int cudaCommon_transferNiftiToArrayOnDevice<float4>(float4*, const nifti_image*);
/* *************************************************************** */
template <class DataType, class NiftiType>
int cudaCommon_transferNiftiToArrayOnDevice1(DataType *array1Cuda, DataType *array2Cuda, const nifti_image *img) {
    if (sizeof(DataType) != sizeof(NiftiType)) {
        reg_print_fct_error("cudaCommon_transferNiftiToArrayOnDevice1");
        reg_print_msg_error("The host and device arrays are of different types");
        return EXIT_FAILURE;
    } else {
        const size_t voxelNumber = NiftiImage::calcVoxelNumber(img, 3);
        const size_t memSize = voxelNumber * sizeof(DataType);
        const NiftiType *array1 = static_cast<NiftiType*>(img->data);
        const NiftiType *array2 = &array1[voxelNumber];
        NR_CUDA_SAFE_CALL(cudaMemcpy(array1Cuda, array1, memSize, cudaMemcpyHostToDevice));
        NR_CUDA_SAFE_CALL(cudaMemcpy(array2Cuda, array2, memSize, cudaMemcpyHostToDevice));
    }
    return EXIT_SUCCESS;
}
/* *************************************************************** */
template <class DataType>
int cudaCommon_transferNiftiToArrayOnDevice(DataType *array1Cuda, DataType *array2Cuda, const nifti_image *img) {
    if (sizeof(DataType) == sizeof(float4)) {
        if ((img->datatype != NIFTI_TYPE_FLOAT32) || (img->dim[5] < 2) || (img->dim[4] > 1)) {
            reg_print_fct_error("cudaCommon_transferNiftiToArrayOnDevice");
            reg_print_msg_error("The specified image is not a single precision deformation field image");
            return EXIT_FAILURE;
        }
        const float *niftiImgValues = static_cast<float*>(img->data);
        const size_t voxelNumber = NiftiImage::calcVoxelNumber(img, 3);
        unique_ptr<float4[]> array1(new float4[voxelNumber]());
        unique_ptr<float4[]> array2(new float4[voxelNumber]());
        for (size_t i = 0; i < voxelNumber; i++)
            array1[i].x = *niftiImgValues++;
        for (size_t i = 0; i < voxelNumber; i++)
            array2[i].x = *niftiImgValues++;
        if (img->dim[5] >= 2) {
            for (size_t i = 0; i < voxelNumber; i++)
                array1[i].y = *niftiImgValues++;
            for (size_t i = 0; i < voxelNumber; i++)
                array2[i].y = *niftiImgValues++;
        }
        if (img->dim[5] >= 3) {
            for (size_t i = 0; i < voxelNumber; i++)
                array1[i].z = *niftiImgValues++;
            for (size_t i = 0; i < voxelNumber; i++)
                array2[i].z = *niftiImgValues++;
        }
        if (img->dim[5] >= 4) {
            for (size_t i = 0; i < voxelNumber; i++)
                array1[i].w = *niftiImgValues++;
            for (size_t i = 0; i < voxelNumber; i++)
                array2[i].w = *niftiImgValues++;
        }
        NR_CUDA_SAFE_CALL(cudaMemcpy(array1Cuda, array1.get(), voxelNumber * sizeof(float4), cudaMemcpyHostToDevice));
        NR_CUDA_SAFE_CALL(cudaMemcpy(array2Cuda, array2.get(), voxelNumber * sizeof(float4), cudaMemcpyHostToDevice));
    } else { // All these else could be removed but the nvcc compiler would warn for unreachable statement
        switch (img->datatype) {
        case NIFTI_TYPE_FLOAT32:
            return cudaCommon_transferNiftiToArrayOnDevice1<DataType, float>(array1Cuda, array2Cuda, img);
        default:
            reg_print_fct_error("cudaCommon_transferNiftiToArrayOnDevice");
            reg_print_msg_error("The image data type is not supported");
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
template int cudaCommon_transferNiftiToArrayOnDevice<float>(float*, float*, const nifti_image*);
template int cudaCommon_transferNiftiToArrayOnDevice<double>(double*, double*, const nifti_image*);
template int cudaCommon_transferNiftiToArrayOnDevice<float4>(float4*, float4*, const nifti_image*); // for deformation field
/* *************************************************************** */
template <class DataType, class NiftiType>
int cudaCommon_transferNiftiToArrayOnDevice1(cudaArray *arrayCuda, const nifti_image *img) {
    if (sizeof(DataType) != sizeof(NiftiType)) {
        reg_print_fct_error("cudaCommon_transferNiftiToArrayOnDevice1");
        reg_print_msg_error("The host and device arrays are of different types");
        return EXIT_FAILURE;
    } else {
        cudaMemcpy3DParms copyParams{};
        copyParams.extent = make_cudaExtent(std::abs(img->dim[1]), std::abs(img->dim[2]), std::abs(img->dim[3]));
        copyParams.srcPtr = make_cudaPitchedPtr(img->data,
                                                copyParams.extent.width * sizeof(DataType),
                                                copyParams.extent.width,
                                                copyParams.extent.height);
        copyParams.dstArray = arrayCuda;
        copyParams.kind = cudaMemcpyHostToDevice;
        NR_CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
    }
    return EXIT_SUCCESS;
}
/* *************************************************************** */
template <class DataType>
int cudaCommon_transferNiftiToArrayOnDevice(cudaArray *arrayCuda, const nifti_image *img) {
    if (sizeof(DataType) == sizeof(float4)) {
        if ((img->datatype != NIFTI_TYPE_FLOAT32) || (img->dim[5] < 2) || (img->dim[4] > 1)) {
            reg_print_fct_error("cudaCommon_transferNiftiToArrayOnDevice");
            reg_print_msg_error("The specified image is not a single precision deformation field image");
            return EXIT_FAILURE;
        }
        const float *niftiImgValues = static_cast<float*>(img->data);
        const size_t voxelNumber = NiftiImage::calcVoxelNumber(img, 3);
        unique_ptr<float4[]> array(new float4[voxelNumber]());
        for (size_t i = 0; i < voxelNumber; i++)
            array[i].x = *niftiImgValues++;
        if (img->dim[5] >= 2) {
            for (size_t i = 0; i < voxelNumber; i++)
                array[i].y = *niftiImgValues++;
        }
        if (img->dim[5] >= 3) {
            for (size_t i = 0; i < voxelNumber; i++)
                array[i].z = *niftiImgValues++;
        }
        if (img->dim[5] == 3) {
            for (size_t i = 0; i < voxelNumber; i++)
                array[i].w = *niftiImgValues++;
        }
        cudaMemcpy3DParms copyParams{};
        copyParams.extent = make_cudaExtent(std::abs(img->dim[1]), std::abs(img->dim[2]), std::abs(img->dim[3]));
        copyParams.srcPtr = make_cudaPitchedPtr(array.get(),
                                                copyParams.extent.width * sizeof(DataType),
                                                copyParams.extent.width,
                                                copyParams.extent.height);
        copyParams.dstArray = arrayCuda;
        copyParams.kind = cudaMemcpyHostToDevice;
        NR_CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
    } else { // All these else could be removed but the nvcc compiler would warn for unreachable statement
        switch (img->datatype) {
        case NIFTI_TYPE_FLOAT32:
            return cudaCommon_transferNiftiToArrayOnDevice1<DataType, float>(arrayCuda, img);
        default:
            reg_print_fct_error("cudaCommon_transferNiftiToArrayOnDevice");
            reg_print_msg_error("The image data type is not supported");
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
template int cudaCommon_transferNiftiToArrayOnDevice<int>(cudaArray*, const nifti_image*);
template int cudaCommon_transferNiftiToArrayOnDevice<float>(cudaArray*, const nifti_image*);
template int cudaCommon_transferNiftiToArrayOnDevice<double>(cudaArray*, const nifti_image*);
template int cudaCommon_transferNiftiToArrayOnDevice<float4>(cudaArray*, const nifti_image*); // for deformation field
/* *************************************************************** */
template <class DataType, class NiftiType>
int cudaCommon_transferNiftiToArrayOnDevice1(cudaArray *array1Cuda, cudaArray *array2Cuda, const nifti_image *img) {
    if (sizeof(DataType) != sizeof(NiftiType)) {
        reg_print_fct_error("cudaCommon_transferNiftiToArrayOnDevice1");
        reg_print_msg_error("The host and device arrays are of different types");
        return EXIT_FAILURE;
    } else {
        NiftiType *array1 = static_cast<NiftiType*>(img->data);
        NiftiType *array2 = &array1[NiftiImage::calcVoxelNumber(img, 3)];
        cudaMemcpy3DParms copyParams{};
        copyParams.extent = make_cudaExtent(std::abs(img->dim[1]), std::abs(img->dim[2]), std::abs(img->dim[3]));
        copyParams.kind = cudaMemcpyHostToDevice;
        // First timepoint
        copyParams.srcPtr = make_cudaPitchedPtr(array1,
                                                copyParams.extent.width * sizeof(DataType),
                                                copyParams.extent.width,
                                                copyParams.extent.height);
        copyParams.dstArray = array1Cuda;
        NR_CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
        // Second timepoint
        copyParams.srcPtr = make_cudaPitchedPtr(array2,
                                                copyParams.extent.width * sizeof(DataType),
                                                copyParams.extent.width,
                                                copyParams.extent.height);
        copyParams.dstArray = array2Cuda;
        NR_CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
    }
    return EXIT_SUCCESS;
}
/* *************************************************************** */
template <class DataType>
int cudaCommon_transferNiftiToArrayOnDevice(cudaArray *array1Cuda, cudaArray *array2Cuda, const nifti_image *img) {
    if (sizeof(DataType) == sizeof(float4)) {
        if ((img->datatype != NIFTI_TYPE_FLOAT32) || (img->dim[5] < 2) || (img->dim[4] > 1)) {
            reg_print_fct_error("cudaCommon_transferNiftiToArrayOnDevice1");
            reg_print_msg_error("The specified image is not a single precision deformation field image");
            return EXIT_FAILURE;
        }
        const float *niftiImgValues = static_cast<float*>(img->data);
        const size_t voxelNumber = NiftiImage::calcVoxelNumber(img, 3);
        unique_ptr<float4[]> array1(new float4[voxelNumber]());
        unique_ptr<float4[]> array2(new float4[voxelNumber]());
        for (size_t i = 0; i < voxelNumber; i++)
            array1[i].x = *niftiImgValues++;
        for (size_t i = 0; i < voxelNumber; i++)
            array2[i].x = *niftiImgValues++;
        if (img->dim[5] >= 2) {
            for (size_t i = 0; i < voxelNumber; i++)
                array1[i].y = *niftiImgValues++;
            for (size_t i = 0; i < voxelNumber; i++)
                array2[i].y = *niftiImgValues++;
        }
        if (img->dim[5] >= 3) {
            for (size_t i = 0; i < voxelNumber; i++)
                array1[i].z = *niftiImgValues++;
            for (size_t i = 0; i < voxelNumber; i++)
                array2[i].z = *niftiImgValues++;
        }
        if (img->dim[5] == 3) {
            for (size_t i = 0; i < voxelNumber; i++)
                array1[i].w = *niftiImgValues++;
            for (size_t i = 0; i < voxelNumber; i++)
                array2[i].w = *niftiImgValues++;
        }

        cudaMemcpy3DParms copyParams{};
        copyParams.extent = make_cudaExtent(std::abs(img->dim[1]), std::abs(img->dim[2]), std::abs(img->dim[3]));
        copyParams.kind = cudaMemcpyHostToDevice;
        // First timepoint
        copyParams.srcPtr = make_cudaPitchedPtr(array1.get(),
                                                copyParams.extent.width * sizeof(DataType),
                                                copyParams.extent.width,
                                                copyParams.extent.height);
        copyParams.dstArray = array1Cuda;
        NR_CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
        // Second timepoint
        copyParams.srcPtr = make_cudaPitchedPtr(array2.get(),
                                                copyParams.extent.width * sizeof(DataType),
                                                copyParams.extent.width,
                                                copyParams.extent.height);
        copyParams.dstArray = array2Cuda;
        NR_CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
    } else { // All these else could be removed but the nvcc compiler would warn for unreachable statement
        switch (img->datatype) {
        case NIFTI_TYPE_FLOAT32:
            return cudaCommon_transferNiftiToArrayOnDevice1<DataType, float>(array1Cuda, array2Cuda, img);
        default:
            reg_print_fct_error("cudaCommon_transferNiftiToArrayOnDevice1");
            reg_print_msg_error("The image data type is not supported");
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
template int cudaCommon_transferNiftiToArrayOnDevice<float>(cudaArray*, cudaArray*, const nifti_image*);
template int cudaCommon_transferNiftiToArrayOnDevice<double>(cudaArray*, cudaArray*, const nifti_image*);
template int cudaCommon_transferNiftiToArrayOnDevice<float4>(cudaArray*, cudaArray*, const nifti_image*); // for deformation field
/* *************************************************************** */
template <class DataType>
int cudaCommon_allocateArrayToDevice(cudaArray **arrayCuda, const int *dim) {
    const cudaExtent volumeSize = make_cudaExtent(std::abs(dim[1]), std::abs(dim[2]), std::abs(dim[3]));
    cudaChannelFormatDesc texDesc = cudaCreateChannelDesc<DataType>();
    NR_CUDA_SAFE_CALL(cudaMalloc3DArray(arrayCuda, &texDesc, volumeSize));
    return EXIT_SUCCESS;
}
template int cudaCommon_allocateArrayToDevice<float>(cudaArray**, const int*);
template int cudaCommon_allocateArrayToDevice<double>(cudaArray**, const int*);
template int cudaCommon_allocateArrayToDevice<float4>(cudaArray**, const int*); // for deformation field
/* *************************************************************** */
template <class DataType>
int cudaCommon_allocateArrayToDevice(cudaArray **array1Cuda, cudaArray **array2Cuda, const int *dim) {
    const cudaExtent volumeSize = make_cudaExtent(std::abs(dim[1]), std::abs(dim[2]), std::abs(dim[3]));
    cudaChannelFormatDesc texDesc = cudaCreateChannelDesc<DataType>();
    NR_CUDA_SAFE_CALL(cudaMalloc3DArray(array1Cuda, &texDesc, volumeSize));
    NR_CUDA_SAFE_CALL(cudaMalloc3DArray(array2Cuda, &texDesc, volumeSize));
    return EXIT_SUCCESS;
}
template int cudaCommon_allocateArrayToDevice<float>(cudaArray**, cudaArray**, const int*);
template int cudaCommon_allocateArrayToDevice<double>(cudaArray**, cudaArray**, const int*);
template int cudaCommon_allocateArrayToDevice<float4>(cudaArray**, cudaArray**, const int*); // for deformation field
/* *************************************************************** */
template <class DataType>
int cudaCommon_allocateArrayToDevice(DataType **arrayCuda, const int *dim) {
    const size_t memSize = (size_t)std::abs(dim[1]) * (size_t)std::abs(dim[2]) * (size_t)std::abs(dim[3]) * sizeof(DataType);
    NR_CUDA_SAFE_CALL(cudaMalloc(arrayCuda, memSize));
    return EXIT_SUCCESS;
}
template int cudaCommon_allocateArrayToDevice<float>(float**, const int*);
template int cudaCommon_allocateArrayToDevice<double>(double**, const int*);
template int cudaCommon_allocateArrayToDevice<int>(int**, const int*);
template int cudaCommon_allocateArrayToDevice<float4>(float4**, const int*); // for deformation field
/* *************************************************************** */
template <class DataType>
int cudaCommon_allocateArrayToDevice(DataType **arrayCuda, const size_t& nVoxels) {
    NR_CUDA_SAFE_CALL(cudaMalloc(arrayCuda, nVoxels * sizeof(DataType)));
    return EXIT_SUCCESS;
}
template int cudaCommon_allocateArrayToDevice<float>(float**, const size_t&);
template int cudaCommon_allocateArrayToDevice<double>(double**, const size_t&);
template int cudaCommon_allocateArrayToDevice<int>(int**, const size_t&);
template int cudaCommon_allocateArrayToDevice<float4>(float4**, const size_t&); // for deformation field
/* *************************************************************** */
template <class DataType>
int cudaCommon_allocateArrayToDevice(DataType **array1Cuda, DataType **array2Cuda, const int *dim) {
    const size_t memSize = (size_t)std::abs(dim[1]) * (size_t)std::abs(dim[2]) * (size_t)std::abs(dim[3]) * sizeof(DataType);
    NR_CUDA_SAFE_CALL(cudaMalloc(array1Cuda, memSize));
    NR_CUDA_SAFE_CALL(cudaMalloc(array2Cuda, memSize));
    return EXIT_SUCCESS;
}
template int cudaCommon_allocateArrayToDevice<float>(float**, float**, const int*);
template int cudaCommon_allocateArrayToDevice<double>(double**, double**, const int*);
template int  cudaCommon_allocateArrayToDevice<float4>(float4**, float4**, const int*); // for deformation field
/* *************************************************************** */
template <class DataType>
int cudaCommon_transferFromDeviceToCpu(DataType *cpuPtr, const DataType *cuPtr, const size_t& nElements) {
    NR_CUDA_SAFE_CALL(cudaMemcpy(cpuPtr, cuPtr, nElements * sizeof(DataType), cudaMemcpyDeviceToHost));
    return EXIT_SUCCESS;
}
template int cudaCommon_transferFromDeviceToCpu<float>(float*, const float*, const size_t&);
template int cudaCommon_transferFromDeviceToCpu<double>(double*, const double*, const size_t&);
/* *************************************************************** */
template <class DataType, class NiftiType>
int cudaCommon_transferFromDeviceToNifti1(nifti_image *img, const DataType *arrayCuda) {
    if (sizeof(DataType) != sizeof(NiftiType)) {
        reg_print_fct_error("cudaCommon_transferFromDeviceToNifti1");
        reg_print_msg_error("The host and device arrays are of different types");
        return EXIT_FAILURE;
    } else {
        NR_CUDA_SAFE_CALL(cudaMemcpy(img->data, arrayCuda, img->nvox * sizeof(DataType), cudaMemcpyDeviceToHost));
    }
    return EXIT_SUCCESS;
}
/* *************************************************************** */
template <class DataType>
int cudaCommon_transferFromDeviceToNifti(nifti_image *img, const DataType *arrayCuda) {
    if (sizeof(DataType) == sizeof(float4)) {
        // A nifti 5D volume is expected
        if (img->dim[0] < 5 || img->dim[4]>1 || img->dim[5] < 2 || img->datatype != NIFTI_TYPE_FLOAT32) {
            reg_print_fct_error("cudaCommon_transferFromDeviceToNifti");
            reg_print_msg_error("The nifti image is not a 5D volume");
            return EXIT_FAILURE;
        }
        const size_t voxelNumber = NiftiImage::calcVoxelNumber(img, 3);
        thrust::device_ptr<const float4> arrayCudaPtr(reinterpret_cast<const float4*>(arrayCuda));
        const thrust::host_vector<float4> array(arrayCudaPtr, arrayCudaPtr + voxelNumber);
        float *niftiImgValues = static_cast<float*>(img->data);
        for (size_t i = 0; i < voxelNumber; i++)
            *niftiImgValues++ = array[i].x;
        if (img->dim[5] >= 2) {
            for (size_t i = 0; i < voxelNumber; i++)
                *niftiImgValues++ = array[i].y;
        }
        if (img->dim[5] >= 3) {
            for (size_t i = 0; i < voxelNumber; i++)
                *niftiImgValues++ = array[i].z;
        }
        if (img->dim[5] >= 4) {
            for (size_t i = 0; i < voxelNumber; i++)
                *niftiImgValues++ = array[i].w;
        }
        return EXIT_SUCCESS;
    } else {
        switch (img->datatype) {
        case NIFTI_TYPE_FLOAT32:
            return cudaCommon_transferFromDeviceToNifti1<DataType, float>(img, arrayCuda);
        default:
            reg_print_fct_error("cudaCommon_transferFromDeviceToNifti");
            reg_print_msg_error("The image data type is not supported");
            return EXIT_FAILURE;
        }
    }
}
template int cudaCommon_transferFromDeviceToNifti<float>(nifti_image*, const float*);
template int cudaCommon_transferFromDeviceToNifti<double>(nifti_image*, const double*);
template int cudaCommon_transferFromDeviceToNifti<float4>(nifti_image*, const float4*); // for deformation field
/* *************************************************************** */
template<>
int cudaCommon_transferFromDeviceToNifti(nifti_image *img, const cudaArray *arrayCuda) {
    if (img->datatype != NIFTI_TYPE_FLOAT32) {
        reg_print_fct_error("cudaCommon_transferFromDeviceToNifti");
        reg_print_msg_error("The image data type is not supported");
        return EXIT_FAILURE;
    }
    cudaMemcpy3DParms copyParams{};
    copyParams.extent = make_cudaExtent(std::abs(img->dim[1]), std::abs(img->dim[2]), std::abs(img->dim[3]));
    copyParams.srcArray = const_cast<cudaArray*>(arrayCuda);
    copyParams.dstPtr = make_cudaPitchedPtr(img->data,
                                            copyParams.extent.width * sizeof(float),
                                            copyParams.extent.width,
                                            copyParams.extent.height);
    copyParams.kind = cudaMemcpyDeviceToHost;
    NR_CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
    return EXIT_SUCCESS;
}
/* *************************************************************** */
template <class DataType, class NiftiType>
int cudaCommon_transferFromDeviceToNifti1(nifti_image *img, const DataType *array1Cuda, const DataType *array2Cuda) {
    if (sizeof(DataType) != sizeof(NiftiType)) {
        reg_print_fct_error("cudaCommon_transferFromDeviceToNifti1");
        reg_print_msg_error("The host and device arrays are of different types");
        return EXIT_FAILURE;
    } else {
        const size_t voxelNumber = NiftiImage::calcVoxelNumber(img, 3);
        NiftiType *array1 = static_cast<NiftiType*>(img->data);
        NiftiType *array2 = &array1[voxelNumber];
        NR_CUDA_SAFE_CALL(cudaMemcpy(array1, array1Cuda, voxelNumber * sizeof(DataType), cudaMemcpyDeviceToHost));
        NR_CUDA_SAFE_CALL(cudaMemcpy(array2, array2Cuda, voxelNumber * sizeof(DataType), cudaMemcpyDeviceToHost));
    }
    return EXIT_SUCCESS;
}
/* *************************************************************** */
template <class DataType>
int cudaCommon_transferFromDeviceToNifti(nifti_image *img, const DataType *array1Cuda, const DataType *array2Cuda) {
    if (sizeof(DataType) == sizeof(float4)) {
        // A nifti 5D volume is expected
        if (img->dim[0] < 5 || img->dim[4]>1 || img->dim[5] < 2 || img->datatype != NIFTI_TYPE_FLOAT32) {
            reg_print_fct_error("cudaCommon_transferFromDeviceToNifti");
            reg_print_msg_error("The nifti image is not a 5D volume");
            return EXIT_FAILURE;
        }
        const size_t voxelNumber = NiftiImage::calcVoxelNumber(img, 3);
        thrust::device_ptr<const float4> array1CudaPtr(reinterpret_cast<const float4*>(array1Cuda));
        thrust::device_ptr<const float4> array2CudaPtr(reinterpret_cast<const float4*>(array2Cuda));
        const thrust::host_vector<float4> array1(array1CudaPtr, array1CudaPtr + voxelNumber);
        const thrust::host_vector<float4> array2(array2CudaPtr, array2CudaPtr + voxelNumber);
        float *niftiImgValues = static_cast<float*>(img->data);
        for (size_t i = 0; i < voxelNumber; i++) {
            *niftiImgValues++ = array1[i].x;
        }
        for (size_t i = 0; i < voxelNumber; i++) {
            *niftiImgValues++ = array2[i].x;
        }
        if (img->dim[5] >= 2) {
            for (size_t i = 0; i < voxelNumber; i++) {
                *niftiImgValues++ = array1[i].y;
            }
            for (size_t i = 0; i < voxelNumber; i++) {
                *niftiImgValues++ = array2[i].y;
            }
        }
        if (img->dim[5] >= 3) {
            for (size_t i = 0; i < voxelNumber; i++) {
                *niftiImgValues++ = array1[i].z;
            }
            for (size_t i = 0; i < voxelNumber; i++) {
                *niftiImgValues++ = array2[i].z;
            }
        }
        if (img->dim[5] >= 4) {
            for (size_t i = 0; i < voxelNumber; i++) {
                *niftiImgValues++ = array1[i].w;
            }
            for (size_t i = 0; i < voxelNumber; i++) {
                *niftiImgValues++ = array2[i].w;
            }
        }
        return EXIT_SUCCESS;
    } else {
        switch (img->datatype) {
        case NIFTI_TYPE_FLOAT32:
            return cudaCommon_transferFromDeviceToNifti1<DataType, float>(img, array1Cuda, array2Cuda);
        default:
            reg_print_fct_error("cudaCommon_transferFromDeviceToNifti");
            reg_print_msg_error("The image data type is not supported");
            return EXIT_FAILURE;
        }
    }
}
template int cudaCommon_transferFromDeviceToNifti<float>(nifti_image*, const float*, const float*);
template int cudaCommon_transferFromDeviceToNifti<double>(nifti_image*, const double*, const double*);
template int cudaCommon_transferFromDeviceToNifti<float4>(nifti_image*, const float4*, const float4*); // for deformation field
/* *************************************************************** */
void cudaCommon_free(cudaArray *arrayCuda) {
    if (arrayCuda != nullptr)
        NR_CUDA_SAFE_CALL(cudaFreeArray(arrayCuda));
}
/* *************************************************************** */
template <class DataType>
void cudaCommon_free(DataType *arrayCuda) {
    if (arrayCuda != nullptr)
        NR_CUDA_SAFE_CALL(cudaFree(arrayCuda));
}
template void cudaCommon_free<int>(int*);
template void cudaCommon_free<float>(float*);
template void cudaCommon_free<double>(double*);
template void cudaCommon_free<float4>(float4*);
/* *************************************************************** */
template <class DataType>
int cudaCommon_transferFromDeviceToNiftiSimple(DataType *arrayCuda, const nifti_image *img) {
    NR_CUDA_SAFE_CALL(cudaMemcpy(arrayCuda, img->data, img->nvox * sizeof(DataType), cudaMemcpyHostToDevice));
    return EXIT_SUCCESS;
}
template int cudaCommon_transferFromDeviceToNiftiSimple<int>(int*, const nifti_image*);
template int cudaCommon_transferFromDeviceToNiftiSimple<float>(float*, const nifti_image*);
template int cudaCommon_transferFromDeviceToNiftiSimple<double>(double*, const nifti_image*);
/* *************************************************************** */
template <class DataType>
int cudaCommon_transferFromDeviceToNiftiSimple1(DataType *arrayCuda, const DataType *img, const size_t& nvox) {
    NR_CUDA_SAFE_CALL(cudaMemcpy(arrayCuda, img, nvox * sizeof(DataType), cudaMemcpyHostToDevice));
    return EXIT_SUCCESS;
}
template int cudaCommon_transferFromDeviceToNiftiSimple1<int>(int*, const int*, const size_t&);
template int cudaCommon_transferFromDeviceToNiftiSimple1<float>(float*, const float*, const size_t&);
template int cudaCommon_transferFromDeviceToNiftiSimple1<double>(double*, const double*, const size_t&);
/* *************************************************************** */
template <class DataType>
int cudaCommon_transferArrayFromCpuToDevice(DataType *arrayCuda, const DataType *arrayCpu, const size_t& nElements) {
    NR_CUDA_SAFE_CALL(cudaMemcpy(arrayCuda, arrayCpu, nElements * sizeof(DataType), cudaMemcpyHostToDevice));
    return EXIT_SUCCESS;
}
template int cudaCommon_transferArrayFromCpuToDevice<int>(int*, const int*, const size_t&);
template int cudaCommon_transferArrayFromCpuToDevice<float>(float*, const float*, const size_t&);
template int cudaCommon_transferArrayFromCpuToDevice<double>(double*, const double*, const size_t&);
/* *************************************************************** */
template <class DataType>
int cudaCommon_transferArrayFromDeviceToCpu(DataType *arrayCpu, const DataType *arrayCuda, const size_t& nElements) {
    NR_CUDA_SAFE_CALL(cudaMemcpy(arrayCpu, arrayCuda, nElements * sizeof(DataType), cudaMemcpyDeviceToHost));
    return EXIT_SUCCESS;
}
template int cudaCommon_transferArrayFromDeviceToCpu<int>(int*, const int*, const size_t&);
template int cudaCommon_transferArrayFromDeviceToCpu<float>(float*, const float*, const size_t&);
template int cudaCommon_transferArrayFromDeviceToCpu<double>(double*, const double*, const size_t&);
/* *************************************************************** */
void cudaCommon_destroyTextureObject(cudaTextureObject_t *texObj) {
    NR_CUDA_SAFE_CALL(cudaDestroyTextureObject(*texObj));
    delete texObj;
}
/* *************************************************************** */
UniqueTextureObjectPtr cudaCommon_createTextureObject(const void *devPtr,
                                                      const cudaResourceType& resType,
                                                      const size_t& size,
                                                      const cudaChannelFormatKind& channelFormat,
                                                      const unsigned& channelCount,
                                                      const cudaTextureFilterMode& filterMode,
                                                      const bool& normalizedCoordinates) {
    // Specify texture
    cudaResourceDesc resDesc{};
    resDesc.resType = resType;
    switch (resType) {
    case cudaResourceTypeLinear:
        resDesc.res.linear.devPtr = const_cast<void*>(devPtr);
        resDesc.res.linear.desc.f = channelFormat;
        resDesc.res.linear.desc.x = 32;
        if (channelCount > 1)
            resDesc.res.linear.desc.y = 32;
        if (channelCount > 2)
            resDesc.res.linear.desc.z = 32;
        if (channelCount > 3)
            resDesc.res.linear.desc.w = 32;
        resDesc.res.linear.sizeInBytes = size;
        break;
    case cudaResourceTypeArray:
        resDesc.res.array.array = static_cast<cudaArray*>(const_cast<void*>(devPtr));
        break;
    default:
        reg_print_fct_error("cudaCommon_createTextureObject");
        reg_print_msg_error("Unsupported resource type");
        reg_exit();
    }

    // Specify texture object parameters
    cudaTextureDesc texDesc{};
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.addressMode[2] = cudaAddressModeWrap;
    texDesc.filterMode = filterMode;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = normalizedCoordinates;

    // Create texture object
    UniqueTextureObjectPtr texObj(new cudaTextureObject_t(), cudaCommon_destroyTextureObject);
    NR_CUDA_SAFE_CALL(cudaCreateTextureObject(texObj.get(), &resDesc, &texDesc, nullptr));

    return texObj;
}
/* *************************************************************** */
