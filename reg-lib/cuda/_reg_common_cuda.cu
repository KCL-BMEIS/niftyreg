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

/* *************************************************************** */
template <class NiftiType>
int cudaCommon_transferNiftiToNiftiOnDevice1(nifti_image *image_d, nifti_image *img) {
    const unsigned memSize = img->dim[1] * img->dim[2] * img->dim[3] * sizeof(NiftiType);

    int *g_dim;
    float* g_pixdim;
    NiftiType* g_data;

    NR_CUDA_SAFE_CALL(cudaMalloc((void**)&g_dim, 8 * sizeof(int)));
    NR_CUDA_SAFE_CALL(cudaMalloc((void**)&g_pixdim, 8 * sizeof(float)));
    NR_CUDA_SAFE_CALL(cudaMalloc((void**)&g_data, memSize));

    NiftiType *array_h = static_cast<NiftiType*>(img->data);
    NR_CUDA_SAFE_CALL(cudaMemcpy(image_d, img, sizeof(nifti_image), cudaMemcpyHostToDevice));

    NR_CUDA_SAFE_CALL(cudaMemcpy(image_d->data, array_h, memSize, cudaMemcpyHostToDevice));
    NR_CUDA_SAFE_CALL(cudaMemcpy(image_d->dim, img->dim, 8 * sizeof(int), cudaMemcpyHostToDevice));
    NR_CUDA_SAFE_CALL(cudaMemcpy(image_d->pixdim, img->pixdim, 8 * sizeof(float), cudaMemcpyHostToDevice));

    return EXIT_SUCCESS;
}
template int cudaCommon_transferNiftiToNiftiOnDevice1<float>(nifti_image*, nifti_image*);
template int cudaCommon_transferNiftiToNiftiOnDevice1<double>(nifti_image*, nifti_image*);
/* *************************************************************** */
template <class DataType, class NiftiType>
int cudaCommon_transferNiftiToArrayOnDevice1(DataType *array_d, nifti_image *img) {
    if (sizeof(DataType) != sizeof(NiftiType)) {
        reg_print_fct_error("cudaCommon_transferNiftiToArrayOnDevice1");
        reg_print_msg_error("The host and device arrays are of different types");
        return EXIT_FAILURE;
    } else {
        const unsigned memSize = img->dim[1] * img->dim[2] * img->dim[3] * sizeof(DataType);
        NiftiType *array_h = static_cast<NiftiType*>(img->data);
        NR_CUDA_SAFE_CALL(cudaMemcpy(array_d, array_h, memSize, cudaMemcpyHostToDevice));
    }
    return EXIT_SUCCESS;
}
/* *************************************************************** */
template <class DataType>
int cudaCommon_transferNiftiToArrayOnDevice(DataType *array_d, nifti_image *img) {
    if (sizeof(DataType) == sizeof(float4)) {
        if ((img->datatype != NIFTI_TYPE_FLOAT32) || (img->dim[5] < 2) || (img->dim[4] > 1)) {
            reg_print_fct_error("cudaCommon_transferNiftiToArrayOnDevice");
            reg_print_msg_error("The specified image is not a single precision deformation field image");
            return EXIT_FAILURE;
        }
        float *niftiImgValues = static_cast<float*>(img->data);
        const size_t voxelNumber = CalcVoxelNumber(*img);
        float4 *array_h = (float4*)calloc(voxelNumber, sizeof(float4));
        for (size_t i = 0; i < voxelNumber; i++)
            array_h[i].x = *niftiImgValues++;
        if (img->dim[5] >= 2) {
            for (size_t i = 0; i < voxelNumber; i++)
                array_h[i].y = *niftiImgValues++;
        }
        if (img->dim[5] >= 3) {
            for (size_t i = 0; i < voxelNumber; i++)
                array_h[i].z = *niftiImgValues++;
        }
        if (img->dim[5] >= 4) {
            for (size_t i = 0; i < voxelNumber; i++)
                array_h[i].w = *niftiImgValues++;
        }
        NR_CUDA_SAFE_CALL(cudaMemcpy(array_d, array_h, voxelNumber * sizeof(float4), cudaMemcpyHostToDevice));
        free(array_h);
    } else { // All these else could be removed but the nvcc compiler would warn for unreachable statement
        switch (img->datatype) {
        case NIFTI_TYPE_FLOAT32:
            return cudaCommon_transferNiftiToArrayOnDevice1<DataType, float>(array_d, img);
        default:
            reg_print_fct_error("cudaCommon_transferNiftiToArrayOnDevice");
            reg_print_msg_error("The image data type is not supported");
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
template int cudaCommon_transferNiftiToArrayOnDevice<double>(double*, nifti_image*);
template int cudaCommon_transferNiftiToArrayOnDevice<float>(float*, nifti_image*);
template int cudaCommon_transferNiftiToArrayOnDevice<int>(int*, nifti_image*);
template int cudaCommon_transferNiftiToArrayOnDevice<float4>(float4*, nifti_image*);
/* *************************************************************** */
template <class DataType, class NiftiType>
int cudaCommon_transferNiftiToArrayOnDevice1(DataType *array_d, DataType *array2_d, nifti_image *img) {
    if (sizeof(DataType) != sizeof(NiftiType)) {
        reg_print_fct_error("cudaCommon_transferNiftiToArrayOnDevice1");
        reg_print_msg_error("The host and device arrays are of different types");
        return EXIT_FAILURE;
    } else {
        const unsigned memSize = img->dim[1] * img->dim[2] * img->dim[3] * sizeof(DataType);
        NiftiType *array_h = static_cast<NiftiType*>(img->data);
        NiftiType *array2_h = &array_h[img->dim[1] * img->dim[2] * img->dim[3]];
        NR_CUDA_SAFE_CALL(cudaMemcpy(array_d, array_h, memSize, cudaMemcpyHostToDevice));
        NR_CUDA_SAFE_CALL(cudaMemcpy(array2_d, array2_h, memSize, cudaMemcpyHostToDevice));
    }
    return EXIT_SUCCESS;
}
/* *************************************************************** */
template <class DataType>
int cudaCommon_transferNiftiToArrayOnDevice(DataType *array_d, DataType *array2_d, nifti_image *img) {
    if (sizeof(DataType) == sizeof(float4)) {
        if ((img->datatype != NIFTI_TYPE_FLOAT32) || (img->dim[5] < 2) || (img->dim[4] > 1)) {
            reg_print_fct_error("cudaCommon_transferNiftiToArrayOnDevice");
            reg_print_msg_error("The specified image is not a single precision deformation field image");
            return EXIT_FAILURE;
        }
        float *niftiImgValues = static_cast<float *>(img->data);
        const size_t voxelNumber = CalcVoxelNumber(*img);
        float4 *array_h = (float4*)calloc(voxelNumber, sizeof(float4));
        float4 *array2_h = (float4*)calloc(voxelNumber, sizeof(float4));
        for (size_t i = 0; i < voxelNumber; i++)
            array_h[i].x = *niftiImgValues++;
        for (size_t i = 0; i < voxelNumber; i++)
            array2_h[i].x = *niftiImgValues++;
        if (img->dim[5] >= 2) {
            for (size_t i = 0; i < voxelNumber; i++)
                array_h[i].y = *niftiImgValues++;
            for (size_t i = 0; i < voxelNumber; i++)
                array2_h[i].y = *niftiImgValues++;
        }
        if (img->dim[5] >= 3) {
            for (size_t i = 0; i < voxelNumber; i++)
                array_h[i].z = *niftiImgValues++;
            for (size_t i = 0; i < voxelNumber; i++)
                array2_h[i].z = *niftiImgValues++;
        }
        if (img->dim[5] >= 4) {
            for (size_t i = 0; i < voxelNumber; i++)
                array_h[i].w = *niftiImgValues++;
            for (size_t i = 0; i < voxelNumber; i++)
                array2_h[i].w = *niftiImgValues++;
        }
        NR_CUDA_SAFE_CALL(cudaMemcpy(array_d, array_h, voxelNumber * sizeof(float4), cudaMemcpyHostToDevice));
        NR_CUDA_SAFE_CALL(cudaMemcpy(array2_d, array2_h, voxelNumber * sizeof(float4), cudaMemcpyHostToDevice));
        free(array_h);
        free(array2_h);
    } else { // All these else could be removed but the nvcc compiler would warn for unreachable statement
        switch (img->datatype) {
        case NIFTI_TYPE_FLOAT32:
            return cudaCommon_transferNiftiToArrayOnDevice1<DataType, float>(array_d, array2_d, img);
        default:
            reg_print_fct_error("cudaCommon_transferNiftiToArrayOnDevice");
            reg_print_msg_error("The image data type is not supported");
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
template int cudaCommon_transferNiftiToArrayOnDevice<float>(float*, float*, nifti_image*);
template int cudaCommon_transferNiftiToArrayOnDevice<double>(double*, double*, nifti_image*);
template int cudaCommon_transferNiftiToArrayOnDevice<float4>(float4*, float4*, nifti_image*); // for deformation field
/* *************************************************************** */
template <class DataType, class NiftiType>
int cudaCommon_transferNiftiToArrayOnDevice1(cudaArray *cuArray_d, nifti_image *img) {
    if (sizeof(DataType) != sizeof(NiftiType)) {
        reg_print_fct_error("cudaCommon_transferNiftiToArrayOnDevice1");
        reg_print_msg_error("The host and device arrays are of different types");
        return EXIT_FAILURE;
    } else {
        NiftiType *array_h = static_cast<NiftiType*>(img->data);

        cudaMemcpy3DParms copyParams; memset(&copyParams, 0, sizeof(copyParams));
        copyParams.extent = make_cudaExtent(img->dim[1], img->dim[2], img->dim[3]);
        copyParams.srcPtr = make_cudaPitchedPtr((void*)array_h,
                                                copyParams.extent.width * sizeof(DataType),
                                                copyParams.extent.width,
                                                copyParams.extent.height);
        copyParams.dstArray = cuArray_d;
        copyParams.kind = cudaMemcpyHostToDevice;
        NR_CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
    }
    return EXIT_SUCCESS;
}
/* *************************************************************** */
template <class DataType>
int cudaCommon_transferNiftiToArrayOnDevice(cudaArray *cuArray_d, nifti_image *img) {
    if (sizeof(DataType) == sizeof(float4)) {
        if ((img->datatype != NIFTI_TYPE_FLOAT32) || (img->dim[5] < 2) || (img->dim[4] > 1)) {
            reg_print_fct_error("cudaCommon_transferNiftiToArrayOnDevice");
            reg_print_msg_error("The specified image is not a single precision deformation field image");
            return EXIT_FAILURE;
        }
        float *niftiImgValues = static_cast<float *>(img->data);
        const size_t voxelNumber = CalcVoxelNumber(*img);
        float4 *array_h = (float4*)calloc(voxelNumber, sizeof(float4));

        for (size_t i = 0; i < voxelNumber; i++)
            array_h[i].x = *niftiImgValues++;
        if (img->dim[5] >= 2) {
            for (size_t i = 0; i < voxelNumber; i++)
                array_h[i].y = *niftiImgValues++;
        }
        if (img->dim[5] >= 3) {
            for (size_t i = 0; i < voxelNumber; i++)
                array_h[i].z = *niftiImgValues++;
        }
        if (img->dim[5] == 3) {
            for (size_t i = 0; i < voxelNumber; i++)
                array_h[i].w = *niftiImgValues++;
        }
        cudaMemcpy3DParms copyParams; memset(&copyParams, 0, sizeof(copyParams));
        copyParams.extent = make_cudaExtent(img->dim[1], img->dim[2], img->dim[3]);
        copyParams.srcPtr = make_cudaPitchedPtr((void*)array_h,
                                                copyParams.extent.width * sizeof(DataType),
                                                copyParams.extent.width,
                                                copyParams.extent.height);
        copyParams.dstArray = cuArray_d;
        copyParams.kind = cudaMemcpyHostToDevice;
        NR_CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
        free(array_h);
    } else { // All these else could be removed but the nvcc compiler would warn for unreachable statement
        switch (img->datatype) {
        case NIFTI_TYPE_FLOAT32:
            return cudaCommon_transferNiftiToArrayOnDevice1<DataType, float>(cuArray_d, img);
        default:
            reg_print_fct_error("cudaCommon_transferNiftiToArrayOnDevice");
            reg_print_msg_error("The image data type is not supported");
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
template int cudaCommon_transferNiftiToArrayOnDevice<int>(cudaArray*, nifti_image*);
template int cudaCommon_transferNiftiToArrayOnDevice<float>(cudaArray*, nifti_image*);
template int cudaCommon_transferNiftiToArrayOnDevice<double>(cudaArray*, nifti_image*);
template int cudaCommon_transferNiftiToArrayOnDevice<float4>(cudaArray*, nifti_image*); // for deformation field
/* *************************************************************** */
template <class DataType, class NiftiType>
int cudaCommon_transferNiftiToArrayOnDevice1(cudaArray *cuArray_d, cudaArray *cuArray2_d, nifti_image *img) {
    if (sizeof(DataType) != sizeof(NiftiType)) {
        reg_print_fct_error("cudaCommon_transferNiftiToArrayOnDevice1");
        reg_print_msg_error("The host and device arrays are of different types");
        return EXIT_FAILURE;
    } else {
        NiftiType *array_h = static_cast<NiftiType*>(img->data);
        NiftiType *array2_h = &array_h[img->dim[1] * img->dim[2] * img->dim[3]];

        cudaMemcpy3DParms copyParams; memset(&copyParams, 0, sizeof(copyParams));
        copyParams.extent = make_cudaExtent(img->dim[1], img->dim[2], img->dim[3]);
        copyParams.kind = cudaMemcpyHostToDevice;
        // First timepoint
        copyParams.srcPtr = make_cudaPitchedPtr((void*)array_h,
                                                copyParams.extent.width * sizeof(DataType),
                                                copyParams.extent.width,
                                                copyParams.extent.height);
        copyParams.dstArray = cuArray_d;
        NR_CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
        // Second timepoint
        copyParams.srcPtr = make_cudaPitchedPtr((void*)array2_h,
                                                copyParams.extent.width * sizeof(DataType),
                                                copyParams.extent.width,
                                                copyParams.extent.height);
        copyParams.dstArray = cuArray2_d;
        NR_CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
    }
    return EXIT_SUCCESS;
}
/* *************************************************************** */
template <class DataType>
int cudaCommon_transferNiftiToArrayOnDevice(cudaArray *cuArray_d, cudaArray *cuArray2_d, nifti_image *img) {
    if (sizeof(DataType) == sizeof(float4)) {
        if ((img->datatype != NIFTI_TYPE_FLOAT32) || (img->dim[5] < 2) || (img->dim[4] > 1)) {
            reg_print_fct_error("cudaCommon_transferNiftiToArrayOnDevice1");
            reg_print_msg_error("The specified image is not a single precision deformation field image");
            return EXIT_FAILURE;
        }
        float *niftiImgValues = static_cast<float*>(img->data);
        const size_t voxelNumber = CalcVoxelNumber(*img);
        float4 *array_h = (float4*)calloc(voxelNumber, sizeof(float4));
        float4 *array2_h = (float4*)calloc(voxelNumber, sizeof(float4));

        for (size_t i = 0; i < voxelNumber; i++)
            array_h[i].x = *niftiImgValues++;
        for (size_t i = 0; i < voxelNumber; i++)
            array2_h[i].x = *niftiImgValues++;

        if (img->dim[5] >= 2) {
            for (size_t i = 0; i < voxelNumber; i++)
                array_h[i].y = *niftiImgValues++;
            for (size_t i = 0; i < voxelNumber; i++)
                array2_h[i].y = *niftiImgValues++;
        }

        if (img->dim[5] >= 3) {
            for (size_t i = 0; i < voxelNumber; i++)
                array_h[i].z = *niftiImgValues++;
            for (size_t i = 0; i < voxelNumber; i++)
                array2_h[i].z = *niftiImgValues++;
        }

        if (img->dim[5] == 3) {
            for (size_t i = 0; i < voxelNumber; i++)
                array_h[i].w = *niftiImgValues++;
            for (size_t i = 0; i < voxelNumber; i++)
                array2_h[i].w = *niftiImgValues++;
        }

        cudaMemcpy3DParms copyParams; memset(&copyParams, 0, sizeof(copyParams));
        copyParams.extent = make_cudaExtent(img->dim[1], img->dim[2], img->dim[3]);
        copyParams.kind = cudaMemcpyHostToDevice;
        // First timepoint
        copyParams.srcPtr = make_cudaPitchedPtr((void*)array_h,
                                                copyParams.extent.width * sizeof(DataType),
                                                copyParams.extent.width,
                                                copyParams.extent.height);
        copyParams.dstArray = cuArray_d;
        NR_CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
        free(array_h);
        // Second timepoint
        copyParams.srcPtr = make_cudaPitchedPtr((void*)array2_h,
                                                copyParams.extent.width * sizeof(DataType),
                                                copyParams.extent.width,
                                                copyParams.extent.height);
        copyParams.dstArray = cuArray2_d;
        NR_CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
        free(array2_h);
    } else { // All these else could be removed but the nvcc compiler would warn for unreachable statement
        switch (img->datatype) {
        case NIFTI_TYPE_FLOAT32:
            return cudaCommon_transferNiftiToArrayOnDevice1<DataType, float>(cuArray_d, cuArray2_d, img);
        default:
            reg_print_fct_error("cudaCommon_transferNiftiToArrayOnDevice1");
            reg_print_msg_error("The image data type is not supported");
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
template int cudaCommon_transferNiftiToArrayOnDevice<float>(cudaArray*, cudaArray*, nifti_image*);
template int cudaCommon_transferNiftiToArrayOnDevice<double>(cudaArray*, cudaArray*, nifti_image*);
template int cudaCommon_transferNiftiToArrayOnDevice<float4>(cudaArray*, cudaArray*, nifti_image*); // for deformation field
/* *************************************************************** */
template <class DataType>
int cudaCommon_allocateArrayToDevice(cudaArray **cuArray_d, int *dim) {
    const cudaExtent volumeSize = make_cudaExtent(dim[1], dim[2], dim[3]);
    cudaChannelFormatDesc texDesc = cudaCreateChannelDesc<DataType>();
    NR_CUDA_SAFE_CALL(cudaMalloc3DArray(cuArray_d, &texDesc, volumeSize));
    return EXIT_SUCCESS;
}
template int cudaCommon_allocateArrayToDevice<float>(cudaArray**, int*);
template int cudaCommon_allocateArrayToDevice<double>(cudaArray**, int*);
template int cudaCommon_allocateArrayToDevice<float4>(cudaArray**, int*); // for deformation field
/* *************************************************************** */
template <class DataType>
int cudaCommon_allocateArrayToDevice(cudaArray **cuArray_d, cudaArray **cuArray2_d, int *dim) {
    const cudaExtent volumeSize = make_cudaExtent(dim[1], dim[2], dim[3]);
    cudaChannelFormatDesc texDesc = cudaCreateChannelDesc<DataType>();
    NR_CUDA_SAFE_CALL(cudaMalloc3DArray(cuArray_d, &texDesc, volumeSize));
    NR_CUDA_SAFE_CALL(cudaMalloc3DArray(cuArray2_d, &texDesc, volumeSize));
    return EXIT_SUCCESS;
}
template int cudaCommon_allocateArrayToDevice<float>(cudaArray**, cudaArray**, int*);
template int cudaCommon_allocateArrayToDevice<double>(cudaArray**, cudaArray**, int*);
template int cudaCommon_allocateArrayToDevice<float4>(cudaArray**, cudaArray**, int*); // for deformation field
/* *************************************************************** */
template <class DataType>
int cudaCommon_allocateArrayToDevice(DataType **array_d, int *dim) {
    const unsigned memSize = dim[1] * dim[2] * dim[3] * sizeof(DataType);
    NR_CUDA_SAFE_CALL(cudaMalloc(array_d, memSize));
    return EXIT_SUCCESS;
}
template int cudaCommon_allocateArrayToDevice<float>(float**, int*);
template int cudaCommon_allocateArrayToDevice<double>(double**, int*);
template int cudaCommon_allocateArrayToDevice<int>(int**, int*);
template int cudaCommon_allocateArrayToDevice<float4>(float4**, int*); // for deformation field
/* *************************************************************** */
template <class DataType>
int cudaCommon_allocateArrayToDevice(DataType **array_d, int vox) {
    const unsigned memSize = vox * sizeof(DataType);
    NR_CUDA_SAFE_CALL(cudaMalloc(array_d, memSize));
    return EXIT_SUCCESS;
}
template int cudaCommon_allocateArrayToDevice<float>(float**, int);
template int cudaCommon_allocateArrayToDevice<double>(double**, int);
template int cudaCommon_allocateArrayToDevice<int>(int**, int);
template int cudaCommon_allocateArrayToDevice<float4>(float4**, int); // for deformation field
/* *************************************************************** */
template <class DataType>
int cudaCommon_allocateArrayToDevice(DataType **array_d, DataType **array2_d, int *dim) {
    const unsigned memSize = dim[1] * dim[2] * dim[3] * sizeof(DataType);
    NR_CUDA_SAFE_CALL(cudaMalloc(array_d, memSize));
    NR_CUDA_SAFE_CALL(cudaMalloc(array2_d, memSize));
    return EXIT_SUCCESS;
}
template int cudaCommon_allocateArrayToDevice<float>(float**, float**, int*);
template int cudaCommon_allocateArrayToDevice<double>(double**, double**, int*);
template int  cudaCommon_allocateArrayToDevice<float4>(float4**, float4**, int*); // for deformation field
/* *************************************************************** */
template <class DataType>
int cudaCommon_transferFromDeviceToCpu(DataType *cpuPtr, DataType *cuPtr, const unsigned nElements) {
    NR_CUDA_SAFE_CALL(cudaMemcpy((void*)cpuPtr, (void*)cuPtr, nElements * sizeof(DataType), cudaMemcpyDeviceToHost));
    return EXIT_SUCCESS;
}
template int cudaCommon_transferFromDeviceToCpu<float>(float *cpuPtr, float *cuPtr, const unsigned nElements);
template int cudaCommon_transferFromDeviceToCpu<double>(double *cpuPtr, double *cuPtr, const unsigned nElements);
/* *************************************************************** */
template <class DataType, class NiftiType>
int cudaCommon_transferFromDeviceToNifti1(nifti_image *img, DataType *array_d) {
    if (sizeof(DataType) != sizeof(NiftiType)) {
        reg_print_fct_error("cudaCommon_transferFromDeviceToNifti1");
        reg_print_msg_error("The host and device arrays are of different types");
        return EXIT_FAILURE;
    } else {
        NiftiType *array_h = static_cast<NiftiType*>(img->data);
        NR_CUDA_SAFE_CALL(cudaMemcpy((void*)array_h, (void*)array_d, img->nvox * sizeof(DataType), cudaMemcpyDeviceToHost));
    }
    return EXIT_SUCCESS;
}
template int cudaCommon_transferFromDeviceToNifti1<float, float>(nifti_image *img, float *array_d);
template int cudaCommon_transferFromDeviceToNifti1<double, double>(nifti_image *img, double *array_d);
/* *************************************************************** */
template <class DataType>
int cudaCommon_transferFromDeviceToNifti(nifti_image *img, DataType *array_d) {
    if (sizeof(DataType) == sizeof(float4)) {
        // A nifti 5D volume is expected
        if (img->dim[0] < 5 || img->dim[4]>1 || img->dim[5] < 2 || img->datatype != NIFTI_TYPE_FLOAT32) {
            reg_print_fct_error("cudaCommon_transferFromDeviceToNifti");
            reg_print_msg_error("The nifti image is not a 5D volume");
            return EXIT_FAILURE;
        }

        float4 *array_h;
        const size_t voxelNumber = CalcVoxelNumber(*img);
        NR_CUDA_SAFE_CALL(cudaMallocHost(&array_h, voxelNumber * sizeof(float4)));
        NR_CUDA_SAFE_CALL(cudaMemcpy((void*)array_h, (const void*)array_d, voxelNumber * sizeof(float4), cudaMemcpyDeviceToHost));
        float *niftiImgValues = static_cast<float*>(img->data);

        for (size_t i = 0; i < voxelNumber; i++)
            *niftiImgValues++ = array_h[i].x;
        if (img->dim[5] >= 2) {
            for (size_t i = 0; i < voxelNumber; i++)
                *niftiImgValues++ = array_h[i].y;
        }
        if (img->dim[5] >= 3) {
            for (size_t i = 0; i < voxelNumber; i++)
                *niftiImgValues++ = array_h[i].z;
        }
        if (img->dim[5] >= 4) {
            for (size_t i = 0; i < voxelNumber; i++)
                *niftiImgValues++ = array_h[i].w;
        }
        NR_CUDA_SAFE_CALL(cudaFreeHost(array_h));

        return EXIT_SUCCESS;
    } else {
        switch (img->datatype) {
        case NIFTI_TYPE_FLOAT32:
            return cudaCommon_transferFromDeviceToNifti1<DataType, float>(img, array_d);
        default:
            reg_print_fct_error("cudaCommon_transferFromDeviceToNifti");
            reg_print_msg_error("The image data type is not supported");
            return EXIT_FAILURE;
        }
    }
}
template int cudaCommon_transferFromDeviceToNifti<float>(nifti_image*, float*);
template int cudaCommon_transferFromDeviceToNifti<double>(nifti_image*, double*);
template int cudaCommon_transferFromDeviceToNifti<float4>(nifti_image*, float4*); // for deformation field
/* *************************************************************** */
template<>
int cudaCommon_transferFromDeviceToNifti(nifti_image *img, cudaArray *cuArray_d) {
    if (img->datatype != NIFTI_TYPE_FLOAT32) {
        reg_print_fct_error("cudaCommon_transferFromDeviceToNifti");
        reg_print_msg_error("The image data type is not supported");
        return EXIT_FAILURE;
    }

    cudaMemcpy3DParms copyParams = {0};
    copyParams.extent = make_cudaExtent(img->dim[1], img->dim[2], img->dim[3]);
    copyParams.srcArray = cuArray_d;
    copyParams.dstPtr = make_cudaPitchedPtr((void*)(img->data), copyParams.extent.width * sizeof(float),
                                            copyParams.extent.width, copyParams.extent.height);
    copyParams.kind = cudaMemcpyDeviceToHost;
    NR_CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
    return EXIT_SUCCESS;
}
/* *************************************************************** */
template <class DataType, class NiftiType>
int cudaCommon_transferFromDeviceToNifti1(nifti_image *img, DataType *array_d, DataType *array2_d) {
    if (sizeof(DataType) != sizeof(NiftiType)) {
        reg_print_fct_error("cudaCommon_transferFromDeviceToNifti1");
        reg_print_msg_error("The host and device arrays are of different types");
        return EXIT_FAILURE;
    } else {
        const size_t voxelNumber = CalcVoxelNumber(*img);
        NiftiType *array_h = static_cast<NiftiType*>(img->data);
        NiftiType *array2_h = &array_h[voxelNumber];
        NR_CUDA_SAFE_CALL(cudaMemcpy((void*)array_h, (void*)array_d, voxelNumber * sizeof(DataType), cudaMemcpyDeviceToHost));
        NR_CUDA_SAFE_CALL(cudaMemcpy((void*)array2_h, (void*)array2_d, voxelNumber * sizeof(DataType), cudaMemcpyDeviceToHost));
    }
    return EXIT_SUCCESS;
}
/* *************************************************************** */
template <class DataType>
int cudaCommon_transferFromDeviceToNifti(nifti_image *img, DataType *array_d, DataType *array2_d) {
    if (sizeof(DataType) == sizeof(float4)) {
        // A nifti 5D volume is expected
        if (img->dim[0] < 5 || img->dim[4]>1 || img->dim[5] < 2 || img->datatype != NIFTI_TYPE_FLOAT32) {
            reg_print_fct_error("cudaCommon_transferFromDeviceToNifti");
            reg_print_msg_error("The nifti image is not a 5D volume");
            return EXIT_FAILURE;
        }
        const size_t voxelNumber = CalcVoxelNumber(*img);
        float4 *array_h = nullptr;
        float4 *array2_h = nullptr;
        NR_CUDA_SAFE_CALL(cudaMallocHost(&array_h, voxelNumber * sizeof(float4)));
        NR_CUDA_SAFE_CALL(cudaMallocHost(&array2_h, voxelNumber * sizeof(float4)));
        NR_CUDA_SAFE_CALL(cudaMemcpy((void*)array_h, (const void*)array_d, voxelNumber * sizeof(float4), cudaMemcpyDeviceToHost));
        NR_CUDA_SAFE_CALL(cudaMemcpy((void*)array2_h, (const void*)array2_d, voxelNumber * sizeof(float4), cudaMemcpyDeviceToHost));
        float *niftiImgValues = static_cast<float *>(img->data);
        for (size_t i = 0; i < voxelNumber; i++) {
            *niftiImgValues++ = array_h[i].x;
        }
        for (size_t i = 0; i < voxelNumber; i++) {
            *niftiImgValues++ = array2_h[i].x;
        }
        if (img->dim[5] >= 2) {
            for (size_t i = 0; i < voxelNumber; i++) {
                *niftiImgValues++ = array_h[i].y;
            }
            for (size_t i = 0; i < voxelNumber; i++) {
                *niftiImgValues++ = array2_h[i].y;
            }
        }
        if (img->dim[5] >= 3) {
            for (size_t i = 0; i < voxelNumber; i++) {
                *niftiImgValues++ = array_h[i].z;
            }
            for (size_t i = 0; i < voxelNumber; i++) {
                *niftiImgValues++ = array2_h[i].z;
            }
        }
        if (img->dim[5] >= 4) {
            for (size_t i = 0; i < voxelNumber; i++) {
                *niftiImgValues++ = array_h[i].w;
            }
            for (size_t i = 0; i < voxelNumber; i++) {
                *niftiImgValues++ = array2_h[i].w;
            }
        }
        NR_CUDA_SAFE_CALL(cudaFreeHost(array_h));
        NR_CUDA_SAFE_CALL(cudaFreeHost(array2_h));

        return EXIT_SUCCESS;
    } else {
        switch (img->datatype) {
        case NIFTI_TYPE_FLOAT32:
            return cudaCommon_transferFromDeviceToNifti1<DataType, float>(img, array_d, array2_d);
        default:
            reg_print_fct_error("cudaCommon_transferFromDeviceToNifti");
            reg_print_msg_error("The image data type is not supported");
            return EXIT_FAILURE;
        }
    }
}
template int cudaCommon_transferFromDeviceToNifti<float>(nifti_image*, float*, float*);
template int cudaCommon_transferFromDeviceToNifti<double>(nifti_image*, double*, double*);
template int cudaCommon_transferFromDeviceToNifti<float4>(nifti_image*, float4*, float4*); // for deformation field
/* *************************************************************** */
void cudaCommon_free(cudaArray *cuArray_d) {
    NR_CUDA_SAFE_CALL(cudaFreeArray(cuArray_d));
}
/* *************************************************************** */
template <class DataType>
void cudaCommon_free(DataType *array_d) {
    if (array_d != nullptr)
        NR_CUDA_SAFE_CALL(cudaFree(array_d));
}
template void cudaCommon_free<int>(int*);
template void cudaCommon_free<float>(float*);
template void cudaCommon_free<double>(double*);
template void cudaCommon_free<float4>(float4*);
/* *************************************************************** */
template <class DataType>
int cudaCommon_transferFromDeviceToNiftiSimple(DataType *array_d, nifti_image *img) {
    NR_CUDA_SAFE_CALL(cudaMemcpy(array_d, img->data, img->nvox * sizeof(DataType), cudaMemcpyHostToDevice));
    return EXIT_SUCCESS;
}
template int cudaCommon_transferFromDeviceToNiftiSimple<int>(int*, nifti_image*);
template int cudaCommon_transferFromDeviceToNiftiSimple<float>(float*, nifti_image*);
template int cudaCommon_transferFromDeviceToNiftiSimple<double>(double*, nifti_image*);
/* *************************************************************** */
template <class DataType>
int cudaCommon_transferFromDeviceToNiftiSimple1(DataType *array_d, DataType *img, const unsigned nvox) {
    NR_CUDA_SAFE_CALL(cudaMemcpy(array_d, img, nvox * sizeof(DataType), cudaMemcpyHostToDevice));
    return EXIT_SUCCESS;
}
template int cudaCommon_transferFromDeviceToNiftiSimple1<int>(int*, int*, const unsigned);
template int cudaCommon_transferFromDeviceToNiftiSimple1<float>(float*, float*, const unsigned);
template int cudaCommon_transferFromDeviceToNiftiSimple1<double>(double*, double*, const unsigned);
/* *************************************************************** */
template <class DataType>
int cudaCommon_transferArrayFromCpuToDevice(DataType *array_d, DataType *array_cpu, const unsigned nElements) {
    const unsigned memSize = nElements * sizeof(DataType);
    NR_CUDA_SAFE_CALL(cudaMemcpy(array_d, array_cpu, memSize, cudaMemcpyHostToDevice));
    return EXIT_SUCCESS;
}
template int cudaCommon_transferArrayFromCpuToDevice<int>(int*, int*, const unsigned);
template int cudaCommon_transferArrayFromCpuToDevice<float>(float*, float*, const unsigned);
template int cudaCommon_transferArrayFromCpuToDevice<double>(double*, double*, const unsigned);
/* *************************************************************** */
template <class DataType>
int cudaCommon_transferArrayFromDeviceToCpu(DataType *array_cpu, DataType *array_d, const unsigned nElements) {
    const unsigned memSize = nElements * sizeof(DataType);
    NR_CUDA_SAFE_CALL(cudaMemcpy(array_cpu, array_d, memSize, cudaMemcpyDeviceToHost));
    return EXIT_SUCCESS;
}
template int cudaCommon_transferArrayFromDeviceToCpu<int>(int*, int*, const unsigned);
template int cudaCommon_transferArrayFromDeviceToCpu<float>(float*, float*, const unsigned);
template int cudaCommon_transferArrayFromDeviceToCpu<double>(double*, double*, const unsigned);
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
