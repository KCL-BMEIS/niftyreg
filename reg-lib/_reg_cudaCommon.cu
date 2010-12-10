/*
 *  _reg_cudaCommon.cpp
 *  
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_CUDACOMMON_CPP
#define _REG_CUDACOMMON_CPP

#include "_reg_cudaCommon.h"

/* ******************************** */
/* ******************************** */
template <class DTYPE, class NIFTI_TYPE>
int cudaCommon_transferNiftiToArrayOnDevice1(DTYPE **array_d, nifti_image *img)
{
    if(sizeof(DTYPE)!=sizeof(NIFTI_TYPE)){
        fprintf(stderr, "ERROR:\tcudaCommon_transferNiftiToArrayOnDevice:\n");
        fprintf(stderr, "ERROR:\tThe host and device arrays are of different types.\n");
        return 1;
    }
    else{
        const unsigned int memSize = img->dim[1] * img->dim[2] * img->dim[3] * sizeof(DTYPE);
        NIFTI_TYPE *array_h=static_cast<NIFTI_TYPE *>(img->data);
        CUDA_SAFE_CALL(cudaMemcpy(*array_d, array_h, memSize, cudaMemcpyHostToDevice));
    }
    return 0;
}
/* ******************************** */
template <class DTYPE>
int cudaCommon_transferNiftiToArrayOnDevice(DTYPE **array_d, nifti_image *img)
{
    if( sizeof(DTYPE)==sizeof(float4) ){
        if( (img->datatype!=NIFTI_TYPE_FLOAT32) || (img->dim[5]<2) || (img->dim[4]>1)){
            fprintf(stderr, "ERROR:\tcudaCommon_transferNiftiToDevice:\n");
            fprintf(stderr, "ERROR:\tThe specified image is not a single precision deformation field image\n");
            return 1;
        }
        float *niftiImgValues = static_cast<float *>(img->data);
        float4 *array_h=(float4 *)calloc(img->nx*img->ny*img->nz,sizeof(float4));
        const int voxelNumber = img->nx*img->ny*img->nz;
        for(int i=0; i<voxelNumber; i++)
            array_h[i].x= *niftiImgValues++;
        if(img->dim[5]>=2){
            for(int i=0; i<voxelNumber; i++)
                array_h[i].y= *niftiImgValues++;
        }
        if(img->dim[5]>=3){
            for(int i=0; i<voxelNumber; i++)
                array_h[i].z= *niftiImgValues++;
        }
        if(img->dim[5]>=4){
            for(int i=0; i<voxelNumber; i++)
                array_h[i].w= *niftiImgValues++;
        }
        CUDA_SAFE_CALL(cudaMemcpy(*array_d, array_h, img->nx*img->ny*img->nz*sizeof(float4), cudaMemcpyHostToDevice));
        free(array_h);
    }
    else{ // All these else could be removed but the nvcc compiler would warn for unreachable statement
        switch(img->datatype){
#ifdef _NR_DEV
            case NIFTI_TYPE_UINT8:
                return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,unsigned char>(array_d, img);
            case NIFTI_TYPE_INT8:
                return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,char>(array_d, img);
            case NIFTI_TYPE_UINT16:
                return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,unsigned short>(array_d, img);
            case NIFTI_TYPE_INT16:
                return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,short>(array_d, img);
            case NIFTI_TYPE_UINT32:
                return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,unsigned int>(array_d, img);
            case NIFTI_TYPE_INT32:
                return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,int>(array_d, img);
            case NIFTI_TYPE_FLOAT64:
                return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,double>(array_d, img);
#endif
            case NIFTI_TYPE_FLOAT32:
                return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,float>(array_d, img);
            default:
                fprintf(stderr, "ERROR:\tcudaCommon_transferNiftiToArrayOnDevice:\n");
                fprintf(stderr, "ERROR:\tThe image data type is not supported\n");
                return 1;
        }
    }
    return 0;
}
#ifdef _NR_DEV
template int cudaCommon_transferNiftiToArrayOnDevice<char>(char **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<unsigned char>(unsigned char **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<short>(short **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<unsigned short>(unsigned short **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<int>(int **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<unsigned int>(unsigned int **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<double>(double **, nifti_image *);
#endif
template int cudaCommon_transferNiftiToArrayOnDevice<float>(float **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<float4>(float4 **, nifti_image *);
/* ******************************** */

template <class DTYPE, class NIFTI_TYPE>
int cudaCommon_transferNiftiToArrayOnDevice1(DTYPE **array_d, DTYPE **array2_d, nifti_image *img)
{
    if(sizeof(DTYPE)!=sizeof(NIFTI_TYPE)){
        fprintf(stderr, "ERROR:\tcudaCommon_transferNiftiToArrayOnDevice:\n");
        fprintf(stderr, "ERROR:\tThe host and device arrays are of different types.\n");
        return 1;
    }
    else{
        const unsigned int memSize = img->dim[1] * img->dim[2] * img->dim[3] * sizeof(DTYPE);
        NIFTI_TYPE *array_h=static_cast<NIFTI_TYPE *>(img->data);
        NIFTI_TYPE *array2_h=&array_h[img->dim[1] * img->dim[2] * img->dim[3]];
        CUDA_SAFE_CALL(cudaMemcpy(*array_d, array_h, memSize, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(*array2_d, array2_h, memSize, cudaMemcpyHostToDevice));
    }
    return 0;
}
/* ******************************** */
template <class DTYPE>
int cudaCommon_transferNiftiToArrayOnDevice(DTYPE **array_d, DTYPE **array2_d, nifti_image *img)
{
    if( sizeof(DTYPE)==sizeof(float4) ){
        if( (img->datatype!=NIFTI_TYPE_FLOAT32) || (img->dim[5]<2) || (img->dim[4]>1)){
            fprintf(stderr, "ERROR:\tcudaCommon_transferNiftiToDevice:\n");
            fprintf(stderr, "ERROR:\tThe specified image is not a single precision deformation field image\n");
            return 1;
        }
        float *niftiImgValues = static_cast<float *>(img->data);
        float4 *array_h=(float4 *)calloc(img->nx*img->ny*img->nz,sizeof(float4));
        float4 *array2_h=(float4 *)calloc(img->nx*img->ny*img->nz,sizeof(float4));
        const int voxelNumber = img->nx*img->ny*img->nz;
        for(int i=0; i<voxelNumber; i++)
            array_h[i].x= *niftiImgValues++;
        for(int i=0; i<voxelNumber; i++)
            array2_h[i].x= *niftiImgValues++;
        if(img->dim[5]>=2){
            for(int i=0; i<voxelNumber; i++)
                array_h[i].y= *niftiImgValues++;
            for(int i=0; i<voxelNumber; i++)
                array2_h[i].y= *niftiImgValues++;
        }
        if(img->dim[5]>=3){
            for(int i=0; i<voxelNumber; i++)
                array_h[i].z= *niftiImgValues++;
            for(int i=0; i<voxelNumber; i++)
                array2_h[i].z= *niftiImgValues++;
        }
        if(img->dim[5]>=4){
            for(int i=0; i<voxelNumber; i++)
                array_h[i].w= *niftiImgValues++;
            for(int i=0; i<voxelNumber; i++)
                array2_h[i].w= *niftiImgValues++;
        }
        CUDA_SAFE_CALL(cudaMemcpy(*array_d, array_h, img->nx*img->ny*img->nz*sizeof(float4), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(*array2_d, array2_h, img->nx*img->ny*img->nz*sizeof(float4), cudaMemcpyHostToDevice));
        free(array_h);
        free(array2_h);
    }
    else{ // All these else could be removed but the nvcc compiler would warn for unreachable statement
        switch(img->datatype){
#ifdef _NR_DEV
            case NIFTI_TYPE_UINT8:
                return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,unsigned char>(array_d, array2_d, img);
            case NIFTI_TYPE_INT8:
                return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,char>(array_d, array2_d, img);
            case NIFTI_TYPE_UINT16:
                return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,unsigned short>(array_d, array2_d, img);
            case NIFTI_TYPE_INT16:
                return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,short>(array_d, array2_d, img);
            case NIFTI_TYPE_UINT32:
                return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,unsigned int>(array_d, array2_d, img);
            case NIFTI_TYPE_INT32:
                return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,int>(array_d, array2_d, img);
            case NIFTI_TYPE_FLOAT64:
                return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,double>(array_d, array2_d, img);
#endif
            case NIFTI_TYPE_FLOAT32:
                return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,float>(array_d, array2_d, img);
            default:
                fprintf(stderr, "ERROR:\tcudaCommon_transferNiftiToArrayOnDevice:\n");
                fprintf(stderr, "ERROR:\tThe image data type is not supported\n");
                return 1;
        }
    }
    return 0;
}
#ifdef _NR_DEV
template int cudaCommon_transferNiftiToArrayOnDevice<char>(char **,char **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<unsigned char>(unsigned char **,unsigned char **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<short>(short **,short **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<unsigned short>(unsigned short **,unsigned short **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<int>(int **,int **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<unsigned int>(unsigned int **,unsigned int **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<double>(double **,double **, nifti_image *);
#endif
template int cudaCommon_transferNiftiToArrayOnDevice<float>(float **,float **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<float4>(float4 **,float4 **, nifti_image *); // for deformation field
/* ******************************** */
/* ******************************** */
template <class DTYPE, class NIFTI_TYPE>
int cudaCommon_transferNiftiToArrayOnDevice1(cudaArray **cuArray_d, nifti_image *img)
{
    if(sizeof(DTYPE)!=sizeof(NIFTI_TYPE)){
        fprintf(stderr, "ERROR:\tcudaCommon_transferNiftiToArrayOnDevice:\n");
        fprintf(stderr, "ERROR:\tThe host and device arrays are of different types.\n");
        return 1;
    }
    else{
        NIFTI_TYPE *array_h=static_cast<NIFTI_TYPE *>(img->data);

        cudaMemcpy3DParms copyParams = {0};
        copyParams.extent = make_cudaExtent(img->dim[1], img->dim[2], img->dim[3]);
        copyParams.srcPtr = make_cudaPitchedPtr((void *) array_h,
                                                copyParams.extent.width*sizeof(DTYPE),
                                                copyParams.extent.width,
                                                copyParams.extent.height);
        copyParams.dstArray = *cuArray_d;
        copyParams.kind = cudaMemcpyHostToDevice;
        CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
    }
    return 0;
}
/* ******************************** */
template <class DTYPE>
int cudaCommon_transferNiftiToArrayOnDevice(cudaArray **cuArray_d, nifti_image *img)
{
    if( sizeof(DTYPE)==sizeof(float4) ){
        if( (img->datatype!=NIFTI_TYPE_FLOAT32) || (img->dim[5]<2) || (img->dim[4]>1) ){
            fprintf(stderr, "ERROR:\tcudaCommon_transferNiftiToDevice:\n");
            fprintf(stderr, "ERROR:\tThe specified image is not a single precision deformation field image\n");
            return 1;
        }
        float *niftiImgValues = static_cast<float *>(img->data);
        float4 *array_h=(float4 *)calloc(img->nx*img->ny*img->nz,sizeof(float4));

        for(int i=0; i<img->nx*img->ny*img->nz; i++)
            array_h[i].x= *niftiImgValues++;

        if(img->dim[5]>=2){
            for(int i=0; i<img->nx*img->ny*img->nz; i++)
                array_h[i].y= *niftiImgValues++;
        }

        if(img->dim[5]>=3){
            for(int i=0; i<img->nx*img->ny*img->nz; i++)
                array_h[i].z= *niftiImgValues++;
        }

        if(img->dim[5]==3){
            for(int i=0; i<img->nx*img->ny*img->nz; i++)
                array_h[i].w= *niftiImgValues++;
        }
        cudaMemcpy3DParms copyParams = {0};
        copyParams.extent = make_cudaExtent(img->dim[1], img->dim[2], img->dim[3]);
        copyParams.srcPtr = make_cudaPitchedPtr((void *) array_h,
                                                copyParams.extent.width*sizeof(DTYPE),
                                                copyParams.extent.width,
                                                copyParams.extent.height);
        copyParams.dstArray = *cuArray_d;
        copyParams.kind = cudaMemcpyHostToDevice;
        CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
        free(array_h);
    }
    else{ // All these else could be removed but the nvcc compiler would warn for unreachable statement
        switch(img->datatype){
#ifdef _NR_DEV
            case NIFTI_TYPE_UINT8:
                return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,unsigned char>(cuArray_d, img);
            case NIFTI_TYPE_INT8:
                return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,char>(cuArray_d, img);
            case NIFTI_TYPE_UINT16:
                return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,unsigned short>(cuArray_d, img);
            case NIFTI_TYPE_INT16:
                return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,short>(cuArray_d, img);
            case NIFTI_TYPE_UINT32:
                return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,unsigned int>(cuArray_d, img);
            case NIFTI_TYPE_INT32:
                return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,int>(cuArray_d, img);
            case NIFTI_TYPE_FLOAT64:
                return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,double>(cuArray_d, img);
#endif
            case NIFTI_TYPE_FLOAT32:
                return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,float>(cuArray_d, img);
            default:
                fprintf(stderr, "ERROR:\tcudaCommon_transferNiftiToArrayOnDevice:\n");
                fprintf(stderr, "ERROR:\tThe image data type is not supported\n");
                return 1;
        }
    }
    return 0;
}
#ifdef _NR_DEV
template int cudaCommon_transferNiftiToArrayOnDevice<char>(cudaArray **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<unsigned char>(cudaArray **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<short>(cudaArray **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<unsigned short>(cudaArray **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<int>(cudaArray **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<unsigned int>(cudaArray **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<double>(cudaArray **, nifti_image *);
#endif
template int cudaCommon_transferNiftiToArrayOnDevice<float>(cudaArray **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<float4>(cudaArray **, nifti_image *); // for deformation field
/* ******************************** */
/* ******************************** */
template <class DTYPE, class NIFTI_TYPE>
int cudaCommon_transferNiftiToArrayOnDevice1(cudaArray **cuArray_d, cudaArray **cuArray2_d, nifti_image *img)
{
    if(sizeof(DTYPE)!=sizeof(NIFTI_TYPE)){
        fprintf(stderr, "ERROR:\tcudaCommon_transferNiftiToArrayOnDevice:\n");
        fprintf(stderr, "ERROR:\tThe host and device arrays are of different types.\n");
        return 1;
    }
    else{
        NIFTI_TYPE *array_h = static_cast<NIFTI_TYPE *>(img->data);
        NIFTI_TYPE *array2_h = &array_h[img->dim[1]*img->dim[2]*img->dim[3]];

        cudaMemcpy3DParms copyParams = {0};
        copyParams.extent = make_cudaExtent(img->dim[1], img->dim[2], img->dim[3]);
        copyParams.kind = cudaMemcpyHostToDevice;
        // First timepoint
        copyParams.srcPtr = make_cudaPitchedPtr((void *) array_h,
                                                copyParams.extent.width*sizeof(DTYPE),
                                                copyParams.extent.width,
                                                copyParams.extent.height);
        copyParams.dstArray = *cuArray_d;
        CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
        // Second timepoint
        copyParams.srcPtr = make_cudaPitchedPtr((void *) array2_h,
                                                copyParams.extent.width*sizeof(DTYPE),
                                                copyParams.extent.width,
                                                copyParams.extent.height);
        copyParams.dstArray = *cuArray2_d;
        CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
    }
    return 0;
}
/* ******************************** */
template <class DTYPE>
int cudaCommon_transferNiftiToArrayOnDevice(cudaArray **cuArray_d, cudaArray **cuArray2_d, nifti_image *img)
{
    if( sizeof(DTYPE)==sizeof(float4) ){
        if( (img->datatype!=NIFTI_TYPE_FLOAT32) || (img->dim[5]<2) || (img->dim[4]>1) ){
            fprintf(stderr, "ERROR:\tcudaCommon_transferNiftiToDevice:\n");
            fprintf(stderr, "ERROR:\tThe specified image is not a single precision deformation field image\n");
            return 1;
        }
        float *niftiImgValues = static_cast<float *>(img->data);
        float4 *array_h=(float4 *)calloc(img->nx*img->ny*img->nz,sizeof(float4));
        float4 *array2_h=(float4 *)calloc(img->nx*img->ny*img->nz,sizeof(float4));

        for(int i=0; i<img->nx*img->ny*img->nz; i++)
            array_h[i].x= *niftiImgValues++;
        for(int i=0; i<img->nx*img->ny*img->nz; i++)
            array2_h[i].x= *niftiImgValues++;

        if(img->dim[5]>=2){
            for(int i=0; i<img->nx*img->ny*img->nz; i++)
                array_h[i].y= *niftiImgValues++;
            for(int i=0; i<img->nx*img->ny*img->nz; i++)
                array2_h[i].y= *niftiImgValues++;
        }

        if(img->dim[5]>=3){
            for(int i=0; i<img->nx*img->ny*img->nz; i++)
                array_h[i].z= *niftiImgValues++;
            for(int i=0; i<img->nx*img->ny*img->nz; i++)
                array2_h[i].z= *niftiImgValues++;
        }

        if(img->dim[5]==3){
            for(int i=0; i<img->nx*img->ny*img->nz; i++)
                array_h[i].w= *niftiImgValues++;
            for(int i=0; i<img->nx*img->ny*img->nz; i++)
                array2_h[i].w= *niftiImgValues++;
        }

        cudaMemcpy3DParms copyParams = {0};
        copyParams.extent = make_cudaExtent(img->dim[1], img->dim[2], img->dim[3]);
        copyParams.kind = cudaMemcpyHostToDevice;
        // First timepoint
        copyParams.srcPtr = make_cudaPitchedPtr((void *) array_h,
                                                copyParams.extent.width*sizeof(DTYPE),
                                                copyParams.extent.width,
                                                copyParams.extent.height);
        copyParams.dstArray = *cuArray_d;
        CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
        free(array_h);
        // Second timepoint
        copyParams.srcPtr = make_cudaPitchedPtr((void *) array2_h,
                                                copyParams.extent.width*sizeof(DTYPE),
                                                copyParams.extent.width,
                                                copyParams.extent.height);
        copyParams.dstArray = *cuArray2_d;
        CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
        free(array2_h);
    }
    else{ // All these else could be removed but the nvcc compiler would warn for unreachable statement
        switch(img->datatype){
#ifdef _NR_DEV
            case NIFTI_TYPE_UINT8:
                return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,unsigned char>(cuArray_d, cuArray2_d, img);
            case NIFTI_TYPE_INT8:
                return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,char>(cuArray_d, cuArray2_d, img);
            case NIFTI_TYPE_UINT16:
                return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,unsigned short>(cuArray_d, cuArray2_d, img);
            case NIFTI_TYPE_INT16:
                return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,short>(cuArray_d, cuArray2_d, img);
            case NIFTI_TYPE_UINT32:
                return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,unsigned int>(cuArray_d, cuArray2_d, img);
            case NIFTI_TYPE_INT32:
                return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,int>(cuArray_d, cuArray2_d, img);
            case NIFTI_TYPE_FLOAT64:
                return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,double>(cuArray_d, cuArray2_d, img);
#endif
            case NIFTI_TYPE_FLOAT32:
                return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,float>(cuArray_d, cuArray2_d, img);
            default:
                fprintf(stderr, "ERROR:\tcudaCommon_transferNiftiToArrayOnDevice:\n");
                fprintf(stderr, "ERROR:\tThe image data type is not supported\n");
                return 1;
        }
    }
    return 0;
}
#ifdef _NR_DEV
template int cudaCommon_transferNiftiToArrayOnDevice<char>(cudaArray **, cudaArray **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<unsigned char>(cudaArray **, cudaArray **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<short>(cudaArray **, cudaArray **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<unsigned short>(cudaArray **, cudaArray **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<int>(cudaArray **, cudaArray **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<unsigned int>(cudaArray **, cudaArray **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<double>(cudaArray **, cudaArray **, nifti_image *);
#endif
template int cudaCommon_transferNiftiToArrayOnDevice<float>(cudaArray **, cudaArray **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<float4>(cudaArray **, cudaArray **, nifti_image *); // for deformation field
/* ******************************** */
/* ******************************** */
template <class DTYPE>
int cudaCommon_allocateArrayToDevice(cudaArray **cuArray_d, int *dim)
{
    const cudaExtent volumeSize = make_cudaExtent(dim[1], dim[2], dim[3]);
    cudaChannelFormatDesc texDesc = cudaCreateChannelDesc<DTYPE>();
    CUDA_SAFE_CALL(cudaMalloc3DArray(cuArray_d, &texDesc, volumeSize));
    return 0;
}
#ifdef _NR_DEV
template int cudaCommon_allocateArrayToDevice<char>(cudaArray **, int *);
template int cudaCommon_allocateArrayToDevice<unsigned char>(cudaArray **, int *);
template int cudaCommon_allocateArrayToDevice<short>(cudaArray **, int *);
template int cudaCommon_allocateArrayToDevice<unsigned short>(cudaArray **, int *);
template int cudaCommon_allocateArrayToDevice<int>(cudaArray **, int *);
template int cudaCommon_allocateArrayToDevice<unsigned int>(cudaArray **, int *);
template int cudaCommon_allocateArrayToDevice<double>(cudaArray **, int *);
#endif
template int cudaCommon_allocateArrayToDevice<float>(cudaArray **, int *);
template int cudaCommon_allocateArrayToDevice<float4>(cudaArray **, int *); // for deformation field
/* ******************************** */
/* ******************************** */
template <class DTYPE>
int cudaCommon_allocateArrayToDevice(cudaArray **cuArray_d, cudaArray **cuArray2_d, int *dim)
{
    const cudaExtent volumeSize = make_cudaExtent(dim[1], dim[2], dim[3]);
    cudaChannelFormatDesc texDesc = cudaCreateChannelDesc<DTYPE>();
    CUDA_SAFE_CALL(cudaMalloc3DArray(cuArray_d, &texDesc, volumeSize));
    CUDA_SAFE_CALL(cudaMalloc3DArray(cuArray2_d, &texDesc, volumeSize));
    return 0;
}
#ifdef _NR_DEV
template int cudaCommon_allocateArrayToDevice<char>(cudaArray **,cudaArray **, int *);
template int cudaCommon_allocateArrayToDevice<unsigned char>(cudaArray **,cudaArray **, int *);
template int cudaCommon_allocateArrayToDevice<short>(cudaArray **,cudaArray **, int *);
template int cudaCommon_allocateArrayToDevice<unsigned short>(cudaArray **,cudaArray **, int *);
template int cudaCommon_allocateArrayToDevice<int>(cudaArray **,cudaArray **, int *);
template int cudaCommon_allocateArrayToDevice<unsigned int>(cudaArray **,cudaArray **, int *);
template int cudaCommon_allocateArrayToDevice<double>(cudaArray **,cudaArray **, int *);
#endif
template int cudaCommon_allocateArrayToDevice<float>(cudaArray **,cudaArray **, int *);
template int cudaCommon_allocateArrayToDevice<float4>(cudaArray **,cudaArray **, int *); // for deformation field
/* ******************************** */
/* ******************************** */
template <class DTYPE>
int cudaCommon_allocateArrayToDevice(DTYPE **array_d, int *dim)
{
    const unsigned int memSize = dim[1] * dim[2] * dim[3] * sizeof(DTYPE);
    CUDA_SAFE_CALL(cudaMalloc(array_d, memSize));
    return 0;
}
#ifdef _NR_DEV
template int cudaCommon_allocateArrayToDevice<char>(char **, int *);
template int cudaCommon_allocateArrayToDevice<unsigned char>(unsigned char **, int *);
template int cudaCommon_allocateArrayToDevice<short>(short **, int *);
template int cudaCommon_allocateArrayToDevice<unsigned short>(unsigned short **, int *);
template int cudaCommon_allocateArrayToDevice<int>(int **, int *);
template int cudaCommon_allocateArrayToDevice<unsigned int>(unsigned int **, int *);
template int cudaCommon_allocateArrayToDevice<double>(double **, int *);
#endif
template int cudaCommon_allocateArrayToDevice<float>(float **, int *);
template int cudaCommon_allocateArrayToDevice<float4>(float4 **, int *); // for deformation field
/* ******************************** */
/* ******************************** */
template <class DTYPE>
int cudaCommon_allocateArrayToDevice(DTYPE **array_d, DTYPE **array2_d, int *dim)
{
    const unsigned int memSize = dim[1] * dim[2] * dim[3] * sizeof(DTYPE);
    CUDA_SAFE_CALL(cudaMalloc(array_d, memSize));
    CUDA_SAFE_CALL(cudaMalloc(array2_d, memSize));
    return 0;
}
#ifdef _NR_DEV
template int cudaCommon_allocateArrayToDevice<char>(char **, char **, int *);
template int cudaCommon_allocateArrayToDevice<unsigned char>(unsigned char **, unsigned char **, int *);
template int cudaCommon_allocateArrayToDevice<short>(short **, short **, int *);
template int cudaCommon_allocateArrayToDevice<unsigned short>(unsigned short **, unsigned short **, int *);
template int cudaCommon_allocateArrayToDevice<int>(int **, int **, int *);
template int cudaCommon_allocateArrayToDevice<unsigned int>(unsigned int **, unsigned int **, int *);
template int cudaCommon_allocateArrayToDevice<double>(double **, double **, int *);
#endif
template int cudaCommon_allocateArrayToDevice<float>(float **, float **, int *);
template int  cudaCommon_allocateArrayToDevice<float4>(float4 **, float4 **, int *); // for deformation field
/* ******************************** */
/* ******************************** */
template <class DTYPE, class NIFTI_TYPE>
int cudaCommon_transferFromDeviceToNifti1(nifti_image *img, DTYPE **array_d)
{
	if(sizeof(DTYPE)!=sizeof(NIFTI_TYPE)){
		fprintf(stderr, "ERROR:\tcudaCommon_transferFromDeviceToNifti:\n");
		fprintf(stderr, "ERROR:\tThe host and device arrays are of different types.\n");
		return 1;
	}
	else{
		NIFTI_TYPE *array_h=static_cast<NIFTI_TYPE *>(img->data);
        CUDA_SAFE_CALL(cudaMemcpy((void *)array_h, (void *)*array_d, img->nvox*sizeof(DTYPE), cudaMemcpyDeviceToHost));
	}
	return 0;
}
/* ******************************** */
template <class DTYPE>
int cudaCommon_transferFromDeviceToNifti(nifti_image *img, DTYPE **array_d)
{
    if(sizeof(DTYPE)==sizeof(float4)){
        // A nifti 5D volume is expected
        if(img->dim[0]<5 || img->dim[4]>1 || img->dim[5]<2 || img->datatype!=NIFTI_TYPE_FLOAT32){
            fprintf(stderr, "ERROR:\tcudaCommon_transferFromDeviceToNifti:\n");
            fprintf(stderr, "ERROR:\tThe nifti image is not a 5D volume.\n");
            return 1;
        }
        const int voxelNumber = img->nx*img->ny*img->nz;
        float4 *array_h;
        CUDA_SAFE_CALL(cudaMallocHost(&array_h, voxelNumber*sizeof(float4)));
        CUDA_SAFE_CALL(cudaMemcpy((void *)array_h, (const void *)*array_d, voxelNumber*sizeof(float4), cudaMemcpyDeviceToHost));
        float *niftiImgValues = static_cast<float *>(img->data);
        for(int i=0; i<voxelNumber; i++)
            *niftiImgValues++ = array_h[i].x;
        if(img->dim[5]>=2){
            for(int i=0; i<voxelNumber; i++)
                *niftiImgValues++ = array_h[i].y;
        }
        if(img->dim[5]>=3){
            for(int i=0; i<voxelNumber; i++)
                *niftiImgValues++ = array_h[i].z;
        }
        if(img->dim[5]>=4){
            for(int i=0; i<voxelNumber; i++)
                *niftiImgValues++ = array_h[i].w;
        }
        CUDA_SAFE_CALL(cudaFreeHost(array_h));

        return 0;
    }
    else{
        switch(img->datatype){
#ifdef _NR_DEV
            case NIFTI_TYPE_UINT8:
                return cudaCommon_transferFromDeviceToNifti1<DTYPE,unsigned char>(img, array_d);
            case NIFTI_TYPE_INT8:
                return cudaCommon_transferFromDeviceToNifti1<DTYPE,char>(img, array_d);
            case NIFTI_TYPE_UINT16:
                return cudaCommon_transferFromDeviceToNifti1<DTYPE,unsigned short>(img, array_d);
            case NIFTI_TYPE_INT16:
                return cudaCommon_transferFromDeviceToNifti1<DTYPE,short>(img, array_d);
            case NIFTI_TYPE_UINT32:
                return cudaCommon_transferFromDeviceToNifti1<DTYPE,unsigned int>(img, array_d);
            case NIFTI_TYPE_INT32:
                return cudaCommon_transferFromDeviceToNifti1<DTYPE,int>(img, array_d);
            case NIFTI_TYPE_FLOAT64:
                return cudaCommon_transferFromDeviceToNifti1<DTYPE,double>(img, array_d);
#endif
            case NIFTI_TYPE_FLOAT32:
                return cudaCommon_transferFromDeviceToNifti1<DTYPE,float>(img, array_d);
            default:
                fprintf(stderr, "ERROR:\tcudaCommon_transferFromDeviceToNifti:\n");
                fprintf(stderr, "ERROR:\tThe image data type is not supported\n");
                return 1;
        }
    }
}
#ifdef _NR_DEV
template int cudaCommon_transferFromDeviceToNifti<char>(nifti_image *, char **);
template int cudaCommon_transferFromDeviceToNifti<unsigned char>(nifti_image *, unsigned char **);
template int cudaCommon_transferFromDeviceToNifti<short>(nifti_image *, short **);
template int cudaCommon_transferFromDeviceToNifti<unsigned short>(nifti_image *, unsigned short **);
template int cudaCommon_transferFromDeviceToNifti<int>(nifti_image *, int **);
template int cudaCommon_transferFromDeviceToNifti<unsigned int>(nifti_image *, unsigned int **);
template int cudaCommon_transferFromDeviceToNifti<double>(nifti_image *, double **);
#endif
template int cudaCommon_transferFromDeviceToNifti<float>(nifti_image *, float **);
template int cudaCommon_transferFromDeviceToNifti<float4>(nifti_image *, float4 **); // for deformation field
/* ******************************** */
/* ******************************** */
template <class DTYPE, class NIFTI_TYPE>
int cudaCommon_transferFromDeviceToNifti1(nifti_image *img, DTYPE **array_d, DTYPE **array2_d)
{
    if(sizeof(DTYPE)!=sizeof(NIFTI_TYPE)){
        fprintf(stderr, "ERROR:\tcudaCommon_transferFromDeviceToNifti:\n");
        fprintf(stderr, "ERROR:\tThe host and device arrays are of different types.\n");
        return 1;
    }
    else{
        unsigned int voxelNumber=img->nx*img->ny*img->nz;
        NIFTI_TYPE *array_h=static_cast<NIFTI_TYPE *>(img->data);
        NIFTI_TYPE *array2_h=&array_h[voxelNumber];
        CUDA_SAFE_CALL(cudaMemcpy((void *)array_h, (void *)*array_d, voxelNumber*sizeof(DTYPE), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy((void *)array2_h, (void *)*array2_d, voxelNumber*sizeof(DTYPE), cudaMemcpyDeviceToHost));
    }
    return 0;
}
/* ******************************** */
template <class DTYPE>
int cudaCommon_transferFromDeviceToNifti(nifti_image *img, DTYPE **array_d, DTYPE **array2_d)
{
    if(sizeof(DTYPE)==sizeof(float4)){
        // A nifti 5D volume is expected
        if(img->dim[0]<5 || img->dim[4]>1 || img->dim[5]<2 || img->datatype!=NIFTI_TYPE_FLOAT32){
            fprintf(stderr, "ERROR:\tcudaCommon_transferFromDeviceToNifti:\n");
            fprintf(stderr, "ERROR:\tThe nifti image is not a 5D volume.\n");
            return 1;
        }
        const int voxelNumber = img->nx*img->ny*img->nz;
        float4 *array_h=NULL;
        float4 *array2_h=NULL;
        CUDA_SAFE_CALL(cudaMallocHost(&array_h, voxelNumber*sizeof(float4)));
        CUDA_SAFE_CALL(cudaMallocHost(&array2_h, voxelNumber*sizeof(float4)));
        CUDA_SAFE_CALL(cudaMemcpy((void *)array_h, (const void *)*array_d, voxelNumber*sizeof(float4), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy((void *)array2_h, (const void *)*array2_d, voxelNumber*sizeof(float4), cudaMemcpyDeviceToHost));
        float *niftiImgValues = static_cast<float *>(img->data);
        for(int i=0; i<voxelNumber; i++){
            *niftiImgValues++ = array_h[i].x;
        }
        for(int i=0; i<voxelNumber; i++){
            *niftiImgValues++ = array2_h[i].x;
        }
        if(img->dim[5]>=2){
            for(int i=0; i<voxelNumber; i++){
                *niftiImgValues++ = array_h[i].y;
            }
            for(int i=0; i<voxelNumber; i++){
                *niftiImgValues++ = array2_h[i].y;
            }
        }
        if(img->dim[5]>=3){
            for(int i=0; i<voxelNumber; i++){
                *niftiImgValues++ = array_h[i].z;
            }
            for(int i=0; i<voxelNumber; i++){
                *niftiImgValues++ = array2_h[i].z;
            }
        }
        if(img->dim[5]>=4){
            for(int i=0; i<voxelNumber; i++){
                *niftiImgValues++ = array_h[i].w;
            }
            for(int i=0; i<voxelNumber; i++){
                *niftiImgValues++ = array2_h[i].w;
            }
        }
        CUDA_SAFE_CALL(cudaFreeHost(array_h));
        CUDA_SAFE_CALL(cudaFreeHost(array2_h));

        return 0;
    }
    else{
        switch(img->datatype){
#ifdef _NR_DEV
            case NIFTI_TYPE_UINT8:
                return cudaCommon_transferFromDeviceToNifti1<DTYPE,unsigned char>(img, array_d, array2_d);
            case NIFTI_TYPE_INT8:
                return cudaCommon_transferFromDeviceToNifti1<DTYPE,char>(img, array_d, array2_d);
            case NIFTI_TYPE_UINT16:
                return cudaCommon_transferFromDeviceToNifti1<DTYPE,unsigned short>(img, array_d, array2_d);
            case NIFTI_TYPE_INT16:
                return cudaCommon_transferFromDeviceToNifti1<DTYPE,short>(img, array_d, array2_d);
            case NIFTI_TYPE_UINT32:
                return cudaCommon_transferFromDeviceToNifti1<DTYPE,unsigned int>(img, array_d, array2_d);
            case NIFTI_TYPE_INT32:
                return cudaCommon_transferFromDeviceToNifti1<DTYPE,int>(img, array_d, array2_d);
            case NIFTI_TYPE_FLOAT64:
                return cudaCommon_transferFromDeviceToNifti1<DTYPE,double>(img, array_d, array2_d);
#endif
            case NIFTI_TYPE_FLOAT32:
                return cudaCommon_transferFromDeviceToNifti1<DTYPE,float>(img, array_d, array2_d);
            default:
                fprintf(stderr, "ERROR:\tcudaCommon_transferFromDeviceToNifti:\n");
                fprintf(stderr, "ERROR:\tThe image data type is not supported\n");
                return 1;
        }
    }
}
#ifdef _NR_DEV
template int cudaCommon_transferFromDeviceToNifti<char>(nifti_image *, char **, char **);
template int cudaCommon_transferFromDeviceToNifti<unsigned char>(nifti_image *, unsigned char **, unsigned char **);
template int cudaCommon_transferFromDeviceToNifti<short>(nifti_image *, short **, short **);
template int cudaCommon_transferFromDeviceToNifti<unsigned short>(nifti_image *, unsigned short **, unsigned short **);
template int cudaCommon_transferFromDeviceToNifti<int>(nifti_image *, int **, int **);
template int cudaCommon_transferFromDeviceToNifti<unsigned int>(nifti_image *, unsigned int **, unsigned int **);
template int cudaCommon_transferFromDeviceToNifti<double>(nifti_image *, double **, double **);
#endif
template int cudaCommon_transferFromDeviceToNifti<float>(nifti_image *, float **, float **);
template int cudaCommon_transferFromDeviceToNifti<float4>(nifti_image *, float4 **, float4 **); // for deformation field
/* ******************************** */
/* ******************************** */
void cudaCommon_free(cudaArray **cuArray_d){
	CUDA_SAFE_CALL(cudaFreeArray(*cuArray_d));
	return;
}
/* ******************************** */
/* ******************************** */
template <class DTYPE>
void cudaCommon_free(DTYPE **array_d){
    CUDA_SAFE_CALL(cudaFree(*array_d));
	return;
}
template void cudaCommon_free<int>(int **);
template void cudaCommon_free<float>(float **);
template void cudaCommon_free<float4>(float4 **);
/* ******************************** */
/* ******************************** */
#endif
