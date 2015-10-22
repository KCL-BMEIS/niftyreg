#include "optimizeKernel.h"

#include "cublas_v2.h"
#include "cusolverDn.h"

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/gather.h>
#include <thrust/sort.h>
#include <cmath>

#include "_reg_maths.h"
#include "_reg_tools.h"
#include "_reg_blockMatching.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))
/* *************************************************************** */
template<class DTYPE>
__device__ __inline__ void reg_mat44_mul_cuda(float* mat, DTYPE const* in, DTYPE *out) {
    out[0] = (DTYPE)((double)mat[0 * 4 + 0] * (double)in[0] + (double)mat[0 * 4 + 1] * (double)in[1] + (double)mat[0 * 4 + 2] * (double)in[2] + (double)mat[0 * 4 + 3]);
    out[1] = (DTYPE)((double)mat[1 * 4 + 0] * (double)in[0] + (double)mat[1 * 4 + 1] * (double)in[1] + (double)mat[1 * 4 + 2] * (double)in[2] + (double)mat[1 * 4 + 3]);
    out[2] = (DTYPE)((double)mat[2 * 4 + 0] * (double)in[0] + (double)mat[2 * 4 + 1] * (double)in[1] + (double)mat[2 * 4 + 2] * (double)in[2] + (double)mat[2 * 4 + 3]);
    return;
}
/* *************************************************************** */
__device__ double getSquareDistance3Dcu(float * first_point3D, float * second_point3D) {
    return sqrt(((double)first_point3D[0] - (double)second_point3D[0]) *
                ((double)first_point3D[0] - (double)second_point3D[0]) +
                ((double)first_point3D[1] - (double)second_point3D[1]) *
                ((double)first_point3D[1] - (double)second_point3D[1]) +
                ((double)first_point3D[2] - (double)second_point3D[2]) *
                ((double)first_point3D[2] - (double)second_point3D[2]));
}
/* *************************************************************** */
void checkCublasStatus(cublasStatus_t status)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        reg_print_fct_error("checkCublasStatus");
        reg_print_msg_error("!!!! CUBLAS  error");
        reg_exit(0);
    }
}
/* *************************************************************** */
void checkCUSOLVERStatus(cusolverStatus_t status, char* msg) {

    if (status != CUSOLVER_STATUS_SUCCESS) {
        if (status == CUSOLVER_STATUS_NOT_INITIALIZED) {
            reg_print_fct_error("the library was not initialized.");
        }
        else if (status == CUSOLVER_STATUS_INTERNAL_ERROR) {
            reg_print_fct_error(" an internal operation failed.");
        }
        reg_exit(0);
    }
}
/* *************************************************************** */
void checkDevInfo(int *devInfo) {
    int * hostDevInfo = (int*)malloc(sizeof(int));
    cudaMemcpy(hostDevInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (hostDevInfo < 0)
        printf("parameter: %d is wrong\n", hostDevInfo);
    if (hostDevInfo > 0)
        printf("%d superdiagonals of an intermediate bidiagonal form B did not converge to zero.\n", hostDevInfo);
    else
        printf(" %d: operation successful\n", hostDevInfo);
    free(hostDevInfo);
}
/* *************************************************************** */
void downloadMat44(mat44 *lastTransformation, float* transform_d) {
    float* tempMat = (float*)malloc(16 * sizeof(float));
    cudaMemcpy(tempMat, transform_d, 16 * sizeof(float), cudaMemcpyDeviceToHost);
    cPtrToMat44(lastTransformation, tempMat);
    free(tempMat);
}
/* *************************************************************** */
void uploadMat44(mat44 lastTransformation, float* transform_d) {
    float* tempMat = (float*)malloc(16 * sizeof(float));
    mat44ToCptr(lastTransformation, tempMat);
    cudaMemcpy(transform_d, tempMat, 16 * sizeof(float), cudaMemcpyHostToDevice);
    free(tempMat);
}
/* *************************************************************** */
//threads: 512 | blocks:numEquations/512
__global__ void transformWarpedPointsKernel(float* transform, float* in, float* out, unsigned int definedBlockNum)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < definedBlockNum) {
        const unsigned int posIdx = 3 * tid;
        in += posIdx;
        out += posIdx;
        reg_mat44_mul_cuda<float>(transform, in, out);
    }
}
/* *************************************************************** */
//blocks: 1 | threads: 12
__global__ void trimAndInvertSingularValuesKernel(float* sigma)
{
    sigma[threadIdx.x] = (sigma[threadIdx.x] < 0.0001) ? 0.0f : (float) ((double) 1.0 / (double) sigma[threadIdx.x]);
}
/* *************************************************************** */
//launched as ldm blocks n threads
__global__ void scaleV(float* V, const unsigned int ldm, const unsigned int n, float*w)
{
    unsigned int k = blockIdx.x;
    unsigned int j = threadIdx.x;
    V[IDX2C(j, k, ldm)] = (float)((double)V[IDX2C(j, k, ldm)] * (double)w[j]);
}
/* *************************************************************** */
//threads: 16 | blocks:1
__global__ void permuteAffineMatrix(float* transform)
{
    __shared__ float buffer[16];
    const unsigned int i = threadIdx.x;

    buffer[i] = transform[i];
    __syncthreads();
    const unsigned int idx33 = (i / 3) * 4 + i % 3;
    const unsigned int idx34 = (i % 3) * 4 + 3;

    if (i < 9) transform[idx33] = buffer[i];
    else if (i < 12)transform[idx34] = buffer[i];
    else transform[i] = buffer[i];

}
/* *************************************************************** */
//threads: 512 | blocks:numEquations/512
__global__ void populateMatrixA(float* A, float *reference, unsigned int numBlocks)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int c = tid * 3;
    //	const unsigned int n = 12;
    const unsigned int lda = numBlocks * 3;

    if (tid < numBlocks) {
        reference += c;
        //IDX2C(i,j,ld)
        A[IDX2C(c, 0, lda)] = reference[0];
        A[IDX2C(c, 1, lda)] = reference[1];
        A[IDX2C(c, 2, lda)] = reference[2];
        A[IDX2C(c, 3, lda)] = A[IDX2C(c, 4, lda)] = A[IDX2C(c, 5, lda)] = A[IDX2C(c, 6, lda)] = A[IDX2C(c, 7, lda)] = A[IDX2C(c, 8, lda)] = A[IDX2C(c, 10, lda)] = A[IDX2C(c, 11, lda)] = 0.0f;
        A[IDX2C(c, 9, lda)] = 1.0f;

        A[IDX2C((c + 1), 3, lda)] = reference[0];
        A[IDX2C((c + 1), 4, lda)] = reference[1];
        A[IDX2C((c + 1), 5, lda)] = reference[2];
        A[IDX2C((c + 1), 0, lda)] = A[IDX2C((c + 1), 1, lda)] = A[IDX2C((c + 1), 2, lda)] = A[IDX2C((c + 1), 6, lda)] = A[IDX2C((c + 1), 7, lda)] = A[IDX2C((c + 1), 8, lda)] = A[IDX2C((c + 1), 9, lda)] = A[IDX2C((c + 1), 11, lda)] = 0.0f;
        A[IDX2C((c + 1), 10, lda)] = 1.0f;

        A[IDX2C((c + 2), 6, lda)] = reference[0];
        A[IDX2C((c + 2), 7, lda)] = reference[1];
        A[IDX2C((c + 2), 8, lda)] = reference[2];
        A[IDX2C((c + 2), 0, lda)] = A[IDX2C((c + 2), 1, lda)] = A[IDX2C((c + 2), 2, lda)] = A[IDX2C((c + 2), 3, lda)] = A[IDX2C((c + 2), 4, lda)] = A[IDX2C((c + 2), 5, lda)] = A[IDX2C((c + 2), 9, lda)] = A[IDX2C((c + 2), 10, lda)] = 0.0f;
        A[IDX2C((c + 2), 11, lda)] = 1.0f;
    }
}
/* *************************************************************** */
//threads: 512 | blocks:numEquations/512
__global__ void populateLengthsKernel(float* lengths, float* warped_d, float* newWarped_d, unsigned int numEquations)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int c = tid * 3;

    if (tid < numEquations) {
        newWarped_d += c;
        warped_d += c;
        lengths[tid] = getSquareDistance3Dcu(warped_d, newWarped_d);
    }

}
/* *************************************************************** */
//launched as 1 block 1 thread
__global__ void outputMatFlat(float* mat, const unsigned int ldm, const unsigned int n, char* msg)
{
    for (int i = 0; i < ldm * n; ++i)
        printf("%f | ", mat[i]);
    printf("\n");
}
/* *************************************************************** */
//launched as 1 block 1 thread
__global__ void outputMat(float* mat, const unsigned int ldm, const unsigned int n, char* msg)
{
    for (int i = 0; i < ldm; ++i) {
        printf("%d ", i);
        for (int j = 0; j < n; ++j) {
            printf("%f ", mat[IDX2C(i, j, ldm)]);
        }
        printf("\n");
    }
    printf("\n");
}
/* *************************************************************** */
/*
* the function computes the SVD of a matrix A
* A = V* x S x U, where V* is a (conjugate) transpose of V
* */
void cusolverSVD(float* A_d, unsigned int m, unsigned int n, float* S_d, float* VT_d, float* U_d) {

    //CAST float* to double*
    /*
    double* Adouble_d;
    double* Sdouble_d;
    double* VTdouble_d;
    double* Udouble_d;

    cudaMalloc((void **) &Adouble_d, m*n*sizeof(double));
    cudaMalloc((void **) &Sdouble_d, xx*sizeof(double));
    cudaMalloc((void **) &VTdouble_d, xx*sizeof(double));
    cudaMalloc((void **) &Udouble_d, xx*sizeof(double));

    cudaMemcpy(b_d, a_d, nBytes, cudaMemcpyDeviceToDevice);
    */

    const int lda = m;
    const int ldu = m;
    const int ldvt = n;

    /*
    * 'A': all m columns of U are returned in array
    * 'S': the first min(m,n) columns of U (the left singular vectors) are returned in the array
    * 'O': the first min(m,n) columns of U (the left singular vectors) are overwritten on the array
    * 'N': no columns of U (no left singular vectors) are computed
    */
    const char jobu = 'A';

    /*
    * 'A': all N rows of V**T are returned in the array
    * 'S': the first min(m,n) rows of V**T (the right singular vectors) are returned in the array
    * 'O': the first min(m,n) rows of V**T (the right singular vectors) are overwritten on the array
    * 'N': no rows of V**T (no right singular vectors) are computed
    */
    const char jobvt = 'A';

    cusolverDnHandle_t gH = NULL;
    int Lwork;
    //device ptrs
    float *Work;
    float *rwork;
    int *devInfo;

    //init cusolver compute SVD and shut down
    checkCUSOLVERStatus(cusolverDnCreate(&gH), "cusolverDnCreate");
    checkCUSOLVERStatus(cusolverDnSgesvd_bufferSize(gH, m, n, &Lwork), "cusolverDnSgesvd_bufferSize");

    cudaMalloc(&Work, Lwork * sizeof(float));
    cudaMalloc(&rwork, Lwork * sizeof(float));
    cudaMalloc(&devInfo, sizeof(int));

    checkCUSOLVERStatus(cusolverDnSgesvd(gH, jobu, jobvt, m, n, A_d, lda, S_d, U_d, ldu, VT_d, ldvt, Work, Lwork, NULL, devInfo), "cusolverDnSgesvd");
    checkCUSOLVERStatus(cusolverDnDestroy(gH), "cusolverDnDestroy");

    //free vars
    cudaFree(devInfo);
    cudaFree(rwork);
    cudaFree(Work);

}
/* *************************************************************** */
/*
* the function computes the Pseudoinverse from the products of the SVD factorisation of A
* R = V x inv(S) x U*
* */
void cublasPseudoInverse(float* transformation, float *R_d, float* warped_d, float *VT_d, float* Sigma_d, float *U_d, const unsigned int m, const unsigned int n) {
    // First we make sure that the really small singular values
    // are set to 0. and compute the inverse by taking the reciprocal of the entries

    trimAndInvertSingularValuesKernel <<<1, n >>>(Sigma_d);	//test 3

    cublasHandle_t handle;

    const float alpha = 1.f;
    const float beta = 0.f;

    const int ldvt = n;//VT's lead dimension
    const int ldu = m;//U's lead dimension
    const int ldr = n;//Pseudoinverse's r lead dimension

    const int rowsVTandR = n;//VT and r's num rows
    const int colsUandR = m;//U and r's num cols
    const int colsVtRowsU = n;//VT's cols and U's rows

    // V x inv(S) in place | We scale eaach row with the corresponding singular value as V is transpose
    scaleV <<<n, n >>>(VT_d, n, n, Sigma_d);

    //Initialize CUBLAS perform ops and shut down
    checkCublasStatus(cublasCreate(&handle));

    //now R = V x inv(S) x U*
    checkCublasStatus(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, rowsVTandR, colsUandR, colsVtRowsU, &alpha, VT_d, ldvt, U_d, ldu, &beta, R_d, ldr));

    //finally M=Rxb, where M is our affine matrix and b a vector containg the warped points
    checkCublasStatus(cublasSgemv(handle, CUBLAS_OP_N, n, m, &alpha, R_d, ldr, warped_d, 1, &beta, transformation, 1));
    checkCublasStatus(cublasDestroy(handle));
    permuteAffineMatrix <<<1, 16 >>>(transformation);
    cudaThreadSynchronize();

}
/* *************************************************************** */
double sortAndReduce(float* lengths_d,
                        float* reference_d,
                        float* warped_d,
                        float* newWarped_d,
                        const unsigned int numBlocks,
                        const unsigned int numToKeep,
                        const unsigned int m) {
    //populateLengthsKernel
    populateLengthsKernel <<< numBlocks, 512 >>>(lengths_d, warped_d, newWarped_d, m / 3);

    // The initial vector with all the input points
    thrust::device_ptr<float> reference_d_ptr(reference_d);
    thrust::device_vector<float> vecReference_d(reference_d_ptr, reference_d_ptr + m);

    thrust::device_ptr<float> warped_d_ptr(warped_d);
    thrust::device_vector<float> vecWarped_d(warped_d_ptr, warped_d_ptr + m);

    thrust::device_ptr<float> lengths_d_ptr(lengths_d);
    thrust::device_vector<float> vec_lengths_d(lengths_d_ptr, lengths_d_ptr + m / 3);

    // initialize indices vector to [0,1,2,..m]
    thrust::counting_iterator<int> iter(0);
    thrust::device_vector<int> indices(m);
    thrust::copy(iter, iter + indices.size(), indices.begin());

    //sort an indices array by lengths as key. Then use it to sort reference and warped arrays

    thrust::sort_by_key(vec_lengths_d.begin(), vec_lengths_d.end(), indices.begin());
    thrust::gather(indices.begin(), indices.end(), vecReference_d.begin(), vecReference_d.begin());//end()?
    thrust::gather(indices.begin(), indices.end(), vecWarped_d.begin(), vecWarped_d.begin());//end()?

    return thrust::reduce(lengths_d_ptr, lengths_d_ptr + numToKeep, 0, thrust::plus<double>());

}
/* *************************************************************** */
//OPTIMIZER-----------------------------------------------
// estimate an affine transformation using least square
void getAffineMat3D(float* AR_d, float* Sigma_d, float* VT_d, float* U_d, float* reference_d, float* warped_d, float *transformation, const unsigned int numBlocks, unsigned int m, unsigned int n) {

    //populate A
    populateMatrixA <<< numBlocks, 512 >>>(AR_d, reference_d, m / 3); //test 2

    //calculate SVD on the GPU
    cusolverSVD(AR_d, m, n, Sigma_d, VT_d, U_d);
    //calculate the pseudoinverse
    cublasPseudoInverse(transformation, AR_d, warped_d, VT_d, Sigma_d, U_d, m, n);

}
/* *************************************************************** */
void affineLocalSearch3DCuda(mat44 *cpuMat, float* final_d, float *AR_d, float* Sigma_d, float* U_d, float* VT_d, float * newWarpedPos_d, float* referencePos_d, float* warpedPos_d, float* lengths_d, const unsigned int numBlocks, const unsigned int num_to_keep, const unsigned int m, const unsigned int n) {

    double lastDistance = std::numeric_limits<double>::max();

    float* lastTransformation_d;
    cudaMalloc(&lastTransformation_d, 16 * sizeof(float));

    //get initial affine matrix
    getAffineMat3D(AR_d, Sigma_d, VT_d, U_d, referencePos_d, warpedPos_d, final_d, numBlocks, m, n);

    for (unsigned int count = 0; count < MAX_ITERATIONS; ++count) {

        // Transform the points in the reference
        transformWarpedPointsKernel <<< numBlocks, 512 >>>(final_d, referencePos_d, newWarpedPos_d, m / 3); //test 1
        double distance = sortAndReduce(lengths_d, referencePos_d, warpedPos_d, newWarpedPos_d, numBlocks, num_to_keep, m);

        // If the change is not substantial or we are getting worst, we return
        if ((distance > lastDistance) || (lastDistance - distance) < TOLERANCE) break;

        lastDistance = distance;

        cudaMemcpy(lastTransformation_d, final_d, 16 * sizeof(float), cudaMemcpyDeviceToDevice);
        getAffineMat3D(AR_d, Sigma_d, VT_d, U_d, referencePos_d, warpedPos_d, final_d, numBlocks, m, n);
    }

    //async cudamemcpy here
    cudaMemcpy(final_d, lastTransformation_d, 16 * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(lastTransformation_d);
}
/* *************************************************************** */
void optimize_affine3D_cuda(mat44* cpuMat,
                            float* final_d,
                            float* A_d,
                            float* U_d,
                            float* Sigma_d,
                            float* VT_d,
                            float* lengths_d,
                            float* reference_d,
                            float* warped_d,
                            float* newWarped_d,
                            unsigned int m,
                            unsigned int n,
                            const unsigned int numToKeep,
                            bool ilsIn,
                            bool isAffine) {

    //m | blockMatchingParams->activeBlockNumber * 3
    //n | 12
    const unsigned int numEquations = m;
    const unsigned int numBlocks = (numEquations % 512) ? (numEquations / 512) + 1 : numEquations / 512;

    uploadMat44(*cpuMat, final_d);
    transformWarpedPointsKernel <<< numBlocks, 512 >>>(final_d, warped_d, newWarped_d, m / 3); //test 1
    cudaMemcpy(warped_d, newWarped_d, m * sizeof(float), cudaMemcpyDeviceToDevice);

    // run the local search optimization routine
    affineLocalSearch3DCuda(cpuMat, final_d, A_d, Sigma_d, U_d, VT_d, newWarped_d, reference_d, warped_d, lengths_d, numBlocks, numToKeep, m, n);

    downloadMat44(cpuMat, final_d);

}
