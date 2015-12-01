#include "_reg_tools.h"
#include "_reg_maths_eigen.h"
#include "_reg_ReadWriteMatrix.h"

#ifdef _USE_CUDA
#include "cusolverDn.h"
#include "_reg_common_cuda.h"
#include "optimizeKernel.h"
#endif
//STD
#include <algorithm>

#define EPS 0.000001

#ifdef _USE_CUDA
/***********************/
/* CUDA ERROR CHECKING */
/***********************/
void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) { exit(code); }
    }
}
void gpuErrchk(cudaError_t ans) { gpuAssert((ans), __FILE__, __LINE__); }


/* ******************************** */
template<typename T>
void cudaCommon_transfer2DMatrixFromCpuToDevice(T* M_d, T** M_h, unsigned int m, unsigned int n) {

    T *tmpMat_h = (T*)malloc(m*n * sizeof(T));
    matmnToCptr<T>(M_h, tmpMat_h, m, n);
    NR_CUDA_SAFE_CALL(cudaMemcpy(M_d, tmpMat_h, m*n * sizeof(T), cudaMemcpyHostToDevice));
    free(tmpMat_h);

}
template void cudaCommon_transfer2DMatrixFromCpuToDevice<float>(float* M_d, float** M_h, unsigned int m, unsigned int n);
template void cudaCommon_transfer2DMatrixFromCpuToDevice<double>(double* M_d, double** M_h, unsigned int m, unsigned int n);
/* ******************************** */
/* ******************************** */
template<typename T>
void cudaCommon_transferFromDeviceTo2DMatrixCpu(T* M_d, T** M_h, unsigned int m, unsigned int n) {

    T *tmpMat_h = (T*)malloc(m*n * sizeof(T));
    NR_CUDA_SAFE_CALL(cudaMemcpy(tmpMat_h, M_d, m*n * sizeof(T), cudaMemcpyDeviceToHost));
    cPtrToMatmn<T>(M_h, tmpMat_h, m, n);
    free(tmpMat_h);

}
template void cudaCommon_transferFromDeviceTo2DMatrixCpu<float>(float* M_d, float** M_h, unsigned int m, unsigned int n);
template void cudaCommon_transferFromDeviceTo2DMatrixCpu<double>(double* M_d, double** M_h, unsigned int m, unsigned int n);
#endif

int main(int argc, char **argv)
{
    //NOT REALLY PLATFORM... HAVE TO CHANGE THAT LATER
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <inputSVDMatrix> <expectedUMatrix> <expectedSMatrix> <expectedVMatrix> <platform>\n", argv[0]);
        return EXIT_FAILURE;
    }

    char *inputSVDMatrixFilename = argv[1];
    char *expectedUMatrixFilename = argv[2];
    char *expectedSMatrixFilename = argv[3];
    char *expectedVMatrixFilename = argv[4];
    int platformCode = atoi(argv[5]);

    std::pair<size_t, size_t> inputMatrixSize = reg_tool_sizeInputMatrixFile(inputSVDMatrixFilename);
    size_t m = inputMatrixSize.first;
    size_t n = inputMatrixSize.second;
    size_t min_size = std::min(m, n);
    size_t max_size = std::max(m, n);
#ifndef NDEBUG
    std::cout << "min_size=" << min_size << std::endl;
#endif

    float **inputSVDMatrix = reg_tool_ReadMatrixFile<float>(inputSVDMatrixFilename, m, n);

#ifndef NDEBUG
    std::cout << "inputSVDMatrix[i][j]=" << std::endl;
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            std::cout << inputSVDMatrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
#endif

    float ** expectedSMatrix = reg_tool_ReadMatrixFile<float>(expectedSMatrixFilename, min_size, min_size);
    float **test_SMatrix = reg_matrix2DAllocate<float>(min_size, min_size);

    //more row than columns
    if (m > n) {

        float ** expectedUMatrix = reg_tool_ReadMatrixFile<float>(expectedUMatrixFilename, m, n);
        float ** expectedVMatrix = reg_tool_ReadMatrixFile<float>(expectedVMatrixFilename, min_size, min_size);

        float **test_UMatrix = reg_matrix2DAllocate<float>(m, n);
        float **test_VMatrix = reg_matrix2DAllocate<float>(min_size, min_size);

        //For the old version of the function:
        float **inputSVDMatrixNotTouched = reg_tool_ReadMatrixFile<float>(inputSVDMatrixFilename, m, n);
        double *test_SVect = (double*)malloc(min_size*sizeof(double));
        //SVD
#ifdef _USE_CUDA
        if(platformCode != 1) {
#endif
            //svd<float>(inputSVDMatrix, m, n, test_SVect, test_VMatrix);
            //U
            for (size_t i = 0; i < m; i++) {
                for (size_t j = 0; j < n; j++) {
                    test_UMatrix[i][j] = inputSVDMatrix[i][j];
                }
            }
#ifdef _USE_CUDA
        }
        else{
            double* inputSVDMatrix_d;
            NR_CUDA_SAFE_CALL(cudaCommon_allocateArrayToDevice<double>(&inputSVDMatrix_d, m * n));
            double **inputSVDMatrix_h = reg_tool_ReadMatrixFile<double>(inputSVDMatrixFilename, m, n);
            cudaCommon_transfer2DMatrixFromCpuToDevice<double>(inputSVDMatrix_d,inputSVDMatrix_h,m,n);

            double* Sigma_d;
            NR_CUDA_SAFE_CALL(cudaCommon_allocateArrayToDevice<double>(&Sigma_d, min_size));
            double* U_d;
            NR_CUDA_SAFE_CALL(cudaCommon_allocateArrayToDevice<double>(&U_d, max_size * max_size));
            double* VT_d;
            NR_CUDA_SAFE_CALL(cudaCommon_allocateArrayToDevice<double>(&VT_d, min_size * min_size));

            //CUDA EXECUTION
            //cusolverSVD(inputSVDMatrix_d, m, n, Sigma_d, VT_d, U_d);
            // --- device side SVD workspace and matrices
            int Lwork = 0;
            int *devInfo;
            gpuErrchk(cudaMalloc(&devInfo, sizeof(int)));
            cusolverStatus_t stat;

            // --- CUDA solver initialization
            cusolverDnHandle_t solver_handle;
            cusolverDnCreate(&solver_handle);

            stat = cusolverDnDgesvd_bufferSize(solver_handle, m, n, &Lwork);
            if(stat != CUSOLVER_STATUS_SUCCESS ) std::cout << "Initialization of cuSolver failed. \n";

            double *work_d;
            gpuErrchk(cudaMalloc(&work_d, Lwork * sizeof(double)));

            // --- CUDA SVD execution
            stat = cusolverDnDgesvd(solver_handle, 'A', 'A', m, n, inputSVDMatrix_d, m, Sigma_d, U_d, max_size, VT_d, min_size, work_d, Lwork, NULL, devInfo);
            //stat = cusolverDnSgesvd(solver_handle, 'N', 'N', M, N, d_A, M, d_S, d_U, M, d_V, N, work, work_size, NULL, devInfo);
            cudaDeviceSynchronize();

            int devInfo_h = 0;
            gpuErrchk(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
            std::cout << "devInfo = " << devInfo_h << "\n";

            switch(stat){
            case CUSOLVER_STATUS_SUCCESS:           std::cout << "SVD computation success\n";                       break;
            case CUSOLVER_STATUS_NOT_INITIALIZED:   std::cout << "Library cuSolver not initialized correctly\n";    break;
            case CUSOLVER_STATUS_INVALID_VALUE:     std::cout << "Invalid parameters passed\n";                     break;
            case CUSOLVER_STATUS_INTERNAL_ERROR:    std::cout << "Internal operation failed\n";                     break;
            }

            if (devInfo_h == 0 && stat == CUSOLVER_STATUS_SUCCESS) std::cout    << "SVD successful\n\n";

            // --- Moving the results from device to host
            gpuErrchk(cudaMemcpy(test_SVect, Sigma_d, n * sizeof(double), cudaMemcpyDeviceToHost));

            for(int i = 0; i < n; i++) std::cout << "d_S["<<i<<"] = " << test_SVect[i] << std::endl;

            cusolverDnDestroy(solver_handle);
        }
    }
#endif
    /*
            //RETRIEVE THE RESULTS FROM THE GPU
            float **test_UMatrixCUDA = reg_matrix2DAllocate<float>(m, m);
            cudaCommon_transferArrayFromDeviceToCpu<float>(test_SVect, &Sigma_d, min_size);
            cudaCommon_transferFromDeviceTo2DMatrixCpu<float>(VT_d, test_VMatrix, min_size, min_size);
            test_VMatrix = reg_matrix2DTranspose<float>(test_VMatrix, min_size, min_size);
            cudaCommon_transferFromDeviceTo2DMatrixCpu<float>(U_d, test_UMatrixCUDA, m, m);

#ifndef NDEBUG
            std::cout << "test_UMatrixCUDA[i][j]=" << std::endl;
            for (size_t i = 0; i < m; i++) {
                for (size_t j = 0; j < m; j++) {
                    std::cout << test_UMatrixCUDA[i][j] << " ";
                }
                std::cout << std::endl;
            }
#endif

        }
#endif
        //S
        for (size_t i = 0; i < min_size; i++) {
            for (size_t j = 0; j < min_size; j++) {
                if (i == j) {
                    test_SMatrix[i][j] = test_SVect[i];
                }
                else {
                    test_SMatrix[i][j] = 0;
                }
            }
        }

#ifndef NDEBUG
        std::cout << "test_UMatrix[i][j]=" << std::endl;
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                std::cout << test_UMatrix[i][j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "test_SMatrix[i][j]=" << std::endl;
        for (size_t i = 0; i < min_size; i++) {
            for (size_t j = 0; j < min_size; j++) {
                std::cout << test_SMatrix[i][j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "test_VMatrix[i][j]=" << std::endl;
        for (size_t i = 0; i < min_size; i++) {
            for (size_t j = 0; j < min_size; j++) {
                std::cout << test_VMatrix[i][j] << " ";
            }
            std::cout << std::endl;
        }
#endif
        //The sign of the vector are different between Matlab and Eigen so let's take the absolute value and let's check that U*S*V' = M
        float max_difference = 0;

        for (size_t i = 0; i < min_size; i++) {
            for (size_t j = 0; j < min_size; j++) {
                float difference = fabsf(test_SMatrix[i][j]) - fabsf(expectedSMatrix[i][j]);
                max_difference = std::max(difference, max_difference);
                if (difference > EPS){
                    fprintf(stderr, "reg_test_svd - checking S - Error in the SVD computation %.8g (>%g)\n", difference, EPS);
                    return EXIT_FAILURE;
                }
                difference = fabsf(test_VMatrix[i][j]) - fabsf(expectedVMatrix[i][j]);
                max_difference = std::max(difference, max_difference);
                if (difference > EPS){
                    fprintf(stderr, "reg_test_svd - checking V - Error in the SVD computation %.8g (>%g)\n", difference, EPS);
                    return EXIT_FAILURE;
                }
            }
        }
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                float difference = fabsf(test_UMatrix[i][j]) - fabsf(expectedUMatrix[i][j]);
                max_difference = std::max(difference, max_difference);
                if (difference > EPS){
                    fprintf(stderr, "reg_test_svd - checking U - Error in the SVD computation %.8g (>%g)\n", difference, EPS);
                    return EXIT_FAILURE;
                }
            }
        }
        //check that U*S*V' = M
        float ** US = reg_matrix2DMultiply(test_UMatrix, m, n, test_SMatrix, min_size, min_size, false);
        float ** VT = reg_matrix2DTranspose(test_VMatrix, min_size, min_size);
        float ** test_inputMatrix = reg_matrix2DMultiply(US, m, min_size, VT, min_size, min_size, false);
#ifndef NDEBUG
        std::cout << "test_inputMatrix[i][j]=" << std::endl;
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                std::cout << test_inputMatrix[i][j] << " ";
            }
            std::cout << std::endl;
        }
#endif
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                float difference = fabsf(inputSVDMatrixNotTouched[i][j] - test_inputMatrix[i][j]);
                max_difference = std::max(difference, max_difference);
                if (difference > EPS){
                    fprintf(stderr, "reg_test_svd - checking that U*S*V' = M - Error in the SVD computation %.8g (>%g)\n", difference, EPS);
                    return EXIT_FAILURE;
                }
            }
        }

        // Free the allocated variables
        for (size_t i = 0; i < m; i++) {
            free(inputSVDMatrix[i]);
            free(inputSVDMatrixNotTouched[i]);
            free(expectedUMatrix[i]);
            free(test_UMatrix[i]);
        }
        for (size_t j = 0; j < min_size; j++) {
            free(expectedSMatrix[j]);
            free(expectedVMatrix[j]);
            free(test_SMatrix[j]);
            free(test_VMatrix[j]);
        }
        free(inputSVDMatrix);
        free(inputSVDMatrixNotTouched);
        free(expectedUMatrix);
        free(expectedSMatrix);
        free(expectedVMatrix);
        free(test_UMatrix);
        free(test_SMatrix);
        free(test_VMatrix);
        free(test_SVect);
        //
#ifndef NDEBUG
        fprintf(stdout, "reg_test_svd ok: %g ( <%g )\n", max_difference, EPS);
#endif
        return EXIT_SUCCESS;
    }
    //more colums than rows
    else {

        float ** expectedUMatrix = reg_tool_ReadMatrixFile<float>(expectedUMatrixFilename, min_size, min_size);
        float ** expectedVMatrix = reg_tool_ReadMatrixFile<float>(expectedVMatrixFilename, n, m);

        float **test_UMatrix = reg_matrix2DAllocate<float>(min_size, min_size);
        float **test_VMatrix = reg_matrix2DAllocate<float>(n, m);

        svd<float>(inputSVDMatrix, m, n, &test_UMatrix, &test_SMatrix, &test_VMatrix);
#ifndef NDEBUG
        std::cout << "test_UMatrix[i][j]=" << std::endl;
        for (size_t i = 0; i < min_size; i++) {
            for (size_t j = 0; j < min_size; j++) {
                std::cout << test_UMatrix[i][j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "test_SMatrix[i][j]=" << std::endl;
        for (size_t i = 0; i < min_size; i++) {
            for (size_t j = 0; j < min_size; j++) {
                std::cout << test_SMatrix[i][j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "test_VMatrix[i][j]=" << std::endl;
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < m; j++) {
                std::cout << test_VMatrix[i][j] << " ";
            }
            std::cout << std::endl;
        }
#endif
        //The sign of the vector are different between Matlab and Eigen so let's take the absolute value and let's check that U*S*V' = M
        float max_difference = 0;

        for (size_t i = 0; i < min_size; i++) {
            for (size_t j = 0; j < min_size; j++) {
                float difference = fabsf(test_SMatrix[i][j]) - fabsf(expectedSMatrix[i][j]);
                max_difference = std::max(difference, max_difference);
                if (difference > EPS){
                    fprintf(stderr, "reg_test_svd - Error in the SVD computation %.8g (>%g)\n", difference, EPS);
                    return EXIT_FAILURE;
                }
                difference = fabsf(test_UMatrix[i][j]) - fabsf(test_UMatrix[i][j]);
                max_difference = std::max(difference, max_difference);
                if (difference > EPS){
                    fprintf(stderr, "reg_test_svd - Error in the SVD computation %.8g (>%g)\n", difference, EPS);
                    return EXIT_FAILURE;
                }
            }
        }
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < m; j++) {
                float difference = fabsf(test_VMatrix[i][j]) - fabsf(test_VMatrix[i][j]);
                max_difference = std::max(difference, max_difference);
                if (difference > EPS){
                    fprintf(stderr, "reg_test_svd - Error in the SVD computation %.8g (>%g)\n", difference, EPS);
                    return EXIT_FAILURE;
                }
            }
        }

        //check that U*S*V' = M
        float ** US = reg_matrix2DMultiply(test_UMatrix, min_size, min_size, test_SMatrix, min_size, min_size, false);
        float ** VT = reg_matrix2DTranspose(test_VMatrix, n, m);
        float ** test_inputMatrix = reg_matrix2DMultiply(US, min_size, min_size, VT, m, n, false);
#ifndef NDEBUG
        std::cout << "test_inputMatrix[i][j]=" << std::endl;
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                std::cout << test_inputMatrix[i][j] << " ";
            }
            std::cout << std::endl;
        }
#endif
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                float difference = fabsf(inputSVDMatrix[i][j] - test_inputMatrix[i][j]);
                max_difference = std::max(difference, max_difference);
                if (difference > EPS){
                    fprintf(stderr, "reg_test_svd - checking that U*S*V' = M - Error in the SVD computation %.8g (>%g)\n", difference, EPS);
                    return EXIT_FAILURE;
                }
            }
        }

        // Free the allocated variables
        for (size_t i = 0; i < min_size; i++) {
            free(inputSVDMatrix[i]);
            free(expectedUMatrix[i]);
            free(test_UMatrix[i]);
            free(expectedSMatrix[i]);
            free(test_SMatrix[i]);
        }
        for (size_t j = 0; j < n; j++) {
            free(expectedVMatrix[j]);
            free(test_VMatrix[j]);
        }
        free(inputSVDMatrix);
        free(expectedUMatrix);
        free(expectedSMatrix);
        free(expectedVMatrix);
        free(test_UMatrix);
        free(test_SMatrix);
        free(test_VMatrix);
        //
#ifndef NDEBUG
        fprintf(stdout, "reg_test_svd ok: %g (<%g)\n", max_difference, EPS);
#endif
        return EXIT_SUCCESS;
    }
    */
}
