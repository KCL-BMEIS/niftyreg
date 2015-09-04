#include "_reg_maths.h"
#include "_reg_globalTrans.h"
//STD
#include <algorithm>

#define EPS 0.000001

int main(int argc, char **argv)
{
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <inputSVDMatrix> <expectedUMatrix> <expectedSMatrix> <expectedVMatrix>\n", argv[0]);
        return EXIT_FAILURE;
    }

    char *inputSVDMatrixFilename = argv[1];
    char *expectedUMatrixFilename = argv[2];
    char *expectedSMatrixFilename = argv[3];
    char *expectedVMatrixFilename = argv[4];

    std::pair<size_t, size_t> inputMatrixSize = reg_tool_sizeInputMatrixFile(inputSVDMatrixFilename);
    size_t m = inputMatrixSize.first;
    size_t n = inputMatrixSize.second;
    size_t min_size = std::min(m, n);
#ifndef NDEBUG
    std::cout << "m=" << m << std::endl;
    std::cout << "n=" << n << std::endl;
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

        svd<float>(inputSVDMatrix, m, n, &test_UMatrix, &test_SMatrix, &test_VMatrix);
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
                    fprintf(stderr, "reg_test_svd - Error in the SVD computation %.8g (>%g)\n", difference, EPS);
                    return EXIT_FAILURE;
                }
                difference = fabsf(test_VMatrix[i][j]) - fabsf(expectedVMatrix[i][j]);
                max_difference = std::max(difference, max_difference);
                if (difference > EPS){
                    fprintf(stderr, "reg_test_svd - Error in the SVD computation %.8g (>%g)\n", difference, EPS);
                    return EXIT_FAILURE;
                }
            }
        }
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                float difference = fabsf(test_UMatrix[i][j]) - fabsf(expectedUMatrix[i][j]);
                max_difference = std::max(difference, max_difference);
                if (difference > EPS){
                    fprintf(stderr, "reg_test_svd - Error in the SVD computation %.8g (>%g)\n", difference, EPS);
                    return EXIT_FAILURE;
                }
            }
        }
        //check that U*S*V' = M
        float ** US = reg_matrix2DMultiply(test_UMatrix, m, n, test_SMatrix, min_size, min_size);
        float ** VT = reg_matrix2DTranspose(test_VMatrix, min_size, min_size);
        float ** test_inputMatrix = reg_matrix2DMultiply(US, m, min_size, VT, min_size, min_size);
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
        for (size_t i = 0; i < m; i++) {
            free(inputSVDMatrix[i]);
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
        free(expectedUMatrix);
        free(expectedSMatrix);
        free(expectedVMatrix);
        free(test_UMatrix);
        free(test_SMatrix);
        free(test_VMatrix);
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
        float ** US = reg_matrix2DMultiply(test_UMatrix, min_size, min_size, test_SMatrix, min_size, min_size);
        float ** VT = reg_matrix2DTranspose(test_VMatrix, n, m);
        float ** test_inputMatrix = reg_matrix2DMultiply(US, min_size, min_size, VT, m, n);
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
}
