#include "nifti1_io.h"
#include "_reg_maths.h"
#include "_reg_maths_eigen.h"
#include "_reg_ReadWriteMatrix.h"
//STD
#include <algorithm>

#define EPS 0.000001

int check_matrix_difference(mat44 matrix1, mat44 matrix2, char *name, float &max_difference)
{
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float difference = fabsf(matrix1.m[i][j] - matrix2.m[i][j]);
            max_difference = std::max(difference, max_difference);
            if (difference > EPS){
                fprintf(stderr, "reg_test_matrix_operation - %s failed %g>%g\n",
                    name, difference, EPS);
                return EXIT_FAILURE;
            }
        }
    }
    return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{

    if (argc != 9) {
        fprintf(stderr, "Usage: %s <inputMatrix1> <inputMatrix2>\
                                                <expectedMultMatrix> <expectedAddMatrix> <expectedSubMatrix> \
                                                                        <expectedExpMatrix> <expectedLogMatrix> <expectedInvMatrix> \n", argv[0]);
        return EXIT_FAILURE;
    }

    char *inputMatrix1Filename = argv[1];
    char *inputMatrix2Filename = argv[2];
    char *expectedMultMatrixFilename = argv[3];
    char *expectedAddMatrixFilename = argv[4];
    char *expectedSubMatrixFilename = argv[5];
    char *expectedExpMatrixFilename = argv[6];
    char *expectedLogMatrixFilename = argv[7];
    char *expectedInvMatrixFilename = argv[8];

    std::pair<size_t, size_t> inputMatrix1Size = reg_tool_sizeInputMatrixFile(inputMatrix1Filename);
    size_t m = inputMatrix1Size.first;
    size_t n = inputMatrix1Size.second;

    if (m != 4 || n != 4) {
        fprintf(stderr, "The input matrices have to be 4x4 matrices");
        return EXIT_FAILURE;
    }

    std::pair<size_t, size_t> inputMatrix2Size = reg_tool_sizeInputMatrixFile(inputMatrix2Filename);
    size_t m2 = inputMatrix2Size.first;
    size_t n2 = inputMatrix2Size.second;

    if (m2 != 4 || n2 != 4) {
        fprintf(stderr, "The input matrices have to be 4x4 matrices");
        return EXIT_FAILURE;
    }

    mat44 *inputMatrix1 = reg_tool_ReadMat44File(inputMatrix1Filename);
    mat44 *inputMatrix2 = reg_tool_ReadMat44File(inputMatrix2Filename);
    mat44 *expectedMultMatrix = reg_tool_ReadMat44File(expectedMultMatrixFilename);
    mat44 *expectedAddMatrix = reg_tool_ReadMat44File(expectedAddMatrixFilename);
    mat44 *expectedSubMatrix = reg_tool_ReadMat44File(expectedSubMatrixFilename);
    mat44 *expectedExpMatrix = reg_tool_ReadMat44File(expectedExpMatrixFilename);
    mat44 *expectedLogMatrix = reg_tool_ReadMat44File(expectedLogMatrixFilename);
    mat44 *expectedInvMatrix = reg_tool_ReadMat44File(expectedInvMatrixFilename);

    ///////////////////////
    float max_difference = 0;

    if (check_matrix_difference(*expectedMultMatrix, (*inputMatrix1)*(*inputMatrix2), (char *) "matrix multiplication", max_difference)) return EXIT_FAILURE;

    if (check_matrix_difference(*expectedMultMatrix, reg_mat44_mul(inputMatrix1, inputMatrix2), (char *) "matrix multiplication", max_difference)) return EXIT_FAILURE;

    if (check_matrix_difference(*expectedAddMatrix, (*inputMatrix1) + (*inputMatrix2), (char *) "matrix addition", max_difference)) return EXIT_FAILURE;

    if (check_matrix_difference(*expectedAddMatrix, reg_mat44_add(inputMatrix1, inputMatrix2), (char *) "matrix addition", max_difference)) return EXIT_FAILURE;

    if (check_matrix_difference(*expectedSubMatrix, (*inputMatrix1) - (*inputMatrix2), (char *) "matrix subtraction", max_difference)) return EXIT_FAILURE;

    if (check_matrix_difference(*expectedSubMatrix, reg_mat44_minus(inputMatrix1, inputMatrix2), (char *) "matrix subtraction", max_difference)) return EXIT_FAILURE;

    if (check_matrix_difference(*expectedExpMatrix, reg_mat44_expm(inputMatrix1), (char *) "matrix exponentiation", max_difference)) return EXIT_FAILURE;

    if (check_matrix_difference(*expectedLogMatrix, reg_mat44_logm(inputMatrix1), (char *) "matrix logarithm", max_difference)) return EXIT_FAILURE;

    if (check_matrix_difference(*expectedInvMatrix, reg_mat44_inv(inputMatrix1), (char *) "reg_mat44_inv matrix inverse", max_difference)) return EXIT_FAILURE;

    if (check_matrix_difference(*expectedInvMatrix, nifti_mat44_inverse(*inputMatrix1), (char *) "nifti_mat44_inverse matrix inverse", max_difference)) return EXIT_FAILURE;

    ////////////////////////
#ifndef NDEBUG
    fprintf(stdout, "reg_test_matrix_operation ok: %g (<%g)\n", max_difference, EPS);
#endif
    return EXIT_SUCCESS;
}

