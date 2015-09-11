#include "nifti1_io.h"
#include "_reg_maths.h"
#include "_reg_globalTrans.h"
#include "_reg_blockMatching.h"
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
                fprintf(stderr, "reg_test_leastSquares - %s failed %g>%g\n",
                    name, difference, EPS);
                return EXIT_FAILURE;
            }
        }
    }
    return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{

    if (argc != 6) {
        fprintf(stderr, "Usage: %s <inputMatrix1> <inputMatrix2> <percentToKeep> <isAffine> <expectedLTSMatrix> \n", argv[0]);
        return EXIT_FAILURE;
    }

    char *inputMatrix1Filename = argv[1];
    char *inputMatrix2Filename = argv[2];
    float percentToKeep = atoi(argv[3]);
    bool isAffine = atoi(argv[4]);
    char *expectedLTSMatrixFilename = argv[5];

    std::pair<size_t, size_t> inputMatrix1Size = reg_tool_sizeInputMatrixFile(inputMatrix1Filename);
    size_t m1 = inputMatrix1Size.first;
    size_t n1 = inputMatrix1Size.second;
    std::pair<size_t, size_t> inputMatrix2Size = reg_tool_sizeInputMatrixFile(inputMatrix2Filename);
    size_t m2 = inputMatrix2Size.first;
    size_t n2 = inputMatrix2Size.second;

    if (m1 != m2 || n1 != n2) {
        fprintf(stderr, "The input matrices must have the same size");
        return EXIT_FAILURE;
    }

    float **inputMatrix1 = reg_tool_ReadMatrixFile<float>(inputMatrix1Filename, m1, n1);
    float **inputMatrix2 = reg_tool_ReadMatrixFile<float>(inputMatrix2Filename, m2, n2);
    mat44 *expectedLSMatrix = reg_tool_ReadMat44File(expectedLTSMatrixFilename);
    ////////////////////////
    float max_difference = 0;
    //2-D
    if (n1 == 2) {
        std::vector<_reg_sorted_point2D> points;
        mat44* test_leastSquares = (mat44 *)malloc(sizeof(mat44));
        reg_mat44_eye(test_leastSquares);
        for (unsigned int i = 0; i < m1; i++) {
            points.push_back(_reg_sorted_point2D(inputMatrix1[i], inputMatrix2[i], 0.0));
        }
        if (isAffine) {
            estimate_affine_transformation2D(points, test_leastSquares);
#ifndef NDEBUG
            reg_mat44_disp(test_leastSquares, "test_affine_transformation2D");
#endif
            if (check_matrix_difference(*expectedLSMatrix, *test_leastSquares, (char *) "LTS matrices 2D affine", max_difference)) return EXIT_FAILURE;
        }
        else {
            estimate_rigid_transformation2D(points, test_leastSquares);
#ifndef NDEBUG
            reg_mat44_disp(test_leastSquares, "test_rigid_transformation2D");
#endif
            if (check_matrix_difference(*expectedLSMatrix, *test_leastSquares, (char *) "LTS matrices 2D rigid", max_difference)) return EXIT_FAILURE;
        }

        ////////////////////////
        // FREE THE MEMORY: ////
        ////////////////////////
        //for (std::vector<float *>::iterator it = points1.begin(); it != points1.end(); ++it) {
        //    free(*it);
        //}
        //for (std::vector<float *>::iterator it = points2.begin(); it != points2.end(); ++it) {
        //    free(*it);
        //}
        free(expectedLSMatrix);
        reg_matrix2DDeallocate(m2, inputMatrix2);
        reg_matrix2DDeallocate(m1, inputMatrix1);
    }
    else if (n1 == 3) {
        std::vector<_reg_sorted_point3D> points;
        mat44* test_leastSquares = (mat44 *)malloc(sizeof(mat44));
        reg_mat44_eye(test_leastSquares);
        for (unsigned int i = 0; i < m1; i++) {
            points.push_back(_reg_sorted_point3D(inputMatrix1[i], inputMatrix2[i], 0.0));
        }
        if (isAffine) {
            estimate_affine_transformation3D(points, test_leastSquares);
#ifndef NDEBUG
            reg_mat44_disp(test_leastSquares, "test_affine_transformation3D");
#endif
            if (check_matrix_difference(*expectedLSMatrix, *test_leastSquares, (char *) "LTS matrices 3D affine", max_difference)) return EXIT_FAILURE;
        }
        else {
            estimate_rigid_transformation3D(points, test_leastSquares);
#ifndef NDEBUG
            reg_mat44_disp(test_leastSquares, "test_rigid_transformation3D");
#endif
            if (check_matrix_difference(*expectedLSMatrix, *test_leastSquares, (char *) "LTS matrices 3D rigid", max_difference)) return EXIT_FAILURE;
        }

        ////////////////////////
        // FREE THE MEMORY: ////
        ////////////////////////
        //for (std::vector<float *>::iterator it = points1.begin(); it != points1.end(); ++it) {
        //    free(*it);
        //}
        //for (std::vector<float *>::iterator it = points2.begin(); it != points2.end(); ++it) {
        //    free(*it);
        //}
        free(expectedLSMatrix);
        reg_matrix2DDeallocate(m2, inputMatrix2);
        reg_matrix2DDeallocate(m1, inputMatrix1);
    }
    else {
        fprintf(stderr, "The input matrix dimensions are not supported");
        return EXIT_FAILURE;
    }
    //

    //
#ifndef NDEBUG
    fprintf(stdout, "reg_test_leastSquares ok: %g (<%g)\n", max_difference, EPS);
#endif
    return EXIT_SUCCESS;
}

