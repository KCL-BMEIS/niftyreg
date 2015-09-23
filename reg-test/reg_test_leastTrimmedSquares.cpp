#include "nifti1_io.h"
#include "_reg_maths.h"
#include "_reg_ReadWriteMatrix.h"
#include "_reg_globalTrans.h"
//STD
#include <algorithm>
//
#include "OptimiseKernel.h"
#include "Platform.h"

#include "Content.h"
#ifdef _USE_CUDA
#include "CudaContent.h"
#endif
#ifdef _USE_OPENCL
#include "CLContent.h"
#endif

#define EPS 0.000001

int check_matrix_difference(mat44 matrix1, mat44 matrix2, char *name, float &max_difference)
{
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float difference = fabsf(matrix1.m[i][j] - matrix2.m[i][j]);
            max_difference = std::max(difference, max_difference);
            if (difference > EPS){
                fprintf(stderr, "reg_test_leastTrimmedSquares - %s failed %g>%g\n",
                    name, difference, EPS);
                return EXIT_FAILURE;
            }
        }
    }
    return EXIT_SUCCESS;
}

void test(Content *con, int platformCode, bool isAffine, bool ils, bool svd) {

    Platform *platform = new Platform(platformCode);

    Kernel *optimiseKernel = platform->createKernel(OptimiseKernel::getName(), con);
    optimiseKernel->castTo<OptimiseKernel>()->calculate(isAffine,ils, svd);

    delete optimiseKernel;
    delete platform;
}

int main(int argc, char **argv)
{

    if (argc != 7) {
        fprintf(stderr, "Usage: %s <inputPoints1> <inputPoints2> <percentToKeep> <isAffine> <expectedLTSMatrix> <platformCode> \n", argv[0]);
        return EXIT_FAILURE;
    }

    char *inputMatrix1Filename = argv[1];
    char *inputMatrix2Filename = argv[2];
    unsigned int percentToKeep = atoi(argv[3]);
    bool isAffine = atoi(argv[4]);
    char *expectedLTSMatrixFilename = argv[5];
    int platformCode = atoi(argv[6]);

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
    // Platforms
    Content *con = NULL;
    if (platformCode == NR_PLATFORM_CPU) {
        con = new Content();
    }
#ifdef _USE_CUDA
    else if (platformCode == NR_PLATFORM_CUDA) {
        con = new CudaContent();
    }
#endif
#ifdef _USE_OPENCL
    else if (platformCode == NR_PLATFORM_CL) {
        con = new ClContent();
    }
#endif
    else {
        reg_print_msg_error("The platform code is not suppoted");
        return EXIT_FAILURE;
    }
    ////////////////////////
    float max_difference = 0;
    unsigned int num_points = m1;
    //I think it is a bit durty... what I am going to do
    _reg_blockMatchingParam* blockMatchingParams = new  _reg_blockMatchingParam();
    
    blockMatchingParams->blockNumber[0] = 1;
    blockMatchingParams->blockNumber[1] = 1;

    blockMatchingParams->activeBlockNumber = num_points;
    blockMatchingParams->definedActiveBlock = num_points;
    blockMatchingParams->percent_to_keep = percentToKeep;

    mat44* test_LTS = (mat44 *)malloc(sizeof(mat44));
    reg_mat44_eye(test_LTS);
    con->setTransformationMatrix(test_LTS);

    //2-D
    if (n1 == 2) {

        blockMatchingParams->blockNumber[2] = 1;
        blockMatchingParams->referencePosition = new float[num_points * n1];
        blockMatchingParams->warpedPosition = new float[num_points * n1];

        unsigned int compteur = 0;
        for (unsigned int j = 0; j < num_points; j++) {
            blockMatchingParams->referencePosition[compteur] = inputMatrix1[j][0];
            blockMatchingParams->referencePosition[compteur + 1] = inputMatrix1[j][1];
            blockMatchingParams->warpedPosition[compteur] = inputMatrix2[j][0];
            blockMatchingParams->warpedPosition[compteur + 1] = inputMatrix2[j][1];
            compteur +=n1;
        }
    }
    else if (n1 == 3) {

        blockMatchingParams->blockNumber[2] = 2;
        blockMatchingParams->referencePosition = new float[num_points * n1];
        blockMatchingParams->warpedPosition = new float[num_points * n1];
        unsigned int compteur = 0;
        for (unsigned int j = 0; j < num_points; j++) {
            blockMatchingParams->referencePosition[compteur] = inputMatrix1[j][0];
            blockMatchingParams->referencePosition[compteur + 1] = inputMatrix1[j][1];
            blockMatchingParams->referencePosition[compteur + 2] = inputMatrix1[j][2];
            blockMatchingParams->warpedPosition[compteur] = inputMatrix2[j][0];
            blockMatchingParams->warpedPosition[compteur + 1] = inputMatrix2[j][1];
            blockMatchingParams->warpedPosition[compteur + 2] = inputMatrix2[j][2];
            compteur +=n1;
        }
    }
    else {
        fprintf(stderr, "The input matrix dimensions are not supported");
        return EXIT_FAILURE;
    }

    con->setBlockMatchingParams(blockMatchingParams);
    test(con, platformCode, isAffine, 1, 1);

#ifndef NDEBUG
    reg_mat44_disp(con->getTransformationMatrix(), (char *) "test_optimize_2D");
#endif

    if (check_matrix_difference(*expectedLSMatrix, *con->getTransformationMatrix(), (char *) "LTS matrices 2D affine - rigid", max_difference)) return EXIT_FAILURE;

    ////////////////////////
    // FREE THE MEMORY: ////
    ////////////////////////
    delete con;
    free(expectedLSMatrix);
    reg_matrix2DDeallocate(m2, inputMatrix2);
    reg_matrix2DDeallocate(m1, inputMatrix1);

#ifndef NDEBUG
    fprintf(stdout, "reg_test_leastTrimmedSquares ok: %g (<%g)\n", max_difference, EPS);
#endif
    return EXIT_SUCCESS;
}

