#include "_reg_ReadWriteImage.h"
#include "_reg_ReadWriteMatrix.h"
#include "_reg_localTrans_regul.h"
#include "_reg_tools.h"

#include "AffineDeformationFieldKernel.h"

#define EPS 0.000001

int main(int argc, char **argv)
{
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <refImage> <inputTrans> <expectedValue> <type>\n", argv[0]);
        return EXIT_FAILURE;
    }

    char *inputRefImageName = argv[1];
    char *inputTransFileName = argv[2];
    char *expectedValueFileName = argv[3];
    int computationType = atoi(argv[4]);

    // Read the input reference image
    nifti_image *referenceImage = reg_io_ReadImageFile(inputRefImageName);
    if (referenceImage == NULL) {
        reg_print_msg_error("The input reference image could not be read");
        return EXIT_FAILURE;
    }
    // Read the transformation file
    nifti_image *transImage = reg_io_ReadImageFile(inputTransFileName);
    if (transImage == NULL) {
        reg_print_msg_error("The transformation image could not be read");
        return EXIT_FAILURE;
    }

    // Compute the linear elasticity value
    double obtainedValue;
    switch(computationType){
    case 0: // Approximation based on the control point grid
       obtainedValue = reg_spline_approxLinearEnergy(transImage);
       break;
    case 1: // Dense based on the control point grid
       obtainedValue = reg_spline_linearEnergy(referenceImage, transImage);
       break;
    case 2: // Dense based on the deformation field
       obtainedValue = reg_defField_linearEnergy(transImage);
       break;
    default:
       reg_print_msg_error("Unexpected computation type");
       reg_exit();
    }

    // Read the expected value
    std::pair<size_t, size_t> inputMatrixSize = reg_tool_sizeInputMatrixFile(expectedValueFileName);
    size_t m = inputMatrixSize.first;
    size_t n = inputMatrixSize.second;
    if(m != 1 && n!= 1)
    {
       fprintf(stderr,"[NiftyReg ERROR] Error when reading the expected constraint value: %s\n",
               expectedValueFileName);
       return EXIT_FAILURE;
    }
    float **inputMatrix = reg_tool_ReadMatrixFile<float>(expectedValueFileName, m, n);
    float expectedValue = inputMatrix[0][0];
    double max_difference = fabs(obtainedValue-expectedValue);


    reg_matrix2DDeallocate(m, inputMatrix);
    nifti_image_free(referenceImage);
    nifti_image_free(transImage);

    if (max_difference > EPS){
        fprintf(stderr, "reg_test_linearElasticity error too large: %g (|%g-%g| > %g)\n",
                max_difference, obtainedValue, expectedValue, EPS);
        return EXIT_FAILURE;
    }
#ifndef NDEBUG
    fprintf(stdout, "reg_test_linearElasticity ok: %g (<%g)\n",
            max_difference, EPS);
#endif

    return EXIT_SUCCESS;
}

