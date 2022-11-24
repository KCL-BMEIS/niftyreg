#include "_reg_ReadWriteMatrix.h"
#include "_reg_tools.h"
#include "_reg_common_cuda.h"

#include <catch2/catch_test_macros.hpp>

#include "AladinContent.h"
#ifdef _USE_CUDA
#include "CUDAAladinContent.h"
#endif
#ifdef _USE_OPENCL
#include "CLAladinContent.h"
#endif

#define EPS_SINGLE 0.0001

/*
    This test file contains the following unit tests:
    test function: Multiplication of a nifti_image by a scalar value
*/

typedef std::tuple<std::string, nifti_image*, float> test_data;


TEST_CASE("Multiply Nifti by scalar", "[MultImage]") {
    // Create a reference 2D image
    int dim[8] = { 2, 2, 2, 1, 1, 1, 1, 1 };
    nifti_image* reference2D = nifti_make_new_nim(
        dim,
        NIFTI_TYPE_FLOAT32,
        true);
    reg_checkAndCorrectDimension(reference2D);

    // Create a reference 3D image
    dim[0] = 3;
    dim[3] = 2;
    nifti_image* reference3D = nifti_make_new_nim(
        dim,
        NIFTI_TYPE_FLOAT32,
        true);
    reg_checkAndCorrectDimension(reference3D);

    // Fill the reference images with integer values
    float* ref2DPtr = static_cast<float*>(reference2D->data);
    float* ref3DPtr = static_cast<float*>(reference3D->data);
    for (auto i = 0; i < reference2D->nvox; ++i) {
        ref2DPtr[i] = i;
    }
    for (auto i = 0; i < reference3D->nvox; ++i) {
        ref3DPtr[i] = i;
    }

    // Generate the different use cases
    std::vector<test_data> test_use_cases;

    test_use_cases.emplace_back(test_data(
        "Multiply 2D by 0",
        reference2D,
        0)
    );
    test_use_cases.emplace_back(test_data(
        "Multiply 2D by 1.5",
        reference2D,
        1.5f)
    );
    test_use_cases.emplace_back(test_data(
        "Multiply 2D by -12.1",
        reference2D,
        -12.1f)
    );
    test_use_cases.emplace_back(test_data(
        "Multiply 3D by 0",
        reference3D,
        0)
    );
    test_use_cases.emplace_back(test_data(
        "Multiply 3D by 1.5",
        reference3D,
        1.5f)
    );
    test_use_cases.emplace_back(test_data(
        "Multiply 3D by -12.1",
        reference3D,
        -12.1f)
    );

    // Loop over all generated test cases to run all tests
    for (auto&& test_use_case : test_use_cases) {
        // Retrieve test information
        std::string test_name;
        nifti_image* reference;
        float mult_value;
        std::tie(test_name, reference, mult_value) = test_use_case;

        // Create a nifti_image to store the result
        nifti_image* result = nifti_copy_nim_info(reference);
        result->data = (void*)malloc(result->nvox * result->nbyper);

        std::tie(test_name, reference, mult_value) = test_use_case;

        SECTION(test_name + " " + "CPU") {
            // Run the multiplication
            reg_tools_multiplyValueToImage(reference, result, mult_value);
            // Check all results
            float* resPtr = static_cast<float*>(result->data);
            float* refPtr = static_cast<float*>(reference->data);
            for (auto i = 0; i < result->nvox; ++i) {
                REQUIRE(fabs(resPtr[i] - refPtr[i] * mult_value) < EPS_SINGLE);
            }
        }

        SECTION(test_name + " " + "CUDA") {
            // Transfer the two nifti images onto the device
            float* reference_cuda;
            cudaCommon_allocateArrayToDevice<float>(&reference_cuda, reference->nvox);
            cudaCommon_transferNiftiToArrayOnDevice(&reference_cuda, reference);
            float* result_cuda;
            cudaCommon_allocateArrayToDevice<float>(&result_cuda, result->nvox);
            // Run the multiplication
            reg_tools_multiplyValueToImage(reference, result, mult_value);
            // Erase the content of the result image on host
            memset(result->data, 0, result->nvox * result->nbyper);
            // Transfer result image back from device
            cudaCommon_transferFromDeviceToNifti<float>(result, &result_cuda);            
            // Check all results
            float* resPtr = static_cast<float*>(result->data);
            float* refPtr = static_cast<float*>(reference->data);
            for (auto i = 0; i < result->nvox; ++i) {
                REQUIRE(fabs(resPtr[i] - refPtr[i] * mult_value) < EPS_SINGLE);
            }
            cudaCommon_free(&result_cuda);
            cudaCommon_free(&reference_cuda);
        }

        nifti_image_free(result);
    }
    nifti_image_free(reference2D);
    nifti_image_free(reference3D);
}
