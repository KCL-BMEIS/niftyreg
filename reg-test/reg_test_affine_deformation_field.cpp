#include "_reg_ReadWriteMatrix.h"
#include "_reg_tools.h"

#include "Kernel.h"
#include "AffineDeformationFieldKernel.h"
#include "Platform.h"

#include <catch2/catch_test_macros.hpp>

#include "AladinContent.h"
#ifdef _USE_CUDA
#include "CUDAAladinContent.h"
#endif
#ifdef _USE_OPENCL
#include "CLAladinContent.h"
#endif

#define EPS_SINGLE 0.0001

TEST_CASE("Affine deformation field", "[Affine]") {
    SECTION("2D") {
        // Create a reference 2D image
        int dim[8]= {2, 2, 2, 1, 1, 1, 1, 1};
        nifti_image *reference = nifti_make_new_nim(
                dim,
                NIFTI_TYPE_FLOAT32,
                true);
        reg_checkAndCorrectDimension(reference);

        // Generate the different use cases
        typedef std::tuple<std::string, mat44 *, float *, float *> test_data;
        std::vector<test_data> test_use_cases;
        // Identity use case
        auto *identity = new mat44;
        reg_mat44_eye(identity);
        // Test order [0,0] [1,0] [0,1] [1,1]
        float identity_result_x[4] = {0, 1, 0, 1};
        float identity_result_y[4] = {0, 0, 1, 1};
        test_use_cases.emplace_back(test_data(
                "- identity",
                identity,
                identity_result_x,
                identity_result_y)
        );
        // Translation
        auto *translation = new mat44;
        reg_mat44_eye(translation);
        translation->m[0][3] = -0.5;
        translation->m[1][3] = 1.5;
        // Test order [0,0] [1,0] [0,1] [1,1]
        float translation_result_x[4] = {-0.5, .5, -0.5, .5};
        float translation_result_y[4] = {1.5, 1.5, 2.5, 2.5};
        test_use_cases.emplace_back(test_data(
                "- translation",
                translation,
                translation_result_x,
                translation_result_y)
        );


        for(auto && test_use_case: test_use_cases) {

            std::string test_name;
            mat44 *test_mat;
            float *test_res_x;
            float *test_res_y;
            std::tie(test_name, test_mat, test_res_x, test_res_y) = test_use_case;

            SECTION(test_name) {
                typedef std::tuple<AladinContent *, std::string, int> content_desc;
                std::vector<content_desc> listContent;
                // Compute the transformation field
                listContent.push_back(content_desc(
                        new AladinContent(
                                reference,
                                nullptr,
                                nullptr,
                                test_mat,
                                sizeof(float)),
                        "-- CPU",
                        0));
#ifdef _USE_CUDA
                listContent.push_back(content_desc(
                        new CudaAladinContent(
                                reference,
                                nullptr,
                                nullptr,
                                test_mat,
                                sizeof(float)),
                        "-- CUDA",
                        1));
#endif
#ifdef _USE_OPENCL
                listContent.push_back(content_desc(
                        new ClAladinContent(
                                reference,
                                nullptr,
                                nullptr,
                                test_mat,
                                sizeof(float)),
                        "-- OpenCL",
                        2));
#endif
                for (auto &&content: listContent) {

                    AladinContent *con;
                    std::string desc;
                    int plat_value;
                    std::tie(con, desc, plat_value) = content;

                    SECTION(desc) {
                        auto *platform = new Platform(plat_value);

                        Kernel *affineDeformKernel = platform->createKernel(
                                AffineDeformationFieldKernel::getName(),
                                con);
                        affineDeformKernel->castTo<AffineDeformationFieldKernel>()->calculate();

                        nifti_image *defField =
                                con->getCurrentDeformationField();

                        // Check all values
                        auto *defFieldPtrX = static_cast<float *>(defField->data);
                        auto *defFieldPtrY = &defFieldPtrX[defField->nx *
                                                           defField->ny];
                        for (int i = 0; i < defField->nx*defField->ny; ++i) {
                            REQUIRE(fabs(
                                    defFieldPtrX[i] - test_res_x[i]) <
                                    EPS_SINGLE);
                            REQUIRE(fabs(
                                    defFieldPtrY[i] - test_res_y[i]) <
                                    EPS_SINGLE);
                        }
                        delete affineDeformKernel;
                        delete platform;
                    }
                    delete con;
                }
                listContent.clear();
            }
            delete test_mat;
        }
        test_use_cases.clear();
        nifti_image_free(reference);
    }
}

/*
    // Read the input reference image
    nifti_image *referenceImage = reg_io_ReadImageFile(inputRefImageName);
    if (referenceImage == NULL) {
        reg_print_msg_error("The input reference image could not be read");
        return EXIT_FAILURE;
    }
    // Read the input affine matrix
    mat44 *inputMatrix = (mat44 *)malloc(sizeof(mat44));
    reg_tool_ReadAffineFile(inputMatrix, inputMatFileName);

    // Read the input deformation field image image
    nifti_image *inputDeformationField = reg_io_ReadImageFile(inputDefImageName);
    if (inputDeformationField == NULL){
        reg_print_msg_error("The input deformation field image could not be read");
        return EXIT_FAILURE;
    }
    // Check the dimension of the input images
    if (referenceImage->nx != inputDeformationField->nx ||
            referenceImage->ny != inputDeformationField->ny ||
            referenceImage->nz != inputDeformationField->nz ||
            (referenceImage->nz > 1 ? 3 : 2) != inputDeformationField->nu){
        reg_print_msg_error("The input reference and deformation field images do not have corresponding sizes");
        return EXIT_FAILURE;
    }

    // Create a deformation field
    nifti_image *test_field = nifti_copy_nim_info(inputDeformationField);
    test_field->data = (void *) malloc(test_field->nvox*test_field->nbyper);

    // Compute the affine deformation field
    AladinContent *con = NULL;
    if (platformCode == NR_PLATFORM_CPU) {
        con = new AladinContent(referenceImage, NULL, NULL, inputMatrix, sizeof(float));
    }
#ifdef _USE_CUDA
    else if (platformCode == NR_PLATFORM_CUDA) {
        con = new CudaAladinContent(referenceImage, NULL, NULL, inputMatrix, sizeof(float));
    }
#endif
#ifdef _USE_OPENCL
    else if (platformCode == NR_PLATFORM_CL) {
        con = new ClAladinContent(referenceImage, NULL, NULL, inputMatrix, sizeof(float));
    }
#endif
    else {
        reg_print_msg_error("The platform code is not suppoted");
        return EXIT_FAILURE;
    }
    //Check if the platform used is double capable
    bool isDouble = con->isCurrentComputationDoubleCapable();
    double proper_eps = EPS;
    if(isDouble == 0) {
        proper_eps = EPS_SINGLE;
    }

    //CPU or GPU code
    reg_tools_changeDatatype<float>(referenceImage);
    test(con, platformCode);
    test_field = con->getCurrentDeformationField();

    // Compute the difference between the computed and inputed deformation field
    nifti_image *diff_field = nifti_copy_nim_info(inputDeformationField);
    diff_field->data = (void *) malloc(diff_field->nvox*diff_field->nbyper);
    reg_tools_substractImageToImage(inputDeformationField, test_field, diff_field);
    reg_tools_abs_image(diff_field);
    double max_difference = reg_tools_getMaxValue(diff_field, -1);

    nifti_image_free(referenceImage);
    nifti_image_free(inputDeformationField);

    delete con;
    free(inputMatrix);

    if (max_difference > proper_eps){
        fprintf(stderr, "reg_test_affine_deformation_field error too large: %g (>%g)\n",
                max_difference, proper_eps);
        return EXIT_FAILURE;
    }
#ifndef NDEBUG
    fprintf(stdout, "reg_test_affine_deformation_field ok: %g (<%g)\n",
            max_difference, proper_eps);
#endif

    return EXIT_SUCCESS;
}
 */

