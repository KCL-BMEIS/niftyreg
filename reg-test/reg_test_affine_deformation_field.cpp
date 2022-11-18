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

/*
    This test file contains the following unit tests:
    test function: creation of a deformation field from an affine matrix
    In 2D and 3D
    identity
    translation
    affine
*/


typedef std::tuple<std::string, nifti_image *, mat44 *, float *, float *, float *> test_data;
typedef std::tuple<AladinContent *, std::string, int> content_desc;

TEST_CASE("Affine deformation field", "[AffineDefField]") {
    // Create a reference 2D image
    int dim[8]= {2, 2, 2, 1, 1, 1, 1, 1};
    nifti_image *reference2D = nifti_make_new_nim(
            dim,
            NIFTI_TYPE_FLOAT32,
            true);
    reg_checkAndCorrectDimension(reference2D);

    // Create a reference 3D image
    dim[0]= 3;
    dim[3]= 2;
    nifti_image *reference3D = nifti_make_new_nim(
            dim,
            NIFTI_TYPE_FLOAT32,
            true);
    reg_checkAndCorrectDimension(reference3D);

    // Generate the different use cases
    std::vector<test_data> test_use_cases;

    // Identity use case - 2D
    auto *identity = new mat44;
    reg_mat44_eye(identity);
    // Test order [0,0] [1,0] [0,1] [1,1]
    float identity_result_2x[4] = {0, 1, 0, 1};
    float identity_result_2y[4] = {0, 0, 1, 1};
    test_use_cases.emplace_back(test_data(
            "identity 2D",
            reference2D,
            identity,
            identity_result_2x,
            identity_result_2y,
            nullptr)
    );
    // Identity use case - 3D
    // Test order [0,0,0] [1,0,0] [0,1,0] [1,1,0],[0,0,1] [1,0,1] [0,1,1] [1,1,1] 
    float identity_result_3x[8] = {0, 1, 0, 1, 0, 1, 0, 1};
    float identity_result_3y[8] = {0, 0, 1, 1, 0, 0, 1, 1};
    float identity_result_3z[8] = {0, 0, 0, 0, 1, 1, 1, 1};
    test_use_cases.emplace_back(test_data(
            "identity 3D",
            reference3D,
            identity,
            identity_result_3x,
            identity_result_3y,
            identity_result_3z)
    );

    // Translation - 2D
    auto *translation = new mat44;
    reg_mat44_eye(translation);
    translation->m[0][3] = -0.5;
    translation->m[1][3] = 1.5;
    translation->m[2][3] = 0.75;
    // Test order [0,0] [1,0] [0,1] [1,1]
    float translation_result_2x[4] = {-0.5, .5, -0.5, .5};
    float translation_result_2y[4] = {1.5, 1.5, 2.5, 2.5};
    test_use_cases.emplace_back(test_data(
            "translation 2D",
            reference2D,
            translation,
            translation_result_2x,
            translation_result_2y,
            nullptr)
    );

    // Translation - 3D
    // Test order [0,0,0] [1,0,0] [0,1,0] [1,1,0],[0,0,1] [1,0,1] [0,1,1] [1,1,1] 
    float translation_result_3x[8] = {-0.5, .5, -0.5, .5, -0.5, .5, -0.5, .5};
    float translation_result_3y[8] = {1.5, 1.5, 2.5, 2.5, 1.5, 1.5, 2.5, 2.5};
    float translation_result_3z[8] = {.75, .75, .75, .75, 1.75, 1.75, 1.75, 1.75};
    test_use_cases.emplace_back(test_data(
            "translation 3D",
            reference3D,
            translation,
            translation_result_3x,
            translation_result_3y,
            translation_result_3z)
    );

    
    // Full affine - 2D
    // Test order [0,0] [1,0] [0,1] [1,1]
    auto *affine = new mat44;
    reg_mat44_eye(affine);
    affine->m[0][3] = -0.5;
    affine->m[1][3] = 1.5;
    affine->m[2][3] = 0.75;
    for (auto i=0; i<4; ++i){
        for (auto j=0; j<4; ++j){
            affine->m[i][j] +=  static_cast<float>((((float) rand() / (RAND_MAX))-.5)/10.);
        }
    }
    float affine_result_2x[4];
    float affine_result_2y[4];
    for (auto i=0; i<4;++i){
        auto x = identity_result_2x[i];
        auto y = identity_result_2y[i];
        affine_result_2x[i] = affine->m[0][3] + affine->m[0][0]*x + affine->m[0][1]*y;
        affine_result_2y[i] = affine->m[1][3] + affine->m[1][0]*x + affine->m[1][1]*y;
        
    }
    test_use_cases.emplace_back(test_data(
            "full affine 2D",
            reference2D,
            affine,
            affine_result_2x,
            affine_result_2y,
            nullptr)
    );
    // Full affine - 3D
    // Test order [0,0,0] [1,0,0] [0,1,0] [1,1,0],[0,0,1] [1,0,1] [0,1,1] [1,1,1]
    float affine_result_3x[8];
    float affine_result_3y[8];
    float affine_result_3z[8];
    for (auto i=0; i<8;++i){
        auto x = identity_result_3x[i];
        auto y = identity_result_3y[i];
        auto z = identity_result_3z[i];
        affine_result_3x[i] = affine->m[0][3] +
            affine->m[0][0]*x + affine->m[0][1]*y + affine->m[0][2]*z;
        affine_result_3y[i] = affine->m[1][3] +
            affine->m[1][0]*x + affine->m[1][1]*y + affine->m[1][2]*z;
        affine_result_3z[i] = affine->m[2][3] +
            affine->m[2][0]*x + affine->m[2][1]*y + affine->m[2][2]*z;        
    }
    test_use_cases.emplace_back(test_data(
            "affine 3D",
            reference3D,
            affine,
            affine_result_3x,
            affine_result_3y,
            affine_result_3z)
    );

    // Loop over all generated test cases to create all content and run all tests
    for(auto && test_use_case: test_use_cases) {

        // Retrieve test information
        std::string test_name;
        nifti_image *reference;
        mat44 *test_mat;
        float *test_res_x;
        float *test_res_y;
        float *test_res_z;
        std::tie(test_name, reference, test_mat, test_res_x, test_res_y, test_res_z) = 
            test_use_case;

        // Accumate all required contents with a vector
        std::vector<content_desc> listContent;
        listContent.push_back(content_desc(
                new AladinContent(
                        reference,
                        nullptr,
                        nullptr,
                        test_mat,
                        sizeof(float)),
                "CPU",
                0));
#ifdef _USE_CUDA
        listContent.push_back(content_desc(
                new CudaAladinContent(
                        reference,
                        nullptr,
                        nullptr,
                        test_mat,
                        sizeof(float)),
                "CUDA",
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
                "OpenCL",
                2));
#endif
        // Loop over all possibles contents for each test
        for (auto &&content: listContent) {

            AladinContent *con;
            std::string desc;
            int plat_value;
            std::tie(con, desc, plat_value) = content;
            SECTION(test_name + " " + desc){
                // Initialise the platform to run current content and retrieve deformation field
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
                                                    defField->ny *
                                                    defField->nz];
                auto *defFieldPtrZ = &defFieldPtrY[defField->nx *
                                                    defField->ny *
                                                    defField->nz];
                for (int i = 0; i < defField->nx*defField->ny*defField->nz; ++i) {
                    REQUIRE(fabs(
                            defFieldPtrX[i] - test_res_x[i]) <
                            EPS_SINGLE);
                    REQUIRE(fabs(
                            defFieldPtrY[i] - test_res_y[i]) <
                            EPS_SINGLE);
                    if(test_res_z != nullptr){
                        REQUIRE(fabs(
                                defFieldPtrZ[i] - test_res_z[i]) <
                                EPS_SINGLE);
                    }
                }
                delete affineDeformKernel;
                delete platform;
                delete con;
            }
        }
        listContent.clear();
    }
    test_use_cases.clear();
    nifti_image_free(reference2D);
    nifti_image_free(reference3D);
    free(identity);
    free(translation);
    free(affine);
}
