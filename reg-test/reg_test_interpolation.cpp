#include "_reg_ReadWriteMatrix.h"
#include "_reg_tools.h"

#include "Kernel.h"
#include "ResampleImageKernel.h"
#include "Platform.h"

#include <list>
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
    test function: image resampling
    In 2D and 3D
    identity
    translation
    affine
*/


typedef std::tuple<std::string, nifti_image*, nifti_image*, float*> test_data;
typedef std::tuple<AladinContent*, std::string, int> content_desc;

TEST_CASE("Resampling", "[resampling]") {
    // Create a reference 2D image
    int dim[8] = { 2, 2, 2, 1, 1, 1, 1, 1 };
    nifti_image* reference2D = nifti_make_new_nim(
        dim,
        NIFTI_TYPE_FLOAT32,
        true);
    reg_checkAndCorrectDimension(reference2D);

    // Fill image with distance from identity
    auto* ref2dPrt = static_cast<float*>(reference2D->data);
    for (float y = 0; y<reference2D->ny; ++y) {
        for (float x = 0; x < reference2D->nx; ++x) {
            *ref2dPrt = sqrtf(x*x + y*y);
            ref2dPrt++;
        }
    }

    // Create a corresponding deformation field

    // Create a reference 3D image
    dim[0] = 3; dim[3] = 2;
    nifti_image* reference3D = nifti_make_new_nim(
        dim,
        NIFTI_TYPE_FLOAT32,
        true);
    reg_checkAndCorrectDimension(reference3D);

    // Fill image with distance from identity
    auto* ref3dPrt = static_cast<float*>(reference3D->data);
    for (float z = 0; z < reference3D->nz; ++z) {
        for (float y = 0; y < reference3D->ny; ++y) {
            for (float x = 0; x < reference3D->nx; ++x) {
                *ref3dPrt = sqrtf(x * x + y * y + z * z);
                ref3dPrt++;
            }
        }
    }

    // Generate the different use cases
    std::vector<test_data> test_use_cases;

    // Identity use case - 2D
    // First create an identity displacement field and then convert it into a deformation
    nifti_image* id_field_2D = nifti_copy_nim_info(reference2D);
    id_field_2D->ndim = id_field_2D->dim[0] = 5;
    id_field_2D->nu = id_field_2D->dim[5] = 2;
    id_field_2D->nvox = id_field_2D->nx * id_field_2D->ny * id_field_2D->nu;
    id_field_2D->data = (void *)calloc(id_field_2D->nvox, id_field_2D->nbyper);
    reg_getDeformationFromDisplacement(id_field_2D);
    float res2[4];
    memcpy(res2, reference2D->data, reference2D->nvox*sizeof(float));
    // create the test case
    test_use_cases.emplace_back(test_data(
        "identity 2D",
        reference2D,
        id_field_2D,
        res2)
    );

    // Identity use case - 3D
    nifti_image* id_field_3D = nifti_copy_nim_info(reference3D);
    id_field_3D->ndim = id_field_3D->dim[0] = 5;
    id_field_3D->nu = id_field_3D->dim[5] = 3;
    id_field_3D->nvox = id_field_3D->nx * id_field_3D->ny * id_field_3D->nz * id_field_3D->nu;
    id_field_3D->data = (void*)calloc(id_field_3D->nvox, id_field_3D->nbyper);
    reg_getDeformationFromDisplacement(id_field_3D);
    float res3[8];
    memcpy(res3, reference3D->data, reference3D->nvox * sizeof(float));
    // create the test case
    test_use_cases.emplace_back(test_data(
        "identity 3D",
        reference3D,
        id_field_3D,
        res3)
    );

    // Loop over all generated test cases to create all content and run all tests
    for (auto&& test_use_case : test_use_cases) {

        // Retrieve test information
        std::string test_name;
        nifti_image *reference;
        nifti_image *def_field;
        float *test_res;
        std::tie(test_name, reference, def_field, test_res) =
            test_use_case;

        // Accumate all required contents with a vector
        std::vector<content_desc> listContent;
        listContent.push_back(content_desc(
            new AladinContent(
                reference,
                reference,
                nullptr,
                sizeof(float)),
            "CPU",
            NR_PLATFORM_CPU));
#ifdef _USE_CUDA
        listContent.push_back(content_desc(
            new CudaAladinContent(
                reference,
                reference,
                nullptr,
                sizeof(float)),
            "CUDA",
            NR_PLATFORM_CUDA));
#endif
#ifdef _USE_OPENCL
        listContent.push_back(content_desc(
            new ClAladinContent(
                reference,
                reference,
                nullptr,
                sizeof(float)),
            "OpenCL",
            NR_PLATFORM_CL));
#endif
        // Loop over all possibles contents for each test
        for (auto&& content : listContent) {

            AladinContent* con;
            std::string desc;
            int plat_value;
            std::tie(con, desc, plat_value) = content;

            SECTION(test_name + " " + desc) {
                // Create and set a warped image to host the computation
                nifti_image* warped = nifti_copy_nim_info(reference);
                warped->data = (void*)malloc(warped->nvox * warped->nbyper);
                con->setCurrentWarped(warped);
                // Set the deformation field
                con->setCurrentDeformationField(def_field);
                // Set an empty mask to consider all voxels
                int* tempMask = (int*)calloc(reference->nvox, sizeof(int));
                con->setCurrentReferenceMask(tempMask, warped->nvox);
                // Initialise the platform to run current content and retrieve deformation field
                auto* platform = new Platform(plat_value);
                Kernel* resampleKernel = platform->createKernel(
                    ResampleImageKernel::getName(),
                    con);
                // args = interpolation and padding
                std::list<int> interp = { 0, 1, 3 };
                for (auto it : interp) {
                    resampleKernel->castTo<ResampleImageKernel>()->calculate(
                        it,
                        0);
                    warped = con->getCurrentWarped(reference->datatype);

                    // Check all values
                    auto* warpedPtr = static_cast<float*>(warped->data);
                    for (int i = 0; i < warped->nx * warped->ny * warped->nz; ++i) {
                        std::cout << i << " " << static_cast<float*>(reference->data)[i] << " " << warpedPtr[i] << " " << test_res[i] << std::endl;
                        REQUIRE(fabs(
                            warpedPtr[i] - test_res[i]) <
                            EPS_SINGLE);
                    }
                }
                delete resampleKernel;
                delete platform;
                free(tempMask);
                delete con;
            }
        }
        listContent.clear();
    }
    test_use_cases.clear();
    // Only free-ing ref as the rest if cleared by content destructor
    nifti_image_free(reference2D);
    nifti_image_free(reference3D);
}
