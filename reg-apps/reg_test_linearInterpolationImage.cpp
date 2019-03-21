//
// Created by Lucas Fidon on 07/03/19.
//

#include "_reg_tools.h"
#include "_reg_resampling.h"
#include "_reg_ReadWriteImage.h"
#include "_reg_localTrans.h"

#define EPS 1e-6
#define EPS_PERTURB 1e-3



int main(int argc, char **argv) {
    if (argc!=2) {
        std::cout << "Usage: " << argv[0] << " <floImage>" << std::endl;
        return EXIT_FAILURE;
    }

    // create the 2x2 2D image for the test
    nifti_image *img = NULL;
    img=reg_io_ReadImageFile(argv[1]);
    reg_tools_changeDatatype<float>(img);
    float *imgPtr = static_cast<float *>(img->data);
    std::cout << "img dimension: " << img->nx << " x " << img->ny << " x " << img->nz << std::endl;
    std::cout << "img spacing: " << img->dx << " x " << img->dy << " x " << img->dz << std::endl;
    imgPtr[0] = 3.f;
    imgPtr[1] = 4.f;
    imgPtr[2] = 1.f;
    imgPtr[3] = 2.f;

    // create the deformation field
    nifti_image* def = NULL;
    float gridSpacing[3];
    gridSpacing[0] = gridSpacing[1] = gridSpacing[2] = 1.f;
    def = nifti_copy_nim_info(img);
    def->dim[5] = def->nu = 2;
    def->nvox *= 2;
    def->intent_p1 = DEF_FIELD;
    def->scl_slope = 1.f;
    def->scl_inter = 0.f;
    def->data = (void *)calloc(def->nvox, def->nbyper);
    float *defPtr = static_cast<float *>(def->data);
    std::cout << "def dimension: " << def->nx << " x " << def->ny << " x " << def->nz  << " x " << def->nu << std::endl;

    defPtr[0] = 1. / 3.;
    defPtr[1] = 2. / 3.;
    defPtr[2] = 1. / 2;
    defPtr[3] = 1.;
    defPtr[4] = 1. / 3.;
    defPtr[5] = 1. / 6.;
    defPtr[6] = 1.;
    defPtr[7] = 1.;

    // test linear interpolation
    nifti_image *imgWarped = nifti_copy_nim_info(img);
    imgWarped->data = (void *)calloc(imgWarped->nvox, imgWarped->nbyper);
    float *imgWarpedPtr = static_cast<float *>(imgWarped->data);
    std::cout << "imgWarped dimension: " << imgWarped->nx << " x " << imgWarped->ny << " x " << imgWarped->nz << std::endl;
    reg_resampleImage(img,  // in
                      imgWarped,  // out
                      def,  // deformation field
                      NULL,  // mask
                      1,  // interpolation order
                      0);  // padding value
    std::cout << "imgWarped = [" << imgWarpedPtr[0] << " " << imgWarpedPtr[1] << " " << imgWarpedPtr[2]
            << " " << imgWarpedPtr[3] << "]" << std::endl;
    if (fabs(imgWarpedPtr[0] - (3. - 1./3.)) > EPS) {
//        EXIT_FAILURE;
          std::cout << "The linear interpolation test Failed..." << std::endl;
          reg_exit();
    }
    if (fabs(imgWarpedPtr[1] - (3. + 1./3.)) > EPS) {
//        EXIT_FAILURE;
        std::cout << "The linear interpolation test Failed..." << std::endl;
        reg_exit();
    }
    if (fabs(imgWarpedPtr[2] - (1. + 1./2.)) > EPS) {
//        EXIT_FAILURE;
        std::cout << "The linear interpolation test Failed..." << std::endl;
        reg_exit();
    }
    if (fabs(imgWarpedPtr[3] - 2.) > EPS) {
//        EXIT_FAILURE;
        std::cout << "The linear interpolation test Failed..." << std::endl;
        reg_exit();
    }

    // test gradient of linearly interpolated image
    defPtr[0] = 1. / 3.;
    defPtr[1] = 2. / 3.;
    defPtr[2] = 1. / 2;
    defPtr[3] = 1. - EPS_PERTURB;  // perturbation to avoid borders
    defPtr[4] = 1. / 3.;
    defPtr[5] = 1. / 6.;
    defPtr[6] = 1. - EPS_PERTURB;
    defPtr[7] = 1. - EPS_PERTURB;
    nifti_image *imgWarGrad = nifti_copy_nim_info(def);
    imgWarGrad->data = (void *)calloc(imgWarGrad->nvox, imgWarGrad->nbyper);
    float *imgWarGradPtr = static_cast<float *>(imgWarGrad->data);

    reg_getImageGradient(img,  // non warped image
                         imgWarGrad,  // out
                         def,  // deformation field
                         NULL,  // mask
                         1,  // interp
                         0,  // padding
                         0);
    std::cout << "imgWarGradX = [" << imgWarGradPtr[0] << " " << imgWarGradPtr[1] << " " << imgWarGradPtr[2]
              << " " << imgWarGradPtr[3] << "]" << std::endl;
    std::cout << "imgWarGradY = [" << imgWarGradPtr[4] << " " << imgWarGradPtr[5] << " " << imgWarGradPtr[6]
              << " " << imgWarGradPtr[7] << "]" << std::endl;
    for (int i=0; i<4; ++i) {
        if (fabs(imgWarGradPtr[i] - 1.) > EPS) {
//            EXIT_FAILURE;
            std::cout << "The gradient test Failed..." << std::endl;
            reg_exit();
        }
        if (fabs(imgWarGradPtr[4+i] + 2.) > EPS) {
//            EXIT_FAILURE;
            std::cout << "The gradient test Failed..." << std::endl;
            reg_exit();
        }
    }

    nifti_image_free(img);
    nifti_image_free(def);
    nifti_image_free(imgWarped);
    nifti_image_free(imgWarGrad);

    std::cout << "The Test passed!" << std::endl;
    EXIT_SUCCESS;
}