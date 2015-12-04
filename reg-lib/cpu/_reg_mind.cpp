/*
 *  _reg_mind.cpp
 *
 *
 *  Created by Benoit Presles on 01/12/2015.
 *  Copyright (c) 2015, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_mind.h"

/* *************************************************************** */
reg_mind::reg_mind()
    : reg_ssd()
{
    //4TM
    //platform = new Platform(NR_PLATFORM_CPU);
    //convolutionKernel = this->platform->createKernel(ConvolutionKernel::getName(), NULL);
#ifndef NDEBUG
   reg_print_msg_debug("reg_mind constructor called");
#endif
}
/* *************************************************************** */
void reg_mind::InitialiseMeasure(nifti_image *refImgPtr,
                                nifti_image *floImgPtr,
                                int *maskRefPtr,
                                nifti_image *warFloImgPtr,
                                nifti_image *warFloGraPtr,
                                nifti_image *forVoxBasedGraPtr,
                                int *maskFloPtr,
                                nifti_image *warRefImgPtr,
                                nifti_image *warRefGraPtr,
                                nifti_image *bckVoxBasedGraPtr)
{
   // Set the pointers using the parent class function
   reg_ssd::InitialiseMeasure(refImgPtr,
                                  floImgPtr,
                                  maskRefPtr,
                                  warFloImgPtr,
                                  warFloGraPtr,
                                  forVoxBasedGraPtr,
                                  maskFloPtr,
                                  warRefImgPtr,
                                  warRefGraPtr,
                                  bckVoxBasedGraPtr);

#ifndef NDEBUG
   reg_print_msg_debug("reg_mind::InitialiseMeasure().");
#endif
}
/* *************************************************************** */
reg_mind::~reg_mind() {}
/* *************************************************************** */
void reg_mind::GetVoxelBasedSimilarityMeasureGradient()
{

}
/* *************************************************************** */
double reg_mind::GetSimilarityMeasureValue() {
    return 0;
}
/* *************************************************************** */
template <class InputTYPE>
void reg_mind::GetMINDImageDesciptor1(nifti_image* inputImgPtr, nifti_image* MINDImgPtr) {

    InputTYPE* MINDImgDataPtr = (InputTYPE*) MINDImgPtr->data;
    //Mean image
    nifti_image *mean_img = nifti_copy_nim_info(inputImgPtr);
    mean_img->data=(void *)calloc(inputImgPtr->nvox,inputImgPtr->nbyper);
    InputTYPE* meanImgDataPtr = (InputTYPE*) mean_img->data;

    mat44* affineTransformation = (mat44 *)calloc(1,sizeof(mat44));

    // Create a deformation field from the matrix
    nifti_image *deformationFieldImage = nifti_copy_nim_info(inputImgPtr);
    deformationFieldImage->dim[0]=deformationFieldImage->ndim=5;
    deformationFieldImage->dim[1]=deformationFieldImage->nx=inputImgPtr->nx;
    deformationFieldImage->dim[2]=deformationFieldImage->ny=inputImgPtr->ny;
    deformationFieldImage->dim[3]=deformationFieldImage->nz=inputImgPtr->nz;
    deformationFieldImage->dim[4]=deformationFieldImage->nt=1;
    deformationFieldImage->pixdim[4]=deformationFieldImage->dt=1.0;
    deformationFieldImage->dim[5]=deformationFieldImage->nu=inputImgPtr->nz>1?3:2;
    deformationFieldImage->dim[6]=deformationFieldImage->nv=1;
    deformationFieldImage->dim[7]=deformationFieldImage->nw=1;
    deformationFieldImage->nvox =(size_t)deformationFieldImage->nx*
                                         deformationFieldImage->ny*deformationFieldImage->nz*
                                         deformationFieldImage->nt*deformationFieldImage->nu;
    deformationFieldImage->scl_slope=1.f;
    deformationFieldImage->scl_inter=0.f;
    deformationFieldImage->datatype = inputImgPtr->datatype;
    deformationFieldImage->nbyper = sizeof(InputTYPE);
    deformationFieldImage->data = (void *)calloc(deformationFieldImage->nvox, deformationFieldImage->nbyper);
    deformationFieldImage->intent_p1=DEF_FIELD;

    //warpedImage
    nifti_image *warpedImage = nifti_copy_nim_info(inputImgPtr);
    warpedImage->dim[0]=warpedImage->ndim=inputImgPtr->dim[0];
    warpedImage->dim[4]=warpedImage->nt=inputImgPtr->dim[4];
    warpedImage->cal_min=inputImgPtr->cal_min;
    warpedImage->cal_max=inputImgPtr->cal_max;
    warpedImage->scl_slope=inputImgPtr->scl_slope;
    warpedImage->scl_inter=inputImgPtr->scl_inter;
    warpedImage->datatype = inputImgPtr->datatype;
    warpedImage->nbyper = inputImgPtr->nbyper;
    warpedImage->nvox = (size_t)warpedImage->dim[1] * (size_t)warpedImage->dim[2] *
          (size_t)warpedImage->dim[3] * (size_t)warpedImage->dim[4];
    warpedImage->data = (void *)calloc(warpedImage->nvox, warpedImage->nbyper);

    //Convolution
    nifti_image *diff_image = nifti_copy_nim_info(inputImgPtr);
    diff_image->data = (void *) malloc(diff_image->nvox*diff_image->nbyper);

    float sigma = -0.5;//voxel based

    int dim = (inputImgPtr->nz > 1) ? 3 : 2;

    if (dim == 2) {
        //2D version
        const int samplingNbr = 4;
        int RSampling2D_x[samplingNbr] = {-1, 1,  0, 0};
        int RSampling2D_y[samplingNbr] = { 0, 0, -1, 1};

        for(int i=0;i<samplingNbr;i++) {
            reg_mat44_eye(affineTransformation);
            affineTransformation->m[0][3] = RSampling2D_x[i];//in mm
            affineTransformation->m[1][3] = RSampling2D_y[i];//in mm
            //But we want it in pixels ->
            affineTransformation->m[0][3] = affineTransformation->m[0][3]*inputImgPtr->dx;
            affineTransformation->m[1][3] = affineTransformation->m[1][3]*inputImgPtr->dy;
            //Invert maybe...
            //*affineTransformation = nifti_mat44_inverse(*affineTransformation);
            affineTransformation->m[2][2] = 0;

            // Initialise the deformation field with an identity transformation
            reg_tools_multiplyValueToImage(deformationFieldImage,deformationFieldImage,0.f);
            reg_getDeformationFromDisplacement(deformationFieldImage);
            //CREATE THE DEF FIELD from the affine matrix
            reg_affine_getDeformationField(affineTransformation,
                                           deformationFieldImage,
                                           false,
                                           NULL);
            /////////////////////////////////////////////////////////////////////////
            reg_resampleImage(inputImgPtr,
                              warpedImage,
                              deformationFieldImage,
                              NULL,
                              1,
                              0);
            ///////////////////////////////////////////////////////////////////////
            reg_tools_substractImageToImage(inputImgPtr, warpedImage, diff_image);

            reg_tools_multiplyImageToImage(diff_image, diff_image, diff_image);

            //this->convolutionKernel->castTo<ConvolutionKernel>()->calculate(diff_image, &sigma, 0, NULL, NULL, NULL);
            reg_tools_kernelConvolution(diff_image, &sigma, 0, NULL, NULL, NULL);
            //Mean
            reg_tools_addImageToImage(mean_img, diff_image, mean_img);

            //Let's store the result - we assume that MINDImgPtr has been well initialized
            //MIND datatype
            unsigned int index = i * diff_image->nvox;
            memcpy(&MINDImgDataPtr[index], diff_image->data, diff_image->nbyper * diff_image->nvox);
        }
        //Let's calculate the mean over the t values
        reg_tools_divideValueToImage(mean_img,mean_img,samplingNbr);
        //Let's calculate the MIND desccriptor
        for(int z=0;z<MINDImgPtr->nz;z++) {
            for(int y=0;y<MINDImgPtr->ny;y++) {
                for(int x=0;x<MINDImgPtr->nx;x++) {

                    int currentMeanIndex = x+
                                           mean_img->nx * y +
                                           mean_img->nx * mean_img->ny * z;
                    InputTYPE max_t = 0;
                    for(int t=0;t<MINDImgPtr->nt;t++) {
                            int currentMINDIndex = x+
                                                   MINDImgPtr->nx * y +
                                                   MINDImgPtr->nx * MINDImgPtr->ny * z +
                                                   MINDImgPtr->nx * MINDImgPtr->ny * MINDImgPtr->nz * t;
                            MINDImgDataPtr[currentMINDIndex] =
                                    exp(-MINDImgDataPtr[currentMINDIndex]/meanImgDataPtr[currentMeanIndex]);
                            max_t = std::max(max_t,MINDImgDataPtr[currentMINDIndex]);
                    }

                    for(int t=0;t<MINDImgPtr->nt;t++) {
                        int currentMINDIndex = x+
                                               MINDImgPtr->nx * y +
                                               MINDImgPtr->nx * MINDImgPtr->ny * z +
                                               MINDImgPtr->nx * MINDImgPtr->ny * MINDImgPtr->nz * t;

                        MINDImgDataPtr[currentMINDIndex] = MINDImgDataPtr[currentMINDIndex] / max_t;

                    }
                }
            }
        }

    } else if (dim == 3) {
        //3D version
        const int samplingNbr = 6;
        int RSampling3D_x[samplingNbr] = {-1, 1,  0, 0,  0, 0};
        int RSampling3D_y[samplingNbr] = {0,  0, -1, 1,  0, 0};
        int RSampling3D_z[samplingNbr] = {0,  0,  0, 0, -1, 1};

        for(int i=0;i<samplingNbr;i++) {
            reg_mat44_eye(affineTransformation);
            affineTransformation->m[0][3] = RSampling3D_x[i];
            affineTransformation->m[1][3] = RSampling3D_y[i];
            affineTransformation->m[2][3] = RSampling3D_z[i];
            //But we want it in pixels ->
            affineTransformation->m[0][3] = affineTransformation->m[0][3]*inputImgPtr->dx;
            affineTransformation->m[1][3] = affineTransformation->m[1][3]*inputImgPtr->dy;
            affineTransformation->m[2][3] = affineTransformation->m[2][3]*inputImgPtr->dz;
            // Update the sform
            /*
            if(inputImgPtr->sform_code>0)
            {
                inputImgCopy->sto_xyz = reg_mat44_mul(affineTransformation, &(inputImgPtr->sto_xyz));
            }
            else
            {
                inputImgCopy->sform_code = 1;
                inputImgCopy->sto_xyz = reg_mat44_mul(affineTransformation, &(inputImgPtr->qto_xyz));
            }
            inputImgCopy->sto_ijk = nifti_mat44_inverse(inputImgCopy->sto_xyz);
            */
            // Create a deformation field from the matrix
            // Initialise the deformation field with an identity transformation
            reg_tools_multiplyValueToImage(deformationFieldImage,deformationFieldImage,0.f);
            reg_getDeformationFromDisplacement(deformationFieldImage);
            //CREATE THE DEF FIELD from the affine matrix
            reg_affine_getDeformationField(affineTransformation,
                                                     deformationFieldImage,
                                                     false,
                                                     NULL);
            /////////////////////////////////////////////////////////////////////////
            reg_resampleImage(inputImgPtr,
                              warpedImage,
                              deformationFieldImage,
                              NULL,
                              1,
                              0);
            ///////////////////////////////////////////////////////////////////////
            //Convolution
            //I think I will have to make some padding - resampling before that...
            reg_tools_substractImageToImage(inputImgPtr, warpedImage, diff_image);
            reg_tools_multiplyImageToImage(diff_image, diff_image, diff_image);
            //this->convolutionKernel->castTo<ConvolutionKernel>()->calculate(diff_image, &sigma, 0, NULL, NULL, NULL);
            reg_tools_kernelConvolution(diff_image, &sigma, 0, NULL, NULL, NULL);
            //Mean
            reg_tools_addImageToImage(mean_img, diff_image, mean_img);

            //Let's store the result - we assume that MINDImgPtr has been well initialized
            //MIND datatype
            unsigned int index = i * diff_image->nvox;
            memcpy(&MINDImgDataPtr[index], diff_image->data, diff_image->nbyper * diff_image->nvox);
        }
        //Let's calculate the mean over the t values
        reg_tools_divideValueToImage(mean_img,mean_img,samplingNbr);
        //Let's calculate the MIND desccriptor
        for(int z=0;z<MINDImgPtr->nz;z++) {
            for(int y=0;y<MINDImgPtr->ny;y++) {
                for(int x=0;x<MINDImgPtr->nx;x++) {

                    int currentMeanIndex = x+
                                           mean_img->nx * y +
                                           mean_img->nx * mean_img->ny * z;
                    InputTYPE max_t = 0;
                    for(int t=0;t<MINDImgPtr->nt;t++) {
                            int currentMINDIndex = x+
                                                   MINDImgPtr->nx * y +
                                                   MINDImgPtr->nx * MINDImgPtr->ny * z +
                                                   MINDImgPtr->nx * MINDImgPtr->ny * MINDImgPtr->nz * t;
                            MINDImgDataPtr[currentMINDIndex] =
                                    exp(-MINDImgDataPtr[currentMINDIndex]/meanImgDataPtr[currentMeanIndex]);
                            max_t = std::max(max_t,MINDImgDataPtr[currentMINDIndex]);
                    }

                    for(int t=0;t<MINDImgPtr->nt;t++) {
                        int currentMINDIndex = x+
                                               MINDImgPtr->nx * y +
                                               MINDImgPtr->nx * MINDImgPtr->ny * z +
                                               MINDImgPtr->nx * MINDImgPtr->ny * MINDImgPtr->nz * t;

                        MINDImgDataPtr[currentMINDIndex] = MINDImgDataPtr[currentMINDIndex] / max_t;

                    }
                }
            }
        }

    } else {
        //Error
        reg_print_fct_error("The input image has to be a 2D or a 3D image");
    }
    //FREE MEMORY
    nifti_image_free(diff_image);
    nifti_image_free(warpedImage);
    nifti_image_free(deformationFieldImage);
    nifti_image_free(mean_img);
}
/* *************************************************************** */
void reg_mind::GetMINDImageDesciptor(nifti_image* inputImgPtr, nifti_image* MINDImgPtr) {
#ifndef NDEBUG
    reg_print_fct_debug("reg_mind -- GetMINDImageDesciptor()");
#endif
    //SECURITY
    if(inputImgPtr->datatype != MINDImgPtr->datatype) {
        reg_print_fct_error("reg_mind -- GetMINDImageDesciptor");
        reg_print_msg_error("The input image and the MIND image must have the same datatype !");
        reg_exit(EXIT_FAILURE);
    }

    switch (inputImgPtr->datatype)
    {
        case NIFTI_TYPE_FLOAT32:
            GetMINDImageDesciptor1<float>(inputImgPtr, MINDImgPtr);
            break;
        case NIFTI_TYPE_FLOAT64:
            GetMINDImageDesciptor1<double>(inputImgPtr, MINDImgPtr);
            break;
        default:
            reg_print_fct_error("reg_mind -- GetMINDImageDesciptor");
            reg_print_msg_error("Input image datatype not supported");
            reg_exit(EXIT_FAILURE);
            break;
    }
}
