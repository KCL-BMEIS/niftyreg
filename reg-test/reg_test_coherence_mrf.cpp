#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include "_reg_mrf.h"
#include "_reg_ReadWriteImage.h"
#include "_reg_localTrans.h"
#include "_reg_sad.h"
#include <numeric>
#include "mattias.h"
#include "_reg_mindssc.h"

int main(int argc, char **argv)
{
    if(argc!=3) {
        fprintf(stderr, "Usage: %s <refImage> <warpedFloatingImage>\n", argv[0]);
        return EXIT_FAILURE;
    }
    //IO
    char *inputRefImageName=argv[1];
    char *inputWarpedFlotingImageName=argv[2];
    //char *inputControlPointImage=argv[3];
    // Read the input images
    nifti_image *referenceImage = reg_io_ReadImageFile(inputRefImageName);
    if(referenceImage==NULL){
        reg_print_msg_error("The input reference image could not be read");
        return EXIT_FAILURE;
    }
    //this coherence test only works for 3-D images:
    int dim = referenceImage->nz > 1 ? 3 : 2;
    if (dim != 3) {
        reg_print_msg_error("This coherence test only works for 3-D images");
        reg_exit();
    }
    reg_tools_changeDatatype<float>(referenceImage);
    float* referenceImageDataPtr = static_cast<float*>(referenceImage->data);

    nifti_image *warpedFloatingImage = reg_io_ReadImageFile(inputWarpedFlotingImageName);
    if(warpedFloatingImage==NULL){
        reg_print_msg_error("The warped input floating image could not be read");
        return EXIT_FAILURE;
    }
    if (referenceImage->nx != warpedFloatingImage->nx &&
        referenceImage->ny != warpedFloatingImage->ny &&
        referenceImage->nz != warpedFloatingImage->nz) {
        reg_print_msg_error("This coherence test only works for 3-D images");
        reg_exit();
    }
    reg_tools_changeDatatype<float>(warpedFloatingImage);
    float* warpedFloatingImageDataPtr = static_cast<float*>(warpedFloatingImage->data);
    /********************************************************************************/
    int m,n,o,p; //dimensions (should be equal, ignore more than 3D)
    m=referenceImage->nx;
    n=referenceImage->ny;
    o=referenceImage->nz;
    int sz=m*n*o; //number of voxels

    //SETTINGS FOR CONTROL POINT SPACING AND LABEL SPACE
    int label_quant=3; //step-size/quantisation of discrete displacements
    int label_hw=6; //half-width of search space
    //L={±0,±label_quant,..,±label_quant*label_hw}^3 voxels
    int grid_step=8; //spacing between control points in voxels
    //HERE and elsewhere THE B-SPLINE GRID DEFINITION NEEDS TO BE ADAPTED TO NIFTY-REG
    //e.g. having control points outside the image domain, etc.

    int mind_hw=1; //patch-radius for SSD in MIND descriptors //2 before
    const int mind_length=12; //length of each descriptors (don't change)


    int label_num=(label_hw*2+1)*(label_hw*2+1)*(label_hw*2+1); //|L| number of displacements
    int m1=m/grid_step; int n1=n/grid_step; int o1=o/grid_step; //dimensions of grid
    int sz1=m1*n1*o1; //number of control points

    //convert images into their MIND representations (12 x m x n x o, dimensions)
    float* mind_fixed=new float[mind_length*sz];
    nifti_image *mind_fixedNifty = nifti_copy_nim_info(referenceImage);
    mind_fixedNifty->ndim = mind_fixedNifty->dim[0] = 4;
    mind_fixedNifty->nt = mind_fixedNifty->dim[4] = mind_length;
    mind_fixedNifty->nvox = mind_fixedNifty->nvox*mind_length;
    mind_fixedNifty->data=(void *)calloc(mind_fixedNifty->nvox,mind_fixedNifty->nbyper);
    float* mind_fixedNiftyDataPtr = static_cast<float*>(mind_fixedNifty->data);
    float* mind_moving=new float[mind_length*sz];

    //input each image (with dimensions m x n x o) and half_width of filter
    //Mattias version
    descriptor(mind_fixed,referenceImageDataPtr,m,n,o,mind_hw);
    for(int z=0;z<o;z++) {
        for (int y=0;y<n;y++) {
            for (int x=0;x<m;x++) {
                for (int l=0;l<mind_length;l++) {
                    mind_fixedNiftyDataPtr[(x+y*m+z*m*n+l*m*n*o)]=mind_fixed[l+(x+y*m+z*m*n)*mind_length];
                }
            }
        }
    }
    //DEBUG
    reg_io_WriteImageFile(mind_fixedNifty,"mind_fixedNifty.nii");
    //DEBUG
    descriptor(mind_moving,warpedFloatingImageDataPtr,m,n,o,mind_hw);
    //My version
    //MINDSSC image
    nifti_image *MINDSSC_refimg = nifti_copy_nim_info(referenceImage);
    MINDSSC_refimg->ndim = MINDSSC_refimg->dim[0] = 4;
    MINDSSC_refimg->nt = MINDSSC_refimg->dim[4] = mind_length;
    MINDSSC_refimg->nvox = MINDSSC_refimg->nvox*mind_length;
    MINDSSC_refimg->data=(void *)calloc(MINDSSC_refimg->nvox,MINDSSC_refimg->nbyper);
    float* MINDSSC_refimgDataPtr = static_cast<float*>(MINDSSC_refimg->data);

    // Compute the MIND descriptor
    int *mask_ref = (int *)calloc(referenceImage->nvox, sizeof(int));
    GetMINDSSCImageDesciptor(referenceImage,MINDSSC_refimg, mask_ref);
    free(mask_ref);
    //
    //MINDSSC image
    nifti_image *MINDSSC_warimg = nifti_copy_nim_info(warpedFloatingImage);
    MINDSSC_warimg->ndim = MINDSSC_warimg->dim[0] = 4;
    MINDSSC_warimg->nt = MINDSSC_warimg->dim[4] = mind_length;
    MINDSSC_warimg->nvox = MINDSSC_warimg->nvox*mind_length;
    MINDSSC_warimg->data=(void *)calloc(MINDSSC_warimg->nvox,MINDSSC_warimg->nbyper);
    float* MINDSSC_warimgDataPtr = static_cast<float*>(MINDSSC_warimg->data);

    // Compute the MIND descriptor
    int *mask_warped = (int *)calloc(warpedFloatingImage->nvox, sizeof(int));
    GetMINDSSCImageDesciptor(warpedFloatingImage,MINDSSC_warimg, mask_warped);
    free(mask_warped);
    //DEBUG
    reg_io_WriteImageFile(MINDSSC_refimg,"MINDSSC_refimg.nii");
    //DEBUG
    //Let's compare the 2 MINDSSC:
    for(int z=0;z<o;z++) {
        for (int y=0;y<n;y++) {
            for (int x=0;x<m;x++) {
                for (int l=0;l<mind_length;l++) {
                    std::cout<<"MIND VALUE 1="<<MINDSSC_refimgDataPtr[(x+y*m+z*m*n+l*m*n*o)]<<std::endl;
                    std::cout<<"MIND VALUE 2="<<mind_fixed[l+(x+y*m+z*m*n)*mind_length]<<std::endl;
                    bool areTheSame =
                            (MINDSSC_refimgDataPtr[(x+y*m+z*m*n+l*m*n*o)]==mind_fixed[l+(x+y*m+z*m*n)*mind_length]);
                    if(areTheSame == 1) {
                        reg_print_msg_debug("OKOKOK");
                    } else {
                        reg_print_msg_debug("NOTNOTNOT");
                    }
                }
            }
        }
    }

    //
    reg_print_msg_debug("---- MIND SSC coherence test ----");
    //Mattias version:

    //
    std::cout<<"---- computeSimilarityCost coherence test ----"<<std::endl;

    /*
    nifti_image *controlPointImage = reg_io_ReadImageFile(inputControlPointImage);
    if(controlPointImage==NULL){
        reg_print_msg_error("The input control point image could not be read");
        return EXIT_FAILURE;
    }
    reg_tools_changeDatatype<float>(controlPointImage);
    */
    //Let's create a control point image:
    nifti_image *controlPointImage = NULL;
    float spacing[3]={10.f,10.f,10.f};
    reg_createControlPointGrid<float>(&controlPointImage,
                               referenceImage,
                               spacing);
//    //
//    int label_quant = 3;
//    int label_hw = 6;
//    float alphaValue = 0.5;
//    //
//    reg_mrf* reg_mrfObject = new reg_mrf(referenceImage,floatingImage,controlPointImage,label_quant,label_hw,alphaValue);
//    std::cout << "object creation done" << std::endl;
//    reg_mrfObject->ComputeSimilarityCost();
//    std::cout << "Computing cost done" << std::endl;
//    reg_mrfObject->regularisationMST();
//    std::cout << "MST done" << std::endl;
//    reg_mrfObject->upsampleDisplacements();
//    std::cout << "upsample displacement done" << std::endl;
//    reg_mrfObject->warpFloatingImage();
//    std::cout << "warping done done" << std::endl;
    /*
    //reg_sad* sadMeasure = new reg_sad();
    reg_mrf2* reg_mrf2Object = new reg_mrf2(referenceImage,floatingImage,controlPointImage,label_quant,label_hw,alphaValue,sadMeasure);
    reg_mrf2Object->ComputeSimilarityCost();
    //Compare the 2 dataCost results:
    float* dataCost1 = reg_mrfObject->GetDataCost();
    nifti_image* dataCost2 = reg_mrf2Object->GetDataCost();
    float* dataCost2Data = static_cast<float*> (dataCost2->data);
    //Do the comparaison now
    float sumDiff = 0;
    float diff = 0;
    for(int i=0;i<dataCost2->nvox;i++) {
        //DEBUG
        std::cout<<"dataCost1[i]="<<dataCost1[i]<<std::endl;
        std::cout<<"dataCost2Data[i]="<<dataCost2Data[i]<<std::endl;
        //DEBUG
        diff = std::abs(dataCost2Data[i]-dataCost1[i]);
        sumDiff+=diff;
    }
    //FOR THE MOMENT - print sumDiff
    std::cout<<"sumDiff="<<sumDiff<<std::endl;
    */
    nifti_image_free(referenceImage);
    nifti_image_free(warpedFloatingImage);
    nifti_image_free(controlPointImage);

    return EXIT_SUCCESS;
}
