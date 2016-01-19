#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include "_reg_mrf.h"
#include "_reg_ReadWriteImage.h"
#include "_reg_localTrans.h"
#include "_reg_sad.h"

int main(int argc, char **argv)
{
    if(argc!=3)
    {
        fprintf(stderr, "Usage: %s <refImage> <floImage>\n", argv[0]);
        return EXIT_FAILURE;
    }
    //
    std::cout<<"---- ComputeSimilarityCost coherence test ----"<<std::endl;
    //IO
    char *inputRefImageName=argv[1];
    char *inputFloImageName=argv[2];
    //char *inputControlPointImage=argv[3];
    // Read the input images
    nifti_image *referenceImage = reg_io_ReadImageFile(inputRefImageName);
    if(referenceImage==NULL){
        reg_print_msg_error("The input reference image could not be read");
        return EXIT_FAILURE;
    }
    reg_tools_changeDatatype<float>(referenceImage);
    nifti_image *floatingImage = reg_io_ReadImageFile(inputFloImageName);
    if(floatingImage==NULL){
        reg_print_msg_error("The input floating image could not be read");
        return EXIT_FAILURE;
    }
    reg_tools_changeDatatype<float>(floatingImage);

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
    nifti_image_free(floatingImage);
    nifti_image_free(controlPointImage);

    return EXIT_SUCCESS;
}
