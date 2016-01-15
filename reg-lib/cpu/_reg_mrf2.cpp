#include "_reg_mrf2.h"
/*****************************************************/
_reg_mrf2::_reg_mrf2(nifti_image *fixedImage,
                     nifti_image *movingImage,
                     nifti_image* controlPointImage,
                     int label_quant,
                     int label_hw,
                     float alphaValue)
{
    this->referenceImage = fixedImage;
    this->movingImage = movingImage;
    if(this->referenceImage->nz > 1) {
        this->dimImage = 3;
    } else {
        this->dimImage = 2;
    }
    //
    this->controlPointImage = controlPointImage;
    this->label_quant = label_quant;
    this->label_hw = label_hw;
    this->alpha = alphaValue;
    this->label_len=(label_hw*2+1); //length and total size of displacement space //13
    this->label_num = pow(label_hw*2+1,dimImage); //|L| number of displacements //169 in 2D -- 2197 in 3D
    this->displacement_array = new int[this->label_len];
    for(int i=0;i<this->label_len;i++){
        this->displacement_array[i]=-label_hw*label_quant+i*label_hw;
    }
#ifndef NDEBUG
    for(int i=0;i<this->label_len;i++){
        reg_print_msg_debug("displacement_array[i]="+this->displacement_array[i]);
    }
#endif
}
template <class DTYPE>
void _reg_mrf2::ComputeSimilarityCost()
{
    //warpedImage
    nifti_image* warpedImage = nifti_copy_nim_info(this->referenceImage);
    warpedImage->data = (void *)calloc(warpedImage->nvox,warpedImage->nbyper);

    //First: deform the moving image according to the current displacement
    nifti_image* deformationFieldImage = nifti_copy_nim_info(controlPointImage);
    deformationFieldImage->dim[0]=deformationFieldImage->ndim=5;
    deformationFieldImage->dim[4]=deformationFieldImage->nt=1;
    deformationFieldImage->dim[5]=deformationFieldImage->nu=this->controlPointImage->nz>1?3:2;
    deformationFieldImage->nvox = (size_t)deformationFieldImage->nx *
                             deformationFieldImage->ny *
                             deformationFieldImage->nz *
                             deformationFieldImage->nu;
    deformationFieldImage->datatype=NIFTI_TYPE_FLOAT32;
    deformationFieldImage->nbyper=sizeof(float);
    deformationFieldImage->data = (void *)calloc(deformationFieldImage->nvox,
                                            deformationFieldImage->nbyper);
    deformationFieldImage->intent_code=NIFTI_INTENT_VECTOR;
    memset(deformationFieldImage->intent_name, 0, 16);
    strcpy(deformationFieldImage->intent_name,"NREG_TRANS");
    deformationFieldImage->intent_p1==DISP_FIELD;

    float* deformationFieldImageData = static_cast<float> (deformationFieldImage->data);

    for(int z=0;z<controlPointImage->nz;z++){ //iterate over all control points
        for(int y=0;y<controlPointImage->ny;y++){
            for(int x=0;x<controlPointImage->nx;x++){

                for(int k=0;k<this->label_len;k++){ //iterate over all displacements
                    for(int j=0;j<this->label_len;j++){
                        for(int i=0;i<this->label_len;i++){
                            int dx = this->displacement_array[i];
                            int dy = this->displacement_array[j];
                            int dz = this->displacement_array[k];

                            int dfx = x+
                                      y*deformationFieldImage->nx+
                                      z*deformationFieldImage->nx*deformationFieldImage->ny+
                                      1*deformationFieldImage->nx*deformationFieldImage->ny*deformationFieldImage->nz+
                                      0*deformationFieldImage->nx*deformationFieldImage->ny*deformationFieldImage->nz*deformationFieldImage->nt;

                            int dfy = x+
                                      y*deformationFieldImage->nx+
                                      z*deformationFieldImage->nx*deformationFieldImage->ny+
                                      1*deformationFieldImage->nx*deformationFieldImage->ny*deformationFieldImage->nz+
                                      1*deformationFieldImage->nx*deformationFieldImage->ny*deformationFieldImage->nz*deformationFieldImage->nt;

                            int dfz = x+
                                      y*deformationFieldImage->nx+
                                      z*deformationFieldImage->nx*deformationFieldImage->ny+
                                      1*deformationFieldImage->nx*deformationFieldImage->ny*deformationFieldImage->nz+
                                      2*deformationFieldImage->nx*deformationFieldImage->ny*deformationFieldImage->nz*deformationFieldImage->nt;

                            deformationFieldImageData[dfx]=dx;
                            deformationFieldImageData[dfy]=dy;
                            deformationFieldImageData[dfz]=dz;

                            reg_getDeformationFromDisplacement(deformationFieldImage);

                            reg_resampleImage(this->movingImage,warpedImage,deformationFieldImage,NULL,1,0.0);

                            //Second: mask the reference image to get only the good part
                            int *maskImage = (int *)calloc(this->referenceImage->nvox, sizeof(int));


                            for(int i=0;i<this->referenceImage->nt;++i) {
                               dataCostmeasure->SetActiveTimepoint(i);
                            }

                            dataCostmeasure->InitialiseMeasure(this->referenceImage,
                                                              warpedImage,
                                                              maskImage,
                                                              warpedImage,
                                                              NULL,
                                                              NULL);

                            double measure=dataCostmeasure->GetSimilarityMeasureValue();

                            //Third - store in a nifty image

                        }
                    }
                }
            }

        }
    }


    nifti_image_free(deformationFieldImage);
    deformationFieldImage=NULL;
}
