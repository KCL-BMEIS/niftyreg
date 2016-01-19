#include "_reg_mrf2.h"
/*****************************************************/
reg_mrf2::reg_mrf2(nifti_image *fixedImage,
                   nifti_image *movingImage,
                   nifti_image* controlPointImage,
                   int label_quant,
                   int label_hw,
                   float alphaValue,
                   reg_measure* dataCostmeasure)
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
    this->dataCostmeasure = dataCostmeasure;
    //Output declaration:
    dataCost = nifti_copy_nim_info(controlPointImage);
    dataCost->data = (void *)malloc(controlPointImage->nvox*label_num*controlPointImage->nbyper);
    reg_tools_changeDatatype<float>(dataCost);
}
/**********************************************************************************************************/
nifti_image* reg_mrf2::GetDataCost() {
    return this->dataCost;
}
/**********************************************************************************************************/
void reg_mrf2::ComputeSimilarityCost()
{
    //warpedImage
    nifti_image* warpedImage = nifti_copy_nim_info(this->referenceImage);
    warpedImage->data = (void *)calloc(warpedImage->nvox,warpedImage->nbyper);

    //First: deform the moving image according to the current displacement
    nifti_image* deformationFieldImage = nifti_copy_nim_info(controlPointImage);
    deformationFieldImage->dim[0]=deformationFieldImage->ndim=5;
    deformationFieldImage->dim[4]=deformationFieldImage->nt=1;
    deformationFieldImage->dim[5]=deformationFieldImage->nu=dimImage;
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

    float* deformationFieldImageData = static_cast<float*> (deformationFieldImage->data);
    float* dataCostData = static_cast<float*> (dataCost->data);

    if(dimImage == 3) {
        for(int z=0;z<controlPointImage->nz-1;z++){ //iterate over all control points
            for(int y=0;y<controlPointImage->ny-1;y++){
                for(int x=0;x<controlPointImage->nx-1;x++){
                    int controlPoint_index = x + y*controlPointImage->nx + z*controlPointImage->nx*controlPointImage->ny;

                    for(int k=0;k<this->label_len;k++){ //iterate over all displacements
                        for(int j=0;j<this->label_len;j++){
                            for(int i=0;i<this->label_len;i++){

                                int dx = this->displacement_array[i];
                                int dy = this->displacement_array[j];
                                int dz = this->displacement_array[k];
                                int displacement_index = dx + dy*this->label_len + dz*label_len*label_len;

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
                                //void reg_tools_binaryImage2int(nifti_image *img, int *array, int &activeVoxelNumber);
                                nifti_image* maskImage = nifti_copy_nim_info(this->referenceImage);
                                maskImage->data = (void *)calloc(maskImage->nvox,maskImage->nbyper);
                                reg_tools_changeDatatype<int>(maskImage);
                                int* maskImageData = static_cast<int*> (maskImage->data);
                                //control points in pixel to reference in pixel
                                int xm_i = x*controlPointImage->dx/referenceImage->dx;
                                int xm_e = (x+1)*controlPointImage->dx/referenceImage->dx;
                                int ym_i = y*controlPointImage->dy/referenceImage->dy;
                                int ym_e = (y+1)*controlPointImage->dy/referenceImage->dy;
                                int zm_i = z*controlPointImage->dz/referenceImage->dz;
                                int zm_e = (z+1)*controlPointImage->dz/referenceImage->dz;
                                for(int zm=zm_i;zm<zm_e;zm++) {
                                    for(int ym=ym_i;ym<ym_e;ym++) {
                                        for(int xm=xm_i;xm<xm_e;xm++) {
                                            maskImageData[xm+ym*maskImage->nx+zm*maskImage->ny*maskImage->nz]=1;
                                        }
                                    }
                                }
                                int *maskArray = (int *)calloc(this->referenceImage->nvox, sizeof(int));
                                int activeVoxelNumber=0;
                                reg_tools_binaryImage2int(maskImage, maskArray, activeVoxelNumber);

                                for(int i=0;i<this->referenceImage->nt;++i) {
                                    dataCostmeasure->SetActiveTimepoint(i);
                                }

                                dataCostmeasure->InitialiseMeasure(this->referenceImage,
                                                                   warpedImage,
                                                                   maskArray,
                                                                   warpedImage,
                                                                   NULL,
                                                                   NULL);

                                double measure=dataCostmeasure->GetSimilarityMeasureValue();
                                //REVERT THE MOVE
                                deformationFieldImageData[dfx]=0;
                                deformationFieldImageData[dfy]=0;
                                deformationFieldImageData[dfz]=0;
                                //Third - store in a nifty image
                                dataCostData[displacement_index+label_num*(controlPoint_index)]=measure;//4D image;
                            }
                        }
                    }
                }

            }
        }
    } else {
        //iterate over all control points
        for(int y=0;y<controlPointImage->ny-1;y++){
            for(int x=0;x<controlPointImage->nx-1;x++){
                int controlPoint_index = x + y*controlPointImage->nx;

                //iterate over all displacements
                for(int j=0;j<this->label_len;j++){
                    for(int i=0;i<this->label_len;i++){

                        int dx = this->displacement_array[i];
                        int dy = this->displacement_array[j];

                        int displacement_index = dx + dy*this->label_len;

                        int dfx = x+
                                  y*deformationFieldImage->nx+
                                  1*deformationFieldImage->nx*deformationFieldImage->ny*deformationFieldImage->nz+
                                  0*deformationFieldImage->nx*deformationFieldImage->ny*deformationFieldImage->nz*deformationFieldImage->nt;

                        int dfy = x+
                                  y*deformationFieldImage->nx+
                                  1*deformationFieldImage->nx*deformationFieldImage->ny*deformationFieldImage->nz+
                                  1*deformationFieldImage->nx*deformationFieldImage->ny*deformationFieldImage->nz*deformationFieldImage->nt;

                        int dfz = x+
                                  y*deformationFieldImage->nx+
                                  1*deformationFieldImage->nx*deformationFieldImage->ny*deformationFieldImage->nz+
                                  2*deformationFieldImage->nx*deformationFieldImage->ny*deformationFieldImage->nz*deformationFieldImage->nt;

                        deformationFieldImageData[dfx]=dx;
                        deformationFieldImageData[dfy]=dy;

                        reg_getDeformationFromDisplacement(deformationFieldImage);

                        reg_resampleImage(this->movingImage,warpedImage,deformationFieldImage,NULL,1,0.0);

                        //Second: mask the reference image to get only the good part
                        //void reg_tools_binaryImage2int(nifti_image *img, int *array, int &activeVoxelNumber);
                        nifti_image* maskImage = nifti_copy_nim_info(this->referenceImage);
                        maskImage->data = (void *)calloc(maskImage->nvox,maskImage->nbyper);
                        reg_tools_changeDatatype<int>(maskImage);
                        int* maskImageData = static_cast<int*> (maskImage->data);
                        //control points in pixel to reference in pixel
                        int xm_i = x*controlPointImage->dx/referenceImage->dx;
                        int xm_e = (x+1)*controlPointImage->dx/referenceImage->dx;
                        int ym_i = y*controlPointImage->dy/referenceImage->dy;
                        int ym_e = (y+1)*controlPointImage->dy/referenceImage->dy;
                        for(int ym=ym_i;ym<ym_e;ym++) {
                            for(int xm=xm_i;xm<xm_e;xm++) {
                                maskImageData[xm+ym*maskImage->nx]=1;
                            }
                        }
                        int *maskArray = (int *)calloc(this->referenceImage->nvox, sizeof(int));
                        int activeVoxelNumber=0;
                        reg_tools_binaryImage2int(maskImage, maskArray, activeVoxelNumber);

                        for(int i=0;i<this->referenceImage->nt;++i) {
                            dataCostmeasure->SetActiveTimepoint(i);
                        }

                        dataCostmeasure->InitialiseMeasure(this->referenceImage,
                                                           warpedImage,
                                                           maskArray,
                                                           warpedImage,
                                                           NULL,
                                                           NULL);

                        double measure=dataCostmeasure->GetSimilarityMeasureValue();
                        //REVERT THE MOVE
                        deformationFieldImageData[dfx]=0;
                        deformationFieldImageData[dfy]=0;
                        //Third - store in a nifty image
                        dataCostData[displacement_index+label_num*(controlPoint_index)]=measure;//4D image;
                    }
                }
            }
        }
    }


    nifti_image_free(deformationFieldImage);
    deformationFieldImage=NULL;
}
