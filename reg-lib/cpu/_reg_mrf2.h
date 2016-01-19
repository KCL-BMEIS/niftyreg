#ifndef _REG_MRF2_H
#define _REG_MRF2_H

#include "_reg_measure.h"
#include "_reg_resampling.h"

class reg_mrf2
{
    public:
        reg_mrf2(nifti_image* fixedImage,
                  nifti_image* movingImage,
                  nifti_image *controlPointImage,
                  int label_quant,
                  int label_hw,
                  float alphaValue,
                  reg_measure *dataCostmeasure);

        nifti_image* GetDataCost();

        void ComputeSimilarityCost();

    protected:
        nifti_image* referenceImage;
        nifti_image* movingImage;
        nifti_image* controlPointImage;
        //SETTINGS FOR CONTROL POINT SPACING AND LABEL SPACE
        int label_quant; //step-size/quantisation of discrete displacements - default = 3
        int label_hw; //half-width of search space // default = 6
        //L={±0,±label_quant,..,±label_quant*label_hw}^3 voxels
        float alpha; //smoothness of displacement field, higher value smoother field
        //MEASURE OF SIMILARITY FOR THE data term
        reg_measure* dataCostmeasure;

    private:
        int dimImage;
        int label_len;
        int label_num;
        int* displacement_array;
        //Output
        nifti_image* dataCost;
};

#endif // _REG_MRF2_H
