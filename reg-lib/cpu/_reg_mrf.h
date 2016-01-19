#ifndef _REG_MRF_H
#define _REG_MRF_H

#include "nifti1_io.h"
#include "_reg_measure.h"
#include "_reg_resampling.h"

//FOR THE MOMENT -- HAS TO PUT IF USE SSE
#include <emmintrin.h>
#include <xmmintrin.h>

//STD
#include <queue>
#include <algorithm>
#include <math.h>

class reg_mrf
{
    public:
        //Constructor
        reg_mrf(nifti_image* referenceImage,
                 nifti_image* movingImage,
                 nifti_image* controlPointImage,
                 int label_quant,
                 int label_hw,
                 float alphaValue);

        //Destructor
        ~reg_mrf();

        //Getters and Setters
        int GetLabel_quant();
        void SetLabel_quant(int label_quant);

        int GetLabel_hw();
        void SetLabel_hw(int label_hw);

        float* GetDataCost();

        //int GetGrid_step();
        //void SetGrid_step(int grid_step);

        //Calculate the similarity (MIND-SAD) for every control point and every displacement
        void ComputeSimilarityCost();

        //uses squared difference penalty
        void regularisationMST();

        //upsampleDisplacements
        void upsampleDisplacements();

        //upsampleDisplacements
        void warpFloatingImage();

        //Run function
        //void Run();

    protected:
        nifti_image* referenceImage;
        nifti_image* floatingImage;
        nifti_image* controlPointImage;
        //SETTINGS FOR CONTROL POINT SPACING AND LABEL SPACE
        int label_quant; //step-size/quantisation of discrete displacements - default = 3
        int label_hw; //half-width of search space // default = 6
        //L={±0,±label_quant,..,±label_quant*label_hw}^3 voxels

        float alpha; //smoothness of displacement field, higher value smoother field

        nifti_image* warpedImage;

    private:
        int dimImage;
        int grid_step[3]; //spacing between control points in voxels - default 8
        int label_len; //default 6*2+1
        int label_num; //label_len^dimImage
        //int m1=m/grid_step[0]; int n1=n/grid_step[1]; int o1=o/grid_step[2]; //dimensions of grid
        //int sz1=m1*n1*o1; //number of control points
        float* dataCost;
        float* regularisedCost;
        int* optimalDisplacement;

        float* fieldLR; //displacement field (on grid level)
        float* fieldHR; //(and dense voxels) three components for 3D displacement

};
/********************************************************************************************************/
struct Edge{
    float weight;
    int startIndex;
    int endIndex;
    friend bool operator<(Edge a,Edge b){
        return a.weight>b.weight;
    }
};
/********************************************************************************************************/
extern "C++"
void edgeGraph(float* edgeWeightMatrix, int* index_neighbours, float* fixed, int m, int n, int o, int* grid_step);
/********************************************************************************************************/
extern "C++"
void primsMST(int* orderedList,int* parentsList,float* edgeWeight,float* edgeWeightMatrix,int* index_neighbours,int m1,int n1, int o1);
/********************************************************************************************************/
extern "C++"
void regularisation(float* marginals,int* selected,float* dataCost,int* ordered,int* parents,float* edgeweights,int label_hw,int m1,int n1, int o1);
/********************************************************************************************************/
extern "C++"
void dt3x(float* r,int* indr,int rl,float dx,float dy,float dz);
/********************************************************************************************************/
extern "C++"
void interp3(float* interp,float* input,float* x1,float* y1,float* z1,int m,int n,int o,int m2,int n2,int o2,bool flag);
#endif // _REG_MRF_H
