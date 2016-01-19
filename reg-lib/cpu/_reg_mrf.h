#ifndef _REG_MRF_H
#define _REG_MRF_H

#include "_reg_measure.h"
#include <cmath>

struct Edge{
    float weight;
    int startIndex;
    int endIndex;
    friend bool operator<(Edge a,Edge b){
        return a.weight>b.weight;
    }
};

class reg_mrf
{
public:
   reg_mrf(reg_measure *_measure,
           nifti_image *_controlPointImage,
           int discrete_radius,
           int _discrete_increment,
           float _reg_weight);
   ~reg_mrf();
   void Run();

private:
   reg_measure *measure;
   nifti_image* controlPointImage;
   int discrete_radius;
   int discrete_increment;
   float regularisation_weight;
   float *discretised_measure;
   //weights and indices of potential edges (six per vertex in 3D)
   float *edgeWeightMatrix;
   float *index_neighbours;
   float* edgeWeight;
   bool initialised;

   void Initialise();
   void GetGraph(nifti_image* controlPointGridImage,
                 float* edgeWeightMatrix,
                 float* index_neighbours);
   void GetDiscretisedMeasure();
   void Optimise();
};
/********************************************************************************************************/
extern "C++"
template <class DTYPE>
void GetGraph_core3D(nifti_image* controlPointGridImage,
                     float* edgeWeightMatrix,
                     float* index_neighbours,
                     nifti_image *refImage,
                     int *mask);
extern "C++"
template <class DTYPE>
void GetGraph_core2D(nifti_image* controlPointGridImage,
                     float* edgeWeightMatrix,
                     float* index_neighbours,
                     nifti_image *refImage,
                     int *mask);
/********************************************************************************************************/
#endif // _REG_MRF_H
