#ifndef _REG_MRF_H
#define _REG_MRF_H

#include "_reg_measure.h"
#include <cmath>
#include <queue>
#include <algorithm>
#include "_reg_mrf_fastdt2.h"

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
   int dim;
   nifti_image* controlPointImage;
   int discrete_radius;
   int discrete_increment;
   float regularisation_weight;
   float *discretised_measure;
   //weights and indices of potential edges (6 per vertex in 3D - 4 in 2D)
   float *edgeWeightMatrix;
   float *index_neighbours;
   //
   int* orderedList;
   int* parentsList;
   float* edgeWeight;
   //
   float* regularisedCost;
   int* optimalDisplacement;
   //
   bool initialised;

   void Initialise();
   void GetDiscretisedMeasure();
   void Optimise();
   //
   void GetGraph();
   void GetPrimsMST();
   void GetRegularisation();
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
