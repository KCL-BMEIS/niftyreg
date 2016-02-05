#ifndef _REG_MRF_H
#define _REG_MRF_H

#include "_reg_measure.h"
#include <cmath>
#include <queue>
#include <algorithm>

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
           nifti_image *_referenceImage,
           nifti_image *_controlPointImage,
           int discrete_radius,
           int _discrete_increment,
           float _reg_weight);

   ~reg_mrf();
   void Run();
   //For the unit tests - let's put these functions "public"
   //but only Run() should be called.
   void Initialise();
   void GetDiscretisedMeasure();
   void Optimise();
   //
   void GetGraph();
   void GetPrimsMST();
   void GetRegularisation();
   //
   float* GetDiscretisedMeasurePtr();

private:
   reg_measure *measure;
   int dim;
   nifti_image* referenceImage;
   nifti_image* controlPointImage;
   int discrete_radius;
   int discrete_increment;
   int *discrete_valueArray;//int because voxel displacements
   //
   float *discretised_measure;
   float regularisation_weight;
   //weights and indices of potential edges (6 per vertex in 3D - 4 in 2D)
   float *edgeWeightMatrix;
   float *index_neighbours;
   //
   int* orderedList;
   int* parentsList;
   float* edgeWeight;
   //
   float* regularisedCost;
   int* optimalDisplacement;//int because voxel displacements
   //
   bool initialised;
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

extern "C++"
void dt1sq(float *val,int* ind,int len,float offset,int k,int* v,float* z,float* f,int* ind1);
extern "C++"
void dt3x(float* r,int* indr,int rl,float dx,float dy,float dz);
/********************************************************************************************************/
#endif // _REG_MRF_H
