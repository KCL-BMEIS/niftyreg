#ifndef _REG_MRF_H
#define _REG_MRF_H

#include "_reg_measure.h"

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
   float *edgeWeightMatrix;
   float *index_neighbours;
   bool initialised;

   void Initialise();
   void GetDiscretisedMeasure();
   void Optimise();
};

#endif // _REG_MRF_H
