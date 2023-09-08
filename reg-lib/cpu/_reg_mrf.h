/**
 * @file _reg_mrf.h
 * @author Benoit Presles
 * @author Mattias Heinrich
 * @date 01/01/2016
 * @brief reg_mrf class for discrete optimisation
 *
 * Copyright (c) 2016-2018, University College London
 * Copyright (c) 2018, NiftyReg Developers.
 * All rights reserved.
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#pragma once

#include "_reg_measure.h"
#include "_reg_localTrans_regul.h"
#include <cmath>
#include <queue>
#include <algorithm>
#include "_reg_maths.h"

struct Edge{
   float weight;
   int startIndex;
   int endIndex;
   friend bool operator<(Edge a,Edge b){
      return a.weight>b.weight;
      //return a.weight<b.weight;
   }
};

class reg_mrf
{
public:
    /// @brief Constructor
   reg_mrf(int _discrete_radius,
           int _discrete_increment,
           float _reg_weight,
           int _img_dim,
           size_t _node_number);
   /// @brief Constructor
   reg_mrf(reg_measure *_measure,
           nifti_image *_referenceImage,
           nifti_image *_controlPointImage,
           int discrete_radius,
           int _discrete_increment,
           float _reg_weight);
   /// @brief Destructor
   ~reg_mrf();
   void Run();
   //4 the tests
   void GetDiscretisedMeasure();
   float* GetDiscretisedMeasurePtr();
   void SetDiscretisedMeasure(float* dm);
   //
   void GetRegularisation();
   //
   void GetOptimalLabel();
   int* GetOptimalLabelPtr();
   //
   int* GetOrderedListPtr();
   int* GetParentsListPtr();
   float* GetEdgeWeightPtr();
   //
   void SetOrderedList(int* ol);
   void SetParentsList(int* pl);
   void SetEdgeWeight(float* ew);
   //
   void GetPrimsMST(float *, int *, int, int, bool);

private:
   void Initialise();
   void UpdateNodePositions();
   void GetGraph(float *, int *);

   reg_measure *measure; ///< Measure of similarity object to use for the data term
   nifti_image* referenceImage; ///< Reference image in which the transformation is parametrised
   nifti_image* controlPointImage; ///< Control point image that contains the transformation to optimise
   int discrete_radius; ///< Radius of the discretised grid
   int discrete_increment; ///< Increment step size in the discretised grid
   float regularisation_weight; ///< Weight given to the regularisation

   int image_dim; ///< Dimension of the reference image
   size_t node_number; ///< Number of nodes in the tree

   float **discrete_values_mm; ///< All discretised values in millimetre

   int* orderedList; ///< Ordered list of nodes from the root to the leaves
   int* parentsList; ///< List that gives parent's index for each node
   float* edgeWeight; ///< Weight of edge between two nodes

   int label_1D_num; ///< Number of discretised values per axis
   int label_nD_num; ///< Total number of discretised values

   nifti_image *input_transformation;
   float *discretised_measures; ///< All discretised measures of similarity
   float* regularised_cost; ///< Discretised cost that embeds data term and regularisation cost
   int* optimal_label_index; ///< Optimimal label index for each node

   bool initialised; ///< Variable to access if the object has been initialised
};
/********************************************************************************************************/
template <class DataType>
void GetGraph_core3D(nifti_image* controlPointGridImage,
                     float* edgeWeightMatrix,
                     float* index_neighbours,
                     nifti_image *refImage,
                     int *mask);
template <class DataType>
void GetGraph_core2D(nifti_image* controlPointGridImage,
                     float* edgeWeightMatrix,
                     float* index_neighbours,
                     nifti_image *refImage,
                     int *mask);
void dt1sq(float *val,int* ind,int len,float offset,int k,int* v,float* z,float* f,int* ind1);
void dt3x(float* r,int* indr,int rl,float dx,float dy,float dz);
/********************************************************************************************************/
