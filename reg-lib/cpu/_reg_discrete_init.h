/**
 * @file _reg_mrf.h
 * @author Benoit Presles
 * @author Mattias Heinrich
 * @date 01/01/2016
 * @brief reg_mrf class for discrete optimisation
 *
 * Copyright (c) 2016, University College London. All rights reserved.
 * Centre for Medical Image Computing (CMIC)
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _reg_discrete_init_H
#define _reg_discrete_init_H

#include "_reg_measure.h"
#include "_reg_optimiser.h"
#include "_reg_localTrans_regul.h"
#include "_reg_localTrans.h"
#include "_reg_ReadWriteImage.h"
#include <cmath>
#include <queue>
#include <algorithm>

/** @brief Given two input images a discretisation of the measure of similarity is performed.
 * The returned transformation is a balanced between the best discretised measure and a regularisation
 * term (bending energy).
 */
class reg_discrete_init
{
public:
   /// @brief Constructor
   reg_discrete_init(reg_measure *_measure,
                     nifti_image *_referenceImage,
                     nifti_image *_controlPointImage,
                     int discrete_radius,
                     int _discrete_increment,
                     int _reg_max_it,
                     float _reg_weight);
   /// @brief Destructor
   ~reg_discrete_init();
   void Run();

private:
   void GetDiscretisedMeasure();
   void AddL2Penalisation(float);
   void GetRegularisedMeasure();
   void getOptimalLabel();
   void UpdateTransformation();

   reg_measure *measure; ///< Measure of similarity object to use for the data term
   nifti_image* referenceImage; ///< Reference image in which the transformation is parametrised
   nifti_image* controlPointImage; ///< Control point image that contains the transformation to optimise
   int discrete_radius; ///< Radius of the discretised grid
   int discrete_increment; ///< Increment step size in the discretised grid
   float regularisation_weight; ///< Weight given to the regularisation

   int image_dim; ///< Dimension of the reference image
   size_t node_number; ///< Number of nodes in the tree

   float **discrete_values_mm; ///< All discretised values in millimeter

   int label_1D_num; ///< Number of discretised values per axis
   int label_nD_num; ///< Total number of discretised values

   nifti_image *input_transformation;
   float *discretised_measures; ///< All discretised measures of similarity
   float *regularised_measures; ///< All combined measures
   int* optimal_label_index; ///< Optimimal label index for each node
   int regularisation_convergence;
   int reg_max_it; ///< Maximal number of iteration in the regularisation strategy

   float l2_weight;
   float* l2_penalisation;
};
/********************************************************************************************************/
#endif // _reg_discrete_init_H
