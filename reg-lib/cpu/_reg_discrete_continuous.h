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

#ifndef _REG_DISCRETE_CONTINUOUS_H
#define _REG_DISCRETE_CONTINUOUS_H

#include "_reg_measure.h"
#include "_reg_optimiser.h"
#include "_reg_localTrans_regul.h"
#include <cmath>
#include <queue>
#include <algorithm>

class reg_discrete_continuous : public InterfaceOptimiser
{
public:
   /// @brief Constructor
   reg_discrete_continuous(reg_measure *_measure,
                           nifti_image *_referenceImage,
                           nifti_image *_controlPointImage,
                           int discrete_radius,
                           int _discrete_increment,
                           float _reg_weight);
   /// @brief Destructor
   ~reg_discrete_continuous();
   void Run();

private:
   void GetDiscretisedMeasure();
   void getOptimalLabel();
   void StoreOptimalMeasureTransformation();
   void ContinuousRegularisation();
   /// @brief Returns the registration current objective function value
   double GetObjectiveFunctionValue();
   /// @bried Compute the cost function gradient
   void GetObjectiveFunctionGradient();
   /// @brief The transformation parameters are optimised
   void UpdateParameters(float);
   /// @brief Ben TODO
   void UpdateBestObjFunctionValue();

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

   float *discretised_measures; ///< All discretised measures of similarity
   int* optimal_label_index; ///< Optimimal label index for each node

   reg_conjugateGradient<float> *optimiser;
   nifti_image *optimal_measure_transformation; ///< Transformation grid that contains the optimal transformation based on the measure discrete optimisation
   nifti_image *regularisation_gradient; ///< blablabla

};
/********************************************************************************************************/
#endif // _REG_DISCRETE_CONTINUOUS_H
