/*
 *  _reg_bspline.h
 *  
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_TRANSFORMATION_H
#define _REG_TRANSFORMATION_H

#include "nifti1_io.h"
#include "_reg_globalTransformation.h"
#include "float.h"
#include <limits>
#include "_reg_maths.h"
#include "_reg_tools.h"

#if _USE_SSE
	#include <emmintrin.h>
#endif

/* *************************************************************** */
/** reg_spline_getDeformationField
 * Compute a dense deformation field in the space of a reference
 * image from a grid of control point.
 * - The computation is performed only at the voxel in the mask
 * - A composition with the current deformation field is performed if composition is true,
 * otherwise and blank deformation is used as initialisation
 * - Cubic B-Spline are used if bspline is true, cubic spline otherwise
 */
extern "C++"
void reg_spline_getDeformationField(nifti_image *splineControlPoint,
                                    nifti_image *referenceImage,
                                    nifti_image *deformationField,
                                    int *mask,
                                    bool composition,
                                    bool bspline
                                    );
/* *************************************************************** */
/** reg_bspline_bendingEnergy
 * Compute and return the average bending energy computed using cubic b-spline
 * Value is approximated at the control point position only.
 */
extern "C++"
double reg_bspline_bendingEnergy(nifti_image *splineControlPoint);
/* *************************************************************** */
/** reg_bspline_bendingEnergyGradient
 * Compute and return the approximated (at the control point position)
 * bending energy gradient for each control point
 */
extern "C++"
void reg_bspline_bendingEnergyGradient(nifti_image *splineControlPoint,
                                       nifti_image *targetImage,
                                       nifti_image *gradientImage,
                                       float weight
                                       );
/* *************************************************************** */
/** reg_bspline_linearEnergy
  * Compute and return the linear elastic energy term approximated
  * at the control point positions only.
  */
extern "C++"
void reg_bspline_linearEnergy(nifti_image *splineControlPoint, double *values);
/* *************************************************************** */
extern "C++"
void reg_bspline_linearEnergyGradient(nifti_image *splineControlPoint,
                                      nifti_image *targetImage,
                                      nifti_image *gradientImage,
                                      float weight0,
                                      float weight1,
                                      float weight2
                                      );
/* *************************************************************** */
/** reg_bspline_GetJacobianMap
 * Compute the Jacobian determinant map using a cubic b-spline parametrisation
 * or a cubic spline parametrisation
 */
extern "C++"
void reg_bspline_GetJacobianMap(nifti_image *splineControlPoint,
                                nifti_image *jacobianImage
                                );
/* *************************************************************** */
/** reg_bspline_jacobian
 * Compute the average Jacobian determinant
 */
extern "C++"
double reg_bspline_jacobian(nifti_image *splineControlPoint,
                            nifti_image *targetImage,
                            bool approx
                            );
/* *************************************************************** */
/** reg_bspline_jacobianDeterminantGradient
 * Compute the gradient Jacobian determinant at every control point position
 * using a cubic b-spline parametrisation
 */
extern "C++"
void reg_bspline_jacobianDeterminantGradient(nifti_image *splineControlPoint,
                                             nifti_image *targetImage,
                                             nifti_image *gradientImage,
                                             float weight,
                                             bool approx
                                             );
/* *************************************************************** */
/** reg_bspline_GetJacobianMatrix
 * Compute the Jacobian matrix at every voxel position
 * using a cubic b-spline parametrisation
 */
extern "C++"
void reg_bspline_GetJacobianMatrix(nifti_image *splineControlPoint,
                                   nifti_image *jacobianImage
                                   );
/* *************************************************************** */
/** reg_bspline_correctFolding
 * Correct the folding in the transformation parametrised through
 * cubic B-Spline
 */
extern "C++"
double reg_bspline_correctFolding(nifti_image *splineControlPoint,
                                  nifti_image *targetImage,
                                  bool approx
                                  );
/* *************************************************************** */
/** reg_voxelCentric2NodeCentric
 * Upsample an image from voxel space to node space
 */
extern "C++"
void reg_voxelCentric2NodeCentric(nifti_image *nodeImage,
                                  nifti_image *voxelImage,
                                  float weight,
                                  bool update
                                  );
/* *************************************************************** */
/** reg_bspline_refineControlPointGrid
 * Refine a control point grid
 */
extern "C++"
void reg_bspline_refineControlPointGrid(nifti_image *targetImage,
                                        nifti_image *splineControlPoint
                                        );
/* *************************************************************** */
/** reg_bspline_initialiseControlPointGridWithAffine
 * Initialise a lattice of control point to generate a global deformation
 */
extern "C++"
int reg_bspline_initialiseControlPointGridWithAffine(mat44 *affineTransformation,
                                                     nifti_image *controlPointImage
                                                     );
/* *************************************************************** */
/** reg_getDisplacementFromDeformation(nifti_image *)
  * This function converts a control point grid containing deformation
  * into a control point grid containing displacements.
  * The conversion is done using the appropriate qform/sform
**/
int reg_getDisplacementFromDeformation(nifti_image *controlPointImage);
/* *************************************************************** */
/** reg_getDeformationFromDisplacement(nifti_image *)
  * This function converts a control point grid containing displacements
  * into a control point grid containing deformation.
  * The conversion is done using the appropriate qform/sform
**/
int reg_getDeformationFromDisplacement(nifti_image *controlPointImage);
/* *************************************************************** */
/** reg_getJacobianImage
 * Compute the Jacobian determinant at every voxel position
 * from a deformation field. A linear interpolation is
 * assumed
 */
extern "C++"
void reg_defField_getJacobianMap(nifti_image *deformationField,
                                 nifti_image *jacobianImage);
/* *************************************************************** */
/** reg_getJacobianImage
 * Compute the Jacobian matrix at every voxel position
 * from a deformation field. A linear interpolation is
 * assumed
 */
extern "C++"
void reg_defField_getJacobianMatrix(nifti_image *deformationField,
                                    nifti_image *jacobianImage);
/* *************************************************************** */
/** reg_composeDefField
  * Preforms a deformation field composition.
  * The deformation field image is applied to the second image:
  * dfToUpdate. Both images are expected to contain deformation
  * field.
  * Only voxel within the mask are considered.
  */
extern "C++"
void reg_defField_compose(nifti_image *deformationField,
                         nifti_image *dfToUpdate,
                         int *mask);
/* *************************************************************** */
/** reg_spline_cppComposition(nifti_image* img1, nifti_image* img2, bool type)
  * This function compose the a first control point image with a second one:
  * T(x)=Grid1(Grid2(x)).
  * Grid1 and Grid2 have to contain deformation.
  * Cubic B-Spline can be used (bspline=true) or Cubic Spline (bspline=false)
 **/
extern "C++"
int reg_spline_cppComposition(nifti_image *grid1,
                              nifti_image *grid2,
                              bool bspline
                              );
/* *************************************************************** */
/** reg_bspline_GetJacobianMapFromVelocityField(nifti_image *img1, nifti_image *img2)
  * This function computed a Jacobian determinant map by integrating the velocity field
 **/
extern "C++"
int reg_bspline_GetJacobianMapFromVelocityField(nifti_image* velocityFieldImage,
                                                nifti_image* jacobianImage
                                                );
/* *************************************************************** */
/** reg_getDeformationFieldFromVelocityGrid(nifti_image *img1, nifti_image *img2, int *mask);
  * The deformation field (img2) is computed by integrating a velocity field (img1).
  * Only the voxel within the mask will be considered. If Mask is set to NULL then
  * all the voxels will be included within the mask.
 **/
extern "C++"
void reg_getDeformationFieldFromVelocityGrid(nifti_image *velocityFieldGrid,
                                             nifti_image *deformationFieldImage,
                                             nifti_image **intermediateDeformationField,
                                             bool approx);
extern "C++"
void reg_getInverseDeformationFieldFromVelocityGrid(nifti_image *velocityFieldGrid,
                                                    nifti_image *deformationFieldImage,
                                                    nifti_image **intermediateDeformationField,
                                                    bool approx);
/* *************************************************************** */
#endif
