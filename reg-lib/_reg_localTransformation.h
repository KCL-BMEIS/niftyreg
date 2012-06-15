/**
 * @file _reg_localTransformation.h
 * @author Marc Modat
 * @date 25/03/2009
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


/* *********************************************** */
/* ****      CUBIC SPLINE BASED FUNCTIONS     **** */
/* *********************************************** */

/* *************************************************************** */
/** reg_createControlPointGrid
 * Generate a control point grid image based on the dimension of a
 * reference image and on a spacing.
 * The function set the qform and sform code to overlay the reference
 * image.
 */
extern "C++" template <class DTYPE>
void reg_createControlPointGrid(nifti_image **controlPointGridImage,
                                nifti_image *referenceImage,
                                float *spacingMillimeter);

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
void reg_bspline_linearEnergy(nifti_image *splineControlPoint,
                              double *values
                              );
/* *************************************************************** */
extern "C++"
void reg_bspline_linearEnergyGradient(nifti_image *splineControlPoint,
                                      nifti_image *targetImage,
                                      nifti_image *gradientImage,
                                      float weight0,
                                      float weight1
                                      );
/* *************************************************************** */
/** reg_bspline_L2norm_displacement
  * Compute and return the L2 norm of the displacement approximated
  * at the control point positions only.
  */
extern "C++"
double reg_bspline_L2norm_displacement(nifti_image *splineControlPoint);
/* *************************************************************** */
/** reg_bspline_L2norm_dispGradient
  * Compute the gradient of the L2 norm of the displacement approximated
  * at the control point positions only.
  */
extern "C++"
void reg_bspline_L2norm_dispGradient(nifti_image *splineControlPoint,
                                     nifti_image *referenceImage,
                                     nifti_image *gradientImage,
                                     float weight);
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
/** reg_bspline_GetJacobianMatrixFull
 * Compute the Jacobian matrix at every voxel position
 * using a cubic b-spline parametrisation
 */
extern "C++"
void reg_bspline_GetJacobianMatrixFull(nifti_image *referenceImage,
                                       nifti_image *splineControlPoint,
                                       mat33 *jacobianImage
                                       );
/* *************************************************************** */
/** reg_bspline_GetJacobianMatrix
 * Compute the Jacobian matrix at every voxel position
 * using a cubic b-spline parametrisation
 * This function is similar to reg_bspline_GetJacobianMatrixFull but it
 * assumes that the splineControlPoint grid is used to parametrise the
 * transformation in referenceImage. It leads to faster computation.
 */
extern "C++"
void reg_bspline_GetJacobianMatrix(nifti_image *referenceImage,
                                   nifti_image *splineControlPoint,
                                   mat33 *jacobianImage
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
/** reg_spline_cppComposition(nifti_image* grid1, nifti_image* grid2, bool type)
  * This function compose the a first control point image with a second one:
  * Grid2(x) <= Grid1(Grid2(x)).
  * Grid1 and Grid2 have to contain either displacement or deformation.
  * The output will be a deformation field if grid1 is a deformation,
  * The output will be a displacement field if grid1 is a displacement.
  * Cubic B-Spline can be used (bspline=true) or Cubic Spline (bspline=false)
 **/
extern "C++"
int reg_spline_cppComposition(nifti_image *grid1,
                              nifti_image *grid2,
                              bool displacement1,
                              bool displacement2,
                              bool bspline
                              );
/* *************************************************************** */


/* *********************************************** */
/* ****   DEFORMATION FIELD BASED FUNCTIONS   **** */
/* *********************************************** */

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
                                    mat33 *jacobianMatrices);
/* *************************************************************** */
/** reg_defField_compose
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

/* *********************************************** */
/* ****     VELOCITY FIELD BASED FUNCTIONS    **** */
/* *********************************************** */

/* *************************************************************** */
/** reg_bspline_GetJacobianMatricesFromVelocityField(nifti_image *ref, nifti_image *vel, mat33 *mat)
  * This function computed Jacobian matrices by integrating the velocity field
 **/
extern "C++"
int reg_bspline_GetJacobianMatricesFromVelocityField(nifti_image* referenceImage,
                                                     nifti_image* velocityFieldImage,
                                                     mat33* jacobianMatrices
                                                     );
/* *************************************************************** */
/** reg_bspline_GetJacobianDetFromVelocityField(nifti_image *det, nifti_image *vel)
  * This function computed a Jacobian determinant map by integrating the velocity field
 **/
extern "C++"
int reg_bspline_GetJacobianDetFromVelocityField(nifti_image* jacobianDetImage,
                                                nifti_image* velocityFieldImage
                                                );
/* *************************************************************** */
/** reg_getDeformationFieldFromVelocityGrid(nifti_image *img1, nifti_image *img2, int *mask);
  * The deformation field (img2) is computed by integrating a velocity field (img1).
  * Only the voxel within the mask will be considered. If Mask is set to NULL then
  * all the voxels will be included within the mask.
 **/
extern "C++"
void reg_bspline_getDeformationFieldFromVelocityGrid(nifti_image *velocityFieldGrid,
                                                     nifti_image *deformationFieldImage);
/* *************************************************************** */


/* *********************************************** */
/* ****            OTHER FUNCTIONS            **** */
/* *********************************************** */

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
/** compute_BCH_update(nifti_image *,nifti_image *,int)
  * This function compute the BCH update using an initial verlocity field
  * and its gradient.
  * The type encodes the number of component of the serie to be considered:
  * 0 - w=u+v
  * 1 - w=u+v+0.5*[u,v]
  * 2 - w=u+v+0.5*[u,v]+[u,[u,v]]/12
  * 3 - w=u+v+0.5*[u,v]+[u,[u,v]]/12-[v,[u,v]]/12
  * 4 - w=u+v+0.5*[u,v]+[u,[u,v]]/12-[v,[u,v]]/12-[v,[u,[u,g]]]/24
**/
extern "C++"
void compute_BCH_update(nifti_image *img1,
                        nifti_image *img2,
                        int type);

/* *************************************************************** */
/** reg_GetDeconvolvedSplineCoefficents(nifti_image *)
  * to write
**/
extern "C++"
void reg_spline_GetDeconvolvedCoefficents(nifti_image *img);

/* *************************************************************** */
#endif
