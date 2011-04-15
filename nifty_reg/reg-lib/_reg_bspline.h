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

#ifndef _REG_BSPLINE_H
#define _REG_BSPLINE_H

#include "nifti1_io.h"
#include "_reg_affineTransformation.h"
#include "float.h"
#include <limits>

#if _USE_SSE
	#include <emmintrin.h>
#endif


/* *************************************************************** */
/** Get_BSplineBasisValues(DTYPE basis, DTYPE *values)
 * The four cubic b-spline values are computed from the relative position.
 */
extern "C++" template<class DTYPE>
void Get_BSplineBasisValues(DTYPE basis,
                            DTYPE *values);
/* *************************************************************** */
/** Get_BSplineBasisValues(DTYPE basis, DTYPE *values, DTYPE *first)
 * The four cubic b-spline values and first derivatives
 * are computed from the relative position.
 */
extern "C++" template<class DTYPE>
void Get_BSplineBasisValues(DTYPE basis,
                            DTYPE *values,
                            DTYPE *first);
/* *************************************************************** */
/** Get_BSplineBasisValues(DTYPE basis, DTYPE *values, DTYPE *first, DTYPE *second)
 * The four cubic b-spline values, first and second derivatives
 * are computed from the relative position.
 */
extern "C++" template<class DTYPE>
void Get_BSplineBasisValues(DTYPE basis,
                            DTYPE *values,
                            DTYPE *first,
                            DTYPE *second);
/* *************************************************************** */
/** Get_SplineBasisValues(DTYPE basis, DTYPE *values)
 * The four cubic spline values are computed from the relative position.
 */
extern "C++" template<class DTYPE>
void Get_SplineBasisValues(DTYPE basis,
                           DTYPE *values);
/* *************************************************************** */
/** Get_SplineBasisValues(DTYPE basis, DTYPE *values, DTYPE *first)
 * The four cubic spline values and first derivatives
 * are computed from the relative position.
 */
extern "C++" template<class DTYPE>
void Get_SplineBasisValues(DTYPE basis,
                           DTYPE *values,
                           DTYPE *first);
/* *************************************************************** */
/** Get_SplineBasisValues(DTYPE basis, DTYPE *values, DTYPE *first, DTYPE *second)
 * The four cubic spline values, first and second derivatives
 * are computed from the relative position.
 */
extern "C++" template<class DTYPE>
void Get_SplineBasisValues(DTYPE basis,
                           DTYPE *values,
                           DTYPE *first,
                           DTYPE *second);
/* *************************************************************** */
/** get_splineDisplacement
 * Extract 16x2 control point positions from a grid
 * StartX and StartY defined the control point of coordinate
 * [0,0]. Control point in a 4-by-4 box will be extracted
 */
template <class ImageTYPE>
void get_splineDisplacement(int startX,
                            int startY,
                            nifti_image *splineControlPoint,
                            ImageTYPE *splineX,
                            ImageTYPE *splineY,
                            ImageTYPE *dispX,
                            ImageTYPE *dispY);
/* *************************************************************** */
/** get_splineDisplacement
 * Extract 64x3 control point positions from a grid
 * StartX, StartY and StartZ defined the control point of coordinate
 * [0,0,0]. Control point in a 4-by-4-by-4 box will be extracted
 */
template <class ImageTYPE>
void get_splineDisplacement(int startX,
                            int startY,
                            int startZ,
                            nifti_image *splineControlPoint,
                            ImageTYPE *splineX,
                            ImageTYPE *splineY,
                            ImageTYPE *splineZ,
                            ImageTYPE *dispX,
                            ImageTYPE *dispY,
                            ImageTYPE *dispZ);
/* *************************************************************** */
/** getReorientationMatrix
 * Compute the transformation matrix to diagonalise the input matrix
 */
extern "C++"
void getReorientationMatrix(nifti_image *splineControlPoint,
                            mat33 *desorient,
                            mat33 *reorient);
/* *************************************************************** */
/** getReorientationMatrix
 * Compute a dense deformation field in the space of a reference
 * image from a grid of control point.
 * - The computation is performed only at the voxel in the mask
 * - A composition with the current deformation field is performed if composition is true,
 * otherwise and blank deformation is used as initialisation
 * - Cubic B-Spline are used if bspline is true, cubic spline otherwise
 */
extern "C++"
void reg_bspline(nifti_image *splineControlPoint,
                 nifti_image *referenceImage,
                 nifti_image *deformationField,
                 int *mask,
                 bool composition,
                 bool bspline
                 );
/* *************************************************************** */
/** reg_bspline_bendingEnergy
 * Compute and return the average bending energy computed using cubic b-spline
 * If approx=true, the computation is performed at the control point initial position only
 */
extern "C++" template<class PrecisionTYPE>
PrecisionTYPE reg_bspline_bendingEnergy(nifti_image *splineControlPoint,
                                        nifti_image *targetImage,
                                        int approx
                                        );
/* *************************************************************** */
/** reg_bspline_bendingEnergyGradient
 * Compute and return the approximated (at the control point position)
 * bending energy gradient for each control point
 */
extern "C++" template<class PrecisionTYPE>
void reg_bspline_bendingEnergyGradient(nifti_image *splineControlPoint,
                                       nifti_image *targetImage,
                                       nifti_image *gradientImage,
                                       float weight
                                       );
/* *************************************************************** */
/** reg_bspline_GetJacobianMap
 * Compute the Jacobian determinant map using a cubic b-spline parametrisation
 */
extern "C++"
void reg_bspline_GetJacobianMap(nifti_image *splineControlPoint,
                                nifti_image *jacobianImage
                                );
/* *************************************************************** */
/** reg_bspline_jacobian
 * Compute the average Jacobian determinant
 */
extern "C++" template<class PrecisionTYPE>
PrecisionTYPE reg_bspline_jacobian(nifti_image *splineControlPoint,
                                   nifti_image *targetImage,
                                   int type
                                   );
/* *************************************************************** */
/** reg_bspline_jacobianDeterminantGradient
 * Compute the gradient Jacobian determinant at every control point position
 * using a cubic b-spline parametrisation
 */
extern "C++" template<class PrecisionTYPE>
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
extern "C++" template<class PrecisionTYPE>
PrecisionTYPE reg_bspline_correctFolding(nifti_image *splineControlPoint,
                                         nifti_image *targetImage,
                                         bool approx
                                         );
/* *************************************************************** */
/** */
extern "C++"
void reg_voxelCentric2NodeCentric(nifti_image *nodeImage,
                                  nifti_image *voxelImage,
                                  float weight
                                  );
/* *************************************************************** */
extern "C++"
void reg_bspline_refineControlPointGrid(nifti_image *targetImage,
                                        nifti_image *splineControlPoint
                                        );
/* *************************************************************** */
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

#endif
