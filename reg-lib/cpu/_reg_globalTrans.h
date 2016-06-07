/**
 * @file _reg_globalTrans.h
 * @author Marc Modat
 * @date 25/03/2009
 * @brief library that contains the function related to global transformation
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_AFFINETRANS_H
#define _REG_AFFINETRANS_H

#include "nifti1_io.h"
#include "_reg_tools.h"
/* *************************************************************** */
/// @brief Structure that is used to store the distance between two corresponding voxel
struct _reg_sorted_point3D
{
    float reference[3];
    float warped[3];

    double distance;

    _reg_sorted_point3D(float * t, float * r, double d)
        :distance(d)
    {
        reference[0] = t[0];
        reference[1] = t[1];
        reference[2] = t[2];

        warped[0] = r[0];
        warped[1] = r[1];
        warped[2] = r[2];
    }

    bool operator <(const _reg_sorted_point3D &sp) const
    {
        return (sp.distance < distance);
    }
};
typedef struct _reg_sorted_point3D _reg_sorted_point3D;
/* *************************************************************** */
/// @brief Structure that is used to store the distance between two corresponding pixel
struct _reg_sorted_point2D
{
    float reference[2];
    float warped[2];

    double distance;

    _reg_sorted_point2D(float * t, float * r, double d)
        :distance(d)
    {
        reference[0] = t[0];
        reference[1] = t[1];

        warped[0] = r[0];
        warped[1] = r[1];
    }
    bool operator <(const _reg_sorted_point2D &sp) const
    {
        return (sp.distance < distance);
    }
};
typedef struct _reg_sorted_point2D _reg_sorted_point2D;
/* *************************************************************** */
/** @brief This Function compute a deformation field based
 * on affine transformation matrix
 * @param affine This matrix contains the affine transformation
 * used to parametrise the transformation
 * @param deformationField Image that contains the deformation field
 * that is being updated
 */
extern "C++"
void reg_affine_getDeformationField(mat44 *affine,
                                    nifti_image *deformationField,
                                    bool compose=false,
                                    int *mask = NULL);
/* *************************************************************** */
void optimize_2D(float* referencePosition, float* warpedPosition,
    unsigned int definedActiveBlock, int percent_to_keep, int max_iter, double tol,
    mat44* final, bool affine);
/* *************************************************************** */
void estimate_affine_transformation2D(std::vector<_reg_sorted_point2D> &points, mat44* transformation);
/* *************************************************************** */
void estimate_rigid_transformation2D(std::vector<_reg_sorted_point2D> &points, mat44* transformation);
/* *************************************************************** */
void optimize_3D(float* referencePosition, float* warpedPosition,
    unsigned int definedActiveBlock, int percent_to_keep, int max_iter, double tol,
    mat44* final, bool affine);
/* *************************************************************** */
void estimate_affine_transformation3D(std::vector<_reg_sorted_point3D> &points, mat44* transformation);
/* *************************************************************** */
void estimate_rigid_transformation3D(std::vector<_reg_sorted_point3D> &points, mat44* transformation);
/* *************************************************************** */
#endif
