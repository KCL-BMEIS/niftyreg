/**
 * @file reg_nrrd.h
 * @brief NiftyReg interface to the NRRD library
 * @author Marc Modat
 * @date 30/05/2012
 *
 * Created by Marc Modat on 30/05/2012.
 *  Copyright (c) 2012-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#pragma once

#include "NrrdIO.h"
#include "_reg_tools.h"

/* *************************************************************** */
/** @brief Convert a NRRD image into a nifti image
 * Note that the NRRD image is not freed
 * @param image Input image in NRRD format
 * @return Returns a pointer nifti_image structure
 */
nifti_image *reg_io_nrdd2nifti(Nrrd *image);
/* *************************************************************** */
/** @brief Convert a nifti image into a NRRD image
 * Note that the nifti image is not freed
 * @param image Nifti image to be converted
 * @return Returns a NRRD image
 */
Nrrd *reg_io_nifti2nrrd(nifti_image *image);
/* *************************************************************** */
/** @brief Read a NRRD image from the disk
 * @param filename Path of the nrrd image to read
 * @return Returns a NRRD image read from the disk
 */
Nrrd *reg_io_readNRRDfile(const char *filename);
/* *************************************************************** */
/** @brief Save a NRRD image on the disk
 * @param image NRRD image to be saved on the disk
 * @param filename Name of the NRRD image on the disk
 */
void reg_io_writeNRRDfile(Nrrd *image, const char *filename);
/* *************************************************************** */
