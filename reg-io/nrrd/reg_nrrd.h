/*
 *  reg_nrrd.h
 *
 *
 *  Created by Marc Modat on 30/05/2012.
 *  Copyright (c) 2012, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_NRRD_H
#define _REG_NRRD_H

#include "nifti1_io.h"
#include "NrrdIO.h"
#include "_reg_tools.h"
#include "_reg_maths.h"

/* *************************************************************** */
/** reg_io_nrdd2nifti
  * Convert a NRRD image into a nifti image
  * Note that the NRRD image is not freed
  */
nifti_image *reg_io_nrdd2nifti(Nrrd *image);
/* *************************************************************** */
/** reg_io_nifti2nrrd
  * Convert a nifti image into a NRRD image
  * Note that the nifti image is not freed
  */
Nrrd *reg_io_nifti2nrrd(nifti_image *image);
/* *************************************************************** */
Nrrd *reg_io_readNRRDfile(const char *filename);
/* *************************************************************** */
void reg_io_writeNRRDfile(Nrrd *image, const char *filename);
/* *************************************************************** */


#endif
