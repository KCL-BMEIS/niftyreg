/*
 *  reg_png.h
 *
 *
 *  Created by Marc Modat on 30/05/2012.
 *  Copyright (c) 2012, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_PNG_H
#define _REG_PNG_H

#include "nifti1_io.h"
#include "readpng.h"
#include "_reg_tools.h"

/* *************************************************************** */
/** reg_readPNGfile
  * This function read a png file from the hard-drive and convert
  * it into a nifti_structure. using this function, you can either
  * read the full image or only the header information
  */
nifti_image *reg_io_readPNGfile(char *filename, bool readData);
/* *************************************************************** */
/** reg_writePNGfile
  * This function first convert a nifti image into a png and then
  * save the png file.
  */
void reg_io_writePNGfile(nifti_image *image, const char *filename);
/* *************************************************************** */

#endif
