/**
 * @file _reg_ReadWriteImage.h
 * @author Marc Modat
 * @date 30/05/2012
 *  Created by Marc Modat on 30/05/2012.
 *  Copyright (c) 2012, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_READWRITEIMAGE_H
#define _REG_READWRITEIMAGE_H

#include "nifti1_io.h"
#include <string>

#ifdef _USE_NR_PNG
    #include "reg_png.h"
#endif

#ifdef _USE_NR_NRRD
    #include "reg_nrrd.h"
#endif

#define NR_NII_FORMAT 0
#define NR_PNG_FORMAT 1
#define NR_NRRD_FORMAT 2

/* *************************************************************** */
/** reg_checkFileFormat
  * The function checks the file format using the provided filename
  * Nifti is returned by default if no format are specified
  * @param filename Filename of the input images
  * @return Code that encode the file format
  */
int reg_io_checkFileFormat(const char *filename);
/* *************************************************************** */
/** reg_ReadImageFile
  * The function expects a filename and returns a nifti_image structure
  * The function will use to correct library and will return a NULL image
  * if the image can not be read
  */
nifti_image *reg_io_ReadImageFile(char *filename);
/* *************************************************************** */
/** reg_ReadImageHeader
  * The function expects a filename and returns a nifti_image structure
  * The function will use to correct library and will return a NULL image
  * if the image can not be read
  * Only the header information is read and the actual data is not store
  */
nifti_image *reg_io_ReadImageHeader(char *filename);
/* *************************************************************** */
/** reg_WriteImageHeader
  * The function expects a filename and nifti_image structure
  * The image will be converted to the format specified in the
  * filename before being saved
  */
void reg_io_WriteImageFile(nifti_image *image, const char *filename);
/* *************************************************************** */
#endif
