/**
 * @file _reg_ReadWriteImage.h
 * @author Marc Modat
 * @date 30/05/2012
 * @brief IO interface to the NiftyReg project. It uses the nifti, nrrd and png libraries.
 *
 *  Created by Marc Modat on 30/05/2012.
 *  Copyright (c) 2012-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#pragma once

#include <string>
#include "_reg_tools.h"

#include "reg_png.h"
#ifdef USE_NRRD
#include "reg_nrrd.h"
#endif
/** @defgroup NIFTYREG_FILEFORMAT_TYPE
 *  @brief Codes to define the image file format
 *  @{
 */
#define NR_NII_FORMAT 0
#define NR_PNG_FORMAT 1
#ifdef USE_NRRD
#define NR_NRRD_FORMAT 2
#endif
/* @} */

/* *************************************************************** */
/** The function checks the file format using the provided filename
  * Nifti is returned by default if no format are specified
  * @param filename Filename of the input images
  * @return Code, NIFTYREG_FILEFORMAT_TYPE, that encode the file format
  */
int reg_io_checkFileFormat(const std::string& filename);
/* *************************************************************** */
/** The function expects a filename and returns a NiftiImage
  * The function will use to correct library and will return an empty image
  * if the image can not be read
  * @param filename Filename of the input images
  * @param onlyHeader If true, only the header information is read and the actual data is not stored
  * @return Image as a NiftiImage
  */
NiftiImage reg_io_ReadImageFile(const char *filename, const bool onlyHeader = false);
/* *************************************************************** */
/** The function expects a filename and a nifti_image
  * The image will be converted to the format specified in the
  * filename before being saved
  * @param image Nifti image to be saved
  * @param filename Filename of the output images
  */
void reg_io_WriteImageFile(nifti_image *image, const char *filename);
/* *************************************************************** */
