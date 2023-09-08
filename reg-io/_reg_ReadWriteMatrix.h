/**
 * @file _reg_ReadWriteMatrix.h
 * @author Benoit Presles
 * @date 17/09/2015
 * @brief library that contains the functions related to matrix
 *
 *  Created by Benoit Presles on 17/09/2015.
 *  Copyright (c) 2015-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#pragma once

#include "_reg_tools.h"

/** @brief Read a text file that contains a affine transformation
 * and store it into a mat44 structure. This function can also read
 * affine parametrisation from Flirt (FSL package) and convert it
 * to a standard millimetre parametrisation
 * @param mat Structure that will be updated with the affine
 * transformation matrix
 * @param referenceImage Reference image of the current transformation
 * @param floatingImage Floating image of the current transformation.
 * Note that referenceImage and floating image have to be defined but
 * are only used when dealing with a Flirt affine matrix.
 * @param filename Filename for the text file that contains the matrix
 * to read
 * @param flirtFile If this flag is set to true the matrix is converted
 * from a Flirt (FSL) parametrisation to a standard parametrisation
 */
void reg_tool_ReadAffineFile(mat44 *mat,
                             nifti_image *referenceImage,
                             nifti_image *floatingImage,
                             char *fileName,
                             bool flirtFile);

/**
* @brief Read a file that contains a 4-by-4 matrix and store it into
* a mat44 structure
* @param mat structure that store the affine transformation matrix
* @param filename Filename of the text file that contains the matrix to read
**/
void reg_tool_ReadAffineFile(mat44 *mat,
                             char *filename);

/**
* @brief Read a file that contains a 4-by-4 matrix and store it into
* a mat44 structure
* @param filename Filename of the text file that contains the matrix to read
* @return mat44 structure that store the matrix
**/
mat44* reg_tool_ReadMat44File(char *fileName);

/** @brief This function save a 4-by-4 matrix to the disk as a text file
 * @param mat Matrix to be saved on the disk
 * @param filename Name of the text file to save on the disk
 */
void reg_tool_WriteAffineFile(const mat44 *mat,
                              const char *fileName);

/**
* @brief Read a file that contains a m-by-n matrix and return its size
* @param filename Filename of the text file that contains the matrix to read
* @return pair of values that contains the matrix size
**/
std::pair<size_t, size_t> reg_tool_sizeInputMatrixFile(char *filename);
/**
* @brief Read a file that contains a m-by-n matrix and store it into
* an appropriate structure
* @param filename Filename of the text file that contains the matrix to read
* @param nbLine number of line of the input matrix
* @param nbColumn number of column of the input matrix
* @return a pointer to a 2D array that points the read matrix
**/
template <class T>
T** reg_tool_ReadMatrixFile(char *filename,
                            size_t nbLine,
                            size_t nbColumn);

/**
* @brief Write a file that contains a m-by-n matrix into a text file
* @param filename Filename of the text file to be written
* @param mat Input matrix to be saved
* @param nbLine number of line of the input matrix
* @param nbColumn number of column of the input matrix
**/
template <class T>
void reg_tool_WriteMatrixFile(char *filename,
                              T **mat,
                              size_t nbLine,
                              size_t nbColumn);
