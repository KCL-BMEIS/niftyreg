/**
 * @file _reg_ReadWriteMatrix.h
 * @author Benoit Presles
 * @date 17/09/2015
 * @brief library that contains the functions related to matrix
 *
 *  Created by Benoit Presles on 17/09/2015.
 *  Copyright (c) 2015, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */


#ifndef _REG_READWRITEMATRIX_H
#define _REG_READWRITEMATRIX_H

#include "nifti1_io.h"
//STD
#include <fstream>
#include <utility>

/** @brief Read a text file that contains a affine transformation
 * and store it into a mat44 structure. This function can also read
 * affine parametrisation from Flirt (FSL package) and convert it
 * to a standard millimeter parametrisation
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
extern "C++"
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
extern "C++"
void reg_tool_ReadAffineFile(mat44 *mat,
                             char *filename);

/**
* @brief Read a file that contains a 4-by-4 matrix and store it into
* a mat44 structure
* @param filename Filename of the text file that contains the matrix to read
* @return mat44 structure that store the matrix
**/
extern "C++"
mat44* reg_tool_ReadMat44File(char *fileName);

/** @brief This function save a 4-by-4 matrix to the disk as a text file
 * @param mat Matrix to be saved on the disk
 * @param filename Name of the text file to save on the disk
 */
extern "C++"
void reg_tool_WriteAffineFile(mat44 *mat,
                              const char *fileName);

/**
* @brief Read a file that contains a m-by-n matrix and return its size
* @param filename Filename of the text file that contains the matrix to read
* @return pair of values that contains the matrix size
**/
extern "C++"
std::pair<size_t, size_t> reg_tool_sizeInputMatrixFile(char *filename);
/**
* @brief Read a file that contains a m-by-n matrix and store it into
* an appropriate structure
* @param filename Filename of the text file that contains the matrix to read
* @param nbLine number of line of the imput matrix
* @param nbColumn number of column of the imput matrix
* @return a pointer to a 2D array that points the read matrix
**/
extern "C++" template <class T>
T** reg_tool_ReadMatrixFile(char *filename,
                            size_t nbLine,
                            size_t nbColumn);

/**
* @brief Write a file that contains a m-by-n matrix into a text file
* @param filename Filename of the text file to be written
* @param mat Input matrix to be saved
* @param nbLine number of line of the imput matrix
* @param nbColumn number of column of the imput matrix
**/
extern "C++" template <class T>
void reg_tool_WriteMatrixFile(char *filename,
                              T **mat,
                              size_t nbLine,
                              size_t nbColumn);

#endif // _REG_READWRITEMATRIX_H

