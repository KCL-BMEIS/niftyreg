/*
 *  _reg_blockMatching.cpp
 *
 *
 *  Created by Marc Modat and Pankaj Daga on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_blockMatching.h"
#include "_reg_globalTrans.h"
#include <map>
#include <iostream>
#include <cmath>
/* *************************************************************** */
template<class DTYPE>
void _reg_set_active_blocks(nifti_image *referenceImage, _reg_blockMatchingParam *params, int *mask, bool runningOnGPU) {

   float *varianceArray = (float *)malloc(params->totalBlockNumber * sizeof(float));
   int *indexArray = (int *)malloc(params->totalBlockNumber * sizeof(int));

   int *maskPtr = &mask[0];

   int unusableBlock = 0;
   size_t index;
   DTYPE *referenceValues = NULL;
   if (referenceImage->nz > 1) {
      referenceValues = (DTYPE *)malloc(BLOCK_3D_SIZE * sizeof(DTYPE));
   }
   else {
      referenceValues = (DTYPE *)malloc(BLOCK_2D_SIZE * sizeof(DTYPE));
   }
   DTYPE *referencePtr = static_cast<DTYPE *>(referenceImage->data);
   int blockIndex = 0;

   if (referenceImage->nz > 1) {
      // Version using 3D blocks
      for (unsigned int k = 0; k < params->blockNumber[2]; k++) {
         for (unsigned int j = 0; j < params->blockNumber[1]; j++) {
            for (unsigned int i = 0; i < params->blockNumber[0]; i++) {

               for (unsigned int n = 0; n < BLOCK_3D_SIZE; n++)
                  referenceValues[n] = (DTYPE)std::numeric_limits<float>::quiet_NaN();

               float mean = 0.0f;
               float voxelNumber = 0.0f;
               int coord = 0;
               for (unsigned int z = k * BLOCK_WIDTH; z < (k + 1) * BLOCK_WIDTH; z++) {
                  if (z < (unsigned int)referenceImage->nz) {
                     index = z * referenceImage->nx * referenceImage->ny;
                     DTYPE *referencePtrZ = &referencePtr[index];
                     int *maskPtrZ = &maskPtr[index];
                     for (unsigned int y = j * BLOCK_WIDTH; y < (j + 1) * BLOCK_WIDTH; y++) {
                        if (y < (unsigned int)referenceImage->ny) {
                           index = y * referenceImage->nx + i * BLOCK_WIDTH;
                           DTYPE *referencePtrXYZ = &referencePtrZ[index];
                           int *maskPtrXYZ = &maskPtrZ[index];
                           for (unsigned int x = i * BLOCK_WIDTH; x < (i + 1) * BLOCK_WIDTH; x++) {
                              if (x < (unsigned int)referenceImage->nx) {
                                 referenceValues[coord] = *referencePtrXYZ;
                                 if (referenceValues[coord] == referenceValues[coord] && *maskPtrXYZ > -1) {
                                    mean += (float)referenceValues[coord];
                                    voxelNumber++;
                                 }
                              }
                              referencePtrXYZ++;
                              maskPtrXYZ++;
                              coord++;
                           }
                        }
                     }
                  }
               }
               mean /= voxelNumber;

               //Let's calculate the variance of the block
               float variance = 0.0f;
               for (int i = 0; i < BLOCK_3D_SIZE; i++) {
                  if (referenceValues[i] == referenceValues[i])
                     variance += (mean - (float)referenceValues[i]) * (mean - (float)referenceValues[i]);
               }
               variance /= voxelNumber;

               if (voxelNumber > BLOCK_3D_SIZE / 2 && variance > 0) {
                  varianceArray[blockIndex] = variance;
               }
               else {
                  varianceArray[blockIndex] = -1;
                  unusableBlock++;
               }
               indexArray[blockIndex] = blockIndex;
               blockIndex++;
            }
         }
      }
   }
   else {
      // Version using 2D blocks
      for (unsigned int j = 0; j < params->blockNumber[1]; j++) {
         for (unsigned int i = 0; i < params->blockNumber[0]; i++) {

            for (unsigned int n = 0; n < BLOCK_2D_SIZE; n++)
               referenceValues[n] = (DTYPE)std::numeric_limits<float>::quiet_NaN();

            float mean = 0.0f;
            float voxelNumber = 0.0f;
            int coord = 0;

            for (unsigned int y = j * BLOCK_WIDTH; y < (j + 1) * BLOCK_WIDTH; y++) {
               if (y < (unsigned )referenceImage->ny) {
                  index = y * referenceImage->nx + i * BLOCK_WIDTH;
                  DTYPE *referencePtrXY = &referencePtr[index];
                  int *maskPtrXY = &maskPtr[index];
                  for (unsigned int x = i * BLOCK_WIDTH; x < (i + 1) * BLOCK_WIDTH; x++) {
                     if (x < (unsigned)referenceImage->nx) {
                        referenceValues[coord] = *referencePtrXY;
                        if (referenceValues[coord] == referenceValues[coord] && *maskPtrXY > -1) {
                           mean += (float)referenceValues[coord];
                           voxelNumber++;
                        }
                     }
                     referencePtrXY++;
                     maskPtrXY++;
                     coord++;
                  }
               }
            }
            mean /= voxelNumber;

            //Let's calculate the variance of the block
            float variance = 0.0f;
            for (int i = 0; i < BLOCK_2D_SIZE; i++) {
               if (referenceValues[i] == referenceValues[i])
                  variance += (mean - (float)referenceValues[i]) * (mean - (float)referenceValues[i]);
            }
            variance /= voxelNumber;

            if (voxelNumber > BLOCK_2D_SIZE / 2 && variance > 0) {
               varianceArray[blockIndex] = variance;
            }
            else {
               varianceArray[blockIndex] = -1;
               unusableBlock++;
            }
            indexArray[blockIndex] = blockIndex;
            blockIndex++;
         }
      }
   }
   free(referenceValues);

   params->activeBlockNumber = params->activeBlockNumber < ((int)params->totalBlockNumber - unusableBlock) ? params->activeBlockNumber : (params->totalBlockNumber - unusableBlock);
   //params->activeBlockNumber = params->totalBlockNumber - unusableBlock;

   reg_heapSort(varianceArray, indexArray, params->totalBlockNumber);
   int *indexArrayPtr = &indexArray[params->totalBlockNumber - 1];
   int count = 0;
   for (int i = 0; i < params->activeBlockNumber; i++) {
      params->totalBlock[*indexArrayPtr--] = count++;
   }
   for (int i=params->activeBlockNumber; i<params->totalBlockNumber; ++i) {
      params->totalBlock[*indexArrayPtr--] = -1;
   }

   count = 0;
   if (runningOnGPU) {
      for (int i = 0; i < params->totalBlockNumber; ++i) {
         if (params->totalBlock[i] != -1) {
            params->totalBlock[i] = -1;
            params->totalBlock[count] = i;
            ++count;
         }
      }
   }

   free(varianceArray);
   free(indexArray);
}
/* *************************************************************** */
void initialise_block_matching_method(nifti_image * reference,
                                      _reg_blockMatchingParam *params,
                                      int percentToKeep_block,
                                      int percentToKeep_opt,
                                      int stepSize_block,
                                      int *mask,
                                      bool runningOnGPU) {
   if (params->totalBlock != NULL) {
      free(params->totalBlock);
      params->totalBlock = NULL;
   }

   if (params->referencePosition != NULL) {
      free(params->referencePosition);
      params->referencePosition = NULL;
   }
   if (params->warpedPosition != NULL) {
      free(params->warpedPosition);
      params->warpedPosition = NULL;
   }

   params->voxelCaptureRange = 3;
   params->blockNumber[0] = (int)std::ceil((double)reference->nx / (double)BLOCK_WIDTH);
   params->blockNumber[1] = (int)std::ceil((double)reference->ny / (double)BLOCK_WIDTH);
   if (reference->nz > 1) {
      params->blockNumber[2] = (int)std::ceil((double)reference->nz / (double)BLOCK_WIDTH);
      params->dim = 3;
   }
   else {
      params->blockNumber[2] = 1;
      params->dim = 2;
   }
   params->totalBlockNumber = params->blockNumber[0] * params->blockNumber[1] * params->blockNumber[2];

   params->stepSize = stepSize_block;

   params->percent_to_keep = percentToKeep_opt;

   //number of block that the user wants to keep after _reg_set_active_blocks it will be: the min between params->totalBlockNumber * percentToKeep_block and params->totalBlockNumber - unsuable blocks
   params->activeBlockNumber = (int)((double) params->totalBlockNumber * ((double) percentToKeep_block / (double) 100));
   params->totalBlock = (int *)malloc(params->totalBlockNumber * sizeof(int));

   switch (reference->datatype) {
   case NIFTI_TYPE_FLOAT32:
      _reg_set_active_blocks<float>(reference, params, mask, runningOnGPU);
      break;
   case NIFTI_TYPE_FLOAT64:
      _reg_set_active_blocks<double>(reference, params, mask, runningOnGPU);
      break;
   default:
      reg_print_fct_error("initialise_block_matching_method()");
      reg_print_msg_error("The reference image data type is not supported");
      reg_exit();
      ;
   }
   if (params->activeBlockNumber < 2) {
      reg_print_fct_error("initialise_block_matching_method()");
      reg_print_msg_error("There are no active blocks");
      reg_exit();
   }
#ifndef NDEBUG
   char text[255];
   sprintf(text, "There are %i active block(s) out of %i.",
           params->activeBlockNumber, params->totalBlockNumber);
   reg_print_msg_debug(text)
      #endif
         //params->activeBlock = (int *)malloc(params->activeBlockNumber * sizeof(int));
         params->referencePosition = (float *)malloc(params->activeBlockNumber * params->dim * sizeof(float));
   params->warpedPosition = (float *)malloc(params->activeBlockNumber * params->dim * sizeof(float));

#ifndef NDEBUG
   reg_print_msg_debug("block matching initialisation done.");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template<typename DTYPE>
void block_matching_method2D(nifti_image * reference, nifti_image * warped, _reg_blockMatchingParam *params, int *mask) {
   DTYPE *referencePtr = static_cast<DTYPE *>(reference->data);
   DTYPE *warpedPtr = static_cast<DTYPE *>(warped->data);

   mat44 *referenceMatrix_xyz;
   if (reference->sform_code > 0)
      referenceMatrix_xyz = &(reference->sto_xyz);
   else
      referenceMatrix_xyz = &(reference->qto_xyz);

   unsigned int referenceIndex_start_x;
   unsigned int referenceIndex_start_y;
   unsigned int referenceIndex_end_x;
   unsigned int referenceIndex_end_y;
   int warpedIndex_start_x;
   int warpedIndex_start_y;
   int warpedIndex_end_x;
   int warpedIndex_end_y;

   unsigned int referenceIndex;
   unsigned int warpedIndex;

   unsigned int blockIndex = 0;

   int index, l, m, x, y, z = 0;
   unsigned int i, j;
   int *maskPtr_XY;
   DTYPE *referencePtr_XY, *warpedPtr_XY;
   DTYPE value, bestCC, referenceMean, warpedMean, referenceVar, warpedVar;
   DTYPE voxelNumber, localCC, referenceTemp, warpedTemp;
   float bestDisplacement[3], referencePosition_temp[3], tempPosition[3];

   DTYPE referenceValues[BLOCK_2D_SIZE];
   bool referenceOverlap[BLOCK_2D_SIZE];
   DTYPE warpedValues[BLOCK_2D_SIZE];
   bool warpedOverlap[BLOCK_2D_SIZE];

   params->definedActiveBlockNumber = 0;

   for (j = 0; j < params->blockNumber[1]; j++) {
      referenceIndex_start_y = j * BLOCK_WIDTH;
      referenceIndex_end_y = referenceIndex_start_y + BLOCK_WIDTH;

      for (i = 0; i < params->blockNumber[0]; i++) {
         referenceIndex_start_x = i * BLOCK_WIDTH;
         referenceIndex_end_x = referenceIndex_start_x + BLOCK_WIDTH;

         if (params->totalBlock[blockIndex] > -1) {

            referenceIndex = 0;
            memset(referenceOverlap, 0, BLOCK_2D_SIZE * sizeof(bool));

            for (y = (int) referenceIndex_start_y; y < (int) referenceIndex_end_y; y++) {
               if (y < reference->ny) {
                  index = y * reference->nx + referenceIndex_start_x;
                  for (x = (int) referenceIndex_start_x; x < (int) referenceIndex_end_x; x++) {
                     if (x < reference->nx) {
                        referencePtr_XY = &referencePtr[index];
                        maskPtr_XY = &mask[index];
                        value = *referencePtr_XY;
                        if (value == value && *maskPtr_XY > -1) {
                           referenceValues[referenceIndex] = value;
                           referenceOverlap[referenceIndex] = 1;
                        }
                     }
                     index++;
                     referenceIndex++;
                  }
               }
               else
                  referenceIndex += BLOCK_WIDTH;
            }
            bestCC = params->voxelCaptureRange > 3 ? 0.9 : 0.0;
            bestDisplacement[0] = std::numeric_limits<float>::quiet_NaN();
            bestDisplacement[1] = 0.f;
            bestDisplacement[2] = 0.f;

            // iteration over the warped blocks
            for (m = -1 * params->voxelCaptureRange; m <= params->voxelCaptureRange; m += params->stepSize) {
               warpedIndex_start_y = referenceIndex_start_y + m;
               warpedIndex_end_y = warpedIndex_start_y + BLOCK_WIDTH;
               for (l = -1 * params->voxelCaptureRange; l <= params->voxelCaptureRange; l += params->stepSize) {
                  warpedIndex_start_x = referenceIndex_start_x + l;
                  warpedIndex_end_x = warpedIndex_start_x + BLOCK_WIDTH;

                  warpedIndex = 0;
                  memset(warpedOverlap, 0, BLOCK_2D_SIZE * sizeof(bool));

                  for (y = warpedIndex_start_y; y < warpedIndex_end_y; y++) {
                     if (-1 < y && y < warped->ny) {
                        index = y * warped->nx + warpedIndex_start_x;
                        for (x = warpedIndex_start_x; x < warpedIndex_end_x; x++) {
                           if (-1 < x && x < warped->nx) {
                              warpedPtr_XY = &warpedPtr[index];
                              value = *warpedPtr_XY;
                              if (value == value && *maskPtr_XY > -1) {
                                 warpedValues[warpedIndex] = value;
                                 warpedOverlap[warpedIndex] = 1;
                              }
                           }
                           index++;
                           warpedIndex++;
                        }
                     }
                     else
                        warpedIndex += BLOCK_WIDTH;
                  }
                  referenceMean = 0.0;
                  warpedMean = 0.0;
                  voxelNumber = 0.0;
                  for (int a = 0; a < BLOCK_2D_SIZE; a++) {
                     if (referenceOverlap[a] && warpedOverlap[a]) {
                        referenceMean += referenceValues[a];
                        warpedMean += warpedValues[a];
                        voxelNumber++;
                     }
                  }

                  if (voxelNumber > BLOCK_2D_SIZE / 2) {
                     referenceMean /= voxelNumber;
                     warpedMean /= voxelNumber;

                     referenceVar = 0.0;
                     warpedVar = 0.0;
                     localCC = 0.0;

                     for (int a = 0; a < BLOCK_2D_SIZE; a++) {
                        if (referenceOverlap[a] && warpedOverlap[a]) {
                           referenceTemp = (referenceValues[a] - referenceMean);
                           warpedTemp = (warpedValues[a] - warpedMean);
                           referenceVar += (referenceTemp)* (referenceTemp);
                           warpedVar += (warpedTemp)* (warpedTemp);
                           localCC += (referenceTemp)* (warpedTemp);
                        }
                     }

                     localCC = (referenceVar * warpedVar) > 0.0 ? fabs(localCC / sqrt(referenceVar * warpedVar)) : 0.0;
                     //localCC = fabs(localCC / sqrt(referenceVar * warpedVar));

                     if (localCC > bestCC) {
                        bestCC = localCC + 1.0e-7f;
                        bestDisplacement[0] = (float)l;
                        bestDisplacement[1] = (float)m;
                     }
                  }
               }
            }

            referencePosition_temp[0] = (float)(i * BLOCK_WIDTH);
            referencePosition_temp[1] = (float)(j * BLOCK_WIDTH);
            referencePosition_temp[2] = 0.0f;

            bestDisplacement[0] += referencePosition_temp[0];
            bestDisplacement[1] += referencePosition_temp[1];
            bestDisplacement[2] = 0.0f;

            reg_mat44_mul(referenceMatrix_xyz, referencePosition_temp, tempPosition);
            z = 2 * params->totalBlock[blockIndex];

            params->referencePosition[z] = tempPosition[0];
            params->referencePosition[z + 1] = tempPosition[1];

            reg_mat44_mul(referenceMatrix_xyz, bestDisplacement, tempPosition);

            params->warpedPosition[z] = tempPosition[0];
            params->warpedPosition[z + 1] = tempPosition[1];
            if (bestDisplacement[0] == bestDisplacement[0]) {
               params->definedActiveBlockNumber++;
            }
         }
         blockIndex++;
      }
   }

}
/* *************************************************************** */
template<typename DTYPE>
void block_matching_method3D(nifti_image * reference,
                             nifti_image * warped,
                             _reg_blockMatchingParam *params,
                             int *mask) {
   DTYPE *referencePtr = static_cast<DTYPE *>(reference->data);
   DTYPE *warpedPtr = static_cast<DTYPE *>(warped->data);

   mat44 *referenceMatrix_xyz;
   if (reference->sform_code > 0)
      referenceMatrix_xyz = &(reference->sto_xyz);
   else
      referenceMatrix_xyz = &(reference->qto_xyz);

   unsigned int referenceIndex_start_x;
   unsigned int referenceIndex_start_y;
   unsigned int referenceIndex_start_z;
   unsigned int referenceIndex_end_x;
   unsigned int referenceIndex_end_y;
   unsigned int referenceIndex_end_z;
   int warpedIndex_start_x;
   int warpedIndex_start_y;
   int warpedIndex_start_z;
   int warpedIndex_end_x;
   int warpedIndex_end_y;
   int warpedIndex_end_z;

   int index, l, m, n, x, y, z;
   int i, j, k; //Need to be int for VC++ compiler and OpenMP
   int *maskPtr_Z, *maskPtr_XYZ;
   DTYPE *referencePtr_Z, *referencePtr_XYZ, *warpedPtr_Z, *warpedPtr_XYZ;
   DTYPE value, bestCC, referenceMean, warpedMean, referenceVar, warpedVar;
   DTYPE voxelNumber, localCC, referenceTemp, warpedTemp;
   float bestDisplacement[3], referencePosition_temp[3], tempPosition[3];
   size_t referenceIndex, warpedIndex, blockIndex, tid = 0;

#if defined (_OPENMP)
   int threadNumber = omp_get_max_threads();
   if (threadNumber > 16)
      omp_set_num_threads(16);
   DTYPE referenceValues[16][BLOCK_3D_SIZE];
   DTYPE warpedValues[16][BLOCK_3D_SIZE];
   bool referenceOverlap[16][BLOCK_3D_SIZE];
   bool warpedOverlap[16][BLOCK_3D_SIZE];
#else
   DTYPE referenceValues[1][BLOCK_3D_SIZE];
   DTYPE warpedValues[1][BLOCK_3D_SIZE];
   bool referenceOverlap[1][BLOCK_3D_SIZE];
   bool warpedOverlap[1][BLOCK_3D_SIZE];
#endif

   params->definedActiveBlockNumber = 0;

#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(params, reference, warped, referencePtr, warpedPtr, mask, referenceMatrix_xyz, \
   referenceOverlap, warpedOverlap, referenceValues, warpedValues) \
   private(i, j, k, l, m, n, x, y, z, blockIndex, referenceIndex, \
   index, tid, referencePtr_Z, referencePtr_XYZ, warpedPtr_Z, warpedPtr_XYZ, \
   maskPtr_Z, maskPtr_XYZ, value, bestCC, bestDisplacement, \
   referenceIndex_start_x, referenceIndex_start_y, referenceIndex_start_z, \
   referenceIndex_end_x, referenceIndex_end_y, referenceIndex_end_z, \
   warpedIndex_start_x, warpedIndex_start_y, warpedIndex_start_z, \
   warpedIndex_end_x, warpedIndex_end_y, warpedIndex_end_z, \
   warpedIndex, referencePosition_temp, tempPosition, referenceTemp, warpedTemp, \
   referenceMean, referenceVar, warpedMean, warpedVar, voxelNumber,localCC)
#endif
   for (k = 0; k < (int)params->blockNumber[2]; k++) {
#if defined (_OPENMP)
      tid = omp_get_thread_num();
#endif
      blockIndex = k * params->blockNumber[0] * params->blockNumber[1];
      referenceIndex_start_z = k * BLOCK_WIDTH;
      referenceIndex_end_z = referenceIndex_start_z + BLOCK_WIDTH;

      for (j = 0; j < (int)params->blockNumber[1]; j++) {
         referenceIndex_start_y = j * BLOCK_WIDTH;
         referenceIndex_end_y = referenceIndex_start_y + BLOCK_WIDTH;

         for (i = 0; i < (int)params->blockNumber[0]; i++) {
            referenceIndex_start_x = i * BLOCK_WIDTH;
            referenceIndex_end_x = referenceIndex_start_x + BLOCK_WIDTH;

            if (params->totalBlock[blockIndex] > -1) {
               referenceIndex = 0;
               memset(referenceOverlap[tid], 0, BLOCK_3D_SIZE * sizeof(bool));
               for (z = (int)referenceIndex_start_z; z < (int)referenceIndex_end_z; z++) {
                  if (z < reference->nz) {
                     index = z * reference->nx * reference->ny;
                     referencePtr_Z = &referencePtr[index];
                     maskPtr_Z = &mask[index];
                     for (y = (int)referenceIndex_start_y; y < (int)referenceIndex_end_y; y++) {
                        if (y < reference->ny) {
                           index = y * reference->nx + referenceIndex_start_x;
                           for (x = (int)referenceIndex_start_x; x < (int)referenceIndex_end_x; x++) {
                              if (x < reference->nx) {
                                 referencePtr_XYZ = &referencePtr_Z[index];
                                 maskPtr_XYZ = &maskPtr_Z[index];
                                 value = *referencePtr_XYZ;
                                 if (value == value && *maskPtr_XYZ > -1) {
                                    referenceValues[tid][referenceIndex] = value;
                                    referenceOverlap[tid][referenceIndex] = 1;
                                 }
                              }
                              index++;
                              referenceIndex++;
                           }
                        }
                        else
                           referenceIndex += BLOCK_WIDTH;
                     }
                  }
                  else
                     referenceIndex += BLOCK_WIDTH * BLOCK_WIDTH;
               }
               bestCC = params->voxelCaptureRange > 3 ? 0.9 : 0.0; //only when misaligned images are registered
               bestDisplacement[0] = std::numeric_limits<float>::quiet_NaN();
               bestDisplacement[1] = 0.f;
               bestDisplacement[2] = 0.f;

               // iteration over the warped blocks
               for (n = -1 * params->voxelCaptureRange; n <= params->voxelCaptureRange; n += params->stepSize) {
                  warpedIndex_start_z = referenceIndex_start_z + n;
                  warpedIndex_end_z = warpedIndex_start_z + BLOCK_WIDTH;
                  for (m = -1 * params->voxelCaptureRange; m <= params->voxelCaptureRange; m += params->stepSize) {
                     warpedIndex_start_y = referenceIndex_start_y + m;
                     warpedIndex_end_y = warpedIndex_start_y + BLOCK_WIDTH;
                     for (l = -1 * params->voxelCaptureRange; l <= params->voxelCaptureRange; l += params->stepSize) {

                        warpedIndex_start_x = referenceIndex_start_x + l;
                        warpedIndex_end_x = warpedIndex_start_x + BLOCK_WIDTH;
                        warpedIndex = 0;
                        memset(warpedOverlap[tid], 0, BLOCK_3D_SIZE * sizeof(bool));
                        for (z = warpedIndex_start_z; z < warpedIndex_end_z; z++) {
                           if (-1 < z && z < warped->nz) {
                              index = z * warped->nx * warped->ny;
                              warpedPtr_Z = &warpedPtr[index];
                              maskPtr_Z = &mask[index];
                              for (y = warpedIndex_start_y; y < warpedIndex_end_y; y++) {
                                 if (-1 < y && y < warped->ny) {
                                    index = y * warped->nx + warpedIndex_start_x;
                                    for (x = warpedIndex_start_x; x < warpedIndex_end_x; x++) {
                                       if (-1 < x && x < warped->nx) {
                                          warpedPtr_XYZ = &warpedPtr_Z[index];
                                          maskPtr_XYZ = &maskPtr_Z[index];
                                          value = *warpedPtr_XYZ;
                                          if (value == value && *maskPtr_XYZ > -1) {
                                             warpedValues[tid][warpedIndex] = value;
                                             warpedOverlap[tid][warpedIndex] = 1;
                                          }
                                       }
                                       index++;
                                       warpedIndex++;
                                    }
                                 }
                                 else
                                    warpedIndex += BLOCK_WIDTH;
                              }
                           }
                           else
                              warpedIndex += BLOCK_WIDTH * BLOCK_WIDTH;
                        }
                        referenceMean = 0.0;
                        warpedMean = 0.0;
                        voxelNumber = 0.0;
                        for (int a = 0; a < BLOCK_3D_SIZE; a++) {
                           if (referenceOverlap[tid][a] && warpedOverlap[tid][a]) {
                              referenceMean += referenceValues[tid][a];
                              warpedMean += warpedValues[tid][a];
                              voxelNumber++;
                           }
                        }

                        if (voxelNumber > BLOCK_3D_SIZE / 2) {
                           referenceMean /= voxelNumber;
                           warpedMean /= voxelNumber;

                           referenceVar = 0.0;
                           warpedVar = 0.0;
                           localCC = 0.0;

                           for (int a = 0; a < BLOCK_3D_SIZE; a++) {
                              if (referenceOverlap[tid][a] && warpedOverlap[tid][a]) {
                                 referenceTemp = (referenceValues[tid][a] - referenceMean);
                                 warpedTemp = (warpedValues[tid][a] - warpedMean);
                                 referenceVar += (referenceTemp)* (referenceTemp);
                                 warpedVar += (warpedTemp)* (warpedTemp);
                                 localCC += (referenceTemp)* (warpedTemp);
                              }
                           }
                           localCC = (referenceVar * warpedVar) > 0.0 ? fabs(localCC / sqrt(referenceVar * warpedVar)) : 0.0;

                           if (localCC > bestCC) {
                              bestCC = localCC + 1.0e-7f;
                              bestDisplacement[0] = (float)l;
                              bestDisplacement[1] = (float)m;
                              bestDisplacement[2] = (float)n;
                           }
                        }
                     }
                  }
               }
               //if (bestDisplacement[0] == bestDisplacement[0]) {
               referencePosition_temp[0] = (float)(i * BLOCK_WIDTH);
               referencePosition_temp[1] = (float)(j * BLOCK_WIDTH);
               referencePosition_temp[2] = (float)(k * BLOCK_WIDTH);

               bestDisplacement[0] += referencePosition_temp[0];
               bestDisplacement[1] += referencePosition_temp[1];
               bestDisplacement[2] += referencePosition_temp[2];

               reg_mat44_mul(referenceMatrix_xyz, referencePosition_temp, tempPosition);
               z = 3 * params->totalBlock[blockIndex];
               params->referencePosition[z] = tempPosition[0];
               params->referencePosition[z+1] = tempPosition[1];
               params->referencePosition[z+2] = tempPosition[2];

               reg_mat44_mul(referenceMatrix_xyz, bestDisplacement, tempPosition);
               params->warpedPosition[z] = tempPosition[0];
               params->warpedPosition[z + 1] = tempPosition[1];
               params->warpedPosition[z + 2] = tempPosition[2];
               if (bestDisplacement[0] == bestDisplacement[0]) {
                  params->definedActiveBlockNumber++;
               }
            }
            blockIndex++;
         }
      }
   }

#if defined (_OPENMP)
   omp_set_num_threads(threadNumber);
#endif
}
/* *************************************************************** */
// Block matching interface function
void block_matching_method(nifti_image * reference, nifti_image * warped, _reg_blockMatchingParam *params, int *mask) {
   if (reference->datatype != warped->datatype) {
      reg_print_fct_error("block_matching_method");
      reg_print_msg_error("Both input images are expected to be of the same type");
   }
   if (reference->nz == 1) {
      switch (reference->datatype) {
      case NIFTI_TYPE_FLOAT64:
         block_matching_method2D<double>(reference, warped, params, mask);
         break;
      case NIFTI_TYPE_FLOAT32:
         block_matching_method2D<float>(reference, warped, params, mask);
         break;
      default:
         reg_print_fct_error("block_matching_method");
         reg_print_msg_error("The reference image data type is not supported");
         reg_exit();
      }
   } else {
      switch (reference->datatype) {
      case NIFTI_TYPE_FLOAT64:
         block_matching_method3D<double>(reference, warped, params, mask);
         break;
      case NIFTI_TYPE_FLOAT32:
         block_matching_method3D<float>(reference, warped, params, mask);
         break;
      default:
         reg_print_fct_error("block_matching_method");
         reg_print_msg_error("The reference image data type is not supported");
         reg_exit();
      }
   }
}
/* *************************************************************** */
// Find the optimal transformation - affine or rigid
void optimize(_reg_blockMatchingParam *params,
              mat44 *transformation_matrix,
              bool affine)
{
   // The block matching provide correspondences in millimeters
   // in the space of the reference image. All warped image coordinates
   // are updated to be in the original warped space
   //    mat44 inverseMatrix = nifti_mat44_inverse(*transformation_matrix);
   if (params->blockNumber[2] == 1)  // 2D images
   {
      //First let's check if we have enough correpondance points to estimate a transfomation
      if(affine) {
         //3 = minimum number of corespondances needed
         if(params->definedActiveBlockNumber < 6)
         {
            char text[255];
            sprintf(text, "%i correspondances between blocks were found", params->definedActiveBlockNumber);
            reg_print_msg_error(text);
            reg_print_msg_error("Not enough correspondences were found - it is impossible to estimate an affine transfomation");
            reg_exit();
         }
      } else {
         if(params->definedActiveBlockNumber < 4)
         {
            char text[255];
            sprintf(text, "%i correspondances between blocks were found", params->definedActiveBlockNumber);
            reg_print_msg_error(text);
            reg_print_msg_error("Not enough correspondences were found - it is impossible to estimate a rigid transfomation");
            reg_exit();
         }
      }

      float in[2];
      float out[2];
      std::vector<float> referencePositionVect;
      std::vector<float> warpedPositionVect;
      int nbNonNaNBlock = 0;
      for (size_t i = 0; i < static_cast<size_t>(params->activeBlockNumber); ++i) {
         size_t index = 2 * i;
         in[0] = params->warpedPosition[index];
         in[1] = params->warpedPosition[index + 1];
         //Can have undefined = NaN in the warped image now -
         //to not loose the correspondance - so check that:
         if(in[0] == in[0]){
            reg_mat33_mul(transformation_matrix, in, out);

            referencePositionVect.push_back(params->referencePosition[index]);
            referencePositionVect.push_back(params->referencePosition[index+1]);
            warpedPositionVect.push_back(out[0]);
            warpedPositionVect.push_back(out[1]);
            nbNonNaNBlock++;
         }
      }
      optimize_2D(&referencePositionVect[0], &warpedPositionVect[0],
            nbNonNaNBlock, params->percent_to_keep,
            MAX_ITERATIONS, TOLERANCE,
            transformation_matrix, affine);
   }
   else  // 3D images
   {
      //First let's check if we have enough correpondance points to estimate a transfomation
      if(affine) {
         //4 = minimum number of corespondances needed
         if(params->definedActiveBlockNumber < 8)
         {
            char text[255];
            sprintf(text, "%i correspondances between blocks were found", params->definedActiveBlockNumber);
            reg_print_msg_error(text);
            reg_print_msg_error("Not enough correspondances were found - it is impossible to estimate an affine tranfomation");
            reg_exit();
         }
      } else {
         if(params->definedActiveBlockNumber < 4)
         {
            char text[255];
            sprintf(text, "%i correspondances between blocks were found", params->definedActiveBlockNumber);
            reg_print_msg_error(text);
            reg_print_msg_error("Not enough correspondances were found - it is impossible to estimate a rigid tranfomation");
            reg_exit();
         }
      }

      float in[3];
      float out[3];
      std::vector<float> referencePositionVect;
      std::vector<float> warpedPositionVect;
      int nbNonNaNBlock = 0;
      for (size_t i = 0; i < static_cast<size_t>(params->activeBlockNumber); ++i) {
         size_t index = 3 * i;
         in[0] = params->warpedPosition[index];
         in[1] = params->warpedPosition[index + 1];
         in[2] = params->warpedPosition[index + 2];
         //Can have undefined = NaN in the warped image now -
         //to not loose the correspondance - so check that:
         if(in[0] == in[0]){
            reg_mat44_mul(transformation_matrix, in, out);

            referencePositionVect.push_back(params->referencePosition[index]);
            referencePositionVect.push_back(params->referencePosition[index+1]);
            referencePositionVect.push_back(params->referencePosition[index+2]);
            warpedPositionVect.push_back(out[0]);
            warpedPositionVect.push_back(out[1]);
            warpedPositionVect.push_back(out[2]);
            nbNonNaNBlock++;
         }
      }
      optimize_3D(&referencePositionVect[0], &warpedPositionVect[0],
            nbNonNaNBlock, params->percent_to_keep,
            MAX_ITERATIONS, TOLERANCE,
            transformation_matrix, affine);
   }
}
/* *************************************************************** */
