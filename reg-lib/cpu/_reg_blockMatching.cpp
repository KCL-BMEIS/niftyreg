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
#include <limits>
#include <cmath>
/* *************************************************************** */
//template<class DTYPE>
//void _reg_set_all_active_blocks(nifti_image *referenceImage, _reg_blockMatchingParam *params, int *mask, bool runningOnGPU) {
//    const size_t totalBlockNumber = params->blockNumber[0] * params->blockNumber[1] * params->blockNumber[2];
//    etc ...
//}
template<class DTYPE>
void _reg_set_active_blocks(nifti_image *referenceImage, _reg_blockMatchingParam *params, int *mask, bool runningOnGPU) {
    const size_t totalBlockNumber = params->blockNumber[0] * params->blockNumber[1] * params->blockNumber[2];
    float *varianceArray = (float *)malloc(totalBlockNumber * sizeof(float));
    int *indexArray = (int *)malloc(totalBlockNumber * sizeof(int));

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
        for (int k = 0; k < params->blockNumber[2]; k++) {
            for (int j = 0; j < params->blockNumber[1]; j++) {
                for (int i = 0; i < params->blockNumber[0]; i++) {

                    for (unsigned int n = 0; n < BLOCK_3D_SIZE; n++)
                        referenceValues[n] = (DTYPE)std::numeric_limits<float>::quiet_NaN();

                    float mean = 0.0f;
                    float voxelNumber = 0.0f;
                    int coord = 0;
                    for (int z = k * BLOCK_WIDTH; z < (k + 1) * BLOCK_WIDTH; z++) {
                        if (z < referenceImage->nz) {
                            index = z * referenceImage->nx * referenceImage->ny;
                            DTYPE *referencePtrZ = &referencePtr[index];
                            int *maskPtrZ = &maskPtr[index];
                            for (int y = j * BLOCK_WIDTH; y < (j + 1) * BLOCK_WIDTH; y++) {
                                if (y < referenceImage->ny) {
                                    index = y * referenceImage->nx + i * BLOCK_WIDTH;
                                    DTYPE *referencePtrXYZ = &referencePtrZ[index];
                                    int *maskPtrXYZ = &maskPtrZ[index];
                                    for (int x = i * BLOCK_WIDTH; x < (i + 1) * BLOCK_WIDTH; x++) {
                                        if (x < referenceImage->nx) {
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
        for (int j = 0; j < params->blockNumber[1]; j++) {
            for (int i = 0; i < params->blockNumber[0]; i++) {

                for (unsigned int n = 0; n < BLOCK_2D_SIZE; n++)
                    referenceValues[n] = (DTYPE)std::numeric_limits<float>::quiet_NaN();

                float mean = 0.0f;
                float voxelNumber = 0.0f;
                int coord = 0;

                for (int y = j * BLOCK_WIDTH; y < (j + 1) * BLOCK_WIDTH; y++) {
                    if (y < referenceImage->ny) {
                        index = y * referenceImage->nx + i * BLOCK_WIDTH;
                        DTYPE *referencePtrXY = &referencePtr[index];
                        int *maskPtrXY = &maskPtr[index];
                        for (int x = i * BLOCK_WIDTH; x < (i + 1) * BLOCK_WIDTH; x++) {
                            if (x < referenceImage->nx) {
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

    //DEBUG TEMP
    //unusableBlock = 0;
    //DEBUG TEMP

    params->activeBlockNumber = params->activeBlockNumber < ((int)totalBlockNumber - unusableBlock) ? params->activeBlockNumber : (totalBlockNumber - unusableBlock);

    reg_heapSort(varianceArray, indexArray, totalBlockNumber);
    //DEBUG
    //for(int i=0;i<totalBlockNumber;i++) {
    //std::cout<<"varianceArray[i]="<< varianceArray[i] << std::endl;
    //std::cout<<"indexArray[i]="<< indexArray[i] << std::endl;
    //}
    //DEBUG
    int *indexArrayPtr = &indexArray[totalBlockNumber - 1];
    int count = 0;
    for (int i = 0; i < params->activeBlockNumber; i++) {
        params->activeBlock[*indexArrayPtr--] = count++;
    }
    for (size_t i = params->activeBlockNumber; i < totalBlockNumber; ++i) {
        params->activeBlock[*indexArrayPtr--] = -1;
    }
    //DEBUG
    //for(int i=0;i<totalBlockNumber;i++) {
    //std::cout<<"i="<< i << std::endl;
    //std::cout<<"params->activeBlock[i]="<< params->activeBlock[i] << std::endl;
    //}
    //DEBUG
    count = 0;
    if (runningOnGPU) {
        for (size_t i = 0; i < totalBlockNumber; ++i) {
            if (params->activeBlock[i] != -1) {
                params->activeBlock[i] = -1;
                params->activeBlock[count] = i;
                ++count;
            }
        }
    }

    free(varianceArray);
    free(indexArray);
}
/* *************************************************************** */
void initialise_block_matching_method(nifti_image * reference, _reg_blockMatchingParam *params, int percentToKeep_block, int percentToKeep_opt, int stepSize_block, int *mask, bool runningOnGPU) {
    if (params->activeBlock != NULL) {
        free(params->activeBlock);
        params->activeBlock = NULL;
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
    //Why ceil and not floor ?
    params->blockNumber[0] = (int)std::ceil((double)reference->nx / (double)BLOCK_WIDTH);
    params->blockNumber[1] = (int)std::ceil((double)reference->ny / (double)BLOCK_WIDTH);
    if (reference->nz > 1) {
        params->blockNumber[2] = (int)std::ceil((double)reference->nz / (double)BLOCK_WIDTH);
    }
    else {
        params->blockNumber[2] = 1;
    }

    params->stepSize = stepSize_block;

    params->percent_to_keep = percentToKeep_opt;
    params->activeBlockNumber = params->blockNumber[0] * params->blockNumber[1] * params->blockNumber[2] * percentToKeep_block / 100;

    params->activeBlock = (int *)malloc(params->blockNumber[0] * params->blockNumber[1] * params->blockNumber[2] * sizeof(int));
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
        reg_exit(1);
        ;
    }
    if (params->activeBlockNumber < 2) {
        reg_print_fct_error("initialise_block_matching_method()");
        reg_print_msg_error("There are no active blocks");
        reg_exit(1);
    }
#ifndef NDEBUG
    char text[255];
    sprintf(text, "There are %i active block(s) out of %i.",
        params->activeBlockNumber, params->blockNumber[0] * params->blockNumber[1] * params->blockNumber[2]);
    reg_print_msg_debug(text)
#endif
    //if (reference->nz > 1) {
    params->referencePosition = (float *)malloc(params->activeBlockNumber * 3 * sizeof(float));
    params->warpedPosition = (float *)malloc(params->activeBlockNumber * 3 * sizeof(float));
    //} else {
    //	params->referencePosition = (float *) malloc(params->activeBlockNumber * 2 * sizeof(float));
    //	params->warpedPosition = (float *) malloc(params->activeBlockNumber * 2 * sizeof(float));
    //}
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

    int referenceIndex_start_x;
    int referenceIndex_start_y;
    int referenceIndex_end_x;
    int referenceIndex_end_y;
    int warpedIndex_start_x;
    int warpedIndex_start_y;
    int warpedIndex_end_x;
    int warpedIndex_end_y;

    unsigned int referenceIndex;
    unsigned int warpedIndex;

    unsigned int blockIndex = 0;
    unsigned int activeBlockIndex = 0;
    params->definedActiveBlock = 0;

    int index, i, j, k, l, m, n, x, y, z = 0;
    int *maskPtr_XY;
    DTYPE *referencePtr_XY, *warpedPtr_XY;
    DTYPE value, bestCC, referenceMean, warpedMean, referenceVar, warpedVar;
    DTYPE voxelNumber, localCC, referenceTemp, warpedTemp;
    float bestDisplacement[3], referencePosition_temp[3], tempPosition[3];

    DTYPE referenceValues[BLOCK_2D_SIZE];
    bool referenceOverlap[BLOCK_2D_SIZE];
    DTYPE warpedValues[BLOCK_2D_SIZE];
    bool warpedOverlap[BLOCK_2D_SIZE];

    float *temp_reference_position = (float *)malloc(2 * params->activeBlockNumber * sizeof(float));
    float *temp_warped_position = (float *)malloc(2 * params->activeBlockNumber * sizeof(float));

    for (i = 0; i < 2 * params->activeBlockNumber; i += 2) {
        temp_reference_position[i] = std::numeric_limits<float>::quiet_NaN();
    }


    for (j = 0; j < params->blockNumber[1]; j++) {
        referenceIndex_start_y = j * BLOCK_WIDTH;
        referenceIndex_end_y = referenceIndex_start_y + BLOCK_WIDTH;

        for (i = 0; i < params->blockNumber[0]; i++) {
            referenceIndex_start_x = i * BLOCK_WIDTH;
            referenceIndex_end_x = referenceIndex_start_x + BLOCK_WIDTH;

            if (params->activeBlock[blockIndex] > -1) {

                referenceIndex = 0;
                memset(referenceOverlap, 0, BLOCK_2D_SIZE * sizeof(bool));

                for (y = referenceIndex_start_y; y < referenceIndex_end_y; y++) {
                    if (-1 < y && y < reference->ny) {
                        index = y * reference->nx + referenceIndex_start_x;
                        referencePtr_XY = &referencePtr[index];
                        maskPtr_XY = &mask[index];
                        for (x = referenceIndex_start_x; x < referenceIndex_end_x; x++) {
                            if (-1 < x && x < reference->nx) {
                                value = *referencePtr_XY;
                                //Why ????? value==value is for NaN
                                //if (value == value && value != 0. && *maskPtr_XY > -1) {
                                if (value == value && *maskPtr_XY > -1) {
                                    referenceValues[referenceIndex] = value;
                                    referenceOverlap[referenceIndex] = 1;
                                }
                            }
                            referencePtr_XY++;
                            maskPtr_XY++;
                            referenceIndex++;
                        }
                    }
                    else
                        referenceIndex += BLOCK_WIDTH;
                }
                bestCC = params->voxelCaptureRange > 3 ? 0.9 : 0.0; //only when misaligned images are registered
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
                                warpedPtr_XY = &warpedPtr[index];

                                for (x = warpedIndex_start_x; x < warpedIndex_end_x; x++) {
                                    if (-1 < x && x < warped->nx) {
                                        value = *warpedPtr_XY;
                                        //if (value == value && value != 0. && *maskPtr_XY > -1) {
                                        if (value == value) {
                                            warpedValues[warpedIndex] = value;
                                            warpedOverlap[warpedIndex] = 1;
                                        }
                                    }
                                    warpedPtr_XY++;
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

                            localCC = (referenceVar * warpedVar) > 0.0 ? fabs(localCC / sqrt(referenceVar * warpedVar)) : 0;

                            if (localCC > bestCC) {
                                bestCC = localCC;
                                bestDisplacement[0] = (float)l;
                                bestDisplacement[1] = (float)m;
                            }
                        }
                    }
                }

                if (bestDisplacement[0] == bestDisplacement[0]) {
                    referencePosition_temp[0] = (float)(i * BLOCK_WIDTH);
                    referencePosition_temp[1] = (float)(j * BLOCK_WIDTH);
                    referencePosition_temp[2] = 0.0f;

                    if (i == 4 && j == 1) {
                        std::cout << "2nd block" << std::endl;
                    }

                    bestDisplacement[0] += referencePosition_temp[0];
                    bestDisplacement[1] += referencePosition_temp[1];
                    bestDisplacement[2] = 0.0f;

                    reg_mat44_mul(referenceMatrix_xyz, referencePosition_temp, tempPosition);
                    temp_reference_position[z] = tempPosition[0];
                    temp_reference_position[z + 1] = tempPosition[1];
                    reg_mat44_mul(referenceMatrix_xyz, bestDisplacement, tempPosition);
                    temp_warped_position[z] = tempPosition[0];
                    temp_warped_position[z + 1] = tempPosition[1];
                    z = z + 2;
                }
                else {
                    //NAN - IN THEORIE WE SHOULD NEVER ENTER HERE IN REAL LIFE - ONLY FOR THE UNIT TESTS

                    bestDisplacement[0] = 0.0;
                    bestDisplacement[1] = 0.0;

                    referencePosition_temp[0] = (float)(i * BLOCK_WIDTH);
                    referencePosition_temp[1] = (float)(j * BLOCK_WIDTH);
                    referencePosition_temp[2] = 0.0f;

                    if (i == 4 && j == 1) {
                        std::cout << "2nd block" << std::endl;
                    }

                    bestDisplacement[0] += referencePosition_temp[0];
                    bestDisplacement[1] += referencePosition_temp[1];
                    bestDisplacement[2] = 0.0f;

                    reg_mat44_mul(referenceMatrix_xyz, referencePosition_temp, tempPosition);
                    temp_reference_position[z] = tempPosition[0];
                    temp_reference_position[z + 1] = tempPosition[1];
                    reg_mat44_mul(referenceMatrix_xyz, bestDisplacement, tempPosition);
                    temp_warped_position[z] = tempPosition[0];
                    temp_warped_position[z + 1] = tempPosition[1];
                    z = z + 2;
                }
            }
            blockIndex++;
        }
    }

    // Removing the NaNs and defining the number of active block
    params->definedActiveBlock = 0;
    j = 0;
    for (i = 0; i < 2 * params->activeBlockNumber; i += 2) {
        if (temp_reference_position[i] == temp_reference_position[i]) {
            params->referencePosition[j] = temp_reference_position[i];
            params->referencePosition[j + 1] = temp_reference_position[i + 1];
            params->warpedPosition[j] = temp_warped_position[i];
            params->warpedPosition[j + 1] = temp_warped_position[i + 1];
            params->definedActiveBlock++;
            j += 2;
        }
    }
    free(temp_reference_position);
    free(temp_warped_position);
}
/* *************************************************************** */
template<typename DTYPE>
void block_matching_method3D(nifti_image * reference, nifti_image * warped, _reg_blockMatchingParam *params, int *mask) {
    DTYPE *referencePtr = static_cast<DTYPE *>(reference->data);
    DTYPE *warpedPtr = static_cast<DTYPE *>(warped->data);

    mat44 *referenceMatrix_xyz;
    if (reference->sform_code > 0)
        referenceMatrix_xyz = &(reference->sto_xyz);
    else
        referenceMatrix_xyz = &(reference->qto_xyz);

    int referenceIndex_start_x;
    int referenceIndex_start_y;
    int referenceIndex_start_z;
    int referenceIndex_end_x;
    int referenceIndex_end_y;
    int referenceIndex_end_z;
    int warpedIndex_start_x;
    int warpedIndex_start_y;
    int warpedIndex_start_z;
    int warpedIndex_end_x;
    int warpedIndex_end_y;
    int warpedIndex_end_z;

    int index, i, j, k, l, m, n, x, y, z;
    int *maskPtr_Z, *maskPtr_XYZ;
    DTYPE *referencePtr_Z, *referencePtr_XYZ, *warpedPtr_Z, *warpedPtr_XYZ;
    DTYPE value, bestCC, referenceMean, warpedMean, referenceVar, warpedVar;
    DTYPE voxelNumber, localCC, referenceTemp, warpedTemp;
    float bestDisplacement[3], referencePosition_temp[3], tempPosition[3];
    size_t referenceIndex, warpedIndex, blockIndex, tid = 0;
    params->definedActiveBlock = 0;
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

    float *temp_reference_position = (float *)malloc(3 * params->activeBlockNumber * sizeof(float));
    float *temp_warped_position = (float *)malloc(3 * params->activeBlockNumber * sizeof(float));
    for (i = 0; i < 3 * params->activeBlockNumber; i += 3)
        temp_reference_position[i] = std::numeric_limits<float>::quiet_NaN();

#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(params, reference, warped, referencePtr, warpedPtr, mask, referenceMatrix_xyz, \
          referenceOverlap, warpedOverlap, referenceValues, warpedValues, \
          temp_reference_position, temp_warped_position) \
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
    for (k = 0; k < params->blockNumber[2]; k++) {
#if defined (_OPENMP)
        tid = omp_get_thread_num();
#endif
        blockIndex = k * params->blockNumber[0] * params->blockNumber[1];
        referenceIndex_start_z = k * BLOCK_WIDTH;
        referenceIndex_end_z = referenceIndex_start_z + BLOCK_WIDTH;

        for (j = 0; j < params->blockNumber[1]; j++) {
            referenceIndex_start_y = j * BLOCK_WIDTH;
            referenceIndex_end_y = referenceIndex_start_y + BLOCK_WIDTH;

            for (i = 0; i < params->blockNumber[0]; i++) {
                referenceIndex_start_x = i * BLOCK_WIDTH;
                referenceIndex_end_x = referenceIndex_start_x + BLOCK_WIDTH;

                if (params->activeBlock[blockIndex] > -1) {
                    referenceIndex = 0;
                    memset(referenceOverlap[tid], 0, BLOCK_3D_SIZE * sizeof(bool));
                    for (z = referenceIndex_start_z; z < referenceIndex_end_z; z++) {
                        if (-1 < z && z < reference->nz) {
                            index = z * reference->nx * reference->ny;
                            referencePtr_Z = &referencePtr[index];
                            maskPtr_Z = &mask[index];
                            for (y = referenceIndex_start_y; y < referenceIndex_end_y; y++) {
                                if (-1 < y && y < reference->ny) {
                                    index = y * reference->nx + referenceIndex_start_x;
                                    referencePtr_XYZ = &referencePtr_Z[index];
                                    maskPtr_XYZ = &maskPtr_Z[index];
                                    for (x = referenceIndex_start_x; x < referenceIndex_end_x; x++) {
                                        if (-1 < x && x < reference->nx) {
                                            value = *referencePtr_XYZ;
                                            if (value == value && *maskPtr_XYZ > -1) {
                                                referenceValues[tid][referenceIndex] = value;
                                                referenceOverlap[tid][referenceIndex] = 1;
                                            }
                                        }
                                        referencePtr_XYZ++;
                                        maskPtr_XYZ++;
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
                                        int *maskPtr_Z = &mask[index];
                                        for (y = warpedIndex_start_y; y < warpedIndex_end_y; y++) {
                                            if (-1 < y && y < warped->ny) {
                                                index = y * warped->nx + warpedIndex_start_x;
                                                warpedPtr_XYZ = &warpedPtr_Z[index];
                                                int *maskPtr_XYZ = &maskPtr_Z[index];
                                                for (x = warpedIndex_start_x; x < warpedIndex_end_x; x++) {
                                                    if (-1 < x && x < warped->nx) {
                                                        value = *warpedPtr_XYZ;
                                                        if (value == value && *maskPtr_XYZ > -1) {
                                                            warpedValues[tid][warpedIndex] = value;
                                                            warpedOverlap[tid][warpedIndex] = 1;
                                                        }
                                                    }
                                                    warpedPtr_XYZ++;
                                                    warpedIndex++;
                                                    maskPtr_XYZ++;
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

                                    localCC = fabs(localCC / sqrt(referenceVar * warpedVar));
                                    /*bool predicate = i * BLOCK_WIDTH == 16 && j * BLOCK_WIDTH == 24 && k * BLOCK_WIDTH == 24;
                                     if (predicate && 0.981295 - localCC < 0.04 && fabs(0.981295 - localCC) >= 0)
                                     printf("C|%d-%d-%d|%.0f|TMN:%f|TVR:%f|RMN:%f|RVR:%f|LCC:%lf|BCC:%lf\n", l, m, n, voxelNumber, referenceMean, referenceVar, warpedMean, warpedVar, localCC, bestCC);
                                     //*/

                                    //hack for Marc's integration tests
                                    //									if (localCC > bestCC || (fabs(localCC - 0.981295)<0.000001 && fabs(bestCC-0.981295)<0.000001)) {
                                    if (localCC > bestCC) {
                                        bestCC = localCC;
                                        bestDisplacement[0] = (float)l;
                                        bestDisplacement[1] = (float)m;
                                        bestDisplacement[2] = (float)n;
                                    }
                                    /*bool predicate = i * BLOCK_WIDTH == 16 && j * BLOCK_WIDTH == 24 && k * BLOCK_WIDTH == 24;
                                     if (predicate )
                                     printf("C|%d-%d-%d|%f-%f-%f\n", l, m, n, bestDisplacement[0], bestDisplacement[1], bestDisplacement[2]);*/

                                }
                            }
                        }
                    }
                    if (bestDisplacement[0] == bestDisplacement[0]) {
                        referencePosition_temp[0] = (float)(i * BLOCK_WIDTH);
                        referencePosition_temp[1] = (float)(j * BLOCK_WIDTH);
                        referencePosition_temp[2] = (float)(k * BLOCK_WIDTH);

                        bestDisplacement[0] += referencePosition_temp[0];
                        bestDisplacement[1] += referencePosition_temp[1];
                        bestDisplacement[2] += referencePosition_temp[2];

                        reg_mat44_mul(referenceMatrix_xyz, referencePosition_temp, tempPosition);
                        z = 3 * params->activeBlock[blockIndex];
                        temp_reference_position[z] = tempPosition[0];
                        temp_reference_position[z + 1] = tempPosition[1];
                        temp_reference_position[z + 2] = tempPosition[2];
                        reg_mat44_mul(referenceMatrix_xyz, bestDisplacement, tempPosition);
                        temp_warped_position[z] = tempPosition[0];
                        temp_warped_position[z + 1] = tempPosition[1];
                        temp_warped_position[z + 2] = tempPosition[2];
                    }
                }
                blockIndex++;
            }
        }
    }

    // Removing the NaNs and defining the number of active block
    params->definedActiveBlock = 0;
    j = 0;
    for (i = 0; i < 3 * params->activeBlockNumber; i += 3) {
        if (temp_reference_position[i] == temp_reference_position[i]) {
            params->referencePosition[j] = temp_reference_position[i];
            params->referencePosition[j + 1] = temp_reference_position[i + 1];
            params->referencePosition[j + 2] = temp_reference_position[i + 2];
            params->warpedPosition[j] = temp_warped_position[i];
            params->warpedPosition[j + 1] = temp_warped_position[i + 1];
            params->warpedPosition[j + 2] = temp_warped_position[i + 2];
            params->definedActiveBlock++;
            j += 3;
        }
    }
    free(temp_reference_position);
    free(temp_warped_position);

#if defined (_OPENMP)
    omp_set_num_threads(threadNumber);
#endif
}
/* *************************************************************** */
template<typename DTYPE>
void block_matching_method2D3D(nifti_image * reference, nifti_image * warped, _reg_blockMatchingParam *params, int *mask, int dim) {

    DTYPE *referencePtr = static_cast<DTYPE *>(reference->data);
    DTYPE *warpedPtr = static_cast<DTYPE *>(warped->data);

    //DEBUG
    /*std::cout << "warped voxel values" << std::endl;
    for (int j = 0; j < warped->ny; j++) {
        for (int i = 0; i < warped->nx; i++) {
            int index = i + warped->nx * j;
            if (i == (warped->nx - 2)) {
                std::cout << "value=" << warpedPtr[index] << std::endl;
            }
        }
    }*/
    //DEBUG

    unsigned int BLOCK_SIZE = 0;
    //dim is 2 or 3!
    BLOCK_SIZE = std::pow(BLOCK_WIDTH,dim);

    mat44 *referenceMatrix_xyz;
    if (reference->sform_code > 0) {
        referenceMatrix_xyz = &(reference->sto_xyz);
    }
    else {
        referenceMatrix_xyz = &(reference->qto_xyz);
    }

    int referenceIndex_start_x;
    int referenceIndex_start_y;
    int referenceIndex_start_z;
    int referenceIndex_end_x;
    int referenceIndex_end_y;
    int referenceIndex_end_z;
    int warpedIndex_start_x;
    int warpedIndex_start_y;
    int warpedIndex_start_z;
    int warpedIndex_end_x;
    int warpedIndex_end_y;
    int warpedIndex_end_z;

    int index, i, j, k, l, m, n, x, y, z;
    int *maskPtr_Z, *maskPtr_XYZ;
    DTYPE *referencePtr_Z, *referencePtr_XYZ, *warpedPtr_Z, *warpedPtr_XYZ;
    DTYPE value, bestCC, referenceMean, warpedMean, referenceVar, warpedVar;
    DTYPE voxelNumber, localCC, referenceTemp, warpedTemp;
    float bestDisplacement[3], referencePosition_temp[3], tempPosition[3];
    size_t referenceIndex, warpedIndex, blockIndex, tid = 0, positionIndex = 0;

    //Let's see OPENMP later...
    DTYPE* referenceValues = reg_matrix1DAllocate<DTYPE>(BLOCK_SIZE);
    DTYPE* warpedValues = reg_matrix1DAllocate<DTYPE>(BLOCK_SIZE);
    bool* referenceOverlap = reg_matrix1DAllocate<bool>(BLOCK_SIZE);
    bool* warpedOverlap = reg_matrix1DAllocate<bool>(BLOCK_SIZE);

    float *temp_reference_position = (float *)malloc(3 * params->activeBlockNumber * sizeof(float));
    float *temp_warped_position = (float *)malloc(3 * params->activeBlockNumber * sizeof(float));
    for (i = 0; i < 3 * params->activeBlockNumber; i += 3) {
        temp_reference_position[i] = std::numeric_limits<float>::quiet_NaN();
    }

    for (k = 0; k < params->blockNumber[2]; k++) {
        blockIndex = k * params->blockNumber[0] * params->blockNumber[1];
        referenceIndex_start_z = k * BLOCK_WIDTH;
        referenceIndex_end_z = referenceIndex_start_z + BLOCK_WIDTH;
        //2D
        if (dim == 2) {
            referenceIndex_end_z = 1;
        }

        for (j = 0; j < params->blockNumber[1]; j++) {
            referenceIndex_start_y = j * BLOCK_WIDTH;
            referenceIndex_end_y = referenceIndex_start_y + BLOCK_WIDTH;

            for (i = 0; i < params->blockNumber[0]; i++) {
                referenceIndex_start_x = i * BLOCK_WIDTH;
                referenceIndex_end_x = referenceIndex_start_x + BLOCK_WIDTH;
                //DEBUG
                //std::cout << "params->activeBlock[blockIndex]=" << params->activeBlock[blockIndex] << std::endl;
                //DEBUG
                if (params->activeBlock[blockIndex] > -1) {
                    referenceIndex = 0;
                    memset(referenceOverlap, 0, BLOCK_SIZE * sizeof(bool));

                    for (z = referenceIndex_start_z; z < referenceIndex_end_z; z++) {
                        if (-1 < z && z < reference->nz) {
                            index = z * reference->nx * reference->ny;
                            //2D-3D - NOT NECESSARY IN THEORY...
                            //if (dim == 2) {
                            //    referencePtr_Z = referencePtr;
                            //    maskPtr_Z = mask;
                            //}
                            //else {
                                referencePtr_Z = &referencePtr[index];
                                maskPtr_Z = &mask[index];
                            //}
                            

                            for (y = referenceIndex_start_y; y < referenceIndex_end_y; y++) {
                                if (-1 < y && y < reference->ny) {
                                    index = y * reference->nx + referenceIndex_start_x;
                                    referencePtr_XYZ = &referencePtr_Z[index];
                                    maskPtr_XYZ = &maskPtr_Z[index];

                                    for (x = referenceIndex_start_x; x < referenceIndex_end_x; x++) {
                                        if (-1 < x && x < reference->nx) {
                                            value = *referencePtr_XYZ;
                                            if (value == value && *maskPtr_XYZ > -1) {
                                                referenceValues[referenceIndex] = value;
                                                referenceOverlap[referenceIndex] = 1;
                                            }
                                        }
                                        referencePtr_XYZ++;
                                        maskPtr_XYZ++;
                                        referenceIndex++;
                                    }
                                }
                                else {
                                    referenceIndex += BLOCK_WIDTH;
                                }
                            }
                        }
                        else {
                            referenceIndex += BLOCK_WIDTH * BLOCK_WIDTH;
                        }
                    }
                    bestCC = params->voxelCaptureRange > 3 ? 0.9 : 0.0; //only when misaligned images are registered
                    bestDisplacement[0] = std::numeric_limits<float>::quiet_NaN();
                    bestDisplacement[1] = 0.f;
                    bestDisplacement[2] = 0.f;

                    //DEBUG
                    if (i == 1 && j==1) {
                        std::cout << "BP" << std::endl;
                    }
                    //DEBUG

                    // iteration over the warped blocks
                    for (n = -1 * params->voxelCaptureRange; n <= params->voxelCaptureRange; n += params->stepSize) {
                        warpedIndex_start_z = referenceIndex_start_z + n;
                        warpedIndex_end_z = warpedIndex_start_z + BLOCK_WIDTH;
                        if (dim == 2) {
                            n = params->voxelCaptureRange + 1;
                            warpedIndex_start_z = 0;
                            warpedIndex_end_z = 1;
                        }

                        for (m = -1 * params->voxelCaptureRange; m <= params->voxelCaptureRange; m += params->stepSize) {
                            warpedIndex_start_y = referenceIndex_start_y + m;
                            warpedIndex_end_y = warpedIndex_start_y + BLOCK_WIDTH;
                            for (l = -1 * params->voxelCaptureRange; l <= params->voxelCaptureRange; l += params->stepSize) {

                                warpedIndex_start_x = referenceIndex_start_x + l;
                                warpedIndex_end_x = warpedIndex_start_x + BLOCK_WIDTH;
                                warpedIndex = 0;
                                memset(warpedOverlap, 0, BLOCK_SIZE * sizeof(bool));

                                for (z = warpedIndex_start_z; z < warpedIndex_end_z; z++) {
                                    if (-1 < z && z < warped->nz) {
                                        index = z * warped->nx * warped->ny;
                                        warpedPtr_Z = &warpedPtr[index];
                                        int *maskPtr_Z = &mask[index];

                                        for (y = warpedIndex_start_y; y < warpedIndex_end_y; y++) {
                                            if (-1 < y && y < warped->ny) {
                                                index = y * warped->nx + warpedIndex_start_x;
                                                warpedPtr_XYZ = &warpedPtr_Z[index];
                                                int *maskPtr_XYZ = &maskPtr_Z[index];
                                                for (x = warpedIndex_start_x; x < warpedIndex_end_x; x++) {
                                                    if (-1 < x && x < warped->nx) {
                                                        value = *warpedPtr_XYZ;
                                                        if (value == value && *maskPtr_XYZ > -1) {
                                                            warpedValues[warpedIndex] = value;
                                                            warpedOverlap[warpedIndex] = 1;
                                                        }
                                                    }
                                                    warpedPtr_XYZ++;
                                                    warpedIndex++;
                                                    maskPtr_XYZ++;
                                                }
                                            }
                                            else {
                                                warpedIndex += BLOCK_WIDTH;
                                            }
                                        }
                                    }
                                    else {
                                        warpedIndex += BLOCK_WIDTH * BLOCK_WIDTH;
                                    }
                                }
                                referenceMean = 0.0;
                                warpedMean = 0.0;
                                voxelNumber = 0.0;
                                //DEBUG
                                if (i == 1 && j == 1) {
                                    std::cout << "BP" << std::endl;
                                }
                                //DEBUG
                                for (int a = 0; a < BLOCK_SIZE; a++) {
                                    if (referenceOverlap[a] && warpedOverlap[a]) {
                                        //DEBUG
                                        if (i == 1 && j == 1) {
                                            std::cout << "referenceValues[a]=" << referenceValues[a] << std::endl;
                                            std::cout << "warpedValues[a]=" << warpedValues[a] << std::endl;
                                        }
                                        //DEBUG
                                        referenceMean += referenceValues[a];
                                        warpedMean += warpedValues[a];
                                        voxelNumber++;
                                    }
                                }

                                if (voxelNumber > BLOCK_SIZE / 2) {
                                    referenceMean /= voxelNumber;
                                    warpedMean /= voxelNumber;

                                    referenceVar = 0.0;
                                    warpedVar = 0.0;
                                    localCC = 0.0;

                                    for (int a = 0; a < BLOCK_SIZE; a++) {
                                        if (referenceOverlap[a] && warpedOverlap[a]) {
                                            referenceTemp = (referenceValues[a] - referenceMean);
                                            warpedTemp = (warpedValues[a] - warpedMean);
                                            referenceVar += (referenceTemp)* (referenceTemp);
                                            warpedVar += (warpedTemp)* (warpedTemp);
                                            localCC += (referenceTemp)* (warpedTemp);
                                        }
                                    }
                                    //To be consistent with the variables name
                                    referenceVar = referenceVar / voxelNumber;
                                    warpedVar = warpedVar / voxelNumber;
                                    localCC = localCC / voxelNumber;

                                    localCC = (referenceVar * warpedVar) > 0.0 ? fabs(localCC / sqrt(referenceVar * warpedVar)) : 0;
                                    if (localCC > bestCC) {
                                        bestCC = localCC;
                                        bestDisplacement[0] = (float)l;
                                        bestDisplacement[1] = (float)m;
                                        bestDisplacement[2] = dim == 2 ? 0 : (float)n;
                                    }
                                }
                            }
                        }
                    }

                    referencePosition_temp[0] = (float)(i * BLOCK_WIDTH);
                    referencePosition_temp[1] = (float)(j * BLOCK_WIDTH);
                    referencePosition_temp[2] = (float)(k * BLOCK_WIDTH);

                    bestDisplacement[0] += referencePosition_temp[0];
                    bestDisplacement[1] += referencePosition_temp[1];
                    bestDisplacement[2] += referencePosition_temp[2];

                    reg_mat44_mul(referenceMatrix_xyz, referencePosition_temp, tempPosition);

                    //DEBUG
                    if (tempPosition[0] == -86 && tempPosition[1] == 20) {
                        std::cout << "BP" << std::endl;
                    }
                    //DEBUG

                    positionIndex = 3 * params->activeBlock[blockIndex];
                    temp_reference_position[positionIndex] = tempPosition[0];
                    temp_reference_position[positionIndex + 1] = tempPosition[1];
                    temp_reference_position[positionIndex + 2] = tempPosition[2];
                    reg_mat44_mul(referenceMatrix_xyz, bestDisplacement, tempPosition);
                    temp_warped_position[positionIndex] = tempPosition[0];
                    temp_warped_position[positionIndex + 1] = tempPosition[1];
                    temp_warped_position[positionIndex + 2] = tempPosition[2];
                }
                else {
                    //It is an unasable block, should never enter here normally
                    //reg_print_msg_warn("It is an unasable block, should never enter here normally");
                }
                blockIndex++;
            }
        }
    }

    // Removing the NaNs and defining the number of active block
    params->definedActiveBlock = 0;
    j = 0;
    for (i = 0; i < 3 * params->activeBlockNumber; i += 3) {
        //I think we should not put the if here because otherwise we loose the correspondance
        //if (temp_reference_position[i] == temp_reference_position[i]) {
        params->referencePosition[j] = temp_reference_position[i];
        params->referencePosition[j + 1] = temp_reference_position[i + 1];
        params->referencePosition[j + 2] = temp_reference_position[i + 2];
        params->warpedPosition[j] = temp_warped_position[i];
        params->warpedPosition[j + 1] = temp_warped_position[i + 1];
        params->warpedPosition[j + 2] = temp_warped_position[i + 2];
        if (temp_reference_position[i] == temp_reference_position[i]) {
            params->definedActiveBlock++;
        }
        j += 3;
        //}
    }
    free(temp_reference_position);
    free(temp_warped_position);
    free(referenceValues);
    free(warpedValues);
    free(referenceOverlap);
    free(warpedOverlap);
}
/* *************************************************************** */
// Block matching interface function
void block_matching_method(nifti_image * reference, nifti_image * warped, _reg_blockMatchingParam *params, int *mask) {
    if (reference->datatype != warped->datatype) {
        reg_print_fct_error("block_matching_method");
        reg_print_msg_error("Both input images are expected to be of the same type");
    }
    //TODO: create a unique function for 2D and 3D !
    //if (reference->nz == 1) {
    //	switch (reference->datatype) {
    //	case NIFTI_TYPE_FLOAT64:
    //		block_matching_method2D<double>(reference, warped, params, mask);
    //		break;
    //	case NIFTI_TYPE_FLOAT32:
    //		block_matching_method2D<float>(reference, warped, params, mask);
    //		break;
    //	default:
    //		reg_print_fct_error("block_matching_method");
    //		reg_print_msg_error("The reference image data type is not supported");
    //		reg_exit(1);
    //	}
    //} else {
    //	switch (reference->datatype) {
    //	case NIFTI_TYPE_FLOAT64:
    //		block_matching_method3D<double>(reference, warped, params, mask);
    //		break;
    //	case NIFTI_TYPE_FLOAT32:
    //		block_matching_method3D<float>(reference, warped, params, mask);
    //		break;
    //	default:
    //		reg_print_fct_error("block_matching_method");
    //		reg_print_msg_error("The reference image data type is not supported");
    //		reg_exit(1);
    //	}
    //}
    int dim = 0;
    if (reference->nz == 1) {
        dim = 2;
    }
    else {
        dim = 3;
    }
    switch (reference->datatype) {
    case NIFTI_TYPE_FLOAT64:
        block_matching_method2D3D<double>(reference, warped, params, mask, dim);
        break;
    case NIFTI_TYPE_FLOAT32:
        block_matching_method2D3D<float>(reference, warped, params, mask, dim);
        break;
    default:
        reg_print_fct_error("block_matching_method");
        reg_print_msg_error("The reference image data type is not supported");
        reg_exit(1);
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
        float in[2];
        float out[2];
        for (size_t i = 0; i < static_cast<size_t>(params->activeBlockNumber); ++i)
        {
            size_t index = 2 * i;
            in[0] = params->warpedPosition[index];
            in[1] = params->warpedPosition[index + 1];
            reg_mat33_mul(transformation_matrix, in, out);
            params->warpedPosition[index] = out[0];
            params->warpedPosition[index + 1] = out[1];
        }
        optimize_2D(params->referencePosition, params->warpedPosition,
            params->definedActiveBlock, params->percent_to_keep,
            MAX_ITERATIONS, TOLERANCE,
            transformation_matrix, affine);
    }
    else  // 3D images
    {
        float in[3];
        float out[3];
        for (size_t i = 0; i < static_cast<size_t>(params->activeBlockNumber); ++i)
        {
            size_t index = 3 * i;
            in[0] = params->warpedPosition[index];
            in[1] = params->warpedPosition[index + 1];
            in[2] = params->warpedPosition[index + 2];
            reg_mat44_mul(transformation_matrix, in, out);
            params->warpedPosition[index] = out[0];
            params->warpedPosition[index + 1] = out[1];
            params->warpedPosition[index + 2] = out[2];
        }
        optimize_3D(params->referencePosition, params->warpedPosition,
            params->definedActiveBlock, params->percent_to_keep,
            MAX_ITERATIONS, TOLERANCE,
            transformation_matrix, affine);
    }
}
