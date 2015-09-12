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
template<class DTYPE>
void _reg_set_active_blocks(nifti_image *targetImage, _reg_blockMatchingParam *params, int *mask, bool runningOnGPU) {
	const size_t totalBlockNumber = params->blockNumber[0] * params->blockNumber[1] * params->blockNumber[2];
	float *varianceArray = (float *) malloc(totalBlockNumber * sizeof(float));
	int *indexArray = (int *) malloc(totalBlockNumber * sizeof(int));

	int *maskPtr = &mask[0];

	int unusableBlock = 0;
	size_t index;

	DTYPE *targetValues = (DTYPE *) malloc(BLOCK_SIZE * sizeof(DTYPE));
	DTYPE *targetPtr = static_cast<DTYPE *>(targetImage->data);
	int blockIndex = 0;

	if (targetImage->nz > 1) {
		// Version using 3D blocks
		for (int k = 0; k < params->blockNumber[2]; k++) {
			for (int j = 0; j < params->blockNumber[1]; j++) {
				for (int i = 0; i < params->blockNumber[0]; i++) {
					for (unsigned int n = 0; n < BLOCK_SIZE; n++)
						targetValues[n] = (DTYPE) std::numeric_limits<float>::quiet_NaN();
					float mean = 0.0f;
					float voxelNumber = 0.0f;
					int coord = 0;
					for (int z = k * BLOCK_WIDTH; z < (k + 1) * BLOCK_WIDTH; z++) {
						if (z < targetImage->nz) {
							index = z * targetImage->nx * targetImage->ny;
							DTYPE *targetPtrZ = &targetPtr[index];
							int *maskPtrZ = &maskPtr[index];
							for (int y = j * BLOCK_WIDTH; y < (j + 1) * BLOCK_WIDTH; y++) {
								if (y < targetImage->ny) {
									index = y * targetImage->nx + i * BLOCK_WIDTH;
									DTYPE *targetPtrXYZ = &targetPtrZ[index];
									int *maskPtrXYZ = &maskPtrZ[index];
									for (int x = i * BLOCK_WIDTH; x < (i + 1) * BLOCK_WIDTH; x++) {
										if (x < targetImage->nx) {
											targetValues[coord] = *targetPtrXYZ;
											if (targetValues[coord] == targetValues[coord] && targetValues[coord] != 0. && *maskPtrXYZ > -1) {
												mean += (float) targetValues[coord];
												voxelNumber++;
											}
										}
										targetPtrXYZ++;
										maskPtrXYZ++;
										coord++;
									}
								}
							}
						}
					}
					if (voxelNumber > BLOCK_SIZE / 2) {
						float variance = 0.0f;
						for (int i = 0; i < BLOCK_SIZE; i++) {
							if (targetValues[i] == targetValues[i])
								variance += (mean - (float) targetValues[i]) * (mean - (float) targetValues[i]);
						}

						variance /= voxelNumber;
						varianceArray[blockIndex] = variance;
					} else {
						varianceArray[blockIndex] = -1;
						unusableBlock++;
					}
					indexArray[blockIndex] = blockIndex;
					blockIndex++;
				}
			}
		}
	} else {
		// Version using 2D blocks
		for (int j = 0; j < params->blockNumber[1]; j++) {
			for (int i = 0; i < params->blockNumber[0]; i++) {

				for (unsigned int n = 0; n < BLOCK_2D_SIZE; n++)
					targetValues[n] = (DTYPE) std::numeric_limits<float>::quiet_NaN();
				float mean = 0.0f;
				float voxelNumber = 0.0f;
				int coord = 0;

				for (int y = j * BLOCK_WIDTH; y < (j + 1) * BLOCK_WIDTH; y++) {
					if (y < targetImage->ny) {
						index = y * targetImage->nx + i * BLOCK_WIDTH;
						DTYPE *targetPtrXY = &targetPtr[index];
						int *maskPtrXY = &maskPtr[index];
						for (int x = i * BLOCK_WIDTH; x < (i + 1) * BLOCK_WIDTH; x++) {
							if (x < targetImage->nx) {
								targetValues[coord] = *targetPtrXY;
								if (targetValues[coord] == targetValues[coord] && targetValues[coord] != 0. && *maskPtrXY > -1) {
									mean += (float) targetValues[coord];
									voxelNumber++;
								}
							}
							targetPtrXY++;
							maskPtrXY++;
							coord++;
						}
					}
				}
				if (voxelNumber > BLOCK_2D_SIZE / 2) {
					float variance = 0.0f;
					for (int i = 0; i < BLOCK_2D_SIZE; i++) {
						if (targetValues[i] == targetValues[i])
							variance += (mean - (float) targetValues[i]) * (mean - (float) targetValues[i]);
					}

					variance /= voxelNumber;
					varianceArray[blockIndex] = variance;
				} else {
					varianceArray[blockIndex] = -1;
					unusableBlock++;
				}
				indexArray[blockIndex] = blockIndex;
				blockIndex++;
			}
		}
	}
	free(targetValues);

	params->activeBlockNumber = params->activeBlockNumber < ((int) totalBlockNumber - unusableBlock) ? params->activeBlockNumber : (totalBlockNumber - unusableBlock);

	reg_heapSort(varianceArray, indexArray, totalBlockNumber);

	int *indexArrayPtr = &indexArray[totalBlockNumber - 1];
	int count = 0;
	for (int i = 0; i < params->activeBlockNumber; i++) {
		params->activeBlock[*indexArrayPtr--] = count++;
	}
	for (size_t i = params->activeBlockNumber; i < totalBlockNumber; ++i) {
		params->activeBlock[*indexArrayPtr--] = -1;
	}

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
void initialise_block_matching_method(nifti_image * target, _reg_blockMatchingParam *params, int percentToKeep_block, int percentToKeep_opt, int stepSize_block, int *mask, bool runningOnGPU) {
	if (params->activeBlock != NULL) {
		free(params->activeBlock);
		params->activeBlock = NULL;
	}
	if (params->targetPosition != NULL) {
		free(params->targetPosition);
		params->targetPosition = NULL;
	}
	if (params->resultPosition != NULL) {
		free(params->resultPosition);
		params->resultPosition = NULL;
	}

	params->voxelCaptureRange = 3;
	params->blockNumber[0] = (int) reg_ceil((float) target->nx / (float) BLOCK_WIDTH);
	params->blockNumber[1] = (int) reg_ceil((float) target->ny / (float) BLOCK_WIDTH);
	if (target->nz > 1)
		params->blockNumber[2] = (int) reg_ceil((float) target->nz / (float) BLOCK_WIDTH);
	else
		params->blockNumber[2] = 1;

	params->stepSize = stepSize_block;

	params->percent_to_keep = percentToKeep_opt;
	params->activeBlockNumber = params->blockNumber[0] * params->blockNumber[1] * params->blockNumber[2] * percentToKeep_block / 100;

	params->activeBlock = (int *) malloc(params->blockNumber[0] * params->blockNumber[1] * params->blockNumber[2] * sizeof(int));
	switch (target->datatype) {
	case NIFTI_TYPE_FLOAT32:
		_reg_set_active_blocks<float>(target, params, mask, runningOnGPU);
		break;
	case NIFTI_TYPE_FLOAT64:
		_reg_set_active_blocks<double>(target, params, mask, runningOnGPU);
		break;
	default:
		reg_print_fct_error("initialise_block_matching_method()");
		reg_print_msg_error("The target image data type is not supported");
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
	sprintf(text,"There are %i active block(s) out of %i.",
			  params->activeBlockNumber, params->blockNumber[0] * params->blockNumber[1] * params->blockNumber[2]);
	reg_print_msg_debug(text)
#endif
	if (target->nz > 1) {
		params->targetPosition = (float *) malloc(params->activeBlockNumber * 3 * sizeof(float));
		params->resultPosition = (float *) malloc(params->activeBlockNumber * 3 * sizeof(float));
	} else {
		params->targetPosition = (float *) malloc(params->activeBlockNumber * 2 * sizeof(float));
		params->resultPosition = (float *) malloc(params->activeBlockNumber * 2 * sizeof(float));
	}
#ifndef NDEBUG
	reg_print_msg_debug("block matching initialisation done.");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template<typename PrecisionTYPE, typename TargetImageType, typename ResultImageType>
void block_matching_method2D(nifti_image * target, nifti_image * result, _reg_blockMatchingParam *params, int *mask) {
	TargetImageType *targetPtr = static_cast<TargetImageType *>(target->data);
	ResultImageType *resultPtr = static_cast<ResultImageType *>(result->data);

	TargetImageType *targetValues = (TargetImageType *) malloc(BLOCK_2D_SIZE * sizeof(TargetImageType));
	bool *targetOverlap = (bool *) malloc(BLOCK_2D_SIZE * sizeof(bool));
	ResultImageType *resultValues = (ResultImageType *) malloc(BLOCK_2D_SIZE * sizeof(ResultImageType));
	bool *resultOverlap = (bool *) malloc(BLOCK_2D_SIZE * sizeof(bool));

	mat44 *targetMatrix_xyz;
	if (target->sform_code > 0)
		targetMatrix_xyz = &(target->sto_xyz);
	else
		targetMatrix_xyz = &(target->qto_xyz);

	int targetIndex_start_x;
	int targetIndex_start_y;
	int targetIndex_end_x;
	int targetIndex_end_y;
	int resultIndex_start_x;
	int resultIndex_start_y;
	int resultIndex_end_x;
	int resultIndex_end_y;

	unsigned int targetIndex;
	unsigned int resultIndex;

	unsigned int blockIndex = 0;
	unsigned int activeBlockIndex = 0;
	params->definedActiveBlock = 0;
	int index;

	for (int j = 0; j < params->blockNumber[1]; j++) {
		targetIndex_start_y = j * BLOCK_WIDTH;
		targetIndex_end_y = targetIndex_start_y + BLOCK_WIDTH;

		for (int i = 0; i < params->blockNumber[0]; i++) {
			targetIndex_start_x = i * BLOCK_WIDTH;
			targetIndex_end_x = targetIndex_start_x + BLOCK_WIDTH;

			if (params->activeBlock[blockIndex] > -1) {

				targetIndex = 0;
				memset(targetOverlap, 0, BLOCK_2D_SIZE * sizeof(bool));

				for (int y = targetIndex_start_y; y < targetIndex_end_y; y++) {
					if (-1 < y && y < target->ny) {
						index = y * target->nx + targetIndex_start_x;
						TargetImageType *targetPtr_XY = &targetPtr[index];
						int *maskPtr_XY = &mask[index];
						for (int x = targetIndex_start_x; x < targetIndex_end_x; x++) {
							if (-1 < x && x < target->nx) {
								TargetImageType value = *targetPtr_XY;
								if (value == value && value != 0. && *maskPtr_XY > -1) {
									targetValues[targetIndex] = value;
									targetOverlap[targetIndex] = 1;
								}
							}
							targetPtr_XY++;
							maskPtr_XY++;
							targetIndex++;
						}
					} else
						targetIndex += BLOCK_WIDTH;
				}
				PrecisionTYPE bestCC = params->voxelCaptureRange > 3 ? 0.9 : 0.0; //only when misaligned images are registered
				float bestDisplacement[3] = { std::numeric_limits<float>::quiet_NaN(), 0.f, 0.f };

				// iteration over the result blocks
				for (int m = -1 * params->voxelCaptureRange; m <= params->voxelCaptureRange; m += params->stepSize) {
					resultIndex_start_y = targetIndex_start_y + m;
					resultIndex_end_y = resultIndex_start_y + BLOCK_WIDTH;
					for (int l = -1 * params->voxelCaptureRange; l <= params->voxelCaptureRange; l += params->stepSize) {
						resultIndex_start_x = targetIndex_start_x + l;
						resultIndex_end_x = resultIndex_start_x + BLOCK_WIDTH;

						resultIndex = 0;
						memset(resultOverlap, 0, BLOCK_2D_SIZE * sizeof(bool));

						for (int y = resultIndex_start_y; y < resultIndex_end_y; y++) {
							if (-1 < y && y < result->ny) {
								index = y * result->nx + resultIndex_start_x;
								ResultImageType *resultPtr_XY = &resultPtr[index];
								int *maskPtr_XY = &mask[index];
								for (int x = resultIndex_start_x; x < resultIndex_end_x; x++) {
									if (-1 < x && x < result->nx) {
										ResultImageType value = *resultPtr_XY;
										if (value == value && value != 0. && *maskPtr_XY > -1) {
											resultValues[resultIndex] = value;
											resultOverlap[resultIndex] = 1;
										}
									}
									resultPtr_XY++;
									resultIndex++;
									maskPtr_XY++;
								}
							} else
								resultIndex += BLOCK_WIDTH;
						}
						PrecisionTYPE targetMean = 0.0;
						PrecisionTYPE resultMean = 0.0;
						PrecisionTYPE voxelNumber = 0.0;
						for (int a = 0; a < BLOCK_2D_SIZE; a++) {
							if (targetOverlap[a] && resultOverlap[a]) {
								targetMean += (PrecisionTYPE) targetValues[a];
								resultMean += (PrecisionTYPE) resultValues[a];
								voxelNumber++;
							}
						}

						if (voxelNumber > BLOCK_2D_SIZE / 2) {
							targetMean /= voxelNumber;
							resultMean /= voxelNumber;

							PrecisionTYPE targetVar = 0.0;
							PrecisionTYPE resultVar = 0.0;
							PrecisionTYPE localCC = 0.0;

							for (int a = 0; a < BLOCK_2D_SIZE; a++) {
								if (targetOverlap[a] && resultOverlap[a]) {
									PrecisionTYPE targetTemp = (PrecisionTYPE) (targetValues[a] - targetMean);
									PrecisionTYPE resultTemp = (PrecisionTYPE) (resultValues[a] - resultMean);
									targetVar += (targetTemp) * (targetTemp);
									resultVar += (resultTemp) * (resultTemp);
									localCC += (targetTemp) * (resultTemp);
								}
							}

							localCC = (targetVar * resultVar) > 0.0 ? fabs(localCC / sqrt(targetVar * resultVar)) : 0;

							if (localCC > bestCC) {
								bestCC = localCC;
								bestDisplacement[0] = (float) l;
								bestDisplacement[1] = (float) m;
							}
						}
					}
				}

				if (std::isfinite(bestDisplacement[0])) {
					float targetPosition_temp[3];
					targetPosition_temp[0] = (float) (i * BLOCK_WIDTH);
					targetPosition_temp[1] = (float) (j * BLOCK_WIDTH);
					targetPosition_temp[2] = 0.0f;

					bestDisplacement[0] += targetPosition_temp[0];
					bestDisplacement[1] += targetPosition_temp[1];
					bestDisplacement[2] = 0.0f;

					float tempPosition[3];
					reg_mat44_mul(targetMatrix_xyz, targetPosition_temp, tempPosition);
					params->targetPosition[activeBlockIndex] = tempPosition[0];
					params->targetPosition[activeBlockIndex + 1] = tempPosition[1];
					reg_mat44_mul(targetMatrix_xyz, bestDisplacement, tempPosition);
					params->resultPosition[activeBlockIndex] = tempPosition[0];
					params->resultPosition[activeBlockIndex + 1] = tempPosition[1];
					activeBlockIndex += 2;
					params->definedActiveBlock++;
				}
			}
			blockIndex++;
		}
	}
	free(resultValues);
	free(targetValues);
	free(targetOverlap);
	free(resultOverlap);
}
/* *************************************************************** */
template<typename DTYPE>
void block_matching_method3D(nifti_image * target, nifti_image * result, _reg_blockMatchingParam *params, int *mask) {
	DTYPE *targetPtr = static_cast<DTYPE *>(target->data);
	DTYPE *resultPtr = static_cast<DTYPE *>(result->data);

	mat44 *targetMatrix_xyz;
	if (target->sform_code > 0)
		targetMatrix_xyz = &(target->sto_xyz);
	else
		targetMatrix_xyz = &(target->qto_xyz);

	int targetIndex_start_x;
	int targetIndex_start_y;
	int targetIndex_start_z;
	int targetIndex_end_x;
	int targetIndex_end_y;
	int targetIndex_end_z;
	int resultIndex_start_x;
	int resultIndex_start_y;
	int resultIndex_start_z;
	int resultIndex_end_x;
	int resultIndex_end_y;
	int resultIndex_end_z;

	int index, i, j, k, l, m, n, x, y, z;
	int *maskPtr_Z, *maskPtr_XYZ;
	DTYPE *targetPtr_Z, *targetPtr_XYZ, *resultPtr_Z, *resultPtr_XYZ;
	DTYPE value, bestCC, targetMean, resultMean, targetVar, resultVar;
	DTYPE voxelNumber, localCC, targetTemp, resultTemp;
	float bestDisplacement[3], targetPosition_temp[3], tempPosition[3];
	size_t targetIndex, resultIndex, blockIndex, tid = 0;
	params->definedActiveBlock = 0;
#if defined (_OPENMP)
	int threadNumber = omp_get_max_threads();
	if(threadNumber>16)
	omp_set_num_threads(16);
	DTYPE targetValues[16][BLOCK_SIZE];
	DTYPE resultValues[16][BLOCK_SIZE];
	bool targetOverlap[16][BLOCK_SIZE];
	bool resultOverlap[16][BLOCK_SIZE];
#else
	DTYPE targetValues[1][BLOCK_SIZE];
	DTYPE resultValues[1][BLOCK_SIZE];
	bool targetOverlap[1][BLOCK_SIZE];
	bool resultOverlap[1][BLOCK_SIZE];
#endif

	float *temp_target_position = (float *) malloc(3 * params->activeBlockNumber * sizeof(float));
	float *temp_result_position = (float *) malloc(3 * params->activeBlockNumber * sizeof(float));
	for (i = 0; i < 3 * params->activeBlockNumber; i += 3)
		temp_target_position[i] = std::numeric_limits<float>::quiet_NaN();

#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(params, target, result, targetPtr, resultPtr, mask, targetMatrix_xyz, \
          targetOverlap, resultOverlap, targetValues, resultValues, \
          temp_target_position, temp_result_position) \
   private(i, j, k, l, m, n, x, y, z, blockIndex, targetIndex, \
           index, tid, targetPtr_Z, targetPtr_XYZ, resultPtr_Z, resultPtr_XYZ, \
           maskPtr_Z, maskPtr_XYZ, value, bestCC, bestDisplacement, \
           targetIndex_start_x, targetIndex_start_y, targetIndex_start_z, \
           targetIndex_end_x, targetIndex_end_y, targetIndex_end_z, \
           resultIndex_start_x, resultIndex_start_y, resultIndex_start_z, \
           resultIndex_end_x, resultIndex_end_y, resultIndex_end_z, \
           resultIndex, targetPosition_temp, tempPosition, targetTemp, resultTemp, \
           targetMean, targetVar, resultMean, resultVar, voxelNumber,localCC)
#endif
   for (k = 0; k < params->blockNumber[2]; k++) {
#if defined (_OPENMP)
      tid = omp_get_thread_num();
#endif
		blockIndex = k * params->blockNumber[0] * params->blockNumber[1];
		targetIndex_start_z = k * BLOCK_WIDTH;
		targetIndex_end_z = targetIndex_start_z + BLOCK_WIDTH;

		for (j = 0; j < params->blockNumber[1]; j++) {
			targetIndex_start_y = j * BLOCK_WIDTH;
			targetIndex_end_y = targetIndex_start_y + BLOCK_WIDTH;

			for (i = 0; i < params->blockNumber[0]; i++) {
				targetIndex_start_x = i * BLOCK_WIDTH;
				targetIndex_end_x = targetIndex_start_x + BLOCK_WIDTH;

				if (params->activeBlock[blockIndex] > -1) {
					targetIndex = 0;
					memset(targetOverlap[tid], 0, BLOCK_SIZE * sizeof(bool));
					for (z = targetIndex_start_z; z < targetIndex_end_z; z++) {
						if (-1 < z && z < target->nz) {
							index = z * target->nx * target->ny;
							targetPtr_Z = &targetPtr[index];
							maskPtr_Z = &mask[index];
							for (y = targetIndex_start_y; y < targetIndex_end_y; y++) {
								if (-1 < y && y < target->ny) {
									index = y * target->nx + targetIndex_start_x;
									targetPtr_XYZ = &targetPtr_Z[index];
									maskPtr_XYZ = &maskPtr_Z[index];
									for (x = targetIndex_start_x; x < targetIndex_end_x; x++) {
										if (-1 < x && x < target->nx) {
											value = *targetPtr_XYZ;
											if (value == value && *maskPtr_XYZ > -1) {
												targetValues[tid][targetIndex] = value;
												targetOverlap[tid][targetIndex] = 1;
											}
										}
										targetPtr_XYZ++;
										maskPtr_XYZ++;
										targetIndex++;
									}
								} else
									targetIndex += BLOCK_WIDTH;
							}
						} else
							targetIndex += BLOCK_WIDTH * BLOCK_WIDTH;
					}
					bestCC = params->voxelCaptureRange > 3 ? 0.9 : 0.0; //only when misaligned images are registered
					bestDisplacement[0] = std::numeric_limits<float>::quiet_NaN();
					bestDisplacement[1] = 0.f;
					bestDisplacement[2] = 0.f;

					// iteration over the result blocks
					for (n = -1 * params->voxelCaptureRange; n <= params->voxelCaptureRange; n += params->stepSize) {
						resultIndex_start_z = targetIndex_start_z + n;
						resultIndex_end_z = resultIndex_start_z + BLOCK_WIDTH;
						for (m = -1 * params->voxelCaptureRange; m <= params->voxelCaptureRange; m += params->stepSize) {
							resultIndex_start_y = targetIndex_start_y + m;
							resultIndex_end_y = resultIndex_start_y + BLOCK_WIDTH;
							for (l = -1 * params->voxelCaptureRange; l <= params->voxelCaptureRange; l += params->stepSize) {

								resultIndex_start_x = targetIndex_start_x + l;
								resultIndex_end_x = resultIndex_start_x + BLOCK_WIDTH;
								resultIndex = 0;
								memset(resultOverlap[tid], 0, BLOCK_SIZE * sizeof(bool));
								for (z = resultIndex_start_z; z < resultIndex_end_z; z++) {
									if (-1 < z && z < result->nz) {
										index = z * result->nx * result->ny;
										resultPtr_Z = &resultPtr[index];
										int *maskPtr_Z = &mask[index];
										for (y = resultIndex_start_y; y < resultIndex_end_y; y++) {
											if (-1 < y && y < result->ny) {
												index = y * result->nx + resultIndex_start_x;
												resultPtr_XYZ = &resultPtr_Z[index];
												int *maskPtr_XYZ = &maskPtr_Z[index];
												for (x = resultIndex_start_x; x < resultIndex_end_x; x++) {
													if (-1 < x && x < result->nx) {
														value = *resultPtr_XYZ;
														if (value == value && *maskPtr_XYZ > -1) {
															resultValues[tid][resultIndex] = value;
															resultOverlap[tid][resultIndex] = 1;
														}
													}
													resultPtr_XYZ++;
													resultIndex++;
													maskPtr_XYZ++;
												}
											} else
												resultIndex += BLOCK_WIDTH;
										}
									} else
										resultIndex += BLOCK_WIDTH * BLOCK_WIDTH;
								}
								targetMean = 0.0;
								resultMean = 0.0;
								voxelNumber = 0.0;
								for (int a = 0; a < BLOCK_SIZE; a++) {
									if (targetOverlap[tid][a] && resultOverlap[tid][a]) {
										targetMean += targetValues[tid][a];
										resultMean += resultValues[tid][a];
										voxelNumber++;
									}
								}

								if (voxelNumber > BLOCK_SIZE / 2) {
									targetMean /= voxelNumber;
									resultMean /= voxelNumber;

									targetVar = 0.0;
									resultVar = 0.0;
									localCC = 0.0;

									for (int a = 0; a < BLOCK_SIZE; a++) {
										if (targetOverlap[tid][a] && resultOverlap[tid][a]) {
											targetTemp = (targetValues[tid][a] - targetMean);
											resultTemp = (resultValues[tid][a] - resultMean);
											targetVar += (targetTemp) * (targetTemp);
											resultVar += (resultTemp) * (resultTemp);
											localCC += (targetTemp) * (resultTemp);
										}
									}

									localCC = fabs(localCC / sqrt(targetVar * resultVar));
									/*bool predicate = i * BLOCK_WIDTH == 16 && j * BLOCK_WIDTH == 24 && k * BLOCK_WIDTH == 24;
									 if (predicate && 0.981295 - localCC < 0.04 && fabs(0.981295 - localCC) >= 0)
									 printf("C|%d-%d-%d|%.0f|TMN:%f|TVR:%f|RMN:%f|RVR:%f|LCC:%lf|BCC:%lf\n", l, m, n, voxelNumber, targetMean, targetVar, resultMean, resultVar, localCC, bestCC);
									 //*/

									//hack for Marc's integration tests
//									if (localCC > bestCC || (fabs(localCC - 0.981295)<0.000001 && fabs(bestCC-0.981295)<0.000001)) {
									if (localCC > bestCC) {
										bestCC = localCC;
										bestDisplacement[0] = (float) l;
										bestDisplacement[1] = (float) m;
										bestDisplacement[2] = (float) n;
									}
									/*bool predicate = i * BLOCK_WIDTH == 16 && j * BLOCK_WIDTH == 24 && k * BLOCK_WIDTH == 24;
									 if (predicate )
									 printf("C|%d-%d-%d|%f-%f-%f\n", l, m, n, bestDisplacement[0], bestDisplacement[1], bestDisplacement[2]);*/

								}
							}
						}
					}
					if (bestDisplacement[0] == bestDisplacement[0]) {
						targetPosition_temp[0] = (float) (i * BLOCK_WIDTH);
						targetPosition_temp[1] = (float) (j * BLOCK_WIDTH);
						targetPosition_temp[2] = (float) (k * BLOCK_WIDTH);

						bestDisplacement[0] += targetPosition_temp[0];
						bestDisplacement[1] += targetPosition_temp[1];
						bestDisplacement[2] += targetPosition_temp[2];

						reg_mat44_mul(targetMatrix_xyz, targetPosition_temp, tempPosition);
						z = 3 * params->activeBlock[blockIndex];
						temp_target_position[z] = tempPosition[0];
						temp_target_position[z + 1] = tempPosition[1];
						temp_target_position[z + 2] = tempPosition[2];
						reg_mat44_mul(targetMatrix_xyz, bestDisplacement, tempPosition);
						temp_result_position[z] = tempPosition[0];
						temp_result_position[z + 1] = tempPosition[1];
						temp_result_position[z + 2] = tempPosition[2];
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
		if (temp_target_position[i] == temp_target_position[i]) {
			params->targetPosition[j] = temp_target_position[i];
			params->targetPosition[j + 1] = temp_target_position[i + 1];
			params->targetPosition[j + 2] = temp_target_position[i + 2];
			params->resultPosition[j] = temp_result_position[i];
			params->resultPosition[j + 1] = temp_result_position[i + 1];
			params->resultPosition[j + 2] = temp_result_position[i + 2];
			params->definedActiveBlock++;
			j += 3;
		}
	}
	free(temp_target_position);
	free(temp_result_position);

#if defined (_OPENMP)
	omp_set_num_threads(threadNumber);
#endif
}
/* *************************************************************** */
// Block matching interface function
void block_matching_method(nifti_image * target, nifti_image * result, _reg_blockMatchingParam *params, int *mask) {
	if (target->datatype != result->datatype) {
		reg_print_fct_error("block_matching_method");
		reg_print_msg_error("Both input images are expected to be of the same type");
	}
	if (target->nz == 1) {
		switch (target->datatype) {
		case NIFTI_TYPE_FLOAT64:
			block_matching_method2D<double, double, double>(target, result, params, mask);
			break;
		case NIFTI_TYPE_FLOAT32:
			block_matching_method2D<float, float, float>(target, result, params, mask);
			break;
		default:
			reg_print_fct_error("block_matching_method")
			;
			reg_print_msg_error("The target image data type is not supported")
			;
			reg_exit(1)
			;
		}
	} else {
		switch (target->datatype) {
		case NIFTI_TYPE_FLOAT64:
			block_matching_method3D<double>(target, result, params, mask);
			break;
		case NIFTI_TYPE_FLOAT32:
			block_matching_method3D<float>(target, result, params, mask);
			break;
		default:
			reg_print_fct_error("block_matching_method")
			;
			reg_print_msg_error("The target image data type is not supported")
			;
			reg_exit(1)
			;
		}
	}
}
/* *************************************************************** */
// Find the optimal transformation - affine or rigid
void optimize(	_reg_blockMatchingParam *params,
               mat44 *transformation_matrix,
               bool affine)
{
   // The block matching provide correspondences in millimeters
   // in the space of the reference image. All warped image coordinates
   // are updated to be in the original floating space
//    mat44 inverseMatrix = nifti_mat44_inverse(*transformation_matrix);
   if(params->blockNumber[2]==1)  // 2D images
   {
      float in[2];
      float out[2];
      for(size_t i=0; i<static_cast<size_t>(params->activeBlockNumber); ++i)
      {
         size_t index=2*i;
         in[0]=params->resultPosition[index];
         in[1]=params->resultPosition[index+1];
         reg_mat33_mul(transformation_matrix,in,out);
         params->resultPosition[ index ]=out[0];
         params->resultPosition[index+1]=out[1];
      }
      optimize_2D(params->targetPosition, params->resultPosition,
                  params->definedActiveBlock, params->percent_to_keep,
                  MAX_ITERATIONS, TOLERANCE,
                  transformation_matrix, affine);
   }
   else  // 3D images
   {
      float in[3];
      float out[3];
      for(size_t i=0; i<static_cast<size_t>(params->activeBlockNumber); ++i)
      {
         size_t index=3*i;
         in[0]=params->resultPosition[index];
         in[1]=params->resultPosition[index+1];
         in[2]=params->resultPosition[index+2];
         reg_mat44_mul(transformation_matrix,in,out);
         params->resultPosition[ index ]=out[0];
         params->resultPosition[index+1]=out[1];
         params->resultPosition[index+2]=out[2];
      }
      optimize_3D(params->targetPosition, params->resultPosition,
                  params->definedActiveBlock, params->percent_to_keep,
                  MAX_ITERATIONS, TOLERANCE,
                  transformation_matrix, affine);
   }
}
