
//for now no templates opencl 2. supports them!
__inline void interpolantCubicSpline(float ratio, float *basis) {
	if (ratio < 0.0f) ratio = 0.0f; //reg_rounding error
	float FF = ratio*ratio;
	basis[0] = ((ratio * ((2.0f - ratio)*ratio - 1.0f)) / 2.0f);
	basis[1] = ((FF * (3.0f*ratio - 5.0) + 2.0f) / 2.0f);
	basis[2] = ((ratio * ((4.0f - 3.0f*ratio)*ratio + 1.0f)) / 2.0f);
	basis[3] = ((ratio - 1.0f) * FF / 2.0f);
}

 __inline int cl_reg_floor(float a) {
	return a > 0 ? (int)a : (int)(a - 1);
}

__inline float  reg_round(float a) {
return a>0.0 ?(int)(a+0.5):(int)(a-0.5);}

__inline void reg_mat44_mul_cl(__global float const* mat, float const* in, float *out) {
	out[0] = mat[0 * 4 + 0] * in[0] +
		mat[0 * 4 + 1] * in[1] +
		mat[0 * 4 + 2] * in[2] +
		mat[0 * 4 + 3];
	out[1] = mat[1 * 4 + 0] * in[0] +
		mat[1 * 4 + 1] * in[1] +
		mat[1 * 4 + 2] * in[2] +
		mat[1 * 4 + 3];
	out[2] = mat[2 * 4 + 0] * in[0] +
		mat[2 * 4 + 1] * in[1] +
		mat[2 * 4 + 2] * in[2] +
		mat[2 * 4 + 3];
	return;
}

__kernel void  CubicSplineResampleImage3D(__global float* floatingImage, __global float* deformationField, __global float* warpedImage,__global int* mask,__global /*mat44*/float* sourceIJKMatrix, long2 voxelNumber, uint3 fi_xyz, uint2 wi_tu, float paddingValue) {
	//long resultVoxelNumber = (long)warpedImage->nx*warpedImage->ny*warpedImage->nz;vn.x
	//long sourceVoxelNumber = (long)floatingImage->nx*floatingImage->ny*floatingImage->nz;vn.y

	__global float *sourceIntensityPtr = (floatingImage);
	__global float *resultIntensityPtr = (warpedImage);
	__global float *deformationFieldPtrX = (deformationField);
	__global float *deformationFieldPtrY = &deformationFieldPtrX[voxelNumber.x];
	__global float *deformationFieldPtrZ = &deformationFieldPtrY[voxelNumber.x];

	__global int *maskPtr = &mask[0];
	long index = get_group_id(0)*get_local_size(0) + get_local_id(0);
	while (index < voxelNumber.x) {


		// Iteration over the different volume along the 4th axis
		for (unsigned int t = 0; t < wi_tu.x*wi_tu.y; t++) {


			__global float *resultIntensity = &resultIntensityPtr[t*voxelNumber.x];
			__global float *sourceIntensity = &sourceIntensityPtr[t*voxelNumber.y];

			float xBasis[4], yBasis[4], zBasis[4], relative;
			int a, b, c, Y, Z, previous[3];

			__global float *zPointer, *yzPointer, *xyzPointer;
			float xTempNewValue, yTempNewValue, intensity, world[3], position[3];



			intensity = (0.0f);

			if ((maskPtr[index]) > -1) {
				world[0] = deformationFieldPtrX[index];
				world[1] = deformationFieldPtrY[index];
				world[2] = deformationFieldPtrZ[index];

				/* real -> voxel; source space */
				reg_mat44_mul_cl(sourceIJKMatrix, world, position);

				previous[0] = (cl_reg_floor(position[0]));
				previous[1] = (cl_reg_floor(position[1]));
				previous[2] = (cl_reg_floor(position[2]));

				// basis values along the x axis
				relative = position[0] - previous[0];
				relative = relative > 0 ? relative : 0;
				interpolantCubicSpline(relative, xBasis);
				// basis values along the y axis
				relative = position[1] - previous[1];
				relative = relative > 0 ? relative : 0;
				interpolantCubicSpline(relative, yBasis);
				// basis values along the z axis
				relative = position[2] - previous[2];
				relative = relative > 0 ? relative : 0;
				interpolantCubicSpline(relative, zBasis);

				--previous[0];
				--previous[1];
				--previous[2];

				for (c = 0; c < 4; c++) {
					Z = previous[2] + c;
					zPointer = &sourceIntensity[Z*fi_xyz.x*fi_xyz.y];
					yTempNewValue = 0.0;
					for (b = 0; b < 4; b++) {
						Y = previous[1] + b;
						yzPointer = &zPointer[Y*fi_xyz.x];
						xyzPointer = &yzPointer[previous[0]];
						xTempNewValue = 0.0;
						for (a = 0; a < 4; a++) {
							if (-1 < (previous[0] + a) && (previous[0] + a) < fi_xyz.x &&
								-1 < Z && Z < fi_xyz.z &&
								-1 < Y && Y < fi_xyz.y) {
								xTempNewValue += *xyzPointer * xBasis[a];
							}
							else {
								// paddingValue
								xTempNewValue += paddingValue * xBasis[a];
							}
							xyzPointer++;
						}
						yTempNewValue += xTempNewValue * yBasis[b];
					}
					intensity += yTempNewValue * zBasis[c];
				}
			}

			resultIntensity[index] = intensity;
		}
	index += get_num_groups(0)*get_local_size(0);
	}
}

__kernel void TrilinearResampleImage(__global float* floatingImage, __global float* deformationField, __global float* warpedImage, __global int* mask, __global /*mat44*/float* sourceIJKMatrix, long2 voxelNumber, uint3 fi_xyz, uint2 wi_tu, float paddingValue) {

	//if( get_local_id(0) == 0 ) printf("block: %zu \n", get_group_id(0));

	//targetVoxelNumber voxelNumber.x
	// sourceVoxelNumber voxelNumber.y

	//intensity images
	__global float *sourceIntensityPtr = (floatingImage);//best to be a texture
	__global float *resultIntensityPtr = (warpedImage);

	//deformation field image
	__global float *deformationFieldPtrX = (deformationField);
	__global float *deformationFieldPtrY = &deformationFieldPtrX[voxelNumber.x];
	__global float *deformationFieldPtrZ = &deformationFieldPtrY[voxelNumber.x];

	__global int *maskPtr = &mask[0];

	// The resampling scheme is applied along each time

	__private long index = get_group_id(0)*get_local_size(0) + get_local_id(0);
	while (index < voxelNumber.x) {
		for (unsigned int t = 0; t<wi_tu.x*wi_tu.y; t++) {


			__global float *resultIntensity = &resultIntensityPtr[t*voxelNumber.x];
			__global float *sourceIntensity = &sourceIntensityPtr[t*voxelNumber.y];

			float xBasis[2], yBasis[2], zBasis[2], relative;
			int a, b, c, X, Y, Z, previous[3];

			__global float *zPointer, *xyzPointer;
			float xTempNewValue, yTempNewValue, intensity, world[3], position[3];

			//for( index = 0; index<targetVoxelNumber; index++ ) {

			intensity = paddingValue;

			if (maskPtr[index]>-1) {

				intensity = 0;

				world[0] = deformationFieldPtrX[index];
				world[1] = deformationFieldPtrY[index];
				world[2] = deformationFieldPtrZ[index];

				/* real -> voxel; source space */
				reg_mat44_mul_cl(sourceIJKMatrix, world, position);

				previous[0] = cl_reg_floor(position[0]);
				previous[1] = cl_reg_floor(position[1]);
				previous[2] = cl_reg_floor(position[2]);

				// basis values along the x axis
				relative = position[0] - previous[0];
				xBasis[0] = (1.0 - relative);
				xBasis[1] = relative;
				// basis values along the y axis
				relative = position[1] - previous[1];
				yBasis[0] = (1.0 - relative);
				yBasis[1] = relative;
				// basis values along the z axis
				relative = position[2] - previous[2];
				zBasis[0] = (1.0 - relative);
				zBasis[1] = relative;

				// For efficiency reason two interpolation are here, with and without using a padding value
				if (paddingValue==paddingValue) {
					// Interpolation using the padding value
					for (c = 0; c<2; c++) {
						Z = previous[2] + c;
						if (Z>-1 && Z < fi_xyz.z) {
							zPointer = &sourceIntensity[Z*fi_xyz.x*fi_xyz.y];
							yTempNewValue = 0.0;
							for (b = 0; b<2; b++) {
								Y = previous[1] + b;
								if (Y>-1 && Y < fi_xyz.y) {
									xyzPointer = &zPointer[Y*fi_xyz.x + previous[0]];
									xTempNewValue = 0.0;
									for (a = 0; a<2; a++) {
										X = previous[0] + a;
										if (X>-1 && X < fi_xyz.x) {
											xTempNewValue += *xyzPointer * xBasis[a];
										} // X
										else xTempNewValue += paddingValue * xBasis[a];
										xyzPointer++;
									} // a
									yTempNewValue += xTempNewValue * yBasis[b];
								} // Y
								else yTempNewValue += paddingValue * yBasis[b];
							} // b
							intensity += yTempNewValue * zBasis[c];
						} // Z
						else intensity += paddingValue * zBasis[c];
					} // c
				} // padding value is defined
				else if (previous[0] >= 0.f && previous[0] < (fi_xyz.x - 1) &&
					previous[1] >= 0.f && previous[1] < (fi_xyz.y - 1) &&
					previous[2] >= 0.f && previous[2] < (fi_xyz.z - 1)) {
					for (c = 0; c < 2; c++) {
						Z = previous[2] + c;
						zPointer = &sourceIntensity[Z*fi_xyz.x*fi_xyz.y];
						yTempNewValue = 0.0;
						for (b = 0; b < 2; b++) {
							Y = previous[1] + b;
							xyzPointer = &zPointer[Y*fi_xyz.x + previous[0]];
							xTempNewValue = 0.0;
							for (a = 0; a < 2; a++) {
								X = previous[0] + a;
								xTempNewValue += *xyzPointer * xBasis[a];
								xyzPointer++;
							} // a
							yTempNewValue += xTempNewValue * yBasis[b];
						} // b
						intensity += yTempNewValue * zBasis[c];
					} // c
				} // padding value is not defined
				// The voxel is outside of the source space and thus set to NaN here
				else intensity = paddingValue;
			} // voxel is in the mask

			resultIntensity[index] = intensity;

			//}
		}
		index += get_num_groups(0)*get_local_size(0);
	}

}



__kernel void NearestNeighborResampleImage(__global float *floatingImage,__global  float *deformationField,__global  float *warpedImage,__global  int *mask,__global  /*mat44*/float* sourceIJKMatrix, long2 voxelNumber, uint3 fi_xyz, uint2 wi_tu, float paddingValue) {

	// The resampling scheme is applied along each time

	__global float *sourceIntensityPtr = (floatingImage);
	__global float *resultIntensityPtr = (warpedImage);
	__global float *deformationFieldPtrX = (deformationField);
	__global float *deformationFieldPtrY = &deformationFieldPtrX[voxelNumber.x];
	__global float *deformationFieldPtrZ = &deformationFieldPtrY[voxelNumber.x];

	__global int *maskPtr = &mask[0];


	long index = get_group_id(0)*get_local_size(0) + get_local_id(0);	
	while (index < voxelNumber.x) {

		for (int t = 0; t<wi_tu.x*wi_tu.x; t++) {

			__global float *resultIntensity = &resultIntensityPtr[t*voxelNumber.x];
			__global float *sourceIntensity = &sourceIntensityPtr[t*voxelNumber.y];

			float intensity;
			float world[3];
			float position[3];
			int previous[3];

			if (maskPtr[index]>-1) {
				world[0] = (float)deformationFieldPtrX[index];
				world[1] = (float)deformationFieldPtrY[index];
				world[2] = (float)deformationFieldPtrZ[index];

				/* real -> voxel; source space */
				reg_mat44_mul_cl(sourceIJKMatrix, world, position);

				previous[0] = (int)reg_round(position[0]);
				previous[1] = (int)reg_round(position[1]);
				previous[2] = (int)reg_round(position[2]);

				if (-1 < previous[2] && previous[2] < fi_xyz.z &&
					-1 < previous[1] && previous[1] < fi_xyz.y &&
					-1 < previous[0] && previous[0] < fi_xyz.x) {
					intensity = sourceIntensity[(previous[2] * fi_xyz.y + previous[1]) * fi_xyz.x + previous[0]];
					resultIntensity[index] = intensity;
				}
				else resultIntensity[index] = paddingValue;
			}
			else resultIntensity[index] = paddingValue;


		}

		index += get_num_groups(0)*get_local_size(0);
	}
}
