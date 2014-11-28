#define SINC_KERNEL_RADIUS 3
#define SINC_KERNEL_SIZE SINC_KERNEL_RADIUS*2
//#define M_PI 3.141593f   /* pi             */

__inline void interpWindowedSincKernel(float relative, float *basis) {
    if (relative < 0.0f)
        relative = 0.0f; //reg_rounding error
    int j = 0;
    float sum = 0.;
    for (int i = -SINC_KERNEL_RADIUS; i < SINC_KERNEL_RADIUS; ++i) {
        float x = relative - (float)(i);
        if (x == 0.0f)
            basis[j] = 1.0f;
        else if (fabs(x) >= (float)(SINC_KERNEL_RADIUS))
            basis[j] = 0;
        else {
            float pi_x = 3.141593f * x;
            basis[j] = (SINC_KERNEL_RADIUS) * sin(pi_x) * sin(pi_x / SINC_KERNEL_RADIUS) / (pi_x * pi_x);
        }
        sum += basis[j];
        j++;
    }
    for (int i = 0; i < SINC_KERNEL_SIZE; ++i)
        basis[i] /= sum;
}
/* *************************************************************** */
/* *************************************************************** */
__inline void interpCubicSplineKernel(float relative, float *basis) {
    if (relative < 0.0f)
        relative = 0.0f; //reg_rounding error
    float FF = relative * relative;
    basis[0] = (relative * ((2.0f - relative) * relative - 1.0f)) / 2.0f;
    basis[1] = (FF * (3.0f * relative - 5.0f) + 2.0f) / 2.0f;
    basis[2] = (relative * ((4.0f - 3.0f * relative) * relative + 1.0f)) / 2.0f;
    basis[3] = (relative - 1.0f) * FF / 2.0f;
}
/* *************************************************************** */
/* *************************************************************** */
__inline void interpLinearKernel(float relative, float *basis) {
    if (relative < 0.0f)
        relative = 0.0f; //reg_rounding error
    basis[1] = relative;
    basis[0] = 1.0f - relative;
}

/* *************************************************************** */
/* *************************************************************** */
__inline void interpNearestNeighKernel(float relative, float *basis) {
    if (relative < 0.0f)
        relative = 0.0f; //reg_rounding error
    basis[0] = basis[1] = 0.0f;
    if (relative > 0.5f)
        basis[1] = 1;
    else
        basis[0] = 1;
}
/* *************************************************************** */
/* *************************************************************** */
__inline float interpLoop(__global float* floatingIntensity,float* xBasis, float* yBasis, float* zBasis,  int* previous, uint3 fi_xyz, float paddingValue, unsigned int kernel_size){
    float intensity = paddingValue;
    for (int c = 0; c < kernel_size; c++) {
        int Z = previous[2] + c;
        bool zInBounds = -1 < Z && Z < fi_xyz.z;
        float yTempNewValue = 0.0f;
        for (int b = 0; b < kernel_size; b++) {
            int Y = previous[1] + b;
            bool yInBounds = -1 < Y && Y < fi_xyz.y;
            float xTempNewValue = 0.0f;
            for (int a = 0; a < kernel_size; a++) {
            	int X = previous[0] + a;
				bool xInBounds = -1 < X && (X + a) < fi_xyz.x;
				const unsigned int idx = Z * fi_xyz.x * fi_xyz.y + Y * fi_xyz.x + X;
				xTempNewValue += (xInBounds && yInBounds  && zInBounds)? floatingIntensity[idx] * xBasis[a]:paddingValue * xBasis[a];
            }
            yTempNewValue += xTempNewValue * yBasis[b];
        }
        intensity += yTempNewValue * zBasis[c];
    }
    return intensity;
}
 

 __inline int cl_reg_floor(float a) {
	return a > 0.0f ? (int)a : (int)(a - 1);
}

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

__kernel void ResampleImage3D(__global float* floatingImage, __global float* deformationField, __global float* warpedImage,__global int* mask,__global float* sourceIJKMatrix, long2 voxelNumber, uint3 fi_xyz, uint2 wi_tu, float paddingValue, int kernelType) {

    __global float *sourceIntensityPtr = (floatingImage);
    __global float *resultIntensityPtr = (warpedImage);
    __global float *deformationFieldPtrX = (deformationField);
    __global float *deformationFieldPtrY = &deformationFieldPtrX[voxelNumber.x];
    __global float *deformationFieldPtrZ = &deformationFieldPtrY[voxelNumber.x];
    
    __global int *maskPtr = &mask[0];
    
    
    long index = get_group_id(0)*get_local_size(0) + get_local_id(0);
    while (index < voxelNumber.x) {
        

        for (unsigned int t = 0; t < wi_tu.x * wi_tu.y; t++) {

            __global float *resultIntensity = &resultIntensityPtr[t * voxelNumber.x];
            __global float *floatingIntensity = &sourceIntensityPtr[t * voxelNumber.y];
        
            


            if (maskPtr[index] > -1) {

                int  previous[3];
                float world[3], position[3], relative[3], intensity;
                
                world[0] = (deformationFieldPtrX[index]);
                world[1] = (deformationFieldPtrY[index]);
                world[2] = (deformationFieldPtrZ[index]);
            
                // real -> voxel; floating space
                reg_mat44_mul_cl(sourceIJKMatrix, world, position);
            
                previous[0] = cl_reg_floor(position[0]);
                previous[1] = cl_reg_floor(position[1]);
                previous[2] = cl_reg_floor(position[2]);
            
                relative[0] = position[0] - (float)(previous[0]);
                relative[1] = position[1] - (float)(previous[1]);
                relative[2] = position[2] - (float)(previous[2]);
                
                
                
                if (kernelType == 0) {

                        float xBasisIn[2], yBasisIn[2], zBasisIn[2];
                        interpNearestNeighKernel(relative[0], xBasisIn);
                        interpNearestNeighKernel(relative[1], yBasisIn);
                        interpNearestNeighKernel(relative[2], zBasisIn);
                        intensity = interpLoop(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, paddingValue, 2);
                }else if (kernelType == 1){

                        float xBasisIn[2], yBasisIn[2], zBasisIn[2];
                        interpLinearKernel(relative[0], xBasisIn);
                        interpLinearKernel(relative[1], yBasisIn);
                        interpLinearKernel(relative[2], zBasisIn);
                        intensity = interpLoop(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, paddingValue, 2);
                }else if (kernelType == 4){
                
                        previous[0] -= SINC_KERNEL_RADIUS;
                        previous[1] -= SINC_KERNEL_RADIUS;
                        previous[2] -= SINC_KERNEL_RADIUS;
                        float xBasisIn[6], yBasisIn[6], zBasisIn[6];
                        interpWindowedSincKernel(relative[0], xBasisIn);
                        interpWindowedSincKernel(relative[1], yBasisIn);
                        interpWindowedSincKernel(relative[2], zBasisIn);
                        intensity = interpLoop(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, paddingValue, 6);
                }else{
                
                        previous[0] --;
                        previous[1] --;
                        previous[2] --;
                        float xBasisIn[4], yBasisIn[4], zBasisIn[4];
                        interpCubicSplineKernel(relative[0], xBasisIn);
                        interpCubicSplineKernel(relative[1], yBasisIn);
                        interpCubicSplineKernel(relative[2], zBasisIn);
                        intensity = interpLoop(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, paddingValue, 4);
                }

                resultIntensity[index] = intensity;
            }
        }
        index += get_num_groups(0)*get_local_size(0);
    }
<<<<<<< HEAD
}
=======
}
>>>>>>> branch 't_dev/cl_aladin' of ssh://thanasio@git.code.sf.net/p/niftyreg/git
