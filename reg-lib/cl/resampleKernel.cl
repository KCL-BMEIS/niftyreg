//To enable double precision
#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define DOUBLE_SUPPORT_AVAILABLE
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#define DOUBLE_SUPPORT_AVAILABLE
#else
#warning "double precision floating point not supported by OpenCL implementation.";
#endif

#if defined(DOUBLE_SUPPORT_AVAILABLE)

// double
typedef double real_t;
typedef double2 real2_t;
typedef double3 real3_t;
typedef double4 real4_t;
typedef double8 real8_t;
typedef double16 real16_t;
#define PI 3.14159265358979323846

#else

// float
typedef float real_t;
typedef float2 real2_t;
typedef float3 real3_t;
typedef float4 real4_t;
typedef float8 real8_t;
typedef float16 real16_t;
#define PI 3.14159265359f

#endif

#define SINC_KERNEL_RADIUS 3
#define SINC_KERNEL_SIZE SINC_KERNEL_RADIUS*2

/* *************************************************************** */
/* *************************************************************** */
__inline void interpWindowedSincKernel(real_t relative, real_t *basis)
{
    if (relative < (real_t) 0.0) {
        relative = (real_t) 0.0; //reg_rounding error
    }
    int j = 0;
    real_t sum = (real_t) 0.0;
    for (int i = -SINC_KERNEL_RADIUS; i < SINC_KERNEL_RADIUS; ++i) {
        real_t x = relative - (real_t)(i);
        if (x == (real_t) 0.0) {
            basis[j] = (real_t) 1.0;
        }
        else if (fabs(x) >= (real_t)(SINC_KERNEL_RADIUS)) {
            basis[j] = (real_t) 0.0;
        }
        else {
            real_t pi_x = PI * x;
            basis[j] = ((real_t)SINC_KERNEL_RADIUS) * sin(pi_x) * sin(pi_x / (real_t)SINC_KERNEL_RADIUS) / (pi_x * pi_x);
        }
        sum += basis[j];
        j++;
    }
    for (int i = 0; i < SINC_KERNEL_SIZE; ++i) {
        basis[i] /= sum;
    }
}
/* *************************************************************** */
/* *************************************************************** */
__inline void interpCubicSplineKernel(real_t relative, real_t *basis)
{
    if (relative < (real_t) 0.0) {
        relative = (real_t) 0.0; //reg_rounding error
    }
    real_t FF = relative * relative;
    basis[0] = (relative * (((real_t) 2.0 - relative) * relative - (real_t) 1.0)) / (real_t) 2.0;
    basis[1] = (FF * ((real_t) 3.0 * relative - 5.0) + 2.0) / (real_t) 2.0;
    basis[2] = (relative * (((real_t) 4.0 - (real_t) 3.0 * relative) * relative + (real_t) 1.0)) / (real_t) 2.0;
    basis[3] = (relative - (real_t) 1.0) * FF / (real_t) 2.0;

}
/* *************************************************************** */
/* *************************************************************** */
__inline void interpLinearKernel(real_t relative, real_t *basis)
{
    if (relative < (real_t) 0.0) relative = (real_t) 0.0; //reg_rounding error
    basis[1] = relative;
    basis[0] = (real_t) 1.0 - relative;
}
/* *************************************************************** */
/* *************************************************************** */
__inline void interpNearestNeighKernel(real_t relative, real_t *basis)
{
    if (relative < (real_t) 0.0) {
        relative = 0.0; //reg_rounding error
    }
    basis[0] = basis[1] = (real_t) 0.0;
    if (relative >= (real_t) 0.5){
        basis[1] = (real_t) 1.0;
    }
    else {
        basis[0] = (real_t) 1.0;
    }
}
/* *************************************************************** */
/* *************************************************************** */
__inline real_t interpLoop2D(__global float* floatingIntensity,
    real_t* xBasis,
    real_t* yBasis,
    real_t* zBasis,
    int *previous,
    uint3 fi_xyz,
    float paddingValue,
    unsigned int kernel_size)
{
    real_t intensity = (real_t) 0.0;
    
        for (unsigned int b = 0; b < kernel_size; b++) {
            int Y = previous[1] + b;
            bool yInBounds = -1 < Y && Y < fi_xyz.y;
            real_t xTempNewValue = (real_t) 0.0;
            
            for (unsigned int a = 0; a < kernel_size; a++) {
                int X = previous[0] + a;
                bool xInBounds = -1 < X && X < fi_xyz.x;

                const unsigned int idx = Y * fi_xyz.x + X;

                xTempNewValue += (xInBounds && yInBounds) ? floatingIntensity[idx] * xBasis[a] : paddingValue * xBasis[a];
            }
            intensity += xTempNewValue * yBasis[b];
    }
    return intensity;

}
/* *************************************************************** */
/* *************************************************************** */
__inline real_t interpLoop3D(__global float* floatingIntensity,
    real_t* xBasis,
    real_t* yBasis,
    real_t* zBasis,
    int *previous,
    uint3 fi_xyz,
    float paddingValue,
    unsigned int kernel_size)
{
    real_t intensity = (real_t) 0.0;
    for (unsigned int c = 0; c < kernel_size; c++) {
        int Z = previous[2] + c;
        bool zInBounds = -1 < Z && Z < fi_xyz.z;
        real_t yTempNewValue = (real_t) 0.0;
        for (unsigned int b = 0; b < kernel_size; b++) {
            int Y = previous[1] + b;
            bool yInBounds = -1 < Y && Y < fi_xyz.y;
            real_t xTempNewValue = (real_t) 0.0;
            for (unsigned int a = 0; a < kernel_size; a++) {
                int X = previous[0] + a;
                bool xInBounds = -1 < X && X < fi_xyz.x;
                const unsigned int idx = Z * fi_xyz.x * fi_xyz.y + Y * fi_xyz.x + X;

                xTempNewValue += (xInBounds && yInBounds  && zInBounds) ? floatingIntensity[idx] * xBasis[a] : paddingValue * xBasis[a];
            }
            yTempNewValue += xTempNewValue * yBasis[b];
        }
        intensity += yTempNewValue * zBasis[c];
    }

    return intensity;
}
/* *************************************************************** */
/* *************************************************************** */
__inline int cl_reg_floor(real_t a)
{
    return a > 0.0 ? (int)a : (int)(a - 1);
}
/* *************************************************************** */
/* *************************************************************** */
__inline void reg_mat44_mul_cl(__global float const* mat,
    float const* in,
    float *out)
{
    out[0] = (float)((real_t)mat[0 * 4 + 0] * (real_t)in[0] +
        (real_t)mat[0 * 4 + 1] * (real_t)in[1] +
        (real_t)mat[0 * 4 + 2] * (real_t)in[2] +
        (real_t)mat[0 * 4 + 3]);
    out[1] = (float)((real_t)mat[1 * 4 + 0] * (real_t)in[0] +
        (real_t)mat[1 * 4 + 1] * (real_t)in[1] +
        (real_t)mat[1 * 4 + 2] * (real_t)in[2] +
        (real_t)mat[1 * 4 + 3]);
    out[2] = (float)((real_t)mat[2 * 4 + 0] * (real_t)in[0] +
        (real_t)mat[2 * 4 + 1] * (real_t)in[1] +
        (real_t)mat[2 * 4 + 2] * (real_t)in[2] +
        (real_t)mat[2 * 4 + 3]);
    return;
}
/* *************************************************************** */
/* *************************************************************** */
float cl_reg_round(float a)
{
    return (float)((a) > 0.0f ? (int)((a)+0.5) : (int)((a)-0.5));
}
/* *************************************************************** */
/* *************************************************************** */
__kernel void ResampleImage2D(__global float* floatingImage,
    __global float* deformationField,
    __global float* warpedImage,
    __global int *mask,
    __global float* sourceIJKMatrix,
    long2 voxelNumber,
    uint3 fi_xyz,
    uint2 wi_tu,
    float paddingValue,
    int kernelType,
    int datatype)
{
    __global float *sourceIntensityPtr = (floatingImage);
    __global float *resultIntensityPtr = (warpedImage);
    __global float *deformationFieldPtrX = (deformationField);
    __global float *deformationFieldPtrY = &deformationFieldPtrX[voxelNumber.x];

    __global int *maskPtr = &mask[0];


    long index = get_group_id(0)*get_local_size(0) + get_local_id(0);
    while (index < voxelNumber.x) {

        for (unsigned int t = 0; t < wi_tu.x * wi_tu.y; t++) {

            __global float *resultIntensity = &resultIntensityPtr[t * voxelNumber.x];
            __global float *floatingIntensity = &sourceIntensityPtr[t * voxelNumber.y];
            real_t intensity = paddingValue;

            if (maskPtr[index] > -1) {
                int  previous[3];
                float world[3], position[3];
                real_t relative[3];

                world[0] = (deformationFieldPtrX[index]);
                world[1] = (deformationFieldPtrY[index]);
                world[2] = 0.0f;

                // real -> voxel; floating space
                reg_mat44_mul_cl(sourceIJKMatrix, world, position);

                previous[0] = cl_reg_floor(position[0]);
                previous[1] = cl_reg_floor(position[1]);

                relative[0] = (real_t)position[0] - (real_t)(previous[0]);
                relative[1] = (real_t)position[1] - (real_t)(previous[1]);

                if (kernelType == 0) {

                    real_t xBasisIn[2], yBasisIn[2], zBasisIn[2];
                    interpNearestNeighKernel(relative[0], xBasisIn);
                    interpNearestNeighKernel(relative[1], yBasisIn);
                    intensity = interpLoop2D(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, paddingValue, 2);

                }
                else if (kernelType == 1){

                    real_t xBasisIn[2], yBasisIn[2], zBasisIn[2];
                    interpLinearKernel(relative[0], xBasisIn);
                    interpLinearKernel(relative[1], yBasisIn);
                    intensity = interpLoop2D(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, paddingValue, 2);

                }
                else if (kernelType == 4){

                    previous[0] -= SINC_KERNEL_RADIUS;
                    previous[1] -= SINC_KERNEL_RADIUS;
                    previous[2] -= SINC_KERNEL_RADIUS;
                    real_t xBasisIn[6], yBasisIn[6], zBasisIn[6];
                    interpWindowedSincKernel(relative[0], xBasisIn);
                    interpWindowedSincKernel(relative[1], yBasisIn);
                    intensity = interpLoop2D(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, paddingValue, 6);
                }
                else{

                    previous[0]--;
                    previous[1]--;
                    previous[2]--;
                    real_t xBasisIn[4], yBasisIn[4], zBasisIn[4];
                    interpCubicSplineKernel(relative[0], xBasisIn);
                    interpCubicSplineKernel(relative[1], yBasisIn);
                    intensity = interpLoop2D(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, paddingValue, 4);
                }
            }
            resultIntensity[index] = (float)intensity;
        }
        index += get_num_groups(0)*get_local_size(0);
    }
}
/* *************************************************************** */
/* *************************************************************** */
__kernel void ResampleImage3D(__global float* floatingImage,
    __global float* deformationField,
    __global float* warpedImage,
    __global int *mask,
    __global float* sourceIJKMatrix,
    long2 voxelNumber,
    uint3 fi_xyz,
    uint2 wi_tu,
    float paddingValue,
    int kernelType,
    int datatype)
{
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
            real_t intensity = paddingValue;

            if (maskPtr[index] > -1) {
                int  previous[3];
                float world[3], position[3];
                real_t relative[3];

                world[0] = (deformationFieldPtrX[index]);
                world[1] = (deformationFieldPtrY[index]);
                world[2] = (deformationFieldPtrZ[index]);

                // real -> voxel; floating space
                reg_mat44_mul_cl(sourceIJKMatrix, world, position);

                previous[0] = cl_reg_floor(position[0]);
                previous[1] = cl_reg_floor(position[1]);
                previous[2] = cl_reg_floor(position[2]);

                relative[0] = (real_t)position[0] - (real_t)(previous[0]);
                relative[1] = (real_t)position[1] - (real_t)(previous[1]);
                relative[2] = (real_t)position[2] - (real_t)(previous[2]);

                if (kernelType == 0) {

                    real_t xBasisIn[2], yBasisIn[2], zBasisIn[2];
                    interpNearestNeighKernel(relative[0], xBasisIn);
                    interpNearestNeighKernel(relative[1], yBasisIn);
                    interpNearestNeighKernel(relative[2], zBasisIn);
                    intensity = interpLoop3D(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, paddingValue, 2);

                }
                else if (kernelType == 1){

                    real_t xBasisIn[2], yBasisIn[2], zBasisIn[2];
                    interpLinearKernel(relative[0], xBasisIn);
                    interpLinearKernel(relative[1], yBasisIn);
                    interpLinearKernel(relative[2], zBasisIn);
                    intensity = interpLoop3D(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, paddingValue, 2);

                }
                else if (kernelType == 4){

                    previous[0] -= SINC_KERNEL_RADIUS;
                    previous[1] -= SINC_KERNEL_RADIUS;
                    previous[2] -= SINC_KERNEL_RADIUS;
                    real_t xBasisIn[6], yBasisIn[6], zBasisIn[6];
                    interpWindowedSincKernel(relative[0], xBasisIn);
                    interpWindowedSincKernel(relative[1], yBasisIn);
                    interpWindowedSincKernel(relative[2], zBasisIn);
                    intensity = interpLoop3D(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, paddingValue, 6);
                }
                else{

                    previous[0] --;
                    previous[1] --;
                    previous[2] --;
                    real_t xBasisIn[4], yBasisIn[4], zBasisIn[4];
                    interpCubicSplineKernel(relative[0], xBasisIn);
                    interpCubicSplineKernel(relative[1], yBasisIn);
                    interpCubicSplineKernel(relative[2], zBasisIn);
                    intensity = interpLoop3D(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, paddingValue, 4);
                }
            }
            resultIntensity[index] = (float)intensity;
        }
        index += get_num_groups(0)*get_local_size(0);
    }
}
/* *************************************************************** */
/* *************************************************************** */
