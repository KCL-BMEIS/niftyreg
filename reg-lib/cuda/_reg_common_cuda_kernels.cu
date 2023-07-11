/*
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 */

#pragma once

/* *************************************************************** */
__device__ __inline__ float2 operator*(float a, float2 b) {
    return { a * b.x, a * b.y };
}
__device__ __inline__ float3 operator*(float a, float3 b) {
    return { a * b.x, a * b.y, a * b.z };
}
__device__ __inline__ float3 operator*(float3 a, float3 b) {
    return { a.x * b.x, a.y * b.y, a.z * b.z };
}
__device__ __inline__ float4 operator*(float4 a, float4 b) {
    return { a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w };
}
__device__ __inline__ float4 operator*(float a, float4 b) {
    return { a * b.x, a * b.y, a * b.z, 0.0f };
}
/* *************************************************************** */
__device__ __inline__ float2 operator/(float2 a, float2 b) {
    return { a.x / b.x, a.y / b.y };
}
__device__ __inline__ float3 operator/(float3 a, float b) {
    return { a.x / b, a.y / b, a.z / b };
}
__device__ __inline__ float3 operator/(float3 a, float3 b) {
    return { a.x / b.x, a.y / b.y, a.z / b.z };
}
/* *************************************************************** */
__device__ __inline__ float2 operator+(float2 a, float2 b) {
    return { a.x + b.x, a.y + b.y };
}
__device__ __inline__ float4 operator+(float4 a, float4 b) {
    return { a.x + b.x, a.y + b.y, a.z + b.z, 0.0f };
}
__device__ __inline__ float3 operator+(float3 a, float3 b) {
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}
/* *************************************************************** */
__device__ __inline__ float3 operator-(float3 a, float3 b) {
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}
__device__ __inline__ float4 operator-(float4 a, float4 b) {
    return { a.x - b.x, a.y - b.y, a.z - b.z, 0.f };
}
/* *************************************************************** */
__device__ __inline__ void reg_mat33_mul_cuda(const mat33& mat, const float (&in)[3], const float& weight, float (&out)[3], const bool& is3d) {
    out[0] = weight * (mat.m[0][0] * in[0] + mat.m[0][1] * in[1] + mat.m[0][2] * in[2]);
    out[1] = weight * (mat.m[1][0] * in[0] + mat.m[1][1] * in[1] + mat.m[1][2] * in[2]);
    out[2] = is3d ? weight * (mat.m[2][0] * in[0] + mat.m[2][1] * in[1] + mat.m[2][2] * in[2]) : 0;
}
/* *************************************************************** */
__device__ __inline__ void reg_mat44_mul_cuda(const mat44& mat, const float (&in)[3], float (&out)[3], const bool& is3d) {
    out[0] = mat.m[0][0] * in[0] + mat.m[0][1] * in[1] + mat.m[0][2] * in[2] + mat.m[0][3];
    out[1] = mat.m[1][0] * in[0] + mat.m[1][1] * in[1] + mat.m[1][2] * in[2] + mat.m[1][3];
    out[2] = is3d ? mat.m[2][0] * in[0] + mat.m[2][1] * in[1] + mat.m[2][2] * in[2] + mat.m[2][3] : 0;
}
/* *************************************************************** */
__device__ __inline__ void reg_div_cuda(const int num, const int denom, int& quot, int& rem) {
    // This will be optimised by the compiler into a single div instruction
    quot = num / denom;
    rem = num % denom;
}
/* *************************************************************** */
