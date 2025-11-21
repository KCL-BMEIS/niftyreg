/*
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the root folder
 */

#pragma once

/* *************************************************************** */
template<typename T>
__device__ __inline__ float2 operator*(const T& a, const float2& b) {
    return { static_cast<float>(a) * b.x, static_cast<float>(a) * b.y };
}
template<typename T>
__device__ __inline__ float2 operator*(const float2& a, const T& b) {
    return b * a;
}
__device__ __inline__ float2 operator*(const float2& a, const float2& b) {
    return { a.x * b.x, a.y * b.y };
}
/* *************************************************************** */
template<typename T>
__device__ __inline__ float3 operator*(const T& a, const float3& b) {
    return { static_cast<float>(a) * b.x, static_cast<float>(a) * b.y, static_cast<float>(a) * b.z };
}
template<typename T>
__device__ __inline__ float3 operator*(const float3& a, const T& b) {
    return b * a;
}
__device__ __inline__ float3 operator*(const float3& a, const float3& b) {
    return { a.x * b.x, a.y * b.y, a.z * b.z };
}
/* *************************************************************** */
template<typename T>
__device__ __inline__ float4 operator*(const T& a, const float4& b) {
    return { static_cast<float>(a) * b.x, static_cast<float>(a) * b.y, static_cast<float>(a) * b.z, static_cast<float>(a) * b.w };
}
template<typename T>
__device__ __inline__ float4 operator*(const float4& a, const T& b) {
    return b * a;
}
__device__ __inline__ float4 operator*(const float4& a, const float4& b) {
    return { a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w };
}
/* *************************************************************** */
template<typename T>
__device__ __inline__ float2 operator/(const T& a, const float2& b) {
    return { static_cast<float>(a) / b.x, static_cast<float>(a) / b.y };
}
template<typename T>
__device__ __inline__ float2 operator/(const float2& a, const T& b) {
    return { a.x / static_cast<float>(b), a.y / static_cast<float>(b) };
}
__device__ __inline__ float2 operator/(const float2& a, const float2& b) {
    return { a.x / b.x, a.y / b.y };
}
/* *************************************************************** */
template<typename T>
__device__ __inline__ float3 operator/(const T& a, const float3& b) {
    return { static_cast<float>(a) / b.x, static_cast<float>(a) / b.y, static_cast<float>(a) / b.z };
}
template<typename T>
__device__ __inline__ float3 operator/(const float3& a, const T& b) {
    return { a.x / static_cast<float>(b), a.y / static_cast<float>(b), a.z / static_cast<float>(b) };
}
__device__ __inline__ float3 operator/(const float3& a, const float3& b) {
    return { a.x / b.x, a.y / b.y, a.z / b.z };
}
/* *************************************************************** */
template<typename T>
__device__ __inline__ float4 operator/(const T& a, const float4& b) {
    return { static_cast<float>(a) / b.x, static_cast<float>(a) / b.y, static_cast<float>(a) / b.z, static_cast<float>(a) / b.w };
}
template<typename T>
__device__ __inline__ float4 operator/(const float4& a, const T& b) {
    return { a.x / static_cast<float>(b), a.y / static_cast<float>(b), a.z / static_cast<float>(b), a.w / static_cast<float>(b) };
}
__device__ __inline__ float4 operator/(const float4& a, const float4& b) {
    return { a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w };
}
/* *************************************************************** */
template<typename T>
__device__ __inline__ float2 operator+(const T& a, const float2& b) {
    return { static_cast<float>(a) + b.x, static_cast<float>(a) + b.y };
}
template<typename T>
__device__ __inline__ float2 operator+(const float2& a, const T& b) {
    return b + a;
}
__device__ __inline__ float2 operator+(const float2& a, const float2& b) {
    return { a.x + b.x, a.y + b.y };
}
/* *************************************************************** */
template<typename T>
__device__ __inline__ float3 operator+(const T& a, const float3& b) {
    return { static_cast<float>(a) + b.x, static_cast<float>(a) + b.y, static_cast<float>(a) + b.z };
}
template<typename T>
__device__ __inline__ float3 operator+(const float3& a, const T& b) {
    return b + a;
}
__device__ __inline__ float3 operator+(const float3& a, const float3& b) {
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}
/* *************************************************************** */
template<typename T>
__device__ __inline__ float4 operator+(const T& a, const float4& b) {
    return { static_cast<float>(a) + b.x, static_cast<float>(a) + b.y, static_cast<float>(a) + b.z, static_cast<float>(a) + b.w };
}
template<typename T>
__device__ __inline__ float4 operator+(const float4& a, const T& b) {
    return b + a;
}
__device__ __inline__ float4 operator+(const float4& a, const float4& b) {
    return { a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
}
/* *************************************************************** */
template<typename T>
__device__ __inline__ float2 operator-(const T& a, const float2& b) {
    return { static_cast<float>(a) - b.x, static_cast<float>(a) - b.y };
}
template<typename T>
__device__ __inline__ float2 operator-(const float2& a, const T& b) {
    return { a.x - static_cast<float>(b), a.y - static_cast<float>(b) };
}
__device__ __inline__ float2 operator-(const float2& a, const float2& b) {
    return { a.x - b.x, a.y - b.y };
}
/* *************************************************************** */
template<typename T>
__device__ __inline__ float3 operator-(const T& a, const float3& b) {
    return { static_cast<float>(a) - b.x, static_cast<float>(a) - b.y, static_cast<float>(a) - b.z };
}
template<typename T>
__device__ __inline__ float3 operator-(const float3& a, const T& b) {
    return { a.x - static_cast<float>(b), a.y - static_cast<float>(b), a.z - static_cast<float>(b) };
}
__device__ __inline__ float3 operator-(const float3& a, const float3& b) {
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}
/* *************************************************************** */
template<typename T>
__device__ __inline__ float4 operator-(const T& a, const float4& b) {
    return { static_cast<float>(a) - b.x, static_cast<float>(a) - b.y, static_cast<float>(a) - b.z, static_cast<float>(a) - b.w };
}
template<typename T>
__device__ __inline__ float4 operator-(const float4& a, const T& b) {
    return { a.x - static_cast<float>(b), a.y - static_cast<float>(b), a.z - static_cast<float>(b), a.w - static_cast<float>(b) };
}
__device__ __inline__ float4 operator-(const float4& a, const float4& b) {
    return { a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w };
}
/* *************************************************************** */
__device__ __inline__ double2 operator+(const double2& a, const double2& b) {
    return { a.x + b.x, a.y + b.y };
}
/* *************************************************************** */
__device__ __inline__ float2 make_float2(const float4& a) {
    return { a.x, a.y };
}
/* *************************************************************** */
__device__ __inline__ float3 make_float3(const float4& a) {
    return { a.x, a.y, a.z };
}
/* *************************************************************** */
__device__ __inline__ float4 make_float4(const float3& a) {
    return { a.x, a.y, a.z, 0.f };
}
/* *************************************************************** */
