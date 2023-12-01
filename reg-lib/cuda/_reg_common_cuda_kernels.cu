/*
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 */

#pragma once

/* *************************************************************** */
template<bool is3d>
__device__ __inline__ void reg_mat33_mul_cuda(const mat33 mat, const float (&in)[3], const double weight, float (&out)[3]) {
    out[0] = weight * (mat.m[0][0] * in[0] + mat.m[1][0] * in[1] + mat.m[2][0] * in[2]);
    out[1] = weight * (mat.m[0][1] * in[0] + mat.m[1][1] * in[1] + mat.m[2][1] * in[2]);
    if constexpr (is3d)
        out[2] = weight * (mat.m[0][2] * in[0] + mat.m[1][2] * in[1] + mat.m[2][2] * in[2]);
}
/* *************************************************************** */
template<bool is3d>
__device__ __inline__ void reg_mat44_mul_cuda(const mat44 mat, const float (&in)[3], float (&out)[3]) {
    out[0] = double(mat.m[0][0]) * double(in[0]) + double(mat.m[0][1]) * double(in[1]) + double(mat.m[0][2]) * double(in[2]) + double(mat.m[0][3]);
    out[1] = double(mat.m[1][0]) * double(in[0]) + double(mat.m[1][1]) * double(in[1]) + double(mat.m[1][2]) * double(in[2]) + double(mat.m[1][3]);
    if constexpr (is3d)
        out[2] = double(mat.m[2][0]) * double(in[0]) + double(mat.m[2][1]) * double(in[1]) + double(mat.m[2][2]) * double(in[2]) + double(mat.m[2][3]);
}
/* *************************************************************** */
__device__ __inline__ mat33 reg_mat33_mul_cuda(const mat33 a, const mat33 b) {
    mat33 c;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            c.m[i][j] = a.m[i][0] * b.m[0][j] + a.m[i][1] * b.m[1][j] + a.m[i][2] * b.m[2][j];
    return c;
}
/* *************************************************************** */
__device__ __inline__ mat33 reg_mat33_inverse_cuda(const mat33 r) {
    /*  INPUT MATRIX:  */
    const double r11 = r.m[0][0]; const double r12 = r.m[0][1]; const double r13 = r.m[0][2];  /* [ r11 r12 r13 ] */
    const double r21 = r.m[1][0]; const double r22 = r.m[1][1]; const double r23 = r.m[1][2];  /* [ r21 r22 r23 ] */
    const double r31 = r.m[2][0]; const double r32 = r.m[2][1]; const double r33 = r.m[2][2];  /* [ r31 r32 r33 ] */

    double deti = (r11 * r22 * r33 - r11 * r32 * r23 - r21 * r12 * r33 +
                   r21 * r32 * r13 + r31 * r12 * r23 - r31 * r22 * r13);

    if (deti != 0.0) deti = 1.0 / deti;

    mat33 q;
    q.m[0][0] = float(deti * (r22 * r33 - r32 * r23));
    q.m[0][1] = float(deti * (-r12 * r33 + r32 * r13));
    q.m[0][2] = float(deti * (r12 * r23 - r22 * r13));

    q.m[1][0] = float(deti * (-r21 * r33 + r31 * r23));
    q.m[1][1] = float(deti * (r11 * r33 - r31 * r13));
    q.m[1][2] = float(deti * (-r11 * r23 + r21 * r13));

    q.m[2][0] = float(deti * (r21 * r32 - r31 * r22));
    q.m[2][1] = float(deti * (-r11 * r32 + r31 * r12));
    q.m[2][2] = float(deti * (r11 * r22 - r21 * r12));

    return q;
}
/* *************************************************************** */
__device__ __inline__ float reg_mat33_determ_cuda(const mat33 r) {
    /*  INPUT MATRIX:  */
    const double r11 = r.m[0][0]; const double r12 = r.m[0][1]; const double r13 = r.m[0][2];  /* [ r11 r12 r13 ] */
    const double r21 = r.m[1][0]; const double r22 = r.m[1][1]; const double r23 = r.m[1][2];  /* [ r21 r22 r23 ] */
    const double r31 = r.m[2][0]; const double r32 = r.m[2][1]; const double r33 = r.m[2][2];  /* [ r31 r32 r33 ] */

    return float(r11 * r22 * r33 - r11 * r32 * r23 - r21 * r12 * r33 +
                 r21 * r32 * r13 + r31 * r12 * r23 - r31 * r22 * r13);
}
/* *************************************************************** */
__device__ __inline__ float reg_mat33_rownorm_cuda(const mat33 a) {
    float r1 = fabs(a.m[0][0]) + fabs(a.m[0][1]) + fabs(a.m[0][2]);
    const float r2 = fabs(a.m[1][0]) + fabs(a.m[1][1]) + fabs(a.m[1][2]);
    const float r3 = fabs(a.m[2][0]) + fabs(a.m[2][1]) + fabs(a.m[2][2]);
    if (r1 < r2) r1 = r2;
    if (r1 < r3) r1 = r3;
    return r1;
}
/* *************************************************************** */
__device__ __inline__ float reg_mat33_colnorm_cuda(const mat33 a) {
    float r1 = fabs(a.m[0][0]) + fabs(a.m[1][0]) + fabs(a.m[2][0]);
    const float r2 = fabs(a.m[0][1]) + fabs(a.m[1][1]) + fabs(a.m[2][1]);
    const float r3 = fabs(a.m[0][2]) + fabs(a.m[1][2]) + fabs(a.m[2][2]);
    if (r1 < r2) r1 = r2;
    if (r1 < r3) r1 = r3;
    return r1;
}
/* *************************************************************** */
__device__ __inline__ mat33 reg_mat33_polar_cuda(mat33 x) {
    // Force matrix to be nonsingular
    float gam = reg_mat33_determ_cuda(x);
    while (gam == 0.0) {        // Perturb matrix
        gam = 0.00001f * (0.001f + reg_mat33_rownorm_cuda(x));
        x.m[0][0] += gam; x.m[1][1] += gam; x.m[2][2] += gam;
        gam = reg_mat33_determ_cuda(x);
    }

    mat33 z;
    float gmi, dif = 1.0f;
    int k = 0;
    while (1) {
        const mat33 y = reg_mat33_inverse_cuda(x);
        if (dif > 0.3) {     // Far from convergence
            const float alp = sqrt(reg_mat33_rownorm_cuda(x) * reg_mat33_colnorm_cuda(x));
            const float bet = sqrt(reg_mat33_rownorm_cuda(y) * reg_mat33_colnorm_cuda(y));
            gam = sqrt(bet / alp);
            gmi = 1.f / gam;
        } else {
            gam = gmi = 1.0f;  // Close to convergence
        }
        z.m[0][0] = 0.5f * (gam * x.m[0][0] + gmi * y.m[0][0]);
        z.m[0][1] = 0.5f * (gam * x.m[0][1] + gmi * y.m[1][0]);
        z.m[0][2] = 0.5f * (gam * x.m[0][2] + gmi * y.m[2][0]);
        z.m[1][0] = 0.5f * (gam * x.m[1][0] + gmi * y.m[0][1]);
        z.m[1][1] = 0.5f * (gam * x.m[1][1] + gmi * y.m[1][1]);
        z.m[1][2] = 0.5f * (gam * x.m[1][2] + gmi * y.m[2][1]);
        z.m[2][0] = 0.5f * (gam * x.m[2][0] + gmi * y.m[0][2]);
        z.m[2][1] = 0.5f * (gam * x.m[2][1] + gmi * y.m[1][2]);
        z.m[2][2] = 0.5f * (gam * x.m[2][2] + gmi * y.m[2][2]);

        dif = (fabs(z.m[0][0] - x.m[0][0]) + fabs(z.m[0][1] - x.m[0][1]) +
               fabs(z.m[0][2] - x.m[0][2]) + fabs(z.m[1][0] - x.m[1][0]) +
               fabs(z.m[1][1] - x.m[1][1]) + fabs(z.m[1][2] - x.m[1][2]) +
               fabs(z.m[2][0] - x.m[2][0]) + fabs(z.m[2][1] - x.m[2][1]) +
               fabs(z.m[2][2] - x.m[2][2]));

        k = k + 1;
        if (k > 100 || dif < 3.e-6) break;  // Convergence or exhaustion
        x = z;
    }

    return z;
}
/* *************************************************************** */
__device__ __inline__ void reg_div_cuda(const int num, const int denom, int& quot, int& rem) {
    // This will be optimised by the compiler into a single div instruction
    quot = num / denom;
    rem = num % denom;
}
/* *************************************************************** */
template<bool is3d>
__device__ __inline__ int3 reg_indexToDims_cuda(const int index, const int3 dims) {
    int quot = 0, rem;
    if constexpr (is3d)
        reg_div_cuda(index, dims.x * dims.y, quot, rem);
    else rem = index;
    const int z = quot;
    reg_div_cuda(rem, dims.x, quot, rem);
    const int y = quot, x = rem;
    return { x, y, z };
}
/* *************************************************************** */
__device__ __inline__ int3 reg_indexToDims_cuda(const int index, const int3 dims) {
    return dims.z > 1 ? reg_indexToDims_cuda<true>(index, dims) : reg_indexToDims_cuda<false>(index, dims);
}
/* *************************************************************** */
