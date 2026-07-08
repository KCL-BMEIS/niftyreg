#include "CudaLts.hpp"
#include "Maths.hpp"

#include <cusolverDn.h>
#include <cublas_v2.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <cfloat>

/* *************************************************************** */
namespace NiftyReg::Cuda {
/* *************************************************************** */
namespace {
constexpr int kReduceThreads = 256;
/* *************************************************************** */
// Create-once.
cusolverDnHandle_t GetSolver() {
    struct Handle {
        cusolverDnHandle_t h = nullptr;
        Handle() { cusolverDnCreate(&h); }
        ~Handle() { if (h) cusolverDnDestroy(h); }
    };
    static Handle handle;
    return handle.h;
}
/* *************************************************************** */
// Reusable device scratch (allocated once per OptimizeLts
struct Scratch {
    bool affine;
    double *sums = nullptr;                 // rigid: Sref[3], Swarped[3], Scross[9]
    double *A = nullptr, *U = nullptr, *V = nullptr, *S = nullptr, *svdWork = nullptr;  // rigid 3x3 SVD
    int svdLwork = 0; gesvdjInfo_t gesvdjParams = nullptr;
    // affine: reduce to the 4x4 normal system (like rigid's 3x3), then a tiny Cholesky solve - far
    // cheaper than DDgels on the tall NxN matrix. normalSums = MtM(10 unique) + MtW(12); An = 4x4
    // col-major; Bn = 4x3 RHS (overwritten with the solution by Dpotrs).
    double *normalSums = nullptr, *An = nullptr, *Bn = nullptr, *potrfWork = nullptr;
    int potrfLwork = 0;
    int *info = nullptr;
    mat44 *final = nullptr, *best = nullptr;        // device transforms

    Scratch(bool affineIn, int maxCount) : affine(affineIn) {
        cusolverDnHandle_t solver = GetSolver();
        NR_CUDA_SAFE_CALL(cudaMalloc(&info, sizeof(int)));
        NR_CUDA_SAFE_CALL(cudaMalloc(&final, sizeof(mat44)));
        NR_CUDA_SAFE_CALL(cudaMalloc(&best, sizeof(mat44)));
        if (!affine) {
            NR_CUDA_SAFE_CALL(cudaMalloc(&sums, 15 * sizeof(double)));
            NR_CUDA_SAFE_CALL(cudaMalloc(&A, 9 * sizeof(double)));
            NR_CUDA_SAFE_CALL(cudaMalloc(&U, 9 * sizeof(double)));
            NR_CUDA_SAFE_CALL(cudaMalloc(&V, 9 * sizeof(double)));
            NR_CUDA_SAFE_CALL(cudaMalloc(&S, 3 * sizeof(double)));
            cusolverDnCreateGesvdjInfo(&gesvdjParams);
            cusolverDnDgesvdj_bufferSize(solver, CUSOLVER_EIG_MODE_VECTOR, 0, 3, 3, A, 3, S, U, 3, V, 3, &svdLwork, gesvdjParams);
            NR_CUDA_SAFE_CALL(cudaMalloc(&svdWork, svdLwork * sizeof(double)));
        } else {
            (void)maxCount;
            NR_CUDA_SAFE_CALL(cudaMalloc(&normalSums, 22 * sizeof(double)));
            NR_CUDA_SAFE_CALL(cudaMalloc(&An, 16 * sizeof(double)));
            NR_CUDA_SAFE_CALL(cudaMalloc(&Bn, 12 * sizeof(double)));
            cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER, 4, An, 4, &potrfLwork);
            NR_CUDA_SAFE_CALL(cudaMalloc(&potrfWork, potrfLwork * sizeof(double)));
        }
    }
    ~Scratch() {
        cudaFree(info); cudaFree(final); cudaFree(best);
        if (!affine) { cudaFree(sums); cudaFree(A); cudaFree(U); cudaFree(V); cudaFree(S); cudaFree(svdWork);
                       if (gesvdjParams) cusolverDnDestroyGesvdjInfo(gesvdjParams); }
        else { cudaFree(normalSums); cudaFree(An); cudaFree(Bn); cudaFree(potrfWork); }
    }
};
/* *************************************************************** */
// --- device kernels (all single-block or single-thread; the linear systems are tiny) ---
// Σref, Σwarped, Σ(ref⊗warped) over `count` points -> sums[15]  (single block)
__global__ void ReduceRigidSums(const float* ref, const float* warped, int count, double* sums) {
    double loc[15]; for (int i = 0; i < 15; ++i) loc[i] = 0;
    for (int k = threadIdx.x; k < count; k += blockDim.x) {
        const double r0 = ref[k * 3], r1 = ref[k * 3 + 1], r2 = ref[k * 3 + 2];
        const double w0 = warped[k * 3], w1 = warped[k * 3 + 1], w2 = warped[k * 3 + 2];
        loc[0] += r0; loc[1] += r1; loc[2] += r2; loc[3] += w0; loc[4] += w1; loc[5] += w2;
        loc[6] += r0 * w0; loc[7] += r0 * w1; loc[8] += r0 * w2;
        loc[9] += r1 * w0; loc[10] += r1 * w1; loc[11] += r1 * w2;
        loc[12] += r2 * w0; loc[13] += r2 * w1; loc[14] += r2 * w2;
    }
    __shared__ double red[kReduceThreads];
    for (int i = 0; i < 15; ++i) {
        red[threadIdx.x] = loc[i]; __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s]; __syncthreads(); }
        if (threadIdx.x == 0) sums[i] = red[0];
        __syncthreads();
    }
}
// covariance A (3x3 col-major, A[col*3+row]) = Scross[row,col] - Sref[row]*Swarped[col]/count
__global__ void BuildCovariance(const double* sums, int count, double* A) {
    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b)
        A[b * 3 + a] = sums[6 + a * 3 + b] - sums[a] * sums[3 + b] / count;
}
// R = V*U^T (col-major U,V: element(row,col)=X[col*3+row]) + reflection guard, t = cw - R*cr -> out
__global__ void AssembleRigid(const double* sums, int count, const double* U, const double* V, mat44* out) {
    double Vl[9]; for (int i = 0; i < 9; ++i) Vl[i] = V[i];
    double R[3][3];
    auto rot = [&]() { for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) {
        double s = 0; for (int k = 0; k < 3; ++k) s += Vl[k * 3 + a] * U[k * 3 + b]; R[a][b] = s; } };
    rot();
    const double det = R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1])
                     - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0])
                     + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]);
    if (det < 0) { Vl[6] = -Vl[6]; Vl[7] = -Vl[7]; Vl[8] = -Vl[8]; rot(); }
    const double cr[3] = { sums[0] / count, sums[1] / count, sums[2] / count };
    const double cw[3] = { sums[3] / count, sums[4] / count, sums[5] / count };
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) out->m[i][j] = (i == j);
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) out->m[a][b] = (float)R[a][b];
        out->m[a][3] = (float)(cw[a] - (R[a][0] * cr[0] + R[a][1] * cr[1] + R[a][2] * cr[2]));
    }
}
// X (4x3 col-major, X[col*4+row]) -> affine mat44
__global__ void AssembleAffine(const double* X, mat44* out) {
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) out->m[i][j] = (i == j);
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 4; ++c) out->m[r][c] = (float)X[r * 4 + c];
}
// Affine normal equations: accumulate MtM (10 unique entries of the symmetric 4x4, M row = [x y z 1])
// and MtW (12 = 4 params x 3 coords) over `count` points -> s[22]  (single block).
__global__ void ReduceAffineNormal(const float* ref, const float* warped, int count, double* s) {
    double loc[22]; for (int i = 0; i < 22; ++i) loc[i] = 0;
    for (int k = threadIdx.x; k < count; k += blockDim.x) {
        const double x = ref[k * 3], y = ref[k * 3 + 1], z = ref[k * 3 + 2];
        const double wx = warped[k * 3], wy = warped[k * 3 + 1], wz = warped[k * 3 + 2];
        loc[0] += x * x; loc[1] += x * y; loc[2] += x * z; loc[3] += x;   // MtM row 0
        loc[4] += y * y; loc[5] += y * z; loc[6] += y;                    // MtM row 1
        loc[7] += z * z; loc[8] += z;                                     // MtM row 2
        loc[9] += 1;                                                      // MtM [3][3] = count
        loc[10] += x * wx; loc[11] += x * wy; loc[12] += x * wz;          // MtW rows (param x,y,z,1) x (wx,wy,wz)
        loc[13] += y * wx; loc[14] += y * wy; loc[15] += y * wz;
        loc[16] += z * wx; loc[17] += z * wy; loc[18] += z * wz;
        loc[19] += wx; loc[20] += wy; loc[21] += wz;
    }
    __shared__ double red[kReduceThreads];
    for (int i = 0; i < 22; ++i) {
        red[threadIdx.x] = loc[i]; __syncthreads();
        for (int st = blockDim.x / 2; st > 0; st >>= 1) { if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st]; __syncthreads(); }
        if (threadIdx.x == 0) s[i] = red[0];
        __syncthreads();
    }
}
// Fill the 4x4 normal matrix An (col-major, symmetric) and the 4x3 RHS Bn (col-major) from s[22].
__global__ void BuildAffineNormal(const double* s, double* An, double* Bn) {
    double m[4][4];
    m[0][0] = s[0]; m[0][1] = m[1][0] = s[1]; m[0][2] = m[2][0] = s[2]; m[0][3] = m[3][0] = s[3];
    m[1][1] = s[4]; m[1][2] = m[2][1] = s[5]; m[1][3] = m[3][1] = s[6];
    m[2][2] = s[7]; m[2][3] = m[3][2] = s[8];
    m[3][3] = s[9];
    for (int col = 0; col < 4; ++col) for (int row = 0; row < 4; ++row) An[col * 4 + row] = m[row][col];
    // Bn[coord*4 + param] = (MtW)[param][coord];  s layout: s[10 + param*3 + coord]
    for (int p = 0; p < 4; ++p) for (int c = 0; c < 3; ++c) Bn[c * 4 + p] = s[10 + p * 3 + c];
}
// residual[k] = ||final*ref_k - warped_k||   (Euclidean, matches CPU SquareDistance3d)
__global__ void ComputeResidual(const float* ref, const float* warped, int count, const mat44* final, float* residual) {
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < count; k += gridDim.x * blockDim.x) {
        float in[3] = { ref[k * 3], ref[k * 3 + 1], ref[k * 3 + 2] }, out[3];
        Mat44Mul<float>(*final, in, out);
        residual[k] = (float)SquareDistance3d(out, &warped[k * 3]);
    }
}
// gather the numToKeep lowest-residual points (idx sorted by residual)
__global__ void Gather(const float* ref, const float* warped, const int* idx, int numToKeep, float* keptRef, float* keptWarped) {
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < numToKeep; k += gridDim.x * blockDim.x) {
        const int s = idx[k] * 3;
        keptRef[k * 3] = ref[s]; keptRef[k * 3 + 1] = ref[s + 1]; keptRef[k * 3 + 2] = ref[s + 2];
        keptWarped[k * 3] = warped[s]; keptWarped[k * 3 + 1] = warped[s + 1]; keptWarped[k * 3 + 2] = warped[s + 2];
    }
}
/* *************************************************************** */
// Estimate into scratch.final (device), reading `count` interleaved points from ref/warped (device).
void Estimate(const float* ref, const float* warped, int count, Scratch& s) {
    constexpr int one = 1;
    if (!s.affine) {
        ReduceRigidSums<<<one, dim3(kReduceThreads)>>>(ref, warped, count, s.sums);
        BuildCovariance<<<one, one>>>(s.sums, count, s.A);
        cusolverDnDgesvdj(GetSolver(), CUSOLVER_EIG_MODE_VECTOR, 0, 3, 3, s.A, 3, s.S, s.U, 3, s.V, 3, s.svdWork, s.svdLwork, s.info, s.gesvdjParams);
        AssembleRigid<<<one, one>>>(s.sums, count, s.U, s.V, s.final);
    } else {
        ReduceAffineNormal<<<one, dim3(kReduceThreads)>>>(ref, warped, count, s.normalSums);
        BuildAffineNormal<<<one, one>>>(s.normalSums, s.An, s.Bn);
        // Cholesky-solve the tiny 4x4 SPD normal system for the 3 RHS (An X = Bn, X overwrites Bn).
        cusolverDnDpotrf(GetSolver(), CUBLAS_FILL_MODE_LOWER, 4, s.An, 4, s.potrfWork, s.potrfLwork, s.info);
        cusolverDnDpotrs(GetSolver(), CUBLAS_FILL_MODE_LOWER, 4, 3, s.An, 4, s.Bn, 4, s.info);
        AssembleAffine<<<one, one>>>(s.Bn, s.final);
    }
}
/* *************************************************************** */
} // unnamed namespace
/* *************************************************************** */
void OptimizeLts(_reg_blockMatchingParam *params,
                 mat44 *transformationMatrix,
                 const float *referencePositionCuda,
                 const float *warpedPositionCuda,
                 bool affine) {
    // Device-resident path for 3D (rigid + affine); 2D uses the (verified) CPU fallback for now.
    if (params->dim != 3) {
        const size_t count = static_cast<size_t>(params->activeBlockNumber) * params->dim;
        NR_CUDA_SAFE_CALL(cudaMemcpy(params->referencePosition, referencePositionCuda, count * sizeof(float), cudaMemcpyDeviceToHost));
        NR_CUDA_SAFE_CALL(cudaMemcpy(params->warpedPosition, warpedPositionCuda, count * sizeof(float), cudaMemcpyDeviceToHost));
        optimize(params, transformationMatrix, affine);
        return;
    }

    const int n = params->activeBlockNumber;

    // Finite working set (drop NaN = unmatched): warpedWork = current_matrix * warped, refWork = ref.
    thrust::device_vector<int> keep(n);
    auto keepEnd = thrust::copy_if(thrust::device, thrust::make_counting_iterator(0), thrust::make_counting_iterator(n), keep.begin(),
        [warpedPositionCuda] __device__(const int i) { return warpedPositionCuda[i * 3] == warpedPositionCuda[i * 3]; });
    const int m = static_cast<int>(keepEnd - keep.begin());   // one sync (buffer sizing), once per call

    thrust::device_vector<float> refWork(m * 3), warpedWork(m * 3), residual(m), keptRef, keptWarped;
    thrust::device_vector<int> idx(m);
    const int* keepPtr = thrust::raw_pointer_cast(keep.data());
    float* refWorkPtr = thrust::raw_pointer_cast(refWork.data());
    float* warpedWorkPtr = thrust::raw_pointer_cast(warpedWork.data());
    const mat44 current = *transformationMatrix;
    thrust::for_each_n(thrust::device, thrust::make_counting_iterator(0), m,
        [keepPtr, refWorkPtr, warpedWorkPtr, referencePositionCuda, warpedPositionCuda, current] __device__(const int k) {
            const int src = keepPtr[k] * 3;
            float in[3] = { warpedPositionCuda[src], warpedPositionCuda[src + 1], warpedPositionCuda[src + 2] }, out[3];
            Mat44Mul<float>(current, in, out);
            refWorkPtr[k * 3] = referencePositionCuda[src]; refWorkPtr[k * 3 + 1] = referencePositionCuda[src + 1]; refWorkPtr[k * 3 + 2] = referencePositionCuda[src + 2];
            warpedWorkPtr[k * 3] = out[0]; warpedWorkPtr[k * 3 + 1] = out[1]; warpedWorkPtr[k * 3 + 2] = out[2];
        });

    const int numToKeep = (int)(m * (params->percent_to_keep / 100.0f));
    keptRef.resize(numToKeep * 3); keptWarped.resize(numToKeep * 3);
    float* residualPtr = thrust::raw_pointer_cast(residual.data());
    int* idxPtr = thrust::raw_pointer_cast(idx.data());
    float* keptRefPtr = thrust::raw_pointer_cast(keptRef.data());
    float* keptWarpedPtr = thrust::raw_pointer_cast(keptWarped.data());

    Scratch s(affine, m);
    const int gridM = (m + kReduceThreads - 1) / kReduceThreads;

    Estimate(refWorkPtr, warpedWorkPtr, m, s);                 // initial estimate over all points
    NR_CUDA_SAFE_CALL(cudaMemcpy(s.best, s.final, sizeof(mat44), cudaMemcpyDeviceToDevice));  // s.best = lastTransformation
    double lastDistance = DBL_MAX;

    // Device-resident estimate; the host reads back ONLY the trimmed-residual scalar each iteration
    // (one 8-byte D->H) to assess convergence
    for (int count = 0; count < MAX_ITERATIONS; ++count) {
        ComputeResidual<<<gridM, dim3(kReduceThreads)>>>(refWorkPtr, warpedWorkPtr, m, s.final, residualPtr);
        thrust::sequence(thrust::device, idx.begin(), idx.end());
        thrust::sort_by_key(thrust::device, residual.begin(), residual.end(), idx.begin());
        const double distance = thrust::reduce(thrust::device, residual.begin(), residual.begin() + numToKeep, 0.0, thrust::plus<double>());
        if (distance > lastDistance || (lastDistance - distance) < TOLERANCE) {
            NR_CUDA_SAFE_CALL(cudaMemcpy(s.final, s.best, sizeof(mat44), cudaMemcpyDeviceToDevice));  // rollback to lastTransformation
            break;
        }
        lastDistance = distance;
        NR_CUDA_SAFE_CALL(cudaMemcpy(s.best, s.final, sizeof(mat44), cudaMemcpyDeviceToDevice));      // save lastTransformation
        Gather<<<(numToKeep + kReduceThreads - 1) / kReduceThreads, dim3(kReduceThreads)>>>(refWorkPtr, warpedWorkPtr, idxPtr, numToKeep, keptRefPtr, keptWarpedPtr);
        Estimate(keptRefPtr, keptWarpedPtr, numToKeep, s);
    }

    NR_CUDA_SAFE_CALL(cudaMemcpy(transformationMatrix, s.final, sizeof(mat44), cudaMemcpyDeviceToHost));
}
/* *************************************************************** */
} // namespace NiftyReg::Cuda
/* *************************************************************** */
