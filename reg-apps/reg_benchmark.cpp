// OpenCL is not supported here
#undef USE_OPENCL

// Reach Content::SetDeformationField, which is only public when NR_TESTING is defined, so a
// synthetic deformation field can be injected directly into the content.
#define NR_TESTING

// CPU vs CUDA benchmark for reg-lib Compute operations.
//
// This is a standalone executable. It is deliberately independent of the test infrastructure
// (reg-test / Catch2) so it builds whenever USE_CUDA is enabled, regardless of BUILD_TESTING.
// It times an operation in-process through the Platform/Content/Compute abstraction, so it measures
// the compute path itself rather than the whole reg_resample binary (no file I/O).
//
// It benchmarks: 2D/3D linear image resampling, the LTS affine/rigid estimator, and the NMI
// similarity-value computation (CPU 1-thread vs N-threads vs CUDA). Adding more resampling-style
// functionality is a matter of writing another task factory and pushing its tasks into the list in
// main() - the timing / thread-sweep / printing harness (runTask) is generic over a BenchmarkTask
// and is never duplicated per operation; LTS and NMI have their own runXxxBench harnesses.
//
// For each task it reports, as means over the timed runs:
//   - CPU-1        : CPU, single thread
//   - CPU-N        : CPU, omp_get_max_threads() threads (only if compiled with OpenMP)
//   - GPU(kernel)  : the compute call only (device-synchronised, excludes the device->host copy)
//   - GPU(+copy)   : the compute call plus GetWarped() (device->host download of the result)
//   - GPU(app)     : the whole per-invocation GPU pipeline a one-shot reg_resample-style run pays -
//                    device allocation + host->device upload of floating/deformation-field + kernel +
//                    device->host download (but NOT file I/O, and NOT the one-time CUDA context init,
//                    which is reported separately at startup). The content is (re)built inside the
//                    timed region so the alloc+upload cost is measured; the synthetic input images are
//                    generated once up front so image generation is not counted.
// For CPU-1/CPU-N/GPU(kernel)/GPU(+copy) the content is built once and warm-up runs exclude that setup.
// It also does a quick CPU-vs-CUDA sanity check (flags a NaN mismatch between the platforms).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "_reg_localTrans.h"     // reg_createDeformationField
#include "_reg_blockMatching.h"  // _reg_blockMatchingParam (the LTS correspondences container)
#include "AladinContent.h"       // AladinContent (SetBlockMatchingParams, public under NR_TESTING)
#include "AladinContentCreator.h"// AladinContentCreator (build AladinContent via the platform)
#include "LtsKernel.h"           // LtsKernel::GetName() / Calculate(bool affine)
#include "CudaAladinContent.h"   // CudaAladinContent (device correspondence buffers)
#include "CudaLts.hpp"           // Cuda::OptimizeLts (the device LTS - benchmarked directly, since
                                 // CudaLtsKernel defaults to the CPU optimize)
#include "Platform.h"            // Platform / Content / Compute / Kernel / NiftiImage / Mat44Eye
#include "DefContentCreator.h"   // DefContentCreator / DefContent (the NMI value benchmark content)
#include "_reg_nmi.h"            // reg_nmi (ApproximatePw / SetTimePointWeight; NMI similarity value)

using std::unique_ptr;
using std::vector;
using std::string;
using dim_t = NiftiImage::dim_t;

/* ---------------------------------------------------------------------------------------------- */
/* Synthetic inputs (self-contained; this file must not depend on the test helpers)               */
/* ---------------------------------------------------------------------------------------------- */

// Install an identity sform (world coordinates == voxel coordinates) on an image.
static void setIdentitySform(NiftiImage& img) {
    mat44 eye;
    Mat44Eye(&eye);
    img->sform_code = 1;
    img->sto_xyz = eye;
    img->sto_ijk = eye;
    img->qform_code = 0;
}

// A float32 image with identity sform, filled with distinct fractional values (unique per voxel).
static NiftiImage makeImage(const std::vector<NiftiImage::dim_t>& dims) {
    NiftiImage img(dims, NIFTI_TYPE_FLOAT32);
    setIdentitySform(img);
    auto ptr = img.data();
    const size_t n = img.nVoxels();
    for (size_t i = 0; i < n; ++i)
        ptr[i] = static_cast<float>(i) + 0.5f;
    return img;
}

// An identity deformation field perturbed by a small seeded amount, so sampling coordinates are
// fractional (exercising the interpolation) rather than landing on integer voxels.
static NiftiImage makePerturbedField(const NiftiImage& reference, unsigned seed, float amplitude = 0.4f) {
    NiftiImage field;
    reg_createDeformationField<float>(field, reference); // identity, world coordinates
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> d(-amplitude, amplitude);
    auto ptr = field.data();
    const size_t n = field.nVoxels();
    for (size_t i = 0; i < n; ++i)
        ptr[i] = static_cast<float>(ptr[i]) + d(gen);
    return field;
}

// NaN-aware max absolute difference between two images: matched NaNs are ignored, a NaN on only one
// side is flagged. Handy for CPU-vs-CUDA comparisons.
struct DiffResult { double maxAbs = 0; bool nanMismatch = false; };
static DiffResult maxAbsDiff(const NiftiImage& a, const NiftiImage& b) {
    DiffResult r;
    const auto pa = a.data();
    const auto pb = b.data();
    const size_t n = std::min(a.nVoxels(), b.nVoxels());
    for (size_t i = 0; i < n; ++i) {
        const float va = static_cast<float>(pa[i]);
        const float vb = static_cast<float>(pb[i]);
        const bool na = std::isnan(va), nb = std::isnan(vb);
        if (na || nb) {
            if (na != nb) r.nanMismatch = true;
            continue;
        }
        r.maxAbs = std::max(r.maxAbs, static_cast<double>(std::fabs(va - vb)));
    }
    return r;
}

/* ---------------------------------------------------------------------------------------------- */
/* Timing + statistics                                                                            */
/* ---------------------------------------------------------------------------------------------- */

struct Stats {
    bool available = false;
    double meanMs = 0, medianMs = 0, minMs = 0, maxMs = 0, sdMs = 0;
};

// Run `body` `warmup` times (discarded) then `runs` times (timed) and summarise the wall-clock.
static Stats timeIt(int warmup, int runs, const std::function<void()>& body) {
    for (int i = 0; i < warmup; ++i) body();
    vector<double> t;
    t.reserve(runs);
    for (int i = 0; i < runs; ++i) {
        const auto start = std::chrono::steady_clock::now();
        body();
        const auto end = std::chrono::steady_clock::now();
        t.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    std::sort(t.begin(), t.end());
    Stats s;
    s.available = true;
    const int n = static_cast<int>(t.size());
    double sum = 0;
    for (double v : t) sum += v;
    s.meanMs = sum / n;
    s.minMs = t.front();
    s.maxMs = t.back();
    s.medianMs = (n % 2) ? t[n / 2] : 0.5 * (t[n / 2 - 1] + t[n / 2]);
    double ss = 0;
    for (double v : t) ss += (v - s.meanMs) * (v - s.meanMs);
    s.sdMs = (n > 1) ? std::sqrt(ss / (n - 1)) : 0;
    return s;
}

/* ---------------------------------------------------------------------------------------------- */
/* Benchmark task abstraction                                                                     */
/* ---------------------------------------------------------------------------------------------- */

// The generated inputs for one task, produced once up front so image generation is never counted in
// a timed region. Held by shared_ptr in the task so the (copyable) std::function closures can share it.
struct Inputs {
    NiftiImage reference, floating, field;
};

// A content+compute attached to a platform (the deformation field uploaded). The Content owns copies
// of the reference/floating and the field, so it is self-sufficient once built.
struct Built {
    unique_ptr<Content> content;
    unique_ptr<Compute> compute;
};

// One unit of benchmark work: an operation at a concrete problem size, buildable on any platform.
// To add a new operation, write a factory returning one of these; the harness does the rest.
struct BenchmarkTask {
    string op;    // e.g. "resample linear 3D"
    string size;  // e.g. "128 x 128 x 128"
    std::shared_ptr<Inputs> inputs;                     // generated once (not timed)
    std::function<Built(Platform&, Inputs&)> attach;    // device alloc + host->device upload + compute
    std::function<void(Compute&)> run;                  // the timed compute call
    // Download the result (device->host on CUDA) and return it by reference - returning by value
    // would deep-copy the whole image on the host each call, which for large images dominates the
    // measured copy-back and hides the true transfer cost.
    std::function<NiftiImage&(Content&)> fetch;
};

// Linear image-resampling task, 2D or 3D, at a cubic size `n`.
static BenchmarkTask makeResampleTask(bool is3d, int n) {
    constexpr int kLinear = 1;
    constexpr float kPad = 0.f;
    const vector<dim_t> dims = is3d ? vector<dim_t>{ n, n, n } : vector<dim_t>{ n, n };
    const unsigned seed = static_cast<unsigned>(n) * 2u + (is3d ? 1u : 0u);

    BenchmarkTask t;
    t.op = string("resample linear ") + (is3d ? "3D" : "2D");
    t.size = is3d ? (std::to_string(n) + " x " + std::to_string(n) + " x " + std::to_string(n))
                  : (std::to_string(n) + " x " + std::to_string(n));
    t.inputs = std::make_shared<Inputs>();
    t.inputs->reference = makeImage(dims);
    t.inputs->floating = makeImage(dims);
    t.inputs->field = makePerturbedField(t.inputs->reference, seed);
    // Build the Content/Compute for a platform. On CUDA this allocates the device buffers and uploads
    // the floating/reference (in Create) and the deformation field (in SetDeformationField) - i.e. the
    // per-invocation host->device cost. The field is copied because SetDeformationField consumes it.
    t.attach = [](Platform& platform, Inputs& in) -> Built {
        Built b;
        unique_ptr<ContentCreator> creator{ platform.CreateContentCreator(ContentType::Base) };
        b.content.reset(creator->Create(in.reference, in.floating));  // null mask -> all active
        NiftiImage fieldCopy(in.field);
        b.content->SetDeformationField(std::move(fieldCopy));
        b.compute.reset(platform.CreateCompute(*b.content));
        return b;
    };
    t.run = [](Compute& compute) { compute.ResampleImage(kLinear, kPad); };
    t.fetch = [](Content& content) -> NiftiImage& { return content.GetWarped(); };
    return t;
}

/* ---------------------------------------------------------------------------------------------- */
/* Harness: time one task on CPU (1 and max threads) and CUDA (kernel, +copyback), + sanity check */
/* ---------------------------------------------------------------------------------------------- */

static void runTask(const BenchmarkTask& task, Platform& cpu, Platform& cuda,
                    int warmup, int runs, int maxThreads) {
    Inputs& in = *task.inputs;

    // --- CPU (build once; time at 1 thread, then at max threads) ---
    Built cpuBuilt = task.attach(cpu, in);
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    const Stats cpu1 = timeIt(warmup, runs, [&] { task.run(*cpuBuilt.compute); });
    const NiftiImage cpuWarped = task.fetch(*cpuBuilt.content);  // deterministic; reference result

    Stats cpuN;
#ifdef _OPENMP
    if (maxThreads > 1) {
        omp_set_num_threads(maxThreads);
        cpuN = timeIt(warmup, runs, [&] { task.run(*cpuBuilt.compute); });
    }
    // Run the GPU-path host work under the same OMP policy as reg_aladin/-platf 1 (max threads),
    // mirroring the LTS harness. For resampling this does NOT change the GPU numbers - the resample
    // runs entirely on the device (Cuda::ResampleImage), with no CPU/OMP fallback - it only keeps the
    // thread policy consistent across the two harnesses. The speedup below is CPU-N / GPU(kernel).
    omp_set_num_threads(maxThreads);
#endif

    // --- CUDA: kernel-only, then compute + device->host copy (content built once) ---
    Built gpuBuilt = task.attach(cuda, in);
    const Stats gpuKernel = timeIt(warmup, runs, [&] {
        task.run(*gpuBuilt.compute);
        cudaDeviceSynchronize();  // the compute launch may be async; include full kernel time
    });
    const Stats gpuCopy = timeIt(warmup, runs, [&] {
        task.run(*gpuBuilt.compute);
        task.fetch(*gpuBuilt.content);  // GetWarped() downloads (and synchronises)
    });
    const NiftiImage gpuWarped = task.fetch(*gpuBuilt.content);

    // --- CUDA: whole per-invocation pipeline (alloc + upload + kernel + download), content rebuilt
    // inside the timed region. Excludes the one-time context init (already up) and file I/O. ---
    const Stats gpuApp = timeIt(warmup, runs, [&] {
        Built b = task.attach(cuda, in);
        task.run(*b.compute);
        cudaDeviceSynchronize();
        task.fetch(*b.content);
    });

    // --- sanity check ---
    const DiffResult diff = maxAbsDiff(cpuWarped, gpuWarped);

    // --- print one row ---
    auto ms = [](const Stats& s) { return s.available ? s.meanMs : std::nan(""); };
    const double cpuNms = cpuN.available ? cpuN.meanMs : cpu1.meanMs;  // fall back to 1-thread
    const double speedup = gpuKernel.meanMs > 0 ? cpuNms / gpuKernel.meanMs : 0;

    std::printf("%-20s %-18s ", task.op.c_str(), task.size.c_str());
    std::printf("%10.3f ", ms(cpu1));
    if (cpuN.available) std::printf("%10.3f ", ms(cpuN));
    else                std::printf("%10s ", "n/a");
    std::printf("%12.4f %12.4f %12.4f %10.1fx\n", ms(gpuKernel), ms(gpuCopy), ms(gpuApp), speedup);
    // Correctness is covered by the regression tests; here only flag a gross CPU/CUDA divergence.
    if (diff.nanMismatch)
        std::printf("  ** WARNING: CPU/CUDA NaN mismatch for %s %s **\n", task.op.c_str(), task.size.c_str());
    std::fflush(stdout);
}

/* ---------------------------------------------------------------------------------------------- */
/* LTS (Least Trimmed Squares) benchmark                                                          */
/* ---------------------------------------------------------------------------------------------- */
// The LTS affine/rigid estimator (`optimize`, reg-lib/cpu/_reg_blockMatching.cpp) is the reg_aladin
// CPU bottleneck and the target of the GPU-LTS port. It is not a Compute operation, so it does NOT
// reuse the BenchmarkTask/runTask harness above; it goes through the AladinContent + LtsKernel
// kernel abstraction. The CUDA arm (CudaLtsKernel::Calculate) today only round-trips the
// correspondence positions device<->host and then runs the SAME CPU optimize() (no device solve
// yet) - it is wired here as scaffolding so that a future GPU LTS is timed by this identical harness.
// optimize() reads only blockNumber[2]/dim/activeBlockNumber/definedActiveBlockNumber/percent_to_keep
// and the referencePosition/warpedPosition arrays, so we inject hand-built correspondences directly
// (warped[i] = A*reference[i] for a known A) and skip real block matching entirely.

// A known, well-conditioned affine (rotation * anisotropic scale/shear + translation).
static mat44 ltsAffine3d() {
    mat44 a; Mat44Eye(&a);
    a.m[0][0] = 1.10f; a.m[0][1] = 0.05f; a.m[0][2] = 0.02f; a.m[0][3] = 5.f;
    a.m[1][0] = -0.03f; a.m[1][1] = 0.95f; a.m[1][2] = 0.04f; a.m[1][3] = -3.f;
    a.m[2][0] = 0.01f; a.m[2][1] = -0.02f; a.m[2][2] = 1.03f; a.m[2][3] = 7.f;
    return a;
}
// A known rigid transform (R = Rz*Ry*Rx + translation).
static mat44 ltsRigid3d() {
    const double rx = 0.20, ry = -0.15, rz = 0.30;
    const double cx = cos(rx), sx = sin(rx), cy = cos(ry), sy = sin(ry), cz = cos(rz), sz = sin(rz);
    mat44 r; Mat44Eye(&r);
    r.m[0][0] = (float)(cz * cy); r.m[0][1] = (float)(cz * sy * sx - sz * cx); r.m[0][2] = (float)(cz * sy * cx + sz * sx);
    r.m[1][0] = (float)(sz * cy); r.m[1][1] = (float)(sz * sy * sx + cz * cx); r.m[1][2] = (float)(sz * sy * cx - cz * sx);
    r.m[2][0] = (float)(-sy);     r.m[2][1] = (float)(cy * sx);                r.m[2][2] = (float)(cy * cx);
    r.m[0][3] = 6.f; r.m[1][3] = -4.f; r.m[2][3] = 9.f;
    return r;
}

// Build a fresh _reg_blockMatchingParam with n 3D correspondences warped[i] = A*reference[i].
// Returned by `new`; whichever AladinContent it is handed to owns it and deletes it (freeing the
// malloc'd arrays), so build a separate one per platform - never share across two contents.
static _reg_blockMatchingParam* makeLtsParams(int n, int keep, const mat44& a, unsigned seed) {
    auto* p = new _reg_blockMatchingParam();
    p->dim = 3;
    p->blockNumber[0] = 1; p->blockNumber[1] = 1; p->blockNumber[2] = 2;  // [2]!=1 -> 3D path
    p->activeBlockNumber = n;
    p->definedActiveBlockNumber = n;
    p->percent_to_keep = keep;
    const size_t len = static_cast<size_t>(n) * 3;
    p->referencePosition = static_cast<float*>(malloc(len * sizeof(float)));
    p->warpedPosition = static_cast<float*>(malloc(len * sizeof(float)));
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> d(-100.f, 100.f);
    for (int i = 0; i < n; ++i) {
        float in[3], out[3];
        for (int c = 0; c < 3; ++c) in[c] = d(gen);
        Mat44Mul<float>(a, in, out);
        for (int c = 0; c < 3; ++c) { p->referencePosition[i * 3 + c] = in[c]; p->warpedPosition[i * 3 + c] = out[c]; }
    }
    return p;
}

// A content+kernel attached to a platform. Heap-allocated (returned by unique_ptr) so the mat44 and
// images the Content points at keep stable addresses. `mat` holds the fitted transform after Calculate.
struct LtsBuilt {
    NiftiImage ref, flo;
    unique_ptr<mat44> mat;
    unique_ptr<AladinContent> content;
    unique_ptr<Kernel> kernel;
};

// Build an AladinContent on `platform` with block-matching args 0,0,0 (no auto block matching),
// inject the hand-built correspondences (uploads to device on CUDA), and create the LTS kernel.
// SetBlockMatchingParams MUST precede CreateKernel (the CUDA kernel ctor reads the params back).
static unique_ptr<LtsBuilt> buildLts(Platform& platform, const vector<dim_t>& dummyDim,
                                     _reg_blockMatchingParam* bmp) {
    auto b = std::make_unique<LtsBuilt>();
    b->ref = makeImage(dummyDim);
    b->flo = makeImage(dummyDim);
    b->mat.reset(new mat44);
    Mat44Eye(b->mat.get());
    unique_ptr<ContentCreator> creator{ platform.CreateContentCreator(ContentType::Aladin) };
    auto* aladinCreator = dynamic_cast<AladinContentCreator*>(creator.get());
    b->content.reset(aladinCreator->Create(b->ref, b->flo, nullptr, b->mat.get(), sizeof(float), 0, 0, 0));
    b->content->SetBlockMatchingParams(bmp);  // uploads positions to device on CUDA
    b->kernel.reset(platform.CreateKernel(LtsKernel::GetName(), b->content.get()));
    return b;
}

// Recovery accuracy: max ||rec*p - gt*p|| (mm) over a fixed probe set spanning the correspondence
// coordinate range. Warped points are gt*ref, so a correct fit gives ~0; a candidate that degrades
// the solve shows up here. Also used to report the accuracy delta between optimisation candidates.
static double ltsAccuracyMm(const mat44& rec, const mat44& gt) {
    const float probes[9][3] = {
        { 0, 0, 0 }, { 80, 80, 80 }, { -80, 80, 80 }, { 80, -80, 80 }, { 80, 80, -80 },
        { -80, -80, 80 }, { -80, 80, -80 }, { 80, -80, -80 }, { -80, -80, -80 } };
    double worst = 0;
    for (auto& p : probes) {
        float in[3] = { p[0], p[1], p[2] }, ra[3], rb[3];
        Mat44Mul<float>(rec, in, ra);
        Mat44Mul<float>(gt, in, rb);
        double s = 0;
        for (int d = 0; d < 3; ++d) s += static_cast<double>(ra[d] - rb[d]) * static_cast<double>(ra[d] - rb[d]);
        worst = std::max(worst, std::sqrt(s));
    }
    return worst;
}

// Time one LTS problem (3D, affine or rigid, n correspondences) on CPU (1 and max threads) and CUDA.
static void runLtsBench(const char* type, int n, bool affine, Platform& cpu, Platform& cuda,
                        int warmup, int runs, int maxThreads) {
    constexpr int kKeep = 50;  // reg_aladin default LTS keep fraction (-pi)
    const vector<dim_t> dummy{ 8, 8, 8 };
    const mat44 a = affine ? ltsAffine3d() : ltsRigid3d();

    // --- CPU: build once, time at 1 thread then max threads ---
    unique_ptr<LtsBuilt> cpuBuilt = buildLts(cpu, dummy, makeLtsParams(n, kKeep, a, static_cast<unsigned>(n)));
    LtsKernel* cpuKernel = cpuBuilt->kernel->castTo<LtsKernel>();
    const auto cpuBody = [&] { Mat44Eye(cpuBuilt->mat.get()); cpuKernel->Calculate(affine); };
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    const Stats cpu1 = timeIt(warmup, runs, cpuBody);
    const mat44 cpuMat = *cpuBuilt->mat;  // recovered transform, for the sanity check

    Stats cpuN;
#ifdef _OPENMP
    if (maxThreads > 1) {
        omp_set_num_threads(maxThreads);
        cpuN = timeIt(warmup, runs, cpuBody);
        omp_set_num_threads(1);
    }
#endif

    // --- CUDA: the actual device LTS (Cuda::OptimizeLts), called directly - CudaLtsKernel defaults
    // to the CPU optimize, so going through the kernel would measure the CPU. GPU(call) = OptimizeLts
    // on a prebuilt content (positions already on device); GPU(app) = build content + upload +
    // OptimizeLts per call. ---
    unique_ptr<LtsBuilt> gpuBuilt = buildLts(cuda, dummy, makeLtsParams(n, kKeep, a, static_cast<unsigned>(n)));
    auto* gpuContent = dynamic_cast<CudaAladinContent*>(gpuBuilt->content.get());
    _reg_blockMatchingParam* gpuParams = gpuContent->AladinContent::GetBlockMatchingParams();
    const float* gpuRefPos = gpuContent->GetReferencePositionCuda();
    const float* gpuWarpedPos = gpuContent->GetWarpedPositionCuda();
    const Stats gpuCall = timeIt(warmup, runs, [&] {
        Mat44Eye(gpuBuilt->mat.get());
        NiftyReg::Cuda::OptimizeLts(gpuParams, gpuBuilt->mat.get(), gpuRefPos, gpuWarpedPos, affine);
    });
    const mat44 gpuMat = *gpuBuilt->mat;

    const Stats gpuApp = timeIt(warmup, runs, [&] {
        unique_ptr<LtsBuilt> b = buildLts(cuda, dummy, makeLtsParams(n, kKeep, a, static_cast<unsigned>(n)));
        auto* c = dynamic_cast<CudaAladinContent*>(b->content.get());
        Mat44Eye(b->mat.get());
        NiftyReg::Cuda::OptimizeLts(c->AladinContent::GetBlockMatchingParams(), b->mat.get(),
                                    c->GetReferencePositionCuda(), c->GetWarpedPositionCuda(), affine);
    });

    // --- sanity: CPU-optimize vs device-LTS recovered matrix on identical inputs (clean data) ---
    double maxDiff = 0;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            maxDiff = std::max(maxDiff, static_cast<double>(std::fabs(cpuMat.m[i][j] - gpuMat.m[i][j])));

    // --- recovery accuracy (max probe displacement vs the ground-truth transform, mm) ---
    const double accMm = ltsAccuracyMm(cpuMat, a);

    // --- print one row ---
    const double cpuNms = cpuN.available ? cpuN.meanMs : cpu1.meanMs;
    const double gpuSpeedup = gpuCall.meanMs > 0 ? cpuNms / gpuCall.meanMs : 0;
    std::printf("%-7s %-4s %9d %10.3f ", type, "3D", n, cpu1.meanMs);
    if (cpuN.available) std::printf("%10.3f ", cpuN.meanMs);
    else                std::printf("%10s ", "n/a");
    std::printf("%12.4f %12.4f %10.2fx %11.2e\n", gpuCall.meanMs, gpuApp.meanMs, gpuSpeedup, accMm);
    if (maxDiff > 1e-3)
        std::printf("  ** WARNING: CPU/CUDA LTS matrix diff %.3g for %s N=%d **\n", maxDiff, type, n);
    std::fflush(stdout);
}

static void runLtsBenchmarks(Platform& cpu, Platform& cuda, int warmup, int runs, int maxThreads) {
    std::printf("\nLTS estimation (optimize() via LtsKernel) - affine/rigid point-set fit, keep=50%% (reg_aladin -pi default)\n");
    std::printf("NOTE: CUDA LTS currently round-trips the correspondence positions device<->host then runs the CPU\n");
    std::printf("      optimize() (no device solve yet); the GPU columns are scaffolding to compare a future GPU LTS.\n");
    std::printf("GPU(call) = LtsKernel::Calculate on a prebuilt content; GPU(app) = build content + upload + Calculate.\n");
    std::printf("The GPU CPU-fallback runs at max threads (as reg_aladin -platf 1 does), so GPU(call) tracks CPU-N today.\n");
    std::printf("GPUspeedup = CPU-N / GPU(call) (~1 now: the GPU path is the CPU-N solve + a device round-trip).\n");
    std::printf("recov(mm) = max probe displacement of the recovered vs ground-truth transform (fit accuracy).\n");
    std::printf("%-7s %-4s %9s %10s %10s %12s %12s %11s %11s\n",
                "type", "dim", "points", "CPU-1", "CPU-N", "GPU(call)", "GPU(app)", "GPUspeedup", "recov(mm)");
    std::printf("%-7s %-4s %9s %10s %10s %12s %12s %11s %11s\n",
                "----", "---", "------", "-----", "-----", "---------", "--------", "----------", "---------");
    const vector<int> sizes = { 1000, 10000, 100000, 200000 };
    for (int n : sizes) runLtsBench("affine", n, /*affine*/true, cpu, cuda, warmup, runs, maxThreads);
    for (int n : sizes) runLtsBench("rigid", n, /*affine*/false, cpu, cuda, warmup, runs, maxThreads);
}

/* ---------------------------------------------------------------------------------------------- */
/* NMI similarity-value benchmark                                                                 */
/* ---------------------------------------------------------------------------------------------- */
// Times reg_nmi::GetSimilarityMeasureValue() - the joint-histogram fill + entropy computation that
// runs once per optimisation iteration (and again inside every NMI gradient step). The dominant cost
// is the per-voxel histogram fill; the approximation path (the default, integer binning) is the one
// parallelised with OpenMP. We pin ApproximatePw() on both platforms so CPU-1, CPU-N and GPU all
// compute the same thing and the comparison is fair.
//
// Setup mirrors reg_test_nmi.cpp / reg_test_regr_measure.cpp and goes entirely through the base
// Platform / DefContentCreator / MeasureCreator virtuals - the CUDA platform produces a
// CudaDefContent + reg_nmi_gpu automatically, so no CUDA-specific types are referenced here.
// SetWarped() uploads the warped image to the device on CUDA, so no resample step is needed.

// A float32 image filled with random integer intensities in [2, 65] (the NMI bin range for the
// default 68 bins: reg_nmi rescales the reference/floating to [2, bin-3]; the warped is used as-is,
// so it must already sit in that range). At least one voxel hits each end so the reference rescale
// is stable. Integer values keep the approximation-path histogram fill in its intended regime.
static NiftiImage makeNmiImage(const std::vector<NiftiImage::dim_t>& dims, unsigned seed) {
    NiftiImage img(dims, NIFTI_TYPE_FLOAT32);
    setIdentitySform(img);
    auto ptr = img.data();
    const size_t n = img.nVoxels();
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> d(2, 65);
    if (n > 0) ptr[0] = 2.f;
    if (n > 1) ptr[1] = 65.f;
    for (size_t i = 2; i < n; ++i)
        ptr[i] = static_cast<float>(d(gen));
    return img;
}

// A DefContent + NMI measure attached to a platform, warped image uploaded and the measure
// initialised. Everything is held so the timed GetSimilarityMeasureValue() call stays valid.
struct NmiBuilt {
    NiftiImage ref, flo, war;
    unique_ptr<Content> content;
    unique_ptr<MeasureCreator> measureCreator;
    unique_ptr<reg_measure> measure;  // a reg_nmi (reg_nmi_gpu on CUDA), driven via the base pointer
};

static NmiBuilt buildNmi(Platform& platform, const NiftiImage& reference, const NiftiImage& warped) {
    NmiBuilt b;
    b.ref = reference;  // fresh per-platform copies (InitialiseMeasure rescales the reference in place)
    b.flo = reference;  // floating is required by Create but unused by the value (warped drives it)
    b.war = warped;
    unique_ptr<ContentCreator> cc{ platform.CreateContentCreator(ContentType::Def) };
    auto* defCreator = dynamic_cast<DefContentCreator*>(cc.get());
    b.content.reset(defCreator->Create(b.ref, b.flo));   // CudaDefContent on the CUDA platform
    b.content->SetWarped(NiftiImage(b.war));              // uploads host->device on CUDA
    b.measureCreator.reset(platform.CreateMeasureCreator());
    b.measure.reset(b.measureCreator->Create(MeasureType::Nmi));  // reg_nmi_gpu on the CUDA platform
    auto* nmi = dynamic_cast<reg_nmi*>(b.measure.get());
    for (int t = 0; t < b.ref->nt; ++t)
        nmi->SetTimePointWeight(t, 1.0);
    nmi->ApproximatePw();  // pin the (default) approximation path so all three columns match
    auto* defContent = dynamic_cast<DefContent*>(b.content.get());
    b.measureCreator->Initialise(*b.measure, *defContent);
    return b;
}

// Time one NMI value computation (2D or 3D, cubic size n) on CPU (1 and max threads) and CUDA.
static void runNmiBench(bool is3d, int n, Platform& cpu, Platform& cuda,
                        int warmup, int runs, int maxThreads) {
    const vector<dim_t> dims = is3d ? vector<dim_t>{ n, n, n } : vector<dim_t>{ n, n };
    const unsigned seed = static_cast<unsigned>(n) * 2u + (is3d ? 1u : 0u);
    const NiftiImage reference = makeNmiImage(dims, seed);
    const NiftiImage warped = makeNmiImage(dims, seed + 100u);
    const size_t voxels = reference.nVoxels();

    // --- CPU (build once; time at 1 thread, then at max threads) ---
    NmiBuilt cpuBuilt = buildNmi(cpu, reference, warped);
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    const Stats cpu1 = timeIt(warmup, runs, [&] { cpuBuilt.measure->GetSimilarityMeasureValue(); });
    const double cpuVal = cpuBuilt.measure->GetSimilarityMeasureValue();

    Stats cpuN;
#ifdef _OPENMP
    if (maxThreads > 1) {
        omp_set_num_threads(maxThreads);
        cpuN = timeIt(warmup, runs, [&] { cpuBuilt.measure->GetSimilarityMeasureValue(); });
    }
    omp_set_num_threads(maxThreads);  // keep the GPU-path host work under the reg_aladin -platf 1 policy
#endif

    // --- CUDA (build once; the value call runs the device kernel and reduces to the host) ---
    NmiBuilt gpuBuilt = buildNmi(cuda, reference, warped);
    const Stats gpu = timeIt(warmup, runs, [&] {
        gpuBuilt.measure->GetSimilarityMeasureValue();
        cudaDeviceSynchronize();
    });
    const double gpuVal = gpuBuilt.measure->GetSimilarityMeasureValue();

    // --- print one row ---
    const double cpuNms = cpuN.available ? cpuN.meanMs : cpu1.meanMs;
    const double ompSpeedup = cpuN.available && cpuN.meanMs > 0 ? cpu1.meanMs / cpuN.meanMs : 0;
    const double gpuSpeedup = gpu.meanMs > 0 ? cpuNms / gpu.meanMs : 0;
    const std::string size = is3d ? (std::to_string(n) + "^3") : (std::to_string(n) + "^2");
    std::printf("%-8s %-10s %12zu %10.3f ", is3d ? "NMI 3D" : "NMI 2D", size.c_str(), voxels, cpu1.meanMs);
    if (cpuN.available) std::printf("%10.3f %8.2fx ", cpuN.meanMs, ompSpeedup);
    else                std::printf("%10s %9s ", "n/a", "n/a");
    std::printf("%12.4f %10.2fx\n", gpu.meanMs, gpuSpeedup);
    if (std::fabs(cpuVal - gpuVal) > 1e-4)
        std::printf("  ** WARNING: CPU/CUDA NMI value diff %.3g (cpu=%.6f cuda=%.6f) **\n",
                    std::fabs(cpuVal - gpuVal), cpuVal, gpuVal);
    std::fflush(stdout);
}

static void runNmiBenchmarks(Platform& cpu, Platform& cuda, int warmup, int runs, int maxThreads) {
    std::printf("\nNMI similarity value (reg_nmi::GetSimilarityMeasureValue, approximation path pinned)\n");
    std::printf("The joint-histogram fill (per-voxel) is the OpenMP-parallelised hot loop; entropies are O(bins).\n");
    std::printf("OMPspeedup = CPU-1 / CPU-N; GPUspeedup = CPU-N / GPU. GPU includes device sync (thrust reduce).\n");
    std::printf("%-8s %-10s %12s %10s %10s %8s %12s %10s\n",
                "measure", "size", "voxels", "CPU-1", "CPU-N", "OMPspd", "GPU", "GPUspd");
    std::printf("%-8s %-10s %12s %10s %10s %8s %12s %10s\n",
                "-------", "----", "------", "-----", "-----", "------", "---", "------");
    const vector<int> sizes3d = { 64, 128, 256 };
    const vector<int> sizes2d = { 512, 2048, 4096 };
    for (int n : sizes3d) runNmiBench(/*is3d*/true, n, cpu, cuda, warmup, runs, maxThreads);
    for (int n : sizes2d) runNmiBench(/*is3d*/false, n, cpu, cuda, warmup, runs, maxThreads);
}

/* ---------------------------------------------------------------------------------------------- */

int main(int argc, char** argv) {
    int runs = 10, warmup = 2;
    unsigned gpuIdx = 999;  // 999 = automatic (best card), matching reg_resample
    // 3D and 2D cubic sizes to sweep (kept moderate; edit here to change the sweep).
    vector<int> sizes3d = {32, 128, 256};
    vector<int> sizes2d = {128, 1024, 4096};

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) runs = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-w") && i + 1 < argc) warmup = std::atoi(argv[++i]);
        else if ((!std::strcmp(argv[i], "-gpuid") || !std::strcmp(argv[i], "--gpuid")) && i + 1 < argc)
            gpuIdx = static_cast<unsigned>(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) {
            std::printf("Usage: %s [-n timed_runs] [-w warmup_runs] [-gpuid <int>]\n", argv[0]);
            std::printf("  Benchmarks 2D and 3D linear resampling on CPU (1 and max threads) vs CUDA,\n");
            std::printf("  then the LTS affine/rigid estimator (CPU thread-scaling + CUDA-path scaffolding).\n");
            std::printf("  -gpuid <int>  Id of the GPU card to use [automatic]\n");
            return 0;
        }
    }

    int maxThreads = 1;
#ifdef _OPENMP
    maxThreads = omp_get_max_threads();
#endif

    // One-time CUDA context initialisation - the fixed price a fresh process pays before any GPU work.
    // Measured as the first CUDA runtime call (creates the primary context on the current device).
    // A one-shot GPU reg_resample-style run pays this once, on top of the per-invocation GPU(app) work.
    const auto initStart = std::chrono::steady_clock::now();
    cudaFree(0);
    const double cudaInitMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - initStart).count();

    if (!Platform::IsCudaEnabled()) {
        std::fprintf(stderr, "CUDA is not available on this platform; nothing to benchmark.\n");
        return 1;
    }

    Platform cpu(PlatformType::Cpu);
    Platform cuda(PlatformType::Cuda);
    cuda.SetGpuIdx(gpuIdx);

    std::printf("reg-lib CPU vs CUDA resampling benchmark\n");
    std::printf("timed runs: %d (+ %d warm-up) per configuration\n", runs, warmup);
    if (gpuIdx == 999) std::printf("GPU: automatic (best card)\n");
    else               std::printf("GPU: id %u\n", gpuIdx);
#ifdef _OPENMP
    std::printf("OpenMP: yes, CPU-N uses %d threads\n", maxThreads);
#else
    std::printf("OpenMP: no (CPU-N column shown as n/a)\n");
#endif
    std::printf("CUDA context init (one-time per process): %.1f ms\n", cudaInitMs);
    std::printf("all timings are means in ms. GPU(kernel) = kernel only; GPU(+copy) adds the device->host\n");
    std::printf("download; GPU(app) = alloc + upload + kernel + download per call. A one-shot GPU run's\n");
    std::printf("latency ~ %.1f ms (init) + GPU(app).\n\n", cudaInitMs);
    std::printf("%-20s %-18s %10s %10s %12s %12s %12s %11s\n",
                "operation", "size", "CPU-1", "CPU-N", "GPU(kernel)", "GPU(+copy)", "GPU(app)", "speedup");
    std::printf("%-20s %-18s %10s %10s %12s %12s %12s %11s\n",
                "---------", "----", "-----", "-----", "-----------", "----------", "--------", "-------");

    // The functionality list. Extend by pushing more tasks (e.g. gradient, spline) here; runTask
    // handles any BenchmarkTask uniformly.
    vector<BenchmarkTask> tasks;
    for (int n : sizes3d) tasks.push_back(makeResampleTask(/*is3d*/true, n));
    for (int n : sizes2d) tasks.push_back(makeResampleTask(/*is3d*/false, n));

    for (const BenchmarkTask& task : tasks)
        runTask(task, cpu, cuda, warmup, runs, maxThreads);

    // LTS (reg_aladin affine/rigid fit): CPU thread-scaling + the CUDA-path baseline (scaffolding).
    runLtsBenchmarks(cpu, cuda, warmup, runs, maxThreads);

    // NMI similarity value: CPU 1-thread vs N-threads (the OpenMP win) vs CUDA.
    runNmiBenchmarks(cpu, cuda, warmup, runs, maxThreads);

    return 0;
}
