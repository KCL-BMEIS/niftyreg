// OpenCL is not supported here
#undef USE_OPENCL

// CPU vs CUDA benchmark for reg-lib Compute operations.
//
// This is a standalone executable
// It times an operation in-process through the Platform/Content/Compute abstraction, so it measures
// the compute path itself rather than the whole reg_resample binary (no file I/O).
//
// Currently it benchmarks 2D and 3D linear image resampling. It is written so that adding more
// functionality is a matter of writing another task factory and pushing its
// tasks into the list in main() - the timing / thread-sweep / printing harness (runTask) is generic
// over a BenchmarkTask and is never duplicated per operation.
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

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "reg_test_common.h"

using std::unique_ptr;
using std::vector;
using std::string;
using dim_t = NiftiImage::dim_t;

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
        omp_set_num_threads(1);
    }
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
            std::printf("  Benchmarks 2D and 3D linear resampling on CPU (1 and max threads) vs CUDA.\n");
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

    return 0;
}
