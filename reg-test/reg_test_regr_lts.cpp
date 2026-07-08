#include "reg_test_common.h"
#include "_reg_blockMatching.h"

#include "LtsKernel.h"
#include "CpuLtsKernel.h"
#include "CudaLtsKernel.h"

/**
 *  Regression test for the LTS estimator: CPU (CpuLtsKernel) vs CUDA (CudaLtsKernel).
 *
 */

namespace {
mat44 makeAffine(unsigned dim) {
    mat44 a; Mat44Eye(&a);
    if (dim == 3) {
        a.m[0][0] = 1.10f; a.m[0][1] = 0.05f; a.m[0][2] = 0.02f; a.m[0][3] = 5.f;
        a.m[1][0] = -0.03f; a.m[1][1] = 0.95f; a.m[1][2] = 0.04f; a.m[1][3] = -3.f;
        a.m[2][0] = 0.01f; a.m[2][1] = -0.02f; a.m[2][2] = 1.03f; a.m[2][3] = 7.f;
    } else {
        a.m[0][0] = 1.20f; a.m[0][1] = 0.10f; a.m[0][3] = 4.f;
        a.m[1][0] = -0.05f; a.m[1][1] = 0.90f; a.m[1][3] = -6.f;
    }
    return a;
}
mat44 makeRigid(unsigned dim) {
    mat44 r; Mat44Eye(&r);
    if (dim == 3) {
        const double rx = 0.20, ry = -0.15, rz = 0.30;
        const double cx = cos(rx), sx = sin(rx), cy = cos(ry), sy = sin(ry), cz = cos(rz), sz = sin(rz);
        r.m[0][0] = (float)(cz * cy); r.m[0][1] = (float)(cz * sy * sx - sz * cx); r.m[0][2] = (float)(cz * sy * cx + sz * sx);
        r.m[1][0] = (float)(sz * cy); r.m[1][1] = (float)(sz * sy * sx + cz * cx); r.m[1][2] = (float)(sz * sy * cx - cz * sx);
        r.m[2][0] = (float)(-sy);     r.m[2][1] = (float)(cy * sx);                r.m[2][2] = (float)(cy * cx);
        r.m[0][3] = 6.f; r.m[1][3] = -4.f; r.m[2][3] = 9.f;
    } else {
        const double th = 0.35;
        r.m[0][0] = (float)cos(th); r.m[0][1] = (float)-sin(th); r.m[0][3] = 4.f;
        r.m[1][0] = (float)sin(th); r.m[1][1] = (float)cos(th);  r.m[1][3] = -7.f;
    }
    return r;
}
// Build a _reg_blockMatchingParam (malloc'd, owned by the caller) with n correspondences
// warped = A * ref over a seeded cloud, with the last `nOutliers` corrupted by a gross offset.
_reg_blockMatchingParam* buildParams(unsigned dim, int n, int nOutliers, int keep, const mat44& a,
                                     std::mt19937& gen, float noiseSigma = 0.f) {
    auto* p = new _reg_blockMatchingParam();
    p->dim = dim;
    p->blockNumber[0] = 1; p->blockNumber[1] = 1; p->blockNumber[2] = dim == 3 ? 2u : 1u;
    p->activeBlockNumber = n;
    p->definedActiveBlockNumber = n;
    p->percent_to_keep = keep;
    const size_t len = static_cast<size_t>(n) * dim;
    p->referencePosition = static_cast<float*>(malloc(len * sizeof(float)));
    p->warpedPosition = static_cast<float*>(malloc(len * sizeof(float)));
    std::uniform_real_distribution<float> distr(-50.f, 50.f), gross(-100.f, 100.f);
    std::normal_distribution<float> noise(0.f, noiseSigma);
    for (int i = 0; i < n; ++i) {
        float in[3] = { 0, 0, 0 }, out[3] = { 0, 0, 0 };
        for (unsigned d = 0; d < dim; ++d) in[d] = distr(gen);
        if (dim == 3) Mat44Mul<float>(a, in, out); else Mat44Mul<float, false>(a, in, out);
        if (noiseSigma > 0.f)
            for (unsigned d = 0; d < dim; ++d) out[d] += noise(gen);
        if (i >= n - nOutliers)
            for (unsigned d = 0; d < dim; ++d) out[d] += gross(gen);
        for (unsigned d = 0; d < dim; ++d) {
            p->referencePosition[i * dim + d] = in[d];
            p->warpedPosition[i * dim + d] = out[d];
        }
    }
    return p;
}
// Max ||A*p - B*p|| (mm) over a fixed probe set spanning the coordinate range.
double maxProbeDisplacement(const mat44& a, const mat44& b, unsigned dim) {
    const float probes[9][3] = { {0,0,0},{40,40,40},{-40,40,40},{40,-40,40},{40,40,-40},
                                 {-40,-40,40},{-40,40,-40},{40,-40,-40},{-40,-40,-40} };
    double worst = 0;
    for (auto& pr : probes) {
        float in[3] = { pr[0], pr[1], pr[2] }, pa[3] = { 0,0,0 }, pb[3] = { 0,0,0 };
        if (dim == 3) { Mat44Mul<float>(a, in, pa); Mat44Mul<float>(b, in, pb); }
        else { Mat44Mul<float, false>(a, in, pa); Mat44Mul<float, false>(b, in, pb); }
        double s = 0; for (unsigned d = 0; d < dim; ++d) s += (double)(pa[d] - pb[d]) * (pa[d] - pb[d]);
        worst = std::max(worst, sqrt(s));
    }
    return worst;
}
} // namespace

class LtsTest {
protected:
    // name, CPU matrix, CUDA matrix, ground-truth A, dim, tolerance (mm), whether to check recovery
    using TestCase = std::tuple<std::string, unique_ptr<mat44>, unique_ptr<mat44>, mat44, unsigned, double, bool>;
    inline static vector<TestCase> testCases;

public:
    LtsTest() {
        if (!testCases.empty())
            return;
        std::mt19937 gen(0);
        constexpr int n = 250;

        // Scenarios chosen to isolate the sources of CPU-vs-GPU divergence:
        //  clean      - exact fit, no trim: both solve exactly -> ~bit-identical baseline.
        //  outliers   - exact inliers + gross outliers, trim active.
        //  noisy      - Gaussian-perturbed, keep 100% (NO trim): isolates the reduction-order +
        //               solve-algorithm difference (rigid gesvdj-vs-JacobiSVD; affine Cholesky-vs-QR).
        //  noisy-trim - Gaussian-perturbed, keep 60%: adds the trim-selection divergence on top.
        struct Scen { const char* suffix; int nOutliers; int keep; float noise; double tol; bool checkRecover; };
        const Scen scenarios[] = {
            { "clean",      0,             100, 0.0f, 0.05, true  },
            { "outliers",   (3 * n) / 10,  60,  0.0f, 0.05, true  },
            { "noisy",      0,             100, 0.5f, 1.00, false },
            { "noisy-trim", 0,             60,  0.5f, 1.00, false },
        };
        for (unsigned dim : { 2u, 3u }) {
            for (int ttype = 0; ttype <= 1; ++ttype) {   // 0 = rigid, 1 = affine
                const bool affine = ttype == 1;
                for (const Scen& scen : scenarios) {
                    const int nOutliers = scen.nOutliers;
                    const int keep = scen.keep;
                    const mat44 gt = affine ? makeAffine(dim) : makeRigid(dim);
                    _reg_blockMatchingParam* templateParams = buildParams(dim, n, nOutliers, keep, gt, gen, scen.noise);

                    vector<NiftiImage::dim_t> imgDim = dim == 3 ? vector<NiftiImage::dim_t>{ 8, 8, 8 }
                                                                : vector<NiftiImage::dim_t>{ 8, 8 };
                    NiftiImage refCpu = makeImage(imgDim), floCpu = makeImage(imgDim);
                    NiftiImage refCuda = makeImage(imgDim), floCuda = makeImage(imgDim);

                    unique_ptr<mat44> matCpu{ new mat44 }; Mat44Eye(matCpu.get());
                    unique_ptr<mat44> matCuda{ new mat44 }; Mat44Eye(matCuda.get());

                    unique_ptr<AladinContent> contentCpu{ new AladinContent(
                        refCpu, floCpu, nullptr, matCpu.get(), sizeof(float), 0, 0, 0) };
                    unique_ptr<AladinContent> contentCuda{ new CudaAladinContent(
                        refCuda, floCuda, nullptr, matCuda.get(), sizeof(float), 0, 0, 0) };
                    contentCpu->SetBlockMatchingParams(new _reg_blockMatchingParam(templateParams));
                    contentCuda->SetBlockMatchingParams(new _reg_blockMatchingParam(templateParams));

                    unique_ptr<LtsKernel> kernelCpu{ new CpuLtsKernel(contentCpu.get()) };
                    unique_ptr<LtsKernel> kernelCuda{ new CudaLtsKernel(contentCuda.get()) };
                    kernelCpu->Calculate(affine);
                    kernelCuda->Calculate(affine);

                    const std::string name = std::string(affine ? "affine " : "rigid ") +
                        (dim == 3 ? "3D " : "2D ") + scen.suffix;
                    testCases.push_back({ name, std::move(matCpu), std::move(matCuda), gt, dim, scen.tol, scen.checkRecover });
                    delete templateParams;
                }
            }
        }
    }
};

TEST_CASE_METHOD(LtsTest, "Regression LTS", "[regression]") {
    for (auto&& testCase : this->testCases) {
        auto&& [name, matCpu, matCuda, gt, dim, tol, checkRecover] = testCase;
        SECTION(name) {
            NR_COUT << "\n**************** Section " << name << " ****************" << std::endl;
            const double recoverCpu = maxProbeDisplacement(*matCpu, gt, dim);
            const double recoverCuda = maxProbeDisplacement(*matCuda, gt, dim);
            const double cpuVsCuda = maxProbeDisplacement(*matCpu, *matCuda, dim);
            double maxEntry = 0;
            for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j)
                maxEntry = std::max(maxEntry, std::fabs((double)matCpu->m[i][j] - (double)matCuda->m[i][j]));
            NR_COUT << "recover(CPU)=" << recoverCpu << " recover(CUDA)=" << recoverCuda
                    << " CPU-vs-CUDA=" << cpuVsCuda << " mm  maxEntryDiff=" << maxEntry << std::endl;
            if (checkRecover) {
                REQUIRE(recoverCpu < tol);    // CPU recovers the known transform
                REQUIRE(recoverCuda < tol);   // CUDA recovers the known transform (real GPU LTS test)
            }
            REQUIRE(cpuVsCuda < tol);         // CPU and CUDA agree (the discrepancy under study)
        }
    }
}
