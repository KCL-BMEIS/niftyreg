#include "reg_test_common.h"
#include "_reg_blockMatching.h"

/**
 *  Analytical unit test for the Least Trimmed Squares (LTS) affine/rigid estimator
 *
 *  This test drives `optimize` directly with hand-built correspondences and asserts
 *  the recovered transform against a known ground-truth matrix:
 *    - `optimize` fits `final : reference -> warped` (with an identity input matrix).
 *    - So building `warped[i] = A * reference[i]` must recover `final == A`.
 *  It covers exact recovery, rigid orthonormality, outlier trimming (and that trimming
 *  is actually necessary), NaN correspondences, the minimum-correspondence counts, small
 *  noise, and determinism.
 */

namespace {

// ---- ground-truth matrix builders -----------------------------------------
mat44 makeAffine3d() {
    mat44 a; Mat44Eye(&a);
    a.m[0][0] = 1.10f; a.m[0][1] = 0.05f; a.m[0][2] = 0.02f; a.m[0][3] = 5.0f;
    a.m[1][0] = -0.03f; a.m[1][1] = 0.95f; a.m[1][2] = 0.04f; a.m[1][3] = -3.0f;
    a.m[2][0] = 0.01f; a.m[2][1] = -0.02f; a.m[2][2] = 1.03f; a.m[2][3] = 7.0f;
    return a;
}
mat44 makeAffine2d() {
    mat44 a; Mat44Eye(&a);
    a.m[0][0] = 1.20f; a.m[0][1] = 0.10f; a.m[0][3] = 4.0f;
    a.m[1][0] = -0.05f; a.m[1][1] = 0.90f; a.m[1][3] = -6.0f;
    return a;
}
mat44 makeRigid3d(double rx, double ry, double rz, double tx, double ty, double tz) {
    const double cx = cos(rx), sx = sin(rx), cy = cos(ry), sy = sin(ry), cz = cos(rz), sz = sin(rz);
    // R = Rz * Ry * Rx
    mat44 r; Mat44Eye(&r);
    r.m[0][0] = (float)(cz * cy);
    r.m[0][1] = (float)(cz * sy * sx - sz * cx);
    r.m[0][2] = (float)(cz * sy * cx + sz * sx);
    r.m[1][0] = (float)(sz * cy);
    r.m[1][1] = (float)(sz * sy * sx + cz * cx);
    r.m[1][2] = (float)(sz * sy * cx - cz * sx);
    r.m[2][0] = (float)(-sy);
    r.m[2][1] = (float)(cy * sx);
    r.m[2][2] = (float)(cy * cx);
    r.m[0][3] = (float)tx; r.m[1][3] = (float)ty; r.m[2][3] = (float)tz;
    return r;
}
mat44 makeRigid2d(double theta, double tx, double ty) {
    mat44 r; Mat44Eye(&r);
    r.m[0][0] = (float)cos(theta); r.m[0][1] = (float)-sin(theta); r.m[0][3] = (float)tx;
    r.m[1][0] = (float)sin(theta); r.m[1][1] = (float)cos(theta); r.m[1][3] = (float)ty;
    return r;
}

// ---- point-cloud helpers ---------------------------------------------------
// A well-conditioned random reference cloud in [-50, 50]^dim (interleaved x,y[,z]).
std::vector<float> makeCloud(std::mt19937& gen, unsigned dim, int n) {
    std::uniform_real_distribution<float> distr(-50.f, 50.f);
    std::vector<float> pts((size_t)n * dim);
    for (auto& v : pts) v = distr(gen);
    return pts;
}
// warped[i] = A * reference[i]
std::vector<float> warpBy(const std::vector<float>& ref, unsigned dim, const mat44& a) {
    std::vector<float> w(ref.size());
    const int n = (int)(ref.size() / dim);
    for (int i = 0; i < n; ++i) {
        float in[3] = { 0, 0, 0 }, out[3] = { 0, 0, 0 };
        for (unsigned d = 0; d < dim; ++d) in[d] = ref[i * dim + d];
        if (dim == 3) Mat44Mul<float>(a, in, out);
        else Mat44Mul<float, false>(a, in, out);
        for (unsigned d = 0; d < dim; ++d) w[i * dim + d] = out[d];
    }
    return w;
}

// Run the LTS estimator on hand-built correspondences and return the recovered matrix.
// Pass an identity input matrix so `optimize` leaves the warped points untouched and fits
// `recovered : reference -> warped`. The position arrays are malloc'd so the struct's
// destructor (which free()s them) matches.
mat44 runLts(unsigned dim, const std::vector<float>& ref, const std::vector<float>& warp,
             int activeBlockNumber, int definedActiveBlockNumber, int percentToKeep, bool affine) {
    _reg_blockMatchingParam params;
    params.dim = dim;
    params.blockNumber[0] = 1;
    params.blockNumber[1] = 1;
    params.blockNumber[2] = dim == 2 ? 1u : 2u;   // blockNumber[2]==1 selects the 2D path
    params.activeBlockNumber = activeBlockNumber;
    params.definedActiveBlockNumber = definedActiveBlockNumber;
    params.percent_to_keep = percentToKeep;
    const size_t len = (size_t)activeBlockNumber * dim;
    params.referencePosition = (float*)malloc(len * sizeof(float));
    params.warpedPosition = (float*)malloc(len * sizeof(float));
    std::copy(ref.begin(), ref.end(), params.referencePosition);
    std::copy(warp.begin(), warp.end(), params.warpedPosition);

    mat44 recovered; Mat44Eye(&recovered);
    optimize(&params, &recovered, affine);
    return recovered;
}

// Largest ||A*p - B*p|| over the probe points (physically meaningful, in mm).
double maxProbeDisplacement(const mat44& a, const mat44& b, const std::vector<float>& probes, unsigned dim) {
    double worst = 0;
    const int n = (int)(probes.size() / dim);
    for (int i = 0; i < n; ++i) {
        float in[3] = { 0, 0, 0 }, pa[3] = { 0, 0, 0 }, pb[3] = { 0, 0, 0 };
        for (unsigned d = 0; d < dim; ++d) in[d] = probes[i * dim + d];
        if (dim == 3) { Mat44Mul<float>(a, in, pa); Mat44Mul<float>(b, in, pb); }
        else { Mat44Mul<float, false>(a, in, pa); Mat44Mul<float, false>(b, in, pb); }
        double sq = 0;
        for (unsigned d = 0; d < dim; ++d) sq += Square((double)pa[d] - (double)pb[d]);
        worst = std::max(worst, sqrt(sq));
    }
    return worst;
}

// Determinant of the 3x3 (or 2x2) linear part of a mat44.
double linearDet(const mat44& m, unsigned dim) {
    if (dim == 2)
        return (double)m.m[0][0] * m.m[1][1] - (double)m.m[0][1] * m.m[1][0];
    return (double)m.m[0][0] * ((double)m.m[1][1] * m.m[2][2] - (double)m.m[1][2] * m.m[2][1])
         - (double)m.m[0][1] * ((double)m.m[1][0] * m.m[2][2] - (double)m.m[1][2] * m.m[2][0])
         + (double)m.m[0][2] * ((double)m.m[1][0] * m.m[2][1] - (double)m.m[1][1] * m.m[2][0]);
}
// Largest deviation of R^T R from identity over the linear part (orthonormality check).
double maxOrthonormalityError(const mat44& m, unsigned dim) {
    double worst = 0;
    for (unsigned i = 0; i < dim; ++i)
        for (unsigned j = 0; j < dim; ++j) {
            double dot = 0;
            for (unsigned k = 0; k < dim; ++k) dot += (double)m.m[k][i] * (double)m.m[k][j];
            worst = std::max(worst, fabs(dot - (i == j ? 1.0 : 0.0)));
        }
    return worst;
}

constexpr double TOL_EXACT = 1e-3;   // recovery of an exact (float) correspondence set, in mm
constexpr double TOL_MIN = 1e-2;     // minimum-point / fewer-equation cases
constexpr double TOL_ORTHO = 1e-4;   // rigid rotation orthonormality / |det - 1|

} // namespace

TEST_CASE("LTS estimation", "[unit]") {
    std::mt19937 gen(0);
    constexpr int N = 120;

    // ---- Group A: exact recovery, no outliers (percent_to_keep = 100) ------
    SECTION("Affine 3D exact recovery") {
        const auto ref = makeCloud(gen, 3, N);
        const auto gt = makeAffine3d();
        const auto rec = runLts(3, ref, warpBy(ref, 3, gt), N, N, 100, true);
        REQUIRE(maxProbeDisplacement(rec, gt, ref, 3) < TOL_EXACT);
    }
    SECTION("Affine 2D exact recovery") {
        const auto ref = makeCloud(gen, 2, N);
        const auto gt = makeAffine2d();
        const auto rec = runLts(2, ref, warpBy(ref, 2, gt), N, N, 100, true);
        REQUIRE(maxProbeDisplacement(rec, gt, ref, 2) < TOL_EXACT);
    }
    SECTION("Rigid 3D exact recovery + orthonormality") {
        const auto ref = makeCloud(gen, 3, N);
        const auto gt = makeRigid3d(0.20, -0.15, 0.30, 6.0, -4.0, 9.0);
        const auto rec = runLts(3, ref, warpBy(ref, 3, gt), N, N, 100, false);
        REQUIRE(maxProbeDisplacement(rec, gt, ref, 3) < TOL_EXACT);
        REQUIRE(fabs(linearDet(rec, 3) - 1.0) < TOL_ORTHO);
        REQUIRE(maxOrthonormalityError(rec, 3) < TOL_ORTHO);
    }
    SECTION("Rigid 2D exact recovery + orthonormality") {
        const auto ref = makeCloud(gen, 2, N);
        const auto gt = makeRigid2d(0.35, 4.0, -7.0);
        const auto rec = runLts(2, ref, warpBy(ref, 2, gt), N, N, 100, false);
        REQUIRE(maxProbeDisplacement(rec, gt, ref, 2) < TOL_EXACT);
        REQUIRE(fabs(linearDet(rec, 2) - 1.0) < TOL_ORTHO);
        REQUIRE(maxOrthonormalityError(rec, 2) < TOL_ORTHO);
    }
    // A reflected (improper) correspondence set forces det(V*U^T) < 0, exercising the reflection
    // guard in EstimateRigidLeastSquares. The estimator must still return a *proper* rotation
    // (det +1, orthonormal), never a reflection.
    SECTION("Rigid reflection guard (improper input -> proper rotation)") {
        const auto ref = makeCloud(gen, 3, N);
        mat44 refl; Mat44Eye(&refl);
        refl.m[0][0] = -1.0f;                                   // reflect across x=0 (det of linear part = -1)
        refl.m[0][3] = 3.0f; refl.m[1][3] = -2.0f; refl.m[2][3] = 5.0f;
        const auto rec = runLts(3, ref, warpBy(ref, 3, refl), N, N, 100, false);
        REQUIRE(fabs(linearDet(rec, 3) - 1.0) < TOL_ORTHO);     // proper rotation, not a reflection
        REQUIRE(maxOrthonormalityError(rec, 3) < TOL_ORTHO);
    }
    SECTION("Identity recovery") {
        const auto ref = makeCloud(gen, 3, N);
        mat44 gt; Mat44Eye(&gt);
        const auto rec = runLts(3, ref, ref, N, N, 100, true);   // warped == reference
        REQUIRE(maxProbeDisplacement(rec, gt, ref, 3) < TOL_EXACT);
    }
    SECTION("Pure translation recovery") {
        const auto ref = makeCloud(gen, 3, N);
        mat44 gt; Mat44Eye(&gt);
        gt.m[0][3] = 12.5f; gt.m[1][3] = -8.0f; gt.m[2][3] = 3.25f;
        const auto rec = runLts(3, ref, warpBy(ref, 3, gt), N, N, 100, false);
        REQUIRE(maxProbeDisplacement(rec, gt, ref, 3) < TOL_EXACT);
    }

    // ---- Group B: outlier trimming ------
    // 90 exact inliers + 30 gross outliers; keeping <=75% must reject the outliers.
    SECTION("Affine 3D with outliers (trimming recovers)") {
        const auto ref = makeCloud(gen, 3, N);
        auto warp = warpBy(ref, 3, makeAffine3d());
        std::uniform_real_distribution<float> gross(-100.f, 100.f);
        for (int i = 90; i < N; ++i)                       // corrupt the last 30
            for (unsigned d = 0; d < 3; ++d) warp[i * 3 + d] += gross(gen);
        const auto rec = runLts(3, ref, warp, N, N, 70, true);   // keep 70% -> 84 pts, all inliers
        REQUIRE(maxProbeDisplacement(rec, makeAffine3d(), ref, 3) < TOL_EXACT);
    }
    SECTION("Rigid 3D with outliers (trimming recovers)") {
        const auto ref = makeCloud(gen, 3, N);
        const auto gt = makeRigid3d(0.20, -0.15, 0.30, 6.0, -4.0, 9.0);
        auto warp = warpBy(ref, 3, gt);
        std::uniform_real_distribution<float> gross(-100.f, 100.f);
        for (int i = 90; i < N; ++i)
            for (unsigned d = 0; d < 3; ++d) warp[i * 3 + d] += gross(gen);
        const auto rec = runLts(3, ref, warp, N, N, 70, false);
        REQUIRE(maxProbeDisplacement(rec, gt, ref, 3) < TOL_EXACT);
    }
    SECTION("Trimming is necessary (no trim -> corrupted fit)") {
        // Identical contaminated data, but keep 100% (no trimming): the fit must be far off.
        const auto ref = makeCloud(gen, 3, N);
        const auto gt = makeAffine3d();
        auto warp = warpBy(ref, 3, gt);
        std::uniform_real_distribution<float> gross(-100.f, 100.f);
        for (int i = 90; i < N; ++i)
            for (unsigned d = 0; d < 3; ++d) warp[i * 3 + d] += gross(gen);
        const auto recTrim = runLts(3, ref, warp, N, N, 70, true);    // with trimming
        const auto recAll = runLts(3, ref, warp, N, N, 100, true);    // without trimming
        REQUIRE(maxProbeDisplacement(recTrim, gt, ref, 3) < TOL_EXACT);
        REQUIRE(maxProbeDisplacement(recAll, gt, ref, 3) > 1.0);      // corrupted by outliers
    }

    // ---- Group C: edge cases -----------------------------------------------
    SECTION("NaN correspondences are skipped") {
        const auto ref = makeCloud(gen, 3, N);
        const auto gt = makeAffine3d();
        auto warp = warpBy(ref, 3, gt);
        const int nNaN = 10;
        for (int i = 0; i < nNaN; ++i)                     // poison 10 warped points
            warp[i * 3] = std::numeric_limits<float>::quiet_NaN();
        // activeBlockNumber = N (loop count), definedActiveBlockNumber = non-NaN count
        const auto rec = runLts(3, ref, warp, N, N - nNaN, 100, true);
        REQUIRE(maxProbeDisplacement(rec, gt, ref, 3) < TOL_EXACT);
    }
    SECTION("Minimum correspondences (3D affine, 8 points)") {
        const auto ref = makeCloud(gen, 3, 8);
        const auto gt = makeAffine3d();
        const auto rec = runLts(3, ref, warpBy(ref, 3, gt), 8, 8, 100, true);
        REQUIRE(maxProbeDisplacement(rec, gt, ref, 3) < TOL_MIN);
    }
    SECTION("Minimum correspondences (3D rigid, 4 points)") {
        const auto ref = makeCloud(gen, 3, 4);
        const auto gt = makeRigid3d(0.20, -0.15, 0.30, 6.0, -4.0, 9.0);
        const auto rec = runLts(3, ref, warpBy(ref, 3, gt), 4, 4, 100, false);
        REQUIRE(maxProbeDisplacement(rec, gt, ref, 3) < TOL_MIN);
        REQUIRE(fabs(linearDet(rec, 3) - 1.0) < TOL_ORTHO);
    }
    SECTION("Small Gaussian noise (least-squares averaging)") {
        const auto ref = makeCloud(gen, 3, N);
        const auto gt = makeAffine3d();
        auto warp = warpBy(ref, 3, gt);
        std::normal_distribution<float> noise(0.f, 0.2f);
        for (auto& v : warp) v += noise(gen);
        const auto rec = runLts(3, ref, warp, N, N, 100, true);
        REQUIRE(maxProbeDisplacement(rec, gt, ref, 3) < 0.5);   // well below the 0.2 mm per-point noise
    }
    // Verify recovery even when half the correspondences are gross outliers (the LTS
    // 50% breakdown point) for both transform models.
    SECTION("Aladin default keep=50% with 50% outliers (affine 3D)") {
        const auto ref = makeCloud(gen, 3, N);
        const auto gt = makeAffine3d();
        auto warp = warpBy(ref, 3, gt);
        std::uniform_real_distribution<float> gross(-100.f, 100.f);
        for (int i = N / 2; i < N; ++i)                    // corrupt half the points
            for (unsigned d = 0; d < 3; ++d) warp[i * 3 + d] += gross(gen);
        const auto rec = runLts(3, ref, warp, N, N, 50, true);
        REQUIRE(maxProbeDisplacement(rec, gt, ref, 3) < TOL_EXACT);
    }
    SECTION("Aladin default keep=50% with 50% outliers (rigid 3D)") {
        const auto ref = makeCloud(gen, 3, N);
        const auto gt = makeRigid3d(0.20, -0.15, 0.30, 6.0, -4.0, 9.0);
        auto warp = warpBy(ref, 3, gt);
        std::uniform_real_distribution<float> gross(-100.f, 100.f);
        for (int i = N / 2; i < N; ++i)
            for (unsigned d = 0; d < 3; ++d) warp[i * 3 + d] += gross(gen);
        const auto rec = runLts(3, ref, warp, N, N, 50, false);
        REQUIRE(maxProbeDisplacement(rec, gt, ref, 3) < TOL_EXACT);
        REQUIRE(fabs(linearDet(rec, 3) - 1.0) < TOL_ORTHO);
    }
    SECTION("Determinism (identical result across runs)") {
        const auto ref = makeCloud(gen, 3, N);
        const auto warp = warpBy(ref, 3, makeAffine3d());
        const auto r1 = runLts(3, ref, warp, N, N, 100, true);
        const auto r2 = runLts(3, ref, warp, N, N, 100, true);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                REQUIRE(r1.m[i][j] == r2.m[i][j]);
    }
}
