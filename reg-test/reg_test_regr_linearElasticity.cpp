#include "reg_test_common.h"
#include "CudaF3dContent.h"

/**
 *  Linear elasticity (approximate linear energy) CPU vs CUDA diagnostic regression test.
 *  For every case we evaluate the approximate linear energy value and its gradient on three backends:
 *    - CPU  single precision (NIFTI_TYPE_FLOAT32 control point grid)
 *    - CUDA single precision
 *    - CPU  double precision (NIFTI_TYPE_FLOAT64 control point grid) -> treated as ground truth,
 *      since reg_spline_approxLinearEnergy / reg_spline_approxLinearEnergyGradient dispatch on the
 *      control point grid datatype. CUDA device storage is float only, so the reference is CPU.
 *
 *  We then report, per case:
 *    - accuracy vs the double reference:  |CPU_f - ref|   vs   |CUDA_f - ref|
 *    - the signed CPU_f - CUDA_f difference (mean over voxels): a consistent sign across voxels and
 *      cases indicates a directional bias (bug); a zero-mean scatter indicates rounding.
 *  across three input regimes (identity / near-identity / large-random) in 2D and 3D.
 *
 *  A hard REQUIRE(|CPU_f - CUDA_f| < EPS) is kept as the pass gate; the accuracy/bias lines use CHECK
 *  so a run fully characterises the discrepancy even while the gate is red.
**/

class LinearElasticityTest {
protected:
    // name, valueCpuF, valueCudaF, valueRefD, gradCpuF, gradCudaF, gradRefD
    using TestCase = std::tuple<std::string, double, double, double, NiftiImage, NiftiImage, NiftiImage>;

    inline static vector<TestCase> testCases;

    // The three input regimes exercised for each dimensionality
    enum class Regime { Identity, NearIdentity, LargeRandom };

    static std::string RegimeName(Regime regime) {
        switch (regime) {
        case Regime::Identity: return "identity";
        case Regime::NearIdentity: return "near-identity";
        case Regime::LargeRandom: return "large-random";
        }
        return "";
    }

    // A non-identity, anisotropic, sheared sform so the mm<->voxel reorientation (which feeds the
    // polar decomposition) is non-trivial - closer to a real image orientation than identity.
    static mat44 SkewedSform(int dim) {
        mat44 m;
        Mat44Eye(&m);
        m.m[0][0] = 1.3f; m.m[0][1] = 0.10f;
        m.m[1][0] = 0.05f; m.m[1][1] = 0.8f;
        if (dim == 3) {
            m.m[0][2] = 0.07f;
            m.m[1][2] = 0.04f;
            m.m[2][0] = 0.02f; m.m[2][1] = 0.06f; m.m[2][2] = 1.15f;
        }
        return m;
    }

public:
    LinearElasticityTest() {
        if (!testCases.empty())
            return;

        // Deterministic seed so the discrepancy is reproducible run to run
        std::mt19937 gen(0);
        std::uniform_real_distribution<float> distrLarge(0, 10);
        std::uniform_real_distribution<float> distrSmall(-0.1f, 0.1f);

        // A linear-in-weight scalar factor on the gradient; use 1 so absolute diffs are readable
        // and never scaled below EPS. It does not affect whether CPU and CUDA agree.
        constexpr float weight = 1.f;

        // Create the platforms
        Platform platformCpu(PlatformType::Cpu);
        Platform platformCuda(PlatformType::Cuda);

        for (const int dim : { 2, 3 }) {
          for (const bool skewed : { false, true }) {
            // Reference/floating images (unused by the linear-energy path, but required by the content)
            constexpr NiftiImage::dim_t size = 8;
            vector<NiftiImage::dim_t> refDim(dim, size);
            NiftiImage reference(refDim, NIFTI_TYPE_FLOAT32);
            NiftiImage floating(refDim, NIFTI_TYPE_FLOAT32);
            // The grid inherits the reference geometry, so set the sform before creating it
            if (skewed)
                setSform(reference, SkewedSform(dim));

            for (const Regime regime : { Regime::Identity, Regime::NearIdentity, Regime::LargeRandom }) {
                // Master single-precision control point grid, initialised to identity
                NiftiImage controlPointGrid = CreateControlPointGrid(reference);
                auto cpgPtr = controlPointGrid.data();
                for (size_t j = 0; j < controlPointGrid.nVoxels(); j++) {
                    switch (regime) {
                    case Regime::Identity:
                        break;  // leave the identity transformation untouched
                    case Regime::NearIdentity:
                        cpgPtr[j] = static_cast<float>(cpgPtr[j]) + distrSmall(gen);
                        break;
                    case Regime::LargeRandom:
                        cpgPtr[j] = distrLarge(gen);
                        break;
                    }
                }

                // Double-precision copy with identical coefficient values (float -> double is exact)
                NiftiImage controlPointGridDouble(controlPointGrid);
                controlPointGridDouble.changeDatatype(NIFTI_TYPE_FLOAT64);

                const std::string testName = std::to_string(dim) + "D " + RegimeName(regime) +
                                             (skewed ? " skewed-sform" : " identity-sform");

                // Per-backend image copies (each content owns / mutates its own buffers)
                NiftiImage referenceCpu(reference), referenceCuda(reference), referenceRef(reference);
                NiftiImage floatingCpu(floating), floatingCuda(floating), floatingRef(floating);
                NiftiImage cpgCpu(controlPointGrid), cpgCuda(controlPointGrid), cpgRef(controlPointGridDouble);

                // Contents: CPU float, CUDA float, CPU double (reference).
                // bytes = sizeof(float) throughout - it governs the (unused) warped/deformation
                // precision, not the control point grid datatype that drives the linear-energy path.
                unique_ptr<F3dContent> contentCpu{ new F3dContent(
                    referenceCpu, floatingCpu, cpgCpu, nullptr, nullptr, nullptr, sizeof(float)) };
                unique_ptr<F3dContent> contentCuda{ new CudaF3dContent(
                    referenceCuda, floatingCuda, cpgCuda, nullptr, nullptr, nullptr, sizeof(float)) };
                unique_ptr<F3dContent> contentRef{ new F3dContent(
                    referenceRef, floatingRef, cpgRef, nullptr, nullptr, nullptr, sizeof(float)) };

                // Computes
                unique_ptr<Compute> computeCpu{ platformCpu.CreateCompute(*contentCpu) };
                unique_ptr<Compute> computeCuda{ platformCuda.CreateCompute(*contentCuda) };
                unique_ptr<Compute> computeRef{ platformCpu.CreateCompute(*contentRef) };

                // Approximate linear energy value
                const double valueCpu = computeCpu->ApproxLinearEnergy();
                const double valueCuda = computeCuda->ApproxLinearEnergy();
                const double valueRef = computeRef->ApproxLinearEnergy();

                // Approximate linear energy gradient (accumulates into the zero-initialised gradient)
                computeCpu->ApproxLinearEnergyGradient(weight);
                computeCuda->ApproxLinearEnergyGradient(weight);
                computeRef->ApproxLinearEnergyGradient(weight);

                testCases.push_back({ testName, valueCpu, valueCuda, valueRef,
                                      std::move(contentCpu->GetTransformationGradient()),
                                      std::move(contentCuda->GetTransformationGradient()),
                                      std::move(contentRef->GetTransformationGradient()) });
            }
          }
        }
    }
};

TEST_CASE_METHOD(LinearElasticityTest, "Regression Linear Elasticity", "[regression]") {
    // Loop over all generated test cases
    for (auto&& testCase : testCases) {
        auto&& [testName, valueCpu, valueCuda, valueRef, gradCpu, gradCuda, gradRef] = testCase;

        SECTION(testName) {
            NR_COUT << "\n**************** Section " << testName << " ****************" << std::endl;
            NR_COUT << std::fixed << std::setprecision(10);

            // ---- Value: accuracy vs double reference + CPU/CUDA agreement ----
            const double valueCpuErr = abs(valueCpu - valueRef);
            const double valueCudaErr = abs(valueCuda - valueRef);
            NR_COUT << "Value  cpu=" << valueCpu << " cuda=" << valueCuda << " ref=" << valueRef << std::endl;
            NR_COUT << "Value  |cpu-ref|=" << valueCpuErr << "  |cuda-ref|=" << valueCudaErr
                    << "  (cpu-cuda)=" << (valueCpu - valueCuda) << std::endl;
            // Soft check: is CUDA systematically farther from truth than CPU?
            CHECK(valueCudaErr <= valueCpuErr + double(EPS));

            // ---- Gradient: per-voxel accuracy, agreement and signed bias ----
            const auto gradCpuPtr = gradCpu.data();
            const auto gradCudaPtr = gradCuda.data();
            const auto gradRefPtr = gradRef.data();
            REQUIRE(gradCpu.nVoxels() == gradCuda.nVoxels());
            REQUIRE(gradCpu.nVoxels() == gradRef.nVoxels());

            double maxCpuErr = 0, maxCudaErr = 0, maxCpuCudaDiff = 0;
            double sumSigned = 0;   // sum of (cpu - cuda) to expose directional bias
            size_t maxDiffIndex = 0;
            for (size_t i = 0; i < gradCpu.nVoxels(); ++i) {
                const double cpuVal = gradCpuPtr[i];
                const double cudaVal = gradCudaPtr[i];
                const double refVal = gradRefPtr[i];
                maxCpuErr = std::max(maxCpuErr, abs(cpuVal - refVal));
                maxCudaErr = std::max(maxCudaErr, abs(cudaVal - refVal));
                const double diff = abs(cpuVal - cudaVal);
                if (diff > maxCpuCudaDiff) {
                    maxCpuCudaDiff = diff;
                    maxDiffIndex = i;
                }
                sumSigned += cpuVal - cudaVal;
            }
            const double meanSigned = sumSigned / static_cast<double>(gradCpu.nVoxels());

            NR_COUT << "Grad   max|cpu-ref|=" << maxCpuErr << "  max|cuda-ref|=" << maxCudaErr << std::endl;
            NR_COUT << "Grad   max|cpu-cuda|=" << maxCpuCudaDiff << " @ voxel " << maxDiffIndex
                    << "  mean(cpu-cuda)=" << meanSigned << std::endl;

            // Soft check: is CUDA systematically less accurate than CPU on the gradient?
            CHECK(maxCudaErr <= maxCpuErr + double(EPS));

            // Pass gate: the CUDA gradient is aligned to the CPU oracle bit-for-bit (FMA off)
            REQUIRE(maxCpuCudaDiff == 0);
            // The scalar energy still reduces in a different order (thrust vs OpenMP), so it is only
            // required to agree within EPS, not bit-exactly
            REQUIRE(abs(valueCpu - valueCuda) < EPS);
        }
    }
}

/**
 *  Finite-difference consistency check of the approximate linear energy GRADIENT against the
 *  approximate linear energy VALUE, entirely on the CPU (double precision) so no CPU/CUDA precision
 *  is involved. It answers: "is reg_spline_approxLinearEnergyGradient the actual derivative of
 *  reg_spline_approxLinearEnergy?" For each control point coefficient c_i we compute the central
 *  finite difference of the (weighted) energy, weight * (E(c_i + h) - E(c_i - h)) / (2h), and compare
 *  it to the analytical gradient entry. A correct gradient matches to O(h^2); a systematic ratio or
 *  residual exposes a definition mismatch (e.g. diagonal-only vs full symmetric tensor, or a
 *  normalisation factor between value and gradient).
**/
class LinearElasticityGradientFiniteDiffTest {
protected:
    // name, analytical gradient, numerical gradient (both double)
    using TestCase = std::tuple<std::string, NiftiImage, NiftiImage>;

    inline static vector<TestCase> testCases;

public:
    LinearElasticityGradientFiniteDiffTest() {
        if (!testCases.empty())
            return;

        std::mt19937 gen(1);
        std::uniform_real_distribution<float> distr(-0.2f, 0.2f);

        constexpr float weight = 1.f;
        constexpr double h = 1e-3;   // finite-difference step (coefficients are O(1..10) mm)

        Platform platformCpu(PlatformType::Cpu);

        for (const int dim : { 2, 3 }) {
            constexpr NiftiImage::dim_t size = 6;
            vector<NiftiImage::dim_t> refDim(dim, size);
            NiftiImage reference(refDim, NIFTI_TYPE_FLOAT32);
            NiftiImage floating(refDim, NIFTI_TYPE_FLOAT32);

            // Double-precision, near-identity control point grid (small so the polar derivative,
            // which the analytical gradient omits, is a second-order effect and does not mask the
            // structural value/gradient mismatch)
            NiftiImage controlPointGrid = CreateControlPointGrid(reference);
            controlPointGrid.changeDatatype(NIFTI_TYPE_FLOAT64);
            {
                auto cpgPtr = controlPointGrid.data();
                for (size_t j = 0; j < controlPointGrid.nVoxels(); j++)
                    cpgPtr[j] = static_cast<double>(cpgPtr[j]) + distr(gen);
            }

            const std::string testName = std::to_string(dim) + "D near-identity";

            // Analytical gradient
            NiftiImage refA(reference), floA(floating), cpgA(controlPointGrid);
            unique_ptr<F3dContent> contentA{ new F3dContent(refA, floA, cpgA, nullptr, nullptr, nullptr, sizeof(double)) };
            unique_ptr<Compute> computeA{ platformCpu.CreateCompute(*contentA) };
            computeA->ApproxLinearEnergyGradient(weight);
            NiftiImage analytical = contentA->GetTransformationGradient();

            // Numerical gradient via central finite differences on the value
            NiftiImage refN(reference), floN(floating), cpgN(controlPointGrid);
            unique_ptr<F3dContent> contentN{ new F3dContent(refN, floN, cpgN, nullptr, nullptr, nullptr, sizeof(double)) };
            unique_ptr<Compute> computeN{ platformCpu.CreateCompute(*contentN) };
            NiftiImage& liveCpg = contentN->GetControlPointGrid();
            auto cpgPtr = liveCpg.data();

            NiftiImage numerical(analytical, NiftiImage::Copy::ImageInfoAndAllocData);
            auto numPtr = numerical.data();
            for (size_t i = 0; i < liveCpg.nVoxels(); ++i) {
                const double c = cpgPtr[i];
                cpgPtr[i] = c + h;
                const double ePlus = computeN->ApproxLinearEnergy();
                cpgPtr[i] = c - h;
                const double eMinus = computeN->ApproxLinearEnergy();
                cpgPtr[i] = c;   // restore
                numPtr[i] = weight * (ePlus - eMinus) / (2 * h);
            }

            testCases.push_back({ testName, std::move(analytical), std::move(numerical) });
        }
    }
};

// Tagged [!mayfail]: this documents a *separate*, currently-deferred CPU-side issue - the analytical
// -le gradient is not the derivative of the -le energy (diagonal-only vs full symmetric tensor, and a
// value/gradient normalisation mismatch). It is identical on CPU and CUDA, so it is not a CPU/CUDA
// alignment problem. The tag records the inconsistency without failing the suite; remove it if/when
// the gradient definition is reconciled with the energy.
TEST_CASE_METHOD(LinearElasticityGradientFiniteDiffTest, "Linear Elasticity Gradient Finite Difference", "[unit][!mayfail]") {
    for (auto&& testCase : testCases) {
        auto&& [testName, analytical, numerical] = testCase;

        SECTION(testName) {
            NR_COUT << "\n**************** FD Section " << testName << " ****************" << std::endl;
            NR_COUT << std::fixed << std::setprecision(10);

            const auto anaPtr = analytical.data();
            const auto numPtr = numerical.data();

            // First pass: magnitude of the numerical gradient, to set a relative significance floor
            double maxAbs = 0;
            for (size_t i = 0; i < numerical.nVoxels(); ++i)
                maxAbs = std::max(maxAbs, abs(static_cast<double>(numPtr[i])));
            const double significant = 0.1 * maxAbs;   // ignore near-zero entries when forming ratios

            double maxAbsDiff = 0;
            double sumRatio = 0; size_t ratioCount = 0;
            size_t maxDiffIndex = 0;
            for (size_t i = 0; i < analytical.nVoxels(); ++i) {
                const double a = anaPtr[i];
                const double n = numPtr[i];
                const double diff = abs(a - n);
                if (diff > maxAbsDiff) { maxAbsDiff = diff; maxDiffIndex = i; }
                if (abs(n) > significant) { sumRatio += a / n; ratioCount++; }   // analytical/numerical ratio
            }
            const double meanRatio = ratioCount ? sumRatio / static_cast<double>(ratioCount) : 0;

            NR_COUT << "FD   max|analytical-numerical|=" << maxAbsDiff << " @ voxel " << maxDiffIndex
                    << "  (max|numerical|=" << maxAbs << ")" << std::endl;
            NR_COUT << "FD   sample analytical/numerical @" << maxDiffIndex << ": "
                    << static_cast<double>(anaPtr[maxDiffIndex]) << " / "
                    << static_cast<double>(numPtr[maxDiffIndex]) << std::endl;
            NR_COUT << "FD   mean(analytical/numerical) over significant entries = " << meanRatio
                    << " (over " << ratioCount << " entries)" << std::endl;

            // Diagnostic: a correct gradient has ratio ~1 and small residual. We report but do not
            // hard-fail here, so the run characterises the mismatch.
            CHECK(maxAbsDiff <= 1e-3 * (maxAbs + 1.0));
        }
    }
}
