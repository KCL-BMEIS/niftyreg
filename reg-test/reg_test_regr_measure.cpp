#include "reg_test_common.h"
#include "_reg_nmi.h"
#include "CudaF3dContent.h"
#include "CudaMeasureCreator.hpp"

/**
 *  Measure regression tests to ensure the CPU and CUDA versions yield the same output
 *  Test classes:
 *   - NMI
 *   - SSD
 *   - LNCC
 *
 *  The case above compares the measures at one small image size. The second case in this file sweeps
 *  the NMI value over image size instead, because that is the axis along which the two backends part
 *  company: the joint histogram and the entropies are accumulated in different orders, so the
 *  difference grows with the number of contributing voxels and a single small image cannot see it.
 *  It also covers the NaN-padded case, which an identity transformation never produces.
 */

class MeasureTest {
protected:
    using TestData = std::tuple<std::string, NiftiImage, NiftiImage, NiftiImage, NiftiImage, MeasureType, bool>;
    using TestCase = std::tuple<std::string, double, double, NiftiImage, NiftiImage, NiftiImage, NiftiImage>;

    inline static vector<TestCase> testCases;

public:
    MeasureTest() {
        if (!testCases.empty())
            return;

        // Create a random number generator
        std::mt19937 gen(0);
        std::uniform_real_distribution<float> distr(0, 1);

        // Create 2D reference, floating, control point grid and local weight similarity images
        constexpr NiftiImage::dim_t size = 16;
        constexpr NiftiImage::dim_t timePoints = 2;
        vector<NiftiImage::dim_t> dim{ size, size, 1, timePoints };
        NiftiImage reference2d(dim, NIFTI_TYPE_FLOAT32);
        NiftiImage floating2d(dim, NIFTI_TYPE_FLOAT32);
        NiftiImage controlPointGrid2d(CreateControlPointGrid(reference2d));
        NiftiImage localWeightSim2d(dim, NIFTI_TYPE_FLOAT32);

        // Create 3D reference, floating, control point grid and local weight similarity images
        dim[2] = size;
        NiftiImage reference3d(dim, NIFTI_TYPE_FLOAT32);
        NiftiImage floating3d(dim, NIFTI_TYPE_FLOAT32);
        NiftiImage controlPointGrid3d(CreateControlPointGrid(reference3d));
        NiftiImage localWeightSim3d(dim, NIFTI_TYPE_FLOAT32);

        // Fill images with random values
        auto ref2dPtr = reference2d.data();
        auto flo2dPtr = floating2d.data();
        auto localWeightSim2dPtr = localWeightSim2d.data();
        for (size_t i = 0; i < reference2d.nVoxels(); ++i) {
            ref2dPtr[i] = distr(gen);
            flo2dPtr[i] = distr(gen);
            localWeightSim2dPtr[i] = distr(gen);
        }

        // Fill images with random values
        auto ref3dPtr = reference3d.data();
        auto flo3dPtr = floating3d.data();
        auto localWeightSim3dPtr = localWeightSim3d.data();
        for (size_t i = 0; i < reference3d.nVoxels(); ++i) {
            ref3dPtr[i] = distr(gen);
            flo3dPtr[i] = distr(gen);
            localWeightSim3dPtr[i] = distr(gen);
        }

        // Create floating images with a DIFFERENT size than the reference. The backward similarity
        // of a symmetric registration runs in floating space, so a floating voxel count that differs
        // from the reference's exercises the backward measure with a distinct active voxel number -
        vector<NiftiImage::dim_t> dimBig{ size, size / 2, 1, timePoints };  // 2D: 128 voxels/time point
        NiftiImage floatingBig2d(dimBig, NIFTI_TYPE_FLOAT32);
        dimBig[1] = size;
        dimBig[2] = size / 2;  // 3D: 16 x 16 x 8 = 2048 voxels/time point
        NiftiImage floatingBig3d(dimBig, NIFTI_TYPE_FLOAT32);
        auto floBig2dPtr = floatingBig2d.data();
        for (size_t i = 0; i < floatingBig2d.nVoxels(); ++i)
            floBig2dPtr[i] = distr(gen);
        auto floBig3dPtr = floatingBig3d.data();
        for (size_t i = 0; i < floatingBig3d.nVoxels(); ++i)
            floBig3dPtr[i] = distr(gen);

        // Create the data container for the regression test
        const std::string measureNames[]{ "NMI"s, "SSD"s, "DTI"s, "LNCC"s, "KLD"s, "MIND"s, "MINDSSC"s };
        constexpr MeasureType testMeasures[]{ MeasureType::Nmi, MeasureType::Ssd, MeasureType::Lncc };
        vector<TestData> testData;
        for (auto&& measure : testMeasures) {
            for (int sym = 0; sym < 2; ++sym) {
                testData.emplace_back(TestData(
                    measureNames[int(measure)] + " 2D"s + (sym ? " Symmetric" : ""),
                    reference2d,
                    floating2d,
                    controlPointGrid2d,
                    localWeightSim2d,
                    measure,
                    sym
                ));
                testData.emplace_back(TestData(
                    measureNames[int(measure)] + " 3D"s + (sym ? " Symmetric" : ""),
                    reference3d,
                    floating3d,
                    controlPointGrid3d,
                    localWeightSim3d,
                    measure,
                    sym
                ));
            }
        }

        // Symmetric cases with reference and floating of different sizes (see floatingBig above).
        // Only the measures that reused the forward voxel count on the backward path are affected.
        constexpr MeasureType asymMeasures[]{ MeasureType::Nmi, MeasureType::Ssd };
        for (auto&& measure : asymMeasures) {
            testData.emplace_back(TestData(
                measureNames[int(measure)] + " 2D Symmetric DifferentSize"s,
                reference2d, floatingBig2d, controlPointGrid2d, localWeightSim2d, measure, true));
            testData.emplace_back(TestData(
                measureNames[int(measure)] + " 3D Symmetric DifferentSize"s,
                reference3d, floatingBig3d, controlPointGrid3d, localWeightSim3d, measure, true));
        }

        // Create the platforms
        Platform platformCpu(PlatformType::Cpu);
        Platform platformCuda(PlatformType::Cuda);

        // Create the measures
        unique_ptr<MeasureCreator> measureCreatorCpu{ new MeasureCreator() };
        unique_ptr<MeasureCreator> measureCreatorCuda{ new CudaMeasureCreator() };

        // Create the content creators
        unique_ptr<F3d2ContentCreator> contentCreatorCpu{ dynamic_cast<F3d2ContentCreator*>(platformCpu.CreateContentCreator(ContentType::F3d2)) };
        unique_ptr<F3d2ContentCreator> contentCreatorCuda{ dynamic_cast<F3d2ContentCreator*>(platformCuda.CreateContentCreator(ContentType::F3d2)) };

        for (auto&& testData : testData) {
            // Get the test data
            auto&& [testName, reference, floating, controlPointGrid, localWeightSim, measureType, isSymmetric] = testData;

            // Create images
            NiftiImage referenceCpu(reference), referenceCuda(reference);
            NiftiImage floatingCpu(floating), floatingCuda(floating);
            NiftiImage controlPointGridCpu(controlPointGrid), controlPointGridCuda(controlPointGrid);
            // The backward control point grid is parametrised in floating space, so build it from
            // the floating image (identical to the forward grid when the two images share a size).
            NiftiImage controlPointGridBw(CreateControlPointGrid(floating));
            NiftiImage controlPointGridCpuBw(controlPointGridBw), controlPointGridCudaBw(controlPointGridBw);
            NiftiImage localWeightSimCpu(localWeightSim), localWeightSimCuda(localWeightSim);

            // Create the contents
            auto contentsCpu = contentCreatorCpu->Create(referenceCpu, floatingCpu, controlPointGridCpu, controlPointGridCpuBw, &localWeightSimCpu, nullptr, nullptr, nullptr, nullptr, sizeof(float));
            auto contentsCuda = contentCreatorCuda->Create(referenceCuda, floatingCuda, controlPointGridCuda, controlPointGridCudaBw, &localWeightSimCuda, nullptr, nullptr, nullptr, nullptr, sizeof(float));
            if (!isSymmetric) {
                delete contentsCpu.second;
                delete contentsCuda.second;
                contentsCpu.second = nullptr;
                contentsCuda.second = nullptr;
            }
            unique_ptr<F3dContent> contentCpu{ contentsCpu.first }, contentCpuBw{ contentsCpu.second };
            unique_ptr<F3dContent> contentCuda{ contentsCuda.first }, contentCudaBw{ contentsCuda.second };

            // Create the computes
            unique_ptr<Compute> computeCpu{ platformCpu.CreateCompute(*contentCpu) };
            unique_ptr<Compute> computeCuda{ platformCuda.CreateCompute(*contentCuda) };
            unique_ptr<Compute> computeCpuBw, computeCudaBw;
            if (isSymmetric) {
                computeCpuBw.reset(platformCpu.CreateCompute(*contentCpuBw));
                computeCudaBw.reset(platformCuda.CreateCompute(*contentCudaBw));
            }

            // Create the measures
            unique_ptr<reg_measure> measureCpu{ measureCreatorCpu->Create(measureType) };
            unique_ptr<reg_measure> measureCuda{ measureCreatorCuda->Create(measureType) };

            // Initialise the measures
            for (int t = 0; t < referenceCpu->nt; t++) {
                measureCpu->SetTimePointWeight(t, 1.5);
                measureCuda->SetTimePointWeight(t, 1.5);
            }
            measureCreatorCpu->Initialise(*measureCpu, *contentCpu, contentCpuBw.get());
            measureCreatorCuda->Initialise(*measureCuda, *contentCuda, contentCudaBw.get());

            // Compute the similarity measure value for CPU
            computeCpu->GetDeformationField(false, true);
            computeCpu->ResampleImage(1, std::numeric_limits<float>::quiet_NaN());
            if (isSymmetric) {
                computeCpuBw->GetDeformationField(false, true);
                computeCpuBw->ResampleImage(1, std::numeric_limits<float>::quiet_NaN());
            }
            const double simMeasureCpu = measureCpu->GetSimilarityMeasureValue();

            // Compute the similarity measure value for CUDA
            computeCuda->GetDeformationField(false, true);
            computeCuda->ResampleImage(1, std::numeric_limits<float>::quiet_NaN());
            if (isSymmetric) {
                computeCudaBw->GetDeformationField(false, true);
                computeCudaBw->ResampleImage(1, std::numeric_limits<float>::quiet_NaN());
            }
            const double simMeasureCuda = measureCuda->GetSimilarityMeasureValue();

            // Compute the similarity measure gradients
            contentCpu->ZeroVoxelBasedMeasureGradient();
            contentCuda->ZeroVoxelBasedMeasureGradient();
            if (isSymmetric) {
                contentCpuBw->ZeroVoxelBasedMeasureGradient();
                contentCudaBw->ZeroVoxelBasedMeasureGradient();
            }
            for (int t = 0; t < referenceCpu->nt; t++) {
                // Compute the similarity measure gradient for CPU
                computeCpu->GetImageGradient(1, std::numeric_limits<float>::quiet_NaN(), t);
                if (isSymmetric)
                    computeCpuBw->GetImageGradient(1, std::numeric_limits<float>::quiet_NaN(), t);
                measureCpu->GetVoxelBasedSimilarityMeasureGradient(t);

                // Compute the similarity measure gradient for CUDA
                computeCuda->GetImageGradient(1, std::numeric_limits<float>::quiet_NaN(), t);
                if (isSymmetric)
                    computeCudaBw->GetImageGradient(1, std::numeric_limits<float>::quiet_NaN(), t);
                measureCuda->GetVoxelBasedSimilarityMeasureGradient(t);
            }

            // Save the results for testing. For symmetric registrations also keep the backward
            // voxel-based gradient (computed in floating space) so its CPU/CUDA agreement is
            // checked too; left empty otherwise.
            NiftiImage voxelBasedGradCpuBw, voxelBasedGradCudaBw;
            if (isSymmetric) {
                voxelBasedGradCpuBw = std::move(contentCpuBw->GetVoxelBasedMeasureGradient());
                voxelBasedGradCudaBw = std::move(contentCudaBw->GetVoxelBasedMeasureGradient());
            }
            testCases.push_back({ testName, simMeasureCpu, simMeasureCuda, std::move(contentCpu->GetVoxelBasedMeasureGradient()),
                                std::move(contentCuda->GetVoxelBasedMeasureGradient()),
                                std::move(voxelBasedGradCpuBw), std::move(voxelBasedGradCudaBw) });
        }
    }
};

TEST_CASE_METHOD(MeasureTest, "Regression Measure", "[regression]") {
    // Loop over all generated test cases
    for (auto&& testCase : testCases) {
        // Retrieve test information
        auto&& [testName, simMeasureCpu, simMeasureCuda, voxelBasedGradCpu, voxelBasedGradCuda,
                voxelBasedGradCpuBw, voxelBasedGradCudaBw] = testCase;

        SECTION(testName) {
            // The comparison below is only meaningful if the operation ran at all
            RequireNonZero(voxelBasedGradCpu, "the voxel-based measure gradient");
            NR_COUT << "\n**************** Section " << testName << " ****************" << std::endl;

            // Increase the precision for the output
            NR_COUT << std::fixed << std::setprecision(10);

            // Check the similarity measure values
            NR_COUT << "Similarity measure: " << simMeasureCpu << " " << simMeasureCuda << std::endl;
            REQUIRE(fabs(simMeasureCpu - simMeasureCuda) < EPS);

            // Check the voxel-based similarity measure gradients
            const auto voxelBasedGradCpuPtr = voxelBasedGradCpu.data();
            const auto voxelBasedGradCudaPtr = voxelBasedGradCuda.data();
            double maxGradDiff = 0, maxGradMag = 0;
            for (size_t i = 0; i < voxelBasedGradCpu.nVoxels(); ++i) {
                const float cpuVal = voxelBasedGradCpuPtr[i];
                const float cudaVal = voxelBasedGradCudaPtr[i];
                const float diff = fabs(cpuVal - cudaVal);
                maxGradDiff = std::max<double>(maxGradDiff, diff);
                maxGradMag = std::max<double>(maxGradMag, fabs(cpuVal));
                if (diff > 0)
                    NR_COUT << i << " " << cpuVal << " " << cudaVal << std::endl;
                REQUIRE(diff == 0);
            }
            // Diagnostic: magnitude of the CPU vs CUDA gap (is it ~1e-6 EPS-scale or ~1e-13 bit-scale?)
            NR_COUT << "Value  |cpu-cuda|=" << fabs(simMeasureCpu - simMeasureCuda) << std::endl;
            NR_COUT << "Grad   max|cpu-cuda|=" << maxGradDiff << "  (max|cpu|=" << maxGradMag
                    << ", relative=" << (maxGradMag > 0 ? maxGradDiff / maxGradMag : 0) << ")" << std::endl;

            // Check the backward voxel-based gradient too (symmetric cases only)
            if (voxelBasedGradCpuBw) {
                const auto voxelBasedGradCpuBwPtr = voxelBasedGradCpuBw.data();
                const auto voxelBasedGradCudaBwPtr = voxelBasedGradCudaBw.data();
                for (size_t i = 0; i < voxelBasedGradCpuBw.nVoxels(); ++i) {
                    const float cpuVal = voxelBasedGradCpuBwPtr[i];
                    const float cudaVal = voxelBasedGradCudaBwPtr[i];
                    if (fabs(cpuVal - cudaVal) > 0)
                        NR_COUT << "Bw " << i << " " << cpuVal << " " << cudaVal << std::endl;
                    REQUIRE(fabs(cpuVal - cudaVal) == 0);
                }
            }
        }
    }
}

namespace {

// An image with intensity content spread over enough distinct values to fill an NMI histogram.
// Quantised, with a large zero background, as a real acquisition is: the intensity rescaling NMI
// applies before binning then lands many voxels on bin boundaries.
NiftiImage MakeSizedImage(NiftiImage::dim_t size, bool is3D, double phase) {
    std::vector<NiftiImage::dim_t> dims(is3D ? 3 : 2, size);
    NiftiImage img(dims, NIFTI_TYPE_FLOAT32);
    setIdentitySform(img);
    const int nx = img->nx, ny = img->ny, nz = img->nz;
    auto ptr = img.data();
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const double x = 6.0 * M_PI * i / nx, y = 6.0 * M_PI * j / ny;
                const double z = nz > 1 ? 6.0 * M_PI * k / nz : 0.0;
                const double wave = 120.0 * (1.0 + std::sin(x + phase) * std::cos(y) * std::cos(z));
                const double radius = std::sqrt(std::pow(i - nx / 2.0, 2) + std::pow(j - ny / 2.0, 2) +
                                                (nz > 1 ? std::pow(k - nz / 2.0, 2) : 0.0));
                ptr[(static_cast<size_t>(k) * ny + j) * nx + i] =
                    radius > 0.42 * nx ? 0.f : static_cast<float>(std::round(wave));
            }
    return img;
}

double NmiValueForSize(Platform& platform, MeasureCreator *measureCreator, const NiftiImage& reference,
                       const NiftiImage& floating, bool cuda, float shift) {
    NiftiImage ref(reference), flo(floating), grid = CreateControlPointGrid(reference);
    if (shift != 0.f) {
        // Push the transform partly off the field of view so the warped image carries NaN padding,
        // which NMI has to exclude from its histogram. An identity grid never produces a single NaN.
        auto gridPtr = grid.data();
        for (size_t i = 0; i < grid.nVoxels(); ++i)
            gridPtr[i] = static_cast<float>(gridPtr[i]) + shift;
    }
    unique_ptr<F3dContent> content{ cuda ? new CudaF3dContent(ref, flo, grid) : new F3dContent(ref, flo, grid) };
    unique_ptr<Compute> compute{ platform.CreateCompute(*content) };
    unique_ptr<reg_measure> measure{ measureCreator->Create(MeasureType::Nmi) };
    for (int t = 0; t < ref->nt; ++t)
        measure->SetTimePointWeight(t, 1.0);
    measureCreator->Initialise(*measure, *content, nullptr);
    compute->GetDeformationField(false, true);
    compute->ResampleImage(1, std::numeric_limits<float>::quiet_NaN());
    return measure->GetSimilarityMeasureValue();
}

} // namespace

TEST_CASE("Regression Measure - NMI value against image size", "[regression]") {
    /*
        The NMI value is the last thing keeping plain reg_f3d from being bit-exact across the backends:
        every operation in the iteration chain is exact and the regularisation terms match, but the
        first objective evaluation of a full-resolution run already differs in the similarity term, and
        the line search eventually turns that into a different trajectory.

        The gate is the same tolerance the case above applies to this value, deliberately: the point is
        to record how the difference scales, not to fail on a number that is expected to be non-zero.
        The printed table is the output that matters.
    */
    Platform platformCpu(PlatformType::Cpu);
    Platform platformCuda(PlatformType::Cuda);
    unique_ptr<MeasureCreator> measureCreatorCpu{ new MeasureCreator() };
    unique_ptr<MeasureCreator> measureCreatorCuda{ new CudaMeasureCreator() };

    for (const bool is3D : { false, true })
        for (const float shift : { 0.f, 3.5f }) {
            const std::string name = std::string(is3D ? "3D" : "2D") +
                (shift == 0.f ? ", identity" : ", NaN padding");
            SECTION(name) {
                NR_COUT << "\n**************** NMI value vs size, " << name << " ****************" << std::endl;
                for (const NiftiImage::dim_t size : { 8, 16, 32, 48 }) {
                    const NiftiImage reference = MakeSizedImage(size, is3D, 0.0);
                    const NiftiImage floating = MakeSizedImage(size, is3D, 0.7);
                    const double cpu = NmiValueForSize(platformCpu, measureCreatorCpu.get(), reference,
                                                       floating, false, shift);
                    const double cuda = NmiValueForSize(platformCuda, measureCreatorCuda.get(), reference,
                                                        floating, true, shift);
                    const double difference = std::abs(cpu - cuda);
                    NR_COUT << "  size " << std::setw(3) << size << (is3D ? "^3" : "^2")
                            << ": CPU = " << std::fixed << std::setprecision(12) << cpu
                            << ", CUDA = " << cuda << ", |diff| = " << std::scientific << difference
                            << (difference == 0 ? "  (bit-exact)" : "") << std::endl;
                    INFO("image size " << size << ", |cpu - cuda| = " << difference);
                    REQUIRE(difference < EPS);
                }
            }
        }
}
