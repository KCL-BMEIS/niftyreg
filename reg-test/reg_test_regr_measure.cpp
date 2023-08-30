#include "reg_test_common.h"
#include "_reg_nmi.h"
#include "CudaF3dContent.h"
#include "CudaMeasure.h"
#include <iomanip>

/**
 *  Measure regression tests to ensure the CPU and CUDA versions yield the same output
 *  Test classes:
 *   - NMI
 *   - SSD
 */

class MeasureTest {
protected:
    using TestData = std::tuple<std::string, NiftiImage, NiftiImage, NiftiImage, NiftiImage, MeasureType, bool>;
    using TestCase = std::tuple<std::string, double, double, NiftiImage, NiftiImage>;

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
        vector<NiftiImage::dim_t> dim{ size, size };
        NiftiImage reference2d(dim, NIFTI_TYPE_FLOAT32);
        NiftiImage floating2d(dim, NIFTI_TYPE_FLOAT32);
        NiftiImage controlPointGrid2d(CreateControlPointGrid(reference2d));
        NiftiImage localWeightSim2d(dim, NIFTI_TYPE_FLOAT32);

        // Create 3D reference, floating, control point grid and local weight similarity images
        dim.push_back(size);
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

        // Create the data container for the regression test
        const std::string measureNames[]{ "NMI"s, "SSD"s, "DTI"s, "LNCC"s, "KLD"s, "MIND"s, "MINDSSC"s };
        const MeasureType testMeasures[]{ MeasureType::Nmi, MeasureType::Ssd };
        vector<TestData> testData;
        for (auto&& measure : testMeasures) {
            for (int sym = 0; sym < 2; ++sym) {
                testData.emplace_back(TestData(
                    measureNames[(int)measure] + " 2D"s + (sym ? " Symmetric" : ""),
                    reference2d,
                    floating2d,
                    controlPointGrid2d,
                    localWeightSim2d,
                    measure,
                    sym
                ));
                testData.emplace_back(TestData(
                    measureNames[(int)measure] + " 3D"s + (sym ? " Symmetric" : ""),
                    reference3d,
                    floating3d,
                    controlPointGrid3d,
                    localWeightSim3d,
                    measure,
                    sym
                ));
            }
        }

        // Create the platforms
        Platform platformCpu(PlatformType::Cpu);
        Platform platformCuda(PlatformType::Cuda);

        // Create the measures
        unique_ptr<Measure> measureCreatorCpu{ new Measure() };
        unique_ptr<Measure> measureCreatorCuda{ new CudaMeasure() };

        for (auto&& testData : testData) {
            // Get the test data
            auto&& [testName, reference, floating, controlPointGrid, localWeightSim, measureType, isSymmetric] = testData;

            // Create images
            NiftiImage referenceCpu(reference), referenceCuda(reference);
            NiftiImage floatingCpu(floating), floatingCuda(floating);
            NiftiImage controlPointGridCpu(controlPointGrid), controlPointGridCuda(controlPointGrid);
            NiftiImage controlPointGridCpuBw(controlPointGrid), controlPointGridCudaBw(controlPointGrid);
            NiftiImage localWeightSimCpu(localWeightSim), localWeightSimCuda(localWeightSim);

            // Create the contents
            unique_ptr<F3dContent> contentCpu{ new F3dContent(
                referenceCpu,
                floatingCpu,
                controlPointGridCpu,
                localWeightSimCpu,
                nullptr,
                nullptr,
                sizeof(float)
            ) };
            unique_ptr<F3dContent> contentCuda{ new CudaF3dContent(
                referenceCuda,
                floatingCuda,
                controlPointGridCuda,
                localWeightSimCuda,
                nullptr,
                nullptr,
                sizeof(float)
            ) };
            unique_ptr<F3dContent> contentCpuBw, contentCudaBw;
            if (isSymmetric) {
                contentCpuBw.reset(new F3dContent(
                    floatingCpu,
                    referenceCpu,
                    controlPointGridCpuBw,
                    nullptr,
                    nullptr,
                    nullptr,
                    sizeof(float)
                ));
                contentCudaBw.reset(new CudaF3dContent(
                    floatingCuda,
                    referenceCuda,
                    controlPointGridCudaBw,
                    nullptr,
                    nullptr,
                    nullptr,
                    sizeof(float)
                ));
            }

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
            for (int i = 0; i < referenceCpu->nt; ++i) {
                measureCpu->SetTimepointWeight(i, 1.0);
                measureCuda->SetTimepointWeight(i, 1.0);
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
            NiftiImage warpedCuda(contentCuda->F3dContent::GetWarped());
            warpedCuda.copyData(contentCpu->GetWarped());
            warpedCuda.disown();
            contentCuda->UpdateWarped();
            // computeCuda->GetDeformationField(false, true);
            // computeCuda->ResampleImage(1, std::numeric_limits<float>::quiet_NaN());
            if (isSymmetric) {
                NiftiImage warpedCudaBw(contentCudaBw->F3dContent::GetWarped());
                warpedCudaBw.copyData(contentCpuBw->GetWarped());
                warpedCudaBw.disown();
                contentCudaBw->UpdateWarped();
                // computeCudaBw->GetDeformationField(false, true);
                // computeCudaBw->ResampleImage(1, std::numeric_limits<float>::quiet_NaN());
            }
            const double simMeasureCuda = measureCuda->GetSimilarityMeasureValue();

            // Compute the similarity measure gradient for CPU
            int timepoint = 0;
            contentCpu->ZeroVoxelBasedMeasureGradient();
            computeCpu->GetImageGradient(1, std::numeric_limits<float>::quiet_NaN(), timepoint);
            if (isSymmetric) {
                contentCpuBw->ZeroVoxelBasedMeasureGradient();
                computeCpuBw->GetImageGradient(1, std::numeric_limits<float>::quiet_NaN(), timepoint);
            }
            measureCpu->GetVoxelBasedSimilarityMeasureGradient(timepoint);

            // Compute the similarity measure gradient for CUDA
            contentCuda->ZeroVoxelBasedMeasureGradient();
            // computeCuda->GetImageGradient(1, std::numeric_limits<float>::quiet_NaN(), timepoint);
            NiftiImage warpedGradCuda(contentCuda->F3dContent::GetWarpedGradient());
            warpedGradCuda.copyData(contentCpu->GetWarpedGradient());
            warpedGradCuda.disown();
            contentCuda->UpdateWarpedGradient();
            if (isSymmetric) {
                contentCudaBw->ZeroVoxelBasedMeasureGradient();
                // computeCudaBw->GetImageGradient(1, std::numeric_limits<float>::quiet_NaN(), timepoint);
                NiftiImage warpedGradCudaBw(contentCudaBw->F3dContent::GetWarpedGradient());
                warpedGradCudaBw.copyData(contentCpuBw->GetWarpedGradient());
                warpedGradCudaBw.disown();
                contentCudaBw->UpdateWarpedGradient();
            }
            measureCuda->GetVoxelBasedSimilarityMeasureGradient(timepoint);

            // Get the voxel-based similarity measure gradients
            NiftiImage voxelBasedGradCpu(contentCpu->GetVoxelBasedMeasureGradient(), NiftiImage::Copy::Image);
            NiftiImage voxelBasedGradCuda(contentCuda->GetVoxelBasedMeasureGradient(), NiftiImage::Copy::Image);

            // Save for testing
            testCases.push_back({ testName, simMeasureCpu, simMeasureCuda, std::move(voxelBasedGradCpu), std::move(voxelBasedGradCuda) });
        }
    }
};

TEST_CASE_METHOD(MeasureTest, "Regression Measure", "[regression]") {
    // Loop over all generated test cases
    for (auto&& testCase : testCases) {
        // Retrieve test information
        auto&& [testName, simMeasureCpu, simMeasureCuda, voxelBasedGradCpu, voxelBasedGradCuda] = testCase;

        SECTION(testName) {
            NR_COUT << "\n**************** Section " << testName << " ****************" << std::endl;

            // Increase the precision for the output
            NR_COUT << std::fixed << std::setprecision(10);

            // Check the similarity measure values
            NR_COUT << "Similarity measure: " << simMeasureCpu << " " << simMeasureCuda << std::endl;
            REQUIRE(fabs(simMeasureCpu - simMeasureCuda) < EPS);

            // Check the voxel-based similarity measure gradients
            const auto voxelBasedGradCpuPtr = voxelBasedGradCpu.data();
            const auto voxelBasedGradCudaPtr = voxelBasedGradCuda.data();
            for (size_t i = 0; i < voxelBasedGradCpu.nVoxels(); ++i) {
                const float cpuVal = voxelBasedGradCpuPtr[i];
                const float cudaVal = voxelBasedGradCudaPtr[i];
                const double diff = fabs(cpuVal - cudaVal);
                if(diff>EPS)
                    NR_COUT << i << " " << cpuVal << " " << cudaVal << std::endl;
                REQUIRE(diff < EPS);
            }
        }
    }
}
