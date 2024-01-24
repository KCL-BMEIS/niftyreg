#include "reg_test_common.h"
#include "CudaF3dContent.h"

/**
 *  Exponentiate gradient regression test to ensure the CPU and CUDA versions yield the same output
**/

class ExponentiateGradientTest {
protected:
    using TestData = std::tuple<std::string, NiftiImage, NiftiImage, NiftiImage, NiftiImage, NiftiImage>;
    using TestCase = std::tuple<std::string, NiftiImage, NiftiImage>;

    inline static vector<TestCase> testCases;

public:
    ExponentiateGradientTest() {
        if (!testCases.empty())
            return;

        // Create a random number generator
        std::mt19937 gen(0);
        std::uniform_real_distribution<float> distr(-1, 1);

        // Create reference images
        constexpr NiftiImage::dim_t dimSize = 4;
        NiftiImage reference2d({ dimSize, dimSize }, NIFTI_TYPE_FLOAT32);
        NiftiImage reference3d({ dimSize, dimSize, dimSize }, NIFTI_TYPE_FLOAT32);

        // Create deformation fields
        NiftiImage deformationField2d = CreateDeformationField(reference2d);
        NiftiImage deformationField3d = CreateDeformationField(reference3d);

        // Create control point grids and fill them with random values
        NiftiImage controlPointGrid2d = CreateControlPointGrid(reference2d);
        NiftiImage controlPointGridBw2d = CreateControlPointGrid(reference2d);
        NiftiImage controlPointGrid3d = CreateControlPointGrid(reference3d);
        NiftiImage controlPointGridBw3d = CreateControlPointGrid(reference3d);
        controlPointGridBw2d->intent_p1 = SPLINE_VEL_GRID;
        controlPointGridBw3d->intent_p1 = SPLINE_VEL_GRID;
        auto cpp2dPtr = controlPointGrid2d.data();
        auto cppBw2dPtr = controlPointGridBw2d.data();
        auto cpp3dPtr = controlPointGrid3d.data();
        auto cppBw3dPtr = controlPointGridBw3d.data();
        for (auto i = 0; i < controlPointGrid2d.nVoxels(); i++) {
            cpp2dPtr[i] = distr(gen);
            cppBw2dPtr[i] = distr(gen);
        }
        for (auto i = 0; i < controlPointGrid3d.nVoxels(); i++) {
            cpp3dPtr[i] = distr(gen);
            cppBw3dPtr[i] = distr(gen);
        }

        // Create voxel-based measure gradients and fill them with random values
        NiftiImage voxelBasedGrad2d(deformationField2d, NiftiImage::Copy::ImageInfoAndAllocData);
        NiftiImage voxelBasedGrad3d(deformationField3d, NiftiImage::Copy::ImageInfoAndAllocData);
        auto voxelBasedGrad2dPtr = voxelBasedGrad2d.data();
        auto voxelBasedGrad3dPtr = voxelBasedGrad3d.data();
        for (auto i = 0; i < voxelBasedGrad2d.nVoxels(); i++)
            voxelBasedGrad2dPtr[i] = distr(gen);
        for (auto i = 0; i < voxelBasedGrad3d.nVoxels(); i++)
            voxelBasedGrad3dPtr[i] = distr(gen);

        // Fill the matrices with random values
        voxelBasedGrad2d->sform_code = 0;
        voxelBasedGrad3d->sform_code = 1;
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                voxelBasedGrad2d->qto_ijk.m[j][k] = j == k ? distr(gen) : 0;
                voxelBasedGrad3d->sto_ijk.m[j][k] = j == k ? distr(gen) : 0;
                deformationField2d->sto_xyz.m[j][k] = j == k ? distr(gen) : 0;
                deformationField3d->sto_xyz.m[j][k] = j == k ? distr(gen) : 0;
            }
        }
        voxelBasedGrad2d->qto_xyz = nifti_mat44_inverse(voxelBasedGrad2d->qto_ijk);
        voxelBasedGrad3d->sto_xyz = nifti_mat44_inverse(voxelBasedGrad3d->sto_ijk);

        // Add the test data
        vector<TestData> testData;
        testData.emplace_back(TestData(
            "2D",
            std::move(reference2d),
            std::move(deformationField2d),
            std::move(controlPointGrid2d),
            std::move(controlPointGridBw2d),
            std::move(voxelBasedGrad2d)
        ));
        testData.emplace_back(TestData(
            "3D",
            std::move(reference3d),
            std::move(deformationField3d),
            std::move(controlPointGrid3d),
            std::move(controlPointGridBw3d),
            std::move(voxelBasedGrad3d)
        ));

        // Create the platforms
        Platform platformCpu(PlatformType::Cpu);
        Platform platformCuda(PlatformType::Cuda);

        for (auto&& testData : testData) {
            // Get the test data
            auto&& [testName, reference, defField, controlPointGrid, controlPointGridBw, voxelBasedGrad] = testData;

            // Create images
            NiftiImage referenceCpu(reference), referenceCuda(reference);
            NiftiImage referenceBwCpu(reference), referenceBwCuda(reference);
            NiftiImage defFieldCpu(defField), defFieldCuda(defField);
            NiftiImage cppCpu(controlPointGrid), cppCuda(controlPointGrid);
            NiftiImage cppBwCpu(controlPointGridBw), cppBwCuda(controlPointGridBw);

            // Create the contents
            unique_ptr<F3dContent> contentCpu{ new F3dContent(referenceCpu, referenceCpu, cppCpu) };
            unique_ptr<F3dContent> contentCuda{ new CudaF3dContent(referenceCuda, referenceCuda, cppCuda) };
            unique_ptr<F3dContent> contentBwCpu{ new F3dContent(referenceBwCpu, referenceBwCpu, cppBwCpu) };
            unique_ptr<F3dContent> contentBwCuda{ new CudaF3dContent(referenceBwCuda, referenceBwCuda, cppBwCuda) };

            // Set the deformation fields
            contentCpu->SetDeformationField(defFieldCpu.disown());
            contentCuda->SetDeformationField(defFieldCuda.disown());

            // Set the voxel-based measure gradient images
            NiftiImage voxelGrad = contentCpu->GetVoxelBasedMeasureGradient();
            voxelGrad->sform_code = voxelBasedGrad->sform_code;
            voxelGrad->qto_ijk = voxelBasedGrad->qto_ijk;
            voxelGrad->qto_xyz = voxelBasedGrad->qto_xyz;
            voxelGrad->sto_ijk = voxelBasedGrad->sto_ijk;
            voxelGrad->sto_xyz = voxelBasedGrad->sto_xyz;
            voxelGrad.copyData(voxelBasedGrad);
            voxelGrad.disown();
            contentCpu->UpdateVoxelBasedMeasureGradient();
            voxelGrad = contentCuda->DefContent::GetVoxelBasedMeasureGradient();
            voxelGrad->sform_code = voxelBasedGrad->sform_code;
            voxelGrad->qto_ijk = voxelBasedGrad->qto_ijk;
            voxelGrad->qto_xyz = voxelBasedGrad->qto_xyz;
            voxelGrad->sto_ijk = voxelBasedGrad->sto_ijk;
            voxelGrad->sto_xyz = voxelBasedGrad->sto_xyz;
            voxelGrad.copyData(voxelBasedGrad);
            voxelGrad.disown();
            contentCuda->UpdateVoxelBasedMeasureGradient();

            // Create the computes
            unique_ptr<Compute> computeCpu{ platformCpu.CreateCompute(*contentCpu) };
            unique_ptr<Compute> computeCuda{ platformCuda.CreateCompute(*contentCuda) };

            // Resample gradient
            computeCpu->ExponentiateGradient(*contentBwCpu);
            computeCuda->ExponentiateGradient(*contentBwCuda);

            // Get the results
            NiftiImage voxelGradCpu(contentCpu->GetVoxelBasedMeasureGradient(), NiftiImage::Copy::Image);
            NiftiImage voxelGradCuda(contentCuda->GetVoxelBasedMeasureGradient(), NiftiImage::Copy::Image);

            // Save for testing
            testCases.push_back({ testName, std::move(voxelGradCpu), std::move(voxelGradCuda) });
        }
    }
};

TEST_CASE_METHOD(ExponentiateGradientTest, "Regression Exponentiate Gradient", "[regression]") {
    // Loop over all generated test cases
    for (auto&& testCase : testCases) {
        // Retrieve test information
        auto&& [sectionName, voxelGradCpu, voxelGradCuda] = testCase;

        SECTION(sectionName) {
            NR_COUT << "\n**************** Section " << sectionName << " ****************" << std::endl;

            // Increase the precision for the output
            NR_COUT << std::fixed << std::setprecision(10);

            // Check the results
            const auto voxelGradCpuPtr = voxelGradCpu.data();
            const auto voxelGradCudaPtr = voxelGradCuda.data();
            for (size_t i = 0; i < voxelGradCpu.nVoxels(); i++) {
                const float voxelGradCpuVal = voxelGradCpuPtr[i];
                const float voxelGradCudaVal = voxelGradCudaPtr[i];
                const float diff = abs(voxelGradCpuVal - voxelGradCudaVal);
                if (diff > 0) {
                    NR_COUT << "[i]=" << i;
                    NR_COUT << " | diff=" << diff;
                    NR_COUT << " | CPU=" << voxelGradCpuVal;
                    NR_COUT << " | CUDA=" << voxelGradCudaVal << std::endl;
                }
                REQUIRE(diff == 0);
            }
        }
    }
}
