#include "reg_test_common.h"
#include "CudaDefContent.h"

/**
 *  Resample gradient regression test to ensure the CPU and CUDA versions yield the same output
**/

class ResampleGradientTest {
protected:
    using TestData = std::tuple<std::string, NiftiImage, NiftiImage, NiftiImage>;
    using TestCase = std::tuple<std::string, NiftiImage, NiftiImage>;

    inline static vector<TestCase> testCases;

public:
    ResampleGradientTest() {
        if (!testCases.empty())
            return;

        // Create a random number generator
        std::mt19937 gen(0);
        std::uniform_real_distribution<float> distr(-1, 1);

        // Create reference images
        constexpr NiftiImage::dim_t dimSize = 4;
        NiftiImage reference2d({ dimSize, dimSize }, NIFTI_TYPE_FLOAT32);
        NiftiImage reference3d({ dimSize, dimSize, dimSize }, NIFTI_TYPE_FLOAT32);

        // Create deformation fields and fill them with random values
        NiftiImage deformationField2d = CreateDeformationField(reference2d);
        NiftiImage deformationField3d = CreateDeformationField(reference3d);
        auto deformationField2dPtr = deformationField2d.data();
        auto deformationField3dPtr = deformationField3d.data();
        for (size_t i = 0; i < deformationField2d.nVoxels(); i++)
            deformationField2dPtr[i] = distr(gen);
        for (size_t i = 0; i < deformationField3d.nVoxels(); i++)
            deformationField3dPtr[i] = distr(gen);

        // Create voxel-based measure gradients and fill them with random values
        NiftiImage voxelBasedGrad2d(deformationField2d, NiftiImage::Copy::ImageInfoAndAllocData);
        NiftiImage voxelBasedGrad3d(deformationField3d, NiftiImage::Copy::ImageInfoAndAllocData);
        auto voxelBasedGrad2dPtr = voxelBasedGrad2d.data();
        auto voxelBasedGrad3dPtr = voxelBasedGrad3d.data();
        for (size_t i = 0; i < voxelBasedGrad2d.nVoxels(); i++)
            voxelBasedGrad2dPtr[i] = distr(gen);
        for (size_t i = 0; i < voxelBasedGrad3d.nVoxels(); i++)
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
            std::move(voxelBasedGrad2d)
        ));
        testData.emplace_back(TestData(
            "3D",
            std::move(reference3d),
            std::move(deformationField3d),
            std::move(voxelBasedGrad3d)
        ));

        // Create the platforms
        Platform platformCpu(PlatformType::Cpu);
        Platform platformCuda(PlatformType::Cuda);

        for (auto&& testData : testData) {
            // Get the test data
            auto&& [testName, reference, defField, voxelBasedGrad] = testData;

            // Create images
            NiftiImage referenceCpu(reference), referenceCuda(reference);
            NiftiImage defFieldCpu(defField), defFieldCuda(defField);

            // Create the contents
            unique_ptr<DefContent> contentCpu{ new DefContent(referenceCpu, referenceCpu) };
            unique_ptr<DefContent> contentCuda{ new CudaDefContent(referenceCuda, referenceCuda) };

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
            contentCpu->UpdateVoxelBasedMeasureGradient();
            voxelGrad = contentCuda->DefContent::GetVoxelBasedMeasureGradient();
            voxelGrad->sform_code = voxelBasedGrad->sform_code;
            voxelGrad->qto_ijk = voxelBasedGrad->qto_ijk;
            voxelGrad->qto_xyz = voxelBasedGrad->qto_xyz;
            voxelGrad->sto_ijk = voxelBasedGrad->sto_ijk;
            voxelGrad->sto_xyz = voxelBasedGrad->sto_xyz;
            voxelGrad.copyData(voxelBasedGrad);
            contentCuda->UpdateVoxelBasedMeasureGradient();

            // Create the computes
            unique_ptr<Compute> computeCpu{ platformCpu.CreateCompute(*contentCpu) };
            unique_ptr<Compute> computeCuda{ platformCuda.CreateCompute(*contentCuda) };

            // Resample gradient
            NiftiImage warpedCpu = computeCpu->ResampleGradient(1, -2.f);
            NiftiImage warpedCuda = computeCuda->ResampleGradient(1, -2.f);

            // Save for testing
            testCases.push_back({ testName, std::move(warpedCpu), std::move(warpedCuda) });
        }
    }
};

TEST_CASE_METHOD(ResampleGradientTest, "Regression Resample Gradient", "[regression]") {
    // Loop over all generated test cases
    for (auto&& testCase : testCases) {
        // Retrieve test information
        auto&& [sectionName, warpedCpu, warpedCuda] = testCase;

        SECTION(sectionName) {
            NR_COUT << "\n**************** Section " << sectionName << " ****************" << std::endl;

            // Increase the precision for the output
            NR_COUT << std::fixed << std::setprecision(10);

            // Check the results
            const auto warpedCpuPtr = warpedCpu.data();
            const auto warpedCudaPtr = warpedCuda.data();
            for (size_t i = 0; i < warpedCpu.nVoxels(); i++) {
                const float warpedCpuVal = warpedCpuPtr[i];
                const float warpedCudaVal = warpedCudaPtr[i];
                const float diff = abs(warpedCpuVal - warpedCudaVal);
                if (diff > 0) {
                    NR_COUT << "[i]=" << i;
                    NR_COUT << " | diff=" << diff;
                    NR_COUT << " | CPU=" << warpedCpuVal;
                    NR_COUT << " | CUDA=" << warpedCudaVal << std::endl;
                }
                REQUIRE(diff == 0);
            }
        }
    }
}
