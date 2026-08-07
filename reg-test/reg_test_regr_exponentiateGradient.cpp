#include "reg_test_common.h"
#include "CudaF3dContent.h"

/**
 *  Gradient accumulation through exponentiation, CPU against CUDA
 *  (reg_f3d2::ExponentiateGradient, implemented by Compute::ExponentiateGradient).
 *
 *  The backward velocity grid's intent_p2 sets the number of squaring steps, and therefore how many
 *  times the composition loop runs. It has to be set explicitly: at its default of 0 the loop body
 *  never executes and the function collapses to a division by 2^0, leaving the gradient untouched on
 *  both backends and comparing equal for the wrong reason. Each case sets it to 6, the value reg_f3d2
 *  uses, and asserts the gradient actually changed before comparing the two.
 *
 *  Each backend is also run twice, resetting the gradient in between, so that the second call has to
 *  reproduce the first. Both implementations pool scratch buffers across calls, and a field left dirty
 *  by one call can only show up in a later one.
**/

class ExponentiateGradientTest {
protected:
    using TestData = std::tuple<std::string, NiftiImage, NiftiImage, NiftiImage, NiftiImage, NiftiImage>;
    // name, input gradient, cpu 1st call, cpu 2nd call, cuda 1st call, cuda 2nd call
    using TestCase = std::tuple<std::string, NiftiImage, NiftiImage, NiftiImage, NiftiImage, NiftiImage>;

    static constexpr int squaringSteps = 6;   // what reg_f3d2 sets on the velocity grid

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
        // The number of squaring steps drives the composition loop in Compute::ExponentiateGradient.
        // It has to be set: intent_p2 defaults to 0, which makes the loop body run zero times and the
        // whole function collapse to a division by 2^0, i.e. this test would compare a no-op.
        controlPointGridBw2d->intent_p2 = squaringSteps;
        controlPointGridBw3d->intent_p2 = squaringSteps;
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
            contentCpu->SetDeformationField(std::move(defFieldCpu));
            contentCuda->SetDeformationField(std::move(defFieldCuda));

            // Set the voxel-based measure gradient images
            NiftiImage& voxelGradCpu = contentCpu->GetVoxelBasedMeasureGradient();
            voxelGradCpu->sform_code = voxelBasedGrad->sform_code;
            voxelGradCpu->qto_ijk = voxelBasedGrad->qto_ijk;
            voxelGradCpu->qto_xyz = voxelBasedGrad->qto_xyz;
            voxelGradCpu->sto_ijk = voxelBasedGrad->sto_ijk;
            voxelGradCpu->sto_xyz = voxelBasedGrad->sto_xyz;
            voxelGradCpu.copyData(voxelBasedGrad);
            contentCpu->UpdateVoxelBasedMeasureGradient();
            NiftiImage& voxelGradCuda = contentCuda->DefContent::GetVoxelBasedMeasureGradient();
            voxelGradCuda->sform_code = voxelBasedGrad->sform_code;
            voxelGradCuda->qto_ijk = voxelBasedGrad->qto_ijk;
            voxelGradCuda->qto_xyz = voxelBasedGrad->qto_xyz;
            voxelGradCuda->sto_ijk = voxelBasedGrad->sto_ijk;
            voxelGradCuda->sto_xyz = voxelBasedGrad->sto_xyz;
            voxelGradCuda.copyData(voxelBasedGrad);
            contentCuda->UpdateVoxelBasedMeasureGradient();

            // Create the computes
            unique_ptr<Compute> computeCpu{ platformCpu.CreateCompute(*contentCpu) };
            unique_ptr<Compute> computeCuda{ platformCuda.CreateCompute(*contentCuda) };

            // Exponentiate the gradient
            computeCpu->ExponentiateGradient(*contentBwCpu);
            computeCuda->ExponentiateGradient(*contentBwCuda);
            NiftiImage gradCpu1(contentCpu->GetVoxelBasedMeasureGradient(), NiftiImage::Copy::Image);
            NiftiImage gradCuda1(contentCuda->GetVoxelBasedMeasureGradient(), NiftiImage::Copy::Image);

            // Reset the gradient and run again, so the second call has to reproduce the first while
            // reusing whatever scratch the first one pooled
            contentCpu->GetVoxelBasedMeasureGradient().copyData(voxelBasedGrad);
            contentCpu->UpdateVoxelBasedMeasureGradient();
            contentCuda->DefContent::GetVoxelBasedMeasureGradient().copyData(voxelBasedGrad);
            contentCuda->UpdateVoxelBasedMeasureGradient();
            computeCpu->ExponentiateGradient(*contentBwCpu);
            computeCuda->ExponentiateGradient(*contentBwCuda);

            // Save the results for testing
            testCases.push_back({ testName,
                                  NiftiImage(voxelBasedGrad, NiftiImage::Copy::Image),
                                  std::move(gradCpu1),
                                  std::move(contentCpu->GetVoxelBasedMeasureGradient()),
                                  std::move(gradCuda1),
                                  std::move(contentCuda->GetVoxelBasedMeasureGradient()) });
        }
    }
};

TEST_CASE_METHOD(ExponentiateGradientTest, "Regression Exponentiate Gradient", "[regression]") {
    // Loop over all generated test cases
    for (auto&& testCase : testCases) {
        // Retrieve test information
        auto&& [sectionName, inputGrad, voxelGradCpu, voxelGradCpu2, voxelGradCuda, voxelGradCuda2] = testCase;

        SECTION(sectionName) {
            NR_COUT << "\n**************** Section " << sectionName << " ****************" << std::endl;

            // Increase the precision for the output
            NR_COUT << std::fixed << std::setprecision(10);

            const auto inputPtr = inputGrad.data();
            const auto voxelGradCpuPtr = voxelGradCpu.data();
            const auto voxelGradCpu2Ptr = voxelGradCpu2.data();
            const auto voxelGradCudaPtr = voxelGradCuda.data();
            const auto voxelGradCuda2Ptr = voxelGradCuda2.data();

            // The exponentiation has to have done something: with intent_p2 left at 0 the loop body
            // never runs and every check below would pass on an untouched gradient
            bool changed = false;
            for (size_t i = 0; i < voxelGradCpu.nVoxels() && !changed; i++)
                changed = static_cast<float>(voxelGradCpuPtr[i]) != static_cast<float>(inputPtr[i]);
            REQUIRE(changed);

            // Check the results
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

                // Repeating the call has to give the same answer, whatever scratch was pooled
                REQUIRE(static_cast<float>(voxelGradCpu2Ptr[i]) == voxelGradCpuVal);
                REQUIRE(static_cast<float>(voxelGradCuda2Ptr[i]) == voxelGradCudaVal);
            }
        }
    }
}
