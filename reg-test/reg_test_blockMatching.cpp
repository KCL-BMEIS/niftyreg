#include "reg_test_common.h"
#include "_reg_blockMatching.h"
#include "BlockMatchingKernel.h"
#include "CpuAffineDeformationFieldKernel.h"
#include "CpuResampleImageKernel.h"


/**
 *  Block matching regression test to ensure the CPU and CUDA versions yield the same output
 */

#define OFFSET 1

class BMTest {
protected:
    using TestData = std::tuple<std::string, NiftiImage, NiftiImage, int*>;
    using TestCase = std::tuple<std::string, unique_ptr<_reg_blockMatchingParam>>;

    inline static vector<TestCase> testCases;

public:
    BMTest() {
        if (!testCases.empty())
            return;

        // Create a random number generator
        std::mt19937 gen(0);
        std::uniform_real_distribution<float> distr(0, 1);

        // Create 2D and 3D reference images
        constexpr NiftiImage::dim_t size = 64;
        vector<NiftiImage::dim_t> dim{ size, size };
        NiftiImage reference2d(dim, NIFTI_TYPE_FLOAT32);
        dim.push_back(size);
        NiftiImage reference3d(dim, NIFTI_TYPE_FLOAT32);

        // Fill images with random values
        const auto ref2dPtr = reference2d.data();
        for (auto ref2dItr = ref2dPtr.begin(); ref2dItr != ref2dPtr.end(); ++ref2dItr) {
            *ref2dItr = distr(gen);
        }
        const auto ref3dPtr = reference3d.data();
        for (auto ref3dItr = ref3dPtr.begin(); ref3dItr != ref3dPtr.end(); ++ref3dItr) {
            *ref3dItr = distr(gen);
        }

        // Create a translation matrix to apply OFFSET voxels along each axis
        mat44 translationMatrix;
        Mat44Eye(&translationMatrix);
        translationMatrix.m[0][3] = -OFFSET;
        translationMatrix.m[1][3] = -OFFSET;
        translationMatrix.m[2][3] = -OFFSET;

        // Create a mask so that voxel at the boundary are ignored
        unique_ptr<int[]> mask2d{ new int[reference2d.nVoxels()] };
        unique_ptr<int[]> mask3d{ new int[reference3d.nVoxels()] };
        int *mask2dPtr = mask2d.get();
        int *mask3dPtr = mask3d.get();
        // set all values to -1
        for (int y = 0; y < reference2d->ny; ++y)
            for (int x = 0; x < reference2d->nx; ++x)
                *mask2dPtr++ = -1;
        for (int z = 0; z < reference3d->nz; ++z)
            for (int y = 0; y < reference3d->ny; ++y)
                for (int x = 0; x < reference3d->nx; ++x)
                    *mask3dPtr++ = -1;
        // Set the internal values to 1
        for (int y = OFFSET; y < reference2d->ny - OFFSET; ++y) {
            mask2dPtr = &mask2d[y * reference2d->nx + OFFSET];
            for (int x = OFFSET; x < reference2d->nx - OFFSET; ++x) {
                *mask2dPtr++ = 1;
            }
        }
        for (int z = OFFSET; z < reference3d->nz - OFFSET; ++z) {
            for (int y = OFFSET; y < reference3d->ny - OFFSET; ++y) {
                mask3dPtr = &mask3d[(z * reference3d->ny + y) * reference3d->nx + OFFSET];
                for (int x = OFFSET; x < reference3d->nx - OFFSET; ++x) {
                    *mask3dPtr++ = 1;
                }
            }
        }

        // Apply the transformation in 2D
        unique_ptr<AladinContent> contentResampling2d{ new AladinContent(reference2d, reference2d) };
        contentResampling2d->SetTransformationMatrix(&translationMatrix);
        unique_ptr<AffineDeformationFieldKernel> affineDeformKernel2d{ new CpuAffineDeformationFieldKernel(contentResampling2d.get()) };
        affineDeformKernel2d->Calculate();
        unique_ptr<ResampleImageKernel> resampleKernel2d{ new CpuResampleImageKernel(contentResampling2d.get()) };
        resampleKernel2d->Calculate(0, std::numeric_limits<float>::quiet_NaN());

        // Apply the transformation in 3D
        unique_ptr<AladinContent> contentResampling3d{ new AladinContent(reference3d, reference3d) };
        contentResampling3d->SetTransformationMatrix(&translationMatrix);
        unique_ptr<AffineDeformationFieldKernel> affineDeformKernel3d{ new CpuAffineDeformationFieldKernel(contentResampling3d.get()) };
        affineDeformKernel3d->Calculate();
        unique_ptr<ResampleImageKernel> resampleKernel3d{ new CpuResampleImageKernel(contentResampling3d.get()) };
        resampleKernel3d->Calculate(0, 0);

        // Create the data container for the regression test
        vector<TestData> testData;
        testData.emplace_back(TestData(
            "BlockMatching 2D",
            reference2d,
            contentResampling2d->GetWarped(),
            mask2d.get()
        ));
        contentResampling2d.release();
        testData.emplace_back(TestData(
            "BlockMatching 3D",
            reference3d,
            contentResampling3d->GetWarped(),
            mask3d.get()
        ));
        contentResampling3d.release();

        for (auto&& data : testData) {
            // Get the test data
            auto&& [testName, reference, warped, mask] = data;

            for (auto&& platformType : PlatformTypes) {
                // Create images
                NiftiImage referenceTest(reference);

                // Create the contents
                shared_ptr<Platform> platform{ new Platform(platformType) };
                unique_ptr<AladinContentCreator> contentCreator{
                    dynamic_cast<AladinContentCreator*>(platform->CreateContentCreator(ContentType::Aladin))
                };
                unique_ptr<AladinContent> content{ contentCreator->Create(
                    referenceTest,
                    referenceTest,
                    mask,
                    nullptr,
                    sizeof(float),
                    100,
                    100,
                    1
                ) };
                content->SetWarped(NiftiImage(warped));

                // Initialise the block matching
                unique_ptr<Kernel> bmKernel{ platform->CreateKernel(BlockMatchingKernel::GetName(), content.get()) };

                // Do the computation
                bmKernel->castTo<BlockMatchingKernel>()->Calculate();

                // Retrieve the information
                unique_ptr<_reg_blockMatchingParam> blockMatchingParams{ new _reg_blockMatchingParam(content->GetBlockMatchingParams()) };

                testCases.push_back({ testName + " " + platform->GetName(), std::move(blockMatchingParams) });
            } // loop over platforms
        }
    }
};

TEST_CASE_METHOD(BMTest, "Block Matching", "[unit]") {
    // Loop over all generated test cases
    for (auto&& testCase : this->testCases) {
        // Retrieve test information
        auto&& [testName, blockMatchingParams] = testCase;

        SECTION(testName) {
            NR_COUT << "\n**************** Section " << testName << " ****************" << std::endl;

            // Increase the precision for the output
            NR_COUT << std::fixed << std::setprecision(10);

            // Loop over the block and ensure all values are identical
            for (int b = 0; b < blockMatchingParams->activeBlockNumber; ++b) {
                for (int d = 0; d < (int)blockMatchingParams->dim; ++d) {
                    const int i = b * (int)blockMatchingParams->dim + d;
                    const auto diffPos = blockMatchingParams->warpedPosition[i] - blockMatchingParams->referencePosition[i];
                    const auto diff = abs(diffPos - OFFSET);
                    if (diff > 0) {
                        NR_COUT << "[" << b << "/" << blockMatchingParams->activeBlockNumber << ":" << d << "] ";
                        NR_COUT << diffPos << std::endl;
                    }
                    REQUIRE(diff == 0);
                }
            }
        }
    }
}

TEST_CASE("Block matching recovers per-region displacements", "[unit]") {
    /*
        The case above applies ONE translation to the whole image, so a block matcher that somehow
        produced a single global answer would pass it. Here the two halves of the image are shifted
        by different integer amounts (+1 and +2 voxels along x), the seam and image borders are
        masked out, and every active block must report exactly the shift of the half it sits in -
        per-block correctness, not a global average.
    */
    constexpr int size = 64, seam = 32, margin = 8;
    constexpr int shiftLeft = 1, shiftRight = 2;
    std::mt19937 gen(0);
    std::uniform_real_distribution<float> distr(0, 1);

    for (const bool is3D : { false, true }) {
        SECTION(is3D ? "3D" : "2D") {
            std::vector<NiftiImage::dim_t> dims(is3D ? 3 : 2, size);
            NiftiImage reference(dims, NIFTI_TYPE_FLOAT32);
            NiftiImage warped(dims, NIFTI_TYPE_FLOAT32);
            const int nx = reference->nx, ny = reference->ny, nz = reference->nz;
            auto refPtr = reference.data();
            for (size_t i = 0; i < reference.nVoxels(); ++i)
                refPtr[i] = distr(gen);

            // warped(x) = ref(x - s): content moves by +s, block matching reports +s
            auto warPtr = warped.data();
            for (int k = 0; k < nz; ++k)
                for (int j = 0; j < ny; ++j)
                    for (int i = 0; i < nx; ++i) {
                        const int s = i < seam ? shiftLeft : shiftRight;
                        const int src = i - s;
                        const size_t index = (static_cast<size_t>(k) * ny + j) * nx + i;
                        warPtr[index] = src >= 0
                            ? static_cast<float>(refPtr[(static_cast<size_t>(k) * ny + j) * nx + src])
                            : std::numeric_limits<float>::quiet_NaN();
                    }

            // Mask out the borders and the seam band, where a block's search window would mix shifts
            std::vector<int> mask(reference.nVoxels(), -1);
            for (int k = 0; k < nz; ++k)
                for (int j = margin; j < ny - margin; ++j)
                    for (int i = margin; i < nx - margin; ++i) {
                        if (i >= seam - margin && i < seam + margin) continue;
                        mask[(static_cast<size_t>(k) * ny + j) * nx + i] = 1;
                    }
            if (is3D)   // trim z borders too
                for (int k = 0; k < nz; ++k)
                    if (k < margin || k >= nz - margin)
                        std::fill_n(mask.begin() + static_cast<size_t>(k) * ny * nx, ny * nx, -1);

            Platform platform(PlatformType::Cpu);
            unique_ptr<AladinContentCreator> creator{
                dynamic_cast<AladinContentCreator*>(platform.CreateContentCreator(ContentType::Aladin)) };
            NiftiImage ref(reference);
            unique_ptr<AladinContent> content{ creator->Create(ref, ref, mask.data(), nullptr,
                                                               sizeof(float), 100, 100, 1) };
            content->SetWarped(NiftiImage(warped));
            unique_ptr<Kernel> bmKernel{ platform.CreateKernel(BlockMatchingKernel::GetName(), content.get()) };
            bmKernel->castTo<BlockMatchingKernel>()->Calculate();
            unique_ptr<_reg_blockMatchingParam> params{ new _reg_blockMatchingParam(content->GetBlockMatchingParams()) };

            size_t leftBlocks = 0, rightBlocks = 0;
            for (int b = 0; b < params->activeBlockNumber; ++b) {
                const float *refPos = &params->referencePosition[b * params->dim];
                const float *warPos = &params->warpedPosition[b * params->dim];
                if (warPos[0] != warPos[0]) continue;   // unmatched block
                const int expected = refPos[0] < seam ? shiftLeft : shiftRight;
                (refPos[0] < seam ? leftBlocks : rightBlocks)++;
                INFO("block at (" << refPos[0] << "," << refPos[1] << "): expected shift " << expected);
                REQUIRE(warPos[0] - refPos[0] == static_cast<float>(expected));
                REQUIRE(warPos[1] - refPos[1] == 0.f);
                if (is3D) REQUIRE(warPos[2] - refPos[2] == 0.f);
            }
            NR_COUT << "  " << (is3D ? "3D" : "2D") << ": " << leftBlocks << " left blocks (+1), "
                    << rightBlocks << " right blocks (+2)" << std::endl;
            // Both regions must actually be represented, or the per-region claim is vacuous
            REQUIRE(leftBlocks > 0);
            REQUIRE(rightBlocks > 0);
        }
    }
}

TEST_CASE("Block matching excludes zero-variance blocks", "[unit]") {
    /*
        _reg_set_active_blocks marks a block unusable when its variance is not strictly positive.
        An image that is constant everywhere except one textured
        quadrant must therefore yield active blocks only inside that quadrant - a matcher fed flat
        blocks would otherwise report meaningless correspondences from a constant signal.
    */
    constexpr int size = 64, textured = 24;
    std::mt19937 gen(1);
    std::uniform_real_distribution<float> distr(0, 1);
    for (const bool is3D : { false, true }) {
        SECTION(is3D ? "3D" : "2D") {
            std::vector<NiftiImage::dim_t> dims(is3D ? 3 : 2, size);
            NiftiImage reference(dims, NIFTI_TYPE_FLOAT32);
            const int nx = reference->nx, ny = reference->ny, nz = reference->nz;
            auto refPtr = reference.data();
            for (int k = 0; k < nz; ++k)
                for (int j = 0; j < ny; ++j)
                    for (int i = 0; i < nx; ++i) {
                        const bool inTexture = i < textured && j < textured && (!is3D || k < textured);
                        refPtr[(static_cast<size_t>(k) * ny + j) * nx + i] = inTexture ? distr(gen) : 0.5f;
                    }

            Platform platform(PlatformType::Cpu);
            unique_ptr<AladinContentCreator> creator{
                dynamic_cast<AladinContentCreator*>(platform.CreateContentCreator(ContentType::Aladin)) };
            NiftiImage ref(reference);
            unique_ptr<AladinContent> content{ creator->Create(ref, ref, nullptr, nullptr,
                                                               sizeof(float), 100, 100, 1) };
            content->SetWarped(NiftiImage(reference));
            unique_ptr<Kernel> bmKernel{ platform.CreateKernel(BlockMatchingKernel::GetName(), content.get()) };
            bmKernel->castTo<BlockMatchingKernel>()->Calculate();
            unique_ptr<_reg_blockMatchingParam> params{ new _reg_blockMatchingParam(content->GetBlockMatchingParams()) };

            REQUIRE(params->activeBlockNumber > 0);
            for (int b = 0; b < params->activeBlockNumber; ++b) {
                const float *refPos = &params->referencePosition[b * params->dim];
                // Every active block must touch the textured quadrant: a block whose whole 4-voxel
                // extent lies in the constant region has zero variance and must have been excluded
                INFO("active block at (" << refPos[0] << "," << refPos[1] << ")");
                REQUIRE(refPos[0] < static_cast<float>(textured));
                REQUIRE(refPos[1] < static_cast<float>(textured));
                if (is3D) REQUIRE(refPos[2] < static_cast<float>(textured));
            }
            NR_COUT << "  " << (is3D ? "3D" : "2D") << ": " << params->activeBlockNumber
                    << " active blocks, all within the textured quadrant" << std::endl;
        }
    }
}
