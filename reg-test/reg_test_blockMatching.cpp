#include "reg_test_common.h"
#include "_reg_blockMatching.h"
#include "CpuBlockMatchingKernel.h"
#include "CudaBlockMatchingKernel.h"
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
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> distr(0, 1);

        // Create a reference 2D and 3D images
        constexpr NiftiImage::dim_t size = 64;
        vector<NiftiImage::dim_t> dim{ size, size };
        NiftiImage reference2d(dim, NIFTI_TYPE_FLOAT32);
        dim.push_back(size);
        NiftiImage reference3d(dim, NIFTI_TYPE_FLOAT32);

        // Fill images with random values
        const auto ref2dPtr = reference2d.data();
        for(auto ref2dItr = ref2dPtr.begin(); ref2dItr!=ref2dPtr.end(); ++ref2dItr){
            *ref2dItr = distr(gen);
        }
        const auto ref3dPtr = reference3d.data();
        for(auto ref3dItr = ref3dPtr.begin(); ref3dItr!=ref3dPtr.end(); ++ref3dItr){
            *ref3dItr = distr(gen);
        }

        // Create a translation matrix to apply OFFSET voxels along each axis
        mat44 translationMatrix;
        reg_mat44_eye(&translationMatrix);
        translationMatrix.m[0][3] = -OFFSET;
        translationMatrix.m[1][3] = -OFFSET;
        translationMatrix.m[2][3] = -OFFSET;

        // Create a mask so that voxel at the boundary are ignored
        int *mask2D = new int[reference2d.nVoxels()];
        int *mask3D = new int[reference3d.nVoxels()];
        int *mask2dPtr = mask2D;
        int *mask3dPtr = mask3D;
        // set all values to -1
        for(int y=0; y<reference2d->ny;++y)
            for(int x=0; x<reference2d->nx;++x)
                *mask2dPtr++ = -1;
        for(int z=0; z<reference3d->nz;++z)
            for(int y=0; y<reference3d->ny;++y)
                for(int x=0; x<reference3d->nx;++x)
                    *mask3dPtr++ = -1;
        // Set the internal values to 1
        for(int y=OFFSET; y<reference2d->ny-OFFSET;++y){
            mask2dPtr = &mask2D[y*reference2d->nx+OFFSET];
            for(int x=OFFSET; x<reference2d->nx-OFFSET;++x){
                *mask2dPtr++ = 1;
            }
        }
        for(int z=OFFSET; z<reference3d->nz-OFFSET;++z){
            for(int y=OFFSET; y<reference3d->ny-OFFSET;++y){
                mask3dPtr = &mask3D[(z*reference3d->ny+y)*reference3d->nx+OFFSET];
                for(int x=OFFSET; x<reference3d->nx-OFFSET;++x){
                    *mask3dPtr++ = 1;
                }
            }
        }

        // Apply the transformation in 2D
        unique_ptr<AladinContent> contentResampling2D{ new AladinContent(
                reference2d,
                reference2d
        ) };
        contentResampling2D->SetTransformationMatrix(&translationMatrix);
        std::unique_ptr<AffineDeformationFieldKernel> affineDeformKernel2D{
            new CpuAffineDeformationFieldKernel(contentResampling2D.get())
            };
        affineDeformKernel2D->Calculate();
        std::unique_ptr<ResampleImageKernel> resampleKernel2D{
            new CpuResampleImageKernel(contentResampling2D.get())
            };
        resampleKernel2D->Calculate(0, std::numeric_limits<float>::quiet_NaN());

        // Apply the transformation in 3D
        unique_ptr<AladinContent> contentResampling3D{ new AladinContent(
                reference3d,
                reference3d
        ) };
        contentResampling3D->SetTransformationMatrix(&translationMatrix);
        std::unique_ptr<AffineDeformationFieldKernel> affineDeformKernel3D{
            new CpuAffineDeformationFieldKernel(contentResampling3D.get())
            };
        affineDeformKernel3D->Calculate();
        std::unique_ptr<ResampleImageKernel> resampleKernel3D{
            new CpuResampleImageKernel(contentResampling3D.get())
            };
        resampleKernel3D->Calculate(0, 0);

        // Create the data container for the regression test
        vector<TestData> testData;
        testData.emplace_back(TestData(
            "BlockMatching 2D",
            reference2d,
            NiftiImage(contentResampling2D->GetWarped()),
            mask2D
        ));
        contentResampling2D.release();
        testData.emplace_back(TestData(
            "BlockMatching 3D",
            reference3d,
            NiftiImage(contentResampling3D->GetWarped()),
            mask3D
        ));
        contentResampling3D.release();

        for (auto&& data : testData) {
            // Get the test data
            auto&& [testName, reference, warped, mask] = data;

            for (auto&& platformType : PlatformTypes) {

                // Create images
                NiftiImage referenceTest(reference);
                NiftiImage warpedTest(warped);

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
                    1) };
                content->SetWarped(warpedTest.disown());

                // Inititialise the block matching
                unique_ptr<Kernel> bmKernel{ platform->CreateKernel(
                    BlockMatchingKernel::GetName(), content.get()
                    ) };

                // Do the computation
                bmKernel->castTo<BlockMatchingKernel>()->Calculate();

                // Retrieve the information
                unique_ptr<_reg_blockMatchingParam> blockMatchingParams{
                    new _reg_blockMatchingParam(content->GetBlockMatchingParams())
                    };
    
                testCases.push_back({ testName + " " + platform->GetName(), std::move(blockMatchingParams) });
            } // loop over platforms
        }
        delete mask2D;
        delete mask3D;
    }
};

TEST_CASE_METHOD(BMTest, "BlockMatching", "[unit]") {
    // Loop over all generated test cases
    for (auto&& testCase : this->testCases) {
        // Retrieve test information
        auto&& [testName, blockMatchingParams] = testCase;

        SECTION(testName) {
            std::cout << "\n**************** Section " << testName << " ****************" << std::endl;

            // Loop over the block and ensure all values are identical
            for (int b = 0; b < blockMatchingParams->activeBlockNumber; ++b) {
                for(int d = 0; d<(int)blockMatchingParams->dim; ++d){
                    const int i = b*(int)blockMatchingParams->dim+d;
                    const auto diffPos = blockMatchingParams->warpedPosition[i] - blockMatchingParams->referencePosition[i];
                    if(fabs(diffPos - OFFSET) > EPS){
                        std::cout << "[" << b << "/" << blockMatchingParams->activeBlockNumber << ":" << d << "] ";
                        std::cout << diffPos << std::endl; std::cout.flush();
                    }
                    REQUIRE(fabs(diffPos - OFFSET) < EPS);
                }
            }
        }
    }
}
