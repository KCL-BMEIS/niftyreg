// OpenCL and CUDA are not supported for this test yet
#undef _USE_OPENCL
#undef _USE_CUDA

#include "reg_test_common.h"
#include "_reg_tools.h"
#include "_reg_ReadWriteImage.h"
#include "_reg_nmi.h"

/*
    This test file contains the following unit tests:
    test function: NMI gradient.
    The anylitical formulation is compared against an approximation
*/

class NMIGradientTest {
public:
    NMIGradientTest() {
        if (!testCases.empty())
            return;

        // Create a number generator
        std::mt19937 gen(0);
        // Images will be rescaled between 2 and bin-3
        // Default bin value is 68 (64+4 for Parzen windowing)
        const unsigned binNumber = 8;
        const float padding = 2; //std::numeric_limits<float>::quiet_NaN();
        std::uniform_real_distribution<float> distr(2, binNumber-3);

        // Create reference and floating 2D images
        vector<NiftiImage::dim_t> dim{ 4, 4 };
        NiftiImage reference2d(dim, NIFTI_TYPE_FLOAT32);
        NiftiImage floating2d(dim, NIFTI_TYPE_FLOAT32);

        // Create reference and floating 3D images
        dim.push_back(4);
        NiftiImage reference3d(dim, NIFTI_TYPE_FLOAT32);
        NiftiImage floating3d(dim, NIFTI_TYPE_FLOAT32);

        // Fill images with random values
        auto ref2dPtr = static_cast<float *>(reference2d->data);
        auto flo2dPtr = static_cast<float *>(floating2d->data);
        // Ensure at least one pixel contains the max and one the min
        ref2dPtr[0] = flo2dPtr[1] = 2.f;
        ref2dPtr[1] = flo2dPtr[0] = binNumber-3;
        for (size_t i = 2; i < reference2d.nVoxels(); ++i)
        {
            ref2dPtr[i] = distr(gen);
            flo2dPtr[i] = distr(gen);
        }

        // Fill images with random values
        auto ref3dPtr = reference3d.data();
        auto flo3dPtr = floating3d.data();
        // Ensure at least one pixel contains the max and one the min
        ref3dPtr[0] = flo3dPtr[1] = 2.f;
        ref3dPtr[1] = flo3dPtr[0] = binNumber-3;
        for (size_t i = 2; i < reference3d.nVoxels(); ++i) {
            ref3dPtr[i] = distr(gen);
            flo3dPtr[i] = distr(gen);
        }

        // Create the object to compute the expected values
        vector<TestData> testData;
        testData.emplace_back(TestData(
            "NMI 2D",
            reference2d,
            floating2d
        ));
        testData.emplace_back(TestData(
            "NMI 3D",
            reference3d,
            floating3d
        ));
        for (auto&& data : testData) {
            for (auto&& platformType : PlatformTypes) {
                // Create the platform
                shared_ptr<Platform> platform{ new Platform(platformType) };
                auto td = data;
                auto&& [testName, reference, floating] = td;
                // Create the content creator
                unique_ptr<DefContentCreator> contentCreator{
                    dynamic_cast<DefContentCreator*>(platform->CreateContentCreator(ContentType::Def))
                };
                // Create the content
                unique_ptr<DefContent> content{ contentCreator->Create(reference, floating) };
                // Add some displacements to the deformation field to avoid grid effect
                float *defPtr = static_cast<float *>(content->GetDeformationField()->data);
                for(unsigned index=0; index<content->GetDeformationField()->nvox;++index)
                    defPtr[index] += 0.1f;
                // Compute the warped image given the current transformation
                unique_ptr<Compute> compute{ platform->CreateCompute(*content) };
                compute->ResampleImage(1, padding);
                compute->GetImageGradient(1, padding, 0);
                // Create the measure
                unique_ptr<Measure> measure{ platform->CreateMeasure() };
                // Use NMI as a measure
                unique_ptr<reg_nmi> measure_nmi{ dynamic_cast<reg_nmi*>(measure->Create(MeasureType::Nmi)) };
                measure_nmi->SetTimepointWeight(0, 1.0); // weight initially set to default value of 1.0
                measure_nmi->SetRefAndFloatBinNumbers(binNumber, binNumber, 0);
                measure->Initialise(*measure_nmi, *content);
                // Compute the NMI gradient
                measure_nmi->GetVoxelBasedSimilarityMeasureGradient(0);
                // Create an image to store the gradient values
                NiftiImage gradientImage(content->GetVoxelBasedMeasureGradient(), NiftiImage::Copy::Image);
                // Create an image to store the expected gradient values
                NiftiImage expectedGradientImage(content->GetDeformationField(), NiftiImage::Copy::Image);
                // Apply perturbations to each value in the deformation field
                float *gradPtr = static_cast<float *>(expectedGradientImage->data);
                const float delta = 0.00001;
                for(unsigned index=0; index<expectedGradientImage.nVoxels();++index){
                    float current_value = defPtr[index];
                    // compute the NMI when removing delta(s)
                    defPtr[index] = current_value - delta;
                    compute->ResampleImage(1, padding);
                    const double nmi_pre = measure_nmi->GetSimilarityMeasureValue();
                    // compute the NMI when adding delta(s)
                    defPtr[index] = current_value + delta;
                    compute->ResampleImage(1, padding);
                    const double nmi_post = measure_nmi->GetSimilarityMeasureValue();
                    // Compute the difference
                    gradPtr[index] = -(nmi_post - nmi_pre) / (2. * delta);
                    defPtr[index] = current_value;
                }
                testCases.push_back({testName + " " + platform->GetName(),
                                     std::move(gradientImage), std::move(expectedGradientImage)});
            }
        }
    }

protected:
    using TestData = std::tuple<std::string, NiftiImage, NiftiImage>;
    using TestCase = std::tuple<std::string, NiftiImage, NiftiImage>;
    inline static vector<TestCase> testCases;
};

TEST_CASE_METHOD(NMIGradientTest, "NMI Gradient", "[unit]") {
    // Loop over all generated test cases
    for (auto&& testCase : testCases) {
        // Retrieve test information
        auto&& [testName, result, expected] = testCase;

        SECTION(testName) {
            std::cout << "\n**************** Section " << testName << " ****************" << std::endl;

            float *resPtr = static_cast<float *>(result->data);
            float *expPtr = static_cast<float *>(expected->data);
            float resMean = reg_tools_getMeanValue(result);
            float expMean = reg_tools_getMeanValue(expected);
            float resStdd = reg_tools_getSTDValue(result);
            float expStdd = reg_tools_getSTDValue(expected);
            double corr = 0;
            for(unsigned i=0; i<expected.nVoxels();++i)
                corr += (resPtr[i]-resMean)*(expPtr[i]-expMean);
            
            corr /= resStdd*expStdd*result.nVoxels();
            std::cout << "Correlation = " << corr << std::endl;
            const double norm = std::max(fabs(reg_tools_getMinValue(expected, 0)),
                                         fabs(reg_tools_getMaxValue(expected, 0)));
            for(unsigned i=0; i<expected.nVoxels();++i){
                const double ratio = fabs(resPtr[i] - expPtr[i])/norm;
                if (ratio > .1){
                    std::cout << "[i]=" << i;
                    std::cout << " | ratio=" << ratio;
                    std::cout << " | Result=" << resPtr[i];
                    std::cout << " | Expected=" << expPtr[i] << std::endl;
                }
            }
            REQUIRE(corr > 0.99);
        }
    }
}