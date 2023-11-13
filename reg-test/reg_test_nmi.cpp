// OpenCL is not supported for this test yet
#undef USE_OPENCL

#include "reg_test_common.h"
#include "_reg_tools.h"
#include "_reg_nmi.h"

/*
    This test file contains the following unit tests:
    test function: NMI computation
*/

class NmiTest {
public:
    NmiTest() {
        if (!testCases.empty())
            return;

        // Create a number generator
        std::mt19937 gen(0);
        // Images will be rescaled between 2 and bin-3
        // Default bin value is 68 (64+4 for Parzen windowing)
        std::uniform_real_distribution<float> distr(2, 65);

        // Create reference and floating 2D images
        vector<NiftiImage::dim_t> dim{ 60, 62 };
        NiftiImage reference2d(dim, NIFTI_TYPE_FLOAT32);
        NiftiImage floating2d(dim, NIFTI_TYPE_FLOAT32);

        // Create reference and floating 3D images
        dim.push_back(64);
        NiftiImage reference3d(dim, NIFTI_TYPE_FLOAT32);
        NiftiImage floating3d(dim, NIFTI_TYPE_FLOAT32);

        // Fill images with random values
        auto ref2dPtr = reference2d.data();
        auto flo2dPtr = floating2d.data();
        // Ensure at least one pixel contains the max and one the min
        ref2dPtr[0] = flo2dPtr[0] = 2.f;
        ref2dPtr[1] = flo2dPtr[1] = 65.f;
        for (size_t i = 2; i < reference2d.nVoxels(); ++i) {
            ref2dPtr[i] = (int)distr(gen); // cast to integer to not use PW
            flo2dPtr[i] = (int)distr(gen);
        }

        // Fill images with random values
        auto ref3dPtr = reference3d.data();
        auto flo3dPtr = floating3d.data();
        // Ensure at least one pixel contains the max and one the min
        ref3dPtr[0] = flo3dPtr[0] = 2.f;
        ref3dPtr[1] = flo3dPtr[1] = 65.f;
        for (size_t i = 2; i < reference3d.nVoxels(); ++i) {
            ref3dPtr[i] = (int)distr(gen);
            flo3dPtr[i] = (int)distr(gen);
        }

        // Create the object to compute the expected values
        vector<TestData> testData;
        testData.emplace_back(TestData(
            "NMI 2D",
            reference2d,
            floating2d,
            GetNmiPw(reference2d, floating2d)
        ));
        testData.emplace_back(TestData(
            "NMI 3D",
            reference3d,
            floating3d,
            GetNmiPw(reference3d, floating3d)
        ));
        for (auto&& data : testData) {
            for (auto&& platformType : PlatformTypes) {
                // Create the platform
                shared_ptr<Platform> platform{ new Platform(platformType) };
                // Make a copy of the test data
                auto [testName, reference, floating, expected] = data;
                // Create the content creator
                unique_ptr<DefContentCreator> contentCreator{
                    dynamic_cast<DefContentCreator*>(platform->CreateContentCreator(ContentType::Def))
                };
                // Create the content
                unique_ptr<DefContent> content{ contentCreator->Create(reference, floating) };
                // Initialise the warped image using floating image
                content->SetWarped(floating.disown());
                // Create the measure
                unique_ptr<Measure> measure{ platform->CreateMeasure() };
                // Use NMI as a measure
                unique_ptr<reg_nmi> measure_nmi{ dynamic_cast<reg_nmi*>(measure->Create(MeasureType::Nmi)) };
                measure_nmi->SetTimePointWeight(0, 1.0); // weight initially set to default value of 1.0
                measure->Initialise(*measure_nmi, *content);
                const double nmi = measure_nmi->GetSimilarityMeasureValue();

                testCases.push_back({ testName + " " + platform->GetName(), nmi, expected });
            }
        }
    }

protected:
    using TestData = std::tuple<std::string, NiftiImage, NiftiImage, double>;
    using TestCase = std::tuple<std::string, double, double>;
    inline static vector<TestCase> testCases;

    double GetNmiPw(const NiftiImage& ref, const NiftiImage& flo) {
        // Allocate a joint histogram and fill it with zeros
        double jh[68][68];
        for (unsigned i = 0; i < 68; ++i)
            for (unsigned j = 0; j < 68; ++j)
                jh[i][j] = 0;
        // Fill it with the intensity values
        const auto refPtr = ref.data();
        const auto floPtr = flo.data();
        for (auto refItr = refPtr.begin(), floItr = floPtr.begin(); refItr != refPtr.end(); ++refItr, ++floItr)
            jh[(int)*refItr][(int)*floItr]++;
        // Convert the histogram into an image to later apply the convolution
        vector<NiftiImage::dim_t> dim{ 68, 68 };
        NiftiImage jointHistogram(dim, NIFTI_TYPE_FLOAT64);
        double *jhPtr = static_cast<double*>(jointHistogram->data);
        // Convert the occurrences to probabilities
        for (unsigned i = 0; i < 68; ++i)
            for (unsigned j = 0; j < 68; ++j)
                *jhPtr++ = jh[i][j] / ref.nVoxels();
        // Apply a convolution to mimic the parzen windowing
        float sigma[1] = { 1.f };
        reg_tools_kernelConvolution(jointHistogram, sigma, ConvKernelType::Cubic);
        // Restore the jh array
        jhPtr = static_cast<double*>(jointHistogram->data);
        for (unsigned i = 0; i < 68; ++i)
            for (unsigned j = 0; j < 68; ++j)
                jh[i][j] = *jhPtr++;
        // Compute the entropies
        double ref_ent = 0.;
        double flo_ent = 0.;
        double joi_ent = 0.;
        for (unsigned i = 0; i < 68; ++i) {
            double ref_pro = 0.;
            double flo_pro = 0.;
            for (unsigned j = 0; j < 68; ++j) {
                flo_pro += jh[i][j];
                ref_pro += jh[j][i];
                if (jh[i][j] > 0.)
                    joi_ent -= jh[i][j] * log(jh[i][j]);
            }
            if (ref_pro > 0)
                ref_ent -= ref_pro * log(ref_pro);
            if (flo_pro > 0)
                flo_ent -= flo_pro * log(flo_pro);
        }
        double nmi = (ref_ent + flo_ent) / joi_ent;
        return nmi;
    }
};

TEST_CASE_METHOD(NmiTest, "NMI", "[unit]") {
    // Loop over all generated test cases
    for (auto&& testCase : testCases) {
        // Retrieve test information
        auto&& [testName, result, expected] = testCase;

        SECTION(testName) {
            NR_COUT << "\n**************** Section " << testName << " ****************" << std::endl;

            // Increase the precision for the output
            NR_COUT << std::fixed << std::setprecision(10);

            const auto diff = abs(result - expected);
            if (diff > 0)
                NR_COUT << "Result=" << result << " | Expected=" << expected << std::endl;
            REQUIRE(diff < EPS);
        }
    }
}
