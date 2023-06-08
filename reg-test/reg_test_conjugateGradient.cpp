// OpenCL is not supported for this test
#undef _USE_OPENCL

#include "reg_test_common.h"

#define EPS 0.000001

/*
    This test file contains the following unit tests:
    test functions: conjugate gradient
    In 2D and 3D
    Update control point grid
    Update transformation gradient
*/


class ConjugateGradientTest: public InterfaceOptimiser {
protected:
    using TestData = std::tuple<std::string, NiftiImage, NiftiImage, NiftiImage, NiftiImage>;
    using TestCase = std::tuple<shared_ptr<Platform>, unique_ptr<F3dContent>, TestData, bool, bool, bool, float>;

    inline static vector<TestCase> testCases;

public:
    ConjugateGradientTest() {
        if (!testCases.empty())
            return;

        // Create a random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> distr(0, 1);

        // Create a reference 2D image
        vector<NiftiImage::dim_t> dimFlo{ 4, 4 };
        NiftiImage reference2d(dimFlo, NIFTI_TYPE_FLOAT32);

        // Fill image with distance from identity
        const auto ref2dPtr = reference2d.data();
        auto ref2dIt = ref2dPtr.begin();
        for (int y = 0; y < reference2d->ny; ++y)
            for (int x = 0; x < reference2d->nx; ++x)
                *ref2dIt++ = sqrtf(static_cast<float>(x * x + y * y));

        // Create a reference 3D image
        dimFlo.push_back(4);
        NiftiImage reference3d(dimFlo, NIFTI_TYPE_FLOAT32);

        // Fill image with distance from identity
        const auto ref3dPtr = reference3d.data();
        auto ref3dIt = ref3dPtr.begin();
        for (int z = 0; z < reference3d->nz; ++z)
            for (int y = 0; y < reference3d->ny; ++y)
                for (int x = 0; x < reference3d->nx; ++x)
                    *ref3dIt++ = sqrtf(static_cast<float>(x * x + y * y + z * z));

        // Generate the different test cases
        // Test 2D
        NiftiImage controlPointGrid2d = CreateControlPointGrid(reference2d);
        NiftiImage bestControlPointGrid2d(controlPointGrid2d, NiftiImage::Copy::ImageInfoAndAllocData);
        NiftiImage transformationGradient2d(controlPointGrid2d, NiftiImage::Copy::ImageInfoAndAllocData);
        auto bestCpp2dPtr = bestControlPointGrid2d.data();
        auto transGrad2dPtr = transformationGradient2d.data();
        for (size_t i = 0; i < transformationGradient2d.nVoxels(); ++i) {
            bestCpp2dPtr[i] = distr(gen);
            transGrad2dPtr[i] = distr(gen);
        }

        // Add the test data
        vector<TestData> testData;
        testData.emplace_back(TestData(
            "2D",
            std::move(reference2d),
            std::move(controlPointGrid2d),
            std::move(bestControlPointGrid2d),
            std::move(transformationGradient2d)
        ));

        // Test 3D
        NiftiImage controlPointGrid3d = CreateControlPointGrid(reference3d);
        NiftiImage bestControlPointGrid3d(controlPointGrid3d, NiftiImage::Copy::ImageInfoAndAllocData);
        NiftiImage transformationGradient3d(controlPointGrid3d, NiftiImage::Copy::ImageInfoAndAllocData);
        auto bestCpp3dPtr = bestControlPointGrid3d.data();
        auto transGrad3dPtr = transformationGradient3d.data();
        for (size_t i = 0; i < transformationGradient3d.nVoxels(); ++i) {
            bestCpp3dPtr[i] = distr(gen);
            transGrad3dPtr[i] = distr(gen);
        }

        // Add the test data
        testData.emplace_back(TestData(
            "3D",
            std::move(reference3d),
            std::move(controlPointGrid3d),
            std::move(bestControlPointGrid3d),
            std::move(transformationGradient3d)
        ));

        // Add platforms, optimise*, and scale to the test data
        distr = std::uniform_real_distribution<float>(0, 10);
        for (auto&& testData : testData) {
            for (auto&& platformType : PlatformTypes) {
                shared_ptr<Platform> platform{ new Platform(platformType) };
                unique_ptr<F3dContentCreator> contentCreator{ dynamic_cast<F3dContentCreator*>(platform->CreateContentCreator(ContentType::F3d)) };
                for (int optimiseX = 0; optimiseX < 2; optimiseX++) {
                    for (int optimiseY = 0; optimiseY < 2; optimiseY++) {
                        for (int optimiseZ = 0; optimiseZ < 2; optimiseZ++) {
                            // Make a copy of the test data
                            auto td = testData;
                            auto&& [testName, reference, controlPointGrid, bestControlPointGrid, transGrad] = td;
                            // Add content
                            unique_ptr<F3dContent> content{ contentCreator->Create(reference, reference, controlPointGrid) };
                            testCases.push_back({ platform, std::move(content), std::move(td), optimiseX, optimiseY, optimiseZ, distr(gen) });
                        }
                    }
                }
            }
        }
    }

    void UpdateControlPointPosition(NiftiImage& currentDof,
                                    const NiftiImage& bestDof,
                                    const NiftiImage& gradient,
                                    const float& scale,
                                    const bool& optimiseX,
                                    const bool& optimiseY,
                                    const bool& optimiseZ) {
        // Update the values for the x-axis displacement
        if (optimiseX) {
            auto currentDofPtr = currentDof.data(0);
            const auto bestDofPtr = bestDof.data(0);
            const auto gradientPtr = gradient.data(0);
            for (size_t i = 0; i < currentDofPtr.length(); ++i)
                currentDofPtr[i] = static_cast<float>(bestDofPtr[i]) + scale * static_cast<float>(gradientPtr[i]);
        }
        // Update the values for the y-axis displacement
        if (optimiseY) {
            auto currentDofPtr = currentDof.data(1);
            const auto bestDofPtr = bestDof.data(1);
            const auto gradientPtr = gradient.data(1);
            for (size_t i = 0; i < currentDofPtr.length(); ++i)
                currentDofPtr[i] = static_cast<float>(bestDofPtr[i]) + scale * static_cast<float>(gradientPtr[i]);
        }
        // Update the values for the z-axis displacement
        if (optimiseZ && currentDof->nz > 1) {
            auto currentDofPtr = currentDof.data(2);
            const auto bestDofPtr = bestDof.data(2);
            const auto gradientPtr = gradient.data(2);
            for (size_t i = 0; i < currentDofPtr.length(); ++i)
                currentDofPtr[i] = static_cast<float>(bestDofPtr[i]) + scale * static_cast<float>(gradientPtr[i]);
        }
    }

    void UpdateGradientValues(NiftiImage& gradient, const bool& firstCall) {
        // Create array1 and array2
        static NiftiImage array1;
        static NiftiImage array2;
        if (firstCall) {
            array1 = NiftiImage(gradient, NiftiImage::Copy::ImageInfoAndAllocData);
            array2 = NiftiImage(gradient, NiftiImage::Copy::ImageInfoAndAllocData);
        }

        auto gradientPtr = gradient.data();
        auto array1Ptr = array1.data();
        auto array2Ptr = array2.data();

        if (firstCall) {
            // Initialise array1 and array2
            for (size_t i = 0; i < gradient.nVoxels(); i++)
                array2Ptr[i] = array1Ptr[i] = -static_cast<float>(gradientPtr[i]);
        } else {
            // Calculate gam
            double dgg = 0, gg = 0;
            for (size_t i = 0; i < gradient.nVoxels(); i++) {
                gg += static_cast<float>(array2Ptr[i]) * static_cast<float>(array1Ptr[i]);
                dgg += (static_cast<float>(gradientPtr[i]) + static_cast<float>(array1Ptr[i])) * static_cast<float>(gradientPtr[i]);
            }
            const double gam = dgg / gg;

            // Update gradient values
            for (size_t i = 0; i < gradient.nVoxels(); i++) {
                array1Ptr[i] = -static_cast<float>(gradientPtr[i]);
                array2Ptr[i] = static_cast<float>(array1Ptr[i]) + gam * static_cast<float>(array2Ptr[i]);
                gradientPtr[i] = -static_cast<float>(array2Ptr[i]);
            }
        }
    }

    // Required for InterfaceOptimiser
    virtual double GetObjectiveFunctionValue() { return 0; }
    virtual void UpdateParameters(float) {}
    virtual void UpdateBestObjFunctionValue() {}
};

TEST_CASE_METHOD(ConjugateGradientTest, "Conjugate gradient", "[ConjugateGradient]") {
    // Loop over all generated test cases
    for (auto&& testCase : testCases) {
        // Retrieve test information
        auto&& [platform, content, testData, optimiseX, optimiseY, optimiseZ, scale] = testCase;
        auto&& [testName, reference, controlPointGrid, bestControlPointGrid, transGrad] = testData;
        const std::string sectionName = testName + " " + platform->GetName() + " " + (optimiseX ? "X" : "noX") + " " + (optimiseY ? "Y" : "noY") + " " + (optimiseZ ? "Z" : "noZ") + " scale = " + std::to_string(scale);

        SECTION(sectionName) {
            std::cout << "******** UpdateControlPointPosition " << sectionName << " ********" << std::endl;

            // Set the control point grid
            NiftiImage img = content->GetControlPointGrid();
            // Use bestControlPointGrid to store bestDof during initialisation of the optimiser
            img.copyData(bestControlPointGrid);
            img.disown();
            content->UpdateControlPointGrid();

            // Set the transformation gradient
            img = content->GetTransformationGradient();
            img.copyData(transGrad);
            img.disown();
            content->UpdateTransformationGradient();

            // Create a copy of the control point grid for expected results
            NiftiImage controlPointGridExpected = bestControlPointGrid;

            // Update the control point position
            unique_ptr<reg_optimiser<float>> optimiser{ platform->template CreateOptimiser<float>(*content, *this, 0, true, optimiseX, optimiseY, optimiseZ) };
            unique_ptr<Compute> compute{ platform->CreateCompute(*content) };
            compute->UpdateControlPointPosition(optimiser->GetCurrentDof(), optimiser->GetBestDof(), optimiser->GetGradient(), scale, optimiseX, optimiseY, optimiseZ);
            UpdateControlPointPosition(controlPointGridExpected, bestControlPointGrid, transGrad, scale, optimiseX, optimiseY, optimiseZ);

            // Check the results
            img = content->GetControlPointGrid();
            const auto cppPtr = img.data();
            const auto cppExpPtr = controlPointGridExpected.data();
            img.disown();
            for (size_t i = 0; i < controlPointGridExpected.nVoxels(); ++i) {
                const float cppVal = cppPtr[i];
                const float cppExpVal = cppExpPtr[i];
                std::cout << i << " " << cppVal << " " << cppExpVal << std::endl;
                REQUIRE(fabs(cppVal - cppExpVal) < EPS);
            }

            // Update the gradient values
            // Only run once by discarding other optimiseX, optimiseY, optimiseZ combinations
            if (!optimiseX && !optimiseY && !optimiseZ) {
                std::cout << "******** UpdateGradientValues " << sectionName << " ********" << std::endl;

                // Initialise the conjugate gradient
                optimiser->UpdateGradientValues();
                UpdateGradientValues(transGrad, true);
                // Fill the gradient with random values
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<float> distr(0, 1);
                auto gradientPtr = transGrad.data();
                for (size_t i = 0; i < transGrad.nVoxels(); i++)
                    gradientPtr[i] = distr(gen);
                // Update the transformation gradient
                img = content->GetTransformationGradient();
                img.copyData(transGrad);
                img.disown();
                content->UpdateTransformationGradient();
                // Get the gradient values
                optimiser->UpdateGradientValues();
                UpdateGradientValues(transGrad, false);

                // Check the results
                img = content->GetTransformationGradient();
                const auto gradPtr = img.data();
                const auto gradExpPtr = transGrad.data();
                img.disown();
                for (size_t i = 0; i < transGrad.nVoxels(); ++i) {
                    const float gradVal = gradPtr[i];
                    const float gradExpVal = gradExpPtr[i];
                    std::cout << i << " " << gradVal << " " << gradExpVal << std::endl;
                    REQUIRE(fabs(gradVal - gradExpVal) < EPS);
                }
            }
            // Ensure the termination of content before CudaContext
            content.reset();
        }
    }
}
