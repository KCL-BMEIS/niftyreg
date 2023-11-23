// OpenCL is not supported for this test
#undef USE_OPENCL

#include "reg_test_common.h"

/*
    This test file contains the following unit tests:
    test functions: conjugate gradient
    In 2D and 3D
    Update control point grid
    Update transformation gradient
*/


class ConjugateGradientTest: public InterfaceOptimiser {
protected:
    using TestData = std::tuple<std::string, NiftiImage, NiftiImage, NiftiImage, NiftiImage, NiftiImage, NiftiImage>;
    using TestCase = std::tuple<shared_ptr<Platform>, unique_ptr<F3dContent>, unique_ptr<F3dContent>, TestData, bool, bool, bool, float>;

    inline static vector<TestCase> testCases;

public:
    ConjugateGradientTest() {
        if (!testCases.empty())
            return;

        // Create a random number generator
        std::mt19937 gen(0);
        std::uniform_real_distribution<float> distr(0, 1);

        // Create a reference 2D image
        vector<NiftiImage::dim_t> dimFlo{ 4, 4 };
        NiftiImage reference2d(dimFlo, NIFTI_TYPE_FLOAT32);

        // Fill image with distance from identity
        const auto ref2dPtr = reference2d.data();
        auto ref2dItr = ref2dPtr.begin();
        for (int y = 0; y < reference2d->ny; ++y)
            for (int x = 0; x < reference2d->nx; ++x)
                *ref2dItr++ = sqrtf(static_cast<float>(x * x + y * y));

        // Create a reference 3D image
        dimFlo.push_back(4);
        NiftiImage reference3d(dimFlo, NIFTI_TYPE_FLOAT32);

        // Fill image with distance from identity
        const auto ref3dPtr = reference3d.data();
        auto ref3dItr = ref3dPtr.begin();
        for (int z = 0; z < reference3d->nz; ++z)
            for (int y = 0; y < reference3d->ny; ++y)
                for (int x = 0; x < reference3d->nx; ++x)
                    *ref3dItr++ = sqrtf(static_cast<float>(x * x + y * y + z * z));

        // Generate the different test cases
        // Test 2D
        NiftiImage controlPointGrid2d = CreateControlPointGrid(reference2d);
        NiftiImage controlPointGridBw2d(controlPointGrid2d);
        NiftiImage bestControlPointGrid2d(controlPointGrid2d, NiftiImage::Copy::ImageInfoAndAllocData);
        NiftiImage transformationGradient2d(controlPointGrid2d, NiftiImage::Copy::ImageInfoAndAllocData);
        NiftiImage transformationGradientBw2d(controlPointGrid2d, NiftiImage::Copy::ImageInfoAndAllocData);
        auto bestCpp2dPtr = bestControlPointGrid2d.data();
        auto transGrad2dPtr = transformationGradient2d.data();
        auto transGradBw2dPtr = transformationGradientBw2d.data();
        for (size_t i = 0; i < transformationGradient2d.nVoxels(); ++i) {
            bestCpp2dPtr[i] = distr(gen);
            transGrad2dPtr[i] = distr(gen);
            transGradBw2dPtr[i] = distr(gen);
        }

        // Add the test data
        vector<TestData> testData;
        testData.emplace_back(TestData(
            "2D",
            std::move(reference2d),
            std::move(controlPointGrid2d),
            std::move(controlPointGridBw2d),
            std::move(bestControlPointGrid2d),
            std::move(transformationGradient2d),
            std::move(transformationGradientBw2d)
        ));

        // Test 3D
        NiftiImage controlPointGrid3d = CreateControlPointGrid(reference3d);
        NiftiImage controlPointGridBw3d(controlPointGrid3d);
        NiftiImage bestControlPointGrid3d(controlPointGrid3d, NiftiImage::Copy::ImageInfoAndAllocData);
        NiftiImage transformationGradient3d(controlPointGrid3d, NiftiImage::Copy::ImageInfoAndAllocData);
        NiftiImage transformationGradientBw3d(controlPointGrid3d, NiftiImage::Copy::ImageInfoAndAllocData);
        auto bestCpp3dPtr = bestControlPointGrid3d.data();
        auto transGrad3dPtr = transformationGradient3d.data();
        auto transGradBw3dPtr = transformationGradientBw3d.data();
        for (size_t i = 0; i < transformationGradient3d.nVoxels(); ++i) {
            bestCpp3dPtr[i] = distr(gen);
            transGrad3dPtr[i] = distr(gen);
            transGradBw3dPtr[i] = distr(gen);
        }

        // Add the test data
        testData.emplace_back(TestData(
            "3D",
            std::move(reference3d),
            std::move(controlPointGrid3d),
            std::move(controlPointGridBw3d),
            std::move(bestControlPointGrid3d),
            std::move(transformationGradient3d),
            std::move(transformationGradientBw3d)
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
                            auto&& [testName, reference, controlPointGrid, controlPointGridBw, bestControlPointGrid, transGrad, transGradBw] = td;
                            // Add content
                            unique_ptr<F3dContent> content{ contentCreator->Create(reference, reference, controlPointGrid) };
                            unique_ptr<F3dContent> contentBw{ contentCreator->Create(reference, reference, controlPointGridBw) };
                            testCases.push_back({ platform, std::move(content), std::move(contentBw), std::move(td), optimiseX, optimiseY, optimiseZ, distr(gen) });
                        }
                    }
                }
            }
        }
    }

    void UpdateControlPointPosition(NiftiImage& currentDof,
                                    const NiftiImage& bestDof,
                                    const NiftiImage& gradient,
                                    const float scale,
                                    const bool optimiseX,
                                    const bool optimiseY,
                                    const bool optimiseZ) {
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

    void UpdateGradientValues(NiftiImage& gradient, const bool firstCall, const bool isSymmetric, NiftiImage *gradientBw) {
        // Create array1 and array2
        static NiftiImage array1, array1Bw;
        static NiftiImage array2, array2Bw;
        if (firstCall) {
            array1 = array2 = NiftiImage(gradient, NiftiImage::Copy::ImageInfoAndAllocData);
            if (isSymmetric)
                array1Bw = array2Bw = NiftiImage(*gradientBw, NiftiImage::Copy::ImageInfoAndAllocData);
        }

        auto gradientPtr = gradient.data();
        auto array1Ptr = array1.data();
        auto array2Ptr = array2.data();
        NiftiImageData gradientBwPtr, array1BwPtr, array2BwPtr;
        if (isSymmetric) {
            gradientBwPtr = gradientBw->data();
            array1BwPtr = array1Bw.data();
            array2BwPtr = array2Bw.data();
        }

        if (firstCall) {
            // Initialise array1 and array2
            for (size_t i = 0; i < gradient.nVoxels(); i++)
                array2Ptr[i] = array1Ptr[i] = -static_cast<float>(gradientPtr[i]);
            if (isSymmetric) {
                for (size_t i = 0; i < gradientBw->nVoxels(); i++)
                    array2BwPtr[i] = array1BwPtr[i] = -static_cast<float>(gradientBwPtr[i]);
            }
        } else {
            // Calculate gam
            double dgg = 0, gg = 0;
            for (size_t i = 0; i < gradient.nVoxels(); i++) {
                gg += static_cast<float>(array2Ptr[i]) * static_cast<float>(array1Ptr[i]);
                dgg += (static_cast<float>(gradientPtr[i]) + static_cast<float>(array1Ptr[i])) * static_cast<float>(gradientPtr[i]);
            }
            double gam = dgg / gg;
            if (isSymmetric) {
                double dggBw = 0, ggBw = 0;
                for (size_t i = 0; i < gradientBw->nVoxels(); i++) {
                    ggBw += static_cast<float>(array2BwPtr[i]) * static_cast<float>(array1BwPtr[i]);
                    dggBw += (static_cast<float>(gradientBwPtr[i]) + static_cast<float>(array1BwPtr[i])) * static_cast<float>(gradientBwPtr[i]);
                }
                gam = (dgg + dggBw) / (gg + ggBw);
            }

            // Update gradient values
            for (size_t i = 0; i < gradient.nVoxels(); i++) {
                array1Ptr[i] = -static_cast<float>(gradientPtr[i]);
                array2Ptr[i] = static_cast<float>(array1Ptr[i]) + gam * static_cast<float>(array2Ptr[i]);
                gradientPtr[i] = -static_cast<float>(array2Ptr[i]);
            }
            if (isSymmetric) {
                for (size_t i = 0; i < gradientBw->nVoxels(); i++) {
                    array1BwPtr[i] = -static_cast<float>(gradientBwPtr[i]);
                    array2BwPtr[i] = static_cast<float>(array1BwPtr[i]) + gam * static_cast<float>(array2BwPtr[i]);
                    gradientBwPtr[i] = -static_cast<float>(array2BwPtr[i]);
                }
            }
        }
    }

    // Required for InterfaceOptimiser
    virtual double GetObjectiveFunctionValue() { return 0; }
    virtual void UpdateParameters(float) {}
    virtual void UpdateBestObjFunctionValue() {}
};

TEST_CASE_METHOD(ConjugateGradientTest, "Conjugate Gradient", "[unit]") {
    // Loop over all generated test cases
    for (auto&& testCase : testCases) {
        // Retrieve test information
        auto&& [platform, content, contentBw, testData, optimiseX, optimiseY, optimiseZ, scale] = testCase;
        auto&& [testName, reference, controlPointGrid, controlPointGridBw, bestControlPointGrid, transGrad, transGradBw] = testData;
        const std::string sectionName = testName + " " + platform->GetName() + " " + (optimiseX ? "X" : "noX") + " " + (optimiseY ? "Y" : "noY") + " " + (optimiseZ ? "Z" : "noZ") + " scale = " + std::to_string(scale);

        SECTION(sectionName) {
            NR_COUT << "\n**************** UpdateControlPointPosition " << sectionName << " ****************" << std::endl;

            // Increase the precision for the output
            NR_COUT << std::fixed << std::setprecision(10);

            // Set the control point grid
            NiftiImage img = content->GetControlPointGrid();
            // Use bestControlPointGrid to store bestDof during initialisation of the optimiser
            img.copyData(bestControlPointGrid);
            img.disown();
            content->UpdateControlPointGrid();

            // Set the transformation gradients
            img = content->GetTransformationGradient();
            img.copyData(transGrad);
            img.disown();
            content->UpdateTransformationGradient();
            img = contentBw->GetTransformationGradient();
            img.copyData(transGradBw);
            img.disown();
            contentBw->UpdateTransformationGradient();

            // Create a copy of the control point grid for expected results
            NiftiImage controlPointGridExpected = bestControlPointGrid;

            // Update the control point position
            unique_ptr<Optimiser<float>> optimiser{ platform->template CreateOptimiser<float>(*content, *this, 0, true, optimiseX, optimiseY, optimiseZ) };
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
                const auto diff = abs(cppVal - cppExpVal);
                if (diff > 0)
                    NR_COUT << i << " " << cppVal << " " << cppExpVal << std::endl;
                REQUIRE(diff == 0);
            }

            // Update the gradient values
            // Only run once by discarding other optimiseX, optimiseY, optimiseZ combinations
            if (!optimiseX && !optimiseY && !optimiseZ) {
                for (int isSymmetric = 0; isSymmetric < 2; isSymmetric++) {
                    NR_COUT << "\n**************** UpdateGradientValues " << sectionName + (isSymmetric ? " Symmetric" : "") << " ****************" << std::endl;

                    // Create a random number generator
                    std::random_device rd;
                    std::mt19937 gen(rd());
                    std::uniform_real_distribution<float> distr(0, 1);

                    // Create a symmetric optimiser if required
                    if (isSymmetric)
                        optimiser.reset(platform->template CreateOptimiser<float>(*content, *this, 0, true, optimiseX, optimiseY, optimiseZ, contentBw.get()));

                    // Initialise the conjugate gradients
                    optimiser->UpdateGradientValues();
                    UpdateGradientValues(transGrad, true, isSymmetric, &transGradBw);

                    // Fill the gradients with random values
                    auto gradientPtr = transGrad.data();
                    auto gradientBwPtr = transGradBw.data();
                    for (size_t i = 0; i < transGrad.nVoxels(); i++) {
                        gradientPtr[i] = distr(gen);
                        if (isSymmetric)
                            gradientBwPtr[i] = distr(gen);
                    }
                    // Update the transformation gradients
                    img = content->GetTransformationGradient();
                    img.copyData(transGrad);
                    img.disown();
                    content->UpdateTransformationGradient();
                    if (isSymmetric) {
                        img = contentBw->GetTransformationGradient();
                        img.copyData(transGradBw);
                        img.disown();
                        contentBw->UpdateTransformationGradient();
                    }

                    // Get the gradient values
                    optimiser->UpdateGradientValues();
                    UpdateGradientValues(transGrad, false, isSymmetric, &transGradBw);

                    // Check the results
                    img = content->GetTransformationGradient();
                    const auto gradPtr = img.data();
                    const auto gradExpPtr = transGrad.data();
                    img.disown();
                    NiftiImageData gradBwPtr, gradExpBwPtr;
                    if (isSymmetric) {
                        img = contentBw->GetTransformationGradient();
                        gradBwPtr = img.data();
                        gradExpBwPtr = transGradBw.data();
                        img.disown();
                    }
                    for (size_t i = 0; i < transGrad.nVoxels(); ++i) {
                        const float gradVal = gradPtr[i];
                        const float gradExpVal = gradExpPtr[i];
                        const auto diff = abs(gradVal - gradExpVal);
                        if (diff > EPS)
                            NR_COUT << i << " " << gradVal << " " << gradExpVal << std::endl;
                        REQUIRE(diff < EPS);
                        if (isSymmetric) {
                            const float gradBwVal = gradBwPtr[i];
                            const float gradExpBwVal = gradExpBwPtr[i];
                            const auto diff = abs(gradBwVal - gradExpBwVal);
                            if (diff > EPS)
                                NR_COUT << i << " " << gradBwVal << " " << gradExpBwVal << " backwards" << std::endl;
                            REQUIRE(diff < EPS);
                        }
                    }
                }
            }
            // Ensure the termination of content before CudaContext
            content.reset();
            contentBw.reset();
        }
    }
}
