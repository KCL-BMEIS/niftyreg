// OpenCL is not supported for this test
#undef USE_OPENCL

#include "reg_test_common.h"

/*
    This test file contains the following unit tests:
    test functions:
    In 2D and 3D
    Maximal length
    Normalise gradient
*/


class NormaliseGradientTest {
protected:
    using TestData = std::tuple<std::string, NiftiImage, NiftiImage, NiftiImage>;
    using TestCase = std::tuple<shared_ptr<Platform>, unique_ptr<F3dContent>, TestData, bool, bool, bool>;

    inline static vector<TestCase> testCases;

public:
    NormaliseGradientTest() {
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
        NiftiImage transformationGradient2d(controlPointGrid2d, NiftiImage::Copy::ImageInfoAndAllocData);
        auto transGrad2dPtr = transformationGradient2d.data();
        for (size_t i = 0; i < transformationGradient2d.nVoxels(); ++i)
            transGrad2dPtr[i] = distr(gen);

        // Add the test data
        vector<TestData> testData;
        testData.emplace_back(TestData(
            "2D",
            std::move(reference2d),
            std::move(controlPointGrid2d),
            std::move(transformationGradient2d)
        ));

        // Test 3D
        NiftiImage controlPointGrid3d = CreateControlPointGrid(reference3d);
        NiftiImage transformationGradient3d(controlPointGrid3d, NiftiImage::Copy::ImageInfoAndAllocData);
        auto transGrad3dPtr = transformationGradient3d.data();
        for (size_t i = 0; i < transformationGradient3d.nVoxels(); ++i)
            transGrad3dPtr[i] = distr(gen);

        // Add the test data
        testData.emplace_back(TestData(
            "3D",
            std::move(reference3d),
            std::move(controlPointGrid3d),
            std::move(transformationGradient3d)
        ));

        // Add platforms and optimise* to the test data
        for (auto&& testData : testData) {
            for (auto&& platformType : PlatformTypes) {
                shared_ptr<Platform> platform{ new Platform(platformType) };
                unique_ptr<F3dContentCreator> contentCreator{ dynamic_cast<F3dContentCreator*>(platform->CreateContentCreator(ContentType::F3d)) };
                for (int optimiseX = 0; optimiseX < 2; optimiseX++) {
                    for (int optimiseY = 0; optimiseY < 2; optimiseY++) {
                        for (int optimiseZ = 0; optimiseZ < 2; optimiseZ++) {
                            // Make a copy of the test data
                            auto td = testData;
                            auto&& [testName, reference, controlPointGrid, testGrad] = td;
                            // Add content
                            unique_ptr<F3dContent> content{ contentCreator->Create(reference, reference, controlPointGrid) };
                            testCases.push_back({ platform, std::move(content), std::move(td), optimiseX, optimiseY, optimiseZ });
                        }
                    }
                }
            }
        }
    }

    template<typename T>
    T GetMaximalLength(const nifti_image* transformationGradient, const bool& optimiseX, const bool& optimiseY, const bool& optimiseZ) {
        if (!optimiseX && !optimiseY && !optimiseZ) return 0;
        const size_t nVoxelsPerVolume = NiftiImage::calcVoxelNumber(transformationGradient, 3);
        const T *ptrX = static_cast<T*>(transformationGradient->data);
        const T *ptrY = &ptrX[nVoxelsPerVolume];
        const T *ptrZ = &ptrY[nVoxelsPerVolume];
        T maxGradLength = 0;

        if (transformationGradient->nz > 1) {
            for (size_t i = 0; i < nVoxelsPerVolume; i++) {
                T valX = 0, valY = 0, valZ = 0;
                if (optimiseX)
                    valX = *ptrX++;
                if (optimiseY)
                    valY = *ptrY++;
                if (optimiseZ)
                    valZ = *ptrZ++;
                maxGradLength = std::max(sqrt(valX * valX + valY * valY + valZ * valZ), maxGradLength);
            }
        } else {
            for (size_t i = 0; i < nVoxelsPerVolume; i++) {
                T valX = 0, valY = 0;
                if (optimiseX)
                    valX = *ptrX++;
                if (optimiseY)
                    valY = *ptrY++;
                maxGradLength = std::max(sqrt(valX * valX + valY * valY), maxGradLength);
            }
        }

        return maxGradLength;
    }

    template<typename T>
    void NormaliseGradient(nifti_image* transformationGradient, const T& maxGradLength, const bool& optimiseX, const bool& optimiseY, const bool& optimiseZ) {
        if (maxGradLength == 0 || (!optimiseX && !optimiseY && !optimiseZ)) return;
        const size_t nVoxelsPerVolume = NiftiImage::calcVoxelNumber(transformationGradient, 3);
        T *ptrX = static_cast<T*>(transformationGradient->data);
        T *ptrY = &ptrX[nVoxelsPerVolume];
        T *ptrZ = &ptrY[nVoxelsPerVolume];
        if (transformationGradient->nz > 1) {
            for (size_t i = 0; i < nVoxelsPerVolume; ++i) {
                T valX = 0, valY = 0, valZ = 0;
                if (optimiseX)
                    valX = ptrX[i];
                if (optimiseY)
                    valY = ptrY[i];
                if (optimiseZ)
                    valZ = ptrZ[i];
                ptrX[i] = valX / maxGradLength;
                ptrY[i] = valY / maxGradLength;
                ptrZ[i] = valZ / maxGradLength;
            }
        } else {
            for (size_t i = 0; i < nVoxelsPerVolume; ++i) {
                T valX = 0, valY = 0;
                if (optimiseX)
                    valX = ptrX[i];
                if (optimiseY)
                    valY = ptrY[i];
                ptrX[i] = valX / maxGradLength;
                ptrY[i] = valY / maxGradLength;
            }
        }
    }
};

TEST_CASE_METHOD(NormaliseGradientTest, "Normalise gradient", "[NormaliseGradient]") {
    // Loop over all generated test cases
    for (auto&& testCase : testCases) {
        // Retrieve test information
        auto&& [platform, content, testData, optimiseX, optimiseY, optimiseZ] = testCase;
        auto&& [testName, reference, controlPointGrid, testGrad] = testData;
        const std::string sectionName = testName + " " + platform->GetName() + " " + (optimiseX ? "X" : "noX") + " " + (optimiseY ? "Y" : "noY") + " " + (optimiseZ ? "Z" : "noZ");

        SECTION(sectionName) {
            NR_COUT << "\n**************** Section " << sectionName << " ****************" << std::endl;

            // Increase the precision for the output
            NR_COUT << std::fixed << std::setprecision(10);

            // Set the transformation gradient image to host the computation
            NiftiImage transGrad = content->GetTransformationGradient();
            transGrad.copyData(testGrad);
            transGrad.disown();
            content->UpdateTransformationGradient();

            // Calculate the maximal length
            unique_ptr<Compute> compute{ platform->CreateCompute(*content) };
            const auto maxLength = static_cast<float>(compute->GetMaximalLength(optimiseX, optimiseY, optimiseZ));
            const auto testLength = GetMaximalLength<float>(testGrad, optimiseX, optimiseY, optimiseZ);
            // Check the results
            REQUIRE(fabs(maxLength - testLength) < EPS);

            // Normalise the gradient
            compute->NormaliseGradient(maxLength, optimiseX, optimiseY, optimiseZ);
            NormaliseGradient<float>(testGrad, testLength, optimiseX, optimiseY, optimiseZ);

            // Check the results
            transGrad = content->GetTransformationGradient();
            const auto transGradPtr = transGrad.data();
            const auto testGradPtr = testGrad.data();
            transGrad.disown();
            for (size_t i = 0; i < testGrad.nVoxels(); ++i) {
                const float transGradVal = transGradPtr[i];
                const float testGradVal = testGradPtr[i];
                const float diff = abs(transGradVal - testGradVal);
                if (diff > EPS)
                    NR_COUT << i << " " << transGradVal << " " << testGradVal << std::endl;
                REQUIRE(diff < EPS);
            }
            // Ensure the termination of content before CudaContext
            content.reset();
        }
    }
}
