// OpenCL is not supported for this test yet
#undef USE_OPENCL

#include "reg_test_common.h"
#include "_reg_tools.h"
#include "_reg_nmi.h"

/*
    NMI on the CPU: the value, and the boundary behaviour of the joint histogram it is built from.

    The first case cross-checks the value against GetNmiPw below. That reference is an independent
    CONSTRUCTION rather than a copy: it bins the integer intensities into a 68x68 array and convolves
    the histogram once, where production accumulates a Parzen window per voxel - two different routes
    to the same numbers. It does restate the entropy formula, so treat it as a cross-check of the
    histogram construction, not an oracle for the whole measure; and it is deliberately confined to
    the interior - the intensities are cast to integer and held in [2, 65] of 68 bins, three clear of
    either end, so the Parzen window is never partial and never clipped. The boundary lives in the
    property cases below.

    The cases after it cover what that leaves out, against properties rather than a second
    implementation: the Parzen kernel's partition of unity, symmetry, support and derivative; the mass
    a value loses when its window is clipped at a bin edge; the agreement between excluding a voxel by
    NaN and excluding it by mask; and the order-independence the OpenMP histogram fill relies on.

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
                content->SetWarped(NiftiImage(floating));
                // Create the measure creator
                unique_ptr<MeasureCreator> measureCreator{ platform->CreateMeasureCreator() };
                // Use NMI as a measure
                unique_ptr<reg_nmi> measure_nmi{ dynamic_cast<reg_nmi*>(measureCreator->Create(MeasureType::Nmi)) };
                measure_nmi->SetTimePointWeight(0, 1.0); // weight initially set to default value of 1.0
                measureCreator->Initialise(*measure_nmi, *content);
                const double nmi = measure_nmi->GetSimilarityMeasureValue();
                // Save the results for testing
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

namespace {

constexpr unsigned short kBins = 68;   // the default: 64 usable + 4 for the Parzen window

// GetBasisSplineValue is templated on PrecisionType but writes its constants as float literals
// (its `2.f / 3.f` is evaluated wholly in float before it ever meets the template type), so
// the double instantiation carries only float precision - about 2e-8 at the peak. The bound below is
// set by that, not by double arithmetic. See the case at the end of this file, which pins it.
constexpr double kKernelTolerance = 1e-7;

// The four taps the histogram fill in reg_getNmiValue uses for a value, following the order in which
// bins are visited, but computing the weights from the kernel rather than restating the loop
double WindowMass(double value, int binNumber) {
    double mass = 0;
    for (int bin = int(value - 1); bin < int(value + 3); ++bin)
        if (0 <= bin && bin < binNumber)
            mass += GetBasisSplineValue<double>(value - bin);
    return mass;
}

NiftiImage MakeIntensityImage(const std::vector<NiftiImage::dim_t>& dims, unsigned seed,
                              float low, float high) {
    NiftiImage img(dims, NIFTI_TYPE_FLOAT32);
    setIdentitySform(img);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> distr(low, high);
    auto ptr = img.data();
    for (size_t i = 0; i < img.nVoxels(); ++i)
        ptr[i] = distr(gen);
    // Pin the extremes so the range is exactly [low, high]. reg_nmi rescales the whole image into
    // [2, bins-3] during InitialiseMeasure, over every voxel and regardless of the mask, so the
    // binning depends on where the extremes sit. Voxels 2 and 3 are used because the exclusion tests
    // below drop every seventh voxel and must not disturb the range - otherwise the two ways of
    // excluding voxels would be compared under two different binnings.
    ptr[2] = low;
    ptr[3] = high;
    return img;
}

double ComputeNmi(const NiftiImage& reference, const NiftiImage& floating, int *mask = nullptr) {
    NiftiImage ref(reference), flo(floating);
    Platform platform(PlatformType::Cpu);
    unique_ptr<DefContentCreator> contentCreator{
        dynamic_cast<DefContentCreator*>(platform.CreateContentCreator(ContentType::Def)) };
    unique_ptr<DefContent> content{ contentCreator->Create(ref, flo, nullptr, mask) };
    content->SetWarped(NiftiImage(floating));
    unique_ptr<MeasureCreator> measureCreator{ platform.CreateMeasureCreator() };
    unique_ptr<reg_nmi> measure{ dynamic_cast<reg_nmi*>(measureCreator->Create(MeasureType::Nmi)) };
    measure->SetTimePointWeight(0, 1.0);
    measureCreator->Initialise(*measure, *content);
    return measure->GetSimilarityMeasureValue();
}

} // namespace

TEST_CASE("The Parzen kernel is a partition of unity over its window", "[unit]") {
    // Each voxel adds GetBasisSplineValue over four consecutive bins. Those four weights summing to
    // one is what makes the joint histogram's total equal the voxel count, and therefore what makes
    // dividing by that total produce probabilities rather than something merely proportional to them.
    for (int step = 0; step <= 40; ++step) {
        const double fraction = step / 40.0;
        const double value = 30.0 + fraction;   // well inside the range, so nothing is clipped
        const double mass = WindowMass(value, kBins);
        INFO("value " << value << ", window mass " << mass);
        REQUIRE(std::abs(mass - 1.0) < kKernelTolerance);
    }
}

TEST_CASE("The Parzen kernel is symmetric, compact and non-negative", "[unit]") {
    for (int step = -60; step <= 60; ++step) {
        const double x = step / 20.0;
        const double value = GetBasisSplineValue<double>(x);
        INFO("x = " << x);
        REQUIRE(value >= 0.0);                                          // a histogram weight
        REQUIRE(value == GetBasisSplineValue<double>(-x));               // even function
        if (std::abs(x) >= 2.0) REQUIRE(value == 0.0);                   // support is four bins wide
        else REQUIRE(value > 0.0);
    }
    // The peak sits at the centre, and a cubic B-spline's value there is 2/3
    REQUIRE(std::abs(GetBasisSplineValue<double>(0.0) - 2.0 / 3.0) < kKernelTolerance);
    // At an integer offset of one, 1/6 either side - the smearing an integer-valued image still gets,
    // which is why NMI of an image against itself is not exactly 2
    REQUIRE(std::abs(GetBasisSplineValue<double>(1.0) - 1.0 / 6.0) < kKernelTolerance);
}

TEST_CASE("The Parzen derivative matches the kernel it differentiates", "[unit]") {
    // GetBasisSplineDerivativeValue drives the NMI gradient. Central differences, in double.
    constexpr double h = 1e-6;
    for (int step = -40; step <= 40; ++step) {
        const double x = step / 20.0;
        if (std::abs(std::abs(x) - 1.0) < h || std::abs(std::abs(x) - 2.0) < h) continue;  // knots
        const double analytic = GetBasisSplineDerivativeValue<double>(x);
        const double numeric = (GetBasisSplineValue<double>(x + h) - GetBasisSplineValue<double>(x - h)) / (2 * h);
        INFO("x = " << x << ", analytic " << analytic << ", numeric " << numeric);
        REQUIRE(std::abs(analytic - numeric) < 1e-6);
    }
}

TEST_CASE("A value at the bin edges loses the clipped part of its window", "[unit]") {
    /*
        The fill skips bins outside [0, binNumber) rather than folding them back, so a voxel whose
        value sits within one bin of either end contributes less than one to the histogram. That is a
        deliberate choice - it is why callers are told to rescale into [2, bins-3] - but it means the
        marginals stop being normalised if anything ever lands there, and nothing said so.
    */
    struct Expectation { double value; double expectedMass; };
    // At value 0 the taps are bins -1, 0, 1, 2 and only three survive; the lost tap is B(0 - (-1)) = 1/6
    const std::vector<Expectation> cases{
        { 0.0,                     1.0 - 1.0 / 6.0 },
        { 1.0,                     1.0 },                       // first fully interior value
        { double(kBins - 2),       1.0 },                       // last fully interior value
        { double(kBins - 1),       1.0 - 1.0 / 6.0 },
    };
    for (const auto& [value, expectedMass] : cases) {
        const double mass = WindowMass(value, kBins);
        NR_COUT << "  value " << std::setw(4) << value << " contributes " << std::fixed
                << std::setprecision(6) << mass << " to the histogram" << std::endl;
        INFO("value " << value << " contributes " << mass << ", expected " << expectedMass);
        REQUIRE(std::abs(mass - expectedMass) < kKernelTolerance);
    }
    // Everything strictly inside contributes exactly one, whatever its fractional part
    for (int bin = 1; bin <= kBins - 3; ++bin)
        for (const double fraction : { 0.0, 0.25, 0.5, 0.75 }) {
            const double mass = WindowMass(bin + fraction, kBins);
            INFO("value " << bin + fraction);
            REQUIRE(std::abs(mass - 1.0) < kKernelTolerance);
        }
}

TEST_CASE("NaN voxels are excluded exactly as masked voxels are", "[unit]") {
    /*
        Two independent exclusions in reg_getNmiValue, one for the mask and one for NaN, reaching the
        same histogram. Excluding a set of voxels one way or the other must give the same number - if
        they ever disagreed, a padded warped image and a masked one would score differently for the
        same overlap.
    */
    for (const bool is3D : { false, true }) {
        SECTION(is3D ? "3D" : "2D") {
            std::vector<NiftiImage::dim_t> dims{ 24, 26 };
            if (is3D) dims.push_back(22);
            const NiftiImage reference = MakeIntensityImage(dims, 1, 2.f, 65.f);
            const NiftiImage floating = MakeIntensityImage(dims, 2, 2.f, 65.f);
            const size_t volume = reference.nVoxels();

            // Exclude every seventh voxel, once by NaN and once by mask
            NiftiImage withNan(reference, NiftiImage::Copy::Image);
            std::vector<int> mask(volume, 0);
            auto nanPtr = withNan.data();
            size_t excluded = 0;
            for (size_t i = 0; i < volume; ++i)
                if (i % 7 == 0) {
                    nanPtr[i] = std::numeric_limits<float>::quiet_NaN();
                    mask[i] = -1;
                    ++excluded;
                }

            const double viaNan = ComputeNmi(withNan, floating);
            const double viaMask = ComputeNmi(reference, floating, mask.data());
            const double unrestricted = ComputeNmi(reference, floating);

            NR_COUT << "  " << (is3D ? "3D" : "2D") << ": " << excluded << " of " << volume
                    << " voxels excluded; NaN " << std::fixed << std::setprecision(12) << viaNan
                    << ", mask " << viaMask << ", neither " << unrestricted << std::endl;

            // Excluding the voxels has to change the answer, or the comparison is vacuous
            INFO("excluding " << excluded << " voxels changed nothing, so this proves nothing");
            REQUIRE(viaNan != unrestricted);
            // And the two ways of excluding them have to agree
            INFO("NaN " << viaNan << " vs mask " << viaMask);
            REQUIRE(std::abs(viaNan - viaMask) < 1e-9);
        }
    }
}

TEST_CASE("The joint histogram does not depend on voxel order", "[unit]") {
    /*
        A histogram is a sum over voxels, so it is order-independent by construction - and the
        approximated fill relies on that, building per-thread partial histograms and merging them
        (in reg_getNmiValue). Permuting the voxels of both images together leaves every intensity
        pair intact, so the histogram, the entropies and the value must all be unchanged.
    */
    std::vector<NiftiImage::dim_t> dims{ 20, 22, 18 };
    const NiftiImage reference = MakeIntensityImage(dims, 3, 2.f, 65.f);
    const NiftiImage floating = MakeIntensityImage(dims, 4, 2.f, 65.f);
    const size_t volume = reference.nVoxels();

    NiftiImage shuffledRef(reference, NiftiImage::Copy::Image);
    NiftiImage shuffledFlo(floating, NiftiImage::Copy::Image);
    {
        std::vector<size_t> order(volume);
        std::iota(order.begin(), order.end(), size_t{ 0 });
        std::shuffle(order.begin(), order.end(), std::mt19937(5));
        const auto refPtr = reference.data();
        const auto floPtr = floating.data();
        auto outRef = shuffledRef.data();
        auto outFlo = shuffledFlo.data();
        for (size_t i = 0; i < volume; ++i) {
            outRef[i] = static_cast<float>(refPtr[order[i]]);
            outFlo[i] = static_cast<float>(floPtr[order[i]]);
        }
    }

    const double original = ComputeNmi(reference, floating);
    const double shuffled = ComputeNmi(shuffledRef, shuffledFlo);
    NR_COUT << "  original " << std::fixed << std::setprecision(12) << original
            << ", shuffled " << shuffled << ", difference " << std::scientific
            << std::abs(original - shuffled) << std::endl;
    INFO("original " << original << " vs shuffled " << shuffled);
    REQUIRE(std::abs(original - shuffled) < 1e-9);
}

TEST_CASE("The Parzen kernel's double instantiation carries only float precision", "[unit]") {
    /*
        The kernel's `2.f / 3.f` is a float constant expression: it is folded to the nearest float
        and only then widened to the template's type. So GetBasisSplineValue<double>(0) returns
        0.66666668653488159 rather than 2/3, an error of about 2e-8 per weight, which enters the joint
        histogram of every double instantiation.

        This is pinned rather than corrected. Writing the constants in the template's type would make
        the double instantiation exact, but it moves NMI values, so it belongs to whoever is willing to
        rebaseline them. Until then the bound above is set by this, not by double arithmetic.
    */
    const double peak = GetBasisSplineValue<double>(0.0);
    const double exact = 2.0 / 3.0;
    NR_COUT << "  kernel peak in double: " << std::setprecision(17) << peak
            << ", exact 2/3: " << exact << ", error " << std::scientific << std::abs(peak - exact)
            << std::endl;

    // The float instantiation is as exact as float allows, so nothing shows there
    REQUIRE(std::abs(double(GetBasisSplineValue<float>(0.f)) - double(2.f / 3.f)) == 0.0);
    // The double one is not exact, and is off by exactly the float representation error of 2/3
    INFO("peak " << peak << " vs 2/3 " << exact);
    REQUIRE(peak != exact);
    REQUIRE(std::abs(peak - exact) == std::abs(double(2.f / 3.f) - exact));
    REQUIRE(std::abs(peak - exact) < kKernelTolerance);
}
