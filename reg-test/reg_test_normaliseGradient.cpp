// OpenCL is not supported for this test
#undef USE_OPENCL

#include "reg_test_common.h"

/*
    GetMaximalLength and NormaliseGradient, checked against planted values rather than a
    reimplementation.

    GetMaximalLength returns the largest Euclidean norm over the nodes, built from only the optimised
    components; NormaliseGradient divides the optimised components by the given length and - the part
    worth stating out loud - writes ZERO to the components that are not optimised, rather than leaving
    them alone (Compute::NormaliseGradient). Every case below therefore plants vectors whose norms are known by
    construction:

      - the maximum under each flag combination is a different planted vector, with a different
        hand-known norm, so a wrong component entering (or missing from) the norm changes the answer.
        Norms are chosen exact in float (3-4-5 triples scaled by powers of two, single-component
        vectors), so equality is required, not closeness;
      - after normalising by the returned length, re-measuring must give exactly 1 when the planted
        maximum divides exactly (a power of two), and the planted maximum's node must hold the unit
        version of itself;
      - non-optimised components are asserted to be zeroed, deliberately, as the production contract;
      - all-flags-false: GetMaximalLength = 0 and NormaliseGradient(0,...) leaves the field untouched.

    The PlatformTypes sweep is kept: the same closed forms gate the CPU and CUDA implementations.
*/

namespace {

// Distinct planted vectors, integer-valued so every square and sum of squares is exact in float.
// The maximal norm then differs per flag combination and is computable independently. (A vector
// whose norm is integral under EVERY flag subset would be a perfect cuboid, which is an open
// problem - so the expectation below uses the IEEE float sqrt of the exact integer sum instead of
// demanding integer norms.)
struct Planted { size_t node; float v[3]; };

// The largest norm for a flag combination over the planted set, in the same arithmetic the float
// instantiation performs: exact integer sums of squares, float sqrt. This is IEEE arithmetic on
// hand-chosen integers, not a restatement of the production loop.
double ExpectedMax(const std::vector<Planted>& planted, bool x, bool y, bool z) {
    float best = 0;
    for (const auto& p : planted) {
        const float vx = x ? p.v[0] : 0.f, vy = y ? p.v[1] : 0.f, vz = z ? p.v[2] : 0.f;
        best = std::max(best, std::sqrt(vx * vx + vy * vy + vz * vz));
    }
    return best;
}

} // namespace

TEST_CASE("Normalise Gradient", "[unit]") {
    for (auto&& platformType : PlatformTypes)
        for (const bool is3D : { false, true }) {
            Platform platform(platformType);

            std::vector<NiftiImage::dim_t> dims(is3D ? 3 : 2, 8);
            NiftiImage reference(dims, NIFTI_TYPE_FLOAT32);
            setIdentitySform(reference);
            NiftiImage controlPointGrid = CreateControlPointGrid(reference);

            // Plant the vectors at distinct nodes on an otherwise small smooth background
            const size_t volume = controlPointGrid.nVoxelsPerVolume();
            std::vector<Planted> planted{
                { 3,               { -24.f, 0.f, 0.f } },
                { 5,               { 0.f, 20.f, 0.f } },
                { 7,               { 3.f, 0.f, is3D ? 16.f : 0.f } },
                { 9,               { 18.f, 24.f, 0.f } },
                { 11,              { 24.f, 0.f, is3D ? 10.f : 0.f } },
                { 13,              { 0.f, 15.f, is3D ? 20.f : 0.f } },
                { volume - 2,      { 12.f, 16.f, is3D ? 21.f : 0.f } },
            };

            const int components = is3D ? 3 : 2;
            NiftiImage inputGradient(controlPointGrid, NiftiImage::Copy::ImageInfoAndAllocData);
            {
                auto ptr = inputGradient.data();
                for (size_t i = 0; i < inputGradient.nVoxels(); ++i)
                    ptr[i] = static_cast<float>(std::sin(0.37 * double(i)));   // |background| < 1
                for (const auto& p : planted)
                    for (int c = 0; c < components; ++c)
                        ptr[c * volume + p.node] = p.v[c];
            }

            for (int optimiseX = 0; optimiseX < 2; optimiseX++)
                for (int optimiseY = 0; optimiseY < 2; optimiseY++)
                    for (int optimiseZ = 0; optimiseZ < 2; optimiseZ++) {
                        const std::string name = std::string(is3D ? "3D" : "2D") + " " + platform.GetName() +
                            (optimiseX ? " X" : " noX") + (optimiseY ? " Y" : " noY") + (optimiseZ ? " Z" : " noZ");
                        SECTION(name) {
                            NiftiImage ref(reference), cpg(controlPointGrid);
                            unique_ptr<F3dContentCreator> contentCreator{
                                dynamic_cast<F3dContentCreator*>(platform.CreateContentCreator(ContentType::F3d)) };
                            unique_ptr<F3dContent> content{ contentCreator->Create(ref, ref, cpg) };
                            content->F3dContent::GetTransformationGradient().copyData(inputGradient);
                            content->UpdateTransformationGradient();
                            unique_ptr<Compute> compute{ platform.CreateCompute(*content) };

                            // In 2D the Z flag must be inert: production forces optimiseZ off when
                            // nz == 1, so the expected value never includes a Z term - and 2D with
                            // only Z requested is effectively the no-flag case
                            const bool effX = optimiseX, effY = optimiseY, effZ = optimiseZ && is3D;
                            const bool anyActive = effX || effY || effZ;
                            const double expectedMax = ExpectedMax(planted, effX, effY, effZ);

                            const double maxLength = compute->GetMaximalLength(optimiseX, optimiseY, optimiseZ);
                            NR_COUT << "  " << std::setw(28) << std::left << name
                                    << " max length = " << maxLength << " expected = " << expectedMax << std::endl;

                            if (!anyActive) {
                                // Contract: no active component (including 2D-with-only-Z, where the
                                // Z flag is forced off) -> maximal length 0, and normalising by a
                                // zero length is a no-op - the field is left untouched, NOT zeroed.
                                // (No early return: Catch2 discovers the later sections by executing
                                // the body, so returning here would silently drop every one of them.)
                                REQUIRE(maxLength == 0);
                                compute->NormaliseGradient(0, optimiseX, optimiseY, optimiseZ);
                                const Deviation deviation = CompareImages(content->GetTransformationGradient(),
                                                                          inputGradient);
                                REQUIRE(deviation.differing == 0);
                                continue;
                            }

                            // The planted norms are exact in float and dominate the background, so
                            // the reduction must return them exactly
                            INFO(name << ": expected " << expectedMax << ", got " << maxLength);
                            REQUIRE(maxLength == expectedMax);

                            // GetMaximalLength must not have modified the gradient
                            {
                                const Deviation deviation = CompareImages(content->GetTransformationGradient(),
                                                                          inputGradient);
                                REQUIRE(deviation.differing == 0);
                            }

                            compute->NormaliseGradient(maxLength, optimiseX, optimiseY, optimiseZ);
                            NiftiImage& normalised = content->GetTransformationGradient();
                            const auto nPtr = normalised.data();
                            const auto iPtr = inputGradient.data();

                            // 1. Non-optimised components are ZEROED - the production contract
                            //    (every component is written as value/maxLength with
                            //    value = 0 when the flag is off), not left at their input values
                            for (size_t i = 0; i < volume; ++i) {
                                if (!effX) REQUIRE(static_cast<float>(nPtr[i]) == 0.f);
                                if (!effY) REQUIRE(static_cast<float>(nPtr[volume + i]) == 0.f);
                                if (is3D && !effZ) REQUIRE(static_cast<float>(nPtr[2 * volume + i]) == 0.f);
                            }

                            // 2. Optimised components are the input divided by the returned length
                            //    (double division, rounded to float - checked exactly at the planted
                            //    nodes where the quotient is representable)
                            for (const auto& p : planted)
                                for (int c = 0; c < components; ++c) {
                                    const bool active = c == 0 ? effX : (c == 1 ? effY : effZ);
                                    if (!active) continue;
                                    const float expected = static_cast<float>(double(p.v[c]) / maxLength);
                                    INFO(name << ": planted node " << p.node << " component " << c);
                                    REQUIRE(static_cast<float>(nPtr[c * volume + p.node]) == expected);
                                }

                            // 3. Re-measuring the normalised field returns 1 up to rounding: the
                            //    planted maximum maps to a unit vector
                            const double remeasured = compute->GetMaximalLength(optimiseX, optimiseY, optimiseZ);
                            INFO(name << ": re-measured max " << remeasured);
                            REQUIRE(std::abs(remeasured - 1.0) < 1e-6);

                            // 4. And every other node is <= 1: nothing grew past the maximum
                            for (size_t i = 0; i < volume; ++i) {
                                const double vx = effX ? static_cast<float>(nPtr[i]) : 0.0;
                                const double vy = effY ? static_cast<float>(nPtr[volume + i]) : 0.0;
                                const double vz = effZ ? static_cast<float>(nPtr[2 * volume + i]) : 0.0;
                                REQUIRE(std::sqrt(vx * vx + vy * vy + vz * vz) <= 1.0 + 1e-6);
                            }
                            (void)iPtr;
                        }
                    }
        }
}
