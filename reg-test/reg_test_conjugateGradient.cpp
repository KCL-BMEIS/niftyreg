// OpenCL is not supported for this test
#undef USE_OPENCL

#include "reg_test_common.h"

/*
    The conjugate gradient optimiser, checked against hand-computed Polak-Ribiere updates rather than
    a transcription of the production loop.

    The production update (ConjugateGradient::UpdateGradientValues) keeps two state arrays and computes

        first call:  array1 = array2 = -g,   gradient unchanged
        update:      beta   = sum((g_new + array1) . g_new) / sum(array2 . array1)
                     array1 = -g_new
                     array2 = -g_new + beta * array2_old
                     g_out  = -array2 = g_new + beta * (previous direction)

    Feeding UNIFORM gradient fields makes every sum a closed form: with the field set to a then b,
    beta = b(b-a)/a^2 and every output element is b + beta*a. The values below are chosen so beta and
    the outputs are exact binary fractions, so the checks are equalities - including on CUDA, where a
    reduction order can differ but sums of identical dyadic values are exact in any order.

    What each case pins:
      - first call leaves the gradient untouched (steepest descent);
      - a REPEATED gradient leaves it untouched again: b == a gives beta = 0 exactly, which any sign
        or association error in the beta sums breaks;
      - one general update matches b + beta*a per element;
      - a second consecutive update matches the two-step closed form, pinning the state carry
        (array2 must hold the previous direction, not the previous gradient);
      - the symmetric variant pools the sums: beta = (b(b-a) + d(d-c)) / (a^2 + c^2) with the
        backward field at c then d, applied to BOTH sides;
      - RestartOptimisation() discards the direction history (next update behaves as a first call)
        but keeps the iteration count; Perturbation(0) does the same and resets the count.

    UpdateControlPointPosition is asserted against its definition, best + scale * gradient, per
    optimise flag - the only arithmetic involved.
*/

namespace {

struct CgFixture {
    NiftiImage reference, controlPointGrid, controlPointGridBw;
    unique_ptr<F3dContent> content, contentBw;
    unique_ptr<Optimiser<float>> optimiser;
    size_t volume = 0;
    int components = 0;

    struct NullOptimisable: public InterfaceOptimiser {
        virtual double GetObjectiveFunctionValue() override { return 0; }
        virtual void UpdateParameters(float) override {}
        virtual void UpdateBestObjFunctionValue() override {}
    };
    NullOptimisable callbacks;

    CgFixture(Platform& platform, bool is3D, bool isSymmetric) {
        std::vector<NiftiImage::dim_t> dims(is3D ? 3 : 2, 4);
        reference = NiftiImage(dims, NIFTI_TYPE_FLOAT32);
        setIdentitySform(reference);
        controlPointGrid = CreateControlPointGrid(reference);
        controlPointGridBw = controlPointGrid;
        components = is3D ? 3 : 2;

        unique_ptr<F3dContentCreator> creator{
            dynamic_cast<F3dContentCreator*>(platform.CreateContentCreator(ContentType::F3d)) };
        content.reset(creator->Create(reference, reference, controlPointGrid));
        volume = content->F3dContent::GetTransformationGradient().nVoxelsPerVolume();
        if (isSymmetric)
            contentBw.reset(creator->Create(reference, reference, controlPointGridBw));
        optimiser.reset(platform.CreateOptimiser<float>(*content, callbacks, 0, true, true, true, true,
                                                        contentBw.get()));
    }

    void SetGradient(F3dContent& con, float value) {
        NiftiImage& gradient = con.F3dContent::GetTransformationGradient();
        auto ptr = gradient.data();
        for (size_t i = 0; i < gradient.nVoxels(); ++i)
            ptr[i] = value;
        con.UpdateTransformationGradient();
    }

    void RequireGradientIs(F3dContent& con, float expected, const std::string& what) {
        NiftiImage& gradient = con.GetTransformationGradient();
        const auto ptr = gradient.data();
        size_t differing = 0;
        for (size_t i = 0; i < gradient.nVoxels(); ++i)
            if (static_cast<float>(ptr[i]) != expected) ++differing;
        INFO(what << ": expected every element == " << expected << ", " << differing << " differ, first is "
             << static_cast<float>(ptr[0]));
        REQUIRE(differing == 0);
    }
};

} // namespace

TEST_CASE("Conjugate gradient closed-form updates", "[unit]") {
    for (auto&& platformType : PlatformTypes)
        for (const bool is3D : { false, true }) {
            Platform platform(platformType);

            SECTION(std::string(is3D ? "3D" : "2D") + " " + platform.GetName() + " forward-only") {
                CgFixture f(platform, is3D, false);

                // First call: steepest descent, gradient untouched
                f.SetGradient(*f.content, 2.f);
                f.optimiser->UpdateGradientValues();
                f.RequireGradientIs(*f.content, 2.f, "first call");

                // Repeated gradient: beta = a(a-a)/a^2 = 0, direction re-initialises to -g
                f.SetGradient(*f.content, 2.f);
                f.optimiser->UpdateGradientValues();
                f.RequireGradientIs(*f.content, 2.f, "repeated gradient");

                // General update: a = 2 (state now holds direction -2), b = 3
                // beta = 3(3-2)/2^2 = 0.75; out = 3 + 0.75*2 = 4.5
                f.SetGradient(*f.content, 3.f);
                f.optimiser->UpdateGradientValues();
                f.RequireGradientIs(*f.content, 4.5f, "update a=2, b=3");

                // Second consecutive update, pinning the state carry: the previous direction is
                // -4.5 (not the previous gradient -3), so with e = -4.5:
                // beta = sum((e - 3) e) / sum((-4.5)(-3)) = 33.75 / 13.5 = 2.5
                // out = e + 2.5 * 4.5 * ... = -array2 = -( -e + 2.5 * (-4.5) ) = e + 11.25 = 6.75
                f.SetGradient(*f.content, -4.5f);
                f.optimiser->UpdateGradientValues();
                f.RequireGradientIs(*f.content, 6.75f, "chained update e=-4.5");
            }

            SECTION(std::string(is3D ? "3D" : "2D") + " " + platform.GetName() + " symmetric") {
                CgFixture f(platform, is3D, true);

                // Initialise both sides: forward a = 2, backward c = 2
                f.SetGradient(*f.content, 2.f);
                f.SetGradient(*f.contentBw, 2.f);
                f.optimiser->UpdateGradientValues();
                f.RequireGradientIs(*f.content, 2.f, "symmetric first call, forward");
                f.RequireGradientIs(*f.contentBw, 2.f, "symmetric first call, backward");

                // Update with forward b = 3, backward d = 1 (equal DOF counts):
                // beta = (b(b-a) + d(d-c)) / (a^2 + c^2) = (3 - 1) / 8 = 0.25
                // forward out = 3 + 0.25*2 = 3.5; backward out = 1 + 0.25*2 = 1.5
                f.SetGradient(*f.content, 3.f);
                f.SetGradient(*f.contentBw, 1.f);
                f.optimiser->UpdateGradientValues();
                f.RequireGradientIs(*f.content, 3.5f, "symmetric update, forward");
                f.RequireGradientIs(*f.contentBw, 1.5f, "symmetric update, backward");
            }

            SECTION(std::string(is3D ? "3D" : "2D") + " " + platform.GetName() + " restart and perturbation") {
                CgFixture f(platform, is3D, false);

                // Build up direction state
                f.SetGradient(*f.content, 2.f);
                f.optimiser->UpdateGradientValues();
                f.SetGradient(*f.content, 3.f);
                f.optimiser->UpdateGradientValues();
                f.RequireGradientIs(*f.content, 4.5f, "pre-restart update");

                // Restart discards the direction history but keeps the iteration count
                constexpr size_t iterAdvance = 3;
                for (size_t it = 0; it < iterAdvance; ++it)
                    f.optimiser->IncrementCurrentIterationNumber();
                f.optimiser->RestartOptimisation();
                REQUIRE(f.optimiser->GetCurrentIterationNumber() == iterAdvance);

                // Next update behaves as a first call: gradient untouched
                f.SetGradient(*f.content, 5.f);
                f.optimiser->UpdateGradientValues();
                f.RequireGradientIs(*f.content, 5.f, "first update after restart");

                // And the one after that is a first-order update from the fresh state:
                // a = 5, b = 10: beta = 10*5/25 = 2; out = 10 + 2*5 = 20
                f.SetGradient(*f.content, 10.f);
                f.optimiser->UpdateGradientValues();
                f.RequireGradientIs(*f.content, 20.f, "second update after restart");

                // Perturbation(0) also resets the direction state and additionally zeroes the
                // iteration count; a zero-length perturbation leaves the control points untouched
                for (size_t it = 0; it < iterAdvance; ++it)
                    f.optimiser->IncrementCurrentIterationNumber();
                f.optimiser->Perturbation(0);
                REQUIRE(f.optimiser->GetCurrentIterationNumber() == 0);

                f.SetGradient(*f.content, 7.f);
                f.optimiser->UpdateGradientValues();
                f.RequireGradientIs(*f.content, 7.f, "first update after perturbation");
            }
        }
}

TEST_CASE("Conjugate gradient control point update", "[unit]") {
    /*
        UpdateControlPointPosition writes best + scale * gradient into the optimised components and
        leaves the others at their current values. That IS the definition, so the expectation is the
        same expression evaluated in double on the same inputs - with a fixed non-uniform best/gradient
        pair so a component mix-up cannot cancel.
    */
    for (auto&& platformType : PlatformTypes)
        for (const bool is3D : { false, true })
            for (const bool optimiseX : { true, false })
                for (const bool optimiseY : { true, false })
                    for (const bool optimiseZ : { true, false }) {
                        Platform platform(platformType);
                        const std::string name = std::string(is3D ? "3D" : "2D") + " " + platform.GetName() +
                            (optimiseX ? " X" : " noX") + (optimiseY ? " Y" : " noY") + (optimiseZ ? " Z" : " noZ");
                        SECTION(name) {
                            constexpr float scale = 0.75f;   // exact in float
                            CgFixture f(platform, is3D, false);

                            // Distinct per-element best DOF and gradient
                            NiftiImage& cpg = f.content->F3dContent::GetControlPointGrid();
                            NiftiImage best(cpg, NiftiImage::Copy::Image);
                            {
                                auto bPtr = best.data();
                                for (size_t i = 0; i < best.nVoxels(); ++i)
                                    bPtr[i] = static_cast<float>(0.5 + 0.25 * (i % 7));
                                cpg.copyData(best);
                                f.content->UpdateControlPointGrid();
                            }
                            NiftiImage gradient(cpg, NiftiImage::Copy::ImageInfoAndAllocData);
                            {
                                auto gPtr = gradient.data();
                                for (size_t i = 0; i < gradient.nVoxels(); ++i)
                                    gPtr[i] = static_cast<float>(2.0 - 0.5 * (i % 5));
                                f.content->F3dContent::GetTransformationGradient().copyData(gradient);
                                f.content->UpdateTransformationGradient();
                            }

                            // Recreate the optimiser so it snapshots this best DOF
                            f.optimiser.reset(platform.CreateOptimiser<float>(*f.content, f.callbacks, 0, true,
                                                                              optimiseX, optimiseY, optimiseZ));
                            unique_ptr<Compute> compute{ platform.CreateCompute(*f.content) };
                            compute->UpdateControlPointPosition(f.optimiser->GetCurrentDof(),
                                                                f.optimiser->GetBestDof(),
                                                                f.optimiser->GetGradient(),
                                                                scale, optimiseX, optimiseY, optimiseZ);

                            const auto resultPtr = f.content->GetControlPointGrid().data();
                            const auto bestPtr = best.data();
                            const auto gradPtr = gradient.data();
                            const bool flags[3] = { optimiseX, optimiseY, optimiseZ && is3D };
                            for (int c = 0; c < f.components; ++c)
                                for (size_t i = 0; i < f.volume; ++i) {
                                    const size_t index = c * f.volume + i;
                                    const float expected = flags[c]
                                        ? static_cast<float>(bestPtr[index]) + scale * static_cast<float>(gradPtr[index])
                                        : static_cast<float>(bestPtr[index]);
                                    INFO(name << ": component " << c << " element " << i);
                                    REQUIRE(static_cast<float>(resultPtr[index]) == expected);
                                }
                        }
                    }
}
