// OpenCL is not supported for this test
#undef USE_OPENCL

#include "reg_test_common.h"
#include "_reg_tools.h"
#include <set>

/*
    The reg_tools image utilities, checked against closed forms.

    Every operation here is simple enough to have an exact per-voxel specification, so no case needs
    a reference implementation: arithmetic against hand values and inverse-pair identities, rescaling
    and thresholding against their defining formulas, the statistics against hand-computed numbers
    (pinning the conventions: the standard deviation is the POPULATION one, both honour
    scl_slope/scl_inter), the deformation/displacement conversion against the sform-applied voxel
    positions, and the pyramid against invariants a smoothing-and-halving must preserve (a constant
    image, the geometry, and - away from the boundary - a linear ramp, which symmetric smoothing
    leaves untouched).

    The convolution family is deliberately absent: it has its own regression and property tests.
*/

namespace {

NiftiImage MakeValueImage(const std::vector<NiftiImage::dim_t>& dims, const std::vector<float>& values) {
    NiftiImage img(dims, NIFTI_TYPE_FLOAT32);
    setIdentitySform(img);
    auto ptr = img.data();
    REQUIRE(img.nVoxels() == values.size());
    for (size_t i = 0; i < values.size(); ++i)
        ptr[i] = values[i];
    return img;
}

void RequireImageValues(const NiftiImage& img, const std::vector<float>& expected, const std::string& what) {
    const auto ptr = img.data();
    REQUIRE(img.nVoxels() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        INFO(what << ": voxel " << i);
        REQUIRE(static_cast<float>(ptr[i]) == expected[i]);
    }
}

} // namespace

TEST_CASE("Tools: image arithmetic", "[unit]") {
    const std::vector<NiftiImage::dim_t> dims{ 2, 2 };
    const NiftiImage a = MakeValueImage(dims, { 1.f, -2.f, 4.f, 0.5f });
    const NiftiImage b = MakeValueImage(dims, { 2.f, 4.f, -1.f, 0.25f });

    SECTION("image-image, hand values") {
        NiftiImage out(a, NiftiImage::Copy::ImageInfoAndAllocData);
        reg_tools_addImageToImage(a, b, out);
        RequireImageValues(out, { 3.f, 2.f, 3.f, 0.75f }, "a + b");
        reg_tools_subtractImageFromImage(a, b, out);
        RequireImageValues(out, { -1.f, -6.f, 5.f, 0.25f }, "a - b");
        reg_tools_multiplyImageToImage(a, b, out);
        RequireImageValues(out, { 2.f, -8.f, -4.f, 0.125f }, "a * b");
        reg_tools_divideImageToImage(a, b, out);
        RequireImageValues(out, { 0.5f, -0.5f, -4.f, 2.f }, "a / b");
    }
    SECTION("image-value, hand values") {
        NiftiImage out(a, NiftiImage::Copy::ImageInfoAndAllocData);
        reg_tools_addValueToImage(a, out, 1.5f);
        RequireImageValues(out, { 2.5f, -0.5f, 5.5f, 2.f }, "a + 1.5");
        reg_tools_subtractValueFromImage(a, out, 1.5f);
        RequireImageValues(out, { -0.5f, -3.5f, 2.5f, -1.f }, "a - 1.5");
        reg_tools_multiplyValueToImage(a, out, -2.f);
        RequireImageValues(out, { -2.f, 4.f, -8.f, -1.f }, "a * -2");
        reg_tools_divideValueToImage(a, out, 4.f);
        RequireImageValues(out, { 0.25f, -0.5f, 1.f, 0.125f }, "a / 4");
    }
    SECTION("inverse pairs return the input exactly") {
        // The values are dyadic, so (a + b) - b and (a * b) / b are exact in float
        NiftiImage tmp(a, NiftiImage::Copy::ImageInfoAndAllocData);
        NiftiImage back(a, NiftiImage::Copy::ImageInfoAndAllocData);
        reg_tools_addImageToImage(a, b, tmp);
        reg_tools_subtractImageFromImage(tmp, b, back);
        RequireImageValues(back, { 1.f, -2.f, 4.f, 0.5f }, "(a + b) - b");
        reg_tools_multiplyImageToImage(a, b, tmp);
        reg_tools_divideImageToImage(tmp, b, back);
        RequireImageValues(back, { 1.f, -2.f, 4.f, 0.5f }, "(a * b) / b");
    }
}

TEST_CASE("Tools: intensity rescale and threshold", "[unit]") {
    SECTION("rescale maps [min, max] to [newMin, newMax] linearly") {
        // Values 10..20: rescaled = (v - 10) / 10 * (newMax - newMin) + newMin, exact for these
        NiftiImage img = MakeValueImage({ 5 }, { 10.f, 12.5f, 15.f, 17.5f, 20.f });
        reg_intensityRescale(img, 0, 2.f, 6.f);
        RequireImageValues(img, { 2.f, 3.f, 4.f, 5.f, 6.f }, "rescaled");
    }
    SECTION("threshold clamps into [lower, upper]") {
        NiftiImage img = MakeValueImage({ 5 }, { -3.f, -1.f, 0.f, 1.f, 3.f });
        reg_thresholdImage<float>(img, -1.f, 1.f);
        RequireImageValues(img, { -1.f, -1.f, 0.f, 1.f, 1.f }, "thresholded");
    }
    SECTION("binarise: zero against non-zero, then against a threshold") {
        NiftiImage img = MakeValueImage({ 5 }, { 0.f, -2.f, 0.f, 0.5f, 3.f });
        reg_tools_binarise_image(img);
        RequireImageValues(img, { 0.f, 1.f, 0.f, 1.f, 1.f }, "binarised (non-zero)");

        // Convention pinned from production: value < thr -> 0, >= thr -> 1
        NiftiImage img2 = MakeValueImage({ 5 }, { 0.f, 0.999f, 1.f, 1.001f, -5.f });
        reg_tools_binarise_image(img2, 1.f);
        RequireImageValues(img2, { 0.f, 0.f, 1.f, 1.f, 0.f }, "binarised (threshold, >= is 1)");
    }
    SECTION("binary image to int array") {
        const NiftiImage img = MakeValueImage({ 4 }, { 0.f, 1.f, 1.f, 0.f });
        std::vector<int> array(4, -7);
        reg_tools_binaryImage2int(img, array.data());
        // Pinned: background is -1, foreground counts from 0
        REQUIRE(array[0] == -1);
        REQUIRE(array[1] > -1);
        REQUIRE(array[2] > -1);
        REQUIRE(array[3] == -1);
    }
}

TEST_CASE("Tools: NaN masking", "[unit]") {
    SECTION("nanMask sets NaN exactly where the mask is background") {
        const NiftiImage img = MakeValueImage({ 4 }, { 1.f, 2.f, 3.f, 4.f });
        const NiftiImage mask = MakeValueImage({ 4 }, { 1.f, 0.f, 1.f, 0.f });
        NiftiImage out(img, NiftiImage::Copy::ImageInfoAndAllocData);
        reg_tools_nanMask_image(img, mask, out);
        const auto ptr = out.data();
        REQUIRE(static_cast<float>(ptr[0]) == 1.f);
        REQUIRE(std::isnan(static_cast<float>(ptr[1])));
        REQUIRE(static_cast<float>(ptr[2]) == 3.f);
        REQUIRE(std::isnan(static_cast<float>(ptr[3])));
    }
    SECTION("removeNanFromMask deactivates exactly the NaN voxels") {
        const NiftiImage img = MakeValueImage({ 4 }, { 1.f, std::numeric_limits<float>::quiet_NaN(),
                                                       3.f, std::numeric_limits<float>::quiet_NaN() });
        std::vector<int> mask{ 0, 0, -1, 0 };
        reg_tools_removeNanFromMask(img, mask.data());
        REQUIRE(mask[0] == 0);
        REQUIRE(mask[1] == -1);   // NaN -> background
        REQUIRE(mask[2] == -1);   // already background, stays
        REQUIRE(mask[3] == -1);
    }
}

TEST_CASE("Tools: image statistics", "[unit]") {
    // {1, 2, 3, 4}: mean 2.5; POPULATION std = sqrt(5/4) - the convention production implements
    // (reg_tools_getSTDValue divides by nvox, not nvox - 1)
    const NiftiImage img = MakeValueImage({ 4 }, { 1.f, 2.f, 3.f, 4.f });
    REQUIRE(reg_tools_getMeanValue(img) == 2.5f);
    REQUIRE(std::abs(reg_tools_getSTDValue(img) - std::sqrt(1.25f)) < 1e-6f);

    SECTION("scl_slope and scl_inter are honoured") {
        NiftiImage scaled(img, NiftiImage::Copy::Image);
        scaled->scl_slope = 2.f;
        scaled->scl_inter = 10.f;
        // Values become 12, 14, 16, 18: mean 15, std doubled
        REQUIRE(reg_tools_getMeanValue(scaled) == 15.f);
        REQUIRE(std::abs(reg_tools_getSTDValue(scaled) - 2.f * std::sqrt(1.25f)) < 1e-5f);
    }
    SECTION("mean RMS of vector images") {
        // Identical images: 0. Images differing by the constant vector (3, 4): per-voxel distance 5.
        NiftiImage vecA({ 2, 2, 1, 1, 2 }, NIFTI_TYPE_FLOAT32);
        setIdentitySform(vecA);
        { auto p = vecA.data(); for (size_t i = 0; i < vecA.nVoxels(); ++i) p[i] = float(i); }
        REQUIRE(reg_tools_getMeanRMS(vecA, vecA) == 0.0);

        NiftiImage vecB(vecA, NiftiImage::Copy::Image);
        {
            auto p = vecB.data();
            const size_t volume = vecB.nVoxelsPerVolume();
            for (size_t i = 0; i < volume; ++i) {
                p[i] = static_cast<float>(p[i]) + 3.f;              // x components
                p[volume + i] = static_cast<float>(p[volume + i]) + 4.f;   // y components
            }
        }
        REQUIRE(std::abs(reg_tools_getMeanRMS(vecA, vecB) - 5.0) < 1e-6);
    }
}

TEST_CASE("Tools: deformation and displacement conversion", "[unit]") {
    /*
        A deformation field stores positions, a displacement field stores position minus the voxel's
        own world coordinate: converting between them adds or subtracts sform * index per voxel. An
        anisotropic, sheared sform makes that world coordinate a genuine matrix product.
    */
    for (const bool is3D : { false, true }) {
        SECTION(is3D ? "3D" : "2D") {
            std::vector<NiftiImage::dim_t> dims(is3D ? 3 : 2, 4);
            NiftiImage reference(dims, NIFTI_TYPE_FLOAT32);
            setAnisotropicSform(reference);
            NiftiImage field = CreateDeformationField(reference);   // identity deformation
            const mat44 sform = reference->sto_xyz;

            // The identity deformation minus the world coordinates is the zero displacement
            NiftiImage displacement(field, NiftiImage::Copy::Image);
            reg_getDisplacementFromDeformation(displacement);
            {
                const auto ptr = displacement.data();
                for (size_t i = 0; i < displacement.nVoxels(); ++i) {
                    INFO("displacement voxel " << i);
                    REQUIRE(std::abs(static_cast<float>(ptr[i])) < 1e-5f);
                }
            }

            // And the round trip restores the positions: sform * index, checked in double
            reg_getDeformationFromDisplacement(displacement);
            const size_t volume = displacement.nVoxelsPerVolume();
            const auto ptr = displacement.data();
            const int nx = displacement->nx, ny = displacement->ny, nz = displacement->nz;
            const int components = is3D ? 3 : 2;
            for (int k = 0; k < nz; ++k)
                for (int j = 0; j < ny; ++j)
                    for (int i = 0; i < nx; ++i) {
                        const size_t index = (static_cast<size_t>(k) * ny + j) * nx + i;
                        for (int c = 0; c < components; ++c) {
                            const double expected = double(sform.m[c][0]) * i + double(sform.m[c][1]) * j +
                                                    double(sform.m[c][2]) * k + double(sform.m[c][3]);
                            INFO("voxel (" << i << "," << j << "," << k << ") component " << c);
                            REQUIRE(std::abs(static_cast<double>(ptr[c * volume + index]) - expected) < 1e-5);
                        }
                    }
        }
    }
}

TEST_CASE("Tools: gradient component zeroing", "[unit]") {
    NiftiImage gradient({ 2, 2, 2, 1, 3 }, NIFTI_TYPE_FLOAT32);
    setIdentitySform(gradient);
    const size_t volume = gradient.nVoxelsPerVolume();
    { auto p = gradient.data(); for (size_t i = 0; i < gradient.nVoxels(); ++i) p[i] = float(i + 1); }

    reg_setGradientToZero(gradient, false, true, false);   // zero only the y components
    const auto p = gradient.data();
    for (size_t i = 0; i < volume; ++i) {
        REQUIRE(static_cast<float>(p[i]) == float(i + 1));                    // x untouched
        REQUIRE(static_cast<float>(p[volume + i]) == 0.f);                    // y zeroed
        REQUIRE(static_cast<float>(p[2 * volume + i]) == float(2 * volume + i + 1));   // z untouched
    }
}

TEST_CASE("Tools: datatype conversion round trip", "[unit]") {
    const NiftiImage source = MakeValueImage({ 4 }, { -1.5f, 0.f, 2.25f, 100.f });
    NiftiImage img(source, NiftiImage::Copy::Image);
    reg_tools_changeDatatype<double>(img);
    REQUIRE(img->datatype == NIFTI_TYPE_FLOAT64);
    reg_tools_changeDatatype<float>(img);
    REQUIRE(img->datatype == NIFTI_TYPE_FLOAT32);
    RequireImageValues(img, { -1.5f, 0.f, 2.25f, 100.f }, "float->double->float");
}

TEST_CASE("Tools: header utilities", "[unit]") {
    SECTION("real image spacing is the column norm of the rotated sform") {
        // A pure rotation keeps unit spacing whatever the angle; scaling the columns scales it
        NiftiImage img({ 4, 4, 4 }, NIFTI_TYPE_FLOAT32);
        const double theta = 0.5;
        mat44 m;
        Mat44Eye(&m);
        m.m[0][0] = static_cast<float>(2 * std::cos(theta)); m.m[0][1] = static_cast<float>(-3 * std::sin(theta));
        m.m[1][0] = static_cast<float>(2 * std::sin(theta)); m.m[1][1] = static_cast<float>(3 * std::cos(theta));
        m.m[2][2] = 1.5f;
        setSform(img, m);
        float spacing[3];
        reg_getRealImageSpacing(img, spacing);
        REQUIRE(std::abs(spacing[0] - 2.f) < 1e-5f);
        REQUIRE(std::abs(spacing[1] - 3.f) < 1e-5f);
        REQUIRE(std::abs(spacing[2] - 1.5f) < 1e-5f);
    }
    SECTION("removeSCLInfo folds slope and intercept into the values") {
        NiftiImage img = MakeValueImage({ 3 }, { 1.f, 2.f, 3.f });
        img->scl_slope = 2.f;
        img->scl_inter = -1.f;
        reg_tools_removeSCLInfo(img);
        REQUIRE(img->scl_slope == 1.f);
        REQUIRE(img->scl_inter == 0.f);
        RequireImageValues(img, { 1.f, 3.f, 5.f }, "scl folded");
    }
}

TEST_CASE("Tools: downsampling and pyramids", "[unit]") {
    SECTION("a constant image stays constant and the geometry halves") {
        NiftiImage img({ 8, 8, 8 }, NIFTI_TYPE_FLOAT32);
        setIdentitySform(img);
        { auto p = img.data(); for (size_t i = 0; i < img.nVoxels(); ++i) p[i] = 3.f; }

        bool downsampleAxis[8] = { false, true, true, true, false, false, false, false };
        reg_downsampleImage<float>(img, true, downsampleAxis);

        REQUIRE(img->nx == 4);
        REQUIRE(img->ny == 4);
        REQUIRE(img->nz == 4);
        REQUIRE(std::abs(img->dx - 2.f) < 1e-5f);   // spacing doubles as dims halve
        const auto p = img.data();
        for (size_t i = 0; i < img.nVoxels(); ++i) {
            INFO("voxel " << i);
            // Gaussian smoothing of a constant is the constant; the resampling then reads it back
            REQUIRE(std::abs(static_cast<float>(p[i]) - 3.f) < 1e-4f);
        }
    }
    SECTION("a linear world-coordinate ramp survives downsampling away from the boundary") {
        // Symmetric smoothing preserves linear functions in the interior, and the downsampled voxel
        // centres sit at known world positions - so the interior values have a closed form
        NiftiImage img({ 16, 16, 16 }, NIFTI_TYPE_FLOAT32);
        setIdentitySform(img);
        const double c[3] = { 2.0, -1.0, 0.5 };
        {
            auto p = img.data();
            for (int k = 0; k < 16; ++k)
                for (int j = 0; j < 16; ++j)
                    for (int i = 0; i < 16; ++i)
                        p[(static_cast<size_t>(k) * 16 + j) * 16 + i] =
                            static_cast<float>(c[0] * i + c[1] * j + c[2] * k);
        }
        bool downsampleAxis[8] = { false, true, true, true, false, false, false, false };
        reg_downsampleImage<float>(img, true, downsampleAxis);

        const mat44 voxelToReal = img->sto_xyz;
        const auto p = img.data();
        const int n = img->nx;
        double maxDeviation = 0;
        size_t checked = 0;
        for (int k = 2; k < n - 2; ++k)
            for (int j = 2; j < n - 2; ++j)
                for (int i = 2; i < n - 2; ++i) {
                    double world[3];
                    for (int d = 0; d < 3; ++d)
                        world[d] = double(voxelToReal.m[d][0]) * i + double(voxelToReal.m[d][1]) * j +
                                   double(voxelToReal.m[d][2]) * k + double(voxelToReal.m[d][3]);
                    const double expected = c[0] * world[0] + c[1] * world[1] + c[2] * world[2];
                    const double actual = p[(static_cast<size_t>(k) * n + j) * n + i];
                    maxDeviation = std::max(maxDeviation, std::abs(actual - expected));
                    ++checked;
                }
        NR_COUT << "  downsampled ramp: " << checked << " interior voxels, max deviation "
                << std::scientific << maxDeviation << std::endl;
        REQUIRE(checked > 0);
        REQUIRE(maxDeviation < 1e-2);
    }
    SECTION("image and mask pyramids: geometry per level, full mask stays full") {
        /*
            reg_createImagePyramid only downsamples an axis while half of it stays at least 32
            voxels, per axis: 64x64x16 halves x and y once (64->32) and never z,
            and the coarsest level repeats the previous one because 32/2 < 32. Pinned - the floor is
            what keeps small images out of the pyramid entirely.
        */
        NiftiImage img({ 64, 64, 16 }, NIFTI_TYPE_FLOAT32);
        setIdentitySform(img);
        { auto p = img.data(); for (size_t i = 0; i < img.nVoxels(); ++i) p[i] = 1.f; }

        constexpr unsigned levels = 3;
        vector<NiftiImage> pyramid(levels);
        reg_createImagePyramid<float>(img, pyramid, levels, levels);
        REQUIRE(pyramid[2]->nx == 64);   // finest level keeps the input size
        REQUIRE(pyramid[2]->nz == 16);
        REQUIRE(pyramid[1]->nx == 32);   // x and y halve once
        REQUIRE(pyramid[1]->ny == 32);
        REQUIRE(pyramid[1]->nz == 16);   // z is already below the floor and never halves
        REQUIRE(pyramid[0]->nx == 32);   // 32/2 < 32: the floor stops further halving
        REQUIRE(pyramid[0]->nz == 16);

        vector<unique_ptr<int[]>> maskPyramid(levels);
        NiftiImage mask({ 64, 64, 16 }, NIFTI_TYPE_UINT8);
        setIdentitySform(mask);
        { auto p = mask.data(); for (size_t i = 0; i < mask.nVoxels(); ++i) p[i] = 1; }
        reg_createMaskPyramid<float>(mask, maskPyramid, levels, levels);
        for (unsigned l = 0; l < levels; ++l) {
            const size_t voxels = pyramid[l].nVoxelsPerVolume();
            size_t active = 0;
            for (size_t i = 0; i < voxels; ++i)
                if (maskPyramid[l][i] > -1) ++active;
            INFO("level " << l);
            REQUIRE(active == voxels);   // a full mask must stay full at every level
        }
    }
}

TEST_CASE("Tools: arithmetic across every supported datatype", "[unit]") {
    /*
        Every public tools function dispatches over the full set of NIfTI datatypes, and only the
        float32 case is exercised by the tests above. The same closed forms hold for every type when
        the operands and results are small integers each type represents exactly - so one sweep pins
        all the dispatch cases with the oracle unchanged.
    */
    const int datatypes[] = { NIFTI_TYPE_UINT8, NIFTI_TYPE_INT8, NIFTI_TYPE_UINT16, NIFTI_TYPE_INT16,
                              NIFTI_TYPE_UINT32, NIFTI_TYPE_INT32, NIFTI_TYPE_FLOAT32, NIFTI_TYPE_FLOAT64 };
    // Without an explicit code, changeDatatype infers the type from sizeof alone and supports only
    // uchar/float/double - so the integer types pass their code explicitly
    const auto toType = [](NiftiImage img, int datatype) {
        switch (datatype) {
        case NIFTI_TYPE_UINT8: reg_tools_changeDatatype<unsigned char>(img, datatype); break;
        case NIFTI_TYPE_INT8: reg_tools_changeDatatype<char>(img, datatype); break;
        case NIFTI_TYPE_UINT16: reg_tools_changeDatatype<unsigned short>(img, datatype); break;
        case NIFTI_TYPE_INT16: reg_tools_changeDatatype<short>(img, datatype); break;
        case NIFTI_TYPE_UINT32: reg_tools_changeDatatype<unsigned>(img, datatype); break;
        case NIFTI_TYPE_INT32: reg_tools_changeDatatype<int>(img, datatype); break;
        case NIFTI_TYPE_FLOAT32: reg_tools_changeDatatype<float>(img, datatype); break;
        default: reg_tools_changeDatatype<double>(img, datatype); break;
        }
        return img;
    };
    const auto backToFloat = [](NiftiImage img) {
        reg_tools_changeDatatype<float>(img);
        return img;
    };

    // Small positive integers, all representable in every type, with exact quotients
    const NiftiImage a = MakeValueImage({ 2, 2 }, { 8.f, 12.f, 20.f, 6.f });
    const NiftiImage b = MakeValueImage({ 2, 2 }, { 2.f, 3.f, 5.f, 2.f });

    for (const int datatype : datatypes) {
        SECTION("datatype " + std::to_string(datatype)) {
            const NiftiImage at = toType(a, datatype);
            const NiftiImage bt = toType(b, datatype);
            REQUIRE(at->datatype == datatype);

            NiftiImage out(at, NiftiImage::Copy::ImageInfoAndAllocData);
            reg_tools_addImageToImage(at, bt, out);
            RequireImageValues(backToFloat(out), { 10.f, 15.f, 25.f, 8.f }, "a + b");
            reg_tools_subtractImageFromImage(at, bt, out);
            RequireImageValues(backToFloat(out), { 6.f, 9.f, 15.f, 4.f }, "a - b");
            reg_tools_multiplyImageToImage(at, bt, out);
            RequireImageValues(backToFloat(out), { 16.f, 36.f, 100.f, 12.f }, "a * b");
            reg_tools_divideImageToImage(at, bt, out);
            RequireImageValues(backToFloat(out), { 4.f, 4.f, 4.f, 3.f }, "a / b");
            reg_tools_addValueToImage(at, out, 5.f);
            RequireImageValues(backToFloat(out), { 13.f, 17.f, 25.f, 11.f }, "a + 5");
            reg_tools_multiplyValueToImage(at, out, 2.f);
            RequireImageValues(backToFloat(out), { 16.f, 24.f, 40.f, 12.f }, "a * 2");

            // The statistics and the value transforms share the same dispatch
            REQUIRE(reg_tools_getMeanValue(at) == 11.5f);
            NiftiImage thresholded(at, NiftiImage::Copy::Image);
            switch (datatype) {   // reg_thresholdImage is templated on the threshold type
            case NIFTI_TYPE_FLOAT64: reg_thresholdImage<double>(thresholded, 7., 15.); break;
            default: reg_thresholdImage<float>(thresholded, 7.f, 15.f); break;
            }
            RequireImageValues(backToFloat(thresholded), { 8.f, 12.f, 15.f, 7.f }, "threshold [7,15]");

            NiftiImage binarised(at, NiftiImage::Copy::Image);
            reg_tools_binarise_image(binarised, 10.f);
            RequireImageValues(backToFloat(binarised), { 0.f, 1.f, 1.f, 0.f }, "binarise thr 10");

            NiftiImage rescaled(at, NiftiImage::Copy::Image);
            reg_intensityRescale(rescaled, 0, 0.f, 7.f);   // [6, 20] -> [0, 7]: exact halves
            RequireImageValues(backToFloat(rescaled), { 1.f, 3.f, 7.f, 0.f }, "rescale [0,7]");
        }
    }
}

TEST_CASE("Tools: header repair", "[unit]") {
    /*
        reg_checkAndCorrectDimension is what every image passes through on load (called from
        reg_io_ReadImageFile): each repair rule is a closed form on a deliberately broken header.
    */
    NiftiImage img({ 8, 8 }, NIFTI_TYPE_FLOAT32);
    setIdentitySform(img);
    // Break the header: zeroed trailing dims, zero slope, zero pixdims on the degenerate axes
    img->dim[3] = img->nz = 0;
    img->dim[4] = img->nt = 0;
    img->dim[5] = img->nu = -2;
    img->scl_slope = 0.f;
    img->dz = img->pixdim[3] = 0.f;

    reg_checkAndCorrectDimension(img);

    REQUIRE(img->nz == 1);          // degenerate dims are forced to one
    REQUIRE(img->nt == 1);
    REQUIRE(img->nu == 1);
    REQUIRE(img->ndim == 2);        // and ndim is recomputed from the highest non-trivial dim
    REQUIRE(img->scl_slope == 1.f); // a zero slope means unscaled, not annihilated
    REQUIRE(img->dz == 1.f);        // degenerate axes get unit spacing
    REQUIRE(img->pixdim[3] == 1.f);
}

TEST_CASE("Tools: image file name detection", "[unit]") {
    // The suffix table the command-line apps use to distinguish an image argument from a text matrix
    for (const auto& [name, isImage] : {
             std::pair{ "brain.nii", true }, std::pair{ "brain.nii.gz", true },
             std::pair{ "brain.hdr", true }, std::pair{ "brain.img", true },
             std::pair{ "brain.img.gz", true },
             std::pair{ "affine.txt", false }, std::pair{ "matrix.mat", false } }) {
        INFO(name);
        REQUIRE(reg_isAnImageFileName(name) == isImage);
    }
}

TEST_CASE("Tools: label-preserving smoothing", "[unit]") {
    /*
        reg_tools_labelKernelConvolution (reg_tools -smoL) smooths a LABEL image: each voxel takes
        the label with the largest smoothed weight in its neighbourhood. Two properties define it
        without restating the kernel:
          - closure: the output can only contain labels the input contained - a value-smoothing
            convolution would blend 0 and 4 into non-labels;
          - a uniform label image is a fixed point.
    */
    SECTION("uniform image is a fixed point") {
        NiftiImage img({ 8, 8, 8 }, NIFTI_TYPE_FLOAT32);
        setIdentitySform(img);
        { auto p = img.data(); for (size_t i = 0; i < img.nVoxels(); ++i) p[i] = 3.f; }
        reg_tools_labelKernelConvolution(img, 1.f, 1.f, 1.f, nullptr, nullptr);
        RequireImageValues(img, std::vector<float>(img.nVoxels(), 3.f), "uniform labels");
    }
    SECTION("closure over the input label set") {
        NiftiImage img({ 8, 8, 8 }, NIFTI_TYPE_FLOAT32);
        setIdentitySform(img);
        const std::set<float> labels{ 0.f, 4.f, 7.f };
        {
            auto p = img.data();
            std::mt19937 gen(0);
            std::uniform_int_distribution<int> pick(0, 2);
            const float values[3] = { 0.f, 4.f, 7.f };
            for (size_t i = 0; i < img.nVoxels(); ++i)
                p[i] = values[pick(gen)];
        }
        reg_tools_labelKernelConvolution(img, 1.f, 1.f, 1.f, nullptr, nullptr);
        const auto p = img.data();
        for (size_t i = 0; i < img.nVoxels(); ++i) {
            INFO("voxel " << i << " = " << static_cast<float>(p[i]));
            REQUIRE(labels.count(static_cast<float>(p[i])) == 1);
        }
    }
}

TEST_CASE("Tools: invalid inputs are rejected, not computed", "[unit]") {
    // NR_FATAL_ERROR throws, so the rejection paths are testable: operands of different sizes or
    // types must raise rather than read out of bounds or reinterpret memory
    const NiftiImage a = MakeValueImage({ 2, 2 }, { 1.f, 2.f, 3.f, 4.f });
    NiftiImage out(a, NiftiImage::Copy::ImageInfoAndAllocData);

    NiftiImage differentSize = MakeValueImage({ 3 }, { 1.f, 2.f, 3.f });
    REQUIRE_THROWS_AS(reg_tools_addImageToImage(a, differentSize, out), std::runtime_error);

    NiftiImage differentType(a, NiftiImage::Copy::Image);
    reg_tools_changeDatatype<double>(differentType);
    REQUIRE_THROWS_AS(reg_tools_addImageToImage(a, differentType, out), std::runtime_error);

    // changeDatatype without an explicit code infers the type from sizeof and supports only
    // uchar/float/double: a 2-byte request without its code is rejected, not guessed
    NiftiImage forShort(a, NiftiImage::Copy::Image);
    REQUIRE_THROWS_AS(reg_tools_changeDatatype<short>(forShort), std::runtime_error);
}
