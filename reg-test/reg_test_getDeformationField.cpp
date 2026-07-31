// OpenCL is not supported for this test
#undef USE_OPENCL

#include "reg_test_common.h"

/*
    reg_spline_getDeformationField: the deformation field a cubic spline control point grid generates.

    The first case checks three concrete transformations - identity, translation, scaling - on every
    available platform. The rest generalise that to properties of the spline, on the CPU, so that a
    passing test says the result is right rather than merely unchanged. The properties, and what each
    one pins down:

      1. Affine reproduction. A uniform cubic B-spline reproduces affine functions exactly when its
         coefficients are the function's values at the nodes. A grid holding A(p_node) must therefore
         evaluate to A(p_voxel) at every voxel, which fixes the basis weights, the grid's origin shift
         and its spacing all at once.

      2. Reproduction across the grid boundary. Where a voxel's four-tap window extends past the last
         node, get_SlidedValues continues the lattice: c(-1,j,k) = c(0,j,k) - s_x, with s_x the first
         column of the grid's voxel-to-real matrix. For a grid holding A(p_node) that gives
         A(p_node) - s_x, while the true value is A(p_node) - A_lin*s_x. The two agree exactly when
         A_lin is the identity, so a pure translation is reproduced exactly even out there, and a
         general linear map is not. Asserting both halves pins the extrapolation: without the shift
         term the translation case fails, and without the deviation the comparison would be vacuous.

      3. Node interpolation. get_SplineBasisValues(0) is (0,1,0,0), so with bspline=false a voxel whose
         grid coordinate is an exact integer must return that control point's coefficient unchanged.
         This is the defining difference between the interpolating variant and the approximating one.

      4. Per-voxel independence. Composition maps each voxel through the same pure function, so
         permuting the input must permute the output identically. The evaluation carries state between
         voxels - a tap cache, and in 3D a refetch heuristic keyed on the sub-voxel offset - and this
         is what distinguishes a cache from a leak.

      5. Agreement between the two 3D evaluations. At a control point spacing of exactly five voxels
         the 3D code precomputes a 125x64 table of basis products instead of evaluating the basis per
         voxel. Since that is reg_f3d's default spacing, it is the path most runs take, and it is
         checked both against the per-voxel evaluation it replaces and against property 1 directly.

      6. The identity. An identity grid must give an identity field. It is an affine, so property 1
         covers it in principle, but it is worth its own cases: the residual decides whether a voxel at
         the edge of the field of view is interpolated or padded, and grid refinement has to preserve
         it across pyramid levels.

    Tolerances are float rounding on coordinates of the stated magnitude unless a case says otherwise.
    Cases that cannot assert an exact value report the quantity they measure instead of gating on it.
*/


class GetDeformationFieldTest {
protected:
    using TestData = std::tuple<std::string, NiftiImage, NiftiImage, NiftiImage>;
    using TestDataComp = std::tuple<std::string, NiftiImage, NiftiImage, NiftiImage, NiftiImage>;
    using TestCase = std::tuple<std::string, NiftiImage, NiftiImage>;

    inline static vector<TestCase> testCases;

public:
    GetDeformationFieldTest() {
        if (!testCases.empty())
            return;

        // Create reference images
        constexpr NiftiImage::dim_t size = 5;
        NiftiImage reference2d({ size, size }, NIFTI_TYPE_FLOAT32);
        NiftiImage reference3d({ size, size, size }, NIFTI_TYPE_FLOAT32);

        // Data container for the test data
        vector<TestData> testData;

        // Identity transformation tests
        // Create an affine transformation b-spline parametrisation
        NiftiImage controlPointGrid2d = CreateControlPointGrid(reference2d);
        NiftiImage controlPointGrid3d = CreateControlPointGrid(reference3d);
        // Create the expected deformation field result with an identity
        NiftiImage expDefField2d = CreateDeformationField(reference2d);
        NiftiImage expDefField3d = CreateDeformationField(reference3d);
        testData.emplace_back(TestData(
            "2D ID",
            reference2d,
            controlPointGrid2d,
            expDefField2d
        ));
        testData.emplace_back(TestData(
            "3D ID",
            reference3d,
            controlPointGrid3d,
            expDefField3d
        ));

        // Translation transformation tests - translation of 2 along each axis
        float *cpp2dPtr = static_cast<float*>(controlPointGrid2d->data);
        float *cpp3dPtr = static_cast<float*>(controlPointGrid3d->data);
        float *expDefField2dPtr = static_cast<float*>(expDefField2d->data);
        float *expDefField3dPtr = static_cast<float*>(expDefField3d->data);
        for (size_t i = 0; i < controlPointGrid2d.nVoxels(); i++)
            cpp2dPtr[i] += 2.f;
        for (size_t i = 0; i < controlPointGrid3d.nVoxels(); i++)
            cpp3dPtr[i] += 2.f;
        for (size_t i = 0; i < expDefField2d.nVoxels(); i++)
            expDefField2dPtr[i] += 2.f;
        for (size_t i = 0; i < expDefField3d.nVoxels(); i++)
            expDefField3dPtr[i] += 2.f;

        testData.emplace_back(TestData(
            "2D Trans",
            reference2d,
            controlPointGrid2d,
            expDefField2d
        ));
        testData.emplace_back(TestData(
            "3D Trans",
            reference3d,
            controlPointGrid3d,
            expDefField3d
        ));

        // Scaling transformation tests
        for (size_t i = 0; i < controlPointGrid2d.nVoxels(); i++)
            cpp2dPtr[i] = (cpp2dPtr[i] - 2.f) * 1.1f;
        for (size_t i = 0; i < controlPointGrid3d.nVoxels(); i++)
            cpp3dPtr[i] = (cpp3dPtr[i] - 2.f) * 1.1f;
        for (size_t i = 0; i < expDefField2d.nVoxels(); i++)
            expDefField2dPtr[i] = (expDefField2dPtr[i] - 2.f) * 1.1f;
        for (size_t i = 0; i < expDefField3d.nVoxels(); i++)
            expDefField3dPtr[i] = (expDefField3dPtr[i] - 2.f) * 1.1f;

        testData.emplace_back(TestData(
            "2D Scaling",
            reference2d,
            controlPointGrid2d,
            expDefField2d
        ));
        testData.emplace_back(TestData(
            "3D Scaling",
            reference3d,
            controlPointGrid3d,
            expDefField3d
        ));

        // Run the actual computation with the provided input data
        for (auto&& data : testData) {
            for (auto&& platformType : PlatformTypes) {
                unique_ptr<Platform> platform{ new Platform(platformType) };
                unique_ptr<F3dContentCreator> contentCreator{ dynamic_cast<F3dContentCreator*>(platform->CreateContentCreator(ContentType::F3d)) };
                // Make a copy of the test data
                auto [testName, reference, controlPointGrid, expDefField] = data;
                // Create the content and the compute
                unique_ptr<F3dContent> content{ contentCreator->Create(reference, reference, controlPointGrid) };
                unique_ptr<Compute> compute{ platform->CreateCompute(*content) };
                // Compute the deformation field
                compute->GetDeformationField(false, true); // no composition - use bspline
                // Save the results for testing
                testCases.push_back({ testName + " "s + platform->GetName(), std::move(content->GetDeformationField()), std::move(expDefField) });
            }
        }

        // Data container for the test data related to composition
        vector<TestDataComp> testDataComp;

        // Ensures composition of identity transformation yield identity
        NiftiImage defField2d = CreateDeformationField(reference2d);
        NiftiImage defField3d = CreateDeformationField(reference3d);
        reg_tools_multiplyValueToImage(expDefField2d, expDefField2d, 0.f);
        reg_tools_multiplyValueToImage(expDefField3d, expDefField3d, 0.f);
        reg_tools_multiplyValueToImage(controlPointGrid2d, controlPointGrid2d, 0.f);
        reg_tools_multiplyValueToImage(controlPointGrid3d, controlPointGrid3d, 0.f);
        reg_getDeformationFromDisplacement(expDefField2d);
        reg_getDeformationFromDisplacement(expDefField3d);
        reg_getDeformationFromDisplacement(controlPointGrid2d);
        reg_getDeformationFromDisplacement(controlPointGrid3d);
        testDataComp.emplace_back(TestDataComp(
            "2D Composition ID",
            reference2d,
            controlPointGrid2d,
            defField2d,
            expDefField2d
        ));
        testDataComp.emplace_back(TestDataComp(
            "3D Composition ID",
            reference3d,
            controlPointGrid3d,
            defField3d,
            expDefField3d
        ));

        // Ensures composition from zooming and and out goes back identity ID
        float *defField2dPtr = static_cast<float*>(defField2d->data);
        float *defField3dPtr = static_cast<float*>(defField3d->data);
        for (size_t i = 0; i < controlPointGrid2d.nVoxels(); i++)
            cpp2dPtr[i] *= 1.1f;
        for (size_t i = 0; i < controlPointGrid3d.nVoxels(); i++)
            cpp3dPtr[i] *= 1.1f;
        for (size_t i = 0; i < defField2d.nVoxels(); i++)
            defField2dPtr[i] /= 1.1f;
        for (size_t i = 0; i < defField3d.nVoxels(); i++)
            defField3dPtr[i] /= 1.1f;
        testDataComp.emplace_back(TestDataComp(
            "2D Composition Scaling",
            reference2d,
            controlPointGrid2d,
            defField2d,
            expDefField2d
        ));
        testDataComp.emplace_back(TestDataComp(
            "3D Composition Scaling",
            reference3d,
            controlPointGrid3d,
            defField3d,
            expDefField3d
        ));

        for (auto&& data : testDataComp) {
            for (auto&& platformType : PlatformTypes) {
                unique_ptr<Platform> platform{ new Platform(platformType) };
                unique_ptr<F3dContentCreator> contentCreator{ dynamic_cast<F3dContentCreator*>(platform->CreateContentCreator(ContentType::F3d)) };
                // Make a copy of the test data
                auto [testName, reference, controlPointGrid, defField, expDefField] = data;
                // Create the content and the compute
                unique_ptr<F3dContent> content{ contentCreator->Create(reference, reference, controlPointGrid) };
                unique_ptr<Compute> compute{ platform->CreateCompute(*content) };
                // Compute the deformation field
                content->SetDeformationField(std::move(defField));
                compute->GetDeformationField(true, true); // with composition - use bspline
                // Save the results for testing
                testCases.push_back({ testName + " "s + platform->GetName(), std::move(content->GetDeformationField()), std::move(expDefField) });
            }
        }
    }
};

TEST_CASE_METHOD(GetDeformationFieldTest, "Deformation Field from B-spline Grid", "[unit]") {
    // Loop over all generated test cases
    for (auto&& testCase : testCases) {
        // Retrieve test information
        auto&& [testName, result, expected] = testCase;

        SECTION(testName) {
            NR_COUT << "\n**************** Section " << testName << " ****************" << std::endl;

            // Increase the precision for the output
            NR_COUT << std::fixed << std::setprecision(10);

            const auto resPtr = result.data();
            const auto expPtr = expected.data();
            for (auto i = 0; i < expected.nVoxels(); i++) {
                const float resVal = resPtr[i];
                const float expVal = expPtr[i];
                const float diff = abs(resVal - expVal);
                // The deformation is a float cubic B-spline weighted sum, so it reproduces the
                // affine (grid) transformation only to float precision, not bit-exactly. The
                // rounding is relative to the coordinate magnitude (~1 ULP), so the tolerance
                // scales with the expected value.
                const float tol = EPS * (std::abs(expVal) > 1.f ? std::abs(expVal) : 1.f);
                if (diff > tol) {
                    NR_COUT << "[i]=" << i;
                    NR_COUT << " | diff=" << diff;
                    NR_COUT << " | Result=" << resVal;
                    NR_COUT << " | Expected=" << expVal << std::endl;
                }
                REQUIRE(diff <= tol);
            }
        }
    }
}

namespace {

constexpr float kTranslationTolerance = 1e-4f;   // float rounding on coordinates of order 100 mm

NiftiImage MakeReference(bool is3D, NiftiImage::dim_t size, bool anisotropic) {
    std::vector<NiftiImage::dim_t> dims(is3D ? 3 : 2, size);
    NiftiImage img(dims, NIFTI_TYPE_FLOAT32);
    if (anisotropic) {
        setAnisotropicSform(img);
    } else {
        setIdentitySform(img);
        img->dx = img->pixdim[1] = 1.f;
        img->dy = img->pixdim[2] = 1.f;
        img->dz = img->pixdim[3] = 1.f;
    }
    return img;
}

NiftiImage MakeGrid(const NiftiImage& reference, float spacingInVoxels) {
    NiftiImage grid;
    const float spacing[3]{ reference->dx * spacingInVoxels, reference->dy * spacingInVoxels,
                            reference->dz * spacingInVoxels };
    reg_createControlPointGrid<float>(grid, reference, spacing);
    return grid;
}

mat44 Translation(float x, float y, float z) {
    mat44 m;
    Mat44Eye(&m);
    m.m[0][3] = x; m.m[1][3] = y; m.m[2][3] = z;
    return m;
}

// A general linear map: the reproduction property still holds inside the grid, but its linear part is
// not the identity, so it cannot survive the extrapolation past the grid edge
mat44 LinearMap() {
    mat44 m;
    Mat44Eye(&m);
    m.m[0][0] = 1.25f; m.m[0][1] = 0.125f;  m.m[0][3] = -1.5f;
    m.m[1][1] = 0.75f; m.m[1][2] = 0.0625f; m.m[1][3] = 2.5f;
    m.m[2][0] = 0.25f; m.m[2][2] = 1.5f;    m.m[2][3] = -0.75f;
    return m;
}

NiftiImage Evaluate(const NiftiImage& reference, const NiftiImage& grid, bool bspline,
                    int *mask = nullptr, bool forceNoLut = false) {
    NiftiImage field = CreateDeformationField(reference);
    NiftiImage gridCopy(grid);
    reg_spline_getDeformationField(gridCopy, field, mask, false, bspline, forceNoLut);
    return field;
}

// Whether this geometry takes the 3D look-up-table branch of the cubic spline evaluation, which behaves
// differently enough from the general path that several cases below have to distinguish them
bool TakesLutPath(const NiftiImage& reference, const NiftiImage& grid, bool forceNoLut) {
    return !forceNoLut && reference->nz > 1 &&
        grid->dx / reference->dx == 5.f && grid->dy / reference->dy == 5.f &&
        grid->dz / reference->dz == 5.f;
}

// Whether a voxel's four-tap window lies wholly inside the grid along every axis. The window starts at
// int(voxel / gridVoxelSpacing) and spans four nodes, so it fits while that index is at most nx-4.
bool WindowInsideGrid(const NiftiImage& field, const NiftiImage& grid, int x, int y, int z) {
    const auto fits = [](int voxel, float gridSpacing, float fieldSpacing, int nodes) {
        const int pre = static_cast<int>(static_cast<float>(voxel) / (gridSpacing / fieldSpacing));
        return pre >= 0 && pre + 3 <= nodes - 1;
    };
    return fits(x, grid->dx, field->dx, grid->nx) &&
           fits(y, grid->dy, field->dy, grid->ny) &&
           (field->nz == 1 || fits(z, grid->dz, field->dz, grid->nz));
}

// Split a comparison into the voxels the grid covers and those whose window extrapolates. The counts
// are kept so a case can assert that the extrapolated band exists before drawing conclusions from it.
struct SplitDeviation {
    Deviation interior, border;
    size_t interiorVoxels = 0, borderVoxels = 0;
};

SplitDeviation CompareBySupport(const NiftiImage& actual, const NiftiImage& expected,
                                const NiftiImage& grid) {
    SplitDeviation split;
    const size_t volume = actual.nVoxelsPerVolume();
    const int nx = actual->nx, ny = actual->ny, nz = actual->nz;
    const int components = nz > 1 ? 3 : 2;
    const auto actualPtr = actual.data();
    const auto expectedPtr = expected.data();
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const bool inside = WindowInsideGrid(actual, grid, i, j, k);
                Deviation& target = inside ? split.interior : split.border;
                ++(inside ? split.interiorVoxels : split.borderVoxels);
                const size_t index = (static_cast<size_t>(k) * ny + j) * nx + i;
                for (int c = 0; c < components; ++c) {
                    const size_t offset = c * volume + index;
                    const double difference = std::abs(static_cast<double>(actualPtr[offset]) -
                                                       static_cast<double>(expectedPtr[offset]));
                    if (difference > 0) ++target.differing;
                    target.max = std::max(target.max, difference);
                }
            }
    return split;
}

// Fill a deformation field with widely scattered positions, most of them outside the grid's support,
// so every tap extrapolates
void ScatterPositions(NiftiImage& field, float spread) {
    const size_t volume = field.nVoxelsPerVolume();
    const int components = field->nz > 1 ? 3 : 2;
    auto ptr = field.data();
    for (size_t i = 0; i < volume; ++i)
        for (int c = 0; c < components; ++c)
            // Deterministic, irrational-ish strides so no two voxels coincide and the positions do not
            // line up with the grid
            ptr[c * volume + i] = static_cast<float>(ptr[c * volume + i]) +
                spread * static_cast<float>(std::sin(0.7139 * double(i) + 2.113 * c));
}

void ReverseVoxelOrder(NiftiImage& field) {
    const size_t volume = field.nVoxelsPerVolume();
    const int components = field->nz > 1 ? 3 : 2;
    auto ptr = field.data();
    for (int c = 0; c < components; ++c)
        for (size_t i = 0; i < volume / 2; ++i) {
            const size_t a = c * volume + i, b = c * volume + (volume - 1 - i);
            const float temp = static_cast<float>(ptr[a]);
            ptr[a] = static_cast<float>(ptr[b]);
            ptr[b] = temp;
        }
}


NiftiImage EvaluateIdentityGrid(const NiftiImage& reference, float spacingInVoxels, bool forceNoLut) {
    NiftiImage grid = MakeGrid(reference, spacingInVoxels);
    NiftiImage field = CreateDeformationField(reference);
    reg_spline_getDeformationField(grid, field, nullptr, false, true, forceNoLut);
    return field;
}

// The largest coordinate the field holds, so a residual can be read as a relative quantity
double CoordinateMagnitude(const NiftiImage& field) {
    double magnitude = 0;
    const auto ptr = field.data();
    for (size_t i = 0; i < field.nVoxels(); ++i)
        magnitude = std::max(magnitude, std::abs(static_cast<double>(ptr[i])));
    return magnitude;
}

// How many voxels the two fields classify differently as inside the reference's field of view, split
// by how close to the boundary they sit. The classification is what actually reaches the similarity
// measure: a voxel outside is padded with NaN and dropped from the joint histogram, one inside
// contributes to it.
//
// The split is the assertable part. A residual of ~1e-5 voxels can only change the verdict for a
// position already within 1e-5 of a face, so every disagreement must lie in the boundary shell. One
// appearing further in would mean the residual had stopped being rounding.
struct FovFlips {
    size_t total = 0;
    size_t awayFromBoundary = 0;   // must be zero
};

FovFlips FovDisagreements(const NiftiImage& a, const NiftiImage& b, const NiftiImage& reference,
                          float margin = 0.5f) {
    const mat44 realToVoxel = reference->sform_code > 0 ? reference->sto_ijk : reference->qto_ijk;
    const size_t volume = a.nVoxelsPerVolume();
    const int components = reference->nz > 1 ? 3 : 2;
    const int bounds[3]{ reference->nx - 1, reference->ny - 1, reference->nz - 1 };
    const auto aPtr = a.data();
    const auto bPtr = b.data();

    const auto voxelCoords = [&](const NiftiImageData& data, size_t index, float (&voxel)[3]) {
        float world[3]{ 0, 0, 0 };
        for (int c = 0; c < components; ++c)
            world[c] = static_cast<float>(data[c * volume + index]);
        Mat44Mul(realToVoxel, world, voxel);
    };
    const auto inside = [&](const float (&voxel)[3]) {
        for (int c = 0; c < components; ++c)
            if (voxel[c] < 0.f || voxel[c] > static_cast<float>(bounds[c])) return false;
        return true;
    };
    // How far this position sits from the nearest face, in voxels
    const auto distanceToBoundary = [&](const float (&voxel)[3]) {
        float distance = std::numeric_limits<float>::max();
        for (int c = 0; c < components; ++c)
            distance = std::min({ distance, std::abs(voxel[c]), std::abs(static_cast<float>(bounds[c]) - voxel[c]) });
        return distance;
    };

    FovFlips flips;
    for (size_t i = 0; i < volume; ++i) {
        float aVoxel[3]{}, bVoxel[3]{};
        voxelCoords(aPtr, i, aVoxel);
        voxelCoords(bPtr, i, bVoxel);
        if (inside(aVoxel) == inside(bVoxel)) continue;
        ++flips.total;
        if (distanceToBoundary(aVoxel) > margin && distanceToBoundary(bVoxel) > margin)
            ++flips.awayFromBoundary;
    }
    return flips;
}

} // namespace

TEST_CASE("Spline deformation field reproduces an affine transformation", "[unit]") {
    // The baseline reproduction property, on a grid that covers the whole field. Both transformations
    // must be reproduced here; it is only at the boundary, in the next case, that they part company.
    for (const bool is3D : { false, true })
        for (const bool anisotropic : { false, true })
            for (const auto& [label, affine] : { std::pair{ "translation", Translation(3.5f, -2.25f, 1.75f) },
                                                 std::pair{ "general linear map", LinearMap() } }) {
                const std::string name = std::string(is3D ? "3D" : "2D") +
                    (anisotropic ? ", anisotropic" : ", identity sform") + ", " + label;
                SECTION(name) {
                    const NiftiImage reference = MakeReference(is3D, 24, anisotropic);
                    NiftiImage grid = MakeGrid(reference, kProductionGridSpacing);
                    ApplyAffineToGrid(grid, affine);

                    const Deviation deviation = CompareImages(Evaluate(reference, grid, true),
                                                              ExpectedAffineField(reference, affine));
                    ReportDeviation(name, deviation);
                    INFO(name << ": max deviation " << deviation.max);
                    REQUIRE(deviation.max < kTranslationTolerance);
                }
            }
}

TEST_CASE("Spline deformation field extrapolates past the grid by continuing it affinely", "[unit]") {
    /*
        The boundary the old reference test transcribed but could never reach. The grid is built for a
        small reference and the field evaluated over a much larger one, so the outer voxels' four-tap
        windows leave the grid on every side and get_SlidedValues has to supply them.

        A translation must still be reproduced exactly out there, because the sliding extrapolation
        continues the grid by exactly one voxel-spacing vector per node of overhang - which is what a
        translated node lattice does. A general linear map must not be, because the extrapolation
        applies the spacing vector rather than the map's linear part to it.

        forceNoLut is set throughout: in 3D at this spacing the table-driven evaluation would run
        instead, and it never extrapolates, so the boundary would go unexercised.
    */
    for (const bool is3D : { false, true })
        for (const bool anisotropic : { false, true }) {
            const std::string suffix = std::string(is3D ? "3D" : "2D") +
                (anisotropic ? ", anisotropic" : ", identity sform");

            SECTION("translation, " + suffix) {
                const NiftiImage small = MakeReference(is3D, 10, anisotropic);
                const NiftiImage large = MakeReference(is3D, 30, anisotropic);
                const mat44 affine = Translation(3.5f, -2.25f, 1.75f);
                NiftiImage grid = MakeGrid(small, kProductionGridSpacing);
                ApplyAffineToGrid(grid, affine);

                const NiftiImage actual = Evaluate(large, grid, true, nullptr, true);
                const SplitDeviation split = CompareBySupport(actual, ExpectedAffineField(large, affine), grid);
                ReportDeviation("translation, " + suffix + ", interior", split.interior);
                ReportDeviation("translation, " + suffix + ", extrapolated", split.border);

                // The extrapolated band must exist, or this case asserts nothing about the boundary
                INFO(split.borderVoxels << " extrapolated voxels, " << split.interiorVoxels << " interior");
                REQUIRE(split.borderVoxels > 0);
                REQUIRE(split.interiorVoxels > 0);

                // Exact on both sides of the grid edge: this is what pins the shift term in
                // get_SlidedValues. Dropping it leaves the border voxels clamped to the edge node.
                INFO("interior max " << split.interior.max << ", extrapolated max " << split.border.max);
                REQUIRE(split.interior.max < kTranslationTolerance);
                REQUIRE(split.border.max < kTranslationTolerance);
            }

            SECTION("general linear map, " + suffix) {
                const NiftiImage small = MakeReference(is3D, 10, anisotropic);
                const NiftiImage large = MakeReference(is3D, 30, anisotropic);
                const mat44 affine = LinearMap();
                NiftiImage grid = MakeGrid(small, kProductionGridSpacing);
                ApplyAffineToGrid(grid, affine);

                const NiftiImage actual = Evaluate(large, grid, true, nullptr, true);
                const SplitDeviation split = CompareBySupport(actual, ExpectedAffineField(large, affine), grid);
                ReportDeviation("linear map, " + suffix + ", interior", split.interior);
                ReportDeviation("linear map, " + suffix + ", extrapolated", split.border);

                INFO(split.borderVoxels << " extrapolated voxels, " << split.interiorVoxels << " interior");
                REQUIRE(split.borderVoxels > 0);
                REQUIRE(split.interiorVoxels > 0);

                // Reproduction still holds wherever the window is inside the grid
                INFO("interior max " << split.interior.max);
                REQUIRE(split.interior.max < kTranslationTolerance);
                // And must fail outside it, by far more than rounding. If this ever passes, the
                // extrapolation has silently changed and the translation case above became vacuous.
                INFO("extrapolated max " << split.border.max);
                REQUIRE(split.border.max > 100 * kTranslationTolerance);
                REQUIRE(std::isfinite(split.border.max));
            }
        }
}

TEST_CASE("The look-up-table path leaves voxels the grid does not span untouched", "[unit]") {
    /*
        The table-driven evaluation does not iterate over voxels - it
        iterates over grid cells, xPre running to splineControlPoint->nx - 4, and fills the
        5x5x5 voxel block each cell covers. Two consequences follow, and neither holds for the
        per-voxel evaluation:

          - the four-tap window is inside the grid by construction, so nothing is ever extrapolated;
          - a voxel outside every cell's block is never written, keeping whatever the field held.

        reg_createControlPointGrid sizes a grid as ceil(n*d/spacing + 3), so (nx-3)*5 >= n and a grid
        built that way always spans its reference. The behaviour is therefore only reachable by a
        caller that supplies its own grid, and it is pinned here so that such a caller finds it
        documented rather than silent.
    */
    const NiftiImage small = MakeReference(true, 10, false);
    const NiftiImage large = MakeReference(true, 30, false);
    const mat44 affine = Translation(3.5f, -2.25f, 1.75f);
    NiftiImage grid = MakeGrid(small, kProductionGridSpacing);
    ApplyAffineToGrid(grid, affine);
    REQUIRE(TakesLutPath(large, grid, false));

    const NiftiImage untouched = CreateDeformationField(large);
    const NiftiImage viaLut = Evaluate(large, grid, true, nullptr, false);
    const NiftiImage viaGeneral = Evaluate(large, grid, true, nullptr, true);
    const NiftiImage expected = ExpectedAffineField(large, affine);

    const SplitDeviation lut = CompareBySupport(viaLut, expected, grid);
    const SplitDeviation general = CompareBySupport(viaGeneral, expected, grid);
    ReportDeviation("look-up table, interior", lut.interior);
    ReportDeviation("look-up table, beyond the grid", lut.border);
    ReportDeviation("general path, beyond the grid", general.border);

    // Both agree wherever the grid spans the field
    REQUIRE(lut.interior.max < kTranslationTolerance);
    REQUIRE(general.interior.max < kTranslationTolerance);
    // The general path extrapolates and reproduces the translation there; the table does not
    REQUIRE(general.border.max < kTranslationTolerance);
    REQUIRE(lut.border.max > 100 * kTranslationTolerance);

    // And what the table leaves behind is precisely the field's prior contents, not a partial result
    const size_t volume = viaLut.nVoxelsPerVolume();
    const int nx = viaLut->nx, ny = viaLut->ny, nz = viaLut->nz;
    const auto lutPtr = viaLut.data();
    const auto untouchedPtr = untouched.data();
    size_t unwritten = 0;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                // The blocks the table fills reach (nodes - 3) * 5 voxels along each axis
                if (i < (grid->nx - 3) * 5 && j < (grid->ny - 3) * 5 && k < (grid->nz - 3) * 5) continue;
                const size_t index = (static_cast<size_t>(k) * ny + j) * nx + i;
                for (int c = 0; c < 3; ++c)
                    REQUIRE(static_cast<float>(lutPtr[c * volume + index]) ==
                            static_cast<float>(untouchedPtr[c * volume + index]));
                ++unwritten;
            }
    NR_COUT << "  look-up table left " << unwritten << " of " << volume << " voxels unwritten" << std::endl;
    REQUIRE(unwritten > 0);
}

TEST_CASE("Spline composition reproduces a translation from any starting position", "[unit]") {
    /*
        Composition reads a position out of the deformation field, locates it in the grid and evaluates
        there - so scattering the input positions well outside the grid drives every tap through the
        extrapolation, on the composition code path rather than the plain one.

        With the grid holding a translation, reproduction holds at every real position, inside the grid
        or not, so the answer is the input plus that translation - an exact expectation for arbitrary
        input positions, which is what makes the scattering safe to do.
    */
    for (const bool is3D : { false, true }) {
        SECTION(is3D ? "3D" : "2D") {
            const NiftiImage reference = MakeReference(is3D, 16, false);
            const mat44 affine = Translation(3.5f, -2.25f, is3D ? 1.75f : 0.f);
            NiftiImage grid = MakeGrid(reference, kProductionGridSpacing);
            ApplyAffineToGrid(grid, affine);

            NiftiImage field = CreateDeformationField(reference);
            ScatterPositions(field, 60.f);   // several times the grid extent, so taps land far outside
            const NiftiImage input(field, NiftiImage::Copy::Image);

            NiftiImage gridCopy(grid);
            reg_spline_getDeformationField(gridCopy, field, nullptr, true, true);

            const NiftiImage expected = [&] {
                NiftiImage shifted(input, NiftiImage::Copy::Image);
                const size_t volume = shifted.nVoxelsPerVolume();
                const int components = is3D ? 3 : 2;
                auto ptr = shifted.data();
                for (size_t i = 0; i < volume; ++i) {
                    float position[3]{ 0, 0, 0 }, transformed[3];
                    for (int c = 0; c < components; ++c)
                        position[c] = static_cast<float>(ptr[c * volume + i]);
                    Mat44Mul(affine, position, transformed);
                    for (int c = 0; c < components; ++c)
                        ptr[c * volume + i] = transformed[c];
                }
                return shifted;
            }();

            RequireChanged(input, field, "the composed deformation field");
            const Deviation deviation = CompareImages(field, expected);
            ReportDeviation(std::string(is3D ? "3D" : "2D") + " composition, translated grid", deviation);
            INFO("max deviation " << deviation.max);
            // Positions reach ~100 mm here, so the bound is scaled accordingly
            REQUIRE(deviation.max < 1e-3);
        }
    }
}

TEST_CASE("Spline composition treats every voxel independently", "[unit]") {
    /*
        Composition is a pure per-voxel map, so permuting the input must permute the output the same
        way. Nothing about the spline is assumed here - only that voxels do not influence each other.

        The tap cache (oldXpre/oldYpre) and the 3D refetch heuristic (`basis <= oldBasis || x == 0`)
        are the reason this is worth asserting: both carry state across voxels, and both are only
        sound because the state is a cache of a pure function of the tap indices. Reversing the input
        is what distinguishes a cache from a leak.

        Note the scope of the property. It detects a voxel picking up a neighbour's result, but not a
        voxel being skipped, since the field is written in place and a skipped voxel simply keeps the
        value it came in with. Skipping is what the translation case above rules out.
    */
    for (const bool is3D : { false, true })
        for (const bool scattered : { false, true }) {
            const std::string name = std::string(is3D ? "3D" : "2D") +
                (scattered ? ", positions outside the grid" : ", identity positions");
            SECTION(name) {
                const NiftiImage reference = MakeReference(is3D, 16, false);
                NiftiImage grid = MakeGrid(reference, kProductionGridSpacing);
                ApplyAffineToGrid(grid, LinearMap());

                NiftiImage forward = CreateDeformationField(reference);
                if (scattered) ScatterPositions(forward, 60.f);
                NiftiImage reversed(forward, NiftiImage::Copy::Image);
                ReverseVoxelOrder(reversed);

                NiftiImage gridA(grid), gridB(grid);
                reg_spline_getDeformationField(gridA, forward, nullptr, true, true);
                reg_spline_getDeformationField(gridB, reversed, nullptr, true, true);
                ReverseVoxelOrder(reversed);

                // A pure per-voxel map gives bit-identical results under any ordering
                const Deviation deviation = CompareImages(forward, reversed);
                ReportDeviation(name, deviation);
                INFO(name << ": " << deviation.differing << " voxels depend on iteration order, max "
                     << deviation.max);
                REQUIRE(deviation.differing == 0);
            }
        }
}

TEST_CASE("Interpolating spline returns the control point value at the nodes", "[unit]") {
    /*
        With bspline=false the basis is the interpolating (Catmull-Rom) variant, whose weights at an
        exact knot are (0,1,0,0). A two-voxel grid spacing puts a node on every even voxel, so the
        field there must equal that node's coefficient exactly - which is the defining difference
        between this path and the approximating B-spline, and the only exact check available for it.

        reg_createControlPointGrid shifts the grid origin back by one node, so the node sitting on
        voxel v is at grid index v/2 + 1.
    */
    for (const bool is3D : { false, true }) {
        SECTION(is3D ? "3D" : "2D") {
            constexpr float spacingInVoxels = 2.f;
            const NiftiImage reference = MakeReference(is3D, 16, false);
            NiftiImage grid = MakeGrid(reference, spacingInVoxels);

            // Perturb the grid so the coefficients are distinguishable from the identity
            const size_t gridVolume = grid.nVoxelsPerVolume();
            auto gridPtr = grid.data();
            for (size_t i = 0; i < grid.nVoxels(); ++i)
                gridPtr[i] = static_cast<float>(gridPtr[i]) +
                    1.7f * static_cast<float>(std::sin(0.53 * double(i)));

            const NiftiImage field = Evaluate(reference, grid, false);
            const size_t volume = field.nVoxelsPerVolume();
            const int nx = field->nx, ny = field->ny, nz = field->nz;
            const int components = is3D ? 3 : 2;
            const auto fieldPtr = field.data();
            const auto coefficients = grid.data();

            size_t checked = 0;
            double maxDeviation = 0;
            for (int k = 0; k < nz; k += is3D ? 2 : 1)
                for (int j = 0; j < ny; j += 2)
                    for (int i = 0; i < nx; i += 2) {
                        // The node coinciding with this voxel, allowing for the one-node origin shift
                        const int nodeX = i / 2 + 1, nodeY = j / 2 + 1, nodeZ = is3D ? k / 2 + 1 : 0;
                        if (nodeX >= grid->nx || nodeY >= grid->ny || (is3D && nodeZ >= grid->nz))
                            continue;
                        const size_t voxel = (static_cast<size_t>(k) * ny + j) * nx + i;
                        const size_t node = (static_cast<size_t>(nodeZ) * grid->ny + nodeY) * grid->nx + nodeX;
                        for (int c = 0; c < components; ++c) {
                            const double actual = static_cast<double>(fieldPtr[c * volume + voxel]);
                            const double expected = static_cast<double>(coefficients[c * gridVolume + node]);
                            maxDeviation = std::max(maxDeviation, std::abs(actual - expected));
                        }
                        ++checked;
                    }

            NR_COUT << "  " << (is3D ? "3D" : "2D") << " node interpolation: " << checked
                    << " nodes checked, max deviation " << std::scientific << maxDeviation << std::endl;
            INFO("checked " << checked << " nodes, max deviation " << maxDeviation);
            REQUIRE(checked > 0);
            REQUIRE(maxDeviation < 1e-5);
        }
    }
}

TEST_CASE("Masked voxels follow the convention each path defines", "[unit]") {
    /*
        The two paths disagree, and the disagreement is worth pinning rather than discovering again.

        Without composition the accumulator is zeroed, gated on the mask, then written
        unconditionally, so a masked-out voxel is set to zero - a position at the world origin, not
        an identity. With composition the whole body sits inside the mask test, so a masked-out
        voxel keeps whatever it held.

        And the 3D table-driven evaluation is a third convention again: its write is guarded by
        `x < nx && mask[index] > -1`, so it skips the voxel as the composition path does rather
        than zeroing it as the per-voxel path does. Which of the two a masked-out voxel gets therefore
        depends on whether the control point spacing is exactly five voxels, which is reg_f3d's
        default. Masked voxels are excluded from every similarity measure, so this does not move a
        registration; it is pinned so that the choice is visible.
    */
    for (const bool is3D : { false, true }) {
        SECTION(is3D ? "3D" : "2D") {
            const NiftiImage reference = MakeReference(is3D, 12, false);
            NiftiImage grid = MakeGrid(reference, kProductionGridSpacing);
            ApplyAffineToGrid(grid, LinearMap());

            const size_t volume = CreateDeformationField(reference).nVoxelsPerVolume();
            const int components = is3D ? 3 : 2;
            // Mask out the second half of the voxels
            std::vector<int> mask(volume);
            for (size_t i = 0; i < volume; ++i) mask[i] = i < volume / 2 ? 0 : -1;

            SECTION("no composition, general path, writes zero") {
                NiftiImage field = CreateDeformationField(reference);
                NiftiImage gridCopy(grid);
                reg_spline_getDeformationField(gridCopy, field, mask.data(), false, true, true);
                const auto ptr = field.data();
                for (size_t i = volume / 2; i < volume; ++i)
                    for (int c = 0; c < components; ++c)
                        REQUIRE(static_cast<float>(ptr[c * volume + i]) == 0.f);
            }

            SECTION("no composition, look-up-table path, skips the voxel") {
                NiftiImage field = CreateDeformationField(reference);
                const NiftiImage input(field, NiftiImage::Copy::Image);
                NiftiImage gridCopy(grid);
                reg_spline_getDeformationField(gridCopy, field, mask.data(), false, true, false);
                const auto ptr = field.data();
                const auto inputPtr = input.data();
                const bool lut = TakesLutPath(reference, grid, false);
                for (size_t i = volume / 2; i < volume; ++i)
                    for (int c = 0; c < components; ++c) {
                        const float actual = static_cast<float>(ptr[c * volume + i]);
                        // 2D has no table, so it falls back to the zeroing convention above
                        REQUIRE(actual == (lut ? static_cast<float>(inputPtr[c * volume + i]) : 0.f));
                    }
            }

            SECTION("composition leaves the voxel untouched") {
                NiftiImage field = CreateDeformationField(reference);
                ScatterPositions(field, 5.f);
                const NiftiImage input(field, NiftiImage::Copy::Image);
                NiftiImage gridCopy(grid);
                reg_spline_getDeformationField(gridCopy, field, mask.data(), true, true);
                const auto ptr = field.data();
                const auto inputPtr = input.data();
                for (size_t i = volume / 2; i < volume; ++i)
                    for (int c = 0; c < components; ++c)
                        REQUIRE(static_cast<float>(ptr[c * volume + i]) ==
                                static_cast<float>(inputPtr[c * volume + i]));
                // And the unmasked half did change, so the call was not simply a no-op
                bool changed = false;
                for (size_t i = 0; i < volume / 2 && !changed; ++i)
                    for (int c = 0; c < components && !changed; ++c)
                        changed = static_cast<float>(ptr[c * volume + i]) !=
                                  static_cast<float>(inputPtr[c * volume + i]);
                REQUIRE(changed);
            }
        }
    }
}

TEST_CASE("Deformation field look-up table is in use", "[unit]") {
    // Guards the cases that follow: if the table were not taken, the two calls would return
    // bit-identical fields and those comparisons would be testing one code path against itself.
    const NiftiImage reference = MakeReference(true, 28, false);
    const NiftiImage grid = MakeGrid(reference, kProductionGridSpacing);
    REQUIRE(grid->dx / reference->dx == kProductionGridSpacing);
    REQUIRE(grid->dy / reference->dy == kProductionGridSpacing);
    REQUIRE(grid->dz / reference->dz == kProductionGridSpacing);

    const Deviation d = CompareImages(Evaluate(reference, grid, true, nullptr, false), Evaluate(reference, grid, true, nullptr, true));
    NR_COUT << "  table vs per-voxel: " << d.differing << " values differ, max "
            << std::scientific << d.max << std::endl;
    REQUIRE(d.differing > 0);
}

TEST_CASE("Deformation field look-up table agrees with the per-voxel evaluation", "[unit]") {
    for (const bool anisotropic : { false, true }) {
        SECTION(anisotropic ? "anisotropic" : "isotropic") {
            const NiftiImage reference = MakeReference(true, 28, anisotropic);
            NiftiImage grid = MakeGrid(reference, kProductionGridSpacing);
            // A smooth, bounded perturbation so the field is not just the identity
            const size_t volume = grid.nVoxelsPerVolume();
            auto ptr = grid.data();
            for (size_t i = 0; i < grid.nVoxels(); ++i)
                ptr[i] = static_cast<float>(ptr[i]) +
                    2.f * static_cast<float>(std::sin(0.37 * static_cast<double>(i % volume)));

            const Deviation d = CompareImages(Evaluate(reference, grid, true, nullptr, false), Evaluate(reference, grid, true, nullptr, true));
            NR_COUT << "  " << (anisotropic ? "anisotropic" : "isotropic  ")
                    << ": max deviation from the per-voxel evaluation = " << std::scientific << d.max << std::endl;
            INFO("max deviation " << d.max);
            // Rounding only: the table holds the same basis products, accumulated in a different
            // order. Observed here is 1.3e-5 isotropic and 2.7e-5 anisotropic on coordinates of order
            // 50 mm, i.e. float epsilon. The bound keeps a margin for the SSE variant of these
            // functions, which is a third arrangement of the same sum.
            REQUIRE(d.max < 1e-4);
        }
    }
}

TEST_CASE("Deformation field look-up table reproduces affine transformations", "[unit]") {
    // Property 1 applied to the table directly, so that it is pinned to the correct values and not
    // merely to whatever the per-voxel evaluation produces.
    mat44 translation;
    Mat44Eye(&translation);
    translation.m[0][3] = 3.5f; translation.m[1][3] = -2.25f; translation.m[2][3] = 1.75f;

    mat44 linear;
    Mat44Eye(&linear);
    linear.m[0][0] = 1.25f; linear.m[0][1] = 0.125f; linear.m[0][3] = -1.5f;
    linear.m[1][1] = 0.75f; linear.m[1][2] = 0.0625f; linear.m[1][3] = 2.5f;
    linear.m[2][0] = 0.25f; linear.m[2][2] = 1.5f;    linear.m[2][3] = -0.75f;

    for (const auto& [label, affine] : { std::pair{ "translation", translation },
                                         std::pair{ "general linear map", linear } }) {
        for (const bool forceNoLut : { false, true }) {
            SECTION(std::string(label) + (forceNoLut ? ", per-voxel" : ", table")) {
                const NiftiImage reference = MakeReference(true, 28, false);
                NiftiImage grid = MakeGrid(reference, kProductionGridSpacing);
                ApplyAffineToGrid(grid, affine);

                const NiftiImage actual = Evaluate(reference, grid, true, nullptr, forceNoLut);
                const Deviation d = CompareImages(actual, ExpectedAffineField(reference, affine));
                NR_COUT << "  " << std::setw(20) << std::left << label
                        << (forceNoLut ? " per-voxel" : " table    ")
                        << ": max deviation from the analytic field = " << std::scientific << d.max << std::endl;
                INFO(label << (forceNoLut ? " per-voxel" : " table") << ", max deviation " << d.max);
                // Observed 1.1e-5 for the translation and 1.5e-5 for the linear map, against 1.1e-5
                // and 1.9e-5 for the per-voxel evaluation: the table is no less accurate than the
                // code it replaces, which is the point of this case.
                REQUIRE(d.max < 1e-4);
            }
        }
    }
}

TEST_CASE("Deformation field look-up table is not used at other spacings", "[unit]") {
    // Only an exactly-5-voxel spacing takes the table, so forceNoLut must make no difference at all
    // elsewhere, and the 2D evaluation has no table at any spacing.
    for (const auto& [label, is3D, spacing] : { std::tuple{ "3D, 2 voxels", true, 2.f },
                                                std::tuple{ "3D, 4 voxels", true, 4.f },
                                                std::tuple{ "2D, 5 voxels", false, 5.f } }) {
        SECTION(label) {
            const NiftiImage reference = MakeReference(is3D, 28, false);
            const NiftiImage grid = MakeGrid(reference, spacing);
            const Deviation d = CompareImages(Evaluate(reference, grid, true, nullptr, false), Evaluate(reference, grid, true, nullptr, true));
            INFO(label << ": " << d.differing << " values differ, max " << d.max);
            REQUIRE(d.differing == 0);
        }
    }
}

TEST_CASE("An identity control point grid evaluates to the identity field", "[unit]") {
    // The identity is an affine, so reproduction makes this exact but for rounding. Anything larger
    // than rounding would mean the grid's origin shift or its spacing had gone wrong.
    for (const bool is3D : { false, true })
        for (const bool anisotropic : { false, true })
            for (const float spacing : { 2.f, 4.f, kProductionGridSpacing }) {
                const std::string name = std::string(is3D ? "3D" : "2D") +
                    (anisotropic ? ", anisotropic" : ", identity sform") +
                    ", spacing " + std::to_string(int(spacing));
                SECTION(name) {
                    const NiftiImage reference = MakeReference(is3D, 24, anisotropic);
                    const NiftiImage expected = CreateDeformationField(reference);   // already the identity
                    const NiftiImage actual = EvaluateIdentityGrid(reference, spacing, false);

                    const Deviation deviation = CompareImages(actual, expected);
                    const double magnitude = CoordinateMagnitude(expected);
                    NR_COUT << "  " << std::setw(44) << std::left << name
                            << " residual = " << std::scientific << std::setprecision(3) << deviation.max
                            << " over coordinates up to " << magnitude
                            << " (" << deviation.differing << " voxels non-exact)" << std::endl;

                    INFO(name << ": residual " << deviation.max << " on coordinates up to " << magnitude);
                    // A few ulp of the largest coordinate. Deliberately loose - the point is the order
                    // of magnitude, not the digits, which move with any reordering of the accumulation.
                    REQUIRE(deviation.max < 1e-4 * std::max(1.0, magnitude));
                }
            }
}

TEST_CASE("The identity residual is what flips voxels across the field-of-view boundary", "[unit]") {
    /*
        The residual above is harmless in itself. What makes it matter is that the deformation field's
        extreme voxels sit exactly on the field-of-view boundary, so a residual of either sign decides
        whether the resampler interpolates or pads with NaN.

        Two evaluations of the same identity grid that differ only in rounding therefore disagree about
        which voxels are in the field of view. Comparing the look-up-table branch against the per-voxel
        one measures exactly the disagreement that separates the CPU and CUDA objectives, since CUDA
        has no table and evaluates per voxel.
    */
    for (const bool anisotropic : { false, true }) {
        const std::string name = std::string("3D, spacing 5") + (anisotropic ? ", anisotropic" : "");
        SECTION(name) {
            const NiftiImage reference = MakeReference(true, 30, anisotropic);
            const NiftiImage viaLut = EvaluateIdentityGrid(reference, kProductionGridSpacing, false);
            const NiftiImage viaPerVoxel = EvaluateIdentityGrid(reference, kProductionGridSpacing, true);
            const NiftiImage exact = CreateDeformationField(reference);

            const Deviation between = CompareImages(viaLut, viaPerVoxel);
            const FovFlips flips = FovDisagreements(viaLut, viaPerVoxel, reference);
            const FovFlips lutFlips = FovDisagreements(viaLut, exact, reference);
            const FovFlips perVoxelFlips = FovDisagreements(viaPerVoxel, exact, reference);

            NR_COUT << "  " << name << ": table vs per-voxel " << std::scientific << between.max
                    << " over " << between.differing << " values; field-of-view disagreements: "
                    << flips.total << " between the two (" << lutFlips.total << " table vs exact, "
                    << perVoxelFlips.total << " per-voxel vs exact), of "
                    << reference.nVoxelsPerVolume() << " voxels" << std::endl;

            // The two paths must actually differ, or there is nothing here to measure
            INFO("table and per-voxel agree bit for bit, so this case measures nothing");
            REQUIRE(between.differing > 0);

            // Both stay at rounding scale: neither path is wrong, they merely round differently
            REQUIRE(between.max < 1e-4);

            // The count itself is a property of the geometry - an identity field puts every surface
            // voxel exactly on a face, which is the worst case - so it is reported, not gated. What is
            // gated is where the flips can occur: a residual of this size cannot change the verdict
            // for a position half a voxel inside the boundary, so any such flip would mean the
            // difference had stopped being rounding and become an error.
            INFO(flips.total << " voxels change field-of-view classification, of which "
                 << flips.awayFromBoundary << " are more than half a voxel from any face");
            REQUIRE(flips.awayFromBoundary == 0);
            // Every flip is a surface voxel, so they cannot outnumber the surface
            REQUIRE(flips.total < reference.nVoxelsPerVolume());
        }
    }
}

TEST_CASE("A refined identity grid still evaluates to the identity", "[unit]") {
    // Grid refinement runs between pyramid levels and halves the spacing. It has to preserve the
    // transformation exactly, so refining an identity grid must leave an identity grid - if it did
    // not, every level change would inject a spurious deformation.
    for (const bool is3D : { false, true }) {
        SECTION(is3D ? "3D" : "2D") {
            const NiftiImage reference = MakeReference(is3D, 24, false);
            NiftiImage grid = MakeGrid(reference, kProductionGridSpacing);
            const NiftiImage before = EvaluateIdentityGrid(reference, kProductionGridSpacing, true);

            NiftiImage referenceForRefinement(reference);   // the API takes a non-const nifti_image*
            reg_spline_refineControlPointGrid(grid, referenceForRefinement);

            NiftiImage field = CreateDeformationField(reference);
            reg_spline_getDeformationField(grid, field, nullptr, false, true, true);

            const NiftiImage expected = CreateDeformationField(reference);
            const Deviation deviation = CompareImages(field, expected);
            const Deviation againstCoarse = CompareImages(field, before);
            NR_COUT << "  " << (is3D ? "3D" : "2D") << " refined identity grid: residual "
                    << std::scientific << deviation.max << ", vs the coarse grid's field "
                    << againstCoarse.max << std::endl;

            // The refined grid has more nodes at half the spacing, so it is a different
            // parametrisation - but of the same identity transformation
            INFO("refined residual " << deviation.max);
            REQUIRE(deviation.max < 1e-4);
        }
    }
}

/* ******************************* linear spline grids ******************************* */

namespace {

// A linear spline grid: the same node lattice, interpolated with the (1-t, t) hat basis instead of
// the cubic one. Selected by intent_p1 == LIN_SPLINE_GRID; 3D only in production.
NiftiImage MakeLinearGrid(const NiftiImage& reference, float spacingInVoxels) {
    NiftiImage grid = MakeGrid(reference, spacingInVoxels);
    grid->intent_p1 = LIN_SPLINE_GRID;
    return grid;
}

NiftiImage EvaluateLinear(const NiftiImage& reference, const NiftiImage& grid, int *mask = nullptr) {
    NiftiImage field = CreateDeformationField(reference);
    NiftiImage gridCopy(grid);
    reg_spline_getDeformationField(gridCopy, field, mask, false, true);
    return field;
}

} // namespace

TEST_CASE("Linear spline field reproduces an affine transformation", "[unit]") {
    /*
        Linear interpolation has exact linear precision, so a grid holding A(p_node) must evaluate to
        A(p_voxel) at every voxel - for a translation AND a general linear map alike, unlike the
        boundary behaviour of the cubic case: the two-tap hat stencil never leaves a grid built by
        reg_createControlPointGrid, whose one-node margin covers it everywhere.

        Linear interpolation also INTERPOLATES: at a voxel coinciding with a node the weights are
        (1, 0) and the field must equal that node's coefficient exactly - checked implicitly here
        since node-coincident voxels are part of the sweep and the tolerance is rounding-level.
    */
    for (const bool anisotropic : { false, true })
        for (const auto& [label, affine] : { std::pair{ "translation", Translation(3.5f, -2.25f, 1.75f) },
                                             std::pair{ "general linear map", LinearMap() } }) {
            const std::string name = std::string("3D, ") + (anisotropic ? "anisotropic, " : "identity sform, ") + label;
            SECTION(name) {
                const NiftiImage reference = MakeReference(true, 24, anisotropic);
                NiftiImage grid = MakeLinearGrid(reference, kProductionGridSpacing);
                ApplyAffineToGrid(grid, affine);

                const Deviation deviation = CompareImages(EvaluateLinear(reference, grid),
                                                          ExpectedAffineField(reference, affine));
                ReportDeviation(name, deviation);
                INFO(name << ": max deviation " << deviation.max);
                REQUIRE(deviation.max < 1e-4);
            }
        }
}

TEST_CASE("Linear spline composition reproduces a translation", "[unit]") {
    /*
        The composed path maps each stored position through the grid's real-to-voxel matrix and
        interpolates there; taps that leave the grid slide, as in the cubic composition. Sliding
        extends a translation field exactly - the clamped node holds p_clamped + t and the shift
        adds p_tap - p_clamped - so the answer is position + t everywhere, whether the position
        lands inside the grid or far outside it.
    */
    for (const auto& [label, spread] : { std::pair{ "positions inside the grid", 3.f },
                                         std::pair{ "positions far outside the grid", 60.f } })
    SECTION(label) {
        const NiftiImage reference = MakeReference(true, 16, false);
        const mat44 affine = Translation(3.5f, -2.25f, 1.75f);
        NiftiImage grid = MakeLinearGrid(reference, kProductionGridSpacing);
        ApplyAffineToGrid(grid, affine);

        NiftiImage field = CreateDeformationField(reference);
        ScatterPositions(field, spread);
        const NiftiImage input(field, NiftiImage::Copy::Image);

        NiftiImage gridCopy(grid);
        reg_spline_getDeformationField(gridCopy, field, nullptr, true, true);

        RequireChanged(input, field, "the composed deformation field");
        const size_t volume = field.nVoxelsPerVolume();
        const auto inPtr = input.data();
        const auto outPtr = field.data();
        double maxDeviation = 0;
        for (size_t i = 0; i < volume; ++i)
            for (int c = 0; c < 3; ++c) {
                const double expected = double(inPtr[c * volume + i]) + double(affine.m[c][3]);
                maxDeviation = std::max(maxDeviation,
                                        std::abs(double(outPtr[c * volume + i]) - expected));
            }
        NR_COUT << "  linear composition, translated grid, " << label << ": max deviation "
                << std::scientific << maxDeviation << std::endl;
        REQUIRE(maxDeviation < 1e-4);
    }
}

TEST_CASE("Linear spline field conventions and guards", "[unit]") {
    SECTION("masked voxels: zeroed without composition, untouched with it") {
        // The same split the cubic paths have, pinned for the linear one: the non-composed loop
        // writes its zero-initialised accumulator regardless of the mask, the composed loop skips
        // the voxel entirely
        const NiftiImage reference = MakeReference(true, 12, false);
        NiftiImage grid = MakeLinearGrid(reference, kProductionGridSpacing);
        ApplyAffineToGrid(grid, LinearMap());
        const size_t volume = CreateDeformationField(reference).nVoxelsPerVolume();
        std::vector<int> mask(volume);
        for (size_t i = 0; i < volume; ++i) mask[i] = i < volume / 2 ? 0 : -1;

        NiftiImage field = EvaluateLinear(reference, grid, mask.data());
        {
            const auto ptr = field.data();
            for (size_t i = volume / 2; i < volume; ++i)
                for (int c = 0; c < 3; ++c)
                    REQUIRE(static_cast<float>(ptr[c * volume + i]) == 0.f);
        }

        NiftiImage composed = CreateDeformationField(reference);
        const NiftiImage before(composed, NiftiImage::Copy::Image);
        NiftiImage gridCopy(grid);
        reg_spline_getDeformationField(gridCopy, composed, mask.data(), true, true);
        {
            const auto ptr = composed.data();
            const auto beforePtr = before.data();
            for (size_t i = volume / 2; i < volume; ++i)
                for (int c = 0; c < 3; ++c)
                    REQUIRE(static_cast<float>(ptr[c * volume + i]) ==
                            static_cast<float>(beforePtr[c * volume + i]));
        }
    }
    SECTION("a 2D linear grid is rejected") {
        // Production has no 2D implementation and must say so rather than dispatch to the wrong one
        const NiftiImage reference = MakeReference(false, 12, false);
        NiftiImage grid = MakeLinearGrid(reference, kProductionGridSpacing);
        NiftiImage field = CreateDeformationField(reference);
        REQUIRE_THROWS_AS(reg_spline_getDeformationField(grid, field, nullptr, false, true),
                          std::runtime_error);
    }
    SECTION("float and double instantiations agree on an affine grid") {
        // Both dispatch branches, same oracle: the double field must reproduce the affine too
        const NiftiImage reference = MakeReference(true, 16, false);
        NiftiImage grid = MakeLinearGrid(reference, kProductionGridSpacing);
        ApplyAffineToGrid(grid, LinearMap());

        NiftiImage fieldDouble = CreateDeformationField(reference);
        reg_tools_changeDatatype<double>(fieldDouble);
        NiftiImage gridDouble(grid);
        reg_tools_changeDatatype<double>(gridDouble);
        reg_spline_getDeformationField(gridDouble, fieldDouble, nullptr, false, true);
        reg_tools_changeDatatype<float>(fieldDouble);

        const Deviation deviation = CompareImages(fieldDouble,
                                                  ExpectedAffineField(reference, LinearMap()));
        ReportDeviation("3D double instantiation, linear map", deviation);
        REQUIRE(deviation.max < 1e-4);
    }
}
