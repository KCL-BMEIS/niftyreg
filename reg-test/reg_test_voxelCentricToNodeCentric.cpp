// OpenCL is not supported for this test
#undef USE_OPENCL

#include "reg_test_common.h"

/*
    reg_voxelCentricToNodeCentric: the voxel-based measure gradient sampled onto the control point
    grid, checked against closed-form expectations rather than a second copy of the algorithm.

    The operation, as production defines it:
      1. maps each node through T = voxelGrad_ijk * (ext^-1 *) nodeGrad_xyz into voxel coordinates;
      2. tri/bilinearly interpolates the voxel gradient there, skipping out-of-bounds taps without
         renormalising;
      3. applies the reorientation matrix R TRANSPOSED (R = floating ijk matrix, ext^-1-composed);
      4. scales by weight * prod(ratio), ratio_i = row-norm of nodeGrad sto_xyz / voxelGrad pixdim.

    Each ingredient is pinned by a case whose expected value is computable by hand:

      - A CONSTANT gradient field g. Linear interpolation weights sum to one, so any node whose taps
        all land in-bounds must hold exactly weight_eff * R^T g. Positions are arranged to land on
        integer and half-integer voxel coordinates, where the float weights are exact powers of two,
        so the check is equality. The reorientation matrices have non-zero off-diagonals - with a
        diagonal matrix, R and R^T act identically on every vector and step 3's transpose would be
        untestable.
      - A LINEAR gradient field g(v) = G v + h. Linear interpolation reproduces linear fields, so the
        node value is weight_eff * R^T g(T * node) - this is what pins T itself, since a wrong
        transformation samples the field at the wrong place and a constant field cannot see that.
      - BOUNDARY nodes. The node->voxel map is chosen so nodes fall on and beyond the voxel image's
        edge. With a constant field, the expected value is g scaled by the sum of the in-bounds tap
        weights: 1 inside, 1/4 or 1/2 in the partial band (exact in float), 0 outside. This is the
        branch where a 2D kernel bug once lived, and the one a well-covered interior can never reach.
      - The grid-to-image affine EXTENSION, present and absent: present, both T and R acquire an
        ext^-1 factor.
      - Both sform_code settings, chosen per case - not via hidden state shared between cases.

    The `update` accumulation flag is pinned by a direct production call (Compute always passes
    false): updating twice from a zeroed start must equal exactly twice a single assignment.
*/

namespace {

// ---- fixed geometry ---------------------------------------------------------------------------

// Node grid -> world: uniform scale 1.5 with a -0.75 shift, so node n maps to voxel 1.5n - 0.75
// (voxelGrad_ijk = identity below): n=0 -> -0.75 (partial taps), n=1 -> 0.75, n=2 -> 2.25,
// n=3 -> 3.75, n=4 -> 5.25 (interior), and the ratio weights are 1.5 per axis exactly.
mat44 NodeToWorld(bool is3D) {
    mat44 m;
    Mat44Eye(&m);
    m.m[0][0] = m.m[1][1] = 1.5f;
    m.m[0][3] = m.m[1][3] = -0.75f;
    if (is3D) { m.m[2][2] = 1.5f; m.m[2][3] = -0.75f; }
    return m;
}

// A variant whose sampled positions are exact multiples of 0.5 voxels, where the linear weights
// (0.5, 0.5), (1, 0) are exact in float and the constant-field case can demand equality:
// node n -> voxel 1.5n - 0.5: n=0 -> -0.5 (half-out), n=1 -> 1, n=2 -> 2.5, n=3 -> 4, n=4 -> 5.5.
mat44 NodeToWorldHalf(bool is3D) {
    mat44 m = NodeToWorld(is3D);
    m.m[0][3] = m.m[1][3] = -0.5f;
    if (is3D) m.m[2][3] = -0.5f;
    return m;
}

// Reorientation with non-zero off-diagonals, so R != R^T and the transpose in step 3 is observable
mat44 Reorientation(bool is3D) {
    mat44 m;
    Mat44Eye(&m);
    m.m[0][0] = 0.9f;  m.m[0][1] = 0.2f;
    m.m[1][0] = -0.3f; m.m[1][1] = 1.1f;
    if (is3D) {
        m.m[0][2] = -0.1f; m.m[1][2] = 0.25f;
        m.m[2][0] = 0.15f; m.m[2][1] = -0.2f; m.m[2][2] = 0.8f;
    }
    return m;
}

// R^T g in double, reading the float matrix entries exactly
void ApplyTransposed(const mat44& r, const double g[3], double out[3]) {
    for (int i = 0; i < 3; ++i)
        out[i] = double(r.m[0][i]) * g[0] + double(r.m[1][i]) * g[1] + double(r.m[2][i]) * g[2];
}

// The weight the operation applies on top of the caller's: prod over axes of
// (row-norm of nodeToWorld) / (voxelGrad pixdim). Both are ours to choose; pixdim is 1 here.
double EffectiveWeight(const mat44& nodeToWorld, bool is3D, float weight) {
    double w = weight;
    for (int i = 0; i < (is3D ? 3 : 2); ++i) {
        const double norm = std::sqrt(Square(double(nodeToWorld.m[i][0])) +
                                      Square(double(nodeToWorld.m[i][1])) +
                                      Square(double(nodeToWorld.m[i][2])));
        w *= norm;
    }
    return w;
}

// The in-bounds fraction of the linear stencil at 1D position p over [0, n-1]: the sum of the tap
// weights that land inside. Exact for the multiples of 0.25 used here.
double InBoundsWeight1d(double p, int n) {
    const int pre = static_cast<int>(std::floor(p));
    const double t = p - pre;
    double w = 0;
    if (pre >= 0 && pre < n) w += 1.0 - t;
    if (pre + 1 >= 0 && pre + 1 < n) w += t;
    return w;
}

// ---- driving the operation --------------------------------------------------------------------

struct Fixture {
    NiftiImage reference, floating, controlPointGrid;
    unique_ptr<F3dContent> content;
    unique_ptr<Compute> compute;
    bool is3D;

    // The voxel image is 8 wide so every interior node's stencil fits; the grid that
    // CreateControlPointGrid builds for it has ceil(8/2+3) = 7 nodes per axis, of which the closed
    // forms below use the ones the NodeToWorld maps keep inside or deliberately push outside.
    Fixture(Platform& platform, bool is3D, const mat44& nodeToWorld, const mat44& reorientation,
            bool withExtension, bool useSform)
        : is3D(is3D) {
        std::vector<NiftiImage::dim_t> dims(is3D ? 3 : 2, 8);
        reference = NiftiImage(dims, NIFTI_TYPE_FLOAT32);
        setIdentitySform(reference);
        floating = NiftiImage(reference, NiftiImage::Copy::Image);

        // The reorientation the Compute wrapper passes is the floating image's ijk matrix
        if (useSform) {
            floating->sform_code = 1;
            floating->sto_ijk = reorientation;
            floating->sto_xyz = nifti_mat44_inverse(reorientation);
            floating->qform_code = 0;
        } else {
            floating->sform_code = 0;
            floating->qform_code = 1;
            floating->qto_ijk = reorientation;
            floating->qto_xyz = nifti_mat44_inverse(reorientation);
        }

        controlPointGrid = CreateControlPointGrid(reference);
        unique_ptr<F3dContentCreator> creator{
            dynamic_cast<F3dContentCreator*>(platform.CreateContentCreator(ContentType::F3d)) };
        content.reset(creator->Create(reference, floating, controlPointGrid));

        // Node grid geometry: sform carries NodeToWorld, whose row norms set the ratio weights
        NiftiImage& transGrad = content->F3dContent::GetTransformationGradient();
        transGrad->sform_code = 1;
        transGrad->sto_xyz = nodeToWorld;
        transGrad->sto_ijk = nifti_mat44_inverse(nodeToWorld);
        transGrad->qform_code = 0;

        if (withExtension) {
            // The stored extension is inverted by the operation, so store E and expect E^-1 factors
            const mat44 extension = Reorientation(is3D);   // reuse: full, invertible
            const mat44 invExtension = nifti_mat44_inverse(extension);
            nifti_add_extension(transGrad, reinterpret_cast<const char*>(&invExtension),
                                sizeof(invExtension), NIFTI_ECODE_IGNORE);
        }

        // Voxel gradient geometry: identity ijk, so voxel coordinates == world coordinates
        NiftiImage& voxelGrad = content->F3dContent::GetVoxelBasedMeasureGradient();
        mat44 eye;
        Mat44Eye(&eye);
        voxelGrad->sform_code = 1;
        voxelGrad->sto_ijk = eye;
        voxelGrad->sto_xyz = eye;
        voxelGrad->qform_code = 0;

        compute.reset(platform.CreateCompute(*content));
    }

    // Fill the voxel gradient with g(v) = base + G v (G == nullptr means constant)
    void FillVoxelGradient(const double base[3], const double (*slope)[3] = nullptr) {
        NiftiImage& voxelGrad = content->F3dContent::GetVoxelBasedMeasureGradient();
        const size_t volume = voxelGrad.nVoxelsPerVolume();
        const int nx = voxelGrad->nx, ny = voxelGrad->ny, nz = voxelGrad->nz;
        const int components = is3D ? 3 : 2;
        auto ptr = voxelGrad.data();
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    const size_t index = (static_cast<size_t>(k) * ny + j) * nx + i;
                    for (int c = 0; c < components; ++c) {
                        double value = base[c];
                        if (slope) value += slope[c][0] * i + slope[c][1] * j + slope[c][2] * k;
                        ptr[c * volume + index] = static_cast<float>(value);
                    }
                }
        content->UpdateVoxelBasedMeasureGradient();
    }

    NiftiImage& Run(float weight) {
        compute->VoxelCentricToNodeCentric(weight);
        return content->GetTransformationGradient();
    }
};

// The voxel coordinate node (nx,ny,nz) maps to under nodeToWorld (voxelGrad ijk is identity)
void NodeVoxelCoord(const mat44& nodeToWorld, int n0, int n1, int n2, double out[3]) {
    for (int i = 0; i < 3; ++i)
        out[i] = double(nodeToWorld.m[i][0]) * n0 + double(nodeToWorld.m[i][1]) * n1 +
                 double(nodeToWorld.m[i][2]) * n2 + double(nodeToWorld.m[i][3]);
}

} // namespace

TEST_CASE("Voxel centric to node centric: constant field", "[unit]") {
    /*
        Constant field g, positions on exact multiples of 0.5 voxels: interior nodes must hold
        weight_eff * R^T g bit-exactly (all the arithmetic is exact in float), boundary nodes the
        same scaled by their in-bounds tap weight, and outside nodes zero.
    */
    constexpr float weight = 0.7f;
    const double g[3] = { 2.0, -1.5, 0.5 };

    for (auto&& platformType : PlatformTypes)
        for (const bool is3D : { false, true })
            for (const bool useSform : { true, false }) {
                Platform platform(platformType);
                const std::string name = std::string(is3D ? "3D" : "2D") + " " + platform.GetName() +
                    (useSform ? " sform" : " qform");
                SECTION(name) {
                    const mat44 nodeToWorld = NodeToWorldHalf(is3D);
                    const mat44 reorientation = Reorientation(is3D);
                    Fixture f(platform, is3D, nodeToWorld, reorientation, false, useSform);
                    const double gUsed[3] = { g[0], g[1], is3D ? g[2] : 0.0 };
                    f.FillVoxelGradient(gUsed);

                    NiftiImage& nodeGrad = f.Run(weight);
                    const size_t volume = nodeGrad.nVoxelsPerVolume();
                    const int components = is3D ? 3 : 2;
                    const auto ptr = nodeGrad.data();

                    double rtg[3];
                    ApplyTransposed(reorientation, gUsed, rtg);
                    const double weightEff = EffectiveWeight(nodeToWorld, is3D, weight);

                    const int nNodes[3] = { nodeGrad->nx, nodeGrad->ny, is3D ? nodeGrad->nz : 1 };
                    size_t interior = 0, partial = 0, outside = 0;
                    for (int k = 0; k < nNodes[2]; ++k)
                        for (int j = 0; j < nNodes[1]; ++j)
                            for (int i = 0; i < nNodes[0]; ++i) {
                                double coord[3];
                                NodeVoxelCoord(nodeToWorld, i, j, k, coord);
                                // In-bounds fraction of the stencil, exact for these positions
                                double fraction = InBoundsWeight1d(coord[0], 8) *
                                                  InBoundsWeight1d(coord[1], 8);
                                if (is3D) fraction *= InBoundsWeight1d(coord[2], 8);
                                if (fraction == 1) ++interior;
                                else if (fraction > 0) ++partial;
                                else ++outside;

                                const size_t index = (static_cast<size_t>(k) * nNodes[1] + j) * nNodes[0] + i;
                                for (int c = 0; c < components; ++c) {
                                    const float expected = static_cast<float>(
                                        static_cast<float>(weightEff * rtg[c]) * fraction);
                                    const float actual = ptr[c * volume + index];
                                    INFO("node (" << i << "," << j << "," << k << ") component " << c
                                         << ": in-bounds fraction " << fraction);
                                    // The products are powers of two apart, so demand near-equality
                                    REQUIRE(std::abs(actual - expected) <=
                                            1e-6f * std::max(1.f, std::abs(expected)));
                                }
                            }
                    // The case must actually cover all three bands, or the boundary claim is vacuous
                    INFO("interior " << interior << ", partial " << partial << ", outside " << outside);
                    REQUIRE(interior > 0);
                    REQUIRE(partial > 0);
                    REQUIRE(outside > 0);
                }
            }
}

TEST_CASE("Voxel centric to node centric: linear field pins the transformation", "[unit]") {
    /*
        g(v) = base + G v, sampled at T * node. A constant field cannot detect a wrong node->voxel
        transformation; a linear one makes the sampled value depend on where the node landed.
        Restricted to nodes whose stencils are fully inside, where linear interpolation reproduces
        the field to float rounding.
    */
    constexpr float weight = 1.f;
    const double base[3] = { 0.4, -0.2, 0.1 };
    const double slope[3][3] = { { 0.30, -0.10, 0.05 },
                                 { 0.15, 0.20, -0.10 },
                                 { -0.05, 0.10, 0.25 } };

    for (auto&& platformType : PlatformTypes)
        for (const bool is3D : { false, true })
            for (const bool withExtension : { false, true }) {
                Platform platform(platformType);
                const std::string name = std::string(is3D ? "3D" : "2D") + " " + platform.GetName() +
                    (withExtension ? " with extension" : "");
                SECTION(name) {
                    const mat44 nodeToWorld = NodeToWorld(is3D);
                    const mat44 reorientation = Reorientation(is3D);
                    Fixture f(platform, is3D, nodeToWorld, reorientation, withExtension, true);
                    f.FillVoxelGradient(base, slope);

                    NiftiImage& nodeGrad = f.Run(weight);
                    const size_t volume = nodeGrad.nVoxelsPerVolume();
                    const int components = is3D ? 3 : 2;
                    const auto ptr = nodeGrad.data();

                    // With an extension E stored (the operation inverts it, and we stored E^-1, so
                    // the effective factor is E): T = E * nodeToWorld and R_eff = Mat33(E) * R
                    mat44 t = nodeToWorld;
                    mat44 rEff = reorientation;
                    if (withExtension) {
                        const mat44 extension = Reorientation(is3D);
                        t = extension * nodeToWorld;
                        // mat33 composition as production does it: ext-factor * reorientation
                        mat44 extForR = extension;
                        extForR.m[0][3] = extForR.m[1][3] = extForR.m[2][3] = 0;
                        rEff = extForR * reorientation;
                        for (int i = 0; i < 3; ++i) rEff.m[i][3] = 0;
                    }
                    const double weightEff = EffectiveWeight(nodeToWorld, is3D, weight);

                    const int nNodes[3] = { nodeGrad->nx, nodeGrad->ny, is3D ? nodeGrad->nz : 1 };
                    size_t checked = 0;
                    for (int k = 0; k < nNodes[2]; ++k)
                        for (int j = 0; j < nNodes[1]; ++j)
                            for (int i = 0; i < nNodes[0]; ++i) {
                                double coord[3];
                                NodeVoxelCoord(t, i, j, k, coord);
                                // Only fully-interior stencils: reproduction holds exactly there
                                const double margin = 1e-6;
                                bool inside = coord[0] >= margin && coord[0] <= 7 - margin &&
                                              coord[1] >= margin && coord[1] <= 7 - margin;
                                if (is3D) inside = inside && coord[2] >= margin && coord[2] <= 7 - margin;
                                else if (coord[2] < 0 || coord[2] >= 1) continue;  // 2D z gate below
                                if (!inside) continue;

                                double gAt[3] = { 0, 0, 0 };
                                for (int c = 0; c < components; ++c)
                                    gAt[c] = base[c] + slope[c][0] * coord[0] + slope[c][1] * coord[1] +
                                             (is3D ? slope[c][2] * coord[2] : 0.0);
                                double expected[3];
                                ApplyTransposed(rEff, gAt, expected);

                                const size_t index = (static_cast<size_t>(k) * nNodes[1] + j) * nNodes[0] + i;
                                for (int c = 0; c < components; ++c) {
                                    const double actual = static_cast<float>(ptr[c * volume + index]);
                                    INFO("node (" << i << "," << j << "," << k << ") component " << c);
                                    REQUIRE(std::abs(actual - weightEff * expected[c]) < 1e-4);
                                }
                                ++checked;
                            }
                    INFO("nodes checked: " << checked);
                    REQUIRE(checked > 0);
                }
            }
}

TEST_CASE("Voxel centric to node centric: 2D nodes with an out-of-plane offset", "[unit]") {
    /*
        In 2D the third row of the transformation still decides whether any tap survives: the c-loop
        tests indexZ = Floor(coord_z) + c against nz = 1. An offset of -0.25 makes Floor = -1, so
        only the c = 1 tap is in-plane - and since 2D weights ignore basisZ, the value must be
        IDENTICAL to the zero-offset case. This asymmetry is easy to break in a rewrite (it is where
        an uninitialised-variable bug once lived on the device side), hence pinned.
    */
    constexpr float weight = 0.7f;
    const double g[3] = { 2.0, -1.5, 0.0 };
    for (auto&& platformType : PlatformTypes) {
        Platform platform(platformType);
        SECTION(platform.GetName()) {
            const mat44 reorientation = Reorientation(false);

            mat44 flat = NodeToWorldHalf(false);            // third row: coord_z = 0 for every node
            Fixture fFlat(platform, false, flat, reorientation, false, true);
            fFlat.FillVoxelGradient(g);
            NiftiImage& gradFlat = fFlat.Run(weight);

            mat44 offset = flat;
            offset.m[2][3] = -0.25f;                        // coord_z = -0.25: only the c=1 tap is in-plane
            Fixture fOffset(platform, false, offset, reorientation, false, true);
            fOffset.FillVoxelGradient(g);
            NiftiImage& gradOffset = fOffset.Run(weight);

            const Deviation deviation = CompareImages(gradFlat, gradOffset);
            ReportDeviation(std::string("2D out-of-plane offset, ") + platform.GetName(), deviation);
            REQUIRE(deviation.differing == 0);
            RequireNonZero(gradFlat, "the node-based gradient");
        }
    }
}

TEST_CASE("Voxel centric to node centric: update accumulates exactly", "[unit]") {
    /*
        The update flag switches assignment to accumulation (production line ~1672). Compute always
        passes false, so this is pinned with direct calls: assign, then update once more - the result
        must be exactly twice the assignment, for every node including the boundary ones.
    */
    for (const bool is3D : { false, true }) {
        SECTION(is3D ? "3D" : "2D") {
            std::vector<NiftiImage::dim_t> dims(is3D ? 3 : 2, 8);
            NiftiImage voxelGrad(std::vector<NiftiImage::dim_t>{ 8, 8, NiftiImage::dim_t(is3D ? 8 : 1),
                                                                 1, NiftiImage::dim_t(is3D ? 3 : 2) },
                                 NIFTI_TYPE_FLOAT32);
            setIdentitySform(voxelGrad);
            auto vPtr = voxelGrad.data();
            for (size_t i = 0; i < voxelGrad.nVoxels(); ++i)
                vPtr[i] = static_cast<float>(std::sin(0.37 * double(i)));

            const mat44 nodeToWorld = NodeToWorld(is3D);
            NiftiImage nodeGrad(std::vector<NiftiImage::dim_t>{ 5, 5, NiftiImage::dim_t(is3D ? 5 : 1),
                                                                1, NiftiImage::dim_t(is3D ? 3 : 2) },
                                NIFTI_TYPE_FLOAT32);
            nodeGrad->sform_code = 1;
            nodeGrad->sto_xyz = nodeToWorld;
            nodeGrad->sto_ijk = nifti_mat44_inverse(nodeToWorld);
            nodeGrad->qform_code = 0;

            const mat44 reorientation = Reorientation(is3D);

            NiftiImage assigned(nodeGrad, NiftiImage::Copy::Image);
            reg_voxelCentricToNodeCentric(assigned, voxelGrad, 0.7f, false, &reorientation);
            RequireNonZero(assigned, "the assigned node gradient");

            NiftiImage accumulated(assigned, NiftiImage::Copy::Image);
            reg_voxelCentricToNodeCentric(accumulated, voxelGrad, 0.7f, true, &reorientation);

            const auto aPtr = assigned.data();
            const auto uPtr = accumulated.data();
            for (size_t i = 0; i < assigned.nVoxels(); ++i) {
                INFO("value " << i);
                REQUIRE(static_cast<float>(uPtr[i]) == 2.f * static_cast<float>(aPtr[i]));
            }
        }
    }
}
