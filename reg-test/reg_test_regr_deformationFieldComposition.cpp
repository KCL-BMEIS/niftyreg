// OpenCL is not supported for this test
#undef USE_OPENCL

#include "reg_test_common.h"
#include "CudaF3dContent.h"

/*
    GetDeformationField, CPU against CUDA.

    Both the composed and the non-composed variants are covered. The composed one is what the
    velocity-field flow generation uses (reg_spline_getFlowFieldFromVelocityGrid, CPU and CUDA alike), and
    the non-composed one is how a plain run builds its deformation field, so between them they sit
    under the first objective evaluation of every registration.

    The geometries are chosen so that the comparison is sensitive rather than merely broad:

      - Grid spacings from one voxel to five. A grid denser than the reference puts the four-tap window
        past the edge of the grid for most voxels, which is the only way to exercise the sliding
        extrapolation; a coarse grid keeps every tap comfortably inside and would never reach it.
      - An anisotropic sform with a shifted origin alongside the identity one. Under an identity sform
        the sub-voxel offsets are exact binary fractions, so the basis weights are representable
        exactly and two implementations agree however they arrange the arithmetic. Only an oblique
        real-to-voxel matrix makes the association of the products observable.
      - 2D and 3D, which are separate implementations on both backends.

    Equality is required. Both backends evaluate the same weights over the same taps in the same order,
    so any difference is a difference in the arithmetic, not in the result being approximated.
*/

namespace {

NiftiImage MakeImage(bool is3D, double phase, bool anisotropic) {
    const NiftiImage::dim_t size = 12;
    std::vector<NiftiImage::dim_t> dims(is3D ? 3 : 2, size);
    NiftiImage img(dims, NIFTI_TYPE_FLOAT32);
    mat44 mat;
    Mat44Eye(&mat);
    if (anisotropic) {
        mat.m[0][0] = 1.7f; mat.m[1][1] = 2.3f; mat.m[2][2] = is3D ? 1.1f : 1.f;
        mat.m[0][3] = -3.5f; mat.m[1][3] = 2.25f; mat.m[2][3] = is3D ? -1.75f : 0.f;
    }
    img->sform_code = 1;
    img->sto_xyz = mat;
    img->sto_ijk = nifti_mat44_inverse(mat);
    img->qform_code = 0;
    img->dx = img->pixdim[1] = mat.m[0][0];
    img->dy = img->pixdim[2] = mat.m[1][1];
    img->dz = img->pixdim[3] = is3D ? mat.m[2][2] : 1.f;
    const int nx = img->nx, ny = img->ny, nz = img->nz;
    auto ptr = img.data();
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const double x = 2.0 * M_PI * i / nx, y = 2.0 * M_PI * j / ny;
                const double z = nz > 1 ? 2.0 * M_PI * k / nz : 0.0;
                ptr[(static_cast<size_t>(k) * ny + j) * nx + i] =
                    static_cast<float>(100.0 * (1.5 + std::sin(x + phase) * std::cos(y) * std::cos(z)));
            }
    return img;
}

NiftiImage MakeGrid(const NiftiImage& reference, float spacingInVoxels) {
    NiftiImage grid;
    const float spacing[3]{ reference->dx * spacingInVoxels, reference->dy * spacingInVoxels,
                            reference->dz * spacingInVoxels };
    reg_createControlPointGrid<float>(grid, reference, spacing);
    const size_t volume = grid.nVoxelsPerVolume();
    const int nx = grid->nx, ny = grid->ny, nz = grid->nz;
    const int components = nz > 1 ? 3 : 2;
    auto ptr = grid.data();
    for (int c = 0; c < components; ++c)
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    const size_t index = c * volume + (static_cast<size_t>(k) * ny + j) * nx + i;
                    const double x = 2.0 * M_PI * i / nx, y = 2.0 * M_PI * j / ny;
                    const double z = nz > 1 ? 2.0 * M_PI * k / nz : 0.0;
                    ptr[index] = static_cast<float>(ptr[index]) +
                        0.3f * static_cast<float>(std::sin(x + c) * std::cos(y) * (nz > 1 ? std::cos(z) : 1.0));
                }
    return grid;
}

double MaxDifference(const NiftiImage& a, const NiftiImage& b, size_t& differing) {
    const auto aPtr = a.data();
    const auto bPtr = b.data();
    double maxDiff = 0;
    differing = 0;
    for (size_t i = 0; i < a.nVoxels(); ++i) {
        const double d = std::abs(static_cast<double>(aPtr[i]) - static_cast<double>(bPtr[i]));
        if (d > 0) ++differing;
        maxDiff = std::max(maxDiff, d);
    }
    return maxDiff;
}

void RunCase(bool is3D, bool composition, float spacing, bool anisotropic, const std::string& name) {
    Platform platformCpu(PlatformType::Cpu);
    Platform platformCuda(PlatformType::Cuda);

    const NiftiImage reference = MakeImage(is3D, 0.0, anisotropic);
    const NiftiImage floating = MakeImage(is3D, 0.6, anisotropic);
    NiftiImage gridCpu = MakeGrid(reference, spacing), gridCuda(gridCpu);
    NiftiImage refCpu(reference), floCpu(floating), refCuda(reference), floCuda(floating);

    unique_ptr<F3dContent> contentCpu{ new F3dContent(refCpu, floCpu, gridCpu) };
    unique_ptr<F3dContent> contentCuda{ new CudaF3dContent(refCuda, floCuda, gridCuda) };
    unique_ptr<Compute> computeCpu{ platformCpu.CreateCompute(*contentCpu) };
    unique_ptr<Compute> computeCuda{ platformCuda.CreateCompute(*contentCuda) };

    computeCpu->GetDeformationField(composition, true);
    computeCuda->GetDeformationField(composition, true);

    // A deformation field is never all-zero (it holds positions), so an unwritten one is caught here
    RequireNonZero(contentCpu->GetDeformationField(), "the CPU deformation field");
    RequireNonZero(contentCuda->GetDeformationField(), "the CUDA deformation field");

    size_t differing = 0;
    const double difference = MaxDifference(contentCpu->GetDeformationField(),
                                            contentCuda->GetDeformationField(), differing);
    NR_COUT << "  " << std::setw(48) << std::left << name
            << " max diff = " << std::scientific << std::setprecision(3) << difference
            << " (" << differing << " differing)" << std::endl;
    INFO(name << ": " << differing << " values differ, max " << difference);
    REQUIRE(difference == 0);
}

} // namespace

TEST_CASE("Regression Deformation Field Composition", "[regression]") {
    for (const bool is3D : { false, true }) {
        for (const bool composition : { false, true }) {
            for (const auto& [label, spacing, anisotropic] :
                 { std::tuple{ "grid 2 voxels", 2.f, false },
                   std::tuple{ "grid 2 voxels, anisotropic", 2.f, true },
                   std::tuple{ "grid 1 voxel (as refined)", 1.f, false },
                   std::tuple{ "grid 5 voxels (reg_f3d default)", 5.f, false },
                   std::tuple{ "grid 5 voxels, anisotropic", 5.f, true } }) {
                // The 3D non-composed evaluation at exactly 5 voxels takes the CPU table-driven
                // branch, which rounds differently from the per-voxel evaluation CUDA uses. It is
                // gated by the case below, at the tolerance that branch actually costs.
                if (is3D && !composition && spacing == 5.f)
                    continue;
                const std::string name = std::string(is3D ? "3D" : "2D") +
                    (composition ? ", composition" : ", no composition") + ", " + label;
                SECTION(name) { RunCase(is3D, composition, spacing, anisotropic, name); }
            }
        }
    }
}

TEST_CASE("Regression Deformation Field at the default control point spacing", "[regression]") {
    /*
        The one geometry the sweep above leaves out, and the one nearly every 3D run uses: a control
        point spacing of exactly five voxels, where the CPU precomputes a 125x64 table of basis
        products instead of evaluating the basis per voxel (the look-up-table branch of
        reg_cubic_spline_getDeformationField3D). The table holds
        the same products, but forming them once and reusing them rounds differently from forming them
        per voxel, and CUDA has no equivalent.

        Two things are gated here, and both can fail:

          1. With forceNoLut, the two backends must agree exactly. This is the real cross-backend
             check for this geometry - the kernels, the gathering and the extrapolation all have to be
             right for it to hold - and it is what identifies the table as the entire source of the
             difference rather than merely a contributor.

          2. Without it, the difference must be non-zero and small. Bounding it catches the table
             drifting; requiring it to be non-zero means that if the table is ever dropped, or
             implemented on the device, this case fails and says so instead of quietly passing.
    */
    for (const bool anisotropic : { false, true }) {
        const std::string name = std::string("3D, no composition, grid 5 voxels") +
            (anisotropic ? ", anisotropic" : "");
        SECTION(name) {
            Platform platformCpu(PlatformType::Cpu);
            Platform platformCuda(PlatformType::Cuda);

            const NiftiImage reference = MakeImage(true, 0.0, anisotropic);
            const NiftiImage floating = MakeImage(true, 0.6, anisotropic);
            NiftiImage gridCpu = MakeGrid(reference, 5.f), gridCuda(gridCpu);
            NiftiImage refCpu(reference), floCpu(floating), refCuda(reference), floCuda(floating);

            unique_ptr<F3dContent> contentCpu{ new F3dContent(refCpu, floCpu, gridCpu) };
            unique_ptr<F3dContent> contentCuda{ new CudaF3dContent(refCuda, floCuda, gridCuda) };
            unique_ptr<Compute> computeCpu{ platformCpu.CreateCompute(*contentCpu) };
            unique_ptr<Compute> computeCuda{ platformCuda.CreateCompute(*contentCuda) };

            computeCuda->GetDeformationField(false, true);
            RequireNonZero(contentCuda->GetDeformationField(), "the CUDA deformation field");

            // 1. The per-voxel evaluation, which is what the device computes
            NiftiImage perVoxel = CreateDeformationField(reference);
            NiftiImage gridForNoLut(gridCpu);
            reg_spline_getDeformationField(gridForNoLut, perVoxel, nullptr, false, true, true);
            size_t differingExact = 0;
            const double exact = MaxDifference(perVoxel, contentCuda->GetDeformationField(), differingExact);
            NR_COUT << "  " << std::setw(48) << std::left << name + ", forceNoLut"
                    << " max diff = " << std::scientific << std::setprecision(3) << exact << std::endl;
            INFO(name << ", forceNoLut: " << differingExact << " values differ, max " << exact);
            REQUIRE(exact == 0);

            // 2. The table, which is what a default run actually takes
            computeCpu->GetDeformationField(false, true);
            size_t differingTable = 0;
            const double table = MaxDifference(contentCpu->GetDeformationField(),
                                               contentCuda->GetDeformationField(), differingTable);
            NR_COUT << "  " << std::setw(48) << std::left << name + ", table"
                    << " max diff = " << std::scientific << std::setprecision(3) << table
                    << " (" << differingTable << " differing)" << std::endl;
            INFO(name << ", table: " << differingTable << " values differ, max " << table);
            // Observed ~3e-6 on coordinates of order 30 mm
            REQUIRE(table < 1e-4);
            // If this ever holds, the table and the device agree and the bound above should be
            // tightened to equality rather than left as a licence to differ
            INFO("the table now agrees with the device exactly - replace the bound above with equality");
            REQUIRE(table > 0);
        }
    }
}
