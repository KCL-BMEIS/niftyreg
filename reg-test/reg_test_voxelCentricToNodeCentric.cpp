// OpenCL is not supported for this test
#undef USE_OPENCL

#include "reg_test_common.h"

/*
    This test file contains the following unit tests:
    test functions: The node-based NMI gradient is extracted from the voxel-based NMI gradient
    In 2D and 3D
*/


class VoxelCentricToNodeCentricTest {
protected:
    using TestData = std::tuple<std::string, NiftiImage, NiftiImage, NiftiImage>;
    using TestCase = std::tuple<std::string, NiftiImage, NiftiImage>;

    inline static vector<TestCase> testCases;

public:
    VoxelCentricToNodeCentricTest() {
        if (!testCases.empty())
            return;

        // Create a random number generator
        std::mt19937 gen(0);
        std::uniform_real_distribution<float> distr(0, 1);

        // Create a 2D reference image
        vector<NiftiImage::dim_t> dimFlo{ 4, 4 };
        NiftiImage reference2d(dimFlo, NIFTI_TYPE_FLOAT32);

        // Create a 3D reference image
        dimFlo.push_back(4);
        NiftiImage reference3d(dimFlo, NIFTI_TYPE_FLOAT32);

        // Create the voxel-based measure gradients
        vector<NiftiImage::dim_t> dimGrad{ 4, 4, 1, 1, 2 };
        NiftiImage voxelBasedMeasureGradient2d(dimGrad, NIFTI_TYPE_FLOAT32);
        dimGrad[2] = 4; dimGrad[4] = 3;
        NiftiImage voxelBasedMeasureGradient3d(dimGrad, NIFTI_TYPE_FLOAT32);

        // Create the control point grids
        NiftiImage controlPointGrid2d = CreateControlPointGrid(reference2d);
        NiftiImage controlPointGrid3d = CreateControlPointGrid(reference3d);

        // Create the matrices and fill them with random values
        std::array<mat44, 4> matrices{};
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                for (int k = 0; k < 4; ++k)
                    matrices[i].m[j][k] = j == k ? distr(gen) : 0;

        // Generate the different test cases
        // Test 2D
        auto grad2dPtr = voxelBasedMeasureGradient2d.data();
        for (size_t i = 0; i < voxelBasedMeasureGradient2d.nVoxels(); ++i)
            grad2dPtr[i] = distr(gen);

        // Add the test data
        vector<TestData> testData;
        testData.emplace_back(TestData(
            "2D",
            std::move(reference2d),
            std::move(controlPointGrid2d),
            std::move(voxelBasedMeasureGradient2d)
        ));

        // Test 3D
        auto grad3dPtr = voxelBasedMeasureGradient3d.data();
        for (size_t i = 0; i < voxelBasedMeasureGradient3d.nVoxels(); ++i)
            grad3dPtr[i] = distr(gen);

        // Add the test data
        testData.emplace_back(TestData(
            "3D",
            std::move(reference3d),
            std::move(controlPointGrid3d),
            std::move(voxelBasedMeasureGradient3d)
        ));

        // Add platforms, composition, and bspline to the test data
        for (auto&& testData : testData) {
            for (auto&& platformType : PlatformTypes) {
                unique_ptr<Platform> platform{ new Platform(platformType) };
                unique_ptr<F3dContentCreator> contentCreator{ dynamic_cast<F3dContentCreator*>(platform->CreateContentCreator(ContentType::F3d)) };
                // Make a copy of the test data
                auto [testName, reference, controlPointGrid, voxelBasedMeasureGradient] = testData;
                // Create the content
                unique_ptr<F3dContent> content{ contentCreator->Create(reference, reference, controlPointGrid) };

                // Set the matrices required for computation
                nifti_image *floating = content->Content::GetFloating();
                if (floating->sform_code > 0)
                    floating->sto_ijk = matrices[0];
                else floating->qto_ijk = matrices[0];
                NiftiImage transGrad = content->F3dContent::GetTransformationGradient();
                static int sfc = 0;
                transGrad->sform_code = sfc++ % 2;
                if (transGrad->sform_code > 0)
                    transGrad->sto_xyz = matrices[1];
                else transGrad->qto_xyz = matrices[1];
                const mat44 invMatrix = nifti_mat44_inverse(matrices[2]);
                nifti_add_extension(transGrad, reinterpret_cast<const char*>(&invMatrix), sizeof(invMatrix), NIFTI_ECODE_IGNORE);

                // Set the voxel-based measure gradient to host the computation
                NiftiImage voxelGrad = content->F3dContent::GetVoxelBasedMeasureGradient();
                if (voxelGrad->sform_code > 0)
                    voxelGrad->sto_ijk = matrices[3];
                else voxelGrad->qto_ijk = matrices[3];
                voxelGrad.copyData(voxelBasedMeasureGradient);
                content->UpdateVoxelBasedMeasureGradient();

                // Compute the expected node-based NMI gradient
                const float weight = distr(gen);
                NiftiImage expTransGrad(transGrad, NiftiImage::Copy::ImageInfoAndAllocData);
                VoxelCentricToNodeCentric<float>(floating, expTransGrad, voxelGrad, weight);
                transGrad.disown(); voxelGrad.disown();

                // Extract the node-based NMI gradient from the voxel-based NMI gradient
                unique_ptr<Compute> compute{ platform->CreateCompute(*content) };
                compute->VoxelCentricToNodeCentric(weight);
                transGrad = NiftiImage(content->GetTransformationGradient(), NiftiImage::Copy::Image);

                testCases.push_back({ testName + " "s + platform->GetName() + " Weight="s + std::to_string(weight), std::move(transGrad), std::move(expTransGrad) });
            }
        }
    }

    template<typename DataType>
    void VoxelCentricToNodeCentric(const nifti_image *floating, NiftiImage& nodeGrad, const NiftiImage& voxelGrad, float weight) {
        const mat44 *voxelToMillimetre = floating->sform_code > 0 ? &floating->sto_ijk : &floating->qto_ijk;
        const bool is3d = nodeGrad->nz > 1;

        const size_t nodeNumber = NiftiImage::calcVoxelNumber(nodeGrad, 3);
        auto nodePtr = nodeGrad.data();
        auto nodePtrX = nodePtr.begin();
        auto nodePtrY = nodePtrX + nodeNumber;
        auto nodePtrZ = nodePtrY + nodeNumber;

        const size_t voxelNumber = NiftiImage::calcVoxelNumber(voxelGrad, 3);
        auto voxelPtr = voxelGrad.data();
        auto voxelPtrX = voxelPtr.begin();
        auto voxelPtrY = voxelPtrX + voxelNumber;
        auto voxelPtrZ = voxelPtrY + voxelNumber;

        // The transformation between the image and the grid
        mat44 transformation;
        // Voxel to millimetre in the grid image
        if (nodeGrad->sform_code > 0)
            transformation = nodeGrad->sto_xyz;
        else transformation = nodeGrad->qto_xyz;
        // Affine transformation between the grid and the reference image
        if (nodeGrad->num_ext > 0 && nodeGrad->ext_list[0].edata) {
            mat44 temp = *(reinterpret_cast<mat44*>(nodeGrad->ext_list[0].edata));
            temp = nifti_mat44_inverse(temp);
            transformation = reg_mat44_mul(&temp, &transformation);
        }
        // Millimetre to voxel in the reference image
        if (voxelGrad->sform_code > 0)
            transformation = reg_mat44_mul(&voxelGrad->sto_ijk, &transformation);
        else transformation = reg_mat44_mul(&voxelGrad->qto_ijk, &transformation);

        // The information has to be reoriented
        // Voxel to millimetre contains the orientation of the image that is used
        // to compute the spatial gradient (floating image)
        mat33 reorientation = reg_mat44_to_mat33(voxelToMillimetre);
        if (nodeGrad->num_ext > 0 && nodeGrad->ext_list[0].edata) {
            mat33 temp = reg_mat44_to_mat33(reinterpret_cast<mat44*>(nodeGrad->ext_list[0].edata));
            temp = nifti_mat33_inverse(temp);
            reorientation = nifti_mat33_mul(temp, reorientation);
        }
        // The information has to be weighted
        float ratio[3] = { nodeGrad->dx, nodeGrad->dy, nodeGrad->dz };
        for (int i = 0; i < (is3d ? 3 : 2); ++i) {
            if (nodeGrad->sform_code > 0) {
                ratio[i] = sqrt(Square(nodeGrad->sto_xyz.m[i][0]) +
                                Square(nodeGrad->sto_xyz.m[i][1]) +
                                Square(nodeGrad->sto_xyz.m[i][2]));
            }
            ratio[i] /= voxelGrad->pixdim[i + 1];
            weight *= ratio[i];
        }
        // For each node, the corresponding voxel is computed
        float nodeCoord[3], voxelCoord[3];
        for (int z = 0; z < nodeGrad->nz; z++) {
            nodeCoord[2] = static_cast<float>(z);
            for (int y = 0; y < nodeGrad->ny; y++) {
                nodeCoord[1] = static_cast<float>(y);
                for (int x = 0; x < nodeGrad->nx; x++) {
                    nodeCoord[0] = static_cast<float>(x);
                    reg_mat44_mul(&transformation, nodeCoord, voxelCoord);
                    // Linear interpolation
                    DataType basisX[2], basisY[2], basisZ[2];
                    const int pre[3] = { Floor(voxelCoord[0]), Floor(voxelCoord[1]), Floor(voxelCoord[2]) };
                    basisX[1] = voxelCoord[0] - static_cast<DataType>(pre[0]);
                    basisX[0] = static_cast<DataType>(1) - basisX[1];
                    basisY[1] = voxelCoord[1] - static_cast<DataType>(pre[1]);
                    basisY[0] = static_cast<DataType>(1) - basisY[1];
                    if (is3d) {
                        basisZ[1] = voxelCoord[2] - static_cast<DataType>(pre[2]);
                        basisZ[0] = static_cast<DataType>(1) - basisZ[1];
                    }
                    DataType interpolatedValue[3]{};
                    for (int c = 0; c < 2; ++c) {
                        const int indexZ = pre[2] + c;
                        if (-1 < indexZ && indexZ < voxelGrad->nz) {
                            for (int b = 0; b < 2; ++b) {
                                const int indexY = pre[1] + b;
                                if (-1 < indexY && indexY < voxelGrad->ny) {
                                    for (int a = 0; a < 2; ++a) {
                                        const int indexX = pre[0] + a;
                                        if (-1 < indexX && indexX < voxelGrad->nx) {
                                            const int index = (indexZ * voxelGrad->ny + indexY) * voxelGrad->nx + indexX;
                                            const DataType linearWeight = basisX[a] * basisY[b] * (is3d ? basisZ[c] : 1);
                                            interpolatedValue[0] += linearWeight * static_cast<DataType>(voxelPtrX[index]);
                                            interpolatedValue[1] += linearWeight * static_cast<DataType>(voxelPtrY[index]);
                                            if (is3d)
                                                interpolatedValue[2] += linearWeight * static_cast<DataType>(voxelPtrZ[index]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    DataType reorientedValue[3]{};
                    reorientedValue[0] =
                        reorientation.m[0][0] * interpolatedValue[0] +
                        reorientation.m[1][0] * interpolatedValue[1] +
                        reorientation.m[2][0] * interpolatedValue[2];
                    reorientedValue[1] =
                        reorientation.m[0][1] * interpolatedValue[0] +
                        reorientation.m[1][1] * interpolatedValue[1] +
                        reorientation.m[2][1] * interpolatedValue[2];
                    if (is3d)
                        reorientedValue[2] =
                        reorientation.m[0][2] * interpolatedValue[0] +
                        reorientation.m[1][2] * interpolatedValue[1] +
                        reorientation.m[2][2] * interpolatedValue[2];
                    *nodePtrX++ = reorientedValue[0] * static_cast<DataType>(weight);
                    *nodePtrY++ = reorientedValue[1] * static_cast<DataType>(weight);
                    if (is3d)
                        *nodePtrZ++ = reorientedValue[2] * static_cast<DataType>(weight);
                } // x
            } // y
        } // z
    }
};

TEST_CASE_METHOD(VoxelCentricToNodeCentricTest, "Voxel Centric to Node Centric", "[unit]") {
    // Loop over all generated test cases
    for (auto&& testCase : testCases) {
        // Retrieve test information
        auto&& [sectionName, transGrad, expTransGrad] = testCase;

        SECTION(sectionName) {
            NR_COUT << "\n**************** Section " << sectionName << " ****************" << std::endl;

            // Increase the precision for the output
            NR_COUT << std::fixed << std::setprecision(10);

            // Check the results
            const auto transGradPtr = transGrad.data();
            const auto expTransGradPtr = expTransGrad.data();
            for (size_t i = 0; i < expTransGrad.nVoxels(); ++i) {
                const float transGradVal = transGradPtr[i];
                const float expTransGradVal = expTransGradPtr[i];
                const float diff = abs(transGradVal - expTransGradVal);
                if (diff > 0) {
                    NR_COUT << "[i]=" << i;
                    NR_COUT << " | diff=" << diff;
                    NR_COUT << " | Result=" << transGradVal;
                    NR_COUT << " | Expected=" << expTransGradVal << std::endl;
                }
                REQUIRE(diff < EPS);
            }
        }
    }
}
