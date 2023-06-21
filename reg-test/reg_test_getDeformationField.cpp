// OpenCL is not supported for this test
#undef _USE_OPENCL

#include "reg_test_common.h"
#include <iomanip>

/*
    This test file contains the following unit tests:
    test functions: creation of a deformation field from a control point grid
    In 2D and 3D
    Cubic spline
*/


class GetDeformationFieldTest {
protected:
    using TestData = std::tuple<std::string, NiftiImage, NiftiImage>;
    using TestCase = std::tuple<shared_ptr<Platform>, unique_ptr<F3dContent>, TestData, bool, bool>;

    inline static vector<TestCase> testCases;

public:
    GetDeformationFieldTest() {
        if (!testCases.empty())
            return;

        // Create a random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> distr(0, 1);

        // Create a 2D reference image
        vector<NiftiImage::dim_t> dimFlo{ 4, 4 };
        NiftiImage reference2d(dimFlo, NIFTI_TYPE_FLOAT32);

        // Create a 3D reference image
        dimFlo.push_back(4);
        NiftiImage reference3d(dimFlo, NIFTI_TYPE_FLOAT32);

        // Generate the different test cases
        // Test 2D
        NiftiImage controlPointGrid2d = CreateControlPointGrid(reference2d);
        auto cpp2dPtr = controlPointGrid2d.data();
        for (size_t i = 0; i < controlPointGrid2d.nVoxels(); ++i)
            cpp2dPtr[i] = distr(gen);

        // Add the test data
        vector<TestData> testData;
        testData.emplace_back(TestData(
            "2D",
            std::move(reference2d),
            std::move(controlPointGrid2d)
        ));

        // Test 3D
        NiftiImage controlPointGrid3d = CreateControlPointGrid(reference3d);
        auto cpp3dPtr = controlPointGrid3d.data();
        for (size_t i = 0; i < controlPointGrid3d.nVoxels(); ++i)
            cpp3dPtr[i] = distr(gen);

        // Add the test data
        testData.emplace_back(TestData(
            "3D",
            std::move(reference3d),
            std::move(controlPointGrid3d)
        ));

        // Add platforms, composition, and bspline to the test data
        for (auto&& testData : testData) {
            for (auto&& platformType : PlatformTypes) {
                shared_ptr<Platform> platform{ new Platform(platformType) };
                unique_ptr<F3dContentCreator> contentCreator{ dynamic_cast<F3dContentCreator*>(platform->CreateContentCreator(ContentType::F3d)) };
                for (int composition = 0; composition < 2; composition++) {
                    if (platformType == PlatformType::Cuda && composition)
                        continue;   // CUDA platform does not support composition
                    for (int bspline = 0; bspline < 2; bspline++) {
                        // Make a copy of the test data
                        auto td = testData;
                        auto&& [testName, reference, controlPointGrid] = td;
                        // Add content
                        unique_ptr<F3dContent> content{ contentCreator->Create(reference, reference, controlPointGrid) };
                        testCases.push_back({ platform, std::move(content), std::move(td), composition, bspline });
                    }
                }
            }
        }
    }

    template<class DataType>
    void GetBSplineBasisValues(const DataType basis, DataType (&values)[4]) {
        const DataType ff = basis * basis;
        const DataType fff = ff * basis;
        const DataType mf = static_cast<DataType>(1.0 - basis);
        values[0] = static_cast<DataType>(mf * mf * mf / 6.0);
        values[1] = static_cast<DataType>((3.0 * fff - 6.0 * ff + 4.0) / 6.0);
        values[2] = static_cast<DataType>((-3.0 * fff + 3.0 * ff + 3.0 * basis + 1.0) / 6.0);
        values[3] = static_cast<DataType>(fff / 6.0);
    }

    template<class DataType>
    void GetSplineBasisValues(const DataType basis, DataType(&values)[4]) {
        const DataType ff = basis * basis;
        values[0] = static_cast<DataType>((basis * ((2.0 - basis) * basis - 1.0)) / 2.0);
        values[1] = static_cast<DataType>((ff * (3.0 * basis - 5.0) + 2.0) / 2.0);
        values[2] = static_cast<DataType>((basis * ((4.0 - 3.0 * basis) * basis + 1.0)) / 2.0);
        values[3] = static_cast<DataType>((basis - 1.0) * ff / 2.0);
    }

    void GetGridValues(const int& xPre, const int& yPre, const NiftiImage& controlPointGrid, float *xControlPointCoordinates, float *yControlPointCoordinates) {
        const auto cppPtr = controlPointGrid.data();
        const auto cppPtrX = cppPtr.begin();
        const auto cppPtrY = cppPtrX + NiftiImage::calcVoxelNumber(controlPointGrid, 2);
        size_t coord = 0;
        for (int y = yPre; y < yPre + 4; y++) {
            const bool in = -1 < y && y < controlPointGrid->ny;
            const size_t index = y * controlPointGrid->nx;
            for (int x = xPre; x < xPre + 4; x++) {
                if (in && -1 < x && x < controlPointGrid->nx) {
                    xControlPointCoordinates[coord] = cppPtrX[index + x];
                    yControlPointCoordinates[coord] = cppPtrY[index + x];
                } else {
                    xControlPointCoordinates[coord] = 0;
                    yControlPointCoordinates[coord] = 0;
                }
                coord++;
            }
        }
    }

    void GetGridValues(const int& xPre, const int& yPre, const int& zPre, const NiftiImage& controlPointGrid, float *xControlPointCoordinates, float *yControlPointCoordinates, float *zControlPointCoordinates) {
        const size_t cppVoxelNumber = NiftiImage::calcVoxelNumber(controlPointGrid, 3);
        const auto cppPtr = controlPointGrid.data();
        const auto cppPtrX = cppPtr.begin();
        const auto cppPtrY = cppPtrX + cppVoxelNumber;
        const auto cppPtrZ = cppPtrY + cppVoxelNumber;
        size_t coord = 0, yIndex, zIndex;
        for (int z = zPre; z < zPre + 4; z++) {
            bool in = true;
            if (-1 < z && z < controlPointGrid->nz)
                zIndex = z * controlPointGrid->nx * controlPointGrid->ny;
            else in = false;
            for (int y = yPre; y < yPre + 4; y++) {
                if (in && -1 < y && y < controlPointGrid->ny)
                    yIndex = y * controlPointGrid->nx;
                else in = false;
                for (int x = xPre; x < xPre + 4; x++) {
                    if (in && -1 < x && x < controlPointGrid->nx) {
                        xControlPointCoordinates[coord] = cppPtrX[zIndex + yIndex + x];
                        yControlPointCoordinates[coord] = cppPtrY[zIndex + yIndex + x];
                        zControlPointCoordinates[coord] = cppPtrZ[zIndex + yIndex + x];
                    } else {
                        xControlPointCoordinates[coord] = 0;
                        yControlPointCoordinates[coord] = 0;
                        zControlPointCoordinates[coord] = 0;
                    }
                    coord++;
                }
            }
        }
    }

    template<class DataType>
    void GetDeformationField(const NiftiImage& controlPointGrid, NiftiImage& defField, const int *mask, const bool& composition, const bool& bspline) {
        if (controlPointGrid->nz > 1)
            GetDeformationField3D<DataType>(controlPointGrid, defField, mask, composition, bspline);
        else
            GetDeformationField2D<DataType>(controlPointGrid, defField, mask, composition, bspline);
    }

    template<class DataType>
    void GetDeformationField2D(const NiftiImage& controlPointGrid, NiftiImage& defField, const int *mask, const bool& composition, const bool& bspline) {
        auto defFieldPtr = defField.data();
        auto defFieldPtrX = defFieldPtr.begin();
        auto defFieldPtrY = defFieldPtrX + NiftiImage::calcVoxelNumber(defField, 3);

        const DataType gridVoxelSpacing[2] = { controlPointGrid->dx / defField->dx, controlPointGrid->dy / defField->dy };
        DataType xBasis[4], yBasis[4], xyBasis[16], xControlPointCoordinates[16], yControlPointCoordinates[16];
        int oldXPre = -1, oldYPre = -1;

        if (composition) {  // Composition of deformation fields
            // Read the ijk sform or qform, as appropriate
            const mat44 *realToVoxel = controlPointGrid->sform_code > 0 ? &controlPointGrid->sto_ijk : &controlPointGrid->qto_ijk;

            for (int y = 0; y < defField->ny; y++) {
                size_t index = y * defField->nx;
                for (int x = 0; x < defField->nx; x++) {
                    // The previous position at the current pixel position is read
                    DataType xReal = defFieldPtrX[index];
                    DataType yReal = defFieldPtrY[index];

                    // From real to pixel position in the CPP
                    const DataType xVoxel = realToVoxel->m[0][0] * xReal + realToVoxel->m[0][1] * yReal + realToVoxel->m[0][3];
                    const DataType yVoxel = realToVoxel->m[1][0] * xReal + realToVoxel->m[1][1] * yReal + realToVoxel->m[1][3];

                    // The spline coefficients are computed
                    int xPre = reg_floor(xVoxel);
                    DataType basis = xVoxel - (DataType)xPre--;
                    if (basis < 0) basis = 0; // rounding error
                    if (bspline) GetBSplineBasisValues<DataType>(basis, xBasis);
                    else GetSplineBasisValues<DataType>(basis, xBasis);

                    int yPre = reg_floor(yVoxel);
                    basis = yVoxel - (DataType)yPre--;
                    if (basis < 0) basis = 0; // rounding error
                    if (bspline) GetBSplineBasisValues<DataType>(basis, yBasis);
                    else GetSplineBasisValues<DataType>(basis, yBasis);

                    if (xVoxel >= 0 && xVoxel <= defField->nx - 1 &&
                        yVoxel >= 0 && yVoxel <= defField->ny - 1) {
                        // The control point positions are extracted
                        if (oldXPre != xPre || oldYPre != yPre) {
                            GetGridValues(xPre, yPre, controlPointGrid, xControlPointCoordinates, yControlPointCoordinates);
                            oldXPre = xPre;
                            oldYPre = yPre;
                        }

                        xReal = 0; yReal = 0;
                        if (mask[index] > -1) {
                            for (int b = 0; b < 4; b++) {
                                for (int a = 0; a < 4; a++) {
                                    const DataType xyBasis = xBasis[a] * yBasis[b];
                                    xReal += xControlPointCoordinates[b * 4 + a] * xyBasis;
                                    yReal += yControlPointCoordinates[b * 4 + a] * xyBasis;
                                }
                            }
                        }

                        defFieldPtrX[index] = xReal;
                        defFieldPtrY[index] = yReal;
                    }
                    index++;
                }
            }
        } else {    // If the deformation field is blank - !composition
            for (int y = 0; y < defField->ny; y++) {
                size_t index = y * defField->nx;

                int yPre = (int)((DataType)y / gridVoxelSpacing[1]);
                DataType basis = (DataType)y / gridVoxelSpacing[1] - (DataType)yPre;
                if (basis < 0) basis = 0; // rounding error
                if (bspline) GetBSplineBasisValues<DataType>(basis, yBasis);
                else GetSplineBasisValues<DataType>(basis, yBasis);

                for (int x = 0; x < defField->nx; x++) {
                    int xPre = (int)((DataType)x / gridVoxelSpacing[0]);
                    basis = (DataType)x / gridVoxelSpacing[0] - (DataType)xPre;
                    if (basis < 0) basis = 0; // rounding error
                    if (bspline) GetBSplineBasisValues<DataType>(basis, xBasis);
                    else GetSplineBasisValues<DataType>(basis, xBasis);

                    size_t coord = 0;
                    for (int a = 0; a < 4; a++) {
                        xyBasis[coord++] = xBasis[0] * yBasis[a];
                        xyBasis[coord++] = xBasis[1] * yBasis[a];
                        xyBasis[coord++] = xBasis[2] * yBasis[a];
                        xyBasis[coord++] = xBasis[3] * yBasis[a];
                    }

                    if (oldXPre != xPre || oldYPre != yPre) {
                        GetGridValues(xPre, yPre, controlPointGrid, xControlPointCoordinates, yControlPointCoordinates);
                        oldXPre = xPre;
                        oldYPre = yPre;
                    }

                    DataType xReal = 0, yReal = 0;
                    if (mask[index] > -1) {
                        for (int a = 0; a < 16; a++) {
                            xReal += xControlPointCoordinates[a] * xyBasis[a];
                            yReal += yControlPointCoordinates[a] * xyBasis[a];
                        }
                    }
                    defFieldPtrX[index] = xReal;
                    defFieldPtrY[index] = yReal;
                    index++;
                }
            }
        }
    }

    template<class DataType>
    void GetDeformationField3D(const NiftiImage& controlPointGrid, NiftiImage& defField, const int *mask, const bool& composition, const bool& bspline) {
        DataType xBasis[4], yBasis[4], zBasis[4];
        DataType xControlPointCoordinates[64];
        DataType yControlPointCoordinates[64];
        DataType zControlPointCoordinates[64];

        const size_t defFieldVoxelNumber = NiftiImage::calcVoxelNumber(defField, 3);
        auto defFieldPtr = defField.data();
        auto defFieldPtrX = defFieldPtr.begin();
        auto defFieldPtrY = defFieldPtrX + defFieldVoxelNumber;
        auto defFieldPtrZ = defFieldPtrY + defFieldVoxelNumber;

        if (composition) {  // Composition of deformation fields
            // Read the ijk sform or qform, as appropriate
            const mat44 *realToVoxel = controlPointGrid->sform_code > 0 ? &controlPointGrid->sto_ijk : &controlPointGrid->qto_ijk;
            for (int z = 0; z < defField->nz; z++) {
                size_t index = z * defField->nx * defField->ny;
                int oldPreX = -99; int oldPreY = -99; int oldPreZ = -99;
                for (int y = 0; y < defField->ny; y++) {
                    for (int x = 0; x < defField->nx; x++) {
                        if (mask[index] > -1) {
                            // The previous position at the current pixel position is read
                            DataType real[] = { defFieldPtrX[index], defFieldPtrY[index], defFieldPtrZ[index] };

                            // From real to pixel position in the control point space
                            DataType voxel[3];
                            voxel[0] =
                                realToVoxel->m[0][0] * real[0] +
                                realToVoxel->m[0][1] * real[1] +
                                realToVoxel->m[0][2] * real[2] +
                                realToVoxel->m[0][3];
                            voxel[1] =
                                realToVoxel->m[1][0] * real[0] +
                                realToVoxel->m[1][1] * real[1] +
                                realToVoxel->m[1][2] * real[2] +
                                realToVoxel->m[1][3];
                            voxel[2] =
                                realToVoxel->m[2][0] * real[0] +
                                realToVoxel->m[2][1] * real[1] +
                                realToVoxel->m[2][2] * real[2] +
                                realToVoxel->m[2][3];

                            // The spline coefficients are computed
                            int xPre = reg_floor(voxel[0]);
                            DataType basis = voxel[0] - (DataType)xPre--;
                            if (basis < 0) basis = 0; // rounding error
                            if (bspline) GetBSplineBasisValues<DataType>(basis, xBasis);
                            else GetSplineBasisValues<DataType>(basis, xBasis);

                            int yPre = reg_floor(voxel[1]);
                            basis = voxel[1] - (DataType)yPre--;
                            if (basis < 0) basis = 0; // rounding error
                            if (bspline) GetBSplineBasisValues<DataType>(basis, yBasis);
                            else GetSplineBasisValues<DataType>(basis, yBasis);

                            int zPre = reg_floor(voxel[2]);
                            basis = voxel[2] - (DataType)zPre--;
                            if (basis < 0) basis = 0; // rounding error
                            if (bspline) GetBSplineBasisValues<DataType>(basis, zBasis);
                            else GetSplineBasisValues<DataType>(basis, zBasis);

                            // The control point positions are extracted
                            if (xPre != oldPreX || yPre != oldPreY || zPre != oldPreZ) {
                                GetGridValues(xPre, yPre, zPre, controlPointGrid, xControlPointCoordinates, yControlPointCoordinates, zControlPointCoordinates);
                                oldPreX = xPre;
                                oldPreY = yPre;
                                oldPreZ = zPre;
                            }

                            real[0] = real[1] = real[2] = 0;
                            int coord = 0;
                            for (int c = 0; c < 4; c++) {
                                for (int b = 0; b < 4; b++) {
                                    for (int a = 0; a < 4; a++) {
                                        DataType tempValue = xBasis[a] * yBasis[b] * zBasis[c];
                                        real[0] += xControlPointCoordinates[coord] * tempValue;
                                        real[1] += yControlPointCoordinates[coord] * tempValue;
                                        real[2] += zControlPointCoordinates[coord] * tempValue;
                                        coord++;
                                    }
                                }
                            }
                            defFieldPtrX[index] = real[0];
                            defFieldPtrY[index] = real[1];
                            defFieldPtrZ[index] = real[2];
                        }
                        index++;
                    }
                }
            }
        } else {    // If the deformation field is blank - !composition
            const DataType gridVoxelSpacing[3] = {
                controlPointGrid->dx / defField->dx,
                controlPointGrid->dy / defField->dy,
                controlPointGrid->dz / defField->dz
            };

            for (int z = 0; z < defField->nz; z++) {
                size_t index = z * defField->nx * defField->ny;
                DataType oldBasis = DataType(1.1);

                int zPre = int(DataType(z) / gridVoxelSpacing[2]);
                DataType basis = (DataType)z / gridVoxelSpacing[2] - (DataType)zPre;
                if (basis < 0) basis = 0; // rounding error
                if (bspline) GetBSplineBasisValues<DataType>(basis, zBasis);
                else GetSplineBasisValues<DataType>(basis, zBasis);

                for (int y = 0; y < defField->ny; y++) {
                    int yPre = int(DataType(y) / gridVoxelSpacing[1]);
                    basis = (DataType)y / gridVoxelSpacing[1] - (DataType)yPre;
                    if (basis < 0) basis = 0; // rounding error
                    if (bspline) GetBSplineBasisValues<DataType>(basis, yBasis);
                    else GetSplineBasisValues<DataType>(basis, yBasis);
                    int coord = 0;
                    DataType yzBasis[16];
                    for (int a = 0; a < 4; a++) {
                        yzBasis[coord++] = yBasis[0] * zBasis[a];
                        yzBasis[coord++] = yBasis[1] * zBasis[a];
                        yzBasis[coord++] = yBasis[2] * zBasis[a];
                        yzBasis[coord++] = yBasis[3] * zBasis[a];
                    }

                    for (int x = 0; x < defField->nx; x++) {
                        int xPre = int(DataType(x) / gridVoxelSpacing[0]);
                        basis = (DataType)x / gridVoxelSpacing[0] - (DataType)xPre;
                        if (basis < 0) basis = 0; // rounding error
                        if (bspline) GetBSplineBasisValues<DataType>(basis, xBasis);
                        else GetSplineBasisValues<DataType>(basis, xBasis);
                        coord = 0;
                        DataType xyzBasis[64];
                        for (int a = 0; a < 16; a++) {
                            xyzBasis[coord++] = xBasis[0] * yzBasis[a];
                            xyzBasis[coord++] = xBasis[1] * yzBasis[a];
                            xyzBasis[coord++] = xBasis[2] * yzBasis[a];
                            xyzBasis[coord++] = xBasis[3] * yzBasis[a];
                        }
                        if (basis <= oldBasis || x == 0)
                            GetGridValues(xPre, yPre, zPre, controlPointGrid, xControlPointCoordinates, yControlPointCoordinates, zControlPointCoordinates);
                        oldBasis = basis;

                        DataType real[3]{};
                        if (mask[index] > -1) {
                            for (int a = 0; a < 64; a++) {
                                real[0] += xControlPointCoordinates[a] * xyzBasis[a];
                                real[1] += yControlPointCoordinates[a] * xyzBasis[a];
                                real[2] += zControlPointCoordinates[a] * xyzBasis[a];
                            }
                        }// mask
                        defFieldPtrX[index] = real[0];
                        defFieldPtrY[index] = real[1];
                        defFieldPtrZ[index] = real[2];
                        index++;
                    } // x
                } // y
            } // z
        } // composition
    }
};

TEST_CASE_METHOD(GetDeformationFieldTest, "Get deformation field", "[GetDeformationField]") {
    // Loop over all generated test cases
    for (auto&& testCase : testCases) {
        // Retrieve test information
        auto&& [platform, content, testData, composition, bspline] = testCase;
        auto&& [testName, reference, controlPointGrid] = testData;
        const std::string sectionName = testName + " " + platform->GetName() + " composition=" + std::to_string(composition) + " bspline=" + std::to_string(bspline);

        SECTION(sectionName) {
            std::cout << "\n**************** Section " << sectionName << " ****************" << std::endl;

            // Compute the deformation field
            unique_ptr<Compute> compute{ platform->CreateCompute(*content) };
            compute->GetDeformationField(composition, bspline);
            NiftiImage defFieldExp(content->GetDeformationField(), NiftiImage::Copy::ImageInfoAndAllocData);
            GetDeformationField<float>(controlPointGrid, defFieldExp, content->GetReferenceMask(), composition, bspline);

            // Check the results
            NiftiImage defField = content->GetDeformationField();
            const auto defFieldPtr = defField.data();
            const auto defFieldExpPtr = defFieldExp.data();
            defField.disown();
            // Increase the precision for the output
            std::cout << std::fixed << std::setprecision(10);
            for (size_t i = 0; i < defFieldExp.nVoxels(); ++i) {
                const double defFieldVal = defFieldPtr[i];
                const double defFieldExpVal = defFieldExpPtr[i];
                std::cout << i << " " << defFieldVal << " " << defFieldExpVal << std::endl;
                REQUIRE(fabs(defFieldVal - defFieldExpVal) < EPS);
            }
            // Ensure the termination of content before CudaContext
            content.reset();
        }
    }
}