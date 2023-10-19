#pragma once

#include "F3d2ContentCreator.h"
#include "CudaF3dContent.h"

class CudaF3d2ContentCreator: public F3d2ContentCreator {
public:
    virtual std::pair<F3dContent*, F3dContent*> Create(nifti_image *reference,
                                                       nifti_image *floating,
                                                       nifti_image *controlPointGrid,
                                                       nifti_image *controlPointGridBw,
                                                       nifti_image *localWeightSim = nullptr,
                                                       int *referenceMask = nullptr,
                                                       int *floatingMask = nullptr,
                                                       mat44 *transformationMatrix = nullptr,
                                                       mat44 *transformationMatrixBw = nullptr,
                                                       size_t bytes = sizeof(float)) override {
        auto con = new CudaF3dContent(reference, floating, controlPointGrid, localWeightSim, referenceMask, transformationMatrix, bytes);
        auto conBw = new CudaF3dContent(floating, reference, controlPointGridBw, nullptr, floatingMask, transformationMatrixBw, bytes);
        conBw->SetReferenceCuda(con->GetFloatingCuda());
        conBw->SetFloatingCuda(con->GetReferenceCuda());
        return { con, conBw };
    }
};
