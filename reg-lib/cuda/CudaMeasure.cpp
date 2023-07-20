#include "CudaMeasure.h"
#include "CudaF3dContent.h"
#include "_reg_nmi_gpu.h"
#include "_reg_ssd_gpu.h"

/* *************************************************************** */
reg_measure* CudaMeasure::Create(const MeasureType& measureType) {
    switch (measureType) {
    case MeasureType::Nmi:
        return new reg_nmi_gpu();
    case MeasureType::Ssd:
        return new reg_ssd_gpu();
    case MeasureType::Dti:
        return new reg_dti_gpu();
    case MeasureType::Lncc:
        return new reg_lncc_gpu();
    case MeasureType::Kld:
        return new reg_kld_gpu();
    case MeasureType::Mind:
        reg_print_msg_error("MIND measure type isn't implemented for GPU");
        reg_exit();
    case MeasureType::Mindssc:
        reg_print_msg_error("MIND-SSC measure type isn't implemented for GPU");
        reg_exit();
    }
    reg_print_msg_error("Unsupported measure type");
    reg_exit();
    return nullptr;
}
/* *************************************************************** */
void CudaMeasure::Initialise(reg_measure& measure, F3dContent& con, F3dContent *conBw) {
    // TODO Implement symmetric scheme for CUDA measure types
    reg_measure_gpu& measureGpu = dynamic_cast<reg_measure_gpu&>(measure);
    CudaF3dContent& cudaCon = dynamic_cast<CudaF3dContent&>(con);
    CudaF3dContent *cudaConBw = dynamic_cast<CudaF3dContent*>(conBw);
    measureGpu.InitialiseMeasure(cudaCon.Content::GetReference(),
                                 cudaCon.GetReferenceCuda(),
                                 cudaCon.Content::GetFloating(),
                                 cudaCon.GetFloatingCuda(),
                                 cudaCon.Content::GetReferenceMask(),
                                 cudaCon.GetReferenceMaskCuda(),
                                 cudaCon.GetActiveVoxelNumber(),
                                 cudaCon.Content::GetWarped(),
                                 cudaCon.GetWarpedCuda(),
                                 cudaCon.F3dContent::GetWarpedGradient(),
                                 cudaCon.GetWarpedGradientCuda(),
                                 cudaCon.F3dContent::GetVoxelBasedMeasureGradient(),
                                 cudaCon.GetVoxelBasedMeasureGradientCuda(),
                                 cudaCon.F3dContent::GetLocalWeightSim(),
                                 cudaConBw ? cudaConBw->Content::GetReferenceMask() : nullptr,
                                 cudaConBw ? cudaConBw->GetReferenceMaskCuda() : nullptr,
                                 cudaConBw ? cudaConBw->Content::GetWarped() : nullptr,
                                 cudaConBw ? cudaConBw->GetWarpedCuda() : nullptr,
                                 cudaConBw ? cudaConBw->F3dContent::GetWarpedGradient() : nullptr,
                                 cudaConBw ? cudaConBw->GetWarpedGradientCuda() : nullptr,
                                 cudaConBw ? cudaConBw->F3dContent::GetVoxelBasedMeasureGradient() : nullptr,
                                 cudaConBw ? cudaConBw->GetVoxelBasedMeasureGradientCuda() : nullptr);
}
/* *************************************************************** */
