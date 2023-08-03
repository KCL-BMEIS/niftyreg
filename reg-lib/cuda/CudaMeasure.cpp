#include "CudaMeasure.h"
#include "CudaDefContent.h"
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
    case MeasureType::MindSsc:
        reg_print_msg_error("MIND-SSC measure type isn't implemented for GPU");
        reg_exit();
    }
    reg_print_fct_error("CudaMeasure::Create");
    reg_print_msg_error("Unsupported measure type");
    reg_exit();
    return nullptr;
}
/* *************************************************************** */
void CudaMeasure::Initialise(reg_measure& measure, DefContent& con, DefContent *conBw) {
    reg_measure_gpu& measureGpu = dynamic_cast<reg_measure_gpu&>(measure);
    CudaDefContent& cudaCon = dynamic_cast<CudaDefContent&>(con);
    CudaDefContent *cudaConBw = dynamic_cast<CudaDefContent*>(conBw);
    measureGpu.InitialiseMeasure(cudaCon.Content::GetReference(),
                                 cudaCon.GetReferenceCuda(),
                                 cudaCon.Content::GetFloating(),
                                 cudaCon.GetFloatingCuda(),
                                 cudaCon.Content::GetReferenceMask(),
                                 cudaCon.GetReferenceMaskCuda(),
                                 cudaCon.GetActiveVoxelNumber(),
                                 cudaCon.Content::GetWarped(),
                                 cudaCon.GetWarpedCuda(),
                                 cudaCon.DefContent::GetWarpedGradient(),
                                 cudaCon.GetWarpedGradientCuda(),
                                 cudaCon.DefContent::GetVoxelBasedMeasureGradient(),
                                 cudaCon.GetVoxelBasedMeasureGradientCuda(),
                                 cudaCon.DefContent::GetLocalWeightSim(),
                                 cudaConBw ? cudaConBw->Content::GetReferenceMask() : nullptr,
                                 cudaConBw ? cudaConBw->GetReferenceMaskCuda() : nullptr,
                                 cudaConBw ? cudaConBw->Content::GetWarped() : nullptr,
                                 cudaConBw ? cudaConBw->GetWarpedCuda() : nullptr,
                                 cudaConBw ? cudaConBw->DefContent::GetWarpedGradient() : nullptr,
                                 cudaConBw ? cudaConBw->GetWarpedGradientCuda() : nullptr,
                                 cudaConBw ? cudaConBw->DefContent::GetVoxelBasedMeasureGradient() : nullptr,
                                 cudaConBw ? cudaConBw->GetVoxelBasedMeasureGradientCuda() : nullptr);
}
/* *************************************************************** */
