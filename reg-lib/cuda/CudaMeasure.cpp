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
void CudaMeasure::Initialise(reg_measure& measure, F3dContent& con) {
    reg_measure_gpu *measureGpu = dynamic_cast<reg_measure_gpu*>(&measure);
    CudaF3dContent *cudaCon = dynamic_cast<CudaF3dContent*>(&con);
    measureGpu->InitialiseMeasure(cudaCon->Content::GetReference(),
                                  cudaCon->Content::GetFloating(),
                                  cudaCon->Content::GetReferenceMask(),
                                  cudaCon->Content::GetReference()->nvox,
                                  cudaCon->Content::GetWarped(),
                                  cudaCon->F3dContent::GetWarpedGradient(),
                                  cudaCon->F3dContent::GetVoxelBasedMeasureGradient(),
                                  cudaCon->F3dContent::GetLocalWeightSim(),
                                  cudaCon->GetReferenceCuda(),
                                  cudaCon->GetFloatingCuda(),
                                  cudaCon->GetReferenceMaskCuda(),
                                  cudaCon->GetWarpedCuda(),
                                  cudaCon->GetWarpedGradientCuda(),
                                  cudaCon->GetVoxelBasedMeasureGradientCuda());
}
/* *************************************************************** */
