#include "Measure.h"
#include "_reg_nmi.h"
#include "_reg_ssd.h"
#include "_reg_dti.h"
#include "_reg_lncc.h"
#include "_reg_kld.h"
#include "_reg_mind.h"

/* *************************************************************** */
reg_measure* Measure::Create(const MeasureType measureType) {
    switch (measureType) {
    case MeasureType::Nmi:
        return new reg_nmi();
    case MeasureType::Ssd:
        return new reg_ssd();
    case MeasureType::Dti:
        return new reg_dti();
    case MeasureType::Lncc:
        return new reg_lncc();
    case MeasureType::Kld:
        return new reg_kld();
    case MeasureType::Mind:
        return new reg_mind();
    case MeasureType::MindSsc:
        return new reg_mindssc();
    default:
        NR_FATAL_ERROR("Unsupported measure type");
        return nullptr;
    }
}
/* *************************************************************** */
void Measure::Initialise(reg_measure& measure, DefContent& con, DefContent *conBw) {
    measure.InitialiseMeasure(con.GetReference(),
                              con.GetFloating(),
                              con.GetReferenceMask(),
                              con.GetWarped(),
                              con.GetWarpedGradient(),
                              con.GetVoxelBasedMeasureGradient(),
                              con.GetLocalWeightSim(),
                              conBw ? conBw->GetReferenceMask() : nullptr,
                              conBw ? conBw->GetWarped() : nullptr,
                              conBw ? conBw->GetWarpedGradient() : nullptr,
                              conBw ? conBw->GetVoxelBasedMeasureGradient() : nullptr);
}
/* *************************************************************** */
