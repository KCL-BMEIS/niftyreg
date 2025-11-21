#pragma once

#include "DefContent.h"
#include "_reg_measure.h"

enum class MeasureType { Nmi, Ssd, Dti, Lncc, Kld, Mind, MindSsc };

class MeasureCreator {
public:
    virtual reg_measure* Create(const MeasureType measureType);
    virtual void Initialise(reg_measure& measure, DefContent& con, DefContent *conBw = nullptr);
};
