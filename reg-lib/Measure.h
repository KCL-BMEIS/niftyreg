#pragma once

#include "F3dContent.h"
#include "_reg_measure.h"

enum class MeasureType { Nmi, Ssd, Dti, Lncc, Kld, Mind, Mindssc };

class Measure {
public:
    virtual reg_measure* Create(const MeasureType& measureType);
    virtual void Initialise(reg_measure& measure, F3dContent& con, F3dContent *conBw = nullptr);
};
