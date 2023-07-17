#pragma once

#include "CudaMeasure.h"

class CudaMeasureFactory: public MeasureFactory {
public:
    virtual Measure* Produce() override { return new CudaMeasure(); }
};
