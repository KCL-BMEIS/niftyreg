#pragma once

#include "CudaMeasureCreator.hpp"

class CudaMeasureCreatorFactory: public MeasureCreatorFactory {
public:
    virtual MeasureCreator* Produce() override { return new CudaMeasureCreator(); }
};
