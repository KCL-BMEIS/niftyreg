#pragma once

#include "MeasureCreator.hpp"

class MeasureCreatorFactory {
public:
    virtual ~MeasureCreatorFactory() = default;
    virtual MeasureCreator* Produce() { return new MeasureCreator(); }
};
