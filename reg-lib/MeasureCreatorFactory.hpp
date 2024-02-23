#pragma once

#include "MeasureCreator.hpp"

class MeasureCreatorFactory {
public:
    virtual MeasureCreator* Produce() { return new MeasureCreator(); }
};
