#pragma once

#include "Measure.h"

class MeasureFactory {
public:
    virtual Measure* Produce() { return new Measure(); }
};
