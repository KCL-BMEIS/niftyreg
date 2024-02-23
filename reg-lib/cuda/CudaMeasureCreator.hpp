#pragma once

#include "MeasureCreator.hpp"

class CudaMeasureCreator: public MeasureCreator {
public:
    virtual reg_measure* Create(const MeasureType measureType) override;
    virtual void Initialise(reg_measure& measure, DefContent& con, DefContent *conBw = nullptr) override;
};
