#ifndef RESAMPLEIMAGEKERNEL_H
#define RESAMPLEIMAGEKERNEL_H

#include "Kernel.h"
#include "nifti1_io.h"

class ResampleImageKernel : public Kernel {
public:
    static std::string getName() {
        return "ResampleImageKernel";
    }
    ResampleImageKernel( std::string name) : Kernel(name) {
    }

    virtual ~ResampleImageKernel(){}

    virtual void calculate(int interp, float paddingValue, bool *dti_timepoint = NULL, mat33 * jacMat = NULL) = 0;
};

#endif // RESAMPLEIMAGEKERNEL_H
