#ifndef CLBLOCKMATCHINGKERNEL_H
#define CLBLOCKMATCHINGKERNEL_H

#include "BlockMatchingKernel.h"
#include "CLAladinContent.h"

class CLBlockMatchingKernel : public BlockMatchingKernel {
    public:
       CLBlockMatchingKernel(AladinContent * conIn, std::string name);
       ~CLBlockMatchingKernel();
       void calculate();

    private:
       CLContextSingletton *sContext;
       ClAladinContent *con;
       nifti_image *reference;
       _reg_blockMatchingParam *params;
       cl_kernel kernel;
       cl_context clContext;
       cl_program program;
       cl_command_queue commandQueue;
       cl_mem clTotalBlock;
       cl_mem clReferenceImageArray;
       cl_mem clWarpedImageArray;
       cl_mem clReferencePosition;
       cl_mem clWarpedPosition;
       cl_mem clMask;
       cl_mem clReferenceMat;
};

#endif // CLBLOCKMATCHINGKERNEL_H
