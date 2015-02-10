#include <iostream>
#include "nifti1_io.h"

#include "Content.h"
#include "CLContextSingletton.h"
#include "CLContent.h"
#include "config.h"

#include "CLKernels.h"
#include "_reg_tools.h"
#include <cstring>

//debugging
#include "_reg_resampling.h"
#include "_reg_globalTransformation.h"
#include "_reg_blockMatching.h"
//---

#define SIZE 128
#define BLOCK_SIZE 64

unsigned int min_cl(unsigned int a, unsigned int b)
{
	return (a < b) ? a : b;
}
/* *************************************************************** */
CLConvolutionKernel::CLConvolutionKernel(std::string name) :
		ConvolutionKernel(name) {
	sContext = &CLContextSingletton::Instance();
}
/* *************************************************************** */
void CLConvolutionKernel::calculate(nifti_image *image, float *sigma, int kernelType, int *mask, bool *timePoints, bool *axis)
{
	//cpu atm
	reg_tools_kernelConvolution(image, sigma, kernelType, mask, timePoints, axis);
}
/* *************************************************************** */
CLConvolutionKernel::~CLConvolutionKernel() {}
/* *************************************************************** */
/* *************************************************************** */
CLAffineDeformationFieldKernel::CLAffineDeformationFieldKernel(Content *conIn, std::string nameIn) :
		AffineDeformationFieldKernel(nameIn)
{
	//populate the CLContent object ptr
	con = static_cast<ClContent*>(conIn);

	//path to kernel files
	std::string clInstallPath(CL_KERNELS_PATH);
	std::string clKernel("affineDeformationKernel.cl");

	//get opencl context params
	sContext = &CLContextSingletton::Instance();
	clContext = sContext->getContext();
	commandQueue = sContext->getCommandQueue();
	program = sContext->CreateProgram((clInstallPath + clKernel).c_str());

	cl_int errNum;

	// Create OpenCL kernel
	kernel = clCreateKernel(program, "affineKernel", &errNum);
	sContext->checkErrNum(errNum, "Error setting kernel CLAffineDeformationFieldKernel.");

	//get cpu ptrs
	this->deformationFieldImage = con->Content::getCurrentDeformationField();
	this->affineTransformation = con->Content::getTransformationMatrix();
	this->ReferenceMatrix = (this->deformationFieldImage->sform_code > 0) ? &(this->deformationFieldImage->sto_xyz) : &(this->deformationFieldImage->qto_xyz);

	//get cl ptrs
	clDeformationField = con->getDeformationFieldArrayClmem();
	clMask = con->getMaskClmem();

	//set some final kernel args
	errNum = clSetKernelArg(this->kernel, 2, sizeof(cl_mem), &this->clMask);
	sContext->checkErrNum(errNum, "Error setting clMask.");

}
/* *************************************************************** */
void CLAffineDeformationFieldKernel::calculate(bool compose)
{
	const unsigned int xThreads = 8;
	const unsigned int yThreads = 8;
	const unsigned int zThreads = 8;

	const unsigned int xBlocks = ((this->deformationFieldImage->nx % xThreads) == 0) ? (this->deformationFieldImage->nx / xThreads) : (this->deformationFieldImage->nx / xThreads) + 1;
	const unsigned int yBlocks = ((this->deformationFieldImage->ny % yThreads) == 0) ? (this->deformationFieldImage->ny / yThreads) : (this->deformationFieldImage->ny / yThreads) + 1;
	const unsigned int zBlocks = ((this->deformationFieldImage->nz % zThreads) == 0) ? (this->deformationFieldImage->nz / zThreads) : (this->deformationFieldImage->nz / zThreads) + 1;
	const cl_uint dims = 3;
	const size_t globalWorkSize[dims] = { xBlocks * xThreads, yBlocks * yThreads, zBlocks * zThreads };
	const size_t localWorkSize[dims] = { xThreads, yThreads, zThreads };

	mat44 transformationMatrix = (compose == true) ? *this->affineTransformation : reg_mat44_mul(this->affineTransformation, ReferenceMatrix);

	float* trans = (float *) malloc(16 * sizeof(float));
	mat44ToCptr(transformationMatrix, trans);

	cl_int errNum;

	cl_uint3 pms_d = {{ (cl_uint)this->deformationFieldImage->nx,
							  (cl_uint)this->deformationFieldImage->ny,
							  (cl_uint)this->deformationFieldImage->nz,
							  (cl_uint)0 }};

	cl_mem cltransMat = clCreateBuffer(this->clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * 16, trans, &errNum);
	this->sContext->checkErrNum(errNum, "CLAffineDeformationFieldKernel::calculate failed to allocate memory (cltransMat): ");

	cl_uint composition = compose;
	errNum = clSetKernelArg(this->kernel, 0, sizeof(cl_mem), &cltransMat);
	sContext->checkErrNum(errNum, "Error setting cltransMat.");
	errNum |= clSetKernelArg(this->kernel, 1, sizeof(cl_mem), &this->clDeformationField);
	sContext->checkErrNum(errNum, "Error setting clDeformationField.");
	errNum |= clSetKernelArg(this->kernel, 3, sizeof(cl_uint3), &pms_d);
	sContext->checkErrNum(errNum, "Error setting kernel arguments.");
	errNum |= clSetKernelArg(this->kernel, 4, sizeof(cl_uint), &composition);
	sContext->checkErrNum(errNum, "Error setting kernel arguments.");

	errNum = clEnqueueNDRangeKernel(this->commandQueue, kernel, dims, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	sContext->checkErrNum(errNum, "Error queuing CLAffineDeformationFieldKernel for execution");
	clFinish(commandQueue);

	free(trans);
	clReleaseMemObject(cltransMat);
	return;
}
/* *************************************************************** */
CLAffineDeformationFieldKernel::~CLAffineDeformationFieldKernel()
{
	if (kernel != 0)
		clReleaseKernel(kernel);
	if (program != 0)
		clReleaseProgram(program);
}
/* *************************************************************** */
/* *************************************************************** */
CLResampleImageKernel::CLResampleImageKernel(Content *conIn, std::string name) :
		ResampleImageKernel(name)
{
	//populate the CLContext object ptr
	con = static_cast<ClContent*>(conIn);

	//path to kernel file
	std::string clInstallPath(CL_KERNELS_PATH);
	std::string clKernel("resampleKernel.cl");

	//get opencl context params
	sContext = &CLContextSingletton::Instance();
	clContext = sContext->getContext();
	commandQueue = sContext->getCommandQueue();
	program = sContext->CreateProgram((clInstallPath + clKernel).c_str());

	//get cpu ptrs
	floatingImage = con->Content::getCurrentFloating();
	warpedImage = con->Content::getCurrentWarped();
	mask = con->Content::getCurrentReferenceMask();

	//get cl ptrs
	clCurrentFloating = con->getFloatingImageArrayClmem();
	clCurrentDeformationField = con->getDeformationFieldArrayClmem();
	clCurrentWarped = con->getWarpedImageClmem();
	clMask = con->getMaskClmem();
	floMat = con->getFloMatClmem();

	//init kernel
	kernel = 0;
}
/* *************************************************************** */
void CLResampleImageKernel::calculate(int interp,
												  float paddingValue,
												  bool *dti_timepoint,
												  mat33 *jacMat)
{
	cl_int errNum;
	// Define the DTI indices if required
	if(dti_timepoint!=NULL || jacMat!=NULL){
		reg_print_fct_error("CLResampleImageKernel::calculate");
		reg_print_msg_error("The DTI resampling has not yet been implemented with the OpenCL platform. Exit.");
		reg_exit(1);
	}

	kernel = clCreateKernel(program, "ResampleImage3D", &errNum);
	sContext->checkErrNum(errNum, "Error setting kernel ResampleImage3D.");

	long targetVoxelNumber = (long) warpedImage->nx * warpedImage->ny * warpedImage->nz;
	const unsigned int maxThreads = sContext->getMaxThreads();
	const unsigned int maxBlocks = sContext->getMaxBlocks();

	unsigned int blocks = (targetVoxelNumber % maxThreads) ? (targetVoxelNumber / maxThreads) + 1 : targetVoxelNumber / maxThreads;
	blocks = min_cl(blocks, maxBlocks);

	const cl_uint dims = 1;
	const size_t globalWorkSize[dims] = { blocks * maxThreads };
	const size_t localWorkSize[dims] = { maxThreads };

	int numMats = 0; //needs to be a parameter
	float* jacMat_h = (float*) malloc(9 * numMats * sizeof(float));

	cl_long2 voxelNumber = {{ (cl_long)warpedImage->nx * warpedImage->ny * warpedImage->nz,
									  (cl_long)floatingImage->nx * floatingImage->ny * floatingImage->nz }};
	cl_uint3 fi_xyz = {{ (cl_uint)floatingImage->nx,
								(cl_uint)floatingImage->ny,
								(cl_uint)floatingImage->nz }};
	cl_uint2 wi_tu = {{ (cl_uint)warpedImage->nt,
							  (cl_uint)warpedImage->nu }};

	if (numMats)
		mat33ToCptr(jacMat, jacMat_h, numMats);
	int datatype = con->getFloatingDatatype();

	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &this->clCurrentFloating);
	sContext->checkErrNum(errNum, "Error setting interp kernel arguments 0.");
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &this->clCurrentDeformationField);
	sContext->checkErrNum(errNum, "Error setting interp kernel arguments 1.");
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &this->clCurrentWarped);
	sContext->checkErrNum(errNum, "Error setting interp kernel arguments 2.");
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &this->clMask);
	sContext->checkErrNum(errNum, "Error setting interp kernel arguments 3.");
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &this->floMat);
	sContext->checkErrNum(errNum, "Error setting interp kernel arguments 4.");
	errNum |= clSetKernelArg(kernel, 5, sizeof(cl_long2), &voxelNumber);
	sContext->checkErrNum(errNum, "Error setting interp kernel arguments 5.");
	errNum |= clSetKernelArg(kernel, 6, sizeof(cl_uint3), &fi_xyz);
	sContext->checkErrNum(errNum, "Error setting interp kernel arguments 6.");
	errNum |= clSetKernelArg(kernel, 7, sizeof(cl_uint2), &wi_tu);
	sContext->checkErrNum(errNum, "Error setting interp kernel arguments 7.");
	errNum |= clSetKernelArg(kernel, 8, sizeof(float), &paddingValue);
	sContext->checkErrNum(errNum, "Error setting interp kernel arguments 8.");
	errNum |= clSetKernelArg(kernel, 9, sizeof(cl_int), &interp);
	sContext->checkErrNum(errNum, "Error setting interp kernel arguments 9.");
	errNum |= clSetKernelArg(kernel, 10, sizeof(cl_int), &datatype);
	sContext->checkErrNum(errNum, "Error setting interp kernel arguments 10.");

	errNum = clEnqueueNDRangeKernel(commandQueue, kernel, dims, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	sContext->checkErrNum(errNum, "Error queuing interp kernel for execution: ");

	clFinish(commandQueue);
}
/* *************************************************************** */
CLResampleImageKernel::~CLResampleImageKernel()
{
	if (kernel != 0)
		clReleaseKernel(kernel);
	if (program != 0)
		clReleaseProgram(program);
}
/* *************************************************************** */
/* *************************************************************** */
CLBlockMatchingKernel::CLBlockMatchingKernel(Content *conIn, std::string name) :
		BlockMatchingKernel(name)
{
	//populate the CLContent object ptr
	con = static_cast<ClContent*>(conIn);

	//path to kernel file
	std::string clInstallPath(CL_KERNELS_PATH);
	std::string clKernel("blockMatchingKernel.cl");

	//get opencl context params
	sContext = &CLContextSingletton::Instance();
	clContext = sContext->getContext();
	commandQueue = sContext->getCommandQueue();
	program = sContext->CreateProgram((clInstallPath + clKernel).c_str());

	// Create OpenCL kernel
	cl_int errNum;
	kernel = clCreateKernel(program, "blockMatchingKernel", &errNum);
	sContext->checkErrNum(errNum, "Error setting bm kernel.");

	//get cl ptrs
	clActiveBlock = con->getActiveBlockClmem();
	clReferenceImageArray = con->getReferenceImageArrayClmem();
	clWarpedImageArray = con->getWarpedImageClmem();
	clWarpedPosition = con->getWarpedPositionClmem();
	clReferencePosition = con->getReferencePositionClmem();
	clMask = con->getMaskClmem();
	clReferenceMat = con->getRefMatClmem();

	//get cpu ptrs
	reference = con->Content::getCurrentReference();
	params = con->Content::getBlockMatchingParams();

}
/* *************************************************************** */
void CLBlockMatchingKernel::calculate()
{
	// Copy some required parameters over to the device
	unsigned int *definedBlock_h = (unsigned int*) malloc(sizeof(unsigned int));
	*definedBlock_h = 0;
	cl_int errNum;
	cl_mem definedBlock = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(unsigned int), definedBlock_h, &errNum);
	this->sContext->checkErrNum(errNum, "CLBlockMatchingKernel::calculate failed to allocate memory (definedBlock): ");

	const unsigned int blockRange = params->voxelCaptureRange%4?params->voxelCaptureRange/4+1:params->voxelCaptureRange/4;
	const unsigned int stepSize = params->stepSize;

	const unsigned int numBlocks = blockRange * 2 + 1;
	const unsigned int sMemSize = numBlocks*numBlocks*numBlocks*64;

	cl_uint3 imageSize = {{(cl_uint)this->reference->nx,
								  (cl_uint)this->reference->ny,
								  (cl_uint)this->reference->nz,
								 (cl_uint)0 }};

	errNum = clSetKernelArg(kernel, 0, sMemSize * sizeof(cl_float), NULL);
	sContext->checkErrNum(errNum, "Error setting shared memory.");
	errNum = clSetKernelArg(kernel, 1, sizeof(cl_mem), &this->clWarpedImageArray);
	sContext->checkErrNum(errNum, "Error setting resultImageArray.");
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &this->clReferenceImageArray);
	sContext->checkErrNum(errNum, "Error setting targetImageArray.");
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &this->clWarpedPosition);
	sContext->checkErrNum(errNum, "Error setting resultPosition.");
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &this->clReferencePosition);
	sContext->checkErrNum(errNum, "Error setting targetPosition.");
	errNum |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &this->clActiveBlock);
	sContext->checkErrNum(errNum, "Error setting mask.");
	errNum |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &this->clMask);
	sContext->checkErrNum(errNum, "Error setting mask.");
	errNum |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &this->clReferenceMat);
	sContext->checkErrNum(errNum, "Error setting targetMatrix_xyz.");
	errNum |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &definedBlock);
	sContext->checkErrNum(errNum, "Error setting definedBlock.");
	errNum |= clSetKernelArg(kernel, 9, sizeof(cl_uint3), &imageSize);
	sContext->checkErrNum(errNum, "Error setting image size.");
	errNum |= clSetKernelArg(kernel, 10, sizeof(cl_uint), &blockRange);
	sContext->checkErrNum(errNum, "Error setting blockRange.");
	errNum |= clSetKernelArg(kernel, 11, sizeof(cl_uint), &stepSize);
	sContext->checkErrNum(errNum, "Error setting step size.");

	const size_t globalWorkSize[3] = { (size_t)params->blockNumber[0] * 4,
												  (size_t)params->blockNumber[1] * 4,
												  (size_t)params->blockNumber[2] * 4 };
	const size_t localWorkSize[3] = { 4, 4, 4 };

	errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	sContext->checkErrNum(errNum, "Error queuing blockmatching kernel for execution: ");
	clFinish(commandQueue);

	errNum = clEnqueueReadBuffer(this->commandQueue, definedBlock, CL_TRUE, 0, sizeof(unsigned int), definedBlock_h, 0, NULL, NULL);
	sContext->checkErrNum(errNum, "Error reading  var after for execution: ");
	params->definedActiveBlock = *definedBlock_h;

	free(definedBlock_h);
	clReleaseMemObject(definedBlock);
}
/* *************************************************************** */
CLBlockMatchingKernel::~CLBlockMatchingKernel()
{
	if (kernel != 0)
		clReleaseKernel(kernel);
	if (program != 0)
		clReleaseProgram(program);
}
/* *************************************************************** */
/* *************************************************************** */
CLOptimiseKernel::CLOptimiseKernel(Content *conIn, std::string name) :
		OptimiseKernel(name)
{
	//populate the CLContent object ptr
	con = static_cast<ClContent*>(conIn);

	//get opencl context params
	sContext = &CLContextSingletton::Instance();
	/*clContext = sContext->getContext();*/
	/*commandQueue = sContext->getCommandQueue();*/

	//get necessary cpu ptrs
	transformationMatrix = con->Content::getTransformationMatrix();
	blockMatchingParams = con->Content::getBlockMatchingParams();
}
/* *************************************************************** */
void CLOptimiseKernel::calculate(bool affine, bool ils, bool clsvd)
{
	//cpu atm
	this->blockMatchingParams = con->getBlockMatchingParams();
	optimize(this->blockMatchingParams, this->transformationMatrix, affine);
}
/* *************************************************************** */
CLOptimiseKernel::~CLOptimiseKernel() {}
/* *************************************************************** */
/* *************************************************************** */
