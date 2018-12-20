#ifndef _REG_ALADIN_CPP
#define _REG_ALADIN_CPP

#include "_reg_ReadWriteMatrix.h"
#include "_reg_aladin.h"
#include "_reg_stringFormat.h"
#include "Platform.h"
#include "AffineDeformationFieldKernel.h"
#include "ResampleImageKernel.h"
#include "BlockMatchingKernel.h"
#include "OptimiseKernel.h"
#include "ConvolutionKernel.h"
#include "AladinContent.h"

#ifdef _USE_CUDA
#include "CUDAAladinContent.h"
#endif
#ifdef _USE_OPENCL
#include "CLAladinContent.h"
#include "InfoDevice.h"
#endif

/* *************************************************************** */
template<class T> reg_aladin<T>::reg_aladin()
{
  this->executableName = (char*) "Aladin";
  this->InputReference = NULL;
  this->InputFloating = NULL;
  this->InputReferenceMask = NULL;
  this->ReferencePyramid = NULL;
  this->FloatingPyramid = NULL;
  this->ReferenceMaskPyramid = NULL;
  this->activeVoxelNumber = NULL;

  this->TransformationMatrix = new mat44;
  this->InputTransformName = NULL;

  this->affineTransformation3DKernel = NULL;
  this->blockMatchingKernel = NULL;
  this->optimiseKernel = NULL;
  this->resamplingKernel = NULL;

  this->con = NULL;
  this->blockMatchingParams = NULL;
  this->platform = NULL;

  this->Verbose = true;

  this->MaxIterations = 5;

  this->NumberOfLevels = 3;
  this->LevelsToPerform = 3;

  this->PerformRigid = 1;
  this->PerformAffine = 1;

  this->BlockStepSize = 1;
  this->BlockPercentage = 50;
  this->InlierLts = 50;

  this->AlignCentre = 1;
  this->AlignCentreMass = 0;

  this->Interpolation = 1;

  this->FloatingSigma = 0.0;
  this->ReferenceSigma = 0.0;

  this->ReferenceUpperThreshold = std::numeric_limits<T>::max();
  this->ReferenceLowerThreshold = -std::numeric_limits<T>::max();

  this->FloatingUpperThreshold = std::numeric_limits<T>::max();
  this->FloatingLowerThreshold = -std::numeric_limits<T>::max();

  this->WarpedPaddingValue = std::numeric_limits<T>::quiet_NaN();

  this->funcProgressCallback = NULL;
  this->paramsProgressCallback = NULL;

  this->platformCode = NR_PLATFORM_CPU;
  this->CurrentLevel = 0;
  this->gpuIdx = 999;

#ifndef NDEBUG
   reg_print_msg_debug("reg_aladin constructor called");
#endif
}
/* *************************************************************** */
template<class T> reg_aladin<T>::~reg_aladin()
{
  if (this->TransformationMatrix != NULL)
    delete this->TransformationMatrix;
  this->TransformationMatrix = NULL;

  if(this->ReferencePyramid!=NULL){
    for (unsigned int l = 0; l < this->LevelsToPerform; ++l)
    {
      if(this->ReferencePyramid[l] != NULL)
        nifti_image_free(this->ReferencePyramid[l]);
      this->ReferencePyramid[l] = NULL;
    }
    free(this->ReferencePyramid);
    this->ReferencePyramid = NULL;
  }
  if(this->FloatingPyramid!=NULL){
    for (unsigned int l = 0; l < this->LevelsToPerform; ++l)
    {
      if(this->FloatingPyramid[l] != NULL)
        nifti_image_free(this->FloatingPyramid[l]);
      this->FloatingPyramid[l] = NULL;
    }
    free(this->FloatingPyramid);
    this->FloatingPyramid = NULL;
  }
  if(this->ReferenceMaskPyramid!=NULL){
    for (unsigned int l = 0; l < this->LevelsToPerform; ++l)
    {
      if(this->ReferenceMaskPyramid[l] != NULL)
        free(this->ReferenceMaskPyramid[l]);
      this->ReferenceMaskPyramid[l] = NULL;
    }
    free(this->ReferenceMaskPyramid);
    this->ReferenceMaskPyramid = NULL;
  }
  if(this->activeVoxelNumber!=NULL)
    free(this->activeVoxelNumber);
  if(this->platform!=NULL)
    delete this->platform;
#ifndef NDEBUG
   reg_print_msg_debug("reg_aladin destructor called");
#endif
}
/* *************************************************************** */
template<class T>
bool reg_aladin<T>::TestMatrixConvergence(mat44 *mat)
{
  bool convergence = true;
  if ((fabsf(mat->m[0][0]) - 1.0f) > CONVERGENCE_EPS)
    convergence = false;
  if ((fabsf(mat->m[1][1]) - 1.0f) > CONVERGENCE_EPS)
    convergence = false;
  if ((fabsf(mat->m[2][2]) - 1.0f) > CONVERGENCE_EPS)
    convergence = false;

  if ((fabsf(mat->m[0][1]) - 0.0f) > CONVERGENCE_EPS)
    convergence = false;
  if ((fabsf(mat->m[0][2]) - 0.0f) > CONVERGENCE_EPS)
    convergence = false;
  if ((fabsf(mat->m[0][3]) - 0.0f) > CONVERGENCE_EPS)
    convergence = false;

  if ((fabsf(mat->m[1][0]) - 0.0f) > CONVERGENCE_EPS)
    convergence = false;
  if ((fabsf(mat->m[1][2]) - 0.0f) > CONVERGENCE_EPS)
    convergence = false;
  if ((fabsf(mat->m[1][3]) - 0.0f) > CONVERGENCE_EPS)
    convergence = false;

  if ((fabsf(mat->m[2][0]) - 0.0f) > CONVERGENCE_EPS)
    convergence = false;
  if ((fabsf(mat->m[2][1]) - 0.0f) > CONVERGENCE_EPS)
    convergence = false;
  if ((fabsf(mat->m[2][3]) - 0.0f) > CONVERGENCE_EPS)
    convergence = false;

  return convergence;
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::SetVerbose(bool _verbose)
{
  this->Verbose = _verbose;
}
/* *************************************************************** */
template<class T>
int reg_aladin<T>::Check()
{
  //This does all the initial checking
  if (this->InputReference == NULL)
  {
    reg_print_fct_error("reg_aladin<T>::Check()");
    reg_print_msg_error("No reference image has been specified or it can not be read");
    return EXIT_FAILURE;
  }

  if (this->InputFloating == NULL)
  {
    reg_print_fct_error("reg_aladin<T>::Check()");
    reg_print_msg_error("No floating image has been specified or it can not be read");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
/* *************************************************************** */
template<class T>
int reg_aladin<T>::Print()
{
  if (this->InputReference == NULL)
  {
    reg_print_fct_error("reg_aladin<T>::Print()");
    reg_print_msg_error("No reference image has been specified");
    return EXIT_FAILURE;
  }
  if (this->InputFloating == NULL)
  {
    reg_print_fct_error("reg_aladin<T>::Print()");
    reg_print_msg_error("No floating image has been specified");
    return EXIT_FAILURE;
  }

  /* *********************************** */
  /* DISPLAY THE REGISTRATION PARAMETERS */
  /* *********************************** */
#ifdef NDEBUG
  if(this->Verbose)
  {
#endif
    std::string text;
    reg_print_info(this->executableName, "Parameters");
    text = stringFormat("Platform: %s", this->platform->getName().c_str());
    reg_print_info(this->executableName, text.c_str());
    text = stringFormat("Reference image name: %s", this->InputReference->fname);
    reg_print_info(this->executableName, text.c_str());
    text = stringFormat("\t%ix%ix%i voxels", this->InputReference->nx, this->InputReference->ny, this->InputReference->nz);
    reg_print_info(this->executableName, text.c_str());
    text = stringFormat("\t%gx%gx%g mm", this->InputReference->dx, this->InputReference->dy, this->InputReference->dz);
    reg_print_info(this->executableName, text.c_str());
    text = stringFormat("Floating image name: %s", this->InputFloating->fname);
    reg_print_info(this->executableName, text.c_str());
    text = stringFormat("\t%ix%ix%i voxels", this->InputFloating->nx, this->InputFloating->ny, this->InputFloating->nz);
    reg_print_info(this->executableName, text.c_str());
    text = stringFormat("\t%gx%gx%g mm", this->InputFloating->dx, this->InputFloating->dy, this->InputFloating->dz);
    reg_print_info(this->executableName, text.c_str());
    text = stringFormat("Maximum iteration number: %i", this->MaxIterations);
    reg_print_info(this->executableName, text.c_str());
    text = stringFormat("\t(%i during the first level)", 2 * this->MaxIterations);
    reg_print_info(this->executableName, text.c_str());
    text = stringFormat("Percentage of blocks: %i %%", this->BlockPercentage);
    reg_print_info(this->executableName, text.c_str());
    reg_print_info(this->executableName, "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
#ifdef NDEBUG
  }
#endif
  return EXIT_SUCCESS;
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::SetInputTransform(const char *filename)
{
  this->InputTransformName = (char *) filename;
  return;
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::InitialiseRegistration()
{
#ifndef NDEBUG
  reg_print_fct_debug("reg_aladin::InitialiseRegistration()");
#endif

  this->platform = new Platform(this->platformCode);
  this->platform->setGpuIdx(this->gpuIdx);

  this->Print();

  // CREATE THE PYRAMID IMAGES
  this->ReferencePyramid = (nifti_image **) malloc(this->LevelsToPerform * sizeof(nifti_image *));
  this->FloatingPyramid = (nifti_image **) malloc(this->LevelsToPerform * sizeof(nifti_image *));
  this->ReferenceMaskPyramid = (int **) malloc(this->LevelsToPerform * sizeof(int *));
  this->activeVoxelNumber = (int *) malloc(this->LevelsToPerform * sizeof(int));

  // FINEST LEVEL OF REGISTRATION
  reg_createImagePyramid<T>(this->InputReference,
                            this->ReferencePyramid,
                            this->NumberOfLevels,
                            this->LevelsToPerform);
  reg_createImagePyramid<T>(this->InputFloating,
                            this->FloatingPyramid,
                            this->NumberOfLevels,
                            this->LevelsToPerform);

  if (this->InputReferenceMask != NULL)
    reg_createMaskPyramid<T>(this->InputReferenceMask,
                             this->ReferenceMaskPyramid,
                             this->NumberOfLevels,
                             this->LevelsToPerform,
                             this->activeVoxelNumber);
  else {
    for (unsigned int l = 0; l < this->LevelsToPerform; ++l) {
      this->activeVoxelNumber[l] = this->ReferencePyramid[l]->nx * this->ReferencePyramid[l]->ny * this->ReferencePyramid[l]->nz;
      this->ReferenceMaskPyramid[l] = (int *) calloc(activeVoxelNumber[l], sizeof(int));
    }
  }

  Kernel *convolutionKernel = this->platform->createKernel(ConvolutionKernel::getName(), NULL);
  // SMOOTH THE INPUT IMAGES IF REQUIRED
  for (unsigned int l = 0; l < this->LevelsToPerform; l++) {
    if (this->ReferenceSigma != 0.0) {
      // Only the first image is smoothed
      bool *active = new bool[this->ReferencePyramid[l]->nt];
      float *sigma = new float[this->ReferencePyramid[l]->nt];
      active[0] = true;
      for (int i = 1; i < this->ReferencePyramid[l]->nt; ++i)
        active[i] = false;
      sigma[0] = this->ReferenceSigma;
      convolutionKernel->castTo<ConvolutionKernel>()->calculate(this->ReferencePyramid[l], sigma, 0, NULL, active);
      delete[] active;
      delete[] sigma;
    }
    if (this->FloatingSigma != 0.0) {
      // Only the first image is smoothed
      bool *active = new bool[this->FloatingPyramid[l]->nt];
      float *sigma = new float[this->FloatingPyramid[l]->nt];
      active[0] = true;
      for (int i = 1; i < this->FloatingPyramid[l]->nt; ++i)
        active[i] = false;
      sigma[0] = this->FloatingSigma;
      convolutionKernel->castTo<ConvolutionKernel>()->calculate(this->FloatingPyramid[l], sigma, 0, NULL, active);
      delete[] active;
      delete[] sigma;
    }
  }
  delete convolutionKernel;

  // THRESHOLD THE INPUT IMAGES IF REQUIRED
  for(unsigned int l=0; l<this->LevelsToPerform; l++)
  {
    reg_thresholdImage<T>(this->ReferencePyramid[l],this->ReferenceLowerThreshold, this->ReferenceUpperThreshold);
    reg_thresholdImage<T>(this->FloatingPyramid[l],this->FloatingLowerThreshold, this->FloatingUpperThreshold);
  }

  // Initialise the transformation
  if (this->InputTransformName != NULL)
  {
    if (FILE *aff = fopen(this->InputTransformName, "r")) {
      fclose(aff);
    }
    else
    {
      std::string text;
      text = stringFormat("The specified input affine file (%s) can not be read", this->InputTransformName);
      reg_print_fct_error("reg_aladin<T>::InitialiseRegistration()");
      reg_print_msg_error(text.c_str());
      reg_exit();
    }
    reg_tool_ReadAffineFile(this->TransformationMatrix, this->InputTransformName);
  }
  else  // No input affine transformation
  {
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        this->TransformationMatrix->m[i][j] = 0.0;
      }
      this->TransformationMatrix->m[i][i] = 1.0;
    }
    if (this->AlignCentre && this->AlignCentreMass==0)
    {
      const mat44 *floatingMatrix = (this->InputFloating->sform_code > 0) ? &(this->InputFloating->sto_xyz) : &(this->InputFloating->qto_xyz);
      const mat44 *referenceMatrix = (this->InputReference->sform_code > 0) ? &(this->InputReference->sto_xyz) : &(this->InputReference->qto_xyz);
      //In pixel coordinates
      float floatingCenter[3];
      floatingCenter[0] = (float) (this->InputFloating->nx) / 2.0f;
      floatingCenter[1] = (float) (this->InputFloating->ny) / 2.0f;
      floatingCenter[2] = (float) (this->InputFloating->nz) / 2.0f;
      float referenceCenter[3];
      referenceCenter[0] = (float) (this->InputReference->nx) / 2.0f;
      referenceCenter[1] = (float) (this->InputReference->ny) / 2.0f;
      referenceCenter[2] = (float) (this->InputReference->nz) / 2.0f;
      //From pixel coordinates to real coordinates
      float floatingRealPosition[3];
      reg_mat44_mul(floatingMatrix, floatingCenter, floatingRealPosition);
      float referenceRealPosition[3];
      reg_mat44_mul(referenceMatrix, referenceCenter, referenceRealPosition);
      //Set translation to the transformation matrix
      this->TransformationMatrix->m[0][3] = floatingRealPosition[0] - referenceRealPosition[0];
      this->TransformationMatrix->m[1][3] = floatingRealPosition[1] - referenceRealPosition[1];
      this->TransformationMatrix->m[2][3] = floatingRealPosition[2] - referenceRealPosition[2];
    }
	else if (this->AlignCentreMass == 2)
	{
		float referenceCentre[3] = { 0,0,0 };
		float referenceCount = 0;
		reg_tools_changeDatatype<float>(this->InputReference);
		float *refPtr = static_cast<float *>(this->InputReference->data);
		size_t refIndex = 0;
		for (int z = 0; z < this->InputReference->nz; ++z) {
			for (int y = 0; y < this->InputReference->ny; ++y) {
				for (int x = 0; x < this->InputReference->nx; ++x) {
					float value = refPtr[refIndex];
					referenceCentre[0] += (float)x * value;
					referenceCentre[1] += (float)y * value;
					referenceCentre[2] += (float)z * value;
					referenceCount+=value;
					refIndex++;
				}
			}
		}
		referenceCentre[0] /= referenceCount;
		referenceCentre[1] /= referenceCount;
		referenceCentre[2] /= referenceCount;
		float refCOM[3];
		if (this->InputReference->sform_code > 0)
			reg_mat44_mul(&(this->InputReference->sto_xyz), referenceCentre, refCOM);

		float floatingCentre[3] = { 0,0,0 };
		float floatingCount = 0;
		reg_tools_changeDatatype<float>(this->InputFloating);
		float *floPtr = static_cast<float *>(this->InputFloating->data);
		size_t floIndex = 0;
		for (int z = 0; z < this->InputFloating->nz; ++z) {
			for (int y = 0; y < this->InputFloating->ny; ++y) {
				for (int x = 0; x < this->InputFloating->nx; ++x) {
					float value = floPtr[floIndex];
					floatingCentre[0] += (float)x * value;
					floatingCentre[1] += (float)y * value;
					floatingCentre[2] += (float)z * value;
					floatingCount += value;
					floIndex++;
				}
			}
		}
		floatingCentre[0] /= floatingCount;
		floatingCentre[1] /= floatingCount;
		floatingCentre[2] /= floatingCount;
		float floCOM[3];
		if (this->InputFloating->sform_code > 0)
			reg_mat44_mul(&(this->InputFloating->sto_xyz), floatingCentre, floCOM);
		reg_mat44_eye(this->TransformationMatrix);
		this->TransformationMatrix->m[0][3] = floCOM[0] - refCOM[0];
		this->TransformationMatrix->m[1][3] = floCOM[1] - refCOM[1];
		this->TransformationMatrix->m[2][3] = floCOM[2] - refCOM[2];
	}
  }
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::ClearCurrentInputImage()
{
  nifti_image_free(this->ReferencePyramid[this->CurrentLevel]);
  this->ReferencePyramid[this->CurrentLevel] = NULL;

  nifti_image_free(this->FloatingPyramid[this->CurrentLevel]);
  this->FloatingPyramid[this->CurrentLevel] = NULL;

  free(this->ReferenceMaskPyramid[this->CurrentLevel]);
  this->ReferenceMaskPyramid[this->CurrentLevel] = NULL;
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::createKernels()
{
  this->affineTransformation3DKernel = platform->createKernel(AffineDeformationFieldKernel::getName(), this->con);
  this->resamplingKernel = platform->createKernel(ResampleImageKernel::getName(), this->con);
  if (this->blockMatchingParams != NULL) {
    this->blockMatchingKernel = platform->createKernel(BlockMatchingKernel::getName(), this->con);
    this->optimiseKernel = platform->createKernel(OptimiseKernel::getName(), this->con);
  } else {
    this->blockMatchingKernel = NULL;
    this->optimiseKernel = NULL;
  }
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::clearKernels()
{
  delete this->affineTransformation3DKernel;
  delete this->resamplingKernel;
  if (this->blockMatchingKernel != NULL)
    delete this->blockMatchingKernel;
  if (this->optimiseKernel != NULL)
    delete this->optimiseKernel;
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::GetDeformationField()
{
  this->affineTransformation3DKernel->template castTo<AffineDeformationFieldKernel>()->calculate();
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::GetWarpedImage(int interp, float padding)
{
  this->GetDeformationField();
  this->resamplingKernel->template castTo<ResampleImageKernel>()->calculate(interp, padding);
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::UpdateTransformationMatrix(int type)
{
  this->blockMatchingKernel->template castTo<BlockMatchingKernel>()->calculate();
  this->optimiseKernel->template castTo<OptimiseKernel>()->calculate(type);

#ifndef NDEBUG
  reg_mat44_disp(this->TransformationMatrix, (char *) "[NiftyReg DEBUG] updated forward matrix");
#endif
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::initAladinContent(nifti_image *ref,
                                      nifti_image *flo,
                                      int *mask,
                                      mat44 *transMat,
                                      size_t bytes,
                                      unsigned int blockPercentage,
                                      unsigned int inlierLts,
                                      unsigned int blockStepSize)
{
  if (this->platformCode == NR_PLATFORM_CPU)
    this->con = new AladinContent(ref, flo, mask, transMat, bytes, blockPercentage, inlierLts, blockStepSize);
#ifdef _USE_CUDA
  else if(platformCode == NR_PLATFORM_CUDA)
    this->con = new CudaAladinContent(ref, flo, mask,transMat, bytes, blockPercentage, inlierLts, blockStepSize);
#endif
#ifdef _USE_OPENCL
  else if(platformCode == NR_PLATFORM_CL)
    this->con = new ClAladinContent(ref, flo, mask,transMat, bytes, blockPercentage, inlierLts, blockStepSize);
#endif
  this->blockMatchingParams = this->con->AladinContent::getBlockMatchingParams();
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::initAladinContent(nifti_image *ref,
                                      nifti_image *flo,
                                      int *mask,
                                      mat44 *transMat,
                                      size_t bytes)
{
  if (this->platformCode == NR_PLATFORM_CPU)
    this->con = new AladinContent(ref, flo, mask, transMat, bytes);
#ifdef _USE_CUDA
  else if(platformCode == NR_PLATFORM_CUDA)
    this->con = new CudaAladinContent(ref, flo, mask,transMat, bytes);
#endif
#ifdef _USE_OPENCL
  else if(platformCode == NR_PLATFORM_CL)
    this->con = new ClAladinContent(ref, flo, mask,transMat, bytes);
#endif
  this->blockMatchingParams = this->con->AladinContent::getBlockMatchingParams();
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::clearAladinContent()
{
  delete this->con;
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::resolveMatrix(unsigned int iterations, const unsigned int optimizationFlag)
{
  unsigned int iteration = 0;
  while (iteration < iterations) {
#ifndef NDEBUG
    char text[255];
    sprintf(text, "%s - level: %i/%i - iteration %i/%i",
            optimizationFlag ? (char *)"Affine" : (char *)"Rigid",
            this->CurrentLevel+1, this->NumberOfLevels, iteration+1, iterations);
    reg_print_msg_debug(text);
#endif
    this->GetWarpedImage(this->Interpolation, this->WarpedPaddingValue);
    this->UpdateTransformationMatrix(optimizationFlag);

    iteration++;
  }
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::Run()
{
  this->InitialiseRegistration();

  //Main loop over the levels:
  for (this->CurrentLevel = 0; this->CurrentLevel < this->LevelsToPerform; this->CurrentLevel++)
  {
    this->initAladinContent(this->ReferencePyramid[CurrentLevel], this->FloatingPyramid[CurrentLevel],
                            this->ReferenceMaskPyramid[CurrentLevel], this->TransformationMatrix, sizeof(T), this->BlockPercentage,
                            this->InlierLts, this->BlockStepSize);
    this->createKernels();

    // Twice more iterations are performed during the first level
    // All the blocks are used during the first level
    const unsigned int maxNumberOfIterationToPerform = (CurrentLevel == 0) ? this->MaxIterations*2 : this->MaxIterations;

#ifdef NDEBUG
    if(this->Verbose)
    {
#endif
      this->DebugPrintLevelInfoStart();
#ifdef NDEBUG
    }
#endif

#ifndef NDEBUG
    if (this->con->getCurrentReference()->sform_code > 0)
      reg_mat44_disp(&this->con->getCurrentReference()->sto_xyz, (char *) "[NiftyReg DEBUG] Reference image matrix (sform sto_xyz)");
    else
      reg_mat44_disp(&this->con->getCurrentReference()->qto_xyz, (char *) "[NiftyReg DEBUG] Reference image matrix (qform qto_xyz)");
    if (this->con->getCurrentFloating()->sform_code > 0)
      reg_mat44_disp(&this->con->getCurrentFloating()->sto_xyz, (char *) "[NiftyReg DEBUG] Floating image matrix (sform sto_xyz)");
    else
      reg_mat44_disp(&this->con->getCurrentFloating()->qto_xyz, (char *) "[NiftyReg DEBUG] Floating image matrix (qform qto_xyz)");
#endif

    /* ****************** */
    /* Rigid registration */
    /* ****************** */
    if ((this->PerformRigid && !this->PerformAffine) || (this->PerformAffine && this->PerformRigid && this->CurrentLevel == 0))
    {
      const unsigned int ratio = (this->PerformAffine && this->PerformRigid && this->CurrentLevel == 0) ? 4 : 1;
      resolveMatrix(maxNumberOfIterationToPerform * ratio, RIGID);
    }

    /* ******************* */
    /* Affine registration */
    /* ******************* */
    if (this->PerformAffine)
      resolveMatrix(maxNumberOfIterationToPerform, AFFINE);

    // SOME CLEANING IS PERFORMED
    this->clearKernels();
    this->clearAladinContent();
    this->ClearCurrentInputImage();

#ifdef NDEBUG
    if(this->Verbose)
    {
#endif
      this->DebugPrintLevelInfoEnd();
      reg_print_info(this->executableName, "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -");
#ifdef NDEBUG
    }
#endif

  }

#ifndef NDEBUG
  reg_print_msg_debug("reg_aladin::Run() done");
#endif
  return;
}
/* *************************************************************** */
template<class T>
nifti_image *reg_aladin<T>::GetFinalWarpedImage()
{
  int floatingType = this->InputFloating->datatype; //t_dev ask before touching this!
  // The initial images are used
  if (this->InputReference == NULL || this->InputFloating == NULL || this->TransformationMatrix == NULL) {
    reg_print_fct_error("reg_aladin::GetFinalWarpedImage()");
    reg_print_msg_error("The reference, floating images and the transformation have to be defined");
    reg_exit();
  }

  int *mask = (int *)calloc(this->InputReference->nx*this->InputReference->ny*this->InputReference->nz,
                            sizeof(int));

  reg_aladin<T>::initAladinContent(this->InputReference,
                                   this->InputFloating,
                                   mask,
                                   this->TransformationMatrix,
                                   sizeof(T));
  reg_aladin<T>::createKernels();

  reg_aladin<T>::GetWarpedImage(3, this->WarpedPaddingValue); // cubic spline interpolation
  nifti_image *CurrentWarped = this->con->getCurrentWarped(floatingType);

  free(mask);
  nifti_image *resultImage = nifti_copy_nim_info(CurrentWarped);
  resultImage->cal_min = this->InputFloating->cal_min;
  resultImage->cal_max = this->InputFloating->cal_max;
  resultImage->scl_slope = this->InputFloating->scl_slope;
  resultImage->scl_inter = this->InputFloating->scl_inter;
  resultImage->data = (void *) malloc(resultImage->nvox * resultImage->nbyper);
  memcpy(resultImage->data, CurrentWarped->data, resultImage->nvox * resultImage->nbyper);

  reg_aladin<T>::clearKernels();
  reg_aladin<T>::clearAladinContent();
  return resultImage;
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::DebugPrintLevelInfoStart()
{
  /* Display some parameters specific to the current level */
  char text[255];
  sprintf(text, "Current level %i / %i", this->CurrentLevel + 1, this->NumberOfLevels);
  reg_print_info(this->executableName,text);
  sprintf(text, "reference image size: \t%ix%ix%i voxels\t%gx%gx%g mm",
          this->con->getCurrentReference()->nx,
          this->con->getCurrentReference()->ny,
          this->con->getCurrentReference()->nz,
          this->con->getCurrentReference()->dx,
          this->con->getCurrentReference()->dy,
          this->con->getCurrentReference()->dz);
  reg_print_info(this->executableName,text);
  sprintf(text, "floating image size: \t%ix%ix%i voxels\t%gx%gx%g mm",
          this->con->getCurrentFloating()->nx,
          this->con->getCurrentFloating()->ny,
          this->con->getCurrentFloating()->nz,
          this->con->getCurrentFloating()->dx,
          this->con->getCurrentFloating()->dy,
          this->con->getCurrentFloating()->dz);
  reg_print_info(this->executableName,text);
  if (this->con->getCurrentReference()->nz == 1){
    reg_print_info(this->executableName, "Block size = [4 4 1]");
  }
  else reg_print_info(this->executableName, "Block size = [4 4 4]");
  reg_print_info(this->executableName, "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
  sprintf(text, "Block number = [%i %i %i]", this->blockMatchingParams->blockNumber[0],
      this->blockMatchingParams->blockNumber[1], this->blockMatchingParams->blockNumber[2]);
  reg_print_info(this->executableName,text);
  reg_mat44_disp(this->TransformationMatrix, (char *) "[reg_aladin] Initial transformation matrix:");
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::DebugPrintLevelInfoEnd()
{
  reg_mat44_disp(this->TransformationMatrix, (char *) "[reg_aladin] Final transformation matrix:");
}
/* *************************************************************** */

#endif //#ifndef _REG_ALADIN_CPP
