#ifndef _REG_ALADIN_CPP
#define _REG_ALADIN_CPP

#include "_reg_ReadWriteMatrix.h"
#include "_reg_aladin.h"
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
template<class T>
reg_aladin<T>::reg_aladin(int platformCodeIn)
{
  this->executableName = (char*) "Aladin";

  this->affineTransformation3DKernel = NULL;
  this->blockMatchingKernel = NULL;
  this->optimiseKernel = NULL;
  this->resamplingKernel = NULL;

  if(platformCodeIn == NR_PLATFORM_CPU)
    this->con = new AladinContent(platformCodeIn);
#ifdef _USE_CUDA
  else if(platformCodeIn == NR_PLATFORM_CUDA)
    this->con = new CudaAladinContent();
#endif
#ifdef _USE_OPENCL
  else if(platformCodeIn == NR_PLATFORM_CL)
    this->con = new ClAladinContent();
#endif

  this->maxIterations = 5;

  this->performRigid = 1;
  this->performAffine = 1;

  this->alignCentre = 1;
  this->alignCentreGravity = 0;

  this->interpolation = 1;

  this->funcProgressCallback = NULL;
  this->paramsProgressCallback = NULL;

  this->currentLevel = 0;
}
/* *************************************************************** */
template<class T>
reg_aladin<T>::~reg_aladin()
{
    this->ClearKernels();
    delete this->con;
#ifndef NDEBUG
   reg_print_fct_debug("reg_aladin<T>::~reg_aladin()");
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
  this->verbose = _verbose;
}
/* *************************************************************** */
template<class T>
int reg_aladin<T>::Check()
{
  //This does all the initial checking
  if (this->con->getInputReference() == NULL)
  {
    reg_print_fct_error("reg_aladin<T>::Check()");
    reg_print_msg_error("No reference image has been specified or it can not be read");
    return EXIT_FAILURE;
  }

  if (this->con->getInputFloating() == NULL)
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
  if (this->con->getInputReference() == NULL) {
    reg_print_fct_error("reg_aladin<T>::Print()");
    reg_print_msg_error("No reference image has been specified");
    return EXIT_FAILURE;
  } if (this->con->getInputFloating() == NULL) {
    reg_print_fct_error("reg_aladin<T>::Print()");
    reg_print_msg_error("No floating image has been specified");
    return EXIT_FAILURE;
  }
  /* *********************************** */
  /* DISPLAY THE REGISTRATION PARAMETERS */
  /* *********************************** */
  if(this->verbose) {
    char text[255];
    reg_print_info(this->executableName, "Parameters");
    sprintf(text, "Platform: %s", this->con->getPlatform()->getName().c_str());
    reg_print_info(this->executableName, text);
    sprintf(text, "Reference image name: %s", this->con->getInputReference()->fname);
    reg_print_info(this->executableName, text);
    sprintf(text, "\t%ix%ix%i voxels", this->con->getInputReference()->nx, this->con->getInputReference()->ny, this->con->getInputReference()->nz);
    reg_print_info(this->executableName, text);
    sprintf(text, "\t%gx%gx%g mm", this->con->getInputReference()->dx, this->con->getInputReference()->dy, this->con->getInputReference()->dz);
    reg_print_info(this->executableName, text);
    sprintf(text, "Floating image name: %s", this->con->getInputFloating()->fname);
    reg_print_info(this->executableName, text);
    sprintf(text, "\t%ix%ix%i voxels", this->con->getInputFloating()->nx, this->con->getInputFloating()->ny, this->con->getInputFloating()->nz);
    reg_print_info(this->executableName, text);
    sprintf(text, "\t%gx%gx%g mm", this->con->getInputFloating()->dx, this->con->getInputFloating()->dy, this->con->getInputFloating()->dz);
    reg_print_info(this->executableName, text);
    sprintf(text, "Maximum iteration number: %i", this->maxIterations);
    reg_print_info(this->executableName, text);
    sprintf(text, "\t(%i during the first level)", 2 * this->maxIterations);
    reg_print_info(this->executableName, text);
    sprintf(text, "Percentage of blocks: %i %%", this->con->getPercentageOfBlock());
    reg_print_info(this->executableName, text);
    reg_print_info(this->executableName, "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
  }
  return EXIT_SUCCESS;
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::SetInputReference(nifti_image* inputRefIn)
{
  this->con->setInputReference(inputRefIn);
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::SetInputFloating(nifti_image* inputFloIn)
{
  this->con->setInputFloating(inputFloIn);
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::SetInputReferenceMask(nifti_image* input)
{
  this->con->setInputReferenceMask(input);
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::SetNumberOfLevels(unsigned levelNumber)
{
  this->con->setLevelNumber(levelNumber);
}
/* *************************************************************** */
template<class T>
unsigned reg_aladin<T>::GetNumberOfLevels()
{
  this->con->getLevelNumber();
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::SetLevelsToPerform(unsigned lp)
{
  this->con->setLevelToPerform(lp);
}
/* *************************************************************** */
template<class T>
unsigned reg_aladin<T>::GetLevelsToPerform()
{
  this->con->getLevelToPerform();
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::SetReferenceSigma(float sigma)
{
  this->con->setReferenceSmoothingSigma(sigma);
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::SetFloatingSigma(float sigma)
{
  this->con->setFloatingSmoothingSigma(sigma);
}
/* *************************************************************** */
template<class T>
mat44* reg_aladin<T>::GetTransformationMatrix()
{
  this->con->getTransformationMatrix();
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::SetReferenceLowerThreshold(float th)
{
  this->con->setReferenceThresholdLow(0,th);
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::SetReferenceUpperThreshold(float th)
{
  this->con->setReferenceThresholdUp(0,th);
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::SetFloatingLowerThreshold(float th)
{
  this->con->setFloatingThresholdLow(0,th);
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::SetFloatingUpperThreshold(float th)
{
  this->con->setFloatingThresholdUp(0,th);
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::SetBlockStepSize(int bss)
{
  this->con->setBlockStepSize(bss);
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::SetBlockPercentage(unsigned bp)
{
  this->con->setPercentageOfBlock(bp);
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::SetInlierLts(unsigned ilts)
{
  this->con->setInlierLts(ilts);
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::SetInputTransform(mat44* mat44In)
{
  this->con->setAffineTransformation(mat44In);
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::SetInputFloatingMask(nifti_image*)
{
    reg_print_fct_warn("reg_aladin::SetInputFloatingMask()");
    reg_print_msg_warn("Floating mask not used in the asymmetric global registration");
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::SetCaptureRangeVox(int captureRangeIn)
{
    this->captureRangeVox = captureRangeIn;
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::InitialiseRegistration()
{
#ifndef NDEBUG
  reg_print_fct_debug("reg_aladin::InitialiseRegistration()");
#endif

  this->con->InitialiseGlobalContent();
  this->Print();

  // Initialise the transformation
  if (this->con->getAffineTransformation() == NULL)
  {
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        this->con->getTransformationMatrix()->m[i][j] = 0.0;
      }
      this->con->getTransformationMatrix()->m[i][i] = 1.0;
    }
    if (this->alignCentre)
    {
      const mat44 *floatingMatrix = (this->con->getInputFloating()->sform_code > 0) ? &(this->con->getInputFloating()->sto_xyz) : &(this->con->getInputFloating()->qto_xyz);
      const mat44 *referenceMatrix = (this->con->getInputReference()->sform_code > 0) ? &(this->con->getInputReference()->sto_xyz) : &(this->con->getInputReference()->qto_xyz);
      //In pixel coordinates
      float floatingCenter[3];
      floatingCenter[0] = (float) (this->con->getInputFloating()->nx) / 2.0f;
      floatingCenter[1] = (float) (this->con->getInputFloating()->ny) / 2.0f;
      floatingCenter[2] = (float) (this->con->getInputFloating()->nz) / 2.0f;
      float referenceCenter[3];
      referenceCenter[0] = (float) (this->con->getInputReference()->nx) / 2.0f;
      referenceCenter[1] = (float) (this->con->getInputReference()->ny) / 2.0f;
      referenceCenter[2] = (float) (this->con->getInputReference()->nz) / 2.0f;
      //From pixel coordinates to real coordinates
      float floatingRealPosition[3];
      reg_mat44_mul(floatingMatrix, floatingCenter, floatingRealPosition);
      float referenceRealPosition[3];
      reg_mat44_mul(referenceMatrix, referenceCenter, referenceRealPosition);
      //Set translation to the transformation matrix
      this->con->getTransformationMatrix()->m[0][3] = floatingRealPosition[0] - referenceRealPosition[0];
      this->con->getTransformationMatrix()->m[1][3] = floatingRealPosition[1] - referenceRealPosition[1];
      this->con->getTransformationMatrix()->m[2][3] = floatingRealPosition[2] - referenceRealPosition[2];
    }
  }
}
/* *************************************************************** */
//template<class T>
//void reg_aladin<T>::ClearCurrentImagePyramid()
//{
//    this->con->ClearCurrentImagePyramid(this->currentLevel);
//}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::ClearBlockMatchingParams()
{
    this->con->ClearBlockMatchingParams();
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::CreateKernels()
{
  this->affineTransformation3DKernel = this->con->getPlatform()->createKernel(AffineDeformationFieldKernel::getName(), this->con);
  this->resamplingKernel = this->con->getPlatform()->createKernel(ResampleImageKernel::getName(), this->con);
  if (this->con->AladinContent::getBlockMatchingParams() != NULL) {
    this->blockMatchingKernel = this->con->getPlatform()->createKernel(BlockMatchingKernel::getName(), this->con);
    this->optimiseKernel = this->con->getPlatform()->createKernel(OptimiseKernel::getName(), this->con);
  } else {
    this->blockMatchingKernel = NULL;
    this->optimiseKernel = NULL;
  }
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::ClearKernels()
{
  if(this->affineTransformation3DKernel != NULL) {
      delete this->affineTransformation3DKernel;
      this->affineTransformation3DKernel = NULL;
  }
  if(this->resamplingKernel == NULL) {
      delete this->resamplingKernel;
      this->resamplingKernel = NULL;
  }
  if (this->blockMatchingKernel != NULL) {
      delete this->blockMatchingKernel;
      this->blockMatchingKernel = NULL;
  }
  if (this->optimiseKernel != NULL) {
      delete this->optimiseKernel;
      this->optimiseKernel = NULL;
  }
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::GetDeformationField()
{
  this->affineTransformation3DKernel->template castTo<AffineDeformationFieldKernel>()->calculate();
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::GetWarpedImage(int interp)
{
  this->GetDeformationField();
  this->resamplingKernel->template castTo<ResampleImageKernel>()->calculate(interp, this->con->getWarpedPaddingValue());
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::UpdateTransformationMatrix(int type)
{
  this->blockMatchingKernel->template castTo<BlockMatchingKernel>()->calculate();
  this->optimiseKernel->template castTo<OptimiseKernel>()->calculate(type);

#ifndef NDEBUG
  reg_mat44_disp(this->con->getTransformationMatrix(), (char *) "[NiftyReg DEBUG] updated forward matrix");
#endif
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::InitCurrentLevel(unsigned int cl)
{
  this->con->setCurrentReference(this->con->getReferencePyramid()[cl]);
  this->con->setCurrentFloating(this->con->getFloatingPyramid()[cl]);
  this->con->setCurrentReferenceMask(this->con->getMaskPyramid()[cl], this->con->getActiveVoxelNumber()[cl]);
  this->con->AllocateWarped();
  this->con->AllocateDeformationField();
  this->con->InitBlockMatchingParams();
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::AllocateImages()
{
    this->con->AllocateWarped();
    this->con->AllocateDeformationField();
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::ClearAllocatedImages()
{
    this->con->ClearWarped();
    this->con->ClearDeformationField();
}
/* *************************************************************** */
//template<class T>
//void reg_aladin<T>::clearAladinContent()
//{
//  delete this->con;
//}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::ResolveMatrix(unsigned int iterations, const unsigned int optimizationFlag)
{
  unsigned int iteration = 0;
  while (iteration < iterations) {
#ifndef NDEBUG
    char text[255];
    sprintf(text, "%s - level: %i/%i - iteration %i/%i",
            optimizationFlag ? (char *)"Affine" : (char *)"Rigid",
            this->currentLevel+1, this->con->getLevelToPerform(), iteration+1, iterations);
    reg_print_msg_debug(text);
#endif
    this->GetWarpedImage(this->interpolation);
    this->UpdateTransformationMatrix(optimizationFlag);

    iteration++;
  }
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::Run()
{
  //CPU Init
  this->InitialiseRegistration();

  //Main loop over the levels:
  for (this->currentLevel = 0; this->currentLevel < this->con->getLevelToPerform(); this->currentLevel++)
  {
    this->InitCurrentLevel(this->currentLevel);

    this->CreateKernels();

    // Twice more iterations are performed during the first level
    // All the blocks are used during the first level
    const unsigned int maxNumberOfIterationToPerform = (this->currentLevel == 0) ? this->maxIterations*2 : this->maxIterations;

#ifdef NDEBUG
    if(this->verbose)
    {
#endif
      this->DebugPrintLevelInfoStart();
#ifdef NDEBUG
    }
#endif

#ifndef NDEBUG
      reg_mat44_disp(this->con->getCurrentReferenceMatrix_xyz(), (char *) "[NiftyReg DEBUG] Reference image matrix");
      reg_mat44_disp(this->con->getCurrentFloatingMatrix_xyz(), (char *) "[NiftyReg DEBUG] Floating image matrix");
#endif

    /* ****************** */
    /* Rigid registration */
    /* ****************** */
    if ((this->performRigid && !this->performAffine) || (this->performAffine && this->performRigid && this->currentLevel == 0))
    {
      const unsigned int ratio = (this->performAffine && this->performRigid && this->currentLevel == 0) ? 4 : 1;
      ResolveMatrix(maxNumberOfIterationToPerform * ratio, RIGID);
    }

    /* ******************* */
    /* Affine registration */
    /* ******************* */
    if (this->performAffine)
      ResolveMatrix(maxNumberOfIterationToPerform, AFFINE);

    // SOME CLEANING IS PERFORMED
    this->ClearKernels();
    this->ClearAllocatedImages();
    //this->ClearCurrentImagePyramid();

#ifdef NDEBUG
    if(this->verbose)
    {
#endif
      this->DebugPrintLevelInfoEnd();
      reg_print_info(this->executableName, "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -");
#ifdef NDEBUG
    }
#endif
  }
  this->ClearBlockMatchingParams();
#ifndef NDEBUG
  reg_print_msg_debug("reg_aladin::Run() done");
#endif
  return;
}
/* *************************************************************** */
template<class T>
nifti_image *reg_aladin<T>::GetFinalWarpedImage()
{
  int floatingType = this->con->getInputFloating()->datatype; //t_dev ask before touching this!
  // The initial images are used
  if (this->con->getInputReference() == NULL ||
      this->con->getInputFloating() == NULL ||
      this->con->getTransformationMatrix() == NULL) {
    reg_print_fct_error("reg_aladin::GetFinalWarpedImage()");
    reg_print_msg_error("The reference, floating images and the transformation have to be defined");
    reg_exit();
  }

  int *mask = (int *)calloc(this->con->getInputReference()->nx*
                            this->con->getInputReference()->ny*
                            this->con->getInputReference()->nz,
                            sizeof(int));

  this->con->setCurrentReference(this->con->getInputReference());
  this->con->setCurrentFloating(this->con->getInputFloating());
  this->con->setCurrentReferenceMask(mask, this->con->getInputReference()->nx*
                                     this->con->getInputReference()->ny*
                                     this->con->getInputReference()->nz);
  reg_aladin<T>::AllocateImages();
  reg_aladin<T>::CreateKernels();

  reg_aladin<T>::GetWarpedImage(3); //3 = cubic spline interpolation
  nifti_image *CurrentWarped = this->con->getCurrentWarped(floatingType);

  free(mask);
  nifti_image *resultImage = nifti_copy_nim_info(CurrentWarped);
  resultImage->cal_min = this->con->getInputFloating()->cal_min;
  resultImage->cal_max = this->con->getInputFloating()->cal_max;
  resultImage->scl_slope = this->con->getInputFloating()->scl_slope;
  resultImage->scl_inter = this->con->getInputFloating()->scl_inter;
  resultImage->data = (void *) malloc(resultImage->nvox * resultImage->nbyper);
  memcpy(resultImage->data, CurrentWarped->data, resultImage->nvox * resultImage->nbyper);

  reg_aladin<T>::ClearKernels();
  reg_aladin<T>::ClearAllocatedImages();
  return resultImage;
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::DebugPrintLevelInfoStart()
{
  /* Display some parameters specific to the current level */
  char text[255];
  sprintf(text, "Current level %i / %i", this->currentLevel + 1, this->con->getLevelToPerform());
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
  sprintf(text, "Block number = [%i %i %i]", this->con->getBlockMatchingParams()->blockNumber[0],
      this->con->getBlockMatchingParams()->blockNumber[1], this->con->getBlockMatchingParams()->blockNumber[2]);
  reg_print_info(this->executableName,text);
  reg_mat44_disp(this->con->getTransformationMatrix(), (char *) "[reg_aladin] Initial transformation matrix:");
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::DebugPrintLevelInfoEnd()
{
  reg_mat44_disp(this->con->getTransformationMatrix(), (char *) "[reg_aladin] Final transformation matrix:");
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::SetGpuIdx(unsigned gpuIdxIn){
   this->con->getPlatform()->setGpuIdx(gpuIdxIn);
}
/* *************************************************************** */
#endif //#ifndef _REG_ALADIN_CPP
