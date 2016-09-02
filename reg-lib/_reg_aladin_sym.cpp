#ifndef _REG_ALADIN_SYM_CPP
#define _REG_ALADIN_SYM_CPP

#include "_reg_aladin_sym.h"
#include "_reg_maths_eigen.h"

/* *************************************************************** */
template <class T>
reg_aladin_sym<T>::reg_aladin_sym(int platformCodeIn)
   :reg_aladin<T>::reg_aladin(platformCodeIn)
{
    this->executableName=(char*) "reg_aladin_sym";

    this->bAffineTransformation3DKernel = NULL;
    this->bConvolutionKernel=NULL;
    this->bBlockMatchingKernel=NULL;
    this->bOptimiseKernel=NULL;
    this->bResamplingKernel=NULL;

    if(platformCodeIn == NR_PLATFORM_CPU)
      this->backCon = new AladinContent(platformCodeIn);
#ifdef _USE_CUDA
    else if(platformCodeIn == NR_PLATFORM_CUDA)
      this->backCon = new CudaAladinContent();
#endif
#ifdef _USE_OPENCL
    else if(platformCodeIn == NR_PLATFORM_CL)
      this->backCon = new ClAladinContent();
#endif

#ifndef NDEBUG
    reg_print_msg_debug("reg_aladin_sym constructor called");
#endif

}
/* *************************************************************** */
template <class T>
reg_aladin_sym<T>::~reg_aladin_sym()
{
    if(this->bResamplingKernel != NULL) {
      delete this->bResamplingKernel;
      this->bResamplingKernel = NULL;
    }
    if(this->bAffineTransformation3DKernel != NULL) {
    delete this->bAffineTransformation3DKernel;
    this->bAffineTransformation3DKernel = NULL;
    }
    if(this->bBlockMatchingKernel != NULL) {
    delete this->bBlockMatchingKernel;
    this->bBlockMatchingKernel = NULL;
    }
    if(this->bOptimiseKernel != NULL) {
    delete this->bOptimiseKernel;
    this->bOptimiseKernel = NULL;
    }
    //this->backCon->ClearWarped();
    //this->backCon->ClearDeformationField();
    //this->backCon->ClearActiveVoxelNumber();
    //this->backCon->ClearThresholds();
    this->backCon->ClearMaskPyramid();
    delete this->backCon;
#ifndef NDEBUG
   reg_print_fct_debug("reg_aladin_sym<T>::~reg_aladin_sym()");
#endif
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::SetInputFloatingMask(nifti_image *m)
{
   this->backCon->setInputReferenceMask(m);
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::InitialiseRegistration()
{
#ifndef NDEBUG
   reg_print_fct_debug("reg_aladin_sym::InitialiseRegistration()");
#endif

   reg_aladin<T>::InitialiseRegistration();
   //The reference becomes the floating etc.!
   this->backCon->setInputReference(this->con->getInputFloating());
   this->backCon->setInputFloating(this->con->getInputReference());

   this->backCon->setNbRefTimePoint(this->con->getNbFloTimePoint());
   this->backCon->setNbFloTimePoint(this->con->getNbRefTimePoint());

   this->backCon->setReferenceSmoothingSigma(this->con->getFloatingSmoothingSigma());
   this->backCon->setFloatingSmoothingSigma(this->con->getReferenceSmoothingSigma());

   this->backCon->setRobustRange(this->con->getRobustRange());

   for(int i=0;i<this->backCon->getNbRefTimePoint();i++) {
       this->backCon->setReferenceThresholdUp(i,this->con->getFloatingThresholdUp()[i]);
       this->backCon->setFloatingThresholdUp(i,this->con->getReferenceThresholdUp()[i]);
       //
       this->backCon->setReferenceThresholdLow(i,this->con->getFloatingThresholdLow()[i]);
       this->backCon->setFloatingThresholdLow(i,this->con->getReferenceThresholdLow()[i]);
   }
   if(this->con->isPyramidUsed() == false) {
       this->backCon->doNotUsePyramidalApproach();
   }
   this->backCon->setLevelNumber(this->con->getLevelNumber());
   this->backCon->setLevelToPerform(this->con->getLevelToPerform());

   this->backCon->setReferencePyramid(this->con->getFloatingPyramid());
   this->backCon->setFloatingPyramid(this->con->getReferencePyramid());

   this->backCon->AllocateMaskPyramid();
   this->backCon->AllocateActiveVoxelNumber();

   if(this->backCon->isPyramidUsed() && this->backCon->getInputReferenceMask()!=NULL)
   {
      reg_createMaskPyramid<T>(this->backCon->getInputReferenceMask(),
                               this->backCon->getMaskPyramid(),
                               this->backCon->getLevelNumber(),
                               this->backCon->getLevelToPerform(),
                               this->backCon->getActiveVoxelNumber());
   }
   else if(this->backCon->isPyramidUsed())
   {
      for(unsigned int l=0; l<this->backCon->getLevelToPerform(); ++l)
      {
         this->backCon->setActiveVoxelNumber(l,this->backCon->getReferencePyramid()[l]->nx*this->backCon->getReferencePyramid()[l]->ny*this->backCon->getReferencePyramid()[l]->nz);
         this->backCon->getMaskPyramid()[l]=(int *)calloc(this->backCon->getActiveVoxelNumber()[l],sizeof(int));
      }
   }
   else
   {
       reg_print_fct_error("reg_aladin_sym<T>::InitialiseRegistration()");
       reg_print_msg_error("A pyramid scheme must be used with reg_aladin");
       reg_exit();
   }
   //Platform
   this->backCon->getPlatform()->setGpuIdx(this->con->getPlatform()->getGpuIdx());

   if(this->alignCentreGravity && this->con->getAffineTransformation()==NULL)
   {
      if(!this->backCon->getInputReferenceMask() && !this->con->getInputReferenceMask()){
         reg_print_msg_error("The masks' centre of gravity can only be used when two masks are specified");
         reg_exit();
      }
      float referenceCentre[3]={0,0,0};
      float referenceCount=0;
      reg_tools_changeDatatype<float>(this->con->getInputReferenceMask());
      float *refMaskPtr=static_cast<float *>(this->con->getInputReferenceMask()->data);
      size_t refIndex=0;
      for(int z=0;z<this->con->getInputReferenceMask()->nz;++z){
         for(int y=0;y<this->con->getInputReferenceMask()->ny;++y){
            for(int x=0;x<this->con->getInputReferenceMask()->nx;++x){
               if(refMaskPtr[refIndex]!=0.f){
                  referenceCentre[0]+=x;
                  referenceCentre[1]+=y;
                  referenceCentre[2]+=z;
                  referenceCount++;
               }
               refIndex++;
            }
         }
      }
      referenceCentre[0]/=referenceCount;
      referenceCentre[1]/=referenceCount;
      referenceCentre[2]/=referenceCount;
      float refCOG[3];
      if(this->con->getInputReference()->sform_code>0)
         reg_mat44_mul(&(this->con->getInputReference()->sto_xyz),referenceCentre,refCOG);

      float floatingCentre[3]={0,0,0};
      float floatingCount=0;
      reg_tools_changeDatatype<float>(this->backCon->getInputReferenceMask());
      float *floMaskPtr=static_cast<float *>(this->backCon->getInputReferenceMask()->data);
      size_t floIndex=0;
      for(int z=0;z<this->backCon->getInputReferenceMask()->nz;++z){
         for(int y=0;y<this->backCon->getInputReferenceMask()->ny;++y){
            for(int x=0;x<this->backCon->getInputReferenceMask()->nx;++x){
               if(floMaskPtr[floIndex]!=0.f){
                  floatingCentre[0]+=x;
                  floatingCentre[1]+=y;
                  floatingCentre[2]+=z;
                  floatingCount++;
               }
               floIndex++;
            }
         }
      }
      floatingCentre[0]/=floatingCount;
      floatingCentre[1]/=floatingCount;
      floatingCentre[2]/=floatingCount;
      float floCOG[3];
      if(this->con->getInputFloating()->sform_code>0)
         reg_mat44_mul(&(this->con->getInputFloating()->sto_xyz),floatingCentre,floCOG);
      reg_mat44_eye(this->con->getTransformationMatrix());
      this->con->getTransformationMatrix()->m[0][3]=floCOG[0]-refCOG[0];
      this->con->getTransformationMatrix()->m[1][3]=floCOG[1]-refCOG[1];
      this->con->getTransformationMatrix()->m[2][3]=floCOG[2]-refCOG[2];
   }
   mat44 tmpMat = nifti_mat44_inverse(*(this->con->getTransformationMatrix()));
   this->backCon->AladinContent::setTransformationMatrix(tmpMat);
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::GetBackwardDeformationField()
{
   this->bAffineTransformation3DKernel->template castTo<AffineDeformationFieldKernel>()->calculate();
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::GetWarpedImage(int interp)
{
   reg_aladin<T>::GetWarpedImage(interp);
   this->GetBackwardDeformationField();
   this->bResamplingKernel->template castTo<ResampleImageKernel>()->calculate(interp, this->backCon->getWarpedPaddingValue());

}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::UpdateTransformationMatrix(int type){

  reg_aladin<T>::UpdateTransformationMatrix(type);

  // Update now the backward transformation matrix
  this->bBlockMatchingKernel->template castTo<BlockMatchingKernel>()->calculate();
  this->bOptimiseKernel->template castTo<OptimiseKernel>()->calculate(type);

#ifndef NDEBUG
   reg_mat44_disp(this->con->getTransformationMatrix(), (char *)"[NiftyReg DEBUG] pre-updated forward transformation matrix");
   reg_mat44_disp(this->backCon->getTransformationMatrix(), (char *)"[NiftyReg DEBUG] pre-updated backward transformation matrix");
#endif
   // Forward and backward matrix are inverted
   mat44 fInverted = nifti_mat44_inverse(*(this->con->getTransformationMatrix()));
   mat44 bInverted = nifti_mat44_inverse(*(this->backCon->getTransformationMatrix()));

   // We average the forward and inverted backward matrix
   this->con->setTransformationMatrix(reg_mat44_avg2(this->con->getTransformationMatrix(), &bInverted));
   // We average the inverted forward and backward matrix
   this->backCon->setTransformationMatrix(reg_mat44_avg2(&fInverted, this->backCon->getTransformationMatrix()));
   for(int i=0;i<3;++i){
      this->con->getTransformationMatrix()->m[3][i]=0.f;
      this->backCon->getTransformationMatrix()->m[3][i]=0.f;
   }
   this->con->getTransformationMatrix()->m[3][3]=1.f;
   this->backCon->getTransformationMatrix()->m[3][3]=1.f;
#ifndef NDEBUG
   reg_mat44_disp(this->con->getTransformationMatrix(), (char *)"[NiftyReg DEBUG] updated forward transformation matrix");
   reg_mat44_disp(this->backCon->getTransformationMatrix(), (char *)"[NiftyReg DEBUG] updated backward transformation matrix");
#endif
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::InitCurrentLevel(unsigned int cl)
{
   reg_aladin<T>::InitCurrentLevel(cl);
   this->backCon->setCurrentReference(this->backCon->getReferencePyramid()[cl]);
   this->backCon->setCurrentReferenceMask(this->backCon->getMaskPyramid()[cl], this->backCon->getActiveVoxelNumber()[cl]);
   this->backCon->setTransformationMatrix(this->backCon->getTransformationMatrix());
   this->backCon->setCurrentFloating(this->backCon->getFloatingPyramid()[cl]);
   this->backCon->AllocateWarped();
   this->backCon->AllocateDeformationField();
   this->backCon->InitBlockMatchingParams();
}
/* *************************************************************** */
//template <class T>
//void reg_aladin_sym<T>::ClearCurrentImagePyramid()
//{
//   reg_aladin<T>::ClearCurrentImagePyramid();
//   this->backCon->ClearCurrentImagePyramid(this->currentLevel);
//}
/* *************************************************************** */
template<class T>
void reg_aladin_sym<T>::ClearBlockMatchingParams()
{
    reg_aladin<T>::ClearBlockMatchingParams();
    this->backCon->ClearBlockMatchingParams();
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::CreateKernels()
{
  reg_aladin<T>::CreateKernels();
  this->bAffineTransformation3DKernel = this->backCon->getPlatform()->createKernel (AffineDeformationFieldKernel::getName(), this->backCon);
  this->bBlockMatchingKernel = this->backCon->getPlatform()->createKernel(BlockMatchingKernel::getName(), this->backCon);
  if (this->backCon->AladinContent::getBlockMatchingParams() != NULL) {
    this->bResamplingKernel = this->backCon->getPlatform()->createKernel(ResampleImageKernel::getName(), this->backCon);
    this->bOptimiseKernel = this->backCon->getPlatform()->createKernel(OptimiseKernel::getName(), this->backCon);
  } else {
    this->bResamplingKernel = NULL;
    this->bOptimiseKernel = NULL;
  }
}
/* *************************************************************** */
//template <class T>
//void reg_aladin_sym<T>::ClearAladinContent()
//{
//  reg_aladin<T>::clearAladinContent();
//  delete this->backCon;
//}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::ClearKernels()
{
  reg_aladin<T>::ClearKernels();
  if(this->bResamplingKernel != NULL) {
    delete this->bResamplingKernel;
    this->bResamplingKernel = NULL;
  }
  if(this->bAffineTransformation3DKernel != NULL) {
  delete this->bAffineTransformation3DKernel;
  this->bAffineTransformation3DKernel = NULL;
  }
  if(this->bBlockMatchingKernel != NULL) {
  delete this->bBlockMatchingKernel;
  this->bBlockMatchingKernel = NULL;
  }
  if(this->bOptimiseKernel != NULL) {
  delete this->bOptimiseKernel;
  this->bOptimiseKernel = NULL;
  }
}
/* *************************************************************** */
template<class T>
void reg_aladin_sym<T>::AllocateImages()
{
    reg_aladin<T>::AllocateImages();
    this->backCon->AllocateWarped();
    this->backCon->AllocateDeformationField();
}
/* *************************************************************** */
template<class T>
void reg_aladin_sym<T>::ClearAllocatedImages()
{
    reg_aladin<T>::ClearAllocatedImages();
    this->backCon->ClearWarped();
    this->backCon->ClearDeformationField();
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::DebugPrintLevelInfoStart()
{
   char text[255];
   sprintf(text, "Current level %i / %i", this->currentLevel+1, this->con->getLevelNumber());
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
   if(this->con->getCurrentReference()->nz==1){
      reg_print_info(this->executableName, "Block size = [4 4 1]");
   }
   else reg_print_info(this->executableName, "Block size = [4 4 4]");
   reg_print_info(this->executableName, "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
   sprintf(text, "Forward Block number = [%i %i %i]", this->con->getBlockMatchingParams()->blockNumber[0],
          this->con->getBlockMatchingParams()->blockNumber[1], this->con->getBlockMatchingParams()->blockNumber[2]);
   reg_print_info(this->executableName, text);
   sprintf(text, "Backward Block number = [%i %i %i]", this->backCon->getBlockMatchingParams()->blockNumber[0],
          this->backCon->getBlockMatchingParams()->blockNumber[1], this->backCon->getBlockMatchingParams()->blockNumber[2]);
   reg_print_info(this->executableName, text);
   reg_mat44_disp(this->con->getTransformationMatrix(),
                  (char *)"[reg_aladin_sym] Initial forward transformation matrix:");
   reg_mat44_disp(this->backCon->getTransformationMatrix(),
                  (char *)"[reg_aladin_sym] Initial backward transformation matrix:");
   reg_print_info(this->executableName, "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");

}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::DebugPrintLevelInfoEnd()
{
   reg_mat44_disp(this->con->getTransformationMatrix(), (char *)"[reg_aladin_sym] Final forward transformation matrix:");
   reg_mat44_disp(this->backCon->getTransformationMatrix(), (char *)"[reg_aladin_sym] Final backward transformation matrix:");
}
/* *************************************************************** */
#endif //REG_ALADIN_SYM_CPP
