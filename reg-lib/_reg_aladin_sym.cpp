#include "_reg_aladin_sym.h"
#include "_reg_maths_eigen.h"

/* *************************************************************** */
template <class T>
reg_aladin_sym<T>::reg_aladin_sym ()
   :reg_aladin<T>::reg_aladin()
{
   this->executableName=(char*) "reg_aladin_sym";

   this->InputFloatingMask=nullptr;
   this->FloatingMaskPyramid=nullptr;

   this->BackwardTransformationMatrix=new mat44;

   this->bAffineTransformation3DKernel = nullptr;
   this->bConvolutionKernel=nullptr;
   this->bBlockMatchingKernel=nullptr;
   this->bOptimiseKernel=nullptr;
   this->bResamplingKernel=nullptr;

   this->backCon = nullptr;
   this->BackwardBlockMatchingParams=nullptr;

   this->floatingUpperThreshold=std::numeric_limits<T>::max();
   this->floatingLowerThreshold=-std::numeric_limits<T>::max();

#ifndef NDEBUG
   reg_print_msg_debug("reg_aladin_sym constructor called");
#endif
}
/* *************************************************************** */
template <class T>
reg_aladin_sym<T>::~reg_aladin_sym()
{
   if(this->BackwardTransformationMatrix!=nullptr)
      delete this->BackwardTransformationMatrix;
   this->BackwardTransformationMatrix=nullptr;

   if(this->FloatingMaskPyramid!=nullptr)
   {
      for(unsigned int i=0; i<this->levelsToPerform; ++i)
      {
         if(this->FloatingMaskPyramid[i]!=nullptr)
         {
           if(this->FloatingMaskPyramid!=nullptr)
             free(this->FloatingMaskPyramid[i]);
            this->FloatingMaskPyramid[i]=nullptr;
         }
      }
      free(this->FloatingMaskPyramid);
      this->FloatingMaskPyramid=nullptr;
   }

#ifndef NDEBUG
   reg_print_msg_debug("reg_aladin_sym destructor called");
#endif
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::SetInputFloatingMask(nifti_image *m)
{
   this->InputFloatingMask = m;
   return;
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::InitialiseRegistration()
{
#ifndef NDEBUG
   reg_print_msg_debug("reg_aladin_sym::InitialiseRegistration() called");
#endif

   reg_aladin<T>::InitialiseRegistration();
   this->FloatingMaskPyramid = (int **) malloc(this->levelsToPerform*sizeof(int *));
   if (this->InputFloatingMask!=nullptr)
   {
      reg_createMaskPyramid<T>(this->InputFloatingMask,
                               this->FloatingMaskPyramid,
                               this->numberOfLevels,
                               this->levelsToPerform);
   }
   else
   {
      for(unsigned int l=0; l<this->levelsToPerform; ++l)
      {
         const size_t voxelNumberBw = this->floatingPyramid[l]->nx * this->floatingPyramid[l]->ny * this->floatingPyramid[l]->nz;
         this->FloatingMaskPyramid[l]=(int *)calloc(voxelNumberBw,sizeof(int));
      }
   }

   // CHECK THE THRESHOLD VALUES TO UPDATE THE MASK
   if(this->floatingUpperThreshold!=std::numeric_limits<T>::max())
   {
      for(unsigned int l=0; l<this->levelsToPerform; ++l)
      {
         T *refPtr = static_cast<T *>(this->floatingPyramid[l]->data);
         int *mskPtr = this->FloatingMaskPyramid[l];
         size_t removedVoxel=0;
         for(size_t i=0;
               i<(size_t)this->floatingPyramid[l]->nx*this->floatingPyramid[l]->ny*this->floatingPyramid[l]->nz;
               ++i)
         {
            if(mskPtr[i]>-1)
            {
               if(refPtr[i]>this->floatingUpperThreshold)
               {
                  ++removedVoxel;
                  mskPtr[i]=-1;
               }
            }
         }
      }
   }
   if(this->floatingLowerThreshold!=-std::numeric_limits<T>::max())
   {
      for(unsigned int l=0; l<this->levelsToPerform; ++l)
      {
         T *refPtr = static_cast<T *>(this->floatingPyramid[l]->data);
         int *mskPtr = this->FloatingMaskPyramid[l];
         size_t removedVoxel=0;
         for(size_t i=0;
               i<(size_t)this->floatingPyramid[l]->nx*this->floatingPyramid[l]->ny*this->floatingPyramid[l]->nz;
               ++i)
         {
            if(mskPtr[i]>-1)
            {
               if(refPtr[i]<this->floatingLowerThreshold)
               {
                  ++removedVoxel;
                  mskPtr[i]=-1;
               }
            }
         }
      }
   }

   if(this->alignCentreMass==1 && this->inputTransformName==nullptr)
   {
      if(!this->inputReferenceMask && !this->InputFloatingMask){
         reg_print_msg_error("The masks' centre of mass can only be used when two masks are specified");
         reg_exit();
      }
      float referenceCentre[3]={0,0,0};
      float referenceCount=0;
      reg_tools_changeDatatype<float>(this->inputReferenceMask);
      float *refMaskPtr=static_cast<float *>(this->inputReferenceMask->data);
      size_t refIndex=0;
      for(int z=0;z<this->inputReferenceMask->nz;++z){
         for(int y=0;y<this->inputReferenceMask->ny;++y){
            for(int x=0;x<this->inputReferenceMask->nx;++x){
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
      if(this->inputReference->sform_code>0)
         reg_mat44_mul(&(this->inputReference->sto_xyz),referenceCentre,refCOG);

      float floatingCentre[3]={0,0,0};
      float floatingCount=0;
      reg_tools_changeDatatype<float>(this->InputFloatingMask);
      float *floMaskPtr=static_cast<float *>(this->InputFloatingMask->data);
      size_t floIndex=0;
      for(int z=0;z<this->InputFloatingMask->nz;++z){
         for(int y=0;y<this->InputFloatingMask->ny;++y){
            for(int x=0;x<this->InputFloatingMask->nx;++x){
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
      if(this->inputFloating->sform_code>0)
         reg_mat44_mul(&(this->inputFloating->sto_xyz),floatingCentre,floCOG);
      reg_mat44_eye(this->transformationMatrix);
      this->transformationMatrix->m[0][3]=floCOG[0]-refCOG[0];
      this->transformationMatrix->m[1][3]=floCOG[1]-refCOG[1];
      this->transformationMatrix->m[2][3]=floCOG[2]-refCOG[2];
   }
   *(this->BackwardTransformationMatrix) = nifti_mat44_inverse(*(this->transformationMatrix));

}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::GetBackwardDeformationField()
{
   this->bAffineTransformation3DKernel->template castTo<AffineDeformationFieldKernel>()->Calculate();
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::GetWarpedImage(int interp, float padding)
{
   reg_aladin<T>::GetWarpedImage(interp, padding);
   this->GetBackwardDeformationField();
   this->bResamplingKernel->template castTo<ResampleImageKernel>()->Calculate(interp, padding);

}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::UpdateTransformationMatrix(int type){

  reg_aladin<T>::UpdateTransformationMatrix(type);

  // Update now the backward transformation matrix
  this->bBlockMatchingKernel->template castTo<BlockMatchingKernel>()->Calculate();
  this->bOptimiseKernel->template castTo<OptimiseKernel>()->Calculate(type);

#ifndef NDEBUG
   reg_mat44_disp(this->transformationMatrix, (char *)"[NiftyReg DEBUG] pre-updated forward transformation matrix");
   reg_mat44_disp(this->BackwardTransformationMatrix, (char *)"[NiftyReg DEBUG] pre-updated backward transformation matrix");
#endif
   // Forward and backward matrix are inverted
   mat44 fInverted = nifti_mat44_inverse(*(this->transformationMatrix));
   mat44 bInverted = nifti_mat44_inverse(*(this->BackwardTransformationMatrix));

   // We average the forward and inverted backward matrix
   *(this->transformationMatrix)=reg_mat44_avg2(this->transformationMatrix, &bInverted );
   // We average the inverted forward and backward matrix
   *(this->BackwardTransformationMatrix)=reg_mat44_avg2(&fInverted, this->BackwardTransformationMatrix );
   for(int i=0;i<3;++i){
      this->transformationMatrix->m[3][i]=0.f;
      this->BackwardTransformationMatrix->m[3][i]=0.f;
   }
   this->transformationMatrix->m[3][3]=1.f;
   this->BackwardTransformationMatrix->m[3][3]=1.f;
#ifndef NDEBUG
   reg_mat44_disp(this->transformationMatrix, (char *)"[NiftyReg DEBUG] updated forward transformation matrix");
   reg_mat44_disp(this->BackwardTransformationMatrix, (char *)"[NiftyReg DEBUG] updated backward transformation matrix");
#endif
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::InitAladinContent(nifti_image *ref,
                        nifti_image *flo,
                        int *mask,
                        mat44 *transMat,
                        size_t bytes,
                        unsigned int blockPercentage,
                        unsigned int inlierLts,
                        unsigned int blockStepSize)
{
    reg_aladin<T>::InitAladinContent(ref,
                               flo,
                               mask,
                               transMat,
                               bytes,
                               blockPercentage,
                               inlierLts,
                               blockStepSize);

  if (this->platformType == PlatformType::Cpu)
  this->backCon = new AladinContent(flo, ref, this->FloatingMaskPyramid[this->currentLevel],this->BackwardTransformationMatrix,bytes, blockPercentage, inlierLts, blockStepSize);
#ifdef _USE_CUDA
  else if (this->platformType == PlatformType::Cuda)
  this->backCon = new CudaAladinContent(flo, ref, this->FloatingMaskPyramid[this->currentLevel],this->BackwardTransformationMatrix,bytes, blockPercentage, inlierLts, blockStepSize);
#endif
#ifdef _USE_OPENCL
  else if (this->platformType == PlatformType::OpenCl)
  this->backCon = new ClAladinContent(flo, ref, this->FloatingMaskPyramid[this->currentLevel],this->BackwardTransformationMatrix,bytes, blockPercentage, inlierLts, blockStepSize);
#endif
  this->BackwardBlockMatchingParams = backCon->AladinContent::GetBlockMatchingParams();
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::DeallocateCurrentInputImage()
{
   reg_aladin<T>::DeallocateCurrentInputImage();
   if(this->FloatingMaskPyramid[this->currentLevel]!=nullptr)
      free(this->FloatingMaskPyramid[this->currentLevel]);
   this->FloatingMaskPyramid[this->currentLevel]=nullptr;
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::CreateKernels()
{
  reg_aladin<T>::CreateKernels();
  this->bAffineTransformation3DKernel = this->platform->CreateKernel (AffineDeformationFieldKernel::GetName(), this->backCon);
  this->bBlockMatchingKernel = this->platform->CreateKernel(BlockMatchingKernel::GetName(), this->backCon);
  this->bResamplingKernel = this->platform->CreateKernel(ResampleImageKernel::GetName(), this->backCon);
  this->bOptimiseKernel = this->platform->CreateKernel(OptimiseKernel::GetName(), this->backCon);
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::DeinitAladinContent()
{
  reg_aladin<T>::DeinitAladinContent();
  delete this->backCon;
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::DeallocateKernels()
{
  reg_aladin<T>::DeallocateKernels();
  delete this->bResamplingKernel;
  delete this->bAffineTransformation3DKernel;
  delete this->bBlockMatchingKernel;
  delete this->bOptimiseKernel;
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::DebugPrintLevelInfoStart()
{
   char text[255];
   sprintf(text, "Current level %i / %i", this->currentLevel+1, this->numberOfLevels);
   reg_print_info(this->executableName,text);
   sprintf(text, "reference image size: \t%ix%ix%i voxels\t%gx%gx%g mm",
           this->con->GetReference()->nx,
           this->con->GetReference()->ny,
           this->con->GetReference()->nz,
           this->con->GetReference()->dx,
           this->con->GetReference()->dy,
           this->con->GetReference()->dz);
   reg_print_info(this->executableName,text);
   sprintf(text, "floating image size: \t%ix%ix%i voxels\t%gx%gx%g mm",
           this->con->GetFloating()->nx,
           this->con->GetFloating()->ny,
           this->con->GetFloating()->nz,
           this->con->GetFloating()->dx,
           this->con->GetFloating()->dy,
           this->con->GetFloating()->dz);
   reg_print_info(this->executableName,text);
   if(this->con->GetReference()->nz==1){
      reg_print_info(this->executableName, "Block size = [4 4 1]");
   }
   else reg_print_info(this->executableName, "Block size = [4 4 4]");
   reg_print_info(this->executableName, "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
   sprintf(text, "Forward Block number = [%i %i %i]", this->blockMatchingParams->blockNumber[0],
          this->blockMatchingParams->blockNumber[1], this->blockMatchingParams->blockNumber[2]);
   reg_print_info(this->executableName, text);
   sprintf(text, "Backward Block number = [%i %i %i]", this->BackwardBlockMatchingParams->blockNumber[0],
          this->BackwardBlockMatchingParams->blockNumber[1], this->BackwardBlockMatchingParams->blockNumber[2]);
   reg_print_info(this->executableName, text);
   reg_mat44_disp(this->transformationMatrix,
                  (char *)"[reg_aladin_sym] Initial forward transformation matrix:");
   reg_mat44_disp(this->BackwardTransformationMatrix,
                  (char *)"[reg_aladin_sym] Initial backward transformation matrix:");
   reg_print_info(this->executableName, "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");

}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::DebugPrintLevelInfoEnd()
{
   reg_mat44_disp(this->transformationMatrix, (char *)"[reg_aladin_sym] Final forward transformation matrix:");
   reg_mat44_disp(this->BackwardTransformationMatrix, (char *)"[reg_aladin_sym] Final backward transformation matrix:");
}
/* *************************************************************** */
template class reg_aladin_sym<float>;
