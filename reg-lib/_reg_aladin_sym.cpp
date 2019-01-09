#ifndef _REG_ALADIN_SYM_CPP
#define _REG_ALADIN_SYM_CPP

#include "_reg_aladin_sym.h"
#include "_reg_maths_eigen.h"

/* *************************************************************** */
template <class T>
reg_aladin_sym<T>::reg_aladin_sym ()
   :reg_aladin<T>::reg_aladin()
{
   this->executableName=(char*) "reg_aladin_sym";

   this->InputFloatingMask=NULL;
   this->FloatingMaskPyramid=NULL;
   this->BackwardActiveVoxelNumber=NULL;

   this->BackwardTransformationMatrix=new mat44;

   this->bAffineTransformation3DKernel = NULL;
   this->bConvolutionKernel=NULL;
   this->bBlockMatchingKernel=NULL;
   this->bOptimiseKernel=NULL;
   this->bResamplingKernel=NULL;

   this->backCon = NULL;
   this->BackwardBlockMatchingParams=NULL;

   this->FloatingUpperThreshold=std::numeric_limits<T>::max();
   this->FloatingLowerThreshold=-std::numeric_limits<T>::max();

#ifndef NDEBUG
   reg_print_msg_debug("reg_aladin_sym constructor called");
#endif

}
/* *************************************************************** */
template <class T>
reg_aladin_sym<T>::~reg_aladin_sym()
{
   if(this->BackwardTransformationMatrix!=NULL)
      delete this->BackwardTransformationMatrix;
   this->BackwardTransformationMatrix=NULL;

   if(this->FloatingMaskPyramid!=NULL)
   {
      for(unsigned int i=0; i<this->LevelsToPerform; ++i)
      {
         if(this->FloatingMaskPyramid[i]!=NULL)
         {
           if(this->FloatingMaskPyramid!=NULL)
             free(this->FloatingMaskPyramid[i]);
            this->FloatingMaskPyramid[i]=NULL;
         }
      }
      free(this->FloatingMaskPyramid);
      this->FloatingMaskPyramid=NULL;
   }
   if(this->BackwardActiveVoxelNumber!=NULL)
     free(this->BackwardActiveVoxelNumber);
   this->BackwardActiveVoxelNumber=NULL;

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
   this->FloatingMaskPyramid = (int **) malloc(this->LevelsToPerform*sizeof(int *));
   this->BackwardActiveVoxelNumber= (int *)malloc(this->LevelsToPerform*sizeof(int));
   if (this->InputFloatingMask!=NULL)
   {
      reg_createMaskPyramid<T>(this->InputFloatingMask,
                               this->FloatingMaskPyramid,
                               this->NumberOfLevels,
                               this->LevelsToPerform,
                               this->BackwardActiveVoxelNumber);
   }
   else
   {
      for(unsigned int l=0; l<this->LevelsToPerform; ++l)
      {
         this->BackwardActiveVoxelNumber[l]=this->FloatingPyramid[l]->nx*this->FloatingPyramid[l]->ny*this->FloatingPyramid[l]->nz;
         this->FloatingMaskPyramid[l]=(int *)calloc(this->BackwardActiveVoxelNumber[l],sizeof(int));
      }
   }

   // CHECK THE THRESHOLD VALUES TO UPDATE THE MASK
   if(this->FloatingUpperThreshold!=std::numeric_limits<T>::max())
   {
      for(unsigned int l=0; l<this->LevelsToPerform; ++l)
      {
         T *refPtr = static_cast<T *>(this->FloatingPyramid[l]->data);
         int *mskPtr = this->FloatingMaskPyramid[l];
         size_t removedVoxel=0;
         for(size_t i=0;
               i<(size_t)this->FloatingPyramid[l]->nx*this->FloatingPyramid[l]->ny*this->FloatingPyramid[l]->nz;
               ++i)
         {
            if(mskPtr[i]>-1)
            {
               if(refPtr[i]>this->FloatingUpperThreshold)
               {
                  ++removedVoxel;
                  mskPtr[i]=-1;
               }
            }
         }
         this->BackwardActiveVoxelNumber[l] -= removedVoxel;
      }
   }
   if(this->FloatingLowerThreshold!=-std::numeric_limits<T>::max())
   {
      for(unsigned int l=0; l<this->LevelsToPerform; ++l)
      {
         T *refPtr = static_cast<T *>(this->FloatingPyramid[l]->data);
         int *mskPtr = this->FloatingMaskPyramid[l];
         size_t removedVoxel=0;
         for(size_t i=0;
               i<(size_t)this->FloatingPyramid[l]->nx*this->FloatingPyramid[l]->ny*this->FloatingPyramid[l]->nz;
               ++i)
         {
            if(mskPtr[i]>-1)
            {
               if(refPtr[i]<this->FloatingLowerThreshold)
               {
                  ++removedVoxel;
                  mskPtr[i]=-1;
               }
            }
         }
         this->BackwardActiveVoxelNumber[l] -= removedVoxel;
      }
   }

   if(this->AlignCentreMass==1 && this->InputTransformName==NULL)
   {
      if(!this->InputReferenceMask && !this->InputFloatingMask){
         reg_print_msg_error("The masks' centre of mass can only be used when two masks are specified");
         reg_exit();
      }
      float referenceCentre[3]={0,0,0};
      float referenceCount=0;
      reg_tools_changeDatatype<float>(this->InputReferenceMask);
      float *refMaskPtr=static_cast<float *>(this->InputReferenceMask->data);
      size_t refIndex=0;
      for(int z=0;z<this->InputReferenceMask->nz;++z){
         for(int y=0;y<this->InputReferenceMask->ny;++y){
            for(int x=0;x<this->InputReferenceMask->nx;++x){
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
      if(this->InputReference->sform_code>0)
         reg_mat44_mul(&(this->InputReference->sto_xyz),referenceCentre,refCOG);

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
      if(this->InputFloating->sform_code>0)
         reg_mat44_mul(&(this->InputFloating->sto_xyz),floatingCentre,floCOG);
      reg_mat44_eye(this->TransformationMatrix);
      this->TransformationMatrix->m[0][3]=floCOG[0]-refCOG[0];
      this->TransformationMatrix->m[1][3]=floCOG[1]-refCOG[1];
      this->TransformationMatrix->m[2][3]=floCOG[2]-refCOG[2];
   }
   *(this->BackwardTransformationMatrix) = nifti_mat44_inverse(*(this->TransformationMatrix));

}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::GetBackwardDeformationField()
{
   this->bAffineTransformation3DKernel->template castTo<AffineDeformationFieldKernel>()->calculate();
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::GetWarpedImage(int interp, float padding)
{
   reg_aladin<T>::GetWarpedImage(interp, padding);
   this->GetBackwardDeformationField();
   this->bResamplingKernel->template castTo<ResampleImageKernel>()->calculate(interp, padding);

}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::UpdateTransformationMatrix(int type){

  reg_aladin<T>::UpdateTransformationMatrix(type);

  // Update now the backward transformation matrix
  this->bBlockMatchingKernel->template castTo<BlockMatchingKernel>()->calculate();
  this->bOptimiseKernel->template castTo<OptimiseKernel>()->calculate(type);

#ifndef NDEBUG
   reg_mat44_disp(this->TransformationMatrix, (char *)"[NiftyReg DEBUG] pre-updated forward transformation matrix");
   reg_mat44_disp(this->BackwardTransformationMatrix, (char *)"[NiftyReg DEBUG] pre-updated backward transformation matrix");
#endif
   // Forward and backward matrix are inverted
   mat44 fInverted = nifti_mat44_inverse(*(this->TransformationMatrix));
   mat44 bInverted = nifti_mat44_inverse(*(this->BackwardTransformationMatrix));

   // We average the forward and inverted backward matrix
   *(this->TransformationMatrix)=reg_mat44_avg2(this->TransformationMatrix, &bInverted );
   // We average the inverted forward and backward matrix
   *(this->BackwardTransformationMatrix)=reg_mat44_avg2(&fInverted, this->BackwardTransformationMatrix );
   for(int i=0;i<3;++i){
      this->TransformationMatrix->m[3][i]=0.f;
      this->BackwardTransformationMatrix->m[3][i]=0.f;
   }
   this->TransformationMatrix->m[3][3]=1.f;
   this->BackwardTransformationMatrix->m[3][3]=1.f;
#ifndef NDEBUG
   reg_mat44_disp(this->TransformationMatrix, (char *)"[NiftyReg DEBUG] updated forward transformation matrix");
   reg_mat44_disp(this->BackwardTransformationMatrix, (char *)"[NiftyReg DEBUG] updated backward transformation matrix");
#endif
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::initAladinContent(nifti_image *ref,
                        nifti_image *flo,
                        int *mask,
                        mat44 *transMat,
                        size_t bytes)
{
   reg_aladin<T>::initAladinContent(ref,
                               flo,
                               mask,
                               transMat,
                               bytes);

  if (this->platformCode == NR_PLATFORM_CPU)
  this->backCon = new AladinContent(flo, ref, this->FloatingMaskPyramid[this->CurrentLevel],this->BackwardTransformationMatrix,bytes);
#ifdef _USE_CUDA
  else if (this->platformCode == NR_PLATFORM_CUDA)
  this->backCon = new CudaAladinContent(flo, ref, this->FloatingMaskPyramid[this->CurrentLevel],this->BackwardTransformationMatrix,bytes);
#endif
#ifdef _USE_OPENCL
  else if (this->platformCode == NR_PLATFORM_CL)
  this->backCon = new ClAladinContent(flo, ref, this->FloatingMaskPyramid[this->CurrentLevel],this->BackwardTransformationMatrix,bytes);
#endif
  this->BackwardBlockMatchingParams = backCon->AladinContent::getBlockMatchingParams();
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::initAladinContent(nifti_image *ref,
                        nifti_image *flo,
                        int *mask,
                        mat44 *transMat,
                        size_t bytes,
                        unsigned int blockPercentage,
                        unsigned int inlierLts,
                        unsigned int blockStepSize)
{
    reg_aladin<T>::initAladinContent(ref,
                               flo,
                               mask,
                               transMat,
                               bytes,
                               blockPercentage,
                               inlierLts,
                               blockStepSize);

  if (this->platformCode == NR_PLATFORM_CPU)
  this->backCon = new AladinContent(flo, ref, this->FloatingMaskPyramid[this->CurrentLevel],this->BackwardTransformationMatrix,bytes, blockPercentage, inlierLts, blockStepSize);
#ifdef _USE_CUDA
  else if (this->platformCode == NR_PLATFORM_CUDA)
  this->backCon = new CudaAladinContent(flo, ref, this->FloatingMaskPyramid[this->CurrentLevel],this->BackwardTransformationMatrix,bytes, blockPercentage, inlierLts, blockStepSize);
#endif
#ifdef _USE_OPENCL
  else if (this->platformCode == NR_PLATFORM_CL)
  this->backCon = new ClAladinContent(flo, ref, this->FloatingMaskPyramid[this->CurrentLevel],this->BackwardTransformationMatrix,bytes, blockPercentage, inlierLts, blockStepSize);
#endif
  this->BackwardBlockMatchingParams = backCon->AladinContent::getBlockMatchingParams();
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::ClearCurrentInputImage()
{
   reg_aladin<T>::ClearCurrentInputImage();
   if(this->FloatingMaskPyramid[this->CurrentLevel]!=NULL)
      free(this->FloatingMaskPyramid[this->CurrentLevel]);
   this->FloatingMaskPyramid[this->CurrentLevel]=NULL;
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::createKernels()
{
  reg_aladin<T>::createKernels();
  this->bAffineTransformation3DKernel = this->platform->createKernel (AffineDeformationFieldKernel::getName(), this->backCon);
  this->bBlockMatchingKernel = this->platform->createKernel(BlockMatchingKernel::getName(), this->backCon);
  this->bResamplingKernel = this->platform->createKernel(ResampleImageKernel::getName(), this->backCon);
  this->bOptimiseKernel = this->platform->createKernel(OptimiseKernel::getName(), this->backCon);
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::clearAladinContent()
{
  reg_aladin<T>::clearAladinContent();
  delete this->backCon;
}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::clearKernels()
{
  reg_aladin<T>::clearKernels();
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
   sprintf(text, "Current level %i / %i", this->CurrentLevel+1, this->NumberOfLevels);
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
   sprintf(text, "Forward Block number = [%i %i %i]", this->blockMatchingParams->blockNumber[0],
          this->blockMatchingParams->blockNumber[1], this->blockMatchingParams->blockNumber[2]);
   reg_print_info(this->executableName, text);
   sprintf(text, "Backward Block number = [%i %i %i]", this->BackwardBlockMatchingParams->blockNumber[0],
          this->BackwardBlockMatchingParams->blockNumber[1], this->BackwardBlockMatchingParams->blockNumber[2]);
   reg_print_info(this->executableName, text);
   reg_mat44_disp(this->TransformationMatrix,
                  (char *)"[reg_aladin_sym] Initial forward transformation matrix:");
   reg_mat44_disp(this->BackwardTransformationMatrix,
                  (char *)"[reg_aladin_sym] Initial backward transformation matrix:");
   reg_print_info(this->executableName, "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");

}
/* *************************************************************** */
template <class T>
void reg_aladin_sym<T>::DebugPrintLevelInfoEnd()
{
   reg_mat44_disp(this->TransformationMatrix, (char *)"[reg_aladin_sym] Final forward transformation matrix:");
   reg_mat44_disp(this->BackwardTransformationMatrix, (char *)"[reg_aladin_sym] Final backward transformation matrix:");
}
/* *************************************************************** */
#endif //REG_ALADIN_SYM_CPP
