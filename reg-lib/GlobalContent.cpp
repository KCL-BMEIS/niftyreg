#include "GlobalContent.h"

using namespace std;

/* *************************************************************** */
GlobalContent::GlobalContent() : GlobalContent(NR_PLATFORM_CPU,1,1)
{
}
/* *************************************************************** */
GlobalContent::GlobalContent(int platformCodeIn) : GlobalContent(platformCodeIn,1,1)
{
}
/* *************************************************************** */
GlobalContent::GlobalContent(int platformCodeIn, int refTimePoint,int floTimePoint)
{
    //Little check
    if(refTimePoint != floTimePoint) {
        reg_print_fct_error("GlobalContent::GlobalContent(int platformCodeIn, int refTimePoint,int floTimePoint)");
        reg_print_msg_error("The reference and floating images must have the save number of timepoints");
        reg_exit();
    }
    this->inputReference=NULL; // pointer to external
    this->inputFloating=NULL; // pointer to external
    this->maskImage=NULL; // pointer to external
    this->affineTransformation=NULL;  // pointer to external

    this->nbRefTimePoint = refTimePoint;
    this->nbFloTimePoint = floTimePoint;
    this->referenceSmoothingSigma=0.;
    this->floatingSmoothingSigma=0.;
    this->robustRange = false;
    this->referenceThresholdUp=new float[refTimePoint];
    this->referenceThresholdLow=new float[refTimePoint];
    this->floatingThresholdUp=new float[floTimePoint];
    this->floatingThresholdLow=new float[floTimePoint];
    for(int i=0; i<refTimePoint; i++) {
       this->referenceThresholdUp[i]=std::numeric_limits<float>::max();
       this->referenceThresholdLow[i]=-std::numeric_limits<float>::max();
    }
    for(int i=0; i<floTimePoint; i++) {
       this->floatingThresholdUp[i]=std::numeric_limits<float>::max();
       this->floatingThresholdLow[i]=-std::numeric_limits<float>::max();
    }
    //
    this->usePyramid=true;
    this->levelNumber=3;
    this->levelToPerform=0;
    //
    this->referencePyramid=NULL;
    this->floatingPyramid=NULL;
    this->maskPyramid=NULL;
    this->activeVoxelNumber=NULL;
    //
    this->currentReference = NULL;
    this->refMatrix_xyz = NULL;
    this->refMatrix_ijk = NULL;
    this->currentReferenceMask = NULL;
    this->currentFloating = NULL;
    this->floMatrix_xyz = NULL;
    this->floMatrix_ijk = NULL;
    this->currentDeformationField = NULL;
    this->currentWarped = NULL;
    this->warpedPaddingValue=std::numeric_limits<float>::quiet_NaN();
    //Platform
    this->platform = new Platform(platformCodeIn);
    //this->platform = NULL;
#ifndef NDEBUG
   reg_print_fct_debug("GlobalContent::GlobalContent(int platformCodeIn, int refTimePoint,int floTimePoint)");
#endif
}
/* *************************************************************** */
GlobalContent::~GlobalContent()
{
   //Py
   if(this->referencePyramid!=NULL)
   {
       if(this->usePyramid)
       {
         for(unsigned int i=0; i<this->levelToPerform; i++)
         {
            if(referencePyramid[i]!=NULL)
            {
               nifti_image_free(referencePyramid[i]);
               referencePyramid[i]=NULL;
            }
         }
       }
       else
       {
           if(referencePyramid[0]!=NULL)
           {
              nifti_image_free(referencePyramid[0]);
              referencePyramid[0]=NULL;
           }
       }
      free(this->referencePyramid);
      this->referencePyramid=NULL;
   }
   if(this->maskPyramid!=NULL)
   {
       if(this->usePyramid)
       {
         for(unsigned int i=0; i<this->levelToPerform; i++)
         {
            if(this->maskPyramid[i]!=NULL)
            {
               free(this->maskPyramid[i]);
               this->maskPyramid[i]=NULL;
            }
         }
       }
       else
       {
           if(this->maskPyramid[0]!=NULL)
           {
              free(this->maskPyramid[0]);
              this->maskPyramid[0]=NULL;
           }
       }
      free(this->maskPyramid);
      this->maskPyramid=NULL;
   }
   if(this->floatingPyramid!=NULL)
   {
       if(this->usePyramid)
       {
         for(unsigned int i=0; i<this->levelToPerform; i++)
         {
            if(floatingPyramid[i]!=NULL)
            {
               nifti_image_free(floatingPyramid[i]);
               floatingPyramid[i]=NULL;
            }
         }
       }
       else {
           if(floatingPyramid[0]!=NULL)
           {
              nifti_image_free(floatingPyramid[0]);
              floatingPyramid[0]=NULL;
           }
       }
      free(this->floatingPyramid);
      this->floatingPyramid=NULL;
   }
   if(this->activeVoxelNumber!=NULL)
   {
      free(activeVoxelNumber);
      this->activeVoxelNumber=NULL;
   }
   if(this->referenceThresholdUp!=NULL)
   {
      delete[] this->referenceThresholdUp;
      this->referenceThresholdUp=NULL;
   }
   if(this->referenceThresholdLow!=NULL)
   {
      delete[] this->referenceThresholdLow;
      this->referenceThresholdLow=NULL;
   }
   if(this->floatingThresholdUp!=NULL)
   {
      delete[] this->floatingThresholdUp;
      this->floatingThresholdUp=NULL;
   }
   if(this->floatingThresholdLow!=NULL)
   {
      delete[] this->floatingThresholdLow;
      this->floatingThresholdLow=NULL;
   }
   //
   ClearWarped();
   ClearDeformationField();
   ClearCurrentInputImages();
   if(this->platform != NULL) {
       delete this->platform;
       this->platform = NULL;
   }
#ifndef NDEBUG
   reg_print_fct_debug("GlobalContent::~GlobalContent()");
#endif
}
/* *************************************************************** */
void GlobalContent::InitialiseGlobalContent()
{
    //Py
    unsigned int pyramidalLevelNumber=1;
    if(this->usePyramid == true) {
        pyramidalLevelNumber = this->levelToPerform;
    }
    this->referencePyramid = (nifti_image **)malloc(pyramidalLevelNumber*sizeof(nifti_image *));
    this->floatingPyramid = (nifti_image **)malloc(pyramidalLevelNumber*sizeof(nifti_image *));
    this->maskPyramid = (int **)malloc(pyramidalLevelNumber*sizeof(int *));
    this->activeVoxelNumber= (int *)malloc(pyramidalLevelNumber*sizeof(int));
    //
    // Update the input images threshold if required
    if(this->robustRange==true){
       // Create a copy of the reference image to extract the robust range
       nifti_image *temp_reference = nifti_copy_nim_info(this->inputReference);
       temp_reference->data = (void *)malloc(temp_reference->nvox * temp_reference->nbyper);
       memcpy(temp_reference->data, this->inputReference->data,temp_reference->nvox * temp_reference->nbyper);
       reg_tools_changeDatatype<float>(temp_reference);
       // Extract the robust range of the reference image
       float *refDataPtr = static_cast<float *>(temp_reference->data);
       reg_heapSort(refDataPtr, temp_reference->nvox);
       // Update the reference threshold values if no value has been setup by the user
       if(this->referenceThresholdLow[0]==-std::numeric_limits<float>::max())
          this->referenceThresholdLow[0] = refDataPtr[(int)reg_round((float)temp_reference->nvox*0.02f)];
       if(this->referenceThresholdUp[0]==std::numeric_limits<float>::max())
          this->referenceThresholdUp[0] = refDataPtr[(int)reg_round((float)temp_reference->nvox*0.98f)];
       // Free the temporarly allocated image
       nifti_image_free(temp_reference);

       // Create a copy of the floating image to extract the robust range
       nifti_image *temp_floating = nifti_copy_nim_info(this->inputFloating);
       temp_floating->data = (void *)malloc(temp_floating->nvox * temp_floating->nbyper);
       memcpy(temp_floating->data, this->inputFloating->data,temp_floating->nvox * temp_floating->nbyper);
       reg_tools_changeDatatype<float>(temp_floating);
       // Extract the robust range of the floating image
       float *floDataPtr = static_cast<float *>(temp_floating->data);
       reg_heapSort(floDataPtr, temp_floating->nvox);
       // Update the floating threshold values if no value has been setup by the user
       if(this->floatingThresholdLow[0]==-std::numeric_limits<float>::max())
          this->floatingThresholdLow[0] = floDataPtr[(int)reg_round((float)temp_floating->nvox*0.02f)];
       if(this->floatingThresholdUp[0]==std::numeric_limits<float>::max())
          this->floatingThresholdUp[0] = floDataPtr[(int)reg_round((float)temp_floating->nvox*0.98f)];
       // Free the temporarly allocated image
       nifti_image_free(temp_floating);
    }
    //
    // FINEST LEVEL OF REGISTRATION
    if(this->usePyramid)
    {
       reg_createImagePyramid<float>(this->inputReference, this->referencePyramid, this->levelNumber, this->levelToPerform);
       reg_createImagePyramid<float>(this->inputFloating, this->floatingPyramid, this->levelNumber, this->levelToPerform);
       if (this->maskImage!=NULL)
          reg_createMaskPyramid<float>(this->maskImage, this->maskPyramid, this->levelNumber, this->levelToPerform, this->activeVoxelNumber);
       else
       {
          for(unsigned int l=0; l<this->levelToPerform; ++l)
          {
             this->activeVoxelNumber[l]=this->referencePyramid[l]->nx*this->referencePyramid[l]->ny*this->referencePyramid[l]->nz;
             this->maskPyramid[l]=(int *)calloc(this->activeVoxelNumber[l],sizeof(int));
          }
       }
    }
    else
    {
       reg_createImagePyramid<float>(this->inputReference, this->referencePyramid, 1, 1);
       reg_createImagePyramid<float>(this->inputFloating, this->floatingPyramid, 1, 1);
       if (this->maskImage!=NULL)
          reg_createMaskPyramid<float>(this->maskImage, this->maskPyramid, 1, 1, this->activeVoxelNumber);
       else
       {
          this->activeVoxelNumber[0]=this->referencePyramid[0]->nx*this->referencePyramid[0]->ny*this->referencePyramid[0]->nz;
          this->maskPyramid[0]=(int *)calloc(this->activeVoxelNumber[0],sizeof(int));
       }
    }
    //
    Kernel *convolutionKernel = this->platform->createKernel(ConvolutionKernel::getName(), NULL);
    // SMOOTH THE INPUT IMAGES IF REQUIRED
    for(unsigned int l=0; l<pyramidalLevelNumber; l++)
    {
       if(this->referenceSmoothingSigma!=0.0)
       {
          bool *active = new bool[this->referencePyramid[l]->nt];
          float *sigma = new float[this->referencePyramid[l]->nt];
          active[0]=true;
          for(int i=1; i<this->referencePyramid[l]->nt; ++i)
             active[i]=false;
          sigma[0]=this->referenceSmoothingSigma;
          convolutionKernel->castTo<ConvolutionKernel>()->calculate(this->referencePyramid[l], sigma, 0, NULL, active);
          delete []active;
          delete []sigma;
       }
       if(this->floatingSmoothingSigma!=0.0)
       {
          // Only the first image is smoothed
          bool *active = new bool[this->floatingPyramid[l]->nt];
          float *sigma = new float[this->floatingPyramid[l]->nt];
          active[0]=true;
          for(int i=1; i<this->floatingPyramid[l]->nt; ++i)
             active[i]=false;
          sigma[0]=this->floatingSmoothingSigma;
          convolutionKernel->castTo<ConvolutionKernel>()->calculate(this->floatingPyramid[l], sigma, 0, NULL, active);
          delete []active;
          delete []sigma;
       }
    }
    delete convolutionKernel;
    //
    // THRESHOLD THE INPUT IMAGES IF REQUIRED
    for(unsigned int l=0; l<pyramidalLevelNumber; l++) {
       reg_thresholdImage<float>(this->referencePyramid[l],this->referenceThresholdLow, this->referenceThresholdUp);
       reg_thresholdImage<float>(this->floatingPyramid[l],this->floatingThresholdLow, this->floatingThresholdUp);
    }
}
/* *************************************************************** */
/* *************************************************************** */
/* *************************************************************** */
//getters
/* *************************************************************** */
/* *************************************************************** */
/* *************************************************************** */
nifti_image* GlobalContent::getInputReference() {
    return this->inputReference;
}
nifti_image* GlobalContent::getInputReferenceMask() {
    return this->maskImage;
}
nifti_image* GlobalContent::getInputFloating() {
    return this->inputFloating;
}
mat44* GlobalContent::getAffineTransformation() {
    return this->affineTransformation;
}

unsigned GlobalContent::getNbRefTimePoint() {
    return this->nbRefTimePoint;
}
unsigned GlobalContent::getNbFloTimePoint() {
    return this->nbFloTimePoint;
}

float GlobalContent::getReferenceSmoothingSigma() {
    return this->referenceSmoothingSigma;
}
float GlobalContent::getFloatingSmoothingSigma() {
    return this->floatingSmoothingSigma;
}
//
bool GlobalContent::getRobustRange() {
    return this->robustRange;
}
//
float* GlobalContent::getReferenceThresholdLow() {
    return this->referenceThresholdLow;
}
float* GlobalContent::getReferenceThresholdUp() {
    return this->referenceThresholdUp;
}
float* GlobalContent::getFloatingThresholdLow() {
    return this->floatingThresholdLow;
}
float* GlobalContent::getFloatingThresholdUp() {
    return this->floatingThresholdUp;
}
//
unsigned int GlobalContent::getLevelNumber() {
    return this->levelNumber;
}
unsigned int GlobalContent::getLevelToPerform() {
    return this->levelToPerform;
}
//
nifti_image* GlobalContent::getCurrentReference() {
    return this->currentReference;
}
mat44* GlobalContent::getCurrentReferenceMatrix_xyz() {
    return this->refMatrix_xyz;
}
nifti_image* GlobalContent::getCurrentFloating() {
    return this->currentFloating;
}
mat44* GlobalContent::getCurrentFloatingMatrix_xyz() {
    return this->floMatrix_xyz;
}
nifti_image* GlobalContent::getCurrentWarped(int datatype) {
    return this->currentWarped;
}
float GlobalContent::getWarpedPaddingValue() {
    return this->warpedPaddingValue;
}
nifti_image* GlobalContent::getCurrentDeformationField() {
    return this->currentDeformationField;
}
int* GlobalContent::getCurrentReferenceMask() {
    return this->currentReferenceMask;
}
bool GlobalContent::isPyramidUsed() {
    return this->usePyramid;
}
/////
nifti_image** GlobalContent::getReferencePyramid() {
    return this->referencePyramid;
}
nifti_image** GlobalContent::getFloatingPyramid() {
    return this->floatingPyramid;
}
int** GlobalContent::getMaskPyramid() {
    return this->maskPyramid;
}
int* GlobalContent::getActiveVoxelNumber() {
    return this->activeVoxelNumber;
}
Platform* GlobalContent::getPlatform() {
    return this->platform;
}
/* *************************************************************** */
/* *************************************************************** */
/* *************************************************************** */
//setters
/* *************************************************************** */
/* *************************************************************** */
/* *************************************************************** */
void GlobalContent::setInputReference(nifti_image *r) {
   this->inputReference = r;
}
void GlobalContent::setInputFloating(nifti_image *f) {
   this->inputFloating = f;
}
void GlobalContent::setInputReferenceMask(nifti_image *m) {
   this->maskImage = m;
}
void GlobalContent::setAffineTransformation(mat44 *a) {
   this->affineTransformation=a;
}
void GlobalContent::setNbRefTimePoint(unsigned ntp) {
    this->nbRefTimePoint = ntp;
}
void GlobalContent::setNbFloTimePoint(unsigned ntp) {
    this->nbFloTimePoint = ntp;
}
void GlobalContent::setReferenceSmoothingSigma(float s) {
   this->referenceSmoothingSigma = s;
}
void GlobalContent::setFloatingSmoothingSigma(float s) {
   this->floatingSmoothingSigma = s;
}
void GlobalContent::setRobustRange(bool rr) {
   this->robustRange = rr;
}
void GlobalContent::setReferenceThresholdUp(unsigned int i, float t) {
   this->referenceThresholdUp[i] = t;
}
void GlobalContent::setReferenceThresholdLow(unsigned int i, float t) {
   this->referenceThresholdLow[i] = t;
}
void GlobalContent::setFloatingThresholdUp(unsigned int i, float t) {
   this->floatingThresholdUp[i] = t;
}
void GlobalContent::setFloatingThresholdLow(unsigned int i, float t) {
   this->floatingThresholdLow[i] = t;
}
void GlobalContent::setLevelNumber(unsigned int l) {
   this->levelNumber = l;
}
void GlobalContent::setLevelToPerform(unsigned int l) {
   this->levelToPerform = l;
}
void GlobalContent::useRobustRange() {
   this->robustRange=true;
}
void GlobalContent::doNotUseRobustRange() {
   this->robustRange=false;
}
void GlobalContent::setWarpedPaddingValue(float p) {
   this->warpedPaddingValue = p;
}
void GlobalContent::doNotUsePyramidalApproach() {
   this->usePyramid=false;
}
//
void GlobalContent::setReferencePyramid(nifti_image** rp) {
    this->referencePyramid = rp;
}
void GlobalContent::setFloatingPyramid(nifti_image** fp) {
    this->floatingPyramid = fp;
}
void GlobalContent::setMaskPyramid(int** mp) {
    this->maskPyramid = mp;
}
void GlobalContent::setActiveVoxelNumber(int py, int avn) {
    this->activeVoxelNumber[py] = avn;
}
//
void GlobalContent::setCurrentReference(nifti_image* currentRefIn) {
    this->currentReference = currentRefIn;
    //mat44 tmpMat = (this->currentReference->sform_code > 0) ? (this->currentReference->sto_xyz) : (this->currentReference->qto_xyz);
    this->refMatrix_xyz = &((this->currentReference->sform_code > 0) ? (this->currentReference->sto_xyz) : (this->currentReference->qto_xyz));

    //tmpMat = (this->currentReference->sform_code > 0) ? (this->currentReference->sto_ijk) : (this->currentReference->qto_ijk);
    this->refMatrix_ijk = &((this->currentReference->sform_code > 0) ? (this->currentReference->sto_ijk) : (this->currentReference->qto_ijk));
}
void GlobalContent::setCurrentReferenceMask(int * currentRefMaskIn, size_t nvox) {
    this->currentReferenceMask = currentRefMaskIn;
}
void GlobalContent::setCurrentFloating(nifti_image* currentFloIn) {
    this->currentFloating = currentFloIn;
    this->floMatrix_ijk = &((this->currentFloating->sform_code > 0) ? (this->currentFloating->sto_ijk) :  (this->currentFloating->qto_ijk));
    this->floMatrix_xyz = &((this->currentFloating->sform_code > 0) ? (this->currentFloating->sto_xyz) :  (this->currentFloating->qto_xyz));
}
void GlobalContent::setCurrentDeformationField(nifti_image* currentDeformationFieldIn) {
    this->currentDeformationField = currentDeformationFieldIn;
}
void GlobalContent::setCurrentWarped(nifti_image* currentWarpedImageIn) {
    this->currentWarped = currentWarpedImageIn;
}
/* *************************************************************** */
void GlobalContent::CheckParameters()
{
   // CHECK THAT BOTH INPUT IMAGES ARE DEFINED
   if(this->inputReference==NULL)
   {
      reg_print_fct_error("GlobalContent::CheckParameters()");
      reg_print_msg_error("The reference image is not defined");
      reg_exit();
   }
   if(this->inputFloating==NULL)
   {
      reg_print_fct_error("GlobalContent::CheckParameters()");
      reg_print_msg_error("The floating image is not defined");
      reg_exit();
   }

   // CHECK THE MASK DIMENSION IF IT IS DEFINED
   if(this->maskImage!=NULL)
   {
      if(this->inputReference->nx != this->maskImage->nx ||
            this->inputReference->ny != this->maskImage->ny ||
            this->inputReference->nz != this->maskImage->nz )
      {
         reg_print_fct_error("GlobalContent::CheckParameters()");
         reg_print_msg_error("The reference and mask images have different dimension");
         reg_exit();
      }
   }

   // CHECK THE NUMBER OF LEVEL TO PERFORM
   if(this->levelToPerform>0)
   {
      this->levelToPerform=this->levelToPerform<this->levelNumber?this->levelToPerform:this->levelNumber;
   }
   else this->levelToPerform=this->levelNumber;
   if(this->levelToPerform==0 || this->levelToPerform>this->levelNumber)
      this->levelToPerform=this->levelNumber;

#ifndef NDEBUG
   reg_print_fct_debug("GlobalContent::CheckParameters");
#endif
}
/* *************************************************************** */
void GlobalContent::AllocateWarped()
{
   if(this->currentReference==NULL || this->currentFloating == NULL) {
      reg_print_fct_error("GlobalContent::AllocateWarped()");
      reg_print_msg_error("The reference image or floating image is not defined");
      reg_exit();
   }
   GlobalContent::ClearWarped();
   this->currentWarped = nifti_copy_nim_info(this->currentReference);
   this->currentWarped->dim[0]=this->currentWarped->ndim=this->currentFloating->ndim;
   this->currentWarped->dim[4]=this->currentWarped->nt=this->currentFloating->nt;
   this->currentWarped->pixdim[4]=this->currentWarped->dt=1.0;
   this->currentWarped->nvox =
      (size_t)this->currentWarped->nx *
      (size_t)this->currentWarped->ny *
      (size_t)this->currentWarped->nz *
      (size_t)this->currentWarped->nt;
   this->currentWarped->scl_slope=1.f;
   this->currentWarped->scl_inter=0.f;
   this->currentWarped->datatype = this->currentFloating->datatype;
   this->currentWarped->nbyper = this->currentFloating->nbyper;
   this->currentWarped->data = (void *)calloc(this->currentWarped->nvox, this->currentWarped->nbyper);
#ifndef NDEBUG
   reg_print_fct_debug("GlobalContent::AllocateWarped");
#endif
}
/* *************************************************************** */
void GlobalContent::ClearWarped()
{
   if(this->currentWarped!=NULL) {
       nifti_image_free(this->currentWarped);
       this->currentWarped=NULL;
   }
#ifndef NDEBUG
   reg_print_fct_debug("GlobalContent::ClearWarped");
#endif
}
/* *************************************************************** */
void GlobalContent::AllocateDeformationField()
{
   if(this->currentReference==NULL) {
      reg_print_fct_error("GlobalContent::AllocateDeformationField()");
      reg_print_msg_error("The reference image is not defined");
      reg_exit();
   }
   GlobalContent::ClearDeformationField();
   this->currentDeformationField = nifti_copy_nim_info(this->currentReference);
   this->currentDeformationField->dim[0]=this->currentDeformationField->ndim=5;
   this->currentDeformationField->dim[1]=this->currentDeformationField->nx=this->currentReference->nx;
   this->currentDeformationField->dim[2]=this->currentDeformationField->ny=this->currentReference->ny;
   this->currentDeformationField->dim[3]=this->currentDeformationField->nz=this->currentReference->nz;
   this->currentDeformationField->dim[4]=this->currentDeformationField->nt=1;
   this->currentDeformationField->pixdim[4]=this->currentDeformationField->dt=1.0;
   if(this->currentReference->nz==1)
      this->currentDeformationField->dim[5]=this->currentDeformationField->nu=2;
   else this->currentDeformationField->dim[5]=this->currentDeformationField->nu=3;
   this->currentDeformationField->pixdim[5]=this->currentDeformationField->du=1.0;
   this->currentDeformationField->dim[6]=this->currentDeformationField->nv=1;
   this->currentDeformationField->pixdim[6]=this->currentDeformationField->dv=1.0;
   this->currentDeformationField->dim[7]=this->currentDeformationField->nw=1;
   this->currentDeformationField->pixdim[7]=this->currentDeformationField->dw=1.0;
   this->currentDeformationField->nvox =
      (size_t)this->currentDeformationField->nx *
      (size_t)this->currentDeformationField->ny *
      (size_t)this->currentDeformationField->nz *
      (size_t)this->currentDeformationField->nt *
      (size_t)this->currentDeformationField->nu;
   //this->currentDeformationField->nbyper = sizeof(T);
   this->currentDeformationField->nbyper = sizeof(float);
   //if(sizeof(T)==sizeof(float))
      this->currentDeformationField->datatype = NIFTI_TYPE_FLOAT32;
   //else this->currentDeformationField->datatype = NIFTI_TYPE_FLOAT64;
   this->currentDeformationField->data = (void *)calloc(this->currentDeformationField->nvox,
                                       this->currentDeformationField->nbyper);
   this->currentDeformationField->intent_code=NIFTI_INTENT_VECTOR;
   memset(this->currentDeformationField->intent_name, 0, 16);
   strcpy(this->currentDeformationField->intent_name,"NREG_TRANS");
   this->currentDeformationField->intent_p1=DEF_FIELD;
   this->currentDeformationField->scl_slope=1.f;
   this->currentDeformationField->scl_inter=0.f;

#ifndef NDEBUG
   reg_print_fct_debug("GlobalContent::AllocateDeformationField");
#endif
}
/* *************************************************************** */
void GlobalContent::ClearDeformationField()
{

   if(this->currentDeformationField!=NULL) {
       nifti_image_free(this->currentDeformationField);
       this->currentDeformationField=NULL;
   }

#ifndef NDEBUG
   reg_print_fct_debug("GlobalContent::ClearDeformationField");
#endif
}
/* *************************************************************** */
void GlobalContent::AllocateMaskPyramid()
{
    unsigned int pyramidalLevelNumber=1;
    if(this->usePyramid == true) {
        pyramidalLevelNumber = this->levelToPerform;
    }
    this->maskPyramid = (int **)malloc(pyramidalLevelNumber*sizeof(int *));
 #ifndef NDEBUG
    reg_print_fct_debug("GlobalContent::AllocateMaskPyramid()");
 #endif
}
/* *************************************************************** */
void GlobalContent::ClearMaskPyramid()
{
    if(this->maskPyramid!=NULL)
    {
        if(this->usePyramid)
        {
          for(unsigned int i=0; i<this->levelToPerform; i++)
          {
             if(this->maskPyramid[i]!=NULL)
             {
                free(this->maskPyramid[i]);
                this->maskPyramid[i]=NULL;
             }
          }
        }
        else
        {
            if(this->maskPyramid[0]!=NULL)
            {
               free(this->maskPyramid[0]);
               this->maskPyramid[0]=NULL;
            }
        }
       free(this->maskPyramid);
       this->maskPyramid=NULL;
    }
#ifndef NDEBUG
   reg_print_fct_debug("GlobalContent::ClearMaskPyramid()");
#endif
}
/* *************************************************************** */
void GlobalContent::AllocateActiveVoxelNumber()
{
    unsigned int pyramidalLevelNumber=1;
    if(this->usePyramid == true) {
        pyramidalLevelNumber = this->levelToPerform;
    }
    this->activeVoxelNumber= (int *)malloc(pyramidalLevelNumber*sizeof(int));
#ifndef NDEBUG
    reg_print_fct_debug("GlobalContent::AllocateActiveVoxelNumber()");
#endif
}
/* *************************************************************** */
void GlobalContent::ClearActiveVoxelNumber()
{
    if(this->activeVoxelNumber!=NULL)
    {
       free(activeVoxelNumber);
       this->activeVoxelNumber=NULL;
    }
#ifndef NDEBUG
    reg_print_fct_debug("GlobalContent::ClearActiveVoxelNumber()");
#endif
}
/* *************************************************************** */
void GlobalContent::ClearCurrentInputImages()
{
    this->currentReference=NULL;
    this->currentReferenceMask=NULL;
    this->currentFloating=NULL;
#ifndef NDEBUG
    reg_print_fct_debug("GlobalContent::ClearCurrentInputImages()");
#endif
}
/* *************************************************************** */
void GlobalContent::ClearCurrentImagePyramid(int currentPyramidLevel)
{
    if(this->referencePyramid[currentPyramidLevel] != NULL) {
        nifti_image_free(this->referencePyramid[currentPyramidLevel]);
        this->referencePyramid[currentPyramidLevel]=NULL;
    }
    if(this->floatingPyramid[currentPyramidLevel] != NULL) {
        nifti_image_free(this->floatingPyramid[currentPyramidLevel]);
        this->floatingPyramid[currentPyramidLevel]=NULL;
    }
    if(this->maskPyramid[currentPyramidLevel] != NULL) {
        free(this->maskPyramid[currentPyramidLevel]);
        this->maskPyramid[currentPyramidLevel]=NULL;
    }

#ifndef NDEBUG
    reg_print_fct_debug("GlobalContent::ClearCurrentImagePyramid()");
#endif
}
/* *************************************************************** */
void GlobalContent::ClearThresholds()
{
   if(this->referenceThresholdUp!=NULL)
   {
      delete[] this->referenceThresholdUp;
      this->referenceThresholdUp=NULL;
   }
   if(this->referenceThresholdLow!=NULL)
   {
      delete[] this->referenceThresholdLow;
      this->referenceThresholdLow=NULL;
   }
   if(this->floatingThresholdUp!=NULL)
   {
      delete[] this->floatingThresholdUp;
      this->floatingThresholdUp=NULL;
   }
   if(this->floatingThresholdLow!=NULL)
   {
      delete[] this->floatingThresholdLow;
      this->floatingThresholdLow=NULL;
   }
#ifndef NDEBUG
    reg_print_fct_debug("GlobalContent::ClearThresholds()");
#endif
}
/* *************************************************************** */
bool GlobalContent::isCurrentComputationDoubleCapable()
{
    return true;
}
/* *************************************************************** */
//void GlobalContent::WarpFloatingImage(int interp)
//{
//    Kernel* resamplingKernel = this->platform->createKernel(ResampleImageKernel::getName(), this);
//    resamplingKernel->template castTo<ResampleImageKernel>()->calculate(interp, this->warpedPaddingValue);
//    delete resamplingKernel;
//}
