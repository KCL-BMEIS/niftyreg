/**
 * @file reg_measure.cpp
 * @author Marc Modat
 * @date 28/02/2014
 *
 *  Copyright (c) 2014-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_ReadWriteImage.h"
#include "_reg_resampling.h"
#include "_reg_tools.h"
#include "_reg_nmi.h"
#include "_reg_dti.h"
#include "_reg_ssd.h"
#include "_reg_mind.h"
#include "_reg_kld.h"
#include "_reg_lncc.h"

typedef struct
{
   char *refImageName;
   char *floImageName;
   char *refMaskImageName;
   char *floMaskImageName;
   int interpolation;
   float paddingValue;
   char *outFileName;
} PARAM;
typedef struct
{
   bool refImageFlag;
   bool floImageFlag;
   bool refMaskImageFlag;
   bool floMaskImageFlag;
   bool returnNMIFlag;
   bool returnSSDFlag;
   bool returnLNCCFlag;
   bool returnNCCFlag;
   bool returnMINDFlag;
   bool outFileFlag;
} FLAG;


void PetitUsage(char *exec)
{
   NR_INFO("Usage:\t" << exec << " -ref <referenceImageName> -flo <floatingImageName> [OPTIONS]");
   NR_INFO("\tSee the help for more details (-h)");
}

void Usage(char *exec)
{
   NR_INFO("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
   NR_INFO("Usage:\t" << exec << " -ref <filename> -flo <filename> [OPTIONS]");
   NR_INFO("\t-ref <filename>\tFilename of the reference image (mandatory)");
   NR_INFO("\t-flo <filename>\tFilename of the floating image (mandatory)");
   NR_INFO("\t\tNote that the floating image is resampled into the reference");
   NR_INFO("\t\timage space using the header informations");

   NR_INFO("* * OPTIONS * *");
   NR_INFO("\t-ncc\t\tReturns the NCC value");
   NR_INFO("\t-lncc\t\tReturns the LNCC value");
   NR_INFO("\t-nmi\t\tReturns the NMI value (64 bins are used)");
   NR_INFO("\t-ssd\t\tReturns the SSD value");
   NR_INFO("\n\t-out\t\tText file output where to store the value(s).\n\t\t\tThe stdout is used by default");
#ifdef _OPENMP
   int defaultOpenMPValue=omp_get_num_procs();
   if(getenv("OMP_NUM_THREADS")!=nullptr)
      defaultOpenMPValue=atoi(getenv("OMP_NUM_THREADS"));
   NR_INFO("\t-omp <int>\tNumber of threads to use with OpenMP. [" << defaultOpenMPValue << "/" << omp_get_num_procs() << "]");
#endif
   NR_INFO("\t--version\tPrint current version and exit (" << NR_VERSION << ")");
   NR_INFO("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
}

int main(int argc, char **argv)
{
   PARAM *param = (PARAM *)calloc(1,sizeof(PARAM));
   FLAG *flag = (FLAG *)calloc(1,sizeof(FLAG));

   param->interpolation=3; // Cubic spline interpolation used by default
   param->paddingValue=std::numeric_limits<float>::quiet_NaN();

#ifdef _OPENMP
   // Set the default number of threads
   int defaultOpenMPValue=omp_get_num_procs();
   if(getenv("OMP_NUM_THREADS")!=nullptr)
      defaultOpenMPValue=atoi(getenv("OMP_NUM_THREADS"));
   omp_set_num_threads(defaultOpenMPValue);
#endif

   /* read the input parameter */
   for(int i=1; i<argc; i++)
   {
      if(strcmp(argv[i],"-h")==0 ||
            strcmp(argv[i],"-H")==0 ||
            strcmp(argv[i],"-help")==0 ||
            strcmp(argv[i],"--help")==0 ||
            strcmp(argv[i],"-HELP")==0 ||
            strcmp(argv[i],"--HELP")==0 ||
            strcmp(argv[i],"-Help")==0 ||
            strcmp(argv[i],"--Help")==0
        )
      {
         Usage(argv[0]);
         return EXIT_SUCCESS;
      }
      else if(strcmp(argv[i], "-omp")==0 || strcmp(argv[i], "--omp")==0)
      {
#ifdef _OPENMP
         omp_set_num_threads(atoi(argv[++i]));
#else
         NR_WARN("NiftyReg has not been compiled with OpenMP, the \'-omp\' flag is ignored");
         ++i;
#endif
      }
      else if( strcmp(argv[i], "-version")==0 ||
            strcmp(argv[i], "-Version")==0 ||
            strcmp(argv[i], "-V")==0 ||
            strcmp(argv[i], "-v")==0 ||
            strcmp(argv[i], "--v")==0 ||
            strcmp(argv[i], "--version")==0)
      {
         NR_COUT << NR_VERSION << std::endl;
         return EXIT_SUCCESS;
      }
      else if((strcmp(argv[i],"-ref")==0) || (strcmp(argv[i],"-target")==0) ||
              (strcmp(argv[i],"--ref")==0))
      {
         param->refImageName=argv[++i];
         flag->refImageFlag=1;
      }
      else if((strcmp(argv[i],"-rmask")==0) ||
              (strcmp(argv[i],"--rmask")==0))
      {
         param->refMaskImageName=argv[++i];
         flag->refMaskImageFlag=1;
      }
      else if((strcmp(argv[i],"-flo")==0) || (strcmp(argv[i],"-source")==0) ||
              (strcmp(argv[i],"--flo")==0))
      {
         param->floImageName=argv[++i];
         flag->floImageFlag=1;
      }
      else if((strcmp(argv[i],"-fmask")==0) ||
              (strcmp(argv[i],"--fmask")==0))
      {
         param->floMaskImageName=argv[++i];
         flag->floMaskImageFlag=1;
      }
      else if(strcmp(argv[i], "-inter") == 0 ||
              (strcmp(argv[i],"--inter")==0))
      {
         param->interpolation=atoi(argv[++i]);
      }
      else if(strcmp(argv[i], "-pad") == 0 ||
              (strcmp(argv[i],"--pad")==0))
      {
         param->paddingValue=(float)atof(argv[++i]);
      }
      else if(strcmp(argv[i], "-ncc") == 0 ||
              (strcmp(argv[i],"--ncc")==0))
      {
         flag->returnNCCFlag=true;
      }
      else if(strcmp(argv[i], "-lncc") == 0 ||
              (strcmp(argv[i],"--lncc")==0))
      {
         flag->returnLNCCFlag=true;
      }
      else if(strcmp(argv[i], "-nmi") == 0 ||
              (strcmp(argv[i],"--nmi")==0))
      {
         flag->returnNMIFlag=true;
      }
      else if(strcmp(argv[i], "-ssd") == 0 ||
              (strcmp(argv[i],"--sdd")==0))
      {
         flag->returnSSDFlag=true;
      }
      else if(strcmp(argv[i], "-mind") == 0 ||
              (strcmp(argv[i],"--mind")==0))
      {
         flag->returnMINDFlag=true;
      }
      else if(strcmp(argv[i], "-out") == 0 ||
              (strcmp(argv[i],"--out")==0))
      {
         flag->outFileFlag=true;
         param->outFileName=argv[++i];
      }
      else
      {
         NR_ERROR("Parameter unknown: " << argv[i]);
         PetitUsage(argv[0]);
         return EXIT_FAILURE;
      }
   }

   if(!flag->refImageFlag || !flag->floImageFlag)
   {
      NR_ERROR("The reference and the floating image have both to be defined");
      PetitUsage(argv[0]);
      return EXIT_FAILURE;
   }

   /* Read the reference image */
   NiftiImage refImage = reg_io_ReadImageFile(param->refImageName);
   if(!refImage)
   {
      NR_ERROR("Error when reading the reference image: " << param->refImageName);
      return EXIT_FAILURE;
   }
   reg_tools_changeDatatype<float>(refImage);

   /* Read the floating image */
   NiftiImage floImage = reg_io_ReadImageFile(param->floImageName);
   if(!floImage)
   {
      NR_ERROR("Error when reading the floating image: " << param->floImageName);
      return EXIT_FAILURE;
   }
   reg_tools_changeDatatype<float>(floImage);

   /* Read and create the mask array */
   vector<unique_ptr<int[]>> refMasks(1);
   unique_ptr<int[]>& refMask = refMasks[0];
   size_t refMaskVoxNumber = refImage.nVoxelsPerVolume();
   if(flag->refMaskImageFlag){
      NiftiImage refMaskImage = reg_io_ReadImageFile(param->refMaskImageName);
      if(!refMaskImage)
      {
         NR_ERROR("Error when reading the reference mask image: " << param->refMaskImageName);
         return EXIT_FAILURE;
      }
      reg_createMaskPyramid<float>(refMaskImage, refMasks, 1, 1);
   }
   else{
      refMask = unique_ptr<int[]>(new int[refMaskVoxNumber]());
      for(size_t i=0;i<refMaskVoxNumber;++i) refMask[i]=i;
   }

   /* Create the warped floating image */
   nifti_image *warpedFloImage = nifti_copy_nim_info(refImage);
   warpedFloImage->ndim=warpedFloImage->dim[0]=floImage->ndim;
   warpedFloImage->nt=warpedFloImage->dim[4]=floImage->nt;
   warpedFloImage->nu=warpedFloImage->dim[5]=floImage->nu;
   warpedFloImage->nvox=NiftiImage::calcVoxelNumber(warpedFloImage, warpedFloImage->ndim);
   warpedFloImage->cal_min=floImage->cal_min;
   warpedFloImage->cal_max=floImage->cal_max;
   warpedFloImage->scl_inter=floImage->scl_inter;
   warpedFloImage->scl_slope=floImage->scl_slope;
   warpedFloImage->datatype=floImage->datatype;
   warpedFloImage->nbyper=floImage->nbyper;
   warpedFloImage->data=malloc(warpedFloImage->nvox*warpedFloImage->nbyper);

   /* Create the deformation field */
   nifti_image *defField = nifti_copy_nim_info(refImage);
   defField->ndim=defField->dim[0]=5;
   defField->nt=defField->dim[4]=1;
   defField->nu=defField->dim[5]=refImage->nz>1?3:2;
   defField->nvox=NiftiImage::calcVoxelNumber(defField, defField->ndim);
   defField->datatype=NIFTI_TYPE_FLOAT32;
   defField->nbyper=sizeof(float);
   defField->data=calloc(defField->nvox,defField->nbyper);
   defField->scl_slope=1.f;
   defField->scl_inter=0.f;
   reg_tools_multiplyValueToImage(defField,defField,0.f);
   defField->intent_p1=DISP_FIELD;
   reg_getDeformationFromDisplacement(defField);

   /* Warp the floating image */
   reg_resampleImage(floImage,
                     warpedFloImage,
                     defField,
                     refMask.get(),
                     param->interpolation,
                     param->paddingValue);
   nifti_image_free(defField);

   FILE *outFile=nullptr;
   if(flag->outFileFlag)
      outFile=fopen(param->outFileName, "w");

   /* Compute the NCC if required */
   if(flag->returnNCCFlag){
      float *refPtr = static_cast<float *>(refImage->data);
      float *warPtr = static_cast<float *>(warpedFloImage->data);
      double refMeanValue =0.;
      double warMeanValue =0.;
      refMaskVoxNumber=0;
      for(size_t i=0; i<refImage->nvox; ++i){
         if(refMask[i]>-1 && refPtr[i]==refPtr[i] && warPtr[i]==warPtr[i]){
            refMeanValue += refPtr[i];
            warMeanValue += warPtr[i];
            ++refMaskVoxNumber;
         }
      }
      if(refMaskVoxNumber==0)
         NR_ERROR("No active voxel");
      refMeanValue /= (double)refMaskVoxNumber;
      warMeanValue /= (double)refMaskVoxNumber;
      double refSTDValue =0.;
      double warSTDValue =0.;
      double measure=0.;
      for(size_t i=0; i<refImage->nvox; ++i){
         if(refMask[i]>-1 && refPtr[i]==refPtr[i] && warPtr[i]==warPtr[i]){
            refSTDValue += Square((double)refPtr[i] - refMeanValue);
            warSTDValue += Square((double)warPtr[i] - warMeanValue);
            measure += ((double)refPtr[i] - refMeanValue) *
                  ((double)warPtr[i] - warMeanValue);
         }
      }
      refSTDValue /= (double)refMaskVoxNumber;
      warSTDValue /= (double)refMaskVoxNumber;
      measure /= sqrt(refSTDValue)*sqrt(warSTDValue)*
            (double)refMaskVoxNumber;
      if(outFile!=nullptr)
         fprintf(outFile, "%g\n", measure);
      else NR_COUT << "NCC: " << measure << std::endl;
   }
   /* Compute the LNCC if required */
   if(flag->returnLNCCFlag){
      reg_lncc *lncc_object=new reg_lncc();
      for(int i=0;i<(refImage->nt<warpedFloImage->nt?refImage->nt:warpedFloImage->nt);++i)
         lncc_object->SetTimePointWeight(i,1.0);
      lncc_object->InitialiseMeasure(refImage,
                                    warpedFloImage,
                                    refMask.get(),
                                    warpedFloImage,
                                    nullptr,
                                    nullptr);
      double measure=lncc_object->GetSimilarityMeasureValue();
      if(outFile!=nullptr)
         fprintf(outFile, "%g\n", measure);
      else NR_COUT << "LNCC: " << measure << std::endl;
      delete lncc_object;
   }
   /* Compute the NMI if required */
   if(flag->returnNMIFlag){
      reg_nmi *nmi_object=new reg_nmi();
      for(int i=0;i<(refImage->nt<warpedFloImage->nt?refImage->nt:warpedFloImage->nt);++i)
        nmi_object->SetTimePointWeight(i, 1.0);
      nmi_object->InitialiseMeasure(refImage,
                                    warpedFloImage,
                                    refMask.get(),
                                    warpedFloImage,
                                    nullptr,
                                    nullptr);
      double measure=nmi_object->GetSimilarityMeasureValue();
      if(outFile!=nullptr)
         fprintf(outFile, "%g\n", measure);
      else NR_COUT << "NMI: " << measure << std::endl;
      delete nmi_object;
   }
   /* Compute the SSD if required */
   if(flag->returnSSDFlag){
      reg_ssd *ssd_object=new reg_ssd();
      for(int i=0;i<(refImage->nt<warpedFloImage->nt?refImage->nt:warpedFloImage->nt);++i)
        ssd_object->SetTimePointWeight(i, 1.0);
      ssd_object->InitialiseMeasure(refImage,
                                    warpedFloImage,
                                    refMask.get(),
                                    warpedFloImage,
                                    nullptr,
                                    nullptr,
                                    nullptr);
      double measure=ssd_object->GetSimilarityMeasureValue();
      if(outFile!=nullptr)
         fprintf(outFile, "%g\n", measure);
      else NR_COUT << "SSD: " << measure << std::endl;
      delete ssd_object;
   }
   /* Compute the MIND SSD if required */
   if(flag->returnMINDFlag){
      reg_mind *mind_object=new reg_mind();
      for(int i=0;i<(refImage->nt<warpedFloImage->nt?refImage->nt:warpedFloImage->nt);++i)
        mind_object->SetTimePointWeight(i, 1.0);
      mind_object->InitialiseMeasure(refImage,
                                    warpedFloImage,
                                    refMask.get(),
                                    warpedFloImage,
                                    nullptr,
                                    nullptr);
      double measure=mind_object->GetSimilarityMeasureValue();
      if(outFile!=nullptr)
         fprintf(outFile, "%g\n", measure);
      else NR_COUT << "MIND: " << measure << std::endl;
      delete mind_object;
   }

   // Close the output file if required
   if(outFile!=nullptr)
      fclose(outFile);

   free(flag);
   free(param);
   return EXIT_SUCCESS;
}
