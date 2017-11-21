/**
 * @file reg_aladin.cpp
 * @author Marc Modat, David C Cash and Pankaj Daga
 * @date 12/08/2009
 *
 * Copyright (c) 2009, University College London. All rights reserved.
 * Centre for Medical Image Computing (CMIC)
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_ReadWriteImage.h"
#include "_reg_ReadWriteMatrix.h"
#include "_reg_aladin_sym.h"
#include "_reg_tools.h"
#include "reg_aladin.h"
//#include <libgen.h> //DO NOT WORK ON WINDOWS !

#ifdef _WIN32
#   include <time.h>
#endif

#define PrecisionTYPE float

void PetitUsage(char *exec)
{
   char text[255];
   reg_print_msg_error("");
   reg_print_msg_error("reg_aladin");
   sprintf(text, "Usage:\t%s -ref <referenceImageName> -flo <floatingImageName> [OPTIONS]",exec);
   reg_print_msg_error(text);
   reg_print_msg_error("\tSee the help for more details (-h).");
   reg_print_msg_error("");
   return;
}
void Usage(char *exec)
{
   char text[255];
   reg_print_info(exec, "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
   reg_print_info(exec, "Block Matching algorithm for global registration.");
   reg_print_info(exec, "Based on Modat et al., \"Global image registration using a symmetric block-matching approach\"");
   reg_print_info(exec, "J. Med. Img. 1(2) 024003, 2014, doi: 10.1117/1.JMI.1.2.024003");
   reg_print_info(exec, "For any comment, please contact Marc Modat (m.modat@ucl.ac.uk)");
   reg_print_info(exec, "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
   sprintf(text, "Usage:\t%s -ref <filename> -flo <filename> [OPTIONS].", exec);
   reg_print_info(exec, text);
   reg_print_info(exec, "\t-ref <filename>\tReference image filename (also called Target or Fixed) (mandatory)");
   reg_print_info(exec, "\t-flo <filename>\tFloating image filename (also called Source or moving) (mandatory)");
   reg_print_info(exec, "");
   reg_print_info(exec, "* * OPTIONS * *");
   reg_print_info(exec, "\t-noSym \t\t\tThe symmetric version of the algorithm is used by default. Use this flag to disable it.");
   reg_print_info(exec, "\t-rigOnly\t\tTo perform a rigid registration only. (Rigid+affine by default)");
   reg_print_info(exec, "\t-affDirect\t\tDirectly optimize 12 DoF affine. (Default is rigid initially then affine)");

   reg_print_info(exec, "\t-aff <filename>\t\tFilename which contains the output affine transformation. [outputAffine.txt]");
   reg_print_info(exec, "\t-inaff <filename>\tFilename which contains an input affine transformation. (Affine*Reference=Floating) [none]");

   reg_print_info(exec, "\t-rmask <filename>\tFilename of a mask image in the reference space.");
   reg_print_info(exec, "\t-fmask <filename>\tFilename of a mask image in the floating space. (Only used when symmetric turned on)");
   reg_print_info(exec, "\t-res <filename>\t\tFilename of the resampled image. [outputResult.nii]");

   reg_print_info(exec, "\t-maxit <int>\t\tMaximal number of iterations of the trimmed least square approach to perform per level. [5]");
   reg_print_info(exec, "\t-ln <int>\t\tNumber of levels to use to generate the pyramids for the coarse-to-fine approach. [3]");
   reg_print_info(exec, "\t-lp <int>\t\tNumber of levels to use to run the registration once the pyramids have been created. [ln]");

   reg_print_info(exec, "\t-smooR <float>\t\tStandard deviation in mm (voxel if negative) of the Gaussian kernel used to smooth the Reference image. [0]");
   reg_print_info(exec, "\t-smooF <float>\t\tStandard deviation in mm (voxel if negative) of the Gaussian kernel used to smooth the Floating image. [0]");
   reg_print_info(exec, "\t-refLowThr <float>\tLower threshold value applied to the reference image. [0]");
   reg_print_info(exec, "\t-refUpThr <float>\tUpper threshold value applied to the reference image. [0]");
   reg_print_info(exec, "\t-floLowThr <float>\tLower threshold value applied to the floating image. [0]");
   reg_print_info(exec, "\t-floUpThr <float>\tUpper threshold value applied to the floating image. [0]");
   reg_print_info(exec, "\t-pad <float>\t\tPadding value [nan]");

   reg_print_info(exec, "\t-nac\t\t\tUse the nifti header origin to initialise the transformation. (Image centres are used by default)");
   reg_print_info(exec, "\t-cog\t\t\tUse the input masks centre of mass to initialise the transformation. (Image centres are used by default)");
   reg_print_info(exec, "\t-interp\t\t\tInterpolation order to use internally to warp the floating image.");
   reg_print_info(exec, "\t-iso\t\t\tMake floating and reference images isotropic if required.");

   reg_print_info(exec, "\t-pv <int>\t\tPercentage of blocks to use in the optimisation scheme. [50]");
   reg_print_info(exec, "\t-pi <int>\t\tPercentage of blocks to consider as inlier in the optimisation scheme. [50]");
   reg_print_info(exec, "\t-speeeeed\t\tGo faster");
#if defined(_USE_CUDA) && defined(_USE_OPENCL)
   reg_print_info(exec, "\t-platf <uint>\t\tChoose platform: CPU=0 | Cuda=1 | OpenCL=2 [0]");
#else
#ifdef _USE_CUDA
   reg_print_info(exec, "\t-platf\t\t\tChoose platform: CPU=0 | Cuda=1 [0]");
#endif
#ifdef _USE_OPENCL
   reg_print_info(exec, "\t-platf\t\t\tChoose platform: CPU=0 | OpenCL=2 [0]");
#endif
#endif
#if defined(_USE_CUDA) || defined(_USE_OPENCL)
   reg_print_info(exec, "\t-gpuid <uint>\t\tChoose a custom gpu.");
   reg_print_info(exec, "\t\t\t\tPlease run reg_gpuinfo first to get platform information and their corresponding ids");
#endif
//   reg_print_info(exec, "\t-crv\t\t\tChoose custom capture range for the block matching alg");
#if defined (_OPENMP)
   int defaultOpenMPValue=omp_get_num_procs();
   if(getenv("OMP_NUM_THREADS")!=NULL)
      defaultOpenMPValue=atoi(getenv("OMP_NUM_THREADS"));
   sprintf(text,"\t-omp <int>\t\tNumber of thread to use with OpenMP. [%i/%i]",
          defaultOpenMPValue, omp_get_num_procs());
   reg_print_info(exec, text);
#endif
   reg_print_info(exec, "\t-voff\t\t\tTurns verbose off [on]");
   reg_print_info(exec, "");
   reg_print_info(exec, "\t--version\t\tPrint current version and exit");
   sprintf(text, "\t\t\t\t(%s)",NR_VERSION);
   reg_print_info(exec, text);
   reg_print_info(exec, "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
   return;
}

int main(int argc, char **argv)
{
   if(argc==1)
   {
      //PetitUsage(basename(argv[0])); //DO NOT WORK ON WINDOWS !
      PetitUsage(argv[0]);
      return EXIT_FAILURE;
   }

   char text[2048];

   time_t start;
   time(&start);

   int symFlag=1;

   char *referenceImageName=NULL;
   int referenceImageFlag=0;

   char *floatingImageName=NULL;
   int floatingImageFlag=0;

   char *outputAffineName=NULL;
   int outputAffineFlag=0;

   char *inputAffineName=NULL;
   int inputAffineFlag=0;

   char *referenceMaskName=NULL;
   int referenceMaskFlag=0;

   char *floatingMaskName=NULL;
   int floatingMaskFlag=0;

   char *outputResultName=NULL;
   int outputResultFlag=0;

   int maxIter=5;
   int nLevels=3;
   int levelsToPerform=std::numeric_limits<int>::max();
   int affineFlag=1;
   int rigidFlag=1;
   int blockStepSize=1;
   int blockPercentage=50;
   float inlierLts=50.0f;
   int alignCentre=1;
   int alignCentreOfGravity=0;
   int interpolation=1;
   float floatingSigma=0.0;
   float referenceSigma=0.0;

   float referenceLowerThr=-std::numeric_limits<PrecisionTYPE>::max();
   float referenceUpperThr=std::numeric_limits<PrecisionTYPE>::max();
   float floatingLowerThr=-std::numeric_limits<PrecisionTYPE>::max();
   float floatingUpperThr=std::numeric_limits<PrecisionTYPE>::max();
   float paddingValue=std::numeric_limits<PrecisionTYPE>::quiet_NaN();

   bool iso=false;
   bool verbose=true;
   int captureRangeVox = 3;
   unsigned int platformFlag = NR_PLATFORM_CPU;
   unsigned gpuIdx = 999;

#if defined (_OPENMP)
   // Set the default number of thread
   int defaultOpenMPValue=omp_get_num_procs();
   if(getenv("OMP_NUM_THREADS")!=NULL)
      defaultOpenMPValue=atoi(getenv("OMP_NUM_THREADS"));
   omp_set_num_threads(defaultOpenMPValue);
#endif

   /* read the input parameter */
   for(int i=1; i<argc; i++)
   {
      if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 ||
            strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 ||
            strcmp(argv[i], "--h")==0 || strcmp(argv[i], "--help")==0)
      {
         Usage(argv[0]);
         return EXIT_SUCCESS;
      }
      else if(strcmp(argv[i], "--xml")==0)
      {
         printf("%s",xml_aladin);
         return EXIT_SUCCESS;
      }
      if( strcmp(argv[i], "-version")==0 ||
            strcmp(argv[i], "-Version")==0 ||
            strcmp(argv[i], "-V")==0 ||
            strcmp(argv[i], "-v")==0 ||
            strcmp(argv[i], "--v")==0 ||
            strcmp(argv[i], "--version")==0)
      {
         printf("%s\n",NR_VERSION);
         return EXIT_SUCCESS;
      }
      else if(strcmp(argv[i], "-ref")==0 || strcmp(argv[i], "-target")==0 || strcmp(argv[i], "--ref")==0)
      {
         referenceImageName=argv[++i];
         referenceImageFlag=1;
      }
      else if(strcmp(argv[i], "-flo")==0 || strcmp(argv[i], "-source")==0 || strcmp(argv[i], "--flo")==0)
      {
         floatingImageName=argv[++i];
         floatingImageFlag=1;
      }

      else if(strcmp(argv[i], "-noSym")==0 || strcmp(argv[i], "--noSym")==0)
      {
         symFlag=0;
      }
      else if(strcmp(argv[i], "-aff")==0 || strcmp(argv[i], "--aff")==0)
      {
         outputAffineName=argv[++i];
         outputAffineFlag=1;
      }
      else if(strcmp(argv[i], "-inaff")==0 || strcmp(argv[i], "--inaff")==0)
      {
         inputAffineName=argv[++i];
         inputAffineFlag=1;
      }
      else if(strcmp(argv[i], "-rmask")==0 || strcmp(argv[i], "-tmask")==0 || strcmp(argv[i], "--rmask")==0)
      {
         referenceMaskName=argv[++i];
         referenceMaskFlag=1;
      }
      else if(strcmp(argv[i], "-fmask")==0 || strcmp(argv[i], "-smask")==0 || strcmp(argv[i], "--fmask")==0)
      {
         floatingMaskName=argv[++i];
         floatingMaskFlag=1;
      }
      else if(strcmp(argv[i], "-res")==0 || strcmp(argv[i], "-result")==0 || strcmp(argv[i], "--res")==0)
      {
         outputResultName=argv[++i];
         outputResultFlag=1;
      }
      else if(strcmp(argv[i], "-maxit")==0 || strcmp(argv[i], "--maxit")==0)
      {
         maxIter = atoi(argv[++i]);
      }
      else if(strcmp(argv[i], "-ln")==0 || strcmp(argv[i], "--ln")==0)
      {
         nLevels=atoi(argv[++i]);
      }
      else if(strcmp(argv[i], "-lp")==0 || strcmp(argv[i], "--lp")==0)
      {
         levelsToPerform=atoi(argv[++i]);
      }

      else if(strcmp(argv[i], "-smooR")==0 || strcmp(argv[i], "-smooT")==0 || strcmp(argv[i], "--smooR")==0)
      {
         referenceSigma = (float)(atof(argv[++i]));
      }
      else if(strcmp(argv[i], "-smooF")==0 || strcmp(argv[i], "-smooS")==0 || strcmp(argv[i], "--smooF")==0)
      {
         floatingSigma=(float)(atof(argv[++i]));
      }
      else if(strcmp(argv[i], "-rigOnly")==0 || strcmp(argv[i], "--rigOnly")==0)
      {
         rigidFlag=1;
         affineFlag=0;
      }
      else if(strcmp(argv[i], "-affDirect")==0 || strcmp(argv[i], "--affDirect")==0)
      {
         rigidFlag=0;
         affineFlag=1;
      }
      else if(strcmp(argv[i], "-nac")==0 || strcmp(argv[i], "--nac")==0)
      {
         alignCentre=0;
      }
      else if(strcmp(argv[i], "-cog")==0 || strcmp(argv[i], "--cog")==0)
      {
         alignCentre=0;
         alignCentreOfGravity=1;
      }
      else if(strcmp(argv[i], "-%v")==0 || strcmp(argv[i], "-pv")==0 || strcmp(argv[i], "--pv")==0)
      {
         float value=atof(argv[++i]);
         if(value<0.f || value>100.f){
            reg_print_msg_error("The variance argument is expected to be between 0 and 100");
            return EXIT_FAILURE;
         }
         blockPercentage=value;
      }
      else if(strcmp(argv[i], "-%i")==0 || strcmp(argv[i], "-pi")==0 || strcmp(argv[i], "--pi")==0)
      {
         float value=atof(argv[++i]);
         if(value<0.f || value>100.f){
            reg_print_msg_error("The inlier argument is expected to be between 0 and 100");
            return EXIT_FAILURE;
         }
         inlierLts=value;
      }
      else if(strcmp(argv[i], "-speeeeed")==0 || strcmp(argv[i], "--speeed")==0)
      {
         blockStepSize=2;
      }
      else if(strcmp(argv[i], "-interp")==0 || strcmp(argv[i], "--interp")==0)
      {
         interpolation=atoi(argv[++i]);
      }
      else if(strcmp(argv[i], "-refLowThr")==0 || strcmp(argv[i], "--refLowThr")==0)
      {
         referenceLowerThr=atof(argv[++i]);
      }
      else if(strcmp(argv[i], "-refUpThr")==0 || strcmp(argv[i], "--refUpThr")==0)
      {
         referenceUpperThr=atof(argv[++i]);
      }
      else if(strcmp(argv[i], "-floLowThr")==0 || strcmp(argv[i], "--floLowThr")==0)
      {
         floatingLowerThr=atof(argv[++i]);
      }
      else if(strcmp(argv[i], "-floUpThr")==0 || strcmp(argv[i], "--floUpThr")==0)
      {
         floatingUpperThr=atof(argv[++i]);
      }

      else if(strcmp(argv[i], "-pad")==0 || strcmp(argv[i], "--pad")==0)
      {
         paddingValue=atof(argv[++i]);
      }
      else if(strcmp(argv[i], "-iso")==0 || strcmp(argv[i], "--iso")==0)
      {
         iso=true;
      }
      else if(strcmp(argv[i], "-voff")==0 || strcmp(argv[i], "--voff")==0)
      {
         verbose=false;
      }
      else if(strcmp(argv[i], "-platf")==0 || strcmp(argv[i], "--platf")==0)
      {
         int value=atoi(argv[++i]);
         if(value<NR_PLATFORM_CPU || value>NR_PLATFORM_CL){
            reg_print_msg_error("The platform argument is expected to be 0, 1 or 2 | 0=CPU, 1=CUDA 2=OPENCL");
            return EXIT_FAILURE;
         }
#ifndef _USE_CUDA
            if(value==NR_PLATFORM_CUDA){
               reg_print_msg_warn("The current install of NiftyReg has not been compiled with CUDA");
               reg_print_msg_warn("The CPU platform is used");
               value=0;
            }
#endif
#ifndef _USE_OPENCL
            if(value==NR_PLATFORM_CL){
               reg_print_msg_error("The current install of NiftyReg has not been compiled with OpenCL");
               reg_print_msg_warn("The CPU platform is used");
               value=0;
            }
#endif
         platformFlag=value;
      }
      else if(strcmp(argv[i], "-gpuid")==0 || strcmp(argv[i], "--gpuid")==0)
      {
          gpuIdx = unsigned(atoi(argv[++i]));
      }
      else if(strcmp(argv[i], "-crv")==0 || strcmp(argv[i], "--crv")==0)
      {
          captureRangeVox=atoi(argv[++i]);
      }
      else if(strcmp(argv[i], "-omp")==0 || strcmp(argv[i], "--omp")==0)
      {
#if defined (_OPENMP)
         omp_set_num_threads(atoi(argv[++i]));
#else
         reg_print_msg_warn("NiftyReg has not been compiled with OpenMP, the \'-omp\' flag is ignored");
         ++i;
#endif
      }
      else
      {

         sprintf(text,"Err:\tParameter %s unknown.",argv[i]);
         reg_print_msg_error(text);
         PetitUsage(argv[0]);
         return EXIT_FAILURE;
      }
   }

   if(!referenceImageFlag || !floatingImageFlag)
   {
      sprintf(text ,"Err:\tThe reference and the floating image have to be defined.");
      reg_print_msg_error(text);
      PetitUsage(argv[0]);
      return EXIT_FAILURE;
   }

   // Output the command line
#ifdef NDEBUG
   if(verbose)
   {
#endif
      reg_print_info((argv[0]), "");
      reg_print_info((argv[0]), "Command line:");
      sprintf(text, "\t");
      for(int i=0; i<argc; i++)
         sprintf(text+strlen(text), " %s", argv[i]);
      reg_print_info((argv[0]), text);
      reg_print_info((argv[0]), "");
#ifdef NDEBUG
   }
#endif

   reg_aladin<PrecisionTYPE> *REG;
   if(symFlag)
   {
      REG = new reg_aladin_sym<PrecisionTYPE>;
      if ( (referenceMaskFlag && !floatingMaskName) || (!referenceMaskFlag && floatingMaskName) )
      {
         reg_print_msg_warn("You have one image mask option turned on but not the other.");
         reg_print_msg_warn("This will affect the degree of symmetry achieved.");
      }
   }
   else
   {
      REG = new reg_aladin<PrecisionTYPE>;
      if (floatingMaskFlag)
      {
         reg_print_msg_warn("Note: Floating mask flag only used in symmetric method. Ignoring this option");
      }
   }

   /* Read the reference image and check its dimension */
   nifti_image *referenceHeader = reg_io_ReadImageFile(referenceImageName);
   if(referenceHeader == NULL)
   {
      sprintf(text,"Error when reading the reference image: %s", referenceImageName);
      reg_print_msg_error(text);
      return EXIT_FAILURE;
   }

   /* Read the floating image and check its dimension */
   nifti_image *floatingHeader = reg_io_ReadImageFile(floatingImageName);
   if(floatingHeader == NULL)
   {
      sprintf(text,"Error when reading the floating image: %s", floatingImageName);
      reg_print_msg_error(text);
      return EXIT_FAILURE;
   }

   // Set the reference and floating images
   nifti_image *isoRefImage=NULL;
   nifti_image *isoFloImage=NULL;
   if(iso)
   {
      // make the images isotropic if required
      isoRefImage=reg_makeIsotropic(referenceHeader,1);
      isoFloImage=reg_makeIsotropic(floatingHeader,1);
      REG->SetInputReference(isoRefImage);
      REG->SetInputFloating(isoFloImage);
   }
   else
   {
      REG->SetInputReference(referenceHeader);
      REG->SetInputFloating(floatingHeader);
   }

   /* read the reference mask image */
   nifti_image *referenceMaskImage=NULL;
   nifti_image *isoRefMaskImage=NULL;
   if(referenceMaskFlag)
   {
      referenceMaskImage = reg_io_ReadImageFile(referenceMaskName);
      if(referenceMaskImage == NULL)
      {
         sprintf(text,"Error when reading the reference mask image: %s", referenceMaskName);
         reg_print_msg_error(text);
         return EXIT_FAILURE;
      }
      /* check the dimension */
      for(int i=1; i<=referenceHeader->dim[0]; i++)
      {
         if(referenceHeader->dim[i]!=referenceMaskImage->dim[i])
         {
            reg_print_msg_error("The reference image and its mask do not have the same dimension");
            return EXIT_FAILURE;
         }
      }
      if(iso)
      {
         // make the image isotropic if required
         isoRefMaskImage=reg_makeIsotropic(referenceMaskImage,0);
         REG->SetInputMask(isoRefMaskImage);
      }
      else REG->SetInputMask(referenceMaskImage);
   }
   /* Read the floating mask image */
   nifti_image *floatingMaskImage=NULL;
   nifti_image *isoFloMaskImage=NULL;
   if(floatingMaskFlag && symFlag)
   {
      floatingMaskImage = reg_io_ReadImageFile(floatingMaskName);
      if(floatingMaskImage == NULL)
      {
         sprintf(text,"Error when reading the floating mask image: %s", floatingMaskName);
         reg_print_msg_error(text);
         return EXIT_FAILURE;
      }
      /* check the dimension */
      for(int i=1; i<=floatingHeader->dim[0]; i++)
      {
         if(floatingHeader->dim[i]!=floatingMaskImage->dim[i])
         {
            reg_print_msg_error("The floating image and its mask do not have the same dimension");
            return EXIT_FAILURE;
         }
      }
      if(iso)
      {
         // make the image isotropic if required
         isoFloMaskImage=reg_makeIsotropic(floatingMaskImage,0);
         REG->SetInputFloatingMask(isoFloMaskImage);
      }
      else REG->SetInputFloatingMask(floatingMaskImage);
   }

   REG->SetMaxIterations(maxIter);
   REG->SetNumberOfLevels(nLevels);
   REG->SetLevelsToPerform(levelsToPerform);
   REG->SetReferenceSigma(referenceSigma);
   REG->SetFloatingSigma(floatingSigma);
   REG->SetAlignCentre(alignCentre);
   REG->SetAlignCentreGravity(alignCentreOfGravity);
   REG->SetPerformAffine(affineFlag);
   REG->SetPerformRigid(rigidFlag);
   REG->SetBlockStepSize(blockStepSize);
   REG->SetBlockPercentage(blockPercentage);
   REG->SetInlierLts(inlierLts);
   REG->SetInterpolation(interpolation);
   REG->setCaptureRangeVox(captureRangeVox);
   REG->setPlatformCode(platformFlag);
   REG->setGpuIdx(gpuIdx);

   if (referenceLowerThr != referenceUpperThr)
   {
      REG->SetReferenceLowerThreshold(referenceLowerThr);
      REG->SetReferenceUpperThreshold(referenceUpperThr);
   }

   if (floatingLowerThr != floatingUpperThr)
   {
      REG->SetFloatingLowerThreshold(floatingLowerThr);
      REG->SetFloatingUpperThreshold(floatingUpperThr);
   }

   REG->SetWarpedPaddingValue(paddingValue);

   if(REG->GetLevelsToPerform() > REG->GetNumberOfLevels())
      REG->SetLevelsToPerform(REG->GetNumberOfLevels());

   // Set the input affine transformation if defined
   if(inputAffineFlag==1)
      REG->SetInputTransform(inputAffineName);

   // Set the verbose type
   REG->SetVerbose(verbose);

#ifndef NDEBUG
   reg_print_msg_debug("*******************************************");
   reg_print_msg_debug("*******************************************");
   reg_print_msg_debug("NiftyReg has been compiled in DEBUG mode");
   reg_print_msg_debug("Please re-run cmake to set the variable");
   reg_print_msg_debug("CMAKE_BUILD_TYPE to \"Release\" if required");
   reg_print_msg_debug("*******************************************");
   reg_print_msg_debug("*******************************************");
#endif

#if defined (_OPENMP)
   if(verbose)
   {
      int maxThreadNumber = omp_get_max_threads();
      sprintf(text, "OpenMP is used with %i thread(s)", maxThreadNumber);
      reg_print_info((argv[0]), text);
   }
#endif // _OPENMP

   // Run the registration
   REG->Run();

   // The warped image is saved
   if(iso)
   {
      REG->SetInputReference(referenceHeader);
      REG->SetInputFloating(floatingHeader);
   }
   nifti_image *outputResultImage=REG->GetFinalWarpedImage();
   if(!outputResultFlag) outputResultName=(char *)"outputResult.nii.gz";
   reg_io_WriteImageFile(outputResultImage,outputResultName);
   nifti_image_free(outputResultImage);

   /* The affine transformation is saved */
   if(outputAffineFlag)
      reg_tool_WriteAffineFile(REG->GetTransformationMatrix(), outputAffineName);
   else reg_tool_WriteAffineFile(REG->GetTransformationMatrix(), (char *)"outputAffine.txt");

   nifti_image_free(referenceHeader);
   nifti_image_free(floatingHeader);
   if(isoRefImage!=NULL)
      nifti_image_free(isoRefImage);
   if(isoFloImage!=NULL)
      nifti_image_free(isoFloImage);
   if(referenceMaskImage!=NULL)
      nifti_image_free(referenceMaskImage);
   if(floatingMaskImage!=NULL)
      nifti_image_free(floatingMaskImage);
   if(isoRefMaskImage!=NULL)
      nifti_image_free(isoRefMaskImage);
   if(isoFloMaskImage!=NULL)
      nifti_image_free(isoFloMaskImage);

   delete REG;
#ifdef NDEBUG
   if(verbose)
   {
#endif
      time_t end;
      time(&end);
      int minutes=(int)floorf((end-start)/60.0f);
      int seconds=(int)(end-start - 60*minutes);
      sprintf(text, "Registration performed in %i min %i sec", minutes, seconds);
      reg_print_info((argv[0]), text);
      reg_print_info((argv[0]), "Have a good day !");
#ifdef NDEBUG
   }
#endif
   return EXIT_SUCCESS;
}
