/*
 *  reg_f3d.cpp
 *
 *
 *  Created by Marc Modat on 26/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_ReadWriteImage.h"
#include "_reg_ReadWriteMatrix.h"
#include "_reg_f3d2.h"
#include "reg_f3d.h"
#include <float.h>
#include <limits>
//#include <libgen.h> //DO NOT WORK ON WINDOWS !

#ifdef _WIN32
#   include <time.h>
#endif

void PetitUsage(char *exec)
{
   char text[255];
   reg_print_msg_error("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
   reg_print_msg_error("Fast Free-Form Deformation algorithm for non-rigid registration");
   sprintf(text,"Usage:\t%s -ref <referenceImageName> -flo <floatingImageName> [OPTIONS]",exec);
   reg_print_msg_error(text);
   reg_print_msg_error("\tSee the help for more details (-h)");
   reg_print_msg_error("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
   return;
}
void Usage(char *exec)
{
   char text[255];
   reg_print_info(exec, "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
   reg_print_info(exec, "Fast Free-Form Deformation (F3D) algorithm for non-rigid registration.");
   reg_print_info(exec, "Based on Modat et al., \"Fast Free-Form Deformation using");
   reg_print_info(exec, "graphics processing units\", CMPB, 2010");
   reg_print_info(exec, "For any comment, please contact Marc Modat (m.modat@ucl.ac.uk)");
   reg_print_info(exec, "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
   sprintf(text, "Usage:\t%s -ref <filename> -flo <filename> [OPTIONS].",exec);
   reg_print_info(exec, text);
   reg_print_info(exec, "\t-ref <filename>\tFilename of the reference image (mandatory)");
   reg_print_info(exec, "\t-flo <filename>\tFilename of the floating image (mandatory)");
   reg_print_info(exec, "***************");
   reg_print_info(exec, "*** OPTIONS ***");
   reg_print_info(exec, "***************");
   reg_print_info(exec, "*** Initial transformation options (One option will be considered):");
   reg_print_info(exec, "\t-aff <filename>\t\tFilename which contains an affine transformation (Affine*Reference=Floating)");
   reg_print_info(exec, "\t-incpp <filename>\tFilename ofloatf control point grid input");
   reg_print_info(exec, "\t\t\t\tThe coarse spacing is defined by this file.");
   reg_print_info(exec, "");
   reg_print_info(exec, "*** Output options:");
   reg_print_info(exec, "\t-cpp <filename>\t\tFilename of control point grid [outputCPP.nii]");
   reg_print_info(exec, "\t-res <filename> \tFilename of the resampled image [outputResult.nii]");
   reg_print_info(exec, "");
   reg_print_info(exec, "*** Input image options:");
   reg_print_info(exec, "\t-rmask <filename>\t\tFilename of a mask image in the reference space");
   reg_print_info(exec, "\t-smooR <float>\t\t\tSmooth the reference image using the specified sigma (mm) [0]");
   reg_print_info(exec, "\t-smooF <float>\t\t\tSmooth the floating image using the specified sigma (mm) [0]");
   reg_print_info(exec, "\t--rLwTh <float>\t\t\tLower threshold to apply to the reference image intensities [none]. Identical value for every timepoint.*");
   reg_print_info(exec, "\t--rUpTh <float>\t\t\tUpper threshold to apply to the reference image intensities [none]. Identical value for every timepoint.*");
   reg_print_info(exec, "\t--fLwTh <float>\t\t\tLower threshold to apply to the floating image intensities [none]. Identical value for every timepoint.*");
   reg_print_info(exec, "\t--fUpTh <float>\t\t\tUpper threshold to apply to the floating image intensities [none]. Identical value for every timepoint.*");
   reg_print_info(exec, "\t-rLwTh <timepoint> <float>\tLower threshold to apply to the reference image intensities [none]*");
   reg_print_info(exec, "\t-rUpTh <timepoint> <float>\tUpper threshold to apply to the reference image intensities [none]*");
   reg_print_info(exec, "\t-fLwTh <timepoint> <float>\tLower threshold to apply to the floating image intensities [none]*");
   reg_print_info(exec, "\t-fUpTh <timepoint> <float>\tUpper threshold to apply to the floating image intensities [none]*");
   reg_print_info(exec, "\t* The scl_slope and scl_inter from the nifti header are taken into account for the thresholds");
   reg_print_info(exec, "");
   reg_print_info(exec, "*** Spline options:");
   reg_print_info(exec, "\t-sx <float>\t\tFinal grid spacing along the x axis in mm (in voxel if negative value) [5 voxels]");
   reg_print_info(exec, "\t-sy <float>\t\tFinal grid spacing along the y axis in mm (in voxel if negative value) [sx value]");
   reg_print_info(exec, "\t-sz <float>\t\tFinal grid spacing along the z axis in mm (in voxel if negative value) [sx value]");
   reg_print_info(exec, "");
   reg_print_info(exec, "*** Regularisation options:");
   reg_print_info(exec, "\t-be <float>\t\tWeight of the bending energy (second derivative of the transformation) penalty term [0.001]");
   reg_print_info(exec, "\t-le <float>\t\tWeight of first order penalty term (symmetric and anti-symmetric part of the Jacobian) [0.01]");
   reg_print_info(exec, "\t-jl <float>\t\tWeight of log of the Jacobian determinant penalty term [0.0]");
   reg_print_info(exec, "\t-noAppJL\t\tTo not approximate the JL value only at the control point position");
   reg_print_info(exec, "");
   reg_print_info(exec, "*** Measure of similarity options:");
   reg_print_info(exec, "*** NMI with 64 bins is used expect if specified otherwise");
   reg_print_info(exec, "\t--nmi\t\t\tNMI. Used NMI even when one or several other measures are specified.");
   reg_print_info(exec, "\t--rbn <int>\t\tNMI. Number of bin to use for the reference image histogram. Identical value for every timepoint.");
   reg_print_info(exec, "\t--fbn <int>\t\tNMI. Number of bin to use for the floating image histogram. Identical value for every timepoint.");
   reg_print_info(exec, "\t-rbn <tp> <int>\t\tNMI. Number of bin to use for the reference image histogram for the specified time point.");
   reg_print_info(exec, "\t-rbn <tp> <int>\t\tNMI. Number of bin to use for the floating image histogram for the specified time point.");
   reg_print_info(exec, "\t--lncc <float>\t\tLNCC. Standard deviation of the Gaussian kernel. Identical value for every timepoint");
   reg_print_info(exec, "\t-lncc <tp> <float>\tLNCC. Standard deviation of the Gaussian kernel for the specified timepoint");
   reg_print_info(exec, "\t--ssd\t\t\tSSD. Used for all time points");
   reg_print_info(exec, "\t-ssd <tp>\t\tSSD. Used for the specified timepoint");
   reg_print_info(exec, "\t--mind\t\t\tMIND. Used for all time points");
   reg_print_info(exec, "\t--kld\t\t\tKLD. Used for all time points");
   reg_print_info(exec, "\t-kld <tp>\t\tKLD. Used for the specified timepoint");
   reg_print_info(exec, "\t* For the Kullback–Leibler divergence, reference and floating are expected to be probabilities");
   reg_print_info(exec, "\t-rr\t\t\tIntensities are thresholded between the 2 and 98\% ile.");
//   reg_print_info(exec, "\t-amc\t\t\tTo use the additive NMI for multichannel data (bivariate NMI by default)");
   reg_print_info(exec, "");
   reg_print_info(exec, "*** Optimisation options:");
   reg_print_info(exec, "\t-maxit <int>\t\tMaximal number of iteration at the final level [150]");
   reg_print_info(exec, "\t-ln <int>\t\tNumber of level to perform [3]");
   reg_print_info(exec, "\t-lp <int>\t\tOnly perform the first levels [ln]");
   reg_print_info(exec, "\t-nopy\t\t\tDo not use a pyramidal approach");
   reg_print_info(exec, "\t-noConj\t\t\tTo not use the conjuage gradient optimisation but a simple gradient ascent");
   reg_print_info(exec, "\t-pert <int>\t\tTo add perturbation step(s) after each optimisation scheme");
   reg_print_info(exec, "");
   reg_print_info(exec, "*** F3D2 options:");
   reg_print_info(exec, "\t-vel \t\t\tUse a velocity field integration to generate the deformation");
   reg_print_info(exec, "\t-nogce \t\t\tDo not use the gradient accumulation through exponentiation");
   reg_print_info(exec, "\t-fmask <filename>\tFilename of a mask image in the floating space");
   reg_print_info(exec, "");

   reg_print_info(exec, "*** Platform options:");
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

#if defined (_OPENMP)
   reg_print_info(exec, "");
   reg_print_info(exec, "*** OpenMP-related options:");
   int defaultOpenMPValue=1;
   if(getenv("OMP_NUM_THREADS")!=NULL)
      defaultOpenMPValue=atoi(getenv("OMP_NUM_THREADS"));
   sprintf(text,"\t-omp <int>\t\tNumber of thread to use with OpenMP. [%i/%i]",
          defaultOpenMPValue, omp_get_num_procs());
   reg_print_info(exec, text);
#endif
   reg_print_info(exec, "");
   reg_print_info(exec, "*** Other options:");
   reg_print_info(exec, "\t-smoothGrad <float>\tTo smooth the metric derivative (in mm) [0]");
   reg_print_info(exec, "\t-pad <float>\t\tPadding value [nan]");
   reg_print_info(exec, "\t-voff\t\t\tTo turn verbose off");
#ifdef _GIT_HASH
   reg_print_info(exec, "");
   reg_print_info(exec, "\t--version\t\tPrint current source code git hash key and exit");
   sprintf(text, "\t\t\t\t(%s)" ,_GIT_HASH);
   reg_print_info(exec, text);
#endif
   reg_print_info(exec, "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
   return;
}

int main(int argc, char **argv)
{
   if(argc==1)
   {
      PetitUsage((argv[0]));
      return EXIT_FAILURE;
   }
   time_t start;
   time(&start);
   int verbose=true;

#if defined (_OPENMP)
   // Set the default number of thread
   int defaultOpenMPValue=1;
   if(getenv("OMP_NUM_THREADS")!=NULL)
      defaultOpenMPValue=atoi(getenv("OMP_NUM_THREADS"));
   omp_set_num_threads(defaultOpenMPValue);
#endif

   char text[1024];
   //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
   // Check if any information is required
   for(int i=1; i<argc; i++)
   {
      if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 ||
            strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 ||
            strcmp(argv[i], "--h")==0 || strcmp(argv[i], "--help")==0)
      {
         Usage((argv[0]));
         return EXIT_SUCCESS;
      }
      if(strcmp(argv[i], "--xml")==0)
      {
         printf("%s",xml_f3d);
         return EXIT_SUCCESS;
      }
      if(strcmp(argv[i], "-gpu")==0 || strcmp(argv[i], "--gpu")==0)
      {
         reg_print_msg_error("The reg_f3d GPU capability has been de-activated in the current release.");
         return EXIT_FAILURE;
      }
      if(strcmp(argv[i], "-voff")==0)
      {
         verbose=false;
      }
#ifdef _GIT_HASH
      if( strcmp(argv[i], "-version")==0 ||
            strcmp(argv[i], "-Version")==0 ||
            strcmp(argv[i], "-V")==0 ||
            strcmp(argv[i], "-v")==0 ||
            strcmp(argv[i], "--v")==0 ||
            strcmp(argv[i], "--version")==0)
      {
         printf("%s\n",_GIT_HASH);
         return EXIT_SUCCESS;
      }
#endif
   }
   //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
   // Output the command line
#ifdef NDEBUG
   if(verbose)
   {
#endif
      reg_print_info((argv[0]), "");
      reg_print_info((argv[0]), "Command line:");
      sprintf(text, "\t");
      for(int i=0; i<argc; i++)
         sprintf(text, "%s %s", text, argv[i]);
      reg_print_info((argv[0]), text);
      reg_print_info((argv[0]), "");
#ifdef NDEBUG
   }
#endif

   //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
   // Read the reference and floating image
   nifti_image *referenceImage=NULL;
   nifti_image *floatingImage=NULL;
   for(int i=1; i<argc; i++)
   {
      if((strcmp(argv[i],"-ref")==0) || (strcmp(argv[i],"-target")==0) || (strcmp(argv[i],"--ref")==0))
      {
         referenceImage=reg_io_ReadImageFile(argv[++i]);
         if(referenceImage==NULL)
         {
            reg_print_msg_error("Error when reading the reference image:");
            reg_print_msg_error(argv[i-1]);
            return EXIT_FAILURE;
         }
      }
      if((strcmp(argv[i],"-flo")==0) || (strcmp(argv[i],"-source")==0) || (strcmp(argv[i],"--flo")==0))
      {
         floatingImage=reg_io_ReadImageFile(argv[++i]);
         if(floatingImage==NULL)
         {
            reg_print_msg_error("Error when reading the floating image:");
            reg_print_msg_error(argv[i-1]);
            return EXIT_FAILURE;
         }
      }
   }
   // Check that both reference and floating image have been defined
   if(referenceImage==NULL)
   {
      reg_print_msg_error("Error. No reference image has been defined");
      PetitUsage((argv[0]));
      return EXIT_FAILURE;
   }
   // Read the floating image
   if(floatingImage==NULL)
   {
      reg_print_msg_error("Error. No floating image has been defined");
      PetitUsage((argv[0]));
      return EXIT_FAILURE;
   }
   //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
   // Check the type of registration object to create
   reg_f3d<float> *REG=NULL;
   for(int i=1; i<argc; i++)
   {
      if(strcmp(argv[i], "-vel")==0 || strcmp(argv[i], "--vel")==0)
      {
         REG=new reg_f3d2<float>(referenceImage->nt,floatingImage->nt);
         break;
      }
      if(strcmp(argv[i], "-sym")==0 || strcmp(argv[i], "--sym")==0)
      {
         REG=new reg_f3d_sym<float>(referenceImage->nt,floatingImage->nt);
         break;
      }
   }
   if(REG==NULL)
      REG=new reg_f3d<float>(referenceImage->nt,floatingImage->nt);
   REG->SetReferenceImage(referenceImage);
   REG->SetFloatingImage(floatingImage);

   // Create some pointers that could be used
   mat44 affineMatrix;
   nifti_image *inputCCPImage=NULL;
   nifti_image *referenceMaskImage=NULL;
   nifti_image *floatingMaskImage=NULL;
   char *outputWarpedImageName=NULL;
   char *outputCPPImageName=NULL;
   bool useMeanLNCC=false;
   int refBinNumber=0;
   int floBinNumber=0;
   //
   unsigned int platformFlag = NR_PLATFORM_CPU;
   unsigned gpuIdx = 999;

   /* read the input parameter */
   for(int i=1; i<argc; i++)
   {
      if(strcmp(argv[i],"-ref")==0 || strcmp(argv[i],"-target")==0 ||
            strcmp(argv[i],"--ref")==0 || strcmp(argv[i],"-flo")==0 ||
            strcmp(argv[i],"-source")==0 || strcmp(argv[i],"--flo")==0 )
      {
         // argument has already been parsed
         ++i;
      }
      else if(strcmp(argv[i], "-voff")==0)
      {
         verbose=false;
         REG->DoNotPrintOutInformation();
      }
      else if(strcmp(argv[i], "-aff")==0 || (strcmp(argv[i],"--aff")==0))
      {
         // Check first if the specified affine file exist
         char *affineTransformationName=argv[++i];
         if(FILE *aff=fopen(affineTransformationName, "r"))
         {
            fclose(aff);
         }
         else
         {
            reg_print_msg_error("The specified input affine file can not be read:");
            reg_print_msg_error(affineTransformationName);
            return EXIT_FAILURE;
         }
         // Read the affine matrix
         reg_tool_ReadAffineFile(&affineMatrix,
                                 affineTransformationName);
         // Send the transformation to the registration object
         REG->SetAffineTransformation(&affineMatrix);
      }
      else if(strcmp(argv[i], "-incpp")==0 || (strcmp(argv[i],"--incpp")==0))
      {
         inputCCPImage=reg_io_ReadImageFile(argv[++i]);
         if(inputCCPImage==NULL)
         {
            reg_print_msg_error("Error when reading the input control point grid image:");
            reg_print_msg_error(argv[i-1]);
            return EXIT_FAILURE;
         }
         REG->SetControlPointGridImage(inputCCPImage);
      }
      else if((strcmp(argv[i],"-rmask")==0) || (strcmp(argv[i],"-tmask")==0) || (strcmp(argv[i],"--rmask")==0))
      {
         referenceMaskImage=reg_io_ReadImageFile(argv[++i]);
         if(referenceMaskImage==NULL)
         {
            reg_print_msg_error("Error when reading the reference mask image:");
            reg_print_msg_error(argv[i-1]);
            return EXIT_FAILURE;
         }
         REG->SetReferenceMask(referenceMaskImage);
      }
      else if((strcmp(argv[i],"-res")==0) || (strcmp(argv[i],"-result")==0) || (strcmp(argv[i],"--res")==0))
      {
         outputWarpedImageName=argv[++i];
      }
      else if(strcmp(argv[i], "-cpp")==0 || (strcmp(argv[i],"--cpp")==0))
      {
         outputCPPImageName=argv[++i];
      }
      else if(strcmp(argv[i], "-maxit")==0 || strcmp(argv[i], "--maxit")==0)
      {
         REG->SetMaximalIterationNumber(atoi(argv[++i]));
      }
      else if(strcmp(argv[i], "-sx")==0 || strcmp(argv[i], "--sx")==0)
      {
         REG->SetSpacing(0,(float)atof(argv[++i]));
      }
      else if(strcmp(argv[i], "-sy")==0 || strcmp(argv[i], "--sy")==0)
      {
         REG->SetSpacing(1,(float)atof(argv[++i]));
      }
      else if(strcmp(argv[i], "-sz")==0 || strcmp(argv[i], "--sz")==0)
      {
         REG->SetSpacing(2,(float)atof(argv[++i]));
      }
      else if((strcmp(argv[i],"--nmi")==0) )
      {
         int bin=64;
         if(refBinNumber!=0)
            bin=refBinNumber;
         for(int t=0; t<referenceImage->nt; ++t)
            REG->UseNMISetReferenceBinNumber(t,bin);
         bin=64;
         if(floBinNumber!=0)
            bin=floBinNumber;
         for(int t=0; t<floatingImage->nt; ++t)
            REG->UseNMISetFloatingBinNumber(t,bin);
      }
      else if((strcmp(argv[i],"-rbn")==0) || (strcmp(argv[i],"-tbn")==0))
      {
         int tp=atoi(argv[++i]);
         int bin=atoi(argv[++i]);
         refBinNumber=bin;
         REG->UseNMISetReferenceBinNumber(tp,bin);
      }
      else if((strcmp(argv[i],"--rbn")==0) )
      {
         int bin = atoi(argv[++i]);
         refBinNumber=bin;
         for(int t=0; t<referenceImage->nt; ++t)
            REG->UseNMISetReferenceBinNumber(t,bin);
      }
      else if((strcmp(argv[i],"-fbn")==0) || (strcmp(argv[i],"-sbn")==0))
      {
         int tp=atoi(argv[++i]);
         int bin=atoi(argv[++i]);
         floBinNumber=bin;
         REG->UseNMISetFloatingBinNumber(tp,bin);
      }
      else if((strcmp(argv[i],"--fbn")==0) )
      {
         int bin = atoi(argv[++i]);
         floBinNumber=bin;
         for(int t=0; t<floatingImage->nt; ++t)
            REG->UseNMISetFloatingBinNumber(t,bin);
      }
      else if(strcmp(argv[i], "-ln")==0 || strcmp(argv[i], "--ln")==0)
      {
         REG->SetLevelNumber(atoi(argv[++i]));
      }
      else if(strcmp(argv[i], "-lp")==0 || strcmp(argv[i], "--lp")==0)
      {
         REG->SetLevelToPerform(atoi(argv[++i]));
      }
      else if(strcmp(argv[i], "-be")==0 || strcmp(argv[i], "--be")==0)
      {
         REG->SetBendingEnergyWeight(atof(argv[++i]));
      }
      else if(strcmp(argv[i], "-le")==0)
      {
         REG->SetLinearEnergyWeight(atof(argv[++i]));
      }
      else if(strcmp(argv[i], "-jl")==0 || strcmp(argv[i], "--jl")==0)
      {
         REG->SetJacobianLogWeight(atof(argv[++i]));
      }
      else if(strcmp(argv[i], "-noAppJL")==0 || strcmp(argv[i], "--noAppJL")==0)
      {
         REG->DoNotApproximateJacobianLog();
      }
      else if((strcmp(argv[i],"-smooR")==0) || (strcmp(argv[i],"-smooT")==0) || strcmp(argv[i], "--smooR")==0)
      {
         REG->SetReferenceSmoothingSigma(atof(argv[++i]));
      }
      else if((strcmp(argv[i],"-smooF")==0) || (strcmp(argv[i],"-smooS")==0) || strcmp(argv[i], "--smooF")==0)
      {
         REG->SetFloatingSmoothingSigma(atof(argv[++i]));
      }
      else if((strcmp(argv[i],"-rLwTh")==0) || (strcmp(argv[i],"-tLwTh")==0))
      {
         int tp=atoi(argv[++i]);
         float val=atof(argv[++i]);
         REG->SetReferenceThresholdLow(tp,val);
      }
      else if((strcmp(argv[i],"-rUpTh")==0) || strcmp(argv[i],"-tUpTh")==0)
      {
         int tp=atoi(argv[++i]);
         float val=atof(argv[++i]);
         REG->SetReferenceThresholdUp(tp,val);
      }
      else if((strcmp(argv[i],"-fLwTh")==0) || (strcmp(argv[i],"-sLwTh")==0))
      {
         int tp=atoi(argv[++i]);
         float val=atof(argv[++i]);
         REG->SetFloatingThresholdLow(tp,val);
      }
      else if((strcmp(argv[i],"-fUpTh")==0) || (strcmp(argv[i],"-sUpTh")==0))
      {
         int tp=atoi(argv[++i]);
         float val=atof(argv[++i]);
         REG->SetFloatingThresholdUp(tp,val);
      }
      else if((strcmp(argv[i],"--rLwTh")==0) )
      {
         float threshold = atof(argv[++i]);
         for(int t=0; t<referenceImage->nt; ++t)
            REG->SetReferenceThresholdLow(t,threshold);
      }
      else if((strcmp(argv[i],"--rUpTh")==0) )
      {
         float threshold = atof(argv[++i]);
         for(int t=0; t<referenceImage->nt; ++t)
            REG->SetReferenceThresholdUp(t,threshold);
      }
      else if((strcmp(argv[i],"--fLwTh")==0) )
      {
         float threshold = atof(argv[++i]);
         for(int t=0; t<floatingImage->nt; ++t)
            REG->SetFloatingThresholdLow(t,threshold);
      }
      else if((strcmp(argv[i],"--fUpTh")==0) )
      {
         float threshold = atof(argv[++i]);
         for(int t=0; t<floatingImage->nt; ++t)
            REG->SetFloatingThresholdUp(t,threshold);
      }
      else if(strcmp(argv[i], "-smoothGrad")==0)
      {
         REG->SetGradientSmoothingSigma(atof(argv[++i]));
      }
      else if(strcmp(argv[i], "-ssd")==0)
      {
         REG->UseSSD(atoi(argv[++i]));
      }
      else if(strcmp(argv[i], "--ssd")==0)
      {
         for(int t=0; t<floatingImage->nt; ++t)
            REG->UseSSD(t);
      }
      else if(strcmp(argv[i], "--mind")==0)
      {
          if(referenceImage->nt>1 || floatingImage->nt>1){
              reg_print_msg_error("reg_mind does not support multiple time point image");
              reg_exit();
          }
          REG->UseMIND(0);
      }
      else if(strcmp(argv[i], "--mindssc")==0)
      {
          if(referenceImage->nt>1 || floatingImage->nt>1){
              reg_print_msg_error("reg_mindssc does not support multiple time point image");
              reg_exit();
          }
          REG->UseMIND(0);
      }
      //MRF -- add option for discrete optimization
      else if(strcmp(argv[i], "-kld")==0)
      {
         REG->UseKLDivergence(atoi(argv[++i]));
      }
      else if(strcmp(argv[i], "--kld")==0)
      {
         for(int t=0; t<floatingImage->nt; ++t)
            REG->UseKLDivergence(t);
      }
//        else if(strcmp(argv[i], "-amc")==0){ // HERE TODO
//            REG->UseMultiChannelNMI();
//        }
      else if(strcmp(argv[i], "-rr")==0)
      {
            REG->UseRobustRange();
      }
      else if(strcmp(argv[i], "-lncc")==0)
      {
         int tp=atoi(argv[++i]);
         float stdev = atof(argv[++i]);
         REG->UseLNCC(tp,stdev);
      }
      else if(strcmp(argv[i], "--lncc")==0)
      {
         float stdev = (float)atof(argv[++i]);
         if(stdev!=999999){ // Value specified by the CLI - to be ignored
            for(int t=0; t<referenceImage->nt; ++t)
               REG->UseLNCC(t,stdev);
         }
      }
      else if(strcmp(argv[i], "-lnccMean")==0)
      {
         useMeanLNCC=true;
      }
      else if(strcmp(argv[i], "-dti")==0 || strcmp(argv[i], "--dti")==0)
      {
         bool *timePoint = new bool[referenceImage->nt];
         for(int t=0; t<referenceImage->nt; ++t)
            timePoint[t]=false;
         timePoint[atoi(argv[++i])]=true;
         timePoint[atoi(argv[++i])]=true;
         timePoint[atoi(argv[++i])]=true;
         if(referenceImage->nz>1)
         {
            timePoint[atoi(argv[++i])]=true;
            timePoint[atoi(argv[++i])]=true;
            timePoint[atoi(argv[++i])]=true;
         }
         REG->UseDTI(timePoint);
         delete []timePoint;
      }
      else if(strcmp(argv[i], "-pad")==0)
      {
         REG->SetWarpedPaddingValue(atof(argv[++i]));
      }
      else if(strcmp(argv[i], "-nopy")==0 || strcmp(argv[i], "--nopy")==0)
      {
         REG->DoNotUsePyramidalApproach();
      }
      else if(strcmp(argv[i], "-noConj")==0 || strcmp(argv[i], "--noConj")==0)
      {
         REG->DoNotUseConjugateGradient();
      }
      else if(strcmp(argv[i], "-approxGrad")==0 || strcmp(argv[i], "--approxGrad")==0)
      {
         REG->UseApproximatedGradient();
      }
      else if(strcmp(argv[i], "-interp")==0 || strcmp(argv[i], "--interp")==0)
      {
         int interp=atoi(argv[++i]);
         switch(interp)
         {
         case 0:
            REG->UseNeareatNeighborInterpolation();
            break;
         case 1:
            REG->UseLinearInterpolation();
            break;
         default:
            REG->UseCubicSplineInterpolation();
            break;
         }
      }
//        else if(strcmp(argv[i], "-noAppPW")==0){ // HERE TODO
//            parzenWindowApproximation=false;
//        }
      else if((strcmp(argv[i],"-fmask")==0) || (strcmp(argv[i],"-smask")==0) ||
              (strcmp(argv[i],"--fmask")==0) || (strcmp(argv[i],"--smask")==0))
      {
         floatingMaskImage=reg_io_ReadImageFile(argv[++i]);
         if(floatingMaskImage==NULL)
         {
            reg_print_msg_error("Error when reading the floating mask image:");
            reg_print_msg_error(argv[i-1]);
            return EXIT_FAILURE;
         }
         REG->SetFloatingMask(floatingMaskImage);
      }
      else if(strcmp(argv[i], "-ic")==0 || strcmp(argv[i], "--ic")==0)
      {
         REG->SetInverseConsistencyWeight(atof(argv[++i]));
      }
      else if(strcmp(argv[i], "-nox") ==0)
      {
         REG->NoOptimisationAlongX();
      }
      else if(strcmp(argv[i], "-noy") ==0)
      {
         REG->NoOptimisationAlongY();
      }
      else if(strcmp(argv[i], "-noz") ==0)
      {
         REG->NoOptimisationAlongZ();
      }
      else if(strcmp(argv[i],"-pert")==0 || strcmp(argv[i],"--pert")==0)
      {
         REG->SetPerturbationNumber((size_t)atoi(argv[++i]));
      }
      else if(strcmp(argv[i], "-nogr") ==0)
      {
         REG->NoGridRefinement();
      }
      else if(strcmp(argv[i], "-nogce")==0 || strcmp(argv[i], "--nogce")==0)
      {
         REG->DoNotUseGradientCumulativeExp();
      }
      else if(strcmp(argv[i], "-bch")==0 || strcmp(argv[i], "--bch")==0)
      {
         REG->UseBCHUpdate(atoi(argv[++i]));
      }
////////////////////////
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
////////////////////////
      else if(strcmp(argv[i], "-gpuid")==0 || strcmp(argv[i], "--gpuid")==0)
      {
          gpuIdx = unsigned(atoi(argv[++i]));
      }
//      else if(strcmp(argv[i], "-iso")==0 || strcmp(argv[i], "--iso")==0){
//         iso=true;
//      }
      else if(strcmp(argv[i], "-omp")==0 || strcmp(argv[i], "--omp")==0)
      {
#if defined (_OPENMP)
         omp_set_num_threads(atoi(argv[++i]));
#else
         reg_print_msg_warn("NiftyReg has not been compiled with OpenMP, the \'-omp\' flag is ignored");
         ++i;
#endif
      }
      /* All the following arguments should have already been parsed */
      else if(strcmp(argv[i], "-help")!=0 && strcmp(argv[i], "-Help")!=0 &&
      strcmp(argv[i], "-HELP")!=0 && strcmp(argv[i], "-h")!=0 &&
      strcmp(argv[i], "--h")!=0 && strcmp(argv[i], "--help")!=0 &&
      strcmp(argv[i], "--xml")!=0 && strcmp(argv[i], "-version")!=0 &&
      strcmp(argv[i], "-Version")!=0 && strcmp(argv[i], "-V")!=0 &&
      strcmp(argv[i], "-v")!=0 && strcmp(argv[i], "--v")!=0 &&
      strcmp(argv[i], "-gpu")!=0 && strcmp(argv[i], "--gpu")!=0 &&
      strcmp(argv[i], "-vel")!=0 && strcmp(argv[i], "-sym")!=0)
      {
         reg_print_msg_error("\tParameter unknown:");
         reg_print_msg_error(argv[i]);
         PetitUsage((argv[0]));
         return EXIT_FAILURE;
      }
   }
   if(useMeanLNCC)
      REG->SetLNCCKernelType(2);

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

      // Save the control point image
      nifti_image *outputControlPointGridImage = REG->GetControlPointPositionImage();
      if(outputCPPImageName==NULL) outputCPPImageName=(char *)"outputCPP.nii";
      memset(outputControlPointGridImage->descrip, 0, 80);
      strcpy (outputControlPointGridImage->descrip,"Control point position from NiftyReg (reg_f3d)");
      if(strcmp("NiftyReg F3D2", REG->GetExecutableName())==0)
         strcpy (outputControlPointGridImage->descrip,"Velocity field grid from NiftyReg (reg_f3d2)");
      reg_io_WriteImageFile(outputControlPointGridImage,outputCPPImageName);
      nifti_image_free(outputControlPointGridImage);
      outputControlPointGridImage=NULL;

      // Save the backward control point image
      if(REG->GetSymmetricStatus())
      {
         // _backward is added to the forward control point grid image name
         std::string b(outputCPPImageName);
         if(b.find( ".nii.gz") != std::string::npos)
            b.replace(b.find( ".nii.gz"),7,"_backward.nii.gz");
         else if(b.find( ".nii") != std::string::npos)
            b.replace(b.find( ".nii"),4,"_backward.nii");
         else if(b.find( ".hdr") != std::string::npos)
            b.replace(b.find( ".hdr"),4,"_backward.hdr");
         else if(b.find( ".img.gz") != std::string::npos)
            b.replace(b.find( ".img.gz"),7,"_backward.img.gz");
         else if(b.find( ".img") != std::string::npos)
            b.replace(b.find( ".img"),4,"_backward.img");
         else if(b.find( ".png") != std::string::npos)
            b.replace(b.find( ".png"),4,"_backward.png");
         else if(b.find( ".nrrd") != std::string::npos)
            b.replace(b.find( ".nrrd"),5,"_backward.nrrd");
         else b.append("_backward.nii");
         nifti_image *outputBackwardControlPointGridImage = REG->GetBackwardControlPointPositionImage();
         memset(outputBackwardControlPointGridImage->descrip, 0, 80);
         strcpy (outputBackwardControlPointGridImage->descrip,"Backward Control point position from NiftyReg (reg_f3d)");
         if(strcmp("NiftyReg F3D2", REG->GetExecutableName())==0)
            strcpy (outputBackwardControlPointGridImage->descrip,"Backward velocity field grid from NiftyReg (reg_f3d2)");
         reg_io_WriteImageFile(outputBackwardControlPointGridImage,b.c_str());
         nifti_image_free(outputBackwardControlPointGridImage);
         outputBackwardControlPointGridImage=NULL;
      }

      // Save the warped image(s)
      nifti_image **outputWarpedImage=(nifti_image **)malloc(2*sizeof(nifti_image *));
      outputWarpedImage[0]=NULL;
      outputWarpedImage[1]=NULL;
      outputWarpedImage = REG->GetWarpedImage();
      if(outputWarpedImageName==NULL)
         outputWarpedImageName=(char *)"outputResult.nii";
      memset(outputWarpedImage[0]->descrip, 0, 80);
      strcpy (outputWarpedImage[0]->descrip,"Warped image using NiftyReg (reg_f3d)");
//      if(strcmp("NiftyReg F3D SYM", REG->GetExecutableName())==0)
//      {
//         strcpy (outputWarpedImage[0]->descrip,"Warped image using NiftyReg (reg_f3d_sym)");
//         strcpy (outputWarpedImage[1]->descrip,"Warped image using NiftyReg (reg_f3d_sym)");
//      }
      if(strcmp("NiftyReg F3D2", REG->GetExecutableName())==0)
      {
         strcpy (outputWarpedImage[0]->descrip,"Warped image using NiftyReg (reg_f3d2)");
         strcpy (outputWarpedImage[1]->descrip,"Warped image using NiftyReg (reg_f3d2)");
      }
      if(REG->GetSymmetricStatus())
      {
         if(outputWarpedImage[1]!=NULL)
         {
            std::string b(outputWarpedImageName);
            if(b.find( ".nii.gz") != std::string::npos)
               b.replace(b.find( ".nii.gz"),7,"_backward.nii.gz");
            else if(b.find( ".nii") != std::string::npos)
               b.replace(b.find( ".nii"),4,"_backward.nii");
            else if(b.find( ".hdr") != std::string::npos)
               b.replace(b.find( ".hdr"),4,"_backward.hdr");
            else if(b.find( ".img.gz") != std::string::npos)
               b.replace(b.find( ".img.gz"),7,"_backward.img.gz");
            else if(b.find( ".img") != std::string::npos)
               b.replace(b.find( ".img"),4,"_backward.img");
            else if(b.find( ".png") != std::string::npos)
               b.replace(b.find( ".png"),4,"_backward.png");
            else if(b.find( ".nrrd") != std::string::npos)
               b.replace(b.find( ".nrrd"),5,"_backward.nrrd");
            else b.append("_backward.nii");
            reg_io_WriteImageFile(outputWarpedImage[1],b.c_str());
         }
      }
      reg_io_WriteImageFile(outputWarpedImage[0],outputWarpedImageName);
      if(outputWarpedImage[0]!=NULL)
         nifti_image_free(outputWarpedImage[0]);
      outputWarpedImage[0]=NULL;
      if(outputWarpedImage[1]!=NULL)
         nifti_image_free(outputWarpedImage[1]);
      outputWarpedImage[1]=NULL;
      free(outputWarpedImage);
      outputWarpedImage=NULL;
   // Erase the registration object
   delete REG;

   // Clean the allocated images
   if(referenceImage!=NULL) nifti_image_free(referenceImage);
   if(floatingImage!=NULL) nifti_image_free(floatingImage);
   if(inputCCPImage!=NULL) nifti_image_free(inputCCPImage);
   if(referenceMaskImage!=NULL) nifti_image_free(referenceMaskImage);
   if(floatingMaskImage!=NULL) nifti_image_free(floatingMaskImage);

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
