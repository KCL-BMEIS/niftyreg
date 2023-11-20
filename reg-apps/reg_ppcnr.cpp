/**
 * @file reg_ppcnr.cpp
 * @author Andrew Melbourne
 * @brief Executable for 4D non-rigid and affine registration (Registration to a single time point, timeseries mean, local mean or Progressive Principal Component Registration)
 * @date 17/07/2013
 *
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */


#include "_reg_tools.h"
#include "float.h"
#include <limits>
#include <cmath>
#include <string.h>

#ifdef _WINDOWS
#include <time.h>
#endif

using PrecisionType = float;

typedef struct
{
   char *sourceImageName;
   char *affineMatrixName;
   char *inputCPPName;
   char *targetMaskName;
   char *finalResultName;
   char *pcaMaskName;
   const char *outputImageName;
   char *currentImageName;
   float spacing[3];
   int locality;
   int maxIteration;
   int prinComp;
   int tp;
   const char *outputResultName;
   std::string outputCPPName;
} PARAM;

typedef struct
{
   bool sourceImageFlag;
   bool affineMatrixFlag;
   bool affineFlirtFlag;
   bool prinCompFlag;
   bool meanonly;
   bool outputResultFlag;
   bool outputCPPFlag;
   bool backgroundIndexFlag;
   bool pca0;
   bool pca1;
   bool pca2;
   bool pca3;
   bool aladin;
   bool flirt;
   bool tp;
   bool noinit;
   bool pmask;
   bool locality;
   bool autolevel;
   bool makesourcex;
} FLAG;


void PetitUsage(char *exec)
{
   NR_INFO("PROGRESSIVE PRINCIPAL COMPONENT REGISTRATION (PPCNR)");
   NR_INFO("Fast Free-Form Deformation algorithm for dynamic contrast enhanced (DCE) non-rigid registration");
   NR_INFO("Usage:\t" << exec << " -source <sourceImageName> [OPTIONS]");
   NR_INFO("\t\t\t\t*Note that no target image is needed!");
   NR_INFO("\tSee the help for more details (-h)");
}

void Usage(char *exec)
{
   NR_INFO("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
   NR_INFO("PROGRESSIVE PRINCIPAL COMPONENT REGISTRATION (PPCNR).");
   NR_INFO("Fast Free-Form Deformation algorithm for non-rigid DCE-MRI registration.");
   NR_INFO("This implementation is a re-factoring of the PPCR algorithm in:");
   NR_INFO("Melbourne et al., \"Registration of dynamic contrast-enhanced MRI using a ");
   NR_INFO(" progressive principal component registration (PPCR)\", Phys Med Biol, 2007.");
   NR_INFO("This code has been written by Andrew Melbourne (a.melbourne@cs.ucl.ac.uk)");
   NR_INFO("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
   NR_INFO("Usage:\t" << exec << " -source <filename> [OPTIONS]");
   NR_INFO("\t-source <filename>\tFilename of the source image (mandatory)");
   NR_INFO("\t*Note that no target image is needed!\n");
   NR_INFO("   Or   -makesource  <outputname> <n> <filenames> \tThis will generate a 4D volume from the n filenames (saved to <outputname>).");
   NR_INFO("        -makesourcex <outputname> <n> <filenames> \tAs above but exits before registration step'.");
   NR_INFO("        -distribute  <filename> <basename>\t\tThis will generate individual 3D volumes from the 4D filename (saved to '<basename>X.nii', 4D only).");
   NR_INFO("\n*** Main Options:");
   NR_INFO("\t-result <filename> \tFilename of the resampled image [outputResult.nii].");
   NR_INFO("\t-pmask  <filename> \tFilename of the PCA mask region.");
   NR_INFO("\t-cpp    <filename>\tFilename of final 5D control point grid (non-rigid registration only).");
   NR_INFO("     Or -aff    <filename>\tFilename of final concatenated affine transformation (affine registration only).");
   NR_INFO("\n*** Other Options:");
   NR_INFO("\t-prinComp <int>\t\tNumber of principal component iterations to run [#timepoints/2].");
   NR_INFO("\t-maxit    <int>\t\tNumber of registration iterations to run [max(400/prinComp,100)].");
   NR_INFO("\t-autolevel \t\tAutomatically increase registration level during PPCR (switched off with -ln or -lp options)."); // not with -FLIRT
   NR_INFO("\t-pca0 \t\t\tOutput pca images 1:prinComp without registration step [pcaX.nii]."); // i.e. just print out each PCA image.
   NR_INFO("\t-pca1 \t\t\tOutput pca images 1:prinComp for inspection [pcaX.nii].");
   NR_INFO("\t-pca2 \t\t\tOutput intermediate results 1:prinComp for inspection [outX.nii].");
   NR_INFO("\t-pca3 \t\t\tSave current deformation result [cppX.nii].");
   NR_INFO("\t-pca123 \t\tWrite out everything!.");
   NR_INFO("\n*** Alternative Registration Options:");
   NR_INFO("\t-mean \t\t\tIterative registration to the mean image only (no PPCR)."); // registration to the mean is quite inefficient as it uses the ppcr 4D->4D model.
   NR_INFO("\t-locality <int>\t\tIterative registration to the local mean image (pm <int> images - no PPCR).");
   NR_INFO("\t-tp       <int>\t\tIterative registration to single time point (no PPCR).");
   NR_INFO("\t-noinit \t\tTurn off cpp initialisation from previous iteration.");
   //NR_INFO("\t-flirt \t\t\tfor PPCNR using Flirt affine registration (not tested)");
   NR_INFO("\n*** reg_f3d/reg_aladin options are carried through (use reg_f3d -h or reg_aladin -h to see these options).");
   //system("reg_f3d -h");
   NR_INFO("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
}


int main(int argc, char **argv)
{
   time_t start;
   time(&start);

   PARAM *param = (PARAM *)calloc(1,sizeof(PARAM));
   FLAG *flag = (FLAG *)calloc(1,sizeof(FLAG));
   flag->aladin=0;
   flag->flirt=0;
   flag->pca0=0;
   flag->pca1=0;
   flag->pca2=0;
   flag->pca3=0;
   flag->meanonly=0;
   flag->autolevel=0;
   flag->outputCPPFlag=0;
   flag->outputResultFlag=0;
   flag->makesourcex=0;
   flag->prinCompFlag=0;
   flag->tp=0;
   flag->noinit=0;
   param->tp=0;
   param->maxIteration=-1;

   std::string regCommandAll;
   std::string regCommand("-target anchorx.nii -source floatx.nii");
   std::string regCommandF("flirt -ref anchorx.nii -in floatx.nii -out outputResult.nii.gz");
   std::string style, STYL3;

   /* read the input parameters */
   for(int i=1; i<argc; i++)
   {
      if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 ||
            strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 ||
            strcmp(argv[i], "--h")==0 || strcmp(argv[i], "--help")==0)
      {
         Usage(argv[0]);
         return EXIT_SUCCESS;
      }
#ifdef _GIT_HASH
      else if(strcmp(argv[i], "-version")==0 || strcmp(argv[i], "-Version")==0 ||
            strcmp(argv[i], "-V")==0 || strcmp(argv[i], "-v")==0 ||
            strcmp(argv[i], "--v")==0 || strcmp(argv[i], "--version")==0)
      {
         NR_COUT << _GIT_HASH << std::endl;
         return EXIT_SUCCESS;
      }
#endif
      else if(strcmp(argv[i], "-source") == 0)
      {
         param->sourceImageName=argv[++i];
         flag->sourceImageFlag=1;
      }
      else if(strcmp(argv[i], "-makesource") == 0 || strcmp(argv[i], "-makesourcex")==0)
      {
         if(strcmp(argv[i], "-makesourcex")==0)
         {
            flag->makesourcex=1;
         }
         param->finalResultName=argv[++i];
         nifti_image *source = nifti_image_read(argv[i+2],false);
         nifti_image *makesource = nifti_copy_nim_info(source);
         nifti_image_free(source);
         makesource->ndim=makesource->dim[0] = 4;
         makesource->nt = makesource->dim[4] = atoi(argv[++i]);
         makesource->nvox = NiftiImage::calcVoxelNumber(makesource->nx, makesource->ndim);
         makesource->data = malloc(makesource->nvox * makesource->nbyper);
         char *temp_data = reinterpret_cast<char *>(makesource->data);
         for(int ii=0; ii<makesource->nt; ii++) // fill with file data
         {
            NR_COUT << "Reading '" << argv[i+1] << "' (" << ii+1 << " of " << makesource->nt << ")" << std::endl;
            source = nifti_image_read(argv[++i],true);
            memcpy(&(temp_data[ii*source->nvox*source->nbyper]), source->data, source->nbyper*source->nvox);
            nifti_image_free(source);
         }
         nifti_set_filenames(makesource,param->finalResultName, 0, 0); // might want to set this
         nifti_image_write(makesource);
         nifti_image_free(makesource);
         param->sourceImageName=param->finalResultName;
         flag->sourceImageFlag=1;
      }
      else if(strcmp(argv[i], "-distribute") == 0)
      {
         param->finalResultName=argv[i+2];
         nifti_image *source = nifti_image_read(argv[i+1],true);
         nifti_image *makesource = nifti_copy_nim_info(source);
         makesource->ndim=makesource->dim[0] = 3;
         makesource->nt = makesource->dim[4] = 1;
         makesource->nvox = NiftiImage::calcVoxelNumber(makesource, makesource->ndim);
         makesource->data = malloc(makesource->nvox * makesource->nbyper);
         char *temp_data = reinterpret_cast<char *>(source->data);
         for(int ii=0; ii<source->nt; ii++) // fill with file data
         {
            memcpy(makesource->data, &(temp_data[ii*makesource->nvox*source->nbyper]), makesource->nbyper*makesource->nvox);
            const std::string outname=param->finalResultName + std::to_string(ii) + ".nii"s;
            NR_COUT << "Writing '" << outname << "' (" << ii+1 << " of " << source->nt << ")" << std::endl;
            nifti_set_filenames(makesource,outname, 0, 0); // might want to set this
            nifti_image_write(makesource);
         }
         nifti_image_free(source);
         nifti_image_free(makesource);
         return EXIT_SUCCESS;
      }
      else if(strcmp(argv[i], "-pmask") == 0)
      {
         param->pcaMaskName=argv[++i];
         flag->pmask=1;
      }
      else if(strcmp(argv[i], "-target") == 0)
      {
         NR_ERROR("Target image is not necessary!");
         PetitUsage(argv[0]);
      }
      else if(strcmp(argv[i], "-aff") == 0)  // use ppcnr affine
      {
         param->outputCPPName=argv[++i];
         flag->outputCPPFlag=1;
         flag->aladin=1;
      }
      else if(strcmp(argv[i], "-incpp") == 0)  // remove -incpp option
      {
         NR_ERROR("-incpp will not be used!");
      }
      else if(strcmp(argv[i], "-result") == 0)
      {
         param->outputResultName=argv[++i];
         flag->outputResultFlag=1;
      }
      else if(strcmp(argv[i], "-cpp") == 0)
      {
         param->outputCPPName=argv[++i];
         flag->outputCPPFlag=1;
      }
      else if(strcmp(argv[i], "-prinComp") == 0)  // number of pcs to use
      {
         param->prinComp=atoi(argv[++i]);
         flag->prinCompFlag=1;
      }
      else if(strcmp(argv[i], "-locality") == 0)  // number of local images to form mean
      {
         param->locality=atoi(argv[++i]);
         flag->locality=1;
         flag->meanonly=1;
         flag->tp=0;
      }
      else if(strcmp(argv[i], "-tp") == 0)  // number of local images to form mean
      {
         param->tp=atoi(argv[++i]);
         flag->locality=0;
         flag->meanonly=0;
         flag->tp=1;
      }
      else if(strcmp(argv[i], "-pca0") == 0)  // write pca images without registration
      {
         flag->pca0=1;
         flag->pca1=0;
         flag->pca2=0;
         flag->pca3=0;
      }
      else if(strcmp(argv[i], "-pca1") == 0)  // write pca images during registration
      {
         flag->pca0=0;
         flag->pca1=1;
         flag->pca2=0;
         flag->pca3=0;
      }
      else if(strcmp(argv[i], "-pca2") == 0)  // write output images during registration
      {
         flag->pca0=0;
         flag->pca1=0;
         flag->pca2=1;
         flag->pca3=0;
      }
      else if(strcmp(argv[i], "-pca3") == 0)  // write cpp images during registration
      {
         flag->pca0=0;
         flag->pca1=0;
         flag->pca2=0;
         flag->pca3=1;
      }
      else if(strcmp(argv[i], "-pca123") == 0)  // write all output images during registration
      {
         flag->pca0=0;
         flag->pca1=1;
         flag->pca2=1;
         flag->pca3=1;
      }
      else if(strcmp(argv[i], "-mean") == 0)  // iterative registration to the mean
      {
         flag->meanonly=1;
      }
      else if(strcmp(argv[i], "-flirt") == 0)  // one day there will be a flirt option:)
      {
         flag->flirt=1;
      }
      else if(strcmp(argv[i], "-autolevel") == 0)
      {
         flag->autolevel=1;
      }
      else if(strcmp(argv[i], "-noinit") == 0)
      {
         flag->noinit=1;
      }
      else if(strcmp(argv[i], "-lp") == 0)   // force autolevel select off if lp or ln are present.
      {
         flag->autolevel=0;
         regCommand += " "s + argv[i] + " "s + argv[i + 1];
         ++i;
      }
      else if(strcmp(argv[i], "-ln") == 0)   // force autolevel select off if lp or ln are present.
      {
         flag->autolevel=0;
         regCommand += " "s + argv[i] + " "s + argv[i + 1];
         ++i;
      }
      else if(strcmp(argv[i], "-maxit") == 0)  // extract number of registration iterations for display
      {
         param->maxIteration=atoi(argv[i+1]);
         regCommand += " "s + argv[i] + " "s + argv[i + 1];
         ++i;
      }
      else
      {
         regCommand += " "s + argv[i];
      }
   }
   if(flag->makesourcex)
   {
      return EXIT_SUCCESS;  // stop if being used to concatenate 3D images into 4D object.
   }
   if(flag->tp)
   {
      param->prinComp=1;
   }

   if(!flag->sourceImageFlag)
   {
      NR_ERROR("At least define a source image!");
      Usage(argv[0]);
      return EXIT_FAILURE;
   }

   nifti_image *image = nifti_image_read(param->sourceImageName,true);
   if(image == nullptr)
   {
      NR_ERROR("Error when reading image: " << param->sourceImageName);
      return EXIT_FAILURE;
   }
   reg_tools_changeDatatype<PrecisionType>(image); // FIX DATA TYPE - DOES THIS WORK?

   // --- 2) READ/SET IMAGE MASK (4D VOLUME, [NS, SS]) ---
   nifti_image *mask=nullptr;
   if(flag->pmask)
   {
      mask = nifti_image_read(param->pcaMaskName,true);
      if(mask == nullptr)
      {
         NR_ERROR("Error when reading image: " << param->pcaMaskName);
         return EXIT_FAILURE;
      }
      reg_tools_changeDatatype<PrecisionType>(mask);
   }
   else
   {
      mask = nifti_copy_nim_info(image);
      mask->ndim=mask->dim[0]=3;
      mask->nt=mask->dim[4]=1;
      mask->nvox = NiftiImage::calcVoxelNumber(mask, mask->ndim);
      mask->data = malloc(mask->nvox*mask->nbyper);
      PrecisionType *intensityPtrM = static_cast<PrecisionType *>(mask->data);
      for(size_t i=0; i<mask->nvox; i++) intensityPtrM[i]=1.0;
   }
   PrecisionType masksum=0;
   PrecisionType *intensityPtrM = static_cast<PrecisionType *>(mask->data);
   for(size_t i=0; i<mask->nvox; i++)
   {
      if(intensityPtrM[i]) masksum++;
   }

   if(!flag->prinCompFlag && !flag->locality && !flag->meanonly && !flag->tp)
   {
      param->prinComp=std::min(image->nt/2,25);// Check the number of components
   }
   if(param->prinComp>=image->nt) param->prinComp=image->nt-1;
   if(!flag->outputResultFlag) param->outputResultName="ppcnrfinal-img.nii";
//	if(param->maxIteration<0) param->maxIteration=(int)(400/param->prinComp); // number of registration iterations is automatically set here...
//    param->maxIteration=(param->maxIteration<50)?50:param->maxIteration;
   if(param->tp>image->nt) param->tp=image->nt;
   if(flag->aladin)  // decide whether to use affine or free-form
   {
      regCommandAll += "reg_aladin ";
      style += "aff";
      STYL3 += "AFF";
   }
   else if(flag->flirt)
   {
      style += "aff";
   }
   else
   {
      regCommandAll += "reg_f3d ";
      style += "cpp";
      STYL3 += "CPP";
   }
   if(!flag->outputCPPFlag)
      param->outputCPPName = "ppcnrfinal-"s + style + (flag->aladin || flag->flirt ? ".txt"s : ".nii"s);
   regCommandAll += regCommand;
   NR_COUT << style << std::endl;

   /* ****************** */
   /* DISPLAY THE REGISTRATION PARAMETERS */
   /* ****************** */
   PrintCmdLine(argc, argv, true);

   if(flag->meanonly && !flag->locality)
      NR_COUT << "Iterative registration to the mean only (Algorithm will ignore PCA results)----------------" << std::endl;
   else if(flag->meanonly && flag->locality)
      NR_COUT << "Iterative registration to local mean only (pm" << param->locality << ") (Algorithm will ignore PCA results)----------------" << std::endl;
   else if(flag->tp)
      NR_COUT << "Iterative registration to single time point only (" << param->tp << ") (Algorithm will ignore PCA results)----------------" << std::endl;
   else
      NR_COUT << "PPCNR Parameters\n----------------" << std::endl;
   NR_COUT << "Source image name: " << param->sourceImageName << std::endl;
   if(flag->pmask) NR_COUT << "PCA Mask image name: " << param->pcaMaskName << std::endl;
   NR_COUT << "Number of time points: " << image->nt << std::endl;
   NR_COUT << "Number of principal components: " << param->prinComp << std::endl;
   NR_COUT << "Registration max iterations: " << param->maxIteration << std::endl;

   /* ********************** */
   /* START THE REGISTRATION */
   /* ********************** */
   param->outputImageName="anchor.nii";   // NEED TO GET WORKING AND PUT INTERMEDIATE FILES IN SOURCE DIRECTORY.
   nifti_image *images=nifti_dup(*image); // Need to make a new image that has the same info as the original.

   /* ************************************/
   /* FOR NUMBER OF PRINCIPAL COMPONENTS */
   /* ************************************/

   float levels[3];
   float *vsum = new float [param->prinComp];
   for(int i=0; i<param->prinComp; i++) vsum[i]=0.f;
   float *dall = new float [images->nt*param->prinComp];
   levels[0]=-10.0;
   levels[1]=-5.0;
   levels[2]=-2.5;
   int levelNumber=1;
   if(images->nt<3) levelNumber=3;
   PrecisionType *Mean = new PrecisionType [image->nt];
   PrecisionType *Cov = new PrecisionType [image->nt*image->nt];
   PrecisionType cov;

   for(int prinCompNumber=1; prinCompNumber<=param->prinComp; prinCompNumber++)
   {
      param->spacing[0]=levels[(int)(3.0*prinCompNumber/(param->prinComp+1))]; // choose a reducing level number
      NR_COUT << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n";
      NR_COUT << "RUNNING ITERATION " << prinCompNumber << " of " << param->prinComp << "\n";
      NR_COUT << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n";
      NR_COUT << "Running component " << prinCompNumber << " of " << param->prinComp << "\n";
      if(flag->autolevel)
         NR_COUT << "Running " << levelNumber << " levels at " << param->spacing[0] << " spacing\n";
      NR_COUT << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n";

      // Read images and find image means
      unsigned voxelNumber = image->nvox/image->nt;
      PrecisionType *intensityPtr = static_cast<PrecisionType *>(image->data);
      PrecisionType *intensityPtrM = static_cast<PrecisionType *>(mask->data);
      for(int t=0; t<image->nt; t++)
      {
         Mean[t]=0.f;
         for(size_t i=0; i<voxelNumber; i++)
         {
            if(intensityPtrM[i]) Mean[t] += *intensityPtr++;
         }
         Mean[t]/=masksum;
      }

      // calculate covariance matrix
      intensityPtr = static_cast<PrecisionType *>(image->data);
      intensityPtrM = static_cast<PrecisionType *>(mask->data);
      for(int t=0; t<image->nt; t++)
      {
         PrecisionType *currentIntensityPtr2 = &intensityPtr[t*voxelNumber];
         for(int t2=t; t2<image->nt; t2++)
         {
            PrecisionType *currentIntensityPtr1 = &intensityPtr[t*voxelNumber];
            cov=0.f;
            for(size_t i=0; i<voxelNumber; i++)
            {
               if(intensityPtrM[i]) cov += (*currentIntensityPtr1++ - Mean[t]) * (*currentIntensityPtr2++ - Mean[t2]);
            }
            Cov[t+image->nt*t2]=cov/masksum;
            Cov[t2+image->nt*t]=Cov[t+image->nt*t2]; // covariance matrix is symmetric.
         }
      }

      // calculate eigenvalues/vectors...
      // 1. reduce
      int n=image->nt;
      float EPS=1e-15;
      int l,k,j,i;
      float scale,hh,h,g,f;
      float *d = new float [n];
      float *e = new float [n];
      float *z = new float [n*n];
      for(i=0; i<n; i++)
      {
         for(j=0; j<n; j++)
         {
            z[i+n*j]=Cov[i+n*j];
         }
      }
      for (i=n-1; i>0; i--)
      {
         l=i-1;
         h=scale=0;
         if(l>0)
         {
            for(k=0; k<i; k++)
               scale+=std::abs(z[i+n*k]);
            if (scale==0)
               e[i]=z[i+n*l];
            else
            {
               for(k=0; k<i; k++)
               {
                  z[i+n*k] /= scale;
                  h+=z[i+n*k]*z[i+n*k];
               }
               f=z[i+n*l];
               g=(f>=0 ? -sqrt(h) : sqrt(h));
               e[i]=scale*g;
               h-=f*g;
               z[i+n*l]=f-g;
               f=0;
               for (j=0; j<i; j++)
               {
                  z[j+n*i]=z[i+n*j]/h;
                  g=0;
                  for (k=0; k<j+1; k++)
                     g+=z[j+n*k]*z[i+n*k];
                  for (k=j+1; k<i; k++)
                     g+= z[k+n*j]*z[i+n*k];
                  e[j]=g/h;
                  f+=e[j]*z[i+n*j];
               }
               hh=f/(h+h);
               for (j=0; j<i; j++)
               {
                  f=z[i+n*j];
                  e[j]=g=e[j]-hh*f;
                  for (k=0; k<j+1; k++)
                     z[j+n*k]-=(f*e[k]+g*z[i+n*k]);
               }
            }
         }
         else
            e[i]=z[i+n*l];
         d[i]=h;
      }
      d[0]=0;
      e[0]=0;
      for (i=0; i<n; i++)
      {
         if(d[i]!=0)
         {
            for (j=0; j<i; j++)
            {
               g=0;
               for (k=0; k<i; k++)
                  g+=z[i+n*k]*z[k+n*j];
               for (k=0; k<i; k++)
                  z[k+n*j]-=g*z[k+n*i];
            }
         }
         d[i]=z[i+n*i];
         z[i+n*i]=1.0;
         for (j=0; j<i; j++) z[j+n*i]=z[i+n*j]=0;
      }

      NR_COUT << "Image Means=[" << Mean[0];
      for(int i=1; i<image->nt; i++)
         NR_COUT << "," << Mean[i]; // not sure it's quite right...
      NR_COUT << "]\n";
      for(int i=0; i<image->nt; i++)
      {
         NR_COUT << "Cov=[" << Cov[i+n*0];
         for(int j=1; j<image->nt; j++)
            NR_COUT << "," << Cov[i+n*j];
         NR_COUT << "]\n";
      }

      // 2. diagonalise
      int m,iter;
      float s,r,p,dd,c,b;
      for (i=1; i<n; i++) e[i-1]=e[i];
      e[n-1]=0;
      for (l=0; l<n; l++)
      {
         iter=0;
         do
         {
            for (m=l; m<n-1; m++)
            {
               dd=std::abs(d[m])+std::abs(d[m+1]);
               if(std::abs(e[m])<=EPS*dd) break;
            }
            if(m!=l)
            {
               if(iter++==30) break;
               g=(d[l+1]-d[l])/(2.0*e[l]);
               r=sqrt(g*g+1.0);
               g=d[m]-d[l]+e[l]/(g+std::abs(r)*g/std::abs(g));
               s=c=1.0;
               p=0;
               for (i=m-1; i>=l; i--)
               {
                  f=s*e[i];
                  b=c*e[i];
                  e[i+1]=(r=sqrt(f*f+g*g));
                  if(r<EPS)
                  {
                     d[i+1]-=p;
                     e[m]=0;
                     break;
                  }
                  s=f/r;
                  c=g/r;
                  g=d[i+1]-p;
                  r=(d[i]-g)*s+2.0*c*b;
                  d[i+1]=g+(p=s*r);
                  g=c*r-b;
                  for (k=0; k<n; k++)
                  {
                     f=z[k+n*(i+1)];
                     z[k+n*(i+1)]=s*z[k+n*i]+c*f;
                     z[k+n*i]=c*z[k+n*i]-s*f;
                  }
               }
               if(r<EPS && i>=l) continue;
               d[l]-=p;
               e[l]=g;
               e[m]=0;
            }
         }
         while(m!=l);
      } // Seems to be ok for an arbitrary covariance matrix.

      // 3. sort eigenvalues & eigenvectors
      for(int i=0; i<n-1; i++)
      {
         float p=d[k=i];
         for(int j=i; j<n; j++)
            if(d[j]>=p) p=d[k=j];
         if(k!=i)
         {
            d[k]=d[i];
            d[i]=p;
            if(z != nullptr)
               for(int j=0; j<n; j++)
               {
                  p=z[j+n*i];
                  z[j+n*i]=z[j+n*k];
                  z[j+n*k]=p;
               }
         }
      }
      NR_COUT << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n";
      for(int i=0; i<image->nt; i++)
      {
         NR_COUT << "EVMatrix=[" << z[i+n*0];
         for(int j=1; j<image->nt; j++)
            NR_COUT << "," << z[i+image->nt*j];
         NR_COUT << "]\n";
      }
      NR_COUT << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n";
      NR_COUT << "Eigenvalues=[" << d[0];
      for(int i=0; i<image->nt; i++)
      {
         if(i>0)
            NR_COUT << "," << d[i];
         vsum[prinCompNumber-1]+=d[i];
         dall[i+image->nt*prinCompNumber-1]=d[i];
      }
      NR_COUT << "]\n";
      for(j=0; j<prinCompNumber; j++)
      {
         NR_COUT << "Variances(" << j+1 << ")=[" << 100.0*dall[0+n*j]/vsum[j];
         for(int i=1; i<image->nt; i++)
            NR_COUT << "," << 100.0*dall[i+image->nt*j]/vsum[j];
         NR_COUT << "]\n";
      }
      if(flag->meanonly)
      {
         NR_COUT << "Iterative registration to mean only - eigenvector matrix overwritten.\n";
         for(int i=0; i<image->nt; i++)
            for(int j=0; j<image->nt; j++)
               z[i+image->nt*j]=1.0/sqrtf(image->nt*prinCompNumber); // is this right?! - if using NMI it's rather moot so I'm not too bothered at the moment...
      }
      if(flag->locality) NR_COUT << "Iterative registration to local mean only (pm " << param->locality << " images).\n";
      if(flag->tp) NR_COUT << "Registration to single time point (" << param->tp << ").\n";

      // 4. rebuild images
      nifti_image *imagep=nifti_dup(*image, false); // Need to make a new image that has the same info as the original.
      float dotty,sum;
      if(flag->locality)  // local mean
      {
         PrecisionType *intensityPtr1 = static_cast<PrecisionType *>(image->data);
         PrecisionType *intensityPtr2 = static_cast<PrecisionType *>(imagep->data);
         for(size_t i=0; i<voxelNumber; i++)
         {
            for(int t=0; t<image->nt; t++)
            {
               dotty=0;
               sum=0;
               for(int tt=std::max(t-param->locality,0); tt<=std::min(t+param->locality,image->nt); tt++)
               {
                  dotty += intensityPtr1[tt*voxelNumber+i];
                  sum++;
               }
               intensityPtr2[t*voxelNumber+i]=dotty/sum;
            }
         }
      }
      else if(flag->tp)  // single time point
      {
         PrecisionType *intensityPtr1 = static_cast<PrecisionType *>(image->data);
         PrecisionType *intensityPtr2 = static_cast<PrecisionType *>(imagep->data);
         for(size_t i=0; i<voxelNumber; i++)
         {
            for(int t=0; t<image->nt; t++)
            {
               intensityPtr2[t*voxelNumber+i]=intensityPtr1[param->tp*voxelNumber+i];
            }
         }
      }
      else  // ppcr and mean
      {
         PrecisionType *intensityPtr1 = static_cast<PrecisionType *>(image->data);
         PrecisionType *intensityPtr2 = static_cast<PrecisionType *>(imagep->data);
         for(size_t i=0; i<voxelNumber; i++)
         {
            for(int c=0; c<prinCompNumber; c++) // Add up component contributions
            {
               dotty=0;
               for(int t=0; t<image->nt; t++) // 1) Multiply each element by eigenvector and add (I.e. dot product)
               {
                  dotty += intensityPtr1[t*voxelNumber+i] * z[t+image->nt*c];
               }
               for(int t=0; t<image->nt; t++) // 2) Multiply value above by that eigenvector and write these to the image data
               {
                  intensityPtr2[t*voxelNumber+i] += dotty * z[t+image->nt*c];
               }
            }
         }
      }
      nifti_set_filenames(imagep, ("pca"s + std::to_string(prinCompNumber) + ".nii"s).c_str(), 0, 0);
      if(flag->pca0 | flag->pca1)
         nifti_image_write(imagep);

      if(!flag->pca0)
      {
         /* ****************************/
         /* FOR NUMBER OF 'TIMEPOINTS' */
         /* ****************************/
         // current: images // these are both open: perpetual source
         // target:  imagep //					   pca target
         PrecisionType *intensityPtrP = static_cast<PrecisionType *>(imagep->data); // pointer to pca-anchor data
         PrecisionType *intensityPtrS = static_cast<PrecisionType *>(images->data); // pointer to real source-float data
         PrecisionType *intensityPtrC = static_cast<PrecisionType *>(image->data); // pointer to updated 'current' data
         for(int imageNumber=0; imageNumber<images->nt; imageNumber++)
         {
            // ROLLING FLOAT AND ANCHOR IMAGES
            nifti_image *stores = nifti_copy_nim_info(images);
            stores->ndim=stores->dim[0]=3;
            stores->nt=stores->dim[4]=1;
            stores->nvox = NiftiImage::calcVoxelNumber(stores, stores->ndim);
            stores->data = calloc(stores->nvox,images->nbyper);

            nifti_image *storet = nifti_dup(*stores, false);

            // COPY THE APPROPRIATE VALUES
            PrecisionType *intensityPtrPP = static_cast<PrecisionType *>(storet->data); // 3D real source image (needs current cpp image)
            PrecisionType *intensityPtrSS = static_cast<PrecisionType *>(stores->data); // 3D pca-float data
            memcpy(intensityPtrPP, &intensityPtrP[imageNumber*storet->nvox], storet->nvox*storet->nbyper);
            memcpy(intensityPtrSS, &intensityPtrS[imageNumber*stores->nvox], stores->nvox*stores->nbyper);

            nifti_set_filenames(stores,"outputResult.nii", 0, 0); // Fail-safe for reg_f3d failure
            nifti_image_write(stores);
            nifti_set_filenames(stores,"floatx.nii", 0, 0); // TODO NAME
            nifti_image_write(stores);
            nifti_image_free(stores);
            nifti_set_filenames(storet,"anchorx.nii", 0, 0); // TODO NAME
            nifti_image_write(storet);
            nifti_image_free(storet);

            std::string regCommandB;
            if(!flag->flirt)
            {
               const std::string temp = "float"s + style + std::to_string(imageNumber + 1) + (flag->aladin ? ".txt"s : ".nii"s);
               regCommandB = regCommandAll + " -"s + style + " "s + temp;
               if(flag->autolevel)
               {
                  regCommandB += " -ln "s + std::to_string(levelNumber);
                  if(!flag->aladin)
                     regCommandB += " -sx "s + std::to_string(param->spacing[0]);
               }
               if(prinCompNumber>1 && !flag->noinit)
                  regCommandB += " -in"s + style + temp;
            }
            else  // flirt -ref -in -out -omat -init
            {
               const std::string temp = "float"s + style + std::to_string(imageNumber + 1) + ".txt"s;
               regCommandB = regCommandF + " -omat "s + temp;
               if(prinCompNumber>1 && !flag->noinit)
                  regCommandB += " -init "s + temp + ";gunzip -f outputResult.nii.gz";
            }

            // DO REGISTRATION
            NR_COUT << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n";
            NR_COUT << "RUNNING ITERATION " << prinCompNumber << " of " << param->prinComp << "\n";
            NR_COUT << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n";
            NR_COUT << "Registering image " << imageNumber+1 << " of " << images->nt << "\n";
            NR_COUT << "'" << regCommandB << "'\n";
            //system(regCommandB);

            if(system(regCommandB))
            {
               NR_ERROR("Error while running the following command: "s + regCommandB);
               return EXIT_FAILURE;
            }

            // READ IN RESULT AND MAKE A NEW CURRENT IMAGE 'image'
            stores = nifti_image_read("outputResult.nii",true); // TODO NAME
            PrecisionType *intensityPtrCC = static_cast<PrecisionType *>(stores->data); // 3D result image
            memcpy(&intensityPtrC[imageNumber*stores->nvox], intensityPtrCC, stores->nvox*stores->nbyper);
            nifti_image_free(stores);
         }
      }
      nifti_image_free(imagep);
      nifti_set_filenames(image, ("out"s + std::to_string(prinCompNumber) + ".nii"s).c_str(), 0, 0);
      if(flag->pca2)
         nifti_image_write(image);
      if(flag->pca3)
      {
         const std::string cppname = "cpp"s + std::to_string(prinCompNumber) + ".nii"s;
         if(!flag->aladin & !flag->flirt)
         {
            nifti_image *dof = nifti_image_read(("float"s + style + "1.nii"s).c_str(), true);
            nifti_image *dofs = nifti_copy_nim_info(dof);
            dofs->nt = dofs->dim[4] = images->nt;
            dofs->nvox = dof->nvox*images->nt;
            dofs->data = (PrecisionType *)calloc(dofs->nvox, dof->nbyper);
            PrecisionType *intensityPtrD = static_cast<PrecisionType *>(dofs->data);
            for(int t=0; t<images->nt; t++)
            {
               nifti_image *dof = nifti_image_read(("float"s + style + std::to_string(t + 1) + ".nii"s).c_str(), true);
               PrecisionType *intensityPtrDD = static_cast<PrecisionType *>(dof->data);
               int r=dof->nvox/3.0;
               for(int i=0; i<3; i++)
               {
                  memcpy(&intensityPtrD[i*image->nt*r+t*r], &intensityPtrDD[i*r], dof->nbyper*r);
               }
               nifti_image_free(dof);
            }
            nifti_set_filenames(dofs,cppname.c_str(), 0, 0); // TODO NAME 	// write final dof data
            nifti_image_write(dofs);
            nifti_image_free(dofs);
         }
         else
         {
            std::string final_string = "";
            for(int t=0; t<images->nt; t++)
            {
               std::ifstream ifs("float"s + style + std::to_string(t + 1) + ".txt"s);
               std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
               final_string += str;
            }
            std::ofstream ofs(cppname);
            ofs << final_string;
         }

      }
   } // End PC's
   NR_COUT << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n";
   NR_COUT << "Finished Iterations and now writing outputs...\n";

   // WRITE OUT RESULT IMAGE AND RESULT DOF
   // Read in images and put into single object
   if(!flag->pca0)
   {
      if(!flag->aladin & !flag->flirt)
      {
         nifti_image *dof = nifti_image_read(("float"s + style + "1.nii"s).c_str(),true);
         nifti_image *dofs = nifti_copy_nim_info(dof);
         dofs->nt = dofs->dim[4] = images->nt;
         dofs->nvox = dof->nvox*images->nt;
         dofs->data = (PrecisionType *)calloc(dofs->nvox, dof->nbyper);
         PrecisionType *intensityPtrD = static_cast<PrecisionType *>(dofs->data);
         for(int t=0; t<images->nt; t++)
         {
            const std::string filename = "float"s + style + std::to_string(t + 1) + ".nii"s;
            nifti_image *dof = nifti_image_read(filename.c_str(),true);
            PrecisionType *intensityPtrDD = static_cast<PrecisionType *>(dof->data);
            int r=dof->nvox/3.0;
            for(int i=0; i<3; i++)
               memcpy(&intensityPtrD[i*image->nt*r+t*r], &intensityPtrDD[i*r], dof->nbyper*r);
            nifti_image_free(dof);
            remove(filename.c_str()); // delete spare floatcpp files
         }
         nifti_set_filenames(dofs,param->outputCPPName.c_str(), 0, 0); // TODO NAME 	// write final dof data
         nifti_image_write(dofs);
         nifti_image_free(dofs);
      }
      else
      {
         std::string final_string;
         for(int t=0; t<images->nt; t++)
         {
            const std::string filename = "float"s + style + std::to_string(t + 1) + ".txt"s;
            std::ifstream ifs(filename);
            std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
            final_string += str;
            remove(filename.c_str());
         }
         std::ofstream ofs(param->outputCPPName);
         ofs << final_string;
      }

      // DELETE
      // delete: anchorx.nii floatx.nii outputResult.nii : I think this is all...
      remove("anchorx.nii");  // flakey...
      remove("floatx.nii");
      remove("outputResult.nii");
      remove("outputResult.nii.gz");

      // Write final image data
      nifti_set_filenames(image,param->outputResultName, 0, 0);
      nifti_image_write(image);
   }
   nifti_image_free(image);
   nifti_image_free(mask);

   time_t end;
   time( &end );
   int minutes = Floor(float(end-start)/60.0f);
   int seconds = (int)(end-start - 60*minutes);
   NR_COUT << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n";
   if(flag->locality)
      NR_COUT << "Registration to " << param->locality << "-local mean with " << param->prinComp << " iterations performed in " << minutes << " min " << seconds << " sec\n";
   if(flag->tp)
      NR_COUT << "Single time point registration to image " << param->tp << " performed in " << minutes << " min " << seconds << " sec\n";
   if(flag->meanonly & !flag->locality)
      NR_COUT << "Registration to mean image with " << param->prinComp << " iterations performed in " << minutes << " min " << seconds << " sec\n";
   if(!flag->locality & !flag->meanonly & !flag->tp)
      NR_COUT << "PPCNR registration with " << param->prinComp << " iterations performed in " << minutes << " min " << seconds << " sec\n";
   NR_COUT << "Have a good day!" << std::endl;

   // CHECK CLEAN-UP
   free( flag );
   free( param );

   return EXIT_SUCCESS;
}
