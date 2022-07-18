/**
 * @file reg_resample.cpp
 * @author Marc Modat
 * @date 18/05/2009
 *
 *  Created by Marc Modat on 18/05/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_ReadWriteImage.h"
#include "_reg_ReadWriteMatrix.h"
#include "_reg_resampling.h"
#include "_reg_globalTrans.h"
#include "_reg_localTrans.h"
#include "_reg_localTrans_jac.h"
#include "_reg_tools.h"
#include "reg_resample.h"

typedef struct
{
   char *referenceImageName;
   char *floatingImageName;
   char *inputTransName;
   char *outputResultName;
   char *outputBlankName;
   float sourceBGValue;
   int interpolation;
   float paddingValue;
   float PSF_Algorithm;
} PARAM;
typedef struct
{
   bool referenceImageFlag;
   bool floatingImageFlag;
   bool inputTransFlag;
   bool outputResultFlag;
   bool outputBlankFlag;
   bool outputBlankXYFlag;
   bool outputBlankYZFlag;
   bool outputBlankXZFlag;
   bool isTensor;
   bool usePSF;
} FLAG;


void PetitUsage(char *exec)
{
   fprintf(stderr,"Usage:\t%s -ref <referenceImageName> -flo <floatingImageName> [OPTIONS].\n",exec);
   fprintf(stderr,"\tSee the help for more details (-h).\n");
   return;
}
void Usage(char *exec)
{
   printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
   printf("Usage:\t%s -ref <filename> -flo <filename> [OPTIONS].\n",exec);
   printf("\t-ref <filename>\n\t\tFilename of the reference image (mandatory)\n");
   printf("\t-flo <filename>\n\t\tFilename of the floating image (mandatory)\n\n");
   printf("* * OPTIONS * *\n");
   printf("\t-trans <filename>\n\t\tFilename of the file containing the transformation parametrisation (from reg_aladin, reg_f3d or reg_transform)\n");
   printf("\t-res <filename>\n\t\tFilename of the resampled image [none]\n");
   printf("\t-blank <filename>\n\t\tFilename of the resampled blank grid [none]\n");
   printf("\t-inter <int>\n\t\tInterpolation order (0, 1, 3, 4)[3] (0=NN, 1=LIN; 3=CUB, 4=SINC)\n");
   printf("\t-pad <int>\n\t\tInterpolation padding value [0]\n");
   printf("\t-tensor\n\t\tThe last six timepoints of the floating image are considered to be tensor order as XX, XY, YY, XZ, YZ, ZZ [off]\n");
   printf("\t-psf\n\t\tPerform the resampling in two steps to resample an image to a lower resolution [off]\n");
   printf("\t-psf_alg <0/1>\n\t\tMinimise the matrix metric (0) or the determinant (1) when estimating the PSF [0]\n");
   printf("\t-voff\n\t\tTurns verbose off [on]\n");
#if defined (_OPENMP)
   int defaultOpenMPValue=omp_get_num_procs();
   if(getenv("OMP_NUM_THREADS")!=NULL)
      defaultOpenMPValue=atoi(getenv("OMP_NUM_THREADS"));
   printf("\t-omp <int>\n\t\tNumber of thread to use with OpenMP. [%i/%i]\n",
          defaultOpenMPValue, omp_get_num_procs());
#endif
   printf("\t--version\n\t\tPrint current version and exit (%s)\n",NR_VERSION);
   printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
   return;
}

int main(int argc, char **argv)
{
   PARAM *param = (PARAM *)calloc(1,sizeof(PARAM));
   FLAG *flag = (FLAG *)calloc(1,sizeof(FLAG));

   param->interpolation=3; // Cubic spline interpolation used by default
   param->paddingValue=0;
   param->PSF_Algorithm=0;
   bool verbose=true;

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
      else if(strcmp(argv[i], "--xml")==0)
      {
         printf("%s",xml_resample);
         return EXIT_SUCCESS;
      }
      else if(strcmp(argv[i], "-voff")==0)
      {
         verbose=false;
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
      else if( strcmp(argv[i], "-version")==0 ||
               strcmp(argv[i], "-Version")==0 ||
               strcmp(argv[i], "-V")==0 ||
               strcmp(argv[i], "-v")==0 ||
               strcmp(argv[i], "--v")==0 ||
               strcmp(argv[i], "--version")==0)
      {
         printf("%s\n",NR_VERSION);
         return EXIT_SUCCESS;
      }
      else if((strcmp(argv[i],"-ref")==0) || (strcmp(argv[i],"-target")==0) ||
              (strcmp(argv[i],"--ref")==0))
      {
         param->referenceImageName=argv[++i];
         flag->referenceImageFlag=1;
      }
      else if((strcmp(argv[i],"-flo")==0) || (strcmp(argv[i],"-source")==0) ||
              (strcmp(argv[i],"--flo")==0))
      {
         param->floatingImageName=argv[++i];
         flag->floatingImageFlag=1;
      }
      else if((strcmp(argv[i],"-res")==0) || (strcmp(argv[i],"-result")==0) ||
              (strcmp(argv[i],"--res")==0))
      {
         param->outputResultName=argv[++i];
         flag->outputResultFlag=1;
      }
      else if(strcmp(argv[i], "-trans") == 0 ||
              strcmp(argv[i],"--trans")==0 ||
              strcmp(argv[i],"-aff")==0 || // added for backward compatibility
              strcmp(argv[i],"-def")==0 || // added for backward compatibility
              strcmp(argv[i],"-cpp")==0 )  // added for backward compatibility
      {
         param->inputTransName=argv[++i];
         flag->inputTransFlag=1;
      }
      else if(strcmp(argv[i], "-inter") == 0 ||
              (strcmp(argv[i],"--inter")==0))
      {
         param->interpolation=atoi(argv[++i]);
      }
      else if(strcmp(argv[i], "-NN") == 0)
      {
         param->interpolation=0;
      }
      else if(strcmp(argv[i], "-LIN") == 0 ||
              (strcmp(argv[i],"-TRI")==0))
      {
         param->interpolation=1;
      }
      else if(strcmp(argv[i], "-CUB") == 0 ||
              (strcmp(argv[i],"-SPL")==0))
      {
         param->interpolation=3;
      }
      else if(strcmp(argv[i], "-SINC") == 0)
      {
         param->interpolation=4;
      }
      else if(strcmp(argv[i], "-pad") == 0 ||
              (strcmp(argv[i],"--pad")==0))
      {
         param->paddingValue=(float)atof(argv[++i]);
      }
      else if(strcmp(argv[i], "-blank") == 0 ||
              (strcmp(argv[i],"--blank")==0))
      {
         param->outputBlankName=argv[++i];
         flag->outputBlankFlag=1;
      }
      else if(strcmp(argv[i], "-blankXY") == 0 ||
              (strcmp(argv[i],"--blankXY")==0))
      {
         param->outputBlankName=argv[++i];
         flag->outputBlankXYFlag=1;
      }
      else if(strcmp(argv[i], "-blankYZ") == 0 ||
              (strcmp(argv[i],"--blankYZ")==0))
      {
         param->outputBlankName=argv[++i];
         flag->outputBlankYZFlag=1;
      }
      else if(strcmp(argv[i], "-blankXZ") == 0 ||
              (strcmp(argv[i],"--blankXZ")==0))
      {
         param->outputBlankName=argv[++i];
         flag->outputBlankXZFlag=1;
      }
      else if(strcmp(argv[i], "-tensor") == 0 ||
              (strcmp(argv[i],"--tensor")==0))
      {
         flag->isTensor=true;
      }
      else if(strcmp(argv[i], "-psf") == 0 ||
              (strcmp(argv[i],"--psf")==0))
      {
         flag->usePSF=true;
      }
      else if(strcmp(argv[i], "-psf_alg") == 0 ||
              (strcmp(argv[i],"--psf_alg")==0))
      {
         param->PSF_Algorithm=(float)atof(argv[++i]);
      }
      else
      {
         fprintf(stderr,"Err:\tParameter %s unknown.\n",argv[i]);
         PetitUsage(argv[0]);
         return EXIT_FAILURE;
      }
   }

   if(!flag->referenceImageFlag || !flag->floatingImageFlag)
   {
      fprintf(stderr,"[NiftyReg ERROR] The reference and the floating image have both to be defined.\n");
      PetitUsage(argv[0]);
      return EXIT_FAILURE;
   }

   /* Read the reference image */
   nifti_image *referenceImage = reg_io_ReadImageHeader(param->referenceImageName);
   if(referenceImage == NULL)
   {
      fprintf(stderr,"[NiftyReg ERROR] Error when reading the reference image: %s\n",
              param->referenceImageName);
      return EXIT_FAILURE;
   }

   /* Read the floating image */
   nifti_image *floatingImage = reg_io_ReadImageFile(param->floatingImageName);
   if(floatingImage == NULL)
   {
      fprintf(stderr,"[NiftyReg ERROR] Error when reading the floating image: %s\n",
              param->floatingImageName);
      return EXIT_FAILURE;
   }

   /* *********************************** */
   /* DISPLAY THE RESAMPLING PARAMETERS */
   /* *********************************** */
   if(verbose){
      printf("\n* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
      printf("Command line:\n");
      for(int i=0; i<argc; i++) printf(" %s", argv[i]);
      printf("\n");
      printf("Parameters\n");
      printf("Reference image name: %s\n",referenceImage->fname);
      printf("\t%ix%ix%i voxels, %i volumes\n",referenceImage->nx,referenceImage->ny,referenceImage->nz,referenceImage->nt);
      printf("\t%gx%gx%g mm\n",referenceImage->dx,referenceImage->dy,referenceImage->dz);
      printf("Floating image name: %s\n",floatingImage->fname);
      printf("\t%ix%ix%i voxels, %i volumes\n",floatingImage->nx,floatingImage->ny,floatingImage->nz,floatingImage->nt);
      printf("\t%gx%gx%g mm\n",floatingImage->dx,floatingImage->dy,floatingImage->dz);
      printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n\n");
   }

   /* *********************** */
   /* READ THE TRANSFORMATION */
   /* *********************** */
   nifti_image *inputTransformationImage = NULL;
   mat44 inputAffineTransformation;
   // Check if a transformation has been specified
   if(flag->inputTransFlag)
   {
      // First check if the input filename is an image
      if(reg_isAnImageFileName(param->inputTransName))
      {
         inputTransformationImage=reg_io_ReadImageFile(param->inputTransName);
         if(inputTransformationImage==NULL)
         {
            fprintf(stderr, "[NiftyReg ERROR] Error when reading the provided transformation: %s\n",
                    param->inputTransName);
            return EXIT_FAILURE;
         }
      }
      else
      {
         // the transformation is assumed to be affine
         reg_tool_ReadAffineFile(&inputAffineTransformation,
                                 param->inputTransName);
      }
   }
   else
   {
      // No transformation is specified, an identity transformation is used
      reg_mat44_eye(&inputAffineTransformation);
   }

   // Create a deformation field
   nifti_image *deformationFieldImage = nifti_copy_nim_info(referenceImage);
   deformationFieldImage->dim[0]=deformationFieldImage->ndim=5;
   deformationFieldImage->dim[1]=deformationFieldImage->nx=referenceImage->nx;
   deformationFieldImage->dim[2]=deformationFieldImage->ny=referenceImage->ny;
   deformationFieldImage->dim[3]=deformationFieldImage->nz=referenceImage->nz;
   deformationFieldImage->dim[4]=deformationFieldImage->nt=1;
   deformationFieldImage->pixdim[4]=deformationFieldImage->dt=1.0;
   deformationFieldImage->dim[5]=deformationFieldImage->nu=referenceImage->nz>1?3:2;
   deformationFieldImage->dim[6]=deformationFieldImage->nv=1;
   deformationFieldImage->dim[7]=deformationFieldImage->nw=1;
   deformationFieldImage->nvox =(size_t)deformationFieldImage->nx*
         deformationFieldImage->ny*deformationFieldImage->nz*
         deformationFieldImage->nt*deformationFieldImage->nu;
   deformationFieldImage->scl_slope=1.f;
   deformationFieldImage->scl_inter=0.f;
   if(inputTransformationImage!=NULL)
   {
      deformationFieldImage->datatype = inputTransformationImage->datatype;
      deformationFieldImage->nbyper = inputTransformationImage->nbyper;
   }
   else
   {
      deformationFieldImage->datatype = NIFTI_TYPE_FLOAT32;
      deformationFieldImage->nbyper = sizeof(float);
   }
   deformationFieldImage->data = (void *)calloc(deformationFieldImage->nvox, deformationFieldImage->nbyper);

   // Initialise the deformation field with an identity transformation
   reg_tools_multiplyValueToImage(deformationFieldImage,deformationFieldImage,0.f);
   reg_getDeformationFromDisplacement(deformationFieldImage);
   deformationFieldImage->intent_p1=DEF_FIELD;

   // Compute the transformation to apply
   if(inputTransformationImage!=NULL)
   {
      switch(static_cast<int>(inputTransformationImage->intent_p1))
      {
      case LIN_SPLINE_GRID:
      case CUB_SPLINE_GRID:
         reg_spline_getDeformationField(inputTransformationImage,
                                        deformationFieldImage,
                                        NULL,
                                        false,
                                        true);
         break;
      case DISP_VEL_FIELD:
         reg_getDeformationFromDisplacement(inputTransformationImage);
      case DEF_VEL_FIELD:
         {
            nifti_image *tempFlowField = nifti_copy_nim_info(deformationFieldImage);
            tempFlowField->data = (void *)malloc(tempFlowField->nvox*tempFlowField->nbyper);
            memcpy(tempFlowField->data,deformationFieldImage->data,
                   tempFlowField->nvox*tempFlowField->nbyper);
            reg_defField_compose(inputTransformationImage,
                                 tempFlowField,
                                 NULL);
            tempFlowField->intent_p1=inputTransformationImage->intent_p1;
            tempFlowField->intent_p2=inputTransformationImage->intent_p2;
            reg_defField_getDeformationFieldFromFlowField(tempFlowField,
                                                          deformationFieldImage,
                                                          false);
            nifti_image_free(tempFlowField);
         }
         break;
      case SPLINE_VEL_GRID:
         reg_spline_getDefFieldFromVelocityGrid(inputTransformationImage,
                                                deformationFieldImage,
                                                false);
         break;
      case DISP_FIELD:
         reg_getDeformationFromDisplacement(inputTransformationImage);
      default:
         reg_defField_compose(inputTransformationImage,
                              deformationFieldImage,
                              NULL);
         break;
      }
      nifti_image_free(inputTransformationImage);
      inputTransformationImage=NULL;
   }
   else
   {
      reg_affine_getDeformationField(&inputAffineTransformation,
                                     deformationFieldImage,
                                     false,
                                     NULL);
   }


   /* ************************* */
   /* WARP THE FLOATING IMAGE */
   /* ************************* */
   if(flag->outputResultFlag)
   {
      switch(param->interpolation)
      {
      case 0:
         param->interpolation=0;
         break;
      case 1:
         param->interpolation=1;
         break;
      case 4:
         param->interpolation=4;
         break;
      default:
         param->interpolation=3;
         break;
      }
      nifti_image *warpedImage = nifti_copy_nim_info(referenceImage);
      warpedImage->dim[0]=warpedImage->ndim=floatingImage->dim[0];
      warpedImage->dim[4]=warpedImage->nt=floatingImage->dim[4];
      warpedImage->dim[5]=warpedImage->nu=floatingImage->dim[5];
      warpedImage->cal_min=floatingImage->cal_min;
      warpedImage->cal_max=floatingImage->cal_max;
      warpedImage->scl_slope=floatingImage->scl_slope;
      warpedImage->scl_inter=floatingImage->scl_inter;
      if(param->paddingValue!=param->paddingValue &&
            (floatingImage->datatype!=NIFTI_TYPE_FLOAT32 ||
             floatingImage->datatype!=NIFTI_TYPE_FLOAT64)){
         warpedImage->datatype = NIFTI_TYPE_FLOAT32;
         reg_tools_changeDatatype<float>(floatingImage);
      }
      else warpedImage->datatype = floatingImage->datatype;
      warpedImage->intent_code=floatingImage->intent_code;
      memset(warpedImage->intent_name, 0, 16);
      strcpy(warpedImage->intent_name,floatingImage->intent_name);
      warpedImage->intent_p1=floatingImage->intent_p1;
      warpedImage->intent_p2=floatingImage->intent_p2;
      warpedImage->nbyper = floatingImage->nbyper;
      warpedImage->nvox = (size_t)warpedImage->dim[1] * warpedImage->dim[2] *
            warpedImage->dim[3] * warpedImage->dim[4] * warpedImage->dim[5];
      warpedImage->data = (void *)calloc(warpedImage->nvox, warpedImage->nbyper);

      if((floatingImage->dim[4]==6 || floatingImage->dim[4]==7) && flag->isTensor==true)
      {
#ifndef NDEBUG
         reg_print_msg_debug("DTI-based resampling\n");
#endif
         // Compute first the Jacobian matrices
         mat33 *jacobian = (mat33 *)malloc(deformationFieldImage->nx *
                                           deformationFieldImage->ny *
                                           deformationFieldImage->nz *
                                           sizeof(mat33));
         reg_defField_getJacobianMatrix(deformationFieldImage,
                                        jacobian);
         // resample the DTI image
         bool timepoints[7];
         for(int i=0; i<7; ++i) timepoints[i]=true;
         if(floatingImage->dim[4]==7) timepoints[0]=false;
         reg_resampleImage(floatingImage,
                           warpedImage,
                           deformationFieldImage,
                           NULL,
                           param->interpolation,
                           std::numeric_limits<float>::quiet_NaN(),
                           timepoints,
                           jacobian
                           );
      }
      else{
         if(flag->usePSF){
            // Compute first the Jacobian matrices
            mat33 *jacobian = (mat33 *)malloc(deformationFieldImage->nx *
                                              deformationFieldImage->ny *
                                              deformationFieldImage->nz *
                                              sizeof(mat33));
            reg_defField_getJacobianMatrix(deformationFieldImage,
                                           jacobian);


            reg_resampleImage_PSF(floatingImage,
                                  warpedImage,
                                  deformationFieldImage,
                                  NULL,
                                  param->interpolation,
                                  param->paddingValue,
                                  jacobian,
                                  (char)round(param->PSF_Algorithm));
#ifndef NDEBUG
            reg_print_msg_debug("PSF resampling completed\n");
#endif
            free(jacobian);
         }
         else
         {
            reg_resampleImage(floatingImage,
                              warpedImage,
                              deformationFieldImage,
                              NULL,
                              param->interpolation,
                              param->paddingValue);
         }
      }

      memset(warpedImage->descrip, 0, 80);
      strcpy (warpedImage->descrip,"Warped image using NiftyReg (reg_resample)");
      reg_io_WriteImageFile(warpedImage,param->outputResultName);

      if(verbose)
         printf("[NiftyReg] Resampled image has been saved: %s\n", param->outputResultName);
      nifti_image_free(warpedImage);
   }

   /* *********************** */
   /* RESAMPLE A REGULAR GRID */
   /* *********************** */
   if(flag->outputBlankFlag ||
         flag->outputBlankXYFlag ||
         flag->outputBlankYZFlag ||
         flag->outputBlankXZFlag )
   {
      nifti_image *gridImage = nifti_copy_nim_info(floatingImage);
      gridImage->cal_min=0;
      gridImage->cal_max=255;
      gridImage->scl_slope=1.f;
      gridImage->scl_inter=0.f;
      gridImage->dim[0]=gridImage->ndim=floatingImage->nz>1?3:2;
      gridImage->dim[1]=gridImage->nx=floatingImage->nx;
      gridImage->dim[2]=gridImage->ny=floatingImage->ny;
      gridImage->dim[3]=gridImage->nz=floatingImage->nz;
      gridImage->dim[4]=gridImage->nt=1;
      gridImage->dim[5]=gridImage->nu=1;
      gridImage->nvox=(size_t)gridImage->nx*
            gridImage->ny*gridImage->nz;
      gridImage->datatype = NIFTI_TYPE_UINT8;
      gridImage->nbyper = sizeof(unsigned char);
      gridImage->data = (void *)calloc(gridImage->nvox, gridImage->nbyper);
      unsigned char *gridImageValuePtr = static_cast<unsigned char *>(gridImage->data);
      for(int z=0; z<gridImage->nz; z++)
      {
         for(int y=0; y<gridImage->ny; y++)
         {
            for(int x=0; x<gridImage->nx; x++)
            {
               if(referenceImage->nz>1)
               {
                  if(flag->outputBlankXYFlag)
                  {
                     if( x/10==(float)x/10.0 || y/10==(float)y/10.0)
                        *gridImageValuePtr = 255;
                  }
                  else if(flag->outputBlankYZFlag)
                  {
                     if( y/10==(float)y/10.0 || z/10==(float)z/10.0)
                        *gridImageValuePtr = 255;
                  }
                  else if(flag->outputBlankXZFlag)
                  {
                     if( x/10==(float)x/10.0 || z/10==(float)z/10.0)
                        *gridImageValuePtr = 255;
                  }
                  else
                  {
                     if( x/10==(float)x/10.0 || y/10==(float)y/10.0 || z/10==(float)z/10.0)
                        *gridImageValuePtr = 255;
                  }
               }
               else
               {
                  if( x/10==(float)x/10.0 || x==referenceImage->nx-1 || y/10==(float)y/10.0 || y==referenceImage->ny-1)
                     *gridImageValuePtr = 255;
               }
               gridImageValuePtr++;
            }
         }
      }

      nifti_image *warpedImage = nifti_copy_nim_info(referenceImage);
      warpedImage->cal_min=0;
      warpedImage->cal_max=255;
      warpedImage->scl_slope=1.f;
      warpedImage->scl_inter=0.f;
      warpedImage->dim[0]=warpedImage->ndim=referenceImage->nz>1?3:2;
      warpedImage->dim[1]=warpedImage->nx=referenceImage->nx;
      warpedImage->dim[2]=warpedImage->ny=referenceImage->ny;
      warpedImage->dim[3]=warpedImage->nz=referenceImage->nz;
      warpedImage->dim[4]=warpedImage->nt=1;
      warpedImage->dim[5]=warpedImage->nu=1;
      warpedImage->datatype =NIFTI_TYPE_UINT8;
      warpedImage->nbyper = sizeof(unsigned char);
      warpedImage->data = (void *)calloc(warpedImage->nvox,
                                         warpedImage->nbyper);
      reg_resampleImage(gridImage,
                        warpedImage,
                        deformationFieldImage,
                        NULL,
                        1, // linear interpolation
                        0);
      memset(warpedImage->descrip, 0, 80);
      strcpy (warpedImage->descrip,"Warped regular grid using NiftyReg (reg_resample)");
      reg_io_WriteImageFile(warpedImage,param->outputBlankName);
      nifti_image_free(warpedImage);
      nifti_image_free(gridImage);
      if(verbose)
         printf("[NiftyReg] Resampled grid has been saved: %s\n", param->outputBlankName);
   }

   //   // Tell the CLI that we finished
   //   closeProgress("reg_resample", "Normal exit");

   nifti_image_free(referenceImage);
   nifti_image_free(floatingImage);
   nifti_image_free(deformationFieldImage);

   free(flag);
   free(param);
   return EXIT_SUCCESS;
}
