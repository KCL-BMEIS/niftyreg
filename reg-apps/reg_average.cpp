/*
 *  reg_average.cpp
 *
 *
 *  Created by Marc Modat on 29/10/2011.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "reg_average.h"

#include "_reg_ReadWriteImage.h"
#include "_reg_ReadWriteMatrix.h"
#include "_reg_tools.h"
#include "_reg_resampling.h"
#include "_reg_globalTrans.h"
#include "_reg_localTrans.h"
#include "_reg_maths_eigen.h"

#define PrecisionTYPE float

typedef enum
{
   AVG_INPUT,
   AVG_MAT_LTS,
   AVG_IMG_TRANS,
   AVG_IMG_TRANS_NOAFF
} NREG_AVG_TYPE;

void usage(char *exec)
{
   char text[255];
   reg_print_info(exec, "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
   reg_print_info(exec, "usage:");
   sprintf(text, "\t%s <outputFileName> [OPTIONS]", exec);
   reg_print_info(exec, text);
   reg_print_info(exec, "\t-avg <inputAffineName1> <inputAffineName2> ... <inputAffineNameN>");
   reg_print_info(exec, "\t\tIf the input are images, the intensities are averaged");
   reg_print_info(exec, "\t\tIf the input are affine matrices, out=expm((logm(M1)+logm(M2)+...+logm(MN))/N)");
   reg_print_info(exec, "");
   reg_print_info(exec, "\t-avg_lts <AffineMat1> <AffineMat2> ... <AffineMatN> ");
   reg_print_info(exec, "\t\tIt will estimate the robust average affine matrix by considering half of the matrices as ouliers.");
   reg_print_info(exec, "");
   reg_print_info(exec, "\t-avg_tran <referenceImage> <transformationFileName1> <floatingImage1> ... <transformationFileNameN> <floatingImageN> ");
   reg_print_info(exec, "\t\tAll input images are resampled into the space of <reference image> and averaged");
   reg_print_info(exec, "\t\tA cubic spline interpolation scheme is used for resampling");
   reg_print_info(exec, "");
   reg_print_info(exec, "\t-demean <referenceImage> <transformationFileName1> <floatingImage1> ...  <transformationFileNameN> <floatingImageN>");
   reg_print_info(exec, "\t\tThe demean option enforces the mean of all transformations to be");
   reg_print_info(exec, "\t\tidentity.");
   reg_print_info(exec, "\t\tIf affine transformations are provided, only the non-rigid part is");
   reg_print_info(exec, "\t\tconsidered after removing the rigid components.");
   reg_print_info(exec, "\t\tIf non-linear transformation are provided the mean (euclidean) is ");
   reg_print_info(exec, "\t\tremoved from all input transformations.");
   reg_print_info(exec, "\t\tIf velocity field non-linear parametrisations are used, the affine");
   reg_print_info(exec, "\t\tcomponent is discarded and the mean in the log space is removed.");
   reg_print_info(exec, "");
   reg_print_info(exec, "\t-demean_noaff <referenceImage> <AffineMat1> <NonRigidTrans1> <floatingImage1> ...  <AffineMatN> <NonRigidTransN> <floatingImageN>");
   reg_print_info(exec, "\t\tSame as -demean expect that the specified affine is removed from the");
   reg_print_info(exec, "\t\tnon-linear (euclidean) transformation.");
   reg_print_info(exec, "\t--version\t\tPrint current version and exit");
   sprintf(text, "\t\t\t\t(%s)",NR_VERSION);
   reg_print_info(exec, text);
   reg_print_info(exec, "");
   reg_print_info(exec, "alternative usage:");
   sprintf(text, "\t%s --cmd_file <textFile>", exec);
   reg_print_info(exec, text);
   reg_print_info(exec, "\t\tA text file that contains the full command is provided");
   reg_print_info(exec, "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
}

void average_norm_intensity(nifti_image *image)
{
   PrecisionTYPE *rankedIntensities = (PrecisionTYPE *)malloc(image->nvox*sizeof(PrecisionTYPE));
   memcpy(rankedIntensities,image->data,image->nvox*sizeof(PrecisionTYPE));
   reg_heapSort(rankedIntensities,static_cast<int>(image->nvox));
   PrecisionTYPE lowerValue=rankedIntensities[static_cast<unsigned int>(static_cast<float>(image->nvox)*0.03f)];
   PrecisionTYPE higherValue=rankedIntensities[static_cast<unsigned int>(static_cast<float>(image->nvox)*0.97f)];
   reg_tools_substractValueToImage(image,image,lowerValue);
   reg_tools_multiplyValueToImage(image,image,255.f/(higherValue-lowerValue));
   free(rankedIntensities);
   return;
}

int remove_nan_and_add(nifti_image *averageImage,
                        nifti_image *toAddImage,
                        nifti_image *definedNumImage)
{
   if(averageImage->nvox!=toAddImage->nvox || averageImage->nvox!=definedNumImage->nvox)
   {
      reg_print_msg_error(" All images must have the same size");
      return EXIT_FAILURE;
   }
   PrecisionTYPE *avgImgPtr = static_cast<PrecisionTYPE *>(averageImage->data);
   PrecisionTYPE *addImgPtr = static_cast<PrecisionTYPE *>(toAddImage->data);
   PrecisionTYPE *defImgPtr = static_cast<PrecisionTYPE *>(definedNumImage->data);
   for(size_t i=0; i<averageImage->nvox; ++i){
      PrecisionTYPE value = *addImgPtr;
      if(value==value){
         *avgImgPtr+=value;
         *defImgPtr+=1;
      }
      avgImgPtr++;
      addImgPtr++;
      defImgPtr++;
   }
   return EXIT_SUCCESS;
}

mat44 compute_average_matrices(size_t matrixNumber,
                               char **inputAffName,
                               float lts_inlier=1.f)
{
   // Read all input images
   mat44 *matrices=NULL;
   matrices = (mat44 *)malloc(matrixNumber*sizeof(mat44));
   for(size_t m=0; m<matrixNumber; ++m)
      reg_tool_ReadAffineFile(&matrices[m],inputAffName[m]);
   // Matrix to store the final result is created
   mat44 average_matrix;
   // An array to store the weight given to each matrix is generated
   float *matrixWeight = (float *)malloc(matrixNumber*sizeof(float));
   int *matrixIndexSorted = (int *)malloc(matrixNumber*sizeof(int));
   // Input matrices are logged in place
   for(size_t m=0; m<matrixNumber; ++m)
   {
      matrices[m] = reg_mat44_logm(&matrices[m]);
      matrixWeight[m]=1.;
   }
   // The number of iteration to perform is defined based on the use of lts
   size_t iterationNumber = 1;
   if(lts_inlier<1.f && lts_inlier>0)
      iterationNumber=10;
   for(size_t it=0; it<iterationNumber; ++it){
      double tempValue[16]= {0,0,0,0,
                             0,0,0,0,
                             0,0,0,0,
                             0,0,0,0
                            };
      double weightSum=0;
      // The (weighted) average matrix is computed
      for(size_t m=0; m<matrixNumber; ++m)
      {
         tempValue[0]+= (double)matrices[m].m[0][0] * matrixWeight[m];
         tempValue[1]+= (double)matrices[m].m[0][1] * matrixWeight[m];
         tempValue[2]+= (double)matrices[m].m[0][2] * matrixWeight[m];
         tempValue[3]+= (double)matrices[m].m[0][3] * matrixWeight[m];
         tempValue[4]+= (double)matrices[m].m[1][0] * matrixWeight[m];
         tempValue[5]+= (double)matrices[m].m[1][1] * matrixWeight[m];
         tempValue[6]+= (double)matrices[m].m[1][2] * matrixWeight[m];
         tempValue[7]+= (double)matrices[m].m[1][3] * matrixWeight[m];
         tempValue[8]+= (double)matrices[m].m[2][0] * matrixWeight[m];
         tempValue[9]+= (double)matrices[m].m[2][1] * matrixWeight[m];
         tempValue[10]+=(double)matrices[m].m[2][2] * matrixWeight[m];
         tempValue[11]+=(double)matrices[m].m[2][3] * matrixWeight[m];
         tempValue[12]+=(double)matrices[m].m[3][0] * matrixWeight[m];
         tempValue[13]+=(double)matrices[m].m[3][1] * matrixWeight[m];
         tempValue[14]+=(double)matrices[m].m[3][2] * matrixWeight[m];
         tempValue[15]+=(double)matrices[m].m[3][3] * matrixWeight[m];
         weightSum += matrixWeight[m];
      }
      tempValue[0] /= weightSum;
      tempValue[1] /= weightSum;
      tempValue[2] /= weightSum;
      tempValue[3] /= weightSum;
      tempValue[4] /= weightSum;
      tempValue[5] /= weightSum;
      tempValue[6] /= weightSum;
      tempValue[7] /= weightSum;
      tempValue[8] /= weightSum;
      tempValue[9] /= weightSum;
      tempValue[10]/= weightSum;
      tempValue[11]/= weightSum;
      tempValue[12]/= weightSum;
      tempValue[13]/= weightSum;
      tempValue[14]/= weightSum;
      tempValue[15]/= weightSum;
      // The average matrix is converted into a mat44
      average_matrix.m[0][0]=(float)tempValue[0];
      average_matrix.m[0][1]=(float)tempValue[1];
      average_matrix.m[0][2]=(float)tempValue[2];
      average_matrix.m[0][3]=(float)tempValue[3];
      average_matrix.m[1][0]=(float)tempValue[4];
      average_matrix.m[1][1]=(float)tempValue[5];
      average_matrix.m[1][2]=(float)tempValue[6];
      average_matrix.m[1][3]=(float)tempValue[7];
      average_matrix.m[2][0]=(float)tempValue[8];
      average_matrix.m[2][1]=(float)tempValue[9];
      average_matrix.m[2][2]=(float)tempValue[10];
      average_matrix.m[2][3]=(float)tempValue[11];
      average_matrix.m[3][0]=(float)tempValue[12];
      average_matrix.m[3][1]=(float)tempValue[13];
      average_matrix.m[3][2]=(float)tempValue[14];
      average_matrix.m[3][3]=(float)tempValue[15];

      // The distance between the average and input matrices are computed
      if(lts_inlier<1.f && lts_inlier>0){
         for(size_t m=0; m<matrixNumber; ++m)
         {
            mat44 Minus=matrices[m] - average_matrix;
            mat44 Minus_transpose;
            for(int i=0; i<4; ++i)
               for(int j=0; j<4; ++j)
                  Minus_transpose.m[i][j] = Minus.m[j][i];
            mat44 MTM=Minus_transpose * Minus;
            double trace=0;
            for(size_t i=0; i<4; ++i)
               trace+=MTM.m[i][i];
            if(trace<std::numeric_limits<double>::epsilon())
               trace = std::numeric_limits<double>::epsilon();
            matrixWeight[m]=1.f/(sqrt(trace));
            matrixIndexSorted[m]=m;
         }
         // Sort the computed distances
         reg_heapSort(matrixWeight, matrixIndexSorted, matrixNumber);
         // Re-assign the weights for the next iteration
         memset(matrixWeight, 0, matrixNumber*sizeof(float));
         for(size_t m=matrixNumber-1; m>lts_inlier * matrixNumber; --m)
         {
            matrixWeight[matrixIndexSorted[m]]=1.f;
         }
      }
      // The average matrix is exponentiated
      average_matrix = reg_mat44_expm(&average_matrix);
   } // iteration number
   // Free the allocated array
   free(matrixWeight);
   free(matrixIndexSorted);
   if(matrices!=NULL) free(matrices);
   return average_matrix;
}

mat44 compute_affine_demean(size_t matrixNumber,
                            char **inputAffName)
{
   mat44 demeanMatrix, tempMatrix;
   memset(&demeanMatrix,0,sizeof(mat44));
   for(size_t m=0; m<matrixNumber; ++m)
   {
      // Read the current matrix
      mat44 current_affine;
      reg_tool_ReadAffineFile(&current_affine,inputAffName[m]);
      // extract the rigid matrix from the affine
      float qb,qc,qd,qx,qy,qz,qfac;
      nifti_mat44_to_quatern(current_affine,&qb,&qc,&qd,&qx,&qy,&qz,NULL,NULL,NULL,&qfac);
      tempMatrix=nifti_quatern_to_mat44(qb,qc,qd,qx,qy,qz,1.f,1.f,1.f,qfac);
      // remove the rigid componenent from the affine matrix
      tempMatrix=nifti_mat44_inverse(tempMatrix);
      tempMatrix=reg_mat44_mul(&tempMatrix,&current_affine);
      // sum up all the affine matrices
      tempMatrix = reg_mat44_logm(&tempMatrix);
      demeanMatrix = demeanMatrix + tempMatrix;
   }
   // The average matrix is normalised
   demeanMatrix = reg_mat44_mul(&demeanMatrix,1.f/(float)matrixNumber);
   // The average matrix is exponentiated
   demeanMatrix = reg_mat44_expm(&demeanMatrix);
   // The average matrix is inverted
   demeanMatrix = nifti_mat44_inverse(demeanMatrix);
   return demeanMatrix;
}

int compute_nrr_demean(nifti_image *demean_field,
                       size_t transformationNumber,
                       char **inputNRRName,
                       char **inputAffName=NULL)
{
   // Set the demean field to zero
   reg_tools_multiplyValueToImage(demean_field,demean_field,0.f);
   // iterate over all transformations
   for(size_t t=0; t<transformationNumber; ++t){
      // read the transformation
      nifti_image *transformation = reg_io_ReadImageFile(inputNRRName[t]);
      // Generate the deformation or flow field
      nifti_image *deformationField = nifti_copy_nim_info(demean_field);
      deformationField->data = (void *)calloc(deformationField->nvox,deformationField->nbyper);
      reg_tools_multiplyValueToImage(deformationField,deformationField,0.f);
      deformationField->scl_slope=1.f;
      deformationField->scl_inter=0.f;
      deformationField->intent_p1=DISP_FIELD;
      reg_getDeformationFromDisplacement(deformationField);
      // Generate a deformation field or a flow field depending of the input transformation
      switch(static_cast<int>(transformation->intent_p1))
      {
      case DISP_FIELD:
         reg_getDeformationFromDisplacement(transformation);
      case DEF_FIELD:
         reg_defField_compose(transformation,deformationField,NULL);
         break;
      case CUB_SPLINE_GRID:
         reg_spline_getDeformationField(transformation,deformationField,NULL,true,true);
         break;
      case DISP_VEL_FIELD:
         reg_getDeformationFromDisplacement(transformation);
      case DEF_VEL_FIELD:
         reg_defField_compose(transformation,deformationField,NULL);
         break;
      case SPLINE_VEL_GRID:
         reg_spline_getFlowFieldFromVelocityGrid(transformation,deformationField);
         break;
      default:
         reg_print_msg_error("Unsupported transformation parametrisation type:");
         reg_print_msg_error(transformation->fname);
         return EXIT_FAILURE;
      }
      // The affine component is removed
      if(inputAffName!=NULL || transformation->num_ext>0){
         mat44 affineTransformation;
         if(transformation->num_ext>0)
         {
            affineTransformation=*reinterpret_cast<mat44 *>(transformation->ext_list[0].edata);
            // Note that if the transformation is a flow field, only half-of the affine has be used
            if(transformation->num_ext>1 && deformationField->intent_p1!=DEF_VEL_FIELD)
            {
               affineTransformation=reg_mat44_mul(
                                       reinterpret_cast<mat44 *>(transformation->ext_list[1].edata),
                                       &affineTransformation);
            }
         }
         else reg_tool_ReadAffineFile(&affineTransformation,inputAffName[t]);
         // The affine component is substracted
         nifti_image *tempField = nifti_copy_nim_info(deformationField);
         tempField->data = (void *)malloc(tempField->nvox*tempField->nbyper);
         tempField->scl_slope=1.f;
         tempField->scl_inter=0.f;
         reg_affine_getDeformationField(&affineTransformation, tempField);
         reg_tools_substractImageToImage(deformationField,tempField,deformationField);
         nifti_image_free(tempField);
         if(deformationField->intent_p1==DEF_FIELD)
            deformationField->intent_p1=DISP_FIELD;
         if(deformationField->intent_p1==DEF_VEL_FIELD)
            deformationField->intent_p1=DISP_VEL_FIELD;
      }
      else reg_getDisplacementFromDeformation(deformationField);
      nifti_image_free(transformation);
      // The current field is added to the average image
      reg_tools_addImageToImage(demean_field,deformationField,demean_field);
      nifti_image_free(deformationField);
   } // iteration over transformation: t
   // The average image is normalised by the number of inputs
   reg_tools_divideValueToImage(demean_field,demean_field,transformationNumber);

   return EXIT_SUCCESS;
}

int compute_average_image(nifti_image *averageImage,
                          size_t imageNumber,
                          char **inputImageName,
                          char **inputAffName=NULL,
                          char **inputNRRName=NULL,
                          bool demean=false)
{
   // Compute the matrix required for demeaning if required
   mat44 demeanMatrix;
   nifti_image *demeanField = NULL;
   if(demean && inputAffName!=NULL && inputNRRName==NULL){
      demeanMatrix = compute_affine_demean(imageNumber, inputAffName);
#ifndef NDEBUG
      reg_print_msg_debug("Matrix to use for demeaning computed");
#endif
   }
   if(demean && inputNRRName!=NULL){
      demeanField=nifti_copy_nim_info(averageImage);
      demeanField->ndim=demeanField->dim[0]=5;
      demeanField->nt=demeanField->dim[4]=1;
      demeanField->nu=demeanField->dim[5]=demeanField->nz>1?3:2;
      demeanField->nvox=(size_t)demeanField->nx *
            demeanField->ny * demeanField->nz *
            demeanField->nt * demeanField->nu;
      demeanField->nbyper=sizeof(float);
      demeanField->datatype=NIFTI_TYPE_FLOAT32;
      demeanField->intent_code=NIFTI_INTENT_VECTOR;
      memset(demeanField->intent_name, 0, 16);
      strcpy(demeanField->intent_name,"NREG_TRANS");
      demeanField->scl_slope=1.f;
      demeanField->scl_inter=0.f;
      demeanField->intent_p1=DISP_FIELD;
      demeanField->data=(void *)calloc(demeanField->nvox, demeanField->nbyper);
      compute_nrr_demean(demeanField, imageNumber, inputNRRName, inputAffName);
#ifndef NDEBUG
      reg_print_msg_debug("Displacement field to use for demeaning computed");
#endif
   }

   // Set the average image to zero
   memset(averageImage->data, 0, averageImage->nvox*averageImage->nbyper);
   // Create an image to store the defined value number
   nifti_image *definedValue = nifti_copy_nim_info(averageImage);
   definedValue->data = (void *)calloc(averageImage->nvox, averageImage->nbyper);
   // Loop over all input images
   for(size_t i=0; i<imageNumber; ++i){
      // Generate a deformation field defined by the average final
      nifti_image *deformationField=nifti_copy_nim_info(averageImage);
      deformationField->ndim=deformationField->dim[0]=5;
      deformationField->nt=deformationField->dim[4]=1;
      deformationField->nu=deformationField->dim[5]=deformationField->nz>1?3:2;
      deformationField->nvox=(size_t)deformationField->nx *
            deformationField->ny * deformationField->nz *
            deformationField->nt * deformationField->nu;
      deformationField->nbyper=sizeof(float);
      deformationField->datatype=NIFTI_TYPE_FLOAT32;
      deformationField->intent_code=NIFTI_INTENT_VECTOR;
      memset(deformationField->intent_name, 0, 16);
      strcpy(deformationField->intent_name,"NREG_TRANS");
      deformationField->scl_slope=1.f;
      deformationField->scl_inter=0.f;
      deformationField->intent_p1=DISP_FIELD;
      deformationField->data=(void *)calloc(deformationField->nvox, deformationField->nbyper);
      reg_tools_multiplyValueToImage(deformationField,deformationField,0.f);
      // Set the transformation to identity
      reg_getDeformationFromDisplacement(deformationField);
      // Compute the transformation if required
      if(inputNRRName!=NULL){
         nifti_image *current_transformation = reg_io_ReadImageFile(inputNRRName[i]);
         switch(static_cast<int>(current_transformation->intent_p1)){
         case DISP_FIELD:
            reg_getDeformationFromDisplacement(current_transformation);
         case DEF_FIELD:
            reg_defField_compose(current_transformation, deformationField, NULL);
            break;
         case CUB_SPLINE_GRID:
            reg_spline_getDeformationField(current_transformation, deformationField, NULL, true, true);
            break;
         case SPLINE_VEL_GRID:
            if(current_transformation->num_ext>0)
               nifti_copy_extensions(deformationField,current_transformation);
            reg_spline_getFlowFieldFromVelocityGrid(current_transformation, deformationField);
            break;
         case DISP_VEL_FIELD:
            reg_getDeformationFromDisplacement(current_transformation);
         case DEF_VEL_FIELD:
            reg_defField_compose(current_transformation,deformationField,NULL);
            break;
         default: reg_print_msg_error("Unsupported transformation type")
                  reg_exit();
         }
         nifti_image_free(current_transformation);
         if(demeanField!=NULL){
            if(deformationField->intent_p1==DEF_VEL_FIELD){
               reg_tools_substractImageToImage(deformationField,demeanField,deformationField);
               nifti_image *tempDef = nifti_copy_nim_info(deformationField);
               tempDef->data = (void *)malloc(tempDef->nvox*tempDef->nbyper);
               memcpy(tempDef->data,deformationField->data,tempDef->nvox*tempDef->nbyper);
               tempDef->scl_slope=1.f;
               tempDef->scl_inter=0.f;
               reg_defField_getDeformationFieldFromFlowField(tempDef,deformationField,false);
               deformationField->intent_p1=DEF_FIELD;
               nifti_free_extensions(deformationField);
               nifti_image_free(tempDef);
            }
            else reg_tools_substractImageToImage(deformationField,demeanField,deformationField);
#ifndef NDEBUG
            reg_print_msg_debug("Input non-linear transformation has been demeaned");
#endif
         }
      }
      else if(inputAffName!=NULL){
         mat44 current_affine;
         reg_tool_ReadAffineFile(&current_affine,inputAffName[i]);
         if(demean && inputAffName!=NULL && inputNRRName==NULL){
            current_affine = demeanMatrix * current_affine;
#ifndef NDEBUG
      reg_print_msg_debug("Input affine transformation has been demeaned");
#endif
         }
         reg_affine_getDeformationField(&current_affine, deformationField);
      }
      // Create a warped image file
      nifti_image *warpedImage = nifti_copy_nim_info(averageImage);
      warpedImage->datatype = NIFTI_TYPE_FLOAT32;
      warpedImage->nbyper = sizeof(float);
      warpedImage->data = (void *)malloc(warpedImage->nvox*warpedImage->nbyper);
      // Read the input image
      nifti_image *current_input_image = reg_io_ReadImageFile(inputImageName[i]);
      reg_tools_changeDatatype<PrecisionTYPE>(current_input_image);
      // Apply the transformation
      reg_resampleImage(current_input_image, warpedImage, deformationField, NULL, 3, std::numeric_limits<float>::quiet_NaN());
      nifti_image_free(deformationField);
      nifti_image_free(current_input_image);
      // Add the image to the average
      remove_nan_and_add(averageImage, warpedImage, definedValue);
      nifti_image_free(warpedImage);
   }
   // Clear the allocated demeanField if needed
   if(demeanField!=NULL) nifti_image_free(demeanField);
   // Normalised the average image
   reg_tools_divideImageToImage(averageImage,definedValue, averageImage);
   nifti_image_free(definedValue);
   return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{
   // Check that the number of argument is sufficient
   if(argc<2)
   {
      usage(argv[0]);
      return EXIT_FAILURE;
   }
#if defined (_OPENMP)
   // Set the default number of thread
   int defaultOpenMPValue=omp_get_num_procs();
   if(getenv("OMP_NUM_THREADS")!=NULL)
      defaultOpenMPValue=atoi(getenv("OMP_NUM_THREADS"));
   omp_set_num_threads(defaultOpenMPValue);
#endif

   // Check if help is required
   for(int i=1; i<argc; ++i)
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
         usage(argv[0]);
         return EXIT_SUCCESS;
      }
      // Check if the --xml information is required
      else if(strcmp(argv[i], "--xml")==0)
      {
         printf("%s",xml_average);
         return EXIT_SUCCESS;
      }
      else if(strcmp(argv[i], "-version")==0 || strcmp(argv[i], "-Version")==0 ||
            strcmp(argv[i], "-V")==0 || strcmp(argv[i], "-v")==0 ||
            strcmp(argv[i], "--v")==0 || strcmp(argv[i], "--version")==0)
      {
         printf("%s\n",NR_VERSION);
         return EXIT_SUCCESS;
      }
   }

   // Check if a command text file is provided
   char **pointer_to_command = NULL;
   int arg_num_command = 0;
   if(strcmp(argv[1],"--cmd_file")==0 && argc==3){
      char buffer[512];
      FILE *cmd_file = fopen(argv[2], "r+");
      if(cmd_file==NULL){
         reg_print_msg_error("Error when reading the provided command line file:");
         reg_print_msg_error(argv[2]);
         reg_exit();
      }
      // First path to extract the actual argument number
      while(fscanf(cmd_file," %511s", buffer)==1)
         ++arg_num_command;
      // Allocate the required size
      pointer_to_command = (char **)malloc(arg_num_command*sizeof(char *));
      // Store the arguments
      fseek(cmd_file, 0, SEEK_SET);
      arg_num_command=0;
      while(fscanf(cmd_file," %511s", buffer)==1){
         int length = strchr(buffer, '\0')-buffer+1;
         if(strcmp(buffer, "-omp")==0){
            fscanf(cmd_file," %511s", buffer);
#if defined (_OPENMP)
            omp_set_num_threads(atoi(buffer));
#else
            reg_print_msg_warn("OpenMP flag detected and ignored.");
#endif
#ifndef NDEBUG
            reg_print_msg_debug("OpenMP flag detected");
#if defined (_OPENMP)
            reg_print_msg_debug("OpenMP core number set to:");
            reg_print_msg_debug(buffer);
#endif
#endif
         }
         else{
            pointer_to_command[arg_num_command] = (char *)malloc(length*sizeof(char));
            strcpy(pointer_to_command[arg_num_command], buffer);
            ++arg_num_command;
         }
      }
      fclose(cmd_file);
   }
   else{
      pointer_to_command = argv;
      arg_num_command = argc;
   }

#ifndef NDEBUG
   reg_print_msg_debug("command");
   for(int i=0;i<arg_num_command;++i){
      printf("%s ", pointer_to_command[i]);
   }
   printf("\n");
#endif


   // Set some variables
   int operation;
   bool use_demean=false;
   size_t image_number=0;
   char *referenceImageName=NULL;

   // Set the name of the file to output
   char *outputName = pointer_to_command[1];


   // Check what operation is required
   if(strcmp(pointer_to_command[2],"-avg")==0){
      operation=AVG_INPUT;
      image_number=arg_num_command-3;
   }
   else if(strcmp(pointer_to_command[2],"-avg_lts")==0 || strcmp(pointer_to_command[2],"-lts_avg")==0){
      operation=AVG_MAT_LTS;
      image_number=arg_num_command-3;
   }
   else if(strcmp(pointer_to_command[2],"-avg_tran")==0){
      referenceImageName=pointer_to_command[3];
      operation=AVG_IMG_TRANS;
      image_number=(arg_num_command-4)/2;
   }
   else if(strcmp(pointer_to_command[2],"-demean")==0 || strcmp(pointer_to_command[2],"-demean1")==0 || strcmp(pointer_to_command[2],"-demean2")==0){
      referenceImageName=pointer_to_command[3];
      operation=AVG_IMG_TRANS;
      image_number=(arg_num_command-4)/2;
      use_demean=true;
   }
   else if(strcmp(pointer_to_command[2],"-demean_noaff")==0 || strcmp(pointer_to_command[2],"-demean3")==0){
      referenceImageName=pointer_to_command[3];
      operation=AVG_IMG_TRANS_NOAFF;
      image_number=(arg_num_command-4)/3;
      use_demean=true;
   }
   else
   {
      reg_print_msg_error("unknow operation. Options are \"-avg\", \"-avg_lts\", \"-avg_tran\", ");
      reg_print_msg_error("\"-demean\" or \"-demean_noaff\". Specified argument:");
      reg_print_msg_error(pointer_to_command[2]);
      usage(pointer_to_command[0]);
      return EXIT_FAILURE;
   }

   // Check if the inputs are affine or images
   bool trans_is_affine=true;
   if(operation==AVG_INPUT || operation==AVG_IMG_TRANS){
      std::string n(pointer_to_command[4]);
      if(     n.find( ".nii") != std::string::npos ||
              n.find( ".nii.gz") != std::string::npos ||
              n.find( ".hdr") != std::string::npos ||
              n.find( ".img") != std::string::npos ||
              n.find( ".img.gz") != std::string::npos ||
              n.find( ".png") != std::string::npos ||
              n.find( ".nrrd") != std::string::npos)
      {
         trans_is_affine=false;
      }
   }

   // Parse the input data
   char **input_image_names = NULL;
   char **input_affine_names = NULL;
   char **input_nonrigid_names = NULL;
   if(operation!=AVG_INPUT || trans_is_affine==false){
      input_image_names = (char **)malloc(image_number*sizeof(char *));
   }
   if((operation==AVG_INPUT && trans_is_affine==true) || trans_is_affine || operation==AVG_IMG_TRANS_NOAFF){
      input_affine_names = (char **)malloc(image_number*sizeof(char *));
   }
   if((operation==AVG_IMG_TRANS && trans_is_affine==false) || operation==AVG_IMG_TRANS_NOAFF){
      input_nonrigid_names = (char **)malloc(image_number*sizeof(char *));
   }
   int start=3;
   int increment=1;
   if(operation==AVG_IMG_TRANS){
      start=4;
      increment=2;
   }
   else if(operation==AVG_IMG_TRANS_NOAFF){
      start=4;
      increment=3;
   }
   int index=0;
   for(int i=start; i<arg_num_command; i+=increment){
      if(operation==AVG_INPUT){
         if(trans_is_affine)
            input_affine_names[index] = pointer_to_command[i];
         else input_image_names[index] = pointer_to_command[i];
      }
      if(operation==AVG_MAT_LTS){
         input_affine_names[index] = pointer_to_command[i];
      }
      if(operation==AVG_IMG_TRANS){
         input_image_names[index] = pointer_to_command[i+1];
         if(trans_is_affine)
            input_affine_names[index] = pointer_to_command[i];
         else input_nonrigid_names[index] = pointer_to_command[i];
      }
      if(operation==AVG_IMG_TRANS_NOAFF){
         input_affine_names[index] = pointer_to_command[i];
         input_nonrigid_names[index] = pointer_to_command[i+1];
         input_image_names[index] = pointer_to_command[i+2];
      }
      ++index;
   }

   mat44 avg_output_matrix;
   nifti_image *avg_output_image=NULL;

   // Go over the different operations
   if(operation==AVG_INPUT && trans_is_affine==true){
      // compute the average matrix from the input provided
      avg_output_matrix = compute_average_matrices(image_number, input_affine_names);
   }
   else if(operation==AVG_MAT_LTS){
      // compute the average LTS matrix from the input provided
      avg_output_matrix = compute_average_matrices(image_number, input_affine_names, 0.5f);
   }
   else{
      // Allocate the average warped image
      if(referenceImageName==NULL)
         referenceImageName=input_image_names[0];
      avg_output_image = reg_io_ReadImageFile(referenceImageName);
      // clean the data and reallocate them
      free(avg_output_image->data);
      avg_output_image->scl_slope=1.f;
      avg_output_image->scl_inter=0.f;
      avg_output_image->datatype=NIFTI_TYPE_FLOAT32;
      if(sizeof(PrecisionTYPE)==sizeof(double))
         avg_output_image->datatype=NIFTI_TYPE_FLOAT64;
      avg_output_image->nbyper=sizeof(PrecisionTYPE);
      avg_output_image->data=(void *)calloc(avg_output_image->nvox,avg_output_image->nbyper);
      reg_tools_multiplyValueToImage(avg_output_image, avg_output_image, 0.f);
      // Set the output filename
      nifti_set_filenames(avg_output_image, outputName, 0, 0);
      // Compute the average image
      compute_average_image(avg_output_image,
                            image_number,
                            input_image_names,
                            input_affine_names,
                            input_nonrigid_names,
                            use_demean);
   }
   // Save the output
   if(avg_output_image==NULL)
      reg_tool_WriteAffineFile(&avg_output_matrix, outputName);
   else reg_io_WriteImageFile(avg_output_image, outputName);

   // Free the allocated array
   if(argc!=arg_num_command){
      for(int i=0; i<arg_num_command; ++i)
         free(pointer_to_command[i]);
      free(pointer_to_command);
   }
   if(avg_output_image!=NULL)
      nifti_image_free(avg_output_image);
   if(input_image_names!=NULL){
      free(input_image_names);
   }
   if(input_affine_names!=NULL){
      free(input_affine_names);
   }
   if(input_nonrigid_names!=NULL){
      free(input_nonrigid_names);
   }

   return EXIT_SUCCESS;
}
