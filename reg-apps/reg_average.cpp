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
#ifndef MM_AVERAGE_CPP
#define MM_AVERAGE_CPP

#include "_reg_ReadWriteImage.h"
#include "_reg_tools.h"
#include "_reg_resampling.h"
#include "_reg_globalTransformation.h"
#include "_reg_localTransformation.h"

#include "reg_average.h"

#ifdef _USE_NR_DOUBLE
#define PrecisionTYPE double
#else
#define PrecisionTYPE float
#endif

void usage(char *exec)
{
   printf("usage:\n\t%s <outputFileName> [OPTIONS]\n\n", exec);
   printf("\t-avg <inputFileName1> <inputFileName2> ... <inputFileNameN> \n");
   printf("\t\tIf the input are images, the intensities are averaged\n");
   printf("\t\tIf the input are affine matrices, out=expm((logm(M1)+logm(M2)+...+logm(MN))/N)\n\n");
   printf("\t-demean1 <referenceImage> <AffineMat1> <floatingImage1> ...  <AffineMatN> <floatingImageN>\n");
   printf("\t-demean2  <referenceImage> <NonRigidTrans1> <floatingImage1> ... <NonRigidTransN> <floatingImageN>\n");
   printf("\t-demean3  <referenceImage> <AffineMat1> <NonRigidTrans1> <floatingImage1> ...  <AffineMatN> <NonRigidTransN> <floatingImageN>\n\n");
   printf("Desciptions:\n\n");
   printf("* The demean1 option enforces the mean of all affine matrices to have\n");
   printf("a Jacobian determinant equal to one. This is done by computing the\n");
   printf("average transformation by considering only the scaling and shearing\n");
   printf("arguments.The inverse of this computed average matrix is then removed\n");
   printf("to all input affine matrix beforeresampling all floating images to the\n");
   printf("user-defined reference space\n\n");
   printf("* The demean2\n\n");
   printf("* The demean3\n\n");
}

template <class DTYPE>
void average_norm_intensity(nifti_image *image)
{
   DTYPE *rankedIntensities = (DTYPE *)malloc(image->nvox*sizeof(DTYPE));
   memcpy(rankedIntensities,image->data,image->nvox*sizeof(DTYPE));
   reg_heapSort(rankedIntensities,static_cast<int>(image->nvox));
   DTYPE lowerValue=rankedIntensities[static_cast<unsigned int>(static_cast<float>(image->nvox)*0.03f)];
   DTYPE higherValue=rankedIntensities[static_cast<unsigned int>(static_cast<float>(image->nvox)*0.97f)];
   reg_tools_substractValueToImage(image,image,lowerValue);
   reg_tools_multiplyValueToImage(image,image,255.f/(higherValue-lowerValue));
   free(rankedIntensities);
   return;
}

int main(int argc, char **argv)
{
   // Check that the number of argument is sufficient
   if(argc<2)
   {
      usage(argv[0]);
      return EXIT_FAILURE;
   }
   // Check if the --xml information is required
   if(strcmp(argv[1], "--xml")==0)
   {
      printf("%s",xml_average);
      return 0;
   }
   // Check if help is required
   for(size_t i=1; i<argc; ++i)
   {
      if(strcmp(argv[i],"-h")==0 ||
            strcmp(argv[i],"-H")==0 ||
            strcmp(argv[i],"-help")==0 ||
            strcmp(argv[i],"-HELP")==0 ||
            strcmp(argv[i],"-Help")==0
        )
      {
         usage(argv[0]);
         return EXIT_SUCCESS;
      }
   }
   // Command line
   printf("\nCommand line:\n\t");
   for(size_t i=0; i<argc; ++i)
      printf("%s ",argv[i]);
   printf("\n\n");

   // Set the name of the file to output
   char *outputName = argv[1];

   // Check what operation is required
   int operation;
   if(strcmp(argv[2],"-avg")==0)
      operation=0;
   else if(strcmp(argv[2],"-demean1")==0)
      operation=1;
   else if(strcmp(argv[2],"-demean2")==0)
      operation=2;
   else if(strcmp(argv[2],"-demean3")==0)
      operation=3;
   else
   {
      reg_print_msg_error("unknow operation. Options are \"-avg\", \"-demean1\", \"-demean2\" or \"-demean3\". Specified argument:");
      reg_print_msg_error(argv[2]);
      usage(argv[0]);
      return EXIT_FAILURE;
   }

   // Create the average image or average matrix
   if(operation==0)
   {
      //Check the name of the first file to verify if they are analyse or nifti image
      std::string n(argv[3]);
      if(     n.find( ".nii.gz") != std::string::npos ||
              n.find( ".nii") != std::string::npos ||
              n.find( ".hdr") != std::string::npos ||
              n.find( ".img") != std::string::npos ||
              n.find( ".img.gz") != std::string::npos)
      {
         // Input arguments are image filename
         // Read the first image to average
         nifti_image *tempImage=reg_io_ReadImageHeader(argv[3]);
         if(tempImage==NULL)
         {
            reg_print_msg_error("The following image can not be read:\n");
            reg_print_msg_error(argv[3]);
            return EXIT_FAILURE;
         }
         reg_checkAndCorrectDimension(tempImage);

         // Create the average image
         nifti_image *average_image=nifti_copy_nim_info(tempImage);
         nifti_image_free(tempImage);
         tempImage=NULL;
         average_image->datatype=NIFTI_TYPE_FLOAT32;
         if(sizeof(PrecisionTYPE)==sizeof(double))
            average_image->datatype=NIFTI_TYPE_FLOAT64;
         average_image->nbyper=sizeof(PrecisionTYPE);
         average_image->data=(void *)calloc(average_image->nvox,average_image->nbyper);

         int imageTotalNumber=0;
         for(int i=3; i<argc; ++i)
         {
            nifti_image *tempImage=reg_io_ReadImageFile(argv[i]);
            if(tempImage==NULL)
            {
               reg_print_msg_error("The following image can not be read:\n");
               reg_print_msg_error(argv[i]);
               return EXIT_FAILURE;
            }
            reg_checkAndCorrectDimension(tempImage);
            if(sizeof(PrecisionTYPE)==sizeof(double))
               reg_tools_changeDatatype<double>(tempImage);
            else reg_tools_changeDatatype<float>(tempImage);
            if(average_image->nvox!=tempImage->nvox)
            {
               reg_print_msg_error(" All images must have the same size. Error when processing:\n");
               reg_print_msg_error(argv[i]);
               return EXIT_FAILURE;
            }
            if(sizeof(PrecisionTYPE)==sizeof(double))
               average_norm_intensity<double>(tempImage);
            else average_norm_intensity<float>(tempImage);
            reg_tools_addImageToImage(average_image,tempImage,average_image);
            imageTotalNumber++;
            nifti_image_free(tempImage);
            tempImage=NULL;
         }
         reg_tools_divideValueToImage(average_image,average_image,(float)imageTotalNumber);
         reg_io_WriteImageFile(average_image,outputName);
         nifti_image_free(average_image);
      }
      else
      {
         // input arguments are assumed to be text file name
         // Create an mat44 array to store all input matrices
         const size_t matrixNumber=argc-3;
         mat44 *inputMatrices=(mat44 *)malloc(matrixNumber * sizeof(mat44));
         // Read all the input matrices
         for(size_t m=0; m<matrixNumber; ++m)
         {
            if(FILE *aff=fopen(argv[m+3], "r"))
            {
               fclose(aff);
            }
            else
            {
               reg_print_msg_error("The specified input affine file can not be read\n");
               reg_print_msg_error(argv[m+3]);
               reg_exit(1);
            }
            // Read the current matrix file
            std::ifstream affineFile;
            affineFile.open(argv[m+3]);
            if(affineFile.is_open())
            {
               // Transfer the values into the mat44 array
               int i=0;
               float value1,value2,value3,value4;
               while(!affineFile.eof())
               {
                  affineFile >> value1 >> value2 >> value3 >> value4;
                  inputMatrices[m].m[i][0] = value1;
                  inputMatrices[m].m[i][1] = value2;
                  inputMatrices[m].m[i][2] = value3;
                  inputMatrices[m].m[i][3] = value4;
                  i++;
                  if(i>3) break;
               }
            }
            affineFile.close();
         }
         // All the input matrices are log-ed
         for(size_t m=0; m<matrixNumber; ++m)
         {
            inputMatrices[m] = reg_mat44_logm(&inputMatrices[m]);
         }
         // All the exponentiated matrices are summed up into one matrix
         //temporary double are used to avoid error accumulation
         double tempValue[16]= {0,0,0,0,
                                0,0,0,0,
                                0,0,0,0,
                                0,0,0,0
                               };
         for(size_t m=0; m<matrixNumber; ++m)
         {
            tempValue[0]+= (double)inputMatrices[m].m[0][0];
            tempValue[1]+= (double)inputMatrices[m].m[0][1];
            tempValue[2]+= (double)inputMatrices[m].m[0][2];
            tempValue[3]+= (double)inputMatrices[m].m[0][3];
            tempValue[4]+= (double)inputMatrices[m].m[1][0];
            tempValue[5]+= (double)inputMatrices[m].m[1][1];
            tempValue[6]+= (double)inputMatrices[m].m[1][2];
            tempValue[7]+= (double)inputMatrices[m].m[1][3];
            tempValue[8]+= (double)inputMatrices[m].m[2][0];
            tempValue[9]+= (double)inputMatrices[m].m[2][1];
            tempValue[10]+=(double)inputMatrices[m].m[2][2];
            tempValue[11]+=(double)inputMatrices[m].m[2][3];
            tempValue[12]+=(double)inputMatrices[m].m[3][0];
            tempValue[13]+=(double)inputMatrices[m].m[3][1];
            tempValue[14]+=(double)inputMatrices[m].m[3][2];
            tempValue[15]+=(double)inputMatrices[m].m[3][3];
         }
         // Average matrix is computed
         tempValue[0] /= (double)matrixNumber;
         tempValue[1] /= (double)matrixNumber;
         tempValue[2] /= (double)matrixNumber;
         tempValue[3] /= (double)matrixNumber;
         tempValue[4] /= (double)matrixNumber;
         tempValue[5] /= (double)matrixNumber;
         tempValue[6] /= (double)matrixNumber;
         tempValue[7] /= (double)matrixNumber;
         tempValue[8] /= (double)matrixNumber;
         tempValue[9] /= (double)matrixNumber;
         tempValue[10]/= (double)matrixNumber;
         tempValue[11]/= (double)matrixNumber;
         tempValue[12]/= (double)matrixNumber;
         tempValue[13]/= (double)matrixNumber;
         tempValue[14]/= (double)matrixNumber;
         tempValue[15]/= (double)matrixNumber;
         // The final matrix is exponentiated
         mat44 outputMatrix;
         outputMatrix.m[0][0]=(float)tempValue[0];
         outputMatrix.m[0][1]=(float)tempValue[1];
         outputMatrix.m[0][2]=(float)tempValue[2];
         outputMatrix.m[0][3]=(float)tempValue[3];
         outputMatrix.m[1][0]=(float)tempValue[4];
         outputMatrix.m[1][1]=(float)tempValue[5];
         outputMatrix.m[1][2]=(float)tempValue[6];
         outputMatrix.m[1][3]=(float)tempValue[7];
         outputMatrix.m[2][0]=(float)tempValue[8];
         outputMatrix.m[2][1]=(float)tempValue[9];
         outputMatrix.m[2][2]=(float)tempValue[10];
         outputMatrix.m[2][3]=(float)tempValue[11];
         outputMatrix.m[3][0]=(float)tempValue[12];
         outputMatrix.m[3][1]=(float)tempValue[13];
         outputMatrix.m[3][2]=(float)tempValue[14];
         outputMatrix.m[3][3]=(float)tempValue[15];
         outputMatrix = reg_mat44_expm(&outputMatrix);
         // Free the array containing the input matrices
         free(inputMatrices);
         // The final matrix is saved
         reg_tool_WriteAffineFile(&outputMatrix,outputName);
      }
   }
   else
   {
      /* **** the average image is created after resampling **** */
      // read the reference image
      nifti_image *referenceImage=reg_io_ReadImageFile(argv[3]);
      if(referenceImage==NULL)
      {
         reg_print_msg_error("The reference image cannot be read. Filename:");
         reg_print_msg_error(argv[3]);
         return EXIT_FAILURE;
      }
#ifndef NDEBUG
      reg_print_msg_debug("reg_average: User-specified reference image:");
      reg_print_msg_debug(referenceImage->fname);
#endif
      if(operation==1)
      {
         // Affine parametrisations are provided
         size_t affineNumber = (argc - 4)/2;
         // All affine matrices are read in
         mat44 *affineMatrices = (mat44 *)malloc(affineNumber*sizeof(mat44));
         for(size_t i=4, j=0; i<argc; i+=2,++j)
         {
            if(reg_isAnImageFileName(argv[i]))
            {
               reg_print_msg_error("An affine transformation was expected. Filename:");
               reg_print_msg_error(argv[i]);
               return EXIT_FAILURE;
            }
            reg_tool_ReadAffineFile(&affineMatrices[j],argv[i]);
         }
         // The rigid matrices are removed from all affine matrices
         mat44 tempMatrix, averageMatrix;
         memset(&averageMatrix,0,sizeof(mat44));
         for(size_t i=0; i<affineNumber; ++i)
         {
            float qb,qc,qd,qx,qy,qz,qfac;
            nifti_mat44_to_quatern(affineMatrices[i],&qb,&qc,&qd,&qx,&qy,&qz,NULL,NULL,NULL,&qfac);
            tempMatrix=nifti_quatern_to_mat44(qb,qc,qd,qx,qy,qz,1.f,1.f,1.f,qfac);
            tempMatrix=nifti_mat44_inverse(tempMatrix);
            tempMatrix=reg_mat44_mul(&tempMatrix,&affineMatrices[i]);
            tempMatrix = reg_mat44_logm(&tempMatrix);
            averageMatrix = averageMatrix + tempMatrix;
         }
         // The average matrix is normalised
         averageMatrix = reg_mat44_mul(&averageMatrix,1.f/(float)affineNumber);
         // The average matrix is exponentiated
         averageMatrix = reg_mat44_expm(&averageMatrix);
         // The average matrix is inverted
         averageMatrix = nifti_mat44_inverse(averageMatrix);
         averageMatrix = reg_mat44_logm(&averageMatrix);
         // Demean all the input affine matrices
         float indet=1.f; // HERE
         float outdet=1.f; // HERE
         for(size_t i=0; i<affineNumber; ++i)
         {
            indet *= reg_mat44_det(&affineMatrices[i]);// HERE
            affineMatrices[i] = reg_mat44_logm(&affineMatrices[i]);
            affineMatrices[i] = averageMatrix + affineMatrices[i];
            affineMatrices[i] = reg_mat44_expm(&affineMatrices[i]);
            outdet *= reg_mat44_det(&affineMatrices[i]);// HERE
         }
         printf("Average determinant %g -> %g\n", indet, outdet); // HERE
         // Create a deformation field to be used to resample all the floating images
         nifti_image *deformationField = nifti_copy_nim_info(referenceImage);
         deformationField->dim[0]=deformationField->ndim=5;
         deformationField->dim[4]=deformationField->nt=1;
         deformationField->dim[5]=deformationField->nu=referenceImage->nz>1?3:2;
         deformationField->nvox = (size_t)deformationField->nx *
                                  deformationField->ny * deformationField->nz * deformationField->nu;
         if(deformationField->datatype!=NIFTI_TYPE_FLOAT32 || deformationField->datatype!=NIFTI_TYPE_FLOAT64)
         {
            deformationField->datatype=NIFTI_TYPE_FLOAT32;
            deformationField->nbyper=sizeof(float);
         }
         deformationField->data = (void *)malloc(deformationField->nvox*deformationField->nbyper);
         // Create an average image
         nifti_image *averageImage = nifti_copy_nim_info(referenceImage);
         if(averageImage->datatype!=NIFTI_TYPE_FLOAT32 || averageImage->datatype!=NIFTI_TYPE_FLOAT64)
         {
            averageImage->datatype=NIFTI_TYPE_FLOAT32;
            averageImage->nbyper=sizeof(float);
         }
         averageImage->data = (void *)calloc(averageImage->nvox,averageImage->nbyper);
         // Create a temporary image
         nifti_image *tempImage = nifti_copy_nim_info(averageImage);
         tempImage->data = (void *)malloc(tempImage->nvox*tempImage->nbyper);
         // warp all floating images and sum them up
         for(size_t i=5, j=0; i<argc; i+=2,++j)
         {
            nifti_image *floatingImage = reg_io_ReadImageFile(argv[i]);
            if(floatingImage==NULL)
            {
               reg_print_msg_error("The floating image cannot be read. Filename:");
               reg_print_msg_error(argv[i]);
               return EXIT_FAILURE;
            }
            reg_affine_getDeformationField(&affineMatrices[j],deformationField);
            if(floatingImage->datatype!=tempImage->datatype)
            {
               switch(tempImage->datatype)
               {
               case NIFTI_TYPE_FLOAT32:
                  reg_tools_changeDatatype<float>(floatingImage);
                  break;
               case NIFTI_TYPE_FLOAT64:
                  reg_tools_changeDatatype<double>(floatingImage);
                  break;
               }
            }
            reg_resampleImage(floatingImage,tempImage,deformationField,NULL,3,0.f);
            if(sizeof(PrecisionTYPE)==sizeof(double))
               average_norm_intensity<double>(tempImage);
            else average_norm_intensity<float>(tempImage);
            reg_tools_addImageToImage(averageImage,tempImage,averageImage);
            nifti_image_free(floatingImage);
         }
         // Normalise the intensity by the number of images
         reg_tools_divideValueToImage(averageImage,averageImage,(float)affineNumber);
         // Free the allocated arrays and images
         nifti_image_free(deformationField);
         nifti_image_free(tempImage);
         free(affineMatrices);
         // Save the average image
         reg_io_WriteImageFile(averageImage,outputName);
         // Free the average image
         nifti_image_free(averageImage);
      } // -avg
      else if(operation==2 || operation==3)
      {
         /* **** Create an average image by demeaning the non-rigid transformation **** */
         // First compute an average field to remove from the final field
         nifti_image *averageField = nifti_copy_nim_info(referenceImage);
         averageField->dim[0]=averageField->ndim=5;
         averageField->dim[4]=averageField->nt=1;
         averageField->dim[5]=averageField->nu=averageField->nz>1?3:2;
         averageField->nvox = (size_t)averageField->nx *
                              averageField->ny * averageField->nz * averageField->nu;
         if(averageField->datatype!=NIFTI_TYPE_FLOAT32 || averageField->datatype!=NIFTI_TYPE_FLOAT64)
         {
            averageField->datatype=NIFTI_TYPE_FLOAT32;
            averageField->nbyper=sizeof(float);
         }
         averageField->data = (void *)calloc(averageField->nvox,averageField->nbyper);
         reg_tools_multiplyValueToImage(averageField,averageField,0.f);
         // Iterate over all the transformation parametrisations - Note that I don't store them all to save space
#ifndef NDEBUG
         char msg[256];
         sprintf(msg,"reg_average: Number of input transformations: %i",(argc-4)/operation);
         reg_print_msg_debug(msg);
#endif
         for(size_t i=(operation==2?4:5); i<argc; i+=operation)
         {
            nifti_image *transformation = reg_io_ReadImageFile(argv[i]);
            if(transformation==NULL)
            {
               reg_print_msg_error("The transformation parametrisation cannot be read. Filename:");
               reg_print_msg_error(argv[i]);
               return EXIT_FAILURE;
            }
            if(transformation->ndim!=5)
            {
               reg_print_msg_error("The specified filename does not seem to be a transformation parametrisation. Filename:");
               reg_print_msg_error(transformation->fname);
               return EXIT_FAILURE;
            }
#ifndef NDEBUG
            reg_print_msg_debug("reg_average: Input non-rigid transformation:");
            reg_print_msg_debug(transformation->fname);
#endif
            // Generate the deformation or flow field
            nifti_image *deformationField = nifti_copy_nim_info(averageField);
            deformationField->data = (void *)malloc(deformationField->nvox*deformationField->nbyper);
            reg_tools_multiplyValueToImage(deformationField,deformationField,0.f);
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
            case SPLINE_GRID:
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
               reg_print_msg_error("Unsupported transformation parametrisation type. Filename:");
               reg_print_msg_error(transformation->fname);
               return EXIT_FAILURE;
            }
            // An affine transformation is use to remove the affine component
            if(operation==3 || transformation->num_ext>0)
            {
               mat44 affineTransformation;
               if(transformation->num_ext>0)
               {
                  if(operation==3)
                  {
                     reg_print_msg_warn("The provided non-rigid parametrisation already embbeds an affine transformation");
                     reg_print_msg_warn(transformation->fname);
                  }
                  affineTransformation=*reinterpret_cast<mat44 *>(transformation->ext_list[0].edata);
                  // Note that if the transformation is a flow field, only half-of the affine has be used
                  if(transformation->num_ext>1 &&
                        deformationField->intent_p1!=DEF_VEL_FIELD)
                  {
                     affineTransformation=reg_mat44_mul(
                                             reinterpret_cast<mat44 *>(transformation->ext_list[1].edata),
                                             &affineTransformation);
                  }
               }
               else
               {
                  reg_tool_ReadAffineFile(&affineTransformation,
                                          argv[i-1]);
#ifndef NDEBUG
                  reg_print_msg_debug("reg_average: Input affine transformation. Filename:");
                  reg_print_msg_debug(argv[i-1]);
#endif
               }
               // The affine component is substracted
               nifti_image *tempField = nifti_copy_nim_info(deformationField);
               tempField->data = (void *)malloc(tempField->nvox*tempField->nbyper);
               reg_affine_getDeformationField(&affineTransformation,
                                              tempField);
               reg_tools_substractImageToImage(deformationField,tempField,deformationField);
               nifti_image_free(tempField);
            }
            else reg_getDisplacementFromDeformation(deformationField);
            reg_tools_addImageToImage(averageField,deformationField,averageField);
            nifti_image_free(transformation);
            nifti_image_free(deformationField);
         } // iteration over all transformation parametrisation
         // the average def/flow field is normalised by the number of input image
         reg_tools_divideValueToImage(averageField,averageField,(argc-4)/operation);
         // The new de-mean transformation are computed and the floating image resample
         // Create an image to store average image
         nifti_image *averageImage = nifti_copy_nim_info(referenceImage);
         if(averageImage->datatype!=NIFTI_TYPE_FLOAT32 || averageImage->datatype!=NIFTI_TYPE_FLOAT64)
         {
            averageImage->datatype=NIFTI_TYPE_FLOAT32;
            averageImage->nbyper=sizeof(float);
         }
         averageImage->data = (void *)calloc(averageImage->nvox,averageImage->nbyper);
         // Create a temporary image
         nifti_image *tempImage = nifti_copy_nim_info(averageImage);
         tempImage->data = (void *)malloc(tempImage->nvox*tempImage->nbyper);
         // Iterate over all the transformation parametrisations
         for(size_t i=(operation==2?4:5); i<argc; i+=operation)
         {
            nifti_image *transformation = reg_io_ReadImageFile(argv[i]);
            if(transformation==NULL)
            {
               reg_print_msg_error("The transformation parametrisation cannot be read. Filename:");
               reg_print_msg_error(argv[i]);
               return EXIT_FAILURE;
            }
            if(transformation->ndim!=5)
            {
               reg_print_msg_error("The specified filename does not seem to be a transformation parametrisation. Filename");
               reg_print_msg_error(transformation->fname);
               return EXIT_FAILURE;
            }
#ifndef NDEBUG
            reg_print_msg_debug("reg_average: Demeaning transformation:");
            reg_print_msg_debug(transformation->fname);
#endif
            // Generate the deformation or flow field
            nifti_image *deformationField = nifti_copy_nim_info(averageField);
            deformationField->data = (void *)malloc(deformationField->nvox*deformationField->nbyper);
            reg_tools_multiplyValueToImage(deformationField,deformationField,0.f);
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
            case SPLINE_GRID:
               reg_spline_getDeformationField(transformation,deformationField,NULL,true,true);
               break;
            case DISP_VEL_FIELD:
               reg_getDeformationFromDisplacement(transformation);
            case DEF_VEL_FIELD:
               reg_defField_compose(transformation,deformationField,NULL);
               break;
            case SPLINE_VEL_GRID:
               if(transformation->num_ext>0)
                  nifti_copy_extensions(deformationField,transformation);
               reg_spline_getFlowFieldFromVelocityGrid(transformation,deformationField);
               break;
            default:
               reg_print_msg_error("Unsupported transformation parametrisation type. Filename:");
               reg_print_msg_error(transformation->fname);
               return EXIT_FAILURE;
            }
            // The deformation is de-mean
            if(deformationField->intent_p1==DEF_VEL_FIELD)
            {
               reg_tools_substractImageToImage(deformationField,averageField,deformationField);
               nifti_image *tempDef = nifti_copy_nim_info(deformationField);
               tempDef->data = (void *)malloc(tempDef->nvox*tempDef->nbyper);
               memcpy(tempDef->data,deformationField->data,tempDef->nvox*tempDef->nbyper);
               reg_defField_getDeformationFieldFromFlowField(tempDef,deformationField,false);
               deformationField->intent_p1=DEF_FIELD;
               nifti_free_extensions(deformationField);
               nifti_image_free(tempDef);
            }
            else reg_tools_substractImageToImage(deformationField,averageField,deformationField);
            // The floating image is resampled
            nifti_image *floatingImage=reg_io_ReadImageFile(argv[i+1]);
            if(floatingImage==NULL)
            {
               reg_print_msg_error("The floating image cannot be read. Filename:");
               reg_print_msg_error(argv[i+1]);
               return EXIT_FAILURE;
            }
#ifndef NDEBUG
            reg_print_msg_debug("reg_average: Warping floating image:");
            reg_print_msg_debug(floatingImage->fname);
#endif
            if(floatingImage->datatype!=tempImage->datatype)
            {
               switch(tempImage->datatype)
               {
               case NIFTI_TYPE_FLOAT32:
                  reg_tools_changeDatatype<float>(floatingImage);
                  break;
               case NIFTI_TYPE_FLOAT64:
                  reg_tools_changeDatatype<double>(floatingImage);
                  break;
               }
            }
            reg_resampleImage(floatingImage,tempImage,deformationField,NULL,3,0.f);
            reg_tools_addImageToImage(averageImage,tempImage,averageImage);
            nifti_image_free(floatingImage);
            nifti_image_free(deformationField);
         } // iteration over all transformation parametrisation
         // Normalise the average image by the number of input images
         reg_tools_divideValueToImage(averageImage,averageImage,(argc-4)/operation);
         // Free the allocated field
         nifti_image_free(averageField);
         // Save and free the average image
         reg_io_WriteImageFile(averageImage,outputName);
         nifti_image_free(averageImage);
      } // (operation==2 || operation==3)
   } // (-demean)

   return EXIT_SUCCESS;
}

#endif
