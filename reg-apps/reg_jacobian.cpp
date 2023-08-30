/**
 * @file reg_jacobian.cpp
 * @author Marc Modat
 * @date 15/11/2010
 * @brief Executable use to generate Jacobian matrices and determinant
 * images.
 *
 *  Created by Marc Modat on 15/11/2010.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_ReadWriteImage.h"
#include "_reg_ReadWriteMatrix.h"
#include "_reg_globalTrans.h"
#include "_reg_localTrans_jac.h"
#include "_reg_tools.h"
#include "_reg_resampling.h"
#include "reg_jacobian.h"

typedef struct
{
   char *refImageName;
   char *inputTransName;
   char *outputJacDetName;
   char *outputJacMatName;
   char *outputLogDetName;
} PARAM;
typedef struct
{
   bool refImageFlag;
   bool inputTransFlag;
   bool outputJacDetFlag;
   bool outputJacMatFlag;
   bool outputLogDetFlag;
} FLAG;

template <class DataType>
void reg_jacobian_computeLog(nifti_image *image)
{
   DataType *imgPtr=static_cast<DataType *>(image->data);
   for(size_t i=0; i<image->nvox;++i){
      *imgPtr = static_cast<DataType>(log(*imgPtr));
      ++imgPtr;
   }
   return;
}

template <class DataType>
void reg_jacobian_convertMat33ToNii(mat33 *array, nifti_image *image)
{
   const size_t voxelNumber=NiftiImage::calcVoxelNumber(image, 3);
   DataType *ptrXX=static_cast<DataType *>(image->data);
   if(image->nz>1)
   {
      DataType *ptrXY=&ptrXX[voxelNumber];
      DataType *ptrXZ=&ptrXY[voxelNumber];
      DataType *ptrYX=&ptrXZ[voxelNumber];
      DataType *ptrYY=&ptrYX[voxelNumber];
      DataType *ptrYZ=&ptrYY[voxelNumber];
      DataType *ptrZX=&ptrYZ[voxelNumber];
      DataType *ptrZY=&ptrZX[voxelNumber];
      DataType *ptrZZ=&ptrZY[voxelNumber];
      for(size_t voxel=0; voxel<voxelNumber; ++voxel)
      {
         mat33 matrix=array[voxel];
         ptrXX[voxel]=matrix.m[0][0];
         ptrXY[voxel]=matrix.m[0][1];
         ptrXZ[voxel]=matrix.m[0][2];
         ptrYX[voxel]=matrix.m[1][0];
         ptrYY[voxel]=matrix.m[1][1];
         ptrYZ[voxel]=matrix.m[1][2];
         ptrZX[voxel]=matrix.m[2][0];
         ptrZY[voxel]=matrix.m[2][1];
         ptrZZ[voxel]=matrix.m[2][2];
      }
   }
   else
   {
      DataType *ptrXY=&ptrXX[voxelNumber];
      DataType *ptrYX=&ptrXY[voxelNumber];
      DataType *ptrYY=&ptrYX[voxelNumber];
      for(size_t voxel=0; voxel<voxelNumber; ++voxel)
      {
         mat33 matrix=array[voxel];
         ptrXX[voxel]=matrix.m[0][0];
         ptrXY[voxel]=matrix.m[0][1];
         ptrYX[voxel]=matrix.m[1][0];
         ptrYY[voxel]=matrix.m[1][1];
      }

   }
}

void PetitUsage(char *exec)
{
   NR_INFO("Usage:\t" << exec << " -ref <referenceImage> [OPTIONS]");
   NR_INFO("\tSee the help for more details (-h)");
}

void Usage(char *exec)
{
   NR_INFO("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
   NR_INFO("Usage:\t" << exec << " [OPTIONS]");
   NR_INFO("* * INPUT * *");
   NR_INFO("\t-trans <filename>");
   NR_INFO("\t\tFilename of the file containing the transformation (mandatory)");
   NR_INFO("\t-ref <filename>");
   NR_INFO("\t\tFilename of the reference image (required if the transformation is a spline parametrisation)");
   NR_INFO("\n* * OUTPUT * *");
   NR_INFO("\t-jac <filename>");
   NR_INFO("\t\tFilename of the Jacobian determinant map");
   NR_INFO("\t-jacM <filename>");
   NR_INFO("\t\tFilename of the Jacobian matrix map. (9 or 4 values are stored as a 5D nifti)");
   NR_INFO("\t-jacL <filename>");
   NR_INFO("\t\tFilename of the Log of the Jacobian determinant map");
#ifdef _OPENMP
   int defaultOpenMPValue=omp_get_num_procs();
   if(getenv("OMP_NUM_THREADS")!=nullptr)
      defaultOpenMPValue=atoi(getenv("OMP_NUM_THREADS"));
   NR_INFO("\t-omp <int>\n\t\tNumber of threads to use with OpenMP. [" << defaultOpenMPValue << "/" << omp_get_num_procs() << "]");
#endif
   NR_INFO("\t--version\n\t\tPrint current version and exit (" << NR_VERSION << ")");
   NR_INFO("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
}

int main(int argc, char **argv)
{
   if(argc==1){
      PetitUsage(argv[0]);
      return EXIT_FAILURE;
   }

   PARAM *param = (PARAM *)calloc(1,sizeof(PARAM));
   FLAG *flag = (FLAG *)calloc(1,sizeof(FLAG));

#ifdef _OPENMP
   // Set the default number of threads
   int defaultOpenMPValue=omp_get_num_procs();
   if(getenv("OMP_NUM_THREADS")!=nullptr)
      defaultOpenMPValue=atoi(getenv("OMP_NUM_THREADS"));
   omp_set_num_threads(defaultOpenMPValue);
#endif

   // read the input parameters
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
         NR_COUT << xml_jacobian << std::endl;
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
      else if(strcmp(argv[i], "-trans") == 0 ||
              (strcmp(argv[i],"--trans")==0))
      {
         param->inputTransName=argv[++i];
         flag->inputTransFlag=1;
      }
      else if(strcmp(argv[i], "-jac") == 0 ||
              (strcmp(argv[i],"--jac")==0))
      {
         param->outputJacDetName=argv[++i];
         flag->outputJacDetFlag=1;
      }
      else if(strcmp(argv[i], "-jacM") == 0 ||
              (strcmp(argv[i],"--jacM")==0))
      {
         param->outputJacMatName=argv[++i];
         flag->outputJacMatFlag=1;
      }
      else if(strcmp(argv[i], "-jacL") == 0 ||
              (strcmp(argv[i],"--jacL")==0))
      {
         param->outputLogDetName=argv[++i];
         flag->outputLogDetFlag=1;
      }
      else
      {
         NR_ERROR("Parameter unknown: " << argv[i]);
         PetitUsage(argv[0]);
         return EXIT_FAILURE;
      }
   }

   /* ******************* */
   /* READ TRANSFORMATION */
   /* ******************* */
   nifti_image *inputTransformation=nullptr;
   if(flag->inputTransFlag)
   {
      // Check of the input transformation is an affine
      if(!reg_isAnImageFileName(param->inputTransName)){
         mat44 *affineTransformation=(mat44 *)malloc(sizeof(mat44));
         reg_tool_ReadAffineFile(affineTransformation,param->inputTransName);
         NR_COUT << reg_mat44_det<double>(affineTransformation) << std::endl;
         return EXIT_SUCCESS;
      }

      inputTransformation = reg_io_ReadImageFile(param->inputTransName);
      if(inputTransformation == nullptr)
      {
         NR_ERROR("Error when reading the transformation image: " << param->inputTransName);
         return EXIT_FAILURE;
      }
   }
   else
   {
      NR_ERROR("No transformation has been provided");
      return EXIT_FAILURE;
   }

   /* *************************** */
   /* COMPUTE JACOBIAN MAT OR DET */
   /* *************************** */
   // Create a deformation field if needed
   nifti_image *referenceImage=nullptr;
   if(inputTransformation->intent_p1==LIN_SPLINE_GRID ||
         inputTransformation->intent_p1==CUB_SPLINE_GRID ||
         inputTransformation->intent_p1==SPLINE_VEL_GRID){
      if(!flag->refImageFlag){
         NR_ERROR("A reference image has to be specified with a spline parametrisation.");
         return EXIT_FAILURE;
      }
      // Read the reference image
      referenceImage = reg_io_ReadImageHeader(param->refImageName);
      if(referenceImage == nullptr)
      {
         NR_ERROR("Error when reading the reference image.");
         return EXIT_FAILURE;
      }
   }

   if(flag->outputJacDetFlag || flag->outputLogDetFlag){
      // Compute the map of Jacobian determinant
      // Create the Jacobian image
      nifti_image *jacobianImage=nullptr;
      if(referenceImage!=nullptr){
         jacobianImage=nifti_copy_nim_info(referenceImage);
         nifti_image_free(referenceImage);referenceImage=nullptr;
      }
      else jacobianImage=nifti_copy_nim_info(inputTransformation);
      jacobianImage->ndim=jacobianImage->dim[0]=jacobianImage->nz>1?3:2;
      jacobianImage->nu=jacobianImage->dim[5]=1;
      jacobianImage->nt=jacobianImage->dim[4]=1;
      jacobianImage->nvox=NiftiImage::calcVoxelNumber(jacobianImage, jacobianImage->ndim);
      jacobianImage->datatype = inputTransformation->datatype;
      jacobianImage->nbyper = inputTransformation->nbyper;
      jacobianImage->cal_min=0;
      jacobianImage->cal_max=0;
      jacobianImage->scl_slope = 1.0f;
      jacobianImage->scl_inter = 0.0f;
      jacobianImage->data = calloc(jacobianImage->nvox, jacobianImage->nbyper);

      switch((int)inputTransformation->intent_p1){
      case DISP_FIELD:
         reg_getDeformationFromDisplacement(inputTransformation);
      case DEF_FIELD:
         reg_defField_getJacobianMap(inputTransformation,jacobianImage);
         break;
      case DISP_VEL_FIELD:
         reg_getDeformationFromDisplacement(inputTransformation);
      case DEF_VEL_FIELD:
         reg_defField_GetJacobianDetFromFlowField(jacobianImage,inputTransformation);
         break;
      case LIN_SPLINE_GRID:
      case CUB_SPLINE_GRID:
         reg_spline_GetJacobianMap(inputTransformation,jacobianImage);
         break;
      case SPLINE_VEL_GRID:
         reg_spline_GetJacobianDetFromVelocityGrid(jacobianImage,inputTransformation);
         break;
      }
      if(flag->outputJacDetFlag)
         reg_io_WriteImageFile(jacobianImage,param->outputJacDetName);
      if(flag->outputLogDetFlag){
         switch(jacobianImage->datatype){
         case NIFTI_TYPE_FLOAT32:
            reg_jacobian_computeLog<float>(jacobianImage);
            break;
         case NIFTI_TYPE_FLOAT64:
            reg_jacobian_computeLog<double>(jacobianImage);
            break;
         }
         reg_io_WriteImageFile(jacobianImage,param->outputLogDetName);
      }
      nifti_image_free(jacobianImage);jacobianImage=nullptr;
   }
   if(flag->outputJacMatFlag){

      nifti_image *jacobianImage=nullptr;
      if(referenceImage!=nullptr){
         jacobianImage=nifti_copy_nim_info(referenceImage);
         nifti_image_free(referenceImage);referenceImage=nullptr;
      }
      else jacobianImage=nifti_copy_nim_info(inputTransformation);
      jacobianImage->ndim=jacobianImage->dim[0]=5;
      jacobianImage->nu=jacobianImage->dim[5]=jacobianImage->nz>1?9:4;
      jacobianImage->nt=jacobianImage->dim[4]=1;
      jacobianImage->nvox=NiftiImage::calcVoxelNumber(jacobianImage, jacobianImage->ndim);
      jacobianImage->datatype = inputTransformation->datatype;
      jacobianImage->nbyper = inputTransformation->nbyper;
      jacobianImage->cal_min=0;
      jacobianImage->cal_max=0;
      jacobianImage->scl_slope = 1.0f;
      jacobianImage->scl_inter = 0.0f;
      jacobianImage->data = calloc(jacobianImage->nvox, jacobianImage->nbyper);

      mat33 *jacobianMatriceArray = (mat33 *)malloc(NiftiImage::calcVoxelNumber(jacobianImage, 3) * sizeof(mat33));
      // Compute the map of Jacobian matrices
      switch((int)inputTransformation->intent_p1){
      case DISP_FIELD:
         reg_getDeformationFromDisplacement(inputTransformation);
      case DEF_FIELD:
         reg_defField_getJacobianMatrix(inputTransformation,jacobianMatriceArray);
         break;
      case DISP_VEL_FIELD:
         reg_getDeformationFromDisplacement(inputTransformation);
      case DEF_VEL_FIELD:
         reg_defField_GetJacobianMatFromFlowField(jacobianMatriceArray,inputTransformation);
         break;
      case LIN_SPLINE_GRID:
      case CUB_SPLINE_GRID:
         reg_spline_GetJacobianMatrix(jacobianImage,inputTransformation,jacobianMatriceArray);
         break;
      case SPLINE_VEL_GRID:
         reg_spline_GetJacobianMatFromVelocityGrid(jacobianMatriceArray,inputTransformation,jacobianImage);
         break;
      }
      switch(jacobianImage->datatype){
      case NIFTI_TYPE_FLOAT32:
         reg_jacobian_convertMat33ToNii<float>(jacobianMatriceArray,jacobianImage);
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_jacobian_convertMat33ToNii<double>(jacobianMatriceArray,jacobianImage);
         break;
      }
      free(jacobianMatriceArray);jacobianMatriceArray=nullptr;
      reg_io_WriteImageFile(jacobianImage,param->outputJacMatName);
      nifti_image_free(jacobianImage);jacobianImage=nullptr;
   }

   // Free the allocated image
   nifti_image_free(inputTransformation);inputTransformation=nullptr;

   return EXIT_SUCCESS;
}
