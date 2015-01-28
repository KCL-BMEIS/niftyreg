#include "_reg_ReadWriteImage.h"
#include "_reg_blockMatching.h"
#include "_reg_tools.h"

#define EPS 0.000001

int main(int argc, char **argv)
{

   if(argc!=4)
   {
      fprintf(stderr, "Usage: %s <refImage> <warImage> <transType>\n", argv[0]);
      return EXIT_FAILURE;
   }

   char *inputRefImageName=argv[1];
   char *inputWarImageName=argv[2];
   int transType=atoi(argv[3]);

   // Read the input reference image
   nifti_image *referenceImage = reg_io_ReadImageFile(inputRefImageName);
   if(referenceImage==NULL){
      reg_print_msg_error("The input reference image could not be read");
      return EXIT_FAILURE;
   }
   reg_tools_changeDatatype<float>(referenceImage);
   // Read the input floating image
   nifti_image *warpedImage = reg_io_ReadImageFile(inputWarImageName);
   if(warpedImage==NULL){
      reg_print_msg_error("The input floating image could not be read");
      return EXIT_FAILURE;
   }
   reg_tools_changeDatatype<float>(warpedImage);

   // Create a mask
   int *mask=(int *)malloc(referenceImage->nvox*sizeof(int));
   for(size_t i=0;i<referenceImage->nvox;++i)
      mask[i]=i;

   _reg_blockMatchingParam blockMatchingParams;
   initialise_block_matching_method(referenceImage,
                                    &blockMatchingParams,
                                    50,
                                    50,
                                    1,
                                    mask,
                                    false // GPU is not used here
                                   );
   block_matching_method(referenceImage,
                         warpedImage,
                         &blockMatchingParams,
                         mask);
   mat44 recoveredTransformation;
   reg_mat44_eye(&recoveredTransformation);
   recoveredTransformation.m[0][0]=1.02f;
   recoveredTransformation.m[1][1]=0.98f;
   recoveredTransformation.m[2][2]=0.98f;
   recoveredTransformation.m[0][3]=4.f;
   recoveredTransformation.m[1][3]=4.f;
   recoveredTransformation.m[2][3]=4.f;
   optimize(&blockMatchingParams,
            &recoveredTransformation,
            transType);

   nifti_image_free(warpedImage);
   free(mask);

   mat44 rigid2D;
   rigid2D.m[0][0]=1.027961f;rigid2D.m[0][1]=-0.004180538f;rigid2D.m[0][2]=0.f;rigid2D.m[0][3]=3.601387f;
   rigid2D.m[1][0]=0.01252018f;rigid2D.m[1][1]=0.9764945f;rigid2D.m[1][2]=0.f;rigid2D.m[1][3]=3.17229f;
   rigid2D.m[2][0]=0.f;rigid2D.m[2][1]=0.f;rigid2D.m[2][2]=1.f;rigid2D.m[2][3]=0.f;
   rigid2D.m[3][0]=0.f;rigid2D.m[3][1]=0.f;rigid2D.m[3][2]=0.f;rigid2D.m[3][3]=1.f;

   mat44 rigid3D;
   rigid3D.m[0][0]=1.028082f;rigid3D.m[0][1]=-0.004869822f;rigid3D.m[0][2]=0.007795987f;rigid3D.m[0][3]=4.177487f;
   rigid3D.m[1][0]=0.01129405f;rigid3D.m[1][1]=0.9697745f;rigid3D.m[1][2]=0.005026158f;rigid3D.m[1][3]=3.874551f;
   rigid3D.m[2][0]=0.004100456f;rigid3D.m[2][1]=0.01087017f;rigid3D.m[2][2]=1.005741f;rigid3D.m[2][3]=4.011357;
   rigid3D.m[3][0]=0.f;rigid3D.m[3][1]=0.f;rigid3D.m[3][2]=0.f;rigid3D.m[3][3]=1.f;

   mat44 affine2D;
   affine2D.m[0][0]= 0.9999999f;affine2D.m[0][1]=0.0003671125f;affine2D.m[0][2]=0.f;affine2D.m[0][3]=3.652262f;
   affine2D.m[1][0]=-0.0003671125f;affine2D.m[1][1]=0.9999999f;affine2D.m[1][2]=0.f;affine2D.m[1][3]=3.319299f;
   affine2D.m[2][0]=0.f;affine2D.m[2][1]=0.f;affine2D.m[2][2]=1.f;affine2D.m[2][3]=0.f;
   affine2D.m[3][0]=0.f;affine2D.m[3][1]=0.f;affine2D.m[3][2]=0.f;affine2D.m[3][3]=1.f;

   mat44 affine3D;
   affine3D.m[0][0]=0.9999814f;affine3D.m[0][1]=-0.004359253f;affine3D.m[0][2]=0.004272044f;affine3D.m[0][3]=4.355269f;
   affine3D.m[1][0]=0.004345424f;affine3D.m[1][1]=0.9999853f;affine3D.m[1][2]=0.003243448f;affine3D.m[1][3]=4.134418f;
   affine3D.m[2][0]=-0.004286081f;affine3D.m[2][1]=-0.00322482f;affine3D.m[2][2]=0.9999856f;affine3D.m[2][3]=3.725645f;
   affine3D.m[3][0]=0.f;affine3D.m[3][1]=0.f;affine3D.m[3][2]=0.f;affine3D.m[3][3]=1.f;

   mat44 *testMatrix=NULL;
   if(referenceImage->nz>1)
   {
      if(transType==0)
         testMatrix=&affine3D;
      else testMatrix=&rigid3D;
   }
   else
   {
      if(transType==0)
         testMatrix=&affine2D;
      else testMatrix=&rigid2D;

   }

   nifti_image_free(referenceImage);


   reg_mat44_disp(testMatrix,(char *)"expected");
   reg_mat44_disp(&recoveredTransformation,(char *)"recovered");

   mat44 differenceMatrix = *testMatrix - recoveredTransformation;
   for(int i=0;i<4;++i){
      for(int j=0;j<4;++j){
         if(fabsf(differenceMatrix.m[i][j])>EPS){
            fprintf(stderr, "reg_test_fullAffine error too large: %g (>%g) [%i,%i]\n",
                    fabs(differenceMatrix.m[i][j]), EPS, i, j);
            return EXIT_FAILURE;
         }
      }
   }

   return EXIT_SUCCESS;
}

