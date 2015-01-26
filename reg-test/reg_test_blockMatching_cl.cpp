#include "_reg_ReadWriteImage.h"
#include "_reg_blockMatching.h"
#include "_reg_tools.h"

#include"Kernel.h"
#include"Kernels.h"
#include "Platform.h"
#include "cl/CLContent.h"

#define EPS 0.000001

void test(Content* con) {

	Platform *clPlatform = new Platform(NR_PLATFORM_CL);

	Kernel* blockMatchingKernel = clPlatform->createKernel(BlockMatchingKernel::getName(), con);
	blockMatchingKernel->castTo<BlockMatchingKernel>()->calculate();

	delete blockMatchingKernel;
	delete clPlatform;
}

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
   for(size_t i=0;i<referenceImage->nvox;++i) mask[i]=i;

   Content* con = new ClContent(referenceImage, NULL, mask, sizeof(float), 50, 50,1);
   con->setCurrentWarped(warpedImage);
   test(con);

   _reg_blockMatchingParam *blockMatchingParams = con->getBlockMatchingParams();

   mat44 recoveredTransformation;
   reg_mat44_eye(&recoveredTransformation);
   recoveredTransformation.m[0][0]=1.02f;
   recoveredTransformation.m[1][1]=0.98f;
   recoveredTransformation.m[2][2]=0.98f;
   recoveredTransformation.m[0][3]=4.f;
   recoveredTransformation.m[1][3]=4.f;
   recoveredTransformation.m[2][3]=4.f;
   optimize(blockMatchingParams, &recoveredTransformation,  transType);

   mat44 rigid2D;
   rigid2D.m[0][0]=1.020541f;rigid2D.m[0][1]=0.008200279f;rigid2D.m[0][2]=0.f;rigid2D.m[0][3]=3.793443f;
   rigid2D.m[1][0]=0.004867995f;rigid2D.m[1][1]=0.982499f;rigid2D.m[1][2]=0.f;rigid2D.m[1][3]=3.791452f;
   rigid2D.m[2][0]=0.f;rigid2D.m[2][1]=0.f;rigid2D.m[2][2]=1.f;rigid2D.m[2][3]=0.f;
   rigid2D.m[3][0]=0.f;rigid2D.m[3][1]=0.f;rigid2D.m[3][2]=0.f;rigid2D.m[3][3]=1.f;
   mat44 rigid3D;
   rigid3D.m[0][0]=1.024129f;rigid3D.m[0][1]=-0.005762629f;rigid3D.m[0][2]=0.00668848f;rigid3D.m[0][3]=4.136654f;
   rigid3D.m[1][0]=0.003204135f;rigid3D.m[1][1]=0.9722677f;rigid3D.m[1][2]=0.001625132f;rigid3D.m[1][3]=3.815583f;
   rigid3D.m[2][0]=0.008603932f;rigid3D.m[2][1]=0.01103071f;rigid3D.m[2][2]=1.001688f;rigid3D.m[2][3]=3.683818f;
   rigid3D.m[3][0]=0.f;rigid3D.m[3][1]=0.f;rigid3D.m[3][2]=0.f;rigid3D.m[3][3]=1.f;
   mat44 affine2D;
   affine2D.m[0][0]= 0.9997862f;affine2D.m[0][1]=0.02067783f;affine2D.m[0][2]=0.f;affine2D.m[0][3]=3.778748f;
   affine2D.m[1][0]=-0.02067783f;affine2D.m[1][1]=0.9997862f;affine2D.m[1][2]=0.f;affine2D.m[1][3]=3.780472f;
   affine2D.m[2][0]=0.f;affine2D.m[2][1]=0.f;affine2D.m[2][2]=1.f;affine2D.m[2][3]=0.f;
   affine2D.m[3][0]=0.f;affine2D.m[3][1]=0.f;affine2D.m[3][2]=0.f;affine2D.m[3][3]=1.f;
   mat44 affine3D;
   affine3D.m[0][0]=0.9999934f;affine3D.m[0][1]=-0.003248326f;affine3D.m[0][2]=0.001676977f;affine3D.m[0][3]=4.228289f;
   affine3D.m[1][0]=0.003247663f;affine3D.m[1][1]=0.9999947f;affine3D.m[1][2]=0.0004015192f;affine3D.m[1][3]=4.005611f;
   affine3D.m[2][0]=-0.001678258f;affine3D.m[2][1]=-0.0003961027f;affine3D.m[2][2]=0.9999985f;affine3D.m[2][3]=3.649101f;
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

   mat44 differenceMatrix = *testMatrix - recoveredTransformation;
   for(int i=0;i<4;++i){
      for(int j=0;j<4;++j){
         if(fabsf(differenceMatrix.m[i][j])>EPS){
            fprintf(stderr, "reg_test_blockmatching_cl error too large: %g (>%g) [%i,%i]\n", fabs(differenceMatrix.m[i][j]), EPS, i, j);
            return EXIT_FAILURE;
         }
      }
   }

   nifti_image_free(referenceImage);
//   nifti_image_free(warpedImage);
   free(mask);
   delete con;

   return EXIT_SUCCESS;
}
