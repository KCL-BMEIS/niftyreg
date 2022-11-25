#include "_reg_ReadWriteImage.h"
#include "_reg_aladin_sym.h"
#include "_reg_tools.h"

#define EPS 0.000001

int main(int argc, char **argv)
{

   if(argc!=4)
   {
      fprintf(stderr, "Usage: %s <refImage> <floImage> <expectedMatrix>\n", argv[0]);
      return EXIT_FAILURE;
   }

   char *inputRefImageName=argv[1];
   char *inputFloImageName=argv[2];
   char *inputMatFileName=argv[3];

   // Read the input reference image
   nifti_image *referenceImage = reg_io_ReadImageFile(inputRefImageName);
   if(referenceImage==nullptr){
      reg_print_msg_error("The input reference image could not be read");
      return EXIT_FAILURE;
   }
   reg_tools_changeDatatype<float>(referenceImage);
   // Read the input reference image
   nifti_image *floatingImage = reg_io_ReadImageFile(inputFloImageName);
   if(floatingImage==nullptr){
      reg_print_msg_error("The input floating image could not be read");
      return EXIT_FAILURE;
   }
   reg_tools_changeDatatype<float>(floatingImage);

   // Read the input affine matrix
   mat44 *inputMatrix=(mat44 *)malloc(sizeof(mat44));
   reg_tool_ReadAffineFile(inputMatrix, inputMatFileName);

   // Run the affine registration
   reg_aladin_sym<float> *affine=new reg_aladin_sym<float>();
   affine->SetInputReference(referenceImage);
   affine->SetInputFloating(floatingImage);
   affine->SetPlatformCode(NR_PLATFORM_CPU);
   affine->Run();
   mat44 differenceMatrix = *inputMatrix - *(affine->GetTransformationMatrix());

   // Cleaning up
   nifti_image_free(referenceImage);
   nifti_image_free(floatingImage);

   for(int i=0;i<4;++i){
      for(int j=0;j<4;++j){
         if(fabsf(differenceMatrix.m[i][j])>EPS){
            fprintf(stderr, "reg_test_fullAffine error too large: %g (>%g)\n",
                    fabs(differenceMatrix.m[i][j]), EPS);
            reg_mat44_disp(inputMatrix, (char *)"Expected Matrix");
            reg_mat44_disp(affine->GetTransformationMatrix(), (char *)"Obtained Matrix");
            reg_mat44_disp(&differenceMatrix, (char *)"Difference Matrix");
            free(inputMatrix);
            delete affine;
            return EXIT_FAILURE;
         }
      }
   }
   free(inputMatrix);
   delete affine;

   return EXIT_SUCCESS;
}
