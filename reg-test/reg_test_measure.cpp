#include "_reg_ReadWriteImage.h"
#include "_reg_ReadWriteMatrix.h"
#include "_reg_tools.h"
#include "_reg_nmi.h"
#include "_reg_ssd.h"
#include "_reg_mind.h"
#include "_reg_lncc.h"

#define EPS 0.000001

int main(int argc, char **argv)
{

   if(argc!=5)
   {
      fprintf(stderr, "Usage: %s <refImage> <warImage> <LNCC|NMI|SSD|MIND> <expectedValueFile>\n", argv[0]);
      return EXIT_FAILURE;
   }

   double max_difference = EPS;

   char *inputRefImageName=argv[1];
   char *inputWarImageName=argv[2];
   char *measure_type=argv[3];
   char *inputMatrixFilename = argv[4];

   /* Read the reference image */
   nifti_image *refImage = reg_io_ReadImageFile(inputRefImageName);
   if(refImage == NULL)
   {
      fprintf(stderr,"[NiftyReg ERROR] Error when reading the reference image: %s\n",
              inputRefImageName);
      return EXIT_FAILURE;
   }
   reg_tools_changeDatatype<float>(refImage);

   /* Read the warped image */
   nifti_image *warImage = reg_io_ReadImageFile(inputWarImageName);
   if(warImage == NULL)
   {
      fprintf(stderr,"[NiftyReg ERROR] Error when reading the floating image: %s\n",
              inputWarImageName);
      return EXIT_FAILURE;
   }
   reg_tools_changeDatatype<float>(warImage);

   /* Read the expected value */
   std::pair<size_t, size_t> inputMatrixSize = reg_tool_sizeInputMatrixFile(inputMatrixFilename);
   size_t m = inputMatrixSize.first;
   size_t n = inputMatrixSize.second;
   if(m != 1 && n!= 1)
   {
      fprintf(stderr,"[NiftyReg ERROR] Error when reading the expected similarity measure value: %s\n",
              inputMatrixFilename);
      return EXIT_FAILURE;
   }
   float **inputMatrix = reg_tool_ReadMatrixFile<float>(inputMatrixFilename, m, n);

   // Check if the input images have the same size
   for(int i=0;i<8;++i){
      if(refImage->dim[i]!=warImage->dim[i])
      {
         reg_print_msg_error("reg_test_measure: The input images do not have the same size");
         return EXIT_FAILURE;
      }
   }

   int *mask_image=(int *)calloc(refImage->nvox,sizeof(int));

   /* Compute the LNCC if required */
   if(strcmp(measure_type, "SSD")==0)
   {
      reg_ssd *measure_object=new reg_ssd();
      for(int i=0;i<refImage->nt;++i){
         measure_object->SetTimepointWeight(i, 1.);
         measure_object->SetNormaliseTimepoint(i,true);
      }
      measure_object->InitialiseMeasure(refImage,
                                        warImage,
                                        mask_image,
                                        warImage,
                                        NULL,
                                        NULL,
                                        NULL);
      double measure=measure_object->GetSimilarityMeasureValue();

#ifndef NDEBUG
      printf("reg_test_measure: SSD value %iD = %.7g\n",
             (refImage->nz>1?3:2), measure);
#endif
      double expectedValue = inputMatrix[0][0];
      max_difference = fabs(measure-expectedValue);
      //
      if(max_difference>EPS)
      {
         printf("reg_test_measure: Incorrect measure value %.7g (diff=%.7g)\n",
                measure, max_difference);
         return EXIT_FAILURE;
      }
      delete measure_object;
   }
   /* Compute the MIND if required */
   else if(strcmp(measure_type, "MIND")==0)
   {
      reg_mind *measure_object=new reg_mind();
      //Let's normalize between 0..1
      for(int i=0;i<refImage->nt;++i)
         measure_object->SetTimepointWeight(i, 1.);
      measure_object->InitialiseMeasure(refImage,
                                        warImage,
                                        mask_image,
                                        warImage,
                                        NULL,
                                        NULL);
      double measure=measure_object->GetSimilarityMeasureValue();
#ifndef NDEBUG
      printf("reg_test_measure: MIND value %iD = %.7g\n",
             (refImage->nz>1?3:2), measure);
#endif
      double expectedValue = inputMatrix[0][0];
      max_difference = fabs(measure-expectedValue);
      //
      if(max_difference>EPS)
      {
         printf("reg_test_measure: Incorrect measure value %.7g (diff=%.7g)\n",
                measure, max_difference);
         return EXIT_FAILURE;
      }
      delete measure_object;
   }
   else
   {
      reg_print_msg_error("reg_test_measure: Unknown measure type");
      return EXIT_FAILURE;
   }

   // Free the allocated images
   nifti_image_free(refImage);
   nifti_image_free(warImage);
   free(mask_image);
   reg_matrix2DDeallocate(m, inputMatrix);

#ifndef NDEBUG
    fprintf(stdout, "reg_test_measure ok: %g (<%g)\n", max_difference, EPS);
#endif

   return EXIT_SUCCESS;
}
