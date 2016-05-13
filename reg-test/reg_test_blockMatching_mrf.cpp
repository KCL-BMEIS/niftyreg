#include "_reg_ReadWriteImage.h"
#include "_reg_ReadWriteBinary.h"
#include "_reg_mrf.h"
#include "_reg_localTrans.h"
#include "_reg_resampling.h"
#include "_reg_mind.h"

#define EPS 0.000001

int main(int argc, char **argv)
{
   time_t start;
   time(&start);

   if(argc!=4) {
      fprintf(stderr, "Usage: %s <refImage> <warpedImage> <expectedDataCost>\n", argv[0]);
      return EXIT_FAILURE;
   }
   //IO
   char *inputRefImageName=argv[1];
   char *inputWarImageName=argv[2];
   char *expectedDataCostName=argv[3];

   // Read reference image
   nifti_image *referenceImage = reg_io_ReadImageFile(inputRefImageName);
   if(referenceImage==NULL){
       reg_print_msg_error("The input reference image could not be read");
       return EXIT_FAILURE;
   }
   reg_tools_changeDatatype<float>(referenceImage);

   // Read floating image
   nifti_image *warpedImage = reg_io_ReadImageFile(inputWarImageName);
   if(warpedImage==NULL){
       reg_print_msg_error("The warped input floating image could not be read");
       return EXIT_FAILURE;
   }
   reg_tools_changeDatatype<float>(warpedImage);

   //DISP
   int discrete_radius = 18;
   int discrete_increment = 3;
   // Create control point grid image
   int spacing_voxel = 8;
   float grid_step_mm[3]={spacing_voxel*referenceImage->dx,
                          spacing_voxel*referenceImage->dy,
                          spacing_voxel*referenceImage->dz};
   nifti_image *controlPointImage = NULL;
   reg_createControlPointGrid<float>(&controlPointImage,
                                     referenceImage,
                                     grid_step_mm);
   memset(controlPointImage->data,0,
          controlPointImage->nvox*controlPointImage->nbyper);
   reg_tools_multiplyValueToImage(controlPointImage,controlPointImage,0.f);
   reg_getDeformationFromDisplacement(controlPointImage);

   //CP
   size_t nb_CP = controlPointImage->nx * controlPointImage->ny * controlPointImage->nz;
   size_t nb_disp = discrete_radius/discrete_increment*2+1;
   size_t nb_disp3D = nb_disp*nb_disp*nb_disp;
   // Read data cost
   //DURTY - TO CHANGE
   float* expectedDataCost = new float[nb_CP*nb_disp3D];
   readFloatBinaryArray(expectedDataCostName, nb_CP*nb_disp3D, expectedDataCost);
   //DEBUG
   //for (int i = 0;i<32388174;i++) {
   //    std::cout<<expectedDataCost[i]<<std::endl;
   //}
   //DEBUG

   // Create an empty mask
   int *mask = (int *)calloc(referenceImage->nvox, sizeof(int));

   reg_ssd* ssdMeasure = new reg_ssd();
   /* Read and create the mask array */
   for(int i=0;i<referenceImage->nt;++i) {
      ssdMeasure->SetActiveTimepoint(i);
   }
   ssdMeasure->InitialiseMeasure(referenceImage,warpedImage,mask,warpedImage,NULL,NULL);
   reg_mrf* reg_mrfObject =
           new reg_mrf(ssdMeasure,referenceImage,controlPointImage,discrete_radius,discrete_increment,0);//18,3 = default parameters
   reg_mrfObject->GetDiscretisedMeasure();
   //Let's compare the 2 datacosts
   for(size_t i=0;i<nb_CP*nb_disp3D;i++) {
       float currentValue = reg_mrfObject->GetDiscretisedMeasurePtr()[i];
       float expectedValue = expectedDataCost[i];
       if((currentValue - expectedValue) > EPS) {
           reg_print_msg_error("the 2 dataCost are different");
           return EXIT_FAILURE;
       }
   }
   //
   delete[] expectedDataCost;
#ifndef NDEBUG
   printf("All good\n");
#endif
   return EXIT_SUCCESS;
}

