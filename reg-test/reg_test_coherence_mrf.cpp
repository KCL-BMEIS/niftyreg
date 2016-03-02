#include "_reg_ReadWriteImage.h"
#include "_reg_ReadWriteBinary.h"
#include "_reg_mrf.h"
#include "_reg_localTrans.h"
#include "_reg_resampling.h"
#include "_reg_mind.h"

int main(int argc, char **argv)
{
   time_t start;
   time(&start);

   if(argc!=4) {
      fprintf(stderr, "Usage: %s <refImage> <floatingImage> <expectedDataCost>\n", argv[0]);
      return EXIT_FAILURE;
   }
   //IO
   char *inputRefImageName=argv[1];
   char *inputFloImageName=argv[2];
   char *expectedDataCostName=argv[3];

   // Read reference image
   nifti_image *referenceImage = reg_io_ReadImageFile(inputRefImageName);
   if(referenceImage==NULL){
       reg_print_msg_error("The input reference image could not be read");
       return EXIT_FAILURE;
   }
   reg_tools_changeDatatype<float>(referenceImage);

   // Read floating image
   nifti_image *floatingImage = reg_io_ReadImageFile(inputFloImageName);
   if(floatingImage==NULL){
       reg_print_msg_error("The warped input floating image could not be read");
       return EXIT_FAILURE;
   }
   reg_tools_changeDatatype<float>(floatingImage);

   // Read data cost
   //DURTY - TO CHANGE
   float* expectedDataCost = new float[32388174];
   readFloatBinaryArray(expectedDataCostName, 32388174, expectedDataCost);
   //DEBUG
   //for (int i = 0;i<32388174;i++) {
   //    std::cout<<expectedDataCost[i]<<std::endl;
   //}
   //DEBUG
   //
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

   // Create an empty mask
   int *mask = (int *)calloc(referenceImage->nvox, sizeof(int));

   // Create a deformation field
   nifti_image *deformationField = nifti_copy_nim_info(referenceImage);
   deformationField->dim[0]=deformationField->ndim=5;
   deformationField->dim[1]=deformationField->nx=referenceImage->nx;
   deformationField->dim[2]=deformationField->ny=referenceImage->ny;
   deformationField->dim[3]=deformationField->nz=referenceImage->nz;
   deformationField->dim[4]=deformationField->nt=1;
   deformationField->pixdim[4]=deformationField->dt=1.0;
   if(referenceImage->nz==1)
      deformationField->dim[5]=deformationField->nu=2;
   else deformationField->dim[5]=deformationField->nu=3;
   deformationField->pixdim[5]=deformationField->du=1.0;
   deformationField->dim[6]=deformationField->nv=1;
   deformationField->pixdim[6]=deformationField->dv=1.0;
   deformationField->dim[7]=deformationField->nw=1;
   deformationField->pixdim[7]=deformationField->dw=1.0;
   deformationField->nvox =
         (size_t)deformationField->nx *
         (size_t)deformationField->ny *
         (size_t)deformationField->nz *
         (size_t)deformationField->nt *
         (size_t)deformationField->nu;
   deformationField->nbyper = sizeof(float);
   deformationField->datatype = NIFTI_TYPE_FLOAT32;
   deformationField->data = (void *)calloc(deformationField->nvox,
                                           deformationField->nbyper);
   deformationField->intent_code=NIFTI_INTENT_VECTOR;
   memset(deformationField->intent_name, 0, 16);
   strcpy(deformationField->intent_name,"NREG_TRANS");
   deformationField->intent_p1=DEF_FIELD;
   deformationField->scl_slope=1.f;
   deformationField->scl_inter=0.f;
   reg_spline_getDeformationField(controlPointImage,
                                  deformationField,
                                  mask,
                                  false, //composition
                                  true // bspline
                                  );

   // create a warped image
   nifti_image *warpedImage = nifti_copy_nim_info(referenceImage);
   warpedImage->data = (void *)malloc(warpedImage->nvox * warpedImage->nbyper);
   reg_resampleImage(floatingImage,
                     warpedImage,
                     deformationField,
                     mask,
                     1,
                     0.f);

   int mind_length = 12;
   //MINDSSC image
   nifti_image *MINDSSC_refimg = nifti_copy_nim_info(referenceImage);
   MINDSSC_refimg->ndim = MINDSSC_refimg->dim[0] = 4;
   MINDSSC_refimg->nt = MINDSSC_refimg->dim[4] = mind_length;
   MINDSSC_refimg->nvox = MINDSSC_refimg->nvox*mind_length;
   MINDSSC_refimg->data=(void *)calloc(MINDSSC_refimg->nvox,MINDSSC_refimg->nbyper);
   // Compute the MIND descriptor
   GetMINDSSCImageDesciptor(referenceImage,MINDSSC_refimg, mask, 1, 0);

   //MINDSSC image
   nifti_image *MINDSSC_warimg = nifti_copy_nim_info(warpedImage);
   MINDSSC_warimg->ndim = MINDSSC_warimg->dim[0] = 4;
   MINDSSC_warimg->nt = MINDSSC_warimg->dim[4] = mind_length;
   MINDSSC_warimg->nvox = MINDSSC_warimg->nvox*mind_length;
   MINDSSC_warimg->data=(void *)calloc(MINDSSC_warimg->nvox,MINDSSC_warimg->nbyper);
   // Compute the MIND descriptor
   GetMINDSSCImageDesciptor(warpedImage,MINDSSC_warimg, mask, 1, 0);

   reg_ssd* ssdMeasure = new reg_ssd();
   /* Read and create the mask array */
   for(int i=0;i<MINDSSC_refimg->nt;++i) {
      ssdMeasure->SetActiveTimepoint(i);
   }
   ssdMeasure->InitialiseMeasure(MINDSSC_refimg,MINDSSC_warimg,mask,MINDSSC_warimg,NULL,NULL);
   reg_mrf* reg_mrfObject =
           new reg_mrf(ssdMeasure,referenceImage,controlPointImage,18,3,0);//18,3 = default parameters
   reg_mrfObject->GetDiscretisedMeasure();
}
