#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include "_reg_ReadWriteImage.h"
#include "_reg_localTrans.h"
#include "_reg_localTrans_jac.h"
#include "_reg_ssd.h"
#include <numeric>

#include "_reg_mind.h"
#include "_reg_mrf.h"
#include "_reg_discrete_init.h"

int main(int argc, char **argv)
{
   time_t start;
   time(&start);

   if(argc!=5) {
      fprintf(stderr, "Usage: %s <refImage> <floatingImage> <regularisationWeight> <outputImageName>\n", argv[0]);
      return EXIT_FAILURE;
   }
   //IO
   char *inputRefImageName=argv[1];
   char *inputFloImageName=argv[2];
   float regularisationWeight=atof(argv[3]);
   char *outputImageName=argv[4];

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

   // Create control point grid image
   int spacing_voxel = 5;
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
                                  false // bspline
                                  );

   // create a warped image
   nifti_image *warpedImage = nifti_copy_nim_info(referenceImage);
   warpedImage->data = (void *)malloc(warpedImage->nvox * warpedImage->nbyper);
   reg_resampleImage(floatingImage,
                     warpedImage,
                     deformationField,
                     mask,
                     3,
                     0.f);

   int mind_length = 6;
   //MINDSSC image
   nifti_image *MINDSSC_refimg = nifti_copy_nim_info(referenceImage);
   MINDSSC_refimg->ndim = MINDSSC_refimg->dim[0] = 4;
   MINDSSC_refimg->nt = MINDSSC_refimg->dim[4] = mind_length;
   MINDSSC_refimg->nvox = MINDSSC_refimg->nvox*mind_length;
   MINDSSC_refimg->data=(void *)calloc(MINDSSC_refimg->nvox,MINDSSC_refimg->nbyper);
   // Compute the MIND descriptor
   GetMINDImageDesciptor(referenceImage,MINDSSC_refimg, mask, 1, 0);

   //MINDSSC image
   nifti_image *MINDSSC_warimg = nifti_copy_nim_info(warpedImage);
   MINDSSC_warimg->ndim = MINDSSC_warimg->dim[0] = 4;
   MINDSSC_warimg->nt = MINDSSC_warimg->dim[4] = mind_length;
   MINDSSC_warimg->nvox = MINDSSC_warimg->nvox*mind_length;
   MINDSSC_warimg->data=(void *)calloc(MINDSSC_warimg->nvox,MINDSSC_warimg->nbyper);
   // Compute the MIND descriptor
   GetMINDImageDesciptor(warpedImage,MINDSSC_warimg, mask, 1, 0);

   reg_ssd* ssdMeasure = new reg_ssd();

   for(int i=0;i<MINDSSC_refimg->nt;++i)
      ssdMeasure->SetActiveTimepoint(i);
   ssdMeasure->InitialiseMeasure(MINDSSC_refimg,
                                 MINDSSC_warimg,
                                 mask,
                                 MINDSSC_warimg,
                                 NULL,
                                 NULL);

   std::cout << "INIT VALUE = " << ssdMeasure->GetSimilarityMeasureValue() << std::endl;

   reg_discrete_init* reg_dcObject = new reg_discrete_init(ssdMeasure,
                                                           referenceImage,
                                                           controlPointImage,
                                                           18,
                                                           3,
                                                           100,
                                                           regularisationWeight);

   reg_dcObject->Run();

   reg_spline_getDeformationField(controlPointImage,
                                  deformationField,
                                  mask,
                                  false, //composition
                                  false // bspline
                                  );
   reg_resampleImage(floatingImage,
                     warpedImage,
                     deformationField,
                     mask,
                     3,
                     0.f);
   GetMINDImageDesciptor(warpedImage,MINDSSC_warimg, mask, 1, 0);


   std::cout << "FINAL VALUE = " << ssdMeasure->GetSimilarityMeasureValue() << std::endl;

   reg_io_WriteImageFile(controlPointImage, "cpp.nii.gz");

   warpedImage->cal_min = floatingImage->cal_min;
   warpedImage->cal_max = floatingImage->cal_max;
   reg_io_WriteImageFile(warpedImage, outputImageName);

   nifti_image *jac_image = nifti_copy_nim_info(referenceImage);
   jac_image->data=(void *)calloc(jac_image->nvox,jac_image->nbyper);
   reg_spline_GetJacobianMap(controlPointImage, jac_image);

   reg_getDisplacementFromDeformation(deformationField);
   deformationField->dim[4] = deformationField->nt = deformationField->nu;
   deformationField->dim[5] = deformationField->nu = 1;
   deformationField->dim[0] = deformationField->ndim = 4;
   reg_io_WriteImageFile(deformationField, "disp.nii.gz");


   delete reg_dcObject;
   free(mask);
   nifti_image_free(MINDSSC_refimg);
   nifti_image_free(MINDSSC_warimg);
   nifti_image_free(referenceImage);
   nifti_image_free(floatingImage);
   nifti_image_free(warpedImage);
   nifti_image_free(controlPointImage);
   nifti_image_free(deformationField);
   nifti_image_free(jac_image);

   time_t end;
   time(&end);
   int minutes=(int)floorf((end-start)/60.0f);
   int seconds=(int)(end-start - 60*minutes);
   char text[255];
   sprintf(text, "Registration performed in %i min %i sec", minutes, seconds);
   reg_print_info((argv[0]), text);
   reg_print_info((argv[0]), "Have a good day !");

   return EXIT_SUCCESS;
}

