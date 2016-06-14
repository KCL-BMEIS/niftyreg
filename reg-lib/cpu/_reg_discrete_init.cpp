#include "_reg_discrete_init.h"

/*****************************************************/
reg_discrete_init::reg_discrete_init(reg_measure *_measure,
                                     nifti_image *_referenceImage,
                                     nifti_image *_controlPointImage,
                                     int _discrete_radius,
                                     int _discrete_increment,
                                     int _reg_max_it,
                                     float _reg_weight)
{
   this->measure = _measure;
   this->referenceImage = _referenceImage;
   this->controlPointImage = _controlPointImage;
   this->discrete_radius = _discrete_radius;
   this->discrete_increment = _discrete_increment;
   this->regularisation_weight = _reg_weight;
   this->reg_max_it = _reg_max_it;

   if(this->discrete_radius/this->discrete_increment !=
      (float)this->discrete_radius/(float)this->discrete_increment){
      reg_print_fct_error("reg_discrete_init:reg_discrete_init()");
      reg_print_msg_error("The discrete_radius is expected to be a multiple of discretise_increment");
   }

   this->image_dim = this->referenceImage->nz > 1 ? 3 :2;
   this->label_1D_num = (this->discrete_radius / this->discrete_increment ) * 2 + 1;
   this->label_nD_num = static_cast<int>(std::pow((double) this->label_1D_num,this->image_dim));
   this->node_number = (size_t)this->controlPointImage->nx *
         this->controlPointImage->ny * this->controlPointImage->nz;

   this->input_transformation=nifti_copy_nim_info(this->controlPointImage);
   this->input_transformation->data=(float *)malloc(this->node_number*this->image_dim*sizeof(float));

   // Allocate the discretised values in voxel
   int *discrete_values_vox = (int *)malloc(this->label_1D_num*sizeof(int));
   int currentValue = -this->discrete_radius;
   for(int i = 0;i<this->label_1D_num;i++) {
      discrete_values_vox[i]=currentValue;
      currentValue+=this->discrete_increment;
   }

   // Allocate the discretised values in millimeter
   this->discrete_values_mm = (float **)malloc(this->image_dim*sizeof(float *));
   for(int i=0;i<this->image_dim;++i){
      this->discrete_values_mm[i] = (float *)malloc(this->label_nD_num*sizeof(float));
   }
   float disp_vox[3];
   mat44 vox2mm = this->referenceImage->qto_xyz;
   if(this->referenceImage->sform_code>0)
      vox2mm = this->referenceImage->sto_xyz;
   int i=0;
   for(int z=0; z<this->label_1D_num; ++z){
      disp_vox[2]=discrete_values_vox[z];
      for(int y=0; y<this->label_1D_num; ++y){
         disp_vox[1]=discrete_values_vox[y];
         for(int x=0; x<this->label_1D_num; ++x){
            disp_vox[0]=discrete_values_vox[x];
            this->discrete_values_mm[0][i] =
                  disp_vox[0] * vox2mm.m[0][0] +
                  disp_vox[1] * vox2mm.m[0][1] +
                  disp_vox[2] * vox2mm.m[0][2];
            this->discrete_values_mm[1][i] =
                  disp_vox[0] * vox2mm.m[1][0] +
                  disp_vox[1] * vox2mm.m[1][1] +
                  disp_vox[2] * vox2mm.m[1][2];
            this->discrete_values_mm[2][i] =
                  disp_vox[0] * vox2mm.m[2][0] +
                  disp_vox[1] * vox2mm.m[2][1] +
                  disp_vox[2] * vox2mm.m[2][2];
            ++i;
         }
      }
   }
   free(discrete_values_vox);

   //regularization - optimization
   this->optimal_label_index=(int *)malloc(this->node_number*sizeof(int));
   currentValue= (this->label_1D_num-1)/2;
   currentValue = (currentValue*this->label_1D_num+currentValue)*this->label_1D_num+currentValue;
   for(size_t n=0; n<this->node_number; ++n)
      this->optimal_label_index[n]=currentValue;

   //To store the cost data term
   this->discretised_measures = (float *)calloc(this->node_number*this->label_nD_num, sizeof(float));

   //Optimal transformation based on the data term
   this->regularised_measures = (float *)malloc(this->node_number*this->label_nD_num*sizeof(float));

   // Compute the l2 for each label
   l2_weight = 1.e-10f;
   this->l2_penalisation = (float *)malloc(this->label_nD_num*sizeof(float));
   int label_index=0;
   for(float z=-this->discrete_radius; z<=this->discrete_radius; z+=this->discrete_increment)
      for(float y=-this->discrete_radius; y<=this->discrete_radius; y+=this->discrete_increment)
         for(float x=-this->discrete_radius; x<=this->discrete_radius; x+=this->discrete_increment)
            this->l2_penalisation[label_index++] = std::sqrt(x*x+y*y+z*z);
}
/*****************************************************/
/*****************************************************/
reg_discrete_init::~reg_discrete_init()
{
   if(this->discretised_measures!=NULL)
      free(this->discretised_measures);
   this->discretised_measures=NULL;

   if(this->regularised_measures!=NULL)
      free(this->regularised_measures);
   this->regularised_measures=NULL;

   if(this->l2_penalisation!=NULL)
      free(this->l2_penalisation);
   this->l2_penalisation=NULL;

   if(this->optimal_label_index!=NULL)
      free(this->optimal_label_index);
   this->optimal_label_index=NULL;

   for(int i=0; i<this->image_dim; ++i){
      if(this->discrete_values_mm[i]!=NULL)
         free(this->discrete_values_mm[i]);
      this->discrete_values_mm[i]=NULL;
   }
   if(this->discrete_values_mm!=NULL)
      free(this->discrete_values_mm);
   this->discrete_values_mm=NULL;

   if(this->input_transformation!=NULL)
      nifti_image_free(this->input_transformation);
   this->input_transformation=NULL;
}
/*****************************************************/
/*****************************************************/
void reg_discrete_init::GetDiscretisedMeasure()
{
   measure->GetDiscretisedValue(this->controlPointImage,
                                this->discretised_measures,
                                this->discrete_radius,
                                this->discrete_increment);
#ifndef NDEBUG
   reg_print_msg_debug("reg_discrete_init::GetDiscretisedMeasure done");
#endif
}
/*****************************************************/
/*****************************************************/
void reg_discrete_init::getOptimalLabel()
{
   this->regularisation_convergence=0;
   size_t opt_label = 0;
   for(size_t node=0; node<this->node_number; ++node){
      size_t current_optimal = this->optimal_label_index[node];
      opt_label =
            std::max_element(this->regularised_measures+node*this->label_nD_num,
                             this->regularised_measures+(node+1)*this->label_nD_num) -
                            (this->regularised_measures+node*this->label_nD_num);
      this->optimal_label_index[node] = opt_label;
      if(current_optimal != opt_label)
         ++this->regularisation_convergence;
   }
#ifndef NDEBUG
   reg_print_msg_debug("reg_discrete_init::getOptimalLabel done");
#endif
}
/*****************************************************/
/*****************************************************/
void reg_discrete_init::UpdateTransformation()
{
   //Update the control point position
   float *cpPtrX = static_cast<float *>(this->controlPointImage->data);
   float *cpPtrY = &cpPtrX[this->node_number];
   float *cpPtrZ = &cpPtrY[this->node_number];

   float *inputCpPtrX = static_cast<float *>(this->input_transformation->data);
   float *inputCpPtrY = &inputCpPtrX[this->node_number];
   float *inputCpPtrZ = &inputCpPtrY[this->node_number];

   memcpy(cpPtrX, inputCpPtrX, this->node_number*3*sizeof(float));
   //float scaleFactor = 0.5;
   float scaleFactor = 1;

   for(int z=1; z<this->controlPointImage->nz-1; z++) {
      for(int y=1; y<this->controlPointImage->ny-1; y++) {
         size_t node = (z*this->controlPointImage->ny+y)*this->controlPointImage->nx+1;
         for(int x=1; x<this->controlPointImage->nx-1; x++){
            int optimal_id = this->optimal_label_index[node];
            cpPtrX[node] = inputCpPtrX[node] + scaleFactor*this->discrete_values_mm[0][optimal_id];
            cpPtrY[node] = inputCpPtrY[node] + scaleFactor*this->discrete_values_mm[1][optimal_id];
            cpPtrZ[node] = inputCpPtrZ[node] + scaleFactor*this->discrete_values_mm[2][optimal_id];
            ++node;
         }
      }
   }

#ifndef NDEBUG
   reg_print_msg_debug("reg_discrete_init::UpdateTransformation done");
#endif
}
/*****************************************************/
/*****************************************************/
void reg_discrete_init::AddL2Penalisation(float weight)
{
   // Compute the l2 for each label
   float *l2_penalisation = (float *)malloc(this->label_nD_num*sizeof(float));
   int label_index=0;
   for(float z=-this->discrete_radius; z<=this->discrete_radius; z+=this->discrete_increment)
      for(float y=-this->discrete_radius; y<=this->discrete_radius; y+=this->discrete_increment)
         for(float x=-this->discrete_radius; x<=this->discrete_radius; x+=this->discrete_increment)
            l2_penalisation[label_index++] = weight * sqrt(x*x+y*y+z*z);

   // Loop over all control points
   int measure_index, n;
   int _node_number = static_cast<int>(this->node_number);
   int _label_nD_num = this->label_nD_num;
   float *_discretised_measures = &this->discretised_measures[0];
#if defined (_OPENMP)
   #pragma omp parallel for default(none) \
   shared(_node_number, _label_nD_num, _discretised_measures, l2_penalisation) \
   private(measure_index, n, label_index)
#endif
   for(n=0; n<_node_number; ++n){
      measure_index = n * _label_nD_num;
      // Loop over all label
      for(label_index=0; label_index<_label_nD_num; ++label_index){
         _discretised_measures[measure_index] -= l2_penalisation[label_index];
         ++measure_index;
      }
   }

   free(l2_penalisation);
}
/*****************************************************/
/*****************************************************/
void reg_discrete_init::GetRegularisedMeasure()
{
   reg_getDisplacementFromDeformation(this->controlPointImage);
   reg_getDisplacementFromDeformation(this->input_transformation);

   float *cpPtrX = static_cast<float *>(this->controlPointImage->data);
   float *cpPtrY = &cpPtrX[this->node_number];
   float *cpPtrZ = &cpPtrY[this->node_number];

   float *inputCpPtrX = static_cast<float *>(this->input_transformation->data);
   float *inputCpPtrY = &inputCpPtrX[this->node_number];
   float *inputCpPtrZ = &inputCpPtrY[this->node_number];

   float basisXX[27], basisYY[27], basisZZ[27], basisXY[27], basisYZ[27], basisXZ[27];
   float _basisXX, _basisYY, _basisZZ, _basisXY, _basisYZ, _basisXZ;
   float basis[4], first[4], second[4];
   get_BSplineBasisValues<float>(0.f, basis, first, second);
   int i=0;
   for(int c=0; c<3; ++c){
      for(int b=0; b<3; ++b){
         for(int a=0; a<3; ++a){
            basisXX[i]=second[a]*basis[b]*basis[c];
            basisYY[i]=basis[a]*second[b]*basis[c];
            basisZZ[i]=basis[a]*basis[b]*second[c];
            basisXY[i]=first[a]*first[b]*basis[c];
            basisYZ[i]=basis[a]*first[b]*first[c];
            basisXZ[i]=first[a]*basis[b]*first[c];
            ++i;
         }
      }
   }
   _basisXX = basisXX[13]; _basisYY = basisYY[13]; _basisZZ = basisZZ[13];
   _basisXY = basisXY[13]; _basisYZ = basisYZ[13]; _basisXZ = basisXZ[13];

   float splineCoeffX[27], splineCoeffY[27], splineCoeffZ[27];

   size_t node = 0;
   for(int z=0; z<this->controlPointImage->nz; z++) {
      for(int y=0; y<this->controlPointImage->ny; y++) {
         for(int x=0; x<this->controlPointImage->nx; x++){
            // Copy all 27 required control point displacement
            i=0;
            for(int c=z-1; c<z+2; c++){
               for(int b=y-1; b<y+2; b++){
                  for(int a=x-1; a<x+2; a++){
                     if(a>-1 && a<this->controlPointImage->nx &&
                        b>-1 && b<this->controlPointImage->ny &&
                        c>-1 && c<this->controlPointImage->nz){
                        int node_index = (c*this->controlPointImage->ny+b)*this->controlPointImage->nx+a;
                        splineCoeffX[i] = cpPtrX[node_index];
                        splineCoeffY[i] = cpPtrY[node_index];
                        splineCoeffZ[i] = cpPtrZ[node_index];
                     }
                     else{
                        splineCoeffX[i] = 0.f;
                        splineCoeffY[i] = 0.f;
                        splineCoeffZ[i] = 0.f;
                     }
                     ++i;
                  } // a
               } // b
            } // c
            // Set the central control point to no displacement
            splineCoeffX[13] = 0.f;
            splineCoeffY[13] = 0.f;
            splineCoeffZ[13] = 0.f;
            // Compute the second derivative without the central control point
            float XX_x=0.0, YY_x=0.0, ZZ_x=0.0;
            float XY_x=0.0, YZ_x=0.0, XZ_x=0.0;
            float XX_y=0.0, YY_y=0.0, ZZ_y=0.0;
            float XY_y=0.0, YZ_y=0.0, XZ_y=0.0;
            float XX_z=0.0, YY_z=0.0, ZZ_z=0.0;
            float XY_z=0.0, YZ_z=0.0, XZ_z=0.0;
            for(i=0; i<27; i++){
               XX_x += basisXX[i]*splineCoeffX[i];
               YY_x += basisYY[i]*splineCoeffX[i];
               ZZ_x += basisZZ[i]*splineCoeffX[i];
               XY_x += basisXY[i]*splineCoeffX[i];
               YZ_x += basisYZ[i]*splineCoeffX[i];
               XZ_x += basisXZ[i]*splineCoeffX[i];

               XX_y += basisXX[i]*splineCoeffY[i];
               YY_y += basisYY[i]*splineCoeffY[i];
               ZZ_y += basisZZ[i]*splineCoeffY[i];
               XY_y += basisXY[i]*splineCoeffY[i];
               YZ_y += basisYZ[i]*splineCoeffY[i];
               XZ_y += basisXZ[i]*splineCoeffY[i];

               XX_z += basisXX[i]*splineCoeffZ[i];
               YY_z += basisYY[i]*splineCoeffZ[i];
               ZZ_z += basisZZ[i]*splineCoeffZ[i];
               XY_z += basisXY[i]*splineCoeffZ[i];
               YZ_z += basisYZ[i]*splineCoeffZ[i];
               XZ_z += basisXZ[i]*splineCoeffZ[i];
            }
            float *_discrete_values_mm_x = this->discrete_values_mm[0];
            float *_discrete_values_mm_y = this->discrete_values_mm[1];
            float *_discrete_values_mm_z = this->discrete_values_mm[2];
            for(int label=0; label<this->label_nD_num; ++label){

               float valX = inputCpPtrX[node] + *_discrete_values_mm_x++;
               float valY = inputCpPtrY[node] + *_discrete_values_mm_y++;
               float valZ = inputCpPtrZ[node] + *_discrete_values_mm_z++;

               size_t measure_index = node * this->label_nD_num + label;
               this->regularised_measures[measure_index] =
                     (1.f-this->regularisation_weight-this->l2_weight) * this->discretised_measures[measure_index] -
                     this->regularisation_weight * (
                     reg_pow2(XX_x + valX * _basisXX) +
                     reg_pow2(XX_y + valY * _basisXX) +
                     reg_pow2(XX_z + valZ * _basisXX) +
                     reg_pow2(YY_x + valX * _basisYY) +
                     reg_pow2(YY_y + valY * _basisYY) +
                     reg_pow2(YY_z + valZ * _basisYY) +
                     reg_pow2(ZZ_x + valX * _basisZZ) +
                     reg_pow2(ZZ_y + valY * _basisZZ) +
                     reg_pow2(ZZ_z + valZ * _basisZZ) + 2.0 * (
                     reg_pow2(XY_x + valX * _basisXY) +
                     reg_pow2(XY_y + valY * _basisXY) +
                     reg_pow2(XY_z + valZ * _basisXY) +
                     reg_pow2(XZ_x + valX * _basisXZ) +
                     reg_pow2(XZ_y + valY * _basisXZ) +
                     reg_pow2(XZ_z + valZ * _basisXZ) +
                     reg_pow2(YZ_x + valX * _basisYZ) +
                     reg_pow2(YZ_y + valY * _basisYZ) +
                     reg_pow2(YZ_z + valZ * _basisYZ)
                     ) ) - this->l2_weight * this->l2_penalisation[label];
            } // label
            ++node;
         } // x
      } // y
   } // z
   reg_getDeformationFromDisplacement(this->controlPointImage);
   reg_getDeformationFromDisplacement(this->input_transformation);
#ifndef NDEBUG
   reg_print_msg_debug("reg_discrete_init::GetRegularisedMeasure done");
#endif
}
/*****************************************************/
/*****************************************************/
void reg_discrete_init::Run()
{
   char text[255];
   sprintf(text, "Control point number = %lu", this->node_number);
   reg_print_info("reg_discrete_init", text);
   sprintf(text, "Discretised radius (voxel) = %i", this->discrete_radius);
   reg_print_info("reg_discrete_init", text);
   sprintf(text, "Discretised step (voxel) = %i", this->discrete_increment);
   reg_print_info("reg_discrete_init", text);
   sprintf(text, "Discretised label number = %i", this->label_nD_num);
   reg_print_info("reg_discrete_init", text);
   // Store the intial transformation parametrisation
   memcpy(this->input_transformation->data, this->controlPointImage->data,
          this->node_number*this->image_dim*sizeof(float));
   // Compute the discretised data term values
   this->GetDiscretisedMeasure();
   // Add the l2 regularisation
   //this->AddL2Penalisation(1.e-10f);
   // Initialise the regularise with the measure only
   memcpy(this->regularised_measures,
          this->discretised_measures,
          this->label_nD_num*this->node_number*sizeof(float));
   // Extract the best label
   this->getOptimalLabel();
   // Update the control point positions
   this->UpdateTransformation();
   // Run the regularisation optimisation
   for(int i=0; i< this->reg_max_it; ++i){
      this->GetRegularisedMeasure();
      this->getOptimalLabel();
      this->UpdateTransformation();
      sprintf(text, "Regularisation %i/%i - BE=%.2f - [%2.2f%%]",
             i+1, this->reg_max_it,
             reg_spline_approxBendingEnergy(this->controlPointImage),
             100.f*(float)this->regularisation_convergence/this->node_number);
      reg_print_info("reg_discrete_init", text);
      //if(this->regularisation_convergence<this->node_number/100)
      //   break;
   }
#ifndef NDEBUG
   reg_print_msg_debug("reg_discrete_init::Run done");
#endif
}
/*****************************************************/
/*****************************************************/
