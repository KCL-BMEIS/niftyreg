#include "_reg_discrete_init.h"

/*****************************************************/
reg_discrete_init::reg_discrete_init(reg_measure *_measure,
                                     nifti_image *_referenceImage,
                                     nifti_image *_controlPointImage,
                                     int _discrete_radius,
                                     int _discrete_increment,
                                     float _reg_weight)
{
   this->measure = _measure;
   this->referenceImage = _referenceImage;
   this->controlPointImage = _controlPointImage;
   this->discrete_radius = _discrete_radius;
   this->discrete_increment = _discrete_increment;
   this->regularisation_weight = _reg_weight;

   this->image_dim = this->referenceImage->nz > 1 ? 3 :2;
   this->label_1D_num = (this->discrete_radius / this->discrete_increment ) * 2 + 1;
   this->label_nD_num = std::pow(this->label_1D_num,this->image_dim);
   this->node_number = (size_t)this->controlPointImage->nx *
         this->controlPointImage->ny * this->controlPointImage->nz;

   this->input_transformation = (float *)malloc(this->node_number*this->image_dim*sizeof(float));

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

   //To store the cost data term
   this->discretised_measures = (float *)calloc(this->node_number*this->label_nD_num,sizeof(float));

   //Optimal transformation based on the data term
   this->regularised_measures = (float *)malloc(this->node_number*this->label_nD_num*sizeof(float));
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
      free(this->input_transformation);
   this->input_transformation=NULL;
}
/*****************************************************/
/*****************************************************/
void reg_discrete_init::GetDiscretisedMeasure()
{
   measure->GetDiscretisedValue(this->controlPointImage,
                                this->discretised_measures,
                                this->discrete_radius,
                                this->discrete_increment,
                                (1.f-this->regularisation_weight));
#ifndef NDEBUG
   reg_print_msg_debug("reg_discrete_init::GetDiscretisedMeasure done");
#endif
}
/*****************************************************/
/*****************************************************/
void reg_discrete_init::getOptimalLabel()
{
   for(int node=0; node<this->node_number; ++node)
      this->optimal_label_index[node] =
         std::min_element(this->regularised_measures+node*this->label_nD_num,
                          this->regularised_measures+(node+1)*this->label_nD_num) -
         (this->regularised_measures+node*this->label_nD_num);
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

   float *inputCpPtrX = this->input_transformation;
   float *inputCpPtrY = &inputCpPtrX[this->node_number];
   float *inputCpPtrZ = &inputCpPtrY[this->node_number];

   for(int z=1; z<this->controlPointImage->nz-1; z++) {
      for(int y=1; y<this->controlPointImage->ny-1; y++) {
         size_t node = (z*this->controlPointImage->ny+y)*this->controlPointImage->nx+1;
         for(int x=1; x<this->controlPointImage->nx-1; x++){
            int optimal_id = this->optimal_label_index[node];
            cpPtrX[node] = inputCpPtrX[node] + this->discrete_values_mm[0][optimal_id];
            cpPtrY[node] = inputCpPtrY[node] + this->discrete_values_mm[1][optimal_id];
            cpPtrZ[node] = inputCpPtrZ[node] + this->discrete_values_mm[2][optimal_id];
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
void reg_discrete_init::GetRegularisedMeasure()
{
   float *cpPtrX = static_cast<float *>(this->controlPointImage->data);
   float *cpPtrY = &cpPtrX[this->node_number];
   float *cpPtrZ = &cpPtrY[this->node_number];

   float *inputCpPtrX = this->input_transformation;
   float *inputCpPtrY = &inputCpPtrX[this->node_number];
   float *inputCpPtrZ = &inputCpPtrY[this->node_number];

   int node_coord[3];
   for(int z=1; z<this->controlPointImage->nz-1; z++) {
      node_coord[2]=z;
      for(int y=1; y<this->controlPointImage->ny-1; y++) {
         node_coord[1]=y;
         size_t node = (z*this->controlPointImage->ny+y)*this->controlPointImage->nx+1;
         for(int x=1; x<this->controlPointImage->nx-1; x++){
            node_coord[0]=x;
            // Store the initial position
            float position_x = cpPtrX[node];
            float position_y = cpPtrY[node];
            float position_z = cpPtrZ[node];
            for(int label=0; label<this->label_nD_num; ++label){
               // Update the control point position
               cpPtrX[node] = inputCpPtrX[node] + this->discrete_values_mm[0][label];
               cpPtrY[node] = inputCpPtrY[node] + this->discrete_values_mm[1][label];
               cpPtrZ[node] = inputCpPtrZ[node] + this->discrete_values_mm[2][label];
               size_t measure_index = node * this->label_nD_num + label;
               this->regularised_measures[measure_index] = this->discretised_measures[measure_index] +
                     100000.f*this->regularisation_weight * reg_spline_singlePointBendingEnergy(this->controlPointImage,
                                                                                       node_coord);
            }
            cpPtrX[node] = position_x;
            cpPtrY[node] = position_y;
            cpPtrZ[node] = position_z;
            ++node;
         }
      }
   }
#ifndef NDEBUG
   reg_print_msg_debug("reg_discrete_init::GetRegularisedMeasure done");
#endif
}
/*****************************************************/
/*****************************************************/
void reg_discrete_init::Run()
{
   // Store the intial transformation parametrisation
   memcpy(this->input_transformation, this->controlPointImage->data,
          this->node_number*this->image_dim*sizeof(float));
   // Compute the discretised data term values
   this->GetDiscretisedMeasure();
   // Initialise the regularise with the measure only
   memcpy(this->regularised_measures,
          this->discretised_measures,
          this->label_nD_num*this->node_number*sizeof(float));
   // Extract the best label
   this->getOptimalLabel();
   // Update the control point positions
   this->UpdateTransformation();
   // Run the regularisation optimisation
   for(int i=0; i< 10; ++i){
      this->GetRegularisedMeasure();
      this->getOptimalLabel();
      this->UpdateTransformation();
   }
#ifndef NDEBUG
   reg_print_msg_debug("reg_discrete_init::Run done");
#endif
}
/*****************************************************/
/*****************************************************/
