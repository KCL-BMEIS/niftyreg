#include "_reg_discrete_continuous.h"

/*****************************************************/
reg_discrete_continuous::reg_discrete_continuous(reg_measure *_measure,
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


   //To store the cost data term - originaly SAD between images.
   this->discretised_measures = (float *)malloc(this->node_number*this->label_nD_num*sizeof(float));

   //regulatization - optimization
   this->optimal_label_index=(int *)malloc(this->node_number*sizeof(int));

   // Allocate stuff
   ///TODO Ben to complete the comment here
   this->optimal_measure_transformation = nifti_copy_nim_info(this->controlPointImage);
   this->optimal_measure_transformation->data = (void *)malloc(this->optimal_measure_transformation->nvox*
                                                               this->optimal_measure_transformation->nbyper);
   this->regularisation_gradient = nifti_copy_nim_info(this->controlPointImage);
   this->regularisation_gradient->data = (void *)malloc(this->regularisation_gradient->nvox*
                                                        this->regularisation_gradient->nbyper);

   this->optimiser = new reg_conjugateGradient<float>();
}
/*****************************************************/
/*****************************************************/
reg_discrete_continuous::~reg_discrete_continuous()
{
   if(this->discretised_measures!=NULL)
      free(this->discretised_measures);
   this->discretised_measures=NULL;

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

   if(this->optimal_measure_transformation!=NULL)
      nifti_image_free(this->optimal_measure_transformation);
   this->optimal_measure_transformation=NULL;

   if(this->regularisation_gradient!=NULL)
      nifti_image_free(this->regularisation_gradient);
   this->regularisation_gradient=NULL;

   if(this->optimiser!=NULL)
      delete this->optimiser;
}
/*****************************************************/
/*****************************************************/
void reg_discrete_continuous::GetDiscretisedMeasure()
{
   measure->GetDiscretisedValue(this->controlPointImage,
                                this->discretised_measures,
                                this->discrete_radius,
                                this->discrete_increment,
                                (1.f-this->regularisation_weight));
#ifndef NDEBUG
   reg_print_msg_debug("reg_discrete_continuous::GetDiscretisedMeasure done");
#endif
}
/*****************************************************/
/*****************************************************/
void reg_discrete_continuous::getOptimalLabel()
{
   for(int node=0; node<this->node_number; ++node)
      this->optimal_label_index[node]=
         std::min_element(this->discretised_measures+node*this->label_nD_num,
                          this->discretised_measures+(node+1)*this->label_nD_num) -
         (this->discretised_measures+node*this->label_nD_num);
}
/*****************************************************/
/*****************************************************/
void reg_discrete_continuous::StoreOptimalMeasureTransformation()
{
   //Update the control point position
   float *cpPtrX = static_cast<float *>(this->controlPointImage->data);
   float *cpPtrY = &cpPtrX[this->node_number];
   float *cpPtrZ = &cpPtrY[this->node_number];
   //Update the control point position
   float *optimalPtrX = static_cast<float *>(this->optimal_measure_transformation->data);
   float *optimalPtrY = &optimalPtrX[this->node_number];
   float *optimalPtrZ = &optimalPtrY[this->node_number];

   size_t voxel=0;
   for(int z=0; z<this->controlPointImage->nz; z++) {
      for(int y=0; y<this->controlPointImage->ny; y++) {
         for(int x=0; x<this->controlPointImage->nx; x++) {
            int optimal_id = this->optimal_label_index[voxel];
            cpPtrX[voxel] += this->discrete_values_mm[0][optimal_id];
            cpPtrY[voxel] += this->discrete_values_mm[1][optimal_id];
            cpPtrZ[voxel] += this->discrete_values_mm[2][optimal_id];
            optimalPtrX[voxel] = cpPtrX[voxel];
            optimalPtrY[voxel] = cpPtrY[voxel];
            optimalPtrZ[voxel] = cpPtrZ[voxel];
            ++voxel;
         }
      }
   }
#ifndef NDEBUG
   reg_print_msg_debug("reg_discrete_continuous::StoreOptimalMeasureTransformation done");
#endif
}
/*****************************************************/
/*****************************************************/
double reg_discrete_continuous::GetObjectiveFunctionValue()
{
   // Compute the bending energy value
   double bending_energy=reg_spline_approxBendingEnergy(this->controlPointImage); ///TODO. needs to account the interpolant spline here
   // Compute the linear elasticity value
//   double linear_elasticity=reg_spline_approxLinearEnergy(this->controlPointImage); ///TODO. needs to account the interpolant spline here
   // Compute the L2 norm between the measure based control point grid and the current transformation
   double l2_norm_between_images=0.f;

   float *cpPtrX = static_cast<float *>(this->controlPointImage->data);
   float *cpPtrY = &cpPtrX[this->node_number];
   float *cpPtrZ = &cpPtrY[this->node_number];
   float *optimalPtrX = static_cast<float *>(this->optimal_measure_transformation->data);
   float *optimalPtrY = &optimalPtrX[this->node_number];
   float *optimalPtrZ = &optimalPtrY[this->node_number];
   int voxel=0;
   for(int z=0;z<this->controlPointImage->nz;++z){
      for(int y=0;y<this->controlPointImage->ny;++y){
         for(int x=0;x<this->controlPointImage->nx;++x){
            float val_cpp_x = cpPtrX[voxel];
            float val_cpp_y = cpPtrY[voxel];
            float val_cpp_z = cpPtrZ[voxel];
            float val_mea_x = optimalPtrX[voxel];
            float val_mea_y = optimalPtrY[voxel];
            float val_mea_z = optimalPtrZ[voxel];
            l2_norm_between_images += reg_pow2(val_cpp_x - val_mea_x);
            l2_norm_between_images += reg_pow2(val_cpp_y - val_mea_y);
            l2_norm_between_images += reg_pow2(val_cpp_z - val_mea_z);
            ++voxel;
         }
      }
   }
   printf("%g\n", bending_energy - l2_norm_between_images/this->node_number);
   return -bending_energy - l2_norm_between_images/this->node_number;
}
/*****************************************************/
void reg_discrete_continuous::GetObjectiveFunctionGradient()
{
   // Empty the gradient image
   memset(this->regularisation_gradient->data,
          0,
          this->regularisation_gradient->nvox*
          this->regularisation_gradient->nbyper);
   // Compute the bending energy gradient
   reg_spline_approxBendingEnergyGradient(this->controlPointImage,
                                          this->regularisation_gradient,
                                          this->regularisation_weight); ///TODO. needs to account for interpolant spline here
   // Compute the linear elasticity gradient
//   reg_spline_approxLinearEnergyGradient(this->controlPointImage,
//                                         this->regularisation_gradient,
//                                         this->regularisation_weight); ///TODO. needs to account for interpolant spline here
   // Compute the gradient of the L2 norm
   float *cpPtrX = static_cast<float *>(this->controlPointImage->data);
   float *cpPtrY = &cpPtrX[this->node_number];
   float *cpPtrZ = &cpPtrY[this->node_number];
   float *optimalPtrX = static_cast<float *>(this->optimal_measure_transformation->data);
   float *optimalPtrY = &optimalPtrX[this->node_number];
   float *optimalPtrZ = &optimalPtrY[this->node_number];
   float *gradPtrX = static_cast<float *>(this->regularisation_gradient->data);
   float *gradPtrY = &gradPtrX[this->node_number];
   float *gradPtrZ = &gradPtrY[this->node_number];
   int voxel=0;
   for(int z=0;z<this->controlPointImage->nz;++z){
      for(int y=0;y<this->controlPointImage->ny;++y){
         for(int x=0;x<this->controlPointImage->nx;++x){
            float val_cpp_x = cpPtrX[voxel];
            float val_cpp_y = cpPtrY[voxel];
            float val_cpp_z = cpPtrZ[voxel];
            float val_mea_x = optimalPtrX[voxel];
            float val_mea_y = optimalPtrY[voxel];
            float val_mea_z = optimalPtrZ[voxel];
            gradPtrX[voxel] += this->regularisation_weight * 2.f * (val_cpp_x - val_mea_x);
            gradPtrY[voxel] += this->regularisation_weight * 2.f * (val_cpp_y - val_mea_y);
            gradPtrZ[voxel] += this->regularisation_weight * 2.f * (val_cpp_z - val_mea_z);
            ++voxel;
         }
      }
   }
}
/*****************************************************/
void reg_discrete_continuous::UpdateParameters(float scale)
{
   float *currentDOF=this->optimiser->GetCurrentDOF();
   float *bestDOF=this->optimiser->GetBestDOF();
   float *gradient=this->optimiser->GetGradient();

   // Update the values for all axis displacement
   for(size_t i=0; i<this->node_number*this->image_dim; ++i)
   {
      currentDOF[i] = bestDOF[i] + scale * gradient[i];
   }
#ifndef NDEBUG
   reg_print_fct_debug("reg_discrete_continuous::UpdateParameters");
#endif
}
/*****************************************************/
void reg_discrete_continuous::UpdateBestObjFunctionValue()
{
   return;
}
/*****************************************************/
void reg_discrete_continuous::ContinuousRegularisation()
{
   // Create an optimiser
   this->optimiser->Initialise(this->node_number*this->image_dim,
                               this->image_dim,
                               true,
                               true,
                               true,
                               100,
                               0,
                               this,
                               static_cast<float *>(this->controlPointImage->data),
                               static_cast<float *>(this->regularisation_gradient->data));
   // Set up variables used during the gradient descent
   // Set the initial step size for the gradient ascent
   float maxStepSize = this->referenceImage->dx>this->referenceImage->dy?this->referenceImage->dx:this->referenceImage->dy;
   if(this->image_dim)
      maxStepSize = (this->referenceImage->dz>maxStepSize)?this->referenceImage->dz:maxStepSize;
   float currentSize = maxStepSize;
   float smallestSize = currentSize / 1000.f;
   // Optimise the current cost function
   while(true)
   {
      if(currentSize==0)
         break;

      if(optimiser->GetCurrentIterationNumber()>=optimiser->GetMaxIterationNumber()){
         reg_print_msg_warn("The current level reached the maximum number of iteration");
         break;
      }

      // Compute the objective function gradient
      this->GetObjectiveFunctionGradient();

      // Normalise the gradient
      float maxGradValue=0;
      if(this->regularisation_gradient->nz>1)
      {
         float *gradPtrX = static_cast<float *>(this->regularisation_gradient->data);
         float *gradPtrY = &gradPtrX[this->node_number];
         float *gradPtrZ = &gradPtrY[this->node_number];
         for(size_t i=0; i<this->node_number; i++)
         {
            float valX = *gradPtrX++;
            float valY = *gradPtrY++;
            float valZ = *gradPtrZ++;
            float length = (float)(sqrt(valX*valX + valY*valY + valZ*valZ));
            maxGradValue = (length>maxGradValue)?length:maxGradValue;
         }
         gradPtrX = static_cast<float *>(this->regularisation_gradient->data);
         gradPtrY = &gradPtrX[this->node_number];
         gradPtrZ = &gradPtrY[this->node_number];
         for(size_t i=0; i<this->node_number; i++)
         {
            *gradPtrX++ /= maxGradValue;
            *gradPtrY++ /= maxGradValue;
            *gradPtrZ++ /= maxGradValue;
         }
      }
      else
      {
         float *gradPtrX = static_cast<float *>(this->regularisation_gradient->data);
         float *gradPtrY = &gradPtrX[this->node_number];
         for(size_t i=0; i<this->node_number; i++)
         {
            float valX = *gradPtrX++;
            float valY = *gradPtrY++;
            float length = (float)(sqrt(valX*valX + valY*valY));
            maxGradValue = (length>maxGradValue)?length:maxGradValue;
         }
         gradPtrX = static_cast<float *>(this->regularisation_gradient->data);
         gradPtrY = &gradPtrX[this->node_number];
         for(size_t i=0; i<this->node_number; i++)
         {
            *gradPtrX++ /= maxGradValue;
            *gradPtrY++ /= maxGradValue;
         }
      }

      // Initialise the line search initial step size
      currentSize=currentSize>maxStepSize?maxStepSize:currentSize;

      // A line search is performed
      optimiser->Optimise(maxStepSize,smallestSize,currentSize);

   } // while

}
/*****************************************************/
/*****************************************************/
void reg_discrete_continuous::Run()
{
   // Compute the discretised data term values
   this->GetDiscretisedMeasure();
   // Extract the best label
   this->getOptimalLabel();
   // Update the control point positions
   this->StoreOptimalMeasureTransformation();
   // Run the regularisation optimisation
   this->ContinuousRegularisation();
}
/*****************************************************/
/*****************************************************/
