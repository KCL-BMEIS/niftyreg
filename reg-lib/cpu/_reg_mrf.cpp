#include "_reg_mrf.h"


/*****************************************************/
reg_mrf::reg_mrf(reg_measure *_measure,
                 nifti_image *_controlPointImage,
                 int _discrete_radius,
                 int _discrete_increment,
                 float _reg_weight)
{
   this->measure = _measure;
   this->controlPointImage = _controlPointImage;
   this->discrete_radius = _discrete_radius;
   this->discrete_increment = _discrete_increment;
   this->regularisation_weight = _reg_weight;

   // Allocate the discretised value result
   int discrete_value = (this->discrete_radius / this->discrete_increment ) * 2 + 1;
   int controlPointNumber = this->controlPointImage->nx *
         this->controlPointImage->ny * this->controlPointImage->nz;
   this->discretised_measure = (float *)malloc(controlPointNumber*discrete_value*sizeof(float));

   // Allocate the arrays to store the graph
   this->edgeWeightMatrix = (float *)malloc(this->controlPointImage->nvox*2*sizeof(float));
   this->index_neighbours = (float *)malloc(this->controlPointImage->nvox*2*sizeof(float));

   this->initialised = false;
}
/*****************************************************/
reg_mrf::~reg_mrf()
{
   if(this->discretised_measure!=NULL)
      free(this->discretised_measure);
   this->discretised_measure=NULL;
   if(this->edgeWeightMatrix!=NULL)
      free(this->edgeWeightMatrix);
   this->edgeWeightMatrix=NULL;
   if(this->edgeWeightMatrix!=NULL)
      free(this->edgeWeightMatrix);
   this->edgeWeightMatrix=NULL;
}
/*****************************************************/
void reg_mrf::Initialise()
{
   // Create the minimum spamming tree
   reg_print_msg_error("Need to implement reg_mrf::Initialise()");
   reg_exit();
   this->initialised = true;
}
/*****************************************************/
void reg_mrf::GetDiscretisedMeasure()
{
   measure->GetDiscretisedValue(this->controlPointImage,
                                this->discretised_measure,
                                this->discrete_radius,
                                this->discrete_increment);
}
/*****************************************************/
void reg_mrf::Optimise()
{
   // Run the optimisation and update the transformation
   reg_print_msg_error("Need to implement reg_mrf::Optimise()");
   reg_exit();
}
/*****************************************************/
void reg_mrf::Run()
{
   if(this->initialised==false)
      this->Initialise();
   this->GetDiscretisedMeasure();
   this->Optimise();
}
/*****************************************************/
