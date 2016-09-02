#include "F3DContent.h"

F3DContent::F3DContent(int platformCodeIn, int refTime, int floTime) : GlobalContent(platformCodeIn, refTime, floTime)
{
#ifdef BUILD_DEV
    this->linearSpline = false;
#endif
    this->spacing = new float[3];
    this->spacing[0]=-5;
    this->spacing[1]=std::numeric_limits<float>::quiet_NaN();
    this->spacing[2]=std::numeric_limits<float>::quiet_NaN();
    this->inputControlPointGrid = NULL;
    this->currentControlPointGrid = NULL;
}
F3DContent::~F3DContent()
{
    //if(this->inputControlPointGrid != NULL)
    //    nifti_image_free(this->inputControlPointGrid);

    if(this->currentControlPointGrid != NULL)
        nifti_image_free(this->currentControlPointGrid);

    if(this->spacing != NULL)
        delete[] spacing;
}
/* *************************************************************** */
void F3DContent::setInputControlPointGrid(nifti_image* cpg)
{
    this->inputControlPointGrid = cpg;
}
/* *************************************************************** */
nifti_image* F3DContent::getInputControlPointGrid()
{
    return this->inputControlPointGrid;
}
/* *************************************************************** */
void F3DContent::setSpacing(unsigned int i, float s)
{
    this->spacing[i] = s;
}
#ifdef BUILD_DEV
/* *************************************************************** */
bool F3DContent::getLinearSpline()
{
    return this->linearSpline;
}
/* *************************************************************** */
void F3DContent::setLinearSpline(bool ls)
{
    this->linearSpline = ls;
}
#endif
/* *************************************************************** */
float* F3DContent::getSpacing()
{
    return this->spacing;
}
/* *************************************************************** */
/* *************************************************************** */
void F3DContent::setCurrentControlPointGrid(nifti_image* cpg)
{
    this->currentControlPointGrid = cpg;
}
/* *************************************************************** */
nifti_image* F3DContent::getCurrentControlPointGrid()
{
    return this->currentControlPointGrid;
}
/* *************************************************************** */
void F3DContent::AllocateControlPointGrid(float* gridSpacing)
{
    if(this->referencePyramid[0]==NULL) {
       reg_print_fct_error("F3DContent::AllocateControlPointGrid(gridSpacing)");
       reg_print_msg_error("The reference image is not defined");
       reg_exit();
    }
    F3DContent::ClearControlPointGrid();
    reg_createControlPointGrid<float>(&(this->currentControlPointGrid),
                                  this->referencePyramid[0],
                                  gridSpacing);
#ifndef NDEBUG
   reg_print_fct_debug("F3DContent::AllocateControlPointGrid");
#endif
}
/* *************************************************************** */
void F3DContent::AllocateControlPointGrid()
{
    if(this->inputControlPointGrid==NULL) {
       reg_print_fct_error("F3DContent::AllocateControlPointGrid()");
       reg_print_msg_error("The inputControlPointGrid is not defined");
       reg_exit();
    }
    F3DContent::ClearControlPointGrid();
    this->currentControlPointGrid = nifti_copy_nim_info(this->inputControlPointGrid);
    this->currentControlPointGrid->data = (void *)malloc(this->currentControlPointGrid->nvox *
                                                         this->currentControlPointGrid->nbyper);

    if(this->inputControlPointGrid->num_ext>0)
       nifti_copy_extensions(this->currentControlPointGrid,this->inputControlPointGrid);
    memcpy( this->currentControlPointGrid->data, this->inputControlPointGrid->data,
            this->currentControlPointGrid->nvox * this->currentControlPointGrid->nbyper);
    // The final grid spacing is computed
    this->spacing[0] = this->currentControlPointGrid->dx / powf(2.0f, (float)(this->getLevelNumber()-1));
    this->spacing[1] = this->currentControlPointGrid->dy / powf(2.0f, (float)(this->getLevelNumber()-1));
    if(this->currentControlPointGrid->nz>1) {
        this->spacing[2] = this->currentControlPointGrid->dz / powf(2.0f, (float)(this->getLevelNumber()-1));
    }

#ifdef BUILD_DEV
 if(this->linearSpline)
    this->currentControlPointGrid->intent_p1=LIN_SPLINE_GRID;
#endif

#ifndef NDEBUG
   reg_print_fct_debug("F3DContent::AllocateControlPointGrid");
#endif
}
/* *************************************************************** */
void F3DContent::ClearControlPointGrid()
{
   if(this->currentControlPointGrid!=NULL) {
       nifti_image_free(this->currentControlPointGrid);
       this->currentControlPointGrid=NULL;
   }
#ifndef NDEBUG
   reg_print_fct_debug("F3DContent::ClearControlPointGrid");
#endif
}
