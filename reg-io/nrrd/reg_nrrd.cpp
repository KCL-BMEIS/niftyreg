/*
 *  reg_nrrd.cpp
 *
 *
 *  Created by Marc Modat on 30/05/2012.
 *  Copyright (c) 2012, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_NRRD_CPP
#define _REG_NRRD_CPP

#include "reg_nrrd.h"


/* *************************************************************** */
nifti_image *reg_io_nrdd2nifti(Nrrd *nrrdImage)
{
    // Check if the file can be converted
    if(nrrdImage->dim>7){
        fprintf(stderr, "[NiftyReg ERROR] reg_io_nrdd2nifti - The Nifti format only support 7 dimensions\n");
        exit(1);
    }

    // Need first to extract the input image dimension
    int dim[8]={1,1,1,1,1,1,1,1};

    dim[0]=nrrdImage->dim;
    for(int i=0;i<(dim[0]<7?dim[0]:7);++i){
        dim[i+1]=(int)nrrdImage->axis[i].size;
    }

    // The nifti_image pointer is created
    nifti_image *niiImage=NULL;

    // The nifti image is generated based on the nrrd image datatype
    switch(nrrdImage->type){
    case nrrdTypeUChar:
        niiImage=nifti_make_new_nim(dim,NIFTI_TYPE_UINT8,true);
        break;
    case nrrdTypeChar:
        niiImage=nifti_make_new_nim(dim,NIFTI_TYPE_INT8,true);
        break;
    case nrrdTypeUShort:
        niiImage=nifti_make_new_nim(dim,NIFTI_TYPE_UINT16,true);
        break;
    case nrrdTypeShort:
        niiImage=nifti_make_new_nim(dim,NIFTI_TYPE_INT16,true);
        break;
    case nrrdTypeUInt:
        niiImage=nifti_make_new_nim(dim,NIFTI_TYPE_UINT32,true);
        break;
    case nrrdTypeInt:
        niiImage=nifti_make_new_nim(dim,NIFTI_TYPE_INT32,true);
        break;
    case nrrdTypeFloat:
        niiImage=nifti_make_new_nim(dim,NIFTI_TYPE_FLOAT32,true);
        break;
    case nrrdTypeDouble:
        niiImage=nifti_make_new_nim(dim,NIFTI_TYPE_FLOAT64,true);
        break;
    default:
        fprintf(stderr, "[NiftyReg ERROR] reg_io_nrdd2nifti - The data type is not supported\n");
        exit(1);
    }

    // The data are copied over from the nrrd to the nifti structure
    memcpy(niiImage->data, nrrdImage->data, niiImage->nvox*niiImage->nbyper);

    // We set the spacing information for every axis
    if(niiImage->ndim>=1) niiImage->dx=niiImage->pixdim[1]=nrrdImage->axis[0].spacing;
    if(niiImage->ndim>=2) niiImage->dy=niiImage->pixdim[2]=nrrdImage->axis[1].spacing;
    if(niiImage->ndim>=3) niiImage->dz=niiImage->pixdim[3]=nrrdImage->axis[2].spacing;
    if(niiImage->ndim>=4) niiImage->dt=niiImage->pixdim[4]=nrrdImage->axis[3].spacing;
    if(niiImage->ndim>=5) niiImage->du=niiImage->pixdim[5]=nrrdImage->axis[4].spacing;
    if(niiImage->ndim>=6) niiImage->dv=niiImage->pixdim[6]=nrrdImage->axis[5].spacing;
    if(niiImage->ndim>=7) niiImage->dw=niiImage->pixdim[7]=nrrdImage->axis[6].spacing;

    // Set the slope and intersection
    niiImage->scl_inter=0;
    niiImage->scl_slope=1;

    // Set the min and max intensities
    niiImage->cal_min=reg_tools_getMinValue(niiImage);
    niiImage->cal_max=reg_tools_getMaxValue(niiImage);


    // The space orientation is extracted and converted into a matrix
    mat44 qform_orientation_matrix;
    reg_mat44_eye(&qform_orientation_matrix);
    if(     nrrdImage->space==nrrdSpaceRightAnteriorSuperior ||
            nrrdImage->space==nrrdSpaceRightAnteriorSuperiorTime ||
            nrrdImage->space==nrrdSpace3DRightHanded ||
            nrrdImage->space==nrrdSpace3DRightHandedTime ){
        qform_orientation_matrix.m[0][0]=1.f; // NIFTI_L2R
        qform_orientation_matrix.m[1][1]=1.f; // NIFTI_P2A
        qform_orientation_matrix.m[2][2]=1.f; // NIFTI_I2S
    }
    else if(nrrdImage->space==nrrdSpaceLeftAnteriorSuperior ||
            nrrdImage->space==nrrdSpaceLeftAnteriorSuperiorTime ||
            nrrdImage->space==nrrdSpace3DLeftHanded ||
            nrrdImage->space==nrrdSpace3DLeftHandedTime ){
        qform_orientation_matrix.m[0][0]=-1.f;  //NIFTI_R2L
        qform_orientation_matrix.m[1][1]=1.f; // NIFTI_P2A
        qform_orientation_matrix.m[2][2]=1.f; // NIFTI_I2S
    }
    else if(nrrdImage->space!=nrrdSpaceScannerXYZ &&
            nrrdImage->space!=nrrdSpaceScannerXYZTime ){
        fprintf(stderr, "[NiftyReg WARNING] reg_io_nrdd2nifti - nrrd space value unrecognised: the Nifti qform is set to identity\n");
    }

    // The origin is set
    if(niiImage->ndim>=1)
        qform_orientation_matrix.m[0][3]=niiImage->qoffset_x=nrrdImage->spaceOrigin[0];
    if(niiImage->ndim>=2)
        qform_orientation_matrix.m[1][3]=niiImage->qoffset_y=nrrdImage->spaceOrigin[1];
    if(niiImage->ndim>=3)
        qform_orientation_matrix.m[2][3]=niiImage->qoffset_z=nrrdImage->spaceOrigin[2];

    // Extract the quaternions and qfac values
    nifti_mat44_to_quatern(qform_orientation_matrix,
                           &niiImage->quatern_b,
                           &niiImage->quatern_c,
                           &niiImage->quatern_d,
                           &niiImage->qoffset_x,
                           &niiImage->qoffset_y,
                           &niiImage->qoffset_z,
                           &niiImage->dx,
                           &niiImage->dy,
                           &niiImage->dz,
                           &niiImage->qfac);

    // Set the qform matrices
    niiImage->qto_xyz=nifti_quatern_to_mat44(niiImage->quatern_b,
                                             niiImage->quatern_c,
                                             niiImage->quatern_d,
                                             niiImage->qoffset_x,
                                             niiImage->qoffset_y,
                                             niiImage->qoffset_z,
                                             niiImage->dx,
                                             niiImage->dy,
                                             niiImage->dz,
                                             niiImage->qfac);
    niiImage->qto_ijk=nifti_mat44_inverse(niiImage->qto_xyz);

    // The sform has to be set if required
    // Check if the spaceDirection array is set
    if(nrrdImage->axis[0].spaceDirection[0]!=std::numeric_limits<double>::quiet_NaN()){
        niiImage->sform_code=1;
        reg_mat44_eye(&niiImage->sto_xyz);
        for(int i=0;i<(niiImage->ndim<3?niiImage->ndim:3);++i){
            for(int j=0;j<(niiImage->ndim<3?niiImage->ndim:3);++j){
                niiImage->sto_xyz.m[i][j]=(float)nrrdImage->axis[i].spaceDirection[j];
            }
            niiImage->sto_xyz.m[i][3]=niiImage->qto_xyz.m[i][3];
        }
        niiImage->sto_ijk=nifti_mat44_inverse(niiImage->sto_xyz);
    }

    // Set the space unit if it is defined
    if(nrrdImage->spaceUnits[0]!=NULL){
        if(strcmp(nrrdImage->spaceUnits[0],"m")==0)
            niiImage->xyz_units=NIFTI_UNITS_METER;
        else if(strcmp(nrrdImage->spaceUnits[0],"mm")==0)
            niiImage->xyz_units=NIFTI_UNITS_MM;
        else if(strcmp(nrrdImage->spaceUnits[0],"um")==0)
            niiImage->xyz_units=NIFTI_UNITS_MICRON;
    }

    // Set the time unit if it is defined
    if(nrrdImage->axis[3].size>1){
        if(nrrdImage->spaceUnits[4]!=NULL){
            if(strcmp(nrrdImage->spaceUnits[4],"sec"))
                niiImage->time_units=NIFTI_UNITS_SEC;
            else if(strcmp(nrrdImage->spaceUnits[4],"msec"))
                niiImage->time_units=NIFTI_UNITS_MSEC;
        }
    }

    // Check if the nrrd image was a NiftyReg velocity field parametrisation
    if(niiImage->nu>1){
        niiImage->intent_code=NIFTI_INTENT_VECTOR;
        // Check if the image is a stationary field from NiftyReg
        if(nrrdImage->axis[4].label!=NULL){
            std::string str=nrrdImage->axis[4].label;
            size_t it;
            if((it=str.find("NREG_VEL_STEP "))!=std::string::npos){
                str=str.substr(it+13);
                memset(niiImage->intent_name, 0, 16);
                strcpy(niiImage->intent_name,"NREG_VEL_STEP");
                niiImage->intent_p1=atof(str.c_str());
            }
            if(str.find("NREG_CPP_FILE")!=std::string::npos){
                memset(niiImage->intent_name, 0, 16);
                strcpy(niiImage->intent_name,"NREG_CPP_FILE");
            }
        }
    }

    // returns the new nii image
    return niiImage;
}
/* *************************************************************** */
Nrrd *reg_io_nifti2nrrd(nifti_image *niiImage)
{
    // Create a nrrd image
    Nrrd *nrrdImage = nrrdNew();

    // Set the nrrd image size
    size_t size[NRRD_DIM_MAX];
    for(int i=1;i<=niiImage->ndim;++i)
        size[i-1]=niiImage->dim[i];

    // Allocate the nrrd data array based on the nii image data type
    switch(niiImage->datatype){
    case NIFTI_TYPE_UINT8:
        nrrdAlloc_nva(nrrdImage,nrrdTypeUChar,niiImage->ndim,size);
        break;
    case NIFTI_TYPE_INT8:
        nrrdAlloc_nva(nrrdImage,nrrdTypeChar,niiImage->ndim,size);
        break;
    case NIFTI_TYPE_UINT16:
        nrrdAlloc_nva(nrrdImage,nrrdTypeUShort,niiImage->ndim,size);
        break;
    case NIFTI_TYPE_INT16:
        nrrdAlloc_nva(nrrdImage,nrrdTypeShort,niiImage->ndim,size);
        break;
    case NIFTI_TYPE_UINT32:
        nrrdAlloc_nva(nrrdImage,nrrdTypeUInt,niiImage->ndim,size);
        break;
    case NIFTI_TYPE_INT32:
        nrrdAlloc_nva(nrrdImage,nrrdTypeInt,niiImage->ndim,size);
        break;
    case NIFTI_TYPE_FLOAT32:
        nrrdAlloc_nva(nrrdImage,nrrdTypeFloat,niiImage->ndim,size);
        break;
    case NIFTI_TYPE_FLOAT64:
        nrrdAlloc_nva(nrrdImage,nrrdTypeDouble,niiImage->ndim,size);
        break;
    default:
        fprintf(stderr, "[NiftyReg ERROR] reg_io_nifti2nrrd - The data type is not supported\n");
        exit(1);
    }

    // Rescale the nii image intensity if required
    if(niiImage->scl_slope!=1 && niiImage->scl_slope!=0){
        reg_tools_addSubMulDivValue(niiImage, niiImage,niiImage->scl_slope,2); // *(niiImage->scl_slope)
        niiImage->scl_slope=1;
    }
    if(niiImage->scl_inter!=0){
        reg_tools_addSubMulDivValue(niiImage, niiImage,niiImage->scl_inter,0); // +(niiImage->scl_inter)
        niiImage->scl_inter=0;
    }

    // Copy the data from the nifti image to the nrrd image
    memcpy(nrrdImage->data, niiImage->data, niiImage->nvox*niiImage->nbyper);

    // Set the origin of the image
    nrrdImage->spaceDim=niiImage->ndim<3?niiImage->ndim:3;
    if(niiImage->sform_code>0){
        nrrdImage->spaceOrigin[0]=niiImage->sto_xyz.m[0][3];
        nrrdImage->spaceOrigin[1]=niiImage->sto_xyz.m[1][3];
        nrrdImage->spaceOrigin[2]=niiImage->sto_xyz.m[2][3];
    }
    else if(niiImage->qform_code>0){
        nrrdImage->spaceOrigin[0]=niiImage->qoffset_x;
        nrrdImage->spaceOrigin[1]=niiImage->qoffset_y;
        nrrdImage->spaceOrigin[2]=niiImage->qoffset_z;
    }
    else{
        nrrdImage->spaceOrigin[0]=0;
        nrrdImage->spaceOrigin[1]=0;
        nrrdImage->spaceOrigin[2]=0;
    }

    // Set the space if suitable with the nrrd file format
    if(niiImage->qform_code>0 || niiImage->sform_code>0){
        int i_orient, j_orient, k_orient;
        // Use the sform information if it is defined
        if(niiImage->sform_code>0)
            nifti_mat44_to_orientation(niiImage->sto_xyz,&i_orient,&j_orient,&k_orient);
        // Use the qform orientation otherwise
        else nifti_mat44_to_orientation(niiImage->qto_xyz,&i_orient,&j_orient,&k_orient);
        if(i_orient==NIFTI_L2R && j_orient==NIFTI_P2A && k_orient==NIFTI_I2S){
            if(niiImage->nt>1)
                nrrdImage->space=nrrdSpaceRightAnteriorSuperiorTime;
            else nrrdImage->space=nrrdSpaceRightAnteriorSuperior;
        }
        else if(i_orient==NIFTI_R2L && j_orient==NIFTI_P2A && k_orient==NIFTI_I2S){
            if(niiImage->nt>1)
                nrrdImage->space=nrrdSpaceLeftAnteriorSuperiorTime;
            else nrrdImage->space=nrrdSpaceLeftAnteriorSuperior;
        }
        else{
            fprintf(stderr, "[NiftyReg WARNING] reg_io_nifti2nrrd - The nifti qform information can be stored in the space variable.\n");
            fprintf(stderr, "[NiftyReg WARNING] reg_io_nifti2nrrd - The space direction will be used.\n");
        }
    }
    else{
        // if qform and sform are not defined the standard nifti orientation is used
        if(niiImage->nt>1)
            nrrdImage->space=nrrdSpaceRightAnteriorSuperiorTime;
        else nrrdImage->space=nrrdSpaceRightAnteriorSuperior;
    }

    // Set the space direction if qform and sform are defined
    if(niiImage->qform_code>0 || niiImage->sform_code>0){
        // the space direction is initialised to idensity
        nrrdImage->axis[0].spaceDirection[0]=1;
        nrrdImage->axis[0].spaceDirection[1]=0;
        nrrdImage->axis[0].spaceDirection[2]=0;
        nrrdImage->axis[1].spaceDirection[0]=0;
        nrrdImage->axis[1].spaceDirection[1]=1;
        nrrdImage->axis[1].spaceDirection[2]=0;
        nrrdImage->axis[2].spaceDirection[0]=0;
        nrrdImage->axis[2].spaceDirection[1]=0;
        nrrdImage->axis[2].spaceDirection[2]=1;
        for(int i=0;i<(niiImage->ndim<3?niiImage->ndim:3);++i){
            for(int j=0;j<(niiImage->ndim<3?niiImage->ndim:3);++j){
                // The sform is used if defined
                if(niiImage->sform_code>0)
                    nrrdImage->axis[i].spaceDirection[j]=niiImage->sto_xyz.m[i][j];
                // The qform matrix is used otherwise
                else nrrdImage->axis[i].spaceDirection[j]=niiImage->qto_xyz.m[i][j];
            }
        }
    }
    else{
        // Set the spacing values if qform and sform are not defined
        for(int i=0;i<niiImage->ndim;i++)
            nrrdImage->axis[i].spacing=niiImage->pixdim[i+1];
    }
    // Set the units if they are defined
    for(int i=0; i<NRRD_SPACE_DIM_MAX; i++){
      airFree(nrrdImage->spaceUnits[i]);
      nrrdImage->spaceUnits[i] = NULL;
    }
    switch(niiImage->xyz_units){
    case NIFTI_UNITS_METER:
        for(int i=0;i<(niiImage->ndim<3?niiImage->ndim:3);++i){
            nrrdImage->spaceUnits[i]=(char *)malloc(200);
            sprintf(nrrdImage->spaceUnits[i],"m");
        }
        break;
    case NIFTI_UNITS_MM:
        for(int i=0;i<(niiImage->ndim<3?niiImage->ndim:3);++i){
            nrrdImage->spaceUnits[i]=(char *)malloc(200);
            sprintf(nrrdImage->spaceUnits[i],"mm");
        }
        break;
    case NIFTI_UNITS_MICRON:
        for(int i=0;i<(niiImage->ndim<3?niiImage->ndim:3);++i){
            nrrdImage->spaceUnits[i]=(char *)malloc(200);
            sprintf(nrrdImage->spaceUnits[i],"um");
        }
        break;
    }

    // Set the time unit if it is defined
    if(niiImage->ndim>3){
        switch(niiImage->time_units){
        case NIFTI_UNITS_SEC:
            nrrdImage->spaceUnits[4]=(char *)"sec";
            break;
        case NIFTI_UNITS_MSEC:
            nrrdImage->spaceUnits[4]=(char *)"msec";
            break;
        }
    }

    // Check if the image is a vector field
    if(niiImage->nu>1){
        // Check if the image is a stationary field from NiftyReg
        if(niiImage->intent_code==NIFTI_INTENT_VECTOR){
            if(strcmp(niiImage->intent_name,"NREG_VEL_STEP")==0){
                // The number of step is store in the nrrdImage->axis[4].label pointer
                char temp[64];
                sprintf(temp,"NREG_VEL_STEP %f",niiImage->intent_p1);
                std::string str=temp;
                if(nrrdImage->axis[4].label!=NULL) free(nrrdImage->axis[4].label);
                nrrdImage->axis[4].label=(char *)malloc(str.length()*sizeof(char));
                strcpy(nrrdImage->axis[4].label,str.c_str());

            }
            else if(strcmp(niiImage->intent_name,"NREG_CPP_FILE")==0){
                std::string str="NREG_CPP_FILE";
                if(nrrdImage->axis[4].label!=NULL) free(nrrdImage->axis[4].label);
                nrrdImage->axis[4].label=(char *)malloc(str.length()*sizeof(char));
                strcpy(nrrdImage->axis[4].label, str.c_str());
            }
        }
    }

    return nrrdImage;
}
/* *************************************************************** */
Nrrd *reg_io_readNRRDfile(const char *filename)
{
    /* create a nrrd; at this point this is just an empty container */
    Nrrd *nrrdImage = nrrdNew();
    char *err;

    /* read in the nrrd from file */
    if (nrrdLoad(nrrdImage, filename, NULL)){
        err = biffGetDone(NRRD);
        fprintf(stderr, "[NiftyReg ERROR] Can not read the file \"%s\":\n%s\n", filename, err);
        free(err);
        exit(1);
    }
    return nrrdImage;
}
/* *************************************************************** */
void reg_io_writeNRRDfile(Nrrd *image, const char *filename)
{
    // Set the encoding to gziped as it is compiled as part of the NiftyReg project
    NrrdIoState *nio=nrrdIoStateNew();
    if (nrrdEncodingGzip->available()){
        nrrdIoStateEncodingSet(nio, nrrdEncodingGzip);
        nrrdIoStateSet(nio, nrrdIoStateZlibLevel, 9);
    }
    else{
        fprintf(stderr, "[NiftyReg ERROR] Can not compress the file: \"%s\"\n", filename);
    }

    char *err;
    if (nrrdSave(filename, image, nio)){
        err = biffGetDone(NRRD);
        fprintf(stderr, "[NiftyReg ERROR] Can not write the file \"%s\":\n%s\n", filename, err);
        free(err);
        exit(1);
    }
    return;
}
/* *************************************************************** */

#endif
