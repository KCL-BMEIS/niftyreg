#include "_reg_f3d2.h"

#define size 128

int main(int argc, char **argv)
{
    char msg[255];
    sprintf(msg,"Usage: %s dim type",argv[0]);
    if(argc!=3){
        reg_print_msg_error(msg);
        return EXIT_FAILURE;
    }
    const int dim=atoi(argv[1]);
    const int type=atoi(argv[2]);
    if(dim!=2 && dim!=3){
        reg_print_msg_error(msg);
        reg_print_msg_error("Expected value for dim are 2 and 3");
        return EXIT_FAILURE;
    }
    if(type<0 || type>4){
        reg_print_msg_error(msg);
        reg_print_msg_error("Expected value for type are 0, 1, 2, 3 and 4");
        return EXIT_FAILURE;
    }

    // Create some images
    int image_dim[8]={dim,size,size,dim==2?1:size,1,1,1,1};
    nifti_image *reference=nifti_make_new_nim(image_dim,NIFTI_TYPE_FLOAT32,true);
    nifti_image *floating=nifti_make_new_nim(image_dim,NIFTI_TYPE_FLOAT32,true);
    reg_checkAndCorrectDimension(reference);
    reg_checkAndCorrectDimension(floating);
    reference->qform_code = 1;
    floating->qform_code = 1;
    reference->qoffset_x=floating->qoffset_x = -size/2;
    reference->qoffset_y=floating->qoffset_y = -size/2;
    if(dim>2)
        reference->qoffset_z=floating->qoffset_z = -size/2;
    else reference->qoffset_z=floating->qoffset_z = 0.f;
    reference->qto_xyz = nifti_quatern_to_mat44(
                reference->quatern_b,reference->quatern_c,reference->quatern_d,
                reference->qoffset_x,reference->qoffset_y,reference->qoffset_z,
                1.f,1.f,1.f,reference->qfac);
    floating->qto_xyz = nifti_quatern_to_mat44(
                floating->quatern_b,floating->quatern_c,floating->quatern_d,
                floating->qoffset_x,floating->qoffset_y,floating->qoffset_z,
                1.f,1.f,1.f,floating->qfac);
    reference->qto_ijk = nifti_mat44_inverse(reference->qto_xyz);
    floating->qto_ijk = nifti_mat44_inverse(floating->qto_xyz);

    // Fill the image intensities
    float *refPtr = static_cast<float *>(reference->data);
    float *floPtr = static_cast<float *>(floating->data);
    for(int z=0; z<reference->nz; ++z){
        float distZ = fabs(z-reference->nz/2);
        if(dim==2) distZ=0;
        for(int y=0; y<reference->ny; ++y){
            float distY = fabs(y-reference->ny/2);
            for(int x=0; x<reference->nx; ++x){
                float distX = fabs(x-reference->nx/2);
                float distance = sqrtf(reg_pow2(distX)+reg_pow2(distY)+reg_pow2(distZ));
                if(distance<0.4f*(float)size)
                    *refPtr++=1.f;
                else *refPtr++=0.f;
                if(distance<0.3f*(float)size)
                    *floPtr++=1.f;
                else *floPtr++=0.f;
            }
        }
    }
    double initialMeanDifference = reg_tools_getMeanRMS(reference,floating);

    // Aff an affine transformation into the floating image
    mat44 affine_transformation;reg_mat44_eye(&affine_transformation);
    affine_transformation.m[0][0]=+0.9f;affine_transformation.m[0][3]=+10.f;
    affine_transformation.m[1][1]=+1.1f;affine_transformation.m[1][3]=-10.f;
    floating->sform_code = 1;
    floating->sto_xyz = reg_mat44_mul(&affine_transformation,&floating->qto_xyz);
    floating->sto_ijk = nifti_mat44_inverse(floating->sto_xyz);

    // Run the specified registration
    reg_f3d<float> *f3d = NULL;
    switch(type){
    case 0: f3d = new reg_f3d<float>(1,1);
        break;
    case 1: f3d = new reg_f3d<float>(1,1);
        f3d->SetJacobianLogWeight(0.01f);
        break;
    case 2: f3d = new reg_f3d<float>(1,1);
        f3d->SetJacobianLogWeight(0.01f);
        f3d->DoNotApproximateJacobianLog();
        break;
    case 3: f3d = new reg_f3d_sym<float>(1,1);
        break;
    case 4: f3d = new reg_f3d2<float>(1,1);
        break;
    }
    f3d->SetAffineTransformation(&affine_transformation);
    f3d->SetReferenceImage(reference);
    f3d->SetFloatingImage(floating);
    f3d->SetWarpedPaddingValue(0.f);
    f3d->UseSSD(0);
    f3d->SetMaximalIterationNumber(1000);
//    f3d->DoNotPrintOutInformation();
    f3d->Run();
    nifti_image *outWar = f3d->GetWarpedImage()[0];
    nifti_image *outCPP = f3d->GetControlPointPositionImage();
    delete f3d;
    double residualMeanDifference = 100.f*reg_tools_getMeanRMS(reference,outWar)/initialMeanDifference;
    std::cout.precision(2);
    std::cout << "Residual error " << residualMeanDifference << "\%" << std::endl;
    if(residualMeanDifference > 5.f){
        reg_io_WriteImageFile(reference,"test_ref.nii");
        reg_io_WriteImageFile(outWar,"test_war.nii");
        return EXIT_FAILURE;
    }
    if(type>0){
        // We check that all the Jacobian values are positives
        reg_spline_GetJacobianMap(outCPP,outWar);
        float min_jac = reg_tools_getMinValue(outWar);
        float max_jac = reg_tools_getMaxValue(outWar);
        std::cout.precision(5);
        std::cout << "Jacobian range [" << min_jac << ";" << max_jac << "]" << std::endl;
        if(min_jac <= 0.f){
            reg_io_WriteImageFile(outCPP,"test_cpp.nii");
            reg_io_WriteImageFile(outWar,"test_jac.nii");
            return EXIT_FAILURE;
        }
    }
    nifti_image_free(outWar);
    nifti_image_free(outCPP);

    nifti_image_free(reference);
    nifti_image_free(floating);

    return EXIT_SUCCESS;
}
