#include "_reg_localTransformation_gpu.h"
#include "_reg_localTransformation.h"
#include "_reg_tools.h"

#include "_reg_common_gpu.h"

#define SIZE 128
#define SPAC 5

#define EPS 0.001

int main(int argc, char **argv)
{
    if(argc!=3){
        fprintf(stderr, "Usage: %s <dim> <type>\n", argv[0]);
        fprintf(stderr, "<dim>\tImages dimension (2,3)\n");
        fprintf(stderr, "<type>\tTest type:\n");
        fprintf(stderr, "\t\t- deformation field from spline coefficient (\"def\")\n");
        fprintf(stderr, "\t\t- deformation fields composition (\"comp\")\n");
        fprintf(stderr, "\t\t- spline velocity field parametrisation exponentation (\"exp\")\n");
        fprintf(stderr, "\t\t- Bending energy computation (\"be\")\n");
        fprintf(stderr, "\t\t- Bending energy gradient (\"beg\")\n");
        fprintf(stderr, "\t\t- Approximated Jacobian based penalty term computation (\"ajac\")\n");
        fprintf(stderr, "\t\t- Jacobian based penalty term computation (\"jac\")\n");
        fprintf(stderr, "\t\t- Approximated Jacobian based penalty term gradient (\"ajacg\")\n");
        fprintf(stderr, "\t\t- Jacobian based penalty term gradient (\"jacg\")\n");
        return EXIT_FAILURE;
    }
    int dimension=atoi(argv[1]);
    char *type=argv[2];

    // Check and setup the GPU card
    CUcontext ctx;
    if(cudaCommon_setCUDACard(&ctx, true))
        return EXIT_FAILURE;

    // Compute some useful variables
    size_t voxelNumber = SIZE*SIZE*SIZE;
    if(dimension!=3)
        voxelNumber = SIZE*SIZE;

    // Create a displacement field
    int nifti_dim[8]={5,SIZE,SIZE,SIZE,1,dimension,1,1};
    if(dimension!=3){
        nifti_dim[3]=1;
    }
    nifti_image *field = nifti_make_new_nim(nifti_dim,
                                            NIFTI_TYPE_FLOAT32,
                                            true);

    // Create an associated control point grid
    nifti_image *controlPointGrid=NULL;
    float spacing[3]={SPAC,SPAC,SPAC};
    reg_createControlPointGrid<float>(&controlPointGrid,field,spacing);
    float *cppPtr = static_cast<float *>(controlPointGrid->data);
    srand(time(0));
    for(size_t i=0;i<controlPointGrid->nvox; ++i)
		cppPtr[i]= SPAC * ((float)rand()/(float)RAND_MAX - 0.5f); // [-0.5*SPAC ; 0.5*SPAC]
    reg_getDeformationFromDisplacement(controlPointGrid);

    size_t controlPointNumber =
            controlPointGrid->nx *
            controlPointGrid->ny *
            controlPointGrid->nz;

    // Create a mask
    int *mask = (int *)malloc(voxelNumber*sizeof(int));
    for(size_t i=0;i<voxelNumber; ++i)
        mask[i]=i;

    // Create the displacement field on the device
    float4 *field_gpu=NULL;
    cudaCommon_allocateArrayToDevice<float4>(&field_gpu,
                                             voxelNumber);

    // Create the control point grid on the device
    float4 *controlPointGrid_gpu=NULL;
    cudaCommon_allocateArrayToDevice<float4>(&controlPointGrid_gpu,
                                             controlPointGrid->nx *
                                             controlPointGrid->ny *
                                             controlPointGrid->nz);

    // Transfer the ccp coefficients from the host to the device
    cudaCommon_transferNiftiToArrayOnDevice(&controlPointGrid_gpu, controlPointGrid);

    // Create a mask array on the device
    int *mask_gpu = NULL;
    NR_CUDA_SAFE_CALL(cudaMalloc(&mask_gpu, voxelNumber*sizeof(int)));

    // Transfer the mask on the device
    NR_CUDA_SAFE_CALL(cudaMemcpy(mask_gpu, mask, voxelNumber*sizeof(int), cudaMemcpyHostToDevice));

    float maxDifferenceDEF=0;
    float maxDifferenceCOMP=0;
    float maxDifferenceEXP=0;
    float maxDifferenceBE=0;
    float maxDifferenceBEG=0;
    float maxDifferenceAJAC=0;
    float maxDifferenceJAC=0;
    float maxDifferenceAJACG=0;
    float maxDifferenceJACG=0;
    nifti_image *field2 = nifti_copy_nim_info(field);
    field2->data = (void *)malloc(field2->nvox*field2->nbyper);

    if(strcmp(type,"def")==0){
        // Compute the displacement field on the host
        reg_spline_getDeformationField(controlPointGrid,
                                       field,
                                       mask,
                                       false, // no composition
                                       true); // non-interpolant spline are used
        // Compute the displacement field on the device
        reg_spline_getDeformationField_gpu(controlPointGrid,
                                           field,
                                           &controlPointGrid_gpu,
                                           &field_gpu,
                                           &mask_gpu,
                                           voxelNumber,
                                           true);  // non-interpolant spline are used
        // Transfer the device field on the host
        cudaCommon_transferFromDeviceToNifti<float4>(field2, &field_gpu);
        // Compute the difference between both fields
        maxDifferenceDEF=reg_test_compare_images(field,field2);
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] [dim=%i] Deformation field difference: %g\n",
               dimension,
               maxDifferenceDEF);
        nifti_set_filenames(field, "reg_test_def_cpu.nii",0,0);
        nifti_set_filenames(field2,"reg_test_def_gpu.nii",0,0);
        nifti_image_write(field);
        nifti_image_write(field2);
        reg_tools_divideImageToImage(field,field2,field);
        reg_tools_substractValueToImage(field,field,1.f);
        reg_tools_abs_image(field);
        nifti_set_filenames(field,"reg_test_def_diff.nii",0,0);
        nifti_image_write(field);
#endif
    }
    else if(strcmp(type,"comp")==0){
        // generate a deformation field
        reg_spline_getDeformationField(controlPointGrid,
                                       field,
                                       mask,
                                       false, // no composition
                                       true); // non-interpolant spline are used
        // Copy the deformation field
        nifti_image *field_comp_cpu = nifti_copy_nim_info(field);
        field_comp_cpu->data = (void *)malloc(field_comp_cpu->nvox*field_comp_cpu->nbyper);
        memcpy(field_comp_cpu->data, field->data, field->nvox*field->nbyper);
        // Create a second field on the device
        float4 *field_comp_gpu=NULL;
        cudaCommon_allocateArrayToDevice<float4>(&field_comp_gpu,voxelNumber);
        // Transfer the host deformation field to the device fields
        cudaCommon_transferNiftiToArrayOnDevice(&field_gpu, field);
        cudaCommon_transferNiftiToArrayOnDevice(&field_comp_gpu, field);
        // Compose both deformation fields on the host
        reg_defField_compose(field_comp_cpu,field,mask);
        // Compose both deformation fields on the device
        reg_defField_compose_gpu(field, &field_comp_gpu,&field_gpu,&mask_gpu,voxelNumber);
        // Transfer the device field on the host
        cudaCommon_transferFromDeviceToNifti<float4>(field2, &field_gpu);
        // Compute the difference between both fields
        maxDifferenceCOMP=reg_test_compare_images(field,field2);
        // Free extra arrays allocated on the GPU
        nifti_image_free(field_comp_cpu);
        cudaCommon_free(&field_comp_gpu);
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] [dim=%i] Deformation field composition difference: %g\n",
               dimension,
               maxDifferenceCOMP);
        nifti_set_filenames(field,"reg_test_def_cpu.nii",0,0);
        nifti_set_filenames(field2,"reg_test_def_gpu.nii",0,0);
        nifti_image_write(field);
        nifti_image_write(field2);
#endif
    }
    else if(strcmp(type,"exp")==0){
        // Convert the control point grid into a velocity field
        controlPointGrid->intent_code=NIFTI_INTENT_VECTOR;
        controlPointGrid->intent_p1=6;
        memset(controlPointGrid->intent_name, 0, 16);
        strcpy(controlPointGrid->intent_name,"NREG_VEL_STEP");
        // Exponentiate the velocity field using the host
        reg_spline_getDeformationFieldFromVelocityGrid(controlPointGrid,
                                                        field,
                                                       false);
        // Exponentiate the velocity field using the device
        reg_getDeformationFieldFromVelocityGrid_gpu(controlPointGrid,
                                                    field,
                                                    &controlPointGrid_gpu,
                                                    &field_gpu);
        // Transfer the device field on the host
        cudaCommon_transferFromDeviceToNifti<float4>(field2, &field_gpu);
        // Compute the difference between both fields
        maxDifferenceEXP=reg_test_compare_images(field,field2);
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] [dim=%i] Deformation field exponentiation difference: %g\n",
               dimension,
               maxDifferenceEXP);
        nifti_set_filenames(field,"reg_test_def_cpu.nii",0,0);
        nifti_set_filenames(field2,"reg_test_def_gpu.nii",0,0);
        nifti_image_write(field);
        nifti_image_write(field2);
#endif
    }
    else if(strcmp(type,"be")==0){
        double be_cpu = reg_spline_approxBendingEnergy(controlPointGrid);
        double be_gpu = reg_spline_approxBendingEnergy_gpu(controlPointGrid,
                                                           &controlPointGrid_gpu);
        maxDifferenceBE = (be_cpu / be_gpu) - 1.0;
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] [dim=%i] Bending energy difference: %g [=(%g/%g)-1]\n",
               dimension,
               maxDifferenceBE,
               be_cpu,
               be_gpu);
#endif
    }
    else if(strcmp(type,"beg")==0){
        // Allocate two extra nifti_image to store the bending energy gradients
        nifti_image *be_grad_cpu=nifti_copy_nim_info(controlPointGrid);
        nifti_image *be_grad_gpu=nifti_copy_nim_info(controlPointGrid);
        be_grad_cpu->data=(void *)calloc(be_grad_cpu->nvox,be_grad_cpu->nbyper);
        be_grad_gpu->data=(void *)calloc(be_grad_gpu->nvox,be_grad_gpu->nbyper);
        // Allocate an extra cuda array to store the device gradient
        float4 *be_grad_device=NULL;
        cudaCommon_allocateArrayToDevice<float4>(&be_grad_device,
                                                 controlPointNumber);
        // Set the gradients arrays to zero
        reg_tools_multiplyValueToImage(be_grad_cpu,be_grad_cpu,0.f);
        cudaCommon_transferNiftiToArrayOnDevice<float4>(&be_grad_device,be_grad_cpu);
        // Compute the gradient on the host
        reg_spline_approxBendingEnergyGradient(controlPointGrid,
                                               be_grad_cpu,
											   controlPointNumber); // weight
        // Compute the gradient on the device
        reg_spline_approxBendingEnergyGradient_gpu(controlPointGrid,
                                                   &controlPointGrid_gpu,
                                                   &be_grad_device,
												   controlPointNumber); // weight
        // Transfer the device field on the host
        cudaCommon_transferFromDeviceToNifti<float4>(be_grad_gpu, &be_grad_device);
        // Compute the difference between both gradient arrays
        maxDifferenceBEG=reg_test_compare_images(be_grad_cpu,be_grad_gpu);
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] [dim=%i] Bending energy gradient difference: %g\n",
               dimension,
               maxDifferenceBEG);
        nifti_set_filenames(be_grad_cpu,"reg_test_def_cpu.nii",0,0);
        nifti_set_filenames(be_grad_gpu,"reg_test_def_gpu.nii",0,0);
        nifti_image_write(be_grad_cpu);
        nifti_image_write(be_grad_gpu);
        reg_tools_divideImageToImage(be_grad_cpu,be_grad_gpu,be_grad_cpu);
        reg_tools_substractValueToImage(be_grad_cpu,be_grad_cpu,1.f);
        reg_tools_abs_image(be_grad_cpu);
        nifti_set_filenames(be_grad_cpu,"reg_test_def_diff.nii",0,0);
        nifti_image_write(be_grad_cpu);
#endif
        // Free the extra images and array
        nifti_image_free(be_grad_cpu);
        nifti_image_free(be_grad_gpu);
        cudaCommon_free(&be_grad_device);
    }
	else if(strcmp(type,"jac")==0 || strcmp(type,"ajac")==0){
        bool approximation=false;
        if(strcmp(type,"ajac")==0) approximation=true;
        double jac_cpu = reg_spline_getJacobianPenaltyTerm(controlPointGrid,
                                                           field,
                                                           approximation); // aprox
        double jac_gpu = reg_spline_getJacobianPenaltyTerm_gpu(field,
                                                               controlPointGrid,
                                                               &controlPointGrid_gpu,
                                                               approximation);
        if(jac_cpu!=jac_cpu){
            fprintf(stderr,
                    "Error with the computation of the Jacobian based penalty term on the CPU\n");
        }
        if(jac_gpu!=jac_gpu){
            fprintf(stderr,
                    "Error with the computation of the Jacobian based penalty term on the GPU\n");
		}
        printf("approx[%i] | dim [%i] : %g %g\n", approximation, dimension, jac_cpu, jac_gpu);
        if(approximation){
            maxDifferenceAJAC = fabs(jac_cpu - jac_gpu) / jac_cpu;
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] [dim=%i] Approximated Jacobian difference: %g\n",
               dimension,
               maxDifferenceAJAC);
#endif
        }
        else{
            maxDifferenceJAC = fabs(jac_cpu - jac_gpu) / jac_cpu;
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] [dim=%i] Jacobian difference: %g\n",
               dimension,
               maxDifferenceJAC);
#endif
        }
    }
    else if(strcmp(type,"ajacg")==0 || strcmp(type,"jacg")==0){
        bool approximation=false;
        if(strcmp(type,"ajacg")==0) approximation=true;
        // Allocate two extra nifti_image to store the jacobian gradients
        nifti_image *jac_grad_cpu=nifti_copy_nim_info(controlPointGrid);
        nifti_image *jac_grad_gpu=nifti_copy_nim_info(controlPointGrid);
        jac_grad_cpu->data=(void *)calloc(jac_grad_cpu->nvox,jac_grad_cpu->nbyper);
        jac_grad_gpu->data=(void *)malloc(jac_grad_gpu->nvox*jac_grad_gpu->nbyper);
        // Allocate an extra cuda array to store the device gradient
        float4 *jac_grad_device=NULL;
        cudaCommon_allocateArrayToDevice<float4>(&jac_grad_device,
                                                 controlPointNumber);
        // Set the gradients arrays to zero
        reg_tools_multiplyValueToImage(jac_grad_cpu,jac_grad_cpu,0.f);
        cudaCommon_transferNiftiToArrayOnDevice<float4>(&jac_grad_device,jac_grad_cpu);
        // Compute the gradient on the host
        reg_spline_getJacobianPenaltyTermGradient(controlPointGrid,
                                                  field,
                                                  jac_grad_cpu,
                                                  controlPointNumber,
                                                  approximation);
        // Compute the gradient on the device
        reg_spline_getJacobianPenaltyTermGradient_gpu(field,
                                                      controlPointGrid,
                                                      &controlPointGrid_gpu,
                                                      &jac_grad_device,
                                                      controlPointNumber,
                                                      approximation);
        // Transfer the device field on the host
        cudaCommon_transferFromDeviceToNifti<float4>(jac_grad_gpu, &jac_grad_device);
        // Compute the difference between both gradient arrays
        if(approximation){
            maxDifferenceAJACG = reg_test_compare_images(jac_grad_cpu,jac_grad_gpu);
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] [dim=%i] Approximated Jacobian gradient difference: %g\n",
               dimension,
               maxDifferenceAJACG);
        nifti_set_filenames(jac_grad_cpu,"reg_test_def_cpu.nii",0,0);
        nifti_set_filenames(jac_grad_gpu,"reg_test_def_gpu.nii",0,0);
        nifti_image_write(jac_grad_cpu);
        nifti_image_write(jac_grad_gpu);
#endif
        }
        else{
            maxDifferenceJACG = reg_test_compare_images(jac_grad_cpu,jac_grad_gpu);
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] [dim=%i] Jacobian gradient difference: %g\n",
               dimension,
               maxDifferenceJACG);
        nifti_set_filenames(jac_grad_cpu,"reg_test_def_cpu.nii",0,0);
        nifti_set_filenames(jac_grad_gpu,"reg_test_def_gpu.nii",0,0);
        nifti_image_write(jac_grad_cpu);
        nifti_image_write(jac_grad_gpu);
#endif
        }
        // Free the extra images and array
        nifti_image_free(jac_grad_cpu);
        nifti_image_free(jac_grad_gpu);
        cudaCommon_free(&jac_grad_device);
    }
    else{
        fprintf(stderr, "ERROR, nothing has been done. Unkown type: %s\n",type);
    }

    // Clean the allocated arrays
    nifti_image_free(field);
    nifti_image_free(field2);
    nifti_image_free(controlPointGrid);
    free(mask);
    cudaCommon_free(&field_gpu);
    cudaCommon_free(&controlPointGrid_gpu);
    cudaCommon_free(&mask_gpu);

    cudaCommon_unsetCUDACard(&ctx);

    if(maxDifferenceDEF>EPS){
        fprintf(stderr,
                "[dim=%i] Deformation field difference too high: %g\n",
                dimension,
                maxDifferenceDEF);
        return EXIT_FAILURE;
    }
    else if(maxDifferenceCOMP>EPS){
        fprintf(stderr,
                "[dim=%i] Deformation field composition difference too high: %g\n",
                dimension,
                maxDifferenceCOMP);
        return EXIT_FAILURE;
    }
    else if(maxDifferenceEXP>EPS){
        fprintf(stderr,
                "[dim=%i] Velocity field exponentiation difference too high: %g\n",
                dimension,
                maxDifferenceEXP);
        return EXIT_FAILURE;
    }
    else if(maxDifferenceBE>EPS){
        fprintf(stderr,
                "[dim=%i] Bending energy value difference too high: %g\n",
                dimension,
                maxDifferenceBE);
        return EXIT_FAILURE;
    }
    else if(maxDifferenceBEG>EPS){
        fprintf(stderr,
                "[dim=%i] Bending energy gradient difference too high: %g\n",
                dimension,
                maxDifferenceBEG);
        return EXIT_FAILURE;
    }
    else if(maxDifferenceAJAC>EPS){
        fprintf(stderr,
                "[dim=%i] Approx. Jacobian based penalty term value difference too high: %g\n",
                dimension,
                maxDifferenceAJAC);
        return EXIT_FAILURE;
    }
    else if(maxDifferenceJAC>EPS){
        fprintf(stderr,
                "[dim=%i] Jacobian based penalty term value difference too high: %g\n",
                dimension,
                maxDifferenceJAC);
        return EXIT_FAILURE;
    }
    else if(maxDifferenceAJACG>EPS){
        fprintf(stderr,
                "[dim=%i] Approx. Jacobian based penalty term gradient difference too high: %g\n",
                dimension,
                maxDifferenceAJACG);
        return EXIT_FAILURE;
    }
    else if(maxDifferenceJACG>EPS){
        fprintf(stderr,
                "[dim=%i] Jacobian based penalty term gradient difference too high: %g\n",
                dimension,
                maxDifferenceJACG);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
