#include "_reg_f3d.h"

//#define ONLY_ONE_ITERATION

//#define COMPUTE_DEF_AFFINE
#define COMPUTE_DEF_SPLINE_LUT
//#define COMPUTE_DEF_SPLINE
//#define COMPUTE_DEF_COMP
#define COMPUTE_RESAMPLING
#define COMPUTE_SP_GRAD
#define COMPUTE_NMI
#define COMPUTE_NMI_GRAD
#define COMPUTE_BE
#define COMPUTE_BE_GRAD
#define COMPUTE_LE
#define COMPUTE_LE_GRAD
#define COMPUTE_VOX_GRID_CONV

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <img1>  <img2>\n", argv[0]);
        return EXIT_FAILURE;
    }

    char *inputImageOneName = argv[1];
    char *inputImageTwoName = argv[2];

    // Read the input reference image
    nifti_image *inputImageOne = reg_io_ReadImageFile(inputImageOneName);
    if (inputImageOne == NULL) {
        reg_print_msg_error("The first input image could not be read");
        return EXIT_FAILURE;
    }
    reg_tools_changeDatatype<float>(inputImageOne);
    nifti_image *inputImageTwo = reg_io_ReadImageFile(inputImageTwoName);
    if (inputImageTwo == NULL) {
        reg_print_msg_error("The second input image could not be read");
        return EXIT_FAILURE;
    }
    reg_tools_changeDatatype<float>(inputImageTwo);

    // Check that both images have the same size
    for(int i=0;i<8;++i){
        if(inputImageOne->dim[i]!=inputImageTwo->dim[i]){
            reg_print_msg_error("The input images do not have the same side");
            return EXIT_FAILURE;
        }
    }

    // Allocate a warped image
    nifti_image *warpedImage = nifti_copy_nim_info(inputImageOne);
    warpedImage->data = (void *)malloc(warpedImage->nvox*warpedImage->nbyper);

    // Create mask
    int *mask = (int *)calloc(inputImageOne->nvox,sizeof(int));

    // Generate deformation fields
    nifti_image *defFieldOne=nifti_copy_nim_info(inputImageOne);
    defFieldOne->ndim=defFieldOne->dim[0]=5;
    defFieldOne->nt=defFieldOne->dim[4]=1;
    defFieldOne->nu=defFieldOne->dim[5]=defFieldOne->nz>1?3:2;
    defFieldOne->nvox = (size_t)defFieldOne->nx * defFieldOne->ny *
            defFieldOne->nz * defFieldOne->nu;
    defFieldOne->data = (void *)malloc(defFieldOne->nvox*defFieldOne->nbyper);
    nifti_image *defFieldTwo=nifti_copy_nim_info(defFieldOne);
    defFieldTwo->data = (void *)malloc(defFieldTwo->nvox*defFieldTwo->nbyper);
    nifti_image *defFieldThr=nifti_copy_nim_info(defFieldOne);
    defFieldThr->data = (void *)malloc(defFieldThr->nvox*defFieldThr->nbyper);


    // Generate a control point grids
    nifti_image *splineGridOne = NULL;
    float spacing[3] = {
        inputImageOne->dx * 5.f,
        inputImageOne->dz * 5.f,
        inputImageOne->dy * 5.f
    };
    reg_createControlPointGrid<float>(&splineGridOne,
                                      inputImageOne,
                                      spacing);
    nifti_image *splineGridTwo = nifti_copy_nim_info(splineGridOne);
    splineGridTwo->data = (void *)malloc(splineGridTwo->nvox*splineGridTwo->nbyper);

    // Generate an affine matrix
    mat44 affine;reg_mat44_eye(&affine);

    time_t start,end; float total_time;

#ifdef COMPUTE_DEF_AFFINE
    // Compute n deformation field from the affine matrix
#ifdef ONLY_ONE_ITERATION
    const int affine_iteration=1;
#else
    const int affine_iteration=150;
#endif
    time(&start);
    for(int i=0;i<affine_iteration;++i)
        reg_affine_getDeformationField(&affine,
                                       defFieldOne,
                                       false,
                                       mask);
    time(&end);
    total_time=end-start;
    printf("Affine deformation in %g second(s) per iteration [%g]\n",
           total_time/(float)affine_iteration, total_time);
#endif


    // Compute n deformation field from the control point grid
#ifdef ONLY_ONE_ITERATION
    const int spline_iteration=1;
#else
    const int spline_iteration=150;
#endif
#ifdef COMPUTE_DEF_SPLINE
    time(&start);
    for(int i=0;i<spline_iteration;++i)
        reg_spline_getDeformationField(splineGridOne,
                                       defFieldOne,
                                       mask,
                                       false,
                                       true,
                                       true);
    time(&end);
    total_time=end-start;
    printf("BSpline (no lut) deformation in %g second(s) per iteration [%g]\n",
           total_time/(float)spline_iteration, total_time);
#endif
#ifdef COMPUTE_DEF_SPLINE_LUT
    time(&start);
    for(int i=0;i<spline_iteration;++i)
        reg_spline_getDeformationField(splineGridOne,
                                       defFieldOne,
                                       mask);
    time(&end);
    total_time=end-start;
    printf("BSpline (with lut) deformation in %g second(s) per iteration [%g]\n",
           total_time/(float)spline_iteration, total_time);
#endif


#ifdef COMPUTE_DEF_COMP
    reg_spline_getDeformationField(splineGridOne,
                                   defFieldTwo,
                                   mask);
// Compute n composed deformation fields
#ifdef ONLY_ONE_ITERATION
    const int compose_field_iteration=1;
#else
    const int compose_field_iteration=150;
#endif
    time(&start);
    for(int i=0;i<compose_field_iteration;++i){
        reg_defField_compose(defFieldOne, defFieldTwo, mask);
        memcpy(defFieldTwo->data, defFieldOne->data, defFieldTwo->nvox*defFieldTwo->nbyper);
    }
    time(&end);
    total_time=end-start;
    printf("Compose deformation in %g second(s) per iteration [%g]\n",
           total_time/(float)compose_field_iteration, total_time);
#endif
    // generate and initialise a NMI object
    reg_nmi *nmi=new reg_nmi;
    nmi->SetTimepointWeight(0, 1.);
    nmi->SetRefAndFloatBinNumbers(68, 68, 0);
    nmi->InitialiseMeasure(inputImageOne,
                           inputImageTwo,
                           mask,
                           inputImageTwo,
                           defFieldTwo,
                           defFieldThr);

    // Compute the NMI

#ifdef COMPUTE_NMI
#ifdef ONLY_ONE_ITERATION
    const int nmi_iteration=1;
#else
    const int nmi_iteration=150;
#endif
    time(&start);
    for(int i=0;i<nmi_iteration;++i)
        nmi->GetSimilarityMeasureValue();
    time(&end);
    total_time=end-start;
    printf("Compute NMI in %g second(s) per iteration [%g]\n",
           total_time/(float)nmi_iteration, total_time);
#endif

#ifdef COMPUTE_RESAMPLING
    // Warp the floating image the NMI
#ifdef ONLY_ONE_ITERATION
    const int resample_iteration=1;
#else
    const int resample_iteration=150;
#endif
    time(&start);
    for(int i=0;i<resample_iteration;++i)
       reg_resampleImage(inputImageTwo,
                         warpedImage,
                         defFieldOne,
                         mask,
                         1,
                         std::numeric_limits<float>::quiet_NaN());
    time(&end);
    total_time=end-start;
    printf("Resampling in %g second(s) per iteration [%g]\n",
           total_time/(float)resample_iteration, total_time);
#endif

#ifdef COMPUTE_BE
    // Compute the bending energy
#ifdef ONLY_ONE_ITERATION
    const int be_iteration=1;
#else
    const int be_iteration=150;
#endif
    time(&start);
    for(int i=0;i<be_iteration;++i)
       reg_spline_approxBendingEnergy(splineGridOne);
    time(&end);
    total_time=end-start;
    printf("Bending energy in %g second(s) per iteration [%g]\n",
           total_time/(float)be_iteration, total_time);
#endif

#ifdef COMPUTE_BE_GRAD
    // Compute the bending energy gradient
#ifdef ONLY_ONE_ITERATION
    const int be_grad_iteration=1;
#else
    const int be_grad_iteration=15;
#endif
    time(&start);
    for(int i=0;i<be_grad_iteration;++i)
       reg_spline_approxBendingEnergyGradient(splineGridOne,
                                              splineGridTwo,
                                              0.01);
    time(&end);
    total_time=end-start;
    printf("Bending energy gradient in %g second(s) per iteration [%g]\n",
           total_time/(float)be_grad_iteration, total_time);
#endif

#ifdef COMPUTE_LE
    // Compute the linear-elasticity
#ifdef ONLY_ONE_ITERATION
    const int le_iteration=1;
#else
    const int le_iteration=150;
#endif
    time(&start);
    for(int i=0;i<le_iteration;++i)
       reg_spline_approxLinearEnergy(splineGridOne);
    time(&end);
    total_time=end-start;
    printf("Linear elasticity in %g second(s) per iteration [%g]\n",
           total_time/(float)le_iteration, total_time);
#endif

#ifdef COMPUTE_LE_GRAD
    // Compute the linear-elasticity Gradient
#ifdef ONLY_ONE_ITERATION
    const int le_grad_iteration=1;
#else
    const int le_grad_iteration=15;
#endif
    time(&start);
    for(int i=0;i<le_grad_iteration;++i)
       reg_spline_approxLinearEnergyGradient(splineGridOne,
                                             splineGridTwo,
                                             0.01);
    time(&end);
    total_time=end-start;
    printf("Linear elasticity gradient in %g second(s) per iteration [%g]\n",
           total_time/(float)le_grad_iteration, total_time);
#endif

#ifdef COMPUTE_SP_GRAD
    // Compute the spatial gradient
#ifdef ONLY_ONE_ITERATION
    const int spatial_gradient_iteration=1;
#else
    const int spatial_gradient_iteration=15;
#endif
    time(&start);
    for(int i=0;i<spatial_gradient_iteration;++i)
        reg_getImageGradient(inputImageOne,
                             defFieldTwo,
                             defFieldOne,
                             mask,
                             1,
                             std::numeric_limits<float>::quiet_NaN(),
                             0);
    time(&end);
    total_time=end-start;
    printf("Spatial gradient in %g second(s) per iteration [%g]\n",
           total_time/(float)spatial_gradient_iteration, total_time);
#endif


#ifdef COMPUTE_NMI_GRAD
    // Compute the NMI voxel gradient
#ifdef ONLY_ONE_ITERATION
    const int nmi_gradient_iteration=1;
#else
    const int nmi_gradient_iteration=15;
#endif
    time(&start);
    for(int i=0;i<nmi_gradient_iteration;++i)
        nmi->GetVoxelBasedSimilarityMeasureGradient(0);
    time(&end);
    total_time=end-start;
    printf("NMI gradient in %g second(s) per iteration [%g]\n",
           total_time/(float)nmi_gradient_iteration, total_time);
#endif


#ifdef COMPUTE_VOX_GRID_CONV
    // Compute n voxel to grid conversion
#ifdef ONLY_ONE_ITERATION
    const int voxel_to_grid_iteration=1;
#else
    const int voxel_to_grid_iteration=15;
#endif
    time(&start);
    for(int i=0;i<voxel_to_grid_iteration;++i){
       int kernel_type=CUBIC_SPLINE_KERNEL;
       // The voxel based NMI gradient is convolved with a spline kernel
       // Convolution along the x axis
       float currentNodeSpacing[3];
       currentNodeSpacing[0]=currentNodeSpacing[1]=currentNodeSpacing[2]=splineGridOne->dx;
       bool activeAxis[3]= {1,0,0};
       reg_tools_kernelConvolution(defFieldThr,
                                   currentNodeSpacing,
                                   kernel_type,
                                   NULL, // mask
                                   NULL, // all volumes are considered as active
                                   activeAxis
                                   );
       // Convolution along the y axis
       currentNodeSpacing[0]=currentNodeSpacing[1]=currentNodeSpacing[2]=splineGridOne->dy;
       activeAxis[0]=0;
       activeAxis[1]=1;
       reg_tools_kernelConvolution(defFieldThr,
                                   currentNodeSpacing,
                                   kernel_type,
                                   NULL, // mask
                                   NULL, // all volumes are considered as active
                                   activeAxis
                                   );
       // Convolution along the z axis if required
       if(defFieldThr->nz>1)
       {
          currentNodeSpacing[0]=currentNodeSpacing[1]=currentNodeSpacing[2]=splineGridOne->dz;
          activeAxis[1]=0;
          activeAxis[2]=1;
          reg_tools_kernelConvolution(defFieldThr,
                                      currentNodeSpacing,
                                      kernel_type,
                                      NULL, // mask
                                      NULL, // all volumes are considered as active
                                      activeAxis
                                      );
       }

       // The node based NMI gradient is extracted
       mat44 reorientation;
       if(inputImageTwo->sform_code>0)
          reorientation = inputImageTwo->sto_ijk;
       else reorientation = inputImageTwo->qto_ijk;
       reg_voxelCentric2NodeCentric(splineGridTwo,
                                    defFieldThr,
                                    0.1,
                                    false, // no update
                                    &reorientation
                                    );
    }
    time(&end);
    total_time=end-start;
    printf("Grid based gradient in %g second(s) per iteration [%g]\n",
           total_time/(float)voxel_to_grid_iteration, total_time);
#endif

    free(mask);

    nifti_image_free(defFieldOne);
    nifti_image_free(defFieldTwo);
    nifti_image_free(defFieldThr);
    nifti_image_free(splineGridOne);
    nifti_image_free(splineGridTwo);

    nifti_image_free(inputImageOne);
    nifti_image_free(inputImageTwo);

    return EXIT_SUCCESS;
}

