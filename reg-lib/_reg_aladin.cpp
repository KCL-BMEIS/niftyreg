#include "_reg_aladin.h"

/* *************************************************************** */
template<class T>
reg_aladin<T>::reg_aladin() {
    this->executableName = (char*)"Aladin";

    this->affineTransformation.reset(new mat44);
    this->inputTransformName = nullptr;

    this->blockMatchingParams = nullptr;

    this->verbose = true;

    this->maxIterations = 5;

    this->numberOfLevels = 3;
    this->levelsToPerform = 3;

    this->performRigid = 1;
    this->performAffine = 1;

    this->blockStepSize = 1;
    this->blockPercentage = 50;
    this->inlierLts = 50;

    this->alignCentre = 1;
    this->alignCentreMass = 0;

    this->interpolation = 1;    // linear

    this->floatingSigma = 0;
    this->referenceSigma = 0;

    this->referenceLowerThreshold = std::numeric_limits<T>::lowest();
    this->referenceUpperThreshold = std::numeric_limits<T>::max();

    this->floatingLowerThreshold = std::numeric_limits<T>::lowest();
    this->floatingUpperThreshold = std::numeric_limits<T>::max();

    this->warpedPaddingValue = std::numeric_limits<T>::quiet_NaN();

    this->funcProgressCallback = nullptr;
    this->paramsProgressCallback = nullptr;

    this->platformType = PlatformType::Cpu;
    this->currentLevel = 0;
    this->gpuIdx = 999;

    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::SetVerbose(bool _verbose) {
    this->verbose = _verbose;
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::Print() {
    if (!this->inputReference)
        NR_FATAL_ERROR("No reference image has been specified");
    if (!this->inputFloating)
        NR_FATAL_ERROR("No floating image has been specified");

    /* *********************************** */
    /* DISPLAY THE REGISTRATION PARAMETERS */
    /* *********************************** */
    NR_VERBOSE("Parameters");
    NR_VERBOSE("Platform: " << this->platform->GetName());
    NR_VERBOSE("Reference image name: " << (this->inputReference->fname ? this->inputReference->fname : "(in-memory image)"));
    NR_VERBOSE("\t" << this->inputReference->nx << "x" << this->inputReference->ny << "x" << this->inputReference->nz << " voxels");
    NR_VERBOSE("\t" << this->inputReference->dx << "x" << this->inputReference->dy << "x" << this->inputReference->dz << " mm");
    NR_VERBOSE("Floating image name: " << (this->inputFloating->fname ? this->inputFloating->fname : "(in-memory image)"));
    NR_VERBOSE("\t" << this->inputFloating->nx << "x" << this->inputFloating->ny << "x" << this->inputFloating->nz << " voxels");
    NR_VERBOSE("\t" << this->inputFloating->dx << "x" << this->inputFloating->dy << "x" << this->inputFloating->dz << " mm");
    NR_VERBOSE("Maximum iteration number: " << this->maxIterations);
    NR_VERBOSE("\t(" << this->maxIterations * 2 << " during the first level)");
    NR_VERBOSE("Percentage of blocks: " << this->blockPercentage << "%");
    NR_VERBOSE("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::SetInputTransform(const char *filename) {
    this->inputTransformName = (char*)filename;
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::InitialiseRegistration() {
    NR_FUNC_CALLED();

    this->platform.reset(new Platform(this->platformType));
    this->platform->SetGpuIdx(this->gpuIdx);

    this->Print();

    if (this->inputReference->nt > 1 || this->inputReference->nu > 1)
        NR_WARN("The reference image contains more than one volume - only the first volume is used for the optimisation");
    if (this->inputFloating->nt > 1 || this->inputFloating->nu > 1)
        NR_WARN("The floating image contains more than one volume - only the first volume is used for the optimisation");

    this->CheckInputImages();

    // CREATE THE PYRAMID IMAGES
    this->referencePyramid = vector<NiftiImage>(this->levelsToPerform);
    this->floatingPyramid = vector<NiftiImage>(this->levelsToPerform);
    this->referenceMaskPyramid = vector<unique_ptr<int[]>>(this->levelsToPerform);

    // FINEST LEVEL OF REGISTRATION
    reg_createImagePyramid<T>(this->inputReference,
                              this->referencePyramid,
                              this->numberOfLevels,
                              this->levelsToPerform);
    reg_createImagePyramid<T>(this->inputFloating,
                              this->floatingPyramid,
                              this->numberOfLevels,
                              this->levelsToPerform);

    if (this->inputReferenceMask)
        reg_createMaskPyramid<T>(this->inputReferenceMask,
                                 this->referenceMaskPyramid,
                                 this->numberOfLevels,
                                 this->levelsToPerform);
    else
        for (unsigned l = 0; l < this->levelsToPerform; ++l)
            this->referenceMaskPyramid[l].reset(new int[this->referencePyramid[l].nVoxelsPerVolume()]());

    unique_ptr<Kernel> convolutionKernel(this->platform->CreateKernel(ConvolutionKernel::GetName(), nullptr));
    // SMOOTH THE INPUT IMAGES IF REQUIRED
    for (unsigned l = 0; l < this->levelsToPerform; l++) {
        if (this->referenceSigma != 0) {
            // Only the first image is smoothed
            unique_ptr<bool[]> active(new bool[this->referencePyramid[l]->nt]);
            unique_ptr<float[]> sigma(new float[this->referencePyramid[l]->nt]);
            active[0] = true;
            for (int i = 1; i < this->referencePyramid[l]->nt; ++i)
                active[i] = false;
            sigma[0] = this->referenceSigma;
            convolutionKernel->castTo<ConvolutionKernel>()->Calculate(this->referencePyramid[l], sigma.get(), ConvKernelType::Mean, nullptr, active.get());
        }
        if (this->floatingSigma != 0) {
            // Only the first image is smoothed
            unique_ptr<bool[]> active(new bool[this->floatingPyramid[l]->nt]);
            unique_ptr<float[]> sigma(new float[this->floatingPyramid[l]->nt]);
            active[0] = true;
            for (int i = 1; i < this->floatingPyramid[l]->nt; ++i)
                active[i] = false;
            sigma[0] = this->floatingSigma;
            convolutionKernel->castTo<ConvolutionKernel>()->Calculate(this->floatingPyramid[l], sigma.get(), ConvKernelType::Mean, nullptr, active.get());
        }
    }

    // THRESHOLD THE INPUT IMAGES IF REQUIRED
    for (unsigned l = 0; l < this->levelsToPerform; l++) {
        reg_thresholdImage<T>(this->referencePyramid[l], this->referenceLowerThreshold, this->referenceUpperThreshold);
        reg_thresholdImage<T>(this->floatingPyramid[l], this->floatingLowerThreshold, this->floatingUpperThreshold);
    }

    // Initialise the transformation
    if (this->inputTransformName != nullptr) {
        if (FILE *aff = fopen(this->inputTransformName, "r")) {
            fclose(aff);
        } else {
            NR_FATAL_ERROR("The specified input affine file ("s + this->inputTransformName + ") can not be read");
        }
        reg_tool_ReadAffineFile(this->affineTransformation.get(), this->inputTransformName);
    } else { // No input affine transformation
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                this->affineTransformation->m[i][j] = 0;
            }
            this->affineTransformation->m[i][i] = 1;
        }
        if (this->alignCentre && this->alignCentreMass == 0) {
            const mat44 *floatingMatrix = (this->inputFloating->sform_code > 0) ? &(this->inputFloating->sto_xyz) : &(this->inputFloating->qto_xyz);
            const mat44 *referenceMatrix = (this->inputReference->sform_code > 0) ? &(this->inputReference->sto_xyz) : &(this->inputReference->qto_xyz);
            //In pixel coordinates
            float floatingCenter[3];
            floatingCenter[0] = (float)(this->inputFloating->nx) / 2.0f;
            floatingCenter[1] = (float)(this->inputFloating->ny) / 2.0f;
            floatingCenter[2] = (float)(this->inputFloating->nz) / 2.0f;
            float referenceCenter[3];
            referenceCenter[0] = (float)(this->inputReference->nx) / 2.0f;
            referenceCenter[1] = (float)(this->inputReference->ny) / 2.0f;
            referenceCenter[2] = (float)(this->inputReference->nz) / 2.0f;
            //From pixel coordinates to real coordinates
            float floatingRealPosition[3];
            Mat44Mul(*floatingMatrix, floatingCenter, floatingRealPosition);
            float referenceRealPosition[3];
            Mat44Mul(*referenceMatrix, referenceCenter, referenceRealPosition);
            //Set translation to the transformation matrix
            this->affineTransformation->m[0][3] = floatingRealPosition[0] - referenceRealPosition[0];
            this->affineTransformation->m[1][3] = floatingRealPosition[1] - referenceRealPosition[1];
            this->affineTransformation->m[2][3] = floatingRealPosition[2] - referenceRealPosition[2];
        } else if (this->alignCentreMass == 2) {
            float referenceCentre[3] = { 0, 0, 0 };
            float referenceCount = 0;
            reg_tools_changeDatatype<float>(this->inputReference);
            float *refPtr = static_cast<float *>(this->inputReference->data);
            size_t refIndex = 0;
            for (int z = 0; z < this->inputReference->nz; ++z) {
                for (int y = 0; y < this->inputReference->ny; ++y) {
                    for (int x = 0; x < this->inputReference->nx; ++x) {
                        float value = refPtr[refIndex];
                        referenceCentre[0] += (float)x * value;
                        referenceCentre[1] += (float)y * value;
                        referenceCentre[2] += (float)z * value;
                        referenceCount += value;
                        refIndex++;
                    }
                }
            }
            referenceCentre[0] /= referenceCount;
            referenceCentre[1] /= referenceCount;
            referenceCentre[2] /= referenceCount;
            float refCOM[3]{};
            if (this->inputReference->sform_code > 0)
                Mat44Mul(this->inputReference->sto_xyz, referenceCentre, refCOM);

            float floatingCentre[3] = { 0, 0, 0 };
            float floatingCount = 0;
            reg_tools_changeDatatype<float>(this->inputFloating);
            float *floPtr = static_cast<float *>(this->inputFloating->data);
            size_t floIndex = 0;
            for (int z = 0; z < this->inputFloating->nz; ++z) {
                for (int y = 0; y < this->inputFloating->ny; ++y) {
                    for (int x = 0; x < this->inputFloating->nx; ++x) {
                        float value = floPtr[floIndex];
                        floatingCentre[0] += (float)x * value;
                        floatingCentre[1] += (float)y * value;
                        floatingCentre[2] += (float)z * value;
                        floatingCount += value;
                        floIndex++;
                    }
                }
            }
            floatingCentre[0] /= floatingCount;
            floatingCentre[1] /= floatingCount;
            floatingCentre[2] /= floatingCount;
            float floCOM[3]{};
            if (this->inputFloating->sform_code > 0)
                Mat44Mul(this->inputFloating->sto_xyz, floatingCentre, floCOM);
            Mat44Eye(this->affineTransformation.get());
            this->affineTransformation->m[0][3] = floCOM[0] - refCOM[0];
            this->affineTransformation->m[1][3] = floCOM[1] - refCOM[1];
            this->affineTransformation->m[2][3] = floCOM[2] - refCOM[2];
        }
    }
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::DeallocateCurrentInputImage() {
    this->referencePyramid[this->currentLevel] = nullptr;
    this->floatingPyramid[this->currentLevel] = nullptr;
    this->referenceMaskPyramid[this->currentLevel] = nullptr;
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::CreateKernels() {
    this->affineTransformation3DKernel.reset(platform->CreateKernel(AffineDeformationFieldKernel::GetName(), this->con.get()));
    this->resamplingKernel.reset(platform->CreateKernel(ResampleImageKernel::GetName(), this->con.get()));
    if (this->blockMatchingParams) {
        this->blockMatchingKernel.reset(platform->CreateKernel(BlockMatchingKernel::GetName(), this->con.get()));
        this->ltsKernel.reset(platform->CreateKernel(LtsKernel::GetName(), this->con.get()));
    } else {
        this->blockMatchingKernel = nullptr;
        this->ltsKernel = nullptr;
    }
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::DeallocateKernels() {
    this->affineTransformation3DKernel = nullptr;
    this->resamplingKernel = nullptr;
    this->blockMatchingKernel = nullptr;
    this->ltsKernel = nullptr;
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::GetDeformationField() {
    this->affineTransformation3DKernel->template castTo<AffineDeformationFieldKernel>()->Calculate();
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::GetWarpedImage(int interp, float padding) {
    this->GetDeformationField();
    this->resamplingKernel->template castTo<ResampleImageKernel>()->Calculate(interp, padding);
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::UpdateTransformationMatrix(int type) {
    this->blockMatchingKernel->template castTo<BlockMatchingKernel>()->Calculate();
    this->ltsKernel->template castTo<LtsKernel>()->Calculate(type);
    NR_MAT44_DEBUG(*this->affineTransformation, "The updated forward matrix");
    CheckTransformationMatrix(*this->affineTransformation, type ? "affine" : "rigid");
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::CheckInputImages() {
    // The whole pipeline is dimensioned from the reference image: a 2D reference (nz == 1) yields
    // a 2D deformation field without a z component, so a 3D floating image could not be sampled
    // through it, and a 3D reference paired with a 2D floating image allocates a warped image
    // whose voxel count does not match its grid
    const bool referenceIs3d = this->inputReference->nz > 1;
    const bool floatingIs3d = this->inputFloating->nz > 1;
    if (referenceIs3d != floatingIs3d)
        NR_FATAL_ERROR("The reference and floating images must both be 2D or both be 3D - the reference image is "s +
                       (referenceIs3d ? "3D" : "2D") + " and the floating image is " + (floatingIs3d ? "3D" : "2D"));

    // Block matching works on blocks of BLOCK_WIDTH voxels along every axis, so an axis with fewer
    // voxels than that cannot hold a single block. A block is only used when more than half of its
    // voxels are defined, so an axis needs BLOCK_WIDTH + BLOCK_WIDTH / 2 + 1 voxels before a second
    // row of blocks becomes usable; below that every block corner shares the same coordinate along
    // the axis, the correspondences are coplanar, and an affine transformation - which has to
    // observe a scaling and a shearing along that axis - cannot be estimated from them
    constexpr int minimumSize = BLOCK_WIDTH;
    constexpr int singleBlockRowSize = BLOCK_WIDTH + BLOCK_WIDTH / 2 + 1;
    const char *axisNames[] = { "x", "y", "z" };
    const int axisNumber = referenceIs3d ? 3 : 2;
    const auto checkImage = [&](const NiftiImage& image, const std::string& name) {
        const int dims[] = { image->nx, image->ny, image->nz };
        for (int axis = 0; axis < axisNumber; ++axis) {
            if (dims[axis] < minimumSize)
                NR_FATAL_ERROR("The " + name + " image has only " + std::to_string(dims[axis]) + " voxel(s) along the " +
                               axisNames[axis] + "-axis - block matching requires at least " + std::to_string(minimumSize) +
                               " voxels along every axis");
            if (dims[axis] < singleBlockRowSize && this->performAffine)
                NR_WARN("The " << name << " image has only " << dims[axis] << " voxels along the " << axisNames[axis] <<
                        "-axis, which can hold a single row of blocks: the block correspondences are coplanar along it and "
                        "an affine transformation cannot be estimated from them - a rigid transformation (-rigOnly) can");
        }
    };
    checkImage(this->inputReference, "reference");
    checkImage(this->inputFloating, "floating");
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::CheckTransformationMatrix(const mat44& matrix, const std::string& description) {
    // A singular estimate means the block correspondences were degenerate (e.g. all in one plane).
    // Used as is, it would collapse the warped image; in the symmetric algorithm its inverse and its
    // matrix logarithm - which never converges for a singular matrix - are needed to average the
    // forward and backward estimates. Stop with a diagnosis instead
    constexpr double minimumDeterminant = 1e-6;
    const double determinant = Mat44Det<double>(&matrix);
    if (!std::isfinite(determinant) || std::abs(determinant) < minimumDeterminant)
        NR_FATAL_ERROR("The estimated "s + description + " transformation is singular (determinant " + std::to_string(determinant) +
                       "): the block correspondences are degenerate, e.g. they all lie in one plane because an image has fewer than " +
                       std::to_string(BLOCK_WIDTH + BLOCK_WIDTH / 2 + 1) + " voxels along an axis - a rigid transformation (-rigOnly) "
                       "can still be estimated from such correspondences");
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::InitAladinContent(NiftiImage& ref,
                                      NiftiImage& flo,
                                      int *mask,
                                      mat44 *transMat,
                                      size_t bytes,
                                      unsigned blockPercentage,
                                      unsigned inlierLts,
                                      unsigned blockStepSize) {
    unique_ptr<AladinContentCreator> contentCreator{ dynamic_cast<AladinContentCreator*>(this->platform->CreateContentCreator(ContentType::Aladin)) };
    this->con.reset(contentCreator->Create(ref, flo, mask, transMat, bytes, blockPercentage, inlierLts, blockStepSize));
    this->blockMatchingParams = this->con->AladinContent::GetBlockMatchingParams();
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::DeinitAladinContent() {
    this->con = nullptr;
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::ResolveMatrix(unsigned iterations, const unsigned optimizationFlag) {
    unsigned iteration = 0;
    while (iteration < iterations) {
        NR_DEBUG((optimizationFlag ? "Affine" : "Rigid") << " - level: " << this->currentLevel + 1 << "/" << this->numberOfLevels
                 << " - iteration " << iteration + 1 << "/" << iterations);
        this->GetWarpedImage(this->interpolation, this->warpedPaddingValue);
        this->UpdateTransformationMatrix(optimizationFlag);
        iteration++;
    }
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::Run() {
    this->InitialiseRegistration();

    //Main loop over the levels:
    for (this->currentLevel = 0; this->currentLevel < this->levelsToPerform; this->currentLevel++) {
        this->InitAladinContent(this->referencePyramid[currentLevel], this->floatingPyramid[currentLevel],
                                this->referenceMaskPyramid[currentLevel].get(), this->affineTransformation.get(), sizeof(T),
                                this->blockPercentage, this->inlierLts, this->blockStepSize);
        this->CreateKernels();

        // Twice more iterations are performed during the first level
        // All the blocks are used during the first level
        const unsigned maxNumberOfIterationToPerform = (currentLevel == 0) ? this->maxIterations * 2 : this->maxIterations;

        this->DebugPrintLevelInfoStart();

        if (this->con->Content::GetReference()->sform_code > 0)
            NR_MAT44_DEBUG(this->con->Content::GetReference()->sto_xyz, "Reference image matrix (sform sto_xyz)");
        else NR_MAT44_DEBUG(this->con->Content::GetReference()->qto_xyz, "Reference image matrix (qform qto_xyz)");
        if (this->con->Content::GetFloating()->sform_code > 0)
            NR_MAT44_DEBUG(this->con->Content::GetFloating()->sto_xyz, "Floating image matrix (sform sto_xyz)");
        else NR_MAT44_DEBUG(this->con->Content::GetFloating()->qto_xyz, "Floating image matrix (qform qto_xyz)");

        /* ****************** */
        /* Rigid registration */
        /* ****************** */
        if ((this->performRigid && !this->performAffine) || (this->performAffine && this->performRigid && this->currentLevel == 0)) {
            const unsigned ratio = (this->performAffine && this->performRigid && this->currentLevel == 0) ? 4 : 1;
            ResolveMatrix(maxNumberOfIterationToPerform * ratio, RIGID);
        }

        /* ******************* */
        /* Affine registration */
        /* ******************* */
        if (this->performAffine)
            ResolveMatrix(maxNumberOfIterationToPerform, AFFINE);

        // SOME CLEANING IS PERFORMED
        this->DeallocateKernels();
        this->DeinitAladinContent();
        this->DeallocateCurrentInputImage();

        this->DebugPrintLevelInfoEnd();
        NR_VERBOSE("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -");
    }

    NR_FUNC_CALLED();
}
/* *************************************************************** */
template<class T>
NiftiImage reg_aladin<T>::GetFinalWarpedImage() {
    // The initial images are used
    if (!this->inputReference || !this->inputFloating || !this->affineTransformation)
        NR_FATAL_ERROR("The reference, floating images and the transformation have to be defined");

    unique_ptr<int[]> mask(new int[this->inputReference.nVoxelsPerVolume()]());

    // The final warp uses cubic interpolation, which the CUDA resampler does not support; run it on
    // the CPU (mirroring reg_f3d, which forces the CPU platform for its final cubic warp).
    this->platformType = PlatformType::Cpu;
    this->platform.reset(new Platform(PlatformType::Cpu));

    reg_aladin<T>::InitAladinContent(this->inputReference,
                                     this->inputFloating,
                                     mask.get(),
                                     this->affineTransformation.get(),
                                     sizeof(T));
    reg_aladin<T>::CreateKernels();

    reg_aladin<T>::GetWarpedImage(3, this->warpedPaddingValue); // cubic spline interpolation

    NiftiImage warpedImage(this->con->GetWarped());
    warpedImage->cal_min = this->inputFloating->cal_min;
    warpedImage->cal_max = this->inputFloating->cal_max;
    warpedImage->scl_slope = this->inputFloating->scl_slope;
    warpedImage->scl_inter = this->inputFloating->scl_inter;

    reg_aladin<T>::DeallocateKernels();
    reg_aladin<T>::DeinitAladinContent();
    return warpedImage;
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::DebugPrintLevelInfoStart() {
    const NiftiImage& ref = this->con->Content::GetReference();
    const NiftiImage& flo = this->con->Content::GetFloating();
    NR_VERBOSE("Current level " << this->currentLevel + 1 << " / " << this->numberOfLevels);
    NR_VERBOSE("Reference image size:\t" << ref->nx << "x" << ref->ny << "x" << ref->nz << " voxels\t" <<
               ref->dx << "x" << ref->dy << "x" << ref->dz << " mm");
    NR_VERBOSE("Floating image size:\t" << flo->nx << "x" << flo->ny << "x" << flo->nz << " voxels\t" <<
               flo->dx << "x" << flo->dy << "x" << flo->dz << " mm");
    NR_VERBOSE("Block size = [4 4 " << (ref->nz == 1 ? 1 : 4) << "]");
    NR_VERBOSE("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
    NR_VERBOSE("Block number = [" << this->blockMatchingParams->blockNumber[0] << " " <<
               this->blockMatchingParams->blockNumber[1] << " " << this->blockMatchingParams->blockNumber[2] << "]");
    NR_MAT44_VERBOSE(*this->affineTransformation, "Initial transformation matrix:");
    NR_VERBOSE("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::DebugPrintLevelInfoEnd() {
    NR_MAT44_VERBOSE(*this->affineTransformation, "Final transformation matrix:");
}
/* *************************************************************** */
template class reg_aladin<float>;
