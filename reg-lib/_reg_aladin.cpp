#include "_reg_ReadWriteMatrix.h"
#include "_reg_aladin.h"
#include "_reg_stringFormat.h"
#include "Platform.h"
#include "AffineDeformationFieldKernel.h"
#include "ResampleImageKernel.h"
#include "BlockMatchingKernel.h"
#include "OptimiseKernel.h"
#include "ConvolutionKernel.h"
#include "AladinContent.h"

#ifdef _USE_CUDA
#include "CudaAladinContent.h"
#endif
#ifdef _USE_OPENCL
#include "CLAladinContent.h"
#include "InfoDevice.h"
#endif

/* *************************************************************** */
template<class T>
reg_aladin<T>::reg_aladin() {
    this->executableName = (char*)"Aladin";
    this->inputReference = nullptr;
    this->inputFloating = nullptr;
    this->inputReferenceMask = nullptr;
    this->referencePyramid = nullptr;
    this->floatingPyramid = nullptr;
    this->referenceMaskPyramid = nullptr;
    this->activeVoxelNumber = nullptr;

    this->transformationMatrix = new mat44;
    this->inputTransformName = nullptr;

    this->affineTransformation3DKernel = nullptr;
    this->blockMatchingKernel = nullptr;
    this->optimiseKernel = nullptr;
    this->resamplingKernel = nullptr;

    this->con = nullptr;
    this->blockMatchingParams = nullptr;
    this->platform = nullptr;

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

    this->interpolation = 1;

    this->floatingSigma = 0.0;
    this->referenceSigma = 0.0;

    this->referenceUpperThreshold = std::numeric_limits<T>::max();
    this->referenceLowerThreshold = -std::numeric_limits<T>::max();

    this->floatingUpperThreshold = std::numeric_limits<T>::max();
    this->floatingLowerThreshold = -std::numeric_limits<T>::max();

    this->warpedPaddingValue = std::numeric_limits<T>::quiet_NaN();

    this->funcProgressCallback = nullptr;
    this->paramsProgressCallback = nullptr;

    this->platformCode = NR_PLATFORM_CPU;
    this->currentLevel = 0;
    this->gpuIdx = 999;

#ifndef NDEBUG
    reg_print_msg_debug("reg_aladin constructor called");
#endif
}
/* *************************************************************** */
template<class T>
reg_aladin<T>::~reg_aladin() {
    if (this->transformationMatrix != nullptr)
        delete this->transformationMatrix;
    this->transformationMatrix = nullptr;

    if (this->referencePyramid != nullptr) {
        for (unsigned int l = 0; l < this->levelsToPerform; ++l) {
            if (this->referencePyramid[l] != nullptr)
                nifti_image_free(this->referencePyramid[l]);
            this->referencePyramid[l] = nullptr;
        }
        free(this->referencePyramid);
        this->referencePyramid = nullptr;
    }
    if (this->floatingPyramid != nullptr) {
        for (unsigned int l = 0; l < this->levelsToPerform; ++l) {
            if (this->floatingPyramid[l] != nullptr)
                nifti_image_free(this->floatingPyramid[l]);
            this->floatingPyramid[l] = nullptr;
        }
        free(this->floatingPyramid);
        this->floatingPyramid = nullptr;
    }
    if (this->referenceMaskPyramid != nullptr) {
        for (unsigned int l = 0; l < this->levelsToPerform; ++l) {
            if (this->referenceMaskPyramid[l] != nullptr)
                free(this->referenceMaskPyramid[l]);
            this->referenceMaskPyramid[l] = nullptr;
        }
        free(this->referenceMaskPyramid);
        this->referenceMaskPyramid = nullptr;
    }
    if (this->activeVoxelNumber != nullptr)
        free(this->activeVoxelNumber);
    if (this->platform != nullptr)
        delete this->platform;
#ifndef NDEBUG
    reg_print_msg_debug("reg_aladin destructor called");
#endif
}
/* *************************************************************** */
template<class T>
bool reg_aladin<T>::TestMatrixConvergence(mat44 *mat) {
    bool convergence = true;
    if ((fabsf(mat->m[0][0]) - 1.0f) > CONVERGENCE_EPS)
        convergence = false;
    if ((fabsf(mat->m[1][1]) - 1.0f) > CONVERGENCE_EPS)
        convergence = false;
    if ((fabsf(mat->m[2][2]) - 1.0f) > CONVERGENCE_EPS)
        convergence = false;

    if ((fabsf(mat->m[0][1]) - 0.0f) > CONVERGENCE_EPS)
        convergence = false;
    if ((fabsf(mat->m[0][2]) - 0.0f) > CONVERGENCE_EPS)
        convergence = false;
    if ((fabsf(mat->m[0][3]) - 0.0f) > CONVERGENCE_EPS)
        convergence = false;

    if ((fabsf(mat->m[1][0]) - 0.0f) > CONVERGENCE_EPS)
        convergence = false;
    if ((fabsf(mat->m[1][2]) - 0.0f) > CONVERGENCE_EPS)
        convergence = false;
    if ((fabsf(mat->m[1][3]) - 0.0f) > CONVERGENCE_EPS)
        convergence = false;

    if ((fabsf(mat->m[2][0]) - 0.0f) > CONVERGENCE_EPS)
        convergence = false;
    if ((fabsf(mat->m[2][1]) - 0.0f) > CONVERGENCE_EPS)
        convergence = false;
    if ((fabsf(mat->m[2][3]) - 0.0f) > CONVERGENCE_EPS)
        convergence = false;

    return convergence;
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::SetVerbose(bool _verbose) {
    this->verbose = _verbose;
}
/* *************************************************************** */
template<class T>
int reg_aladin<T>::Check() {
    //This does all the initial checking
    if (this->inputReference == nullptr) {
        reg_print_fct_error("reg_aladin<T>::Check()");
        reg_print_msg_error("No reference image has been specified or it can not be read");
        return EXIT_FAILURE;
    }

    if (this->inputFloating == nullptr) {
        reg_print_fct_error("reg_aladin<T>::Check()");
        reg_print_msg_error("No floating image has been specified or it can not be read");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
/* *************************************************************** */
template<class T>
int reg_aladin<T>::Print() {
    if (this->inputReference == nullptr) {
        reg_print_fct_error("reg_aladin<T>::Print()");
        reg_print_msg_error("No reference image has been specified");
        return EXIT_FAILURE;
    }
    if (this->inputFloating == nullptr) {
        reg_print_fct_error("reg_aladin<T>::Print()");
        reg_print_msg_error("No floating image has been specified");
        return EXIT_FAILURE;
    }

    /* *********************************** */
    /* DISPLAY THE REGISTRATION PARAMETERS */
    /* *********************************** */
#ifdef NDEBUG
    if (this->verbose) {
#endif
        std::string text;
        reg_print_info(this->executableName, "Parameters");
        text = stringFormat("Platform: %s", this->platform->GetName().c_str());
        reg_print_info(this->executableName, text.c_str());
        text = stringFormat("Reference image name: %s", this->inputReference->fname);
        reg_print_info(this->executableName, text.c_str());
        text = stringFormat("\t%ix%ix%i voxels", this->inputReference->nx, this->inputReference->ny, this->inputReference->nz);
        reg_print_info(this->executableName, text.c_str());
        text = stringFormat("\t%gx%gx%g mm", this->inputReference->dx, this->inputReference->dy, this->inputReference->dz);
        reg_print_info(this->executableName, text.c_str());
        text = stringFormat("Floating image name: %s", this->inputFloating->fname);
        reg_print_info(this->executableName, text.c_str());
        text = stringFormat("\t%ix%ix%i voxels", this->inputFloating->nx, this->inputFloating->ny, this->inputFloating->nz);
        reg_print_info(this->executableName, text.c_str());
        text = stringFormat("\t%gx%gx%g mm", this->inputFloating->dx, this->inputFloating->dy, this->inputFloating->dz);
        reg_print_info(this->executableName, text.c_str());
        text = stringFormat("Maximum iteration number: %i", this->maxIterations);
        reg_print_info(this->executableName, text.c_str());
        text = stringFormat("\t(%i during the first level)", 2 * this->maxIterations);
        reg_print_info(this->executableName, text.c_str());
        text = stringFormat("Percentage of blocks: %i %%", this->blockPercentage);
        reg_print_info(this->executableName, text.c_str());
        reg_print_info(this->executableName, "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
#ifdef NDEBUG
    }
#endif
    return EXIT_SUCCESS;
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::SetInputTransform(const char *filename) {
    this->inputTransformName = (char*)filename;
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::InitialiseRegistration() {
#ifndef NDEBUG
    reg_print_fct_debug("reg_aladin::InitialiseRegistration()");
#endif

    this->platform = new Platform(this->platformCode);
    this->platform->SetGpuIdx(this->gpuIdx);

    this->Print();

    // CREATE THE PYRAMID IMAGES
    this->referencePyramid = (nifti_image **)malloc(this->levelsToPerform * sizeof(nifti_image *));
    this->floatingPyramid = (nifti_image **)malloc(this->levelsToPerform * sizeof(nifti_image *));
    this->referenceMaskPyramid = (int **)malloc(this->levelsToPerform * sizeof(int *));
    this->activeVoxelNumber = (int *)malloc(this->levelsToPerform * sizeof(int));

    // FINEST LEVEL OF REGISTRATION
    reg_createImagePyramid<T>(this->inputReference,
                              this->referencePyramid,
                              this->numberOfLevels,
                              this->levelsToPerform);
    reg_createImagePyramid<T>(this->inputFloating,
                              this->floatingPyramid,
                              this->numberOfLevels,
                              this->levelsToPerform);

    if (this->inputReferenceMask != nullptr)
        reg_createMaskPyramid<T>(this->inputReferenceMask,
                                 this->referenceMaskPyramid,
                                 this->numberOfLevels,
                                 this->levelsToPerform,
                                 this->activeVoxelNumber);
    else {
        for (unsigned int l = 0; l < this->levelsToPerform; ++l) {
            this->activeVoxelNumber[l] = this->referencePyramid[l]->nx * this->referencePyramid[l]->ny * this->referencePyramid[l]->nz;
            this->referenceMaskPyramid[l] = (int *)calloc(activeVoxelNumber[l], sizeof(int));
        }
    }

    Kernel *convolutionKernel = this->platform->CreateKernel(ConvolutionKernel::GetName(), nullptr);
    // SMOOTH THE INPUT IMAGES IF REQUIRED
    for (unsigned int l = 0; l < this->levelsToPerform; l++) {
        if (this->referenceSigma != 0.0) {
            // Only the first image is smoothed
            bool *active = new bool[this->referencePyramid[l]->nt];
            float *sigma = new float[this->referencePyramid[l]->nt];
            active[0] = true;
            for (int i = 1; i < this->referencePyramid[l]->nt; ++i)
                active[i] = false;
            sigma[0] = this->referenceSigma;
            convolutionKernel->castTo<ConvolutionKernel>()->Calculate(this->referencePyramid[l], sigma, 0, nullptr, active);
            delete[] active;
            delete[] sigma;
        }
        if (this->floatingSigma != 0.0) {
            // Only the first image is smoothed
            bool *active = new bool[this->floatingPyramid[l]->nt];
            float *sigma = new float[this->floatingPyramid[l]->nt];
            active[0] = true;
            for (int i = 1; i < this->floatingPyramid[l]->nt; ++i)
                active[i] = false;
            sigma[0] = this->floatingSigma;
            convolutionKernel->castTo<ConvolutionKernel>()->Calculate(this->floatingPyramid[l], sigma, 0, nullptr, active);
            delete[] active;
            delete[] sigma;
        }
    }
    delete convolutionKernel;

    // THRESHOLD THE INPUT IMAGES IF REQUIRED
    for (unsigned int l = 0; l < this->levelsToPerform; l++) {
        reg_thresholdImage<T>(this->referencePyramid[l], this->referenceLowerThreshold, this->referenceUpperThreshold);
        reg_thresholdImage<T>(this->floatingPyramid[l], this->floatingLowerThreshold, this->floatingUpperThreshold);
    }

    // Initialise the transformation
    if (this->inputTransformName != nullptr) {
        if (FILE *aff = fopen(this->inputTransformName, "r")) {
            fclose(aff);
        } else {
            std::string text;
            text = stringFormat("The specified input affine file (%s) can not be read", this->inputTransformName);
            reg_print_fct_error("reg_aladin<T>::InitialiseRegistration()");
            reg_print_msg_error(text.c_str());
            reg_exit();
        }
        reg_tool_ReadAffineFile(this->transformationMatrix, this->inputTransformName);
    } else { // No input affine transformation
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                this->transformationMatrix->m[i][j] = 0.0;
            }
            this->transformationMatrix->m[i][i] = 1.0;
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
            reg_mat44_mul(floatingMatrix, floatingCenter, floatingRealPosition);
            float referenceRealPosition[3];
            reg_mat44_mul(referenceMatrix, referenceCenter, referenceRealPosition);
            //Set translation to the transformation matrix
            this->transformationMatrix->m[0][3] = floatingRealPosition[0] - referenceRealPosition[0];
            this->transformationMatrix->m[1][3] = floatingRealPosition[1] - referenceRealPosition[1];
            this->transformationMatrix->m[2][3] = floatingRealPosition[2] - referenceRealPosition[2];
        } else if (this->alignCentreMass == 2) {
            float referenceCentre[3] = {0, 0, 0};
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
            float refCOM[3];
            if (this->inputReference->sform_code > 0)
                reg_mat44_mul(&(this->inputReference->sto_xyz), referenceCentre, refCOM);

            float floatingCentre[3] = {0, 0, 0};
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
            float floCOM[3];
            if (this->inputFloating->sform_code > 0)
                reg_mat44_mul(&(this->inputFloating->sto_xyz), floatingCentre, floCOM);
            reg_mat44_eye(this->transformationMatrix);
            this->transformationMatrix->m[0][3] = floCOM[0] - refCOM[0];
            this->transformationMatrix->m[1][3] = floCOM[1] - refCOM[1];
            this->transformationMatrix->m[2][3] = floCOM[2] - refCOM[2];
        }
    }
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::ClearCurrentInputImage() {
    nifti_image_free(this->referencePyramid[this->currentLevel]);
    this->referencePyramid[this->currentLevel] = nullptr;

    nifti_image_free(this->floatingPyramid[this->currentLevel]);
    this->floatingPyramid[this->currentLevel] = nullptr;

    free(this->referenceMaskPyramid[this->currentLevel]);
    this->referenceMaskPyramid[this->currentLevel] = nullptr;
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::CreateKernels() {
    this->affineTransformation3DKernel = platform->CreateKernel(AffineDeformationFieldKernel::GetName(), this->con);
    this->resamplingKernel = platform->CreateKernel(ResampleImageKernel::GetName(), this->con);
    if (this->blockMatchingParams != nullptr) {
        this->blockMatchingKernel = platform->CreateKernel(BlockMatchingKernel::GetName(), this->con);
        this->optimiseKernel = platform->CreateKernel(OptimiseKernel::GetName(), this->con);
    } else {
        this->blockMatchingKernel = nullptr;
        this->optimiseKernel = nullptr;
    }
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::ClearKernels() {
    delete this->affineTransformation3DKernel;
    delete this->resamplingKernel;
    if (this->blockMatchingKernel != nullptr)
        delete this->blockMatchingKernel;
    if (this->optimiseKernel != nullptr)
        delete this->optimiseKernel;
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
    this->optimiseKernel->template castTo<OptimiseKernel>()->Calculate(type);

#ifndef NDEBUG
    reg_mat44_disp(this->transformationMatrix, (char *)"[NiftyReg DEBUG] updated forward matrix");
#endif
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::InitAladinContent(nifti_image *ref,
                                      nifti_image *flo,
                                      int *mask,
                                      mat44 *transMat,
                                      size_t bytes,
                                      unsigned int blockPercentage,
                                      unsigned int inlierLts,
                                      unsigned int blockStepSize) {
    if (this->platformCode == NR_PLATFORM_CPU)
        this->con = new AladinContent(ref, flo, mask, transMat, bytes, blockPercentage, inlierLts, blockStepSize);
#ifdef _USE_CUDA
    else if (platformCode == NR_PLATFORM_CUDA)
        this->con = new CudaAladinContent(ref, flo, mask, transMat, bytes, blockPercentage, inlierLts, blockStepSize);
#endif
#ifdef _USE_OPENCL
    else if (platformCode == NR_PLATFORM_CL)
        this->con = new ClAladinContent(ref, flo, mask, transMat, bytes, blockPercentage, inlierLts, blockStepSize);
#endif
    this->blockMatchingParams = this->con->AladinContent::GetBlockMatchingParams();
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::InitAladinContent(nifti_image *ref,
                                      nifti_image *flo,
                                      int *mask,
                                      mat44 *transMat,
                                      size_t bytes) {
    if (this->platformCode == NR_PLATFORM_CPU)
        this->con = new AladinContent(ref, flo, mask, transMat, bytes);
#ifdef _USE_CUDA
    else if (platformCode == NR_PLATFORM_CUDA)
        this->con = new CudaAladinContent(ref, flo, mask, transMat, bytes);
#endif
#ifdef _USE_OPENCL
    else if (platformCode == NR_PLATFORM_CL)
        this->con = new ClAladinContent(ref, flo, mask, transMat, bytes);
#endif
    this->blockMatchingParams = this->con->AladinContent::GetBlockMatchingParams();
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::ClearAladinContent() {
    delete this->con;
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::ResolveMatrix(unsigned int iterations, const unsigned int optimizationFlag) {
    unsigned int iteration = 0;
    while (iteration < iterations) {
#ifndef NDEBUG
        char text[255];
        sprintf(text, "%s - level: %i/%i - iteration %i/%i",
                optimizationFlag ? (char *)"Affine" : (char *)"Rigid",
                this->currentLevel + 1, this->numberOfLevels, iteration + 1, iterations);
        reg_print_msg_debug(text);
#endif
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
                                this->referenceMaskPyramid[currentLevel], this->transformationMatrix, sizeof(T), this->blockPercentage,
                                this->inlierLts, this->blockStepSize);
        this->CreateKernels();

        // Twice more iterations are performed during the first level
        // All the blocks are used during the first level
        const unsigned int maxNumberOfIterationToPerform = (currentLevel == 0) ? this->maxIterations * 2 : this->maxIterations;

#ifdef NDEBUG
        if (this->verbose) {
#endif
            this->DebugPrintLevelInfoStart();
#ifdef NDEBUG
        }
#endif

#ifndef NDEBUG
        if (this->con->GetCurrentReference()->sform_code > 0)
            reg_mat44_disp(&this->con->GetCurrentReference()->sto_xyz, (char *)"[NiftyReg DEBUG] Reference image matrix (sform sto_xyz)");
        else
            reg_mat44_disp(&this->con->GetCurrentReference()->qto_xyz, (char *)"[NiftyReg DEBUG] Reference image matrix (qform qto_xyz)");
        if (this->con->GetCurrentFloating()->sform_code > 0)
            reg_mat44_disp(&this->con->GetCurrentFloating()->sto_xyz, (char *)"[NiftyReg DEBUG] Floating image matrix (sform sto_xyz)");
        else
            reg_mat44_disp(&this->con->GetCurrentFloating()->qto_xyz, (char *)"[NiftyReg DEBUG] Floating image matrix (qform qto_xyz)");
#endif

        /* ****************** */
        /* Rigid registration */
        /* ****************** */
        if ((this->performRigid && !this->performAffine) || (this->performAffine && this->performRigid && this->currentLevel == 0)) {
            const unsigned int ratio = (this->performAffine && this->performRigid && this->currentLevel == 0) ? 4 : 1;
            ResolveMatrix(maxNumberOfIterationToPerform * ratio, RIGID);
        }

        /* ******************* */
        /* Affine registration */
        /* ******************* */
        if (this->performAffine)
            ResolveMatrix(maxNumberOfIterationToPerform, AFFINE);

        // SOME CLEANING IS PERFORMED
        this->ClearKernels();
        this->ClearAladinContent();
        this->ClearCurrentInputImage();

#ifdef NDEBUG
        if (this->verbose) {
#endif
            this->DebugPrintLevelInfoEnd();
            reg_print_info(this->executableName, "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -");
#ifdef NDEBUG
        }
#endif

    }

#ifndef NDEBUG
    reg_print_msg_debug("reg_aladin::Run() done");
#endif
    return;
}
/* *************************************************************** */
template<class T>
nifti_image* reg_aladin<T>::GetFinalWarpedImage() {
    int floatingType = this->inputFloating->datatype; //t_dev ask before touching this!
    // The initial images are used
    if (this->inputReference == nullptr || this->inputFloating == nullptr || this->transformationMatrix == nullptr) {
        reg_print_fct_error("reg_aladin::GetFinalWarpedImage()");
        reg_print_msg_error("The reference, floating images and the transformation have to be defined");
        reg_exit();
    }

    int *mask = (int *)calloc(this->inputReference->nx * this->inputReference->ny * this->inputReference->nz,
                              sizeof(int));

    reg_aladin<T>::InitAladinContent(this->inputReference,
                                     this->inputFloating,
                                     mask,
                                     this->transformationMatrix,
                                     sizeof(T));
    reg_aladin<T>::CreateKernels();

    reg_aladin<T>::GetWarpedImage(3, this->warpedPaddingValue); // cubic spline interpolation
    nifti_image *currentWarped = this->con->GetCurrentWarped(floatingType);

    free(mask);
    nifti_image *resultImage = nifti_copy_nim_info(currentWarped);
    resultImage->cal_min = this->inputFloating->cal_min;
    resultImage->cal_max = this->inputFloating->cal_max;
    resultImage->scl_slope = this->inputFloating->scl_slope;
    resultImage->scl_inter = this->inputFloating->scl_inter;
    resultImage->data = (void *)malloc(resultImage->nvox * resultImage->nbyper);
    memcpy(resultImage->data, currentWarped->data, resultImage->nvox * resultImage->nbyper);

    reg_aladin<T>::ClearKernels();
    reg_aladin<T>::ClearAladinContent();
    return resultImage;
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::DebugPrintLevelInfoStart() {
    /* Display some parameters specific to the current level */
    char text[255];
    sprintf(text, "Current level %i / %i", this->currentLevel + 1, this->numberOfLevels);
    reg_print_info(this->executableName, text);
    sprintf(text, "reference image size: \t%ix%ix%i voxels\t%gx%gx%g mm",
            this->con->GetCurrentReference()->nx,
            this->con->GetCurrentReference()->ny,
            this->con->GetCurrentReference()->nz,
            this->con->GetCurrentReference()->dx,
            this->con->GetCurrentReference()->dy,
            this->con->GetCurrentReference()->dz);
    reg_print_info(this->executableName, text);
    sprintf(text, "floating image size: \t%ix%ix%i voxels\t%gx%gx%g mm",
            this->con->GetCurrentFloating()->nx,
            this->con->GetCurrentFloating()->ny,
            this->con->GetCurrentFloating()->nz,
            this->con->GetCurrentFloating()->dx,
            this->con->GetCurrentFloating()->dy,
            this->con->GetCurrentFloating()->dz);
    reg_print_info(this->executableName, text);
    if (this->con->GetCurrentReference()->nz == 1) {
        reg_print_info(this->executableName, "Block size = [4 4 1]");
    } else reg_print_info(this->executableName, "Block size = [4 4 4]");
    reg_print_info(this->executableName, "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
    sprintf(text, "Block number = [%i %i %i]", this->blockMatchingParams->blockNumber[0],
            this->blockMatchingParams->blockNumber[1], this->blockMatchingParams->blockNumber[2]);
    reg_print_info(this->executableName, text);
    reg_mat44_disp(this->transformationMatrix, (char *)"[reg_aladin] Initial transformation matrix:");
}
/* *************************************************************** */
template<class T>
void reg_aladin<T>::DebugPrintLevelInfoEnd() {
    reg_mat44_disp(this->transformationMatrix, (char *)"[reg_aladin] Final transformation matrix:");
}
/* *************************************************************** */
