#include "_reg_aladin.h"
//#include "Context.h"
#include "CudaContext.h"
#include "CPUPlatform.h"
#include "CLPlatform.h"
#include "CudaPlatform.h"
#include "kernels.h"

#ifndef _REG_ALADIN_CPP
#define _REG_ALADIN_CPP


/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T> reg_aladin<T>::reg_aladin()
{
	this->ExecutableName = (char*) "Aladin";
	this->InputReference = NULL;
	this->InputFloating = NULL;
	this->InputReferenceMask = NULL;
	this->CurrentReference = NULL;
	this->CurrentFloating = NULL;
	this->CurrentWarped = NULL;
	this->CurrentReferenceMask = NULL;
	this->ReferencePyramid = NULL;
	this->FloatingPyramid = NULL;
	this->ReferenceMaskPyramid = NULL;
	this->CurrentWarped = NULL;
	this->deformationFieldImage = NULL;
	this->activeVoxelNumber = NULL;

	this->deformationFieldImage = NULL;
	TransformationMatrix = new mat44;
	InputTransformName = NULL;

	this->Verbose = true;

	this->MaxIterations = 5;

	this->NumberOfLevels = 3;
	this->LevelsToPerform = 3;

	this->PerformRigid = 1;
	this->PerformAffine = 1;

	this->BlockPercentage = 50;
	this->InlierLts = 50;

	this->AlignCentre = 1;

	this->Interpolation = 1;

	this->FloatingSigma = 0.0;
	this->ReferenceSigma = 0.0;

	this->ReferenceUpperThreshold = std::numeric_limits<T>::max();
	this->ReferenceLowerThreshold = -std::numeric_limits<T>::max();

	this->funcProgressCallback = NULL;
	this->paramsProgressCallback = NULL;

	this->platformCode = 1;

	if (platformCode == 0)
		this->platform = new CPUPlatform();
	else if (platformCode == 1)
		this->platform = new CudaPlatform();
	else
		this->platform = new CLPlatform();
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T> reg_aladin<T>::~reg_aladin()
{
	/*this->ClearWarpedImage();
	this->ClearDeformationField();*/

	if (this->TransformationMatrix != NULL)
		delete this->TransformationMatrix;
	this->TransformationMatrix = NULL;


	for (unsigned int l = 0; l < this->LevelsToPerform; ++l)
	{
		nifti_image_free(this->ReferencePyramid[l]);
		this->ReferencePyramid[l] = NULL;
		nifti_image_free(this->FloatingPyramid[l]);
		this->FloatingPyramid[l] = NULL;
		free(this->ReferenceMaskPyramid[l]);
		this->ReferenceMaskPyramid[l] = NULL;
	}
	free(this->ReferencePyramid);
	this->ReferencePyramid = NULL;
	free(this->FloatingPyramid);
	this->FloatingPyramid = NULL;
	free(this->ReferenceMaskPyramid);
	this->ReferenceMaskPyramid = NULL;
	free(activeVoxelNumber);
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

template <class T>
void reg_aladin<T>::ClearCurrentInputImage()
{
	nifti_image_free(this->ReferencePyramid[this->CurrentLevel]);
	this->ReferencePyramid[this->CurrentLevel] = NULL;
	nifti_image_free(this->FloatingPyramid[this->CurrentLevel]);
	this->FloatingPyramid[this->CurrentLevel] = NULL;
	free(this->ReferenceMaskPyramid[this->CurrentLevel]);
	this->ReferenceMaskPyramid[this->CurrentLevel] = NULL;
	this->CurrentReference = NULL;
	this->CurrentFloating = NULL;
	this->CurrentReferenceMask = NULL;

	/*this->ClearWarpedImage();
	this->ClearDeformationField();*/
}


template <class T>
bool reg_aladin<T>::TestMatrixConvergence(mat44 *mat)
{
	bool convergence = true;
	if ((fabsf(mat->m[0][0]) - 1.0f) > CONVERGENCE_EPS) convergence = false;
	if ((fabsf(mat->m[1][1]) - 1.0f) > CONVERGENCE_EPS) convergence = false;
	if ((fabsf(mat->m[2][2]) - 1.0f) > CONVERGENCE_EPS) convergence = false;

	if ((fabsf(mat->m[0][1]) - 0.0f) > CONVERGENCE_EPS) convergence = false;
	if ((fabsf(mat->m[0][2]) - 0.0f) > CONVERGENCE_EPS) convergence = false;
	if ((fabsf(mat->m[0][3]) - 0.0f) > CONVERGENCE_EPS) convergence = false;

	if ((fabsf(mat->m[1][0]) - 0.0f) > CONVERGENCE_EPS) convergence = false;
	if ((fabsf(mat->m[1][2]) - 0.0f) > CONVERGENCE_EPS) convergence = false;
	if ((fabsf(mat->m[1][3]) - 0.0f) > CONVERGENCE_EPS) convergence = false;

	if ((fabsf(mat->m[2][0]) - 0.0f) > CONVERGENCE_EPS) convergence = false;
	if ((fabsf(mat->m[2][1]) - 0.0f) > CONVERGENCE_EPS) convergence = false;
	if ((fabsf(mat->m[2][3]) - 0.0f) > CONVERGENCE_EPS) convergence = false;

	return convergence;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_aladin<T>::SetVerbose(bool _verbose)
{
	this->Verbose = _verbose;
}

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_aladin<T>::Check()
{
	//This does all the initial checking
	if (this->InputReference == NULL)
	{
		fprintf(stderr, "[NiftyReg ERROR] No reference image has been specified or it can not be read\n");
		return 1;
	}
	reg_checkAndCorrectDimension(this->InputReference);

	if (this->InputFloating == NULL)
	{
		fprintf(stderr, "[NiftyReg ERROR] No floating image has been specified or it can not be read\n");
		return 1;
	}
	reg_checkAndCorrectDimension(this->InputFloating);

	return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_aladin<T>::Print()
{
	if (this->InputReference == NULL)
	{
		fprintf(stderr, "[NiftyReg ERROR] No reference image has been specified\n");
		return 1;
	}
	if (this->InputFloating == NULL)
	{
		fprintf(stderr, "[NiftyReg ERROR] No floating image has been specified\n");
		return 1;
	}

	/* *********************************** */
	/* DISPLAY THE REGISTRATION PARAMETERS */
	/* *********************************** */
#ifdef NDEBUG
	if (this->Verbose)
	{
#endif
		printf("[%s] Parameters\n", this->ExecutableName);
		printf("[%s] Reference image name: %s\n", this->ExecutableName, this->InputReference->fname);
		printf("[%s] \t%ix%ix%i voxels\n", this->ExecutableName, this->InputReference->nx, this->InputReference->ny, this->InputReference->nz);
		printf("[%s] \t%gx%gx%g mm\n", this->ExecutableName, this->InputReference->dx, this->InputReference->dy, this->InputReference->dz);
		printf("[%s] Floating image name: %s\n", this->ExecutableName, this->InputFloating->fname);
		printf("[%s] \t%ix%ix%i voxels\n", this->ExecutableName, this->InputFloating->nx, this->InputFloating->ny, this->InputFloating->nz);
		printf("[%s] \t%gx%gx%g mm\n", this->ExecutableName, this->InputFloating->dx, this->InputFloating->dy, this->InputFloating->dz);
		printf("[%s] Maximum iteration number: %i", this->ExecutableName, this->MaxIterations);
		printf(" (%i during the first level)\n", 2 * this->MaxIterations);
		printf("[%s] Percentage of blocks: %i %%\n", this->ExecutableName, this->BlockPercentage);
		printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
#ifdef NDEBUG
	}
#endif
	return 0;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_aladin<T>::SetInputTransform(const char *filename)
{
	this->InputTransformName = (char *)filename;
	return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_aladin<T>::InitialiseRegistration()
{
#ifndef NDEBUG
	printf("[NiftyReg DEBUG] reg_aladin::InitialiseRegistration() called\n");
#endif

	convolutionKernel = platform->createKernel(ConvolutionKernel::Name(), NULL);
	this->Print();

	// CREATE THE PYRAMIDE IMAGES
	this->ReferencePyramid = (nifti_image **)malloc(this->LevelsToPerform*sizeof(nifti_image *));
	this->FloatingPyramid = (nifti_image **)malloc(this->LevelsToPerform*sizeof(nifti_image *));
	this->ReferenceMaskPyramid = (int **)malloc(this->LevelsToPerform*sizeof(int *));
	this->activeVoxelNumber = (int *)malloc(this->LevelsToPerform*sizeof(int));

	// FINEST LEVEL OF REGISTRATION
	reg_createImagePyramid<T>(this->InputReference, this->ReferencePyramid, this->NumberOfLevels, this->LevelsToPerform);
	reg_createImagePyramid<T>(this->InputFloating, this->FloatingPyramid, this->NumberOfLevels, this->LevelsToPerform);
	if (this->InputReferenceMask != NULL)
		reg_createMaskPyramid<T>(this->InputReferenceMask, this->ReferenceMaskPyramid, this->NumberOfLevels, this->LevelsToPerform, this->activeVoxelNumber);
	else
	{
		for (unsigned int l = 0; l < this->LevelsToPerform; ++l)
		{
			this->activeVoxelNumber[l] = this->ReferencePyramid[l]->nx*this->ReferencePyramid[l]->ny*this->ReferencePyramid[l]->nz;
			this->ReferenceMaskPyramid[l] = (int *)calloc(activeVoxelNumber[l], sizeof(int));
		}
	}

	// CHECK THE THRESHOLD VALUES TO UPDATE THE MASK
	if (this->ReferenceUpperThreshold != std::numeric_limits<T>::max())
	{
		for (unsigned int l = 0; l<this->LevelsToPerform; ++l)
		{
			T *refPtr = static_cast<T *>(this->ReferencePyramid[l]->data);
			int *mskPtr = this->ReferenceMaskPyramid[l];
			size_t removedVoxel = 0;
			for (size_t i = 0; i<(size_t)this->ReferencePyramid[l]->nx*this->ReferencePyramid[l]->ny*this->ReferencePyramid[l]->nz; ++i)
			{
				if (mskPtr[i]>-1)
				{
					if (refPtr[i]>this->ReferenceUpperThreshold)
					{
						++removedVoxel;
						mskPtr[i] = -1;
					}
				}
			}
			this->activeVoxelNumber[l] -= removedVoxel;
		}
	}
	if (this->ReferenceLowerThreshold != -std::numeric_limits<T>::max())
	{
		for (unsigned int l = 0; l < this->LevelsToPerform; ++l)
		{
			T *refPtr = static_cast<T *>(this->ReferencePyramid[l]->data);
			int *mskPtr = this->ReferenceMaskPyramid[l];
			size_t removedVoxel = 0;
			for (size_t i = 0; i<(size_t)this->ReferencePyramid[l]->nx*this->ReferencePyramid[l]->ny*this->ReferencePyramid[l]->nz; ++i)
			{
				if (mskPtr[i]>-1)
				{
					if (refPtr[i] < this->ReferenceLowerThreshold)
					{
						++removedVoxel;
						mskPtr[i] = -1;
					}
				}
			}
			this->activeVoxelNumber[l] -= removedVoxel;
		}
	}

	// SMOOTH THE INPUT IMAGES IF REQUIRED
	for (unsigned int l = 0; l < this->LevelsToPerform; l++)
	{
		if (this->ReferenceSigma != 0.0)
		{
			// Only the first image is smoothed
			bool *active = new bool[this->ReferencePyramid[l]->nt];
			float *sigma = new float[this->ReferencePyramid[l]->nt];
			active[0] = true;
			for (int i = 1; i < this->ReferencePyramid[l]->nt; ++i)
				active[i] = false;
			sigma[0] = this->ReferenceSigma;
			//kernel1
			//reg_tools_kernelConvolution(co,this->FloatingPyramid[l], sigma, 0, NULL, active);

			convolutionKernel->castTo<ConvolutionKernel>()->execute(this->FloatingPyramid[l], sigma, 0, NULL, active);

			delete[]active;
			delete[]sigma;
			delete convolutionKernel;
		}
		if (this->FloatingSigma != 0.0)
		{
			// Only the first image is smoothed
			bool *active = new bool[this->FloatingPyramid[l]->nt];
			float *sigma = new float[this->FloatingPyramid[l]->nt];
			active[0] = true;
			for (int i = 1; i < this->FloatingPyramid[l]->nt; ++i)
				active[i] = false;
			sigma[0] = this->FloatingSigma;
			//reg_tools_kernelConvolution(co, this->FloatingPyramid[l], sigma, 0, NULL, active);

			convolutionKernel->castTo<ConvolutionKernel>()->execute(this->FloatingPyramid[l], sigma, 0, NULL, active);

			delete[]active;
			delete[]sigma;
			delete convolutionKernel;
		}
	}

	// Initialise the transformation
	if (this->InputTransformName != NULL)
	{
		if (FILE *aff = fopen(this->InputTransformName, "r"))
		{
			fclose(aff);
		}
		else
		{
			fprintf(stderr, "The specified input affine file (%s) can not be read\n", this->InputTransformName);
			reg_exit(1);
		}
		reg_tool_ReadAffineFile(this->TransformationMatrix, this->InputTransformName);
	}
	else  // No input affine transformation
	{
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				this->TransformationMatrix->m[i][j] = 0.0;
			}
			this->TransformationMatrix->m[i][i] = 1.0;
		}
		if (this->AlignCentre)
		{
			mat44 *floatingMatrix;
			if (this->InputFloating->sform_code>0)
				floatingMatrix = &(this->InputFloating->sto_xyz);
			else floatingMatrix = &(this->InputFloating->qto_xyz);
			mat44 *referenceMatrix;
			if (this->InputReference->sform_code>0)
				referenceMatrix = &(this->InputReference->sto_xyz);
			else referenceMatrix = &(this->InputReference->qto_xyz);
			float floatingCenter[3];
			floatingCenter[0] = (float)(this->InputFloating->nx) / 2.0f;
			floatingCenter[1] = (float)(this->InputFloating->ny) / 2.0f;
			floatingCenter[2] = (float)(this->InputFloating->nz) / 2.0f;
			float referenceCenter[3];
			referenceCenter[0] = (float)(this->InputReference->nx) / 2.0f;
			referenceCenter[1] = (float)(this->InputReference->ny) / 2.0f;
			referenceCenter[2] = (float)(this->InputReference->nz) / 2.0f;
			float floatingRealPosition[3];
			reg_mat44_mul(floatingMatrix, floatingCenter, floatingRealPosition);
			float referenceRealPosition[3];
			reg_mat44_mul(referenceMatrix, referenceCenter, referenceRealPosition);
			this->TransformationMatrix->m[0][3] = floatingRealPosition[0] - referenceRealPosition[0];
			this->TransformationMatrix->m[1][3] = floatingRealPosition[1] - referenceRealPosition[1];
			this->TransformationMatrix->m[2][3] = floatingRealPosition[2] - referenceRealPosition[2];

		}
	}
}
template <class T>
void reg_aladin<T>::createKernels(){
	affineTransformation3DKernel = platform->createKernel(AffineDeformationFieldKernel::Name(), this->con);
	resamplingKernel = platform->createKernel(ResampleImageKernel::Name(), this->con);
	if (con->bm){
		blockMatchingKernel = platform->createKernel(BlockMatchingKernel::Name(), this->con);
		optimiseKernel = platform->createKernel(OptimiseKernel::Name(), this->con);
	}
}

template <class T>
void reg_aladin<T>::clearKernels(){
	delete affineTransformation3DKernel;
	delete resamplingKernel;
	if (con->bm){
		delete blockMatchingKernel;
		delete optimiseKernel;
	}
}

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_aladin<T>::InitialiseBlockMatching(int CurrentPercentageOfBlockToUse)
{
	initialise_block_matching_method(CurrentReference, blockMatchingParams, CurrentPercentageOfBlockToUse, InlierLts, CurrentReferenceMask, false);

}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_aladin<T>::GetDeformationField()
{
	affineTransformation3DKernel->castTo<AffineDeformationFieldKernel>()->execute();
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_aladin<T>::GetWarpedImage(int interp)
{
	this->GetDeformationField();
	resamplingKernel->castTo<ResampleImageKernel>()->execute(interp, 0);
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_aladin<T>::UpdateTransformationMatrix(int type)
{


	//tempcode 
	/*nifti_image *input1 = CurrentReference;
	const char* input1Name = ( char * )"input_CurrentReference_BlockMatchingKernel.nii";
	reg_io_WriteImageFile(input1, input1Name);*/

	//tempcode 
	/*nifti_image *input2 = CurrentWarped;
	const char* input2Name = ( char * )"input_CurrentWarped_BlockMatchingKernel.nii";
	reg_io_WriteImageFile(input2, input2Name);*/

	blockMatchingKernel->castTo<BlockMatchingKernel>()->execute();//watch the trans matrix!!!!!!

	//reg_mat44_disp(this->TransformationMatrix, (char *)"[DEBUG] pre matrix");
	optimiseKernel->castTo<OptimiseKernel>()->execute(type == AFFINE);
	//reg_mat44_disp(this->TransformationMatrix, (char *)"[DEBUG] after matrix");


#ifndef NDEBUG
	reg_mat44_disp(this->TransformationMatrix, (char *)"[DEBUG] updated matrix");
#endif
}
template <class T>
void reg_aladin<T>::clearContext(){
	delete con;
}
template <class T>
void reg_aladin<T>::initContext(){

	if (platformCode == 0)
		this->con = new Context(this->ReferencePyramid[CurrentLevel], this->FloatingPyramid[CurrentLevel], this->ReferenceMaskPyramid[CurrentLevel], sizeof(T), this->BlockPercentage, InlierLts);
	else if (platformCode == 1)
		this->con = new CudaContext(this->ReferencePyramid[CurrentLevel], this->FloatingPyramid[CurrentLevel], this->ReferenceMaskPyramid[CurrentLevel], sizeof(T), this->BlockPercentage, InlierLts);
	else
		this->con = new Context(this->ReferencePyramid[CurrentLevel], this->FloatingPyramid[CurrentLevel], this->ReferenceMaskPyramid[CurrentLevel], sizeof(T), this->BlockPercentage, InlierLts);

	this->CurrentReference = con->getCurrentReference();
	this->CurrentFloating = con->getCurrentFloating();
	//this->CurrentReferenceMask = con->getCurrentReferenceMask();
	//this->CurrentWarped = con->getCurrentWarped();
	//this->deformationFieldImage = con->getCurrentDeformationField();
	this->blockMatchingParams = con->getBlockMatchingParams();
	con->setTransformationMatrix(this->TransformationMatrix);

}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_aladin<T>::Run()
{
	// Initialise the registration parameters
	this->InitialiseRegistration();

	// Compute the resolution of the progress bar
	unsigned long iProgressStep = 1;
	const unsigned long nProgressSteps = (PerformAffine && PerformRigid) ? MaxIterations * 4 * 2 + MaxIterations*(LevelsToPerform + 1) : MaxIterations*(LevelsToPerform + 1);

	// Compute the progress unit
	const unsigned long progressUnit = (unsigned long)ceil((float)nProgressSteps / 100.0f);
	//Main loop over the levels:
	for (this->CurrentLevel = 0; this->CurrentLevel < this->LevelsToPerform; this->CurrentLevel++)
	{
		std::cout << "====================" << CurrentLevel << ": " << LevelsToPerform << "================================ " << std::endl;

		//this->InitialiseBlockMatching(this->BlockPercentage);
		this->initContext();
		this->createKernels();
		//this->TransformationMatrix = con->getTransformationMatrix();
		// Twice more iterations are performed during the first level
		// All the blocks are used during the first level
		int maxNumberOfIterationToPerform = this->MaxIterations;
		if (CurrentLevel == 0)
		{
			maxNumberOfIterationToPerform *= 2;
			//            percentageOfBlockToUse=100;
		}

		/* initialise the block matching */



#ifdef NDEBUG
		if (this->Verbose)
		{
#endif
			this->DebugPrintLevelInfoStart();
#ifdef NDEBUG
		}
#endif

#ifndef NDEBUG
		if(this->CurrentReference->sform_code>0)
			reg_mat44_disp(&this->CurrentReference->sto_xyz, (char *)"[DEBUG] Reference image matrix (sform sto_xyz)");
		else reg_mat44_disp(&this->CurrentReference->qto_xyz, (char *)"[DEBUG] Reference image matrix (qform qto_xyz)");
		if (this->CurrentFloating->sform_code > 0)
			reg_mat44_disp(&this->CurrentFloating->sto_xyz, (char *)"[DEBUG] Floating image matrix (sform sto_xyz)");
		else reg_mat44_disp(&this->CurrentFloating->qto_xyz, (char *)"[DEBUG] Floating image matrix (qform qto_xyz)");
#endif
		/* ****************** */
		/* Rigid registration */
		/* ****************** */
		//std::cout << "Rigid!" << std::endl;
		int iteration = 0;
		if ((this->PerformRigid && !this->PerformAffine) || (this->PerformAffine && this->PerformRigid && CurrentLevel == 0))
		{
			const unsigned int ratio = (PerformAffine && PerformRigid && CurrentLevel == 0) ? 4 : 1;
			while (iteration < maxNumberOfIterationToPerform*ratio)
			{
#ifndef NDEBUG
				printf("[DEBUG] -Rigid- iteration %i\n", iteration);
#endif
				this->GetWarpedImage(this->Interpolation);
				this->UpdateTransformationMatrix(RIGID);
				if (funcProgressCallback && paramsProgressCallback)
				{
					(*funcProgressCallback)(100.0f * (float)iProgressStep / (float)nProgressSteps, paramsProgressCallback);
				}

				// Announce the progress via CLI
				if ((int)(iProgressStep % progressUnit) == 0) progressXML(100 * iProgressStep / nProgressSteps, "Performing Rigid Registration...");

				iteration++;
				iProgressStep++;
			}
		}

		/* ******************* */
		/* Affine registration */
		/* ******************* */
		iteration = 0;
		if (this->PerformAffine)
		{
			while (iteration < maxNumberOfIterationToPerform)
			{
#ifndef NDEBUG
				printf("[DEBUG] -Affine- iteration %i\n", iteration);
#endif
				//bool print = iteration == 1;
				this->GetWarpedImage(this->Interpolation);
				this->UpdateTransformationMatrix(AFFINE);
				if (funcProgressCallback && paramsProgressCallback)
				{
					(*funcProgressCallback)(100.0f * (float)iProgressStep / (float)nProgressSteps,
						paramsProgressCallback);
				}

				// Announce the progress via CLI
				if ((int)(iProgressStep % progressUnit) == 0)
				{
					progressXML(100 * iProgressStep / nProgressSteps, "Performing Affine Registration...");
				}

				iteration++;
				iProgressStep++;

			}
		}

		// SOME CLEANING IS PERFORMED

		this->ClearCurrentInputImage();
		this->clearKernels();
		delete con;


#ifdef NDEBUG
		if (this->Verbose)
		{
#endif
			this->DebugPrintLevelInfoEnd();
			printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
#ifdef NDEBUG
		}
#endif

	} // level this->LevelsToPerform

	if (funcProgressCallback && paramsProgressCallback)
	{
		(*funcProgressCallback)(100., paramsProgressCallback);
	}

#ifndef NDEBUG
	printf("[NiftyReg DEBUG] reg_aladin::Run() done\n");
#endif
	return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
nifti_image *reg_aladin<T>::GetFinalWarpedImage()
{
	// The initial images are used
	if (this->InputReference == NULL ||
		this->InputFloating == NULL ||
		this->TransformationMatrix == NULL)
	{
		fprintf(stderr, "[NiftyReg ERROR] reg_aladin::GetWarpedImage()\n");
		fprintf(stderr, " * The reference, floating images and the transformation have to be defined\n");
	}

	this->CurrentReference = this->InputReference;
	this->CurrentFloating = this->InputFloating;
	reg_tools_changeDatatype<T>(CurrentReference);
	reg_tools_changeDatatype<T>(CurrentFloating);

	/*std::cout << "sze: " << CurrentReference->nx*CurrentReference->ny*CurrentReference->nz << std::endl;
	float* data = static_cast<float*>(CurrentReference->data);
	std::cout << "val: " << data[CurrentReference->nx*CurrentReference->ny*CurrentReference->nz - 1] << std::endl;*/
	this->CurrentReferenceMask = (int *)calloc(CurrentReference->nx*CurrentReference->ny*CurrentReference->nz, sizeof(int));

	if (platformCode == 0)
		this->con = new Context(this->InputReference, this->InputFloating, this->CurrentReferenceMask, sizeof(T)/*, 50, 50*/);
	else if (platformCode == 1)
		this->con = new CudaContext(this->InputReference, this->InputFloating, this->CurrentReferenceMask, sizeof(T)/*, 50, 50*/);
	else
		this->con = new Context(this->InputReference, this->InputFloating, this->CurrentReferenceMask, sizeof(T)/*, 50, 50*/);




	this->CurrentWarped = con->getCurrentWarped();
	this->deformationFieldImage = con->getCurrentDeformationField();

	con->setTransformationMatrix(this->TransformationMatrix);
	this->createKernels();

	/*reg_aladin<T>::AllocateWarpedImage();
	reg_aladin<T>::AllocateDeformationField();*/

	reg_aladin<T>::GetWarpedImage(3); // cubic spline interpolation
	//reg_aladin<T>::ClearDeformationField();

	nifti_image *resultImage = nifti_copy_nim_info(this->CurrentWarped);
	resultImage->cal_min = this->InputFloating->cal_min;
	resultImage->cal_max = this->InputFloating->cal_max;
	resultImage->scl_slope = this->InputFloating->scl_slope;
	resultImage->scl_inter = this->InputFloating->scl_inter;
	resultImage->data = (void *)malloc(resultImage->nvox*resultImage->nbyper);
	memcpy(resultImage->data, this->CurrentWarped->data, resultImage->nvox*resultImage->nbyper);

	//reg_aladin<T>::ClearWarpedImage();

	this->clearKernels();
	delete con;
	delete platform;
	return resultImage;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_aladin<T>::DebugPrintLevelInfoStart()
{
	/* Display some parameters specific to the current level */
	printf("[%s] Current level %i / %i\n", this->ExecutableName, this->CurrentLevel + 1, this->NumberOfLevels);
	printf("[%s] reference image size: \t%ix%ix%i voxels\t%gx%gx%g mm\n", this->ExecutableName,
		this->CurrentReference->nx, this->CurrentReference->ny, this->CurrentReference->nz,
		this->CurrentReference->dx, this->CurrentReference->dy, this->CurrentReference->dz);
	printf("[%s] floating image size: \t%ix%ix%i voxels\t%gx%gx%g mm\n", this->ExecutableName,
		this->CurrentFloating->nx, this->CurrentFloating->ny, this->CurrentFloating->nz,
		this->CurrentFloating->dx, this->CurrentFloating->dy, this->CurrentFloating->dz);
	if (this->CurrentReference->nz == 1)
		printf("[%s] Block size = [4 4 1]\n", this->ExecutableName);
	else printf("[%s] Block size = [4 4 4]\n", this->ExecutableName);
	printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
	printf("[%s] Block number = [%i %i %i]\n", this->ExecutableName, this->blockMatchingParams->blockNumber[0],
		this->blockMatchingParams->blockNumber[1], this->blockMatchingParams->blockNumber[2]);
	reg_mat44_disp(this->TransformationMatrix, (char *)"[reg_aladin] Initial transformation matrix:");
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_aladin<T>::DebugPrintLevelInfoEnd()
{
	reg_mat44_disp(this->TransformationMatrix, (char *)"[reg_aladin] Final transformation matrix:");
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

#endif //#ifndef _REG_ALADIN_CPP
