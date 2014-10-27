#include "_reg_aladin_sym.h"
#ifndef _REG_ALADIN_SYM_CPP
#define _REG_ALADIN_SYM_CPP

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
reg_aladin_sym<T>::reg_aladin_sym()
	:reg_aladin<T>::reg_aladin()
{
	this->ExecutableName = (char*) "reg_aladin_sym";

	this->InputFloatingMask = NULL;
	this->FloatingMaskPyramid = NULL;
	this->CurrentFloatingMask = NULL;
	this->BackwardActiveVoxelNumber = NULL;

	this->BackwardDeformationFieldImage = NULL;
	this->CurrentBackwardWarped = NULL;
	this->BackwardTransformationMatrix = new mat44;

	this->FloatingUpperThreshold = std::numeric_limits<T>::max();
	this->FloatingLowerThreshold = -std::numeric_limits<T>::max();

#ifndef NDEBUG
	printf("[NiftyReg DEBUG] reg_aladin_sym constructor called\n");
#endif

}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
reg_aladin_sym<T>::~reg_aladin_sym()
{
	/*this->ClearBackwardWarpedImage();
	this->ClearBackwardDeformationField();*/

	if (this->BackwardTransformationMatrix != NULL)
		delete this->BackwardTransformationMatrix;
	this->BackwardTransformationMatrix = NULL;

	if (this->FloatingMaskPyramid != NULL)
	{
		for (unsigned int i = 0; i < this->LevelsToPerform; ++i)
		{
			if (this->FloatingMaskPyramid[i] != NULL)
			{
				free(this->FloatingMaskPyramid[i]);
				this->FloatingMaskPyramid[i] = NULL;
			}
		}
		free(this->FloatingMaskPyramid);
		this->FloatingMaskPyramid = NULL;
	}
	free(this->BackwardActiveVoxelNumber);
	this->BackwardActiveVoxelNumber = NULL;

}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_aladin_sym<T>::SetInputFloatingMask(nifti_image *m)
{
	this->InputFloatingMask = m;
	return;
}

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */


/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_aladin_sym<T>::InitialiseRegistration()
{

#ifndef NDEBUG
	printf("[NiftyReg DEBUG] reg_aladin_sym::InitialiseRegistration() called\n");
#endif

	reg_aladin<T>::InitialiseRegistration();
	this->FloatingMaskPyramid = (int **)malloc(this->LevelsToPerform*sizeof(int *));
	this->BackwardActiveVoxelNumber = (int *)malloc(this->LevelsToPerform*sizeof(int));
	if (this->InputFloatingMask != NULL)
	{
		reg_createMaskPyramid<T>(this->InputFloatingMask,
			this->FloatingMaskPyramid,
			this->NumberOfLevels,
			this->LevelsToPerform,
			this->BackwardActiveVoxelNumber);
	}
	else
	{
		for (unsigned int l = 0; l < this->LevelsToPerform; ++l)
		{
			this->BackwardActiveVoxelNumber[l] = this->FloatingPyramid[l]->nx*this->FloatingPyramid[l]->ny*this->FloatingPyramid[l]->nz;
			this->FloatingMaskPyramid[l] = (int *)calloc(this->BackwardActiveVoxelNumber[l], sizeof(int));
		}
	}

	// CHECK THE THRESHOLD VALUES TO UPDATE THE MASK
	if (this->FloatingUpperThreshold != std::numeric_limits<T>::max())
	{
		for (unsigned int l = 0; l<this->LevelsToPerform; ++l)
		{
			T *refPtr = static_cast<T *>(this->FloatingPyramid[l]->data);
			int *mskPtr = this->FloatingMaskPyramid[l];
			size_t removedVoxel = 0;
			for (size_t i = 0;
				i < (size_t)this->FloatingPyramid[l]->nx*this->FloatingPyramid[l]->ny*this->FloatingPyramid[l]->nz;
				++i)
			{
				if (mskPtr[i] > -1)
				{
					if (refPtr[i]>this->FloatingUpperThreshold)
					{
						++removedVoxel;
						mskPtr[i] = -1;
					}
				}
			}
			this->BackwardActiveVoxelNumber[l] -= removedVoxel;
		}
	}
	if (this->FloatingLowerThreshold != -std::numeric_limits<T>::max())
	{
		for (unsigned int l = 0; l < this->LevelsToPerform; ++l)
		{
			T *refPtr = static_cast<T *>(this->FloatingPyramid[l]->data);
			int *mskPtr = this->FloatingMaskPyramid[l];
			size_t removedVoxel = 0;
			for (size_t i = 0;
				i < (size_t)this->FloatingPyramid[l]->nx*this->FloatingPyramid[l]->ny*this->FloatingPyramid[l]->nz;
				++i)
			{
				if (mskPtr[i] > -1)
				{
					if (refPtr[i] < this->FloatingLowerThreshold)
					{
						++removedVoxel;
						mskPtr[i] = -1;
					}
				}
			}
			this->BackwardActiveVoxelNumber[l] -= removedVoxel;
		}
	}

	//TransformationMatrix maps the initial transform from reference to floating.
	//Invert to get initial transformation from floating to reference
	*(this->BackwardTransformationMatrix) = nifti_mat44_inverse(*(this->TransformationMatrix));
	//Future todo: Square root of transform gives halfway space for both images
	//Forward will be to transform CurrentReference to HalfwayFloating
	//Backward will be to transform CurrentFloating to HalfwayReference
	//Forward transform will update Halfway Reference to new halfway space image
	//Backward transform will update Halfway Floating to new halfway space image
	//*(this->BackwardTransformationMatrix) = reg_mat44_sqrt(this->BackwardTransformationMatrix);
	//*(this->TransformationMatrix) = reg_mat44_sqrt(this->TransformationMatrix);

}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_aladin_sym<T>::GetBackwardDeformationField()
{
	bAffineTransformation3DKernel->castTo<AffineDeformationFieldKernel>()->execute();
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_aladin_sym<T>::GetWarpedImage(int interp)
{
	reg_aladin<T>::GetWarpedImage(interp);
	this->GetBackwardDeformationField();
	//TODO: This needs correction, otherwise we are transforming an image that has already been warped
	/* reg_resampleImage(this->CurrentReference,
					   this->CurrentBackwardWarped,
					   this->BackwardDeformationFieldImage,
					   this->CurrentFloatingMask,
					   interp,
					   0);*/
	bResamplingKernel->castTo<ResampleImageKernel>()->execute(interp, 0);
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_aladin_sym<T>::UpdateTransformationMatrix(int type)
{
	// Update first the forward transformation matrix
	/*block_matching_method(this->CurrentReference, this->CurrentWarped, this->blockMatchingParams, this->CurrentReferenceMask);
	optimize(this->blockMatchingParams, this->TransformationMatrix, type == RIGID);*/

	blockMatchingKernel->castTo<BlockMatchingKernel>()->execute();//watch the trans matrix!!!!!!
	optimiseKernel->castTo<OptimiseKernel>()->execute(type == AFFINE);


	// Update now the backward transformation matrix
	/*block_matching_method(this->CurrentFloating, &this->BackwardBlockMatchingParams, this->CurrentFloatingMask);
	optimize(&this->BackwardBlockMatchingParams, this->BackwardTransformationMatrix, type == RIGID);*/
	bBlockMatchingKernel->castTo<BlockMatchingKernel>()->execute();//watch the trans matrix!!!!!!
	bOptimiseKernel->castTo<OptimiseKernel>()->execute(type == AFFINE);

	// Forward and backward matrix are inverted
	mat44 fInverted = nifti_mat44_inverse(*(this->TransformationMatrix));
	mat44 bInverted = nifti_mat44_inverse(*(this->BackwardTransformationMatrix));

	// We average the forward and inverted backward matrix
	*(this->TransformationMatrix) = reg_mat44_avg2(this->TransformationMatrix, &bInverted );
	// We average the inverted forward and backward matrix
	*(this->BackwardTransformationMatrix) = reg_mat44_avg2(&fInverted, this->BackwardTransformationMatrix );
#ifndef NDEBUG
	reg_mat44_disp(this->TransformationMatrix, (char *)"[DEBUG] updated forward transformation matrix");
	reg_mat44_disp(this->BackwardTransformationMatrix, (char *)"[DEBUG] updated backward transformation matrix");
#endif
}


template <class T>
void reg_aladin_sym<T>::clearContext(){
	delete this->con;
	delete this->backCon;
}
template <class T>
void reg_aladin_sym<T>::initContext(){
	reg_aladin<T>::initContext();

	if (platformCode == 0)
		this->backCon = new Context(this->FloatingPyramid[CurrentLevel], this->ReferencePyramid[CurrentLevel], this->FloatingMaskPyramid[CurrentLevel], sizeof(T), this->BlockPercentage, InlierLts);
	else if (platformCode == 1)
		this->backCon = new CudaContext(this->FloatingPyramid[CurrentLevel], this->ReferencePyramid[CurrentLevel], this->FloatingMaskPyramid[CurrentLevel], sizeof(T), this->BlockPercentage, InlierLts);
	else
		this->backCon = new Context(this->FloatingPyramid[CurrentLevel], this->ReferencePyramid[CurrentLevel], this->FloatingMaskPyramid[CurrentLevel], sizeof(T), this->BlockPercentage, InlierLts);


	
	this->backCon->setTransformationMatrix(this->BackwardTransformationMatrix);
	this->BackwardBlockMatchingParams = this->backCon->getBlockMatchingParams();
	this->CurrentBackwardWarped = this->backCon->getCurrentWarped();
	this->BackwardDeformationFieldImage = this->backCon->getCurrentDeformationField();
}


/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_aladin_sym<T>::ClearCurrentInputImage()
{
	reg_aladin<T>::ClearCurrentInputImage();
	if (this->FloatingMaskPyramid[this->CurrentLevel] != NULL)
		free(this->FloatingMaskPyramid[this->CurrentLevel]);
	this->FloatingMaskPyramid[this->CurrentLevel] = NULL;
	this->CurrentFloatingMask = NULL;

	/*this->ClearBackwardWarpedImage();
	this->ClearBackwardDeformationField();*/
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

template <class T>
void reg_aladin_sym<T>::clearKernels()
{
	reg_aladin<T>::clearKernels();
	delete bResamplingKernel;
	delete bAffineTransformation3DKernel;
	if (backCon->bm){
		delete bBlockMatchingKernel;
		delete bOptimiseKernel;
	}
}

template <class T>
void reg_aladin_sym<T>::DebugPrintLevelInfoStart()
{
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
	printf("[%s] Forward Block number = [%i %i %i]\n", this->ExecutableName, this->blockMatchingParams->blockNumber[0], this->blockMatchingParams->blockNumber[1], this->blockMatchingParams->blockNumber[2]);
	printf("[%s] Backward Block number = [%i %i %i]\n", this->ExecutableName, this->BackwardBlockMatchingParams->blockNumber[0], this->BackwardBlockMatchingParams->blockNumber[1], this->BackwardBlockMatchingParams->blockNumber[2]);
	reg_mat44_disp(this->TransformationMatrix, (char *)"[reg_aladin_sym] Initial forward transformation matrix:");
	reg_mat44_disp(this->BackwardTransformationMatrix, (char *)"[reg_aladin_sym] Initial backward transformation matrix:");
	printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");

}
template <class T>
void reg_aladin_sym<T>::createKernels(){
	bAffineTransformation3DKernel = platform->createKernel (AffineDeformationFieldKernel::Name(), this->backCon);
	bBlockMatchingKernel = platform->createKernel(BlockMatchingKernel::Name(), this->backCon);
	bResamplingKernel = platform->createKernel(ResampleImageKernel::Name(), this->backCon);
	bOptimiseKernel = platform->createKernel(OptimiseKernel::Name(), this->backCon);

	reg_aladin<T>::createKernels();
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_aladin_sym<T>::DebugPrintLevelInfoEnd()
{
	reg_mat44_disp(this->TransformationMatrix,
		(char *)"[reg_aladin_sym] Final forward transformation matrix:");
	reg_mat44_disp(this->BackwardTransformationMatrix,
		(char *)"[reg_aladin_sym] Final backward transformation matrix:");
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
#endif //REG_ALADIN_SYM_CPP
