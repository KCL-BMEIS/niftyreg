/*
 *  _reg_affineTransformation.cpp
 *
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_AFFINETRANSFORMATION_CPP
#define _REG_AFFINETRANSFORMATION_CPP

#include "_reg_affineTransformation.h"

/* *************************************************************** */
mat44 reg_mat44_mul(mat44 *A, mat44 *B)
{
	mat44 R;
	
	for(int i=0; i<4; i++){
		for(int j=0; j<4; j++){
			R.m[i][j] = A->m[i][0]*B->m[0][j] + A->m[i][1]*B->m[1][j] + A->m[i][2]*B->m[2][j] + A->m[i][3]*B->m[3][j];
		}
	}
	
	return R;
}
/* *************************************************************** */
mat44 reg_mat44_add(mat44 *A, mat44 *B)
{
    mat44 R;

    for(int i=0; i<4; i++){
        for(int j=0; j<4; j++){
            R.m[i][j] = A->m[i][j]+B->m[i][j];
        }
    }

    return R;
}
/* *************************************************************** */
void reg_mat44_mul(	mat44 *mat,
                    float in[3],
                    float out[3])
{
    out[0]=mat->m[0][0]*in[0] + mat->m[0][1]*in[1] + mat->m[0][2]*in[2] + mat->m[0][3];
    out[1]=mat->m[1][0]*in[0] + mat->m[1][1]*in[1] + mat->m[1][2]*in[2] + mat->m[1][3];
    out[2]=mat->m[2][0]*in[0] + mat->m[2][1]*in[1] + mat->m[2][2]*in[2] + mat->m[2][3];
    return;
}
/* *************************************************************** */
void reg_mat44_disp(mat44 *mat, char * title)
{
    printf("%s:\n%g\t%g\t%g\t%g\n%g\t%g\t%g\t%g\n%g\t%g\t%g\t%g\n%g\t%g\t%g\t%g\n", title,
           mat->m[0][0], mat->m[0][1], mat->m[0][2], mat->m[0][3],
           mat->m[1][0], mat->m[1][1], mat->m[1][2], mat->m[1][3],
           mat->m[2][0], mat->m[2][1], mat->m[2][2], mat->m[2][3],
           mat->m[3][0], mat->m[3][1], mat->m[3][2], mat->m[3][3]);
}
/* *************************************************************** */
void reg_mat33_disp(mat33 *mat, char * title)
{
    printf("%s:\n%g\t%g\t%g\n%g\t%g\t%g\n%g\t%g\t%g\n", title,
           mat->m[0][0], mat->m[0][1], mat->m[0][2],
           mat->m[1][0], mat->m[1][1], mat->m[1][2],
           mat->m[2][0], mat->m[2][1], mat->m[2][2]);
}
/* *************************************************************** */
/* *************************************************************** */
template <class FieldTYPE>
void reg_affine_positionField2D(mat44 *affineTransformation,
                    nifti_image *targetImage,
                    nifti_image *positionFieldImage)
{
    FieldTYPE *positionFieldPtr = static_cast<FieldTYPE *>(positionFieldImage->data);

    unsigned int positionFieldXIndex=0;
    unsigned int positionFieldYIndex=targetImage->nvox;

    mat44 *targetMatrix;
    if(targetImage->sform_code>0){
        targetMatrix=&(targetImage->sto_xyz);
    }
    else targetMatrix=&(targetImage->qto_xyz);

    mat44 voxelToRealDeformed = reg_mat44_mul(affineTransformation, targetMatrix);

    float index[3];
    float position[3];
    index[2]=0;
    for(int y=0; y<targetImage->ny; y++){
        index[1]=(float)y;
        for(int x=0; x<targetImage->nx; x++){
            index[0]=(float)x;

            reg_mat44_mul(&voxelToRealDeformed, index, position);

            /* the deformation field (real coordinates) is stored */
            positionFieldPtr[positionFieldXIndex++] = position[0];
            positionFieldPtr[positionFieldYIndex++] = position[1];
        }
    }
}
/* *************************************************************** */
template <class FieldTYPE>
void reg_affine_positionField3D(mat44 *affineTransformation,
                    nifti_image *targetImage,
                    nifti_image *positionFieldImage)
{
    FieldTYPE *positionFieldPtr = static_cast<FieldTYPE *>(positionFieldImage->data);
    
    unsigned int positionFieldXIndex=0;
    unsigned int positionFieldYIndex=targetImage->nvox;
    unsigned int positionFieldZIndex=2*targetImage->nvox;
    
    mat44 *targetMatrix;
    if(targetImage->sform_code>0){
        targetMatrix=&(targetImage->sto_xyz);
    }
    else targetMatrix=&(targetImage->qto_xyz);
    
    mat44 voxelToRealDeformed = reg_mat44_mul(affineTransformation, targetMatrix);

    float index[3];
    float position[3];
    for(int z=0; z<targetImage->nz; z++){
        index[2]=(float)z;
        for(int y=0; y<targetImage->ny; y++){
            index[1]=(float)y;
            for(int x=0; x<targetImage->nx; x++){
                index[0]=(float)x;

                reg_mat44_mul(&voxelToRealDeformed, index, position);

                /* the deformation field (real coordinates) is stored */
                positionFieldPtr[positionFieldXIndex++] = position[0];
                positionFieldPtr[positionFieldYIndex++] = position[1];
                positionFieldPtr[positionFieldZIndex++] = position[2];
            }
        }
    }
}
/* *************************************************************** */
void reg_affine_positionField(mat44 *affineTransformation,
								nifti_image *targetImage,
								nifti_image *positionFieldImage)
{
    if(targetImage->nz==1){
        switch(positionFieldImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_affine_positionField2D<float>(affineTransformation, targetImage, positionFieldImage);
                break;
            case NIFTI_TYPE_FLOAT64:
                reg_affine_positionField2D<double>(affineTransformation, targetImage, positionFieldImage);
                break;
            default:
                printf("err\treg_affine_positionField\tThe deformation field data type is not supported\n");
                return;
        }
    }
    else{
        switch(positionFieldImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                reg_affine_positionField3D<float>(affineTransformation, targetImage, positionFieldImage);
                break;
            case NIFTI_TYPE_FLOAT64:
                reg_affine_positionField3D<double>(affineTransformation, targetImage, positionFieldImage);
                break;
            default:
                printf("err\treg_affine_positionField\tThe deformation field data type is not supported\n");
                return;
        }
    }
}
/* *************************************************************** */
/* *************************************************************** */
void reg_tool_ReadAffineFile(	mat44 *mat,
								nifti_image* target,
								nifti_image* source,
								char *fileName,
								bool flirtFile)
{
	std::ifstream affineFile;
	affineFile.open(fileName);
	if(affineFile.is_open()){
		int i=0;
		float value1,value2,value3,value4;
		while(!affineFile.eof()){
			affineFile >> value1 >> value2 >> value3 >> value4;
			mat->m[i][0] = value1;
			mat->m[i][1] = value2;
			mat->m[i][2] = value3;
			mat->m[i][3] = value4;
			i++;
			if(i>3) break;
		}
	}
	affineFile.close();
	
	if(flirtFile){
		mat44 absoluteTarget;
		mat44 absoluteSource;
		for(int i=0;i<4;i++){
			for(int j=0;j<4;j++){
				absoluteTarget.m[i][j]=absoluteSource.m[i][j]=0.0;
			}
		}
		//If the target sform is defined, it is used; qform otherwise;
		mat44 *targetMatrix;
		if(target->sform_code > 0){
			targetMatrix = &(target->sto_xyz);
#ifndef NDEBUG
			printf("[DEBUG] The target sform matrix is defined and used\n");
#endif
		}
		else targetMatrix = &(target->qto_xyz);
		//If the source sform is defined, it is used; qform otherwise;
		mat44 *sourceMatrix;
		if(source->sform_code > 0){
#ifndef NDEBUG
			printf("[DEBUG] The source sform matrix is defined and used\n");
#endif
			sourceMatrix = &(source->sto_xyz);
		}
		else sourceMatrix = &(source->qto_xyz);
		
		for(int i=0;i<3;i++){
			absoluteTarget.m[i][i]=sqrt(targetMatrix->m[0][i]*targetMatrix->m[0][i]
						+ targetMatrix->m[1][i]*targetMatrix->m[1][i]
						+ targetMatrix->m[2][i]*targetMatrix->m[2][i]);
			absoluteSource.m[i][i]=sqrt(sourceMatrix->m[0][i]*sourceMatrix->m[0][i]
						+ sourceMatrix->m[1][i]*sourceMatrix->m[1][i]
						+ sourceMatrix->m[2][i]*sourceMatrix->m[2][i]);
		}
		absoluteTarget.m[3][3]=absoluteSource.m[3][3]=1.0;
#ifndef NDEBUG
		printf("[DEBUG] An flirt affine file is assumed and is converted to a real word affine matrix\n");
		reg_mat44_disp(mat, (char *)"[DEBUG] Matrix read from the input file");
		reg_mat44_disp(targetMatrix, (char *)"[DEBUG] Target Matrix");
		reg_mat44_disp(sourceMatrix, (char *)"[DEBUG] Source Matrix");
		reg_mat44_disp(&(absoluteTarget), (char *)"[DEBUG] Target absolute Matrix");
		reg_mat44_disp(&(absoluteSource), (char *)"[DEBUG] Source absolute Matrix");
#endif
		
		absoluteSource = nifti_mat44_inverse(absoluteSource);
		*mat = nifti_mat44_inverse(*mat);
		
		*mat = reg_mat44_mul(&absoluteSource,mat);
		*mat = reg_mat44_mul(mat, &absoluteTarget);
		*mat = reg_mat44_mul(sourceMatrix,mat);
		mat44 tmp = nifti_mat44_inverse(*targetMatrix);
		*mat = reg_mat44_mul(mat, &tmp);
	}
	
#ifndef NDEBUG
	reg_mat44_disp(mat, (char *)"[DEBUG] Affine matrix");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
void reg_tool_WriteAffineFile(	mat44 *mat,
								char *fileName)
{
	FILE *affineFile;
	affineFile=fopen(fileName, "w");
	for(int i=0;i<4;i++){
		fprintf(affineFile, "%g %g %g %g\n", mat->m[i][0], mat->m[i][1], mat->m[i][2], mat->m[i][3]);
	}
	fclose(affineFile);
}
/* *************************************************************** */
/* *************************************************************** */

#endif
