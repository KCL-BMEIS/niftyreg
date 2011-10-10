#ifndef _REG_MATHS_CPP
#define _REG_MATHS_CPP

#include "_reg_maths.h"

/* *************************************************************** */
/* *************************************************************** */
void reg_LUdecomposition(float *mat,
                         int dim,
                         int *index)
{
    if(index!=NULL)
        free(index);
    index=(int *)calloc(dim,sizeof(int));

    /***** WARNING *****/
    // Implementation from Numerical recipies, might be changed in the future
    /***** WARNING *****/

    float *vv=(float *)malloc(dim*sizeof(float));

    for(int i=0;i<dim;++i){
        float big=0.f;
        float temp;
        for(int j=0;j<dim;++j)
            if( (temp=fabs(mat[i*dim+j]))>big) big=temp;
        if(big==0.f){
            fprintf(stderr, "[NiftyReg] ERROR Singular matrix in the LU decomposition\n");
            exit(1);
        }
        vv[i]=1.f/big;
    }
    for(int j=0;j<dim;++j){
        for(int i=0;i<j;++i){
            float sum=mat[i*dim+j];
            for(int k=0;k<i;k++) sum -= mat[i*dim+k]*mat[k*dim+j];
            mat[i*dim+j]=sum;
        }
        float big=0.f;
        float dum;
        int imax=0;
        for(int i=j;i<dim;++i){
            float sum=mat[i*dim+j];
            for(int k=0;k<j;++k ) sum -= mat[i*dim+k]*mat[k*dim+j];
            mat[i*dim+j]=sum;
            if( (dum=vv[i]*fabs(sum)) >= big ){
                big=dum;
                imax=i;
            }
        }
        if(j != imax){
            for(int k=0;k<dim;++k){
                dum=mat[imax*dim+k];
                mat[imax*dim+k]=mat[j*dim+k];
                mat[j*dim+k]=dum;
            }
            vv[imax]=vv[j];
        }
        index[j]=imax;
        if(mat[j*dim+j]==0) mat[j*dim+j]=1.0e-20f;
        if(j!=dim-1){
            dum=1.f/mat[j*dim+j];
            for(int i=j+1; i<dim;++i) mat[i*dim+j] *= dum;
        }
    }

    free(vv);
    return;
}
/* *************************************************************** */
/* *************************************************************** */

void reg_matrixInvertMultiply(float *mat,
                              int dim,
                              int *index,
                              float *vec)
{
    // Perform the LU decomposition if necessary
    if(index==NULL) reg_LUdecomposition(mat, dim, index);

    /***** WARNING *****/
    // Implementation from Numerical recipies, might be changed in the future
    /***** WARNING *****/

    int ii=-1;

    for(int i=0;i<dim;++i){
        int ip=index[i];
        float sum = vec[ip];
        vec[ip]=vec[i];
        if(ii!=-1)
            for(int j=ii;j<i;++j) sum -= mat[i*dim+j]*vec[j];
        else if(sum) ii=i;
        vec[i]=sum;
    }
    for(int i=dim-1;i>-1;--i){
        float sum=vec[i];
        for(int j=i+1;j<dim;j++) sum -= mat[i*dim+j]*vec[j];
        vec[i]=sum/mat[i*dim+i];
    }
}
/* *************************************************************** */
/* *************************************************************** */
// Heap sort
void reg_heapSort(float *array_tmp, int *index_tmp, int blockNum)
{
    float *array = &array_tmp[-1];
    int *index = &index_tmp[-1];
    int l=(blockNum >> 1)+1;
    int ir=blockNum;
    float val;
    int iVal;
    for(;;){
        if(l>1){
            val=array[--l];
            iVal=index[l];
        }
        else{
            val=array[ir];
            iVal=index[ir];
            array[ir]=array[1];
            index[ir]=index[1];
            if(--ir == 1){
                array[1]=val;
                index[1]=iVal;
                break;
            }
        }
        int i=l;
        int j=l+l;
        while(j<=ir){
            if(j<ir && array[j]<array[j+1]) j++;
            if(val<array[j]){
                array[i]=array[j];
                index[i]=index[j];
                i=j;
                j<<=1;
            }
            else break;
        }
        array[i]=val;
        index[i]=iVal;
    }
}
/* *************************************************************** */
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
/* *************************************************************** */
template <class DTYPE>
void reg_mat44_mul(mat44 *mat,
                    DTYPE *in,
                    DTYPE *out)
{
    out[0]=mat->m[0][0]*in[0] + mat->m[0][1]*in[1] + mat->m[0][2]*in[2] + mat->m[0][3];
    out[1]=mat->m[1][0]*in[0] + mat->m[1][1]*in[1] + mat->m[1][2]*in[2] + mat->m[1][3];
    out[2]=mat->m[2][0]*in[0] + mat->m[2][1]*in[1] + mat->m[2][2]*in[2] + mat->m[2][3];
    return;
}
template void reg_mat44_mul<float>(mat44 *, float*, float*);
template void reg_mat44_mul<double>(mat44 *, double*, double*);
/* *************************************************************** */
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
void reg_getReorientationMatrix(nifti_image *splineControlPoint, mat33 *desorient, mat33 *reorient)
{
    /* In case the matrix is not diagonal, the jacobian has to be reoriented */
    reorient->m[0][0]=splineControlPoint->dx; reorient->m[0][1]=0.0f; reorient->m[0][2]=0.0f;
    reorient->m[1][0]=0.0f; reorient->m[1][1]=splineControlPoint->dy; reorient->m[1][2]=0.0f;
    reorient->m[2][0]=0.0f; reorient->m[2][1]=0.0f; reorient->m[2][2]=splineControlPoint->dz;
    mat33 spline_ijk;
    if(splineControlPoint->sform_code>0){
        spline_ijk.m[0][0]=splineControlPoint->sto_ijk.m[0][0];
        spline_ijk.m[0][1]=splineControlPoint->sto_ijk.m[0][1];
        spline_ijk.m[0][2]=splineControlPoint->sto_ijk.m[0][2];
        spline_ijk.m[1][0]=splineControlPoint->sto_ijk.m[1][0];
        spline_ijk.m[1][1]=splineControlPoint->sto_ijk.m[1][1];
        spline_ijk.m[1][2]=splineControlPoint->sto_ijk.m[1][2];
        spline_ijk.m[2][0]=splineControlPoint->sto_ijk.m[2][0];
        spline_ijk.m[2][1]=splineControlPoint->sto_ijk.m[2][1];
        spline_ijk.m[2][2]=splineControlPoint->sto_ijk.m[2][2];
    }
    else{
        spline_ijk.m[0][0]=splineControlPoint->qto_ijk.m[0][0];
        spline_ijk.m[0][1]=splineControlPoint->qto_ijk.m[0][1];
        spline_ijk.m[0][2]=splineControlPoint->qto_ijk.m[0][2];
        spline_ijk.m[1][0]=splineControlPoint->qto_ijk.m[1][0];
        spline_ijk.m[1][1]=splineControlPoint->qto_ijk.m[1][1];
        spline_ijk.m[1][2]=splineControlPoint->qto_ijk.m[1][2];
        spline_ijk.m[2][0]=splineControlPoint->qto_ijk.m[2][0];
        spline_ijk.m[2][1]=splineControlPoint->qto_ijk.m[2][1];
        spline_ijk.m[2][2]=splineControlPoint->qto_ijk.m[2][2];
    }
    *desorient=nifti_mat33_mul(spline_ijk, *reorient);
    *reorient=nifti_mat33_inverse(*desorient);
}
/* *************************************************************** */
/* *************************************************************** */
#endif // _REG_MATHS_CPP
