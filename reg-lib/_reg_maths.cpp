#ifndef _REG_MATHS_CPP
#define _REG_MATHS_CPP

#include "_reg_maths.h"
#include "_reg_tools.h"

#define mat(i,j,dim) mat[i*dim+j]

/* *************************************************************** */
#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

static double maxarg1,maxarg2;
#define FMAX(a,b) (maxarg1=(a),maxarg2=(b),(maxarg1) > (maxarg2) ?\
(maxarg1) : (maxarg2))

static int iminarg1,iminarg2;
#define IMIN(a,b) (iminarg1=(a),iminarg2=(b),(iminarg1) < (iminarg2) ?\
(iminarg1) : (iminarg2))

static double sqrarg;
#define SQR(a) ((sqrarg=(a)) == 0.0 ? 0.0 : sqrarg*sqrarg)

/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_LUdecomposition(T *mat,
                         size_t dim,
                         size_t *index)
{
    T *vv=(T *)malloc(dim*sizeof(T));
    size_t i,j,k,imax=0;

    for(i=0;i<dim;++i){
        T big=0.f;
        T temp;
        for(j=0;j<dim;++j)
            if( (temp=fabs(mat(i,j,dim)))>big)
                big=temp;
        if(big==0.f){
            fprintf(stderr, "[NiftyReg] ERROR Singular matrix in the LU decomposition\n");
            reg_exit(1);
        }
        vv[i]=1.0/big;
    }
    for(j=0;j<dim;++j){
        for(i=0;i<j;++i){
            T sum=mat(i,j,dim);
            for(k=0;k<i;k++) sum -= mat(i,k,dim)*mat(k,j,dim);
            mat(i,j,dim)=sum;
        }
        T big=0.f;
        T dum;
        for(i=j;i<dim;++i){
            T sum=mat(i,j,dim);
            for(k=0;k<j;++k ) sum -= mat(i,k,dim)*mat(k,j,dim);
            mat(i,j,dim)=sum;
            if( (dum=vv[i]*fabs(sum)) >= big ){
                big=dum;
                imax=i;
            }
        }
        if(j != imax){
            for(k=0;k<dim;++k){
                dum=mat(imax,k,dim);
                mat(imax,k,dim)=mat(j,k,dim);
                mat(j,k,dim)=dum;
            }
            vv[imax]=vv[j];
        }
        index[j]=imax;
        if(mat(j,j,dim)==0) mat(j,j,dim)=1.0e-20;
        if(j!=dim-1){
            dum=1.0/mat(j,j,dim);
            for(i=j+1; i<dim;++i) mat(i,j,dim) *= dum;
        }
    }
    free(vv);
    return;
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_matrixInvertMultiply(T *mat,
                              size_t dim,
                              size_t *index,
                              T *vec)
{
    // Perform the LU decomposition if necessary
    if(index==NULL)
        reg_LUdecomposition(mat, dim, index);

    int ii=0;
	for(int i=0;i<(int)dim;++i){
        int ip=index[i];
        T sum = vec[ip];
        vec[ip]=vec[i];
        if(ii!=0){
            for(int j=ii-1;j<i;++j)
                sum -= mat(i,j,dim)*vec[j];
        }
        else if(sum!=0)
            ii=i+1;
        vec[i]=sum;
    }
	for(int i=(int)dim-1;i>-1;--i){
        T sum=vec[i];
		for(int j=i+1;j<(int)dim;++j)
            sum -= mat(i,j,dim)*vec[j];
        vec[i]=sum/mat(i,i,dim);
    }
}
template void reg_matrixInvertMultiply<float>(float *, size_t, size_t *, float *);
template void reg_matrixInvertMultiply<double>(double *, size_t, size_t *, double *);
/* *************************************************************** */
/* *************************************************************** */
extern "C++" template <class T>
void reg_matrixMultiply(T *mat1,
                        T *mat2,
                        int *dim1,
                        int *dim2,
                        T *&res)
{
    // First check that the dimension are appropriate
    if(dim1[1]!=dim2[0]){
        fprintf(stderr, "Matrices can not be multiplied due to their size: [%i %i] [%i %i]\n",
                dim1[0],dim1[1],dim2[0],dim2[1]);
        reg_exit(1);
    }
    int resDim[2]={dim1[0],dim2[1]};
    // Allocate the result matrix
    if(res!=NULL)
        free(res);
    res=(T *)calloc(resDim[0]*resDim[1],sizeof(T));
    // Multiply both matrices
    for(int j=0; j<resDim[1];++j){
        for(int i=0; i<resDim[0];++i){
            double sum=0.0;
            for(int k=0;k<dim1[1];++k){
                sum += mat1[k*dim1[0]+i] * mat2[j*dim2[0]+k];
            }
            res[j*resDim[0]+i]=sum;
        } // i
    } // j
}
template void reg_matrixMultiply<float>(float * ,float * ,int *, int * , float *&);
template void reg_matrixMultiply<double>(double * ,double * ,int *, int * , double *&);
/* *************************************************************** */
/* *************************************************************** */
extern "C++" template <class T>
void reg_matrixInverse(T *mat,
                       int *dim)
{
    // First check that the dimension are appropriate
    if(dim[1]!=dim[0]){
        fprintf(stderr, "Matrices is expected to be square [%i %i]. Return\n",
                dim[0],dim[1]);
        return;
    }

    for (int i=1; i < dim[0]; i++) {
        mat[i] /= mat[0]; // normalize row 0
    }
    for (int i=1; i < dim[0]; i++)  {
        for (int j=i; j < dim[0]; j++)  { // do a column of L
            T sum = 0.0;
            for (int k = 0; k < i; k++){
                sum += mat[j*dim[0]+k] * mat[k*dim[0]+i];
            }
            mat[j*dim[0]+i] -= sum;
        }
        if (i == dim[0]-1) continue;
        for (int j=i+1; j < dim[0]; j++)  {  // do a row of U
            T sum = 0.0;
            for (int k = 0; k < i; k++){
                sum += mat[i*dim[0]+k]*mat[k*dim[0]+j];
            }
            mat[i*dim[0]+j] =(mat[i*dim[0]+j]-sum) / mat[i*dim[0]+i];
        }
    }
    for ( int i = 0; i < dim[0]; i++ )  // invert L
        for ( int j = i; j < dim[0]; j++ )  {
            T x = 1.0;
            if ( i != j ) {
                x = 0.0;
                for ( int k = i; k < j; k++ ){
                    x -= mat[j*dim[0]+k]*mat[k*dim[0]+i];
                }
            }
            mat[j*dim[0]+i] = x / mat[j*dim[0]+j];
        }
    for ( int i = 0; i < dim[0]; i++ )   // invert U
        for ( int j = i; j < dim[0]; j++ )  {
            if ( i == j ){continue;}
            T sum = 0.0;
            for ( int k = i; k < j; k++ ){
                sum += mat[k*dim[0]+j]*( (i==k) ? 1.0 : mat[i*dim[0]+k] );
            }
            mat[i*dim[0]+j] = -sum;
        }
    for ( int i = 0; i < dim[0]; i++ )   // final inversion
        for ( int j = 0; j < dim[0]; j++ )  {
            T sum = 0.0;
            for ( int k = ((i>j)?i:j); k < dim[0]; k++ ){
                sum += ((j==k)?1.0:mat[j*dim[0]+k])*mat[k*dim[0]+i];
            }
            mat[j*dim[0]+i] = sum;
        }
}
template void reg_matrixInverse<float>(float *, int *);
template void reg_matrixInverse<double>(double *, int *);
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
// Heap sort
void reg_heapSort(float *array_tmp, int blockNum)
{
    float *array = &array_tmp[-1];
    int l=(blockNum >> 1)+1;
    int ir=blockNum;
    float val;
    for(;;){
        if(l>1){
            val=array[--l];
        }
        else{
            val=array[ir];
            array[ir]=array[1];
            if(--ir == 1){
                array[1]=val;
                break;
            }
        }
        int i=l;
        int j=l+l;
        while(j<=ir){
            if(j<ir && array[j]<array[j+1]) j++;
            if(val<array[j]){
                array[i]=array[j];
                i=j;
                j<<=1;
            }
            else break;
        }
        array[i]=val;
    }
}
/* *************************************************************** */
/* *************************************************************** */
template <class MTYPE>
float reg_mat44_det(MTYPE const* A)
{
    double D =
          (double)A->m[0][0]*A->m[1][1]*A->m[2][2]*A->m[3][3]
        - A->m[0][0]*A->m[1][1]*A->m[3][2]*A->m[2][3]
        - A->m[0][0]*A->m[2][1]*A->m[1][2]*A->m[3][3]
        + A->m[0][0]*A->m[2][1]*A->m[3][2]*A->m[1][3]
        + A->m[0][0]*A->m[3][1]*A->m[1][2]*A->m[2][3]
        - A->m[0][0]*A->m[3][1]*A->m[2][2]*A->m[1][3]
        - A->m[1][0]*A->m[0][1]*A->m[2][2]*A->m[3][3]
        + A->m[1][0]*A->m[0][1]*A->m[3][2]*A->m[2][3]
        + A->m[1][0]*A->m[2][1]*A->m[0][2]*A->m[3][3]
        - A->m[1][0]*A->m[2][1]*A->m[3][2]*A->m[0][3]
        - A->m[1][0]*A->m[3][1]*A->m[0][2]*A->m[2][3]
        + A->m[1][0]*A->m[3][1]*A->m[2][2]*A->m[0][3]
        + A->m[2][0]*A->m[0][1]*A->m[1][2]*A->m[3][3]
        - A->m[2][0]*A->m[0][1]*A->m[3][2]*A->m[1][3]
        - A->m[2][0]*A->m[1][1]*A->m[0][2]*A->m[3][3]
        + A->m[2][0]*A->m[1][1]*A->m[3][2]*A->m[0][3]
        + A->m[2][0]*A->m[3][1]*A->m[0][2]*A->m[1][3]
        - A->m[2][0]*A->m[3][1]*A->m[1][2]*A->m[0][3]
        - A->m[3][0]*A->m[0][1]*A->m[1][2]*A->m[2][3]
        + A->m[3][0]*A->m[0][1]*A->m[2][2]*A->m[1][3]
        + A->m[3][0]*A->m[1][1]*A->m[0][2]*A->m[2][3]
        - A->m[3][0]*A->m[1][1]*A->m[2][2]*A->m[0][3]
        - A->m[3][0]*A->m[2][1]*A->m[0][2]*A->m[1][3]
        + A->m[3][0]*A->m[2][1]*A->m[1][2]*A->m[0][3];
    return static_cast<float>(D);
}
template float reg_mat44_det<mat44>(mat44 const* A);
template float reg_mat44_det<reg_mat44d>(reg_mat44d const* A);
/* *************************************************************** */
template <class MTYPE>
mat44 reg_MTYPE_to_mat44(MTYPE *M)
{
    mat44 R;
    for(int i=0;i<4;++i)
        for(int j=0;j<4;++j)
            R.m[i][j]=static_cast<float>(M->m[i][j]);
    return R;
}
/* *************************************************************** */
template <class MTYPE>
MTYPE reg_mat44_to_MTYPE(mat44 *M)
{
    MTYPE R;
    for(int i=0;i<4;++i)
        for(int j=0;j<4;++j)
            R.m[i][j]=M->m[i][j];
    return R;
}
/* *************************************************************** */
//Ported from VNL
template <class MTYPE>
MTYPE reg_mat44_inv(MTYPE const* A)
{
    MTYPE R;
    float detA = reg_mat44_det(A);
    if(detA==0){
        fprintf(stderr,"[NiftyReg ERROR] Cannot invert 4x4 matrix with zero determinant.\n");
        fprintf(stderr,"[NiftyReg ERROR] Returning matrix of zeros\n");
        memset(&R,0,sizeof(MTYPE));
        return R;
    }
    detA = 1.0f / detA;
    R.m[0][0] =  A->m[1][1]*A->m[2][2]*A->m[3][3] - A->m[1][1]*A->m[2][3]*A->m[3][2]
            - A->m[2][1]*A->m[1][2]*A->m[3][3] + A->m[2][1]*A->m[1][3]*A->m[3][2]
            + A->m[3][1]*A->m[1][2]*A->m[2][3] - A->m[3][1]*A->m[1][3]*A->m[2][2];
    R.m[0][1] = -A->m[0][1]*A->m[2][2]*A->m[3][3] + A->m[0][1]*A->m[2][3]*A->m[3][2]
            + A->m[2][1]*A->m[0][2]*A->m[3][3] - A->m[2][1]*A->m[0][3]*A->m[3][2]
            - A->m[3][1]*A->m[0][2]*A->m[2][3] + A->m[3][1]*A->m[0][3]*A->m[2][2];
    R.m[0][2] =  A->m[0][1]*A->m[1][2]*A->m[3][3] - A->m[0][1]*A->m[1][3]*A->m[3][2]
            - A->m[1][1]*A->m[0][2]*A->m[3][3] + A->m[1][1]*A->m[0][3]*A->m[3][2]
            + A->m[3][1]*A->m[0][2]*A->m[1][3] - A->m[3][1]*A->m[0][3]*A->m[1][2];
    R.m[0][3] = -A->m[0][1]*A->m[1][2]*A->m[2][3] + A->m[0][1]*A->m[1][3]*A->m[2][2]
            + A->m[1][1]*A->m[0][2]*A->m[2][3] - A->m[1][1]*A->m[0][3]*A->m[2][2]
            - A->m[2][1]*A->m[0][2]*A->m[1][3] + A->m[2][1]*A->m[0][3]*A->m[1][2];
    R.m[1][0] = -A->m[1][0]*A->m[2][2]*A->m[3][3] + A->m[1][0]*A->m[2][3]*A->m[3][2]
            + A->m[2][0]*A->m[1][2]*A->m[3][3] - A->m[2][0]*A->m[1][3]*A->m[3][2]
            - A->m[3][0]*A->m[1][2]*A->m[2][3] + A->m[3][0]*A->m[1][3]*A->m[2][2];
    R.m[1][1] =  A->m[0][0]*A->m[2][2]*A->m[3][3] - A->m[0][0]*A->m[2][3]*A->m[3][2]
            - A->m[2][0]*A->m[0][2]*A->m[3][3] + A->m[2][0]*A->m[0][3]*A->m[3][2]
            + A->m[3][0]*A->m[0][2]*A->m[2][3] - A->m[3][0]*A->m[0][3]*A->m[2][2];
    R.m[1][2] = -A->m[0][0]*A->m[1][2]*A->m[3][3] + A->m[0][0]*A->m[1][3]*A->m[3][2]
            + A->m[1][0]*A->m[0][2]*A->m[3][3] - A->m[1][0]*A->m[0][3]*A->m[3][2]
            - A->m[3][0]*A->m[0][2]*A->m[1][3] + A->m[3][0]*A->m[0][3]*A->m[1][2];
    R.m[1][3] =  A->m[0][0]*A->m[1][2]*A->m[2][3] - A->m[0][0]*A->m[1][3]*A->m[2][2]
            - A->m[1][0]*A->m[0][2]*A->m[2][3] + A->m[1][0]*A->m[0][3]*A->m[2][2]
            + A->m[2][0]*A->m[0][2]*A->m[1][3] - A->m[2][0]*A->m[0][3]*A->m[1][2];
    R.m[2][0] =  A->m[1][0]*A->m[2][1]*A->m[3][3] - A->m[1][0]*A->m[2][3]*A->m[3][1]
            - A->m[2][0]*A->m[1][1]*A->m[3][3] + A->m[2][0]*A->m[1][3]*A->m[3][1]
            + A->m[3][0]*A->m[1][1]*A->m[2][3] - A->m[3][0]*A->m[1][3]*A->m[2][1];
    R.m[2][1] = -A->m[0][0]*A->m[2][1]*A->m[3][3] + A->m[0][0]*A->m[2][3]*A->m[3][1]
            + A->m[2][0]*A->m[0][1]*A->m[3][3] - A->m[2][0]*A->m[0][3]*A->m[3][1]
            - A->m[3][0]*A->m[0][1]*A->m[2][3] + A->m[3][0]*A->m[0][3]*A->m[2][1];
    R.m[2][2]=  A->m[0][0]*A->m[1][1]*A->m[3][3] - A->m[0][0]*A->m[1][3]*A->m[3][1]
            - A->m[1][0]*A->m[0][1]*A->m[3][3] + A->m[1][0]*A->m[0][3]*A->m[3][1]
            + A->m[3][0]*A->m[0][1]*A->m[1][3] - A->m[3][0]*A->m[0][3]*A->m[1][1];
    R.m[2][3]= -A->m[0][0]*A->m[1][1]*A->m[2][3] + A->m[0][0]*A->m[1][3]*A->m[2][1]
            + A->m[1][0]*A->m[0][1]*A->m[2][3] - A->m[1][0]*A->m[0][3]*A->m[2][1]
            - A->m[2][0]*A->m[0][1]*A->m[1][3] + A->m[2][0]*A->m[0][3]*A->m[1][1];
    R.m[3][0]= -A->m[1][0]*A->m[2][1]*A->m[3][2] + A->m[1][0]*A->m[2][2]*A->m[3][1]
            + A->m[2][0]*A->m[1][1]*A->m[3][2] - A->m[2][0]*A->m[1][2]*A->m[3][1]
            - A->m[3][0]*A->m[1][1]*A->m[2][2] + A->m[3][0]*A->m[1][2]*A->m[2][1];
    R.m[3][1]=  A->m[0][0]*A->m[2][1]*A->m[3][2] - A->m[0][0]*A->m[2][2]*A->m[3][1]
            - A->m[2][0]*A->m[0][1]*A->m[3][2] + A->m[2][0]*A->m[0][2]*A->m[3][1]
            + A->m[3][0]*A->m[0][1]*A->m[2][2] - A->m[3][0]*A->m[0][2]*A->m[2][1];
    R.m[3][2]= -A->m[0][0]*A->m[1][1]*A->m[3][2] + A->m[0][0]*A->m[1][2]*A->m[3][1]
            + A->m[1][0]*A->m[0][1]*A->m[3][2] - A->m[1][0]*A->m[0][2]*A->m[3][1]
            - A->m[3][0]*A->m[0][1]*A->m[1][2] + A->m[3][0]*A->m[0][2]*A->m[1][1];
    R.m[3][3]=  A->m[0][0]*A->m[1][1]*A->m[2][2] - A->m[0][0]*A->m[1][2]*A->m[2][1]
            - A->m[1][0]*A->m[0][1]*A->m[2][2] + A->m[1][0]*A->m[0][2]*A->m[2][1]
            + A->m[2][0]*A->m[0][1]*A->m[1][2] - A->m[2][0]*A->m[0][2]*A->m[1][1];
    return reg_mat44_mul(&R,detA);
}
template mat44 reg_mat44_inv<mat44>(mat44 const* A);
template reg_mat44d reg_mat44_inv<reg_mat44d>(reg_mat44d const* A);
/* *************************************************************** */
/* *************************************************************** */
reg_mat44d reg_mat44_singleToDouble(mat44 const *mat)
{
    reg_mat44d R;
    for(int i=0;i<4;++i)
        for(int j=0;j<4;++j)
            R.m[i][j]=static_cast<double>(mat->m[i][j]);
    return R;
}
/* *************************************************************** */
mat44 reg_mat44_doubleToSingle(reg_mat44d const *mat)
{
    mat44 R;
    for(int i=0;i<4;++i)
        for(int j=0;j<4;++j)
            R.m[i][j]=static_cast<float>(mat->m[i][j]);
    return R;
}
/* *************************************************************** */
/* *************************************************************** */
mat33 reg_mat44_to_mat33(mat44 const* A)
{
    mat33 out;
    out.m[0][0]=A->m[0][0];
    out.m[0][1]=A->m[0][1];
    out.m[0][2]=A->m[0][2];
    out.m[1][0]=A->m[1][0];
    out.m[1][1]=A->m[1][1];
    out.m[1][2]=A->m[1][2];
    out.m[2][0]=A->m[2][0];
    out.m[2][1]=A->m[2][1];
    out.m[2][2]=A->m[2][2];
    return out;
}
/* *************************************************************** */
/* *************************************************************** */
template <class MTYPE>
MTYPE reg_mat44_mul(MTYPE const* A, MTYPE const* B)
{
    MTYPE R;
    for(int i=0; i<4; i++){
        for(int j=0; j<4; j++){
            R.m[i][j] = A->m[i][0]*B->m[0][j] +
                    A->m[i][1]*B->m[1][j] +
                    A->m[i][2]*B->m[2][j] +
                    A->m[i][3]*B->m[3][j];
        }
    }
    return R;
}
template mat44 reg_mat44_mul<mat44>(mat44 const* A, mat44 const* B);
template reg_mat44d reg_mat44_mul<reg_mat44d>(reg_mat44d const* A, reg_mat44d const* B);
/* *************************************************************** */
mat44 operator*(mat44 A,mat44 B)
{
    return reg_mat44_mul(&A,&B);
}
reg_mat44d operator*(reg_mat44d A,reg_mat44d B)
{
    return reg_mat44_mul(&A,&B);
}
/* *************************************************************** */
/* *************************************************************** */
template <class MTYPE>
MTYPE reg_mat44_add(MTYPE const* A, MTYPE const* B)
{
    MTYPE R;
    for(int i=0; i<4; i++){
        for(int j=0; j<4; j++){
            R.m[i][j] = A->m[i][j]+B->m[i][j];
        }
    }
    return R;
}
template mat44 reg_mat44_add<mat44>(mat44 const* A, mat44 const* B);
template reg_mat44d reg_mat44_add<reg_mat44d>(reg_mat44d const* A, reg_mat44d const* B);
/* *************************************************************** */
/* *************************************************************** */
mat44 operator+(mat44 A,mat44 B)
{
    return reg_mat44_add(&A,&B);
}
reg_mat44d operator+(reg_mat44d A,reg_mat44d B)
{
    return reg_mat44_add(&A,&B);
}
/* *************************************************************** */
/* *************************************************************** */
template <class MTYPE>
MTYPE reg_mat44_minus(MTYPE const* A, MTYPE const* B)
{
    MTYPE R;
    for(int i=0; i<4; i++){
        for(int j=0; j<4; j++){
            R.m[i][j] = A->m[i][j]-B->m[i][j];
        }
    }
    return R;
}
template mat44 reg_mat44_minus<mat44>(mat44 const* A, mat44 const* B);
template reg_mat44d reg_mat44_minus<reg_mat44d>(reg_mat44d const* A, reg_mat44d const* B);
/* *************************************************************** */
mat44 operator-(mat44 A,mat44 B)
{
    return reg_mat44_minus(&A,&B);
}
reg_mat44d operator-(reg_mat44d A,reg_mat44d B)
{
    return reg_mat44_minus(&A,&B);
}
/* *************************************************************** */
/* *************************************************************** */
void reg_mat33_eye (mat33 *mat)
{
    mat->m[0][0]=1.f; mat->m[0][1]=mat->m[0][2]=0.f;
    mat->m[1][1]=1.f; mat->m[1][0]=mat->m[1][2]=0.f;
    mat->m[2][2]=1.f; mat->m[2][0]=mat->m[2][1]=0.f;
}
/* *************************************************************** */
template <class MTYPE>
void reg_mat44_eye (MTYPE *mat)
{
    mat->m[0][0]=1.f; mat->m[0][1]=mat->m[0][2]=mat->m[0][3]=0.f;
    mat->m[1][1]=1.f; mat->m[1][0]=mat->m[1][2]=mat->m[1][3]=0.f;
    mat->m[2][2]=1.f; mat->m[2][0]=mat->m[2][1]=mat->m[2][3]=0.f;
    mat->m[3][3]=1.f; mat->m[3][0]=mat->m[3][1]=mat->m[3][2]=0.f;
}
template void reg_mat44_eye<mat44>(mat44 *mat);
template void reg_mat44_eye<reg_mat44d>(reg_mat44d *mat);
/* *************************************************************** */
/* *************************************************************** */
template <class MTYPE>
float reg_mat44_norm_inf(MTYPE const* mat)
{
    float maxval=0.0;
    float newval=0.0;
    for (int i=0; i < 4; i++){
        for (int j=0; j < 4; j++){
            newval = fabsf((float)mat->m[i][j]);
            maxval = (newval > maxval) ? newval : maxval;
        }
    }
    return maxval;
}
template float reg_mat44_norm_inf<mat44>(mat44 const *mat);
template float reg_mat44_norm_inf<reg_mat44d>(reg_mat44d const *mat);
/* *************************************************************** */
template <class DTYPE,class MTYPE>
void reg_mat44_mul(MTYPE const* mat,
                    DTYPE const* in,
                    DTYPE *out)
{
    out[0]=mat->m[0][0]*in[0] + mat->m[0][1]*in[1] + mat->m[0][2]*in[2] + mat->m[0][3];
    out[1]=mat->m[1][0]*in[0] + mat->m[1][1]*in[1] + mat->m[1][2]*in[2] + mat->m[1][3];
    out[2]=mat->m[2][0]*in[0] + mat->m[2][1]*in[1] + mat->m[2][2]*in[2] + mat->m[2][3];
    return;
}
template void reg_mat44_mul<float,mat44>(mat44 const*, float const*, float*);
template void reg_mat44_mul<float,reg_mat44d>(reg_mat44d const*, float const*, float*);
template void reg_mat44_mul<double,mat44>(mat44 const*, double const*, double*);
template void reg_mat44_mul<double,reg_mat44d>(reg_mat44d const*, double const*, double*);
/* *************************************************************** */
/* *************************************************************** */
template <class MTYPE>
MTYPE reg_mat44_mul(MTYPE const* A, double scalar)
{
    MTYPE out;
    out.m[0][0]=A->m[0][0]*scalar;out.m[0][1]=A->m[0][1]*scalar;out.m[0][2]=A->m[0][2]*scalar;out.m[0][3]=A->m[0][3]*scalar;
    out.m[1][0]=A->m[1][0]*scalar;out.m[1][1]=A->m[1][1]*scalar;out.m[1][2]=A->m[1][2]*scalar;out.m[1][3]=A->m[1][3]*scalar;
    out.m[2][0]=A->m[2][0]*scalar;out.m[2][1]=A->m[2][1]*scalar;out.m[2][2]=A->m[2][2]*scalar;out.m[2][3]=A->m[2][3]*scalar;
    out.m[3][0]=A->m[3][0]*scalar;out.m[3][1]=A->m[3][1]*scalar;out.m[3][2]=A->m[3][2]*scalar;out.m[3][3]=A->m[3][3]*scalar;
    return out;
}
template mat44 reg_mat44_mul<mat44>(mat44 const* A, double scalar);
template reg_mat44d reg_mat44_mul<reg_mat44d>(reg_mat44d const* A, double scalar);
/* *************************************************************** */
/* *************************************************************** */
template <class MTYPE>
MTYPE reg_mat44_sqrt(MTYPE const* mat)
{
//reg_mat44_disp(mat,(char *)"beginning reg_mat44_sqrt");
    MTYPE X=*mat;
    MTYPE Y;
    int it=0;
    int maxit=10;
    reg_mat44_eye(&Y);
    MTYPE delX, delY;
    double eps=1.0e-7;
    MTYPE Xsq=reg_mat44_mul(&X,&X);
    MTYPE diffMat = reg_mat44_minus(&Xsq,mat);
    MTYPE XdelY, YdelX;
    while (reg_mat44_norm_inf(&diffMat) > eps)
    {
        delX=reg_mat44_inv(&X);
        delY=reg_mat44_inv(&Y);
        XdelY = X + delY;
        YdelX = Y + delX;
        X=reg_mat44_mul(&XdelY,0.5);
        Y=reg_mat44_mul(&YdelX,0.5);
        Xsq= X * X;
        diffMat = Xsq - *mat;
        it++;
        if(it > maxit)
            break;
    }
    return X;
}
template mat44 reg_mat44_sqrt<mat44>(mat44 const* mat);
template reg_mat44d reg_mat44_sqrt<reg_mat44d>(reg_mat44d const* mat);
/* *************************************************************** */
/* *************************************************************** */
/**
  * Compute the matrix exponential according to "Linear combination of transformations",
  * Marc Alex, Volume 21, Issue 3, ACM SIGGRAPH 2002.
  * and from Kelvin's implementation of the code in NifTK
  */
template <class MTYPE>
MTYPE reg_mat44_expm(MTYPE const* mat, int maxit)
{
    double j = FMAX(0.0,1+reg_floor(log(reg_mat44_norm_inf(mat))/log(2.0)));

    MTYPE A=reg_mat44_mul(mat,pow(2.0,-j));
    MTYPE D,N,X,cX;
    reg_mat44_eye(&D);
    reg_mat44_eye(&N);
    reg_mat44_eye(&X);

    double c = 1.0;
    for(int k=1; k <= maxit; k++){
        c = c * (maxit-k+1.0) / (k*(2*maxit-k+1.0));
        X = reg_mat44_mul(&A,&X);
        cX = reg_mat44_mul(&X,c);
        N = reg_mat44_add(&N,&cX);
        cX = reg_mat44_mul(&cX,pow(-1.0,k));
        D = reg_mat44_add(&D,&cX);
    }
    D=reg_mat44_inv(&D);
    X=reg_mat44_mul(&D,&N);
    for(int i=0; i < reg_round(j); i++){
        X=reg_mat44_mul(&X,&X);
    }
    return X;
}
template mat44 reg_mat44_expm<mat44>(mat44 const* mat, int maxit);
template reg_mat44d reg_mat44_expm<reg_mat44d>(reg_mat44d const* mat, int maxit);
/* *************************************************************** */
/* *************************************************************** */
template <class MTYPE>
MTYPE reg_mat44_logm(MTYPE const* mat)
{
    int k = 0;
    MTYPE I;
    reg_mat44_eye(&I);
    MTYPE A=*mat;
    MTYPE residual = A-I;
    while(reg_mat44_norm_inf(&residual) > 0.25){
        A=reg_mat44_sqrt(&A);
        k++;
        residual = A-I;
        if(k>1.0e7){
            fprintf(stderr, "reg_mat44_logm did not converge after 10e7 iterations.");
            break;
        }
    }
    A = I - A;
    MTYPE X = A, Z = A;
    double i = 1.0;
    while(reg_mat44_norm_inf(&Z) > 1.0e-7){
        Z = Z * A;
        i += 1.0;
        X = X + reg_mat44_mul(&Z, 1.f/i);
        if(i>1.0e7){
            fprintf(stderr, "reg_mat44_logm did not converge after 10e7 iterations.");
            break;
        }
    }
    X = reg_mat44_mul(&X,-1.0);
    X = reg_mat44_mul(&X, pow(2.0,k));
    return X;
}
template mat44 reg_mat44_logm<mat44>(mat44 const* mat);
template reg_mat44d reg_mat44_logm<reg_mat44d>(reg_mat44d const* mat);
/* *************************************************************** */
/* *************************************************************** */
template <class MTYPE>
MTYPE reg_mat44_avg2(MTYPE const* A, MTYPE const* B)
{
    MTYPE out;
    MTYPE logA=reg_mat44_logm(A);
    MTYPE logB=reg_mat44_logm(B);
    logA = reg_mat44_add(&logA,&logB);
    out = reg_mat44_mul(&logA,0.5);
    return reg_mat44_expm(&out);

}
template mat44 reg_mat44_avg2<mat44>(mat44 const* A, mat44 const* B);
template reg_mat44d reg_mat44_avg2<reg_mat44d>(reg_mat44d const* A, reg_mat44d const* B);
/* *************************************************************** */
/* *************************************************************** */
template <class MTYPE>
void reg_mat44_disp(MTYPE *mat, char * title)
{
    printf("%s:\n%g\t%g\t%g\t%g\n%g\t%g\t%g\t%g\n%g\t%g\t%g\t%g\n%g\t%g\t%g\t%g\n", title,
           mat->m[0][0], mat->m[0][1], mat->m[0][2], mat->m[0][3],
           mat->m[1][0], mat->m[1][1], mat->m[1][2], mat->m[1][3],
           mat->m[2][0], mat->m[2][1], mat->m[2][2], mat->m[2][3],
           mat->m[3][0], mat->m[3][1], mat->m[3][2], mat->m[3][3]);
}
template void reg_mat44_disp<mat44>(mat44 *mat, char * title);
template void reg_mat44_disp<reg_mat44d>(reg_mat44d *mat, char * title);
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
void reg_getReorientationMatrix(nifti_image *splineControlPoint, mat33 *reorient)
{
    if(splineControlPoint->sform_code>0)
        *reorient=nifti_mat33_inverse(nifti_mat33_polar(reg_mat44_to_mat33(&splineControlPoint->sto_xyz)));
    else *reorient=nifti_mat33_inverse(nifti_mat33_polar(reg_mat44_to_mat33(&splineControlPoint->qto_xyz)));
}
/* *************************************************************** */
// Calculate pythagorean distance
template <class T>
T pythag(T a, T b)
{
    T absa, absb;
    absa = fabs(a);
    absb = fabs(b);

    if (absa > absb) return (T)(absa * sqrt(1.0f+SQR(absb/absa)));
    else return (absb == 0.0f ? 0.0f : (T)(absb * sqrt(1.0f+SQR(absa/absb))));
}
/* *************************************************************** */
/* *************************************************************** */
/** @brief SVD
  * @param in input matrix to decompose - in place
  * @param m row
  * @param n colomn
  * @param w diagonal term
  * @param v rotation part
  */
template <class T>
void svd(T ** in, size_t m, size_t n, T * w, T ** v)
{
    T * rv1 = (T *)malloc(sizeof(T) * n);
    T anorm, c, f, g, h, s, scale, x, y, z;
    size_t flag,i,its,j,jj,k,l=0,nm=0;

    g = scale = anorm = 0.0f;
    for (i = 1; i <= n; ++i)
    {
        l = i + 1;
        rv1[i-1] = scale * g;
        g = s = scale = 0.0f;

        if ( i <= m)
        {
            for (k = i; k <= m; ++k)
            {
                scale += fabs(in[k-1][i-1]);
            }
            if (scale)
            {
                for (k = i; k <= m; ++k)
                {
                    in[k-1][i-1] /= scale;
                    s += in[k-1][i-1] * in[k-1][i-1];
                }
                f = in[i-1][i-1];
                g = -SIGN(sqrt(s), f);
                h = f * g - s;
                in[i-1][i-1] = f - g;

                for (j = l; j <= n; ++j)
                {
                    for (s = 0.0, k=i; k<=m; ++k) s += in[k-1][i-1]*in[k-1][j-1];
                    f = s/h;
                    for (k = i; k <= m; ++k) in[k-1][j-1] += f * in[k-1][i-1];
                }
                for (k = i; k <= m; ++k)
                {
                    in[k-1][i-1] *= scale;
                }
            }
        }
        w[i-1] = scale * g;
        g = s = scale = 0.0;
        if ((i <= m) && (i != n))
        {
            for (k = l; k <= n; ++k)
            {
                scale += fabs(in[i-1][k-1]);
            }
            if (scale)
            {
                for (k = l; k <= n; ++k)
                {
                    in[i-1][k-1] /= scale;
                    s += in[i-1][k-1] * in[i-1][k-1];
                }
                f = in[i-1][l-1];
                g = -SIGN(sqrt(s), f);
                h = f*g-s;
                in[i-1][l-1] = f - g;

                for (k = l; k <= n; ++k) rv1[k-1] = in[i-1][k-1]/h;
                for (j = l; j <= m; ++j)
                {
                    for (s = 0.0, k = l; k <= n; ++k)
                    {
                        s += in[j-1][k-1] * in[i-1][k-1];
                    }
                    for (k = l; k <= n; ++k)
                    {
                        in[j-1][k-1] += s * rv1[k-1];
                    }
                }

                for (k=l;k<=n;++k) in[i-1][k-1] *= scale;
            }
        }
        anorm = FMAX(anorm, (fabs(w[i-1])+fabs(rv1[i-1])));
    }

    for (i = n; i >= 1; --i)
    {
        if (i < n)
        {
            if (g)
            {
                for (j = l; j <= n; ++j)
                {
                    v[j-1][i-1] = (in[i-1][j-1]/in[i-1][l-1])/g;
                }
                for (j = l; j <= n; ++j)
                {
                    for (s = 0.0, k = l; k <= n; ++k) s += in[i-1][k-1] * v[k-1][j-1];
                    for (k=l;k<=n;++k) v[k-1][j-1] += s * v[k-1][i-1];
                }
            }
            for (j = l; j <= n; ++j) v[i-1][j-1] = v[j-1][i-1] = 0.0;
        }
        v[i-1][i-1] = 1.0f;
        g = rv1[i-1];
        l = i;
    }

    for (i = IMIN(m, n); i >= 1; --i)
    {
        l = i+1;
        g = w[i-1];
        for (j = l; j <= n; ++j) in[i-1][j-1] = 0.0f;
        if (g)
        {
            g = 1.0f/g;
            for (j = l; j <= n; ++j)
            {
                for (s = 0.0, k = l; k <= m; ++k) s += in[k-1][i-1] * in[k-1][j-1];
                f = (s/in[i-1][i-1])*g;
                for (k=i; k <=m; ++k) in[k-1][j-1] += f * in[k-1][i-1];
            }
            for (j=i; j <= m; ++j) in[j-1][i-1] *= g;
        }
        else for (j = i; j <= m; ++j) in[j-1][i-1] = 0.0;
        ++in[i-1][i-1];
    }

    for (k = n; k >= 1; --k)
    {
        for (its = 0; its < 30; ++its)
        {
            flag = 1;
            for (l=k; l >= 1; --l)
            {
                nm = l - 1;
                if ((T)(fabs(rv1[l-1])+anorm) == anorm)
                {
                    flag = 0;
                    break;
                }
                if ((T)(fabs(w[nm-1])+anorm) == anorm) break;
            }

            if (flag)
            {
                c = 0.0f;
                s = 1.0f;
                for (i=l; i<=k; ++i) // changed
                {
                    f = s * rv1[i-1];
                    rv1[i-1] = c * rv1[i-1];
                    if ((T)(fabs(f)+anorm) == anorm) break;
                    g=w[i-1];
                    h=pythag(f,g);
                    w[i-1]=h;
                    h=1.0f/h;
                    c=g*h;
                    s = -f*h;

                    for (j = 1; j <= m; ++j)
                    {
                        y=in[j-1][nm-1];
                        z=in[j-1][i-1];
                        in[j-1][nm-1]=y*c+z*s;
                        in[j-1][i-1]=z*c-y*s;
                    }
                }
            }
            z = w[k-1];
            if (l == k)
            {
                if (z < 0.0f)
                {
                    w[k-1] = -z;
                    for (j = 1; j <= n; ++j) v[j-1][k-1] = -v[j-1][k-1];
                }
                break;
            }

            x = w[l-1];
            nm = k - 1;
            y = w[nm-1];
            g = rv1[nm-1];
            h = rv1[k-1];

            f = ((y-z)*(y+z)+(g-h)*(g+h))/(2.0f*h*y);
            g = pythag(f, (T)1);
            f = ((x-z)*(x+z)+h*((y/(f+SIGN(g,f)))-h))/x;
            c = s = 1.0f;
            for (j = l; j <= nm; ++j)
            {
                i = j + 1;
                g = rv1[i-1];
                y = w[i-1];
                h = s * g;
                g = c * g;
                z = pythag(f, h);
                rv1[j-1] = z;
                c = f/z;
                s = h/z;
                f = x*c+g*s;
                g = g*c-x*s;
                h = y*s;
                y *= c;

                for (jj = 1; jj <= n; ++jj)
                {
                    x = v[jj-1][j-1];
                    z = v[jj-1][i-1];
                    v[jj-1][j-1] = x*c+z*s;
                    v[jj-1][i-1] = z*c-x*s;
                }
                z = pythag(f, h);
                w[j-1] = z;
                if (z)
                {
                    z = 1.0f/z;
                    c = f * z;
                    s = h * z;
                }
                f = c*g+s*y;
                x = c*y-s*g;

                for (jj = 1; jj <= m; ++jj)
                {
                    y = in[jj-1][j-1];
                    z = in[jj-1][i-1];
                    in[jj-1][j-1] = y*c+z*s;
                    in[jj-1][i-1] = z*c-y*s;
                }
            }
            rv1[l-1] = 0.0f;
            rv1[k-1] = f;
            w[k-1] = x;
        }
    }
    free (rv1);
}
template void svd<float>(float ** in, size_t m, size_t n, float * w, float ** v);
template void svd<double>(double ** in, size_t m, size_t n, double * w, double ** v);
/* *************************************************************** */
/* *************************************************************** */
#endif // _REG_MATHS_CPP
