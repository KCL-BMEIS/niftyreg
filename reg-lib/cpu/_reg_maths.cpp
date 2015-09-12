#ifndef _REG_MATHS_CPP
#define _REG_MATHS_CPP

#define USE_EIGEN

#include "_reg_maths.h"
//STD
#include <map>
#include <vector>
// Eigen headers are in there because of the nvcc preprocessing step
#include "Eigen/Core"
#include "Eigen/SVD"
#include "Eigen/unsupported/MatrixFunctions"

#define mat(i,j,dim) mat[i*dim+j]

/* *************************************************************** */
/* *************************************************************** */
void reg_logarithm_tensor(mat33 *in_tensor)
{
    int sm, sn;
    Eigen::Matrix3d tensor, sing;

    // Convert to Eigen format
    for (sm = 0; sm < 3; sm++)
        for (sn = 0; sn < 3; sn++)
            tensor(sm, sn) = static_cast<double>(in_tensor->m[sm][sn]);

    // Decompose the input tensor
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(tensor, Eigen::ComputeThinV | Eigen::ComputeThinU);

    // Set a matrix containing the eigen values
    sing.setZero();
    sing(0, 0) = svd.singularValues()(0);
    sing(1, 1) = svd.singularValues()(1);
    sing(2, 2) = svd.singularValues()(2);

    if (sing(0, 0) <= 0)
        sing(0, 0) = std::numeric_limits<double>::epsilon();
    if (sing(1, 1) <= 0)
        sing(1, 1) = std::numeric_limits<double>::epsilon();
    if (sing(2, 2) <= 0)
        sing(2, 2) = std::numeric_limits<double>::epsilon();

    // Compute Rt log(E) R
    tensor = svd.matrixU() * sing.log() * svd.matrixU().transpose();

    // Convert the result to mat33 format
    for (sm = 0; sm < 3; sm++)
        for (sn = 0; sn < 3; sn++)
            in_tensor->m[sm][sn] = static_cast<float>(tensor(sm, sn));
}
/* *************************************************************** */
void reg_exponentiate_logged_tensor(mat33 *in_tensor)
{
    int sm, sn;
    Eigen::Matrix3d tensor;

    // Convert to Eigen format
    for (sm = 0; sm < 3; sm++)
        for (sn = 0; sn < 3; sn++)
            tensor(sm, sn) = static_cast<double>(in_tensor->m[sm][sn]);

    // Compute Rt exp(E) R
    tensor = tensor.exp();

    // Convert the result to mat33 format
    for (sm = 0; sm < 3; sm++)
        for (sn = 0; sn < 3; sn++)
            in_tensor->m[sm][sn] = static_cast<float>(tensor(sm, sn));
}
/* *************************************************************** */
/* *************************************************************** */
/** @brief SVD
 * @param in input matrix to decompose - in place
 * @param size_m row
 * @param size_n colomn
 * @param w diagonal term
 * @param v rotation part
 */
template<class T>
void svd(T **in, size_t size_m, size_t size_n, T * w, T **v) {
    if (size_m == 0 || size_n == 0) {
        reg_print_fct_error("svd");
        reg_print_msg_error("The specified matrix is empty");
        reg_exit(1);
    }

#ifdef _WIN32
    long sm, sn, sn2;
    long size__m = (long)size_m, size__n = (long)size_n;
#else
    size_t sm, sn, sn2;
    size_t size__m = size_m, size__n = size_n;
#endif
    Eigen::MatrixXd m(size_m, size_n);

    //Convert to Eigen matrix
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(in,m, size__m, size__n) \
   private(sm, sn)
#endif
    for (sm = 0; sm < size__m; sm++)
    {
        for (sn = 0; sn < size__n; sn++)
        {
            m(sm, sn) = static_cast<double>(in[sm][sn]);
        }
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(m, Eigen::ComputeThinU | Eigen::ComputeThinV);

#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(in,svd,v,w, size__n,size__m) \
   private(sn2, sn, sm)
#endif
    for (sn = 0; sn < size__n; sn++) {
        w[sn] = svd.singularValues()(sn);
        for (sn2 = 0; sn2 < size__n; sn2++) {
            v[sn2][sn] = static_cast<T>(svd.matrixV()(sn2, sn));
        }
        for (sm = 0; sm < size__m; sm++) {
            in[sm][sn] = static_cast<T>(svd.matrixU()(sm, sn));
        }
    }
}
template void svd<float>(float **in, size_t m, size_t n, float * w, float **v);
template void svd<double>(double **in, size_t m, size_t n, double * w, double **v);
/* *************************************************************** */
/* *************************************************************** */
/**
* @brief SVD
* @param in input matrix to decompose
* @param size_m row
* @param size_n colomn
* @param U unitary matrices
* @param S diagonal matrix
* @param V unitary matrices
*  X = U*S*V'
*/
template<class T>
void svd(T **in, size_t size_m, size_t size_n, T ***U, T ***S, T ***V) {
    if (in == NULL) {
        reg_print_fct_error("svd");
        reg_print_msg_error("The specified matrix is empty");
        reg_exit(1);
    }

#ifdef _WIN32
    long sm, sn, sn2, min_dim;
    long size__m = (long)size_m, size__n = (long)size_n;
#else
    size_t sm, sn, sn2, min_dim;
    size_t size__m = size_m, size__n = size_n;
#endif
    Eigen::MatrixXd m(size__m, size__n);

    //Convert to Eigen matrix
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(in, m, size__m, size__n) \
   private(sm, sn)
#endif
    for (sm = 0; sm < size__m; sm++)
    {
        for (sn = 0; sn < size__n; sn++)
        {
            m(sm, sn) = static_cast<double>(in[sm][sn]);
        }
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(m, Eigen::ComputeThinU | Eigen::ComputeThinV);

    //std::cout << "Its singular values are:" << std::endl << svd.singularValues() << std::endl;
    //std::cout << "Its left singular vectors are the columns of the thin U matrix:" << std::endl << svd.matrixU() << std::endl;
    //std::cout << "Its right singular vectors are the columns of the thin V matrix:" << std::endl << svd.matrixV() << std::endl;

    int i, j;
    min_dim = std::min(size__m, size__n);
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(in, svd, U, S, V, size__n, size__m, min_dim) \
   private(i, j)
#endif
    //Convert to C matrix
    for (i = 0; i < min_dim; i++) {
        for (j = 0; j < min_dim; j++) {
            if (i == j) {
                (*S)[i][j] = static_cast<T>(svd.singularValues()(i));
            }
            else {
                (*S)[i][j] = 0;
            }
        }
    }

    if (size__m > size__n) {
        //Convert to C matrix
        for (i = 0; i < min_dim; i++) {
            for (j = 0; j < min_dim; j++) {
                (*V)[i][j] = static_cast<T>(svd.matrixV()(i, j));

            }
        }
        for (i = 0; i < size__m; i++) {
            for (j = 0; j < size__n; j++) {
                (*U)[i][j] = static_cast<T>(svd.matrixU()(i, j));
            }
        }
    }
    else {
        //Convert to C matrix
        for (i = 0; i < min_dim; i++) {
            for (j = 0; j < min_dim; j++) {
                (*U)[i][j] = static_cast<T>(svd.matrixU()(i, j));

            }
        }
        for (i = 0; i < size__n; i++) {
            for (j = 0; j < size__m; j++) {
                (*V)[i][j] = static_cast<T>(svd.matrixV()(i, j));
            }
        }
    }

}
template void svd<float>(float **in, size_t size_m, size_t size_n, float ***U, float ***S, float ***V);
template void svd<double>(double **in, size_t size_m, size_t size_n, double ***U, double ***S, double ***V);
/* *************************************************************** */
/* *************************************************************** */
template<class T>
void reg_LUdecomposition(T *mat,
    size_t dim,
    size_t *index)
{
    T *vv = (T *)malloc(dim * sizeof(T));
    size_t i, j, k, imax = 0;

    for (i = 0; i < dim; ++i)
    {
        T big = 0.f;
        T temp;
        for (j = 0; j < dim; ++j)
            if ((temp = fabs(mat(i, j, dim)))>big)
                big = temp;
        if (big == 0.f)
        {
            reg_print_fct_error("reg_LUdecomposition");
            reg_print_msg_error("Singular matrix");
            reg_exit(1);
        }
        vv[i] = 1.0 / big;
    }
    for (j = 0; j < dim; ++j)
    {
        for (i = 0; i < j; ++i)
        {
            T sum = mat(i, j, dim);
            for (k = 0; k < i; k++) sum -= mat(i, k, dim)*mat(k, j, dim);
            mat(i, j, dim) = sum;
        }
        T big = 0.f;
        T dum;
        for (i = j; i < dim; ++i)
        {
            T sum = mat(i, j, dim);
            for (k = 0; k < j; ++k) sum -= mat(i, k, dim)*mat(k, j, dim);
            mat(i, j, dim) = sum;
            if ((dum = vv[i] * fabs(sum)) >= big)
            {
                big = dum;
                imax = i;
            }
        }
        if (j != imax)
        {
            for (k = 0; k < dim; ++k)
            {
                dum = mat(imax, k, dim);
                mat(imax, k, dim) = mat(j, k, dim);
                mat(j, k, dim) = dum;
            }
            vv[imax] = vv[j];
        }
        index[j] = imax;
        if (mat(j, j, dim) == 0) mat(j, j, dim) = 1.0e-20;
        if (j != dim - 1)
        {
            dum = 1.0 / mat(j, j, dim);
            for (i = j + 1; i < dim; ++i) mat(i, j, dim) *= dum;
        }
    }
    free(vv);
    return;
}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
void reg_matrixInvertMultiply(T *mat,
    size_t dim,
    size_t *index,
    T *vec)
{
    // Perform the LU decomposition if necessary
    if (index == NULL)
        reg_LUdecomposition(mat, dim, index);

    int ii = 0;
    for (int i = 0; i < (int)dim; ++i)
    {
        int ip = index[i];
        T sum = vec[ip];
        vec[ip] = vec[i];
        if (ii != 0)
        {
            for (int j = ii - 1; j < i; ++j)
                sum -= mat(i, j, dim)*vec[j];
        }
        else if (sum != 0)
            ii = i + 1;
        vec[i] = sum;
    }
    for (int i = (int)dim - 1; i > -1; --i)
    {
        T sum = vec[i];
        for (int j = i + 1; j < (int)dim; ++j)
            sum -= mat(i, j, dim)*vec[j];
        vec[i] = sum / mat(i, i, dim);
    }
}
template void reg_matrixInvertMultiply<float>(float *, size_t, size_t *, float *);
template void reg_matrixInvertMultiply<double>(double *, size_t, size_t *, double *);
/* *************************************************************** */
/* *************************************************************** */
template<class T>
void reg_matrixMultiply(T *mat1,
    T *mat2,
    size_t *dim1,
    size_t *dim2,
    T * &res)
{
    // First check that the dimension are appropriate
    if (dim1[1] != dim2[0])
    {
        char text[255]; sprintf(text, "Matrices can not be multiplied due to their size: [%zu %zu] [%zu %zu]",
            dim1[0], dim1[1], dim2[0], dim2[1]);
        reg_print_fct_error("reg_matrixMultiply");
        reg_print_msg_error(text);
        reg_exit(1);
    }
    int resDim[2] = { dim1[0], dim2[1] };
    // Allocate the result matrix
    if (res != NULL)
        free(res);
    res = (T *)calloc(resDim[0] * resDim[1], sizeof(T));
    // Multiply both matrices
    for (int j = 0; j < resDim[1]; ++j)
    {
        for (int i = 0; i < resDim[0]; ++i)
        {
            double sum = 0.0;
            for (int k = 0; k < dim1[1]; ++k)
            {
                sum += mat1[k * dim1[0] + i] * mat2[j * dim2[0] + k];
            }
            res[j * resDim[0] + i] = sum;
        } // i
    } // j
}
template void reg_matrixMultiply<float>(float *, float *, size_t *, size_t *, float * &);
template void reg_matrixMultiply<double>(double *, double *, size_t *, size_t *, double * &);
/* *************************************************************** */
/* *************************************************************** */
/* *************************************************************** */
/* *************************************************************** */
template<class T>
T* reg_matrix1DAllocate(size_t arraySize) {
    T* res = (T*)malloc(arraySize*sizeof(T));
    return res;
}
template float* reg_matrix1DAllocate<float>(size_t arraySize);
template double* reg_matrix1DAllocate<double>(size_t arraySize);
/* *************************************************************** */
template<class T>
T* reg_matrix1DAllocateAndInitToZero(size_t arraySize) {
    T* res = (T*)calloc(arraySize, sizeof(T));
    return res;
}
template float* reg_matrix1DAllocateAndInitToZero<float>(size_t arraySize);
template double* reg_matrix1DAllocateAndInitToZero<double>(size_t arraySize);
/* *************************************************************** */
template<class T>
void reg_matrix1DDeallocate(T* mat) {
    free(mat);
}
template void reg_matrix1DDeallocate<float>(float* mat);
template void reg_matrix1DDeallocate<double>(double* mat);
/* *************************************************************** */
template<class T>
T** reg_matrix2DAllocate(size_t arraySizeX, size_t arraySizeY) {
    T** res;
    res = (T**)malloc(arraySizeX*sizeof(T*));
    for (int i = 0; i < arraySizeX; i++) {
        res[i] = (T*)malloc(arraySizeY*sizeof(T));
    }
    return res;
}
template float** reg_matrix2DAllocate<float>(size_t arraySizeX, size_t arraySizeY);
template double** reg_matrix2DAllocate<double>(size_t arraySizeX, size_t arraySizeY);
/* *************************************************************** */
template<class T>
T** reg_matrix2DAllocateAndInitToZero(size_t arraySizeX, size_t arraySizeY) {
    T** res;
    res = (T**)calloc(arraySizeX, sizeof(T*));
    for (int i = 0; i < arraySizeX; i++) {
        res[i] = (T*)calloc(arraySizeY, sizeof(T));
    }
    return res;
}
template float** reg_matrix2DAllocateAndInitToZero<float>(size_t arraySizeX, size_t arraySizeY);
template double** reg_matrix2DAllocateAndInitToZero<double>(size_t arraySizeX, size_t arraySizeY);
/* *************************************************************** */
template<class T>
void reg_matrix2DDeallocate(size_t arraySizeX, T** mat) {
    for (int i = 0; i < arraySizeX; i++) {
        free(mat[i]);
    }
    free(mat);
}
template void reg_matrix2DDeallocate<float>(size_t arraySizeX, float** mat);
template void reg_matrix2DDeallocate<double>(size_t arraySizeX, double** mat);
/* *************************************************************** */
template<class T>
T** reg_matrix2DTranspose(T** mat, size_t arraySizeX, size_t arraySizeY) {
    T** res;
    res = (T**)malloc(arraySizeY*sizeof(T*));
    for (int i = 0; i < arraySizeY; i++) {
        res[i] = (T*)malloc(arraySizeX*sizeof(T));
    }
    for (int i = 0; i < arraySizeX; i++) {
        for (int j = 0; j < arraySizeY; j++) {
            res[j][i] = mat[i][j];
        }
    }
    return res;
}
template float** reg_matrix2DTranspose<float>(float** mat, size_t arraySizeX, size_t arraySizeY);
template double** reg_matrix2DTranspose<double>(double** mat, size_t arraySizeX, size_t arraySizeY);
/* *************************************************************** */
template<class T>
T** reg_matrix2DMultiply(T** mat1, size_t mat1X, size_t mat1Y, T** mat2, size_t mat2X, size_t mat2Y, bool transposeMat2) {
    if (transposeMat2 == false) {
        // First check that the dimension are appropriate
        if (mat1Y != mat2X) {
            char text[255]; sprintf(text, "Matrices can not be multiplied due to their size: [%zu %zu] [%zu %zu]",
                mat1X, mat1Y, mat2X, mat2Y);
            reg_print_fct_error("reg_matrix2DMultiply");
            reg_print_msg_error(text);
            reg_exit(1);
        }
        size_t nbElement = mat1Y;
        T** res;
        res = (T**)malloc(mat1X*sizeof(T*));
        for (size_t i = 0; i < mat1X; i++) {
            res[i] = (T*)malloc(mat2Y*sizeof(T));
        }
        for (size_t i = 0; i < mat1X; i++) {
            for (size_t j = 0; j < mat2Y; j++) {
                res[i][j] = 0;
                for (int k = 0; k < nbElement; k++) {
                    res[i][j] += static_cast<T>(static_cast<double>(mat1[i][k]) * static_cast<double>(mat2[k][j]));
                }
            }
        }
        return res;
    }
    else {
        // First check that the dimension are appropriate
        if (mat1Y != mat2Y) {
            char text[255]; sprintf(text, "Matrices can not be multiplied due to their size: [%zu %zu] [%zu %zu]",
                mat1X, mat1Y, mat2Y, mat2X);
            reg_print_fct_error("reg_matrix2DMultiply");
            reg_print_msg_error(text);
            reg_exit(1);
        }
        size_t nbElement = mat1Y;
        T** res;
        res = (T**)malloc(mat1X*sizeof(T*));
        for (size_t i = 0; i < mat1X; i++) {
            res[i] = (T*)malloc(mat2X*sizeof(T));
        }
        for (size_t i = 0; i < mat1X; i++) {
            for (size_t j = 0; j < mat2X; j++) {
                res[i][j] = 0;
                for (size_t k = 0; k < nbElement; k++) {
                    res[i][j] += static_cast<T>(static_cast<double>(mat1[i][k]) * static_cast<double>(mat2[j][k]));
                }
            }
        }
        return res;
    }
}
template float** reg_matrix2DMultiply<float>(float** mat1, size_t mat1X, size_t mat1Y, float** mat2, size_t mat2X, size_t mat2Y, bool transposeMat2);
template double** reg_matrix2DMultiply<double>(double** mat1, size_t mat1X, size_t mat1Y, double** mat2, size_t mat2X, size_t mat2Y, bool transposeMat2);
/* *************************************************************** */
template<class T>
void reg_matrix2DMultiply(T** mat1, size_t mat1X, size_t mat1Y, T** mat2, size_t mat2X, size_t mat2Y, T** res, bool transposeMat2) {
    if (transposeMat2 == false) {
        // First check that the dimension are appropriate
        if (mat1Y != mat2X) {
            char text[255]; sprintf(text, "Matrices can not be multiplied due to their size: [%zu %zu] [%zu %zu]",
                mat1X, mat1Y, mat2X, mat2Y);
            reg_print_fct_error("reg_matrix2DMultiply");
            reg_print_msg_error(text);
            reg_exit(1);
        }
        size_t nbElement = mat1Y;

        for (size_t i = 0; i < mat1X; i++) {
            for (size_t j = 0; j < mat2Y; j++) {
                res[i][j] = 0;
                for (size_t k = 0; k < nbElement; k++) {
                    res[i][j] += static_cast<T>(static_cast<double>(mat1[i][k]) * static_cast<double>(mat2[k][j]));
                }
            }
        }
    }
    else {
        // First check that the dimension are appropriate
        if (mat1Y != mat2Y) {
            char text[255]; sprintf(text, "Matrices can not be multiplied due to their size: [%zu %zu] [%zu %zu]",
                mat1X, mat1Y, mat2Y, mat2X);
            reg_print_fct_error("reg_matrix2DMultiply");
            reg_print_msg_error(text);
            reg_exit(1);
        }
        size_t nbElement = mat1Y;

        for (size_t i = 0; i < mat1X; i++) {
            for (size_t j = 0; j < mat2X; j++) {
                res[i][j] = 0;
                for (size_t k = 0; k < nbElement; k++) {
                    res[i][j] += static_cast<T>(static_cast<double>(mat1[i][k]) * static_cast<double>(mat2[j][k]));
                }
            }
        }
    }
}
template void reg_matrix2DMultiply<float>(float** mat1, size_t mat1X, size_t mat1Y, float** mat2, size_t mat2X, size_t mat2Y, float** res, bool transposeMat2);
template void reg_matrix2DMultiply<double>(double** mat1, size_t mat1X, size_t mat1Y, double** mat2, size_t mat2X, size_t mat2Y, double** res, bool transposeMat2);
/* *************************************************************** */
// Multiply a matrix with a vector - we assume correct dimension
template<class T>
T* reg_matrix2DVectorMultiply(T** mat, size_t m, size_t n, T* vect) {

    T* res = (T*)malloc(m*sizeof(T));

    for (int i = 0; i < m; i++) {
        res[i] = 0;
        for (int k = 0; k < n; k++) {
            res[i] += static_cast<T>(static_cast<double>(mat[i][k]) * static_cast<double>(vect[k]));
        }
    }
    return res;
}
template float* reg_matrix2DVectorMultiply<float>(float** mat, size_t m, size_t n, float* vect);
template double* reg_matrix2DVectorMultiply<double>(double** mat, size_t m, size_t n, double* vect);
/* *************************************************************** */
template<class T>
void reg_matrix2DVectorMultiply(T** mat, size_t m, size_t n, T* vect, T* res) {
    for (int i = 0; i < m; i++) {
        res[i] = 0;
        for (int k = 0; k < n; k++) {
            res[i] += static_cast<T>(static_cast<double>(mat[i][k]) * static_cast<double>(vect[k]));
        }
    }
}
template void reg_matrix2DVectorMultiply<float>(float** mat, size_t m, size_t n, float* vect, float* res);
template void reg_matrix2DVectorMultiply<double>(double** mat, size_t m, size_t n, double* vect, double* res);
/* *************************************************************** */
template<class T>
T reg_matrix2DDet(T** mat, size_t m, size_t n) {
    if (m != n) {
        char text[255]; sprintf(text, "The matrix have to be square: [%zu %zu]",
            m, n);
        reg_print_fct_error("reg_matrix2DDeterminant");
        reg_print_msg_error(text);
        reg_exit(1);
    }
    double res;
    if (m == 2) {
        res = static_cast<double>(mat[0][0]) * static_cast<double>(mat[1][1]) - static_cast<double>(mat[1][0]) * static_cast<double>(mat[0][1]);
    }
    else if (m == 3) {
        res = (static_cast<double>(mat[0][0]) * (static_cast<double>(mat[1][1]) * static_cast<double>(mat[2][2]) - static_cast<double>(mat[1][2]) * static_cast<double>(mat[2][1]))) -
            (static_cast<double>(mat[0][1]) * (static_cast<double>(mat[1][0]) * static_cast<double>(mat[2][2]) - static_cast<double>(mat[1][2]) * static_cast<double>(mat[2][0]))) +
            (static_cast<double>(mat[0][2]) * (static_cast<double>(mat[1][0]) * static_cast<double>(mat[2][1]) - static_cast<double>(mat[1][1]) * static_cast<double>(mat[2][0])));
    }
    else {
        // Convert to Eigen format
        Eigen::MatrixXd eigenRes(m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                eigenRes(i, j) = static_cast<double>(mat[i][j]);
            }
        }
        res = eigenRes.determinant();
    }
    return static_cast<T>(res);
}
template float reg_matrix2DDet<float>(float** mat, size_t m, size_t n);
template double reg_matrix2DDet<double>(double** mat, size_t m, size_t n);
/* *************************************************************** */
/* *************************************************************** */
/* *************************************************************** */
/* *************************************************************** */
void estimate_rigid_transformation2D(float** points1, float** points2, int num_points, mat44 * transformation)
{

    double centroid_target[2] = { 0.0 };
    double centroid_result[2] = { 0.0 };

    for (unsigned j = 0; j < num_points; ++j) {
        centroid_target[0] += points1[j][0];
        centroid_target[1] += points1[j][1];
        centroid_result[0] += points2[j][0];
        centroid_result[1] += points2[j][1];
    }

    centroid_target[0] /= static_cast<double>(num_points);
    centroid_target[1] /= static_cast<double>(num_points);

    centroid_result[0] /= static_cast<double>(num_points);
    centroid_result[1] /= static_cast<double>(num_points);

    float **u = reg_matrix2DAllocateAndInitToZero<float>(2, 2);
    float * w = reg_matrix1DAllocate<float>(2);
    float **v = reg_matrix2DAllocate<float>(2, 2);
    float **ut = reg_matrix2DAllocate<float>(2, 2);
    float **r = reg_matrix2DAllocate<float>(2, 2);

    // Demean the input points
    for (unsigned j = 0; j < num_points; ++j) {
        points1[j][0] -= centroid_target[0];
        points1[j][1] -= centroid_target[1];

        points2[j][0] -= centroid_result[0];
        points2[j][1] -= centroid_result[1];

        u[0][0] += points1[j][0] * points2[j][0];
        u[0][1] += points1[j][0] * points2[j][1];

        u[1][0] += points1[j][1] * points2[j][0];
        u[1][1] += points1[j][1] * points2[j][1];
    }

    svd(u, 2, 2, w, v);

    // Calculate transpose
    ut[0][0] = u[0][0];
    ut[1][0] = u[0][1];

    ut[0][1] = u[1][0];
    ut[1][1] = u[1][1];

    // Calculate the rotation matrix
    reg_matrix2DMultiply<float>(v, 2, 2, ut, 2, 2, r, false);

    float det = reg_matrix2DDet<float>(r, 2, 2);

    // Take care of possible reflection
    if (det < 0.0f) {
        v[0][1] = -v[0][1];
        v[1][1] = -v[1][1];
        reg_matrix2DMultiply<float>(v, 2, 2, ut, 2, 2, r, false);
    }

    // Calculate the translation
    float t[2];
    t[0] = centroid_result[0] - (r[0][0] * centroid_target[0] +
        r[0][1] * centroid_target[1]);

    t[1] = centroid_result[1] - (r[1][0] * centroid_target[0] +
        r[1][1] * centroid_target[1]);

    transformation->m[0][0] = r[0][0];
    transformation->m[0][1] = r[0][1];
    transformation->m[0][3] = t[0];

    transformation->m[1][0] = r[1][0];
    transformation->m[1][1] = r[1][1];
    transformation->m[1][3] = t[1];

    transformation->m[2][0] = 0.0f;
    transformation->m[2][1] = 0.0f;
    transformation->m[2][2] = 1.0f;
    transformation->m[2][3] = 0.0f;

    transformation->m[0][2] = 0.0f;
    transformation->m[1][2] = 0.0f;
    transformation->m[3][2] = 0.0f;

    transformation->m[3][0] = 0.0f;
    transformation->m[3][1] = 0.0f;
    transformation->m[3][2] = 0.0f;
    transformation->m[3][3] = 1.0f;

    // Do the deletion here
    reg_matrix2DDeallocate(2, u);
    reg_matrix1DDeallocate(w);
    reg_matrix2DDeallocate(2, v);
    reg_matrix2DDeallocate(2, ut);
    reg_matrix2DDeallocate(2, r);
}
/* *************************************************************** */
void estimate_rigid_transformation2D(std::vector<_reg_sorted_point2D> &points, mat44 * transformation)
{

    unsigned int num_points = points.size();
    float** points1 = reg_matrix2DAllocate<float>(num_points, 2);
    float** points2 = reg_matrix2DAllocate<float>(num_points, 2);
    for (unsigned int i = 0; i < num_points; i++) {
        points1[i][0] = points[i].target[0];
        points1[i][1] = points[i].target[1];
        points2[i][0] = points[i].result[0];
        points2[i][1] = points[i].result[1];
    }
    estimate_rigid_transformation2D(points1, points2, num_points, transformation);
    //FREE MEMORY
    reg_matrix2DDeallocate(num_points, points1);
    reg_matrix2DDeallocate(num_points, points2);
}
/* *************************************************************** */
void estimate_rigid_transformation3D(float** points1, float** points2, int num_points, mat44 * transformation)
{

    double centroid_target[3] = { 0.0 };
    double centroid_result[3] = { 0.0 };


    for (unsigned j = 0; j < num_points; ++j)
    {
        centroid_target[0] += points1[j][0];
        centroid_target[1] += points1[j][1];
        centroid_target[2] += points1[j][2];

        centroid_result[0] += points2[j][0];
        centroid_result[1] += points2[j][1];
        centroid_result[2] += points2[j][2];
    }

    centroid_target[0] /= static_cast<double>(num_points);
    centroid_target[1] /= static_cast<double>(num_points);
    centroid_target[2] /= static_cast<double>(num_points);

    centroid_result[0] /= static_cast<double>(num_points);
    centroid_result[1] /= static_cast<double>(num_points);
    centroid_result[2] /= static_cast<double>(num_points);

    float **u  = reg_matrix2DAllocateAndInitToZero<float>(3, 3);
    float * w = reg_matrix1DAllocate<float>(3);
    float **v  = reg_matrix2DAllocate<float>(3, 3);
    float **ut = reg_matrix2DAllocate<float>(3, 3);
    float **r  = reg_matrix2DAllocate<float>(3, 3);

    // Demean the input points
    for (unsigned j = 0; j < num_points; ++j)
    {
        points1[j][0] -= centroid_target[0];
        points1[j][1] -= centroid_target[1];
        points1[j][2] -= centroid_target[2];

        points2[j][0] -= centroid_result[0];
        points2[j][1] -= centroid_result[1];
        points2[j][2] -= centroid_result[2];

        u[0][0] += points1[j][0] * points2[j][0];
        u[0][1] += points1[j][0] * points2[j][1];
        u[0][2] += points1[j][0] * points2[j][2];

        u[1][0] += points1[j][1] * points2[j][0];
        u[1][1] += points1[j][1] * points2[j][1];
        u[1][2] += points1[j][1] * points2[j][2];

        u[2][0] += points1[j][2] * points2[j][0];
        u[2][1] += points1[j][2] * points2[j][1];
        u[2][2] += points1[j][2] * points2[j][2];

    }

    svd(u, 3, 3, w, v);

    // Calculate transpose
    ut[0][0] = u[0][0];
    ut[1][0] = u[0][1];
    ut[2][0] = u[0][2];

    ut[0][1] = u[1][0];
    ut[1][1] = u[1][1];
    ut[2][1] = u[1][2];

    ut[0][2] = u[2][0];
    ut[1][2] = u[2][1];
    ut[2][2] = u[2][2];

    // Calculate the rotation matrix
    reg_matrix2DMultiply<float>(v, 3, 3, ut, 3, 3, r, false);

    float det = reg_matrix2DDet<float>(r, 3, 3);

    // Take care of possible reflection
    if (det < 0.0f)
    {
        v[0][2] = -v[0][2];
        v[1][2] = -v[1][2];
        v[2][2] = -v[2][2];

    }
    // Calculate the rotation matrix
    reg_matrix2DMultiply<float>(v, 3, 3, ut, 3, 3, r, false);

    // Calculate the translation
    float t[3];
    t[0] = centroid_result[0] - (r[0][0] * centroid_target[0] +
        r[0][1] * centroid_target[1] +
        r[0][2] * centroid_target[2]);

    t[1] = centroid_result[1] - (r[1][0] * centroid_target[0] +
        r[1][1] * centroid_target[1] +
        r[1][2] * centroid_target[2]);

    t[2] = centroid_result[2] - (r[2][0] * centroid_target[0] +
        r[2][1] * centroid_target[1] +
        r[2][2] * centroid_target[2]);

    transformation->m[0][0] = r[0][0];
    transformation->m[0][1] = r[0][1];
    transformation->m[0][2] = r[0][2];
    transformation->m[0][3] = t[0];

    transformation->m[1][0] = r[1][0];
    transformation->m[1][1] = r[1][1];
    transformation->m[1][2] = r[1][2];
    transformation->m[1][3] = t[1];

    transformation->m[2][0] = r[2][0];
    transformation->m[2][1] = r[2][1];
    transformation->m[2][2] = r[2][2];
    transformation->m[2][3] = t[2];

    transformation->m[3][0] = 0.0f;
    transformation->m[3][1] = 0.0f;
    transformation->m[3][2] = 0.0f;
    transformation->m[3][3] = 1.0f;

    // Do the deletion here
    reg_matrix2DDeallocate(2, u);
    reg_matrix1DDeallocate(w);
    reg_matrix2DDeallocate(2, v);
    reg_matrix2DDeallocate(2, ut);
    reg_matrix2DDeallocate(2, r);
}
/* *************************************************************** */
void estimate_rigid_transformation3D(std::vector<_reg_sorted_point3D> &points, mat44 * transformation)
{
    unsigned int num_points = points.size();
    float** points1 = reg_matrix2DAllocate<float>(num_points, 3);
    float** points2 = reg_matrix2DAllocate<float>(num_points, 3);
    for (unsigned int i = 0; i < num_points; i++) {
        points1[i][0] = points[i].target[0];
        points1[i][1] = points[i].target[1];
        points1[i][2] = points[i].target[2];
        points2[i][0] = points[i].result[0];
        points2[i][1] = points[i].result[1];
        points2[i][2] = points[i].result[2];
    }
    estimate_rigid_transformation3D(points1, points2, num_points, transformation);
    //FREE MEMORY
    reg_matrix2DDeallocate(num_points, points1);
    reg_matrix2DDeallocate(num_points, points2);
}
/* *************************************************************** */
void estimate_affine_transformation2D(float** points1, float** points2, int num_points, mat44 * transformation)
{
    //We assume same number of points in both arrays    
    int num_equations = num_points * 2;
    unsigned c = 0;
    float** A = reg_matrix2DAllocate<float>(num_equations, 6);

    for (unsigned k = 0; k < num_points; ++k) {
        c = k * 2;

        A[c][0] = points1[k][0];
        A[c][1] = points1[k][1];
        A[c][2] = A[c][3] = A[c][5] = 0.0f;
        A[c][4] = 1.0f;

        A[c + 1][2] = points1[k][0];
        A[c + 1][3] = points1[k][1];
        A[c + 1][0] = A[c + 1][1] = A[c + 1][4] = 0.0f;
        A[c + 1][5] = 1.0f;
    }
    
    float* w  = reg_matrix1DAllocate<float>(6);
    float** v = reg_matrix2DAllocate<float>(6, 6);

    svd(A, num_equations, 6, w, v);

    for (unsigned k = 0; k < 6; ++k) {
        if (w[k] < 0.0001) {
            w[k] = 0.0f;
        }
        else {
            w[k] = static_cast<float>(1.0 / static_cast<double>(w[k]));
        }
    }

    // Now we can compute the pseudoinverse which is given by
    // V*inv(W)*U'
    // First compute the V * inv(w) in place.
    // Simply scale each column by the corresponding singular value
    for (unsigned k = 0; k < 6; ++k) {
        for (unsigned j = 0; j < 6; ++j) {
            v[j][k] = static_cast<float>(static_cast<double>(v[j][k]) * static_cast<double>(w[k]));
        }
    }

    float** r = reg_matrix2DAllocate<float>(6, num_equations);
    reg_matrix2DMultiply<float>(v, 6, 6, A, num_equations, 6, r, true);
    // Now r contains the pseudoinverse
    // Create vector b and then multiple r*b to get the affine paramsA
    float* b = reg_matrix1DAllocate<float>(num_equations);
    for (unsigned k = 0; k < num_points; ++k) {
        c = k * 2;
        b[c] = points2[k][0];
        b[c + 1] = points2[k][1];
    }

    float* transform = reg_matrix1DAllocate<float>(6);
    reg_matrix2DVectorMultiply<float>(r, 6, num_equations, b, transform);

    transformation->m[0][0] = transform[0];
    transformation->m[0][1] = transform[1];
    transformation->m[0][2] = 0.0f;
    transformation->m[0][3] = transform[4];

    transformation->m[1][0] = transform[2];
    transformation->m[1][1] = transform[3];
    transformation->m[1][2] = 0.0f;
    transformation->m[1][3] = transform[5];

    transformation->m[2][0] = 0.0f;
    transformation->m[2][1] = 0.0f;
    transformation->m[2][2] = 1.0f;
    transformation->m[2][3] = 0.0f;

    transformation->m[3][0] = 0.0f;
    transformation->m[3][1] = 0.0f;
    transformation->m[3][2] = 0.0f;
    transformation->m[3][3] = 1.0f;

    // Do the deletion here
    reg_matrix1DDeallocate(transform);
    reg_matrix1DDeallocate(b);
    reg_matrix2DDeallocate(6, r);
    reg_matrix2DDeallocate(6, v);
    reg_matrix1DDeallocate(w);
    reg_matrix2DDeallocate(num_equations, A);
}
/* *************************************************************** */
void estimate_affine_transformation2D(std::vector<_reg_sorted_point2D> &points, mat44 * transformation)
{
    unsigned int num_points = points.size();
    float** points1 = reg_matrix2DAllocate<float>(num_points, 2);
    float** points2 = reg_matrix2DAllocate<float>(num_points, 2);
    for (unsigned int i = 0; i < num_points; i++) {
        points1[i][0] = points[i].target[0];
        points1[i][1] = points[i].target[1];
        points2[i][0] = points[i].result[0];
        points2[i][1] = points[i].result[1];
    }
    estimate_affine_transformation2D(points1, points2, num_points, transformation);
    //FREE MEMORY
    reg_matrix2DDeallocate(num_points, points1);
    reg_matrix2DDeallocate(num_points, points2);
}
/* *************************************************************** */
// estimate an affine transformation using least square
void estimate_affine_transformation3D(float** points1, float** points2, int num_points, mat44 * transformation)
{
    //We assume same number of points in both arrays 

    // Create our A matrix
    // we need at least 4 points. Assuming we have that here.
    int num_equations = num_points * 3;
    unsigned c = 0;
    float** A = reg_matrix2DAllocate<float>(num_equations, 12);

    for (unsigned k = 0; k < num_points; ++k) {
        c = k * 3;
        A[c][0] = points1[k][0];
        A[c][1] = points1[k][1];
        A[c][2] = points1[k][2];
        A[c][3] = A[c][4] = A[c][5] = A[c][6] = A[c][7] = A[c][8] = A[c][10] = A[c][11] = 0.0f;
        A[c][9] = 1.0f;

        A[c + 1][3] = points1[k][0];
        A[c + 1][4] = points1[k][1];
        A[c + 1][5] = points1[k][2];
        A[c + 1][0] = A[c + 1][1] = A[c + 1][2] = A[c + 1][6] = A[c + 1][7] = A[c + 1][8] = A[c + 1][9] = A[c + 1][11] = 0.0f;
        A[c + 1][10] = 1.0f;

        A[c + 2][6] = points1[k][0];
        A[c + 2][7] = points1[k][1];
        A[c + 2][8] = points1[k][2];
        A[c + 2][0] = A[c + 2][1] = A[c + 2][2] = A[c + 2][3] = A[c + 2][4] = A[c + 2][5] = A[c + 2][9] = A[c + 2][10] = 0.0f;
        A[c + 2][11] = 1.0f;
    }

    float* w = reg_matrix1DAllocate<float>(12);
    float** v = reg_matrix2DAllocate<float>(12, 12);
    // Now we can compute our svd
    svd(A, num_equations, 12, w, v);

    // First we make sure that the really small singular values
    // are set to 0. and compute the inverse by taking the reciprocal
    // of the entries
    for (unsigned k = 0; k < 12; ++k) {
        if (w[k] < 0.0001) {
            w[k] = 0.0f;
        }
        else {
            w[k] = static_cast<float>(1.0 / static_cast<double>(w[k]));
        }
    }

    // Now we can compute the pseudoinverse which is given by
    // V*inv(W)*U'
    // First compute the V * inv(w) in place.
    // Simply scale each column by the corresponding singular value
    for (unsigned k = 0; k < 12; ++k) {
        for (unsigned j = 0; j < 12; ++j) {
            v[j][k] = static_cast<float>(static_cast<double>(v[j][k]) * static_cast<double>(w[k]));
        }
    }

    // Now multiply the matrices together
    // Pseudoinverse = v * w * A(transpose)
    float** r = reg_matrix2DAllocate<float>(12, num_equations);
    reg_matrix2DMultiply<float>(v, 12, 12, A, num_equations, 12, r, true);
    // Now r contains the pseudoinverse
    // Create vector b and then multiple rb to get the affine paramsA
    float* b = reg_matrix1DAllocate<float>(num_equations);
    for (unsigned k = 0; k < num_points; ++k) {
        c = k * 3;
        b[c] = points2[k][0];
        b[c + 1] = points2[k][1];
        b[c + 2] = points2[k][2];
    }

    float * transform = reg_matrix1DAllocate<float>(12);
    //mul_matvec(r, 12, num_equations, b, transform);
    reg_matrix2DVectorMultiply<float>(r, 12, num_equations, b, transform);

    transformation->m[0][0] = transform[0];
    transformation->m[0][1] = transform[1];
    transformation->m[0][2] = transform[2];
    transformation->m[0][3] = transform[9];

    transformation->m[1][0] = transform[3];
    transformation->m[1][1] = transform[4];
    transformation->m[1][2] = transform[5];
    transformation->m[1][3] = transform[10];

    transformation->m[2][0] = transform[6];
    transformation->m[2][1] = transform[7];
    transformation->m[2][2] = transform[8];
    transformation->m[2][3] = transform[11];

    transformation->m[3][0] = 0.0f;
    transformation->m[3][1] = 0.0f;
    transformation->m[3][2] = 0.0f;
    transformation->m[3][3] = 1.0f;

    // Do the deletion here
    reg_matrix1DDeallocate(transform);
    reg_matrix1DDeallocate(b);
    reg_matrix2DDeallocate(12, r);
    reg_matrix2DDeallocate(12, v);
    reg_matrix1DDeallocate(w);
    reg_matrix2DDeallocate(num_equations, A);
}
/* *************************************************************** */
// estimate an affine transformation using least square
void estimate_affine_transformation3D(std::vector<_reg_sorted_point3D> &points, mat44 * transformation)
{
    unsigned int num_points = points.size();
    float** points1 = reg_matrix2DAllocate<float>(num_points, 3);
    float** points2 = reg_matrix2DAllocate<float>(num_points, 3);
    for (unsigned int i = 0; i < num_points; i++) {
        points1[i][0] = points[i].target[0];
        points1[i][1] = points[i].target[1];
        points1[i][2] = points[i].target[2];
        points2[i][0] = points[i].result[0];
        points2[i][1] = points[i].result[1];
        points2[i][2] = points[i].result[2];
    }
    estimate_affine_transformation3D(points1, points2, num_points, transformation);
    //FREE MEMORY
    reg_matrix2DDeallocate(num_points, points1);
    reg_matrix2DDeallocate(num_points, points2);
}
/* *************************************************************** */
///LTS 2D
void optimize_2D(float* referencePosition, float* warpedPosition,
    unsigned int definedActiveBlock, int percent_to_keep, int max_iter, double tol,
    mat44 * final, bool affine) {

    // Set the current transformation to identity
    reg_mat44_eye(final);

    const unsigned num_points = definedActiveBlock;
    unsigned long num_equations = num_points * 2;
    // Keep a sorted list of the distance measure
    std::multimap<double, _reg_sorted_point2D> queue;
    std::vector<_reg_sorted_point2D> top_points;

    double distance = 0.0;
    double lastDistance = std::numeric_limits<double>::max();
    unsigned long i;

    // The initial vector with all the input points
    for (unsigned j = 0; j < num_points * 2; j += 2)
    {
        top_points.push_back(_reg_sorted_point2D(&referencePosition[j], &warpedPosition[j], 0.0));
    }
    if (affine) {
        estimate_affine_transformation2D(top_points, final);
    }
    else {
        estimate_rigid_transformation2D(top_points, final);
    }

    const unsigned long num_to_keep = (unsigned long)(num_points * (percent_to_keep / 100.0f));
    num_equations = num_to_keep * 2;
    float * newWarpedPosition = new float[num_points * 2];

    mat44 lastTransformation;
    memset(&lastTransformation, 0, sizeof(mat44));

    for (unsigned count = 0; count < max_iter; ++count)
    {
        // Transform the points in the target
        for (unsigned j = 0; j < num_points * 2; j += 2)
        {
            reg_mat33_mul(final, &referencePosition[j], &newWarpedPosition[j]);
        }
        queue = std::multimap<double, _reg_sorted_point2D>();
        for (unsigned j = 0; j < num_points * 2; j += 2)
        {
            distance = get_square_distance2D(&newWarpedPosition[j], &warpedPosition[j]);
            queue.insert(std::pair<double, _reg_sorted_point2D>(distance,
                _reg_sorted_point2D(&referencePosition[j], &warpedPosition[j], distance)));
        }

        distance = 0.0;
        i = 0;
        top_points.clear();

        for (std::multimap<double, _reg_sorted_point2D>::iterator it = queue.begin();
            it != queue.end(); ++it, ++i)
        {
            if (i >= num_to_keep) break;
            top_points.push_back((*it).second);
            distance += (*it).first;
        }

        // If the change is not substantial, we return
        if ((distance > lastDistance) || (lastDistance - distance) < tol)
        {
            // restore the last transformation
            memcpy(final, &lastTransformation, sizeof(mat44));
            break;
        }
        lastDistance = distance;
        memcpy(&lastTransformation, final, sizeof(mat44));
        if (affine) {
            estimate_affine_transformation2D(top_points, final);
        }
        else {
            estimate_rigid_transformation2D(top_points, final);
        }
    }
    delete[] newWarpedPosition;

}
/* *************************************************************** */
///LTS 3D
void optimize_3D(float *referencePosition, float *warpedPosition,
	unsigned int definedActiveBlock, int percent_to_keep, int max_iter, double tol,
	mat44 *final, bool affine) {

    // Set the current transformation to identity
    reg_mat44_eye(final);

    const unsigned num_points = definedActiveBlock;
    unsigned long num_equations = num_points * 3;
    // Keep a sorted list of the distance measure
    std::multimap<double, _reg_sorted_point3D> queue;
    std::vector<_reg_sorted_point3D> top_points;
    double distance = 0.0;
    double lastDistance = std::numeric_limits<double>::max();
    unsigned long i;

    // The initial vector with all the input points
    for (unsigned j = 0; j < num_points*3; j+=3) {
       top_points.push_back(_reg_sorted_point3D(&referencePosition[j],
                                                &warpedPosition[j],
                                                0.0));
    }
    if (affine) {
        estimate_affine_transformation3D(top_points, final);
    } else {
        estimate_rigid_transformation3D(top_points, final);
    }
    unsigned long num_to_keep = (unsigned long)(num_points * (percent_to_keep/100.0f));
    num_equations = num_to_keep*3;
    float* newWarpedPosition = new float[num_points*3];

    mat44 lastTransformation;
    memset(&lastTransformation,0,sizeof(mat44));

    for (unsigned count = 0; count < max_iter; ++count)
    {
       // Transform the points in the target
       for (unsigned j = 0; j < num_points * 3; j+=3) {
          reg_mat44_mul(final, &referencePosition[j], &newWarpedPosition[j]);
       }
       queue = std::multimap<double, _reg_sorted_point3D>();
       for (unsigned j = 0; j < num_points * 3; j+= 3)
       {
          distance = get_square_distance3D(&newWarpedPosition[j], &warpedPosition[j]);
          queue.insert(std::pair<double,
                       _reg_sorted_point3D>(distance,
                                            _reg_sorted_point3D(&referencePosition[j],
                                                                &warpedPosition[j],
                                                                distance)));
       }

       distance = 0.0;
       i = 0;
       top_points.clear();
       for (std::multimap<double, _reg_sorted_point3D>::iterator it = queue.begin();it != queue.end(); ++it, ++i)
       {
          if (i >= num_to_keep) break;
          top_points.push_back((*it).second);
          distance += (*it).first;
       }

       // If the change is not substantial, we return
       if ((distance > lastDistance) || (lastDistance - distance) < tol)
       {
          memcpy(final, &lastTransformation, sizeof(mat44));
          break;
       }
       lastDistance = distance;
       memcpy(&lastTransformation, final, sizeof(mat44));
       if(affine) {
           estimate_affine_transformation3D(top_points, final);
       } else {
           estimate_rigid_transformation3D(top_points, final);
       }
    }
    delete [] newWarpedPosition;
}
/* *************************************************************** */
/* *************************************************************** */
/* *************************************************************** */
/* *************************************************************** */
// Heap sort
void reg_heapSort(float *array_tmp, int *index_tmp, int blockNum)
{
    float *array = &array_tmp[-1];
    int *index = &index_tmp[-1];
    int l = (blockNum >> 1) + 1;
    int ir = blockNum;
    float val;
    int iVal;
    for (;;)
    {
        if (l > 1)
        {
            val = array[--l];
            iVal = index[l];
        }
        else
        {
            val = array[ir];
            iVal = index[ir];
            array[ir] = array[1];
            index[ir] = index[1];
            if (--ir == 1)
            {
                array[1] = val;
                index[1] = iVal;
                break;
            }
        }
        int i = l;
        int j = l + l;
        while (j <= ir)
        {
            if (j < ir && array[j] < array[j + 1])
                j++;
            if (val < array[j])
            {
                array[i] = array[j];
                index[i] = index[j];
                i = j;
                j <<= 1;
            }
            else
                break;
        }
        array[i] = val;
        index[i] = iVal;
    }
}
/* *************************************************************** */
// Heap sort
template<class DTYPE>
void reg_heapSort(DTYPE *array_tmp, int blockNum)
{
    DTYPE *array = &array_tmp[-1];
    int l = (blockNum >> 1) + 1;
    int ir = blockNum;
    DTYPE val;
    for (;;)
    {
        if (l > 1)
        {
            val = array[--l];
        }
        else
        {
            val = array[ir];
            array[ir] = array[1];
            if (--ir == 1)
            {
                array[1] = val;
                break;
            }
        }
        int i = l;
        int j = l + l;
        while (j <= ir)
        {
            if (j < ir && array[j] < array[j + 1])
                j++;
            if (val < array[j])
            {
                array[i] = array[j];
                i = j;
                j <<= 1;
            }
            else
                break;
        }
        array[i] = val;
    }
}
template void reg_heapSort<float>(float *array_tmp, int blockNum);
template void reg_heapSort<double>(double *array_tmp, int blockNum);
/* *************************************************************** */
/* *************************************************************** */
bool operator==(mat44 A, mat44 B)
{
    for (unsigned i = 0; i < 4; ++i)
    {
        for (unsigned j = 0; j < 4; ++j)
        {
            if (A.m[i][j] != B.m[i][j])
                return false;
        }
    }
    return true;
}
/* *************************************************************** */
bool operator!=(mat44 A, mat44 B)
{
    for (unsigned i = 0; i < 4; ++i)
    {
        for (unsigned j = 0; j < 4; ++j)
        {
            if (A.m[i][j] != B.m[i][j])
                return true;
        }
    }
    return false;
}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
T reg_mat44_det(mat44 const* A)
{
    double D =
        static_cast<double>(A->m[0][0]) * static_cast<double>(A->m[1][1]) * static_cast<double>(A->m[2][2]) * static_cast<double>(A->m[3][3])
        - static_cast<double>(A->m[0][0]) * static_cast<double>(A->m[1][1]) * static_cast<double>(A->m[3][2]) * static_cast<double>(A->m[2][3])
        - static_cast<double>(A->m[0][0]) * static_cast<double>(A->m[2][1]) * static_cast<double>(A->m[1][2]) * static_cast<double>(A->m[3][3])
        + static_cast<double>(A->m[0][0]) * static_cast<double>(A->m[2][1]) * static_cast<double>(A->m[3][2]) * static_cast<double>(A->m[1][3])
        + static_cast<double>(A->m[0][0]) * static_cast<double>(A->m[3][1]) * static_cast<double>(A->m[1][2]) * static_cast<double>(A->m[2][3])
        - static_cast<double>(A->m[0][0]) * static_cast<double>(A->m[3][1]) * static_cast<double>(A->m[2][2]) * static_cast<double>(A->m[1][3])
        - static_cast<double>(A->m[1][0]) * static_cast<double>(A->m[0][1]) * static_cast<double>(A->m[2][2]) * static_cast<double>(A->m[3][3])
        + static_cast<double>(A->m[1][0]) * static_cast<double>(A->m[0][1]) * static_cast<double>(A->m[3][2]) * static_cast<double>(A->m[2][3])
        + static_cast<double>(A->m[1][0]) * static_cast<double>(A->m[2][1]) * static_cast<double>(A->m[0][2]) * static_cast<double>(A->m[3][3])
        - static_cast<double>(A->m[1][0]) * static_cast<double>(A->m[2][1]) * static_cast<double>(A->m[3][2]) * static_cast<double>(A->m[0][3])
        - static_cast<double>(A->m[1][0]) * static_cast<double>(A->m[3][1]) * static_cast<double>(A->m[0][2]) * static_cast<double>(A->m[2][3])
        + static_cast<double>(A->m[1][0]) * static_cast<double>(A->m[3][1]) * static_cast<double>(A->m[2][2]) * static_cast<double>(A->m[0][3])
        + static_cast<double>(A->m[2][0]) * static_cast<double>(A->m[0][1]) * static_cast<double>(A->m[1][2]) * static_cast<double>(A->m[3][3])
        - static_cast<double>(A->m[2][0]) * static_cast<double>(A->m[0][1]) * static_cast<double>(A->m[3][2]) * static_cast<double>(A->m[1][3])
        - static_cast<double>(A->m[2][0]) * static_cast<double>(A->m[1][1]) * static_cast<double>(A->m[0][2]) * static_cast<double>(A->m[3][3])
        + static_cast<double>(A->m[2][0]) * static_cast<double>(A->m[1][1]) * static_cast<double>(A->m[3][2]) * static_cast<double>(A->m[0][3])
        + static_cast<double>(A->m[2][0]) * static_cast<double>(A->m[3][1]) * static_cast<double>(A->m[0][2]) * static_cast<double>(A->m[1][3])
        - static_cast<double>(A->m[2][0]) * static_cast<double>(A->m[3][1]) * static_cast<double>(A->m[1][2]) * static_cast<double>(A->m[0][3])
        - static_cast<double>(A->m[3][0]) * static_cast<double>(A->m[0][1]) * static_cast<double>(A->m[1][2]) * static_cast<double>(A->m[2][3])
        + static_cast<double>(A->m[3][0]) * static_cast<double>(A->m[0][1]) * static_cast<double>(A->m[2][2]) * static_cast<double>(A->m[1][3])
        + static_cast<double>(A->m[3][0]) * static_cast<double>(A->m[1][1]) * static_cast<double>(A->m[0][2]) * static_cast<double>(A->m[2][3])
        - static_cast<double>(A->m[3][0]) * static_cast<double>(A->m[1][1]) * static_cast<double>(A->m[2][2]) * static_cast<double>(A->m[0][3])
        - static_cast<double>(A->m[3][0]) * static_cast<double>(A->m[2][1]) * static_cast<double>(A->m[0][2]) * static_cast<double>(A->m[1][3])
        + static_cast<double>(A->m[3][0]) * static_cast<double>(A->m[2][1]) * static_cast<double>(A->m[1][2]) * static_cast<double>(A->m[0][3]);
    return static_cast<T>(D);
}
template float reg_mat44_det<float>(mat44 const* A);
template double reg_mat44_det<double>(mat44 const* A);
/* *************************************************************** */
/* *************************************************************** */
template<class T>
T reg_mat33_det(mat33 const* A)
{
    double D = static_cast<T>((static_cast<double>(A->m[0][0]) * (static_cast<double>(A->m[1][1]) * static_cast<double>(A->m[2][2]) - static_cast<double>(A->m[1][2]) * static_cast<double>(A->m[2][1]))) -
        (static_cast<double>(A->m[0][1]) * (static_cast<double>(A->m[1][0]) * static_cast<double>(A->m[2][2]) - static_cast<double>(A->m[1][2]) * static_cast<double>(A->m[2][0]))) +
        (static_cast<double>(A->m[0][2]) * (static_cast<double>(A->m[1][0]) * static_cast<double>(A->m[2][1]) - static_cast<double>(A->m[1][1]) * static_cast<double>(A->m[2][0]))));
    return static_cast<T>(D);
}
template float reg_mat33_det<float>(mat33 const* A);
template double reg_mat33_det<double>(mat33 const* A);
/* *************************************************************** */
/* *************************************************************** */
mat33 reg_mat44_to_mat33(mat44 const* A)
{
    mat33 out;
    out.m[0][0] = A->m[0][0];
    out.m[0][1] = A->m[0][1];
    out.m[0][2] = A->m[0][2];
    out.m[1][0] = A->m[1][0];
    out.m[1][1] = A->m[1][1];
    out.m[1][2] = A->m[1][2];
    out.m[2][0] = A->m[2][0];
    out.m[2][1] = A->m[2][1];
    out.m[2][2] = A->m[2][2];
    return out;
}
/* *************************************************************** */
/* *************************************************************** */
mat44 reg_mat44_mul(mat44 const* A, mat44 const* B)
{
    mat44 R;
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            R.m[i][j] = A->m[i][0] * B->m[0][j] +
                A->m[i][1] * B->m[1][j] +
                A->m[i][2] * B->m[2][j] +
                A->m[i][3] * B->m[3][j];
        }
    }
    return R;
}
/* *************************************************************** */
mat44 operator*(mat44 A, mat44 B)
{
    return reg_mat44_mul(&A, &B);
}
/* *************************************************************** */
void reg_mat33_mul(mat44 const* mat,
    float const* in,
    float *out)
{
    out[0] = static_cast<float>(
        static_cast<double>(in[0])*static_cast<double>(mat->m[0][0]) +
        static_cast<double>(in[1])*static_cast<double>(mat->m[0][1]) +
        static_cast<double>(mat->m[0][3]));
    out[1] = static_cast<float>(
        static_cast<double>(in[0])*static_cast<double>(mat->m[1][0]) +
        static_cast<double>(in[1])*static_cast<double>(mat->m[1][1]) +
        static_cast<double>(mat->m[1][3]));
    return;
}
/* *************************************************************** */
void reg_mat33_mul(mat33 const* mat,
    float const* in,
    float *out)
{
    out[0] = static_cast<float>(
        static_cast<double>(in[0])*static_cast<double>(mat->m[0][0]) +
        static_cast<double>(in[1])*static_cast<double>(mat->m[0][1]) +
        static_cast<double>(mat->m[0][2]));
    out[1] = static_cast<float>(
        static_cast<double>(in[0])*static_cast<double>(mat->m[1][0]) +
        static_cast<double>(in[1])*static_cast<double>(mat->m[1][1]) +
        static_cast<double>(mat->m[1][2]));
    return;
}
/* *************************************************************** */
mat33 reg_mat33_mul(mat33 const* A, mat33 const* B)
{
    mat33 R;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            R.m[i][j] = A->m[i][0] * B->m[0][j] +
                A->m[i][1] * B->m[1][j] +
                A->m[i][2] * B->m[2][j];
        }
    }
    return R;
}
/* *************************************************************** */
mat33 operator*(mat33 A, mat33 B)
{
    return reg_mat33_mul(&A, &B);
}
/* *************************************************************** */
/* *************************************************************** */
mat33 reg_mat33_add(mat33 const* A, mat33 const* B)
{
    mat33 R;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            R.m[i][j] = A->m[i][j] + B->m[i][j];
        }
    }
    return R;
}
/* *************************************************************** */
/* *************************************************************** */
mat33 reg_mat33_trans(mat33 A)
{
    mat33 R;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            R.m[j][i] = A.m[i][j];
        }
    }
    return R;
}
/* *************************************************************** */
/* *************************************************************** */
mat33 operator+(mat33 A, mat33 B)
{
    return reg_mat33_add(&A, &B);
}
/* *************************************************************** */
/* *************************************************************** */
mat44 reg_mat44_add(mat44 const* A, mat44 const* B)
{
    mat44 R;
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            R.m[i][j] = A->m[i][j] + B->m[i][j];
        }
    }
    return R;
}
/* *************************************************************** */
/* *************************************************************** */
mat44 operator+(mat44 A, mat44 B)
{
    return reg_mat44_add(&A, &B);
}
/* *************************************************************** */
/* *************************************************************** */
mat33 reg_mat33_minus(mat33 const* A, mat33 const* B)
{
    mat33 R;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            R.m[i][j] = A->m[i][j] - B->m[i][j];
        }
    }
    return R;
}
/* *************************************************************** */
/* *************************************************************** */
void reg_mat33_diagonalize(mat33 const* A, mat33 * Q, mat33 * D)
{
    // A must be a symmetric matrix.
    // returns Q and D such that
    // Diagonal matrix D = QT * A * Q;  and  A = Q*D*QT
    const int maxsteps = 24;  // certainly wont need that many.
    int k0, k1, k2;
    float o[3], m[3];
    float q[4] = { 0.0, 0.0, 0.0, 1.0 };
    float jr[4];
    float sqw, sqx, sqy, sqz;
    float tmp1, tmp2, mq;
    mat33 AQ;
    float thet, sgn, t, c;
    for (int i = 0; i < maxsteps; ++i)
    {
        // quat to matrix
        sqx = q[0] * q[0];
        sqy = q[1] * q[1];
        sqz = q[2] * q[2];
        sqw = q[3] * q[3];
        Q->m[0][0] = (sqx - sqy - sqz + sqw);
        Q->m[1][1] = (-sqx + sqy - sqz + sqw);
        Q->m[2][2] = (-sqx - sqy + sqz + sqw);
        tmp1 = q[0] * q[1];
        tmp2 = q[2] * q[3];
        Q->m[1][0] = 2.0 * (tmp1 + tmp2);
        Q->m[0][1] = 2.0 * (tmp1 - tmp2);
        tmp1 = q[0] * q[2];
        tmp2 = q[1] * q[3];
        Q->m[2][0] = 2.0 * (tmp1 - tmp2);
        Q->m[0][2] = 2.0 * (tmp1 + tmp2);
        tmp1 = q[1] * q[2];
        tmp2 = q[0] * q[3];
        Q->m[2][1] = 2.0 * (tmp1 + tmp2);
        Q->m[1][2] = 2.0 * (tmp1 - tmp2);

        // AQ = A * Q
        AQ.m[0][0] = Q->m[0][0] * A->m[0][0] + Q->m[1][0] * A->m[0][1] + Q->m[2][0] * A->m[0][2];
        AQ.m[0][1] = Q->m[0][1] * A->m[0][0] + Q->m[1][1] * A->m[0][1] + Q->m[2][1] * A->m[0][2];
        AQ.m[0][2] = Q->m[0][2] * A->m[0][0] + Q->m[1][2] * A->m[0][1] + Q->m[2][2] * A->m[0][2];
        AQ.m[1][0] = Q->m[0][0] * A->m[0][1] + Q->m[1][0] * A->m[1][1] + Q->m[2][0] * A->m[1][2];
        AQ.m[1][1] = Q->m[0][1] * A->m[0][1] + Q->m[1][1] * A->m[1][1] + Q->m[2][1] * A->m[1][2];
        AQ.m[1][2] = Q->m[0][2] * A->m[0][1] + Q->m[1][2] * A->m[1][1] + Q->m[2][2] * A->m[1][2];
        AQ.m[2][0] = Q->m[0][0] * A->m[0][2] + Q->m[1][0] * A->m[1][2] + Q->m[2][0] * A->m[2][2];
        AQ.m[2][1] = Q->m[0][1] * A->m[0][2] + Q->m[1][1] * A->m[1][2] + Q->m[2][1] * A->m[2][2];
        AQ.m[2][2] = Q->m[0][2] * A->m[0][2] + Q->m[1][2] * A->m[1][2] + Q->m[2][2] * A->m[2][2];
        // D = Qt * AQ
        D->m[0][0] = AQ.m[0][0] * Q->m[0][0] + AQ.m[1][0] * Q->m[1][0] + AQ.m[2][0] * Q->m[2][0];
        D->m[0][1] = AQ.m[0][0] * Q->m[0][1] + AQ.m[1][0] * Q->m[1][1] + AQ.m[2][0] * Q->m[2][1];
        D->m[0][2] = AQ.m[0][0] * Q->m[0][2] + AQ.m[1][0] * Q->m[1][2] + AQ.m[2][0] * Q->m[2][2];
        D->m[1][0] = AQ.m[0][1] * Q->m[0][0] + AQ.m[1][1] * Q->m[1][0] + AQ.m[2][1] * Q->m[2][0];
        D->m[1][1] = AQ.m[0][1] * Q->m[0][1] + AQ.m[1][1] * Q->m[1][1] + AQ.m[2][1] * Q->m[2][1];
        D->m[1][2] = AQ.m[0][1] * Q->m[0][2] + AQ.m[1][1] * Q->m[1][2] + AQ.m[2][1] * Q->m[2][2];
        D->m[2][0] = AQ.m[0][2] * Q->m[0][0] + AQ.m[1][2] * Q->m[1][0] + AQ.m[2][2] * Q->m[2][0];
        D->m[2][1] = AQ.m[0][2] * Q->m[0][1] + AQ.m[1][2] * Q->m[1][1] + AQ.m[2][2] * Q->m[2][1];
        D->m[2][2] = AQ.m[0][2] * Q->m[0][2] + AQ.m[1][2] * Q->m[1][2] + AQ.m[2][2] * Q->m[2][2];
        o[0] = D->m[1][2];
        o[1] = D->m[0][2];
        o[2] = D->m[0][1];
        m[0] = fabs(o[0]);
        m[1] = fabs(o[1]);
        m[2] = fabs(o[2]);

        k0 = (m[0] > m[1] && m[0] > m[2]) ? 0 : (m[1] > m[2]) ? 1 : 2; // index of largest element of offdiag
        k1 = (k0 + 1) % 3;
        k2 = (k0 + 2) % 3;
        if (o[k0] == 0.0)
        {
            break;                          // diagonal already
        }
        thet = (D->m[k2][k2] - D->m[k1][k1]) / (2.0*o[k0]);
        sgn = (thet > 0.0) ? 1.0 : -1.0;
        thet *= sgn;                      // make it positive
        t = sgn / (thet + ((thet < 1.E6) ? sqrt(thet*thet + 1.0) : thet)); // sign(T)/(|T|+sqrt(T^2+1))
        c = 1.0 / sqrt(t*t + 1.0);        //  c= 1/(t^2+1) , t=s/c
        if (c == 1.0)
        {
            break;                          // no room for improvement - reached machine precision.
        }
        jr[0] = jr[1] = jr[2] = jr[3] = 0.0;
        jr[k0] = sgn*sqrt((1.0 - c) / 2.0);    // using 1/2 angle identity sin(a/2) = sqrt((1-cos(a))/2)
        jr[k0] *= -1.0;                     // since our quat-to-matrix convention was for v*M instead of M*v
        jr[3] = sqrt(1.0f - jr[k0] * jr[k0]);
        if (jr[3] == 1.0)
        {
            break;                          // reached limits of floating point precision
        }
        q[0] = (q[3] * jr[0] + q[0] * jr[3] + q[1] * jr[2] - q[2] * jr[1]);
        q[1] = (q[3] * jr[1] - q[0] * jr[2] + q[1] * jr[3] + q[2] * jr[0]);
        q[2] = (q[3] * jr[2] + q[0] * jr[1] - q[1] * jr[0] + q[2] * jr[3]);
        q[3] = (q[3] * jr[3] - q[0] * jr[0] - q[1] * jr[1] - q[2] * jr[2]);
        mq = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        q[0] /= mq;
        q[1] /= mq;
        q[2] /= mq;
        q[3] /= mq;
    }
}

/* *************************************************************** */
/* *************************************************************** */
mat33 operator-(mat33 A, mat33 B)
{
    return reg_mat33_minus(&A, &B);
}
/* *************************************************************** */
/* *************************************************************** */
void reg_mat33_eye(mat33 *mat)
{
    mat->m[0][0] = 1.f;
    mat->m[0][1] = mat->m[0][2] = 0.f;
    mat->m[1][1] = 1.f;
    mat->m[1][0] = mat->m[1][2] = 0.f;
    mat->m[2][2] = 1.f;
    mat->m[2][0] = mat->m[2][1] = 0.f;
}
/* *************************************************************** */
/* *************************************************************** */
mat44 reg_mat44_minus(mat44 const* A, mat44 const* B)
{
    mat44 R;
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            R.m[i][j] = A->m[i][j] - B->m[i][j];
        }
    }
    return R;
}

/* *************************************************************** */
/* *************************************************************** */
mat44 operator-(mat44 A, mat44 B)
{
    return reg_mat44_minus(&A, &B);
}

/* *************************************************************** */
/* *************************************************************** */
void reg_mat44_eye(mat44 *mat)
{
    mat->m[0][0] = 1.f;
    mat->m[0][1] = mat->m[0][2] = mat->m[0][3] = 0.f;
    mat->m[1][1] = 1.f;
    mat->m[1][0] = mat->m[1][2] = mat->m[1][3] = 0.f;
    mat->m[2][2] = 1.f;
    mat->m[2][0] = mat->m[2][1] = mat->m[2][3] = 0.f;
    mat->m[3][3] = 1.f;
    mat->m[3][0] = mat->m[3][1] = mat->m[3][2] = 0.f;
}
/* *************************************************************** */
/* *************************************************************** */
float reg_mat44_norm_inf(mat44 const* mat)
{
    float maxval = 0.0;
    float newval = 0.0;
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            newval = fabsf(mat->m[i][j]);
            maxval = (newval > maxval) ? newval : maxval;
        }
    }
    return maxval;
}
/* *************************************************************** */
/* *************************************************************** */
void reg_mat44_mul(mat44 const* mat,
    float const* in,
    float *out)
{
    out[0] = static_cast<float>(static_cast<double>(mat->m[0][0]) * static_cast<double>(in[0]) +
        static_cast<double>(mat->m[0][1]) * static_cast<double>(in[1]) +
        static_cast<double>(mat->m[0][2]) * static_cast<double>(in[2]) +
        static_cast<double>(mat->m[0][3]));
    out[1] = static_cast<float>(static_cast<double>(mat->m[1][0]) * static_cast<double>(in[0]) +
        static_cast<double>(mat->m[1][1]) * static_cast<double>(in[1]) +
        static_cast<double>(mat->m[1][2]) * static_cast<double>(in[2]) +
        static_cast<double>(mat->m[1][3]));
    out[2] = static_cast<float>(static_cast<double>(mat->m[2][0]) * static_cast<double>(in[0]) +
        static_cast<double>(mat->m[2][1]) * static_cast<double>(in[1]) +
        static_cast<double>(mat->m[2][2]) * static_cast<double>(in[2]) +
        static_cast<double>(mat->m[2][3]));
    return;
}
/* *************************************************************** */
/* *************************************************************** */
void reg_mat44_mul(mat44 const* mat,
    double const* in,
    double *out)
{
    double matD[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            matD[i][j] = static_cast<double>(mat->m[i][j]);

    out[0] = matD[0][0] * in[0] +
        matD[0][1] * in[1] +
        matD[0][2] * in[2] +
        matD[0][3];
    out[1] = matD[1][0] * in[0] +
        matD[1][1] * in[1] +
        matD[1][2] * in[2] +
        matD[1][3];
    out[2] = matD[2][0] * in[0] +
        matD[2][1] * in[1] +
        matD[2][2] * in[2] +
        matD[2][3];
    return;
}
/* *************************************************************** */
/* *************************************************************** */
mat44 reg_mat44_mul(mat44 const* A, double scalar)
{
    mat44 out;
    out.m[0][0] = A->m[0][0] * scalar;
    out.m[0][1] = A->m[0][1] * scalar;
    out.m[0][2] = A->m[0][2] * scalar;
    out.m[0][3] = A->m[0][3] * scalar;
    out.m[1][0] = A->m[1][0] * scalar;
    out.m[1][1] = A->m[1][1] * scalar;
    out.m[1][2] = A->m[1][2] * scalar;
    out.m[1][3] = A->m[1][3] * scalar;
    out.m[2][0] = A->m[2][0] * scalar;
    out.m[2][1] = A->m[2][1] * scalar;
    out.m[2][2] = A->m[2][2] * scalar;
    out.m[2][3] = A->m[2][3] * scalar;
    out.m[3][0] = A->m[3][0] * scalar;
    out.m[3][1] = A->m[3][1] * scalar;
    out.m[3][2] = A->m[3][2] * scalar;
    out.m[3][3] = A->m[3][3] * scalar;
    return out;
}
/* *************************************************************** */
/* *************************************************************** */
mat44 reg_mat44_sqrt(mat44 const* mat)
{
    mat44 X;
    Eigen::Matrix4d m;
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 4; ++j)
        {
            m(i, j) = static_cast<double>(mat->m[i][j]);
        }
    }
    m = m.sqrt();
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            X.m[i][j] = static_cast<float>(m(i, j));
    return X;
}
/* *************************************************************** */
/* *************************************************************** */
mat44 reg_mat44_expm(mat44 const* mat)
{
    mat44 X;
    Eigen::Matrix4d m;
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            m(i, j) = static_cast<double>(mat->m[i][j]);
        }
    }
    m = m.exp();
    //
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            X.m[i][j] = static_cast<float>(m(i, j));

    return X;
}
/* *************************************************************** */
/* *************************************************************** */
mat44 reg_mat44_logm(mat44 const* mat)
{
    mat44 X;
    Eigen::Matrix4d m;
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            m(i, j) = static_cast<double>(mat->m[i][j]);
        }
    }
    m = m.log();
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            X.m[i][j] = static_cast<float>(m(i, j));
    return X;
}
/* *************************************************************** */
/* *************************************************************** */
mat44 reg_mat44_avg2(mat44 const* A, mat44 const* B)
{
    mat44 out;
    mat44 logA = reg_mat44_logm(A);
    mat44 logB = reg_mat44_logm(B);
    for (int i = 0; i < 4; ++i) {
        logA.m[3][i] = 0.f;
        logB.m[3][i] = 0.f;
    }
    logA = reg_mat44_add(&logA, &logB);
    out = reg_mat44_mul(&logA, 0.5);
    return reg_mat44_expm(&out);

}
/* *************************************************************** */
/* *************************************************************** */
mat44 reg_mat44_inv(mat44 const* mat)
{
    mat44 out;
    Eigen::Matrix4d m, m_inv;
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            m(i, j) = static_cast<double>(mat->m[i][j]);
        }
    }
    m_inv = m.inverse();
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            out.m[i][j] = static_cast<float>(m_inv(i, j));
    //
    return out;

}
/* *************************************************************** */
/* *************************************************************** */
void reg_mat44_disp(mat44 *mat, char * title){
    printf("%s:\n%.7g\t%.7g\t%.7g\t%.7g\n%.7g\t%.7g\t%.7g\t%.7g\n%.7g\t%.7g\t%.7g\t%.7g\n%.7g\t%.7g\t%.7g\t%.7g\n", title,
        mat->m[0][0], mat->m[0][1], mat->m[0][2], mat->m[0][3],
        mat->m[1][0], mat->m[1][1], mat->m[1][2], mat->m[1][3],
        mat->m[2][0], mat->m[2][1], mat->m[2][2], mat->m[2][3],
        mat->m[3][0], mat->m[3][1], mat->m[3][2], mat->m[3][3]);
}

/* *************************************************************** */
/* *************************************************************** */
void reg_mat33_disp(mat33 *mat, char * title){
    printf("%s:\n%g\t%g\t%g\n%g\t%g\t%g\n%g\t%g\t%g\n", title,
        mat->m[0][0], mat->m[0][1], mat->m[0][2],
        mat->m[1][0], mat->m[1][1], mat->m[1][2],
        mat->m[2][0], mat->m[2][1], mat->m[2][2]);
}
/* *************************************************************** */
//is it square distance or just distance?
// Helper function: Get the square of the Euclidean distance
double get_square_distance3D(float * first_point3D, float * second_point3D) {
    return sqrt((first_point3D[0] - second_point3D[0]) * (first_point3D[0] - second_point3D[0]) + (first_point3D[1] - second_point3D[1]) * (first_point3D[1] - second_point3D[1]) + (first_point3D[2] - second_point3D[2]) * (first_point3D[2] - second_point3D[2]));
}
/* *************************************************************** */
//is it square distance or just distance?
double get_square_distance2D(float * first_point2D, float * second_point2D) {
    return sqrt((first_point2D[0] - second_point2D[0]) * (first_point2D[0] - second_point2D[0]) + (first_point2D[1] - second_point2D[1]) * (first_point2D[1] - second_point2D[1]));
}
/* *************************************************************** */
// Calculate pythagorean distance
template<class T>
T pythag(T a, T b)
{
    T absa, absb;
    absa = fabs(a);
    absb = fabs(b);

    if (absa > absb)
        return (T)(absa * sqrt(1.0f + SQR(absb / absa)));
    else
        return (absb == 0.0f ? 0.0f : (T)(absb * sqrt(1.0f + SQR(absa / absb))));
}
#endif // _REG_MATHS_CPP
