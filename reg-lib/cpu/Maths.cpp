#define USE_EIGEN

#include "_reg_tools.h"
#include "Eigen/Core"
#include "unsupported/Eigen/MatrixFunctions"

/* *************************************************************** */
namespace NiftyReg {
/* *************************************************************** */
template<class T>
T** Matrix2dMultiply(T **mat1, const size_t mat1X, const size_t mat1Y, T **mat2, const size_t mat2X, const size_t mat2Y, const bool transposeMat2) {
    if (transposeMat2 == false) {
        // First check that the dimension are appropriate
        if (mat1Y != mat2X)
            NR_FATAL_ERROR("Matrices can not be multiplied due to their size: [" + std::to_string(mat1X) + " " +
                           std::to_string(mat1Y) + "] [" + std::to_string(mat2X) + " " + std::to_string(mat2Y) + "]");

        T **res = Matrix2dAlloc<T>(mat1X, mat2Y);
        for (size_t i = 0; i < mat1X; i++) {
            for (size_t j = 0; j < mat2Y; j++) {
                double resTemp = 0;
                for (size_t k = 0; k < mat1Y; k++)
                    resTemp += static_cast<double>(mat1[i][k]) * static_cast<double>(mat2[k][j]);
                res[i][j] = static_cast<T>(resTemp);
            }
        }
        return res;
    } else {
        // First check that the dimension are appropriate
        if (mat1Y != mat2Y)
            NR_FATAL_ERROR("Matrices can not be multiplied due to their size: [" + std::to_string(mat1X) + " " +
                           std::to_string(mat1Y) + "] [" + std::to_string(mat2Y) + " " + std::to_string(mat2X) + "]");

        T **res = Matrix2dAlloc<T>(mat1X, mat2X);
        for (size_t i = 0; i < mat1X; i++) {
            for (size_t j = 0; j < mat2X; j++) {
                double resTemp = 0;
                for (size_t k = 0; k < mat1Y; k++)
                    resTemp += static_cast<double>(mat1[i][k]) * static_cast<double>(mat2[j][k]);
                res[i][j] = static_cast<T>(resTemp);
            }
        }
        return res;
    }
}
template float** Matrix2dMultiply<float>(float** mat1, const size_t mat1X, const size_t mat1Y, float** mat2, const size_t mat2X, const size_t mat2Y, const bool transposeMat2);
template double** Matrix2dMultiply<double>(double** mat1, const size_t mat1X, const size_t mat1Y, double** mat2, const size_t mat2X, const size_t mat2Y, const bool transposeMat2);
/* *************************************************************** */
template<class T>
void Matrix2dMultiply(T **mat1, const size_t mat1X, const size_t mat1Y, T **mat2, const size_t mat2X, const size_t mat2Y, T **resT, const bool transposeMat2) {
    if (transposeMat2 == false) {
        // First check that the dimension are appropriate
        if (mat1Y != mat2X)
            NR_FATAL_ERROR("Matrices can not be multiplied due to their size: [" + std::to_string(mat1X) + " " +
                           std::to_string(mat1Y) + "] [" + std::to_string(mat2X) + " " + std::to_string(mat2Y) + "]");

        for (size_t i = 0; i < mat1X; i++) {
            for (size_t j = 0; j < mat2Y; j++) {
                double resTemp = 0;
                for (size_t k = 0; k < mat1Y; k++)
                    resTemp += static_cast<double>(mat1[i][k]) * static_cast<double>(mat2[k][j]);
                resT[i][j] = static_cast<T>(resTemp);
            }
        }
    } else {
        // First check that the dimension are appropriate
        if (mat1Y != mat2Y)
            NR_FATAL_ERROR("Matrices can not be multiplied due to their size: [" + std::to_string(mat1X) + " " +
                           std::to_string(mat1Y) + "] [" + std::to_string(mat2Y) + " " + std::to_string(mat2X) + "]");

        for (size_t i = 0; i < mat1X; i++) {
            for (size_t j = 0; j < mat2X; j++) {
                double resTemp = 0;
                for (size_t k = 0; k < mat1Y; k++)
                    resTemp += static_cast<double>(mat1[i][k]) * static_cast<double>(mat2[j][k]);
                resT[i][j] = static_cast<T>(resTemp);
            }
        }
    }
}
template void Matrix2dMultiply<float>(float** mat1, const size_t mat1X, const size_t mat1Y, float** mat2, const size_t mat2X, const size_t mat2Y, float** resT, const bool transposeMat2);
template void Matrix2dMultiply<double>(double** mat1, const size_t mat1X, const size_t mat1Y, double** mat2, const size_t mat2X, const size_t mat2Y, double** resT, const bool transposeMat2);
/* *************************************************************** */
// Multiply a matrix with a vector - we assume correct dimension
template<class T>
T* Matrix2dVectorMultiply(T **mat, const size_t m, const size_t n, T* vect) {
    T* res = Matrix1dAlloc<T>(m);
    for (size_t i = 0; i < m; i++) {
        double resTemp = 0;
        for (size_t k = 0; k < n; k++) {
            resTemp += static_cast<double>(mat[i][k]) * static_cast<double>(vect[k]);
        }
        res[i] = static_cast<T>(resTemp);
    }
    return res;
}
template float* Matrix2dVectorMultiply<float>(float** mat, const size_t m, const size_t n, float* vect);
template double* Matrix2dVectorMultiply<double>(double** mat, const size_t m, const size_t n, double* vect);
/* *************************************************************** */
template<class T>
void Matrix2dVectorMultiply(T **mat, const size_t m, const size_t n, T* vect, T* res) {
    for (size_t i = 0; i < m; i++) {
        double resTemp = 0;
        for (size_t k = 0; k < n; k++) {
            resTemp += static_cast<double>(mat[i][k]) * static_cast<double>(vect[k]);
        }
        res[i] = static_cast<T>(resTemp);
    }
}
template void Matrix2dVectorMultiply<float>(float** mat, const size_t m, const size_t n, float* vect, float* res);
template void Matrix2dVectorMultiply<double>(double** mat, const size_t m, const size_t n, double* vect, double* res);
/* *************************************************************** */
// Heap sort
void HeapSort(float *array_tmp, int *index_tmp, int blockNum) {
    float *array = &array_tmp[-1];
    int *index = &index_tmp[-1];
    int l = (blockNum >> 1) + 1;
    int ir = blockNum;
    float val;
    int iVal;
    for (;;) {
        if (l > 1) {
            val = array[--l];
            iVal = index[l];
        } else {
            val = array[ir];
            iVal = index[ir];
            array[ir] = array[1];
            index[ir] = index[1];
            if (--ir == 1) {
                array[1] = val;
                index[1] = iVal;
                break;
            }
        }
        int i = l;
        int j = l + l;
        while (j <= ir) {
            if (j < ir && array[j] < array[j + 1])
                j++;
            if (val < array[j]) {
                array[i] = array[j];
                index[i] = index[j];
                i = j;
                j <<= 1;
            } else
                break;
        }
        array[i] = val;
        index[i] = iVal;
    }
}
/* *************************************************************** */
// Heap sort
template<class DataType>
void HeapSort(DataType *array_tmp, int blockNum) {
    DataType *array = &array_tmp[-1];
    int l = (blockNum >> 1) + 1;
    int ir = blockNum;
    DataType val;
    for (;;) {
        if (l > 1) {
            val = array[--l];
        } else {
            val = array[ir];
            array[ir] = array[1];
            if (--ir == 1) {
                array[1] = val;
                break;
            }
        }
        int i = l;
        int j = l + l;
        while (j <= ir) {
            if (j < ir && array[j] < array[j + 1])
                j++;
            if (val < array[j]) {
                array[i] = array[j];
                i = j;
                j <<= 1;
            } else
                break;
        }
        array[i] = val;
    }
}
template void HeapSort<float>(float *array_tmp, int blockNum);
template void HeapSort<double>(double *array_tmp, int blockNum);
/* *************************************************************** */
template<class T>
T Mat44Det(const mat44 *A) {
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
template float Mat44Det<float>(const mat44 *A);
template double Mat44Det<double>(const mat44 *A);
/* *************************************************************** */
void Mat33Diagonalize(const mat33 *A, mat33 *Q, mat33 *D) {
    // A must be a symmetric matrix.
    // returns Q and D such that
    // Diagonal matrix D = QT * A * Q;  and  A = Q*D*QT
    const int maxsteps = 24;  // certainly wont need that many.
    int k0, k1, k2;
    float o[3], m[3];
    float q[4] = { 0, 0, 0, 1 };
    float jr[4];
    float sqw, sqx, sqy, sqz;
    float tmp1, tmp2, mq;
    mat33 AQ;
    float thet, sgn, t, c;
    for (int i = 0; i < maxsteps; ++i) {
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
        if (o[k0] == 0) {
            break;                          // diagonal already
        }
        thet = (D->m[k2][k2] - D->m[k1][k1]) / (2.0 * o[k0]);
        sgn = (thet > 0) ? 1 : -1;
        thet *= sgn;                      // make it positive
        t = sgn / (thet + ((thet < 1.E6) ? sqrt(thet * thet + 1.0) : thet)); // sign(T)/(|T|+sqrt(T^2+1))
        c = 1.0 / sqrt(t * t + 1.0);        //  c= 1/(t^2+1) , t=s/c
        if (c == 1.0) {
            break;                          // no room for improvement - reached machine precision.
        }
        jr[0] = jr[1] = jr[2] = jr[3] = 0;
        jr[k0] = sgn * sqrt((1.0 - c) / 2.0);    // using 1/2 angle identity sin(a/2) = sqrt((1-cos(a))/2)
        jr[k0] *= -1.0;                     // since our quat-to-matrix convention was for v*M instead of M*v
        jr[3] = sqrt(1.0f - jr[k0] * jr[k0]);
        if (jr[3] == 1.0) {
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
void Mat44Disp(const mat44 mat, const std::string& title) {
    NR_COUT << title << ":\n"
        << mat.m[0][0] << "\t" << mat.m[0][1] << "\t" << mat.m[0][2] << "\t" << mat.m[0][3] << "\n"
        << mat.m[1][0] << "\t" << mat.m[1][1] << "\t" << mat.m[1][2] << "\t" << mat.m[1][3] << "\n"
        << mat.m[2][0] << "\t" << mat.m[2][1] << "\t" << mat.m[2][2] << "\t" << mat.m[2][3] << "\n"
        << mat.m[3][0] << "\t" << mat.m[3][1] << "\t" << mat.m[3][2] << "\t" << mat.m[3][3] << std::endl;
}
/* *************************************************************** */
/** @brief SVD
* @param in input matrix to decompose - in place
* @param size_m row
* @param size_n colomn
* @param w diagonal term
* @param v rotation part
*/
template<class T>
void Svd(T **in, const size_t size_m, const size_t size_n, T * w, T **v) {
    if (size_m == 0 || size_n == 0)
        NR_FATAL_ERROR("The specified matrix is empty");

#ifdef _WIN32
    long sm, sn, sn2;
    long size__m = (long)size_m, size__n = (long)size_n;
#else
    size_t sm, sn, sn2;
    size_t size__m = size_m, size__n = size_n;
#endif
    Eigen::MatrixXd m(size_m, size_n);

    //Convert to Eigen matrix
#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(in,m, size__m, size__n) \
   private(sn)
#endif
    for (sm = 0; sm < size__m; sm++)
        for (sn = 0; sn < size__n; sn++)
            m(sm, sn) = static_cast<double>(in[sm][sn]);

    Eigen::JacobiSVD<Eigen::MatrixXd> Svd(m, Eigen::ComputeThinU | Eigen::ComputeThinV);

#ifdef _OPENMP
#pragma omp parallel for default(none) \
   shared(in,Svd,v,w, size__n,size__m) \
   private(sn2, sm)
#endif
    for (sn = 0; sn < size__n; sn++) {
        w[sn] = static_cast<T>(Svd.singularValues()(sn));
        for (sn2 = 0; sn2 < size__n; sn2++)
            v[sn2][sn] = static_cast<T>(Svd.matrixV()(sn2, sn));
        for (sm = 0; sm < size__m; sm++)
            in[sm][sn] = static_cast<T>(Svd.matrixU()(sm, sn));
    }
}
template void Svd<float>(float **in, const size_t m, const size_t n, float * w, float **v);
template void Svd<double>(double **in, const size_t m, const size_t n, double * w, double **v);
/* *************************************************************** */
template<class T>
T Matrix2dDet(T **mat, const size_t m, const size_t n) {
    if (m != n)
        NR_FATAL_ERROR("The matrix have to be square: [" + std::to_string(m) + " " + std::to_string(n) + "]");

    double res;
    if (m == 2) {
        res = static_cast<double>(mat[0][0]) * static_cast<double>(mat[1][1]) - static_cast<double>(mat[1][0]) * static_cast<double>(mat[0][1]);
    } else if (m == 3) {
        res = (static_cast<double>(mat[0][0]) * (static_cast<double>(mat[1][1]) * static_cast<double>(mat[2][2]) - static_cast<double>(mat[1][2]) * static_cast<double>(mat[2][1]))) -
            (static_cast<double>(mat[0][1]) * (static_cast<double>(mat[1][0]) * static_cast<double>(mat[2][2]) - static_cast<double>(mat[1][2]) * static_cast<double>(mat[2][0]))) +
            (static_cast<double>(mat[0][2]) * (static_cast<double>(mat[1][0]) * static_cast<double>(mat[2][1]) - static_cast<double>(mat[1][1]) * static_cast<double>(mat[2][0])));
    } else {
        // Convert to Eigen format
        Eigen::MatrixXd eigenRes(m, n);
        for (size_t i = 0; i < m; i++)
            for (size_t j = 0; j < n; j++)
                eigenRes(i, j) = static_cast<double>(mat[i][j]);
        res = eigenRes.determinant();
    }
    return static_cast<T>(res);
}
template float Matrix2dDet<float>(float **mat, const size_t m, const size_t n);
template double Matrix2dDet<double>(double **mat, const size_t m, const size_t n);
/* *************************************************************** */
void Mat33Expm(mat33 *tensorIn) {
    int sm, sn;
    Eigen::Matrix3d tensor;

    // Convert to Eigen format
    for (sm = 0; sm < 3; sm++) {
        for (sn = 0; sn < 3; sn++) {
            float val = tensorIn->m[sm][sn];
            if (val != val) return;
            tensor(sm, sn) = static_cast<double>(val);
        }
    }

    // Compute exp(E)
    tensor = tensor.exp();

    // Convert the result to mat33 format
    for (sm = 0; sm < 3; sm++)
        for (sn = 0; sn < 3; sn++)
            tensorIn->m[sm][sn] = static_cast<float>(tensor(sm, sn));
}
/* *************************************************************** */
mat44 Mat44Expm(const mat44 *mat) {
    mat44 X;
    Eigen::Matrix4d m;
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            m(i, j) = static_cast<double>(mat->m[i][j]);

    m = m.exp();

    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            X.m[i][j] = static_cast<float>(m(i, j));

    return X;
}
/* *************************************************************** */
void Mat33Logm(mat33 *tensorIn) {
    int sm, sn;
    Eigen::Matrix3d tensor;

    // Convert to Eigen format
    bool all_zeros = true;
    double det = 0;
    for (sm = 0; sm < 3; sm++) {
        for (sn = 0; sn < 3; sn++) {
            float val = tensorIn->m[sm][sn];
            if (val != 0.f) all_zeros = false;
            if (val != val) return;
            tensor(sm, sn) = static_cast<double>(val);
        }
    }
    // Actually R case requires invertible and no negative real ev,
    // but the only observed case so far was non-invertible.
    // determinant is not a perfect check for invertibility and
    // identity with zero not great either, but the alternative
    // is a general eigensolver and the logarithm function should
    // suceed unless convergence just isn't happening.
    det = tensor.determinant();
    if (all_zeros || det == 0) {
        Mat33ToNan(tensorIn);
        return;
    }

    // Compute the actual matrix log
    tensor = tensor.log();

    // Convert the result to mat33 format
    for (sm = 0; sm < 3; sm++)
        for (sn = 0; sn < 3; sn++)
            tensorIn->m[sm][sn] = static_cast<float>(tensor(sm, sn));
}
/* *************************************************************** */
mat44 Mat44Logm(const mat44 *mat) {
    mat44 X;
    Eigen::Matrix4d m;
    for (char i = 0; i < 4; ++i)
        for (char j = 0; j < 4; ++j)
            m(i, j) = static_cast<double>(mat->m[i][j]);
    m = m.log();
    for (char i = 0; i < 4; ++i)
        for (char j = 0; j < 4; ++j)
            X.m[i][j] = static_cast<float>(m(i, j));
    return X;
}
/* *************************************************************** */
mat44 Mat44Avg2(const mat44 *A, const mat44 *B) {
    mat44 out;
    mat44 logA = Mat44Logm(A);
    mat44 logB = Mat44Logm(B);
    for (int i = 0; i < 4; ++i) {
        logA.m[3][i] = 0.f;
        logB.m[3][i] = 0.f;
    }
    logA = logA + logB;
    out = logA * 0.5;
    return Mat44Expm(&out);
}
/* *************************************************************** */
} // namespace NiftyReg
/* *************************************************************** */
