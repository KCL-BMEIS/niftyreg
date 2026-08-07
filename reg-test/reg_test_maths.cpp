// OpenCL is not supported for this test
#undef USE_OPENCL

#include "reg_test_common.h"
#include <catch2/catch_template_test_macros.hpp>

/*
    The Maths module's matrix and sorting utilities, checked against mathematical oracles.

    Everything here has a closed form or a defining property, so nothing needs a reference
    implementation:

      - the dense matrix products against hand-computed results on non-square shapes (a transposed
        or swapped index survives a square test but not a 2x3 * 3x4 one);
      - the heap sorts against the definition of sorting: ascending output, a permutation of the
        input, and - for the indexed variant - indices that still point at the values they came in
        with;
      - the determinant against triangularity, singularity and multiplicativity;
      - the symmetric diagonalisation against a matrix with known eigenvalues, orthonormality of the
        eigenvectors, and reconstruction;
      - the matrix exponential and logarithm against the identities exp(0) = I, exp(diag) = diag of
        exponentials, exp of a nilpotent translation generator = I + N (exact, since N^2 = 0),
        det(exp(A)) = e^trace(A), and the round trip;
      - the log-Euclidean average against its closed forms: the average of A with itself is A, of
        two translations the mid translation, of two coaxial rotations the mid rotation.

    EstimateAffineLeastSquares / EstimateRigidLeastSquares are exercised to death through the LTS
    estimator tests and are not repeated here. Mat44Disp only prints.
*/

namespace {

constexpr double kTol = 1e-5;    // float storage of double-computed results

mat44 MakeMat44(std::initializer_list<double> rowMajor) {
    mat44 m{};
    auto it = rowMajor.begin();
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            m.m[i][j] = static_cast<float>(*it++);
    return m;
}

void RequireMat44Near(const mat44& actual, const mat44& expected, double tol, const std::string& what) {
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            INFO(what << ": entry (" << i << "," << j << ")");
            REQUIRE(std::abs(double(actual.m[i][j]) - double(expected.m[i][j])) < tol);
        }
}

mat44 RotationZ(double theta) {
    mat44 r;
    Mat44Eye(&r);
    r.m[0][0] = static_cast<float>(std::cos(theta)); r.m[0][1] = static_cast<float>(-std::sin(theta));
    r.m[1][0] = static_cast<float>(std::sin(theta)); r.m[1][1] = static_cast<float>(std::cos(theta));
    return r;
}

} // namespace

TEMPLATE_TEST_CASE("Maths: dense matrix multiplication", "[unit]", float, double) {
    // 2x3 * 3x4, hand-computed - non-square so a swapped index or a transposed operand cannot pass
    const double m1v[2][3] = { { 1, 2, 3 }, { 4, 5, 6 } };
    const double m2v[3][4] = { { 1, 0, 2, -1 }, { 0, 1, -1, 2 }, { 3, -2, 0, 1 } };
    const double expected[2][4] = { { 10, -4, 0, 6 }, { 22, -7, 3, 12 } };

    TestType **m1 = Matrix2dAlloc<TestType>(2, 3);
    TestType **m2 = Matrix2dAlloc<TestType>(3, 4);
    TestType **m2t = Matrix2dAlloc<TestType>(4, 3);   // m2 transposed, for the transposeMat2 path
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 3; ++j) m1[i][j] = static_cast<TestType>(m1v[i][j]);
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 4; ++j) m2[i][j] = static_cast<TestType>(m2v[i][j]);
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 4; ++j) m2t[j][i] = static_cast<TestType>(m2v[i][j]);

    SECTION("allocating variant") {
        TestType **res = Matrix2dMultiply<TestType>(m1, 2, 3, m2, 3, 4, false);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 4; ++j) {
                INFO("entry (" << i << "," << j << ")");
                REQUIRE(double(res[i][j]) == expected[i][j]);
            }
        Matrix2dDealloc(2, res);
    }
    SECTION("in-place variant") {
        TestType **res = Matrix2dAlloc<TestType>(2, 4);
        Matrix2dMultiply<TestType>(m1, 2, 3, m2, 3, 4, res, false);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 4; ++j)
                REQUIRE(double(res[i][j]) == expected[i][j]);
        Matrix2dDealloc(2, res);
    }
    SECTION("transposed second operand") {
        // Multiplying by m2t with transposeMat2 must give the same product
        TestType **res = Matrix2dMultiply<TestType>(m1, 2, 3, m2t, 4, 3, true);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 4; ++j) {
                INFO("entry (" << i << "," << j << ")");
                REQUIRE(double(res[i][j]) == expected[i][j]);
            }
        Matrix2dDealloc(2, res);
    }
    SECTION("matrix-vector") {
        const double vec[3] = { 2, -1, 3 };
        const double expectedVec[2] = { 9, 21 };
        TestType v[3], out[2];
        for (int i = 0; i < 3; ++i) v[i] = static_cast<TestType>(vec[i]);

        TestType *res = Matrix2dVectorMultiply<TestType>(m1, 2, 3, v);
        REQUIRE(double(res[0]) == expectedVec[0]);
        REQUIRE(double(res[1]) == expectedVec[1]);
        free(res);

        Matrix2dVectorMultiply<TestType>(m1, 2, 3, v, out);
        REQUIRE(double(out[0]) == expectedVec[0]);
        REQUIRE(double(out[1]) == expectedVec[1]);
    }

    Matrix2dDealloc(2, m1);
    Matrix2dDealloc(3, m2);
    Matrix2dDealloc(4, m2t);
}

TEST_CASE("Maths: heap sort", "[unit]") {
    // Sorting has a complete specification: ascending order, and the output is a permutation of the
    // input. For the indexed variant the indices must still point at the values they arrived with.
    std::mt19937 gen(0);
    std::uniform_real_distribution<float> distr(-100.f, 100.f);
    constexpr int n = 257;   // odd, > 1 heap level, not a power of two

    SECTION("indexed variant") {
        std::vector<float> values(n), original(n);
        std::vector<int> index(n);
        for (int i = 0; i < n; ++i) { values[i] = distr(gen); index[i] = i; }
        original = values;

        HeapSort(values.data(), index.data(), n);

        for (int i = 1; i < n; ++i) REQUIRE(values[i - 1] <= values[i]);
        for (int i = 0; i < n; ++i) {
            INFO("position " << i);
            REQUIRE(original[index[i]] == values[i]);   // the index still names its value
        }
        std::vector<int> seen(index.begin(), index.end());
        std::sort(seen.begin(), seen.end());
        for (int i = 0; i < n; ++i) REQUIRE(seen[i] == i);   // and the indices are a permutation
    }
    SECTION("plain variant, float and double") {
        std::vector<float> valuesF(n);
        for (auto& v : valuesF) v = distr(gen);
        std::vector<float> sortedF(valuesF);
        std::sort(sortedF.begin(), sortedF.end());
        HeapSort(valuesF.data(), n);
        REQUIRE(valuesF == sortedF);

        std::vector<double> valuesD(n);
        for (auto& v : valuesD) v = distr(gen);
        std::vector<double> sortedD(valuesD);
        std::sort(sortedD.begin(), sortedD.end());
        HeapSort(valuesD.data(), n);
        REQUIRE(valuesD == sortedD);
    }
    SECTION("already sorted, reversed, and duplicated inputs") {
        for (const int variant : { 0, 1, 2 }) {
            std::vector<float> values(n);
            for (int i = 0; i < n; ++i)
                values[i] = variant == 0 ? float(i) : (variant == 1 ? float(n - i) : float(i % 7));
            std::vector<float> sorted(values);
            std::sort(sorted.begin(), sorted.end());
            HeapSort(values.data(), n);
            INFO("variant " << variant);
            REQUIRE(values == sorted);
        }
    }
}

TEMPLATE_TEST_CASE("Maths: 4x4 determinant", "[unit]", float, double) {
    SECTION("identity and triangular") {
        mat44 eye;
        Mat44Eye(&eye);
        REQUIRE(double(Mat44Det<TestType>(&eye)) == 1.0);

        // Triangular: the determinant is the product of the diagonal, whatever sits above it
        const mat44 tri = MakeMat44({ 2, 5, -3, 7,
                                      0, -1, 4, 2,
                                      0, 0, 3, -8,
                                      0, 0, 0, 0.5 });
        REQUIRE(std::abs(double(Mat44Det<TestType>(&tri)) - (2 * -1 * 3 * 0.5)) < kTol);
    }
    SECTION("singular") {
        // Two equal rows
        const mat44 sing = MakeMat44({ 1, 2, 3, 4,
                                       5, 6, 7, 8,
                                       1, 2, 3, 4,
                                       0, 1, 0, 1 });
        REQUIRE(std::abs(double(Mat44Det<TestType>(&sing))) < kTol);
    }
    SECTION("dense matrices with hand-computed determinants") {
        // Fully dense, including the bottom row: the Leibniz expansion has 24 signed terms and a
        // triangular or affine-shaped input zeroes most of them, so a single wrong sign can hide.
        // These two leave every term alive. det(A) = 72 and det(B) = 8, by exact integer expansion.
        const mat44 a = MakeMat44({ 1, 2, 3, 4,
                                    5, 6, 7, 8,
                                    2, 6, 4, 8,
                                    3, 1, 1, 2 });
        const mat44 b = MakeMat44({ 2, 1, 0, 3,
                                    1, 3, 2, 1,
                                    4, 0, 1, 2,
                                    1, 1, 1, 1 });
        REQUIRE(double(Mat44Det<TestType>(&a)) == 72.0);
        REQUIRE(double(Mat44Det<TestType>(&b)) == 8.0);
        // And multiplicativity on the same dense pair: det(AB) = 576 exactly
        const mat44 ab = const_cast<mat44&>(a) * const_cast<mat44&>(b);
        REQUIRE(double(Mat44Det<TestType>(&ab)) == 576.0);
    }
    SECTION("multiplicativity") {
        mat44 a = MakeMat44({ 1, 0.5, 0, 2,
                              -0.25, 1.5, 0.75, 0,
                              0, 1, 2, -1,
                              0, 0, 0, 1 });
        mat44 b = RotationZ(0.7);
        b.m[2][3] = 3.f;
        const mat44 ab = a * b;
        const double detA = Mat44Det<TestType>(&a);
        const double detB = Mat44Det<TestType>(&b);
        const double detAB = Mat44Det<TestType>(&ab);
        INFO("det(A) = " << detA << ", det(B) = " << detB << ", det(AB) = " << detAB);
        REQUIRE(std::abs(detAB - detA * detB) < 1e-4);
    }
}

TEST_CASE("Maths: symmetric 3x3 diagonalisation", "[unit]") {
    /*
        A = [[2,1,0],[1,2,0],[0,0,5]] has eigenvalues {1, 3, 5} by hand. The oracle asserts what a
        diagonalisation IS: D diagonal holding those eigenvalues, Q orthonormal, and A reconstructed
        from them (production returns A = Q^T D Q; the reconstruction below pins that convention).
    */
    mat33 a{};
    a.m[0][0] = 2; a.m[0][1] = 1; a.m[1][0] = 1; a.m[1][1] = 2; a.m[2][2] = 5;
    mat33 q{}, d{};
    Mat33Diagonalize(&a, &q, &d);

    // D is diagonal and holds {1, 3, 5} in some order
    std::vector<double> eigenvalues;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            if (i != j) {
                INFO("off-diagonal D(" << i << "," << j << ")");
                REQUIRE(std::abs(double(d.m[i][j])) < kTol);
            }
        eigenvalues.push_back(d.m[i][i]);
    }
    std::sort(eigenvalues.begin(), eigenvalues.end());
    REQUIRE(std::abs(eigenvalues[0] - 1.0) < kTol);
    REQUIRE(std::abs(eigenvalues[1] - 3.0) < kTol);
    REQUIRE(std::abs(eigenvalues[2] - 5.0) < kTol);

    // Q is orthonormal: Q Q^T = I
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double dot = 0;
            for (int k = 0; k < 3; ++k) dot += double(q.m[i][k]) * double(q.m[j][k]);
            INFO("QQ^T (" << i << "," << j << ")");
            REQUIRE(std::abs(dot - (i == j ? 1.0 : 0.0)) < kTol);
        }

    // Reconstruction: try both similarity orientations, exactly one must reproduce A
    const auto reconstruct = [&](bool qFirst) {
        double worst = 0;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double sum = 0;
                for (int k = 0; k < 3; ++k)
                    for (int l = 0; l < 3; ++l)
                        sum += (qFirst ? double(q.m[i][k]) : double(q.m[k][i])) * double(d.m[k][l]) *
                               (qFirst ? double(q.m[j][l]) : double(q.m[l][j]));
                worst = std::max(worst, std::abs(sum - double(a.m[i][j])));
            }
        return worst;
    };
    // Production's convention, determined empirically and pinned: A = Q D Q^T
    // (the transposed orientation reconstructs with error ~2, not rounding)
    const double errQdQt = reconstruct(true);
    NR_COUT << "  reconstruction error (A = Q D Q^T): " << std::scientific << errQdQt << std::endl;
    REQUIRE(errQdQt < kTol);
}

TEST_CASE("Maths: matrix exponential", "[unit]") {
    SECTION("exp(0) = I (3x3) and exp(diag) = diag of exponentials") {
        mat33 zero{};
        Mat33Expm(&zero);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                REQUIRE(std::abs(double(zero.m[i][j]) - (i == j ? 1.0 : 0.0)) < kTol);

        mat33 diag{};
        diag.m[0][0] = 0.5f; diag.m[1][1] = -1.f; diag.m[2][2] = 2.f;
        Mat33Expm(&diag);
        REQUIRE(std::abs(double(diag.m[0][0]) - std::exp(0.5)) < kTol);
        REQUIRE(std::abs(double(diag.m[1][1]) - std::exp(-1.0)) < kTol);
        REQUIRE(std::abs(double(diag.m[2][2]) - std::exp(2.0)) < 1e-4);
        REQUIRE(std::abs(double(diag.m[0][1])) < kTol);
    }
    SECTION("exp of a translation generator is exactly I + N") {
        // N has only m[0..2][3] non-zero, so N^2 = 0 and the series terminates after two terms
        mat44 n{};
        n.m[0][3] = 3.5f; n.m[1][3] = -2.25f; n.m[2][3] = 1.75f;
        const mat44 result = Mat44Expm(&n);
        mat44 expected;
        Mat44Eye(&expected);
        expected.m[0][3] = 3.5f; expected.m[1][3] = -2.25f; expected.m[2][3] = 1.75f;
        RequireMat44Near(result, expected, kTol, "exp(translation generator)");
    }
    SECTION("exp of a rotation generator is the rotation") {
        // The generator [[0,-t],[t,0]] exponentiates to the rotation by t - the closed form that
        // defines the matrix exponential on so(2)
        constexpr double theta = 0.6;
        mat44 g{};
        g.m[0][1] = static_cast<float>(-theta);
        g.m[1][0] = static_cast<float>(theta);
        const mat44 result = Mat44Expm(&g);
        RequireMat44Near(result, RotationZ(theta), kTol, "exp(rotation generator)");
    }
    SECTION("det(exp(A)) = e^trace(A)") {
        mat44 a = MakeMat44({ 0.2, 0.1, 0, 1,
                              -0.1, 0.3, 0.05, -2,
                              0, 0.02, -0.1, 0.5,
                              0, 0, 0, 0 });
        const mat44 e = Mat44Expm(&a);
        const double trace = 0.2 + 0.3 - 0.1 + 0.0;
        REQUIRE(std::abs(double(Mat44Det<double>(&e)) - std::exp(trace)) < 1e-4);
    }
}

TEST_CASE("Maths: matrix logarithm", "[unit]") {
    SECTION("log(I) = 0 and the round trip") {
        mat44 eye;
        Mat44Eye(&eye);
        const mat44 logI = Mat44Logm(&eye);
        RequireMat44Near(logI, mat44{}, kTol, "log(I)");

        // log(exp(A)) = A for a small A (within the principal branch)
        mat44 a = MakeMat44({ 0.1, 0.2, 0, 1.5,
                              -0.2, 0.1, 0.05, -0.75,
                              0, -0.05, 0.15, 0.5,
                              0, 0, 0, 0 });
        const mat44 e = Mat44Expm(&a);
        const mat44 back = Mat44Logm(&e);
        RequireMat44Near(back, a, 1e-4, "log(exp(A))");
    }
    SECTION("log of a pure translation is its generator") {
        mat44 t;
        Mat44Eye(&t);
        t.m[0][3] = 4.f; t.m[1][3] = -1.5f; t.m[2][3] = 0.5f;
        const mat44 logT = Mat44Logm(&t);
        mat44 expected{};
        expected.m[0][3] = 4.f; expected.m[1][3] = -1.5f; expected.m[2][3] = 0.5f;
        RequireMat44Near(logT, expected, kTol, "log(translation)");
    }
    SECTION("3x3 log round trip on a symmetric positive-definite tensor") {
        mat33 tensor{};
        tensor.m[0][0] = 2; tensor.m[0][1] = 0.5f; tensor.m[1][0] = 0.5f;
        tensor.m[1][1] = 1.5f; tensor.m[2][2] = 3;
        mat33 roundTrip = tensor;
        Mat33Logm(&roundTrip);
        Mat33Expm(&roundTrip);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                INFO("entry (" << i << "," << j << ")");
                REQUIRE(std::abs(double(roundTrip.m[i][j]) - double(tensor.m[i][j])) < 1e-4);
            }
    }
    SECTION("guards: an all-zero or singular tensor logs to NaN, a NaN input is left alone") {
        // Production refuses the log of a non-invertible tensor by filling it with NaN
        mat33 zero{};
        Mat33Logm(&zero);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                REQUIRE(std::isnan(zero.m[i][j]));

        mat33 singular{};
        singular.m[0][0] = 1; singular.m[1][1] = 1;   // third row/column zero: det = 0
        Mat33Logm(&singular);
        REQUIRE(std::isnan(singular.m[0][0]));

        // A NaN entry makes the function return without touching the rest
        mat33 withNan{};
        withNan.m[0][0] = std::numeric_limits<float>::quiet_NaN();
        withNan.m[1][1] = 7.f;
        Mat33Logm(&withNan);
        REQUIRE(withNan.m[1][1] == 7.f);
    }
}

TEST_CASE("Maths: log-Euclidean average of two transformations", "[unit]") {
    SECTION("average of A with itself is A") {
        mat44 a = RotationZ(0.4);
        a.m[0][3] = 2.f; a.m[1][3] = -1.f;
        const mat44 avg = Mat44Avg2(&a, &a);
        RequireMat44Near(avg, a, 1e-4, "avg(A, A)");
    }
    SECTION("average of two translations is the mid translation") {
        // Translation logs are nilpotent generators, so the log-space mean is exactly the mid vector
        mat44 t1, t2, expected;
        Mat44Eye(&t1); Mat44Eye(&t2); Mat44Eye(&expected);
        t1.m[0][3] = 4.f; t1.m[1][3] = -2.f; t1.m[2][3] = 1.f;
        t2.m[0][3] = 2.f; t2.m[1][3] = 6.f;  t2.m[2][3] = -3.f;
        expected.m[0][3] = 3.f; expected.m[1][3] = 2.f; expected.m[2][3] = -1.f;
        RequireMat44Near(Mat44Avg2(&t1, &t2), expected, kTol, "avg(translations)");
    }
    SECTION("average of two coaxial rotations is the mid rotation") {
        const mat44 r1 = RotationZ(0.2), r2 = RotationZ(0.8);
        RequireMat44Near(Mat44Avg2(&r1, &r2), RotationZ(0.5), 1e-4, "avg(rotations)");
    }
    SECTION("the average is symmetric in its arguments") {
        mat44 a = RotationZ(0.3); a.m[0][3] = 1.5f;
        mat44 b = RotationZ(-0.2); b.m[1][3] = -2.5f;
        RequireMat44Near(Mat44Avg2(&a, &b), Mat44Avg2(&b, &a), kTol, "avg symmetry");
    }
}
