#include "reg_test_common.h"
#include "_reg_splineBasis.h"
#include <catch2/catch_template_test_macros.hpp>

/*
    The spline basis functions, checked against their defining properties.

    Everything in this repo that evaluates a spline is built on these functions - the deformation
    field, the bending energy, the linear elasticity, the Jacobian, and the CUDA counterparts of each.
    They are also duplicated: each family is written once as an array form and once as a switch on the
    weight index, and each is instantiated at both float and double, with every literal typed so that
    an instantiation evaluates wholly in its own precision.

    The checks are properties of the mathematics rather than restatements of the formulae, so a
    passing test says the weights are right and not merely unchanged:

      - partition of unity: the weights sum to 1, so a spline reproduces constants. The first
        derivative weights sum to 0, and the second derivative weights too.
      - symmetry: the cubic B-spline is symmetric about the centre of its support.
      - non-negativity: the cubic B-spline weights are all >= 0. Not decoration - accumulating a
        weighted sum of them in float is only safe against cancellation because of it.
      - the derivative arrays really are the derivatives of the value array (central differences).
      - the values at the knots, which are what separate the two families: the Catmull-Rom variant
        puts all the weight on one node and so interpolates, the B-spline spreads it 1/6, 4/6, 1/6
        and so does not.
      - the switch form agrees with the array form, at every index and outside the support.

    Both instantiations are exercised throughout: a float-typed literal in a double instantiation is
    invisible to any check that only runs one of them.
*/

namespace {

// A sweep that includes the endpoints and avoids landing only on exact binary fractions, since those
// are the values at which a float and a double evaluation cannot disagree
const std::vector<double>& BasisSweep() {
    static const std::vector<double> sweep = [] {
        std::vector<double> values{ 0.0, 1.0, 0.5, 0.25, 0.75 };
        for (int i = 1; i < 20; ++i)
            values.push_back(i / 20.9);   // deliberately not a dyadic rational
        return values;
    }();
    return sweep;
}

template<class DataType>
struct Basis { DataType values[4], first[4], second[4]; };

template<class DataType>
Basis<DataType> BSpline(double basis) {
    Basis<DataType> b{};
    get_BSplineBasisValues<DataType>(static_cast<DataType>(basis), b.values, b.first, b.second);
    return b;
}

template<class DataType>
Basis<DataType> Spline(double basis) {
    Basis<DataType> b{};
    get_SplineBasisValues<DataType>(static_cast<DataType>(basis), b.values, b.first, b.second);
    return b;
}

template<class DataType>
double Sum(const DataType (&values)[4]) {
    return double(values[0]) + double(values[1]) + double(values[2]) + double(values[3]);
}

// Tolerances. The float instantiation accumulates four float weights, so a few ulp of a quantity of
// order 1 is the floor; double is limited only by the finite-difference step where one is used.
template<class DataType> constexpr double Tolerance();
template<> constexpr double Tolerance<float>() { return 1e-6; }
template<> constexpr double Tolerance<double>() { return 1e-12; }

template<class DataType>
const char* TypeName();
template<> const char* TypeName<float>() { return "float"; }
template<> const char* TypeName<double>() { return "double"; }

} // namespace

TEMPLATE_TEST_CASE("Spline basis weights form a partition of unity", "[unit]", float, double) {
    // A spline whose coefficients are all equal must reproduce that constant, which is exactly the
    // statement that the weights sum to one at every position in the support.
    for (const double basis : BasisSweep()) {
        const auto b = BSpline<TestType>(basis);
        const auto s = Spline<TestType>(basis);
        INFO(TypeName<TestType>() << ", basis = " << basis);
        REQUIRE(std::abs(Sum(b.values) - 1.0) < Tolerance<TestType>());
        REQUIRE(std::abs(Sum(s.values) - 1.0) < Tolerance<TestType>());
        // Differentiating a constant gives zero, so the derivative weights must cancel
        REQUIRE(std::abs(Sum(b.first)) < Tolerance<TestType>());
        REQUIRE(std::abs(Sum(s.first)) < Tolerance<TestType>());
        REQUIRE(std::abs(Sum(b.second)) < Tolerance<TestType>());
        REQUIRE(std::abs(Sum(s.second)) < Tolerance<TestType>());
    }
}

TEMPLATE_TEST_CASE("Cubic B-spline basis weights are symmetric and non-negative", "[unit]", float, double) {
    for (const double basis : BasisSweep()) {
        const auto b = BSpline<TestType>(basis);
        const auto mirrored = BSpline<TestType>(1.0 - basis);
        INFO(TypeName<TestType>() << ", basis = " << basis);
        for (int i = 0; i < 4; ++i) {
            // The support is symmetric about its centre: weight i at t equals weight 3-i at 1-t
            REQUIRE(std::abs(double(b.values[i]) - double(mirrored.values[3 - i])) < Tolerance<TestType>());
            // All-positive weights are what makes the float accumulation in the convolution and LNCC
            // paths round to the same value as a double accumulation. If this ever stops holding,
            // those optimisations stop being safe.
            REQUIRE(double(b.values[i]) >= 0.0);
        }
    }
}

TEMPLATE_TEST_CASE("Spline basis derivatives match the value weights", "[unit]", float, double) {
    // The first and second arrays claim to be the derivatives of the value array. Central differences
    // in double, so the reference is independent of the type under test.
    constexpr double h = 1e-5;
    for (const double basis : BasisSweep()) {
        // Stay clear of the ends, where a central difference would step outside the sweep
        if (basis < 2 * h || basis > 1 - 2 * h) continue;
        const auto b = BSpline<TestType>(basis);
        const auto s = Spline<TestType>(basis);
        const auto bPlus = BSpline<double>(basis + h), bMinus = BSpline<double>(basis - h);
        const auto sPlus = Spline<double>(basis + h), sMinus = Spline<double>(basis - h);
        const auto bMid = BSpline<double>(basis), sMid = Spline<double>(basis);
        INFO(TypeName<TestType>() << ", basis = " << basis);
        for (int i = 0; i < 4; ++i) {
            const double bFirst = (bPlus.values[i] - bMinus.values[i]) / (2 * h);
            const double sFirst = (sPlus.values[i] - sMinus.values[i]) / (2 * h);
            const double bSecond = (bPlus.values[i] - 2 * bMid.values[i] + bMinus.values[i]) / (h * h);
            const double sSecond = (sPlus.values[i] - 2 * sMid.values[i] + sMinus.values[i]) / (h * h);
            // The finite-difference truncation error, not the type, sets these bounds
            REQUIRE(std::abs(double(b.first[i]) - bFirst) < 1e-6);
            REQUIRE(std::abs(double(s.first[i]) - sFirst) < 1e-6);
            REQUIRE(std::abs(double(b.second[i]) - bSecond) < 1e-3);
            REQUIRE(std::abs(double(s.second[i]) - sSecond) < 1e-3);
        }
    }
}

TEMPLATE_TEST_CASE("Spline basis weights take their defining values at the knots", "[unit]", float, double) {
    // These four sets of numbers are what distinguish the two families, and they are what the
    // deformation field relies on at every voxel that lands exactly on a control point.
    const auto b0 = BSpline<TestType>(0.0);
    const auto b1 = BSpline<TestType>(1.0);
    const auto s0 = Spline<TestType>(0.0);
    const auto s1 = Spline<TestType>(1.0);
    const double tol = Tolerance<TestType>();

    // The cubic B-spline approximates: at a knot it spreads weight 1/6, 4/6, 1/6 over three nodes,
    // which is why an identity control point grid does not evaluate to exactly the identity field.
    REQUIRE(std::abs(double(b0.values[0]) - 1.0 / 6.0) < tol);
    REQUIRE(std::abs(double(b0.values[1]) - 4.0 / 6.0) < tol);
    REQUIRE(std::abs(double(b0.values[2]) - 1.0 / 6.0) < tol);
    REQUIRE(std::abs(double(b0.values[3])) < tol);
    // Shifting by a full knot moves the same weights along by one node
    REQUIRE(std::abs(double(b1.values[0])) < tol);
    REQUIRE(std::abs(double(b1.values[1]) - 1.0 / 6.0) < tol);
    REQUIRE(std::abs(double(b1.values[2]) - 4.0 / 6.0) < tol);
    REQUIRE(std::abs(double(b1.values[3]) - 1.0 / 6.0) < tol);

    // The Catmull-Rom variant interpolates: all the weight sits on the node itself, so a grid
    // coordinate that is an exact integer returns that control point's value unchanged
    REQUIRE(std::abs(double(s0.values[0])) < tol);
    REQUIRE(std::abs(double(s0.values[1]) - 1.0) < tol);
    REQUIRE(std::abs(double(s0.values[2])) < tol);
    REQUIRE(std::abs(double(s0.values[3])) < tol);
    REQUIRE(std::abs(double(s1.values[0])) < tol);
    REQUIRE(std::abs(double(s1.values[1])) < tol);
    REQUIRE(std::abs(double(s1.values[2]) - 1.0) < tol);
    REQUIRE(std::abs(double(s1.values[3])) < tol);
}

TEMPLATE_TEST_CASE("Single-index basis form agrees with the array form", "[unit]", float, double) {
    // get_BSplineBasisValue evaluates one weight through a switch, duplicating the formulae in
    // get_BSplineBasisValues. Nothing but this test keeps the two in step.
    for (const double basis : BasisSweep()) {
        const auto b = BSpline<TestType>(basis);
        for (int index = 0; index < 4; ++index) {
            TestType value{}, first{}, second{};
            get_BSplineBasisValue<TestType>(static_cast<TestType>(basis), index, value, first, second);
            INFO(TypeName<TestType>() << ", basis = " << basis << ", index = " << index);
            REQUIRE(std::abs(double(value) - double(b.values[index])) < Tolerance<TestType>());
            REQUIRE(std::abs(double(first) - double(b.first[index])) < Tolerance<TestType>());
            REQUIRE(std::abs(double(second) - double(b.second[index])) < Tolerance<TestType>());
        }
    }
}

TEMPLATE_TEST_CASE("Single-index basis form returns zero outside the support", "[unit]", float, double) {
    // The default branch of the switch. Callers index it from a loop bound, so an off-by-one there
    // would silently pick up whatever this returns.
    for (const int index : { -1, 4, 100 }) {
        TestType value{ 1 }, first{ 1 }, second{ 1 };
        get_BSplineBasisValue<TestType>(static_cast<TestType>(0.3), index, value, first, second);
        INFO(TypeName<TestType>() << ", index = " << index);
        REQUIRE(double(value) == 0.0);
        REQUIRE(double(first) == 0.0);
        REQUIRE(double(second) == 0.0);
    }
}

TEMPLATE_TEST_CASE("Resampling cubic kernel properties", "[unit]", float, double) {
    /*
        InterpCubicSplineKernel (reg_test_common.h) supplies the expected values for the
        interpolation, image-gradient and resampling tests. Pinning its defining properties here is
        what makes that legitimate: the helper is then validated independently, not trusted because
        it happens to match the production kernels it checks.

          - partition of unity: weights sum to 1 at any offset, so constants are reproduced;
          - linear precision: sum(w_i * (i - 1)) == t, so linear ramps are reproduced exactly - the
            property the ramp-based resampling and gradient tests rely on;
          - interpolation: at t = 0 the weights are (0,1,0,0), so on-voxel samples return the voxel;
          - the derivative weights sum to 0 (constants have zero gradient) and their first moment is
            1 (a unit ramp has unit gradient).
    */
    for (int step = 0; step <= 20; ++step) {
        const TestType t = static_cast<TestType>(step / 20.0);
        TestType basis[4], derivative[4];
        InterpCubicSplineKernel(t, basis, derivative);

        double sum = 0, firstMoment = 0, derivSum = 0, derivMoment = 0;
        for (int i = 0; i < 4; ++i) {
            sum += double(basis[i]);
            firstMoment += double(basis[i]) * (i - 1);
            derivSum += double(derivative[i]);
            derivMoment += double(derivative[i]) * (i - 1);
        }
        INFO("t = " << t);
        REQUIRE(std::abs(sum - 1.0) < 1e-6);
        REQUIRE(std::abs(firstMoment - double(t)) < 1e-6);
        REQUIRE(std::abs(derivSum) < 1e-6);
        REQUIRE(std::abs(derivMoment - 1.0) < 1e-5);
    }
    // On-voxel: all the weight on the sample itself
    TestType basis[4];
    InterpCubicSplineKernel(TestType(0), basis);
    REQUIRE(double(basis[0]) == 0.0);
    REQUIRE(double(basis[1]) == 1.0);
    REQUIRE(double(basis[2]) == 0.0);
    REQUIRE(double(basis[3]) == 0.0);
}
