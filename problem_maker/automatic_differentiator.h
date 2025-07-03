#pragma once

#include "types.h"
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

namespace autodiff_utils {

template <typename Dynamics, typename Scalar>
Matrix<Scalar> jacobianX(const Dynamics& dynamics, const Vector<Scalar>& x, const Vector<Scalar>& u) {
    using autodiff::dual;

    Vector<dual> xdual = x.template cast<dual>();
    Vector<dual> udual = u.template cast<dual>();

    auto fun = [&dynamics, &udual](const Vector<dual>& xdual_in) {
        return dynamics.f_(xdual_in, udual);
    };

    Matrix<Scalar> J;
    Vector<dual> y;
    autodiff::jacobian(fun, autodiff::wrt(xdual), autodiff::at(xdual), y, J);
    return J;
}

template <typename Dynamics, typename Scalar>
static Matrix<Scalar> jacobianU(const Dynamics& dynamics, const Vector<Scalar>& x, const Vector<Scalar>& u) {
    using autodiff::dual;

    Vector<dual> xdual = x.template cast<dual>();
    Vector<dual> udual = u.template cast<dual>();

    auto fun = [&dynamics, &xdual](const Vector<dual>& udual_in) {
        return dynamics->f_(xdual, udual_in);
    };

    Matrix<Scalar> J;
    Vector<dual> y;
    autodiff::jacobian(fun, autodiff::wrt(udual), autodiff::at(udual), y, J);
    return J;
}

template <typename Cost, typename Scalar>
Vector<Scalar> derivativeX(const Cost& cost, const Vector<Scalar>& x) {
    using autodiff::dual;
    Vector<dual> xdual = x.template cast<dual>();

    auto fun = [&cost](const Vector<dual>& xdual_in)
    {
        return cost.p_(xdual_in);
    };

    Vector<Scalar> grad;
    autodiff::gradient(fun, autodiff::wrt(xdual), autodiff::at(xdual), grad);
    return grad;
}

template <typename Cost, typename Scalar>
Matrix<Scalar> hessianX(const Cost& cost, const Vector<Scalar>& x) {
    using autodiff::dual2nd;
    using autodiff::VectorXdual2nd;

    VectorXdual2nd xdual = x.template cast<dual2nd>();

    auto fun = [&cost](const VectorXdual2nd& xdual_in)
    {
        return cost.p_(xdual_in);
    };

    Matrix<Scalar> H;
    autodiff::hessian(fun, autodiff::wrt(xdual), autodiff::at(xdual), H);
    return H;
}

template <typename Cost, typename Scalar>
Vector<Scalar> derivativeX(const Cost& cost, const Vector<Scalar>& x, const Vector<Scalar>& u) {
    using autodiff::dual;
    Vector<dual> xdual = x.template cast<dual>();
    Vector<dual> udual = u.template cast<dual>();

    auto fun = [&cost, &udual](const Vector<dual>& xdual_in)
    {
        return cost.q_(xdual_in, udual);
    };

    Vector<Scalar> grad;
    autodiff::gradient(fun, autodiff::wrt(xdual), autodiff::at(xdual, udual), grad);
    return grad;
}

template <typename Cost, typename Scalar>
Vector<Scalar> derivativeU(const Cost& cost, const Vector<Scalar>& x, const Vector<Scalar>& u) {
    using autodiff::dual;
    Vector<dual> xdual = x.template cast<dual>();
    Vector<dual> udual = u.template cast<dual>();

    auto fun = [&cost, &xdual](const Vector<dual>& udual_in)
    {
        return cost.q_(xdual, udual_in);
    };

    Vector<Scalar> grad;
    autodiff::gradient(fun, autodiff::wrt(udual), autodiff::at(udual), grad);
    return grad;
}

template <typename Cost, typename Scalar>
Matrix<Scalar> hessianXU(const Cost& cost, const Vector<Scalar>& x, const Vector<Scalar>& u) {
    using autodiff::dual2nd;
    using autodiff::VectorXdual2nd;

    VectorXdual2nd zdual(x.size() + u.size());
    zdual << x.template cast<dual2nd>(), u.template cast<dual2nd>();

    auto fun = [&cost, nx=x.size()](const VectorXdual2nd& zdual_in)
    {
        auto xdual_in = zdual_in.head(nx);
        auto udual_in = zdual_in.tail(zdual_in.size()-nx);
        return cost.q_(xdual_in, udual_in);
    };

    Matrix<Scalar> H;
    autodiff::hessian(fun, autodiff::wrt(zdual), autodiff::at(zdual), H);
    return H;
}

};



// Vector<dual> zdual(x.size() + u.size());
// zdual << x.cast<dual>(), u.cast<dual>();

// auto fun = [&dynamics, nx = x.size()](const Vector<dual>& zdual_in) {
//     auto xdual_in = zdual_in.head(nx);
//     auto udual_in = zdual_in.tail(zdual_in.size() - nx);
//     return dynamics.f_(xdual_in, udual_in);
// };

// Matrix<double> J;
// Vector<dual> y;
// autodiff::jacobian(fun, autodiff::wrt(zdual), autodiff::at(zdual), y, J);

// auto fx = J.leftCols(x.size());
// auto fu = J.rightCols(u.size());