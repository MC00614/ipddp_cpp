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

};
