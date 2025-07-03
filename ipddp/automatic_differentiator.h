#pragma once
#include "types.h"
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <memory>

namespace autodiff_utils {

template <typename Scalar>
Matrix<Scalar> jacobianX(const DiscreteDynamicsBase<Scalar>* dynamics, const Vector<Scalar>& x, const Vector<Scalar>& u) {
    using autodiff::dual;

    Vector<dual> xdual = x.template cast<dual>();
    Vector<dual> udual = u.template cast<dual>();

    auto fun = [dynamics, &u](const Vector<dual>& xdual_in) {
        return dynamics->f(xdual_in, udual);
    };

    Matrix<Scalar> J;
    Vector<dual> y;
    autodiff::jacobian(fun, autodiff::wrt(xdual), autodiff::at(xdual), y, J);
    return J;
}

template <typename Scalar>
static Matrix<Scalar> jacobianU(const DiscreteDynamicsBase<Scalar>* dynamics, const Vector<Scalar>& x, const Vector<Scalar>& u) {
    using autodiff::dual;

    Vector<dual> xdual = x.template cast<dual>();
    Vector<dual> udual = u.template cast<dual>();

    auto fun = [dynamics, &x](const Vector<dual>& udual_in) {
        return dynamics->f(xdual, udual_in);
    };

    Matrix<Scalar> J;
    Vector<dual> y;
    autodiff::jacobian(fun, autodiff::wrt(udual), autodiff::at(udual), y, J);
    return J;
}

};
