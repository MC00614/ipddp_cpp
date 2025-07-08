#pragma once

#include "dynamics/linear_discrete_dynamics.h"
#include "dynamics/discrete_dynamics_base.h"

#include <pybind11/pybind11.h>

template <typename Scalar>
class PyDiscreteDynamics : public DiscreteDynamicsBase<Scalar> {
public:
    using DiscreteDynamicsBase<Scalar>::DiscreteDynamicsBase;

    Eigen::VectorXd f(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override {
        PYBIND11_OVERLOAD_PURE(Eigen::VectorXd, DiscreteDynamicsBase<Scalar>, f, x, u);
    }

    Eigen::MatrixXd fx(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override {
        PYBIND11_OVERLOAD_PURE(Eigen::MatrixXd, DiscreteDynamicsBase<Scalar>, fx, x, u);
    }

    Eigen::MatrixXd fu(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override {
        PYBIND11_OVERLOAD_PURE(Eigen::MatrixXd, DiscreteDynamicsBase<Scalar>, fu, x, u);
    }

    int getDimX() const override {
        PYBIND11_OVERLOAD_PURE(int, DiscreteDynamicsBase<Scalar>, getDimX);
    }

    int getDimU() const override {
        PYBIND11_OVERLOAD_PURE(int, DiscreteDynamicsBase<Scalar>, getDimU);
    }

    Scalar getDT() const override {
        PYBIND11_OVERLOAD_PURE(Scalar, DiscreteDynamicsBase<Scalar>, getDT);
    }
};