#pragma once

#include "cost/scalar_quadratic_stage_cost.h"
#include "cost/quadratic_stage_cost.h"
#include "cost/stage_cost_base.h"

#include <pybind11/pybind11.h>

class PyStageCost : public StageCostBase<double> {
public:
    using StageCostBase<double>::StageCostBase;

    double q(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override {
        PYBIND11_OVERLOAD_PURE(double, StageCostBase<double>, q, x, u);
    }

    Eigen::VectorXd qx(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override {
        PYBIND11_OVERLOAD_PURE(Eigen::VectorXd, StageCostBase<double>, qx, x, u);
    }

    Eigen::VectorXd qu(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override {
        PYBIND11_OVERLOAD_PURE(Eigen::VectorXd, StageCostBase<double>, qu, x, u);
    }

    Eigen::MatrixXd qxx(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override {
        PYBIND11_OVERLOAD_PURE(Eigen::MatrixXd, StageCostBase<double>, qxx, x, u);
    }

    Eigen::MatrixXd quu(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override {
        PYBIND11_OVERLOAD_PURE(Eigen::MatrixXd, StageCostBase<double>, quu, x, u);
    }

    Eigen::MatrixXd qxu(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override {
        PYBIND11_OVERLOAD_PURE(Eigen::MatrixXd, StageCostBase<double>, qxu, x, u);
    }
};
