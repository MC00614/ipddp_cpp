#pragma once

#include "cost/scalar_quadratic_terminal_cost.h"
#include "cost/quadratic_terminal_cost.h"
#include "cost/terminal_cost_base.h"

#include <pybind11/pybind11.h>

class PyTerminalCost : public TerminalCostBase<double> {
public:
    using TerminalCostBase<double>::TerminalCostBase;

    double p(const Eigen::VectorXd& x) const override {
        PYBIND11_OVERLOAD_PURE(double, TerminalCostBase<double>, p, x);
    }

    Eigen::VectorXd px(const Eigen::VectorXd& x) const override {
        PYBIND11_OVERLOAD_PURE(Eigen::VectorXd, TerminalCostBase<double>, px, x);
    }

    Eigen::MatrixXd pxx(const Eigen::VectorXd& x) const override {
        PYBIND11_OVERLOAD_PURE(Eigen::MatrixXd, TerminalCostBase<double>, pxx, x);
    }
};
