#pragma once

#include "constraint/terminal_constraint_base.h"
#include <pybind11/pybind11.h>

class PyTerminalConstraint : public TerminalConstraintBase<double> {
public:
    using TerminalConstraintBase<double>::TerminalConstraintBase;

    Eigen::VectorXd cT(const Eigen::VectorXd& x) const override {
        PYBIND11_OVERLOAD_PURE(Eigen::VectorXd, TerminalConstraintBase, cT, x);
    }

    Eigen::MatrixXd cTx(const Eigen::VectorXd& x) const override {
        PYBIND11_OVERLOAD_PURE(Eigen::MatrixXd, TerminalConstraintBase, cTx, x);
    }

    void setConstraintType(ConstraintType constraint_type) override {
        PYBIND11_OVERLOAD(void, TerminalConstraintBase, setConstraintType, constraint_type);
    }

    void setDimCT(int dim_cT) override {
        PYBIND11_OVERLOAD(void, TerminalConstraintBase, setDimCT, dim_cT);
    }

    ConstraintType getConstraintType() const override {
        PYBIND11_OVERLOAD(ConstraintType, TerminalConstraintBase, getConstraintType);
    }

    int getDimCT() const override {
        PYBIND11_OVERLOAD(int, TerminalConstraintBase, getDimCT);
    }
};
