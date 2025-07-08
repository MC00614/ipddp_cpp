#pragma once

#include "constraint/stage_constraint_base.h"
#include <pybind11/pybind11.h>

class PyStageConstraint : public StageConstraintBase<double> {
public:
    using StageConstraintBase<double>::StageConstraintBase;

    Eigen::VectorXd c(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override {
        PYBIND11_OVERLOAD_PURE(Eigen::VectorXd, StageConstraintBase, c, x, u);
    }

    Eigen::MatrixXd cx(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override {
        PYBIND11_OVERLOAD_PURE(Eigen::MatrixXd, StageConstraintBase, cx, x, u);
    }

    Eigen::MatrixXd cu(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override {
        PYBIND11_OVERLOAD_PURE(Eigen::MatrixXd, StageConstraintBase, cu, x, u);
    }

    void setConstraintType(ConstraintType constraint_type) override {
        PYBIND11_OVERLOAD(void, StageConstraintBase, setConstraintType, constraint_type);
    }

    void setDimC(int dim_c) override {
        PYBIND11_OVERLOAD(void, StageConstraintBase, setDimC, dim_c);
    }

    ConstraintType getConstraintType() const override {
        PYBIND11_OVERLOAD(ConstraintType, StageConstraintBase, getConstraintType);
    }

    int getDimC() const override {
        PYBIND11_OVERLOAD(int, StageConstraintBase, getDimC);
    }
};
