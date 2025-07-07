#pragma once

#include "stage_constraint_base.h"

template <typename Scalar>
class LinearStageConstraint : public StageConstraintBase<Scalar> {
public:
    LinearStageConstraint() {}

    LinearStageConstraint(const Matrix<Scalar>& Cx_in, const Matrix<Scalar>& Cu_in, const Vector<Scalar>& c0_in, ConstraintType type)
        : Cx(Cx_in), Cu(Cu_in), c0(c0_in) {
        this->constraint_type = type;
        this->dim_c = static_cast<int>(c0_in.size());
    }

    virtual Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        return Cx * x + Cu * u + c0;
    }

    virtual Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        return Cx;
    }

    virtual Matrix<Scalar> cu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        return Cu;
    }

    void setCx(const Matrix<Scalar>& Cx_in) {
        Cx = Cx_in;
    }

    void setCu(const Matrix<Scalar>& Cu_in) {
        Cu = Cu_in;
    }

    void setC0(const Vector<Scalar>& c0_in) {
        c0 = c0_in;
    }

private:
    Matrix<Scalar> Cx;
    Matrix<Scalar> Cu;
    Vector<Scalar> c0;
};