#pragma once

#include "terminal_constraint_base.h"

template <typename Scalar>
class LinearTerminalConstraint : public TerminalConstraintBase<Scalar> {
public:
    LinearTerminalConstraint() {
    }

    LinearTerminalConstraint(const Matrix<Scalar>& CTx_in, const Vector<Scalar>& cT0_in, ConstraintType type)
        : CTx(CTx_in), cT0(cT0_in) {
        this->constraint_type = type;
        this->dim_cT = static_cast<int>(cT0_in.size());
    }

    virtual Vector<Scalar> cT(const Vector<Scalar>& x) const override {
        return CTx * x + cT0;
    }

    virtual Matrix<Scalar> cTx(const Vector<Scalar>& x) const override {
        return CTx;
    }

    void setCTx(const Matrix<Scalar>& CTx_in) {
        CTx = CTx_in;
    }

    void setcT0(const Vector<Scalar>& cT0_in) {
        cT0 = cT0_in;
    }

private:
    Matrix<Scalar> CTx;
    Vector<Scalar> cT0;
};
