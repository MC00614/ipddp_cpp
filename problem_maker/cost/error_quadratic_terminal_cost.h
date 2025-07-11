#pragma once

#include "terminal_cost_base.h"

template <typename Scalar>
class ErrorQuadraticTerminalCost : public TerminalCostBase<Scalar> {
public:
    ErrorQuadraticTerminalCost() {
    }

    ErrorQuadraticTerminalCost(const Matrix<Scalar>& Qf_in, const Vector<Scalar>& x_ref) : Qf(Qf_in), x_ref(x_ref) {
    }

    virtual Scalar p(const Vector<Scalar>& x) const override {
        return 0.5 * ((x - x_ref).transpose() * Qf * (x - x_ref)).value();
    }

    virtual Vector<Scalar> px(const Vector<Scalar>& x) const override {
        return Qf * (x - x_ref);
    }

    virtual Matrix<Scalar> pxx(const Vector<Scalar>&) const override {
        return Qf;
    }

    void setQf(const Matrix<Scalar>& Qf_in) {
        Qf = Qf_in;
    }
    
    void setXref(const Vector<Scalar>& x_ref_in) {
        x_ref = x_ref_in;
    }

private:
    Matrix<Scalar> Qf;
    Vector<Scalar> x_ref;
};
