#pragma once

#include "terminal_cost_base.h"

template <typename Scalar>
class QuadraticTerminalCost : public TerminalCostBase<Scalar> {
public:
    QuadraticTerminalCost() {
    }

    QuadraticTerminalCost(const Matrix<Scalar>& Qf_in) : Qf(Qf_in) {
    }

    virtual Scalar p(const Vector<Scalar>& x) const override {
        return 0.5 * x.transpose() * Qf * x;
    }

    virtual Vector<Scalar> px(const Vector<Scalar>& x) const override {
        return Qf * x;
    }

    virtual Matrix<Scalar> pxx(const Vector<Scalar>&) const override {
        return Qf;
    }

    void setQf(const Matrix<Scalar>& Qf_in) {
        Qf = Qf_in;
    }

private:
    Matrix<Scalar> Qf;
};
