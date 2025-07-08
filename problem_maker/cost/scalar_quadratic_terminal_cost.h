#pragma once

#include "terminal_cost_base.h"

template <typename Scalar>
class ScalarQuadraticTerminalCost : public TerminalCostBase<Scalar> {
public:
    ScalarQuadraticTerminalCost() {
    }

    ScalarQuadraticTerminalCost(const Scalar& Qf_in) : Qf(Qf_in) {
    }

    virtual Scalar p(const Vector<Scalar>& x) const override {
        return 0.5 * Qf * x.squaredNorm();
    }

    virtual Vector<Scalar> px(const Vector<Scalar>& x) const override {
        return Qf * x;
    }

    virtual Matrix<Scalar> pxx(const Vector<Scalar>& x) const override {
        return Qf * Matrix<Scalar>::Identity(x.size(), x.size());
    }

    void setQf(const Scalar& Qf_in) {
        Qf = Qf_in;
    }

private:
    Scalar Qf;
    // Need Optimization with minimal cache (in init)
    // Matrix<Scalar> Qf_mat;
};
