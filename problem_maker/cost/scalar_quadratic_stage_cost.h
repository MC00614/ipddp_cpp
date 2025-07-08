#pragma once

#include "stage_cost_base.h"

template <typename Scalar>
class ScalarQuadraticStageCost : public StageCostBase<Scalar> {
public:
    ScalarQuadraticStageCost() {
    }

    ScalarQuadraticStageCost(const Scalar& Q_in, const Scalar& R_in) : Q(Q_in), R(R_in) {
    }

    virtual Scalar q(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        return 0.5 * (Q * x.squaredNorm() + R * u.squaredNorm());
    }

    virtual Vector<Scalar> qx(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        return Q * x;
    }

    virtual Vector<Scalar> qu(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        return R * u;
    }

    virtual Matrix<Scalar> qxx(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        return Q * Matrix<Scalar>::Identity(x.size(), x.size());
    }

    virtual Matrix<Scalar> quu(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        return R * Matrix<Scalar>::Identity(u.size(), u.size());
    }

    virtual Matrix<Scalar> qxu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        return Matrix<Scalar>::Zero(x.rows(), u.rows());
    }

    void setQ(const Scalar& Q_in) {
        Q = Q_in;
    }

    void setR(const Scalar& R_in) {
        R = R_in;
    }

private:
    Scalar Q;
    Scalar R;
    // Need Optimization with minimal cache (in init)
    // Matrix<Scalar> Q_mat;
    // Matrix<Scalar> R_mat;
};
