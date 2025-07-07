#pragma once

#include "stage_cost_base.h"

template <typename Scalar>
class QuadraticStageCost : public StageCostBase<Scalar> {
public:
    QuadraticStageCost() {
    }

    QuadraticStageCost(const Matrix<Scalar>& Q_in, const Matrix<Scalar>& R_in) : Q(Q_in), R(R_in) {
    }

    virtual Scalar q(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        return 0.5 * (x.transpose() * Q * x + u.transpose() * R * u).value();
    }

    virtual Vector<Scalar> qx(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        return Q * x;
    }

    virtual Vector<Scalar> qu(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        return R * u;
    }

    virtual Matrix<Scalar> qxx(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        return Q;
    }

    virtual Matrix<Scalar> quu(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        return R;
    }

    virtual Matrix<Scalar> qxu(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        return Matrix<Scalar>::Zero(Q.rows(), R.rows());
    }

    void setQ(const Matrix<Scalar>& Q_in) {
        Q = Q_in;
    }

    void setR(const Matrix<Scalar>& R_in) {
        R = R_in;
    }

private:
    Matrix<Scalar> Q;
    Matrix<Scalar> R;
};
