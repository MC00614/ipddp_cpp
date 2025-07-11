#pragma once

#include "stage_cost_base.h"

template <typename Scalar>
class VectorStageCost : public StageCostBase<Scalar> {
public:
    VectorStageCost() {
    }

    VectorStageCost(const Vector<Scalar>& Q_in, const Vector<Scalar>& R_in) : Q(Q_in), R(R_in) {
    }

    virtual Scalar q(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        return Q.dot(x) + R.dot(u);
    }

    virtual Vector<Scalar> qx(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        return Q;
    }

    virtual Vector<Scalar> qu(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        return R;
    }

    virtual Matrix<Scalar> qxx(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        return Matrix<Scalar>::Zero(x.size(), x.size());
    }

    virtual Matrix<Scalar> quu(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        return Matrix<Scalar>::Zero(u.size(), u.size());
    }

    virtual Matrix<Scalar> qxu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        return Matrix<Scalar>::Zero(x.rows(), u.rows());
    }

    void setQ(const Vector<Scalar>& Q_in) {
        Q = Q_in;
    }

    void setR(const Vector<Scalar>& R_in) {
        R = R_in;
    }

private:
    Vector<Scalar> Q;
    Vector<Scalar> R;
};
