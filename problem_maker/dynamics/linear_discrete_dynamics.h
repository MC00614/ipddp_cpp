#pragma once

#include "discrete_dynamics_base.h"

template <typename Scalar>
class LinearDiscreteDynamics : public DiscreteDynamicsBase<Scalar> {
public:
    LinearDiscreteDynamics() {
        this->C = Vector<Scalar>::Zero(this->dim_x);
    }

    LinearDiscreteDynamics(const Matrix<Scalar>& A_in, const Matrix<Scalar>& B_in) : A(A_in), B(B_in) {
        this->dim_x = A_in.cols();
        this->dim_u = B_in.cols();
        this->C = Vector<Scalar>::Zero(this->dim_x);
    }

    virtual Vector<Scalar> f(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        return A * x + B * u + C;
    }

    virtual Matrix<Scalar> fx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        return A;
    }

    virtual Matrix<Scalar> fu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        return B;
    }

    void setA(const Matrix<Scalar>& A_in) {
        this->dim_x = A_in.cols();
        A = A_in;
    }

    void setB(const Matrix<Scalar>& B_in) {
        this->dim_u = B_in.cols();
        B = B_in;
    }

    void setC(const Vector<Scalar>& C_in) {
        C = C_in;
    }

private:
    Matrix<Scalar> A;
    Matrix<Scalar> B;
    Vector<Scalar> C;
};
