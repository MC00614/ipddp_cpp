#include "discrete_dynamics_base.hpp"

template <typename Scalar>
class PointMass3D : public DiscreteDynamicsBase<Scalar> {
public:
    PointMass3D() {
        this->nx = 6;
        this->nu = 3;
        this->dt = 0.1;
    }

    Vector<Scalar> f(const Vector<Scalar>& x, const Vector<Scalar>& u) override {
        Vector<Scalar> xnext(6);
        Vector<Scalar> pos = x.segment(0,3);
        Vector<Scalar> vel = x.segment(3,3);

        Vector<Scalar> acc = u;
        acc(2) -= 9.81;

        Vector<Scalar> pos_next = pos + dt_ * vel;
        Vector<Scalar> vel_next = vel + dt_ * acc;

        xnext << pos_next, vel_next;
        return xnext;
    }

    Matrix<Scalar> fx(const Vector<Scalar>& x, const Vector<Scalar>& u) override {
        Matrix<Scalar> Fx = Matrix<Scalar>::Zero(6,6);
        Fx.block(0,0,3,3) = Matrix<Scalar>::Identity(3,3);
        Fx.block(0,3,3,3) = dt_ * Matrix<Scalar>::Identity(3,3);
        Fx.block(3,3,3,3) = Matrix<Scalar>::Identity(3,3);
        return Fx;
    }

    Matrix<Scalar> fu(const Vector<Scalar>& x, const Vector<Scalar>& u) override {
        Matrix<Scalar> Fu = Matrix<Scalar>::Zero(6,3);
        Fu.block(3,0,3,3) = dt_ * Matrix<Scalar>::Identity(3,3);
        return Fu;
    }
};
