#include "discrete_dynamics.h"

template <typename Scalar>
class PointMass3D : public DiscreteDynamics<PointMass3D<Scalar>, Scalar> {
public:
    PointMass3D() {
        this->dim_x = 6;
        this->dim_u = 3;
        this->dt = 0.1;
    }

    template <typename Vector_Dyn>
    Vector_Dyn f(const Vector_Dyn& x, const Vector_Dyn& u) const {
        Vector_Dyn xnext(6);
        auto pos = x.segment(0,3);
        auto vel = x.segment(3,3);

        Vector_Dyn acc = u;
        acc(2) -= 9.81;

        auto pos_next = pos + this->dt * vel;
        auto vel_next = vel + this->dt * acc;

        xnext << pos_next, vel_next;
        return xnext;
    }

    Matrix<Scalar> fx(const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        Matrix<Scalar> Fx = Matrix<Scalar>::Zero(6,6);
        Fx.block(0,0,3,3) = Matrix<Scalar>::Identity(3,3);
        Fx.block(0,3,3,3) = this->dt * Matrix<Scalar>::Identity(3,3);
        Fx.block(3,3,3,3) = Matrix<Scalar>::Identity(3,3);
        return Fx;
    }

    Matrix<Scalar> fu(const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        Matrix<Scalar> Fu = Matrix<Scalar>::Zero(6,3);
        Fu.block(3,0,3,3) = this->dt * Matrix<Scalar>::Identity(3,3);
        return Fu;
    }
};
