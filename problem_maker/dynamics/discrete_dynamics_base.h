#pragma once

#include "types.h"

template <typename Scalar>
class DiscreteDynamicsBase {
public:
    virtual ~DiscreteDynamicsBase() = default;

    virtual Vector<Scalar> f(const Vector<Scalar>& x, const Vector<Scalar>& u) const = 0;
    virtual Matrix<Scalar> fx(const Vector<Scalar>& x, const Vector<Scalar>& u) const = 0;
    virtual Matrix<Scalar> fu(const Vector<Scalar>& x, const Vector<Scalar>& u) const = 0;

    virtual int getDimX() const {
        return this->dim_x;
    };
    virtual int getDimU() const {
        return this->dim_u;
    };
    virtual Scalar getDT() const {
        return this->dt;
    };

protected:
    int dim_x{};
    int dim_u{};
    Scalar dt{};
};