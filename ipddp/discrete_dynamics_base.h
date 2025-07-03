#pragma once

#include "types.h"
#include <stdexcept>

template <typename Scalar>
class DiscreteDynamicsBase {
public:
    DiscreteDynamicsBase() = default;
    virtual ~DiscreteDynamicsBase() = default;

    virtual Vector<Scalar> f(const Vector<Scalar>& x, const Vector<Scalar>& u) = 0;

    virtual Matrix<Scalar> fx(const Vector<Scalar>& x, const Vector<Scalar>& u) {
        return autodiff_utils::jacobianX(this, x, u);
    }

    virtual Matrix<Scalar> fu(const Vector<Scalar>& x, const Vector<Scalar>& u) {
        return autodiff_utils::jacobianU(this, x, u);
    }

    virtual int stateDim() const final { return nx; }
    virtual int controlDim() const final { return nu; }
    virtual Scalar timeStep() const final { return dt; }

protected:
    int nx;
    int nu;
    Scalar dt;
};