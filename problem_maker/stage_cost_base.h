#pragma once

#include "types.h"

template <typename Scalar>
class StageCostBase {
public:
    virtual ~StageCostBase() = default;

    virtual Scalar q(const Vector<Scalar>& x, const Vector<Scalar>& u) const = 0;

    virtual Vector<Scalar> qx(const Vector<Scalar>& x, const Vector<Scalar>& u) const = 0;
    virtual Vector<Scalar> qu(const Vector<Scalar>& x, const Vector<Scalar>& u) const = 0;

    virtual Matrix<Scalar> qxx(const Vector<Scalar>& x, const Vector<Scalar>& u) const = 0;
    virtual Matrix<Scalar> quu(const Vector<Scalar>& x, const Vector<Scalar>& u) const = 0;
    virtual Matrix<Scalar> qxu(const Vector<Scalar>& x, const Vector<Scalar>& u) const = 0;
};