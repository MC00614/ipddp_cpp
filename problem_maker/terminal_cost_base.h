#pragma once

#include "types.h"

template <typename Scalar>
class TerminalCostBase {
public:
    virtual ~TerminalCostBase() = default;

    virtual Scalar p(const Vector<Scalar>& x) const = 0;
    virtual Vector<Scalar> px(const Vector<Scalar>& x) const = 0;
    virtual Matrix<Scalar> pxx(const Vector<Scalar>& x) const = 0;
};