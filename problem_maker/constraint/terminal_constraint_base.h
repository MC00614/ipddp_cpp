#pragma once

#include "types.h"

template <typename Scalar>
class TerminalConstraintBase {
public:
    virtual ~TerminalConstraintBase() = default;

    virtual Vector<Scalar> cT(const Vector<Scalar>& x) const = 0;
    virtual Matrix<Scalar> cTx(const Vector<Scalar>& x) const = 0;

    virtual ConstraintType getConstraintType() const {
        return this->constraint_type;
    };
    virtual int getDimCT() const {
        return this->dim_cT;
    };

protected:
    ConstraintType constraint_type;
    int dim_cT;
};