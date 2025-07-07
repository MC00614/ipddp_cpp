#pragma once

#include "types.h"

template <typename Scalar>
class TerminalConstraintBase {
public:
    virtual ~TerminalConstraintBase() = default;

    virtual Vector<Scalar> cT(const Vector<Scalar>& x) const = 0;
    virtual Matrix<Scalar> cTx(const Vector<Scalar>& x) const = 0;

    virtual void setConstraintType(ConstraintType constraint_type) {
        this->constraint_type = constraint_type;
    };
    virtual void setDimCT(int dim_cT) {
        this->dim_cT = dim_cT;
    };
    
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