// Not divide here.
// Ser EQ/NOC/SOC enum class to divide, in the problem maker.
// In problem maker, we need to stack it following previous method / or can make another approach?

#pragma once

#include "automatic_differentiator.h"
#include "types.h"

template <typename Derived, typename Scalar, DiffMethod diff_method = DiffMethod::Custom>
class TerminalConstraintFunction {
public:
    TerminalConstraintFunction() = default;
    ~TerminalConstraintFunction() = default;

    template <typename Vector_Cst>
    Vector_Cst cT_(const Vector_Cst& x) const {
        return static_cast<const Derived*>(this)->cT(x);
    }

    Matrix<Scalar> cTx_(const Vector<Scalar>& x) const {
        if constexpr (diff_method == DiffMethod::Custom) {
            return static_cast<const Derived*>(this)->cTx(x);
        }
        // else if constexpr (diff_method == DiffMethod::Autodiff) {
        //     return autodiff_utils::jacobianX(*this, x);
        // }
    }

    Matrix<Scalar> cTu_(const Vector<Scalar>& x) const {
        if constexpr (diff_method == DiffMethod::Custom) {
            return static_cast<const Derived*>(this)->cTu(x);
        }
        // else if constexpr (diff_method == DiffMethod::Autodiff) {
        //     return autodiff_utils::jacobianU(*this, x);
        // }
    }

    ConstraintType getConstraintType() const { return constraint_type; }

    int getDimCT() const { return dim_cT; }

protected:
    ConstraintType constraint_type;
    int dim_cT;
};