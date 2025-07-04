// Not divide here.
// Ser EQ/NOC/SOC enum class to divide, in the problem maker.
// In problem maker, we need to stack it following previous method / or can make another approach?

#pragma once

#include "automatic_differentiator.h"
#include "types.h"

template <typename Derived, typename Scalar, DiffMethod diff_method = DiffMethod::Custom>
class StageConstraintFunction {
public:
    StageConstraintFunction() = default;
    ~StageConstraintFunction() = default;

    template <typename Vector_Cst, typename Vector_Dyn>
    Vector_Cst c_(const Vector_Dyn& x, const Vector_Dyn& u) const {
        return static_cast<const Derived*>(this)->c(x, u);
    }

    Matrix<Scalar> cx_(const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        if constexpr (diff_method == DiffMethod::Custom) {
            return static_cast<const Derived*>(this)->cx(x, u);
        }
        // else if constexpr (diff_method == DiffMethod::Autodiff) {
        //     return autodiff_utils::jacobianX(*this, x, u);
        // }
    }

    Matrix<Scalar> cu_(const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        if constexpr (diff_method == DiffMethod::Custom) {
            return static_cast<const Derived*>(this)->cu(x, u);
        }
        // else if constexpr (diff_method == DiffMethod::Autodiff) {
        //     return autodiff_utils::jacobianU(*this, x, u);
        // }
    }

    ConstraintType getConstraintType() const { return constraint_type; }
    
    int getDimC() const { return dim_c; }


protected:
    ConstraintType constraint_type;
    int dim_c;
};