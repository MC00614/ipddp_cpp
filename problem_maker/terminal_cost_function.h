#pragma once

#include "automatic_differentiator.h"
#include "types.h"

template <typename Derived, typename Scalar, DiffMethod diff_method = DiffMethod::Custom>
class TerminalCostFunction {
public:
    TerminalCostFunction() = default;
    ~TerminalCostFunction() = default;

    template <typename Vector_Cst, typename Vector_Dyn>
    Vector_Cst p_(const Vector_Dyn& x) const {
        return static_cast<const Derived*>(this)->p(x);
    }

    Matrix<Scalar> px_(const Vector<Scalar>& x) const {
        if constexpr (diff_method == DiffMethod::Custom) {
            return static_cast<const Derived*>(this)->px(x);
        }
        else if constexpr (diff_method == DiffMethod::Autodiff) {
            return autodiff_utils::derivativeX(*this, x);
        }
    }

    Matrix<Scalar> pxx_(const Vector<Scalar>& x) const {
        if constexpr (diff_method == DiffMethod::Custom) {
            return static_cast<const Derived*>(this)->pxx(x);
        }
        else if constexpr (diff_method == DiffMethod::Autodiff) {
            return autodiff_utils::hessianX(*this, x);
        }
    }
};