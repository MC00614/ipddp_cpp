#pragma once

#include "automatic_differentiator.h"
#include "types.h"

template <typename Derived, typename Scalar, DiffMethod diff_method = DiffMethod::Custom>
class StageCostFunction {
public:
    StageCostFunction() = default;
    ~StageCostFunction() = default;

    template <typename Vector_Cost>
    Vector_Cost q_(const Vector_Cost& x, const Vector_Cost& u) const {
        return static_cast<const Derived*>(this)->stage(x, u);
    }

    Matrix<Scalar> qx_(const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        if constexpr (diff_method == DiffMethod::Custom) {
            return static_cast<const Derived*>(this)->q(x, u);
        }
        else if constexpr (diff_method == DiffMethod::Autodiff) {
            return autodiff_utils::derivativeX(*this, x, u);
        }
    }

    Matrix<Scalar> qu_(const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        if constexpr (diff_method == DiffMethod::Custom) {
            return static_cast<const Derived*>(this)->q(x, u);
        }
        else if constexpr (diff_method == DiffMethod::Autodiff) {
            return autodiff_utils::derivativeU(*this, x, u);
        }
    }

    Matrix<Scalar> qdd_(const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        if constexpr (diff_method == DiffMethod::Custom) {
            return static_cast<const Derived*>(this)->qdd(x, u);
        }
        else if constexpr (diff_method == DiffMethod::Autodiff) {
            return autodiff_utils::hessianXU(*this, x, u);
        }
    }
};