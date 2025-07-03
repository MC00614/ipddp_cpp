#pragma once

#include "automatic_differentiator.h"
#include "types.h"

template <typename Derived, typename Scalar, DiffMethod diff_method = DiffMethod::Custom>
class CostFunction {
public:
    CostFunction() = default;
    ~CostFunction() = default;

    template <typename Vector_Cost>
    Vector_Cost p_(const Vector_Cost& x) const {
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
            return autodiff_utils::HessianXU(*this, x, u);
        }
    }
};