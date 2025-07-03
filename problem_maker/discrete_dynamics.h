#pragma once

#include "automatic_differentiator.h"
#include "types.h"

// FutureWork: Implement calcDiff function to intergrate differentiation of fx and fu
// This will be efficient for solver side, also for tape of Autodiff
// Considering the current status of solver, we leave it.

template <typename Derived, typename Scalar, DiffMethod diff_method = DiffMethod::Custom>
class DiscreteDynamics {
public:
    DiscreteDynamics() = default;
    ~DiscreteDynamics() = default;

    template <typename Vector_Dyn>
    Vector_Dyn f_(const Vector_Dyn& x, const Vector_Dyn& u) const {
        return static_cast<const Derived*>(this)->f(x, u);
    }

    Matrix<Scalar> fx_(const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        if constexpr (diff_method == DiffMethod::Custom) {
            return static_cast<const Derived*>(this)->fx(x, u);
        }
        else if constexpr (diff_method == DiffMethod::Autodiff) {
            return autodiff_utils::jacobianX(*this, x, u);
        }
    }

    Matrix<Scalar> fu_(const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        if constexpr (diff_method == DiffMethod::Custom) {
            return static_cast<const Derived*>(this)->fu(x, u);
        }
        else if constexpr (diff_method == DiffMethod::Autodiff) {
            return autodiff_utils::jacobianU(*this, x, u);
        }
    }

    int getDimX() const { return dim_x; }
    int getDimU() const { return dim_u; }
    Scalar getDT() const { return dt; }

protected:
    int dim_x{};
    int dim_u{};
    Scalar dt{};
};