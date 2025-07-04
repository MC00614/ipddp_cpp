#pragma once

#include <Eigen/Dense>

template <typename Scalar>
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

template <typename Scalar>
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

enum class ConstraintType {
    NO = 0,
    SOC = 1,
    EQ = 2
};

// enum class DiffMethod {
//     Custom,
//     Autodiff
// };
