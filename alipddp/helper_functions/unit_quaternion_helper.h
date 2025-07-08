#pragma once

#include <Eigen/Dense>
#include "types.h"

namespace quaternion_helper
{

inline Eigen::Matrix<double,4,3> getH() {
    Eigen::Matrix<double,4,3> H;
    H << 0,0,0,
            1,0,0,
            0,1,0,
            0,0,1;
    return H;
}

inline Eigen::Vector4d Phi(Eigen::Vector3d w) {
    Eigen::Vector4d phi;
    phi << 1, w;
    // phi.normalize();
    phi /= sqrt(1 + w.squaredNorm());
    return phi;
}

inline Eigen::Matrix4d Lq(const Eigen::Vector4d& q) {
    Eigen::Matrix4d lq;
    lq << q(0), -q(1), -q(2), -q(3),
          q(1),  q(0), -q(3),  q(2),
          q(2),  q(3),  q(0), -q(1),
          q(3), -q(2),  q(1),  q(0);
    return lq;
}

inline void calcE(Eigen::MatrixXd& E, const Eigen::VectorXd& x, int q_idx) {
    E.setIdentity(x.size(), x.size());
    E.block(q_idx, q_idx, 4, 3) = Lq(x.segment(q_idx,4)) * getH();
}

inline void calcEE(Eigen::MatrixXd& EE, const Eigen::VectorXd& x_n, int q_idx) {
    EE.setIdentity(x_n.size(), x_n.size());
    EE.block(q_idx, q_idx, 4, 3) = Lq(x_n.segment(q_idx,4)) * getH();
}

inline void Id(Eigen::MatrixXd id, int q_idx, int dim_rn, double q_q) {
    id = Eigen::MatrixXd::Zero(dim_rn, dim_rn);
    id.block(q_idx, q_idx, 3, 3) = q_q * Eigen::Matrix3d::Identity();
}

} // namespace quaternion_helper