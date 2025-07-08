#pragma once

#include <Eigen/Dense>

namespace quaternion_helper
{

inline const Eigen::Matrix<double, 4, 3>& getH() {
    static const Eigen::Matrix<double, 4, 3> H = [] {
        Eigen::Matrix<double, 4, 3> h;
        h << 0, 0, 0,
             1, 0, 0,
             0, 1, 0,
             0, 0, 1;
        return h;
    }();
    return H;
}
inline Eigen::Vector4d Phi(Eigen::Vector3d w) {
    Eigen::Vector4d phi;
    phi << 1, w;
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

inline void calcE(Eigen::MatrixXd& E, const Eigen::VectorXd& x, const int& q_idx, const int& dim_rn) {
    E.setIdentity(dim_rn + 1, dim_rn);
    E.block(q_idx, q_idx, 4, 3) = Lq(x.segment(q_idx,4)) * getH();
}

inline void calcEE(Eigen::MatrixXd& EE, const Eigen::VectorXd& x_n, const int& q_idx, const int& dim_rn) {
    EE.setIdentity(dim_rn + 1, dim_rn);
    EE.block(q_idx, q_idx, 4, 3) = Lq(x_n.segment(q_idx,4)) * getH();
}

inline void Id(Eigen::MatrixXd& id, const double& q_q, const int& q_idx, const int& dim_rn) {
    id = Eigen::MatrixXd::Zero(dim_rn, dim_rn);
    id.block(q_idx, q_idx, 3, 3) = q_q * Eigen::Matrix3d::Identity();
}

} // namespace quaternion_helper