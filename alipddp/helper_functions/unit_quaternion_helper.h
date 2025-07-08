// #pragma once

// #include <Eigen/Dense>
// #include "types.h"

// namespace quaternion_helper
// {

// inline Eigen::Matrix4d Lq(const Eigen::Vector4d& q) {
//     Eigen::Matrix4d lq;
//     lq << q(0), -q(1), -q(2), -q(3),
//           q(1),  q(0), -q(3),  q(2),
//           q(2),  q(3),  q(0), -q(1),
//           q(3), -q(2),  q(1),  q(0);
//     return lq;
// }

// inline void calcE(int q_idx, const Eigen::VectorXd& x, Eigen::MatrixXd& E) {
//     E.setIdentity(x.size(), x.size());
//     E.block(q_idx, q_idx, 4, 3) = Lq(x.segment(q_idx,4)) * getH();
// }

// inline void calcEE(int q_idx, const Eigen::VectorXd& fv, Eigen::MatrixXd& EE) {
//     EE.setIdentity(fv.size(), fv.size());
//     EE.block(q_idx, q_idx, 4, 3) = Lq(fv.segment(q_idx,4)) * getH();
// }

// inline Eigen::Matrix<double,4,3> getH() {
//     Eigen::Matrix<double,4,3> H;
//     H << 0,0,0,
//          1,0,0,
//          0,1,0,
//          0,0,1;
//     return H;
// }

// inline Eigen::MatrixXd Id(int q_idx, int dim_rn, double q_q) {
//     Eigen::MatrixXd id = Eigen::MatrixXd::Zero(dim_rn, dim_rn);
//     id.block(q_idx, q_idx, 3, 3) = q_q * Eigen::Matrix3d::Identity();
//     return id;
// }

// } // namespace quaternion_helper