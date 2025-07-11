#pragma once

#include <Eigen/Dense>
#include <cmath>

namespace soc_helper
{
inline void LtimesVec(Eigen::Ref<Eigen::VectorXd> out,
                      const Eigen::Ref<const Eigen::VectorXd>& soc,
                      const Eigen::Ref<const Eigen::VectorXd>& vec) {
    const double& s0 = soc(0);
    const int n = soc.size() - 1;
    const Eigen::Ref<const Eigen::VectorXd>& v = soc.tail(n);
    out(0) = s0 * vec(0) + v.dot(vec.tail(n));
    out.tail(n) = vec(0) * v + s0 * vec.tail(n);
}

inline void LinvTimesVec(Eigen::Ref<Eigen::VectorXd> out,
                         const Eigen::Ref<const Eigen::VectorXd>& soc,
                         const Eigen::Ref<const Eigen::VectorXd>& vec) {
    const double& s0 = soc(0);
    const int n = soc.size() - 1;
    const Eigen::Ref<const Eigen::VectorXd>& v = soc.tail(n);
    const double denom = s0 * s0 - v.squaredNorm();
    out(0) = (s0 * vec(0) - v.dot(vec.tail(n))) / denom;
    out.tail(n).noalias() = (- out(0) / s0) * v;
    out.tail(n) += (vec.tail(n) / s0);
}

inline void Linv(Eigen::Ref<Eigen::MatrixXd> out, const Eigen::Ref<const Eigen::VectorXd>& soc) {
    const double s0 = soc(0);
    const int n = soc.size() - 1;
    const Eigen::Ref<const Eigen::VectorXd> v = soc.tail(n);
    const double denom = s0 * s0 - v.squaredNorm();
    out(0,0) = s0 / denom;
    out.block(1,0,n,1) = - v / denom;
    out.block(0,1,1,n) = out.block(1,0,n,1).transpose();
    out.block(1,1,n,n).noalias() = (v * v.transpose()) / (denom * s0);
    out.block(1,1,n,n).diagonal().array() += (1.0 / s0);
}

inline void LinvTimesArrow(Eigen::Ref<Eigen::MatrixXd> out,
                           const Eigen::Ref<const Eigen::VectorXd>& soc_inv,
                           const Eigen::Ref<const Eigen::VectorXd>& soc_arrow) {
    const int n = soc_inv.size() - 1;
    const double s = soc_inv(0);
    const Eigen::Ref<const Eigen::VectorXd>& v = soc_inv.tail(n);
    const double a = soc_arrow(0);
    const Eigen::Ref<const Eigen::VectorXd>& b = soc_arrow.tail(n);
    const double inv_s = 1.0 / s;
    const double denom = s * s - v.squaredNorm();
    const double inv_denom = 1.0 / denom;
    Eigen::VectorXd mvs = - v * inv_s;

    out(0,0) = (s * a - v.dot(b)) * inv_denom;
    out.block(1,0,n,1).noalias() = mvs * out(0,0);
    out.block(0,1,1,n) = (s * b.transpose() - a * v.transpose()) * inv_denom;
    out.block(1,1,n,n).noalias() = mvs * out.block(0,1,1,n);
    out.block(1,1,n,n).diagonal().array() += a * inv_s;
}

inline void LinvArrow(Eigen::Ref<Eigen::MatrixXd> out,
                      const Eigen::Ref<const Eigen::MatrixXd>& l_inv,
                      const Eigen::Ref<const Eigen::VectorXd>& soc) {
    const double s = soc(0);
    const int n = soc.size() - 1;
    const Eigen::Ref<const Eigen::VectorXd>& v = soc.tail(n);
    const double a = l_inv(0,0);
    const Eigen::Ref<const Eigen::VectorXd>& b = l_inv.block(1,0,n,1);
    const Eigen::Ref<const Eigen::MatrixXd>& C = l_inv.block(1,1,n,n);

    out(0,0) = a * s + b.dot(v);
    out.block(0,1,1,n).noalias() = (a * v + s * b).transpose();
    out.block(1,0,n,1).noalias() = s * b;
    out.block(1,0,n,1) += C * v;
    out.block(1,1,n,n).noalias() = s * C;
    out.block(1,1,n,n) += b * v.transpose();
}

inline void addBarrierCost(double& barrier_cost,
                            const Eigen::VectorXd& s,
                            const std::vector<int>& dim_hs,
                            const std::vector<int>& dim_hs_top) {
    for (int i = 0; i < dim_hs.size(); ++i) {
        const int d   = dim_hs[i] - 1;
        const int idx = dim_hs_top[i];
        const double s0 = s(idx);
        const double v_2 = s.segment(idx + 1, d).squaredNorm();
        barrier_cost += 0.5 * std::log(s0 * s0 - v_2);
    }
}

inline bool isFractionToBoundary(const Eigen::VectorXd& s_new,
                        const Eigen::VectorXd& s_old,
                        const double& one_tau,
                        const std::vector<int>& dim_hs,
                        const std::vector<int>& dim_hs_top) {
    for (int i = 0; i < dim_hs.size(); ++i) {
        const int d   = dim_hs[i] - 1;
        const int idx = dim_hs_top[i];
        if (s_new(idx) - s_new.segment(idx + 1, d).norm() < one_tau * (s_old(idx) - s_old.segment(idx + 1, d).norm())){
            return true;
        }
    }
    return false;
}

/*
inline Eigen::VectorXd calcDualResidual(const Eigen::VectorXd& s,
                                        const Eigen::VectorXd& y,
                                        const Eigen::VectorXd& e,
                                        const double& mu,
                                        const std::vector<int>& dim_hs,
                                        const std::vector<int>& dim_hs_top) {
    Eigen::VectorXd rd(s.size());
    for (int i = 0; i < dim_hs.size(); ++i) {
        const int d   = dim_hs[i];
        const int idx = dim_hs_top[i];
        LtimesVec(rd.segment(idx, d), y.segment(idx, d), s.segment(idx, d));
    }
    rd -= mu * e;
    return rd;
}

inline Eigen::VectorXd calcResidual(const Eigen::VectorXd& y,
                                    const Eigen::VectorXd& rp,
                                    const Eigen::VectorXd& rd,
                                    const std::vector<int>& dim_hs,
                                    const std::vector<int>& dim_hs_top) {
    Eigen::VectorXd r(y.size());
    for (int i = 0; i < dim_hs.size(); ++i) {
        const int d   = dim_hs[i];
        const int idx = dim_hs_top[i];
        LtimesVec(r.segment(idx, d), y.segment(idx, d), rp.segment(idx, d));
    }
    return r - rd;
}

inline Eigen::VectorXd calcLinvVector(const Eigen::VectorXd& s,
                                        const Eigen::VectorXd& r,
                                        const std::vector<int>& dim_hs,
                                        const std::vector<int>& dim_hs_top) {
    Eigen::VectorXd out(r.size());
    for (int i = 0; i < dim_hs.size(); ++i) {
        const int d   = dim_hs[i];
        const int idx = dim_hs_top[i];
        LinvTimesVec(out.segment(idx, d), s.segment(idx, d), r.segment(idx, d));
    }
    return out;
}

inline Eigen::MatrixXd calcLinvVectorMatrix(const Eigen::VectorXd& s,
                                            const Eigen::VectorXd& y,
                                            const Eigen::MatrixXd& Mat,
                                            const std::vector<int>& dim_hs,
                                            const std::vector<int>& dim_hs_top,
                                            int dim_hs_max) {
    Eigen::MatrixXd out(Mat.rows(), Mat.cols());
    Eigen::MatrixXd Sinv_Y_block(dim_hs_max, dim_hs_max);
    for (int i = 0; i < dim_hs.size(); ++i) {
        const int d   = dim_hs[i];
        const int idx = dim_hs_top[i];
        Eigen::Ref<Eigen::MatrixXd> Sinv_Y = Sinv_Y_block.topLeftCorner(d, d);
        LinvTimesArrow(Sinv_Y, s.segment(idx, d), y.segment(idx, d));
        out.middleRows(idx, d) = Sinv_Y * Mat.middleRows(idx, d);
    }
    return out;
}
*/

} // namespace soc_helper