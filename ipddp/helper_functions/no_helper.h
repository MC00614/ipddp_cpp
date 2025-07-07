#pragma once

#include <Eigen/Dense>
#include <cmath>

namespace no_helper
{

inline void addBarrierCost(double& barrier_cost, const Eigen::VectorXd& s) {
    barrier_cost += s.array().log().sum();
}

inline bool isFractionToBoundary(const Eigen::VectorXd& s_new,
                                const Eigen::VectorXd& s_old,
                                const double& one_tau) {
    return !((s_new.array() >= one_tau * s_old.array()).all());
}

/*
inline void calcDualResidual(Eigen::VectorXd& out,
                            const Eigen::VectorXd& s,
                            const Eigen::VectorXd& y,
                            const Eigen::VectorXd& e,
                            const double& mu) {
    out = s.array() * y.array() - mu * e.array();
}

inline void calcResidual(Eigen::VectorXd& out,
                        const Eigen::VectorXd& y,
                        const Eigen::VectorXd& rp,
                        const Eigen::VectorXd& rd) {
    out = y.array() * rp.array() - rd.array();
}

inline void calcLinvVector(Eigen::VectorXd& out,
                        const Eigen::VectorXd& s,
                        const Eigen::VectorXd& r) {
    out = r.array() / s.array();
}

inline void calcLinvVectorMatrix(Eigen::MatrixXd& out,
                                const Eigen::VectorXd& s,
                                const Eigen::VectorXd& y,
                                const Eigen::MatrixXd& Mat) {
    out = Mat.array().colwise() * (y.array() / s.array());
}
*/

} // namespace no_helper